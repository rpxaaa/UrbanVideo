import cv2
import base64
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from .state import GraphState
from .config import MODEL_NAME, API_KEY, BASE_URL, is_baseline, get_frame_count
from .video_utils import extract_frames as video_extract_frames, get_sampled_indices
from .memory_builder import build_memory, format_memory_for_prompt, format_path_timeline

# ── Category-specific prompts ─────────────────────────────────────────
# Structured Spatial Chain-of-Thought (Spatial CoT) prompts
# These prompts guide the model through a 4-step spatial reasoning process:
# 1. [Visual Anchors]: Identify core landmarks/objects
# 2. [Coordinate Mapping]: Establish relative positions and depth
# 3. [Topological Reasoning]: Infer movement vectors and spatial relations
# 4. [Final Decision]: Match with options

SPATIAL_COT_BASE = (
    "Please analyze the video frames using the following structured spatial reasoning process:\n"
    "Step 1 [Visual Anchors]: Identify the core landmarks or reference objects in the scene (e.g., specific buildings, roads, waterfronts).\n"
    "Step 2 [Coordinate Mapping]: Describe your relative position to these anchors using a 3x3 grid concept (left/center/right, top/middle/bottom) and depth (near/mid/far).\n"
    "Step 3 [Topological Reasoning]: Based on the visual evidence and any movement history, reason about the spatial relationships (e.g., altitude changes, turns, ego-centric vs. allocentric perspective).\n"
    "Step 4 [Final Decision]: Based on the reasoning above, select the correct option.\n\n"
)

CATEGORY_PROMPTS = {
    # ═══ Navigation & Planning ═══

    "Progress Evaluation": (
        SPATIAL_COT_BASE +
        "You are a drone following navigation instructions. Determine WHICH step you just completed.\n\n"
        "Question: {question}\n\n"
        "CRITICAL: The last frame is NOT necessarily the end of all instructions. "
        "The video may end mid-route. Judge ONLY by visual scene evidence, NOT frame position.\n\n"
        "MOTION CUES (use to judge if a step was executed):\n"
        "  Turn left  → scene shifts RIGHT.   Turn right → scene shifts LEFT.\n"
        "  Ascend     → more sky, ground objects shrink.  Descend → more ground, objects enlarge.\n"
        "  Forward    → center objects grow larger over frames.\n"
        "  Approach X → X becomes larger/closer.  Reach X → X dominates the final frames.\n\n"
        "DECISION STEPS:\n"
        "1. Parse the instruction into Step 1, Step 2, Step 3...\n"
        "2. For each step, check ALL frames for its expected motion cue. Mark [DONE] or [NOT YET].\n"
        "3. Identify the LAST step marked [DONE] before the first [NOT YET].\n"
        "4. If you see clear evidence of step N's completion AND clear evidence that step N+1 has NOT started,\n"
        "   then step N is your answer. If multiple steps could be [DONE], pick the one with strongest visual support.\n\n"
        "Provide your final answer as 'Option: [X]' where X is the letter."
    ),

    "Action Generation": (
        SPATIAL_COT_BASE +
        "You are a drone navigating in a first-person urban view. Determine the SINGLE best next action.\n\n"
        "Question: {question}\n\n"
        "FRAME REFERENCE GUIDE (frames are ordered chronologically):\n"
        "  Frames 1-4: early state / starting position\n"
        "  Frames 5-10: mid-route (most navigation steps execute here)\n"
        "  Frames 11-16: approach phase and final state\n\n"
        "STEP CONFIRMATION CRITERIA — a step is CONFIRMED [DONE] only when you observe its "
        "expected visual outcome in the frames. Use these criteria:\n"
        "  - \"turn left\": buildings/objects shift from center toward the RIGHT side of frame\n"
        "  - \"turn right\": buildings/objects shift from center toward the LEFT side of frame\n"
        "  - \"go up / fly higher / ascend\": ground objects appear smaller, the horizon drops lower, "
        "more sky is visible, rooftop details become visible\n"
        "  - \"go down / descend / fly lower\": ground objects appear larger, more ground details "
        "visible, the horizon rises, less sky visible\n"
        "  - \"fly forward / go straight\": objects in center grow larger, side objects move toward frame edges\n"
        "  - \"approach X\": X becomes larger/more detailed over successive frames, or X has already "
        "been passed (visible behind you or shrinking)\n"
        "  - \"reach X / arrive at X\": X dominates the view, you are very close or at the destination\n"
        "If you are UNSURE whether a step is done, mark it [PENDING] — better to re-execute than skip.\n\n"
        "INSTRUCTION AUDIT (MANDATORY — do this BEFORE selecting an answer):\n"
        "1. Parse the navigation instruction into individual steps. List them as Step 1, Step 2, Step 3, etc.\n"
        "2. For EACH step, inspect ALL video frames systematically using the criteria above and mark it as [DONE] or [PENDING]:\n"
        "   - [DONE] = visual evidence in the frames confirms this step was executed "
        "(look for: the turn happened, the altitude changed, the landmark appeared)\n"
        "   - [PENDING] = no visual evidence confirms completion, or the current state is before this step\n"
        "3. Identify the FIRST [PENDING] step — this is your target. "
        "CRITICAL: the last frame is NOT necessarily the end of the task. "
        "The video may end before all steps complete, or may capture only part of the route.\n"
        "4. Now match this target step against your current 3D state (altitude, orientation, position):\n"
        "   - If target says 'turn right' but you already turned right → mark it [DONE], re-evaluate\n"
        "   - If target says 'go up' and you are still at low altitude → the next action IS going up\n"
        "   - If target says 'approach X' and X is not yet visible → you must move toward where X would be\n"
        "5. Select the option that EXECUTES the first [PENDING] step.\n"
        "6. CONTRADICTION CHECK: for EACH option, state in one sentence why it might be WRONG "
        "(e.g., re-executes a completed step, wrong direction, wrong altitude, premature approach). "
        "Then select the option whose contradiction is weakest.\n\n"
        "In your [Coordinate Mapping], note altitude, heading direction, and nearby landmarks. "
        "In your [Topological Reasoning], match the first [PENDING] step against your current state. "
        "Exclude options that re-execute already-completed steps or are inconsistent with the required movement direction.\n\n"
        "Provide your final answer as 'Option: [X]' where X is the letter."
    ),

    "Landmark Position": (
        SPATIAL_COT_BASE +
        "You are a drone flying in an urban environment. Determine your spatial position relative to a landmark.\n\n"
        "Question: {question}\n\n"
        "VIEWPOINT DISAMBIGUATION protocol:\n"
        "1. Declare your reference frame: are you using screen-based (what the camera sees) or ego-centered "
        "(your own body/drone orientation)? You MUST use ego-centered for the final answer.\n"
        "2. Transformation rule: Screen-left + forward heading → ego-left. Screen-right + forward heading → ego-right. "
        "Screen-top + no forward tilt → ego-above. Screen-bottom + downward tilt → ego-below.\n"
        "3. In your [Topological Reasoning], determine: Are you above, beside, facing, behind, or across from the landmark? "
        "Consider both your 3D position AND your heading direction relative to the landmark's position.\n\n"
        "Provide your final answer as 'Option: [X]' where X is the letter."
    ),

    "Goal Detection": (
        "You are a drone. Determine whether a specific destination is visible from your current position.\n\n"
        "Question: {question}\n\n"
        "FRAME SCAN (go through frames 1-16 chronologically — starting viewpoint is frames 1-6):\n"
        "For EACH frame, record:\n"
        "  Frame N: target BUILDING/STRUCTURE visible? [YES / NO / PARTIAL]\n"
        "  Frame N: target FLOOR/LEVEL/BALCONY visible? [YES / NO]\n"
        "  Frame N: key FEATURE (sign, entrance, color, shape) visible? [YES / NO]\n"
        "  Frame N: distance to target? [far / mid / near]\n\n"
        "BEST EVIDENCE: Which SINGLE frame shows the target most clearly? If no frame clearly\n"
        "shows the target, state this explicitly.\n\n"
        "DECISION RULES:\n"
        "- ALL three = YES in starting frames: destination IS within sight (select 'visible/reached' option).\n"
        "- Building YES but floor/feature NO: general area visible but exact destination NOT yet reached.\n"
        "- Building NO (or only PARTIAL): destination NOT within sight from current position.\n"
        "- If target appears only in LATER frames (after frame 8): you need to move to spot it.\n\n"
        "Provide your final answer as 'Option: [X]' where X is the letter."
    ),

    "High-level Planning": (
        SPATIAL_COT_BASE +
        "You are a drone planning a navigation route in an urban 3D environment.\n\n"
        "Question: {question}\n\n"
        "DRONE AERIAL NAVIGATION PARADIGM — the shortest path between two points in 3D space is a "
        "STRAIGHT LINE through the air, NOT a ground-level walking path along streets or sidewalks. "
        "You can fly OVER buildings, THROUGH open spaces between structures, and DIRECTLY toward "
        "elevated targets.\n\n"
        "In your [Topological Reasoning]:\n"
        "1. Identify the destination's 3D position (which building, which floor/height).\n"
        "2. Determine your current 3D position (altitude, facing direction, nearby structures).\n"
        "3. Project the aerial straight-line path — what is the FIRST intermediate object or location "
        "you must approach to stay on the 3D shortest path? This is NOT a ground navigation problem.\n\n"
        "Provide your final answer as 'Option: [X]' where X is the letter."
    ),

    "Cognitive Map": (
        SPATIAL_COT_BASE +
        "You are an embodied agent. Describe the surrounding environment at your current location.\n\n"
        "Question: {question}\n\n"
        "Build a comprehensive mental picture. In your [Coordinate Mapping], place all identified anchors relative to your viewpoint "
        "(front, left, right, behind, below) ensuring spatial consistency.\n\n"
        "Provide your final answer as 'Option: [X]' where X is the letter."
    ),

    "Association Reasoning": (
        "You are a drone looking for a specific target. The target itself may NOT be directly visible. "
        "You must identify which VISIBLE object or location is the best spatial proxy for reaching the target.\n\n"
        "Question: {question}\n\n"
        "METHOD:\n"
        "1. List ALL distinctive visible objects (buildings, roads, signs, landmarks).\n"
        "2. For each option, answer: Is this object/location VISIBLE in the frames? [YES / NO]\n"
        "3. If the target is NOT visible, evaluate each visible candidate by asking:\n"
        "   - Is it in the SAME building or building complex as the target?\n"
        "   - Is it on the SAME side of the street/road?\n"
        "   - Is it at the SAME altitude/floor level?\n"
        "   - If you approach this candidate, will you be closer to the target?\n"
        "4. Select the option that is the BEST spatial waypoint — the one that shares the most "
        "spatial context with the target and positions you for the final approach.\n\n"
        "Provide your final answer as 'Option: [X]' where X is the letter."
    ),


    # ═══ Recall & Perception ═══

    "Trajectory Captioning": (
        "You are a drone. Summarize your complete movement route from the video, then match to options.\n\n"
        "Question: {question}\n\n"
        "STEP 1 — TRACE YOUR ROUTE chronologically:\n"
        "  a) Starting point: exact location and altitude (frame 1).\n"
        "  b) Each movement segment: turn direction, altitude change, what you fly over/past.\n"
        "  c) Ending point: exact location and altitude (last frame).\n\n"
        "STEP 2 — SUMMARIZE as one sentence:\n"
        "  'Started at [X], flew [direction] over/toward [landmarks], then [turned/ascended/descended],\n"
        "   passed [landmarks], and ended at [Y].'\n\n"
        "STEP 3 — MATCH TO OPTIONS:\n"
        "  Compare your summary against EACH option. Eliminate options that:\n"
        "  - Got the starting point wrong.\n"
        "  - Got the ending point wrong.\n"
        "  - Have turns/altitude changes in the WRONG ORDER.\n"
        "  - Mention landmarks you never passed.\n"
        "  Select the option that matches your summary most accurately.\n\n"
        "Provide your answer as 'Option: [X]' where X is the letter."
    ),

    "Counterfactual": (
        "You are an embodied agent. Consider what would happen if you took a "
        "different action than the one shown in the video.\n\n"
        "Question: {question}\n\n"
        "Think about the spatial consequences of the alternative action: would you "
        "still be aligned with the target? Would the mission still be achievable?\n\n"
        "Provide your answer as 'Option: [X]' where X is the letter."
    ),

    "Duration": (
        "You are comparing the DURATION of two movement segments in a drone flight video.\n\n"
        "Question: {question}\n\n"
        "COMPARISON METHOD:\n"
        "1. Identify WHICH frames belong to segment A and which to segment B.\n"
        "   Look for: when does each segment START (first frame showing that movement)\n"
        "   and END (last frame before the next distinct movement begins).\n"
        "2. Count frames: since frames are evenly spaced in time, MORE frames = LONGER duration.\n"
        "3. Also consider movement complexity: segments with turns, altitude changes,\n"
        "   or maneuvering around obstacles take more time than straight-line flight.\n"
        "4. Compare total frame span × complexity for each segment.\n\n"
        "Provide your answer as 'Option: [X]' where X is the letter."
    ),

    # ═══ Recall / Perception ═══

    "Causal": (
        "Analyze the cause-and-effect relationship shown in the video.\n\n"
        "Question: {question}\n\n"
        "Identify what happened first, what happened as a result, and the causal "
        "chain connecting them.\n\n"
        "Provide your answer as 'Option: [X]' where X is the letter."
    ),

    "Proximity": (
        "Determine which object is closest to a reference point in the scene.\n\n"
        "Question: {question}\n\n"
        "Identify all candidate objects in view and judge which appears nearest to "
        "the reference point based on visual depth cues.\n\n"
        "Provide your answer as 'Option: [X]' where X is the letter."
    ),

    "Scene Recall": (
        "Recall what you saw at a specific moment in the video.\n\n"
        "Question: {question}\n\n"
        "Focus on the relevant part of the video and recall the visual details from "
        "that moment.\n\n"
        "Provide your answer as 'Option: [X]' where X is the letter."
    ),

    "Start/End Position": (
        "Identify your position at the start or end of the video.\n\n"
        "Question: {question}\n\n"
        "Determine whether the question asks about the START (first frames) or END (last frames).\n"
        "For START: focus exclusively on frames 1-3. Identify the exact location, nearby landmarks, "
        "altitude, and what is directly ahead/below/around you.\n"
        "For END: focus exclusively on the last 3 frames. Identify the final location, what you are "
        "facing, and any distinctive structures at that position.\n"
        "Be precise — distinguish between similar-looking locations by noting specific landmarks, "
        "altitude differences, and spatial layout differences.\n\n"
        "Provide your answer as 'Option: [X]' where X is the letter."
    ),

    "Object Recall": (
        "Recall specific objects you saw during navigation.\n\n"
        "Question: {question}\n\n"
        "Identify which moment the question refers to, then recall what objects were "
        "visible at that point.\n\n"
        "Provide your answer as 'Option: [X]' where X is the letter."
    ),

    "Sequence Recall": (
        "You are watching a drone flight video. The question asks what happens NEXT after a specific event, "
        "or what the correct order of events is.\n\n"
        "Question: {question}\n\n"
        "TEMPORAL ANCHORING:\n"
        "1. Scan ALL frames chronologically. Find the EXACT frame range where the described event occurs.\n"
        "   Note the frame numbers. If the event spans multiple frames, note the start and end.\n"
        "2. Look ONLY at frames IMMEDIATELY AFTER that event — these show what happened NEXT.\n"
        "   Do NOT look at frames before the event (they show the past) or far-future frames.\n"
        "3. Describe what you see in those immediately-following frames.\n"
        "4. Match your observation against the options.\n\n"
        "CRITICAL: 'Next step' means what happens RIGHT AFTER the described event, not what happens\n"
        "at the end of the video. If the question asks about event order, trace frame-by-frame\n"
        "and note which object/action appears first, second, third.\n\n"
        "Provide your final answer as 'Option: [X]' where X is the letter."
    ),
}

DEFAULT_PROMPT = (
    "Please answer the following multiple-choice question based on the video.\n\n"
    "Question: {question}\n\n"
    "Think step by step, then provide your final choice as 'Option: [X]' "
    "where X is the option letter."
)

def build_prompt_text(question: str, question_category: str) -> str:
    template = CATEGORY_PROMPTS.get(question_category, DEFAULT_PROMPT)
    return template.format(question=question)


def _uniform_frames(video_path: str, num_frames: int = 16, max_size: int = 768) -> list[str]:
    """Uniform sampling — used only for BASELINE_CATEGORIES."""
    if not video_path:
        return []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []

    step = max(total_frames // num_frames, 1)

    frames_base64 = []
    for i in range(num_frames):
        frame_idx = i * step
        if frame_idx >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
        text = f"Frame {i+1}/{num_frames}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        frames_base64.append(base64.b64encode(buffer).decode('utf-8'))

    cap.release()
    return frames_base64


def reasoner_node(state: GraphState):
    messages = state.get("messages", [])

    if not messages:
        video_path = state.get("video_path")
        question = state.get("question")
        question_category = state.get("question_category", "")

        baseline = is_baseline(question_category)
        num_frames = get_frame_count(question_category)

        if baseline:
            # BASELINE_CATEGORIES: uniform sampling, simple prompt, no memory
            frames = _uniform_frames(video_path, num_frames)
            prompt_text = DEFAULT_PROMPT.format(question=question)
            # Store empty spatial_memory so verifier skips properly
            new_spatial_memory = {}
        else:
            # Spatial / Step-Match: category-specific sampling + Spatial CoT + MemoryBuilder
            frames = video_extract_frames(
                video_path, num_frames, max_size=768,
                question=question, question_category=question_category,
            )

            # Get the frame indices that were sampled (for MemoryBuilder metadata)
            frame_indices = get_sampled_indices(
                video_path, num_frames,
                question=question, question_category=question_category,
            )

            # Build programmatic spatial memory (no LLM call)
            from .frame_selector import CATEGORY_SAMPLING_MODE
            sampling_mode = CATEGORY_SAMPLING_MODE.get(question_category, "uniform")
            spatial_memory = build_memory(
                video_path=video_path,
                frame_indices=frame_indices,
                question_category=question_category,
                sampling_mode=sampling_mode,
                question=question,
            )
            new_spatial_memory = spatial_memory

            prompt_text = build_prompt_text(question, question_category)

            # Inject programmatic spatial memory context
            memory_context = spatial_memory.get("formatted_context", "")
            if memory_context:
                # Add movement timeline for Progress Evaluation
                if question_category == "Progress Evaluation":
                    timeline = format_path_timeline(spatial_memory)
                    if timeline:
                        memory_context += "\n" + timeline + "\n"
                prompt_text = memory_context + "\n" + prompt_text

        content = [{"type": "text", "text": prompt_text}]

        for f in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{f}"}
            })

        initial_msg = HumanMessage(content=content)
        messages_to_send = [initial_msg]
        new_messages = [initial_msg]
    else:
        # Retry: verifier challenge or validator retry — forward existing messages
        messages_to_send = messages
        new_messages = []
        new_spatial_memory = state.get("spatial_memory", {})

    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0,
        timeout=120,
        max_retries=3
    )

    max_custom_retries = 3
    response = None

    for attempt in range(max_custom_retries):
        try:
            print(f"  -> [Reasoner] Sending request to LLM (Model: {MODEL_NAME}), Attempt {attempt + 1}/{max_custom_retries}...")
            response = llm.invoke(messages_to_send)
            print(f"  -> [Reasoner] Received response from LLM.")
            break
        except Exception as e:
            print(f"  -> [Reasoner] API Error on attempt {attempt + 1}: {e}")
            if attempt < max_custom_retries - 1:
                delay = 2 * (2 ** attempt)  # 指数退避: 2s → 4s → 8s
                print(f"  -> [Reasoner] Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"  -> [Reasoner] Failed after {max_custom_retries} attempts.")
                raise e

    new_messages.append(response)

    return {"messages": new_messages, "spatial_memory": new_spatial_memory}

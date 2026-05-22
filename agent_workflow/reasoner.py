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
        "You are an embodied agent following a sequence of navigation instructions in a first-person urban view.\n\n"
        "Question: {question}\n\n"
        "Focus your [Topological Reasoning] on comparing the final frames with the navigation steps: which step did you "
        "just complete before arriving at the final viewpoint? Pay strict attention to turns, altitude changes, and landmarks.\n\n"
        "Progress determination: Compare the visual evidence across the sampled frames with each navigation instruction step. "
        "Determine which step's completion is best supported by the visible scene state, landmarks, and agent position. "
        "Do NOT infer progress from frame position or timestamp alone — the last frame does not necessarily mean the task "
        "is complete. Match against actual visual evidence only.\n\n"
        "Provide your final answer as 'Option: [X]' where X is the letter."
    ),

    "Action Generation": (
        SPATIAL_COT_BASE +
        "You are a drone navigating in a first-person urban view. Determine the SINGLE best next action.\n\n"
        "Question: {question}\n\n"
        "INSTRUCTION AUDIT (MANDATORY — do this BEFORE selecting an answer):\n"
        "1. Parse the navigation instruction into individual steps. List them as Step 1, Step 2, Step 3, etc.\n"
        "2. For EACH step, inspect the video frames and mark it as [DONE] or [PENDING]:\n"
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
        "5. Select the option that EXECUTES the first [PENDING] step.\n\n"
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
        SPATIAL_COT_BASE +
        "You are a drone navigating to a specific destination in an urban environment.\n\n"
        "Question: {question}\n\n"
        "THREE-LEVEL VISIBILITY HIERARCHY — check each level in order:\n"
        "  Level 1: Is the target BUILDING visible in any frame? Identify it by shape, color, height.\n"
        "  Level 2: Is the target FLOOR/LEVEL visible? Look for floor-specific features (balconies, windows, signage).\n"
        "  Level 3: Is the PRECISE target (e.g., a specific balcony or entrance) visually identifiable?\n\n"
        "In your [Visual Anchors], scan each frame systematically using this hierarchy. "
        "In your [Coordinate Mapping], locate the target at the highest level you can confirm. "
        "If Level 3 is not reachable, base your answer on the highest confirmed level.\n\n"
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
        SPATIAL_COT_BASE +
        "You are a drone navigating toward a specific target in an urban environment.\n\n"
        "Question: {question}\n\n"
        "SPATIAL ASSOCIATION protocol — when the target itself is not directly visible, you must identify "
        "which VISIBLE object serves as the best spatial proxy or intermediate waypoint.\n\n"
        "In your [Visual Anchors]: enumerate ALL distinctive visible objects (buildings, roads, landmarks).\n"
        "In your [Coordinate Mapping]: place each object relative to your current viewpoint.\n"
        "In your [Topological Reasoning]:\n"
        "1. Which visible object is physically CLOSEST to the target's known or inferred position?\n"
        "2. Which visible object shares the same spatial context (same building, same side of street, same altitude)?\n"
        "3. Which object, if you approach it, best positions you for the final leg to the target?\n"
        "Select the object with the strongest spatial association to the target.\n\n"
        "Provide your final answer as 'Option: [X]' where X is the letter."
    ),


    # ═══ Recall & Perception ═══

    "Trajectory Captioning": (
        "You are a drone. Summarize your complete movement route from the video.\n\n"
        "Question: {question}\n\n"
        "Trace your FULL 3D trajectory in chronological order:\n"
        "1. Starting point: exact location and altitude at the first frame.\n"
        "2. Movement path: every turn (left/right), altitude change (rise/descend/flat), and direction change.\n"
        "3. Ending point: exact location and altitude at the last frame.\n"
        "4. Overall pattern: describe the complete route as a sequence (e.g., 'flew forward over X, "
        "turned right toward Y, descended to Z').\n\n"
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
        "You are an embodied agent. Compare how long different movement segments take.\n\n"
        "Question: {question}\n\n"
        "Visually estimate the distance and altitude change for each segment being "
        "compared. Which one covers more ground or involves more complex maneuvering?\n\n"
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
        "Recall the order of events or objects you encountered during navigation.\n\n"
        "Question: {question}\n\n"
        "Trace through the video in chronological order to determine the correct "
        "sequence.\n\n"
        "Provide your answer as 'Option: [X]' where X is the letter."
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
                delay = 3 * (2 ** attempt)  # 指数退避: 3s → 6s → 12s
                print(f"  -> [Reasoner] Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"  -> [Reasoner] Failed after {max_custom_retries} attempts.")
                raise e

    new_messages.append(response)

    return {"messages": new_messages, "spatial_memory": new_spatial_memory}

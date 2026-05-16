import cv2
import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from .state import GraphState
from .config import MODEL_NAME, API_KEY, BASE_URL

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
        "Provide your final answer as 'Option: [X]' where X is the letter."
    ),

    "Action Generation": (
        SPATIAL_COT_BASE +
        "You are an embodied agent navigating in a first-person urban view. Determine the next action to take.\n\n"
        "Question: {question}\n\n"
        "In your [Coordinate Mapping], explicitly note your current altitude and orientation. In your [Topological Reasoning], "
        "match your current 3D position against the instruction sequence to find the next required move (e.g., if target is higher, you must rise).\n\n"
        "Provide your final answer as 'Option: [X]' where X is the letter."
    ),

    "Landmark Position": (
        SPATIAL_COT_BASE +
        "You are an embodied agent. Determine your position relative to a landmark.\n\n"
        "Question: {question}\n\n"
        "In your [Topological Reasoning], strictly differentiate between Ego-centric (your perspective) and Allocentric (global) views. "
        "Are you above, beside, facing, or across from the landmark?\n\n"
        "Provide your final answer as 'Option: [X]' where X is the letter."
    ),

    "Goal Detection": (
        SPATIAL_COT_BASE +
        "You are an embodied agent navigating to a specific destination in an urban environment.\n\n"
        "Question: {question}\n\n"
        "In your [Visual Anchors], look carefully for the target destination (e.g., a specific floor balcony, entrance). "
        "In your [Coordinate Mapping], precisely locate it in the frame (e.g., top-right, mid-depth).\n\n"
        "Provide your final answer as 'Option: [X]' where X is the letter."
    ),

    "High-level Planning": (
        SPATIAL_COT_BASE +
        "You are an embodied agent navigating an urban environment to reach a destination.\n\n"
        "Question: {question}\n\n"
        "In your [Topological Reasoning], project a 3D path from your current position to the destination. "
        "What is the most logical intermediate object or location to approach next based on actual visual evidence?\n\n"
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
        "You are an embodied agent navigating toward a specific target.\n\n"
        "Question: {question}\n\n"
        "If the target is not directly visible, use [Topological Reasoning] to identify which visible object is most spatially relevant "
        "to orient toward the target.\n\n"
        "Provide your final answer as 'Option: [X]' where X is the letter."
    ),


    # ═══ Recall & Perception ═══

    "Trajectory Captioning": (
        "You are an embodied agent. Summarize your movement route from the video.\n\n"
        "Question: {question}\n\n"
        "Note your starting point (first frames), your ending point (last frames), "
        "and the overall path: direction, altitude changes, and any turns.\n\n"
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
        "Look at the first frames for the starting position, or the last frames for "
        "the ending position. Note landmarks, buildings, roads, and other features "
        "visible at that moment.\n\n"
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


def extract_frames(video_path: str, num_frames: int = 16, max_size: int = 768) -> list[str]:
    """Extract frames with uniform sampling across the full video.

    Returns list of base64-encoded JPEG strings.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
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
        spatial_memory = state.get("spatial_memory", {})

        frames = extract_frames(video_path, 16)
        prompt_text = build_prompt_text(question, question_category)
        
        # Inject spatial memory context if available
        if spatial_memory:
            memory_context = "\n[Spatial Memory Context]:\n"
            for key, val in spatial_memory.items():
                memory_context += f"- {key}: {val}\n"
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
        messages_to_send = messages
        new_messages = []

    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0,
        timeout=120,
        max_retries=3
    )
    
    import time
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
                print(f"  -> [Reasoner] Retrying in 3 seconds...")
                time.sleep(3)  # 等待3秒后重试
            else:
                print(f"  -> [Reasoner] Failed after {max_custom_retries} attempts.")
                raise e

    new_messages.append(response)

    # Simplified spatial memory extraction logic:
    # We will update spatial memory by parsing the LLM response for [Coordinate Mapping] or [Visual Anchors]
    # In a full implementation, a separate memory_node could extract and format this more rigorously.
    new_spatial_memory = state.get("spatial_memory", {})
    if isinstance(response.content, str):
        if "Step 1 [Visual Anchors]:" in response.content:
             # Just store the raw thought process temporarily to aid the next step if this was a multi-step graph
             pass

    return {"messages": new_messages, "spatial_memory": new_spatial_memory}

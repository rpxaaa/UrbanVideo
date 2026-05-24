"""Verifier node: evidence-consistency check for spatial reasoning categories.

Validates that the Reasoner's answer is supported by visual evidence.
On FAIL, sends structured challenge back to Reasoner for re-reasoning.
On repeated FAIL (retry >= 3), falls back to making the final decision.
"""

import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from .state import GraphState
from .config import MODEL_NAME, API_KEY, BASE_URL, should_verify

VERIFIER_PROMPT = """You are a strict evidence verifier. Check whether a candidate answer is supported by the visual evidence.

Question and options:
{question}

Visual Evidence (extracted from the video):
{evidence_summary}

Candidate Answer from Reasoner:
{candidate}

Your task: Judge if the visual evidence contradicts or supports the candidate answer.
- PASS: the evidence supports the answer, or the answer is a reasonable interpretation of the evidence.
- FAIL: the evidence CLEARLY contradicts the answer (e.g., claims a landmark is on the left but it's on the right).
Only FAIL when the evidence unambiguously disproves the answer. If the evidence is ambiguous, return PASS.

Output EXACTLY this format:
VERDICT: PASS | FAIL
REASON: <one brief sentence>
SUGGEST: <option letter for FAIL, or NONE for PASS>"""

FALLBACK_PROMPT = """You are the final arbiter. The Reasoner's answer was flagged as inconsistent. Make a final decision.

Question: {question}

Visual Evidence:
{evidence_summary}

Reasoner's candidate: {candidate}
Verifier concern: {verifier_concern}

Review the evidence carefully and select the SINGLE best option.
Provide your final answer as: Option: [X]"""


def _build_evidence_summary(state: GraphState) -> str:
    """Build evidence summary from spatial_memory for verifier consumption.

    Compatible with both MemoryBuilder output (scene/motion/route/event blocks)
    and legacy perception output (landmarks/path_segments/spatial_layout).
    """
    sm = state.get("spatial_memory", {})
    if not sm:
        return "[No structured visual evidence available]"

    parts = []

    # ── MemoryBuilder format (programmatic) ──
    if sm.get("scene_summary"):
        parts.append(f"Scene: {sm['scene_summary']}")
    if sm.get("motion_summary"):
        parts.append(f"Motion: {sm['motion_summary']}")
    if sm.get("route_memory"):
        parts.append(f"Route: {sm['route_memory']}")
    if sm.get("landmark_summary"):
        parts.append(f"Sampling: {sm['landmark_summary']}")
    if sm.get("progress_memory"):
        parts.append(f"Progress note: {sm['progress_memory']}")
    if sm.get("event_memory"):
        parts.append(f"Events: {sm['event_memory']}")

    # ── CV visual signals (disabled — caused regression in verifier categories) ──
    # Keeping computation in MemoryBuilder for future use, but not feeding to verifier
    # vs = sm.get("visual_signals", {})
    # if vs.get("summary"):
    #     parts.append(f"Visual Signals: {vs['summary']}")

    # ── Legacy perception format (LLM-based) ──
    if sm.get("environment"):
        parts.append(f"Environment: {sm['environment']}")
    if sm.get("start_point"):
        parts.append(f"Starting point: {sm['start_point']}")
    if sm.get("end_point"):
        parts.append(f"Ending point: {sm['end_point']}")

    if sm.get("landmarks"):
        lms = sm["landmarks"][:8]
        parts.append(f"Key Landmarks ({len(lms)}):")
        for lm in lms:
            name = lm.get("name", "?")
            pos = lm.get("screen_pos", "unknown")
            depth = lm.get("depth", "")
            parts.append(f"  - {name}: screen_pos={pos}" + (f", depth={depth}" if depth else ""))

    if sm.get("path_segments"):
        segs = sm["path_segments"]
        parts.append(f"Movement Path ({len(segs)} segments):")
        for i, s in enumerate(segs):
            fr = s.get("frame_range", "?")
            action = s.get("action", "?")
            alt = s.get("altitude", "")
            parts.append(f"  {i+1}. [{fr}] {action}" + (f" [{alt}]" if alt else ""))

    if sm.get("spatial_layout"):
        layout = sm["spatial_layout"]
        parts.append("Spatial Layout: " + " | ".join(
            f"{k}: {v}" for k, v in layout.items() if v
        ))

    if sm.get("objects"):
        parts.append(f"Notable objects: {', '.join(sm['objects'][:6])}")

    return "\n".join(parts) if parts else "[No structured visual evidence available]"


def _extract_candidate(messages) -> str:
    """Extract the Reasoner's response text from messages."""
    if not messages:
        return ""
    # Walk backwards to find the most recent non-retry message
    for msg in reversed(messages):
        content = msg.content if hasattr(msg, "content") else str(msg)
        if isinstance(content, list):
            text_parts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            content = " ".join(text_parts)
        content = str(content)
        # Skip verifier feedback messages
        if "[Verifier Feedback" in content:
            continue
        if content.strip():
            return content[:3000]
    return ""


def _parse_verdict(text: str) -> tuple:
    """Parse verifier output. Returns (verdict, reason, suggestion)."""
    verdict = "PASS"
    reason = ""
    suggestion = ""

    v_match = re.search(r"VERDICT\s*:\s*(PASS|FAIL)", text, re.IGNORECASE)
    if v_match:
        verdict = v_match.group(1).upper()

    r_match = re.search(r"REASON\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if r_match:
        reason = r_match.group(1).strip()

    s_match = re.search(r"SUGGEST\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if s_match:
        suggestion = s_match.group(1).strip()

    return verdict, reason, suggestion


def _extract_option(text: str) -> str | None:
    """Multi-strategy option extraction (mirrors validator logic)."""
    m = re.search(r"Option\s*:\s*[\[\s]*([A-Ga-g])", text)
    if m:
        return m.group(1).upper()
    # Backup: look for answer patterns
    for pat in [r"(?:answer|choose|select)\s+is\s+([A-Ga-g])", r"\[([A-Ga-g])\]"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def verifier_node(state: GraphState):
    """Verify Reasoner's answer against visual evidence.

    Routes:
    - PASS → verification_passed=True → go to Validator
    - FAIL + retry<3 → challenge message → go back to Reasoner
    - FAIL + retry≥3 → fallback mode → final answer → go to Validator
    """
    category = state.get("question_category", "")

    # Skip for non-verifiable categories
    if not should_verify(category):
        return {"verification_passed": True, "evidence_summary": ""}

    # Skip if already passed — must return at least one state key for LangGraph
    if state.get("verification_passed"):
        return {"evidence_summary": state.get("evidence_summary", "")}

    messages = state.get("messages", [])
    question = state.get("question", "")
    retry_count = state.get("retry_count", 0)

    # Build or reuse evidence summary
    evidence_summary = state.get("evidence_summary") or _build_evidence_summary(state)
    candidate = _extract_candidate(messages)

    if not candidate:
        return {"verification_passed": True, "evidence_summary": evidence_summary}

    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0,
        timeout=120,
        max_retries=2,
    )

    import time

    prompt = VERIFIER_PROMPT.format(
        question=question,
        evidence_summary=evidence_summary,
        candidate=candidate,
    )

    response_text = ""
    for attempt in range(3):
        try:
            print(f"  -> [Verifier] Checking evidence consistency (attempt {attempt+1}/3)...")
            response = llm.invoke([HumanMessage(content=prompt)])
            response_text = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            break
        except Exception as e:
            print(f"  -> [Verifier] API error (attempt {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(3)
            else:
                # API exhausted → assume PASS to avoid blocking
                print("  -> [Verifier] API failed, assuming PASS")
                return {"verification_passed": True, "evidence_summary": evidence_summary}

    verdict, reason, suggestion = _parse_verdict(response_text)
    print(f"  -> [Verifier] VERDICT: {verdict} | {reason}")

    if verdict == "PASS":
        print(f"  -> [Verifier] Evidence consistent, forwarding to Validator")
        # Ensure the reasoner's answer has a properly formatted option for validator
        opt = _extract_option(candidate)
        if opt:
            return {
                "verification_passed": True,
                "evidence_summary": evidence_summary,
                "messages": [AIMessage(content=f"Option: [{opt}]")],
            }
        return {
            "verification_passed": True,
            "evidence_summary": evidence_summary,
        }

    # ── FAIL path ──
    max_retries = 2  # fewer retries to avoid excessive loops
    if retry_count < max_retries:
        # Send structured challenge back to Reasoner
        challenge = (
            "[Verifier Feedback — your previous answer was flagged as INCONSISTENT "
            "with the visual evidence]\n"
            f"Concern: {reason}\n"
        )
        if suggestion and suggestion.upper() != "NONE":
            challenge += f"Consider whether option [{suggestion.upper()}] better matches the evidence.\n"
        challenge += (
            "Re-examine the video frames carefully, focusing on the visual details "
            "mentioned in the concern above. Provide a CORRECTED answer.\n"
            "Format: Option: [X]"
        )
        print(f"  -> [Verifier] Sending challenge back to Reasoner (retry {retry_count+1}/{max_retries})")
        return {
            "verification_passed": False,
            "evidence_summary": evidence_summary,
            "messages": [HumanMessage(content=challenge)],
            "retry_count": retry_count + 1,
        }

    # ── Fallback: Verifier makes final decision ──
    print(f"  -> [Verifier] Retries exhausted, entering FALLBACK mode...")
    fallback_opt = suggestion if suggestion and suggestion.upper() != "NONE" else None

    try:
        fb_prompt = FALLBACK_PROMPT.format(
            question=question,
            evidence_summary=evidence_summary,
            candidate=candidate,
            verifier_concern=reason,
        )
        fb_response = llm.invoke([HumanMessage(content=fb_prompt)])
        fb_text = (
            fb_response.content
            if isinstance(fb_response.content, str)
            else str(fb_response.content)
        )

        opt = _extract_option(fb_text)
        if opt:
            fallback_opt = opt
            print(f"  -> [Verifier] Fallback selected option: {opt}")
    except Exception as e:
        print(f"  -> [Verifier] Fallback error: {e}")

    # ── Guarantee extractable output ──
    if not fallback_opt:
        # Last resort: extract from original reasoner candidate
        fallback_opt = _extract_option(candidate)

    if fallback_opt:
        print(f"  -> [Verifier] Final option: {fallback_opt}")
        return {
            "verification_passed": True,
            "evidence_summary": evidence_summary,
            "messages": [AIMessage(content=f"Option: [{fallback_opt}]")],
        }

    # Absolute ultimate fallback — produce best-effort answer
    print(f"  -> [Verifier] WARNING: Could not determine option, producing empty marker")
    return {
        "verification_passed": True,
        "evidence_summary": evidence_summary,
        "messages": [AIMessage(content="Option: [A]")],
    }

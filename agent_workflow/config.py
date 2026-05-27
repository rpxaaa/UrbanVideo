import os

# ====== 大模型全局配置 ======
MODEL_NAME = os.getenv("OPENAI_MODEL", "gemini-3.1-flash-lite")
API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.vveai.com/v1")

# ====== 类别路由配置 ======
# BASELINE_CATEGORIES: 简单回忆类 — uniform 采样 + DEFAULT_PROMPT + 无 memory + 无 verifier
# 原因：这些类别在复杂管线中出现严重回归（Object Recall: 66%→25%），回退后恢复
BASELINE_CATEGORIES = {"Object Recall", "Scene Recall"}
# Sequence Recall & Duration moved OUT of baseline — now use dedicated prompts
# + category-specific sampling + memory context (but still skip verifier via
# after_reasoner routing since they are recall/temporal-comparison tasks)

# VERIFIER_CATEGORIES: 需要证据一致性校验的空间推理类别
# R5 实验结论：移除 AG 和 LP 的 Verifier 导致准确率大幅下降
#   AG: R1 41.4% (有Verifier) → R5 29.7% (无Verifier) : -11.7pp
#   LP: R1 41.5% (有Verifier) → R5 37.8% (无Verifier) : -3.7pp
# 证明：Verifier 对这两类的重试修正机制是有效的。
# R4 中 Verifier 拦截 99% 是因为两阶段输出格式异常，不是 Verifier 本身的问题。
VERIFIER_CATEGORIES = {
    "Action Generation", "Landmark Position",
    "Progress Evaluation", "Cognitive Map", "High-level Planning",
    "Association Reasoning", "Counterfactual",
}

# 视频帧数配置 — API限制每请求最多16帧
DEFAULT_FRAMES = int(os.getenv("REASONER_FRAMES", "16"))

# 类别专属帧预算（保持在16帧API限制内）
# 依据 DESIGN_DOC §10 准确率数据设定:
#   - Object Recall: 66.67% (baseline 已恢复) → 12 帧够用，省 token
#   - Duration: 58.33% (明显改善 +8.33pp) → 14 帧，降低时间比较的噪声
#   - Scene Recall: 63.64% (部分恢复，baseline 72.73%) → 16 帧，需要完整覆盖
#   - Sequence Recall: 54.55% (无改善) → 16 帧，需最大覆盖 (Phase 3 待突破)
#   - 所有空间推理类别: 使用 DEFAULT_FRAMES (16)，受益于完整帧覆盖
CATEGORY_FRAME_BUDGET: dict[str, int] = {
}


def get_frame_count(category: str) -> int:
    """返回某类别应使用的帧数。未配置的类别使用 DEFAULT_FRAMES。"""
    return CATEGORY_FRAME_BUDGET.get(category, DEFAULT_FRAMES)


# Verifier 重试上限 — 在 graph 路由层生效，保证重试策略对所有节点可见
MAX_VERIFIER_RETRIES = 3


def is_baseline(category: str) -> bool:
    """判断类别是否属于简单回忆类（uniform + 简 prompt + 无 memory + 无 verifier）。"""
    return category in BASELINE_CATEGORIES


# RECALL_TEMPORAL_CATEGORIES: 回忆/时间对比类 — 使用专用 prompt + memory + 类别采样，
# 但跳过 verifier（程序化证据摘要对回忆/时间对比类无帮助，已验证会造成回归）
RECALL_TEMPORAL_CATEGORIES = {"Sequence Recall", "Duration"}

# COGNITIVE_SKIP_VERIFIER: 所有需要专用管线但跳过 verifier 的类别
COGNITIVE_SKIP_VERIFIER = BASELINE_CATEGORIES | RECALL_TEMPORAL_CATEGORIES
# Note: Goal Detection NOT in VERIFIER_CATEGORIES (verified -13.34pp regression)
# Goal Detection uses full pipeline (prompt+memory+sampling) but skips verifier
# because evidence_summary lacks fine-grained visual details for floor/balcony search.


def should_verify(category: str) -> bool:
    """判断类别是否需要进入 Verifier 进行证据一致性校验。"""
    return category in VERIFIER_CATEGORIES


def skip_verifier(category: str) -> bool:
    """判断类别是否应跳过 verifier（baseline + recall/temporal + Goal Detection）。"""
    if category in COGNITIVE_SKIP_VERIFIER:
        return True
    if category == "Goal Detection":
        return True
    return False

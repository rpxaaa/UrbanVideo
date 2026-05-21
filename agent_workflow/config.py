import os

# ====== 大模型全局配置 ======
MODEL_NAME = os.getenv("OPENAI_MODEL", "gemini-3.1-flash-lite")
API_KEY = os.getenv("OPENAI_API_KEY", "sk-yMlVtvCwWi1Wt164CbB023D8Db1241E6AaAe82Ba56EeA435")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.vveai.com/v1")

# ====== 类别路由配置 ======
# BASELINE_CATEGORIES: 简单回忆类 — uniform 采样 + DEFAULT_PROMPT + 无 memory + 无 verifier
# 原因：这些类别在复杂管线中出现严重回归（Object Recall: 66%→25%），回退后恢复
BASELINE_CATEGORIES = {"Object Recall", "Scene Recall", "Sequence Recall", "Duration"}

# VERIFIER_CATEGORIES: 需要证据一致性校验的空间推理类别
VERIFIER_CATEGORIES = {
    "Action Generation", "Progress Evaluation", "Landmark Position",
    "Cognitive Map", "High-level Planning",
    "Association Reasoning", "Counterfactual",
}
# Note: Goal Detection removed — Verifier caused -13.34pp regression
# due to evidence_summary lacking fine-grained visual details for floor/balcony search

# 视频帧数配置
REASONER_FRAMES = int(os.getenv("REASONER_FRAMES", "16"))


def is_baseline(category: str) -> bool:
    """判断类别是否属于简单回忆类（uniform + 简 prompt + 无 memory + 无 verifier）。"""
    return category in BASELINE_CATEGORIES


def should_verify(category: str) -> bool:
    """判断类别是否需要进入 Verifier 进行证据一致性校验。"""
    return category in VERIFIER_CATEGORIES

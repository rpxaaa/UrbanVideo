import os

# ====== 大模型全局配置 ======
MODEL_NAME = os.getenv("OPENAI_MODEL", "your_model_name")
API_KEY = os.getenv("OPENAI_API_KEY", "your_api_key")
BASE_URL = os.getenv("OPENAI_BASE_URL", "your_base_url")

# 视频帧数配置（部分模型限制16帧输入）
PERCEPTION_FRAMES = int(os.getenv("PERCEPTION_FRAMES", "16"))
REASONER_FRAMES = int(os.getenv("REASONER_FRAMES", "16"))

import os

# ====== 大模型全局配置 ======
# MODEL_NAME = os.getenv("OPENAI_MODEL", "qwen3.6:35b")
# API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
# BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1/")
# MODEL_NAME = os.getenv("OPENAI_MODEL", "gemini-3.1-flash-lite")
MODEL_NAME = os.getenv("OPENAI_MODEL", "qwen3-vl-flash")
API_KEY = os.getenv("OPENAI_API_KEY", "sk-yMlVtvCwWi1Wt164CbB023D8Db1241E6AaAe82Ba56EeA435")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.vveai.com/v1")

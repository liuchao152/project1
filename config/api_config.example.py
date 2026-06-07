"""
API 配置文件模板 - 复制此文件为 api_config.py 并填入真实密钥

使用方法：
    1. 复制此文件为 api_config.py
    2. 填入真实的 API 密钥
    3. 不要将 api_config.py 提交到 Git！
"""

# ========================
# 阿里云 DashScope 配置（通义千问）
# ========================
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_API_KEY = "sk-YOUR_API_KEY_HERE"  # ← 修改这里
DEFAULT_MODEL = "qwen3-32b"

ALIYUN_CONFIG = {
    "base_url": DASHSCOPE_BASE_URL,
    "api_key": DASHSCOPE_API_KEY,
    "model": DEFAULT_MODEL,
}

# ========================
# 阿里云 Embedding 配置
# ========================
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_API_KEY = "sk-YOUR_EMBEDDING_KEY_HERE"  # ← 修改这里
EMBEDDING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

EMBEDDING_CONFIG = {
    "base_url": EMBEDDING_BASE_URL,
    "api_key": EMBEDDING_API_KEY,
    "model": EMBEDDING_MODEL,
}

# ========================
# 多端口配置（用于负载均衡）
# ========================
DASHSCOPE_BASE_URLS = [
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
]

# ========================
# 并发配置
# ========================
MAX_WORKERS = 48
PER_HOST_LIMIT = 24

# ========================
# 其他配置
# ========================
TEMPERATURE = 0.0
TOP_P = 0.7
MAX_TOKENS = 6144

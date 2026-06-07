"""
API 配置文件 - 统一管理所有 API 密钥和模型配置

使用方法：
    from config.api_config import ALIYUN_CONFIG, EMBEDDING_CONFIG
    
    # 使用阿里云配置
    client = OpenAI(api_key=ALIYUN_CONFIG["api_key"], base_url=ALIYUN_CONFIG["base_url"])
    
    # 或使用独立变量
    from config.api_config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, DEFAULT_MODEL
"""

# ========================
# 阿里云 DashScope 配置（通义千问）
# ========================
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_API_KEY = "sk-43574ab35a0d45f3bf3d5d161cb75748"
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
EMBEDDING_API_KEY = "sk-172c7005eddb4a5aa2e73f5d200d571c"
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

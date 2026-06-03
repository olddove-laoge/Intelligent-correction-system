"""应用配置：从环境变量读取敏感配置，定义路径常量。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── 项目根目录 ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# ── 文件存储 ────────────────────────────────────────────────────────────
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

HISTORY_DIR = BASE_DIR / "history"
HISTORY_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# ── Flask ───────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"

# ── 阿里云 API ─────────────────────────────────────────────────────────
ALIBABA_CLOUD_ACCESS_KEY_ID = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID", "")
ALIBABA_CLOUD_ACCESS_KEY_SECRET = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET", "")
ALIYUN_OCR_ENDPOINT = os.getenv(
    "ALIYUN_OCR_ENDPOINT", "ocr-api.cn-hangzhou.aliyuncs.com"
)

# ── 大模型 API ─────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")  # deepseek / qwen
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_API_ENDPOINT = os.getenv("LLM_API_ENDPOINT", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "10"))  # 秒

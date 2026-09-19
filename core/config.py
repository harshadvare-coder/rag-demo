import os
from dotenv import load_dotenv

load_dotenv()  # must run before reading env vars

# =========================================================
# LLM / Embedding configuration
# =========================================================

EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")

LLM_API_URL = os.getenv("LLM_API_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL_ID = os.getenv(
    "LLM_MODEL_ID",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0"
)

# =========================================================
# Retrieval / generation hyper-parameters
# =========================================================

RETRIEVAL_TOP_K      = int(os.getenv("RETRIEVAL_TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

LLM_TOP_P       = float(os.getenv("LLM_TOP_P", "0.9"))
LLM_TOP_K       = int(os.getenv("LLM_TOP_K", "50"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.5"))
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# =========================================================
# Chat history
# =========================================================

HISTORY_DIR  = "chat_history"
MAX_HISTORY  = int(os.getenv("MAX_HISTORY", "20"))

# Agents that have their own separate history track
AGENT_KEYS = ("orchestrator", "activity", "booking", "general")

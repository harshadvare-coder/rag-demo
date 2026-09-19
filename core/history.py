import json
import os

from core.config import HISTORY_DIR, MAX_HISTORY, AGENT_KEYS

os.makedirs(HISTORY_DIR, exist_ok=True)


# =========================================================
# Chat History Store
# =========================================================

def session_dir(session_id: str) -> str:
    """Return the folder path for a session's history files."""
    return os.path.join(HISTORY_DIR, session_id)


def history_path(session_id: str, agent: str) -> str:
    """Return the file path for a specific agent's history within a session."""
    return os.path.join(session_dir(session_id), f"{agent}.json")


def load_history(session_id: str, agent: str) -> list:
    """Load history for a specific agent in a session."""
    path = history_path(session_id, agent)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(session_id: str, agent: str, history: list):
    """Save (and trim) history for a specific agent in a session."""
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    os.makedirs(session_dir(session_id), exist_ok=True)
    with open(history_path(session_id, agent), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def append_and_save(session_id: str, agent: str, role: str, content: str):
    """Append a single message to an agent's history and persist it."""
    history = load_history(session_id, agent)
    history.append({"role": role, "content": content})
    save_history(session_id, agent, history)


def load_session_histories(session_id: str) -> dict:
    """Load all per-agent histories for a session in one call."""
    histories = {key: load_history(session_id, key) for key in AGENT_KEYS}
    print(
        f"\nSession: {session_id} | "
        + " ".join(f"{k}={len(v)}" for k, v in histories.items())
    )
    return histories

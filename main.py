import json
import os
import uuid
from typing import Generator, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents import activity_agent, hotel_agent, orchestrator

load_dotenv()


# =========================================================
# Configuration
# =========================================================

EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")

LLM_API_URL = os.getenv("LLM_API_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")

LLM_MODEL_ID = os.getenv(
    "LLM_MODEL_ID",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0"
)

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
LLM_TOP_K = int(os.getenv("LLM_TOP_K", "50"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.5"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))


# =========================================================
# FastAPI
# =========================================================

app = FastAPI()


# =========================================================
# Chat History Store
# =========================================================

HISTORY_DIR = "chat_history"
os.makedirs(HISTORY_DIR, exist_ok=True)

MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))

# Agents that have their own separate history track
AGENT_KEYS = ("orchestrator", "activity", "hotel", "general")


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


# =========================================================
# Embedding API
# =========================================================

def get_embedding(text: str):
    headers = {
        "api-key": EMBEDDING_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {"text": text}

    response = httpx.post(
        EMBEDDING_API_URL,
        headers=headers,
        json=payload,
        timeout=60
    )
    response.raise_for_status()
    data = response.json()
    print("Embedding API response:", data)
    return data["embedding"]


# =========================================================
# Startup — vectorize both agents
# =========================================================

@app.on_event("startup")
def startup():
    print("Starting application...")
    activity_agent.vectorize_faqs(get_embedding)
    hotel_agent.vectorize_faqs(get_embedding)
    print("Application is ready.")


# =========================================================
# Request Model
# =========================================================

class ChatRequest(BaseModel):
    question: str


# =========================================================
# Chat
# =========================================================

@app.post("/chat")
def chat(
    request: ChatRequest,
    x_session_id: Optional[str] = Header(default=None)
):
    question = request.question.strip()

    if not question:
        return {"error": "Question cannot be empty"}

    session_id = x_session_id or str(uuid.uuid4())

    # Load per-agent histories
    orchestrator_history = load_history(session_id, "orchestrator")
    activity_history     = load_history(session_id, "activity")
    hotel_history        = load_history(session_id, "hotel")
    general_history      = load_history(session_id, "general")

    print(
        f"\nSession: {session_id} | "
        f"orchestrator={len(orchestrator_history)} "
        f"activity={len(activity_history)} "
        f"hotel={len(hotel_history)} "
        f"general={len(general_history)} | "
        f"Question: {question}"
    )

    # ----------------------------------------------------------
    # 1. Orchestrator — classify and route (with its own history)
    #    The orchestrator only builds the payload and parses the
    #    response; the actual HTTP call is made here.
    # ----------------------------------------------------------

    routing_payload = orchestrator.build_payload(
        question=question,
        history=orchestrator_history,
        llm_model_id=LLM_MODEL_ID,
    )

    routing_headers = {
        "api-key": LLM_API_KEY,
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Accept": "text/event-stream",
        "Content-Type": "application/json"
    }

    route_raw = ""
    try:
        with httpx.stream(
            "POST",
            LLM_API_URL,
            headers=routing_headers,
            json=routing_payload,
            timeout=30
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                raw = line[len("data:"):].strip()
                if raw == "[DONE]":
                    break
                try:
                    event_data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                delta = None
                if event_data.get("type") == "content_block_delta":
                    delta = event_data.get("delta", {}).get("text", "")
                elif "text" in event_data:
                    delta = event_data["text"]
                elif "content" in event_data:
                    delta = event_data["content"]
                elif "message" in event_data:
                    delta = event_data["message"]
                if delta:
                    route_raw += delta
    except Exception as e:
        print(f"[Orchestrator] API call failed: {e}. Defaulting to 'general'.")

    route = orchestrator.parse_route(route_raw) if route_raw else "general"

    # Save orchestrator turn
    orchestrator_history.append({"role": "user",      "content": question})
    orchestrator_history.append({"role": "assistant",  "content": route})
    save_history(session_id, "orchestrator", orchestrator_history)

    print(f"[Orchestrator] Route selected: {route}")

    # ----------------------------------------------------------
    # 2. Select agent and its history based on route
    # ----------------------------------------------------------

    if route == "activity":
        agent         = activity_agent
        agent_history = activity_history
        agent_key     = "activity"
    elif route == "hotel":
        agent         = hotel_agent
        agent_history = hotel_history
        agent_key     = "hotel"
    else:
        # General / conversational — no RAG
        agent         = None
        agent_history = general_history
        agent_key     = "general"

    # ----------------------------------------------------------
    # 3. Embed question and retrieve from the selected agent
    # ----------------------------------------------------------

    query_embedding = get_embedding(question)

    documents, metadatas, ids = [], [], []

    if agent is not None:
        result = agent.retrieve(
            query_embedding=query_embedding,
            top_k=RETRIEVAL_TOP_K,
            threshold=SIMILARITY_THRESHOLD
        )
        documents = result["documents"]
        metadatas = result["metadatas"]
        ids = result["ids"]

    # ----------------------------------------------------------
    # 4. Build context message for LLM
    # ----------------------------------------------------------

    if documents:
        context = ""
        for i, document in enumerate(documents):
            context += f"\nFAQ {i + 1}:\n{document}\n"
        current_message = (
            f"FAQ CONTEXT:\n{context}\n"
            f"USER QUESTION:\n{question}\n"
            f"Answer the user using only the FAQ context."
        )
    else:
        current_message = (
            f"There is no FAQ context available for this question.\n"
            f"Answer using the conversation history above.\n"
            f"USER QUESTION:\n{question}"
        )

    # ----------------------------------------------------------
    # 5. Pick system prompt from selected agent (or default)
    # ----------------------------------------------------------

    if agent is not None:
        system_prompt = agent.get_system_prompt()
    else:
        with open("prompts/general_prompt.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()

    # ----------------------------------------------------------
    # 6. Sources payload for frontend
    # ----------------------------------------------------------

    sources_payload = json.dumps([
        {
            "id": ids[i],
            "question": metadatas[i]["question"],
            "category": metadatas[i]["category"]
        }
        for i in range(len(documents))
    ])

    # Add user question to agent history
    agent_history.append({"role": "user", "content": question})
    if len(agent_history) > MAX_HISTORY:
        agent_history = agent_history[-MAX_HISTORY:]
    save_history(session_id, agent_key, agent_history)

    # ----------------------------------------------------------
    # 7. Stream LLM response
    # ----------------------------------------------------------

    def stream_llm() -> Generator[str, None, None]:

        yield f"event: session\ndata: {json.dumps({'session_id': session_id})}\n\n"
        yield f"event: route\ndata: {json.dumps({'agent': route})}\n\n"
        yield f"event: sources\ndata: {sources_payload}\n\n"

        headers = {
            "api-key": LLM_API_KEY,
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Accept": "text/event-stream",
            "Content-Type": "application/json"
        }

        # Build messages from this agent's history (all turns except the
        # current user turn, which we append as current_message with context)
        messages = []
        for msg in agent_history[:-1]:
            messages.append({
                "role": msg["role"],
                "content": [{"type": "text", "text": msg["content"]}]
            })
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": current_message}]
        })

        print(f"\n[LLM] Sending {len(messages)} messages to LLM (agent={agent_key}):")
        for m in messages:
            preview = m["content"][0]["text"][:80].replace("\n", " ")
            print(f"  [{m['role']}] {preview}...")

        payload = {
            "model_id": LLM_MODEL_ID,
            "body": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": LLM_MAX_TOKENS,
                "temperature": LLM_TEMPERATURE,
                "top_k": LLM_TOP_K,
                "system": system_prompt,
                "messages": messages
            }
        }

        full_answer = ""

        with httpx.stream(
            "POST",
            LLM_API_URL,
            headers=headers,
            json=payload,
            timeout=120
        ) as response:

            if response.status_code != 200:
                error_body = response.read().decode("utf-8", errors="replace")
                print(f"LLM API error {response.status_code}: {error_body}")
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                if line.startswith("data:"):
                    raw = line[len("data:"):].strip()

                    if raw == "[DONE]":
                        break

                    try:
                        event_data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    delta_text = None
                    event_type = event_data.get("type", "")

                    if event_type == "content_block_delta":
                        delta_text = event_data.get("delta", {}).get("text", "")
                    elif "text" in event_data:
                        delta_text = event_data["text"]
                    elif "content" in event_data:
                        delta_text = event_data["content"]
                    elif "message" in event_data:
                        delta_text = event_data["message"]

                    if delta_text:
                        full_answer += delta_text
                        yield f"event: answer\ndata: {json.dumps({'text': delta_text})}\n\n"

        # Save assistant reply to this agent's history
        agent_history.append({"role": "assistant", "content": full_answer})
        save_history(session_id, agent_key, agent_history)
        print(f"[HISTORY] Saved agent='{agent_key}'. Total messages: {len(agent_history)}")

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        stream_llm(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


# =========================================================
# History
# =========================================================

@app.get("/history/{session_id}")
def get_history(session_id: str):
    """Return all per-agent histories for a session."""
    result = {}
    for key in AGENT_KEYS:
        h = load_history(session_id, key)
        result[key] = {
            "message_count": len(h),
            "history": h
        }
    return {"session_id": session_id, "agents": result}


@app.get("/history/{session_id}/{agent}")
def get_agent_history(session_id: str, agent: str):
    """Return history for a specific agent within a session."""
    if agent not in AGENT_KEYS:
        return {"error": f"Unknown agent '{agent}'. Valid: {AGENT_KEYS}"}
    h = load_history(session_id, agent)
    return {"session_id": session_id, "agent": agent, "message_count": len(h), "history": h}


@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    """Delete the entire session folder and all agent history files inside it."""
    import shutil
    path = session_dir(session_id)
    if os.path.exists(path):
        shutil.rmtree(path)
        return {"session_id": session_id, "cleared": True}
    return {"session_id": session_id, "cleared": False}


@app.delete("/history/{session_id}/{agent}")
def clear_agent_history(session_id: str, agent: str):
    """Delete history for a specific agent within a session."""
    if agent not in AGENT_KEYS:
        return {"error": f"Unknown agent '{agent}'. Valid: {AGENT_KEYS}"}
    path = history_path(session_id, agent)
    if os.path.exists(path):
        os.remove(path)
        return {"session_id": session_id, "agent": agent, "cleared": True}
    return {"session_id": session_id, "agent": agent, "cleared": False}


# =========================================================
# Health
# =========================================================

@app.get("/health")
def health():
    # Each session is a subfolder inside HISTORY_DIR
    unique_sessions = [
        d for d in os.listdir(HISTORY_DIR)
        if os.path.isdir(os.path.join(HISTORY_DIR, d))
    ]
    return {
        "status": "ok",
        "activity_faq_count": activity_agent.get_collection().count(),
        "hotel_faq_count": hotel_agent.get_collection().count(),
        "saved_sessions": len(unique_sessions)
    }

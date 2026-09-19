import json
import os
import uuid
from typing import Optional

from fastapi import FastAPI, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents.activity_agent import ActivityAgent
from agents.general_agent import GeneralAgent
from agents.hotel_booking_agent import HotelBookingAgent
from core.config import LLM_API_URL, LLM_API_KEY, LLM_MODEL_ID, HISTORY_DIR, AGENT_KEYS
from core.embedding import get_embedding
from core.history import load_history, history_path, session_dir
from agents.orchestrator import Orchestrator
from core.startup import run_vectorization

# =========================================================
# Agent singletons
# =========================================================

activity_agent      = ActivityAgent()
general_agent       = GeneralAgent()
hotel_booking_agent = HotelBookingAgent(
    llm_api_url=LLM_API_URL,
    llm_api_key=LLM_API_KEY,
    llm_model_id=LLM_MODEL_ID,
)
orchestrator = Orchestrator(
    llm_api_url=LLM_API_URL,
    llm_api_key=LLM_API_KEY,
    llm_model_id=LLM_MODEL_ID,
)

# =========================================================
# FastAPI app
# =========================================================

app = FastAPI()


@app.on_event("startup")
def startup():
    run_vectorization(
        agents_to_vectorize=[
            (activity_agent,      "ActivityAgent"),
            (hotel_booking_agent, "HotelBookingAgent"),
        ],
        get_embedding_fn=get_embedding,
    )
    print("[Startup] Application is ready. Vectorization running in background.")


# =========================================================
# Request model
# =========================================================

class ChatRequest(BaseModel):
    question: str


# =========================================================
# Routes
# =========================================================

@app.post("/chat")
def chat(
    request: ChatRequest,
    x_session_id: Optional[str] = Header(default=None),
):
    question = request.question.strip()
    if not question:
        return {"error": "Question cannot be empty"}

    session_id = x_session_id or str(uuid.uuid4())

    # Orchestrator: User Query + History → LLM → Agent Name
    agent = orchestrator.classify(question, session_id)

    def _with_preamble(agent_name: str, inner_gen):
        """Prepend session + agent SSE events before the agent's own stream."""
        yield f"event: session\ndata: {json.dumps({'session_id': session_id})}\n\n"
        yield f"event: agent\ndata: {json.dumps({'agent': agent_name})}\n\n"
        yield from inner_gen

    if agent == "HotelBookingAgent":
        generator = _with_preamble(
            "HotelBookingAgent",
            hotel_booking_agent.stream_turn(
                session_id=session_id,
                user_message=question,
                get_embedding_fn=get_embedding,
            ),
        )
    elif agent == "ActivityAgent":
        generator = _with_preamble(
            "ActivityAgent",
            activity_agent.stream_response(session_id, question),
        )
    else:
        generator = _with_preamble(
            "GeneralAgent",
            general_agent.stream_response(session_id, question),
        )

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/booking/{session_id}")
def get_booking_state(session_id: str):
    """Return the current hotel booking state for a session."""
    state = hotel_booking_agent.load_booking_state(session_id)
    return {"session_id": session_id, "booking": state}


@app.delete("/booking/{session_id}")
def cancel_booking(session_id: str):
    """Clear / cancel the in-progress booking for a session and its chat history."""
    hotel_booking_agent.clear_booking_state(session_id)
    path = history_path(session_id, "booking")
    if os.path.exists(path):
        os.remove(path)
    return {"session_id": session_id, "booking_cleared": True}


@app.get("/history/{session_id}")
def get_history(session_id: str):
    """Return all per-agent histories for a session."""
    result = {}
    for key in AGENT_KEYS:
        h = load_history(session_id, key)
        result[key] = {"message_count": len(h), "history": h}
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


@app.get("/health")
def health():
    unique_sessions = [
        d for d in os.listdir(HISTORY_DIR)
        if os.path.isdir(os.path.join(HISTORY_DIR, d))
    ]
    return {
        "status": "ok",
        "activity_faq_count":      activity_agent.get_collection().count(),
        "hotel_booking_faq_count": hotel_booking_agent.get_collection().count(),
        "saved_sessions":          len(unique_sessions),
    }

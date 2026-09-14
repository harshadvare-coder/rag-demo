# from fastapi import FastAPI

# app = FastAPI()

# @app.get("/")
# def home():
#     return {"message":"RAG API is Running"}



import json
import os
import re
import uuid
from typing import Generator, Optional

import chromadb
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


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

# Top-K: number of FAQ chunks to retrieve from ChromaDB
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))

# Similarity threshold: discard chunks with distance above this
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

# LLM sampling params
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

# Max number of messages to keep per session
# (1 exchange = 1 user + 1 assistant = 2 messages)
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))


def history_path(session_id: str) -> str:
    """Return the file path for a session's history."""
    return os.path.join(HISTORY_DIR, f"{session_id}.json")


def load_history(session_id: str) -> list:
    """Load history from disk. Returns empty list if not found."""
    path = history_path(session_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(session_id: str, history: list):
    """Persist history to disk, trimmed to MAX_HISTORY messages."""
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    with open(history_path(session_id), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# =========================================================
# ChromaDB
# =========================================================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="faq_collection"
)


# =========================================================
# Read System Prompt
# =========================================================

with open("system_prompt.txt", "r", encoding="utf-8") as file:
    SYSTEM_PROMPT = file.read()


# =========================================================
# Embedding API
# =========================================================

def get_embedding(text: str):

    headers = {
        "api-key": EMBEDDING_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "text": text
    }

    response = httpx.post(
        EMBEDDING_API_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    print("Embedding API response:")
    print(data)

    # Change this line if your wrapper returns
    # the vector under a different JSON field.
    return data["embedding"]


# =========================================================
# Read FAQ
# =========================================================

def load_faqs():

    with open("faq.txt", "r", encoding="utf-8") as file:
        content = file.read()

    # Split into blocks by the "# id:" header
    block_pattern = re.compile(
        r"# id:\s*(.*?)\s*\|\s*category:\s*(.*?)\n(.*?)(?=\n# id:|\Z)",
        re.DOTALL
    )

    # Each Q/A pair within a block
    qa_pattern = re.compile(
        r"Q:\s*(.*?)\nA:\s*(.*?)(?=\nQ:|\Z)",
        re.DOTALL
    )

    faqs = []

    for block_match in block_pattern.finditer(content):

        block_id = block_match.group(1).strip()
        category = block_match.group(2).strip()
        block_body = block_match.group(3)

        qa_pairs = qa_pattern.findall(block_body)

        for index, (question, answer) in enumerate(qa_pairs):

            # Generate a unique ID per Q&A pair within the block
            unique_id = (
                f"{block_id}_{index + 1}"
                if len(qa_pairs) > 1
                else block_id
            )

            faqs.append({
                "id": unique_id,
                "category": category,
                "question": question.strip(),
                "answer": answer.strip()
            })

    return faqs


# =========================================================
# Vectorize FAQ and Store in ChromaDB
# =========================================================

def vectorize_faqs():

    faqs = load_faqs()

    print(f"Found {len(faqs)} FAQs")

    # Delete all existing entries so stale FAQs
    # (removed from faq.txt) don't linger in ChromaDB.
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)
        print(f"Cleared {len(existing_ids)} stale records from ChromaDB.")

    for index, faq in enumerate(faqs):

        text = (
            f"Question: {faq['question']}\n"
            f"Answer: {faq['answer']}"
        )

        print(
            f"Creating embedding for: "
            f"{faq['question']}"
        )

        embedding = get_embedding(text)

        collection.upsert(
            ids=[faq["id"]],
            documents=[text],
            embeddings=[embedding],
            metadatas=[{
                "faq_id": faq["id"],
                "category": faq["category"],
                "question": faq["question"]
            }]
        )

    print("FAQ vectorization completed.")
    print(
        f"Total records in ChromaDB: "
        f"{collection.count()}"
    )


# =========================================================
# Startup
# =========================================================

@app.on_event("startup")
def startup():

    print("Starting application...")

    vectorize_faqs()

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

    # Get or create session from X-Session-ID header
    session_id = x_session_id or str(uuid.uuid4())

    # Load history from disk
    history = load_history(session_id)

    print(f"Session: {session_id} | History: {len(history)} messages | Question: {question}")

    # -----------------------------------------------------
    # 1. Vectorize user question
    # -----------------------------------------------------

    query_embedding = get_embedding(question)

    # -----------------------------------------------------
    # 2. Search ChromaDB with similarity threshold
    # -----------------------------------------------------

    total_docs = collection.count()

    documents = []
    metadatas = []
    ids = []

    if total_docs > 0:

        n_results = min(RETRIEVAL_TOP_K, total_docs)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        raw_documents = results["documents"][0]
        raw_metadatas = results["metadatas"][0]
        raw_ids = results["ids"][0]
        raw_distances = results["distances"][0]

        print("\nAll retrieved chunks with distances:")
        for i, dist in enumerate(raw_distances):
            status = "✓ kept" if dist <= SIMILARITY_THRESHOLD else "✗ dropped"
            print(f"  [{raw_ids[i]}] distance={dist:.4f}  {status}")

        # Keep only results below the threshold
        for i, distance in enumerate(raw_distances):
            if distance <= SIMILARITY_THRESHOLD:
                documents.append(raw_documents[i])
                metadatas.append(raw_metadatas[i])
                ids.append(raw_ids[i])

        print(
            f"\nRelevant chunks after threshold "
            f"({SIMILARITY_THRESHOLD}): {len(documents)}"
        )

    # -----------------------------------------------------
    # 3. Build Context
    # -----------------------------------------------------

    if documents:
        context = ""
        for i, document in enumerate(documents):
            context += f"\nFAQ {i + 1}:\n{document}\n"
        # Inject FAQ context into the current user message
        current_message = (
            f"FAQ CONTEXT:\n{context}\n"
            f"USER QUESTION:\n{question}\n"
            f"Answer the user using only the FAQ context."
        )
    else:
        # No FAQ match — use conversation history only
        current_message = (
            f"There is no FAQ context available for this question.\n"
            f"Answer using the conversation history above.\n"
            f"USER QUESTION:\n{question}"
        )

    # -----------------------------------------------------
    # 5. Stream LLM response via SSE
    # -----------------------------------------------------

    # Clean sources list — only what the frontend needs
    sources_payload = json.dumps([
        {
            "id": ids[i],
            "question": metadatas[i]["question"],
            "category": metadatas[i]["category"]
        }
        for i in range(len(documents))
    ])

    # Add plain question to history (not the prompt with FAQ context)
    history.append({"role": "user", "content": question})

    # Trim and save to disk
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    save_history(session_id, history)

    def stream_llm() -> Generator[str, None, None]:

        yield f"event: session\ndata: {json.dumps({'session_id': session_id})}\n\n"
        yield f"event: sources\ndata: {sources_payload}\n\n"

        headers = {
            "api-key": LLM_API_KEY,
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Accept": "text/event-stream",
            "Content-Type": "application/json"
        }

        # Build messages from history:
        # - All previous messages use plain content
        # - Current (last) user message uses current_message
        #   which has FAQ context injected if available
        current_history = history

        messages = []
        for msg in current_history[:-1]:  # all except current question
            messages.append({
                "role": msg["role"],
                "content": [{"type": "text", "text": msg["content"]}]
            })

        # Current turn with FAQ context injected
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": current_message}]
        })

        print(f"\n[HISTORY] Sending {len(messages)} messages to LLM:")
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
                # System prompt goes here — separate from messages
                # so it doesn't pollute conversation history
                "system": SYSTEM_PROMPT,
                "messages": messages
            }
        }

        # Accumulate full answer to save in history after streaming
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

        # Save assistant reply to disk
        history.append({"role": "assistant", "content": full_answer})
        save_history(session_id, history)
        print(f"[HISTORY] Saved. Total messages: {len(history)}")

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
    """Return the chat history for a session."""
    history = load_history(session_id)
    return {
        "session_id": session_id,
        "message_count": len(history),
        "history": history
    }


@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    """Clear the chat history for a session."""
    path = history_path(session_id)
    if os.path.exists(path):
        os.remove(path)
    return {"session_id": session_id, "cleared": True}


# =========================================================
# Health
# =========================================================

@app.get("/health")
def health():
    session_files = [
        f for f in os.listdir(HISTORY_DIR)
        if f.endswith(".json")
    ]
    return {
        "status": "ok",
        "faq_count": collection.count(),
        "saved_sessions": len(session_files)
    }


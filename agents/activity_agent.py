import json
import re
import httpx
import chromadb

from core.config import (
    LLM_API_URL,
    LLM_API_KEY,
    LLM_MODEL_ID,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TOP_K,
    RETRIEVAL_TOP_K,
    SIMILARITY_THRESHOLD,
)
from core.embedding import get_embedding
from core.history import load_history, save_history


# =========================================================
# Activity Agent
# Owns the activity_faq.txt knowledge base
# =========================================================


class ActivityAgent:
    def __init__(
        self,
        faq_file: str = "activity_faq.txt",
        prompt_file: str = "prompts/activity_prompt.txt",
        chroma_path: str = "./chroma_db",
        collection_name: str = "activity_faq_collection",
    ):
        self.faq_file = faq_file
        self.prompt_file = prompt_file
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)

    # ----------------------------------------------------------
    # System prompt
    # ----------------------------------------------------------

    def get_system_prompt(self) -> str:
        with open(self.prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()

    # ----------------------------------------------------------
    # FAQ loading
    # ----------------------------------------------------------

    def load_faqs(self) -> list:
        """Parse activity_faq.txt into a list of FAQ dicts."""
        with open(self.faq_file, "r", encoding="utf-8") as f:
            content = f.read()

        block_pattern = re.compile(
            r"# id:\s*(.*?)\s*\|\s*category:\s*(.*?)\n(.*?)(?=\n# id:|\Z)",
            re.DOTALL,
        )
        qa_pattern = re.compile(
            r"Q:\s*(.*?)\nA:\s*(.*?)(?=\nQ:|\Z)",
            re.DOTALL,
        )

        faqs = []
        for block_match in block_pattern.finditer(content):
            block_id = block_match.group(1).strip()
            category = block_match.group(2).strip()
            block_body = block_match.group(3)
            qa_pairs = qa_pattern.findall(block_body)

            for index, (question, answer) in enumerate(qa_pairs):
                unique_id = (
                    f"{block_id}_{index + 1}" if len(qa_pairs) > 1 else block_id
                )
                faqs.append({
                    "id": unique_id,
                    "category": category,
                    "question": question.strip(),
                    "answer": answer.strip(),
                })

        return faqs

    # ----------------------------------------------------------
    # Vectorisation
    # ----------------------------------------------------------

    def vectorize_faqs(self, get_embedding_fn):
        """Embed all activity FAQs and store in ChromaDB."""
        faqs = self.load_faqs()
        print(f"[ActivityAgent] Found {len(faqs)} FAQs")

        existing_ids = self.collection.get()["ids"]
        if existing_ids:
            self.collection.delete(ids=existing_ids)
            print(f"[ActivityAgent] Cleared {len(existing_ids)} stale records.")

        for faq in faqs:
            text = f"Question: {faq['question']}\nAnswer: {faq['answer']}"
            print(f"[ActivityAgent] Embedding: {faq['question']}")
            embedding = get_embedding_fn(text)
            self.collection.upsert(
                ids=[faq["id"]],
                documents=[text],
                embeddings=[embedding],
                metadatas=[{
                    "faq_id": faq["id"],
                    "category": faq["category"],
                    "question": faq["question"],
                }],
            )

        print(f"[ActivityAgent] Vectorization done. Total: {self.collection.count()}")

    # ----------------------------------------------------------
    # Retrieval
    # ----------------------------------------------------------

    def retrieve(self, query_embedding: list, top_k: int, threshold: float) -> dict:
        """Query ChromaDB and return filtered results."""
        total_docs = self.collection.count()
        documents, metadatas, ids = [], [], []

        if total_docs > 0:
            n_results = min(top_k, total_docs)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            for i, distance in enumerate(results["distances"][0]):
                status = "✓ kept" if distance <= threshold else "✗ dropped"
                print(f"  [ActivityAgent][{results['ids'][0][i]}] dist={distance:.4f} {status}")
                if distance <= threshold:
                    documents.append(results["documents"][0][i])
                    metadatas.append(results["metadatas"][0][i])
                    ids.append(results["ids"][0][i])

        return {"documents": documents, "metadatas": metadatas, "ids": ids}

    # ----------------------------------------------------------
    # Collection accessor
    # ----------------------------------------------------------

    def get_collection(self):
        return self.collection

    # ----------------------------------------------------------
    # Stream one user turn — owns retrieval, LLM call, and
    # history persistence for the activity agent.
    # ----------------------------------------------------------

    def stream_response(self, session_id: str, question: str):
        """
        Generator that yields SSE-formatted strings for the activity route.
        Loads history, retrieves relevant FAQs, streams the LLM reply,
        and saves the completed turn — all internally.
        """
        history = load_history(session_id, "activity")

        # ---- Retrieval ----
        query_embedding = get_embedding(question)
        result    = self.retrieve(
            query_embedding=query_embedding,
            top_k=RETRIEVAL_TOP_K,
            threshold=SIMILARITY_THRESHOLD,
        )
        documents = result["documents"]
        metadatas = result["metadatas"]
        ids       = result["ids"]

        # ---- Build context message ----
        if documents:
            context = "".join(f"\nFAQ {i + 1}:\n{doc}\n" for i, doc in enumerate(documents))
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

        # ---- Sources payload ----
        sources_payload = json.dumps([
            {
                "id":       ids[i],
                "question": metadatas[i]["question"],
                "category": metadatas[i]["category"],
            }
            for i in range(len(documents))
        ])

        # ---- Append user turn to history before streaming ----
        history.append({"role": "user", "content": question})
        save_history(session_id, "activity", history)

        yield f"event: sources\ndata: {sources_payload}\n\n"

        # ---- LLM streaming ----
        headers = {
            "api-key":       LLM_API_KEY,
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Accept":        "text/event-stream",
            "Content-Type":  "application/json",
        }

        messages = []
        for msg in history[:-1]:
            messages.append({
                "role":    msg["role"],
                "content": [{"type": "text", "text": msg["content"]}],
            })
        messages.append({
            "role":    "user",
            "content": [{"type": "text", "text": current_message}],
        })

        print(f"\n[LLM] Sending {len(messages)} messages to LLM (agent=activity):")
        for m in messages:
            preview = m["content"][0]["text"][:80].replace("\n", " ")
            print(f"  [{m['role']}] {preview}...")

        payload = {
            "model_id": LLM_MODEL_ID,
            "body": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens":  LLM_MAX_TOKENS,
                "temperature": LLM_TEMPERATURE,
                "top_k":       LLM_TOP_K,
                "system":      self.get_system_prompt(),
                "messages":    messages,
            },
        }

        full_answer = ""

        with httpx.stream(
            "POST", LLM_API_URL,
            headers=headers, json=payload, timeout=120,
        ) as response:
            if response.status_code != 200:
                error_body = response.read().decode("utf-8", errors="replace")
                print(f"LLM API error {response.status_code}: {error_body}")
            response.raise_for_status()

            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
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

        # ---- Save assistant reply ----
        history.append({"role": "assistant", "content": full_answer})
        save_history(session_id, "activity", history)
        print(f"[HISTORY] Saved agent='activity'. Total messages: {len(history)}")

        yield "event: done\ndata: {}\n\n"

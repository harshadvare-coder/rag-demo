import json
import os
import re
from datetime import datetime, date

import httpx
import chromadb

# =========================================================
# Booking Agent
#
# Pure infrastructure — zero business logic.
# Everything the LLM should do is in booking_prompt.txt.
#
# Responsibilities:
#   1. FAQ loading, vectorisation, retrieval (ChromaDB)
#   2. Booking state persistence (JSON on disk)
#   3. Build system prompt = base prompt + FAQ context + state snapshot
#   4. Call LLM, collect reply
#   5. Parse <booking_state> block from reply and merge into state
# =========================================================


class BookingAgent:
    def __init__(
        self,
        faq_file: str        = "hotel_faq.txt",
        prompt_file: str     = "prompts/booking_prompt.txt",
        booking_dir: str     = "booking_sessions",
        chroma_path: str     = "./chroma_db",
        collection_name: str = "booking_faq_collection",
        llm_api_url: str     = "",
        llm_api_key: str     = "",
        llm_model_id: str    = "",
    ):
        self.faq_file      = faq_file
        self.prompt_file   = prompt_file
        self.booking_dir   = booking_dir
        self.llm_api_url   = llm_api_url
        self.llm_api_key   = llm_api_key
        self.llm_model_id  = llm_model_id

        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection    = self.chroma_client.get_or_create_collection(
            name=collection_name
        )
        os.makedirs(self.booking_dir, exist_ok=True)

    # ----------------------------------------------------------
    # FAQ loading
    # ----------------------------------------------------------

    def load_faqs(self) -> list:
        """Parse hotel_faq.txt into a list of FAQ dicts."""
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
            block_id   = block_match.group(1).strip()
            category   = block_match.group(2).strip()
            block_body = block_match.group(3)
            qa_pairs   = qa_pattern.findall(block_body)

            for index, (question, answer) in enumerate(qa_pairs):
                unique_id = (
                    f"{block_id}_{index + 1}" if len(qa_pairs) > 1 else block_id
                )
                faqs.append({
                    "id":       unique_id,
                    "category": category,
                    "question": question.strip(),
                    "answer":   answer.strip(),
                })
        return faqs

    # ----------------------------------------------------------
    # Vectorisation
    # ----------------------------------------------------------

    def vectorize_faqs(self, get_embedding_fn):
        """Embed all hotel FAQs and store in ChromaDB."""
        faqs = self.load_faqs()
        print(f"[BookingAgent] Found {len(faqs)} FAQs")

        existing_ids = self.collection.get()["ids"]
        if existing_ids:
            self.collection.delete(ids=existing_ids)
            print(f"[BookingAgent] Cleared {len(existing_ids)} stale records.")

        for faq in faqs:
            text      = f"Question: {faq['question']}\nAnswer: {faq['answer']}"
            print(f"[BookingAgent] Embedding: {faq['question']}")
            embedding = get_embedding_fn(text)
            self.collection.upsert(
                ids       =[faq["id"]],
                documents =[text],
                embeddings=[embedding],
                metadatas =[{
                    "faq_id":   faq["id"],
                    "category": faq["category"],
                    "question": faq["question"],
                }],
            )
        print(f"[BookingAgent] Vectorization done. Total: {self.collection.count()}")

    # ----------------------------------------------------------
    # Retrieval
    # ----------------------------------------------------------

    def retrieve(
        self,
        query_embedding: list,
        top_k: int       = 3,
        threshold: float = 0.5,
    ) -> dict:
        """Query ChromaDB and return filtered FAQ results."""
        total_docs = self.collection.count()
        documents, metadatas, ids = [], [], []

        if total_docs > 0:
            n_results = min(top_k, total_docs)
            results   = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
            for i, distance in enumerate(results["distances"][0]):
                status = "✓ kept" if distance <= threshold else "✗ dropped"
                print(
                    f"  [BookingAgent][{results['ids'][0][i]}] "
                    f"dist={distance:.4f} {status}"
                )
                if distance <= threshold:
                    documents.append(results["documents"][0][i])
                    metadatas.append(results["metadatas"][0][i])
                    ids.append(results["ids"][0][i])

        return {"documents": documents, "metadatas": metadatas, "ids": ids}

    def get_collection(self):
        return self.collection

    # ----------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------

    def _booking_path(self, session_id: str) -> str:
        return os.path.join(self.booking_dir, f"{session_id}.json")

    def load_booking_state(self, session_id: str) -> dict:
        path = self._booking_path(session_id)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return self._empty_state()

    def save_booking_state(self, session_id: str, state: dict):
        with open(self._booking_path(session_id), "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def clear_booking_state(self, session_id: str):
        path = self._booking_path(session_id)
        if os.path.exists(path):
            os.remove(path)

    @staticmethod
    def _empty_state() -> dict:
        return {
            "step":                   "destination",
            "destination":            None,
            "checkin_date":           None,
            "checkout_date":          None,
            "num_guests":             None,
            "guest_name":             None,
            "guest_email":            None,
            "special_requests":       None,
            "special_requests_asked": False,
            "confirmed":              False,
            "cancelled":              False,
            "created_at":             datetime.utcnow().isoformat(),
        }

    # ----------------------------------------------------------
    # System prompt builder
    # ----------------------------------------------------------

    def get_system_prompt(self) -> str:
        with open(self.prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _build_system_prompt(self, state: dict, faq_context: str = "") -> str:
        """
        base prompt  +  FAQ context (if any)  +  live state snapshot.
        The prompt file contains all behavioural rules including how to
        handle cancellation, post-confirmation FAQ, and state transitions.
        """
        base = self.get_system_prompt()

        # ---- FAQ context ----
        faq_block = ""
        if faq_context:
            faq_block = (
                "\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "RELEVANT HOTEL FAQ CONTEXT\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "Use the information below to answer hotel-related questions.\n"
                "If the question is not covered here, rely on general knowledge.\n\n"
                + faq_context
            )

        # ---- Live booking state snapshot ----
        filled = {
            k: v for k, v in {
                "destination":      state.get("destination"),
                "checkin_date":     state.get("checkin_date"),
                "checkout_date":    state.get("checkout_date"),
                "num_guests":       state.get("num_guests"),
                "guest_name":       state.get("guest_name"),
                "guest_email":      state.get("guest_email"),
                "special_requests": state.get("special_requests"),
            }.items() if v is not None
        }

        missing = [
            f for f in [
                "destination", "checkin_date", "checkout_date",
                "num_guests", "guest_name", "guest_email",
            ] if not state.get(f)
        ]
        if state.get("special_requests") is None and not state.get("special_requests_asked"):
            missing.append("special_requests (optional)")

        state_block = (
            "\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "CURRENT BOOKING STATE  (do NOT show this to the user)\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Flow step         : {state.get('step', 'destination')}\n"
            f"Already collected : {json.dumps(filled) if filled else 'nothing yet'}\n"
            f"Still missing     : {', '.join(missing) if missing else 'none'}\n"
            f"Today's date      : {date.today().strftime('%-d %b %Y')}\n"
            "\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "STATE UPDATE INSTRUCTION\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "After your reply, append on a NEW line a JSON block in\n"
            "<booking_state>…</booking_state> tags.\n"
            "\n"
            "CRITICAL RULES FOR THE STATE BLOCK:\n"
            "- Include ALL currently known field values every time — not just changed fields.\n"
            "- Copy 'Already collected' values above into the block, then add/update new ones.\n"
            "- Never omit a collected field. Never set a collected field to null unless the\n"
            "  user explicitly asked to clear it.\n"
            "- When the booking is confirmed, set confirmed=true and step='confirmed'.\n"
            "- When the user cancels, set cancelled=true and step='cancelled'.\n"
            "- When the booking is already confirmed or cancelled and the user asks a hotel\n"
            "  info question, answer it and emit the state block unchanged (same values).\n"
            "\n"
            "Valid keys: destination, checkin_date, checkout_date, num_guests,\n"
            "            guest_name, guest_email, special_requests,\n"
            "            confirmed (bool), cancelled (bool), step (string).\n"
            "\n"
            "step values: destination | checkin_date | checkout_date | num_guests |\n"
            "             guest_name | guest_email | special_requests | review | confirmed | cancelled\n"
            "\n"
            "Example — confirming a complete booking:\n"
            "<booking_state>\n"
            '{"destination": "Paris", "checkin_date": "20 Sep 2026", "checkout_date": "25 Sep 2026",\n'
            ' "num_guests": 2, "guest_name": "Jane Doe", "guest_email": "jane@example.com",\n'
            ' "special_requests": null, "confirmed": true, "step": "confirmed"}\n'
            "</booking_state>\n"
        )

        return base + faq_block + state_block

    # ----------------------------------------------------------
    # State-update parser
    # ----------------------------------------------------------

    @staticmethod
    def _parse_state_update(raw_reply: str) -> tuple[str, dict]:
        """Strip <booking_state>…</booking_state> from reply, return (clean, updates)."""
        pattern = re.compile(
            r"<booking_state>\s*(.*?)\s*</booking_state>",
            re.DOTALL | re.IGNORECASE,
        )
        match = pattern.search(raw_reply)
        if not match:
            return raw_reply.strip(), {}

        json_str = match.group(1).strip()
        clean    = pattern.sub("", raw_reply).strip()

        try:
            updates = json.loads(json_str)
            if not isinstance(updates, dict):
                updates = {}
        except json.JSONDecodeError:
            updates = {}

        return clean, updates

    def _apply_updates(self, state: dict, updates: dict) -> dict:
        """Merge LLM updates into state. Never overwrite an existing value with None."""
        allowed = {
            "destination", "checkin_date", "checkout_date", "num_guests",
            "guest_name", "guest_email", "special_requests",
            "confirmed", "cancelled", "step",
        }
        for key, value in updates.items():
            if key not in allowed:
                continue
            if value is None and state.get(key) is not None:
                continue  # protect existing values
            state[key] = value

        # Keep step in sync with terminal flags
        if state.get("confirmed") and state.get("step") != "confirmed":
            state["step"] = "confirmed"
        if state.get("cancelled") and state.get("step") != "cancelled":
            state["step"] = "cancelled"

        # Track when special_requests step was reached
        if state.get("step") == "special_requests":
            state["special_requests_asked"] = True

        return state

    # ----------------------------------------------------------
    # LLM call
    # ----------------------------------------------------------

    def _call_llm(self, system_prompt: str, history: list, user_message: str) -> str:
        messages = []
        for msg in history:
            messages.append({
                "role":    msg["role"],
                "content": [{"type": "text", "text": msg["content"]}],
            })
        messages.append({
            "role":    "user",
            "content": [{"type": "text", "text": user_message}],
        })

        payload = {
            "model_id": self.llm_model_id,
            "body": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens":        1024,
                "temperature":       0.3,
                "system":            system_prompt,
                "messages":          messages,
            },
        }
        headers = {
            "api-key":       self.llm_api_key,
            "Authorization": f"Bearer {self.llm_api_key}",
            "Accept":        "text/event-stream",
            "Content-Type":  "application/json",
        }

        full_text = ""
        try:
            with httpx.stream(
                "POST", self.llm_api_url,
                headers=headers, json=payload, timeout=60,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[len("data:"):].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    delta = None
                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {}).get("text", "")
                    elif "text" in event:
                        delta = event["text"]
                    elif "content" in event:
                        delta = event["content"]
                    elif "message" in event:
                        delta = event["message"]

                    if delta:
                        full_text += delta

        except Exception as exc:
            print(f"[BookingAgent] LLM call failed: {exc}")
            full_text = (
                "I'm sorry, I'm having trouble connecting right now. "
                "Please try again in a moment."
            )

        return full_text

    # ----------------------------------------------------------
    # Core — process one user turn
    # ----------------------------------------------------------

    def process_turn(
        self,
        session_id:      str,
        user_message:    str,
        history:         list | None = None,
        get_embedding_fn             = None,
        top_k:           int         = 3,
        threshold:       float       = 0.5,
    ) -> dict:
        """
        For every turn:
          1. Load state
          2. RAG retrieval (if embedding function provided)
          3. Build system prompt (base + FAQ context + state snapshot)
          4. Call LLM
          5. Parse <booking_state> block → merge into state → save
          6. Return reply + state
        No business logic here — the prompt drives everything.
        """
        if history is None:
            history = []

        state = self.load_booking_state(session_id)

        # ---- RAG retrieval ----
        faq_context = ""
        sources     = []

        if get_embedding_fn is not None:
            try:
                query_embedding = get_embedding_fn(user_message)
                rag_result      = self.retrieve(query_embedding, top_k=top_k, threshold=threshold)

                for i, doc in enumerate(rag_result["documents"]):
                    faq_context += f"FAQ {i + 1}:\n{doc}\n\n"

                sources = [
                    {
                        "id":       rag_result["ids"][i],
                        "question": rag_result["metadatas"][i]["question"],
                        "category": rag_result["metadatas"][i]["category"],
                    }
                    for i in range(len(rag_result["documents"]))
                ]

                print(
                    f"[BookingAgent] RAG: {len(sources)} FAQ(s) retrieved"
                    if sources else "[BookingAgent] RAG: no matches above threshold"
                )
            except Exception as exc:
                print(f"[BookingAgent] RAG failed (continuing without context): {exc}")

        # ---- Build prompt + call LLM ----
        system_prompt         = self._build_system_prompt(state, faq_context=faq_context)
        raw_reply             = self._call_llm(system_prompt, history, user_message)

        # ---- Parse and apply state update ----
        clean_reply, updates  = self._parse_state_update(raw_reply)
        state                 = self._apply_updates(state, updates)
        self.save_booking_state(session_id, state)

        print(f"[BookingAgent] session={session_id} step={state['step']} updates={updates}")

        is_confirmed = bool(state.get("confirmed"))
        is_cancelled = bool(state.get("cancelled"))
        is_done      = state["step"] in ("confirmed", "cancelled")

        return {
            "reply":     clean_reply,
            "state":     state,
            "done":      is_done,
            "confirmed": is_confirmed,
            "cancelled": is_cancelled,
            "sources":   sources,
        }

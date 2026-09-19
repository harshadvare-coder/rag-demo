import json

import httpx

from core.config import (
    LLM_API_URL,
    LLM_API_KEY,
    LLM_MODEL_ID,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TOP_K,
)
from core.history import load_history, save_history


# =========================================================
# General Agent
# Handles conversational / off-topic messages with no RAG.
# Owns its own history loading, LLM streaming, and saving.
# =========================================================


class GeneralAgent:
    def __init__(self, prompt_file: str = "prompts/general_prompt.txt"):
        self.prompt_file = prompt_file

    def get_system_prompt(self) -> str:
        with open(self.prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()

    def stream_response(self, session_id: str, question: str):
        """
        Generator that yields SSE-formatted strings for the general route.
        Loads history, streams a conversational LLM reply, and saves the
        completed turn — all internally.
        """
        history = load_history(session_id, "general")

        current_message = (
            f"There is no FAQ context available for this question.\n"
            f"Answer using the conversation history above.\n"
            f"USER QUESTION:\n{question}"
        )

        # Append user turn before streaming
        history.append({"role": "user", "content": question})
        save_history(session_id, "general", history)

        yield "event: sources\ndata: []\n\n"

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

        print(f"\n[LLM] Sending {len(messages)} messages to LLM (agent=GeneralAgent):")
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

        # Save assistant reply
        history.append({"role": "assistant", "content": full_answer})
        save_history(session_id, "general", history)
        print(f"[HISTORY] Saved agent='GeneralAgent'. Total messages: {len(history)}")

        yield "event: done\ndata: {}\n\n"

import json
import re

import httpx


# =========================================================
# Orchestrator
# Builds the routing payload, calls the LLM API, and parses
# the routing decision. main.py only calls classify().
# =========================================================


class Orchestrator:
    def __init__(
        self,
        prompt_file: str = "prompts/orchestrator_prompt.txt",
        llm_api_url: str = "",
        llm_api_key: str = "",
        llm_model_id: str = "",
    ):
        self.prompt_file = prompt_file
        self.llm_api_url = llm_api_url
        self.llm_api_key = llm_api_key
        self.llm_model_id = llm_model_id

    # ----------------------------------------------------------
    # System prompt
    # ----------------------------------------------------------

    def _load_routing_prompt(self) -> str:
        with open(self.prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()

    # ----------------------------------------------------------
    # Payload builder
    # ----------------------------------------------------------

    def _build_payload(self, question: str, history: list) -> dict:
        """Build the LLM request payload for routing classification."""
        messages = []
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": [{"type": "text", "text": msg["content"]}],
            })
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": question}],
        })

        return {
            "model_id": self.llm_model_id,
            "body": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 150,
                "temperature": 0.0,
                "system": self._load_routing_prompt(),
                "messages": messages,
            },
        }

    # ----------------------------------------------------------
    # Route parser
    # ----------------------------------------------------------

    def _parse_route(self, raw_text: str) -> str:
        """
        Parse the raw LLM response text and return the routing decision.
        Returns one of: "activity", "hotel", "booking", "general"
        """
        text = raw_text.strip()
        print(f"[Orchestrator] Raw classification response: {text}")

        # Strip markdown code fences if LLM wraps in ```json ... ```
        if text.startswith("```"):
            text = re.sub(r"^```[a-z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text).strip()

        try:
            parsed = json.loads(text)
            agent = parsed.get("agent", "general").lower().strip()
            reason = parsed.get("reason", "")
            print(f"[Orchestrator] Routed to → '{agent}' | Reason: {reason}")
            if agent in ("activity", "hotel", "booking"):
                return agent
        except Exception as e:
            print(f"[Orchestrator] Parse error: {e}. Defaulting to 'general'.")

        return "general"

    # ----------------------------------------------------------
    # Public entry point — called by main.py
    # ----------------------------------------------------------

    def classify(self, question: str, history: list) -> str:
        """
        Call the LLM API to classify `question` and return the route.
        Returns one of: "activity", "hotel", "booking", "general"
        """
        payload = self._build_payload(question, history)
        headers = {
            "api-key": self.llm_api_key,
            "Authorization": f"Bearer {self.llm_api_key}",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }

        route_raw = ""
        try:
            with httpx.stream(
                "POST",
                self.llm_api_url,
                headers=headers,
                json=payload,
                timeout=30,
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

        return self._parse_route(route_raw) if route_raw else "general"

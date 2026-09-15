import json
import re


# =========================================================
# Orchestrator
# Acts as a pure distributor — builds the routing payload
# and parses the LLM response. No HTTP calls here.
# =========================================================

PROMPT_FILE = "prompts/orchestrator_prompt.txt"


def _load_routing_prompt() -> str:
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_payload(question: str, history: list, llm_model_id: str) -> dict:
    """
    Build the LLM request payload for routing classification.

    `history` is a list of {"role": ..., "content": ...} dicts from
    previous orchestrator turns (question → route pairs).

    Returns a payload dict ready to POST to the LLM API.
    """
    messages = []
    for msg in history:
        messages.append({
            "role": msg["role"],
            "content": [{"type": "text", "text": msg["content"]}]
        })
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": question}]
    })

    return {
        "model_id": llm_model_id,
        "body": {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 150,
            "temperature": 0.0,
            "system": _load_routing_prompt(),
            "messages": messages
        }
    }


def parse_route(raw_text: str) -> str:
    """
    Parse the raw LLM response text and return the routing decision.
    Returns one of: "activity", "hotel", "general"
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
        if agent in ("activity", "hotel"):
            return agent
    except Exception as e:
        print(f"[Orchestrator] Parse error: {e}. Defaulting to 'general'.")

    return "general"

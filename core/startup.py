import threading


# =========================================================
# Startup — vectorize FAQs in background threads
# =========================================================

def run_vectorization(agents_to_vectorize: list, get_embedding_fn):
    """
    Spawn a daemon thread per agent so vectorization (embedding API calls
    + ChromaDB writes) never blocks the uvicorn startup or the event loop.
    Each agent owns its own ChromaDB collection, so threads don't contend.

    :param agents_to_vectorize: list of (agent_instance, name_str) tuples
    :param get_embedding_fn:    callable that accepts a text string and
                                returns an embedding vector
    """
    def vectorize(agent, name):
        try:
            print(f"[Startup] [{name}] Vectorization started...")
            agent.vectorize_faqs(get_embedding_fn)
            print(f"[Startup] [{name}] Vectorization complete.")
        except Exception as e:
            print(f"[Startup] [{name}] Vectorization failed: {e}")

    for agent, name in agents_to_vectorize:
        threading.Thread(target=vectorize, args=(agent, name), daemon=True).start()

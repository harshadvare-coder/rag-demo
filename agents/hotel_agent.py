import re
import chromadb


# =========================================================
# Hotel Agent
# Owns the hotel_faq.txt knowledge base
# =========================================================

chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(name="hotel_faq_collection")


PROMPT_FILE = "prompts/hotel_prompt.txt"


def _load_system_prompt() -> str:
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_faqs(faq_file: str = "hotel_faq.txt") -> list:
    """Parse hotel_faq.txt into a list of FAQ dicts."""

    with open(faq_file, "r", encoding="utf-8") as f:
        content = f.read()

    block_pattern = re.compile(
        r"# id:\s*(.*?)\s*\|\s*category:\s*(.*?)\n(.*?)(?=\n# id:|\Z)",
        re.DOTALL
    )
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
            unique_id = (
                f"{block_id}_{index + 1}" if len(qa_pairs) > 1 else block_id
            )
            faqs.append({
                "id": unique_id,
                "category": category,
                "question": question.strip(),
                "answer": answer.strip()
            })

    return faqs


def vectorize_faqs(get_embedding_fn):
    """Embed all hotel FAQs and store in ChromaDB."""

    faqs = load_faqs()
    print(f"[HotelAgent] Found {len(faqs)} FAQs")

    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)
        print(f"[HotelAgent] Cleared {len(existing_ids)} stale records.")

    for faq in faqs:
        text = f"Question: {faq['question']}\nAnswer: {faq['answer']}"
        print(f"[HotelAgent] Embedding: {faq['question']}")
        embedding = get_embedding_fn(text)
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

    print(f"[HotelAgent] Vectorization done. Total: {collection.count()}")


def retrieve(query_embedding: list, top_k: int, threshold: float) -> dict:
    """Query ChromaDB and return filtered results."""

    total_docs = collection.count()
    documents, metadatas, ids = [], [], []

    if total_docs > 0:
        n_results = min(top_k, total_docs)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        for i, distance in enumerate(results["distances"][0]):
            status = "✓ kept" if distance <= threshold else "✗ dropped"
            print(f"  [HotelAgent][{results['ids'][0][i]}] dist={distance:.4f} {status}")
            if distance <= threshold:
                documents.append(results["documents"][0][i])
                metadatas.append(results["metadatas"][0][i])
                ids.append(results["ids"][0][i])

    return {"documents": documents, "metadatas": metadatas, "ids": ids}


def get_system_prompt() -> str:
    return _load_system_prompt()


def get_collection():
    return collection

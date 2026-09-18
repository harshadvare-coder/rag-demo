import re
import chromadb


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

"""RAG pipeline — document chunking, embedding, and retrieval."""

from __future__ import annotations
import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any


class VectorStore:
    """Pure Python TF-IDF vector store with cosine similarity."""

    def __init__(self):
        self.documents: list[dict] = []
        self.vocabulary: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.tfidf_matrix: list[dict[int, float]] = []
        self.metadata: list[dict] = []

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _compute_idf(self):
        n = len(self.documents)
        df: Counter = Counter()
        for doc in self.documents:
            tokens = set(self._tokenize(doc.get("text", "")))
            for t in tokens:
                df[t] += 1
        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = math.log((1 + n) / (1 + freq)) + 1
        self.vocabulary = {t: i for i, t in enumerate(sorted(self.idf.keys()))}

    def _compute_tfidf(self, text: str) -> dict[int, float]:
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        vec = {}
        for term, count in tf.items():
            if term in self.idf:
                idx = self.vocabulary[term]
                vec[idx] = (count / total) * self.idf[term]
        return vec

    def build_index(self, documents: list[dict]):
        self.documents = documents
        self.metadata = [doc.get("metadata", {}) for doc in documents]
        self._compute_idf()
        self.tfidf_matrix = []
        for doc in documents:
            self.tfidf_matrix.append(self._compute_tfidf(doc.get("text", "")))

    @staticmethod
    def _cosine_similarity(a: dict[int, float], b: dict[int, float]) -> float:
        if not a or not b:
            return 0.0
        common = set(a.keys()) & set(b.keys())
        dot = sum(a[k] * b[k] for k in common)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search(self, query: str, top_k: int = 5, metadata_filter: dict | None = None) -> list[dict]:
        q_vec = self._compute_tfidf(query)
        scored = []
        for i, doc_vec in enumerate(self.tfidf_matrix):
            if metadata_filter:
                meta = self.metadata[i] if i < len(self.metadata) else {}
                if not all(meta.get(k) == v for k, v in metadata_filter.items()):
                    continue
            score = self._cosine_similarity(q_vec, doc_vec)
            if score > 0:
                scored.append((i, score))
        scored.sort(key=lambda x: -x[1])
        results = []
        for idx, score in scored[:top_k]:
            entry = {**self.documents[idx], "score": score}
            if idx < len(self.metadata):
                entry["metadata"] = self.metadata[idx]
            results.append(entry)
        return results

    def add_documents(self, docs: list[dict]):
        self.documents.extend(docs)
        self.metadata.extend([doc.get("metadata", {}) for doc in docs])
        self._compute_idf()
        self.tfidf_matrix = []
        for doc in self.documents:
            self.tfidf_matrix.append(self._compute_tfidf(doc.get("text", "")))

    def save_index(self, path: str):
        data = {
            "documents": self.documents,
            "vocabulary": self.vocabulary,
            "idf": self.idf,
            "tfidf_matrix": self.tfidf_matrix,
            "metadata": self.metadata,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_index(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.documents = data["documents"]
        self.vocabulary = data["vocabulary"]
        self.idf = data["idf"]
        self.tfidf_matrix = data["tfidf_matrix"]
        self.metadata = data.get("metadata", [{} for _ in self.documents])


class RAGPipeline:
    def __init__(self, display: Any = None):
        self.display = display
        self.documents: list[dict] = []
        self.chunks: list[dict] = []
        self.vector_store = VectorStore()

        try:
            import faiss

            self.faiss = faiss
            self._has_faiss = True
        except ImportError:
            self.faiss = None
            self._has_faiss = False

    def build(self, docs_dir: str, chunk_size: int = 512, overlap: int = 50):
        if self.display:
            self.display.print_step(f"Loading documents from {docs_dir}...")

        self.documents = self._load_documents(docs_dir)
        if not self.documents:
            if self.display:
                self.display.print_error("No documents found.")
            return

        if self.display:
            self.display.print_step(f"Loaded {len(self.documents)} documents")

        self.chunks = self._chunk_documents(chunk_size, overlap)
        if self.display:
            self.display.print_step(
                f"Created {len(self.chunks)} chunks (size={chunk_size}, overlap={overlap})"
            )

        self._build_index()

    def query(self, query: str, top_k: int = 5, metadata_filter: dict | None = None) -> dict:
        if not self.chunks:
            return {
                "query": query,
                "results": [],
                "answer": "No documents indexed. Run 'rag build' first.",
            }

        results = self._search(query, top_k, metadata_filter)
        context = "\n\n".join(r["text"] for r in results)

        return {
            "query": query,
            "results": results,
            "context": context,
            "answer": f"Found {len(results)} relevant chunks for: {query}",
        }

    def _load_documents(self, path: str) -> list[dict]:
        docs = []
        p = Path(path)
        if p.is_file():
            return [self._read_file(p)]
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.suffix.lower() in (
                    ".txt",
                    ".md",
                    ".py",
                    ".json",
                    ".yaml",
                    ".yml",
                    ".csv",
                ):
                    doc = self._read_file(f)
                    if doc:
                        docs.append(doc)
        return docs

    def _read_file(self, path: Path) -> dict:
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return {
                "path": str(path),
                "name": path.name,
                "content": content,
                "text": content,
            }
        except Exception:
            return {}

    def _chunk_documents(self, size: int, overlap: int) -> list[dict]:
        chunks = []
        for doc in self.documents:
            text = doc.get("content", "")
            doc_name = doc.get("name", "unknown")
            for i in range(0, len(text), size - overlap):
                chunk_text = text[i : i + size]
                if chunk_text.strip():
                    chunks.append(
                        {
                            "text": chunk_text,
                            "source": doc_name,
                            "offset": i,
                            "metadata": {"source": doc_name, "offset": i},
                        }
                    )
        return chunks

    def _build_index(self):
        self.vector_store.build_index(self.chunks)
        if self.display:
            self.display.print_step(
                f"Built TF-IDF index with {len(self.vector_store.vocabulary)} terms"
            )

    def _search(self, query: str, top_k: int, metadata_filter: dict | None = None) -> list[dict]:
        return self.vector_store.search(query, top_k, metadata_filter)

    def save_index(self, path: str):
        self.vector_store.save_index(path)
        if self.display:
            self.display.print_success(f"Index saved to {path}")

    def load_index(self, path: str):
        self.vector_store.load_index(path)
        self.chunks = self.vector_store.documents
        if self.display:
            self.display.print_success(f"Index loaded from {path} ({len(self.chunks)} chunks)")

    def add_documents(self, docs: list[dict]):
        self.documents.extend(docs)
        self.chunks.extend(docs)
        self.vector_store.add_documents(docs)
        if self.display:
            self.display.print_step(f"Added {len(docs)} documents to index")

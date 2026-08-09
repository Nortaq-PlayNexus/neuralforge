"""RAG pipeline — document chunking, embedding, and retrieval."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any


class RAGPipeline:
    def __init__(self, display: Any = None):
        self.display = display
        self.documents: list[dict] = []
        self.chunks: list[dict] = []

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
            self.display.print_step(f"Created {len(self.chunks)} chunks (size={chunk_size}, overlap={overlap})")

        self._build_index()

    def query(self, query: str, top_k: int = 5) -> dict:
        if not self.chunks:
            return {"query": query, "results": [], "answer": "No documents indexed. Run 'rag build' first."}

        results = self._search(query, top_k)
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
                if f.suffix.lower() in (".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv"):
                    doc = self._read_file(f)
                    if doc:
                        docs.append(doc)
        return docs

    def _read_file(self, path: Path) -> dict:
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return {"path": str(path), "name": path.name, "content": content}
        except Exception:
            return {}

    def _chunk_documents(self, size: int, overlap: int) -> list[dict]:
        chunks = []
        for doc in self.documents:
            text = doc.get("content", "")
            doc_name = doc.get("name", "unknown")
            for i in range(0, len(text), size - overlap):
                chunk_text = text[i:i + size]
                if chunk_text.strip():
                    chunks.append({
                        "text": chunk_text,
                        "source": doc_name,
                        "offset": i,
                    })
        return chunks

    def _build_index(self):
        pass

    def _search(self, query: str, top_k: int) -> list[dict]:
        query_lower = query.lower()
        scored = []
        for chunk in self.chunks:
            text = chunk["text"].lower()
            score = sum(1 for word in query_lower.split() if word in text)
            if score > 0:
                scored.append({**chunk, "score": score})
        scored.sort(key=lambda x: -x["score"])
        return scored[:top_k]

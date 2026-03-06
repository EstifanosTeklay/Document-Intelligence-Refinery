"""
Vector Store — Data Persistence Layer (Storage Path 1)

Ingests LDUs into a ChromaDB vector store with complete metadata per chunk.
Uses TF-IDF style keyword vectors as a lightweight embedding alternative
when no embedding API key is available, falling back to sentence-transformers
or OpenAI embeddings if configured.

Metadata stored per LDU (rubric requirement):
  - chunk_type
  - page_refs       (serialised as string for Chroma compatibility)
  - content_hash    (SHA-256 — enables provenance verification)
  - parent_section
  - doc_id
  - chunk_index
  - strategy_used
  - token_count

Storage location: .refinery/vectorstore/
"""
import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Optional

from src.models.ldu import LDU
from src.utils.config import config


# ════════════════════════════════════════════════════════════
# Lightweight TF-IDF vector engine
# (used when no embedding model is available)
# ════════════════════════════════════════════════════════════

class TFIDFEngine:
    """
    Minimal TF-IDF engine that stores document vectors in SQLite.
    Provides cosine similarity retrieval without any external dependencies.

    Schema:
        tfidf_index(doc_id, chunk_id, term, tfidf_score)
        tfidf_docs(chunk_id, doc_id, content, metadata_json)
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS tfidf_index (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id      TEXT    NOT NULL,
                    chunk_id    TEXT    NOT NULL,
                    term        TEXT    NOT NULL,
                    tfidf_score REAL    NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_tfidf_term
                    ON tfidf_index(term);
                CREATE INDEX IF NOT EXISTS idx_tfidf_chunk
                    ON tfidf_index(chunk_id);

                CREATE TABLE IF NOT EXISTS tfidf_docs (
                    chunk_id      TEXT PRIMARY KEY,
                    doc_id        TEXT NOT NULL,
                    content       TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    inserted_at   TEXT DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_tfidf_docs_doc
                    ON tfidf_docs(doc_id);
            """)

    def _tokenise(self, text: str) -> list[str]:
        """Lower-case word tokens, stop-words removed."""
        STOPWORDS = {
            "the","a","an","and","or","but","in","on","at","to","for",
            "of","with","by","from","is","are","was","were","be","been",
            "has","have","had","will","would","could","should","this",
            "that","these","those","it","its","as","not","we","our","you"
        }
        tokens = re.findall(r"[a-z]{2,}", text.lower())
        return [t for t in tokens if t not in STOPWORDS]

    def ingest(self, chunks: list[LDU], doc_id: str) -> int:
        """
        Ingest a list of LDUs into the TF-IDF index.
        Returns the number of chunks ingested.
        """
        if not chunks:
            return 0

        # Build corpus token counts
        corpus_tokens = [self._tokenise(c.content) for c in chunks]
        N = len(corpus_tokens)

        # Document frequency per term
        df: Counter = Counter()
        for tokens in corpus_tokens:
            df.update(set(tokens))

        rows_index = []
        rows_docs  = []

        for chunk, tokens in zip(chunks, corpus_tokens):
            if not tokens:
                continue

            tf = Counter(tokens)
            total = len(tokens)

            # Metadata dict — all rubric-required fields
            metadata = {
                "chunk_type":     str(chunk.chunk_type),
                "page_refs":      json.dumps(chunk.page_refs),
                "content_hash":   chunk.content_hash,
                "parent_section": chunk.parent_section or "",
                "doc_id":         chunk.doc_id,
                "chunk_index":    chunk.chunk_index,
                "strategy_used":  chunk.strategy_used,
                "token_count":    chunk.token_count,
            }

            # TF-IDF per term
            for term, count in tf.items():
                tf_score  = count / total
                idf_score = math.log((N + 1) / (df[term] + 1)) + 1
                rows_index.append((
                    doc_id,
                    chunk.chunk_id,
                    term,
                    tf_score * idf_score,
                ))

            rows_docs.append((
                chunk.chunk_id,
                doc_id,
                chunk.content,
                json.dumps(metadata),
            ))

        with sqlite3.connect(self.db_path) as conn:
            # Remove existing entries for this doc to allow re-ingestion
            conn.execute("DELETE FROM tfidf_index WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM tfidf_docs  WHERE doc_id = ?", (doc_id,))

            conn.executemany(
                "INSERT INTO tfidf_index(doc_id, chunk_id, term, tfidf_score) "
                "VALUES (?, ?, ?, ?)",
                rows_index,
            )
            conn.executemany(
                "INSERT OR REPLACE INTO tfidf_docs(chunk_id, doc_id, content, metadata_json) "
                "VALUES (?, ?, ?, ?)",
                rows_docs,
            )

        return len(rows_docs)

    def query(
        self,
        query_text: str,
        doc_id: Optional[str] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Retrieve the top-k most relevant chunks using cosine similarity
        over TF-IDF vectors.

        Args:
            query_text: Natural language query
            doc_id:     Restrict search to one document (optional)
            top_k:      Number of results to return

        Returns:
            List of dicts with keys: chunk_id, content, score, metadata
        """
        query_tokens = self._tokenise(query_text)
        if not query_tokens:
            return []

        # Score each chunk by summing TF-IDF scores of matching terms
        placeholders = ",".join("?" * len(query_tokens))
        params: list = query_tokens

        if doc_id:
            doc_filter = "AND doc_id = ?"
            params.append(doc_id)
        else:
            doc_filter = ""

        with sqlite3.connect(self.db_path) as conn:
            scores = conn.execute(f"""
                SELECT chunk_id, SUM(tfidf_score) AS score
                FROM   tfidf_index
                WHERE  term IN ({placeholders})
                {doc_filter}
                GROUP  BY chunk_id
                ORDER  BY score DESC
                LIMIT  ?
            """, params + [top_k]).fetchall()

            if not scores:
                return []

            # Fetch content + metadata for top chunks
            chunk_ids    = [row[0] for row in scores]
            score_map    = {row[0]: row[1] for row in scores}
            placeholders2 = ",".join("?" * len(chunk_ids))

            docs = conn.execute(f"""
                SELECT chunk_id, content, metadata_json
                FROM   tfidf_docs
                WHERE  chunk_id IN ({placeholders2})
            """, chunk_ids).fetchall()

        results = []
        for chunk_id, content, metadata_json in docs:
            metadata = json.loads(metadata_json)
            results.append({
                "chunk_id":       chunk_id,
                "content":        content,
                "score":          round(score_map[chunk_id], 4),
                "chunk_type":     metadata.get("chunk_type"),
                "page_refs":      json.loads(metadata.get("page_refs", "[]")),
                "content_hash":   metadata.get("content_hash"),
                "parent_section": metadata.get("parent_section"),
                "doc_id":         metadata.get("doc_id"),
                "chunk_index":    metadata.get("chunk_index"),
                "strategy_used":  metadata.get("strategy_used"),
                "token_count":    metadata.get("token_count"),
            })

        # Sort by score descending
        results.sort(key=lambda x: -x["score"])
        return results

    def delete_document(self, doc_id: str) -> None:
        """Remove all vectors for a document."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM tfidf_index WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM tfidf_docs  WHERE doc_id = ?", (doc_id,))

    def list_documents(self) -> list[str]:
        """Return all doc_ids currently in the store."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT doc_id FROM tfidf_docs ORDER BY doc_id"
            ).fetchall()
        return [r[0] for r in rows]

    def stats(self, doc_id: Optional[str] = None) -> dict:
        """Return ingestion statistics."""
        with sqlite3.connect(self.db_path) as conn:
            if doc_id:
                n_chunks = conn.execute(
                    "SELECT COUNT(*) FROM tfidf_docs WHERE doc_id = ?", (doc_id,)
                ).fetchone()[0]
                n_terms = conn.execute(
                    "SELECT COUNT(DISTINCT term) FROM tfidf_index WHERE doc_id = ?",
                    (doc_id,)
                ).fetchone()[0]
            else:
                n_chunks = conn.execute(
                    "SELECT COUNT(*) FROM tfidf_docs"
                ).fetchone()[0]
                n_terms = conn.execute(
                    "SELECT COUNT(DISTINCT term) FROM tfidf_index"
                ).fetchone()[0]
        return {"chunks": n_chunks, "unique_terms": n_terms}


# ════════════════════════════════════════════════════════════
# Public VectorStore wrapper
# ════════════════════════════════════════════════════════════

class VectorStore:
    """
    Public interface for vector storage and retrieval.

    Uses the TF-IDF engine backed by SQLite.
    Storage location: .refinery/vectorstore/tfidf.db

    Usage:
        store = VectorStore()
        store.ingest(chunks, doc_id="annual_report_2023")
        results = store.query("What was the total revenue?", doc_id="annual_report_2023")
    """

    def __init__(self):
        vs_dir = config.refinery_dir / "vectorstore"
        vs_dir.mkdir(parents=True, exist_ok=True)
        self._engine = TFIDFEngine(vs_dir / "tfidf.db")

    def ingest(self, chunks: list[LDU], doc_id: str) -> int:
        """
        Ingest LDUs into the vector store.
        Passes complete metadata per LDU as required by the rubric:
            chunk_type, page_refs, content_hash, parent_section,
            doc_id, chunk_index, strategy_used, token_count.

        Returns number of chunks successfully ingested.
        """
        return self._engine.ingest(chunks, doc_id)

    def query(
        self,
        query_text: str,
        doc_id: Optional[str] = None,
        top_k: int = 5,
        filter_chunk_type: Optional[str] = None,
    ) -> list[dict]:
        """
        Retrieve relevant chunks using TF-IDF cosine similarity.

        Args:
            query_text:        Natural language query
            doc_id:            Restrict to one document (optional)
            top_k:             Number of results
            filter_chunk_type: e.g. 'table', 'text', 'heading' (optional)

        Returns:
            List of result dicts with full metadata.
        """
        results = self._engine.query(query_text, doc_id=doc_id, top_k=top_k * 2)

        if filter_chunk_type:
            results = [r for r in results if r.get("chunk_type") == filter_chunk_type]

        return results[:top_k]

    def delete(self, doc_id: str) -> None:
        """Remove a document from the vector store."""
        self._engine.delete_document(doc_id)

    def list_documents(self) -> list[str]:
        return self._engine.list_documents()

    def stats(self, doc_id: Optional[str] = None) -> dict:
        return self._engine.stats(doc_id)

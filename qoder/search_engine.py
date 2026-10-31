"""Seekdb hybrid search engine wrapper."""

import os
import logging
import pyseekdb

logger = logging.getLogger(__name__)


class SearchEngine:
    def __init__(self):
        host = os.environ.get("SEEKDB_HOST")
        if host:
            self.client = pyseekdb.Client(
                host=host,
                port=int(os.environ.get("SEEKDB_PORT", "2881")),
                user=os.environ.get("SEEKDB_USER", "root"),
                password=os.environ.get("SEEKDB_PASSWORD", ""),
                database=os.environ.get("SEEKDB_DATABASE", "test"),
            )
            logger.info("Seekdb server mode: %s:%s", host, os.environ.get("SEEKDB_PORT", "2881"))
        else:
            db_path = os.environ.get("SEEKDB_PATH", "./seekdb_data")
            os.makedirs(db_path, exist_ok=True)
            self.client = pyseekdb.Client(path=db_path)
            logger.info("Seekdb embedded mode: %s", db_path)

        self.collection = self.client.get_or_create_collection(name="notes")
        count = self.collection.count()
        logger.info("Collection 'notes' ready, documents: %s", count)

    def add_documents(self, chunks: list[dict], metadatas: list[dict], ids: list[str]):
        """Add document chunks with metadata to the collection."""
        documents = [c["text"] for c in chunks]
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
        self.collection.refresh_index()
        logger.info("Added %d documents, refreshed index", len(ids))

    def hybrid_search(self, query: str, n_results: int = 5) -> list[dict]:
        """Perform hybrid search: semantic + keyword with RRF fusion.

        Uses pyseekdb's hybrid_search API which combines:
        - Full-text keyword matching (where_document / $contains)
        - Vector semantic similarity (query_texts)
        - RRF (Reciprocal Rank Fusion) to merge and rank results
        """
        if self.collection.count() == 0:
            return []

        try:
            results = self.collection.hybrid_search(
                query={
                    "where_document": {"$contains": query},
                    "n_results": min(n_results * 3, 30),
                },
                knn={
                    "query_texts": [query],
                    "n_results": min(n_results * 3, 30),
                },
                rank={"rrf": {}},
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
            return self._format_results(results)
        except Exception as e:
            logger.warning("hybrid_search failed (%s), falling back to query", e)
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"],
                )
                return self._format_results(results)
            except Exception as e2:
                logger.error("Fallback query also failed: %s", e2)
                return []

    def _format_results(self, results: list[dict]) -> list[dict]:
        """Normalize search results into a list of dicts."""
        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        formatted = []
        ids = results["ids"][0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            formatted.append({
                "id": doc_id,
                "text": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "distance": float(dists[i]) if i < len(dists) else None,
            })
        return formatted

    def count(self) -> int:
        return self.collection.count()

    def delete_by_filename(self, filename: str):
        """Delete all chunks belonging to a specific file."""
        try:
            self.collection.delete(where={"filename": filename})
            self.collection.refresh_index()
            logger.info("Deleted chunks for file: %s", filename)
        except Exception as e:
            logger.warning("Delete by filename failed: %s", e)
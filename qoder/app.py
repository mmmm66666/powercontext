"""Flask web application for note & literature retrieval assistant."""

import os
import uuid
import logging
from datetime import datetime

from flask import Flask, render_template, request, jsonify

from document_processor import extract_text, chunk_text
from search_engine import SearchEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXT = {"txt", "pdf"}

# Lazy-init search engine (avoids import-time side effects)
_engine: SearchEngine | None = None
# Track uploaded files in memory (keyed by filename)
uploaded_files: dict[str, dict] = {}


def get_engine() -> SearchEngine:
    global _engine
    if _engine is None:
        _engine = SearchEngine()
    return _engine


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ── Routes ───────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not _allowed(f.filename):
        return jsonify({"error": f"Unsupported type. Allowed: {', '.join(ALLOWED_EXT)}"}), 400

    filename = f.filename
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(save_path)

    try:
        engine = get_engine()

        # 1. Extract text
        text = extract_text(save_path, filename)
        if not text.strip():
            return jsonify({"error": "No text extracted from file"}), 400

        # 2. Chunk
        chunks = chunk_text(text)
        if not chunks:
            return jsonify({"error": "Failed to chunk document"}), 400

        # 3. Build IDs & metadata
        ids = [f"{filename}_{i}_{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
        metadatas = [
            {
                "filename": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "uploaded_at": datetime.now().isoformat(),
            }
            for i in range(len(chunks))
        ]

        # 4. Store in Seekdb (auto-embedding via pyseekdb)
        engine.add_documents(chunks, metadatas, ids)

        uploaded_files[filename] = {
            "filename": filename,
            "chunks": len(chunks),
            "uploaded_at": datetime.now().isoformat(),
        }

        logger.info("Uploaded '%s': %d chunks", filename, len(chunks))
        return jsonify({
            "message": f"Processed '{filename}' into {len(chunks)} chunks",
            "filename": filename,
            "chunks": len(chunks),
        })

    except Exception as e:
        logger.exception("Upload processing failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/search")
def search():
    query = request.args.get("q", "").strip()
    n = min(int(request.args.get("n", "5")), 20)

    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        engine = get_engine()
        results = engine.hybrid_search(query, n_results=n)
        logger.info("Search '%s' -> %d results", query, len(results))
        return jsonify({"query": query, "results": results, "count": len(results)})
    except Exception as e:
        logger.exception("Search failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/files")
def list_files():
    return jsonify({"files": list(uploaded_files.values()), "total_chunks": get_engine().count()})


@app.route("/api/files/<path:filename>", methods=["DELETE"])
def delete_file(filename):
    try:
        engine = get_engine()
        engine.delete_by_filename(filename)
        uploaded_files.pop(filename, None)
        return jsonify({"message": f"Deleted '{filename}'"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health():
    try:
        engine = get_engine()
        count = engine.count()
        return jsonify({"status": "ok", "seekdb": "connected", "total_chunks": count})
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
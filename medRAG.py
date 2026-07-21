from flask import Flask, request, jsonify
from flask_cors import CORS
from rag.pipeline import RAGPipeline
from hallucination.scorer import HallucinationScorer

app = Flask(__name__)
CORS(app)

rag = RAGPipeline()
scorer = HallucinationScorer()

@app.route("/api/query", methods = ["POST"])
def query():
    data = request.json
    user_query = data.get("query", "").strip()
    if not user_query:
        return jsonify({"error": "Query is required"}), 400
    
    # retrive relevant chunks
    chunks = rag.retrieve(user_query, top_k = 5)

    # generate answer groundede in chunks
    answer = rag.generate(user_query, chunks)

    # score hallucination risk
    score_result = scorer.score(answer, chunks)

    return jsonify({"answer": answer, "sources": chunks, "hallucination": score_result})

@app.route("/api/.ingest", methods = ["POST"])
def ingest():
    data = request.json
    texts = data.get("texts", [])
    if not texts:
        return jsonify({"error": "No texts provided"}), 400
    count = rag.ingest(texts)
    return jsonify({"ingested": count})

@app.route("/api/health", methods = ["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True, port=5001)
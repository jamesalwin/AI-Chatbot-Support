# app.py
from flask import Flask, render_template, request, jsonify, session
from model import EmbeddingChatModel
import os, re
from uuid import uuid4

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET", "replace_this_for_prod")

# load model (embeddings.pkl should be in repo or created at runtime)
model = EmbeddingChatModel(embeddings_path=os.environ.get("EMBEDDINGS_PATH", "embeddings.pkl"))

conversation_memory = {}
ORDER_ID_REGEX = re.compile(r"\b([A-Za-z0-9\-]{5,})\b")

def ensure_session():
    if "sid" not in session:
        session["sid"] = str(uuid4())
    sid = session["sid"]
    if sid not in conversation_memory:
        conversation_memory[sid] = {"history": [], "last_tag": None}
    return sid

@app.route("/")
def index():
    ensure_session()
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    sid = ensure_session()
    data = request.get_json(force=True)
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"success": False, "error": "Empty message"}), 400

    mem = conversation_memory[sid]
    last_tag = mem.get("last_tag")

    # simple follow-up: order id after order_status
    if last_tag == "order_status":
        match = ORDER_ID_REGEX.search(message)
        if match:
            order_id = match.group(1)
            reply = f"Thanks — I found order **{order_id}**. Status: *In transit*. Estimated delivery: 2–4 business days."
            mem["history"].append(("user", message, None))
            mem["history"].append(("bot", reply, "order_status_followup"))
            mem["last_tag"] = "order_status_followup"
            return jsonify({"success": True, "tag": "order_status_followup", "response": reply, "confidence": 0.95})

    # use embedding model
    result = model.predict(message)
    mem["history"].append(("user", message, None))
    mem["history"].append(("bot", result["response"], result["tag"]))
    mem["last_tag"] = result["tag"]

    if result["confidence"] < 0.45:
        fallback = "Sorry — I didn't quite get that. Could you rephrase or give more details?"
        return jsonify({"success": True, "tag": "unknown", "response": fallback, "confidence": result["confidence"]})

    return jsonify({"success": True, "tag": result["tag"], "response": result["response"], "confidence": result["confidence"]})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

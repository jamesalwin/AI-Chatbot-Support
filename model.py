# model.py
import pickle, random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

class EmbeddingChatModel:
    def __init__(self, embeddings_path="embeddings.pkl"):
        with open(embeddings_path, "rb") as f:
            data = pickle.load(f)
        self.model_name = data.get("model_name", DEFAULT_MODEL_NAME)
        self.tags = data["tags"]
        self.intent_embeddings = data["embeddings"]
        self.responses = data["responses"]
        # lazy load embedder (small)
        self.embedder = SentenceTransformer(self.model_name)

    def predict(self, text):
        # embed and compute cosine similarity
        query_emb = self.embedder.encode([text], convert_to_numpy=True)[0].reshape(1, -1)
        sims = cosine_similarity(query_emb, self.intent_embeddings)[0]
        best_idx = int(np.argmax(sims))
        best_tag = self.tags[best_idx]
        raw_score = float(sims[best_idx])  # -1..1
        confidence = max(0.0, min(1.0, (raw_score + 1) / 2))
        resp_list = self.responses.get(best_tag, [])
        response = random.choice(resp_list) if resp_list else "Sorry, I don't have a response for that yet."
        return {"tag": best_tag, "response": response, "confidence": confidence}

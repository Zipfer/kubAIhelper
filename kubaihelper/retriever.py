import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from typing import List, Dict

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

class Retriever:
    """Encodes corpus and retrieves similar solutions for a given query."""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.ids = []
        self.solutions = []
        self.contexts = []
        self.embeddings = None

    def build_index(self, corpus: List[Dict], use_faiss: bool = True):
        self.contexts = [c["context"] for c in corpus]
        self.solutions = [c["solution"] for c in corpus]
        self.ids = [c["id"] for c in corpus]
        self.embeddings = self.model.encode(
            self.contexts, show_progress_bar=True, normalize_embeddings=True
        )

        if use_faiss and _HAS_FAISS:
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)
        else:
            self.index = None

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        if self.index:
            D, I = self.index.search(np.array([q_emb], dtype="float32"), top_k)
            return [
                {"id": self.ids[i], "score": float(D[0][j]), "solution": self.solutions[i]}
                for j, i in enumerate(I[0])
            ]
        sims = np.dot(self.embeddings, q_emb)
        idxs = np.argsort(-sims)[:top_k]
        return [{"id": self.ids[i], "score": float(sims[i]), "solution": self.solutions[i]} for i in idxs]


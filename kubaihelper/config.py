from pathlib import Path

class Config:
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CORPUS_PATH = Path("data/corpus.jsonl")
    FEEDBACK_PATH = Path("data/feedback.jsonl")
    INDEX_PATH = Path("data/index")
    FINE_TUNED_MODEL_DIR = Path("models/fine_tuned")


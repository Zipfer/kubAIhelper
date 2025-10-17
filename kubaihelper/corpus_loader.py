import json
from typing import List, Dict

class CorpusLoader:
    """Loads and parses the corpus of problems and solutions."""

    @staticmethod
    def load_jsonl(path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line.strip()) for line in f if line.strip()]

    @staticmethod
    def build_id_map(corpus: List[Dict]) -> Dict[str, Dict]:
        return {item["id"]: item for item in corpus}


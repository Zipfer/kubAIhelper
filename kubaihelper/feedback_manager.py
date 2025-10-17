import json
import os
import datetime
from typing import Dict, List

class FeedbackManager:
    """Handles human-in-the-loop feedback storage."""

    def __init__(self, feedback_path: str):
        self.feedback_path = feedback_path
        os.makedirs(os.path.dirname(feedback_path) or ".", exist_ok=True)

    def add_feedback(self, query: str, solution_id: str):
        record = {
            "query": query,
            "solution_id": solution_id,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        }
        with open(self.feedback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[+] Feedback recorded: {record}")

    def load_feedback(self) -> List[Dict]:
        if not os.path.exists(self.feedback_path):
            return []
        with open(self.feedback_path, "r", encoding="utf-8") as f:
            return [json.loads(line.strip()) for line in f if line.strip()]


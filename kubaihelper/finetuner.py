from sentence_transformers import SentenceTransformer, InputExample, losses
import torch
from typing import List, Dict

class FineTuner:
    """Fine-tunes embedding model based on human feedback."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def prepare_examples(self, feedback: List[Dict], corpus_map: Dict[str, Dict]) -> List[InputExample]:
        examples = []
        for fb in feedback:
            q = fb.get("query")
            sol_id = fb.get("solution_id")
            if not q or sol_id not in corpus_map:
                continue
            pos = corpus_map[sol_id]["solution"]
            examples.append(InputExample(texts=[q, pos]))
        return examples

    def train(self, examples: List[InputExample], output_dir: str, epochs: int = 1, batch_size: int = 16):
        model = SentenceTransformer(self.model_name)
        dataloader = torch.utils.data.DataLoader(examples, batch_size=batch_size, shuffle=True)
        loss_fn = losses.MultipleNegativesRankingLoss(model)

        model.fit(
            train_objectives=[(dataloader, loss_fn)],
            epochs=epochs,
            warmup_steps=max(100, int(len(dataloader) * epochs * 0.1)),
            show_progress_bar=True,
            output_path=output_dir,
        )
        print(f"[✓] Fine-tuned model saved to {output_dir}")


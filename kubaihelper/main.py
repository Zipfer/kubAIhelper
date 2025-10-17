import argparse
from kubaihelper.config import Config
from kubaihelper.corpus_loader import CorpusLoader
from kubaihelper.retriever import Retriever
from kubaihelper.feedback_manager import FeedbackManager
from kubaihelper.finetuner import FineTuner

def cli():
    parser = argparse.ArgumentParser(description="AI Kubernetes Helper CLI")
    parser.add_argument("--index", action="store_true", help="Build corpus index")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--feedback", action="store_true", help="Record feedback interactively")
    parser.add_argument("--finetune", action="store_true", help="Fine-tune model on feedback")
    args = parser.parse_args()

    if args.index:
        corpus = CorpusLoader.load_jsonl(Config.CORPUS_PATH)
        retriever = Retriever(Config.DEFAULT_MODEL)
        retriever.build_index(corpus)
        print("[✓] Index built successfully.")

    if args.query:
        corpus = CorpusLoader.load_jsonl(Config.CORPUS_PATH)
        retriever = Retriever(Config.DEFAULT_MODEL)
        retriever.build_index(corpus)
        results = retriever.search(args.query)
        print("\n--- Top Matches ---")
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r['score']:.3f}] {r['solution']}\n")

        if args.feedback:
            choice = input("Pick accepted result number (0 = none): ").strip()
            if choice.isdigit() and int(choice) > 0:
                chosen = results[int(choice) - 1]
                fb = FeedbackManager(Config.FEEDBACK_PATH)
                fb.add_feedback(args.query, chosen["id"])

    if args.finetune:
        fb = FeedbackManager(Config.FEEDBACK_PATH).load_feedback()
        corpus = CorpusLoader.load_jsonl(Config.CORPUS_PATH)
        corpus_map = CorpusLoader.build_id_map(corpus)
        tuner = FineTuner(Config.DEFAULT_MODEL)
        examples = tuner.prepare_examples(fb, corpus_map)
        tuner.train(examples, str(Config.FINE_TUNED_MODEL_DIR))

if __name__ == "__main__":
    cli()


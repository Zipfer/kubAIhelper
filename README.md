# AI Kubernetes Helper

An **AI-based assistant** that analyzes Kubernetes issues, searches similar problems from known sources (e.g., StackOverflow, man pages, internal logs), and **learns over time** from human feedback.

---

## Features

- Modular design (Retriever, Corpus, Feedback, FineTuner)
- Fast semantic search (FAISS + PyTorch embeddings)
- Human-in-the-loop self-improvement
- Easily extendable for Kubernetes diagnostics or other domains

---

## Project Structure
```
kubaihelper/
├── kubaihelper/
│ ├── corpus_loader.py
│ ├── retriever.py
│ ├── feedback_manager.py
│ ├── finetuner.py
│ └── main.py
├── data/
│ ├── corpus.jsonl
│ └── feedback.jsonl
├── requirements.txt
└── README.md
```


---

## Installation

```bash
git clone https://github.com/yourusername/kubaihelper.git
cd kubaihelper
pip install -r requirements.txt
```

## Usage

Build the index
```bash
python -m kubaihelper.main --index`
```
Search for solution
```bash
python -m kubaihelper.main --query "pod stuck in CrashLoopBackOff" --feedback
```
Fine-tune model based on feedback
```bash
python -m kubaihelper.main --finetune

```

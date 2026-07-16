# Customer Support AI

An experiment-driven learning repository for evaluating **local LLMs** on real AI product problems.

This is **not** a production application. It is a lab: small, modular experiments that build on each other over months — starting with customer support ticket classification.

---

## Why this exists

Shipping AI features well requires more than calling an API. You need to understand:

- How prompts behave on real inputs
- How models differ on the same task
- How to measure quality before users see it
- When embeddings, RAG, or human review are actually needed

This repo is a place to learn those product skills hands-on, using **local models via Ollama**, with a clean Python skeleton that stays easy to extend.

**Design principles**

- No overengineering
- No LangChain / LlamaIndex / FastAPI / React / Docker
- No vector databases or RAG (yet)
- Prompts live in files, config lives in `.env`
- One clear module per concern

---

## Current experiment

### Customer Support Ticket Classification

**Goal:** Given a support ticket, classify it into a category and return structured JSON:

```json
{
  "category": "",
  "confidence": 0.0,
  "reason": ""
}
```

**Status:** Project skeleton only. Classifier wiring exists; dataset loading and evaluation are intentionally not implemented yet.

---

## Future roadmap

Experiments will be added incrementally, roughly in this order:

| Stage | Focus |
|-------|--------|
| 1 | **Prompt Engineering** — iterate on `prompts/` and measure impact |
| 2 | **Model Comparison** — same task, different local models |
| 3 | **Few-shot Prompting** — add examples without changing code structure |
| 4 | **Embeddings** — similarity and clustering experiments |
| 5 | **RAG** — retrieval over a knowledge base |
| 6 | **Human-in-the-loop** — review, correct, and feed labels back |
| 7 | **AI Evaluation** — richer metrics, error analysis, reports |

Each stage should land as a small, reviewable change — not a rewrite.

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally
- A pulled model (default: `qwen2.5:7b`)

```bash
ollama pull qwen2.5:7b
```

---

## Setup

```bash
# Clone
git clone https://github.com/<your-username>/ai-product-lab.git
cd ai-product-lab

# Virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Dependencies
pip install -r requirements.txt

# Environment
cp .env.example .env
# Edit .env if your Ollama URL or model name differs
```

Never commit `.env`. Only `.env.example` is tracked.

---

## Run

From the project root:

```bash
python -m src.main
```

With no dataset yet, you should see:

```
No dataset found.
Project setup complete.
Next step: create the first dataset.
```

That means the skeleton is healthy: config loaded, Ollama reachable, prompt present.

---

## Project structure

```
ai-product-lab/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example          # Template for local config
├── config.py             # Loads all settings from .env
├── datasets/
│   ├── raw/              # Source datasets (you add these later)
│   └── processed/        # Cleaned / derived data
├── prompts/
│   └── classifier.txt    # Classification prompt template
├── outputs/
│   ├── predictions/      # Model outputs per run
│   ├── metrics/          # Numeric evaluation results
│   └── reports/          # Human-readable experiment notes
└── src/
    ├── __init__.py
    ├── main.py           # Entry point & startup checks
    ├── loader.py         # Dataset I/O (TODO)
    ├── classifier.py     # Ollama-backed classify(ticket)
    ├── evaluator.py      # Evaluation orchestration (TODO)
    ├── metrics.py        # accuracy, precision, recall, F1, … (TODO)
    └── utils.py          # Prompt loading, env helpers, logging
```

---

## Configuration

Everything configurable comes from `.env` (see `.env.example`):

| Variable | Purpose |
|----------|---------|
| `OLLAMA_BASE_URL` | Local Ollama server URL |
| `MODEL_NAME` | Model tag (e.g. `qwen2.5:7b`) |
| `TEMPERATURE` | Sampling temperature |
| `TOP_P` | Nucleus sampling |
| `MAX_TOKENS` | Max generated tokens |
| `PROMPT_PATH` | Path to prompt template |
| `DATASET_PATH` | Path to ticket dataset |
| `OUTPUT_DIR` | Root for predictions / metrics / reports |

`config.py` reads these via `python-dotenv`. No hardcoded secrets or model settings in source.

---

## How experiments will be added

1. **Add or change a prompt** under `prompts/` — do not hardcode prompts in Python.
2. **Add a dataset** under `datasets/raw/` when ready (none ships with this skeleton).
3. **Extend `loader.py`** to read that format.
4. **Run classification** via `classifier.classify`.
5. **Wire `evaluator.py` / `metrics.py`** once labels exist.
6. **Write results** under `outputs/` and note findings in `outputs/reports/`.

Prefer new files or small modules over growing a single script. Keep each experiment’s “why” in a short report so the lab stays readable months later.

---

## Classifier API (ready)

```python
from src.classifier import classify

result = classify("I was charged twice for my subscription.")
# -> {"category": "...", "confidence": 0.0, "reason": "..."}
```

Requires Ollama running and the model named in `.env` to be available locally.

---

## What is intentionally missing

- No sample / fake ticket dataset
- No evaluation implementation yet
- No RAG, embeddings, or vector DB
- No web UI or API server

Those come later, as explicit experiments — not as premature infrastructure.

---

## License

Add a license when you publish the repository publicly (e.g. MIT).

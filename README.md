# Customer Support AI

An experiment-driven repository for understanding how Large Language Models (LLMs) behave on a real customer support classification problem.

This project is not intended to become a production customer support system.

Instead, it serves as a research playground for investigating how AI systems make decisions, where they fail, and how different techniques improve their performance.

---

## Project Goal

The objective of this repository is to answer questions such as:

- Can a local LLM classify customer support tickets accurately?
- How much does prompt engineering improve performance?
- Which local model performs best?
- When do embeddings help?
- When is RAG actually useful?
- How should AI systems be evaluated?
- When should humans remain in the loop?

Rather than building features immediately, every improvement is treated as an experiment.

---

## Current Progress

### ✅ Phase 1 – Baseline Inference

Completed

Current capabilities:

- Local LLM inference using Ollama
- Qwen 2.5 support
- Customer support dataset
- Ticket taxonomy
- Batch inference pipeline
- Structured JSON responses
- Prediction exports (`run_001.csv`, `run_002.csv`)

Current pipeline:

```
Customer Tickets
        │
        ▼
Dataset Loader
        │
        ▼
Local LLM (Qwen)
        │
        ▼
Structured Prediction
        │
        ▼
Prediction CSV
```

Current output:

```csv
ticket,actual_label,predicted_label,reason
```

Each execution creates a new prediction file instead of overwriting previous runs.

---

## Upcoming Work

The project will evolve through small, measurable experiments.

### Phase 2

Evaluation

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

### Phase 3

Failure Analysis

Questions to investigate:

- Which categories fail most often?
- Why do failures occur?
- Prompt issue or model issue?
- Does missing context cause failures?

---

### Phase 4

Prompt Engineering

- Zero-shot
- One-shot
- Few-shot
- Structured prompting

Measure the impact of prompt quality.

---

### Phase 5

Model Comparison

Compare identical experiments across:

- Qwen
- Mistral
- Gemma
- Llama

---

### Phase 6

Embeddings

Explore:

- Semantic similarity
- Duplicate ticket detection
- Ticket clustering
- Trend detection

---

### Phase 7

Retrieval Augmented Generation (RAG)

Investigate:

- Retrieval quality
- Context engineering
- Context size
- Retrieval failures

---

### Phase 8

Human-in-the-loop

Explore confidence thresholds for:

- Auto classification
- Human review
- Hybrid workflows

---

## Technology Stack

- Python
- Ollama
- Qwen 2.5
- pandas
- tqdm
- python-dotenv

The project intentionally avoids heavy frameworks so the focus remains on understanding AI systems rather than tooling.

---

## Running the Project

Clone the repository.

```bash
git clone git@github.com:SrivatsavChinta/customer-support-ai.git

cd customer-support-ai
```

Create a virtual environment.

```bash
python3 -m venv .venv

source .venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Copy the environment variables.

```bash
cp .env.example .env
```

Download the default model.

```bash
ollama pull qwen2.5:7b
```

Run the experiment.

```bash
python -m src.main
```

---

## Project Structure

```
customer-support-ai/

├── datasets/
│   ├── raw/
│   │   ├── tickets_v1.csv
│   │   └── taxonomy_v1.json
│   └── processed/
│
├── outputs/
│   ├── predictions/
│   │   ├── run_001.csv
│   │   └── run_002.csv
│   ├── metrics/
│   └── reports/
│
├── prompts/
│   └── classifier.txt
│
├── src/
│   ├── main.py
│   ├── loader.py
│   ├── classifier.py
│   ├── evaluator.py
│   ├── metrics.py
│   └── utils.py
│
├── config.py
├── README.md
├── requirements.txt
└── .env.example
```

---

## Current Learnings

Some early observations from the baseline experiment:

- Local LLMs can successfully classify customer support tickets.
- Models often understand the intent but do not always follow a predefined taxonomy.
- Prompt quality significantly influences structured outputs.
- Building an evaluation pipeline is essential before comparing models.

---

## Roadmap

- [x] Project setup
- [x] Local LLM integration
- [x] Customer support dataset
- [x] Baseline inference pipeline
- [ ] Evaluation pipeline
- [ ] Failure analysis
- [ ] Prompt engineering
- [ ] Model comparison
- [ ] Embeddings
- [ ] RAG
- [ ] Human-in-the-loop
- [ ] Final benchmark report

---

## Purpose

This repository is less about building customer support software and more about learning how AI systems behave.

The end goal is to develop a practical understanding of designing, evaluating, and improving AI-powered product experiences through iterative experimentation.
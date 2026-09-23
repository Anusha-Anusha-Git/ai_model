# engineering-ai — Stage 1: Local Model

## Setup

```bash
cd engineering-ai
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python -m app.models.llm
```

First run downloads ~3GB of model weights from Hugging Face into
`~/.cache/huggingface`.

## Success criteria

- No errors during model load
- Response is coherent (accuracy isn't the point yet — RAG and the
  calculation modules handle that in later stages)
- You get a baseline tokens/sec and RAM usage number, needed for the
  Stage 11 deployment sizing decisions

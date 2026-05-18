# Hate Speech Detection (Multilingual) – Flask + SQLite + Tailwind

A simple web app to detect hate/abusive content in multiple languages using a pretrained multilingual toxicity model. Built with Flask, SQLite, and Tailwind CSS.

## Features
- Analyze text for indicators like identity attack, threat, insult, obscene, and overall toxicity
- Multilingual support via a pretrained model
- Save analyses in SQLite and view recent history
- Minimal UI styled with Tailwind (via CDN)

## Tech Stack
- Backend: Flask (Python)
- Model: [Detoxify](https://github.com/unitaryai/detoxify) (multilingual variant, based on XLM-R)
- Database: SQLite (local file `app.db`)
- UI: Tailwind CSS (CDN)

## Setup

1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app

```bash
python app.py
```

The server runs at `http://localhost:3000`. On the first analysis, the model will download weights (one-time) and may take a minute.

## Notes
- This is a demo. For production, consider adding request limits, input sanitation, and model warmup.
- Classification decision is derived from available toxicity sub-scores; tune thresholds in `app.py` as needed.
- If PyTorch installation fails on your platform, consult the official PyTorch install guide for a compatible wheel.

## Project Structure
```
app.py
app.db                # created at runtime
templates/
  base.html
  index.html
  history.html
static/
  app.js
requirements.txt
README.md
```

## License
MIT 
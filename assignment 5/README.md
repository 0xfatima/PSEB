# AI Engineering Program – Module 4: Deep Learning (Part 3) – Train Better and Ship

This repository contains two projects:

| Folder | Project | Status |
| --- | --- | --- |
| [`Project 5`](Project%205/) | **Live Training Dashboard** – City Hospital breast-cancer screening model with live learning curves in Streamlit | Compulsory |
| [`Project 4`](Project%204/) | **Ship the Reader** – digit reader behind a FastAPI endpoint with a Streamlit page | Chosen from Projects 1–4 |

Each folder has its own `README.md` (objectives, implementation, setup, results, limitations), `requirements.txt` and a `screenshots/` folder.

## Quick start

```bash
# Project 5
cd "Project 5"
pip install -r requirements.txt
streamlit run app.py

# Project 4 (three terminals)
cd "Project 4"
pip install -r requirements.txt
python train.py
uvicorn api:app --reload
streamlit run ui.py
```

Both projects use datasets built into scikit-learn, so no downloads are needed and everything runs on a laptop CPU.

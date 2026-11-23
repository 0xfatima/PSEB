# Project 4 – Ship the Reader (Branch Staff)

> *"Branch staff want to try the digit reader in a browser tomorrow. Give them a page, give IT an API, and write down what it cannot do."*

**What it does, in one sentence:** a small neural network reads scanned 8×8 handwritten digits; IT calls it through a FastAPI endpoint and branch staff try it on a Streamlit web page.

---

## 1. Objectives

- Train a digit reader and save it with a config file, so every result is reproducible.
- Put the model behind an API (`POST /predict`) that loads the model **once** and never crashes on bad input.
- Give non-technical staff a browser page that calls the API.
- Write a README that lets a classmate run the whole project **from this file alone**, including honest limitations.

## 2. Architecture

```
train.py ──► model.pt + config.json ──► api.py (FastAPI, POST /predict) ◄── ui.py (Streamlit page)
             saved once                 loads the model once                 64 pixels as JSON
                                        returns digit + confidence           shows the answer
```

| Item | Value |
| --- | --- |
| Model | Multi-layer perceptron: 64 inputs → 64 hidden (ReLU) → 10 outputs |
| Parameters | 4,810 |
| Input | 64 pixel values of one 8×8 grayscale digit, each 0 to 16 (scaled to 0–1 inside) |
| Output | Predicted digit (0–9), confidence (softmax probability), all 10 probabilities |
| Training | Adam, learning rate 0.001, 200 full-batch epochs, seed 0 |
| Data | scikit-learn `load_digits`: 1,797 images, split 80 / 20 stratified → 1,437 train / 360 test |
| Compute | Laptop CPU, trains in a few seconds; no GPU needed |

## 3. Project structure

```
Project 4/
├── model.py           # The network, shared by train.py and api.py
├── train.py           # Trains, evaluates, saves model.pt, config.json, test_indices.json, runs.csv, curves.png
├── api.py             # FastAPI app: GET /health, POST /predict
├── ui.py              # Streamlit page for branch staff
├── tests/
│   └── test_api.py    # API tests with FastAPI's TestClient
├── requirements.txt
├── screenshots/
└── README.md
```

## 4. Implementation

**`train.py`**
1. Loads the digits data and scales pixels to 0–1.
2. Splits 80 / 20, stratified, seed 0, and saves the test indices so the page only shows unseen images.
3. Trains the MLP and records training and test loss every epoch (saved as `curves.png`).
4. Evaluates: test accuracy, recall per digit, confusion matrix and the top mix-ups.
5. Saves `model.pt` (weights) and `config.json` (layer sizes, settings, seed, results) and appends a row to `runs.csv` for experiment tracking.

**`api.py`**
- Loads `config.json` and `model.pt` once at start-up and switches the model to `eval()` mode.
- `POST /predict` takes `{"pixels": [64 numbers]}`.
- Input checks return a clear JSON error with status 422 instead of crashing: wrong number of values, values outside 0–16, or non-numbers.
- Returns `{"digit", "confidence", "probabilities"}`.
- `GET /health` reports the model and its test accuracy. FastAPI also provides a free test page at `/docs`.

**`ui.py`**
- A slider to pick a test image, the 8×8 picture, and a **Predict** button that calls the API.
- Shows the predicted digit, confidence, true label, a correct/wrong message, a low-confidence warning (< 60 %) and a bar chart of all ten probabilities.
- Shows whether the API is online, and has a "bad input" button to demonstrate the input check.
- The page never loads the model itself, so the same API can serve a phone app later.

## 5. Setup

Requires **Python 3.9+**.

```bash
# 1. clone and enter the folder
git clone <your-repo-url>
cd "<your-repo>/Project 4"

# 2. (optional) virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate

# 3. install
pip install -r requirements.txt
```

## 6. How to run – three terminals, in this order

**Terminal 1 – train (one command):**

```bash
python train.py
```

**Terminal 2 – start the API:**

```bash
uvicorn api:app --reload
```

Open <http://localhost:8000/docs>, expand `POST /predict`, press **Try it out** and **Execute** (an example image of a 0 is pre-filled).

Or from the command line:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
     -d "{\"pixels\": [0,0,5,13,9,1,0,0,0,0,13,15,10,15,5,0,0,3,15,2,0,11,8,0,0,4,12,0,0,8,8,0,0,5,8,0,0,9,8,0,0,4,11,0,1,12,7,0,0,2,14,5,10,12,0,0,0,0,6,13,10,0,0,0]}"
```

**Terminal 3 – start the page:**

```bash
streamlit run ui.py
```

A browser tab opens at <http://localhost:8501>. If the API runs elsewhere, set `DIGIT_API_URL` (e.g. `DIGIT_API_URL=http://192.168.1.10:8000 streamlit run ui.py`).

**Run the tests (after training):**

```bash
pytest -q
```

## 7. Results

Reference results from the course solution run (360 test images the model never trained on). Exact numbers can vary slightly with your PyTorch version; `train.py` prints yours and saves them in `config.json`.

**Test accuracy: 0.944**

| Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Recall | 1.00 | 0.91 | 0.89 | 0.90 | 0.97 | 0.98 | 0.98 | 0.97 | 0.95 | 0.90 |

**API replies**

| Request | Reply |
| --- | --- |
| Test image 0 (a real "0") | `{"digit": 0, "confidence": 0.943, ...}` |
| Only 10 pixel values | `{"error": "send exactly 64 pixel values (got 10)"}` (status 422) |

What the average hides: digit **2** is found only 89 % of the time, and the confusion matrix shows mix-ups such as **9 read as 7** and **9 read as 5**.

### Screenshots

| Training output | Learning curves |
| --- | --- |
| ![Training](screenshots/01_training.png) | ![Curves](screenshots/02_curves.png) |

| API `/docs` – good input | API `/docs` – bad input |
| --- | --- |
| ![API docs](screenshots/03_api_docs.png) | ![Bad input](screenshots/04_api_bad_input.png) |

| Streamlit page | Tests |
| --- | --- |
| ![UI](screenshots/05_ui_prediction.png) | ![Tests](screenshots/07_tests.png) |

## 8. Limitations

1. **Data: 8×8 scanned digits only.** The model was trained on tiny, centred, 0–16 grayscale scans. Photos of handwriting, other resolutions, or digits that are shifted or rotated will fail (shifting by one pixel alone drops accuracy sharply, as Project 3 showed).
2. **Known errors.** About 1 in 18 test digits is wrong. Digit 2 is the weakest class (89 % recall) and 9 is often read as 7 or 5. Low-confidence answers should be checked by a person.
3. **Small, old dataset.** 1,797 images from a few dozen writers; the test set has only 360 images, so the 0.944 accuracy is an estimate with a margin of a few points.
4. **Not production-hardened.** No authentication, rate limiting, logging or HTTPS; one prediction per request; the API is meant for a local network trial, not the public internet.
5. **Confidence is not calibrated.** A "90 % confident" answer is not guaranteed to be right 90 % of the time.

## 9. Tech stack

Python · PyTorch · scikit-learn · FastAPI · Uvicorn · Streamlit · pytest

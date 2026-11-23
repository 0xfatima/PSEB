# Project 5 – Live Training Dashboard (City Hospital)

> *"Our trial screening model reaches a training loss of 0.000, and the doctors do not trust it. Build a page where our data team can watch training and validation loss live, fix the curves, and see how many cancers the model finds."*

A Streamlit dashboard that trains a breast-cancer screening network **live**, redraws the training and validation loss curves while it learns, diagnoses the curves (underfitting / overfitting / healthy), keeps every experiment in a downloadable table, lets you check single patients and run a one-time final test.

---

## 1. Objectives

- Watch training and validation loss **while** the model trains, not after.
- Turn the "training loss 0.000" model into one with a **healthy** learning curve, changing one setting at a time.
- Report what matters to doctors: **cancers found** and **false alarms**, not only accuracy.
- Keep every experiment reproducible (settings + seed) and exportable as CSV.
- Evaluate the chosen model **once** on a held-out test set and be honest about its limits.

## 2. Dataset

| Item | Value |
| --- | --- |
| Source | Wisconsin Diagnostic Breast Cancer data, built into scikit-learn (`load_breast_cancer`) |
| Patients | 569 real patients, 30 measurements each (radius, texture, perimeter, area, …) |
| Label | 1 = malignant, 0 = benign (scikit-learn's labels are flipped in code) |
| Split | 60 / 20 / 20 → 341 train / 114 validation / 114 test, stratified, seed 0 |
| Scaling | `StandardScaler` fitted on the **training rows only**, then applied to validation and test |

## 3. Project structure

```
Project 5/
├── app.py             # Streamlit dashboard (the page)
├── trainer.py         # Training engine: data, model, train loop, diagnosis, evaluation
├── requirements.txt
├── REPORT.md          # Experiments, recommendation, final test, why it is not ready
├── screenshots/       # Screenshots of the running dashboard
└── README.md
```

## 4. Implementation

### Workflow

```
Sidebar settings ──► trainer.train() ──► on_epoch callback ──► diagnose() ──► Results
                     one epoch at a time   redraws the chart     reads curves    table + check a patient
```

### `trainer.py` – the engine (no Streamlit inside)

- **`load_data(scale, seed)`** – loads and splits the data, optionally scales it (scaler fitted on training rows only).
- **`build_model(hidden, dropout)`** – `30 → hidden → hidden → 2` with ReLU and optional dropout after each hidden layer.
- **`train(cfg, data, on_epoch)`** – full-batch training with `AdamW` (weight decay as a setting). After every epoch it records both losses with `model.eval()` (dropout off) and calls `on_epoch(epoch, hist)`. It keeps the best checkpoint (lowest validation loss) and, if early stopping is on, stops after `patience` epochs without improvement and reloads the best weights.
- **`diagnose(hist)`** – the Part 2 rule:
  - training loss still over **0.3** → **underfitting**
  - validation loss ends **30 % above its best** → **overfitting**
  - otherwise → **healthy**
  - extra check: validation loss jumping up by more than 50 % between epochs three or more times → **unstable** (usually unscaled inputs or a too-high learning rate)
- **`evaluate(model, X, y, threshold)`** – accuracy, cancers found, cancers missed, false alarms, confusion matrix.
- Running `python trainer.py` reproduces the five experiments from the course slides in the terminal.

**Why a callback?** `trainer.py` knows nothing about Streamlit, so the same engine runs in the dashboard, a notebook or a test.

### `app.py` – the page

1. **Sidebar:** neurons per layer, dropout, weight decay, learning rate, max epochs, early stopping + patience, scaling on/off, decision threshold, seed, and a **Train** button.
2. **Live chart:** an `st.empty()` placeholder redrawn every 5 epochs, plus a progress bar showing the current losses.
3. **Diagnosis:** green (healthy), yellow (over/underfitting) or red (unstable) message explaining the curve.
4. **Four metrics:** epochs run, best validation loss (and its epoch), cancers found, false alarms.
5. **Experiments table:** every run stored in `st.session_state`, downloadable as `experiments.csv`.
6. **Check a patient:** pick a validation patient, see P(malignant), the model's call, the true label and the raw measurements.
7. **Final test (run once):** a checkbox that evaluates the current model on the 114 held-out test patients and warns if you look more than once.

## 5. Setup

Requires **Python 3.9+**. Everything runs on a laptop CPU; no GPU or download of data is needed.

```bash
# 1. clone and enter the folder
git clone <your-repo-url>
cd "<your-repo>/Project 5"

# 2. (optional) virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate

# 3. install
pip install -r requirements.txt
```

## 6. How to run

**The dashboard**

```bash
streamlit run app.py
```

A browser tab opens at `http://localhost:8501`. Choose settings in the sidebar and press **Train**.

**The engine on its own (terminal table of the five experiments)**

```bash
python trainer.py
```

**Suggested experiment path – one change at a time**

| # | Change | What to look for |
| --- | --- | --- |
| 1 | Defaults: 256 neurons, no fixes | Training loss → 0, validation rises after ~epoch 23: overfitting |
| 2 | + dropout 0.3, weight decay 0.05 | Still overfitting: these alone barely move the curve |
| 3 | + early stopping (patience 20) | Stops near the best epoch: healthy |
| 4 | Neurons 256 → 16 (all fixes on) | Healthy, fewest false alarms |
| 5 | Same, scaling **off** | Curves jump around, stops early, finds fewer cancers |

## 7. Results

Reference results from the course solution run (validation set, 43 cancers among 114 patients). Your numbers can differ slightly between PyTorch versions and machines; replace them with your own after running.

| Run | Diagnosis | Cancers found | False alarms |
| --- | --- | --- | --- |
| 256 neurons, no fixes | overfitting | 41 of 43 | 3 |
| + dropout 0.3, decay 0.05 | overfitting | 40 of 43 | 2 |
| + early stopping (20) | healthy | 40 of 43 | 3 |
| **16 neurons + all fixes** | **healthy** | **40 of 43** | **1** |
| same, not scaled | jumps around | 33 of 43 | 2 |

**Final test, run once, 16 neurons + all fixes:** accuracy **0.956** · **41 of 42** cancers found · **4** false alarms.

**Key lessons**

- A training loss of 0.000 is a warning sign, not a success: the validation curve showed overfitting after about epoch 23.
- Dropout and weight decay alone did **not** fix it here; **early stopping plus a smaller model** did the real work.
- Without scaling the 30 measurements (which range from about 0.001 to 2,500) one optimiser step overshoots and the curves jump around.

The full write-up with the recommendation is in [REPORT.md](REPORT.md).

### Screenshots

| Overfitting (defaults) | Healthy (16 neurons + all fixes) |
| --- | --- |
| ![Overfitting run](screenshots/01_overfitting_run.png) | ![Healthy run](screenshots/02_healthy_run.png) |

| Not scaled | Experiments table |
| --- | --- |
| ![Not scaled](screenshots/03_not_scaled.png) | ![Experiments table](screenshots/04_experiments_table.png) |

| Check a patient | Final test |
| --- | --- |
| ![Check a patient](screenshots/05_check_patient.png) | ![Final test](screenshots/06_final_test.png) |

## 8. Limitations

- **One small dataset.** 569 patients from one source; the model has never seen data from City Hospital's own scanners or population.
- **It still misses cancers.** One cancer was missed in the final test. For screening, a missed cancer costs far more than a false alarm, so the 0.5 threshold should be lowered and chosen with the doctors.
- **Small test set.** 114 test patients means the 0.956 accuracy has a wide margin of error (roughly ± 4 points).
- **Measurements, not images.** The model uses 30 pre-computed measurements of cell nuclei; it cannot read a scan by itself.
- **Not a medical device.** A doctor must review every result. No clinical validation, calibration or fairness check across patient groups has been done.
- **Dashboard scope.** Experiments live in the browser session only (download the CSV to keep them); training is full-batch on CPU, which is fine for 341 rows but would not scale to large datasets.

## 9. Tech stack

Python · PyTorch · scikit-learn · pandas · Streamlit

"""
trainer.py - the training engine for the City Hospital live training dashboard.

This file knows nothing about Streamlit. It loads the data, builds the model,
trains it one epoch at a time and calls back `on_epoch(epoch, hist)` after every
epoch, so the same engine works in the dashboard, in a notebook or in a test.

Run it on its own to reproduce the five experiments from the course slides:
    python trainer.py
"""

import copy

import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Default settings: the "big model, no fixes" starting point.
DEFAULTS = {
    "hidden": 256,          # neurons in each of the two hidden layers
    "dropout": 0.0,         # share of neurons switched off at each step
    "weight_decay": 0.0,    # AdamW weight decay
    "lr": 0.001,            # learning rate
    "epochs": 300,          # maximum number of epochs
    "early_stopping": False,
    "patience": 20,         # epochs without improvement before stopping
    "scale": True,          # standardise the 30 measurements
    "seed": 0,
    "threshold": 0.5,       # probability of malignant needed to flag a patient
}

# Diagnosis rule from Part 2 of the module
UNDERFIT_TRAIN_LOSS = 0.3   # training loss still above this -> underfitting
OVERFIT_RATIO = 1.3         # final val loss 30% above its best -> overfitting


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_data(scale=True, seed=0):
    """Load the Wisconsin breast cancer data and split it 60 / 20 / 20.

    569 patients, 30 measurements each. Label 1 = malignant, 0 = benign.
    (scikit-learn stores 0 = malignant, so the labels are flipped here.)
    The scaler is fitted on the training rows only.
    """
    ds = load_breast_cancer()
    X = ds.data.astype(np.float32)
    y = (ds.target == 0).astype(np.int64)  # malignant = 1

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)

    raw_va, raw_te = X_va.copy(), X_te.copy()   # unscaled, for display

    scaler = None
    if scale:
        scaler = StandardScaler().fit(X_tr)     # training rows only
        X_tr = scaler.transform(X_tr).astype(np.float32)
        X_va = scaler.transform(X_va).astype(np.float32)
        X_te = scaler.transform(X_te).astype(np.float32)

    return {
        "X_tr": torch.tensor(X_tr), "y_tr": torch.tensor(y_tr),
        "X_va": torch.tensor(X_va), "y_va": torch.tensor(y_va),
        "X_te": torch.tensor(X_te), "y_te": torch.tensor(y_te),
        "raw_va": raw_va, "raw_te": raw_te,
        "feature_names": list(ds.feature_names),
        "scaled": scale,
        "scaler": scaler,
    }


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_model(hidden=256, dropout=0.0, n_features=30):
    """Two hidden layers of `hidden` neurons, optional dropout, 2 outputs."""
    return nn.Sequential(
        nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, 2),
    )


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train(cfg, data, on_epoch=None):
    """Train with the settings in `cfg`.

    After every epoch both losses are appended to `hist` and
    `on_epoch(epoch, hist)` is called (epoch counts from 1).

    Returns (model, hist). With early stopping on, the model returned is the
    best checkpoint (lowest validation loss), not the last one.
    """
    cfg = {**DEFAULTS, **cfg}
    torch.manual_seed(cfg["seed"])

    model = build_model(cfg["hidden"], cfg["dropout"], data["X_tr"].shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    hist = {"train": [], "val": [], "best_epoch": None, "stopped_early": False}
    best, wait, best_state = float("inf"), 0, None

    for epoch in range(1, cfg["epochs"] + 1):
        # one training step on the full training set
        model.train()
        loss = loss_fn(model(data["X_tr"]), data["y_tr"])
        opt.zero_grad()
        loss.backward()
        opt.step()

        # measure both losses with dropout switched off
        model.eval()
        with torch.no_grad():
            hist["train"].append(loss_fn(model(data["X_tr"]), data["y_tr"]).item())
            hist["val"].append(loss_fn(model(data["X_va"]), data["y_va"]).item())

        # keep the best checkpoint in memory
        if hist["val"][-1] < best:
            best, wait = hist["val"][-1], 0
            best_state = copy.deepcopy(model.state_dict())
            hist["best_epoch"] = epoch
        else:
            wait += 1

        if on_epoch:
            on_epoch(epoch, hist)

        if cfg["early_stopping"] and wait >= cfg["patience"]:
            hist["stopped_early"] = True
            break

    if cfg["early_stopping"] and best_state is not None:
        model.load_state_dict(best_state)   # ship the best model, not the last
    model.eval()
    return model, hist


# --------------------------------------------------------------------------
# Reading the curves
# --------------------------------------------------------------------------
def diagnose(hist):
    """Read the curves and return (verdict, explanation).

    Rule from Part 2:
      * training loss still over 0.3           -> underfitting
      * validation ends 30% above its best      -> overfitting
      * otherwise                               -> healthy
    Extra check: validation loss that jumps up by more than 50% from one
    epoch to the next several times means training is unstable (usually
    because the inputs are not scaled or the learning rate is too high).
    """
    train_l, val_l = hist["train"], hist["val"]
    if not train_l:
        return "no data", "Nothing was trained."

    final_train, final_val, best_val = train_l[-1], val_l[-1], min(val_l)

    jumps = sum(1 for a, b in zip(val_l[:-1], val_l[1:]) if b > 1.5 * a)
    if jumps >= 3:
        return ("unstable",
                f"Validation loss jumped up sharply {jumps} times. The curves "
                "jump around: scale the inputs or lower the learning rate.")
    if final_train > UNDERFIT_TRAIN_LOSS:
        return ("underfitting",
                f"Training loss is still {final_train:.3f} (over "
                f"{UNDERFIT_TRAIN_LOSS}). The model has not learned the "
                "training data: train longer, raise the learning rate or use "
                "a bigger model.")
    if final_val > OVERFIT_RATIO * best_val:
        return ("overfitting",
                f"Validation loss ended at {final_val:.3f}, "
                f"{(final_val / best_val - 1) * 100:.0f}% above its best "
                f"({best_val:.3f} at epoch {hist['best_epoch']}). The model is "
                "memorising: use early stopping, a smaller model, dropout or "
                "weight decay.")
    return ("healthy",
            f"Both losses fell together. Best validation loss {best_val:.3f} "
            f"at epoch {hist['best_epoch']}.")


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def malignant_probability(model, X):
    """Probability of malignant (class 1) for every row of X."""
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(X), dim=1)[:, 1].numpy()


def evaluate(model, X, y, threshold=0.5):
    """Screening numbers: cancers found, false alarms, accuracy."""
    prob = malignant_probability(model, X)
    pred = (prob >= threshold).astype(int)
    y = y.numpy()
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "accuracy": round((tp + tn) / len(y), 3),
        "cancers_found": tp,
        "cancers_total": tp + fn,
        "missed": fn,
        "false_alarms": fp,
        "confusion": [[tn, fp], [fn, tp]],
    }


# --------------------------------------------------------------------------
# Command line: the five experiments from the slides
# --------------------------------------------------------------------------
EXPERIMENTS = [
    ("256 neurons, no fixes", {}),
    ("+ dropout 0.3, decay 0.05", {"dropout": 0.3, "weight_decay": 0.05}),
    ("+ early stopping (20)", {"dropout": 0.3, "weight_decay": 0.05,
                               "early_stopping": True, "patience": 20}),
    ("16 neurons + all fixes", {"hidden": 16, "dropout": 0.3, "weight_decay": 0.05,
                                "early_stopping": True, "patience": 20}),
    ("same, not scaled", {"hidden": 16, "dropout": 0.3, "weight_decay": 0.05,
                          "early_stopping": True, "patience": 20, "scale": False}),
]

if __name__ == "__main__":
    print(f"{'Run':28s} {'Diagnosis':13s} {'Epochs':>6s} {'Best val':>9s} "
          f"{'Cancers found':>14s} {'False alarms':>13s}")
    for name, changes in EXPERIMENTS:
        cfg = {**DEFAULTS, **changes}
        data = load_data(scale=cfg["scale"], seed=cfg["seed"])
        model, hist = train(cfg, data)
        verdict, _ = diagnose(hist)
        m = evaluate(model, data["X_va"], data["y_va"], cfg["threshold"])
        print(f"{name:28s} {verdict:13s} {len(hist['train']):6d} "
              f"{min(hist['val']):9.3f} "
              f"{m['cancers_found']:>6d} of {m['cancers_total']:<5d} "
              f"{m['false_alarms']:13d}")

    # final test, once, on the chosen settings
    cfg = {**DEFAULTS, **EXPERIMENTS[3][1]}
    data = load_data(scale=True, seed=0)
    model, _ = train(cfg, data)
    t = evaluate(model, data["X_te"], data["y_te"], cfg["threshold"])
    print(f"\nFinal test (16 neurons + all fixes): accuracy {t['accuracy']:.3f}, "
          f"{t['cancers_found']} of {t['cancers_total']} cancers found, "
          f"{t['false_alarms']} false alarms")

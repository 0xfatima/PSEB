"""
train.py - train the digit reader and save everything the API needs.

Saves:
    model.pt           the trained weights
    config.json        layer sizes, settings, seed and test results
    test_indices.json  which images are in the test set (used by ui.py)
    runs.csv           one row per training run (experiment tracking)
    curves.png         training and test loss per epoch

Run:
    python train.py
"""

import csv
import json
import os
from datetime import datetime

import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.model_selection import train_test_split

from model import build_model

config = {
    "model": "mlp",
    "sizes": [64, 64, 10],     # 64 pixels -> 64 hidden -> 10 digits
    "lr": 0.001,
    "epochs": 200,
    "seed": 0,
    "test_size": 0.2,
    "pixel_max": 16.0,         # digits pixels go from 0 to 16
}

# ------------------------------------------------------------------ data
torch.manual_seed(config["seed"])
digits = load_digits()
X = digits.data / config["pixel_max"]            # scale pixels to 0-1
y = digits.target
idx = np.arange(len(y))

idx_tr, idx_te = train_test_split(idx, test_size=config["test_size"],
                                  stratify=y, random_state=config["seed"])
X_tr = torch.tensor(X[idx_tr], dtype=torch.float32)
y_tr = torch.tensor(y[idx_tr])
X_te = torch.tensor(X[idx_te], dtype=torch.float32)
y_te = torch.tensor(y[idx_te])
print(f"Training images: {len(idx_tr)} · Test images: {len(idx_te)}")

# ------------------------------------------------------------------ train
model = build_model(config["sizes"])
opt = torch.optim.Adam(model.parameters(), lr=config["lr"])
loss_fn = nn.CrossEntropyLoss()
hist = {"train": [], "test": []}

for epoch in range(config["epochs"]):
    model.train()
    loss = loss_fn(model(X_tr), y_tr)
    opt.zero_grad()
    loss.backward()
    opt.step()

    model.eval()
    with torch.no_grad():
        hist["train"].append(loss_fn(model(X_tr), y_tr).item())
        hist["test"].append(loss_fn(model(X_te), y_te).item())
    if (epoch + 1) % 25 == 0:
        print(f"epoch {epoch + 1:3d}  train loss {hist['train'][-1]:.3f}  "
              f"test loss {hist['test'][-1]:.3f}")

# ------------------------------------------------------------------ evaluate
model.eval()
with torch.no_grad():
    pred = model(X_te).argmax(1).numpy()
test_acc = float((pred == y_te.numpy()).mean())
recall = recall_score(y_te.numpy(), pred, average=None)
cm = confusion_matrix(y_te.numpy(), pred)

print(f"\nTest accuracy: {test_acc:.3f}")
print("Recall per digit:", {d: round(float(r), 2) for d, r in enumerate(recall)})
print("Confusion matrix (rows = true digit, columns = predicted):")
print(cm)

# biggest mix-ups, for the limitations section
mixups = sorted(((int(cm[i, j]), i, j) for i in range(10) for j in range(10)
                 if i != j and cm[i, j] > 0), reverse=True)[:5]
for n, true_d, pred_d in mixups:
    print(f"  {true_d} read as {pred_d}: {n} times")

# ------------------------------------------------------------------ save
here = os.path.dirname(os.path.abspath(__file__))
torch.save(model.state_dict(), os.path.join(here, "model.pt"))

config.update({
    "test_acc": round(test_acc, 3),
    "per_class_recall": [round(float(r), 2) for r in recall],
    "top_mixups": [{"true": t, "predicted": p, "count": n} for n, t, p in mixups],
    "n_train": int(len(idx_tr)),
    "n_test": int(len(idx_te)),
    "final_train_loss": round(hist["train"][-1], 4),
    "final_test_loss": round(hist["test"][-1], 4),
    "trained_at": datetime.now().isoformat(timespec="seconds"),
    "torch_version": torch.__version__,
})
with open(os.path.join(here, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

with open(os.path.join(here, "test_indices.json"), "w") as f:
    json.dump([int(i) for i in idx_te], f)

runs_path = os.path.join(here, "runs.csv")
new_file = not os.path.exists(runs_path)
with open(runs_path, "a", newline="") as f:
    w = csv.writer(f)
    if new_file:
        w.writerow(["date", "model", "sizes", "lr", "epochs", "seed", "test_acc"])
    w.writerow([config["trained_at"], config["model"], config["sizes"],
                config["lr"], config["epochs"], config["seed"], config["test_acc"]])

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.plot(hist["train"], color="#E8833A", label="training loss")
    plt.plot(hist["test"], color="#3A7BD5", linestyle="--", label="test loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Digit reader · test accuracy {test_acc:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(here, "curves.png"), dpi=120)
    print("Saved curves.png")
except ImportError:
    print("matplotlib not installed: skipped curves.png")

print("Saved model.pt, config.json, test_indices.json and runs.csv")

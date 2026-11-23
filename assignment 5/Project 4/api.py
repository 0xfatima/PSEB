"""
api.py - the digit reader behind a FastAPI endpoint.

The model is loaded ONCE when the API starts (loading is slow, predicting is fast).

Run:
    uvicorn api:app --reload
Then open http://localhost:8000/docs to try it in the browser.
"""

import json
import os

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from model import build_model

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.pt")
CONFIG_PATH = os.path.join(HERE, "config.json")

if not (os.path.exists(MODEL_PATH) and os.path.exists(CONFIG_PATH)):
    raise RuntimeError("model.pt or config.json not found. Run `python train.py` first.")

# ------------------------------------------------------------------ load once
with open(CONFIG_PATH) as f:
    config = json.load(f)

model = build_model(config["sizes"])
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

N_PIXELS = config["sizes"][0]           # 64
PIXEL_MAX = config.get("pixel_max", 16.0)

app = FastAPI(
    title="Digit reader API",
    description="Send the 64 pixel values (0 to 16) of one 8x8 scanned digit; "
                "get back the digit and the model's confidence.",
    version="1.0",
)


class Image(BaseModel):
    pixels: list[float]                 # 64 numbers, 0 to 16

    model_config = {
        "json_schema_extra": {
            "example": {"pixels": [0, 0, 5, 13, 9, 1, 0, 0, 0, 0, 13, 15, 10, 15, 5, 0,
                                   0, 3, 15, 2, 0, 11, 8, 0, 0, 4, 12, 0, 0, 8, 8, 0,
                                   0, 5, 8, 0, 0, 9, 8, 0, 0, 4, 11, 0, 1, 12, 7, 0,
                                   0, 2, 14, 5, 10, 12, 0, 0, 0, 0, 6, 13, 10, 0, 0, 0]}
        }
    }


def error(message, status=422):
    return JSONResponse(status_code=status, content={"error": message})


@app.get("/")
def root():
    return {"message": "Digit reader API. POST 64 pixel values to /predict. "
                       "Interactive docs at /docs."}


@app.get("/health")
def health():
    return {"status": "ok", "model": config.get("model", "mlp"),
            "sizes": config["sizes"], "test_acc": config.get("test_acc")}


@app.post("/predict")
def predict(img: Image):
    # ---- input checks: a clear error, never a crash
    if len(img.pixels) != N_PIXELS:
        return error(f"send exactly {N_PIXELS} pixel values "
                     f"(got {len(img.pixels)})")
    if any(p != p for p in img.pixels):                 # NaN check
        return error("pixel values must be numbers, not NaN")
    if any(p < 0 or p > PIXEL_MAX for p in img.pixels):
        return error(f"pixel values must be between 0 and {PIXEL_MAX:g}")

    # ---- predict
    x = torch.tensor(img.pixels, dtype=torch.float32).reshape(1, N_PIXELS) / PIXEL_MAX
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]

    return {
        "digit": int(probs.argmax()),
        "confidence": round(float(probs.max()), 3),
        "probabilities": [round(float(p), 3) for p in probs],
    }

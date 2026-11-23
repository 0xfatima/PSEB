"""
ui.py - a Streamlit page for branch staff to try the digit reader.

The page only talks to the API over HTTP; it never loads the model itself,
so the same API can later serve a phone app or a website.

Run (in a second terminal, with the API already running):
    streamlit run ui.py
"""

import json
import os

import pandas as pd
import requests
import streamlit as st
from sklearn.datasets import load_digits

API_URL = os.environ.get("DIGIT_API_URL", "http://localhost:8000")
HERE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Digit reader", page_icon="🔢")
st.title("🔢 Digit reader")
st.caption("Pick a scanned 8×8 digit, press Predict, and the page asks the API.")


@st.cache_data
def get_digits():
    d = load_digits()
    path = os.path.join(HERE, "test_indices.json")
    if os.path.exists(path):
        with open(path) as f:
            indices = json.load(f)
        source = "test set (images the model never trained on)"
    else:
        indices = list(range(len(d.target)))
        source = "all images (run train.py to restrict to the test set)"
    return d.images, d.data, d.target, indices, source


images, data, target, indices, source = get_digits()

# ---------------------------------------------------------------- API status
try:
    h = requests.get(f"{API_URL}/health", timeout=3).json()
    st.success(f"API online at {API_URL} · model test accuracy {h.get('test_acc')}")
except requests.exceptions.RequestException:
    st.error(f"API not reachable at {API_URL}. Start it first: "
             "`uvicorn api:app --reload`")

# ---------------------------------------------------------------- pick an image
st.write(f"Images from the {source}.")
k = st.slider("Pick a test image", 0, len(indices) - 1, 0)
i = indices[k]

left, right = st.columns([1, 2])
with left:
    st.image(images[i] / 16.0, width=160, clamp=True,
             caption=f"Image #{i} (8×8 pixels)")

with right:
    if st.button("Predict", type="primary"):
        try:
            r = requests.post(f"{API_URL}/predict",
                              json={"pixels": data[i].tolist()}, timeout=5)
            reply = r.json()
            if r.ok:
                true_label = int(target[i])
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted digit", reply["digit"])
                c2.metric("Confidence", f"{reply['confidence']:.1%}")
                c3.metric("True label", true_label)
                if reply["digit"] == true_label:
                    st.success("Correct.")
                else:
                    st.error("Wrong: this is one of the model's known mistakes.")
                if reply["confidence"] < 0.6:
                    st.warning("Low confidence: a person should check this digit.")
                st.bar_chart(pd.DataFrame({"probability": reply["probabilities"]},
                                          index=[str(d) for d in range(10)]))
                with st.expander("Raw API reply"):
                    st.json(reply)
            else:
                st.error(reply.get("error", reply))
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the API: {e}")

# ---------------------------------------------------------------- bad input demo
with st.expander("Try a bad input (shows the API's input check)"):
    if st.button("Send only 10 pixel values"):
        try:
            r = requests.post(f"{API_URL}/predict",
                              json={"pixels": data[i][:10].tolist()}, timeout=5)
            st.write(f"Status code: {r.status_code}")
            st.json(r.json())
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the API: {e}")

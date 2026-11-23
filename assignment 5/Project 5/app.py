"""
app.py - City Hospital live training dashboard (Streamlit).

Run:
    streamlit run app.py

Settings live in the sidebar. Press "Train" and watch training and validation
loss redraw every 5 epochs, read the diagnosis, compare runs in the
experiments table, check single patients and, once, run the final test.
"""

from datetime import datetime

import pandas as pd
import streamlit as st

import trainer

st.set_page_config(page_title="City Hospital · Live Training Dashboard",
                   page_icon="🩺", layout="wide")

COLORS = ["#E8833A", "#3A7BD5"]  # training loss = orange, validation = blue
REDRAW_EVERY = 5


@st.cache_data(show_spinner=False)
def get_data(scale, seed):
    return trainer.load_data(scale=scale, seed=seed)


def loss_frame(hist):
    df = pd.DataFrame({"training loss": hist["train"],
                       "validation loss": hist["val"]})
    df.index = range(1, len(df) + 1)
    df.index.name = "epoch"
    return df


# ---------------------------------------------------------------- state
for key, default in [("runs", []), ("model", None), ("hist", None),
                     ("cfg", None), ("data", None), ("verdict", None),
                     ("final_test_runs", 0)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Settings")
hidden = st.sidebar.select_slider("Neurons per hidden layer",
                                  options=[8, 16, 32, 64, 128, 256], value=256)
dropout = st.sidebar.slider("Dropout", 0.0, 0.6, 0.0, 0.05)
weight_decay = st.sidebar.select_slider("Weight decay",
                                        options=[0.0, 0.001, 0.01, 0.05, 0.1],
                                        value=0.0)
lr = st.sidebar.select_slider("Learning rate",
                              options=[0.0001, 0.0003, 0.001, 0.003, 0.01],
                              value=0.001)
epochs = st.sidebar.slider("Maximum epochs", 50, 500, 300, 10)
early_stopping = st.sidebar.checkbox("Early stopping", value=False)
patience = st.sidebar.slider("Patience (epochs)", 5, 50, 20, 5,
                             disabled=not early_stopping)
scale = st.sidebar.checkbox("Scale the measurements", value=True,
                            help="StandardScaler fitted on the training rows only.")
threshold = st.sidebar.slider("Decision threshold (malignant if p ≥)",
                              0.1, 0.9, 0.5, 0.05)
seed = st.sidebar.number_input("Seed", 0, 9999, 0, 1)
train_clicked = st.sidebar.button("Train", type="primary", use_container_width=True)

cfg = {"hidden": hidden, "dropout": dropout, "weight_decay": weight_decay,
       "lr": lr, "epochs": epochs, "early_stopping": early_stopping,
       "patience": patience, "scale": scale, "seed": int(seed),
       "threshold": threshold}

# ---------------------------------------------------------------- header
st.title("🩺 City Hospital · Live training dashboard")
st.caption("Breast cancer screening model · 569 real patients (Wisconsin data), "
           "30 measurements each · split 341 / 114 / 114, stratified · "
           "1 = malignant")

st.subheader("Learning curves")
chart = st.empty()          # redrawn in place -> looks live
status = st.empty()

# ---------------------------------------------------------------- training
if train_clicked:
    data = get_data(scale, int(seed))
    bar = st.progress(0.0, text="Starting…")

    def redraw(epoch, hist):
        bar.progress(min(epoch / cfg["epochs"], 1.0),
                     text=f"Epoch {epoch} / {cfg['epochs']} · "
                          f"train {hist['train'][-1]:.3f} · "
                          f"val {hist['val'][-1]:.3f}")
        if epoch % REDRAW_EVERY == 0 or epoch == 1:
            chart.line_chart(loss_frame(hist), color=COLORS)

    model, hist = trainer.train(cfg, data, on_epoch=redraw)
    bar.empty()

    verdict, explanation = trainer.diagnose(hist)
    m = trainer.evaluate(model, data["X_va"], data["y_va"], threshold)

    st.session_state.update(model=model, hist=hist, cfg=cfg, data=data,
                            verdict=(verdict, explanation), metrics=m)
    st.session_state.runs.append({
        "run": len(st.session_state.runs) + 1,
        "time": datetime.now().strftime("%H:%M:%S"),
        **{k: cfg[k] for k in ["hidden", "dropout", "weight_decay", "lr",
                               "early_stopping", "patience", "scale",
                               "threshold", "seed"]},
        "epochs_run": len(hist["train"]),
        "best_val_loss": round(min(hist["val"]), 3),
        "best_epoch": hist["best_epoch"],
        "final_train_loss": round(hist["train"][-1], 3),
        "diagnosis": verdict,
        "cancers_found": f"{m['cancers_found']} of {m['cancers_total']}",
        "false_alarms": m["false_alarms"],
        "val_accuracy": m["accuracy"],
    })

# ---------------------------------------------------------------- results
if st.session_state.hist is None:
    chart.info("Choose settings in the sidebar and press **Train**.")
else:
    hist = st.session_state.hist
    run_cfg = st.session_state.cfg
    m = st.session_state.metrics
    verdict, explanation = st.session_state.verdict

    chart.line_chart(loss_frame(hist), color=COLORS)
    note = " (stopped early)" if hist["stopped_early"] else ""
    status.caption(f"Orange = training loss · Blue = validation loss · "
                   f"best validation epoch: {hist['best_epoch']}{note}")

    show = {"healthy": st.success, "overfitting": st.warning,
            "underfitting": st.warning, "unstable": st.error}
    show.get(verdict, st.info)(f"**Diagnosis: {verdict}.** {explanation}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Epochs run", len(hist["train"]))
    c2.metric("Best validation loss", f"{min(hist['val']):.3f}",
              help=f"at epoch {hist['best_epoch']}")
    c3.metric("Cancers found (validation)",
              f"{m['cancers_found']} of {m['cancers_total']}",
              help=f"{m['missed']} missed")
    c4.metric("False alarms (validation)", m["false_alarms"])

# ---------------------------------------------------------------- experiments
st.subheader("Experiments")
if st.session_state.runs:
    runs_df = pd.DataFrame(st.session_state.runs)
    st.dataframe(runs_df, use_container_width=True, hide_index=True)
    c1, c2 = st.columns([1, 1])
    c1.download_button("Download experiments as CSV",
                       runs_df.to_csv(index=False).encode("utf-8"),
                       file_name="experiments.csv", mime="text/csv")
    if c2.button("Clear table"):
        st.session_state.runs = []
        st.rerun()
    st.caption("Change one setting at a time and compare the rows.")
else:
    st.caption("Every run you train is added here.")

# ---------------------------------------------------------------- check a patient
if st.session_state.model is not None:
    st.subheader("Check a patient")
    data = st.session_state.data
    model = st.session_state.model
    n_val = len(data["y_va"])
    i = st.number_input(f"Validation patient (0 – {n_val - 1}); the model never "
                        "trained on these", 0, n_val - 1, 0, 1)
    p = float(trainer.malignant_probability(model, data["X_va"][i:i + 1])[0])
    truth = "malignant" if int(data["y_va"][i]) == 1 else "benign"
    flagged = p >= st.session_state.cfg["threshold"]

    c1, c2, c3 = st.columns(3)
    c1.metric("P(malignant)", f"{p:.3f}")
    c2.metric("Model says", "malignant – refer" if flagged else "benign")
    c3.metric("True label", truth)
    if flagged != (truth == "malignant"):
        st.warning("The model is wrong on this patient.")

    with st.expander("Show this patient's 30 measurements"):
        st.dataframe(pd.DataFrame({"measurement": data["feature_names"],
                                   "value": data["raw_va"][i]}),
                     hide_index=True, use_container_width=True)

    # ------------------------------------------------------------ final test
    st.subheader("Final test (run once)")
    st.caption("The 114 test patients are kept apart. Choose your settings "
               "using the validation numbers first, then tick this once.")
    if st.checkbox("Run the final test on the current model"):
        if "final_shown" not in st.session_state:
            st.session_state.final_test_runs += 1
            st.session_state.final_shown = True
        t = trainer.evaluate(model, data["X_te"], data["y_te"],
                             st.session_state.cfg["threshold"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Test accuracy", f"{t['accuracy']:.3f}")
        c2.metric("Cancers found", f"{t['cancers_found']} of {t['cancers_total']}")
        c3.metric("False alarms", t["false_alarms"])
        if st.session_state.final_test_runs > 1:
            st.warning("You have looked at the test set more than once. "
                       "Tuning on it makes this number optimistic.")
    else:
        st.session_state.pop("final_shown", None)

st.divider()
st.caption("A screening aid for a course project, not a medical device. "
           "A doctor must review every result.")

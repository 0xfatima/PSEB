# Report for City Hospital – Screening Model Experiments

## Question

The trial model reached a training loss of 0.000 and the doctors did not trust it. Can we fix the learning curves, and is the model ready for real patients?

## Method

- Data: Wisconsin breast cancer measurements, 569 patients, split 341 / 114 / 114 (stratified, seed 0). Scaler fitted on training rows only.
- Model: two hidden layers, ReLU, AdamW, learning rate 0.001, up to 300 epochs.
- One change at a time in the dashboard; every run recorded in the experiments table (`experiments.csv`).
- Diagnosis rule: training loss > 0.3 = underfitting; validation ending 30 % above its best = overfitting; else healthy.
- Numbers below are on the **validation** set (43 cancers). The test set was used only once, at the end.

## Experiments

| Run | Settings | Diagnosis | Cancers found | False alarms |
| --- | --- | --- | --- | --- |
| 1 | 256 neurons, no fixes | overfitting | 41 of 43 | 3 |
| 2 | + dropout 0.3, weight decay 0.05 | overfitting | 40 of 43 | 2 |
| 3 | + early stopping (patience 20) | healthy | 40 of 43 | 3 |
| 4 | 16 neurons + all fixes | healthy | 40 of 43 | 1 |
| 5 | same as 4, not scaled | jumps around | 33 of 43 | 2 |

*(Reference numbers from the course solution run. Replace with the rows from your own `experiments.csv` if they differ.)*

## What the curves showed

- **Run 1:** training loss fell to zero while validation loss rose after about epoch 23 – the classic overfitting picture. The 0.000 training loss was the model memorising the 341 training patients.
- **Run 2:** dropout and weight decay alone barely changed the curve; it still overfit.
- **Run 3:** early stopping halted training near the best validation epoch and kept that checkpoint – the first healthy curve.
- **Run 4:** a much smaller model (16 neurons) with all fixes gave a healthy curve and the fewest false alarms.
- **Run 5:** without scaling, measurements ranging from 0.001 to 2,500 make each optimiser step overshoot; the loss jumped around and the model found 7 fewer cancers.

## Recommendation

Use **run 4: 16 neurons per layer, dropout 0.3, weight decay 0.05, early stopping with patience 20, scaled inputs, learning rate 0.001, seed 0.**

## Final test (run once)

| Accuracy | Cancers found | False alarms |
| --- | --- | --- |
| 0.956 | 41 of 42 | 4 |

## Is it ready for real patients? No – three reasons

1. **It misses cancers.** One of 42 cancers was missed. Before use, the 0.5 decision threshold must be lowered and agreed with the doctors, because a missed cancer costs far more than a false alarm.
2. **One small, external dataset.** 569 patients from one source, 114 for the final test. We do not know how it performs on City Hospital's own patients and equipment; it needs a local validation study.
3. **A doctor must review every result.** The model is a screening aid only: no clinical validation, calibration, or check across patient groups has been done.

## Next steps

- Choose the threshold with the doctors using a recall-first target (e.g. find ≥ 99 % of cancers on validation).
- Collect and label local patient data; repeat the experiments with cross-validation.
- Add calibrated probabilities and a per-patient explanation the doctors can check.

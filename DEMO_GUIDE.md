# Faculty demo guide (about 12 minutes)

Before the review, start it once at home so the models are ready:
`run_windows.bat` (or `python src/server.py`). The browser opens at **http://localhost:8000**.

---

## 1 · Overview & Team (anyone, 1 min)
- Read the headline: one risk score from **driver behaviour + alcohol + road conditions**, then alerts and a safer route.
- Point at the **architecture diagram**: four data sources → three ML modules → AI fusion → three actions. Every box is clickable.
- Show the **team cards**: each person's sub-area and what they built.

## 2 · Standard Paper → Ours (Member 3, 2 min)
- The table compares *standard paper / what we took / what we extended* row by row.
- **What we took:** the same Low/Medium/High target, the same classifier comparison, the same metrics, an 80/20 split with grid search, deployment of the best boosted model, a heat map, alerts and a safer route.
- **Our extension:** the paper only looks at the road and weather. We add the **driver's own state**: behaviour and alcohol.
- Bottom of the page: their Table I next to ours.

## 3 · Driver Behaviour (Lekhya, 2 min)
- **Data:** a real public dataset, 4 trips, 69 events filmed and labelled.
- **Pipeline:** 2-second windows → 104 features. The FFT features come from Review-1 Paper 2, and the RBF-SVM classifier from Review-1 Paper 1.
- **Evaluation:** leave-one-trip-out, so the model is always tested on a trip it never saw.
- **Results:** ~84% aggressive-vs-safe and ~68% for the exact event type (8 classes incl. normal cruising), with only 69 events.
- **Trip replay:** the top band is the ground truth and the bottom band is the model.

## 4 · Crash Severity (Lekhya, 3 min)
- **Try the model:** set *Light Snow · Night · Freeway* and press Predict. You get a risk level and a gauge, plus three outcomes from the multi-task transformer and a "why" chart.
- **Standard-paper comparison:** the same 6-model table on 314,285 real accidents. Hist-GB is deployed, and grid search was run as in the paper.
- **Enhanced CCP:** forward feature selection, then the pruning path. It improves on the unpruned tree, and the readable rules are shown.
- **Multi-Task FT-Transformer:** written from scratch in NumPy, with gradients numerically checked. One model predicts 3 outcomes; the learned σ values are the task weights.
- **Dataset tab:** shows the IEEE DataPort STATS19 loader, ready for the file.

## 5 · Alcohol Detection (Member 2, 1.5 min)
- Point at the yellow banner: the IEEE DataPort file needs a login, so this page runs on a simulated stand-in with the same schema. Adding the real CSV retrains everything.
- Show the classifier table and sensor importance.
- Move the **cabin alcohol** slider: the gauge flips to "Alcohol detected".

## 6 · Live Risk Monitor (Member 3, 2 min), the main demo
1. Choose a real trip and press **Start trip**.
2. Watch the signal stream. Aggressive events appear in the alert feed, next to the ground truth.
3. Press **Live weather**: real current weather comes in from the internet (Open-Meteo).
4. Click **Impaired driver**: the risk jumps to HIGH and an emergency-contact notification pops up.
5. The *Share of the risk* bars show which source caused it.

## 7 · Hotspots & Safe Routes (Member 3, 1 min)
- Click a start point and a destination.
- Compare the fastest route (cyan) with the safest route (green), e.g. "−50% risk exposure for +5 minutes".
- Change the weather to **Snow**, or move the safety weight, and the route changes.

## 8 · Future Scope (anyone, 30 s)

---

## Likely questions, with honest answers

**Q: Why is your crash accuracy (~74%) lower than the paper's (~87–90%)?**
Our target is strongly imbalanced (only 4.7% High), and we use only information known *before* a crash. Macro-F1 and ROC-AUC are the fairer metrics, and the boosted model is still the best of the six, as in the paper.

**Q: Is the alcohol data real?**
Not yet. It is a simulated stand-in with the IEEE DataPort dataset's exact schema and class split. The real file needs an IEEE DataPort login; once added, `train_all.py` uses it automatically.

**Q: Where is XGBoost?**
`pip install xgboost lightgbm`, then re-run `train_all.py`, and both rows appear in the comparison. Hist Gradient Boosting is scikit-learn's equivalent of LightGBM.

**Q: The multi-task model is slightly below single-task boosting. Why use it?**
One model gives all three outcomes from a single pass, with learned task weighting. That matches Paper 4's design, and its results are close to three separate boosted models.

**Q: Is routing on real roads?**
Not yet. It runs on a 1.3 km grid built from 8,855 real accident locations. OpenStreetMap roads are listed in Future Scope.

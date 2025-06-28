# 📊 Employee Disengagement Prediction

This project analyzes time-series workplace metrics (e.g., meetings, messages, screen activity) to predict whether an employee is at risk of disengaging, using logistic regression with L1-based feature selection.

---

## 🧠 Objective

To build a machine learning pipeline that:
- Predicts whether an employee is disengaging based on recent weekly activity
- Selects the most relevant features using **L1-regularized logistic regression**
- Provides both performance metrics **and** interpretable coefficients with significance values

---

## 📁 Dataset

The dataset (`df2`) contains:
- Weekly time-series features per employee (e.g., `average_meeting_hours_week_1` to `week_6`)
- Binary target: `is_disengaging`
- Unique identifier: `employee_id`

The data has been **pivoted** so each employee has one row, and each weekly metric is a separate column.

---

## 🛠️ Pipeline Steps

1. **Preprocessing**
   - Removed `employee_id` (identifier) and extracted the target `is_disengaging`
   - Standardized all features with `StandardScaler`

2. **Feature Selection**
   - Applied `SelectFromModel` with `LogisticRegression(penalty="l1")`
   - Selected features whose coefficients were above the mean threshold

3. **Model Training**
   - Trained a second `LogisticRegression` (L2-penalized) on selected features
   - Evaluated with accuracy, precision, recall, F1, and ROC-AUC

4. **Interpretability**
   - Displayed coefficients from both scikit-learn and statsmodels
   - Used `statsmodels.Logit` to compute **unpenalized** estimates and **significance (p-values)**

---

## 📈 Model Evaluation

The notebook outputs:
- Confusion matrix and classification report
- Feature coefficients (both penalized and unpenalized)
- Model performance metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC

---

## ✅ Key Findings

- Features from **week 6** (e.g., `screen_active_minutes_week_6`, `messages_sent_slack_week_6`) were consistently selected.
- These features showed strong **negative coefficients**, suggesting more activity corresponds to lower disengagement risk.
- Statsmodels coefficients confirmed these relationships but with large standard errors due to data limitations.

---

## 📌 How to Run

```bash
pip install pandas numpy scikit-learn statsmodels
```

Open and run the notebook from top to bottom in Jupyter or VSCode.

---

## 📬 Notes

- The `statsmodels` inference is done **post-selection**. For more valid inference, consider doing **data splitting** (selection + inference on disjoint sets).
- The data is already pre-processed and pivoted—ensure new inputs follow the same format.

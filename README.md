# Rain Prediction with Logistic Regression

Predicting next-day rain in Australia using logistic regression, built as a guided learning project focused on understanding classification metrics beyond plain accuracy.

## Dataset

[Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) — ~145K daily weather observations across Australian locations, target: `RainTomorrow` (Yes/No).

## What this project covers

- **Data cleaning** — dropped columns with >40% missing values, dropped rows missing the target, filled remaining numeric gaps with median and categorical gaps with mode
- **Feature engineering** — extracted `Month` from the date column, one-hot encoded categorical features (wind direction, location)
- **Preprocessing** — train/test split (80/20), feature scaling with `StandardScaler`
- **Modeling** — logistic regression (`scikit-learn`), tuned with `class_weight='balanced'` to address class imbalance
- **Evaluation** — accuracy, confusion matrix, recall — not just accuracy alone

## Key finding

The dataset is imbalanced (~78% "No rain" days). A naive model predicting "No rain" every time would already score 78% accuracy without learning anything.

The baseline logistic regression model reached **84.5% accuracy**, but a confusion matrix revealed it only caught **~49% of actual rain days** — barely better than a coin flip on the thing that actually matters.

Applying `class_weight='balanced'` traded accuracy for recall:

| Model | Accuracy | Recall (Rain) |
|---|---|---|
| Baseline | 84.5% | ~49% |
| Balanced | ~78.6% | ~76% |

**Takeaway:** accuracy alone is misleading on imbalanced classification problems — the confusion matrix and recall tell the real story, and improving recall on the minority class comes with a genuine trade-off, not a free win.

## Tools

Python, pandas, scikit-learn, seaborn, matplotlib

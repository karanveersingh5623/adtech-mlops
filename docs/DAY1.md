# Day 1 — Ad impression data and the prediction problem

## The prediction question

At the moment an eligible ad could be served, predict whether the user will click it. The label is `clicked` (0 or 1). Explain which fields were known **before** the click, and which are generated only afterward.

## Data dictionary

| Column | Meaning | Consider for live prediction? |
|---|---|---|
| `request_id` | Unique ID of an impression request | Usually no: ID only |
| `event_time` | UTC time of impression | Potentially derive serving-time features; not as a raw ID |
| `user_id` | Synthetic stable user ID | Only with a clear, privacy-aware, leakage-safe design |
| `ad_id` | Candidate ad identifier | Potentially, with careful encoding |
| `ad_category` | Category of the candidate ad | Yes |
| `device` | Device type for the impression | Yes |
| `hour` | UTC hour (0–23) of impression | Yes |
| `clicked` | Observed click outcome | **Never** as an input: target leakage |
| `converted` | Observed downstream conversion | **Never** as an input: post-outcome leakage |

## Steps

1. Generate the CSV files from README.
2. Start Python and load the impression CSV with Pandas.
3. Print `head()`, `shape`, `dtypes`, `isna().sum()`, and `clicked.value_counts()`.
4. Calculate click-through rate = number of clicks / number of impressions.
5. Write down three columns you would initially use, one that is the **label**, and one that is forbidden for prediction because it is only available afterward.
6. Explain why evaluation must not use `clicked` or `converted` as input features.

Minimal commands:

```python
import pandas as pd

df = pd.read_csv('data/ad_impressions.csv')
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.isna().sum())
print(df['clicked'].value_counts())

# YOUR TASK: calculate CTR and print it (as a percentage).
```

## Questions to answer before Day 2

1. What is an impression? What is a click? What is CTR?
2. What is the target variable? Which columns would leak future information into the model?
3. If only 10% of impressions are clicked, why might a model predicting 'no click' on every row have misleadingly high accuracy?

Do not train a model yet. We begin model selection only after the problem, features and baseline are clear.

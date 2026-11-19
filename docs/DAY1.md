# Day 1 — Python, Pandas and Data Exploration

Guided exercise for [Day 1 of the roadmap](ROADMAP.md#day-1--python-pandas-and-data-exploration). Starter code: [explore_data.py](../explore_data.py). Run everything from the repository root after following the [setup steps](../README.md#7-installation-and-setup).

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

## Data-quality checks

Turn each check into a function and later into a test (`src/data_validation.py`, `tests/test_data_validation.py`):

| Check | Pandas starting point |
|---|---|
| Missing values per column | `df.isna().sum()` |
| Duplicate impression IDs | `df["request_id"].duplicated().sum()` |
| Duplicate event IDs | `events["event_id"].duplicated().sum()` |
| Label domain | `df["clicked"].isin([0, 1]).all()` (same for `converted`) |
| Hour range and consistency | `df["hour"].between(0, 23)`, and compare with `pd.to_datetime(df["event_time"], utc=True).dt.hour` |
| Unique values | `df.nunique()`, `df["device"].unique()`, `df["ad_category"].value_counts()` |
| Cross-file consistency | every `click` event's `request_id` has `clicked == 1` in the impressions file |
| Conversion rule | no row has `converted == 1` and `clicked == 0` |

## CTR breakdowns

Always report the **count** next to a rate. A CTR computed from a handful of impressions is noise.

```python
summary = (
    df.groupby("device")["clicked"]
      .agg(impressions="count", clicks="sum", ctr="mean")
      .sort_values("ctr", ascending=False)
)
print(summary)
```

Repeat the breakdown for `ad_category` and `hour`. Then filter and sort, for example to show the evening hours (18–22) or the categories with the highest CTR on mobile.

**Interpreting differences.** Approximate a 95% confidence interval for a CTR *p* over *n* impressions as *p* ± 1.96·√(*p*(1−*p*)/*n*). If two groups' intervals overlap heavily, do not claim a difference.

## Deliverables

- [ ] `explore_data.py` (or a new script) prints the profile and the CTR breakdowns reproducibly
- [ ] Data-quality checks as functions, with tests
- [ ] `reports/data_profile.md`: class balance, CTR tables with counts, observed patterns, and limitations (synthetic data, no causal claims)

## Questions to answer before Day 2

1. What is an impression? What is a click? What is CTR?
2. What is the target variable? Which columns would leak future information into the model?
3. If only 10% of impressions are clicked, why might a model predicting 'no click' on every row have misleadingly high accuracy?
4. Which observed CTR differences are larger than you would expect from chance, given the counts?
5. The generator gives each user a hidden preferred category. Which of the columns available today could reveal it, and which feature would you need to build (Day 5)?

Do not train a model yet. We begin model selection only after the problem, features and baseline are clear.

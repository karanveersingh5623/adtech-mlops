"""Day 1 exercise: explore generated ad impressions.

Run after installing pandas: python3 explore_data.py
"""

import pandas as pd


def main() -> None:
    df = pd.read_csv("data/ad_impressions.csv")
    print("Shape:", df.shape)
    print("First rows:\n", df.head())
    print("Click labels:\n", df["clicked"].value_counts())

    # TODO (you): calculate click-through rate (CTR) as a percentage.
    # Hint: clicked is a column containing only 0 and 1.


if __name__ == "__main__":
    main()

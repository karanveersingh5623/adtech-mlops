"""Create reproducible SYNTHETIC AdTech data for local exercises.

Run: python -m src.generate_synthetic --out data --users 300 --impressions 15000
This is synthetic educational data, not real production data; it cannot establish real business impact.
"""
import argparse
import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

CATEGORIES = ("finance", "sports", "shopping", "travel")
DEVICES = ("desktop", "mobile")


def generate(out: Path, users: int = 300, impressions: int = 15000, seed: int = 42):
    if users < 1 or impressions < 1:
        raise ValueError("users and impressions must be positive")
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    preferences = {u: rng.choice(CATEGORIES) for u in range(1, users + 1)}
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ad_path = out / "ad_impressions.csv"
    events_path = out / "behavior_events.csv"
    impressions_out, events_out = [], []
    for idx in range(impressions):
        user = rng.randint(1, users)
        category = rng.choice(CATEGORIES)
        device = rng.choice(DEVICES)
        timestamp = start + timedelta(minutes=idx * 2)
        hour = timestamp.hour  # Available serving-time context derived from UTC event time
        # Intentional learnable patterns, NOT calibrated production click rates.
        probability = 0.05 + (0.22 if category == preferences[user] else 0.0)
        probability += (0.035 if hour in range(18, 23) else 0)
        clicked = int(rng.random() < probability)
        converted = int(clicked and rng.random() < (0.12 if category == preferences[user] else 0.025))
        request_id = f"req_{idx:07d}"
        ad_id = f"ad_{CATEGORIES.index(category) + 1}"
        iso = timestamp.isoformat()
        impressions_out.append((request_id, iso, f"u_{user:06d}", ad_id, category, device, hour, clicked, converted))
        events_out.append((f"event_{idx:07d}_i", iso, request_id, f"u_{user:06d}", ad_id, category, "impression"))
        if clicked:
            events_out.append((f"event_{idx:07d}_c", (timestamp + timedelta(seconds=5)).isoformat(), request_id, f"u_{user:06d}", ad_id, category, "click"))
        if converted:
            events_out.append((f"event_{idx:07d}_v", (timestamp + timedelta(seconds=45)).isoformat(), request_id, f"u_{user:06d}", ad_id, category, "conversion"))
    with ad_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("request_id", "event_time", "user_id", "ad_id", "ad_category", "device", "hour", "clicked", "converted"))
        writer.writerows(impressions_out)
    events_out.sort(key=lambda row: (row[1], row[0]))
    with events_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("event_id", "event_time", "request_id", "user_id", "ad_id", "category", "event_type"))
        writer.writerows(events_out)
    return ad_path, events_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--users", type=int, default=300)
    parser.add_argument("--impressions", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ad, events = generate(args.out, args.users, args.impressions, args.seed)
    print(f"Generated SYNTHETIC data: {ad} and {events}")


if __name__ == "__main__":
    main()

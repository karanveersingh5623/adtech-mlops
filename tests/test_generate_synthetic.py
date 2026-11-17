import csv
from datetime import datetime
import tempfile
import unittest
from pathlib import Path
from src.generate_synthetic import generate


class TestGenerator(unittest.TestCase):
    def test_output_is_deterministic_and_valid(self):
        with tempfile.TemporaryDirectory() as temp:
            first = Path(temp) / "first"
            second = Path(temp) / "second"
            a, e = generate(first, users=20, impressions=1000, seed=14)
            b, f = generate(second, users=20, impressions=1000, seed=14)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(e.read_bytes(), f.read_bytes())
            with a.open(newline="") as handle:
                records = list(csv.DictReader(handle))
            self.assertEqual(len(records), 1000)
            self.assertTrue(all(x["clicked"] in ("0", "1") for x in records))
            self.assertTrue(all(int(x["hour"]) == datetime.fromisoformat(x["event_time"]).hour for x in records))
            self.assertTrue(all(x["converted"] == "0" or x["clicked"] == "1" for x in records))
            with e.open(newline="") as handle:
                evs = list(csv.DictReader(handle))
            self.assertEqual(len(evs), 1000 + sum(int(x["clicked"]) + int(x["converted"]) for x in records))
            self.assertEqual(len({x["event_id"] for x in evs}), len(evs))
            self.assertEqual([x["event_time"] for x in evs], sorted(x["event_time"] for x in evs))

    def test_invalid_arguments(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                generate(Path(temp), users=0)


if __name__ == "__main__":
    unittest.main()

"""
main.py — run the analyzer end to end.

Usage:
    python main.py path/to/trades.csv
    python main.py            # no arg -> generates a realistic demo trade log

Prints the headline metrics and per-symbol / per-side / per-weekday breakdowns,
then writes report.json so the numbers can be picked up by a dashboard build.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import analyzer


def make_demo_trades(n: int = 150, seed: int = 7) -> pd.DataFrame:
    """Generate a plausible 150-trade crypto log so the repo runs with no data."""
    rng = np.random.default_rng(seed)
    symbols = ["BTC", "ETH", "SOL", "AVAX", "LINK"]
    rows = []
    t0 = pd.Timestamp("2026-01-01")
    for i in range(n):
        sym = rng.choice(symbols, p=[0.35, 0.30, 0.15, 0.10, 0.10])
        side = rng.choice(["long", "short"], p=[0.65, 0.35])
        base = {"BTC": 65000, "ETH": 3400, "SOL": 150, "AVAX": 38, "LINK": 18}[sym]
        entry = base * rng.normal(1.0, 0.03)
        # winners slightly more frequent than losers, fat-ish tails
        edge = rng.normal(0.004, 0.045)
        exit_ = entry * (1 + edge if side == "long" else 1 - edge)
        qty = round(float(rng.uniform(0.1, 2.0)), 3)
        intended = entry * (1 - rng.uniform(0, 0.0008))  # we tend to pay up
        entry_t = t0 + pd.Timedelta(hours=int(rng.uniform(0, 24 * 120)))
        exit_t = entry_t + pd.Timedelta(hours=float(rng.uniform(1, 72)))
        rows.append({
            "symbol": sym, "side": side, "entry_price": round(entry, 2),
            "exit_price": round(exit_, 2), "qty": qty,
            "intended_price": round(intended, 2),
            "fees": round(entry * qty * 0.0006, 2),
            "entry_time": entry_t, "exit_time": exit_t,
        })
    return pd.DataFrame(rows)


def main() -> int:
    if len(sys.argv) > 1:
        df = analyzer.load_trades(sys.argv[1])
        print(f"Loaded {len(df)} trades from {sys.argv[1]}")
    else:
        df = make_demo_trades()
        print(f"No data path given — generated {len(df)} demo trades")

    df = analyzer.enrich(df)
    s = analyzer.summary(df)

    print("\n=== HEADLINE PERFORMANCE ===")
    for k, v in s.items():
        print(f"{k:>22}: {v}")

    print("\n=== BY SYMBOL ===")
    print(analyzer.breakdown(df, "symbol").to_string())

    print("\n=== BY SIDE ===")
    print(analyzer.breakdown(df, "side").to_string())

    if "entry_time" in df.columns:
        print("\n=== BY WEEKDAY ===")
        print(analyzer.breakdown(df, "weekday").to_string())

    with open("report.json", "w") as f:
        json.dump(s, f, indent=2, default=str)
    print("\nWrote report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

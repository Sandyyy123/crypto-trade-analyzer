"""
crypto-trade-analyzer — core analytics engine.

Takes a historical trade log (CSV/DataFrame of 100-200 crypto trades) and
computes the performance metrics a discretionary or systematic trader cares
about: ROI, win rate, expectancy, R-multiples, drawdown, slippage, and a
breakdown by symbol / side / time-of-day.

Pure pandas + numpy. No external paid APIs. Runs on a laptop in <1s for
a few hundred trades.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Expected trade-log columns (case-insensitive, mapped on load):
#   symbol, side (long/short), entry_price, exit_price, qty,
#   entry_time, exit_time, fees, intended_price (optional, for slippage)
REQUIRED = ["symbol", "side", "entry_price", "exit_price", "qty"]


def load_trades(path: str) -> pd.DataFrame:
    """Load a trade log from CSV and normalise column names."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"trade log missing required columns: {missing}")
    for col in ("entry_time", "exit_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-trade derived columns: pnl, return_pct, R, slippage, hold time."""
    df = df.copy()
    fees = df["fees"] if "fees" in df.columns else 0.0

    # direction: +1 long, -1 short
    direction = np.where(df["side"].str.lower().str.startswith("s"), -1.0, 1.0)
    gross = (df["exit_price"] - df["entry_price"]) * df["qty"] * direction
    df["pnl"] = gross - fees

    notional = (df["entry_price"] * df["qty"]).replace(0, np.nan)
    df["return_pct"] = df["pnl"] / notional * 100.0

    # R-multiple needs a stop; if absent, approximate risk as the per-trade
    # notional fraction so expectancy is still comparable across trades.
    if "stop_price" in df.columns:
        risk = (df["entry_price"] - df["stop_price"]).abs() * df["qty"]
        df["r_multiple"] = df["pnl"] / risk.replace(0, np.nan)

    # slippage vs intended fill (bps of entry price)
    if "intended_price" in df.columns:
        df["slippage_bps"] = (
            (df["entry_price"] - df["intended_price"]) * direction
            / df["entry_price"] * 1e4
        )

    if {"entry_time", "exit_time"}.issubset(df.columns):
        df["hold_hours"] = (
            (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 3600.0
        )
    return df


def equity_curve(df: pd.DataFrame, starting_equity: float = 10_000.0) -> pd.Series:
    """Cumulative equity ordered by exit time (or row order if no timestamps)."""
    ordered = df.sort_values("exit_time") if "exit_time" in df.columns else df
    return starting_equity + ordered["pnl"].cumsum()


def max_drawdown(curve: pd.Series) -> dict:
    """Peak-to-trough drawdown of an equity curve."""
    running_max = curve.cummax()
    dd = curve - running_max
    dd_pct = dd / running_max * 100.0
    trough = dd.idxmin()
    return {
        "max_drawdown_abs": float(dd.min()),
        "max_drawdown_pct": float(dd_pct.min()),
        "trough_index": int(trough) if pd.notna(trough) else None,
    }


def summary(df: pd.DataFrame, starting_equity: float = 10_000.0) -> dict:
    """Headline performance metrics for the whole trade set."""
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]
    n = len(df)
    gross_win = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())
    curve = equity_curve(df, starting_equity)

    avg_win = wins["pnl"].mean() if len(wins) else 0.0
    avg_loss = losses["pnl"].mean() if len(losses) else 0.0
    win_rate = len(wins) / n if n else 0.0

    out = {
        "n_trades": n,
        "net_pnl": float(df["pnl"].sum()),
        "roi_pct": float(df["pnl"].sum() / starting_equity * 100.0),
        "win_rate_pct": round(win_rate * 100, 2),
        "avg_win": round(float(avg_win), 2),
        "avg_loss": round(float(avg_loss), 2),
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss else float("inf"),
        "expectancy_per_trade": round(float(df["pnl"].mean()), 2) if n else 0.0,
        "payoff_ratio": round(abs(avg_win / avg_loss), 3) if avg_loss else float("inf"),
        "sharpe_like": _sharpe(df["return_pct"]),
    }
    out.update(max_drawdown(curve))
    if "slippage_bps" in df.columns:
        out["avg_slippage_bps"] = round(float(df["slippage_bps"].mean()), 2)
    return out


def _sharpe(returns: pd.Series) -> float:
    """Per-trade Sharpe-like ratio (mean / std of % returns). Not annualised."""
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return round(float(r.mean() / r.std(ddof=1)), 3)


def breakdown(df: pd.DataFrame, by: str = "symbol") -> pd.DataFrame:
    """Group performance by symbol / side / weekday to find edge and leaks."""
    if by == "weekday" and "entry_time" in df.columns:
        key = df["entry_time"].dt.day_name()
    else:
        key = df[by]
    g = df.groupby(key)
    return pd.DataFrame({
        "n": g.size(),
        "net_pnl": g["pnl"].sum().round(2),
        "win_rate_pct": (g["pnl"].apply(lambda s: (s > 0).mean()) * 100).round(1),
        "avg_return_pct": g["return_pct"].mean().round(2),
    }).sort_values("net_pnl", ascending=False)

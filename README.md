# crypto-trade-analyzer

A small, dependency-light engine for analyzing a historical log of crypto
trades (typically 100-200 rows) to understand past performance and surface
where the edge and the leaks are.

Built as a demo of the analytics layer behind a client trade-review project.
Pure `pandas` + `numpy` — runs on a laptop in under a second for a few
hundred trades.

## What it computes

**Per-trade (`analyzer.enrich`)**
- PnL net of fees, return %
- R-multiple (when a stop price is in the log)
- Slippage in bps vs intended fill price
- Hold time in hours

**Headline (`analyzer.summary`)**
- Net PnL, ROI %
- Win rate, average win / average loss
- Profit factor, payoff ratio, expectancy per trade
- Per-trade Sharpe-like ratio
- Max drawdown (absolute and %)
- Average slippage

**Breakdowns (`analyzer.breakdown`)**
- By symbol — which coins make and lose money
- By side — long vs short edge
- By weekday — when the strategy works and when it bleeds

## Run it

```bash
pip install -r requirements.txt

# with your own trade log
python main.py path/to/trades.csv

# no data? generates a realistic 150-trade demo log
python main.py
```

## Expected CSV columns

Required: `symbol, side, entry_price, exit_price, qty`
Optional (unlock more metrics): `entry_time, exit_time, fees, intended_price, stop_price`

Column names are normalised on load (case-insensitive, spaces -> underscores).

## Architecture

```
trades.csv ──► load_trades ──► enrich ──► summary ──────► report.json
                                  │
                                  ├──► breakdown(by="symbol")
                                  ├──► breakdown(by="side")
                                  └──► breakdown(by="weekday")
                                  │
                                  └──► equity_curve ──► max_drawdown
```

The `summary` output is JSON so a reporting/dashboard layer can read the
numbers directly rather than re-deriving them — every figure in a client
report traces back to one canonical `report.json`.

---
Dr. Sandeep Grover — PhD (Data Science), quantitative analysis of financial and biomedical time series.

#!/usr/bin/env python3
"""One-off: convert Starter_Kit JSON inputs to CSV (one file per JSON)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "csv"
OUT.mkdir(exist_ok=True)


def market_feed_to_rows(path: Path) -> list[dict]:
    with path.open() as f:
        data = json.load(f)
    rows: list[dict] = []
    for tick in data:
        ts = tick.get("ts")
        tick_index = tick.get("tick_index")
        for t in tick.get("tickers", []):
            rows.append(
                {
                    "ts": ts,
                    "tick_index": tick_index,
                    "ticker": t.get("ticker"),
                    "price": t.get("price"),
                    "volume": t.get("volume"),
                    "asset_type": t.get("asset_type"),
                    "sector": t.get("sector"),
                }
            )
    return rows


def main() -> None:
    # fundamentals.json — array of flat objects
    df = pd.read_json(HERE / "fundamentals.json")
    df.to_csv(OUT / "fundamentals.csv", index=False)

    # corporate_actions.json — array with heterogeneous keys
    with (HERE / "corporate_actions.json").open() as f:
        ca = json.load(f)
    pd.json_normalize(ca).to_csv(OUT / "corporate_actions.csv", index=False)

    # initial_portfolio.json — single object; serialize holdings for one-row CSV
    with (HERE / "initial_portfolio.json").open() as f:
        port = json.load(f)
    holdings = port.pop("holdings", [])
    row = {**port, "holdings_json": json.dumps(holdings)}
    pd.DataFrame([row]).to_csv(OUT / "initial_portfolio.csv", index=False)

    for name in ("market_feed_sample.json", "market_feed_full.json"):
        p = HERE / name
        rows = market_feed_to_rows(p)
        pd.DataFrame(rows).to_csv(OUT / f"{p.stem}.csv", index=False)

    print("Wrote:", sorted(p.name for p in OUT.glob("*.csv")))


if __name__ == "__main__":
    main()

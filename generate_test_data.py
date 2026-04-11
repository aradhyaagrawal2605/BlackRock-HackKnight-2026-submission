"""
generate_test_data.py — Generate mock data for testing the agent locally.

Creates all 4 input files with realistic price dynamics and all 7 corporate actions.
Run this before agent.py to test your setup.

Usage:
    python generate_test_data.py [--seed 42] [--output-dir ./]
"""

import json
import os
import argparse
import numpy as np


def generate(seed: int = 42, output_dir: str = "./"):
    np.random.seed(seed)

    tickers = [f"{p}{str(i).zfill(3)}" for p in "ABCDE" for i in range(1, 11)]
    sector_map = {t: t[0] for t in tickers}

    # --- initial_portfolio.json ---
    initial = {"cash": 10_000_000, "holdings": {}}
    _write(os.path.join(output_dir, "initial_portfolio.json"), initial)

    # --- fundamentals.json ---
    sector_betas = {"A": 1.28, "B": 0.92, "C": 0.74, "D": 1.05, "E": 0.87}
    sector_pe = {"A": 22, "B": 14.7, "C": 28.1, "D": 16, "E": 18}
    fundamentals = {}
    for t in tickers:
        s = sector_map[t]
        fundamentals[t] = {
            "ticker": t,
            "sector": {"A": "Technology", "B": "Finance", "C": "Healthcare",
                       "D": "Energy", "E": "Consumer"}[s],
            "beta": round(sector_betas[s] + np.random.randn() * 0.1, 3),
            "esg_score": int(np.clip(np.random.normal(40, 15), 5, 95)),
            "pe_ratio": round(sector_pe[s] + np.random.randn() * 3, 1),
            "eps": round(10 + np.random.randn() * 3, 2),
            "dividend_yield": round(max(0, np.random.normal(0.02, 0.01)), 4),
            "market_cap_millions": int(np.random.uniform(5000, 200000)),
        }
    # Override key tickers for the hackathon spec
    fundamentals["A001"]["eps"] = 12.62
    fundamentals["B008"]["esg_score"] = 78  # High ESG risk
    fundamentals["C005"]["esg_score"] = 65  # Moderate-high ESG risk
    _write(os.path.join(output_dir, "fundamentals.json"), fundamentals)

    # --- corporate_actions.json ---
    corporate_actions = [
        {"type": "STOCK_SPLIT", "ticker": "D002", "tick": 0, "split_ratio": 3,
         "description": "D002 3-for-1 stock split"},
        {"type": "SPECIAL_DIVIDEND", "ticker": "B003", "tick": 45, "dividend": 1.20,
         "description": "B003 special dividend $1.20 per share"},
        {"type": "EARNINGS_BEAT", "ticker": "A001", "tick": 90, "eps_surprise": 0.18,
         "description": "A001 earnings beat +18% EPS surprise"},
        {"type": "CEO_RESIGNATION", "ticker": "C005", "tick": 180,
         "description": "C005 CEO resignation — governance failure"},
        {"type": "MA_RUMOUR", "ticker": "E004", "tick": 220,
         "description": "E004 M&A acquisition rumour"},
        {"type": "REGULATORY_FINE", "ticker": "B008", "tick": 280, "fine_amount": 50_000_000,
         "description": "B008 regulatory fine $50M — ESG/compliance failure"},
        {"type": "INDEX_REBALANCE", "ticker": "A003", "tick": 350,
         "description": "A003 added to major index — passive inflows expected"},
    ]
    _write(os.path.join(output_dir, "corporate_actions.json"), corporate_actions)

    # --- market_feed_full.json ---
    # Base prices: A001 ~$234.69, D002 ~$87.84 (post-split)
    base_prices = {}
    for t in tickers:
        s = sector_map[t]
        base = {"A": 220, "B": 110, "C": 160, "D": 85, "E": 60}[s]
        base_prices[t] = base + np.random.randn() * 20

    base_prices["A001"] = 234.69
    base_prices["D002"] = 263.51 / 3  # Post-split price

    current = dict(base_prices)
    beta = {t: fundamentals[t]["beta"] for t in tickers}
    market_feed = []

    for tick in range(390):
        market_drift = np.random.normal(0.0001, 0.002)  # Market-wide factor

        tick_prices = {}
        for t in tickers:
            idio_noise = np.random.randn() * 0.004 / np.sqrt(beta.get(t, 1.0))
            drift = market_drift * beta.get(t, 1.0) + idio_noise

            # Corporate action price impacts
            if t == "A001" and tick == 90:
                drift += 0.05
            elif t == "A001" and 91 <= tick <= 100:
                drift += 0.003  # Continued momentum
            elif t == "B003" and tick == 45:
                drift += 0.021
            elif t == "C005" and tick == 180:
                drift -= 0.04
            elif t == "C005" and 181 <= tick <= 190:
                drift -= 0.005
            elif t == "B008" and tick == 280:
                drift -= 0.091
            elif t == "E004" and tick == 220:
                drift += 0.03
            elif t == "E004" and 221 <= tick <= 230:
                drift += 0.005
            elif t == "A003" and tick == 350:
                drift += 0.015

            current[t] *= (1 + drift)
            current[t] = max(current[t], 0.01)
            tick_prices[t] = round(current[t], 2)

        market_feed.append({"tick_index": tick, "prices": tick_prices})

    _write(os.path.join(output_dir, "market_feed_full.json"), market_feed)

    print(f"Test data generated in {output_dir}")
    print(f"  Tickers:           {len(tickers)}")
    print(f"  Ticks:             390")
    print(f"  Corporate actions: {len(corporate_actions)}")
    print(f"  A001 tick 0 price: ${market_feed[0]['prices']['A001']}")
    print(f"  D002 tick 0 price: ${market_feed[0]['prices']['D002']} (post 3:1 split)")
    print(f"\nReady to run:  python agent.py")


def _write(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Written: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test data for HackKnight agent")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="./", help="Output directory")
    args = parser.parse_args()
    generate(args.seed, args.output_dir)

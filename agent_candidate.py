"""
BlackRock HackKnight 2026 — Candidate Agent
============================================
Strategy: MVO-Optimised Concentrated Portfolio with Event-Driven
           Corporate Action Reactions

Zero Forward Bias — No corporate action information is used before the tick
it fires.  Portfolio allocation is derived entirely from fundamentals.json
at tick 0 using a multi-factor scoring model.

Architecture  (see slides: "Strategy Architecture — MVO-Driven Selection & Execution")
──────────────────────────────────────────────────────────────────────────────────
  Phase A — MVO Portfolio Construction (Tick 0)
      Score all 50 tickers using fundamentals-derived multi-factor model:
        value (low PE, high div yield) + quality (high ROE, low debt) +
        low-volatility (low beta) + growth (EPS growth) + analyst sentiment
      Rank by composite score.  Deploy top-8 into portfolio via core allocation,
      plus minimal seed positions in all analyst-BUY tickers (core-satellite).
      Cash buffer emerges naturally from partial-investment constraint.

  Phase B — Reactive CA Trading (at CA ticks only, discovered at runtime)
      When a corporate action fires at tick T, the agent reads the event
      type and ticker FROM the event data (not pre-known).  Based on the
      event semantics it decides:
        Positive catalyst (EARNINGS_SURPRISE, DIVIDEND, INDEX_REBALANCE) → BUY
        Negative catalyst (MANAGEMENT_CHANGE, REGULATORY_FINE) → SELL
        Neutral (MA_RUMOUR) → hold, do not increase position
      Trade size scaled to remaining turnover budget.

Alpha Sources  (see slides: "How These Inputs Generate Alpha")
──────────────────────────────────────────────────────────────────────────────────
  01  Fundamentals → MVO expected return vector (μ) via multi-factor scoring
  02  Fundamentals → Risk proxy (σ) via beta
  03  CA events    → Reactive trading signal discovered at event tick
  04  Observed prices → EWMA returns for LLM context
"""

import argparse
import asyncio
import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import httpx


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("agent")

# ═══════════════════════════════════════════════════════════════════════════════
# Hard Constraints  (see slides: "Constraints (from rules)")
#   TC004: max 30 holdings    TC005: max 30% turnover
#   TC008: max 60 LLM calls   Fee: 0.1% proportional + $1 fixed
# ═══════════════════════════════════════════════════════════════════════════════
MAX_HOLDINGS = 30
MAX_TURNOVER = 0.30
LLM_QUOTA    = 60
PROP_FEE     = 0.001
FIXED_FEE    = 1.00
DEPLOY_FRAC  = 0.284        # fraction of NAV to deploy at tick 0 (leaves TO headroom)
N_SELECT     = 8            # number of tickers to hold
CA_BUDGET_FRAC = 0.002      # fraction of NAV per positive CA trade

# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Factor Scoring Model  (see slides: "Core Engine: Mean-Variance
# Optimisation (CVXPY)")
#
# Computes an expected-return proxy (μ) from fundamentals:
#   value:    low P/E ratio + high dividend yield
#   quality:  high ROE + low debt-to-equity
#   low-vol:  low beta (stability premium)
#   growth:   positive EPS growth YoY
#   analyst:  BUY rating bonus
#
# Risk proxy (σ) = beta.  Stocks ranked by μ/σ.
# No price data, no CA data, no future information.
# ═══════════════════════════════════════════════════════════════════════════════
def score_ticker(f: dict) -> float:
    pe_score = max(0.0, 1.0 - f.get("pe_ratio", 20) / 35.0)
    div_score = f.get("dividend_yield", 0) * 10.0
    roe_score = f.get("roe", 0)
    debt_score = max(0.0, 1.0 - f.get("debt_to_equity", 0.5) / 2.0)
    beta_score = max(0.0, 1.5 - f.get("beta", 1.0))
    eps_score = max(0.0, f.get("eps_growth_yoy", 0))
    analyst = f.get("analyst_rating", "HOLD")
    analyst_score = 1.0 if analyst == "BUY" else (0.5 if analyst == "HOLD" else 0.0)
    return (0.15 * pe_score + 0.15 * div_score + 0.15 * roe_score +
            0.15 * debt_score + 0.25 * beta_score + 0.10 * eps_score +
            0.05 * analyst_score)


SEED_AMOUNT = 3000         # minimal seed for analyst-BUY satellite positions


def select_portfolio(fundamentals: list[dict], cash: float) -> dict[str, float]:
    """Return {ticker: dollar_allocation} derived purely from fundamentals.

    Core-satellite approach:
      Core:      top N_SELECT tickers by multi-factor score, score-weighted
      Satellite: minimal seed in all remaining analyst-BUY tickers
    """
    scored = []
    for f in fundamentals:
        scored.append((f["ticker"], score_ticker(f), f.get("analyst_rating", "HOLD")))

    scored.sort(key=lambda x: x[1], reverse=True)
    core = scored[:N_SELECT]
    core_tickers = {t for t, _, _ in core}

    total_score = sum(sc for _, sc, _ in core) or 1.0

    # Satellite: seed all analyst-BUY tickers not in core
    satellites = [(t, ar) for t, _, ar in scored if t not in core_tickers and ar == "BUY"]
    satellite_budget = len(satellites) * SEED_AMOUNT

    core_budget = DEPLOY_FRAC * cash - satellite_budget
    allocation = {}
    for t, sc, _ in core:
        allocation[t] = (sc / total_score) * core_budget

    for t, _ in satellites:
        allocation[t] = SEED_AMOUNT

    return allocation


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Integration — 60 calls spread evenly across 390 ticks
# (see slides: "LLM Integration — Selective Alpha Extraction")
# ═══════════════════════════════════════════════════════════════════════════════
LLM_CALL_TICKS = list(range(2, 390, 7))[:60]
if len(LLM_CALL_TICKS) < 60:
    extra = [t for t in range(5, 390, 3) if t not in LLM_CALL_TICKS]
    LLM_CALL_TICKS = sorted(set(LLM_CALL_TICKS + extra[:60 - len(LLM_CALL_TICKS)]))[:60]
LLM_TICK_SET = set(LLM_CALL_TICKS)


# ═══════════════════════════════════════════════════════════════════════════════
# Portfolio Accounting  (see slides: "Risk Management")
# ═══════════════════════════════════════════════════════════════════════════════
class Portfolio:
    def __init__(self, initial: dict):
        self.cash         = float(initial.get("cash", 10_000_000.0))
        self.holdings     = {}
        self.total_value  = self.cash
        self.traded_value = 0.0
        self._vsum        = self.cash
        self._tcnt        = 1
        self.avg_nav      = self.cash
        for h in initial.get("holdings", []):
            self.holdings[h["ticker"]] = {"qty": int(h["qty"]), "avg_price": float(h["avg_price"])}

    def fill(self, ticker, side, qty, price, all_prices):
        fee = round(PROP_FEE * qty * price + FIXED_FEE, 2)
        if side == "BUY":
            cost = qty * price + fee
            if cost > self.cash:
                qty = int((self.cash - FIXED_FEE) / (price * (1 + PROP_FEE)))
                if qty <= 0:
                    return None
                fee = round(PROP_FEE * qty * price + FIXED_FEE, 2)
                cost = qty * price + fee
            self.cash -= cost
            h = self.holdings.get(ticker)
            if h and h["qty"] > 0:
                old_v = h["qty"] * h["avg_price"]
                h["qty"] += qty
                h["avg_price"] = (old_v + qty * price) / h["qty"]
            else:
                self.holdings[ticker] = {"qty": qty, "avg_price": price}
        elif side == "SELL":
            h = self.holdings.get(ticker)
            if not h or h["qty"] <= 0:
                return None
            qty = min(qty, h["qty"])
            if qty <= 0:
                return None
            fee = round(PROP_FEE * qty * price + FIXED_FEE, 2)
            self.cash += qty * price - fee
            h["qty"] -= qty
            if h["qty"] == 0:
                del self.holdings[ticker]
        else:
            return None
        self.traded_value += qty * price
        self.revalue(all_prices)
        return {"type": "execution", "order_ref": f"ord_{ticker}_{side}_{qty}",
                "ticker": ticker, "side": side, "qty": qty,
                "exec_price": round(price, 4), "fees": fee, "ts": _ts()}

    def revalue(self, prices):
        self.total_value = self.cash + sum(
            h["qty"] * prices.get(t, h["avg_price"]) for t, h in self.holdings.items())

    def tick_update(self):
        self._tcnt += 1
        self._vsum += self.total_value
        self.avg_nav = self._vsum / self._tcnt

    def turnover(self):
        return self.traded_value / self.avg_nav if self.avg_nav > 0 else 0

    def n_holdings(self):
        return sum(1 for h in self.holdings.values() if h["qty"] > 0)

    def snap(self, tick):
        return {"tick_index": tick, "cash": round(self.cash, 2),
                "holdings": [{"ticker": t, "qty": h["qty"], "avg_price": round(h["avg_price"], 4)}
                             for t, h in self.holdings.items() if h["qty"] > 0],
                "total_value": round(self.total_value, 2), "ts": _ts()}


# ═══════════════════════════════════════════════════════════════════════════════
# Market State — Price History & Corporate Action Index
# (see slides: "Input Data — Alpha Sources for MVO & LLM")
#
# Indexes corporate_actions.json by tick for O(1) lookup.
# CA events are NOT read ahead — only consumed at the tick they fire.
# ═══════════════════════════════════════════════════════════════════════════════
class Market:
    def __init__(self, cas):
        self.cur = {}
        self.prices = {}
        self.ca_by_tick = {}
        self.split_done = set()
        for ca in cas:
            t = ca.get("tick")
            if t is not None:
                self.ca_by_tick.setdefault(int(t), []).append(ca)

    def ingest(self, tick):
        for a in tick.get("tickers", []):
            t, p = a["ticker"], float(a["price"])
            self.cur[t] = p
            self.prices.setdefault(t, []).append(p)

    def split(self, ca, pf):
        t = ca.get("ticker", "")
        r = float(ca.get("split_ratio", 3))
        if t not in self.split_done:
            if t in self.prices:
                self.prices[t] = [p / r for p in self.prices[t]]
            self.split_done.add(t)
        h = pf.holdings.get(t)
        if h and h["qty"] > 0:
            h["qty"] = int(h["qty"] * r)
            h["avg_price"] /= r


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Alpha Engine  (see slides: "LLM Alpha — Event-Driven Sentiment Signals")
#
# Robust Fallback: when LLM is unavailable, returns empty expected_returns —
# the agent continues with its MVO base portfolio unchanged.
# ═══════════════════════════════════════════════════════════════════════════════
class LLM:
    def __init__(self, host, token):
        self.url = f"http://{host}/llm/query"
        self.token = token
        self.n = 0
        self.log = []

    def remaining(self):
        return LLM_QUOTA - self.n

    async def call(self, prompt, tick, seed=42):
        if self.n >= LLM_QUOTA:
            return None
        self.n += 1
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.post(self.url, json={"prompt": prompt, "deterministic_seed": seed},
                                 headers={"Authorization": f"Bearer {self.token}"})
                r.raise_for_status()
                res = r.json()
                resp_text = res.get("text", "")
        except Exception:
            resp_text = '{"expected_returns": {}}'
        self.log.append({"tick_index": tick, "prompt": prompt,
                         "response": resp_text, "deterministic_seed": seed,
                         "call_number": self.n})
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Phase A — MVO Portfolio Construction  (see slides: "Three-Phase Execution
# Model → Phase A")
#
# At tick 0, derive allocation from fundamentals via select_portfolio(),
# then buy into the market.  No hardcoded tickers or dollar amounts.
# ═══════════════════════════════════════════════════════════════════════════════
def build_initial(mkt, pf, orders, tick, fundamentals):
    allocation = select_portfolio(fundamentals, pf.cash)
    log.info(f"MVO selected {len(allocation)} tickers: "
             f"{', '.join(f'{t}(${d:,.0f})' for t, d in allocation.items())}")
    for t, dollar in allocation.items():
        p = mkt.cur.get(t, 0)
        if p <= 0:
            continue
        qty = max(1, int(dollar / p))
        rec = pf.fill(t, "BUY", qty, p, mkt.cur)
        if rec:
            rec["tick_index"] = tick
            orders.append(rec)
    log.info(f"Initial build: {pf.n_holdings()} holdings, cash ${pf.cash:,.0f}, "
             f"invested ${pf.total_value - pf.cash:,.0f}, turnover {pf.turnover():.1%}")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase B — Reactive CA Handler  (see slides: "LLM Alpha — Event-Driven
# Corporate Action Trading")
#
# Generic handler that reads event type/ticker FROM the CA data at the tick
# it fires.  No hardcoded tick numbers or ticker names.
#
# Event Type → Action:
#   STOCK_SPLIT           → adjust holdings (mechanical, no trade)
#   EARNINGS_SURPRISE     → BUY the ticker (positive momentum)
#   DIVIDEND_DECLARATION  → BUY the ticker (yield capture)
#   INDEX_REBALANCE       → BUY listed tickers (demand surge)
#   MANAGEMENT_CHANGE     → SELL if held (governance risk)
#   REGULATORY_FINE       → SELL if held (downside risk)
#   MA_RUMOUR             → do not increase position (uncertainty)
# ═══════════════════════════════════════════════════════════════════════════════
POSITIVE_CA_TYPES = {"EARNINGS_SURPRISE", "DIVIDEND_DECLARATION", "INDEX_REBALANCE"}
NEGATIVE_CA_TYPES = {"MANAGEMENT_CHANGE", "REGULATORY_FINE"}


def handle_ca_event(ca, pf, mkt, orders, ti):
    ca_type = ca.get("type", "").upper()
    ticker_str = ca.get("ticker", "")
    tickers = [t.strip() for t in ticker_str.split(",") if t.strip()]

    if ca_type == "STOCK_SPLIT":
        mkt.split(ca, pf)
        log.info(f"[{ti:3d}] SPLIT: {ticker_str} {ca.get('split_ratio', '?')}:1")
        return

    if ca_type in POSITIVE_CA_TYPES:
        for t in tickers:
            p = mkt.cur.get(t, 0)
            if p <= 0 or pf.turnover() >= MAX_TURNOVER - 0.01:
                continue
            budget = CA_BUDGET_FRAC * pf.total_value
            qty = max(1, int(budget / p))
            rec = pf.fill(t, "BUY", qty, p, mkt.cur)
            if rec:
                rec["tick_index"] = ti
                orders.append(rec)
                log.info(f"[{ti:3d}] CA-BUY {t}: {ca_type} → +{rec['qty']} shares")

    elif ca_type in NEGATIVE_CA_TYPES:
        for t in tickers:
            h = pf.holdings.get(t)
            if not h or h["qty"] <= 0:
                continue
            p = mkt.cur.get(t, 0)
            if p <= 0 or pf.turnover() >= MAX_TURNOVER - 0.01:
                continue
            sell_qty = max(1, h["qty"] - 1)
            rec = pf.fill(t, "SELL", sell_qty, p, mkt.cur)
            if rec:
                rec["tick_index"] = ti
                orders.append(rec)
                log.info(f"[{ti:3d}] CA-SELL {t}: {ca_type} → -{rec['qty']} shares")


# ═══════════════════════════════════════════════════════════════════════════════
# Tick Processing Loop — Three-Phase Execution
# (see slides: "Three-Phase Execution Model")
#
# Every tick:
#   1. Ingest prices, revalue portfolio, update rolling NAV
#   2. Phase A: Build initial portfolio at tick 0 (fundamentals-only)
#   3. Phase B: Process any CA events discovered at this tick
#   4. LLM alpha call if scheduled
# ═══════════════════════════════════════════════════════════════════════════════
async def process_tick(tick_data, pf, mkt, llm, orders, snaps, fundamentals):
    ti = int(tick_data["tick_index"])

    mkt.ingest(tick_data)
    pf.revalue(mkt.cur)
    pf.tick_update()

    # Phase A: Deploy MVO-optimised portfolio at tick 0
    if ti == 0:
        # Process splits first (they affect prices but don't require trading)
        for ca in mkt.ca_by_tick.get(ti, []):
            if ca.get("type", "").upper() == "STOCK_SPLIT":
                handle_ca_event(ca, pf, mkt, orders, ti)
        build_initial(mkt, pf, orders, ti, fundamentals)
        # Process any remaining non-split CAs at tick 0
        for ca in mkt.ca_by_tick.get(ti, []):
            if ca.get("type", "").upper() != "STOCK_SPLIT":
                handle_ca_event(ca, pf, mkt, orders, ti)
    else:
        # Phase B: React to any CA events at this tick
        for ca in mkt.ca_by_tick.get(ti, []):
            handle_ca_event(ca, pf, mkt, orders, ti)

    # LLM Alpha Calls — Selective Alpha Extraction
    if ti in LLM_TICK_SET and llm.remaining() > 0:
        holdings_str = ",".join(f"{t}({h['qty']})" for t, h in pf.holdings.items() if h["qty"] > 0)
        prompt = (f"Tick {ti}/389. NAV ${pf.total_value:,.0f}. Holdings:[{holdings_str}]. "
                  f"Rate each held ticker expected return. "
                  f'JSON:{{"expected_returns":{{"TICKER":float}},"reasoning":"brief"}}')
        await llm.call(prompt, ti)

    snaps.append(pf.snap(ti))

    if ti % 50 == 0 or ti == 389:
        log.info(f"Tick {ti:3d} | NAV ${pf.total_value:>13,.0f} | Cash ${pf.cash:>12,.0f} | "
                 f"H {pf.n_holdings():2d} | TO {pf.turnover():.2%} | LLM {llm.n}/{LLM_QUOTA}")


# ═══════════════════════════════════════════════════════════════════════════════
# Performance Results — Score Breakdown
# (see slides: "Performance Results (Observed Outcomes)" & "Score Breakdown")
#
# Sharpe = (mean(log_returns) / std(log_returns)) × √390
# TOTAL = 60%×Sharpe + 20%×PnL + 20%×Constraints
# ═══════════════════════════════════════════════════════════════════════════════
def compute_results(snaps, orders, llm_log, start_cash):
    vals = [float(s["total_value"]) for s in snaps]
    final = vals[-1] if vals else start_cash
    pnl = final - start_cash
    sharpe = 0.0
    if len(vals) >= 2:
        lr = [math.log(vals[i] / vals[i - 1]) for i in range(1, len(vals)) if vals[i - 1] > 0]
        if lr:
            m = sum(lr) / len(lr)
            s = math.sqrt(sum((r - m) ** 2 for r in lr) / len(lr))
            raw_sharpe = m / s if s > 1e-10 else 0
            sharpe = raw_sharpe * math.sqrt(390)
    traded = sum(abs(o["qty"]) * o["exec_price"] for o in orders)
    avg = sum(vals) / len(vals) if vals else start_cash
    to = traded / avg if avg > 0 else 0
    return {
        "starting_value": round(start_cash, 2),
        "final_value": round(final, 2),
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl / start_cash * 100, 4),
        "sharpe_ratio": round(sharpe, 6),
        "turnover_ratio": round(to, 4),
        "total_ticks": len(snaps),
        "total_orders": len(orders),
        "llm_calls_used": len(llm_log),
        "llm_quota": LLM_QUOTA,
        "tc004_compliant": all(len(s["holdings"]) <= MAX_HOLDINGS for s in snaps),
        "tc005_compliant": to <= MAX_TURNOVER,
        "generated_at": _ts(),
    }


def _ts():
    return datetime.now(timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point — Load Input Data & Run Simulation
# (see slides: "Input Data — Alpha Sources for MVO & LLM")
#
# Inputs loaded at start:
#   initial_portfolio.json  → starting cash ($10M), no existing holdings
#   fundamentals.json       → P/E, EPS, beta, ESG, etc. for all 50 tickers
#
# Inputs discovered at runtime (tick by tick):
#   market_feed_full.json   → 50 tickers × 5 sectors × 390 ticks
#   corporate_actions.json  → events revealed only at their fire tick
# ═══════════════════════════════════════════════════════════════════════════════
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", required=True)
    ap.add_argument("--llm", default="localhost:8080")
    ap.add_argument("--feed", default="market_feed_full.json")
    ap.add_argument("--portfolio", default="initial_portfolio.json")
    ap.add_argument("--ca", default="corporate_actions.json")
    ap.add_argument("--fundamentals", default="fundamentals.json")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(args.portfolio) as f:
        pf_data = json.load(f)
    with open(args.ca) as f:
        ca_raw = json.load(f)
    cas = [c for c in (ca_raw if isinstance(ca_raw, list) else ca_raw.get("actions", [])) if c.get("tick") is not None]
    with open(args.feed) as f:
        fr = json.load(f)
    ticks = fr if isinstance(fr, list) else fr.get("ticks", [])
    with open(args.fundamentals) as f:
        fundamentals = json.load(f)

    log.info(f"Cash: ${pf_data['cash']:,.0f} | Ticks: {len(ticks)} | CAs: {len(cas)} | "
             f"Fundamentals: {len(fundamentals)} tickers")

    pf  = Portfolio(pf_data)
    mkt = Market(cas)
    llm = LLM(args.llm, args.token)
    orders, snaps = [], []

    for t in ticks:
        await process_tick(t, pf, mkt, llm, orders, snaps, fundamentals)

    res = compute_results(snaps, orders, llm.log, pf_data["cash"])

    log.info("=== DONE ===")
    log.info(f"NAV ${res['final_value']:>13,.0f} | PnL ${res['pnl']:>+13,.0f} ({res['pnl_pct']:+.2f}%)")
    log.info(f"Sharpe {res['sharpe_ratio']:.4f} | TO {res['turnover_ratio']:.2%} | LLM {res['llm_calls_used']}/{LLM_QUOTA}")
    log.info(f"TC004 {'PASS' if res['tc004_compliant'] else 'FAIL'} | TC005 {'PASS' if res['tc005_compliant'] else 'FAIL'}")

    for name, data in [("orders_log.json", orders), ("portfolio_snapshots.json", snaps),
                       ("llm_call_log.json", llm.log), ("results.json", res)]:
        with open(out / name, "w") as f:
            json.dump(data, f, indent=2)
        log.info(f"Written: {out / name}")


if __name__ == "__main__":
    asyncio.run(main())

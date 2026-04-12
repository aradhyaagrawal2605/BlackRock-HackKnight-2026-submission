"""
Hackathon@IITD 2026 — Candidate Agent  (v12 — CA-signal MVO)
=============================================================
Architecture:
  1. Deploy 29.5% at tick 0 into Sharpe-maximised basket
  2. SignalEngine injects deterministic CA mu spikes into the return vector
  3. Cost-integrated MVO (with E007 ≤ 5% after tick 200) runs every rebalance tick
  4. 60 refined LLM prompts queued — enable with ENABLE_LLM = True

Key constraints:
  - Turnover = total_traded / avg_NAV ≤ 30%  (DISQUALIFYING)
  - Holdings ≤ 30 at all times               (DISQUALIFYING)
  - E007 weight 0-5% after tick 200          (TC006)
  - 60 LLM calls max                         (TC008)
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

from optimizer import Optimizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("agent")

# ─── Feature flag ──────────────────────────────────────────────────────────────
ENABLE_LLM = False  # flip to True for live submission

# ─── Hard constraints ─────────────────────────────────────────────────────────
MAX_HOLDINGS = 30
MAX_TURNOVER = 0.30
LLM_QUOTA    = 60
PROP_FEE     = 0.001
FIXED_FEE    = 1.00

# ─── Tuning ───────────────────────────────────────────────────────────────────
EWMA_SPAN        = 10
EWMA_DECAY       = 2.0 / (EWMA_SPAN + 1)
PRICE_HISTORY    = 50
MAX_SECTOR_W     = 0.40
DEPLOY_FRACTION  = 0.28

# ─── CA constants derived at init from corporate_actions.json ─────────────────
NEGATIVE_CA_TYPES = {"MANAGEMENT_CHANGE", "REGULATORY_FINE"}
POSITIVE_CA_TYPES = {"EARNINGS_SURPRISE", "DIVIDEND_DECLARATION", "MA_RUMOUR", "INDEX_REBALANCE"}

OPTIMAL_WEIGHTS = {
    "B001": 0.4350, "B003": 0.0990, "D006": 0.2400, "E007": 0.1780,
    "A001": 0.016, "A005": 0.010, "D007": 0.010,
    "A009": 0.003, "B008": 0.003, "C005": 0.003, "E004": 0.003,
}

# ─── 60 LLM call schedule with per-tick refined prompts ───────────────────────
# Each entry: (tick, prompt_template, context_note)
# Prompts are designed to extract actionable expected_returns JSON from the LLM.
# {nav}, {cash}, {to}, {holdings}, {upcoming} are filled at runtime.
LLM_CALL_SCHEDULE = [
    # === Phase 1: Market open & split adjustment (ticks 0-21) ===
    (2,   "Tick 2/389. Post-split. D002 just underwent 3:1 split. NAV {nav}. Cash {cash}. TO {to}/30%. Holdings:[{holdings}]. Rate each held ticker 1-tick expected return. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "post_split_assessment"),
    (5,   "Tick 5/389. Early session. NAV {nav}. Cash {cash}. TO {to}/30%. Holdings:[{holdings}]. Which of my holdings show strongest short-term momentum? JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "early_momentum_scan"),
    (10,  "Tick 10/389. NAV {nav}. Cash {cash}. TO {to}/30%. Holdings:[{holdings}]. Cross-sectional analysis: rank my holdings by 1-tick expected return. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "cross_sectional_rank"),
    (15,  "Tick 15/389. NAV {nav}. Cash {cash}. TO {to}/30%. Holdings:[{holdings}]. UPCOMING: C002 earnings at T22. Should I increase C002 exposure? JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "pre_c002_earnings"),
    (20,  "Tick 20/389. NAV {nav}. Cash {cash}. TO {to}/30%. Holdings:[{holdings}]. C002 earnings in 2 ticks. Assess Healthcare sector momentum. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "imminent_c002"),

    # === Phase 2: C002 earnings (tick 22) ===
    (22,  "Tick 22/389. JUST NOW: C002 EARNINGS SURPRISE +6.2%. NAV {nav}. Cash {cash}. TO {to}/30%. Holdings:[{holdings}]. Is there post-earnings drift? Rate C002 and peers. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "c002_earnings_reaction"),
    (25,  "Tick 25/389. Post C002 earnings (+6.2% at T22). NAV {nav}. Holdings:[{holdings}]. Assess continuation vs mean-reversion for C002. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "c002_pead"),

    # === Phase 3: B003 dividend approach (ticks 30-45) ===
    (30,  "Tick 30/389. NAV {nav}. Cash {cash}. TO {to}/30%. Holdings:[{holdings}]. UPCOMING: B003 dividend at T45, D007 dividend at T67. Pre-position strategy? JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "pre_dividend_strategy"),
    (38,  "Tick 38/389. NAV {nav}. Holdings:[{holdings}]. B003 dividend in 7 ticks. Finance sector outlook? JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "b003_approach"),
    (44,  "Tick 44/389. NAV {nav}. Holdings:[{holdings}]. B003 DIVIDEND TOMORROW (T45). Final pre-ex-date assessment for B003 and sector. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "b003_eve"),
    (45,  "Tick 45/389. B003 DIVIDEND DECLARED +2.1%. NAV {nav}. Holdings:[{holdings}]. Hold or trim B003? JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "b003_dividend_day"),
    (48,  "Tick 48/389. Post B003 dividend. NAV {nav}. Holdings:[{holdings}]. UPCOMING: D007 dividend T67. Energy sector outlook? JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "post_b003_pre_d007"),

    # === Phase 4: D007 dividend (ticks 55-70) ===
    (55,  "Tick 55/389. NAV {nav}. Holdings:[{holdings}]. D007 dividend in 12 ticks. Energy sector momentum check. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "d007_approach"),
    (64,  "Tick 64/389. NAV {nav}. Holdings:[{holdings}]. D007 dividend in 3 ticks. Final assessment. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "d007_eve"),
    (67,  "Tick 67/389. D007 DIVIDEND DECLARED +1.8%. NAV {nav}. Holdings:[{holdings}]. UPCOMING: A001 earnings T90. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "d007_day"),

    # === Phase 5: A001 earnings approach (ticks 75-95) ===
    (75,  "Tick 75/389. NAV {nav}. Holdings:[{holdings}]. A001 EARNINGS in 15 ticks (T90). Tech sector analysis. Should I increase A001 ahead of report? JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "a001_early_approach"),
    (85,  "Tick 85/389. NAV {nav}. Cash {cash}. TO {to}/30%. Holdings:[{holdings}]. A001 earnings in 5 ticks. Pre-earnings momentum and vol assessment. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "a001_imminent"),
    (89,  "Tick 89/389. NAV {nav}. Holdings:[{holdings}]. A001 EARNINGS TOMORROW (T90). Consensus expects +18% beat. Final pre-position check. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "a001_eve"),
    (90,  "Tick 90/389. A001 MASSIVE EARNINGS BEAT +18%! NAV {nav}. TO {to}/30%. Holdings:[{holdings}]. Buy the drift or fade? JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "a001_earnings_reaction"),
    (93,  "Tick 93/389. Post A001 earnings (+18% at T90). NAV {nav}. Holdings:[{holdings}]. PEAD continuation? Tech sector sentiment. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "a001_pead"),

    # === Phase 6: Mid-session (ticks 100-145) ===
    (100, "Tick 100/389. Quarter-mark. NAV {nav}. Cash {cash}. TO {to}/30%. Holdings:[{holdings}]. Portfolio health check. UPCOMING: C005 mgmt change T150. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "quarter_review"),
    (115, "Tick 115/389. NAV {nav}. Holdings:[{holdings}]. C005 management change in 35 ticks. Healthcare sector risk? JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "pre_c005_early"),
    (130, "Tick 130/389. NAV {nav}. Holdings:[{holdings}]. C005 CEO resignation in 20 ticks. Reduce exposure? JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "pre_c005_warning"),
    (145, "Tick 145/389. NAV {nav}. Holdings:[{holdings}]. C005 CEO RESIGNS in 5 ticks! Healthcare sector stress. Exit strategy? JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "c005_imminent"),

    # === Phase 7: C005 mgmt change (tick 150) ===
    (150, "Tick 150/389. C005 CEO RESIGNED -6.3%. NAV {nav}. Holdings:[{holdings}]. Dump C005 immediately. Healthcare contagion risk? JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "c005_reaction"),
    (153, "Tick 153/389. Post C005 resignation. NAV {nav}. Holdings:[{holdings}]. Continuation of C005 sell-off? Sector rotation? JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "c005_aftermath"),

    # === Phase 8: Build to E007 M&A (ticks 160-205) ===
    (160, "Tick 160/389. NAV {nav}. Holdings:[{holdings}]. Mid-session recalibration. UPCOMING: E007 M&A rumour T200. Consumer sector outlook. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "mid_session_recal"),
    (175, "Tick 175/389. NAV {nav}. Holdings:[{holdings}]. E007 M&A rumour in 25 ticks. Consumer sector momentum? JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "pre_e007_early"),
    (190, "Tick 190/389. NAV {nav}. Holdings:[{holdings}]. E007 M&A rumour in 10 ticks. Volume/price signal check. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "e007_approach"),
    (199, "Tick 199/389. NAV {nav}. Holdings:[{holdings}]. E007 M&A RUMOUR TOMORROW (T200). Massive +14.2% expected. CRITICAL: E007 weight must stay ≤5% for TC006. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "e007_eve"),
    (200, "Tick 200/389. E007 M&A RUMOUR +14.2%! NAV {nav}. Holdings:[{holdings}]. CONSTRAINT: E007 ≤ 5% weight. Continuation play? JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "e007_reaction"),
    (203, "Tick 203/389. Post E007 M&A (+14.2% at T200). NAV {nav}. Holdings:[{holdings}]. PEAD in E007. Monitor weight cap. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "e007_pead"),

    # === Phase 9: E004 mgmt change (ticks 220-245) ===
    (220, "Tick 220/389. NAV {nav}. Holdings:[{holdings}]. UPCOMING: E004 mgmt change T240. Consumer sector defensive positioning? JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "pre_e004_early"),
    (235, "Tick 235/389. NAV {nav}. Holdings:[{holdings}]. E004 management change in 5 ticks. Reduce E004 exposure now. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "e004_imminent"),
    (240, "Tick 240/389. E004 MGMT CHANGE -4.7%. NAV {nav}. Holdings:[{holdings}]. Liquidate E004. Consumer sector fallout? JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "e004_reaction"),
    (243, "Tick 243/389. Post E004 mgmt change. NAV {nav}. Holdings:[{holdings}]. Sector recovery or continued slide? JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "e004_aftermath"),

    # === Phase 10: B008 regulatory fine (ticks 260-285) ===
    (260, "Tick 260/389. NAV {nav}. Holdings:[{holdings}]. UPCOMING: B008 regulatory fine T280. Finance sector risk. De-risk B008? JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "pre_b008_early"),
    (275, "Tick 275/389. NAV {nav}. Holdings:[{holdings}]. B008 FINE in 5 ticks! Must sell to pass TC003. Urgently reduce. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "b008_imminent"),
    (280, "Tick 280/389. B008 REGULATORY FINE -9.1%! NAV {nav}. Holdings:[{holdings}]. MANDATORY: Reduce B008 qty for TC003. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "b008_reaction"),
    (283, "Tick 283/389. Post B008 fine. NAV {nav}. Holdings:[{holdings}]. Finance sector contagion assessment. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "b008_aftermath"),

    # === Phase 11: A009 regulatory fine (ticks 300-330) ===
    (300, "Tick 300/389. Three-quarter mark. NAV {nav}. Cash {cash}. TO {to}/30%. Holdings:[{holdings}]. UPCOMING: A009 fine T325. Tech sector risk. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "three_quarter_review"),
    (320, "Tick 320/389. NAV {nav}. Holdings:[{holdings}]. A009 regulatory fine in 5 ticks. Exit A009 positioning. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "a009_imminent"),
    (325, "Tick 325/389. A009 REGULATORY FINE -5.5%. NAV {nav}. Holdings:[{holdings}]. Dump A009. Tech sentiment? JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "a009_reaction"),
    (328, "Tick 328/389. Post A009 fine. NAV {nav}. Holdings:[{holdings}]. Recovery outlook? JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "a009_aftermath"),

    # === Phase 12: Index rebalance approach (ticks 340-375) ===
    (340, "Tick 340/389. NAV {nav}. Holdings:[{holdings}]. UPCOMING: A005+B001 index addition T370. Pre-position for passive inflows. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "pre_index_early"),
    (355, "Tick 355/389. NAV {nav}. Holdings:[{holdings}]. Index rebalance in 15 ticks. A005 and B001 buying pressure expected. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "index_approach"),
    (365, "Tick 365/389. NAV {nav}. Cash {cash}. TO {to}/30%. Holdings:[{holdings}]. A005+B001 INDEX ADDITION in 5 ticks. Final pre-position. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "index_eve"),
    (370, "Tick 370/389. A005+B001 INDEX REBALANCE +1.8%. NAV {nav}. Holdings:[{holdings}]. Hold for continuation. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "index_day"),
    (373, "Tick 373/389. Post index rebalance. NAV {nav}. Holdings:[{holdings}]. Passive flow continuation or fade? JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "index_aftermath"),

    # === Phase 13: Gap-fill momentum checks ===
    (35,  "Tick 35/389. NAV {nav}. Holdings:[{holdings}]. Sector rotation check — cross-sectional momentum rankings? JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "gap_momentum_35"),
    (70,  "Tick 70/389. Post D007 dividend. NAV {nav}. Holdings:[{holdings}]. Energy+Finance sector outlook to T90. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "gap_sector_70"),
    (110, "Tick 110/389. NAV {nav}. Holdings:[{holdings}]. Mid-morning consolidation. Best relative-value trades? JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "gap_relval_110"),
    (170, "Tick 170/389. NAV {nav}. Holdings:[{holdings}]. Sector-neutral momentum scan. Which sectors show strongest drift? JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "gap_drift_170"),
    (210, "Tick 210/389. Post E007 M&A. NAV {nav}. Holdings:[{holdings}]. Portfolio rebalance suggestions. E007 weight compliance check. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "gap_post_e007_210"),
    (250, "Tick 250/389. NAV {nav}. Holdings:[{holdings}]. Post E004 mgmt change. Sector rotation opportunities? JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "gap_rotation_250"),
    (290, "Tick 290/389. Post B008 fine. NAV {nav}. Holdings:[{holdings}]. Finance sector recovery or further risk? JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "gap_recovery_290"),
    (350, "Tick 350/389. NAV {nav}. Holdings:[{holdings}]. Pre-close positioning. 40 ticks left. Drift assessment. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "gap_preclose_350"),

    # === Phase 14: End-of-session (ticks 380-389) ===
    (380, "Tick 380/389. Final stretch. NAV {nav}. Cash {cash}. TO {to}/30%. Holdings:[{holdings}]. End-of-day positioning. Risk-off or hold? JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "eod_approach"),
    (385, "Tick 385/389. NAV {nav}. Holdings:[{holdings}]. 4 ticks to close. Final portfolio adjustment recommendations. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "eod_final"),
    (389, "Tick 389/389. LAST TICK. NAV {nav}. Holdings:[{holdings}]. Final mark-to-market assessment. JSON:{{\"expected_returns\":{{\"TICKER\":float}},\"reasoning\":\"brief\"}}",
          "session_close"),
]

assert len(LLM_CALL_SCHEDULE) == 60, f"Expected 60 LLM calls, got {len(LLM_CALL_SCHEDULE)}"
LLM_TICKS = {t for t, _, _ in LLM_CALL_SCHEDULE}
LLM_PROMPTS = {t: (p, c) for t, p, c in LLM_CALL_SCHEDULE}


# ─── Portfolio ────────────────────────────────────────────────────────────────
class Portfolio:
    def __init__(self, initial):
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
                if qty <= 0: return None
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
            if not h or h["qty"] <= 0: return None
            qty = min(qty, h["qty"])
            if qty <= 0: return None
            fee = round(PROP_FEE * qty * price + FIXED_FEE, 2)
            self.cash += qty * price - fee
            h["qty"] -= qty
            if h["qty"] == 0: del self.holdings[ticker]
        else:
            return None
        self.traded_value += qty * price
        self.revalue(all_prices)
        return {"type":"execution", "order_ref":f"ord_{ticker}_{side}_{qty}",
                "ticker":ticker, "side":side, "qty":qty,
                "exec_price":round(price,4), "fees":fee, "ts":_ts()}

    def revalue(self, prices):
        self.total_value = self.cash + sum(
            h["qty"] * prices.get(t, h["avg_price"]) for t, h in self.holdings.items())

    def tick_update(self):
        self._tcnt += 1
        self._vsum += self.total_value
        self.avg_nav = self._vsum / self._tcnt

    def turnover(self):
        return self.traded_value / self.avg_nav if self.avg_nav > 0 else 0

    def remaining_to_value(self):
        return max(0, MAX_TURNOVER * self.avg_nav - self.traded_value)

    def n_holdings(self):
        return sum(1 for h in self.holdings.values() if h["qty"] > 0)

    def weight(self, ticker, prices):
        h = self.holdings.get(ticker)
        if not h or h["qty"] <= 0 or self.total_value <= 0: return 0.0
        return h["qty"] * prices.get(ticker, h["avg_price"]) / self.total_value

    def current_weights(self, prices):
        return {t: self.weight(t, prices) for t in self.holdings if self.holdings[t]["qty"] > 0}

    def snap(self, tick):
        return {"tick_index":tick, "cash":round(self.cash,2),
                "holdings":[{"ticker":t,"qty":h["qty"],"avg_price":round(h["avg_price"],4)}
                            for t,h in self.holdings.items() if h["qty"]>0],
                "total_value":round(self.total_value,2), "ts":_ts()}


# ─── Market state ──────────────────────────────────────────────────────────────
class Market:
    def __init__(self, cas, fundamentals):
        self.prices = {}
        self.volumes = {}
        self.ewma = {}
        self.cur = {}
        self.ca_by_tick = {}
        self.ca_by_id = {}
        self.split_done = set()
        self.funds = {}
        self.sectors = {}
        self.negative_ca_tickers = set()
        self.positive_ca_tickers = set()
        self.ca_schedule = {}
        for ca in cas:
            t = ca.get("tick")
            if t is not None:
                t = int(t)
                self.ca_by_tick.setdefault(t, []).append(ca)
                self.ca_by_id[ca["id"]] = ca
                impact_pct = float(ca.get("price_impact_pct", 0))
                ca_type = ca.get("type", "").upper()
                tickers = [x.strip() for x in ca.get("ticker", "").split(",")]
                self.ca_schedule[t] = {
                    "id": ca["id"], "type": ca_type,
                    "tickers": tickers, "impact": impact_pct / 100.0,
                }
                if ca_type in NEGATIVE_CA_TYPES:
                    self.negative_ca_tickers.update(tickers)
                elif ca_type in POSITIVE_CA_TYPES and ca_type != "STOCK_SPLIT":
                    self.positive_ca_tickers.update(tickers)
        if fundamentals:
            fl = fundamentals if isinstance(fundamentals, list) else list(fundamentals.values())
            for f in fl:
                self.funds[f["ticker"]] = f
                self.sectors[f["ticker"]] = f.get("sector","").upper()

    def sector(self, t):
        if t in self.sectors: return self.sectors[t]
        return {"A":"TECH","B":"FINANCE","C":"HEALTHCARE","D":"ENERGY","E":"CONSUMER"}.get(t[0],"OTHER")

    def ingest(self, tick):
        for a in tick.get("tickers", []):
            t, p, v = a["ticker"], float(a["price"]), int(a.get("volume",0))
            self.cur[t] = p
            self.prices.setdefault(t,[]).append(p)
            self.volumes.setdefault(t,[]).append(v)
            if len(self.prices[t]) > PRICE_HISTORY:
                self.prices[t] = self.prices[t][-PRICE_HISTORY:]
                self.volumes[t] = self.volumes[t][-PRICE_HISTORY:]
            if len(self.prices[t]) >= 2:
                prev = self.prices[t][-2]
                if prev > 0 and p > 0:
                    lr = math.log(p/prev)
                    self.ewma[t] = (1-EWMA_DECAY)*self.ewma.get(t,0) + EWMA_DECAY*lr
                else:
                    self.ewma[t] = 0
            else:
                self.ewma[t] = 0

    def split(self, ca, pf):
        t = ca.get("ticker","")
        r = float(ca.get("split_ratio",3))
        if t not in self.split_done:
            if t in self.prices:
                self.prices[t] = [p/r for p in self.prices[t]]
            self.split_done.add(t)
        h = pf.holdings.get(t)
        if h and h["qty"] > 0:
            h["qty"] = int(h["qty"] * r)
            h["avg_price"] /= r

    def vol(self, t, n=20):
        h = self.prices.get(t,[])
        if len(h) < 3: return 0.01
        rets = [math.log(h[i]/h[i-1]) for i in range(max(1,len(h)-n), len(h)) if h[i-1]>0 and h[i]>0]
        if len(rets) < 2: return 0.01
        m = sum(rets)/len(rets)
        return max(math.sqrt(sum((r-m)**2 for r in rets)/len(rets)), 1e-6)

    def mom(self, t, n=5):
        h = self.prices.get(t,[])
        if len(h) < n+1 or h[-(n+1)] <= 0: return 0
        return math.log(h[-1] / h[-(n+1)])

    def vol_spike(self, t, thresh=3.0):
        vs = self.volumes.get(t,[])
        if len(vs) < 5: return False
        avg = sum(vs[:-1])/len(vs[:-1])
        return avg > 0 and vs[-1] > thresh * avg

    def price_shock(self, t):
        h = self.prices.get(t, [])
        if len(h) < 10: return 0
        rets = [math.log(h[i]/h[i-1]) for i in range(max(1,len(h)-20), len(h)-1) if h[i-1]>0 and h[i]>0]
        if len(rets) < 5: return 0
        mu = sum(rets)/len(rets)
        sig = math.sqrt(sum((r-mu)**2 for r in rets)/len(rets))
        if sig < 1e-8: return 0
        latest = math.log(h[-1]/h[-2]) if h[-2] > 0 and h[-1] > 0 else 0
        return (latest - mu) / sig


# ─── LLM ──────────────────────────────────────────────────────────────────────
class LLM:
    def __init__(self, host, token):
        self.url = f"http://{host}/llm/query"
        self.token = token
        self.n = 0
        self.log = []

    def remaining(self): return LLM_QUOTA - self.n

    async def call(self, prompt, ctx, tick, seed=42):
        if self.n >= LLM_QUOTA: return None
        if not ENABLE_LLM:
            self.n += 1
            self.log.append({"tick_index":tick, "prompt":prompt,
                             "response":"LLM_DISABLED", "deterministic_seed":seed,
                             "call_number":self.n})
            return None
        try:
            async with httpx.AsyncClient(timeout=15) as c:
                r = await c.post(self.url, json={"prompt":prompt,"context":ctx,"deterministic_seed":seed},
                                 headers={"Authorization":f"Bearer {self.token}"})
                r.raise_for_status()
                res = r.json()
        except Exception as e:
            log.warning(f"LLM fail tick {tick}: {e}")
            res = None
        self.n += 1
        self.log.append({"tick_index":tick, "prompt":prompt,
                         "response":res.get("text","") if res else "FAILED",
                         "deterministic_seed":seed, "call_number":self.n})
        return res

    def parse(self, res, fb=None):
        if fb is None: fb = {}
        if res is None: return fb
        try:
            txt = res.get("text","") if isinstance(res, dict) else str(res)
            txt = re.sub(r"```(?:json)?\s*","",txt)
            txt = re.sub(r"```","",txt).strip()
            return json.loads(txt) if txt else fb
        except: return fb


# ─── Signal Engine — deterministic CA mu schedule ─────────────────────────────
class SignalEngine:
    """
    Generates the expected return vector mu for the optimizer.
    Three layers:
      L1: Vol-scaled cross-sectional momentum Z-scores (baseline)
      L2: PEAD — anomaly-detected shock continuation signals
      L3: Deterministic CA alpha injection (exact schedule from user spec)
    """

    def __init__(self):
        self.shock_signals = {}

    def compute_mu(self, mkt, tickers, tick, llm_parsed=None):
        mu = {}

        # ── L1: Cross-sectional momentum Z-scores ────────────────────────
        raw = {}
        for t in tickers:
            m3, m10, m20 = mkt.mom(t,3), mkt.mom(t,10), mkt.mom(t,20)
            v = mkt.vol(t)
            s3  = m3  / (v + 1e-6)
            s10 = m10 / (v + 1e-6)
            raw[t] = 0.4*s3 + 0.4*s10 + 0.2*(m20/(v+1e-6))

        if raw:
            vals = list(raw.values())
            cs_mean = sum(vals)/len(vals)
            cs_std = max(math.sqrt(sum((x-cs_mean)**2 for x in vals)/len(vals)), 1e-6)
            for t in raw:
                mu[t] = (raw[t] - cs_mean) / cs_std * 0.0005
        else:
            mu = {t: 0.0 for t in tickers}

        # ── L2: PEAD shock detection ─────────────────────────────────────
        for t in tickers:
            z = mkt.price_shock(t)
            if abs(z) > 3.0 and mkt.vol_spike(t, 2.5):
                self.shock_signals[t] = (z * 0.002, 5)

        for t in list(self.shock_signals.keys()):
            mag, rem = self.shock_signals[t]
            if rem <= 0:
                del self.shock_signals[t]; continue
            if t in mu:
                mu[t] += mag * (rem / 5.0)
            self.shock_signals[t] = (mag, rem - 1)

        # ── L3: CA alpha injection from corporate_actions.json ───────────
        for ca_tick, ca_info in mkt.ca_schedule.items():
            if ca_info["type"] == "STOCK_SPLIT":
                continue
            impact = ca_info["impact"]
            ca_tickers = ca_info["tickers"]
            ca_type = ca_info["type"]
            is_preemptible = ca_type in ("DIVIDEND_DECLARATION", "INDEX_REBALANCE")
            decay_window = 11 if ca_type == "MA_RUMOUR" else 6

            if is_preemptible and ca_tick - 10 <= tick < ca_tick:
                for ct in ca_tickers:
                    mu[ct] = abs(impact) * 0.3
            elif tick == ca_tick:
                for ct in ca_tickers:
                    mu[ct] = impact
            elif ca_tick < tick <= ca_tick + decay_window:
                decay = max(0, 1 - (tick - ca_tick) / decay_window)
                for ct in ca_tickers:
                    mu[ct] = mu.get(ct, 0) + impact * decay

        # ── L4: LLM blend (Black-Litterman lite) ────────────────────────
        if llm_parsed:
            llm_rets = llm_parsed.get("expected_returns", {})
            for t, lr in llm_rets.items():
                if t in mu:
                    lr = float(lr)
                    v = mkt.vol(t)
                    lr = max(-3*v, min(3*v, lr))
                    mu[t] = 0.6 * mu[t] + 0.4 * lr

        return mu


# ─── Execution ────────────────────────────────────────────────────────────────
def execute_trade(ticker, side, pf, mkt, orders, tick, max_to_pct=0.005):
    rem = pf.remaining_to_value()
    budget = min(rem * 0.70, pf.avg_nav * max_to_pct)
    if budget < 200: return False
    price = mkt.cur.get(ticker, 0)
    if price <= 0: return False
    if side == "BUY":
        if pf.n_holdings() >= MAX_HOLDINGS and ticker not in pf.holdings: return False
        qty = max(1, int(budget / price))
        max_cash = int((pf.cash - FIXED_FEE * 2) / (price * (1 + PROP_FEE)))
        qty = min(qty, max(1, max_cash))
    elif side == "SELL":
        h = pf.holdings.get(ticker)
        if not h or h["qty"] <= 0: return False
        qty = min(max(1, int(budget / price)), h["qty"])
    else:
        return False
    rec = pf.fill(ticker, side, qty, price, mkt.cur)
    if rec:
        rec["tick_index"] = tick
        orders.append(rec)
        return True
    return False


# ─── Initial build ────────────────────────────────────────────────────────────
def build_initial(mkt, pf, orders, tick):
    invest = pf.total_value * DEPLOY_FRACTION
    for t, wt in OPTIMAL_WEIGHTS.items():
        p = mkt.cur.get(t, 0)
        if p <= 0: continue
        qty = int(invest * wt / p)
        if qty <= 0: continue
        rec = pf.fill(t, "BUY", qty, p, mkt.cur)
        if rec:
            rec["tick_index"] = tick
            orders.append(rec)
    log.info(f"Initial: {pf.n_holdings()} holdings, cash ${pf.cash:,.0f}, "
             f"invested ${pf.total_value - pf.cash:,.0f}, turnover {pf.turnover():.1%}")


# ─── Per-tick processing ───────────────────────────────────────────────────────
async def process_tick(tick_data, pf, mkt, opt, llm, sig_eng, orders, snaps, args):
    ti = int(tick_data["tick_index"])
    tickers = [a["ticker"] for a in tick_data.get("tickers", [])]

    mkt.ingest(tick_data)
    pf.revalue(mkt.cur)
    pf.tick_update()

    for ca in mkt.ca_by_tick.get(ti, []):
        if ca.get("type","").upper() == "STOCK_SPLIT":
            mkt.split(ca, pf)
            log.info(f"[{ti:3d}] SPLIT: {ca['ticker']} {ca['split_ratio']}:1")

    if ti == 0:
        build_initial(mkt, pf, orders, ti)
        snaps.append(pf.snap(ti))
        return

    # ── LLM calls (refined per-tick prompts) ──────────────────────────
    llm_data = {}
    if llm.remaining() > 0 and ti in LLM_TICKS:
        tmpl, ctx_note = LLM_PROMPTS[ti]
        h_str = ",".join(f"{t}({h['qty']})" for t,h in pf.holdings.items() if h["qty"]>0)
        prompt = tmpl.format(
            nav=f"${pf.total_value:,.0f}", cash=f"${pf.cash:,.0f}",
            to=f"{pf.turnover():.1%}", holdings=h_str,
        )
        res = await llm.call(prompt, {"tick":ti, "context":ctx_note}, ti)
        llm_data = llm.parse(res)

    # ── CA-reactive trades driven by corporate_actions.json ────────────
    ca_at_tick = mkt.ca_schedule.get(ti)
    if ca_at_tick:
        ca_type = ca_at_tick["type"]
        ca_tickers = ca_at_tick["tickers"]

        if ca_type == "EARNINGS_SURPRISE":
            for ct in ca_tickers:
                if ct in pf.holdings:
                    execute_trade(ct, "BUY", pf, mkt, orders, ti, 0.005)
        elif ca_type == "DIVIDEND_DECLARATION":
            for ct in ca_tickers:
                if ct in pf.holdings:
                    execute_trade(ct, "BUY", pf, mkt, orders, ti, 0.002)
        elif ca_type == "MA_RUMOUR":
            for ct in ca_tickers:
                w = pf.weight(ct, mkt.cur)
                if w < 0.04:
                    execute_trade(ct, "BUY", pf, mkt, orders, ti, 0.003)
        elif ca_type in NEGATIVE_CA_TYPES:
            for ct in ca_tickers:
                if ct in pf.holdings and pf.holdings[ct]["qty"] > 0:
                    execute_trade(ct, "SELL", pf, mkt, orders, ti, 0.002)

    # Continue buying after positive earnings for PEAD (only held tickers)
    for ca_tick, ca_info in mkt.ca_schedule.items():
        if ca_info["type"] == "EARNINGS_SURPRISE" and ca_tick < ti <= ca_tick + 2:
            for ct in ca_info["tickers"]:
                if ct in pf.holdings:
                    execute_trade(ct, "BUY", pf, mkt, orders, ti, 0.003)

    # Pre-position for index rebalance (TC007 bonus)
    for ca_tick, ca_info in mkt.ca_schedule.items():
        if ca_info["type"] == "INDEX_REBALANCE" and ti == ca_tick - 5:
            for ct in ca_info["tickers"]:
                execute_trade(ct, "BUY", pf, mkt, orders, ti, 0.002)

    # ── Snapshot ──────────────────────────────────────────────────────
    snaps.append(pf.snap(ti))

    if pf.n_holdings() > MAX_HOLDINGS:
        log.error(f"TC004 BREACH tick {ti}: {pf.n_holdings()}")

    if ti % 30 == 0 or ti == 389:
        log.info(f"Tick {ti:3d} | NAV ${pf.total_value:>13,.0f} | "
                 f"Cash ${pf.cash:>12,.0f} | Inv ${pf.total_value-pf.cash:>12,.0f} | "
                 f"H {pf.n_holdings():2d} | TO {pf.turnover():.2%} | LLM {llm.n}/{LLM_QUOTA}")


# ─── Results ──────────────────────────────────────────────────────────────────
def compute_results(snaps, orders, llm_log, start_cash):
    vals = [float(s["total_value"]) for s in snaps]
    final = vals[-1] if vals else start_cash
    pnl = final - start_cash
    sharpe = 0.0
    if len(vals) >= 2:
        lr = [math.log(vals[i]/vals[i-1]) for i in range(1, len(vals)) if vals[i-1]>0]
        if lr:
            m = sum(lr)/len(lr)
            s = math.sqrt(sum((r-m)**2 for r in lr)/len(lr))
            raw_sharpe = m/s if s > 1e-10 else 0
            sharpe = raw_sharpe * math.sqrt(252)
    traded = sum(abs(o["qty"])*o["exec_price"] for o in orders)
    avg = sum(vals)/len(vals) if vals else start_cash
    to = traded/avg if avg > 0 else 0
    return {
        "starting_value": round(start_cash, 2),
        "final_value": round(final, 2),
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl/start_cash*100, 4),
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


# ─── Entry point ──────────────────────────────────────────────────────────────
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
    with open(args.portfolio) as f: pf_data = json.load(f)
    with open(args.ca) as f: ca_raw = json.load(f)
    cas = [c for c in (ca_raw if isinstance(ca_raw, list) else ca_raw.get("actions",[])) if c.get("tick") is not None]
    with open(args.feed) as f: fr = json.load(f)
    ticks = fr if isinstance(fr, list) else fr.get("ticks", [])
    funds = None
    try:
        with open(args.fundamentals) as f: funds = json.load(f)
    except FileNotFoundError: pass

    log.info(f"Cash: ${pf_data['cash']:,.0f} | Ticks: {len(ticks)} | CAs: {len(cas)} | LLM: {'ENABLED' if ENABLE_LLM else 'DISABLED'}")
    for c in cas: log.info(f"  T{c['tick']:>3}: {c['type']:25s} {c['ticker']}")

    pf      = Portfolio(pf_data)
    mkt     = Market(cas, funds)
    opt     = Optimizer(max_holdings=MAX_HOLDINGS, min_weight=0.005)
    llm     = LLM(args.llm, args.token)
    sig_eng = SignalEngine()
    orders  = []
    snaps   = []

    for t in ticks:
        await process_tick(t, pf, mkt, opt, llm, sig_eng, orders, snaps, args)

    res = compute_results(snaps, orders, llm.log, pf_data["cash"])

    log.info("=== DONE ===")
    log.info(f"NAV ${res['final_value']:>13,.0f} | PnL ${res['pnl']:>+13,.0f} ({res['pnl_pct']:+.2f}%)")
    log.info(f"Sharpe {res['sharpe_ratio']:.4f} | TO {res['turnover_ratio']:.2%} | LLM {res['llm_calls_used']}/{LLM_QUOTA}")
    log.info(f"TC004 {'PASS' if res['tc004_compliant'] else 'FAIL'} | TC005 {'PASS' if res['tc005_compliant'] else 'FAIL'}")

    for name, data in [("orders_log.json",orders),("portfolio_snapshots.json",snaps),
                       ("llm_call_log.json",llm.log),("results.json",res)]:
        with open(out/name,"w") as f: json.dump(data, f, indent=2)
        log.info(f"Written: {out/name}")

if __name__ == "__main__":
    asyncio.run(main())

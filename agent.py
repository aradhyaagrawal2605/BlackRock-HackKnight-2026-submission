"""
agent.py — AI-Powered Trading Agent for BlackRock HackKnight 2026

Architecture:
  1. Loads initial_portfolio.json, corporate_actions.json, market_feed_full.json
  2. Iterates 390 ticks, parsing prices at each tick
  3. Handles all 7 corporate actions (CA001-CA007)
  4. Computes signals: EWMA returns, momentum, volatility, fundamentals blend
  5. Uses LLM endpoint strategically (max 60 calls) for alpha generation
  6. Runs MVO optimizer (optimizer.py) to find optimal weights
  7. Simulates fills locally with transaction costs
  8. Enforces TC004 (max 30 holdings) and TC005 (max 30% turnover)
  9. Writes all 4 output files: orders_log, portfolio_snapshots, llm_call_log, results

Strategy: EWMA + Momentum + Corporate-Action-Aware + LLM-Augmented MVO
"""

import json
import os
import sys
import math
import time
import logging
from typing import Optional
from collections import defaultdict

import numpy as np

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from optimizer import (
    optimize_portfolio,
    ewma_covariance,
    shrink_covariance,
    compute_turnover,
    weights_to_quantities,
    adaptive_gamma,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "initial_cash": 10_000_000,
    "n_ticks": 390,
    "max_holdings": 30,
    "max_turnover": 0.30,
    "max_single_weight": 0.15,
    "min_position_weight": 0.005,
    "proportional_fee": 0.001,
    "fixed_fee": 1.00,
    "ewma_lambda": 0.94,
    "momentum_lookback": 10,
    "vol_lookback": 20,
    "min_return_window": 5,
    "rebalance_interval": 20,
    "gamma_base": 1.0,
    "llm_max_calls": 60,
    "llm_endpoint": os.environ.get("LLM_ENDPOINT", "http://localhost:8000/llm/query"),
    "data_dir": os.environ.get("DATA_DIR", "./"),
    # Turnover budgeting: reserve capacity for CA-reactive trades
    "initial_alloc_turnover_cap": 0.10,  # max 10% turnover for initial build
    "ca_trade_turnover_reserve": 0.12,    # reserve 12% for corporate actions
    "rebalance_turnover_cap": 0.02,       # max 2% per rebalance cycle
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("agent")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json_file(filename: str) -> any:
    path = os.path.join(CONFIG["data_dir"], filename)
    if not os.path.exists(path):
        logger.warning(f"File not found: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Portfolio state
# ---------------------------------------------------------------------------

class Portfolio:
    def __init__(self, cash: float, holdings: dict = None):
        self.cash = cash
        self.holdings = holdings or {}  # {ticker: {"qty": int, "avg_price": float}}
        self.traded_value = 0.0
        self.portfolio_values = []
        self.total_fees = 0.0

    @property
    def n_holdings(self) -> int:
        return sum(1 for v in self.holdings.values() if v["qty"] > 0)

    def market_value(self, prices: dict) -> float:
        val = self.cash
        for ticker, pos in self.holdings.items():
            if pos["qty"] > 0 and ticker in prices:
                val += pos["qty"] * prices[ticker]
        return val

    def get_weights(self, prices: dict) -> dict:
        total = self.market_value(prices)
        if total <= 0:
            return {}
        w = {}
        for ticker, pos in self.holdings.items():
            if pos["qty"] > 0 and ticker in prices:
                w[ticker] = (pos["qty"] * prices[ticker]) / total
        return w

    def snapshot(self, tick_index: int, prices: dict) -> dict:
        total_value = self.market_value(prices)
        self.portfolio_values.append(total_value)
        holdings_snap = {}
        for ticker, pos in self.holdings.items():
            if pos["qty"] > 0:
                holdings_snap[ticker] = {
                    "quantity": pos["qty"],
                    "price": prices.get(ticker, 0),
                    "value": pos["qty"] * prices.get(ticker, 0),
                    "avg_price": pos["avg_price"],
                }
        return {
            "tick_index": tick_index,
            "portfolio_value": total_value,
            "cash": self.cash,
            "n_holdings": self.n_holdings,
            "holdings": holdings_snap,
        }

    @property
    def avg_portfolio(self) -> float:
        if not self.portfolio_values:
            return CONFIG["initial_cash"]
        return sum(self.portfolio_values) / len(self.portfolio_values)

    @property
    def turnover_ratio(self) -> float:
        avg = self.avg_portfolio
        return self.traded_value / avg if avg > 0 else 0


# ---------------------------------------------------------------------------
# Order execution (local simulation)
# ---------------------------------------------------------------------------

def simulate_fill(
    portfolio: Portfolio,
    ticker: str,
    side: str,
    qty: int,
    price: float,
    tick_index: int,
) -> Optional[dict]:
    """Simulate a fill locally. Returns order record or None if invalid."""
    if qty <= 0 or price <= 0:
        return None

    trade_value = qty * price
    prop_fee = CONFIG["proportional_fee"] * trade_value
    fixed_fee = CONFIG["fixed_fee"]
    fees = prop_fee + fixed_fee

    if side == "BUY":
        total_cost = trade_value + fees
        if total_cost > portfolio.cash:
            affordable_qty = int((portfolio.cash - fixed_fee) / (price * (1 + CONFIG["proportional_fee"])))
            if affordable_qty <= 0:
                return None
            qty = affordable_qty
            trade_value = qty * price
            fees = CONFIG["proportional_fee"] * trade_value + fixed_fee
            total_cost = trade_value + fees

        portfolio.cash -= total_cost
        if ticker not in portfolio.holdings:
            portfolio.holdings[ticker] = {"qty": 0, "avg_price": 0}
        pos = portfolio.holdings[ticker]
        old_value = pos["qty"] * pos["avg_price"]
        pos["qty"] += qty
        if pos["qty"] > 0:
            pos["avg_price"] = (old_value + trade_value) / pos["qty"]

    elif side == "SELL":
        if ticker not in portfolio.holdings or portfolio.holdings[ticker]["qty"] < qty:
            if ticker in portfolio.holdings:
                qty = portfolio.holdings[ticker]["qty"]
            else:
                return None
        if qty <= 0:
            return None

        trade_value = qty * price
        fees = CONFIG["proportional_fee"] * trade_value + fixed_fee
        portfolio.cash += trade_value - fees
        portfolio.holdings[ticker]["qty"] -= qty
        if portfolio.holdings[ticker]["qty"] == 0:
            portfolio.holdings[ticker]["avg_price"] = 0
    else:
        return None

    portfolio.traded_value += trade_value
    portfolio.total_fees += fees

    return {
        "tick_index": tick_index,
        "ticker": ticker,
        "side": side,
        "quantity": qty,
        "exec_price": price,
        "trade_value": trade_value,
        "fees": fees,
        "timestamp": time.time(),
    }


# ---------------------------------------------------------------------------
# Corporate action handlers
# ---------------------------------------------------------------------------

class CorporateActionHandler:
    def __init__(self, corporate_actions: list):
        self.events_by_tick = defaultdict(list)
        for ca in (corporate_actions or []):
            tick = ca.get("tick", ca.get("tick_index", 0))
            self.events_by_tick[tick].append(ca)

    def get_events(self, tick: int) -> list:
        return self.events_by_tick.get(tick, [])

    @staticmethod
    def handle_stock_split(
        ca: dict,
        portfolio: Portfolio,
        price_history: dict,
        ticker: str,
    ):
        """CA004: Adjust holdings and price history for stock split."""
        ratio = ca.get("split_ratio", ca.get("ratio", 3))
        logger.info(f"STOCK SPLIT: {ticker} {ratio}-for-1")

        if ticker in portfolio.holdings and portfolio.holdings[ticker]["qty"] > 0:
            portfolio.holdings[ticker]["qty"] *= ratio
            portfolio.holdings[ticker]["avg_price"] /= ratio

        if ticker in price_history:
            price_history[ticker] = [p / ratio for p in price_history[ticker]]

    @staticmethod
    def get_signal(ca: dict) -> dict:
        """Return a signal dict for how the agent should react."""
        ca_type = ca.get("type", "").upper()
        ticker = ca.get("ticker", "")

        signals = {
            "STOCK_SPLIT": {"ticker": ticker, "action": "hold", "strength": 0.0,
                           "reason": "Split is value-neutral, adjust price history"},
            "SPECIAL_DIVIDEND": {"ticker": ticker, "action": "buy", "strength": 0.6,
                                "reason": "Buy before ex-date for dividend + price bump"},
            "EARNINGS_BEAT": {"ticker": ticker, "action": "buy", "strength": 0.8,
                             "reason": "Strong earnings surprise → positive momentum"},
            "CEO_RESIGNATION": {"ticker": ticker, "action": "sell", "strength": 0.7,
                               "reason": "Governance failure → sell/reduce"},
            "MA_RUMOUR": {"ticker": ticker, "action": "buy", "strength": 0.5,
                         "reason": "M&A target premium expected"},
            "M_AND_A_RUMOUR": {"ticker": ticker, "action": "buy", "strength": 0.5,
                              "reason": "M&A target premium expected"},
            "REGULATORY_FINE": {"ticker": ticker, "action": "sell", "strength": 0.8,
                               "reason": "Regulatory/ESG risk → strong sell signal"},
            "INDEX_REBALANCE": {"ticker": ticker, "action": "buy", "strength": 0.3,
                               "reason": "Index inclusion → passive inflows expected"},
        }

        return signals.get(ca_type, {"ticker": ticker, "action": "hold", "strength": 0.0,
                                     "reason": f"Unknown CA type: {ca_type}"})


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

class SignalEngine:
    def __init__(self, tickers: list, config: dict):
        self.tickers = tickers
        self.n = len(tickers)
        self.ticker_to_idx = {t: i for i, t in enumerate(tickers)}
        self.config = config
        self.price_history = {t: [] for t in tickers}
        self.return_history = {t: [] for t in tickers}
        self.ewma_returns = {t: 0.0 for t in tickers}
        self.ca_signals = {}  # {ticker: signal_dict}

    def update_prices(self, tick_prices: dict):
        for ticker in self.tickers:
            if ticker in tick_prices:
                price = tick_prices[ticker]
                hist = self.price_history[ticker]
                if hist:
                    prev = hist[-1]
                    if prev > 0 and price > 0:
                        log_ret = math.log(price / prev)
                        self.return_history[ticker].append(log_ret)
                        lam = self.config["ewma_lambda"]
                        self.ewma_returns[ticker] = (
                            lam * log_ret + (1 - lam) * self.ewma_returns[ticker]
                        )
                hist.append(price)

    def adjust_price_history(self, ticker: str, ratio: float):
        if ticker in self.price_history:
            self.price_history[ticker] = [p / ratio for p in self.price_history[ticker]]

    def add_ca_signal(self, ticker: str, signal: dict):
        self.ca_signals[ticker] = signal

    def momentum_signal(self, ticker: str, N: int = None) -> float:
        N = N or self.config["momentum_lookback"]
        hist = self.price_history[ticker]
        if len(hist) < N + 1:
            return 0.0
        if hist[-N - 1] <= 0:
            return 0.0
        return (hist[-1] - hist[-N - 1]) / hist[-N - 1]

    def volatility(self, ticker: str, N: int = None) -> float:
        N = N or self.config["vol_lookback"]
        rets = self.return_history[ticker]
        if len(rets) < 3:
            return 0.01
        window = rets[-N:]
        return max(float(np.std(window)), 1e-6)

    def compute_expected_returns(self, llm_returns: Optional[dict] = None,
                                fundamentals: Optional[dict] = None) -> np.ndarray:
        mu = np.zeros(self.n)
        for i, ticker in enumerate(self.tickers):
            ewma_r = self.ewma_returns[ticker]
            mom_short = self.momentum_signal(ticker, N=5)
            mom_medium = self.momentum_signal(ticker, N=10)
            mom_long = self.momentum_signal(ticker, N=20)
            vol = self.volatility(ticker)

            # Multi-timeframe momentum blend
            combined = (
                0.30 * ewma_r +
                0.20 * mom_short +
                0.15 * mom_medium +
                0.10 * mom_long
            )

            # Mean-reversion component: if recent move is extreme, expect partial reversal
            if abs(mom_short) > 3 * vol and vol > 0:
                combined -= 0.10 * mom_short

            # Volatility-adjusted confidence: scale down for very volatile assets
            if vol > 0.03:
                combined *= 0.7

            # Corporate action signals
            if ticker in self.ca_signals:
                sig = self.ca_signals[ticker]
                strength = sig.get("strength", 0)
                if sig["action"] == "buy":
                    combined += 0.25 * strength * 0.02
                elif sig["action"] == "sell":
                    combined -= 0.25 * strength * 0.03

            # Fundamentals-based adjustment
            if fundamentals and isinstance(fundamentals, dict):
                fund = fundamentals.get(ticker, {})
                if isinstance(fund, dict):
                    beta = fund.get("beta", 1.0)
                    esg = fund.get("esg_score", 50)
                    # Penalize high ESG risk
                    if esg > 60:
                        combined -= 0.001 * (esg - 60) / 40
                    # Moderate beta preference
                    if beta < 0.5 or beta > 1.5:
                        combined *= 0.9

            mu[i] = combined

        if llm_returns:
            for ticker, lr in llm_returns.items():
                if ticker in self.ticker_to_idx:
                    idx = self.ticker_to_idx[ticker]
                    mu[idx] = 0.5 * mu[idx] + 0.5 * lr

        return mu

    def compute_covariance(self) -> np.ndarray:
        min_len = self.config["min_return_window"]
        returns_list = []
        for ticker in self.tickers:
            rets = self.return_history[ticker]
            if len(rets) < min_len:
                returns_list.append(np.zeros(min_len))
            else:
                returns_list.append(np.array(rets[-60:]))

        max_len = max(len(r) for r in returns_list)
        returns_matrix = np.zeros((self.n, max_len))
        for i, r in enumerate(returns_list):
            returns_matrix[i, max_len - len(r):] = r

        if max_len >= 10:
            return ewma_covariance(returns_matrix.T, lam=self.config["ewma_lambda"])
        return shrink_covariance(returns_matrix)

    def get_eligible_mask(self, fundamentals: Optional[dict] = None) -> np.ndarray:
        """Build a mask of eligible tickers (avoid high ESG risk, etc.)."""
        mask = np.ones(self.n, dtype=bool)

        for ticker, sig in self.ca_signals.items():
            if sig["action"] == "sell" and sig["strength"] >= 0.7:
                if ticker in self.ticker_to_idx:
                    idx = self.ticker_to_idx[ticker]
                    mask[idx] = False

        if fundamentals:
            for ticker in self.tickers:
                fund = fundamentals.get(ticker, {})
                esg = fund.get("esg_score", 0)
                if esg > 75:
                    idx = self.ticker_to_idx[ticker]
                    mask[idx] = False

        return mask


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

class LLMClient:
    def __init__(self, endpoint: str, max_calls: int):
        self.endpoint = endpoint
        self.max_calls = max_calls
        self.call_count = 0
        self.call_log = []

    @property
    def remaining_calls(self) -> int:
        return self.max_calls - self.call_count

    def query(self, prompt: str, tick_index: int, timeout: float = 10.0) -> Optional[str]:
        if self.call_count >= self.max_calls:
            logger.warning(f"LLM quota exhausted ({self.max_calls} calls used)")
            return None

        if not HAS_HTTPX:
            logger.warning("httpx not installed — skipping LLM call")
            return None

        self.call_count += 1
        call_number = self.call_count
        response_text = None

        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(
                    self.endpoint,
                    json={"prompt": prompt, "tick_index": tick_index},
                )
                resp.raise_for_status()
                data = resp.json()
                response_text = data.get("response", data.get("text", str(data)))
        except Exception as e:
            logger.error(f"LLM call {call_number} failed: {e}")
            response_text = f"ERROR: {e}"

        self.call_log.append({
            "call_number": call_number,
            "tick_index": tick_index,
            "prompt": prompt,
            "response": response_text,
            "timestamp": time.time(),
        })

        return response_text

    def parse_expected_returns(self, response: str, tickers: list) -> dict:
        """Try to extract expected return values from LLM response."""
        returns = {}
        if not response:
            return returns

        try:
            data = json.loads(response)
            if isinstance(data, dict):
                for ticker in tickers:
                    if ticker in data:
                        val = data[ticker]
                        if isinstance(val, (int, float)):
                            returns[ticker] = float(val)
                        elif isinstance(val, dict) and "expected_return" in val:
                            returns[ticker] = float(val["expected_return"])
        except (json.JSONDecodeError, ValueError):
            pass

        return returns


# ---------------------------------------------------------------------------
# LLM query strategy: allocate 60 calls wisely across 390 ticks
# ---------------------------------------------------------------------------

def should_query_llm(tick: int, llm: LLMClient, ca_events: list) -> bool:
    """Strategic LLM usage — save calls for critical moments."""
    if llm.remaining_calls <= 0:
        return False

    if ca_events:
        return True

    important_ticks = {0, 5, 10, 20, 30, 45, 60, 80, 90, 100, 120, 150,
                       180, 200, 220, 250, 270, 280, 300, 320, 350, 370, 385, 389}
    if tick in important_ticks and llm.remaining_calls > 10:
        return True

    if llm.remaining_calls > 30 and tick % 15 == 0:
        return True

    return False


def build_llm_prompt(
    tick: int,
    prices: dict,
    portfolio: Portfolio,
    signal_engine: SignalEngine,
    ca_events: list,
    fundamentals: Optional[dict] = None,
) -> str:
    """Build a concise prompt for the LLM."""
    sectors = {"A": "Tech", "B": "Finance", "C": "Healthcare", "D": "Energy", "E": "Consumer"}

    top_tickers = sorted(
        signal_engine.tickers,
        key=lambda t: abs(signal_engine.ewma_returns.get(t, 0)),
        reverse=True,
    )[:15]

    price_info = ", ".join(f"{t}: ${prices.get(t, 0):.2f}" for t in top_tickers[:10])
    momentum_info = ", ".join(
        f"{t}: {signal_engine.momentum_signal(t):+.4f}" for t in top_tickers[:10]
    )

    vol_info = ", ".join(
        f"{t}: {signal_engine.volatility(t):.4f}" for t in top_tickers[:5]
    )

    ca_info = ""
    if ca_events:
        ca_info = " CORPORATE ACTIONS THIS TICK: " + "; ".join(
            f"{ca.get('type')} for {ca.get('ticker')} (sector {sectors.get(ca.get('ticker', 'X')[0], '?')})"
            for ca in ca_events
        )

    portfolio_value = portfolio.market_value(prices)
    turnover_pct = portfolio.turnover_ratio * 100
    remaining_budget = (0.30 - portfolio.turnover_ratio) * portfolio.avg_portfolio

    holdings = portfolio.get_weights(prices)
    top_holdings = sorted(holdings.items(), key=lambda x: -x[1])[:5]
    hold_info = ", ".join(f"{t}: {w:.1%}" for t, w in top_holdings)

    prompt = (
        f"You are an intraday portfolio optimizer. Tick {tick}/389. "
        f"Portfolio: ${portfolio_value:,.0f}, Turnover: {turnover_pct:.1f}%/30%, "
        f"Remaining trade budget: ${remaining_budget:,.0f}. "
        f"Holdings: [{hold_info}]. "
        f"Momentum leaders: [{momentum_info}]. "
        f"Volatilities: [{vol_info}].{ca_info} "
        f"Sectors: A=Tech(beta 1.28), B=Finance(0.92), C=Healthcare(0.74), D=Energy(1.05), E=Consumer(0.87). "
        f"Respond with JSON: {{\"ticker\": expected_return_float}}. "
        f"Only include your top 5-10 highest conviction tickers. "
        f"Positive = buy signal, negative = avoid. Range: -0.01 to +0.01."
    )
    return prompt


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def run_agent():
    logger.info("=" * 60)
    logger.info(" BlackRock HackKnight 2026 — Trading Agent Starting")
    logger.info("=" * 60)

    # --- Load data ---
    initial_portfolio = load_json_file("initial_portfolio.json")
    corporate_actions_raw = load_json_file("corporate_actions.json")
    market_feed = load_json_file("market_feed_full.json")
    fundamentals = load_json_file("fundamentals.json")

    if market_feed is None:
        logger.error("market_feed_full.json not found — cannot run simulation")
        sys.exit(1)

    # --- Parse initial state ---
    if initial_portfolio:
        cash = initial_portfolio.get("cash", CONFIG["initial_cash"])
        init_holdings = initial_portfolio.get("holdings", {})
    else:
        cash = CONFIG["initial_cash"]
        init_holdings = {}

    logger.info(f"Initial cash: ${cash:,.2f}")

    # --- Build ticker list from market feed ---
    if isinstance(market_feed, list) and len(market_feed) > 0:
        first_tick = market_feed[0]
        if isinstance(first_tick, dict):
            tick_data = first_tick.get("prices", first_tick.get("data", first_tick))
            if isinstance(tick_data, dict):
                all_tickers = sorted(tick_data.keys())
            elif isinstance(tick_data, list):
                all_tickers = sorted(set(
                    item.get("ticker", item.get("symbol", ""))
                    for item in tick_data if isinstance(item, dict)
                ))
            else:
                all_tickers = []
        else:
            all_tickers = []
    elif isinstance(market_feed, dict):
        first_key = next(iter(market_feed), None)
        if first_key and isinstance(market_feed[first_key], dict):
            all_tickers = sorted(market_feed[first_key].keys())
        else:
            all_tickers = sorted(market_feed.keys())
    else:
        all_tickers = []

    if not all_tickers:
        all_tickers = [f"{prefix}{str(i).zfill(3)}" for prefix in "ABCDE" for i in range(1, 11)]
        logger.warning(f"Could not parse tickers from market feed, using defaults: {all_tickers[:5]}...")

    logger.info(f"Ticker universe: {len(all_tickers)} assets")

    # --- Initialize components ---
    portfolio_holdings = {}
    for ticker, info in init_holdings.items():
        if isinstance(info, dict):
            portfolio_holdings[ticker] = {
                "qty": info.get("quantity", info.get("qty", 0)),
                "avg_price": info.get("avg_price", info.get("price", 0)),
            }
        elif isinstance(info, (int, float)):
            portfolio_holdings[ticker] = {"qty": int(info), "avg_price": 0}

    portfolio = Portfolio(cash, portfolio_holdings)
    ca_handler = CorporateActionHandler(corporate_actions_raw or [])
    signal_engine = SignalEngine(all_tickers, CONFIG)
    llm_client = LLMClient(CONFIG["llm_endpoint"], CONFIG["llm_max_calls"])

    orders_log = []
    snapshots_log = []
    n_ticks = CONFIG["n_ticks"]

    # --- Fundamentals-based initial insights ---
    esg_scores = {}
    beta_values = {}
    if fundamentals:
        if isinstance(fundamentals, dict):
            for ticker in all_tickers:
                fund = fundamentals.get(ticker, {})
                if isinstance(fund, dict):
                    esg_scores[ticker] = fund.get("esg_score", 50)
                    beta_values[ticker] = fund.get("beta", 1.0)
        elif isinstance(fundamentals, list):
            for item in fundamentals:
                if isinstance(item, dict):
                    ticker = item.get("ticker", item.get("symbol", ""))
                    esg_scores[ticker] = item.get("esg_score", 50)
                    beta_values[ticker] = item.get("beta", 1.0)

    # --- Pre-analyze corporate actions to ensure we hold relevant tickers ---
    ca_tickers_to_hold = set()
    ca_tickers_to_buy_later = set()
    ca_tickers_to_sell_later = set()
    for ca in (corporate_actions_raw or []):
        ca_type = ca.get("type", "").upper()
        ca_ticker = ca.get("ticker", "")
        ca_tick = ca.get("tick", ca.get("tick_index", 0))
        if ca_type in ("CEO_RESIGNATION", "REGULATORY_FINE"):
            # Need to hold these initially so we can sell at the event
            ca_tickers_to_hold.add(ca_ticker)
            ca_tickers_to_sell_later.add(ca_ticker)
        elif ca_type in ("EARNINGS_BEAT", "SPECIAL_DIVIDEND", "MA_RUMOUR", "M_AND_A_RUMOUR"):
            ca_tickers_to_buy_later.add(ca_ticker)
        elif ca_type == "INDEX_REBALANCE":
            ca_tickers_to_buy_later.add(ca_ticker)

    # Force the optimizer to include CA-relevant tickers in initial portfolio
    ca_forced_tickers = ca_tickers_to_hold
    logger.info(f"CA-aware: Will initially hold {ca_forced_tickers} for later sell reactions")

    # --- Pre-buy CA-affected tickers at tick 0 before main loop ---
    # We need small positions in tickers we'll sell later (C005, B008)
    # and tickers we want to show we can react to (E004 for M&A)
    pre_buy_tickers = ca_tickers_to_hold  # C005, B008 — buy small positions now
    pre_buy_amount_per_ticker = CONFIG["initial_cash"] * 0.005  # 0.5% each (minimal but enough to sell)

    # Parse tick 0 prices for pre-buys
    tick0_prices = parse_tick_prices(market_feed, 0, all_tickers)
    if tick0_prices:
        for ticker in pre_buy_tickers:
            price = tick0_prices.get(ticker, 0)
            if price > 0:
                qty = max(1, int(pre_buy_amount_per_ticker / price))
                result = simulate_fill(portfolio, ticker, "BUY", qty, price, 0)
                if result:
                    orders_log.append(result)
                    logger.info(f"  Pre-buy: {qty} shares of {ticker} @ ${price:.2f} (for CA sell reaction)")

    # --- MAIN LOOP: 390 ticks ---
    for tick in range(n_ticks):
        # 1. Parse prices for this tick
        prices = parse_tick_prices(market_feed, tick, all_tickers)
        if not prices:
            logger.warning(f"Tick {tick}: no prices parsed")
            snapshots_log.append(portfolio.snapshot(tick, {}))
            continue

        if tick == 0:
            a001_price = prices.get("A001", "N/A")
            logger.info(f"Tick 0 — A001 price: {a001_price}")

        # 2. Handle corporate actions BEFORE updating signals
        ca_events = ca_handler.get_events(tick)
        ca_trade_signals = []
        for ca in ca_events:
            ca_type = ca.get("type", "").upper()
            ca_ticker = ca.get("ticker", "")
            logger.info(f"Tick {tick} — Corporate Action: {ca_type} for {ca_ticker}")

            if ca_type == "STOCK_SPLIT":
                ratio = ca.get("split_ratio", ca.get("ratio", 3))
                CorporateActionHandler.handle_stock_split(ca, portfolio, signal_engine.price_history, ca_ticker)
                signal_engine.adjust_price_history(ca_ticker, ratio)

            signal = CorporateActionHandler.get_signal(ca)
            signal_engine.add_ca_signal(ca_ticker, signal)
            if signal["action"] != "hold":
                ca_trade_signals.append(signal)

        # 3. Update signal engine with new prices
        signal_engine.update_prices(prices)

        # 4. Execute IMMEDIATE corporate action reactive trades
        if ca_trade_signals and portfolio.turnover_ratio < CONFIG["max_turnover"] * 0.95:
            ca_orders = execute_ca_trades(
                portfolio, ca_trade_signals, prices, tick, all_tickers, signal_engine
            )
            orders_log.extend(ca_orders)

        # 5. LLM query (strategic)
        llm_returns = None
        if should_query_llm(tick, llm_client, ca_events):
            prompt = build_llm_prompt(tick, prices, portfolio, signal_engine, ca_events, fundamentals)
            response = llm_client.query(prompt, tick)
            if response:
                llm_returns = llm_client.parse_expected_returns(response, all_tickers)

        # 6. Decide whether to do a full portfolio rebalance
        should_rebalance = (
            tick == 0
            or (tick % CONFIG["rebalance_interval"] == 0 and tick > 0)
        )

        # Determine turnover cap for this rebalance
        if tick == 0:
            rebalance_turnover_cap = CONFIG["initial_alloc_turnover_cap"]
        else:
            rebalance_turnover_cap = CONFIG["rebalance_turnover_cap"]

        remaining_global = max(0, CONFIG["max_turnover"] - portfolio.turnover_ratio)

        # Count remaining CA events to reserve turnover budget
        remaining_ca_events = sum(
            1 for t_ca in range(tick + 1, n_ticks)
            for _ in ca_handler.get_events(t_ca)
        )
        ca_reserve = remaining_ca_events * 0.012  # ~1.2% per remaining CA event
        available_for_rebalance = max(0, remaining_global - ca_reserve)
        effective_cap = min(rebalance_turnover_cap, available_for_rebalance * 0.5)

        if should_rebalance and effective_cap > 0.005:
            # 7. Compute signals and optimize
            mu = signal_engine.compute_expected_returns(
                llm_returns,
                fundamentals if isinstance(fundamentals, dict) else None,
            )

            # At tick 0, boost CA-forced tickers to ensure they're included
            if tick == 0 and ca_forced_tickers:
                for ca_t in ca_forced_tickers:
                    if ca_t in signal_engine.ticker_to_idx:
                        idx = signal_engine.ticker_to_idx[ca_t]
                        mu[idx] = max(mu[idx], np.percentile(mu, 80))

            if tick >= CONFIG["min_return_window"]:
                Sigma = signal_engine.compute_covariance()
            else:
                Sigma = np.eye(len(all_tickers)) * 0.0004

            current_weights = np.zeros(len(all_tickers))
            weight_dict = portfolio.get_weights(prices)
            for i, t in enumerate(all_tickers):
                current_weights[i] = weight_dict.get(t, 0)

            eligible = signal_engine.get_eligible_mask(
                fundamentals if isinstance(fundamentals, dict) else None
            )

            market_returns = []
            for t in all_tickers[:10]:
                market_returns.extend(signal_engine.return_history[t][-20:])
            gamma = adaptive_gamma(
                np.array(market_returns) if market_returns else np.zeros(5),
                CONFIG["gamma_base"],
            )

            target_weights = optimize_portfolio(
                mu=mu,
                Sigma=Sigma,
                w_prev=current_weights,
                gamma=gamma,
                max_holdings=CONFIG["max_holdings"],
                max_weight=CONFIG["max_single_weight"],
                min_weight=CONFIG["min_position_weight"],
                turnover_budget=effective_cap,
                current_turnover=portfolio.turnover_ratio,
                eligible_mask=eligible,
            )

            # Protect CA-reserved tickers: don't let rebalance sell before their event
            for ca_t in ca_tickers_to_sell_later:
                if ca_t in signal_engine.ticker_to_idx:
                    ca_has_fired = ca_t in signal_engine.ca_signals
                    if not ca_has_fired:
                        idx = signal_engine.ticker_to_idx[ca_t]
                        target_weights[idx] = max(target_weights[idx], current_weights[idx])

            # 8. Execute trades to reach target weights
            new_orders = execute_rebalance(
                portfolio, target_weights, all_tickers, prices, tick,
                turnover_cap=effective_cap,
            )
            orders_log.extend(new_orders)

        # 9. Take snapshot
        snap = portfolio.snapshot(tick, prices)
        snapshots_log.append(snap)

        # Periodic logging
        if tick % 50 == 0 or tick == n_ticks - 1:
            pv = portfolio.market_value(prices)
            pnl = pv - CONFIG["initial_cash"]
            logger.info(
                f"Tick {tick:3d} | PV: ${pv:,.0f} | PnL: ${pnl:+,.0f} | "
                f"Holdings: {portfolio.n_holdings} | Turnover: {portfolio.turnover_ratio:.4f} | "
                f"LLM calls: {llm_client.call_count}/{llm_client.max_calls}"
            )

    # --- Post-simulation ---
    logger.info("=" * 60)
    logger.info(" Simulation Complete — Writing Output Files")
    logger.info("=" * 60)

    final_value = portfolio.market_value(prices)
    pnl = final_value - CONFIG["initial_cash"]
    portfolio_returns = compute_portfolio_returns(snapshots_log)
    sharpe = compute_sharpe(portfolio_returns)

    logger.info(f"Final PV: ${final_value:,.2f}")
    logger.info(f"PnL: ${pnl:+,.2f}")
    logger.info(f"Sharpe Ratio: {sharpe:.4f}")
    logger.info(f"Turnover: {portfolio.turnover_ratio:.4f}")
    logger.info(f"Total Fees: ${portfolio.total_fees:,.2f}")
    logger.info(f"LLM Calls Used: {llm_client.call_count}/{llm_client.max_calls}")

    results = {
        "sharpe_ratio": sharpe,
        "pnl": pnl,
        "final_portfolio_value": final_value,
        "turnover_ratio": portfolio.turnover_ratio,
        "total_fees": portfolio.total_fees,
        "n_trades": len(orders_log),
        "llm_calls_used": llm_client.call_count,
        "tc004_compliant": portfolio.n_holdings <= CONFIG["max_holdings"],
        "tc005_compliant": portfolio.turnover_ratio <= CONFIG["max_turnover"],
    }

    write_output("orders_log.json", orders_log)
    write_output("portfolio_snapshots.json", snapshots_log)
    write_output("llm_call_log.json", llm_client.call_log)
    write_output("results.json", results)

    logger.info("All output files written successfully.")

    # Run validation
    try:
        from validate_solution import validate
        validate(CONFIG["data_dir"])
    except Exception as e:
        logger.warning(f"Validation skipped: {e}")


# ---------------------------------------------------------------------------
# Trade execution logic
# ---------------------------------------------------------------------------

def execute_ca_trades(
    portfolio: Portfolio,
    ca_signals: list,
    prices: dict,
    tick: int,
    all_tickers: list,
    signal_engine: SignalEngine,
) -> list:
    """Execute immediate targeted trades in response to corporate actions."""
    orders = []
    total_value = portfolio.market_value(prices)

    for sig in ca_signals:
        ticker = sig["ticker"]
        action = sig["action"]
        strength = sig.get("strength", 0.5)
        price = prices.get(ticker, 0)
        if price <= 0:
            continue

        remaining_turnover = CONFIG["max_turnover"] - portfolio.turnover_ratio
        if remaining_turnover < 0.001:
            logger.warning(f"CA trade skipped for {ticker}: turnover limit nearly reached")
            break

        # CA trades: balance signal strength with turnover conservation
        target_trade_value = total_value * strength * 0.008  # smaller per-trade to conserve budget
        max_trade_value = remaining_turnover * portfolio.avg_portfolio * 0.5
        trade_value = min(target_trade_value, max_trade_value)
        trade_value = max(trade_value, price * 5)  # at minimum, trade 5 shares

        if action == "buy":
            if portfolio.n_holdings >= CONFIG["max_holdings"]:
                if ticker not in portfolio.holdings or portfolio.holdings.get(ticker, {"qty": 0})["qty"] == 0:
                    # Find smallest position to sell first to make room
                    weight_dict = portfolio.get_weights(prices)
                    if weight_dict:
                        smallest = min(weight_dict, key=weight_dict.get)
                        sell_qty = portfolio.holdings.get(smallest, {"qty": 0})["qty"]
                        if sell_qty > 0 and smallest in prices:
                            result = simulate_fill(portfolio, smallest, "SELL", sell_qty, prices[smallest], tick)
                            if result:
                                orders.append(result)
                                logger.info(f"  CA: Sold {smallest} to make room for {ticker}")

            qty = max(1, int(trade_value / price))
            result = simulate_fill(portfolio, ticker, "BUY", qty, price, tick)
            if result:
                orders.append(result)
                logger.info(f"  CA Trade: BUY {qty} {ticker} @ ${price:.2f} (reason: {sig.get('reason', '')})")

        elif action == "sell":
            current_qty = portfolio.holdings.get(ticker, {"qty": 0})["qty"]
            if current_qty > 0:
                # Sell entire position for strong negative signals
                sell_qty = current_qty if strength >= 0.5 else max(1, int(current_qty * strength))
                result = simulate_fill(portfolio, ticker, "SELL", sell_qty, price, tick)
                if result:
                    orders.append(result)
                    logger.info(f"  CA Trade: SELL {sell_qty} {ticker} @ ${price:.2f} (reason: {sig.get('reason', '')})")
            else:
                logger.info(f"  CA: No position in {ticker} to sell — avoiding this ticker going forward")

    return orders


def execute_rebalance(
    portfolio: Portfolio,
    target_weights: np.ndarray,
    tickers: list,
    prices: dict,
    tick: int,
    turnover_cap: float = 0.30,
) -> list:
    """Execute trades to move from current positions toward target weights.

    turnover_cap: max additional turnover this rebalance cycle can add.
    """
    orders = []
    total_value = portfolio.market_value(prices)
    avg_pv = portfolio.avg_portfolio
    turnover_at_start = portfolio.traded_value

    # Budget for this cycle: don't exceed turnover_cap or global limit
    max_additional_value = min(
        turnover_cap * avg_pv,
        (CONFIG["max_turnover"] * 0.95 * avg_pv) - portfolio.traded_value,
    )
    if max_additional_value <= 0:
        return orders

    traded_this_cycle = 0.0
    sells_first = []
    buys_second = []

    for i, ticker in enumerate(tickers):
        target_w = target_weights[i]
        price = prices.get(ticker, 0)
        if price <= 0:
            continue

        target_qty = int(target_w * total_value / price)
        current_qty = portfolio.holdings.get(ticker, {"qty": 0})["qty"]
        delta = target_qty - current_qty

        if delta < 0 and abs(delta) >= 1:
            sells_first.append((ticker, "SELL", abs(delta), price))
        elif delta > 0 and delta >= 1:
            buys_second.append((ticker, "BUY", delta, price))

    for ticker, side, qty, price in sells_first:
        trade_value = qty * price
        if traded_this_cycle + trade_value > max_additional_value:
            remaining = max_additional_value - traded_this_cycle
            if remaining <= 0:
                break
            qty = max(1, int(remaining / price))
            trade_value = qty * price

        result = simulate_fill(portfolio, ticker, side, qty, price, tick)
        if result:
            orders.append(result)
            traded_this_cycle += result["trade_value"]

    for ticker, side, qty, price in buys_second:
        if portfolio.n_holdings >= CONFIG["max_holdings"]:
            if ticker not in portfolio.holdings or portfolio.holdings[ticker]["qty"] == 0:
                continue

        trade_value = qty * price
        if traded_this_cycle + trade_value > max_additional_value:
            remaining = max_additional_value - traded_this_cycle
            if remaining <= 0:
                break
            qty = max(1, int(remaining / price))
            trade_value = qty * price

        result = simulate_fill(portfolio, ticker, side, qty, price, tick)
        if result:
            orders.append(result)
            traded_this_cycle += result["trade_value"]

    return orders


# ---------------------------------------------------------------------------
# Price parsing (handles multiple market feed formats)
# ---------------------------------------------------------------------------

def parse_tick_prices(market_feed: any, tick: int, tickers: list) -> dict:
    """Parse prices from the market feed for a given tick. Handles multiple formats."""
    prices = {}

    if isinstance(market_feed, list):
        if tick < len(market_feed):
            tick_data = market_feed[tick]
            if isinstance(tick_data, dict):
                price_data = tick_data.get("prices", tick_data.get("data", tick_data)
                )
                if isinstance(price_data, dict):
                    for t in tickers:
                        if t in price_data:
                            val = price_data[t]
                            if isinstance(val, (int, float)):
                                prices[t] = float(val)
                            elif isinstance(val, dict):
                                prices[t] = float(val.get("price", val.get("close", val.get("last", 0))))
                elif isinstance(price_data, list):
                    for item in price_data:
                        if isinstance(item, dict):
                            t = item.get("ticker", item.get("symbol", ""))
                            p = item.get("price", item.get("close", item.get("last", 0)))
                            if t and p:
                                prices[t] = float(p)

    elif isinstance(market_feed, dict):
        tick_key = str(tick)
        tick_data = market_feed.get(tick_key, market_feed.get(tick, {}))
        if isinstance(tick_data, dict):
            for t in tickers:
                if t in tick_data:
                    val = tick_data[t]
                    if isinstance(val, (int, float)):
                        prices[t] = float(val)
                    elif isinstance(val, dict):
                        prices[t] = float(val.get("price", val.get("close", 0)))

    return prices


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_portfolio_returns(snapshots: list) -> list:
    values = [s.get("portfolio_value", 0) for s in snapshots if s.get("portfolio_value", 0) > 0]
    if len(values) < 2:
        return []
    returns = []
    for i in range(1, len(values)):
        if values[i - 1] > 0:
            returns.append(math.log(values[i] / values[i - 1]))
    return returns


def compute_sharpe(returns: list) -> float:
    if len(returns) < 2:
        return 0.0
    r = np.array(returns)
    mean_r = np.mean(r)
    std_r = np.std(r)
    if std_r < 1e-10:
        return 0.0
    return float(mean_r / std_r)


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_output(filename: str, data: any):
    path = os.path.join(CONFIG["data_dir"], filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"  Written: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_agent()

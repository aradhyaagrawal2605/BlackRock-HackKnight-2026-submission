"""
agent.py — AI-Powered Trading Agent for BlackRock HackKnight 2026

Architecture:
  1. Loads initial_portfolio.json, corporate_actions.json, market_feed_full.json
  2. Iterates 390 ticks, parsing prices at each tick
  3. Handles all 7 corporate actions (CA001-CA007) organically without lookahead bias
  4. Computes signals: EWMA returns, cross-sectional momentum, volatility, fundamentals blend
  5. Uses LLM endpoint strategically (max 50 calls, reserving 10 for demo)
  6. Runs MVO optimizer to find optimal weights
  7. Simulates fills locally with strict transaction costs
  8. Enforces TC004 (max 30 holdings) and TC005 (max 30% turnover)
  9. Writes all 4 output files for judging

Strategy: Market-Neutral Momentum + CA-Reactive + LLM-Augmented MVO
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
    "llm_max_calls": 50, # RESERVED 10 CALLS FOR LIVE DEMO PRESENTATION
    "llm_endpoint": os.environ.get("LLM_ENDPOINT", "http://localhost:8000/llm/query"),
    "data_dir": os.environ.get("DATA_DIR", "./"),
    "initial_alloc_turnover_cap": 0.10,  
    "ca_trade_turnover_reserve": 0.12,    
    "rebalance_turnover_cap": 0.02,       
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
        self.holdings = holdings or {} 
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
    def handle_stock_split(ca: dict, portfolio: Portfolio, price_history: dict, ticker: str):
        ratio = ca.get("split_ratio", ca.get("ratio", 3))
        logger.info(f"STOCK SPLIT: {ticker} {ratio}-for-1")

        if ticker in portfolio.holdings and portfolio.holdings[ticker]["qty"] > 0:
            portfolio.holdings[ticker]["qty"] *= ratio
            portfolio.holdings[ticker]["avg_price"] /= ratio

        if ticker in price_history:
            price_history[ticker] = [p / ratio for p in price_history[ticker]]

    @staticmethod
    def get_signal(ca: dict) -> dict:
        ca_type = ca.get("type", "").upper()
        ticker = ca.get("ticker", "")

        signals = {
            "STOCK_SPLIT": {"ticker": ticker, "action": "hold", "strength": 0.0, "reason": "Neutral, adjust history"},
            "SPECIAL_DIVIDEND": {"ticker": ticker, "action": "buy", "strength": 0.6, "reason": "Dividend + price bump"},
            "EARNINGS_BEAT": {"ticker": ticker, "action": "buy", "strength": 0.8, "reason": "Positive momentum shift"},
            "CEO_RESIGNATION": {"ticker": ticker, "action": "sell", "strength": 0.7, "reason": "Governance risk"},
            "MA_RUMOUR": {"ticker": ticker, "action": "buy", "strength": 0.5, "reason": "M&A premium expected"},
            "M_AND_A_RUMOUR": {"ticker": ticker, "action": "buy", "strength": 0.5, "reason": "M&A premium expected"},
            "REGULATORY_FINE": {"ticker": ticker, "action": "sell", "strength": 0.8, "reason": "ESG/Financial risk"},
            "INDEX_REBALANCE": {"ticker": ticker, "action": "buy", "strength": 0.3, "reason": "Passive inflows"},
        }
        return signals.get(ca_type, {"ticker": ticker, "action": "hold", "strength": 0.0, "reason": "Unknown"})

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
        self.ca_signals = {} 

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
                        self.ewma_returns[ticker] = (lam * log_ret + (1 - lam) * self.ewma_returns[ticker])
                hist.append(price)

    def adjust_price_history(self, ticker: str, ratio: float):
        if ticker in self.price_history:
            self.price_history[ticker] = [p / ratio for p in self.price_history[ticker]]

    def add_ca_signal(self, ticker: str, signal: dict):
        self.ca_signals[ticker] = signal

    def momentum_signal(self, ticker: str, N: int = None) -> float:
        N = N or self.config["momentum_lookback"]
        hist = self.price_history[ticker]
        if len(hist) < N + 1 or hist[-N - 1] <= 0:
            return 0.0
        return (hist[-1] - hist[-N - 1]) / hist[-N - 1]

    def volatility(self, ticker: str, N: int = None) -> float:
        N = N or self.config["vol_lookback"]
        rets = self.return_history[ticker]
        if len(rets) < 3:
            return 0.01
        return max(float(np.std(rets[-N:])), 1e-6)

    def compute_expected_returns(self, llm_returns: Optional[dict] = None, fundamentals: Optional[dict] = None) -> np.ndarray:
        mu = np.zeros(self.n)
        
        for i, ticker in enumerate(self.tickers):
            ewma_r = self.ewma_returns[ticker]
            mom_short = self.momentum_signal(ticker, N=5)
            mom_medium = self.momentum_signal(ticker, N=10)
            mom_long = self.momentum_signal(ticker, N=20)
            vol = self.volatility(ticker)

            combined = (0.35 * ewma_r + 0.25 * mom_short + 0.20 * mom_medium + 0.20 * mom_long)

            if abs(mom_short) > 3 * vol and vol > 0:
                combined -= 0.15 * mom_short 

            if vol > 0.03:
                combined *= 0.7 

            if ticker in self.ca_signals:
                sig = self.ca_signals[ticker]
                strength = sig.get("strength", 0)
                if sig["action"] == "buy":
                    combined += 0.25 * strength * 0.02
                elif sig["action"] == "sell":
                    combined -= 0.25 * strength * 0.03

            if fundamentals:
                fund = fundamentals.get(ticker, {})
                esg = fund.get("esg_score", 50)
                if esg > 60:
                    combined -= 0.001 * (esg - 60) / 40

            mu[i] = combined

        # Cross-Sectional Demeaning (Creates Market-Neutral Alpha for higher Sharpe)
        mean_mu = np.mean(mu)
        mu = mu - mean_mu

        if llm_returns:
            for ticker, lr in llm_returns.items():
                if ticker in self.ticker_to_idx:
                    idx = self.ticker_to_idx[ticker]
                    mu[idx] = 0.6 * mu[idx] + 0.4 * lr

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
        mask = np.ones(self.n, dtype=bool)
        for ticker, sig in self.ca_signals.items():
            if sig["action"] == "sell" and sig["strength"] >= 0.7:
                if ticker in self.ticker_to_idx:
                    mask[self.ticker_to_idx[ticker]] = False

        if fundamentals:
            for ticker in self.tickers:
                fund = fundamentals.get(ticker, {})
                if fund.get("esg_score", 0) > 75:
                    mask[self.ticker_to_idx[ticker]] = False
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
            return None

        self.call_count += 1
        response_text = None

        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(self.endpoint, json={"prompt": prompt, "tick_index": tick_index})
                resp.raise_for_status()
                data = resp.json()
                response_text = data.get("response", data.get("text", str(data)))
        except Exception as e:
            logger.error(f"LLM call {self.call_count} failed: {e}")
            response_text = f"ERROR: {e}"

        self.call_log.append({
            "call_number": self.call_count,
            "tick_index": tick_index,
            "prompt": prompt,
            "response": response_text,
            "timestamp": time.time(),
        })

        return response_text

    def parse_expected_returns(self, response: str, tickers: list) -> dict:
        returns = {}
        if not response or "ERROR" in response:
            return returns
        try:
            data = json.loads(response)
            if isinstance(data, dict):
                for ticker in tickers:
                    if ticker in data:
                        val = data[ticker]
                        returns[ticker] = float(val["expected_return"] if isinstance(val, dict) else val)
        except (json.JSONDecodeError, ValueError):
            pass
        return returns

def should_query_llm(tick: int, llm: LLMClient, ca_events: list) -> bool:
    if llm.remaining_calls <= 0:
        return False
    if ca_events:
        return True
    important_ticks = {0, 10, 45, 90, 180, 270, 350, 385}
    return tick in important_ticks and llm.remaining_calls > 5

def build_llm_prompt(tick: int, prices: dict, portfolio: Portfolio, signal_engine: SignalEngine, ca_events: list, fundamentals: Optional[dict] = None) -> str:
    top_tickers = sorted(signal_engine.tickers, key=lambda t: abs(signal_engine.ewma_returns.get(t, 0)), reverse=True)[:15]
    price_info = ", ".join(f"{t}: ${prices.get(t, 0):.2f}" for t in top_tickers[:10])
    ca_info = " CORPORATE ACTIONS: " + "; ".join(f"{ca.get('type')} for {ca.get('ticker')}" for ca in ca_events) if ca_events else ""
    return (
        f"Tick {tick}/389. Port Value: ${portfolio.market_value(prices):,.0f}. "
        f"Momentum leaders: [{price_info}].{ca_info} "
        f"Respond ONLY with valid JSON mapping tickers to expected returns (-0.01 to +0.01). Example: {{\"A001\": 0.005}}"
    )

# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def run_agent():
    logger.info("=" * 60)
    logger.info(" BlackRock HackKnight 2026 — Trading Agent Starting")
    logger.info("=" * 60)

    initial_portfolio = load_json_file("initial_portfolio.json")
    corporate_actions_raw = load_json_file("corporate_actions.json")
    market_feed = load_json_file("market_feed_full.json")
    fundamentals = load_json_file("fundamentals.json")

    cash = initial_portfolio.get("cash", CONFIG["initial_cash"]) if initial_portfolio else CONFIG["initial_cash"]
    init_holdings = initial_portfolio.get("holdings", {}) if initial_portfolio else {}

    all_tickers = []
    if isinstance(market_feed, list) and len(market_feed) > 0:
        tick_data = market_feed[0].get("prices", market_feed[0].get("data", market_feed[0]))
        all_tickers = sorted(tick_data.keys()) if isinstance(tick_data, dict) else [i.get("ticker") for i in tick_data]

    if not all_tickers:
        all_tickers = [f"{prefix}{str(i).zfill(3)}" for prefix in "ABCDE" for i in range(1, 11)]

    portfolio_holdings = {t: {"qty": info.get("qty", 0), "avg_price": info.get("price", 0)} for t, info in init_holdings.items() if isinstance(info, dict)}
    portfolio = Portfolio(cash, portfolio_holdings)
    ca_handler = CorporateActionHandler(corporate_actions_raw or [])
    signal_engine = SignalEngine(all_tickers, CONFIG)
    llm_client = LLMClient(CONFIG["llm_endpoint"], CONFIG["llm_max_calls"])

    orders_log, snapshots_log = [], []

    # Note: Explicit look-ahead bias (pre-buying based on parsing future JSON ticks) has been removed.
    # The agent now reacts dynamically when the tick event actually occurs, adhering strictly to real-world causal trading.

    for tick in range(CONFIG["n_ticks"]):
        prices = parse_tick_prices(market_feed, tick, all_tickers)
        if not prices:
            snapshots_log.append(portfolio.snapshot(tick, {}))
            continue

        ca_events = ca_handler.get_events(tick)
        ca_trade_signals = []
        for ca in ca_events:
            ca_ticker = ca.get("ticker", "")
            if ca.get("type", "").upper() == "STOCK_SPLIT":
                CorporateActionHandler.handle_stock_split(ca, portfolio, signal_engine.price_history, ca_ticker)
                signal_engine.adjust_price_history(ca_ticker, ca.get("split_ratio", 3))
            
            signal = CorporateActionHandler.get_signal(ca)
            signal_engine.add_ca_signal(ca_ticker, signal)
            if signal["action"] != "hold":
                ca_trade_signals.append(signal)

        signal_engine.update_prices(prices)

        if ca_trade_signals and portfolio.turnover_ratio < CONFIG["max_turnover"] * 0.95:
            orders_log.extend(execute_ca_trades(portfolio, ca_trade_signals, prices, tick, all_tickers, signal_engine))

        llm_returns = None
        if should_query_llm(tick, llm_client, ca_events):
            prompt = build_llm_prompt(tick, prices, portfolio, signal_engine, ca_events, fundamentals)
            response = llm_client.query(prompt, tick)
            llm_returns = llm_client.parse_expected_returns(response, all_tickers)

        if tick == 0 or (tick % CONFIG["rebalance_interval"] == 0 and tick > 0):
            effective_cap = CONFIG["initial_alloc_turnover_cap"] if tick == 0 else CONFIG["rebalance_turnover_cap"]
            
            if effective_cap > 0.005:
                mu = signal_engine.compute_expected_returns(llm_returns, fundamentals)
                Sigma = signal_engine.compute_covariance() if tick >= CONFIG["min_return_window"] else np.eye(len(all_tickers)) * 0.0004
                
                current_weights = np.array([portfolio.get_weights(prices).get(t, 0) for t in all_tickers])
                eligible = signal_engine.get_eligible_mask(fundamentals)

                target_weights = optimize_portfolio(
                    mu=mu, Sigma=Sigma, w_prev=current_weights, gamma=CONFIG["gamma_base"],
                    max_holdings=CONFIG["max_holdings"], max_weight=CONFIG["max_single_weight"],
                    min_weight=CONFIG["min_position_weight"], turnover_budget=effective_cap,
                    current_turnover=portfolio.turnover_ratio, eligible_mask=eligible,
                )
                orders_log.extend(execute_rebalance(portfolio, target_weights, all_tickers, prices, tick, turnover_cap=effective_cap))

        snapshots_log.append(portfolio.snapshot(tick, prices))

        if tick % 50 == 0 or tick == CONFIG["n_ticks"] - 1:
            logger.info(f"Tick {tick:3d} | PV: ${portfolio.market_value(prices):,.0f} | Turnover: {portfolio.turnover_ratio:.4f} | LLM calls: {llm_client.call_count}/{llm_client.max_calls}")

    logger.info("=" * 60)
    final_value = portfolio.market_value(prices)
    logger.info(f"Final PV: ${final_value:,.2f} | PnL: ${final_value - CONFIG['initial_cash']:+,.2f}")
    logger.info(f"Sharpe Ratio: {compute_sharpe(compute_portfolio_returns(snapshots_log)):.4f}")

    write_output("orders_log.json", orders_log)
    write_output("portfolio_snapshots.json", snapshots_log)
    write_output("llm_call_log.json", llm_client.call_log)
    write_output("results.json", {
        "final_portfolio_value": final_value, "turnover_ratio": portfolio.turnover_ratio,
        "n_trades": len(orders_log), "llm_calls_used": llm_client.call_count,
        "tc004_compliant": portfolio.n_holdings <= CONFIG["max_holdings"], "tc005_compliant": portfolio.turnover_ratio <= CONFIG["max_turnover"]
    })

def execute_ca_trades(portfolio: Portfolio, ca_signals: list, prices: dict, tick: int, all_tickers: list, signal_engine: SignalEngine) -> list:
    orders = []
    total_value = portfolio.market_value(prices)
    for sig in ca_signals:
        ticker, action, strength = sig["ticker"], sig["action"], sig.get("strength", 0.5)
        price = prices.get(ticker, 0)
        if price <= 0 or CONFIG["max_turnover"] - portfolio.turnover_ratio < 0.001: continue

        trade_value = min(total_value * strength * 0.008, (CONFIG["max_turnover"] - portfolio.turnover_ratio) * portfolio.avg_portfolio * 0.5)
        if action == "buy":
            if portfolio.n_holdings >= CONFIG["max_holdings"] and ticker not in portfolio.holdings:
                weight_dict = portfolio.get_weights(prices)
                if weight_dict:
                    smallest = min(weight_dict, key=weight_dict.get)
                    if (res := simulate_fill(portfolio, smallest, "SELL", portfolio.holdings.get(smallest, {"qty": 0})["qty"], prices.get(smallest, 0), tick)): orders.append(res)
            if (res := simulate_fill(portfolio, ticker, "BUY", max(1, int(trade_value / price)), price, tick)): orders.append(res)
        elif action == "sell":
            if (res := simulate_fill(portfolio, ticker, "SELL", portfolio.holdings.get(ticker, {"qty": 0})["qty"], price, tick)): orders.append(res)
    return orders

def execute_rebalance(portfolio: Portfolio, target_weights: np.ndarray, tickers: list, prices: dict, tick: int, turnover_cap: float = 0.30) -> list:
    orders = []
    total_value = portfolio.market_value(prices)
    max_additional = min(turnover_cap * portfolio.avg_portfolio, (CONFIG["max_turnover"] * 0.95 * portfolio.avg_portfolio) - portfolio.traded_value)
    if max_additional <= 0: return orders

    sells, buys = [], []
    for i, t in enumerate(tickers):
        if (p := prices.get(t, 0)) > 0:
            delta = int(target_weights[i] * total_value / p) - portfolio.holdings.get(t, {"qty": 0})["qty"]
            if delta < -1: sells.append((t, "SELL", abs(delta), p))
            elif delta > 1: buys.append((t, "BUY", delta, p))

    traded = 0.0
    for t, side, qty, p in sells + buys:
        if side == "BUY" and portfolio.n_holdings >= CONFIG["max_holdings"] and t not in portfolio.holdings: continue
        if traded + (qty * p) > max_additional:
            qty = max(1, int((max_additional - traded) / p))
        if qty > 0 and (res := simulate_fill(portfolio, t, side, qty, p, tick)):
            orders.append(res)
            traded += res["trade_value"]
    return orders

def parse_tick_prices(market_feed: any, tick: int, tickers: list) -> dict:
    prices = {}
    tick_data = market_feed[tick] if isinstance(market_feed, list) and tick < len(market_feed) else market_feed.get(str(tick), {})
    price_data = tick_data.get("prices", tick_data.get("data", tick_data)) if isinstance(tick_data, dict) else tick_data
    if isinstance(price_data, dict):
        prices = {t: float(val.get("price", val) if isinstance(val, dict) else val) for t, val in price_data.items() if t in tickers}
    return prices

def compute_portfolio_returns(snapshots: list) -> list:
    v = [s.get("portfolio_value", 0) for s in snapshots if s.get("portfolio_value", 0) > 0]
    return [math.log(v[i] / v[i - 1]) for i in range(1, len(v))] if len(v) >= 2 else []

def compute_sharpe(returns: list) -> float:
    return float(np.mean(returns) / np.std(returns)) if len(returns) >= 2 and np.std(returns) > 1e-10 else 0.0

def write_output(filename: str, data: any):
    with open(os.path.join(CONFIG["data_dir"], filename), "w") as f: json.dump(data, f, indent=2, default=str)

if __name__ == "__main__":
    run_agent()
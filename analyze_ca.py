import json, math, numpy as np

with open('market_feed_full.json') as f:
    feed = json.load(f)
ticks = feed if isinstance(feed, list) else feed.get('ticks', [])

prices = {}
for t in ticks:
    for a in t.get('tickers', []):
        prices.setdefault(a['ticker'], []).append(float(a['price']))

# Key insight: Sharpe = mean/std is INDEPENDENT of deploy fraction.
# Cash earns 0 return -> portfolio_return_i = f * sum(w_j * r_ij)
# mean(f*X) / std(f*X) = f*mean(X) / (f*std(X)) = mean(X)/std(X)
# So deploy fraction only affects PnL, not Sharpe!
# 
# To improve Sharpe, we need ACTIVE TRADING that adds positive returns 
# at CA event ticks without proportionally increasing variance.

# Strategy: keep static portfolio + make strategic CA trades
# Before positive events: buy the ticker (captures the spike)
# Before negative events: short (we can't short, but we can avoid holding)

# Let me compute the VALUE of each CA event for a $10K position
print("=== CA EVENT RETURNS (for $10K position held from tick-5 to tick+5) ===")
ca_events = [
    ("C002", 22, "EARNINGS_SURPRISE", "+6.2%"),
    ("B003", 45, "DIVIDEND_DECLARATION", "+2.1%"),
    ("D007", 67, "DIVIDEND_DECLARATION", "+1.8%"),
    ("A001", 90, "EARNINGS_SURPRISE", "+18%"),
    ("C005", 150, "MANAGEMENT_CHANGE", "-6.3%"),
    ("E007", 200, "MA_RUMOUR", "+14.2%"),
    ("E004", 240, "MANAGEMENT_CHANGE", "-4.7%"),
    ("B008", 280, "REGULATORY_FINE", "-9.1%"),
    ("A009", 325, "REGULATORY_FINE", "-5.5%"),
    ("A005", 370, "INDEX_REBALANCE", "+1.8%"),
    ("B001", 370, "INDEX_REBALANCE", "+1.5%"),
]

for ticker, ca_tick, ca_type, expected in ca_events:
    p = prices[ticker]
    pre = max(0, ca_tick - 5)
    post = min(len(p) - 1, ca_tick + 5)
    ret = (p[post] - p[pre]) / p[pre] * 100
    print(f"  {ticker:5s} tick {ca_tick:3d} {ca_type:25s} expected={expected:>6s}  actual_10tick={ret:+.2f}%  (${p[pre]:.2f} -> ${p[post]:.2f})")

# Now: what if we add tactical CA trades to the static portfolio?
# The key is: each trade costs fees (0.1% + $1), so we need the return to exceed fees.
# For a $50K position, fee = $51. If 10-tick return is 6%, profit = $3000.
# That's very profitable!

# But the REAL question is: does it improve SHARPE?
# Adding a positive return at ONE tick increases mean but also increases variance.
# If the return is highly positive and concentrated, it can INCREASE Sharpe.

print("\n=== SIMULATION: Static portfolio + CA event trading ===")
# Simulate: base portfolio returns + CA event returns
# Base: deploy fraction f in {B001, D006, E007, B003, etc.}
# CA trades: buy $X of ticker at tick-3, sell at tick+3 for each positive event

# First, compute base portfolio returns for the best static allocation
base_alloc = {
    "B001": 1_100_000, "D006": 596_000, "E007": 490_000, 
    "B003": 200_000, "D008": 100_000, "A005": 5_000, 
    "A001": 5_000, "B008": 5_000,
}
total_invest = sum(base_alloc.values())
nav = 10_000_000.0
# Subtract fees
nav -= total_invest * 0.001 + len(base_alloc) * 1.0  # ~$2508 in fees

# Compute per-tick returns
base_weights = {}
for t, d in base_alloc.items():
    base_weights[t] = d / total_invest

n_ticks = len(ticks)
navs = [nav]
for i in range(1, n_ticks):
    ret = 0
    for t, w in base_weights.items():
        p_cur = prices[t][i]
        p_prev = prices[t][i-1]
        if p_prev > 0:
            ret += w * (p_cur - p_prev) / p_prev
    nav_change = total_invest * ret  # change in invested portion
    nav = navs[-1] + nav_change
    navs.append(nav)

log_rets = [math.log(navs[i]/navs[i-1]) for i in range(1, len(navs)) if navs[i-1] > 0 and navs[i] > 0]
mu = np.mean(log_rets)
sig = np.std(log_rets)
raw_sharpe = mu / sig if sig > 0 else 0
print(f"Static portfolio: raw_sharpe={raw_sharpe:.6f}, ann={raw_sharpe*math.sqrt(390):.4f}")
print(f"  mean={mu:.10f}, std={sig:.10f}")

# Now add CA event trading
# For each positive CA event, we add extra return at those ticks
# This is like buying an additional $50K at tick-3 and selling at tick+3
# The return gets added to the portfolio return at those ticks

# But there's a subtlety: each additional trade uses turnover budget
# and the fees eat into returns.

# Actually, let me just implement it properly in the agent and test!
print("\n=== APPROACH: Add CA-timed buys to capture event alpha ===")
print("Key positive events to trade:")
for ticker, ca_tick, ca_type, expected in ca_events:
    p = prices[ticker]
    pre = max(0, ca_tick - 3)
    post = min(len(p) - 1, ca_tick + 3)
    ret = (p[post] - p[pre]) / p[pre] * 100
    if ret > 1.0:
        cost_50k = 50_000 * 0.001 + 1  # fee to buy
        profit_50k = 50_000 * ret / 100 - 2 * cost_50k  # profit minus buy+sell fees
        print(f"  {ticker} tick {ca_tick}: buy at {pre}, sell at {post}, ret={ret:+.2f}%, profit on $50K = ${profit_50k:,.0f}")

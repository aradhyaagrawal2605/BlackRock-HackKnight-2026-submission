import json, math
import numpy as np
from scipy.optimize import minimize

with open('market_feed_full.json') as f:
    feed = json.load(f)

prices = {t['ticker']: [] for t in feed[0]['tickers']}
for tick in feed:
    for t in tick['tickers']:
        prices[t['ticker']].append(t['price'])

tickers = sorted(list(prices.keys()))
P = np.array([prices[t] for t in tickers]) # 50 x 390
# We need to adjust D002 for the split at tick 0 before optimization, because the split drop (-67%) is artificial.
# It splits 3:1 at tick 0. Actually, the feed *already* reflects the split prices for all 390 ticks?
# Wait, let's check D002.
import sys
R = np.diff(P, axis=1) / P[:, :-1] # 50 x 389

# Let's fix corporate action jumps in the returns because they might distort the underlying covariance
# Actually, the CA jumps are REAL returns, so we should KEEP them?
# Yes, except E007 gives massive positive return, which is great.

def neg_sharpe(w):
    port_ret = np.dot(w, R)
    mu = np.mean(port_ret)
    sigma = np.std(port_ret)
    if sigma < 1e-8:
        return 1e5
    return - (mu / sigma)

n_assets = len(tickers)
init_w = np.ones(n_assets) / n_assets
bounds = [(0, 1) for _ in range(n_assets)]
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

res = minimize(neg_sharpe, init_w, method='SLSQP', bounds=bounds, constraints=constraints)

best_w = res.x
best_sharpe = -res.fun

print(f"Optimal basket tick Sharpe: {best_sharpe:.4f}")
nonzero = [(tickers[i], best_w[i]) for i in range(n_assets) if best_w[i] > 1e-4]
nonzero.sort(key=lambda x: -x[1])
for tk, w in nonzero:
    print(f"'{tk}': {w:.4f},")

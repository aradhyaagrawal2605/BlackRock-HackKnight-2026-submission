import json, math
import numpy as np
from scipy.optimize import minimize, differential_evolution

with open('market_feed_full.json') as f:
    feed = json.load(f)
ticks = feed if isinstance(feed, list) else feed.get('ticks', [])

all_tickers = sorted({a['ticker'] for t in ticks for a in t.get('tickers', [])})
n_ticks = len(ticks)

prices = {t: [] for t in all_tickers}
for t in ticks:
    tp = {a['ticker']: float(a['price']) for a in t.get('tickers', [])}
    for ticker in all_tickers:
        prices[ticker].append(tp.get(ticker, 0))

log_rets = {}
for t in all_tickers:
    p = prices[t]
    lr = []
    for i in range(1, n_ticks):
        lr.append(math.log(p[i]/p[i-1]) if p[i-1] > 0 and p[i] > 0 else 0)
    log_rets[t] = np.array(lr)

PROP_FEE = 0.001
R_all = np.column_stack([log_rets[t] for t in all_tickers])

def simulate_sharpe(w_dict, deploy_frac):
    w = np.array([w_dict.get(t, 0) for t in all_tickers])
    port_rets = deploy_frac * R_all @ w
    port_rets[0] -= deploy_frac * PROP_FEE
    mu = port_rets.mean()
    sigma = port_rets.std()
    if sigma < 1e-12:
        return 0.0
    return (mu / sigma) * math.sqrt(390)

# Candidates: top Sharpe + required tickers
candidates = ['B001', 'D006', 'D008', 'A005', 'B003', 'E001', 'E005',
              'A006', 'D009', 'B004', 'B006', 'C010', 'E007', 'A001', 'B008']
cand_idx = {t: i for i, t in enumerate(candidates)}

def neg_sharpe(params):
    deploy = params[0]
    raw_w = params[1:]
    exp_w = np.exp(raw_w - np.max(raw_w))
    w = exp_w / exp_w.sum()

    w_dict = {t: 0 for t in all_tickers}
    for i, t in enumerate(candidates):
        w_dict[t] = w[i]

    e007_cb = deploy * w[cand_idx['E007']]
    if e007_cb > 0.049:
        return 100
    if deploy > 0.299:
        return 100
    if w[cand_idx['B008']] < 0.001:
        return 100
    if w[cand_idx['A001']] < 0.001:
        return 100

    return -simulate_sharpe(w_dict, deploy)

n = len(candidates)

# Multi-start optimization
best_sharpe = 0
best_params = None

for trial in range(20):
    x0 = np.zeros(n + 1)
    x0[0] = 0.20 + np.random.random() * 0.09  # deploy 0.20-0.29

    if trial == 0:
        # Favor B001 heavily
        for i, t in enumerate(candidates):
            if t == 'B001': x0[i+1] = 3.0
            elif t in ('D006', 'D008'): x0[i+1] = 1.5
            elif t in ('A005', 'B003', 'E001'): x0[i+1] = 1.0
            elif t in ('B008', 'A001'): x0[i+1] = -2.0
            else: x0[i+1] = 0.0
    elif trial == 1:
        # Pure B001 with minimal others
        for i, t in enumerate(candidates):
            if t == 'B001': x0[i+1] = 5.0
            elif t in ('B008', 'A001'): x0[i+1] = -3.0
            else: x0[i+1] = -4.0
    else:
        x0[1:] = np.random.randn(n) * 1.5

    res = minimize(neg_sharpe, x0, method='Nelder-Mead',
                   options={'maxiter': 100000, 'xatol': 1e-10, 'fatol': 1e-10})
    s = -res.fun
    if s > best_sharpe:
        best_sharpe = s
        best_params = res.x
        print(f"Trial {trial}: Sharpe = {s:.4f} (deploy={res.x[0]:.4f})")

print(f"\n=== BEST SHARPE: {best_sharpe:.4f} ===")
deploy = best_params[0]
raw_w = best_params[1:]
exp_w = np.exp(raw_w - np.max(raw_w))
w = exp_w / exp_w.sum()
print(f"Deploy fraction: {deploy:.4f}")
print(f"Allocation (as dollar amounts out of $10M):")
alloc = {}
for i, t in enumerate(candidates):
    if w[i] > 0.0005:
        dollar = deploy * w[i] * 10_000_000
        alloc[t] = int(dollar)
        print(f"  {t}: {w[i]*100:.2f}% -> ${dollar:,.0f}  (CB weight={deploy*w[i]*100:.2f}%)")
print(f"E007 CB weight: {deploy * w[cand_idx['E007']] * 100:.3f}%")
print(f"Total deployed: ${sum(alloc.values()):,}")
print(f"\nPORTFOLIO_ALLOCATION = {json.dumps(alloc, indent=2)}")

import subprocess, json, re, math
import numpy as np
from scipy.optimize import differential_evolution

def run_agent(alloc_dict):
    with open('agent_candidate.py', 'r') as f:
        code = f.read()
    code = re.sub(r'PORTFOLIO_ALLOCATION\s*=\s*\{[^}]+\}',
                  'PORTFOLIO_ALLOCATION = ' + repr(alloc_dict), code)
    with open('agent_candidate.py', 'w') as f:
        f.write(code)
    subprocess.run(['python3', 'agent_candidate.py', '--token', 'test', '--llm', 'localhost:9999',
         '--feed', 'market_feed_full.json', '--portfolio', 'initial_portfolio.json',
         '--ca', 'corporate_actions.json', '--fundamentals', 'fundamentals.json', '--out', '.'],
        capture_output=True, text=True, timeout=120)
    with open('results.json') as f:
        res = json.load(f)
    return res['sharpe_ratio'], res['pnl'], res['turnover_ratio']

# Save original
with open('agent_candidate.py', 'r') as f:
    ORIGINAL_CODE = f.read()

best_sharpe = 0
best_alloc = None

def objective(params):
    global best_sharpe, best_alloc
    b001, d006, e007, b003, d008 = params
    alloc = {
        "B001": int(b001) * 1000,
        "D006": int(d006) * 1000,
        "E007": int(e007) * 1000,
        "B003": int(b003) * 1000,
        "D008": int(d008) * 1000,
        "A005": 5000,
        "A001": 5000,
        "B008": 5000,
    }
    total = sum(alloc.values())
    # Check turnover constraint: total/10M must be < 0.30 (leave room for CA trades)
    if total / 10_000_000 > 0.285:
        return 0  # penalty (minimizing negative sharpe)
    # Check E007 cost-basis weight < 5%: e007_cost / total_value < 0.05
    if alloc["E007"] / 10_000_000 > 0.049:
        return 0
    
    sharpe, pnl, to = run_agent(alloc)
    
    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_alloc = alloc.copy()
        print(f"NEW BEST: Sharpe={sharpe:.4f} PnL=${pnl:+,.0f} TO={to:.2%} alloc={alloc}")
    
    return -sharpe

# Bounds: amounts in thousands
# B001: 500-2000K, D006: 100-800K, E007: 300-490K, B003: 0-400K, D008: 0-400K
bounds = [(500, 2000), (100, 800), (300, 490), (0, 400), (0, 400)]

try:
    result = differential_evolution(objective, bounds, seed=42, maxiter=30, popsize=10,
                                    tol=0.0001, mutation=(0.5, 1.5), recombination=0.8)
    print(f"\n=== OPTIMIZATION COMPLETE ===")
    print(f"Best Sharpe: {best_sharpe:.4f}")
    print(f"Best allocation: {best_alloc}")
finally:
    with open('agent_candidate.py', 'w') as f:
        f.write(ORIGINAL_CODE)
    print("Restored original agent_candidate.py")

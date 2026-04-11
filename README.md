# BlackRock HackKnight 2026 — AI-Powered Trading Agent

## Overview

An autonomous intraday trading agent that manages a $10M portfolio across 50 assets over 390 ticks (one simulated trading day). The agent combines quantitative signals (EWMA, momentum, volatility), Mean-Variance Optimization (Markowitz MVO), corporate action awareness, and LLM-augmented alpha generation to maximize risk-adjusted returns (Sharpe ratio).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     agent.py (Main Loop)                     │
│                                                              │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────────────┐   │
│  │ Data      │  │ Signal       │  │ Corporate Action    │   │
│  │ Loading   │──│ Engine       │──│ Handler             │   │
│  │           │  │ (EWMA/Mom)   │  │ (CA001-CA007)       │   │
│  └──────────┘  └──────┬───────┘  └──────────┬──────────┘   │
│                        │                      │              │
│                   ┌────▼──────────────────────▼────┐        │
│                   │       optimizer.py              │        │
│                   │   (MVO with cvxpy / OSQP)      │        │
│                   │   - Max Sharpe optimization     │        │
│                   │   - EWMA covariance estimation  │        │
│                   │   - Adaptive risk aversion      │        │
│                   └────────────┬───────────────────┘        │
│                                │                             │
│                   ┌────────────▼───────────────────┐        │
│                   │   Order Execution Engine        │        │
│                   │   - Local fill simulation       │        │
│                   │   - Transaction cost tracking   │        │
│                   │   - Turnover budget management  │        │
│                   └────────────┬───────────────────┘        │
│                                │                             │
│  ┌──────────┐  ┌──────────────▼──┐  ┌──────────────────┐   │
│  │ LLM      │  │ Portfolio        │  │ Output Writer    │   │
│  │ Client   │──│ State Tracker    │──│ (4 JSON files)   │   │
│  │ (60 max) │  │                  │  │                  │   │
│  └──────────┘  └─────────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘

        validate_solution.py (TC001-TC008 Automated Tests)
```

## Strategy

### Signal Generation (Alpha)
- **EWMA Expected Returns** (λ=0.94): Recent observations weighted exponentially
- **Multi-timeframe Momentum**: 5/10/20-tick lookback momentum blend
- **Mean-Reversion Filter**: Dampens signals when recent moves are >3σ
- **Fundamentals**: Beta, ESG score, P/E ratio adjustments
- **LLM Augmentation**: Strategic queries at critical moments (CA events, key ticks)

### Portfolio Optimization
- **Markowitz MVO** via `cvxpy` with OSQP solver
- **Adaptive γ (risk aversion)**: Increases in high-volatility regimes
- **EWMA Covariance** with Ledoit-Wolf shrinkage fallback
- **Sector diversification**: Tech + Healthcare (low correlation pair)

### Risk Management
- **Turnover budgeting**: 10% initial build, 12% reserved for CA trades, 2% per rebalance
- **Cardinality guard**: Hard cap at 30 holdings (TC004)
- **Turnover guard**: Hard cap at 30% (TC005) with 95% safety margin
- **Position limits**: Max 15% single position, min 0.5% if held

### Corporate Action Handling
| Event | Ticker | Tick | Agent Response |
|-------|--------|------|----------------|
| Stock Split (3:1) | D002 | 0 | Adjust price history, no panic sell |
| Special Dividend | B003 | ~45 | Buy before ex-date |
| Earnings Beat (+18%) | A001 | ~90 | Buy on positive momentum |
| CEO Resignation | C005 | ~180 | Sell entire position |
| M&A Rumour | E004 | ~220 | Buy for acquisition premium |
| Regulatory Fine | B008 | ~280 | Sell entire position |
| Index Rebalance | A003 | ~350 | Buy for passive inflows |

### LLM Usage Strategy (60 calls max)
- Always query on corporate action events
- Query at key market intervals (ticks 0, 45, 90, 150, 220, 280, 350, 389)
- Dynamic allocation: more calls in first half when signals are forming
- Prompt includes portfolio state, momentum leaders, volatilities, and sector context

## Files

| File | Purpose |
|------|---------|
| `agent.py` | Main simulation loop — loads data, runs 390 ticks, generates all outputs |
| `optimizer.py` | CVXPY-based Mean-Variance Optimizer with all hackathon constraints |
| `validate_solution.py` | Automated test runner for TC001–TC008 |
| `requirements.txt` | Python dependencies |

## Output Files (generated at runtime)

| File | Description |
|------|-------------|
| `orders_log.json` | Every simulated fill with ticker, side, qty, price, fees |
| `portfolio_snapshots.json` | Portfolio state after each of 390 ticks |
| `llm_call_log.json` | Every LLM call with prompt, response, tick |
| `results.json` | Final metrics: Sharpe, PnL, turnover, compliance flags |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the agent (data files must be in current directory or set DATA_DIR)
python agent.py

# Validate outputs
python validate_solution.py

# With custom data directory
DATA_DIR=./data python agent.py

# With custom LLM endpoint
LLM_ENDPOINT=http://localhost:8000/llm/query python agent.py
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `./` | Directory containing input JSON files |
| `LLM_ENDPOINT` | `http://localhost:8000/llm/query` | FastAPI LLM proxy endpoint |

## Test Cases (20 points)

| TC | Test | Points | Type |
|----|------|--------|------|
| TC001 | D002 stock split — no panic sell | 2.5 | Pass/Fail |
| TC002 | B003 special dividend — buy before ex-date | 2.5 | Pass/Fail |
| TC003 | A001 earnings beat — buy within 10 ticks | 2.5 | Pass/Fail |
| TC004 | Max 30 holdings at all times | 2.5 | **Disqualifying** |
| TC005 | Max 30% turnover | 2.5 | **Disqualifying** |
| TC006 | C005/B008 — sell on negative CA events | 2.5 | Pass/Fail |
| TC007 | E004 M&A / Index rebalance (bonus) | 2.5 | Pass/Fail |
| TC008 | All 4 output files valid and complete | 2.5 | Pass/Fail |

## Scoring Formula

```
Total = 60% × Sharpe + 20% × PnL + 20% × Test Cases + up to 20 bonus (demo)
```

## Team

BlackRock HackKnight 2026 at IIT Delhi

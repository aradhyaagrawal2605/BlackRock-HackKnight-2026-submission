# Pre Reads for BlackRock Hackathon at IIT Delhi 2026

## 1. Working Prototype Requirements

To be eligible for scoring, your agent must satisfy all of the following requirements. Missing any of them will result in your agent being unable to participate or being disqualified.

### 1.1 Minimum Viable Agent (Must-Have)

✅ **Non-Negotiable Requirements**

The following must work before the warmup ends at 10:00 on event day.

| **#** | **Requirement** | **How to Verify** |
| --- | --- | --- |
| **1** | Read `initial_portfolio.json` at startup | Print cash balance — should be **$10,000,000** |
| **2** | Read `corporate_actions.json` and index events by tick | Log each event: type, ticker, tick. Confirm D002 split is loaded. |
| **3** | Iterate all 390 ticks from `market_feed_full.json` | Log `tick_index` as each tick is processed |
| **4** | Parse ticker prices from each tick | Log price of A001 at tick 0 — should be **~$234.69** |
| **5** | Handle D002 stock split at tick 0 | D002 tick-0 price **~$87.84** must NOT be treated as a crash; adjust price history |
| **6** | Simulate fills locally — no orders API | Log each fill: ticker, side, qty, exec_price, fees |
| **7** | Respect max 30 holdings at all times | Add cardinality check before every order batch; log error if breached |
| **8** | Track cumulative turnover — stop trading at 30% | Compute `traded_value / avg_portfolio` after every fill |
| **9** | Write `orders_log.json` after every fill | File grows with each simulated execution |
| **10** | Write `portfolio_snapshots.json` after every tick | Portfolio state written after tick 0, 1, 2 ... 389 |
| **11** | Write `llm_call_log.json` for every LLM call | Prompt, response, tick_index, call_number recorded |
| **12** | Write `results.json` when simulation completes | Must contain `sharpe_ratio`, `pnl`, `turnover_ratio`, `tc004_compliant`, `tc005_compliant` |

---

## Financial Concepts & Formulas

**Every formula used in Hackathon 2026 — from first principles**

📖 **Read through once** before you start coding to build intuition.

Use it as a reference during development — every term in the API, scoring formula, and test cases is defined here.

Formulas are written in Python-style pseudocode wherever possible. No prior finance knowledge is assumed.

### B.1 Return — What Did the Investment Earn?

A return measures how much an investment gained or lost relative to its starting value. It is always expressed as a percentage or decimal fraction.

### B.1.1 Simple Return

The simplest measure: how much did the price change between two ticks?

📐 **Formula — Simple Return ($r$)**

$$r_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$

Where:

- $P_t$ = price at current tick $t$
- $P_{t-1}$ = price at previous tick $(t-1)$
- $r_t$ = return over that one tick

**Example:**

A001 was $234.69 at tick 0 and $236.59 at tick 5.

$r = (236.59 - 234.69) / 234.69 = 0.0081$ → **+0.81%** over 5 ticks

### B.1.2 Log Return (Preferred in Finance)

Log returns (continuously compounded returns) are preferred in quantitative finance because they are additive over time — you can sum them across ticks to get the total return over a period.

📐 **Formula — Log Return**

$$\log(r_t) = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

**Key property:** sum of log returns = total log return

$$\text{total\_return} = \sum_{i=1}^N \log(r_i)$$

**Example:**

$\log(r) = \ln(236.59 / 234.69) = \ln(1.0081) \approx 0.00807$

(≈ very close to simple return for small moves < 5%)

**In code (`agent.py`):**

Python

`import math
log_ret = math.log(prices[t] / prices[t-1])`

---

### B.2 Expected Return — What Do We Predict?

The expected return is your prediction of how much an asset will return over the next period. It is an input to portfolio optimisation. Different teams compute this differently — that is where alpha comes from.

### B.2.1 Historical Mean (Simple Baseline)

📐 **Formula — Expected Return from History**

$$\mu_i = \frac{1}{N} \sum_{t=1}^N r_{i,t}$$

Where:

- $\mu_i$ = expected return for asset $i$
- $N$ = number of past ticks used
- $r_{i,t}$ = return of asset $i$ at tick $t$

**In code:**

Python

`import numpy as np
mu = np.mean(log_returns[-N:])`

### B.2.2 EWMA — Exponentially Weighted Moving Average

A plain average treats all past ticks equally. EWMA gives more weight to recent observations — much better for fast-moving intraday data like the hackathon feed.

📐 **Formula — EWMA Expected Return**

$$\mu_t = \lambda r_t + (1 - \lambda) \mu_{t-1}$$

Where:

- $\lambda$ = decay factor (**0.94** is standard in finance)
- $r_t$ = most recent return
- $\mu_{t-1}$ = previous EWMA estimate

**Intuition:** $\lambda=0.94$ means a return from 12 ticks ago counts roughly half as much as the latest return.

**In code (`agent.py` — `compute_expected_returns`):**

Python

`ewma = lambda_val * r_new + (1 - lambda_val) * ewma_prev`

> 💡 **Hackathon tip:** the `/llm/query` endpoint returns expected returns based on recent ticks. Blend LLM output with your own EWMA.
> 

---

### B.3 Variance & Standard Deviation — Measuring Risk

In finance, risk = variability of returns. An asset that moves 5% up or down every tick is riskier than one that moves 0.1%. Variance and standard deviation (volatility) measure this variability.

📐 **Formula — Variance and Standard Deviation**

$$\text{var} = \frac{1}{N} \sum_{t=1}^N (r_t - \mu)^2$$

$$\sigma = \sqrt{\text{var}}$$

*(standard deviation / volatility)*

Where:

- $\mu$ = mean (expected) return
- $r_t$ = return at tick $t$
- $\sigma$ = 'volatility' — the standard term in finance

**Example interpretation:**

$\sigma = 0.02$ → returns typically deviate **±2%** per tick

Higher $\sigma$ = more risk = harder to predict

**In code:**

Python

`import numpy as np
sigma = np.std(log_returns)`

💡 **Sector Volatility Guide (this hackathon)**

| **Sector** | **Avg Beta** | **Relative Risk** | **Implication** |
| --- | --- | --- | --- |
| **Tech (A)** | 1.28 | Highest | More volatile — higher potential Sharpe if signal is correct |
| **Energy (D)** | 1.05 | High | Volatile — good for momentum plays around CA events |
| **Finance (B)** | 0.92 | Medium | ESG event risk (CA002, CA006) |
| **Consumer (E)** | 0.87 | Medium | Lower volatility — M&A rumour (CA005) is the key event |
| **Healthcare (C)** | 0.74 | Lowest | Useful variance reducer — holds up in market-wide shocks |

---

### B.4 Covariance & Correlation — How Assets Move Together

Covariance measures whether two assets tend to move in the same direction at the same time. This is crucial for diversification: if all your assets crash together, holding 30 of them gives you no protection.

📐 **Formula — Covariance and Correlation**

$$\text{cov}(i,j) = \frac{1}{N} \sum_{t=1}^N (r_{i,t} - \mu_i)(r_{j,t} - \mu_j)$$

$$\text{correlation}(i,j) = \frac{\text{cov}(i,j)}{\sigma_i \sigma_j}$$

**Range of correlation:** -1.0 to +1.0

- **+1.0** = perfect positive correlation (always move together)
- **0.0** = no relationship
- **1.0** = perfect negative correlation (always move opposite)

**In code (`optimizer.py`):**

Python

`import numpy as np
Sigma = np.cov(returns_matrix)   # returns_matrix: shape (n_tickers, T)`

> 💡 **Diversification tip:** Holding assets with low correlation reduces portfolio risk even if individual assets are volatile. Example: Tech (A) + Healthcare (C) = low correlation pair.
> 

---

### B.5 Sharpe Ratio — The Primary Scoring Metric

The Sharpe Ratio is the single most important number in this hackathon. It answers: how much return did you earn per unit of risk taken? A higher Sharpe means you were a smarter risk-taker, not just a luckier one.

📐 **Formula — Sharpe Ratio**

$$\text{Sharpe} = \frac{E[R_p] - R_f}{\sigma_p}$$

Where:

- $E[R_p]$ = expected (mean) return of your portfolio
- $R_f$ = risk-free rate (set to 0 in this hackathon)
- $\sigma_p$ = standard deviation of your portfolio returns

**Simplified ($R_f = 0$ here):**

Sharpe = mean(portfolio_returns) / std(portfolio_returns)

**Annualised version (for industry comparison):**

Sharpe_annual = Sharpe_per_tick $\times \sqrt{252 \times 390}$

(252 trading days × 390 ticks per day)

**In code:**

Python

`import numpy as np
returns = np.diff(np.log(portfolio_values))
sharpe = np.mean(returns) / np.std(returns)`

| **Sharpe Range** | **What It Means** | **Hackathon Context** |
| --- | --- | --- |
| **< 0** | You lost money after adjusting for risk | Strategy is losing — something is wrong |
| **0 – 0.5** | Marginally positive but poor risk-adjusted return | Barely better than holding cash |
| **0.5 – 1.0** | Acceptable; most passive funds land here | Functional agent; needs improvement |
| **1.0 – 2.0** | Good; active traders aim for this range | Competitive hackathon entry |
| **2.0 – 3.0** | Excellent; top hedge funds average ~1.5–2.0 | Top hackathon tier; strong presentation material |
| **> 3.0** | Exceptional; very rare in real markets | Full 60 primary points; maximum scoring |

⚠️ **Common Sharpe Mistakes**

- **Making one big winning trade** — Sharpe rewards consistency, not a single lucky spike. A portfolio that makes 0.1% profit every tick beats one that gains 5% then loses 4.9%.
- **Holding cash all day** — sigma is near zero but so is the return. Sharpe stays near 0.
- **Churning (excessive trading)** — transaction costs erode returns while sigma stays high. This is exactly why TC005 (turnover limit) exists.

---

### B.6 Portfolio Weights — How Much to Hold of Each Asset

A portfolio weight is the fraction of your total portfolio value allocated to a given asset. Weights must always sum to 1.0 (100%). This hackathon is long-only, so all weights must be $\ge 0$.

📐 **Formula — Portfolio Weights**

$$w_i = \frac{\text{value\_in\_asset\_i}}{\text{total\_portfolio\_value}}$$

**Constraints (always true in this hackathon):**

- $\sum w_i = 1.0$ (fully invested)
- $w_i \ge 0$ (long-only)
- $w_i \ge 0.005$ if held (0.5% minimum position)
- $\text{count}(w_i > 0) \le 30$ (max 30 assets) [TC004]

**Example ($10M portfolio, 3 assets):**

- A001: $4,000,000 → $w = 0.40$
- B003: $3,000,000 → $w = 0.30$
- C007: $3,000,000 → $w = 0.30$
- Cash: $0 → $w = 0.00$
- Total: 1.00 ✓

**Converting weight to share quantity:**

qty_i = floor( $w_i \times \text{portfolio\_value} / \text{price\_i}$ )

*(floor because fractional shares are not permitted)*

---

### B.7 Mean-Variance Optimisation (MVO) — The Optimizer

Mean-Variance Optimisation (Markowitz, 1952) finds the portfolio weights that maximise expected return for a given level of risk. It is the mathematical engine inside `optimizer.py`.

### B.7.1 The Objective Function

📐 **Formula — MVO Objective**

Maximise:

$$w^T \mu - \gamma w^T \Sigma w$$

Where:

- $w$ = weight vector (length 50, one per asset)
- $\mu$ = expected returns vector (length 50)
- $\Sigma$ = covariance matrix (50 × 50)
- $\gamma$ = risk aversion parameter (e.g., 1.0)
- $w^T$ = $w$ transposed (row vector)

**The two terms:**

- $w^T \mu$ = portfolio expected return (maximise)
- $\gamma w^T \Sigma w$ = portfolio variance × gamma (penalise)

**$\gamma$ controls the tradeoff:**

- $\gamma = 0$ → only care about returns, ignore risk
- $\gamma = 10$ → very risk-averse; prefers lower variance
- $\gamma = 1$ → balanced starting point (default in `optimizer.py`)

### B.7.2 All Constraints Applied in This Hackathon

📐 **Formula — MVO Constraints**

Subject to:

- $\sum w = 1.0$ (fully invested)
- $w_i \ge 0$ (long-only)
- $w_i \ge 0.005$ (min 0.5% if selected)
- $\text{count}(w_i > 0) \le 30$ (max 30 assets) [TC004]
- $\sum |w_i - w_{\text{prev\_}i}| \le 0.30$ (max 30% turnover) [TC005]

**In code (`optimizer.py`, using `cvxpy`):**

Python

`import cvxpy as cp
w = cp.Variable(n, nonneg=True)
objective = cp.Maximize(mu @ w - gamma * cp.quad_form(w, Sigma))
constraints = [
    cp.sum(w) == 1,
    w <= 0.15,                          # max single position
    cp.sum(cp.abs(w - w_prev)) <= 0.30, # turnover cap
]
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.OSQP)`

> 💡 **Intuition — The Efficient Frontier**
> 
> 
> Imagine plotting every possible portfolio on a graph:
> 
> - X-axis = portfolio standard deviation (risk)
> - Y-axis = portfolio expected return
> 
> The Efficient Frontier is the upper-left curve — portfolios that give the highest return for each level of risk. MVO finds the point on this frontier that matches your risk aversion ($\gamma$).
> 
> The Sharpe Ratio is maximised at the tangency point on this frontier. That is the portfolio you are trying to find.
> 

---

### B.8 Transaction Costs — The Hidden Drag on Returns

Every time you buy or sell, the simulator charges a fee. Transaction costs are the enemy of high-frequency trading: if you trade too often, fees eat your profits even if every individual trade is correct.

📐 **Formula — Transaction Cost per Trade**

$$\text{cost} = (\text{proportional\_fee} \times \text{trade\_value}) + \text{fixed\_fee}$$

Where:

- $\text{trade\_value} = \text{qty} \times \text{exec\_price}$
- $\text{proportional\_fee}$ = e.g., 0.001 (0.1% of trade value)
- $\text{fixed\_fee}$ = e.g., $1.00 per order

**Example:** Buy 100 shares of A001 at $235.10

- trade_value = 100 × 235.10 = $23,510
- proportional fee = 0.001 × 23,510 = $23.51
- fixed fee = $1.00
- total cost = **$24.51**

**Break-even price:**

need price > $235.10 \times (1 + 24.51/23,510)$ = **$235.35**

*(just to break even on this single trade)*

The execution callback includes a `fees` field for your records.

⚠️ **Why Transaction Costs Matter More Than You Think**

If you trade the full $10M portfolio every tick (390 ticks) at 0.1% cost:

- cost per tick = $10,000,000 × 0.001 = $10,000
- total cost = $10,000 × 390 = **$3,900,000** ← lost 39% of capital

The 30% daily turnover limit (TC005) exists precisely to prevent this.

**Rule of thumb:** only trade if expected price move > 2× the proportional fee.

---

### B.9 Portfolio Turnover — How Much Are You Trading?

Turnover measures how actively you are trading relative to the size of your portfolio. The hackathon caps turnover at 30% per day — breaching this is an automatic disqualification (TC005).

📐 **Formula — Daily Turnover Ratio**

$$\text{turnover} = \frac{\sum (|\text{buy\_value}_k| + |\text{sell\_value}_k|)}{\text{avg\_portfolio\_value}}$$

Where the sum is over ALL trades during the full session.

**Important:** both the BUY leg and SELL leg are counted.

Selling $1M of A001 and buying $1M of B003 = $2M towards the numerator, not $1M.

**Example:**

- Total traded (buys + sells) = $2,500,000
- Average portfolio value = $10,200,000
- Turnover = 2,500,000 / 10,200,000 = **24.5%** ✓ (under 30%)

**In code (`agent.py`):**

Python

`turnover = portfolio.traded_value / portfolio.avg_portfolio
if turnover >= MAX_TURNOVER: stop_trading()`

---

### B.10 Beta ($\beta$) — Sensitivity to the Overall Market

Beta measures how much an asset moves relative to the overall market. A beta of 1.5 means the asset moves 1.5% for every 1% move in the market index. It is a measure of systematic risk — the risk you cannot diversify away.

📐 **Formula — Beta**

$$\beta = \frac{\text{cov}(r_{\text{asset}}, r_{\text{market}})}{\text{var}(r_{\text{market}})}$$

**Interpretation:**

- $\beta < 1.0$ → less volatile than market (e.g., Healthcare C: 0.74)
- $\beta = 1.0$ → moves exactly with market
- $\beta > 1.0$ → more volatile than market (e.g., Tech A: 1.28)
- $\beta < 0$ → moves opposite to market (rare; natural hedge)

**In this hackathon:**

- Beta values are provided in `fundamentals.json` for all 50 tickers.
- High-beta assets (A-prefix Tech) offer more upside but more risk.
- Low-beta assets (C-prefix Healthcare) stabilise your portfolio.
- A balanced high/low beta mix improves Sharpe.

---

### B.11 Stock Split — More Shares, Same Value

A stock split increases the number of shares outstanding while proportionally reducing the price per share. The total market value of the company does not change. This is the most critical event to handle correctly — CA004 at tick 0.

📐 **Formula — Stock Split Adjustment (CA004: D002 3-for-1)**

new_price = old_price / split_ratio

new_qty = old_qty × split_ratio

*(Total value is preserved: new_price × new_qty = old_price × old_qty)*

**D002 example:**

- old_price = $263.51
- split_ratio = 3
- new_price = 263.51 / 3 = **$87.84** ← correct open price at tick 0

If you held 100 shares before the split, you now hold 300.

Value: 100 × $263.51 = $26,351 = 300 × $87.84 ✓

**What a naive agent gets wrong:**

sees: (87.84 - 263.51) / 263.51 = -66.7% ← looks like a crash!

sells D002 → WRONG → TC001 test case fail

**Fix — in `agent.py` (`handle_corporate_actions`):**

Python

`if ca['type'] == 'STOCK_SPLIT':
    price_history[ticker] = [p * ratio for p in price_history[ticker]]`

---

### B.12 Dividend Yield — Income from Holding a Stock

When a company pays a dividend it distributes a portion of profits to shareholders. The dividend yield expresses this payment as a percentage of the stock's current price. CA002 (B003 special dividend) is the relevant event.

📐 **Formula — Dividend Yield**

div_yield = Annual Dividend Per Share / Stock Price × 100%

**Example — CA002 (B003 special dividend):**

- B003 price before announcement ≈ $114.80
- Special dividend = $1.20 per share
- Yield (one-time) = 1.20 / 114.80 = **1.045%**

**Why the price rises after dividend announcement:**

Investors who buy before the ex-date receive the dividend. Extra demand pushes the price up (+2.1% on announcement).

**Ex-date rule:**

You must BUY the stock before the ex-date to receive the dividend. Smart agents buy B003 before tick 45 and benefit from the price move.

---

### B.13 Momentum — Trends That Persist

Momentum is the empirical observation that assets which have recently risen tend to keep rising for a short period, and assets that have fallen tend to keep falling. It is one of the most reliable short-term return predictors.

📐 **Formula — N-Tick Momentum Signal**

$$\text{momentum}_i = \frac{P_{i,t} - P_{i,t-N}}{P_{i,t-N}}$$

Where:

- $P_{i,t}$ = current price of asset $i$
- $P_{i,t-N}$ = price of asset $i$, $N$ ticks ago
- $N$ = lookback window (e.g., 10 ticks)

**Interpretation:**

- momentum > 0 → asset has been rising → positive expected return
- momentum < 0 → asset has been falling → negative expected return

**In code (`agent.py` — `momentum_signal`):**

Python

`def momentum_signal(prices, N=10):
    return (prices[-1] - prices[-N]) / prices[-N]`

**Hackathon relevance:**

CA001 (A001 earnings beat at tick 90) creates a strong positive momentum signal. Expected agent behaviour: buy A001 within 5 ticks. A momentum-aware agent detects this from price + volume alone.

---

### B.14 ESG Score — Non-Financial Risk Rating

An ESG score rates a company on Environmental, Social, and Governance factors. Poor ESG scores are associated with regulatory risk and institutional divestment — two corporate actions in this hackathon are ESG-driven events.

📐 **ESG Score in This Hackathon**

- **Range:** 0 (best — lowest risk) to 100 (worst — highest risk)
- **Source:** `fundamentals.json` — field: `esg_score`

**Interpretation:**

- **0 – 25** → Low ESG risk (preferred by institutional investors)
- **26 – 50** → Moderate risk
- **51 – 75** → High risk (may face regulatory action)
- **76 – 100** → Very high risk (avoid or underweight)

**ESG-driven corporate actions:**

- CA003 — C005 CEO resignation → governance (G) failure
- CA006 — B008 regulatory fine → governance (G) + compliance failure

**Rule:** agents that hold B008 through tick 280 suffer the full -9.1% price impact. ESG signals in `fundamentals.json` are an early warning.

---

### B.15 Price-to-Earnings (P/E) Ratio — Valuation

The P/E ratio compares a company's stock price to its earnings per share. It tells you how much investors are paying for each dollar of profit. It is one of the most widely used equity valuation metrics.

📐 **Formula — P/E Ratio**

P/E = Stock Price / Earnings Per Share (EPS)

**Example:**

- A001 price = $234.69, EPS = $12.62
- P/E = 234.69 / 12.62 = **18.6x**

**Interpretation:**

- P/E < 15 → potentially undervalued (Finance B: avg P/E 14.7)
- P/E > 25 → growth expectations priced in (Healthcare C: avg P/E 28.1)

**In earnings-surprise events (CA001 — A001 beats EPS by 18%):**

Price rises but EPS estimate also rises → P/E compresses. This is why analyst upgrades follow earnings beats.

---

### B.16 Quick Glossary

| **Term** | **Plain English** | **Where Used** |
| --- | --- | --- |
| **Return** | % change in price from one tick to the next | Expected returns, Sharpe |
| **Log Return** | $\ln(P_t / P_{t-1})$ — additive, preferred in finance | EWMA, covariance, Sharpe |
| **Expected Return** | Your prediction of next-tick return for an asset | `/llm/query`, MVO objective |
| **Volatility ($\sigma$)** | Standard deviation of returns — measures risk | Sharpe denominator, MVO |
| **Covariance** | How much two assets move together | Covariance matrix ($\Sigma$) |
| **Correlation** | Covariance scaled to −1..+1 range | Diversification analysis |
| **Sharpe Ratio** | Return per unit of risk — primary scoring metric | 60% of total score |
| **Portfolio Weight** | Fraction of portfolio value in each asset (sum=1) | MVO output, order sizing |
| **MVO** | Optimiser that maximises Sharpe subject to constraints | `optimizer.py` / `cvxpy` |
| **Beta ($\beta$)** | How much asset moves per 1% market move | `fundamentals.json`, risk |
| **P/E Ratio** | Price ÷ Earnings per share; valuation measure | `fundamentals.json` |
| **Dividend Yield** | Annual dividend / price; income return component | CA002 (B003 dividend) |
| **ESG Score** | Environmental/Social/Governance risk (0=best) | CA003, CA006, avoid list |
| **Momentum** | Recent price trend — rising assets tend to keep rising | CA001 trade signal |
| **Turnover** | Total traded value / avg portfolio; capped at 30% | TC005 (disqualifying) |
| **Transaction Cost** | Fee per trade = proportional + fixed | Erodes PnL on over-trading |
| **Stock Split** | Share count multiplied, price divided — no value change | CA004 (D002 at tick 0) |
| **Tick** | One time-step in the simulation (1 minute real-time) | 390 ticks in one day |
| **Alpha** | Return above what the market predicts — your edge | What you are competing on |
| **EWMA** | Exponentially Weighted Moving Average — recent data weighted more | Expected return |
| **Cardinality** | Number of distinct assets held at one time; max 30 | TC004 (disqualifying) |
| **Long-Only** | You can only buy, not short-sell; weights $\ge 0$ | Hard constraint |
| **PnL** | Total portfolio value change from start to end | 20% of total score |
| **Gamma ($\gamma$)** | Risk aversion parameter in MVO — higher = more conservative | `optimizer.py` tunable |
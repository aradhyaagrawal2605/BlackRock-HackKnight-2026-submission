"""
validate_solution.py — Automated Test Cases for BlackRock HackKnight 2026

Validates the four output files against 8 test cases (TC001-TC008):
  TC001: Stock split handling (D002 3-for-1 at tick 0) — CA004
  TC002: Special dividend reaction (B003 at tick ~45) — CA002
  TC003: Earnings beat reaction (A001 at tick ~90) — CA001
  TC004: Cardinality constraint (max 30 holdings) — DISQUALIFYING
  TC005: Turnover constraint (max 30%) — DISQUALIFYING
  TC006: Corporate action — CEO resignation (C005 at tick ~180), regulatory fine (B008 at tick ~280)
  TC007: Bonus — M&A rumour (E004), index rebalance
  TC008: Output file completeness and schema validation

Usage:
    python validate_solution.py [--output-dir ./]
"""

import json
import os
import sys
import argparse
import math
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def safe_get(d: dict, *keys, default=None):
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d


# ---------------------------------------------------------------------------
# Test case implementations
# ---------------------------------------------------------------------------

class TestResult:
    def __init__(self, tc_id: str, name: str, passed: bool, weight: float,
                 details: str = "", disqualifying: bool = False):
        self.tc_id = tc_id
        self.name = name
        self.passed = passed
        self.weight = weight
        self.details = details
        self.disqualifying = disqualifying

    @property
    def score(self) -> float:
        return self.weight if self.passed else 0.0


def tc001_stock_split(snapshots: list, orders: list, corporate_actions: list) -> TestResult:
    """
    TC001: D002 stock split (3-for-1) at tick 0.
    Agent must NOT treat the split as a crash and sell D002 in the first few ticks.
    The price ~$87.84 is correct post-split, not a -66% crash from ~$263.51.
    """
    tc_id, name, weight = "TC001", "Stock Split Handling (D002)", 2.5
    details = []
    passed = True

    early_sells = [
        o for o in orders
        if o.get("ticker") == "D002"
        and o.get("side", "").upper() == "SELL"
        and o.get("tick_index", 999) <= 5
    ]
    if early_sells:
        passed = False
        details.append(f"Agent panic-sold D002 within first 5 ticks ({len(early_sells)} sell orders)")

    for snap in snapshots:
        if snap.get("tick_index", -1) == 0:
            holdings = snap.get("holdings", {})
            d002 = holdings.get("D002", {})
            if isinstance(d002, dict):
                qty = d002.get("quantity", d002.get("qty", 0))
                price = d002.get("price", d002.get("avg_price", 0))
                if price > 0 and abs(price - 87.84) / 87.84 > 0.05:
                    details.append(f"D002 price at tick 0 looks wrong: {price} (expected ~87.84)")
            break

    if not details:
        details.append("D002 split handled correctly — no panic sells detected")

    return TestResult(tc_id, name, passed, weight, "; ".join(details))


def tc002_special_dividend(orders: list, snapshots: list) -> TestResult:
    """
    TC002: B003 special dividend announced around tick 45.
    Agent should buy B003 before ex-date to capture the +2.1% price move.
    """
    tc_id, name, weight = "TC002", "Special Dividend Reaction (B003)", 2.5

    b003_buys_before = [
        o for o in orders
        if o.get("ticker") == "B003"
        and o.get("side", "").upper() == "BUY"
        and o.get("tick_index", 999) <= 55
    ]

    if b003_buys_before:
        return TestResult(tc_id, name, True, weight,
                         f"Agent bought B003 {len(b003_buys_before)} time(s) around dividend announcement")
    else:
        b003_any_buy = any(o.get("ticker") == "B003" and o.get("side", "").upper() == "BUY" for o in orders)
        if b003_any_buy:
            return TestResult(tc_id, name, False, weight,
                             "Agent bought B003 but too late (after tick 55)")
        return TestResult(tc_id, name, False, weight,
                         "Agent never bought B003 around the dividend event")


def tc003_earnings_beat(orders: list) -> TestResult:
    """
    TC003: A001 earnings beat at tick ~90 (+18% EPS surprise).
    Agent should buy A001 within ~10 ticks of the event.
    """
    tc_id, name, weight = "TC003", "Earnings Beat Reaction (A001)", 2.5

    a001_buys = [
        o for o in orders
        if o.get("ticker") == "A001"
        and o.get("side", "").upper() == "BUY"
        and 85 <= o.get("tick_index", -1) <= 100
    ]

    if a001_buys:
        return TestResult(tc_id, name, True, weight,
                         f"Agent bought A001 {len(a001_buys)} time(s) around earnings beat (ticks 85-100)")
    return TestResult(tc_id, name, False, weight,
                     "Agent did not buy A001 near the earnings beat event (tick ~90)")


def tc004_cardinality(snapshots: list) -> TestResult:
    """
    TC004: DISQUALIFYING — max 30 holdings at any tick.
    """
    tc_id, name, weight = "TC004", "Cardinality Constraint (max 30)", 2.5
    max_seen = 0
    worst_tick = -1

    for snap in snapshots:
        holdings = snap.get("holdings", {})
        n_held = sum(
            1 for v in holdings.values()
            if (isinstance(v, dict) and (v.get("quantity", v.get("qty", 0)) > 0))
            or (isinstance(v, (int, float)) and v > 0)
        )
        if n_held > max_seen:
            max_seen = n_held
            worst_tick = snap.get("tick_index", -1)

    passed = max_seen <= 30
    details = f"Max holdings observed: {max_seen} (tick {worst_tick})"
    if not passed:
        details += " — DISQUALIFICATION: exceeded 30 holdings"

    return TestResult(tc_id, name, passed, weight, details, disqualifying=True)


def tc005_turnover(results: dict, orders: list, snapshots: list) -> TestResult:
    """
    TC005: DISQUALIFYING — total turnover must not exceed 30%.
    """
    tc_id, name, weight = "TC005", "Turnover Constraint (max 30%)", 2.5

    turnover = results.get("turnover_ratio", results.get("turnover", None))
    if turnover is not None:
        passed = turnover <= 0.30
        details = f"Reported turnover: {turnover:.4f}"
        if not passed:
            details += " — DISQUALIFICATION: exceeded 30% turnover"
        return TestResult(tc_id, name, passed, weight, details, disqualifying=True)

    total_traded = sum(
        abs(o.get("quantity", o.get("qty", 0)) * o.get("exec_price", o.get("price", 0)))
        for o in orders
    )

    portfolio_values = []
    for snap in snapshots:
        pv = snap.get("portfolio_value", snap.get("total_value", 0))
        if pv > 0:
            portfolio_values.append(pv)

    avg_pv = sum(portfolio_values) / len(portfolio_values) if portfolio_values else 10_000_000
    turnover = total_traded / avg_pv if avg_pv > 0 else 0

    passed = turnover <= 0.30
    details = f"Computed turnover: {turnover:.4f} (traded: ${total_traded:,.0f}, avg PV: ${avg_pv:,.0f})"
    if not passed:
        details += " — DISQUALIFICATION: exceeded 30% turnover"

    return TestResult(tc_id, name, passed, weight, details, disqualifying=True)


def tc006_corporate_actions(orders: list, snapshots: list) -> TestResult:
    """
    TC006: CEO resignation (C005 at tick ~180) and regulatory fine (B008 at tick ~280).
    Agent should reduce or exit C005 and B008 after these events.
    """
    tc_id, name, weight = "TC006", "Corporate Action Response (C005, B008)", 2.5
    checks = []

    c005_sells = [
        o for o in orders
        if o.get("ticker") == "C005"
        and o.get("side", "").upper() == "SELL"
        and 175 <= o.get("tick_index", -1) <= 200
    ]
    if c005_sells:
        checks.append("C005 CEO resignation: correctly reduced position")
    else:
        checks.append("C005 CEO resignation: no sell detected (ticks 175-200)")

    b008_sells = [
        o for o in orders
        if o.get("ticker") == "B008"
        and o.get("side", "").upper() == "SELL"
        and 275 <= o.get("tick_index", -1) <= 300
    ]
    if b008_sells:
        checks.append("B008 regulatory fine: correctly reduced position")
    else:
        checks.append("B008 regulatory fine: no sell detected (ticks 275-300)")

    passed = len(c005_sells) > 0 or len(b008_sells) > 0
    return TestResult(tc_id, name, passed, weight, "; ".join(checks))


def tc007_bonus(orders: list, snapshots: list) -> TestResult:
    """
    TC007: Bonus — M&A rumour (E004) and index rebalance events.
    """
    tc_id, name, weight = "TC007", "Bonus: M&A Rumour / Index Rebalance", 2.5

    e004_buys = [
        o for o in orders
        if o.get("ticker") == "E004"
        and o.get("side", "").upper() == "BUY"
    ]

    if e004_buys:
        return TestResult(tc_id, name, True, weight,
                         f"Agent traded E004 (M&A rumour target) — {len(e004_buys)} buy(s)")
    return TestResult(tc_id, name, False, weight,
                     "Agent did not trade E004 for the M&A rumour event (bonus)")


def tc008_output_completeness(
    orders_log: Any,
    snapshots: Any,
    llm_log: Any,
    results: Any,
) -> TestResult:
    """
    TC008: All 4 output files exist and have valid schema.
    """
    tc_id, name, weight = "TC008", "Output File Completeness", 2.5
    issues = []

    if orders_log is None:
        issues.append("orders_log.json missing")
    elif not isinstance(orders_log, list):
        issues.append("orders_log.json must be a list")

    if snapshots is None:
        issues.append("portfolio_snapshots.json missing")
    elif not isinstance(snapshots, list):
        issues.append("portfolio_snapshots.json must be a list")
    else:
        if len(snapshots) < 390:
            issues.append(f"portfolio_snapshots.json has {len(snapshots)} entries (expected 390)")

    if llm_log is None:
        issues.append("llm_call_log.json missing")
    elif not isinstance(llm_log, list):
        issues.append("llm_call_log.json must be a list")

    if results is None:
        issues.append("results.json missing")
    elif isinstance(results, dict):
        required_keys = ["sharpe_ratio", "pnl", "turnover_ratio"]
        for k in required_keys:
            if k not in results:
                issues.append(f"results.json missing key: {k}")
    else:
        issues.append("results.json must be a dict")

    passed = len(issues) == 0
    details = "All files valid" if passed else "; ".join(issues)
    return TestResult(tc_id, name, passed, weight, details)


# ---------------------------------------------------------------------------
# Main validation runner
# ---------------------------------------------------------------------------

def validate(output_dir: str = "./") -> dict:
    orders_log = load_json(os.path.join(output_dir, "orders_log.json"))
    snapshots = load_json(os.path.join(output_dir, "portfolio_snapshots.json"))
    llm_log = load_json(os.path.join(output_dir, "llm_call_log.json"))
    results = load_json(os.path.join(output_dir, "results.json"))

    orders = orders_log if isinstance(orders_log, list) else []
    snaps = snapshots if isinstance(snapshots, list) else []
    res = results if isinstance(results, dict) else {}
    corp_actions = load_json(os.path.join(output_dir, "corporate_actions.json"))
    corp_actions = corp_actions if isinstance(corp_actions, list) else []

    test_results = [
        tc001_stock_split(snaps, orders, corp_actions),
        tc002_special_dividend(orders, snaps),
        tc003_earnings_beat(orders),
        tc004_cardinality(snaps),
        tc005_turnover(res, orders, snaps),
        tc006_corporate_actions(orders, snaps),
        tc007_bonus(orders, snaps),
        tc008_output_completeness(orders_log, snapshots, llm_log, results),
    ]

    disqualified = any(tr.disqualifying and not tr.passed for tr in test_results)
    total_score = sum(tr.score for tr in test_results) if not disqualified else 0.0
    max_score = sum(tr.weight for tr in test_results)

    print("\n" + "=" * 72)
    print(" BlackRock HackKnight 2026 — Automated Validation Report")
    print("=" * 72)

    for tr in test_results:
        status = "PASS" if tr.passed else ("FAIL [DQ]" if tr.disqualifying else "FAIL")
        print(f"  [{status:>9s}] {tr.tc_id}: {tr.name} ({tr.score:.1f}/{tr.weight:.1f})")
        print(f"             {tr.details}")

    print("-" * 72)
    if disqualified:
        print(f"  DISQUALIFIED — Score: 0 / {max_score:.1f}")
    else:
        print(f"  Total Score: {total_score:.1f} / {max_score:.1f}")
    print("=" * 72)

    # Return summary dict
    summary = {
        "total_score": total_score if not disqualified else 0.0,
        "max_score": max_score,
        "disqualified": disqualified,
        "tests": [
            {
                "tc_id": tr.tc_id,
                "name": tr.name,
                "passed": tr.passed,
                "score": tr.score,
                "weight": tr.weight,
                "details": tr.details,
            }
            for tr in test_results
        ],
    }
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate HackKnight 2026 solution output")
    parser.add_argument("--output-dir", default="./", help="Directory containing output JSON files")
    args = parser.parse_args()
    summary = validate(args.output_dir)
    if summary["disqualified"]:
        sys.exit(1)
    sys.exit(0)

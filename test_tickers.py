"""
IntrinsiQ — Ticker Test Suite
==============================
Tests FCF extraction, key metrics, and all 5 valuation models across
5 S&P 500 tickers per major sector. Run from the valuation_app/ directory:

    python test_tickers.py

Outputs a color-coded summary table + saves full results to test_results.csv
"""

import sys
import os
import time
import traceback
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from src.data_fetcher import DataFetcher
from src.models import (
    dcf_valuation, ddm_valuation, ev_ebitda_valuation,
    pe_relative_valuation, regression_valuation, inverse_dcf,
    compute_blended_value, compute_dynamic_wacc, suggest_dcf_assumptions,
)
from src.peers import get_peers_for_ticker

# ─────────────────────────────────────────────────────────────────────────────
#  Test universe — 5 tickers per sector
# ─────────────────────────────────────────────────────────────────────────────

TEST_UNIVERSE = {
    "Technology":            ["AAPL", "MSFT", "NVDA", "GOOGL", "META"],
    "Healthcare":            ["JNJ",  "LLY",  "UNH",  "ABBV",  "TMO"],
    "Financials":            ["JPM",  "BAC",  "GS",   "V",     "BRK-B"],
    "Consumer Staples":      ["KO",   "PEP",  "PG",   "WMT",   "COST"],
    "Consumer Discretionary":["AMZN", "TSLA", "MCD",  "HD",    "NKE"],
    "Energy":                ["XOM",  "CVX",  "COP",  "SLB",   "PSX"],
    "Industrials":           ["CAT",  "BA",   "UNP",  "HON",   "RTX"],
    "Communication Services":["NFLX", "DIS",  "CMCSA","T",     "VZ"],
    "Utilities":             ["NEE",  "DUK",  "SO",   "D",     "AEP"],
    "Real Estate":           ["PLD",  "AMT",  "EQIX", "SPG",   "O"],
    "Basic Materials":       ["LIN",  "APD",  "NEM",  "FCX",   "NUE"],
}

# DCF assumptions are computed dynamically per ticker using compute_dynamic_wacc
# These are fallback defaults only
DCF_DEFAULTS = dict(
    wacc=0.09,
    growth_rate_stage1=0.10,
    growth_rate_stage2=0.06,
    terminal_growth=0.025,
    stage1_years=5,
    stage2_years=5,
)

# ─────────────────────────────────────────────────────────────────────────────
#  ANSI colors
# ─────────────────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def green(s):  return f"{GREEN}{s}{RESET}"
def yellow(s): return f"{YELLOW}{s}{RESET}"
def red(s):    return f"{RED}{s}{RESET}"
def blue(s):   return f"{BLUE}{s}{RESET}"
def cyan(s):   return f"{CYAN}{s}{RESET}"
def bold(s):   return f"{BOLD}{s}{RESET}"
def dim(s):    return f"{DIM}{s}{RESET}"

def status_icon(ok, warn=False):
    if ok:    return green("✓")
    if warn:  return yellow("~")
    return red("✗")


# ─────────────────────────────────────────────────────────────────────────────
#  Single ticker test
# ─────────────────────────────────────────────────────────────────────────────

def test_ticker(ticker: str, fetcher: DataFetcher, peer_sample: list) -> dict:
    result = {
        "ticker": ticker,
        "name": "",
        "sector": "",
        "price": None,
        "fcf": None,
        "fcf_source": "",
        "ocf": None,
        "has_dividend": False,
        # WACC
        "wacc": None, "coe": None,
        # Model results
        "dcf_value": None,   "dcf_ok": False,
        "ddm_value": None,   "ddm_ok": False,
        "ev_value": None,    "ev_ok": False,
        "pe_value": None,    "pe_ok": False,
        "reg_value": None,   "reg_ok": False,
        "inv_growth": None,  "inv_ok": False,
        "blended": None,
        "upside_pct": None,
        # Diagnostics
        "errors": [],
        "warnings": [],
        "fetch_time_s": 0,
    }

    t0 = time.time()

    try:
        m = fetcher.get_key_metrics(ticker)
        if not m or not m.get("current_price"):
            result["errors"].append("No price data returned")
            return result

        result["name"]         = m.get("name", ticker)
        result["sector"]       = m.get("sector", "N/A")
        result["price"]        = m.get("current_price")
        result["fcf"]          = m.get("free_cashflow")
        result["ocf"]          = m.get("operating_cashflow")
        result["has_dividend"] = (m.get("dividend_rate") or 0) > 0

        # FCF source check
        fcf = m.get("free_cashflow", 0)
        if fcf and fcf > 0:
            result["fcf_source"] = "positive ✓"
        elif fcf and fcf < 0:
            result["fcf_source"] = "negative !"
            result["warnings"].append(f"Negative FCF: ${fcf:,.0f}")
        else:
            result["fcf_source"] = "missing ✗"
            result["warnings"].append("FCF is zero/missing — DCF/InvDCF will fail")

        # Get small peer set for comps models
        peer_data = []
        if peer_sample:
            try:
                peer_data = fetcher.get_peers_data(peer_sample[:4])
                time.sleep(0.2)
            except Exception as e:
                result["warnings"].append(f"Peer fetch partial: {e}")

        # ── Dynamic WACC per ticker ───────────────────────
        wacc_data = compute_dynamic_wacc(
            beta=m.get("beta", 1.0) or 1.0,
            total_debt=m.get("total_debt", 0) or 0,
            market_cap=m.get("market_cap", 1e9) or 1e9,
            sector=m.get("sector", ""),
        )
        dcf_sugg = suggest_dcf_assumptions(m)
        result["wacc"] = wacc_data["wacc"]
        result["coe"]  = wacc_data["cost_of_equity"]

        ticker_dcf = dict(
            wacc=wacc_data["wacc"],
            growth_rate_stage1=dcf_sugg.get("g1", 10.0) / 100,
            growth_rate_stage2=dcf_sugg.get("g2", 6.0) / 100,
            terminal_growth=dcf_sugg.get("terminal", 2.5) / 100,
            stage1_years=5,
            stage2_years=5,
        )

        # ── DCF ──────────────────────────────────────────
        try:
            if fcf and fcf > 0:
                r = dcf_valuation(
                    base_fcf=fcf, **ticker_dcf,
                    net_debt=m.get("net_debt", 0),
                    shares_outstanding=m.get("shares_outstanding", 1),
                )
                if not r.get("error") and r.get("intrinsic_value", 0) > 0:
                    result["dcf_value"] = r["intrinsic_value"]
                    result["dcf_ok"] = True
                else:
                    result["errors"].append(f"DCF: {r.get('error', 'zero value')}")
            else:
                result["errors"].append("DCF: skipped (no positive FCF)")
        except Exception as e:
            result["errors"].append(f"DCF exception: {e}")

        # ── DDM ──────────────────────────────────────────
        try:
            r = ddm_valuation(
                current_dividend=m.get("dividend_rate", 0) or 0,
                dividend_growth_rate=0.05,
                cost_of_equity=0.09,
            )
            if not r.get("error") and r.get("intrinsic_value"):
                result["ddm_value"] = r["intrinsic_value"]
                result["ddm_ok"] = True
            else:
                if not result["has_dividend"]:
                    result["warnings"].append("DDM: no dividend (expected for growth stocks)")
                else:
                    result["errors"].append(f"DDM: {r.get('error','failed')}")
        except Exception as e:
            result["errors"].append(f"DDM exception: {e}")

        # ── EV/EBITDA ─────────────────────────────────────
        try:
            r = ev_ebitda_valuation(
                target_ebitda=m.get("ebitda", 0) or 0,
                peer_metrics=peer_data,
                net_debt=m.get("net_debt", 0),
                shares_outstanding=m.get("shares_outstanding", 1),
            )
            if not r.get("error") and r.get("intrinsic_value", 0) > 0:
                result["ev_value"] = r["intrinsic_value"]
                result["ev_ok"] = True
            else:
                result["errors"].append(f"EV/EBITDA: {r.get('error','failed')}")
        except Exception as e:
            result["errors"].append(f"EV/EBITDA exception: {e}")

        # ── P/E ───────────────────────────────────────────
        try:
            eps = m.get("eps_forward") or m.get("eps", 0) or 0
            r = pe_relative_valuation(eps=eps, peer_metrics=peer_data)
            if not r.get("error") and r.get("intrinsic_value", 0) > 0:
                result["pe_value"] = r["intrinsic_value"]
                result["pe_ok"] = True
            else:
                result["errors"].append(f"P/E: {r.get('error','failed')}")
        except Exception as e:
            result["errors"].append(f"P/E exception: {e}")

        # ── Regression ────────────────────────────────────
        try:
            r = regression_valuation(target_metrics=m, peer_metrics=peer_data)
            if not r.get("error") and r.get("intrinsic_value", 0) > 0:
                result["reg_value"] = r["intrinsic_value"]
                result["reg_ok"] = True
            else:
                result["errors"].append(f"Regression: {r.get('error','failed')}")
        except Exception as e:
            result["errors"].append(f"Regression exception: {e}")

        # ── Inverse DCF ───────────────────────────────────
        try:
            if fcf and fcf > 0 and result["price"]:
                r = inverse_dcf(
                    current_price=result["price"], base_fcf=fcf,
                    wacc=DCF_DEFAULTS["wacc"],
                    terminal_growth=DCF_DEFAULTS["terminal_growth"],
                    net_debt=m.get("net_debt", 0),
                    shares_outstanding=m.get("shares_outstanding", 1),
                )
                if not r.get("error") and r.get("implied_growth") is not None:
                    result["inv_growth"] = r["implied_growth"]
                    result["inv_ok"] = True
                else:
                    result["errors"].append(f"InvDCF: {r.get('error','failed')}")
            else:
                result["errors"].append("InvDCF: skipped (no positive FCF)")
        except Exception as e:
            result["errors"].append(f"InvDCF exception: {e}")

        # ── Blended ───────────────────────────────────────
        model_results = {
            "dcf":        {"intrinsic_value": result["dcf_value"], "error": None if result["dcf_ok"] else "skip"},
            "ddm":        {"intrinsic_value": result["ddm_value"], "error": None if result["ddm_ok"] else "skip"},
            "ev_ebitda":  {"intrinsic_value": result["ev_value"],  "error": None if result["ev_ok"]  else "skip"},
            "pe_relative":{"intrinsic_value": result["pe_value"],  "error": None if result["pe_ok"]  else "skip"},
            "regression": {"intrinsic_value": result["reg_value"], "error": None if result["reg_ok"] else "skip"},
        }
        weights = {"dcf": 0.35, "ddm": 0.10, "ev_ebitda": 0.20, "pe_relative": 0.20, "regression": 0.15}
        b = compute_blended_value(model_results, weights)
        if b.get("blended_value") and result["price"]:
            result["blended"]    = b["blended_value"]
            result["upside_pct"] = ((b["blended_value"] / result["price"]) - 1) * 100

    except Exception as e:
        result["errors"].append(f"FATAL: {traceback.format_exc()}")

    result["fetch_time_s"] = round(time.time() - t0, 1)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Print helpers
# ─────────────────────────────────────────────────────────────────────────────

def fmt_price(v):
    if v is None: return dim("N/A")
    return f"${v:>8.2f}"

def fmt_pct(v):
    if v is None: return dim("  N/A  ")
    color = GREEN if v > 10 else (RED if v < -10 else YELLOW)
    return f"{color}{v:>+6.1f}%{RESET}"

def fmt_fcf(v, src):
    if v is None or v == 0:
        return red("   MISSING  ")
    color = GREEN if v > 0 else RED
    return f"{color}${v/1e9:>5.2f}B{RESET} {dim(src[:8])}"

def model_check(ok):
    return green("●") if ok else red("○")


# ─────────────────────────────────────────────────────────────────────────────
#  Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_tests(sectors=None, delay=1.5):
    fetcher   = DataFetcher()
    all_results = []
    sector_summaries = {}

    universe = {k: v for k, v in TEST_UNIVERSE.items() if sectors is None or k in sectors}
    total_tickers = sum(len(v) for v in universe.values())

    print()
    print(bold("═" * 90))
    print(bold(f"  🔬 IntrinsiQ — Ticker Test Suite"))
    print(bold(f"  McMillin Analytics"))
    print(bold(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  {total_tickers} tickers across {len(universe)} sectors"))
    print(bold("═" * 90))
    print()

    tested = 0
    for sector, tickers in universe.items():
        print(bold(f"\n  {'─'*80}"))
        print(bold(f"  📁  {sector.upper()}"))
        print(bold(f"  {'─'*80}"))
        print(f"  {bold('Ticker'):<12} {bold('Price'):>9} {bold('FCF'):>14}  "
              f"{bold('WACC'):>7}  {bold('D C E P R'):>9}  {bold('Blended'):>9}  {bold('Upside'):>8}  {bold('Issues')}")
        print(f"  {'─'*88}")

        sector_ok = 0
        sector_results = []

        for i, ticker in enumerate(tickers):
            # Use other tickers in sector as peers
            peers = [t for t in tickers if t != ticker]

            print(f"  {cyan(ticker):<20}", end="", flush=True)

            r = test_ticker(ticker, fetcher, peers)
            all_results.append({**r, "sector": sector})
            sector_results.append(r)

            # Model flags: D=DCF C=DDM E=EV P=PE R=Reg
            models = (f"{model_check(r['dcf_ok'])} "
                      f"{model_check(r['ddm_ok'])} "
                      f"{model_check(r['ev_ok'])} "
                      f"{model_check(r['pe_ok'])} "
                      f"{model_check(r['reg_ok'])}")

            issue_count = len(r["errors"])
            warn_count  = len(r["warnings"])
            issues_str  = (red(f"{issue_count} err") if issue_count else "") + \
                          (" " if issue_count and warn_count else "") + \
                          (yellow(f"{warn_count} warn") if warn_count else "") or green("clean")

            wacc_str = f"{r['wacc']*100:.1f}%" if r.get("wacc") else "N/A"
            print(f"{fmt_price(r['price']):>9}  "
                  f"{fmt_fcf(r['fcf'], r['fcf_source']):>14}  "
                  f"{dim(wacc_str):>7}  "
                  f"{models}  "
                  f"{fmt_price(r['blended']):>9}  "
                  f"{fmt_pct(r['upside_pct']):>8}  "
                  f"{issues_str}")

            # Print errors indented
            for err in r["errors"]:
                print(f"    {dim('└─')} {red(err[:80])}")
            for wrn in r["warnings"]:
                print(f"    {dim('└─')} {yellow(wrn[:80])}")

            # Count sector success
            models_ok = sum([r["dcf_ok"], r["ev_ok"], r["pe_ok"], r["reg_ok"]])
            if models_ok >= 2:
                sector_ok += 1

            tested += 1
            if i < len(tickers) - 1:
                time.sleep(delay)

        # Sector summary
        pct = sector_ok / len(tickers) * 100
        color = GREEN if pct >= 80 else (YELLOW if pct >= 50 else RED)
        print(f"\n  Sector pass rate: {color}{sector_ok}/{len(tickers)} ({pct:.0f}%){RESET}")
        sector_summaries[sector] = {"pass": sector_ok, "total": len(tickers), "pct": pct}

    # ── Overall summary ───────────────────────────────────────────────────
    print()
    print(bold("═" * 90))
    print(bold("  OVERALL RESULTS SUMMARY"))
    print(bold("═" * 90))
    print(f"\n  {'Sector':<28} {'Pass Rate':>10}  {'FCF Issues':>12}  {'Blended':>10}")
    print(f"  {'─'*70}")

    total_pass = 0
    total_all  = 0
    for sector, tickers in universe.items():
        s = sector_summaries[sector]
        sector_results_list = [r for r in all_results if r["sector"] == sector]
        fcf_issues = sum(1 for r in sector_results_list if not r["fcf"] or r["fcf"] <= 0)
        has_blended = sum(1 for r in sector_results_list if r["blended"])
        color = GREEN if s["pct"] >= 80 else (YELLOW if s["pct"] >= 50 else RED)
        print(f"  {sector:<28} {color}{s['pass']}/{s['total']} ({s['pct']:.0f}%){RESET:>6}"
              f"  {(red(str(fcf_issues)) if fcf_issues else green('0')):>12}"
              f"  {green(str(has_blended)) + dim('/'+str(len(tickers))):>10}")
        total_pass += s["pass"]
        total_all  += s["total"]

    overall_pct = total_pass / total_all * 100
    color = GREEN if overall_pct >= 80 else (YELLOW if overall_pct >= 50 else RED)
    print(f"\n  {'─'*70}")
    print(bold(f"  {'TOTAL':<28} {color}{total_pass}/{total_all} ({overall_pct:.0f}%){RESET}"))

    # FCF-specific summary
    fcf_ok     = sum(1 for r in all_results if r.get("fcf") and r["fcf"] > 0)
    fcf_neg    = sum(1 for r in all_results if r.get("fcf") and r["fcf"] < 0)
    fcf_miss   = sum(1 for r in all_results if not r.get("fcf") or r["fcf"] == 0)

    print(f"\n  FCF Extraction: {green(str(fcf_ok))} positive  |  {yellow(str(fcf_neg))} negative  |  {red(str(fcf_miss))} missing")

    # ── Save CSV ──────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    csv_cols = ["sector", "ticker", "name", "price", "fcf", "fcf_source", "ocf",
                "wacc", "coe",
                "dcf_value", "ddm_value", "ev_value", "pe_value", "reg_value",
                "blended", "upside_pct", "inv_growth",
                "dcf_ok", "ddm_ok", "ev_ok", "pe_ok", "reg_ok", "inv_ok",
                "errors", "warnings", "fetch_time_s"]
    df = df[[c for c in csv_cols if c in df.columns]]
    out_path = "test_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  {green('✓')} Full results saved to {bold(out_path)}")
    print(bold("\n" + "═" * 90 + "\n"))

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IntrinsiQ Ticker Test Suite")
    parser.add_argument("--sectors", nargs="+", help="Only test specific sectors (default: all)")
    parser.add_argument("--delay",   type=float, default=1.5,
                        help="Seconds between tickers to avoid rate limits (default: 1.5)")
    parser.add_argument("--fast",    action="store_true",
                        help="Reduce delay to 0.5s (risk rate limiting)")
    args = parser.parse_args()

    delay = 0.5 if args.fast else args.delay

    print(f"\n  Delay between tickers: {delay}s  (use --fast to speed up, --delay N to change)")
    print(f"  Estimated runtime: ~{total_t := sum(len(v) for v in TEST_UNIVERSE.values()) * delay / 60:.1f} min for full suite\n")

    df = run_tests(sectors=args.sectors, delay=delay)

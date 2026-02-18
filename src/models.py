"""
Valuation Models Module
=======================
Implements 6 valuation methodologies:
  1. DCF  — Discounted Cash Flow
  2. DDM  — Dividend Discount Model (Gordon Growth)
  3. COMP_EV — EV/EBITDA Comparable Companies
  4. COMP_PE — P/E Relative Valuation
  5. REG  — Multiple Regression on Peer Multiples
  6. INV_DCF — Inverse DCF (implied growth rate)
"""

import numpy as np
import pandas as pd
from typing import Optional
from scipy.optimize import brentq
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════
#  1. DCF — Discounted Cash Flow
# ══════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════
#  Net Debt Sanitization
# ══════════════════════════════════════════════════════════

# Companies with captive finance arms carry lending portfolio debt on the
# consolidated balance sheet (DE, CAT, F, GM, GE, HMC, TM etc).
# This debt is NOT corporate debt — it funds loans to customers.
# Including it destroys DCF equity value. We cap net debt at a
# sector-appropriate EBITDA multiple to screen it out.

SECTOR_NET_DEBT_CAP = {
    # Max net debt as multiple of EBITDA
    "Industrials":            4.0,
    "Consumer Discretionary": 3.0,
    "Consumer Cyclical":      3.0,
    "Technology":             2.0,
    "Healthcare":             3.5,
    "Consumer Staples":       3.5,
    "Consumer Defensive":     3.5,
    "Energy":                 2.5,
    "Communication Services": 4.0,
    "Utilities":              6.0,
    "Real Estate":            8.0,
    "Basic Materials":        3.0,
    "Financials":             None,   # banks — don't cap, use P/BV instead
    "Financial Services":     None,
}

def sanitize_net_debt(net_debt: float, ebitda: float, sector: str) -> tuple:
    """
    Cap net debt at a sector-appropriate EBITDA multiple.
    Returns (sanitized_net_debt, was_capped, cap_applied).
    Handles companies with captive finance arms (DE, CAT, F, GM etc).
    """
    if net_debt is None or net_debt <= 0:
        return (net_debt or 0), False, None
    if not ebitda or ebitda <= 0:
        return net_debt, False, None

    cap_multiple = SECTOR_NET_DEBT_CAP.get(sector)
    if cap_multiple is None:
        return net_debt, False, None  # financials — don't touch

    max_debt = ebitda * cap_multiple
    if net_debt > max_debt:
        return max_debt, True, cap_multiple
    return net_debt, False, None

def dcf_valuation(
    base_fcf: float,
    wacc: float,
    growth_rate_stage1: float,
    growth_rate_stage2: float,
    terminal_growth: float,
    net_debt: float,
    shares_outstanding: float,
    stage1_years: int = 5,
    stage2_years: int = 5,
) -> dict:
    """
    Two-stage DCF model:
      Stage 1: High-growth phase (stage1_years)
      Stage 2: Transition phase (stage2_years)
      Terminal: Gordon Growth at terminal_growth rate
    """
    if wacc <= terminal_growth:
        return {"error": "WACC must exceed terminal growth rate", "intrinsic_value": 0}

    projected_fcfs = []
    pv_fcfs = []
    current_fcf = base_fcf

    # Stage 1
    for i in range(stage1_years):
        current_fcf *= 1 + growth_rate_stage1
        pv = current_fcf / (1 + wacc) ** (i + 1)
        projected_fcfs.append(current_fcf)
        pv_fcfs.append(pv)

    # Stage 2 — transition from stage2 growth rate down to terminal
    # growth_rate_stage2 is the starting rate for this phase, blending to terminal
    for i in range(stage2_years):
        blend = (i + 1) / stage2_years  # goes 0→1 across stage 2
        g = growth_rate_stage2 * (1 - blend) + terminal_growth * blend
        current_fcf *= 1 + g
        pv = current_fcf / (1 + wacc) ** (stage1_years + i + 1)
        projected_fcfs.append(current_fcf)
        pv_fcfs.append(pv)

    total_years = stage1_years + stage2_years

    # Terminal value (Gordon Growth on last projected FCF)
    terminal_fcf = projected_fcfs[-1] * (1 + terminal_growth)
    tv = terminal_fcf / (wacc - terminal_growth)
    pv_tv = tv / (1 + wacc) ** total_years

    pv_sum = sum(pv_fcfs)
    enterprise_value = pv_sum + pv_tv
    equity_value = enterprise_value - net_debt
    intrinsic_value = equity_value / shares_outstanding if shares_outstanding else 0

    return {
        "intrinsic_value": intrinsic_value,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "pv_sum": pv_sum,
        "pv_tv": pv_tv,
        "tv": tv,
        "terminal_fcf": terminal_fcf,
        "projected_fcfs": projected_fcfs,
        "pv_fcfs": pv_fcfs,
        "tv_pct": pv_tv / (pv_sum + pv_tv) if (pv_sum + pv_tv) else 0,
        "years": total_years,
        "error": None,
    }


# ══════════════════════════════════════════════════════════
#  2. DDM — Dividend Discount Model
# ══════════════════════════════════════════════════════════

def ddm_valuation(
    current_dividend: float,
    dividend_growth_rate: float,
    cost_of_equity: float,
) -> dict:
    """
    Gordon Growth Model: P = D1 / (r - g)
    Also computes multi-stage version if growth > terminal:
      Stage 1: 5yr high growth, then stable
    """
    if current_dividend <= 0:
        return {
            "error": "No dividend paid — DDM not applicable",
            "intrinsic_value": None,
        }

    if cost_of_equity <= dividend_growth_rate:
        return {
            "error": "Cost of equity must exceed dividend growth rate",
            "intrinsic_value": None,
        }

    # Simple Gordon Growth
    d1 = current_dividend * (1 + dividend_growth_rate)
    gordon_value = d1 / (cost_of_equity - dividend_growth_rate)

    # Two-stage: 5yr high growth then stable
    stable_growth = min(dividend_growth_rate, 0.04)
    pv_dividends = []
    d = current_dividend
    for i in range(5):
        d *= 1 + dividend_growth_rate
        pv_dividends.append(d / (1 + cost_of_equity) ** (i + 1))

    terminal_div = d * (1 + stable_growth)
    tv = terminal_div / (cost_of_equity - stable_growth)
    pv_tv = tv / (1 + cost_of_equity) ** 5
    two_stage_value = sum(pv_dividends) + pv_tv

    # Use two-stage as primary if growth > stable, else gordon
    primary = two_stage_value if dividend_growth_rate > stable_growth else gordon_value

    return {
        "intrinsic_value": primary,
        "gordon_value": gordon_value,
        "two_stage_value": two_stage_value,
        "d1": d1,
        "pv_dividends": pv_dividends,
        "pv_terminal": pv_tv,
        "implied_yield": d1 / gordon_value if gordon_value else 0,
        "error": None,
    }


# ══════════════════════════════════════════════════════════
#  3. EV/EBITDA Comps
# ══════════════════════════════════════════════════════════

def ev_ebitda_valuation(
    target_ebitda: float,
    peer_metrics: list,
    net_debt: float,
    shares_outstanding: float,
    multiple_override: Optional[float] = None,
) -> dict:
    """
    Values target using median EV/EBITDA of comparable companies.
    peer_metrics: list of dicts with keys: ticker, ev_ebitda, name
    """
    if target_ebitda <= 0:
        return {"error": "EBITDA must be positive for EV/EBITDA valuation", "intrinsic_value": None}

    valid_peers = [
        p for p in peer_metrics
        if p.get("ev_ebitda") and 0 < p["ev_ebitda"] < 200
    ]

    if len(valid_peers) < 2 and not multiple_override:
        return {"error": "Insufficient peer EV/EBITDA data", "intrinsic_value": None}

    multiples = [p["ev_ebitda"] for p in valid_peers]
    median_multiple = multiple_override or np.median(multiples)
    mean_multiple = np.mean(multiples) if multiples else median_multiple
    min_multiple = np.min(multiples) if multiples else median_multiple
    max_multiple = np.max(multiples) if multiples else median_multiple

    # Bear / Base / Bull
    bear_ev = target_ebitda * (np.percentile(multiples, 25) if multiples else median_multiple * 0.75)
    base_ev = target_ebitda * median_multiple
    bull_ev = target_ebitda * (np.percentile(multiples, 75) if multiples else median_multiple * 1.25)

    def ev_to_price(ev):
        return max(0, (ev - net_debt) / shares_outstanding) if shares_outstanding else 0

    return {
        "intrinsic_value": ev_to_price(base_ev),
        "bear_value": ev_to_price(bear_ev),
        "bull_value": ev_to_price(bull_ev),
        "implied_ev": base_ev,
        "median_multiple": round(median_multiple, 2),
        "mean_multiple": round(mean_multiple, 2),
        "min_multiple": round(min_multiple, 2),
        "max_multiple": round(max_multiple, 2),
        "peer_multiples": multiples,
        "peer_names": [p.get("ticker", "?") for p in valid_peers],
        "peer_count": len(valid_peers),
        "error": None,
    }


# ══════════════════════════════════════════════════════════
#  4. P/E Relative Valuation
# ══════════════════════════════════════════════════════════

def pe_relative_valuation(
    eps: float,
    peer_metrics: list,
    use_forward_pe: bool = True,
    multiple_override: Optional[float] = None,
) -> dict:
    """
    Values target using median P/E of comparable companies.
    """
    if eps <= 0:
        return {"error": "EPS must be positive for P/E valuation", "intrinsic_value": None}

    pe_key = "pe_forward" if use_forward_pe else "pe_trailing"
    valid_peers = [
        p for p in peer_metrics
        if p.get(pe_key) and 0 < p[pe_key] < 300
    ]

    # Fallback to trailing if forward not available
    if not valid_peers and use_forward_pe:
        pe_key = "pe_trailing"
        valid_peers = [p for p in peer_metrics if p.get(pe_key) and 0 < p[pe_key] < 300]

    if len(valid_peers) < 2 and not multiple_override:
        return {"error": "Insufficient peer P/E data", "intrinsic_value": None}

    pes = [p[pe_key] for p in valid_peers]
    median_pe = multiple_override or np.median(pes)

    bear_pe = np.percentile(pes, 25) if pes else median_pe * 0.75
    bull_pe = np.percentile(pes, 75) if pes else median_pe * 1.25

    return {
        "intrinsic_value": eps * median_pe,
        "bear_value": eps * bear_pe,
        "bull_value": eps * bull_pe,
        "median_pe": round(median_pe, 2),
        "mean_pe": round(np.mean(pes), 2) if pes else median_pe,
        "peer_pes": pes,
        "peer_names": [p.get("ticker", "?") for p in valid_peers],
        "peer_count": len(valid_peers),
        "pe_type": pe_key,
        "error": None,
    }


# ══════════════════════════════════════════════════════════
#  5. Multiple Regression Valuation
# ══════════════════════════════════════════════════════════

def regression_valuation(
    target_metrics: dict,
    peer_metrics: list,
) -> dict:
    """
    Predicts fair value by regressing on EV/EBITDA and P/E multiples
    across peers, then applying the predicted multiple to the target's
    own EBITDA and EPS.

    Predicts multiples (not raw price) to avoid share-price scale issues
    — e.g. COST at $994 vs KO at $79 would poison a raw-price regression.

    Two sub-models:
      1. Predict EV/EBITDA → apply to target EBITDA → derive price
      2. Predict P/E       → apply to target EPS    → derive price
    Average the two for the final estimate.
    """
    FEATURES = [
        "revenue_growth", "net_margin", "operating_margin",
        "earnings_growth", "pb_ratio", "ps_ratio",
    ]

    # Build peer DataFrame — normalize by fundamentals, not raw price
    rows = []
    for p in peer_metrics:
        price = p.get("current_price")
        ev_ebitda = p.get("ev_ebitda")
        pe = p.get("pe_trailing") or p.get("pe_forward")
        if not price or not (ev_ebitda or pe):
            continue
        row = {"ticker": p.get("ticker", ""), "price": price,
               "ev_ebitda": ev_ebitda, "pe": pe}
        for feat in FEATURES:
            row[feat] = p.get(feat, np.nan)
        rows.append(row)

    if len(rows) < 3:
        return {
            "error": f"Need ≥3 peers with multiples data (have {len(rows)})",
            "intrinsic_value": None,
        }

    df = pd.DataFrame(rows)

    # Select features with enough non-null values
    available = [
        f for f in FEATURES
        if f in df.columns and df[f].notna().sum() >= max(2, len(df) * 0.4)
    ]
    if not available:
        available = [f for f in FEATURES if f in df.columns]

    estimates = []
    r2_scores = []

    def _run_multiple_regression(df, target_col, available, target_metrics):
        """Regress on a multiple, return predicted multiple."""
        sub = df[df[target_col].notna()].copy()
        if len(sub) < 3:
            return None, None

        # Winsorize to remove outliers — widen bounds for small peer sets
        q_lo = 0.10 if len(sub) >= 8 else 0.0
        q_hi = 0.90 if len(sub) >= 8 else 1.0
        lo, hi = sub[target_col].quantile(q_lo), sub[target_col].quantile(q_hi)
        sub = sub[(sub[target_col] >= lo) & (sub[target_col] <= hi)]
        if len(sub) < 2:
            return None, None

        feat_cols = [f for f in available if sub[f].notna().sum() >= 2]
        if not feat_cols:
            return float(sub[target_col].median()), 0.0

        for f in feat_cols:
            sub[f] = sub[f].fillna(sub[f].median())

        X = sub[feat_cols].values
        y = sub[target_col].values

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        mdl = Ridge(alpha=1.0)
        mdl.fit(X_s, y)
        y_pred = mdl.predict(X_s)
        r2 = max(0, r2_score(y, y_pred))

        # Build target feature vector
        t_row = []
        for f in feat_cols:
            v = target_metrics.get(f, np.nan)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = float(sub[f].median())
            t_row.append(v)

        pred_multiple = float(mdl.predict(scaler.transform([t_row]))[0])
        # Clamp to sane range (between 50th/95th of peers)
        pred_multiple = np.clip(pred_multiple, sub[target_col].quantile(0.10), sub[target_col].quantile(0.90))
        return pred_multiple, r2

    # ── Model 1: EV/EBITDA multiple ───────────────────────────────────────
    pred_ev_mult, r2_ev = _run_multiple_regression(df, "ev_ebitda", available, target_metrics)
    if pred_ev_mult and pred_ev_mult > 0:
        target_ebitda = target_metrics.get("ebitda", 0) or 0
        net_debt      = target_metrics.get("net_debt", 0) or 0
        shares        = target_metrics.get("shares_outstanding", 1) or 1
        if target_ebitda > 0 and shares > 0:
            ev = pred_ev_mult * target_ebitda
            equity_val = ev - net_debt
            price_from_ev = equity_val / shares
            if price_from_ev > 0:
                estimates.append(price_from_ev)
                r2_scores.append(r2_ev or 0)

    # ── Model 2: P/E multiple ─────────────────────────────────────────────
    pred_pe_mult, r2_pe = _run_multiple_regression(df, "pe", available, target_metrics)
    if pred_pe_mult and pred_pe_mult > 0:
        eps = target_metrics.get("eps_forward") or target_metrics.get("eps", 0) or 0
        if eps > 0:
            price_from_pe = pred_pe_mult * eps
            if price_from_pe > 0:
                estimates.append(price_from_pe)
                r2_scores.append(r2_pe or 0)

    if not estimates:
        return {"error": "Could not compute multiple-based estimate", "intrinsic_value": None}

    # Weighted average by R²
    if sum(r2_scores) > 0:
        weights = [r / sum(r2_scores) for r in r2_scores]
        predicted_price = sum(e * w for e, w in zip(estimates, weights))
    else:
        predicted_price = float(np.mean(estimates))

    # Sanity clamp — regression can blow up when EBITDA is large and multiples
    # are noisy (energy, industrials). Cap at 4x the median peer price.
    peer_prices = [p.get("current_price", 0) for p in peer_metrics if p.get("current_price", 0) > 0]
    if peer_prices:
        peer_median = float(np.median(peer_prices))
        target_price = target_metrics.get("current_price", 0) or 0
        # Use whichever reference is higher so we don't clip legitimate upsides
        price_ref = max(peer_median, target_price)
        predicted_price = min(predicted_price, price_ref * 4.0)

    predicted_price = max(0, predicted_price)
    spread = np.std(estimates) if len(estimates) > 1 else predicted_price * 0.15

    return {
        "intrinsic_value": predicted_price,
        "bear_value": max(0, predicted_price - spread),
        "bull_value": predicted_price + spread,
        "r_squared": round(float(np.mean(r2_scores)), 4),
        "features_used": available,
        "coefficients": {},
        "intercept": 0,
        "residual_std": round(spread, 2),
        "peer_count": len(rows),
        "feature_importance": {},
        "ev_multiple_used": round(pred_ev_mult, 2) if pred_ev_mult else None,
        "pe_multiple_used": round(pred_pe_mult, 2) if pred_pe_mult else None,
        "error": None,
    }


# ══════════════════════════════════════════════════════════
#  6. Inverse DCF — Implied Growth Rate
# ══════════════════════════════════════════════════════════

def inverse_dcf(
    current_price: float,
    base_fcf: float,
    wacc: float,
    terminal_growth: float,
    net_debt: float,
    shares_outstanding: float,
    stage1_years: int = 5,
    stage2_years: int = 5,
) -> dict:
    """
    Solves for the FCF growth rate implied by the current market price.
    Uses scipy brentq root-finding on the DCF equation.
    """
    if base_fcf <= 0:
        return {"error": "Positive FCF required for Inverse DCF", "implied_growth": None}

    if wacc <= terminal_growth:
        return {"error": "WACC must exceed terminal growth rate", "implied_growth": None}

    def price_diff(growth_rate):
        result = dcf_valuation(
            base_fcf=base_fcf,
            wacc=wacc,
            growth_rate_stage1=growth_rate,
            growth_rate_stage2=max(growth_rate * 0.5, terminal_growth),
            terminal_growth=terminal_growth,
            net_debt=net_debt,
            shares_outstanding=shares_outstanding,
            stage1_years=stage1_years,
            stage2_years=stage2_years,
        )
        if result.get("error"):
            return -current_price
        return result["intrinsic_value"] - current_price

    # Find bracket
    try:
        # Check if solution exists
        low_val = price_diff(-0.40)
        high_val = price_diff(0.80)

        if low_val * high_val > 0:
            # Try wider bracket
            try:
                high_val = price_diff(2.0)
                if low_val * high_val > 0:
                    return {
                        "error": "Cannot solve — price implies extreme growth assumptions",
                        "implied_growth": None,
                        "low_check": low_val,
                        "high_check": high_val,
                    }
                implied_g = brentq(price_diff, -0.40, 2.0, xtol=1e-6, maxiter=200)
            except Exception:
                return {"error": "Root finding failed — unusual FCF/price relationship", "implied_growth": None}
        else:
            implied_g = brentq(price_diff, -0.40, 0.80, xtol=1e-6, maxiter=200)
    except ValueError as e:
        return {"error": f"Root finding error: {e}", "implied_growth": None}

    # Sensitivity: what price at different growth scenarios
    scenarios = {}
    for g in [-0.10, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        r = dcf_valuation(
            base_fcf=base_fcf, wacc=wacc,
            growth_rate_stage1=g,
            growth_rate_stage2=max(g * 0.5, terminal_growth),
            terminal_growth=terminal_growth,
            net_debt=net_debt, shares_outstanding=shares_outstanding,
            stage1_years=stage1_years, stage2_years=stage2_years,
        )
        if not r.get("error"):
            scenarios[g] = r["intrinsic_value"]

    return {
        "implied_growth": implied_g,
        "current_price": current_price,
        "scenarios": scenarios,
        "wacc_used": wacc,
        "terminal_growth_used": terminal_growth,
        "error": None,
    }



# ══════════════════════════════════════════════════════════
#  Dynamic WACC / Cost of Equity Calculator
# ══════════════════════════════════════════════════════════

# Sector-level effective tax rate estimates
SECTOR_TAX_RATES = {
    "Technology": 0.15,
    "Healthcare": 0.17,
    "Financials": 0.22,
    "Consumer Staples": 0.22,
    "Consumer Discretionary": 0.21,
    "Energy": 0.24,
    "Industrials": 0.21,
    "Communication Services": 0.20,
    "Utilities": 0.23,
    "Real Estate": 0.10,
    "Basic Materials": 0.22,
}

# Sector-level pre-tax cost of debt estimates
SECTOR_COST_OF_DEBT = {
    "Technology": 0.038,
    "Healthcare": 0.040,
    "Financials": 0.045,
    "Consumer Staples": 0.038,
    "Consumer Discretionary": 0.045,
    "Energy": 0.048,
    "Industrials": 0.042,
    "Communication Services": 0.050,
    "Utilities": 0.042,
    "Real Estate": 0.045,
    "Basic Materials": 0.045,
}

def compute_dynamic_wacc(
    beta: float,
    total_debt: float,
    market_cap: float,
    sector: str = "",
    risk_free_rate: float = 0.0425,
    equity_risk_premium: float = 0.055,
) -> dict:
    """
    Compute WACC and Cost of Equity dynamically from company fundamentals.

    Uses CAPM for cost of equity:
        CoE = Rf + Beta * ERP

    Capital structure weights from market values:
        E_weight = Market Cap / (Market Cap + Total Debt)
        D_weight = Total Debt / (Market Cap + Total Debt)

    Sector-calibrated tax rate and cost of debt.
    """
    # Sanitize inputs
    beta       = max(0.1, min(beta or 1.0, 3.5))
    market_cap = max(market_cap or 1e9, 1e6)
    total_debt = max(total_debt or 0, 0)

    # Cost of equity via CAPM
    cost_of_equity = risk_free_rate + beta * equity_risk_premium

    # Capital structure weights
    total_capital = market_cap + total_debt
    equity_weight = market_cap / total_capital
    debt_weight   = total_debt / total_capital

    # Sector defaults
    tax_rate      = SECTOR_TAX_RATES.get(sector, 0.21)
    cost_of_debt  = SECTOR_COST_OF_DEBT.get(sector, 0.043)

    # After-tax cost of debt
    after_tax_debt = cost_of_debt * (1 - tax_rate)

    # WACC
    wacc = equity_weight * cost_of_equity + debt_weight * after_tax_debt

    # Apply sector floor/ceiling — utilities/REITs can go lower, high-beta tech higher
    sector_floors = {
        "Utilities": 0.055, "Real Estate": 0.060,
        "Consumer Staples": 0.065, "Healthcare": 0.070,
    }
    wacc = max(wacc, sector_floors.get(sector, 0.065))
    wacc = min(wacc, 0.18)  # Cap at 18%

    return {
        "wacc": round(wacc, 4),
        "cost_of_equity": round(cost_of_equity, 4),
        "cost_of_debt": round(cost_of_debt, 4),
        "after_tax_debt": round(after_tax_debt, 4),
        "equity_weight": round(equity_weight, 4),
        "debt_weight": round(debt_weight, 4),
        "beta": round(beta, 2),
        "tax_rate": tax_rate,
        "risk_free_rate": risk_free_rate,
        "equity_risk_premium": equity_risk_premium,
    }

def suggest_dcf_assumptions(metrics: dict) -> dict:
    """
    Suggest sensible DCF growth rate defaults based on company fundamentals.
    Uses sector-specific floors so mature/stable companies don't get
    unrealistically low growth rates from noisy TTM data.
    Returns a dict of suggested slider values (% for sliders).
    """
    sector      = metrics.get("sector", "")
    rev_growth  = metrics.get("revenue_growth", 0) or 0
    earn_growth = metrics.get("earnings_growth", 0) or 0
    beta        = metrics.get("beta", 1.0) or 1.0

    # Sector-specific g1 floors (historical FCF CAGR anchors)
    # These prevent TTM noise from dragging defaults too low
    SECTOR_G1_FLOOR = {
        "Technology":             0.12,  # tech compounds fast
        "Consumer Discretionary": 0.10,
        "Consumer Cyclical":      0.10,
        "Healthcare":             0.08,
        "Financials":             0.06,
        "Financial Services":     0.06,
        "Consumer Staples":       0.06,  # KO/PEP ~6-8% FCF CAGR
        "Consumer Defensive":     0.06,
        "Energy":                 0.05,
        "Industrials":            0.07,
        "Communication Services": 0.08,
        "Utilities":              0.04,
        "Real Estate":            0.05,
        "Basic Materials":        0.05,
    }

    # Sector-specific g1 caps
    SECTOR_G1_CAP = {
        "Technology":             0.35,
        "Consumer Discretionary": 0.30,
        "Consumer Cyclical":      0.30,
        "Healthcare":             0.20,
        "Financials":             0.15,
        "Financial Services":     0.15,
        "Consumer Staples":       0.12,
        "Consumer Defensive":     0.12,
        "Energy":                 0.15,
        "Industrials":            0.15,
        "Communication Services": 0.20,
        "Utilities":              0.08,
        "Real Estate":            0.10,
        "Basic Materials":        0.12,
    }

    floor = SECTOR_G1_FLOOR.get(sector, 0.06)
    cap   = SECTOR_G1_CAP.get(sector, 0.20)

    # Blend TTM data with sector floor — TTM gets 40% weight, floor 60%
    # This prevents a single bad quarter from tanking the assumption
    raw_g1 = (earn_growth * 0.6 + rev_growth * 0.4)
    raw_g1 = max(-0.05, min(raw_g1, 0.50))  # sanity clamp
    blended_g1 = raw_g1 * 0.4 + floor * 0.6 if (rev_growth or earn_growth) else floor

    g1 = round(min(max(blended_g1, floor), cap) * 100, 1)
    g2 = round(g1 * 0.60, 1)   # stage 2 decelerates ~40% from stage 1

    # Terminal: 2.5% for most, 3.0% for high-growth sectors
    terminal = 3.0 if sector in ("Technology", "Consumer Discretionary", "Consumer Cyclical") else 2.5

    return {
        "g1": g1,
        "g2": max(round(terminal + 0.5, 1), g2),  # g2 must be above terminal
        "terminal": terminal,
    }

# ══════════════════════════════════════════════════════════
#  Blended Valuation
# ══════════════════════════════════════════════════════════

def compute_blended_value(
    model_results: dict,
    weights: dict,
) -> dict:
    """
    Computes weighted average intrinsic value from all available models.
    weights: dict of model_key -> weight (0-1), should sum to ~1
    Skips models with errors or None values and renormalizes.
    """
    contributions = {}
    total_weight = 0.0

    for key, weight in weights.items():
        result = model_results.get(key, {})
        if not result:
            continue
        if result.get("error") or result.get("intrinsic_value") is None:
            continue
        val = result["intrinsic_value"]
        if val is not None and val > 0 and np.isfinite(val) and weight > 0:
            contributions[key] = {"value": val, "weight": weight}
            total_weight += weight

    if not contributions or total_weight == 0:
        return {"blended_value": None, "contributions": {}, "error": "No valid model outputs"}

    # Renormalize
    blended = 0.0
    normalized_contributions = {}
    for key, item in contributions.items():
        norm_weight = item["weight"] / total_weight
        blended += item["value"] * norm_weight
        normalized_contributions[key] = {
            "value": item["value"],
            "raw_weight": item["weight"],
            "effective_weight": norm_weight,
            "contribution": item["value"] * norm_weight,
        }

    return {
        "blended_value": blended,
        "contributions": normalized_contributions,
        "active_models": list(contributions.keys()),
        "skipped_models": [k for k in weights if k not in contributions],
        "error": None,
    }


# ══════════════════════════════════════════════════════════
#  Sensitivity Analysis
# ══════════════════════════════════════════════════════════

def sensitivity_dcf(
    base_fcf: float,
    net_debt: float,
    shares_outstanding: float,
    terminal_growth: float = 0.025,
    wacc_range: list = None,
    growth_range: list = None,
    stage1_years: int = 5,
    stage2_years: int = 5,
) -> pd.DataFrame:
    """
    Returns a DataFrame of intrinsic values for a grid of WACC × Growth rate.
    """
    if wacc_range is None:
        wacc_range = [0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12]
    if growth_range is None:
        growth_range = [-0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25]

    rows = []
    for g in growth_range:
        row = {}
        for w in wacc_range:
            r = dcf_valuation(
                base_fcf=base_fcf, wacc=w,
                growth_rate_stage1=g,
                growth_rate_stage2=max(g * 0.5, terminal_growth),
                terminal_growth=terminal_growth,
                net_debt=net_debt, shares_outstanding=shares_outstanding,
                stage1_years=stage1_years, stage2_years=stage2_years,
            )
            row[f"{w:.0%}"] = round(r.get("intrinsic_value", 0), 2) if not r.get("error") else 0
        rows.append(row)

    df = pd.DataFrame(rows, index=[f"{g:.0%}" for g in growth_range])
    df.index.name = "Growth Rate"
    df.columns.name = "WACC"
    return df


MODEL_LABELS = {
    "dcf": "DCF",
    "ddm": "DDM",
    "ev_ebitda": "EV/EBITDA",
    "pe_relative": "P/E Relative",
    "regression": "Regression",
}

MODEL_COLORS = {
    "dcf": "#3b82f6",
    "ddm": "#a855f7",
    "ev_ebitda": "#06b6d4",
    "pe_relative": "#f59e0b",
    "regression": "#10b981",
}


# ══════════════════════════════════════════════════════════
#  Financial Sector Models (Banks, Insurance, Fin Services)
# ══════════════════════════════════════════════════════════

def excess_return_valuation(
    book_value_per_share: float,
    roe: float,
    cost_of_equity: float,
    growth_rate: float,
    terminal_growth: float = 0.03,
    stage1_years: int = 5,
) -> dict:
    """
    Residual Income / Excess Return Model for financial companies.

    Value = BV + PV of excess returns during high-growth phase
                + PV of terminal excess return (stable phase)

    Excess Return per year = (ROE - CoE) × Book Value
    Book value grows at retention ratio × ROE each year.

    This is the theoretically correct model for banks:
    - If ROE > CoE → premium to book (bank creates value)
    - If ROE < CoE → discount to book (bank destroys value)
    - Returns exactly book value when ROE = CoE
    """
    if not book_value_per_share or book_value_per_share <= 0:
        return {"error": "No book value data", "intrinsic_value": None}
    if not roe or roe <= 0:
        return {"error": "No ROE data", "intrinsic_value": None}
    if cost_of_equity <= terminal_growth:
        return {"error": "Cost of equity must exceed terminal growth", "intrinsic_value": None}

    # Retention ratio drives book value growth
    # Assume payout = 1 - (g / ROE), minimum 20% retention
    retention = max(0.20, min(0.80, growth_rate / roe if roe > 0 else 0.40))

    bv = book_value_per_share
    pv_excess = 0.0

    for i in range(stage1_years):
        excess_return = (roe - cost_of_equity) * bv
        pv_excess += excess_return / (1 + cost_of_equity) ** (i + 1)
        bv *= (1 + roe * retention)  # book value grows with retained earnings

    # Terminal excess return (stable phase) — ROE fades toward CoE + small spread
    terminal_roe = cost_of_equity + (roe - cost_of_equity) * 0.5  # fade half the spread
    terminal_excess = (terminal_roe - cost_of_equity) * bv
    pv_terminal = (terminal_excess / (cost_of_equity - terminal_growth)) / \
                  (1 + cost_of_equity) ** stage1_years

    intrinsic_value = book_value_per_share + pv_excess + pv_terminal

    return {
        "intrinsic_value": max(0, intrinsic_value),
        "book_value_per_share": book_value_per_share,
        "roe": roe,
        "cost_of_equity": cost_of_equity,
        "excess_return_pv": pv_excess,
        "terminal_value_pv": pv_terminal,
        "premium_to_book": (intrinsic_value / book_value_per_share - 1) if book_value_per_share else 0,
        "error": None,
    }


def pbv_relative_valuation(
    target_metrics: dict,
    peer_metrics: list,
) -> dict:
    """
    P/BV Relative Valuation for financial companies.
    Adjusts peer P/BV multiples for ROE differentials using the
    theoretical relationship: Fair P/BV = ROE / CoE

    Approach:
    1. Compute ROE-adjusted fair P/BV for each peer
    2. Apply median adjusted multiple to target book value
    """
    if not target_metrics.get("book_value") or target_metrics["book_value"] <= 0:
        return {"error": "No book value data for target", "intrinsic_value": None}

    target_bv   = target_metrics.get("book_value", 0)
    target_roe  = target_metrics.get("roe", 0) or 0
    target_coe  = target_metrics.get("cost_of_equity", 0.10) or 0.10

    # Collect peer P/BV and ROE data
    peer_rows = []
    for p in peer_metrics:
        pb  = p.get("pb_ratio", 0) or 0
        roe = p.get("roe", 0) or 0
        if pb > 0 and roe > 0:
            # ROE-adjusted P/BV: normalize by ROE/CoE ratio
            coe_est = 0.10  # use 10% as peer CoE proxy
            adj_pb = pb / (roe / coe_est) if roe > 0 else pb
            peer_rows.append({"pb": pb, "roe": roe, "adj_pb": adj_pb})

    if len(peer_rows) < 2:
        # Fall back to simple median P/BV
        peer_pbs = [p.get("pb_ratio", 0) for p in peer_metrics if p.get("pb_ratio", 0) > 0]
        if not peer_pbs:
            return {"error": "Insufficient peer P/BV data", "intrinsic_value": None}
        median_pb = float(np.median(peer_pbs))
        intrinsic = median_pb * target_bv
        return {
            "intrinsic_value": max(0, intrinsic),
            "pb_multiple_used": round(median_pb, 2),
            "method": "simple_median",
            "error": None,
        }

    # Use ROE-adjusted median P/BV, then re-apply target's ROE premium
    median_adj_pb = float(np.median([r["adj_pb"] for r in peer_rows]))
    # Scale back up by target ROE / CoE ratio
    roe_coe_ratio = target_roe / target_coe if target_coe > 0 and target_roe > 0 else 1.0
    roe_coe_ratio = np.clip(roe_coe_ratio, 0.3, 5.0)  # sanity clamp
    fair_pb = median_adj_pb * roe_coe_ratio
    intrinsic = fair_pb * target_bv

    return {
        "intrinsic_value": max(0, intrinsic),
        "pb_multiple_used": round(fair_pb, 2),
        "roe_coe_ratio": round(roe_coe_ratio, 2),
        "peer_median_adj_pb": round(median_adj_pb, 2),
        "method": "roe_adjusted",
        "error": None,
    }


def is_financial_sector(sector: str) -> bool:
    """Returns True if the company is in financial services."""
    return sector in ("Financials", "Financial Services", "Insurance")

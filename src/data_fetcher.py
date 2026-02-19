"""
Data fetcher using yfinance with a 24-hour file-based cache.
Implements tiered caching:
  1. In-memory (session-level via st.cache_data)
  2. Disk cache (24-hour TTL using pickle + JSON metadata)
  3. Graceful fallback to stale data on rate limit errors
"""

import os
import json
import time
import pickle
import logging
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

CACHE_DIR = ".valuation_cache"
CACHE_TTL = 86400  # 24 hours


# ─────────────────────────────────────────────
#  Low-level disk cache helpers
# ─────────────────────────────────────────────

def _safe_key(key: str) -> str:
    return hashlib.md5(key.encode()).hexdigest()


def _meta_path(key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{_safe_key(key)}.meta")


def _data_path(key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{_safe_key(key)}.pkl")


def _is_valid(key: str) -> bool:
    mp = _meta_path(key)
    dp = _data_path(key)
    if not os.path.exists(mp) or not os.path.exists(dp):
        return False
    try:
        with open(mp, "r") as f:
            meta = json.load(f)
        return (time.time() - meta.get("ts", 0)) < CACHE_TTL
    except Exception:
        return False


def _has_stale(key: str) -> bool:
    return os.path.exists(_data_path(key))


def _load(key: str):
    dp = _data_path(key)
    if not os.path.exists(dp):
        return None
    try:
        with open(dp, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _save(key: str, data) -> None:
    try:
        with open(_data_path(key), "wb") as f:
            pickle.dump(data, f)
        with open(_meta_path(key), "w") as f:
            json.dump({"ts": time.time(), "dt": datetime.now().isoformat()}, f)
    except Exception as e:
        logger.warning(f"Cache write failed: {e}")


def _fetch_with_retry(func, *args, retries=3, **kwargs):
    """Call func with exponential backoff retries."""
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt + 0.5
                logger.info(f"Retry {attempt+1}/{retries} in {wait:.1f}s — {e}")
                time.sleep(wait)
            else:
                raise


# ─────────────────────────────────────────────
#  Public DataFetcher class
# ─────────────────────────────────────────────

class DataFetcher:
    """
    All yfinance calls go through here.
    Returns (data, from_cache, cache_age_hours) tuples.
    """

    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
            self.available = True
        except ImportError:
            self.available = False
            logger.error("yfinance not installed")

    def _get(self, key: str, fetch_func, *args, **kwargs):
        """Generic cached get."""
        if _is_valid(key):
            data = _load(key)
            if data is not None:
                meta = json.load(open(_meta_path(key)))
                age = (time.time() - meta["ts"]) / 3600
                return data, True, round(age, 1)

        try:
            data = _fetch_with_retry(fetch_func, *args, **kwargs)
            if data is not None:
                _save(key, data)
                return data, False, 0.0
        except Exception as e:
            logger.warning(f"Fetch failed ({key}): {e}")
            # Fallback to stale cache
            if _has_stale(key):
                data = _load(key)
                try:
                    meta = json.load(open(_meta_path(key)))
                    age = (time.time() - meta["ts"]) / 3600
                except Exception:
                    age = -1
                logger.info(f"Using stale cache for {key}")
                return data, True, round(age, 1)

        return None, False, -1

    # ── info ──────────────────────────────────

    def get_info(self, ticker: str):
        key = f"info_{ticker.upper()}"

        def fetch():
            stock = self.yf.Ticker(ticker)
            info = stock.info
            if not info or len(info) < 5:
                raise ValueError("Empty info returned")
            return info

        return self._get(key, fetch)

    # ── financials ────────────────────────────

    def get_financials(self, ticker: str):
        key = f"financials_{ticker.upper()}"

        def fetch():
            stock = self.yf.Ticker(ticker)
            return {
                "income_stmt": stock.income_stmt,
                "balance_sheet": stock.balance_sheet,
                "cash_flow": stock.cashflow,
                "quarterly_income": stock.quarterly_income_stmt,
                "quarterly_cashflow": stock.quarterly_cashflow,
            }

        return self._get(key, fetch)

    # ── price history ─────────────────────────

    def get_history(self, ticker: str, period: str = "2y"):
        key = f"history_{ticker.upper()}_{period}"

        def fetch():
            stock = self.yf.Ticker(ticker)
            hist = stock.history(period=period)
            if hist.empty:
                raise ValueError("Empty price history")
            return hist

        return self._get(key, fetch)

    # ── FCF helpers ────────────────────────────

    @staticmethod
    def _extract_fcf_from_cashflow(cf_df, ticker="") -> tuple:
        """
        Robustly extract Free Cash Flow and Operating Cash Flow from a yfinance
        cashflow DataFrame. Returns (fcf, ocf) or (None, None).

        Strategy:
          1. Look for a direct FreeCashFlow row
          2. Compute OCF - CapEx
          3. Fall back to OCF alone
        """
        if cf_df is None or cf_df.empty:
            return None, None

        # Print available rows for debugging (only in dev)
        logger.debug(f"{ticker} cash flow rows: {list(cf_df.index)}")

        def get_first_valid(df, idx_label):
            """Get first non-NaN numeric value from a row, scanning all columns."""
            try:
                row = df.loc[idx_label]
                # Scan all columns for first non-NaN value (not just iloc[0])
                for val in row:
                    try:
                        f = float(val)
                        import math
                        if not math.isnan(f):
                            return f
                    except (TypeError, ValueError):
                        continue
            except Exception:
                pass
            return None

        def find_row(df, *keywords):
            """Search rows case-insensitively for any of the keywords."""
            for idx in df.index:
                idx_norm = str(idx).lower().replace(" ", "").replace("_", "").replace("-", "")
                for kw in keywords:
                    kw_norm = kw.lower().replace(" ", "").replace("_", "").replace("-", "")
                    if kw_norm == idx_norm or kw_norm in idx_norm:
                        val = get_first_valid(df, idx)
                        if val is not None:
                            return val
            return None

        # ── Step 1: Try direct FreeCashFlow row ──────────────────────────
        # First try exact label match (fastest, most reliable)
        fcf_direct = None
        for exact_label in ["Free Cash Flow", "FreeCashFlow", "FreeCashflow", "Free Cashflow"]:
            if exact_label in cf_df.index:
                fcf_direct = get_first_valid(cf_df, exact_label)
                if fcf_direct is not None:
                    logger.debug(f"{ticker}: FCF found via exact label '{exact_label}' = {fcf_direct:,.0f}")
                    break

        # Fall back to fuzzy search
        if fcf_direct is None:
            fcf_direct = find_row(cf_df,
                "FreeCashFlow", "FreeCashflow",
                "Free Cash Flow", "freecashflow",
            )
        if fcf_direct is not None:
            # Also try to get OCF for reporting
            ocf_direct = find_row(cf_df,
                "OperatingCashFlow", "CashFromOperations",
                "NetCashProvidedByOperatingActivities",
                "TotalCashFromOperatingActivities",
                "OperatingActivities",
            )
            logger.debug(f"{ticker}: FCF from direct row = {fcf_direct:,.0f}")
            return fcf_direct, (ocf_direct or fcf_direct)

        # ── Step 2: OCF - CapEx ───────────────────────────────────────────
        ocf = find_row(cf_df,
            "OperatingCashFlow",
            "CashFromOperations",
            "NetCashProvidedByOperatingActivities",
            "NetCashFromOperatingActivities",
            "TotalCashFromOperatingActivities",
            "CashGeneratedFromOperations",
            "OperatingActivities",
            "NetCashProvidedFromOperatingActivities",
        )

        capex = find_row(cf_df,
            "CapitalExpenditures",
            "CapitalExpenditure",
            "CapEx",
            "PurchaseOfPPE",
            "PurchaseOfPlantAndEquipment",
            "AcquisitionOfPPE",
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PurchaseOfPropertyPlantAndEquipment",
            "NetPPEPurchaseAndSale",
            "InvestmentInPropertyPlantAndEquipment",
        )

        if ocf is not None and capex is not None:
            # CapEx is typically stored as negative; if positive flip it
            fcf = ocf + capex if capex < 0 else ocf - capex
            logger.debug(f"{ticker}: FCF computed = OCF({ocf:,.0f}) - CapEx({capex:,.0f}) = {fcf:,.0f}")
            return fcf, ocf

        # ── Step 3: OCF proxy ─────────────────────────────────────────────
        if ocf is not None:
            logger.debug(f"{ticker}: Using OCF as FCF proxy = {ocf:,.0f}")
            return ocf, ocf

        return None, None

    # ── convenience: extracted metrics ────────

    def get_key_metrics(self, ticker: str) -> dict:
        """
        Returns a flat dict of the key metrics needed for valuation.
        Falls back to 0 / None gracefully.
        """
        info, from_cache, age = self.get_info(ticker)
        fin, _, _ = self.get_financials(ticker)

        if not info:
            return {}

        def safe(val, default=0):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            return val

        # ── Robust FCF / OCF extraction ───────────────────────────────────
        # Priority 1: info dict direct fields
        # Priority 2: compute from cash flow statement (Operating CF - CapEx)
        # Priority 3: OCF as proxy
        # Use None as sentinel (not 0) so we can distinguish "missing" from "zero"
        fcf_info = info.get("freeCashflow")
        ocf_info = info.get("operatingCashflow")
        # Normalize NaN to None
        import math
        def _none_if_nan(v):
            if v is None: return None
            try:
                return None if math.isnan(float(v)) else float(v)
            except Exception:
                return None
        fcf_info = _none_if_nan(fcf_info)
        ocf_info = _none_if_nan(ocf_info)

        fcf_cf, ocf_cf = None, None
        if fin and fin.get("cash_flow") is not None:
            fcf_cf, ocf_cf = self._extract_fcf_from_cashflow(fin["cash_flow"], ticker)

        def best_val(from_info, from_cf, min_abs=1000):
            """
            Pick the best FCF/OCF value using this priority:
              1. Cash flow statement if positive and meaningful (most reliable)
              2. Info dict if positive and meaningful
              3. Info dict if negative but cf is also negative (both bad)
              4. Whatever is non-None
            This prevents Yahoo's stale/negative info dict values from
            overriding a perfectly good positive value in the cash flow stmt.
            """
            cf_good   = from_cf   is not None and from_cf   > min_abs
            info_good = from_info is not None and from_info > min_abs

            # Both positive — prefer cash flow statement (more granular)
            if cf_good and info_good:
                return from_cf
            # Only CF is positive
            if cf_good:
                return from_cf
            # Only info is positive
            if info_good:
                return from_info
            # Neither is positive — fall back gracefully
            if from_cf is not None and abs(from_cf) > min_abs:
                return from_cf
            if from_info is not None and abs(from_info) > min_abs:
                return from_info
            return from_cf or from_info or 0

        fcf_final = best_val(fcf_info, fcf_cf)
        ocf_final = best_val(ocf_info, ocf_cf)
        logger.debug(f"{ticker}: fcf_info={fcf_info}, fcf_cf={fcf_cf} → final={fcf_final}")

        metrics = {
            # Identity
            "ticker": ticker.upper(),
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", ""),
            "description": info.get("longBusinessSummary", ""),
            "ceo": next((o.get("name","") for o in info.get("companyOfficers",[]) if "Chief Executive" in o.get("title","")), ""),
            "city": info.get("city", ""),
            "state": info.get("state", ""),
            "country": info.get("country", ""),
            "website": info.get("website", ""),
            "employees": safe(info.get("fullTimeEmployees"), None),
            "founded": info.get("founded", ""),
            # Price
            "current_price": safe(
                info.get("currentPrice") or
                info.get("regularMarketPrice") or
                info.get("previousClose"), None
            ),
            "market_cap": safe(info.get("marketCap"), None),
            "52w_high": safe(info.get("fiftyTwoWeekHigh")),
            "52w_low": safe(info.get("fiftyTwoWeekLow")),
            "beta": safe(info.get("beta"), 1.0),
            # Shares & debt
            "shares_outstanding": safe(info.get("sharesOutstanding"), 1),
            "total_debt": safe(info.get("totalDebt"), 0),
            "cash": safe(info.get("totalCash") or info.get("cash"), 0),
            "net_debt": safe(info.get("totalDebt"), 0) - safe(info.get("totalCash") or info.get("cash"), 0),
            # Income
            "revenue": safe(info.get("totalRevenue"), 0),
            "revenue_growth": safe(info.get("revenueGrowth"), 0),
            "gross_margin": safe(info.get("grossMargins"), 0),
            "operating_margin": safe(info.get("operatingMargins"), 0),
            "net_margin": safe(info.get("profitMargins"), 0),
            "ebitda": safe(info.get("ebitda"), 0),
            "operating_cashflow": ocf_final,
            "free_cashflow": fcf_final,
            "earnings_growth": safe(info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth"), 0),
            # Per share
            "eps": safe(info.get("trailingEps") or info.get("epsTrailingTwelveMonths"), 0),
            "eps_forward": safe(info.get("forwardEps"), 0),
            "book_value": safe(info.get("bookValue"), 0),
            "roe": safe(info.get("returnOnEquity"), 0),
            "roa": safe(info.get("returnOnAssets"), 0),
            "tangible_book_value": safe(info.get("tangibleBookValue"), 0),
            "dividend_rate": safe(info.get("dividendRate"), 0),
            "dividend_yield": safe(info.get("dividendYield"), 0),
            "payout_ratio": safe(info.get("payoutRatio"), 0),
            # Multiples
            "pe_trailing": safe(info.get("trailingPE"), 0),
            "pe_forward": safe(info.get("forwardPE"), 0),
            "ps_ratio": safe(info.get("priceToSalesTrailing12Months"), 0),
            "pb_ratio": safe(info.get("priceToBook"), 0),
            "ev_ebitda": safe(info.get("enterpriseToEbitda"), 0),
            "ev_revenue": safe(info.get("enterpriseToRevenue"), 0),
            "enterprise_value": safe(info.get("enterpriseValue"), 0),
            # Analyst
            "target_mean_price": safe(info.get("targetMeanPrice"), None),
            "target_low_price": safe(info.get("targetLowPrice"), None),
            "target_high_price": safe(info.get("targetHighPrice"), None),
            "analyst_rating": info.get("recommendationKey", "N/A"),
            "analyst_count": safe(info.get("numberOfAnalystOpinions"), 0),
            # Cache metadata
            "_from_cache": from_cache,
            "_cache_age_hours": age,
        }

        # Derive FCF per share
        if metrics["free_cashflow"] and metrics["shares_outstanding"]:
            metrics["fcf_per_share"] = metrics["free_cashflow"] / metrics["shares_outstanding"]
        else:
            metrics["fcf_per_share"] = 0

        return metrics

    def get_peers_data(self, peer_tickers: list) -> list:
        """Get key metrics for a list of peer tickers. Returns list of dicts."""
        results = []
        for t in peer_tickers:
            try:
                m = self.get_key_metrics(t)
                if m and m.get("current_price"):
                    results.append(m)
                time.sleep(0.3)  # polite delay
            except Exception as e:
                logger.warning(f"Failed to fetch peer {t}: {e}")
        return results

    def clear_cache(self, ticker: str = None):
        """Clear cache for a ticker, or all if ticker is None."""
        if not os.path.exists(CACHE_DIR):
            return
        for fn in os.listdir(CACHE_DIR):
            if ticker is None or ticker.upper() in fn:
                try:
                    os.remove(os.path.join(CACHE_DIR, fn))
                except Exception:
                    pass

    def get_cache_status(self) -> dict:
        """Returns info about the cache directory."""
        if not os.path.exists(CACHE_DIR):
            return {"exists": False, "files": 0, "size_mb": 0}
        files = os.listdir(CACHE_DIR)
        meta_files = [f for f in files if f.endswith(".meta")]
        total_size = sum(
            os.path.getsize(os.path.join(CACHE_DIR, f)) for f in files
        )
        entries = {}
        for mf in meta_files:
            try:
                with open(os.path.join(CACHE_DIR, mf)) as f:
                    meta = json.load(f)
                age_h = (time.time() - meta["ts"]) / 3600
                entries[mf] = {"age_hours": round(age_h, 1), "dt": meta.get("dt", "")}
            except Exception:
                pass
        return {
            "exists": True,
            "files": len(files),
            "entries": len(meta_files),
            "size_mb": round(total_size / 1e6, 2),
            "detail": entries,
        }

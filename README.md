# 🔬 IntrinsiQ | McMillin Analytics — Blended Stock Valuation Engine

> A professional, multi-method intrinsic value calculator built with Streamlit, yfinance, and modern financial modeling techniques.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📐 Valuation Methods

| Method | Description |
|--------|-------------|
| **DCF** | Two-stage Discounted Cash Flow with configurable WACC, Stage 1/2 growth, and Gordon Growth terminal value |
| **DDM** | Dividend Discount Model — Gordon Growth single-stage and two-stage variant |
| **EV/EBITDA Comps** | Comparable company analysis using median EV/EBITDA multiple from industry peers |
| **P/E Relative** | Relative valuation using median forward/trailing P/E across peer set |
| **Multiple Regression** | Ridge regression trained on 8 peer multiples (EV/EBITDA, P/E, P/S, P/B, margins, growth) |
| **Inverse DCF** | Solves for the FCF growth rate implied by today's market price |

All models are **weighted and blended** into a single intrinsic value. Unavailable models (e.g. DDM for non-dividend payers) are automatically excluded and weights renormalized.

---

## 🏗️ Architecture

```
valuation_app/
├── app.py                    # Main Streamlit app (UI, state, orchestration)
├── src/
│   ├── data_fetcher.py       # yfinance wrapper + tiered 24hr disk cache
│   ├── models.py             # All 6 valuation models + blended output
│   ├── visualizations.py     # Plotly charts (dark theme, Bloomberg aesthetic)
│   └── peers.py              # 300+ peer tickers across 40+ industries
├── .streamlit/
│   └── config.toml           # Dark theme config
├── requirements.txt
└── README.md
```

### Caching Strategy

1. **Streamlit `st.cache_data`** — in-memory, session-level, 1hr TTL
2. **Disk cache** (`.valuation_cache/`) — pickle + JSON metadata, 24hr TTL
3. **Stale fallback** — if yfinance rate-limits, serves stale cache with age indicator
4. **Retry with backoff** — 3 attempts with exponential backoff on failures

---

## 🚀 Getting Started

### Run Locally

```bash
git clone https://github.com/yourusername/intrinsiciq
cd intrinsiciq
pip install -r requirements.txt
streamlit run app.py
```

### Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo → select `app.py`
4. Click Deploy — live URL in ~2 minutes ✅

### Run on Kaggle

```python
# Install dependencies
!pip install streamlit yfinance scikit-learn scipy plotly -q

# Run models in notebook mode
import sys
sys.path.insert(0, '/kaggle/working/valuation_app')
from src.data_fetcher import DataFetcher
from src.models import dcf_valuation, compute_blended_value

fetcher = DataFetcher()
metrics = fetcher.get_key_metrics("AAPL")
result = dcf_valuation(
    base_fcf=metrics['free_cashflow'],
    wacc=0.09, growth_rate_stage1=0.12,
    growth_rate_stage2=0.07, terminal_growth=0.025,
    net_debt=metrics['net_debt'],
    shares_outstanding=metrics['shares_outstanding']
)
print(f"DCF Intrinsic Value: ${result['intrinsic_value']:.2f}")
```

---

## 🎛️ Features

- **Dynamic weights** — Adjust the blend of each model with sliders
- **Real-time assumptions** — WACC, growth rates, cost of equity, dividend growth
- **Auto peer discovery** — 300+ pre-mapped tickers across 40+ industries
- **Sensitivity heatmap** — Color-coded WACC × Growth Rate matrix
- **Inverse DCF** — What growth rate does the market imply?
- **Analyst consensus** — Target prices and rating overlay
- **Downloadable results** — Export peer tables, sensitivity matrices

---

## 📊 Screenshots

*Add screenshots after deployment*

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It does not constitute financial advice. Always do your own due diligence before making investment decisions.

---

## 📝 License

MIT — free to use, modify, and distribute.

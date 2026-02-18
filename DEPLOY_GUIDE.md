# 🚀 IntrinsiQ | McMillin Analytics — Deployment & Publishing Guide

---

## OPTION 1 — Streamlit Community Cloud (Recommended, Free)

The fastest path from code to shareable URL.

### Step 1: Push to GitHub

```bash
# Initialize repo (if not already)
cd valuation_app
git init
git add .
git commit -m "feat: initial IntrinsiQ valuation app"

# Create repo on GitHub (via web or CLI)
gh repo create intrinsiciq --public --source=. --remote=origin --push
# OR manual:
git remote add origin https://github.com/YOURUSERNAME/intrinsiciq.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to **https://share.streamlit.io**
2. Click **"New app"**
3. Connect your GitHub account
4. Select repo: `intrinsiciq`, branch: `main`, file: `app.py`
5. Click **"Deploy!"** — live in ~2 minutes

**Your URL:** `https://YOURNAME-intrinsiciq-app-HASH.streamlit.app`

### Step 3: Custom domain (optional, Streamlit Pro)
- Upgrade to Streamlit Teams/Pro for custom domain support
- Or use Cloudflare tunnel for free custom domains

---

## OPTION 2 — Render.com (Free Tier, Always On)

Better for production — no cold starts.

```bash
# Create render.yaml in project root
cat > render.yaml << 'EOF'
services:
  - type: web
    name: intrinsiciq
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
EOF
```

1. Push to GitHub
2. Create account at **render.com**
3. "New Web Service" → connect GitHub repo
4. Render detects `render.yaml` automatically
5. Free tier: https://intrinsiciq.onrender.com

---

## OPTION 3 — Railway.app

```bash
# Install Railway CLI
npm install -g @railway/cli
railway login
railway init
railway up
```

Cost: ~$5/mo for hobby tier (great for always-on)

---

## OPTION 4 — Docker + any VPS (DigitalOcean, AWS, GCP, Azure)

```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
```

```bash
docker build -t intrinsiciq .
docker run -p 8501:8501 intrinsiciq
# Access at http://localhost:8501
```

Deploy to DigitalOcean App Platform:
1. Push Dockerfile to GitHub
2. New App → GitHub repo → DigitalOcean detects Dockerfile
3. ~$5/mo for basic dyno

---

## GITHUB SETUP (Best Practices)

### .gitignore

```gitignore
# Cache (regenerated at runtime)
.valuation_cache/
__pycache__/
*.pyc
.env

# VS Code / IDE
.vscode/
.idea/

# macOS
.DS_Store

# Python
*.egg-info/
dist/
build/
venv/
.venv/
```

### GitHub Actions — Auto-deploy on push

```yaml
# .github/workflows/deploy.yml
name: Deploy to Streamlit Cloud
on:
  push:
    branches: [main]
jobs:
  notify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Trigger Streamlit Redeploy
        run: echo "Streamlit Cloud auto-deploys on push to main ✅"
```

---

## KAGGLE INTEGRATION

### Option A: Notebook version

1. Go to **kaggle.com/notebooks** → New Notebook
2. Upload your `src/` folder as a dataset:
   - Kaggle → Datasets → New Dataset → upload `valuation_app/` as zip
3. In notebook:

```python
# Cell 1: Install deps
!pip install yfinance scikit-learn scipy plotly -q

# Cell 2: Mount your dataset
import sys
sys.path.insert(0, '/kaggle/input/intrinsiciq-valuation-engine')

# Cell 3: Run analysis
from src.data_fetcher import DataFetcher
from src.models import *

fetcher = DataFetcher()

tickers = ["AAPL", "MSFT", "GOOGL", "NVDA"]
results = []

for t in tickers:
    m = fetcher.get_key_metrics(t)
    if not m.get("free_cashflow"): continue
    
    dcf_r = dcf_valuation(
        base_fcf=m['free_cashflow'], wacc=0.09,
        growth_rate_stage1=m.get('earnings_growth', 0.12),
        growth_rate_stage2=0.07, terminal_growth=0.025,
        net_debt=m['net_debt'], shares_outstanding=m['shares_outstanding']
    )
    results.append({
        "Ticker": t, "Price": m['current_price'],
        "DCF Value": dcf_r.get('intrinsic_value', 0),
        "Upside": ((dcf_r.get('intrinsic_value', 0) / m['current_price']) - 1) * 100
    })

import pandas as pd
df = pd.DataFrame(results)
print(df.to_string(index=False))
```

### Option B: Link Kaggle dataset to GitHub

```bash
# Install Kaggle CLI
pip install kaggle

# Set API key: ~/.kaggle/kaggle.json
# {"username": "you", "key": "YOUR_API_KEY"}

# Push dataset
kaggle datasets create -p valuation_app/ --dir-mode zip
```

---

## LINKEDIN POST TEMPLATE

```
🔬 Just shipped: IntrinsiQ by McMillin Analytics — a blended stock valuation app built in Python

What it does:
• 6 valuation methods: DCF (2-stage), DDM, EV/EBITDA comps, P/E relative, 
  Ridge regression on peer multiples, and Inverse DCF
• Weighted blend with real-time assumption tuning
• Auto-pulls comparable companies by sector/industry
• 24-hour data cache + rate-limit protection via yfinance
• Sensitivity heatmap (WACC × Growth)
• Inverse DCF shows what growth rate the market is pricing in

Tech stack: Python · Streamlit · yfinance · scikit-learn · Plotly · SciPy

🔗 Live app: [your-app.streamlit.app]
🐙 GitHub: [github.com/you/intrinsiciq]
📓 Kaggle: [kaggle.com/you/intrinsiciq]

For educational purposes only — not financial advice!

#Python #Finance #DataScience #Streamlit #Fintech #OpenSource #Investing
```

---

## COMPARISON OF DEPLOYMENT OPTIONS

| Platform | Cost | Cold Start | Custom Domain | Auto-Deploy | Best For |
|----------|------|------------|---------------|-------------|----------|
| Streamlit Cloud | Free | Yes (30s) | Pro only | ✅ Yes | Portfolio demos |
| Render.com | Free / $7mo | Free tier yes | ✅ Yes | ✅ Yes | Reliable demos |
| Railway | $5/mo | No | ✅ Yes | ✅ Yes | Always-on |
| DigitalOcean Apps | $5/mo | No | ✅ Yes | ✅ Yes | Production |
| AWS/GCP/Azure | $10-30/mo | No | ✅ Yes | Manual | Enterprise |
| Kaggle Notebooks | Free | N/A | N/A | N/A | Research/sharing |

**Recommendation:** Start with Streamlit Cloud (zero cost, zero config), upgrade to Render or Railway once you want always-on availability.

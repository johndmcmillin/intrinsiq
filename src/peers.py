"""
Peer company database organized by sector and industry.
Used for comparable company analysis.
"""

INDUSTRY_PEERS = {
    "Technology": {
        "Internet Content & Information": ["GOOGL", "META", "SNAP", "PINS", "IAC", "ZD"],
        "Software—Application": ["MSFT", "CRM", "NOW", "ADBE", "ORCL", "WDAY", "INTU", "HUBS", "SPLK"],
        "Software—Infrastructure": ["MSFT", "ORCL", "PANW", "FTNT", "CRWD", "ZS", "OKTA"],
        "Semiconductors": ["NVDA", "AMD", "INTC", "QCOM", "AVGO", "TSM", "MU", "AMAT", "LRCX", "KLAC"],
        "Consumer Electronics": ["AAPL", "SONY", "HPQ", "DELL", "LOGI"],
        "Computer Hardware": ["AAPL", "DELL", "HPQ", "NTAP", "STX", "WDC", "PSTG"],
        "Electronic Components": ["TXN", "ADI", "MCHP", "ON", "STM", "SWKS"],
        "Information Technology Services": ["ACN", "IBM", "INFY", "WIT", "CTSH", "EPAM"],
    },
    "Healthcare": {
        "Drug Manufacturers—General": ["JNJ", "PFE", "MRK", "ABBV", "BMY", "LLY", "AMGN", "GILD", "AZN"],
        "Drug Manufacturers—Specialty & Generic": ["AGN", "JAZZ", "PRGO", "HZN", "BHC"],
        "Biotechnology": ["BIIB", "REGN", "VRTX", "ILMN", "MRNA", "BNTX", "SGEN", "ALNY"],
        "Medical Devices": ["MDT", "ABT", "BSX", "SYK", "ZBH", "EW", "ISRG", "BDX"],
        "Health Information Services": ["UNH", "CI", "HUM", "CVS", "MCK", "CAH", "ABC"],
        "Diagnostics & Research": ["TMO", "DHR", "A", "BIO", "IDXX", "NEOG"],
    },
    "Financial Services": {
        "Banks—Diversified": ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC"],
        "Banks—Regional": ["KEY", "RF", "CFG", "HBAN", "MTB", "FITB", "ZION", "CMA"],
        "Insurance—Life": ["MET", "PRU", "AFL", "LNC", "GL", "PFG", "UNM"],
        "Insurance—P&C": ["TRV", "ALL", "CB", "PGR", "HIG", "WRB"],
        "Asset Management": ["BLK", "TROW", "IVZ", "AMG", "BEN", "EV", "APAM"],
        "Capital Markets": ["GS", "MS", "SCHW", "RJF", "SF", "LPLA", "IBKR"],
        "Credit Services": ["V", "MA", "AXP", "DFS", "SYF", "COF"],
        "Insurance—Diversified": ["BRK-B", "AIG", "MFC", "SFG"],
    },
    "Consumer Cyclical": {
        "Auto Manufacturers": ["TSLA", "F", "GM", "TM", "HMC", "STLA", "RIVN", "LCID"],
        "Auto Parts": ["APTV", "LKQ", "BWA", "LEA", "MGA", "DAN"],
        "Specialty Retail": ["HD", "LOW", "TGT", "TJX", "ROST", "BBY", "WSM"],
        "Internet Retail": ["AMZN", "ETSY", "EBAY", "W", "CHWY", "OSTK"],
        "Restaurants": ["MCD", "SBUX", "YUM", "QSR", "DPZ", "CMG", "SHAK", "TXRH"],
        "Hotels & Motels": ["MAR", "HLT", "IHG", "H", "WH", "RCL", "CCL", "NCLH"],
        "Apparel Retail": ["LULU", "NKE", "UAA", "ANF", "AEO", "GPS", "PVH", "RL"],
        "Home Improvement": ["HD", "LOW", "FLOOR", "TILE", "SFM"],
    },
    "Consumer Defensive": {
        "Household & Personal Products": ["PG", "CL", "KMB", "CHD", "CLX", "EL", "COTY"],
        "Beverages—Non-Alcoholic": ["KO", "PEP", "MNST", "CELH", "FIZZ", "COTT"],
        "Beverages—Alcoholic": ["BUD", "TAP", "SAM", "STZ", "DEO", "BF-B"],
        "Food Distribution": ["SYY", "PFGC", "USFD"],
        "Discount Stores": ["WMT", "COST", "DG", "DLTR", "BJ", "PRTY"],
        "Food—Major Diversified": ["KHC", "GIS", "K", "CPB", "MKC", "CAG", "SJM"],
        "Tobacco": ["MO", "PM", "BTI", "LO"],
    },
    "Energy": {
        "Oil & Gas Integrated": ["XOM", "CVX", "COP", "BP", "SHEL", "TTE"],
        "Oil & Gas E&P": ["PXD", "DVN", "MRO", "APA", "FANG", "OXY", "CRC", "SM"],
        "Oil & Gas Midstream": ["ET", "EPD", "MMP", "TRGP", "WMB", "KMI", "OKE"],
        "Oil & Gas Equipment & Services": ["SLB", "HAL", "BKR", "FTI", "RIG", "NOV"],
        "Oil & Gas Refining": ["PSX", "VLO", "MPC", "HFC", "DKL"],
    },
    "Industrials": {
        "Aerospace & Defense": ["BA", "RTX", "LMT", "NOC", "GD", "HII", "TXT", "HWM", "TDG"],
        "Specialty Industrial Machinery": ["CAT", "DE", "EMR", "ETN", "ITW", "PH", "ROK", "AME"],
        "Farm & Heavy Construction Machinery": ["DE", "CAT", "AGCO", "CNH", "PCAR", "TEX", "OSK"],
        "Airlines": ["DAL", "UAL", "AAL", "LUV", "JBLU", "ALK", "HA"],
        "Railroads": ["UNP", "CSX", "NSC", "CNI", "CP", "KSU"],
        "Trucking": ["ODFL", "SAIA", "XPO", "JBHT", "WERN", "KNX"],
        "Waste Management": ["WM", "RSG", "SRCL", "CWST", "ADSW"],
        "Engineering & Construction": ["URI", "PWR", "PRIM", "MTZ", "AECOM", "WSP"],
    },
    "Communication Services": {
        "Telecom Services": ["T", "VZ", "TMUS", "LUMN", "USM", "BCE"],
        "Entertainment": ["DIS", "NFLX", "PARA", "WBD", "FOXA", "LGF-A"],
        "Electronic Gaming & Multimedia": ["EA", "ATVI", "TTWO", "RBLX", "U"],
        "Publishing": ["NYT", "GCI", "MDP", "NWS", "NWSA"],
        "Broadcasting": ["CMCSA", "CHTR", "DISH", "CABO"],
    },
    "Utilities": {
        "Utilities—Regulated Electric": ["NEE", "DUK", "SO", "D", "EXC", "AEE", "WEC", "ES"],
        "Utilities—Regulated Gas": ["SRE", "ATO", "NI", "OGS", "SR", "SWX"],
        "Utilities—Renewable": ["NEE", "BEP", "CWEN", "AY", "CLNE"],
        "Utilities—Diversified": ["AWK", "AWR", "MSEX", "YORW"],
    },
    "Real Estate": {
        "REIT—Retail": ["SPG", "O", "NNN", "BRX", "KIM", "REG", "SKT"],
        "REIT—Residential": ["AMT", "PLD", "CCI", "EQIX", "PSA", "EXR", "CUBE", "NSA"],
        "REIT—Office": ["BXP", "VNO", "SLG", "HIW", "PDM", "OPI"],
        "REIT—Healthcare": ["WELL", "PEAK", "OHI", "SBRA", "LTC"],
        "REIT—Industrial": ["PLD", "DRE", "STAG", "FR", "EGP"],
    },
    "Consumer Staples": {
        "Household & Personal Products": ["PG", "CL", "KMB", "CHD", "CLX", "EL", "COTY"],
        "Beverages—Non-Alcoholic": ["KO", "PEP", "MNST", "CELH", "FIZZ"],
        "Beverages—Alcoholic": ["BUD", "TAP", "SAM", "STZ", "DEO", "BF-B"],
        "Food Distribution": ["SYY", "PFGC", "USFD"],
        "Discount Stores": ["WMT", "COST", "DG", "DLTR", "BJ"],
        "Food—Major Diversified": ["KHC", "GIS", "K", "CPB", "MKC", "CAG", "SJM"],
        "Tobacco": ["MO", "PM", "BTI", "LO"],
        "Beverages—Non-Alcoholic (broad)": ["KO", "PEP", "MNST", "CELH", "KDP", "FIZZ"],
    },
    "Basic Materials": {
        "Specialty Chemicals": ["DD", "DOW", "LYB", "EMN", "ALB", "LTHM", "KRTX"],
        "Gold": ["NEM", "GOLD", "KGC", "AEM", "WPM", "FNV", "RGLD"],
        "Steel": ["NUE", "STLD", "CLF", "X", "CMC", "SCHN"],
        "Copper": ["FCX", "SCCO", "TRQ", "HBM"],
        "Agricultural Inputs": ["MOS", "NTR", "CF", "ICL", "CTVA"],
    },
}


def _normalize(s: str) -> str:
    """Normalize industry string for fuzzy matching — strip dashes, hyphens, spaces."""
    return s.lower().replace("—", " ").replace("-", " ").replace("&", "and").strip()

def get_peers_for_ticker(ticker: str, info: dict, max_peers: int = 8) -> list:
    """
    Get peer companies for a given ticker based on sector/industry info from yfinance.
    Returns a list of ticker symbols.
    yfinance sector/industry naming is inconsistent — uses normalized fuzzy matching.
    """
    sector   = info.get("sector", "")
    industry = info.get("industry", "")
    ind_norm = _normalize(industry)

    # Try the sector directly, then try mapped aliases
    SECTOR_ALIASES = {
        "Consumer Defensive":    "Consumer Defensive",
        "Consumer Cyclical":     "Consumer Cyclical",
        "Financial Services":    "Financial Services",
        "Consumer Staples":      "Consumer Defensive",
        "Consumer Discretionary":"Consumer Cyclical",
    }
    sector_key = SECTOR_ALIASES.get(sector, sector)

    if sector_key in INDUSTRY_PEERS:
        sector_dict = INDUSTRY_PEERS[sector_key]

        # Fuzzy match industry string
        best_peers = None
        best_score = 0
        for ind_key, peers in sector_dict.items():
            key_norm = _normalize(ind_key)
            # Score: count matching words
            words_a = set(ind_norm.split())
            words_b = set(key_norm.split())
            score = len(words_a & words_b)
            if score > best_score:
                best_score = score
                best_peers = peers

        if best_peers and best_score > 0:
            return [p for p in best_peers if p.upper() != ticker.upper()][:max_peers]

        # Fall back to first industry in sector
        first_industry = list(sector_dict.values())[0]
        return [p for p in first_industry if p.upper() != ticker.upper()][:max_peers]

    # Generic fallback
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B"]


def get_all_sectors() -> list:
    return list(INDUSTRY_PEERS.keys())


def get_industries_for_sector(sector: str) -> list:
    return list(INDUSTRY_PEERS.get(sector, {}).keys())


def get_peers_manual(sector: str, industry: str) -> list:
    return INDUSTRY_PEERS.get(sector, {}).get(industry, [])

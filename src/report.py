"""
IntrinsiQ™ PDF Report Generator
McMillin Analytics — Equity Valuation Intelligence
"""

import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# ── Brand Colors — Dark Navy + Gold ───────────────────────────────────────────
NAVY        = colors.HexColor("#0d1b2a")   # deep navy background
NAVY_MID    = colors.HexColor("#162032")   # slightly lighter navy for panels
NAVY_LIGHT  = colors.HexColor("#1e2d40")   # card/section backgrounds
GOLD        = colors.HexColor("#c9a84c")   # primary gold accent
GOLD_LIGHT  = colors.HexColor("#e8c96a")   # lighter gold for headings
GOLD_PALE   = colors.HexColor("#f5e6b8")   # very pale gold for dark bg text
GREEN       = colors.HexColor("#2ecc71")   # undervalued signal
RED         = colors.HexColor("#e74c3c")   # overvalued signal
ORANGE      = colors.HexColor("#e67e22")   # slightly overvalued
BLUE        = colors.HexColor("#3b9dd4")   # neutral accent
SLATE       = colors.HexColor("#7f8c9a")   # muted text on light bg
LIGHT_SLATE = colors.HexColor("#a0adb8")   # secondary text
WHITE       = colors.white
OFF_WHITE   = colors.HexColor("#f4f1ea")   # warm off-white for body bg
LIGHT_GRAY  = colors.HexColor("#f0ece3")   # panel bg on light sections
MID_GRAY    = colors.HexColor("#d5cfc4")   # dividers on light bg
BORDER      = colors.HexColor("#2a3f55")   # borders on dark bg

# Aliases kept for compatibility
PURPLE      = GOLD
LIGHT_PURPLE = GOLD_LIGHT
DARK_BLUE   = NAVY_MID
TEAL        = BLUE

# ── Styles ────────────────────────────────────────────────────────────────────
def make_styles():
    return {
        "letterhead_title": ParagraphStyle(
            "letterhead_title",
            fontName="Helvetica-Bold",
            fontSize=22,
            textColor=GOLD_LIGHT,
            spaceAfter=2,
            letterSpacing=3,
        ),
        "letterhead_sub": ParagraphStyle(
            "letterhead_sub",
            fontName="Helvetica",
            fontSize=8,
            textColor=GOLD_PALE,
            spaceAfter=2,
            letterSpacing=2,
        ),
        "section_header": ParagraphStyle(
            "section_header",
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=GOLD,
            spaceBefore=12,
            spaceAfter=4,
            letterSpacing=1.5,
        ),
        "ticker": ParagraphStyle(
            "ticker",
            fontName="Helvetica-Bold",
            fontSize=28,
            textColor=LIGHT_PURPLE,
            spaceAfter=2,
        ),
        "company_name": ParagraphStyle(
            "company_name",
            fontName="Helvetica",
            fontSize=13,
            textColor=LIGHT_SLATE,
            spaceAfter=4,
        ),
        "valuation_label": ParagraphStyle(
            "valuation_label",
            fontName="Helvetica",
            fontSize=8,
            textColor=SLATE,
            spaceAfter=1,
        ),
        "valuation_value": ParagraphStyle(
            "valuation_value",
            fontName="Helvetica-Bold",
            fontSize=26,
            textColor=GREEN,
            spaceAfter=2,
        ),
        "disclaimer": ParagraphStyle(
            "disclaimer",
            fontName="Helvetica",
            fontSize=7,
            textColor=SLATE,
            leading=10,
        ),
        "normal": ParagraphStyle(
            "normal",
            fontName="Helvetica",
            fontSize=8.5,
            textColor=NAVY,
            leading=12,
        ),
        "label": ParagraphStyle(
            "label",
            fontName="Helvetica",
            fontSize=7.5,
            textColor=SLATE,
        ),
        "value": ParagraphStyle(
            "value",
            fontName="Helvetica-Bold",
            fontSize=8.5,
            textColor=NAVY,
        ),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt(v, prefix="$", decimals=2):
    if v is None:
        return "N/A"
    try:
        return f"{prefix}{float(v):,.{decimals}f}"
    except Exception:
        return "N/A"

def fmt_pct(v, decimals=1):
    if v is None:
        return "N/A"
    try:
        v = float(v)
        sign = "+" if v > 0 else ""
        return f"{sign}{v:.{decimals}f}%"
    except Exception:
        return "N/A"

def valuation_label(pct):
    if pct is None:
        return ("N/A", SLATE)
    if pct > 40:   return ("Significantly Overvalued",  RED)
    if pct > 15:   return ("Overvalued",                colors.HexColor("#ea580c"))
    if pct > 5:    return ("Slightly Overvalued",        ORANGE)
    if pct > -5:   return ("Fairly Valued",              GREEN)
    if pct > -15:  return ("Slightly Undervalued",       colors.HexColor("#059669"))
    if pct > -40:  return ("Undervalued",                BLUE)
    return ("Significantly Undervalued",                 PURPLE)

def section_rule(story):
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GRAY))
    story.append(Spacer(1, 4))

def dark_rule(story):
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1, color=GOLD))
    story.append(Spacer(1, 6))


# ── Section builders ──────────────────────────────────────────────────────────

def build_letterhead(story, styles, generated_at):
    """Top letterhead with brand identity."""
    header_data = [[
        Paragraph("IntrinsiQ™", styles["letterhead_title"]),
        Paragraph(
            f"Equity Valuation Report<br/>"
            f"<font color='#a0adb8'>Generated {generated_at}</font>",
            ParagraphStyle("rh", fontName="Helvetica", fontSize=8,
                           textColor=GOLD_PALE, alignment=TA_RIGHT)
        )
    ]]
    header_table = Table(header_data, colWidths=[3.5*inch, 4*inch])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("BACKGROUND", (0,0), (-1,-1), NAVY),
        ("TOPPADDING", (0,0), (-1,-1), 12),
        ("BOTTOMPADDING", (0,0), (-1,-1), 12),
        ("LEFTPADDING", (0,0), (0,-1), 16),
        ("RIGHTPADDING", (-1,0), (-1,-1), 16),
        ("ROUNDEDCORNERS", [6, 6, 6, 6]),
    ]))
    story.append(header_table)
    story.append(Paragraph(
        "BY MCMILLIN ANALYTICS  ·  EQUITY VALUATION INTELLIGENCE",
        ParagraphStyle("sub", fontName="Helvetica", fontSize=7,
                       textColor=GOLD, alignment=TA_CENTER,
                       spaceBefore=4, spaceAfter=10, letterSpacing=1)
    ))


def build_company_header(story, styles, m, price, bv, upside):
    """Company identity + price vs intrinsic value block."""
    val_text, val_color = valuation_label(upside)

    left = [
        Paragraph(m.get("name", ""), ParagraphStyle(
            "co_name", fontName="Helvetica-Bold", fontSize=17,
            textColor=GOLD_LIGHT, spaceBefore=6, spaceAfter=5, leading=20,
        )),
        Paragraph(
            f"{m.get('ticker','')}  ·  {m.get('sector','—')}",
            ParagraphStyle("co_sub", fontName="Helvetica", fontSize=9, textColor=LIGHT_SLATE, spaceBefore=0, spaceAfter=4)
        ),
        Paragraph(
            f"{m.get('industry','—')} &nbsp;·&nbsp; Mkt Cap: {m.get('market_cap_fmt', '—')}",
            ParagraphStyle("meta", fontName="Helvetica", fontSize=8, textColor=SLATE)
        ),
    ]

    right_rows = [
        ["Market Price",   fmt(price)],
        ["Intrinsic Value", fmt(bv)],
        ["Signal",          f"{val_text}  ({fmt_pct(upside)})"],
    ]

    right_table = Table(right_rows, colWidths=[1.3*inch, 1.8*inch])
    right_table.setStyle(TableStyle([
        ("FONTNAME",  (0,0), (0,-1), "Helvetica"),
        ("FONTNAME",  (1,0), (1,-1), "Helvetica-Bold"),
        ("FONTSIZE",  (0,0), (-1,-1), 9),
        ("TEXTCOLOR", (0,0), (0,-1), LIGHT_SLATE),
        ("TEXTCOLOR", (1,0), (1,0), OFF_WHITE),
        ("TEXTCOLOR", (1,1), (1,1), GOLD_LIGHT),
        ("TEXTCOLOR", (1,2), (1,2), val_color),
        ("ALIGN",     (1,0), (1,-1), "RIGHT"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))

    data = [[left, right_table]]
    tbl = Table(data, colWidths=[4*inch, 3.5*inch])
    tbl.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("BACKGROUND", (0,0), (-1,-1), NAVY_MID),
        ("TOPPADDING", (0,0), (-1,-1), 18),
        ("BOTTOMPADDING", (0,0), (-1,-1), 18),
        ("LEFTPADDING", (0,0), (0,-1), 16),
        ("RIGHTPADDING", (-1,0), (-1,-1), 16),
        ("ROUNDEDCORNERS", [6,6,6,6]),
        ("BOX", (0,0), (-1,-1), 1, GOLD),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 10))


def build_key_metrics(story, styles, m):
    """Key financial metrics grid."""
    story.append(Paragraph("KEY FINANCIAL METRICS", styles["section_header"]))
    section_rule(story)

    def safe(v, fmt_fn=None, suffix=""):
        if v is None or v == 0:
            return "—"
        return (fmt_fn(v) if fmt_fn else str(v)) + suffix

    metrics = [
        ("P/E Ratio",       safe(m.get("pe_ratio"),      lambda x: f"{x:.1f}x")),
        ("Forward P/E",     safe(m.get("forward_pe"),    lambda x: f"{x:.1f}x")),
        ("EPS (TTM)",       safe(m.get("eps"),           lambda x: f"${x:.2f}")),
        ("Price/Book",      safe(m.get("price_to_book"), lambda x: f"{x:.1f}x")),
        ("Div. Yield",      safe(m.get("dividend_yield"),lambda x: f"{x*100:.2f}%" if x < 1 else f"{x:.2f}%")),
        ("Beta",            safe(m.get("beta"),          lambda x: f"{x:.2f}")),
        ("Revenue (TTM)",   safe(m.get("revenue"),       lambda x: f"${x/1e9:.1f}B" if x >= 1e9 else f"${x/1e6:.0f}M")),
        ("FCF",             safe(m.get("free_cashflow"), lambda x: f"${x/1e9:.1f}B" if abs(x) >= 1e9 else f"${x/1e6:.0f}M")),
        ("D/E Ratio",       safe(m.get("debt_to_equity"),lambda x: f"{x:.2f}x")),
        ("ROE",             safe(m.get("roe"),           lambda x: f"{x*100:.1f}%")),
        ("Profit Margin",   safe(m.get("profit_margin"), lambda x: f"{x*100:.1f}%")),
        ("52W Range",       f"{safe(m.get('fifty_two_week_low'), lambda x: f'${x:.0f}')} – {safe(m.get('fifty_two_week_high'), lambda x: f'${x:.0f}')}"),
    ]

    # 3-column grid
    rows = []
    for i in range(0, len(metrics), 3):
        row = []
        for label, val in metrics[i:i+3]:
            row.append(Paragraph(label, styles["label"]))
            row.append(Paragraph(val, styles["value"]))
        # Pad if needed
        while len(row) < 6:
            row.append(Paragraph("", styles["label"]))
            row.append(Paragraph("", styles["value"]))
        rows.append(row)

    col_w = [1.2*inch, 0.9*inch, 1.2*inch, 0.9*inch, 1.2*inch, 0.9*inch]
    tbl = Table(rows, colWidths=col_w)
    tbl.setStyle(TableStyle([
        ("FONTSIZE",  (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("LINEBELOW", (0,0), (-1,-2), 0.3, MID_GRAY),
        ("BACKGROUND", (0,0), (-1,-1), WHITE),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 10))


def build_valuation_bridge(story, styles, mr, weights, price):
    """Valuation bridge — bar chart approximated as a table."""
    story.append(Paragraph("VALUATION BRIDGE", styles["section_header"]))
    section_rule(story)

    MODEL_LABELS = {
        "dcf":        "DCF (Discounted Cash Flow)",
        "ddm":        "DDM (Dividend Discount)",
        "ev_ebitda":  "EV/EBITDA Multiple",
        "pe_relative":"P/E Relative Valuation",
        "regression": "Regression (Historical)",
    }
    weight_keys = {
        "dcf":        "dcf",
        "ddm":        "ddm",
        "ev_ebitda":  "ev_ebitda",
        "pe_relative":"pe_relative",
        "regression": "regression",
    }

    rows = [["Model", "Weight", "Value", "vs Market", "Bar"]]
    max_val = price * 1.6 if price else 1

    for key, label in MODEL_LABELS.items():
        w = weights.get(weight_keys[key], 0)
        if w == 0:
            continue
        r = mr.get(key) or mr.get(key.replace("_", ""))
        iv = None
        if r and not (isinstance(r, dict) and r.get("error")):
            iv = r.get("intrinsic_value") if isinstance(r, dict) else None

        val_str  = fmt(iv) if iv else "N/A"
        try:
            iv_f    = float(iv) if iv else None
            price_f = float(price) if price else None
            diff_pct = ((price_f - iv_f) / iv_f * 100) if (iv_f and price_f) else None
        except Exception:
            diff_pct = None
        diff_str = fmt_pct(diff_pct) if diff_pct is not None else "—"
        diff_color = RED if (diff_pct and diff_pct > 5) else (GREEN if (diff_pct and diff_pct < -5) else SLATE)

        # Bar approximation — filled blocks
        bar_fill = min(int((iv / max_val) * 20), 20) if iv else 0
        bar = "█" * bar_fill

        rows.append([
            Paragraph(label, ParagraphStyle("bl", fontName="Helvetica", fontSize=8, textColor=colors.HexColor("#1e2d40"))),
            Paragraph(f"{int(w*100)}%", ParagraphStyle("bw", fontName="Helvetica-Bold", fontSize=8, textColor=GOLD, alignment=TA_CENTER)),
            Paragraph(val_str, ParagraphStyle("bv", fontName="Helvetica-Bold", fontSize=8, textColor=GOLD_LIGHT, alignment=TA_RIGHT)),
            Paragraph(diff_str, ParagraphStyle("bd", fontName="Helvetica-Bold", fontSize=8, textColor=diff_color, alignment=TA_RIGHT)),
            Paragraph(bar, ParagraphStyle("bb", fontName="Helvetica", fontSize=7, textColor=GOLD)),
        ])

    col_w = [2.5*inch, 0.6*inch, 0.9*inch, 0.9*inch, 1.6*inch]
    tbl = Table(rows, colWidths=col_w)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  NAVY),
        ("TEXTCOLOR",     (0,0), (-1,0),  GOLD_PALE),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,0),  7.5),
        ("ALIGN",         (1,0), (-1,0),  "CENTER"),
        ("BOTTOMPADDING", (0,0), (-1,0),  6),
        ("TOPPADDING",    (0,0), (-1,0),  6),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [OFF_WHITE, LIGHT_GRAY]),
        ("LINEBELOW",     (0,0), (-1,-1), 0.3, MID_GRAY),
        ("BOTTOMPADDING", (0,1), (-1,-1), 5),
        ("TOPPADDING",    (0,1), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("RIGHTPADDING",  (0,0), (-1,-1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 10))


def build_model_weights(story, styles, weights, bv, price, upside):
    """Model weights summary + blended result."""
    story.append(Paragraph("MODEL WEIGHTS & BLENDED VALUATION", styles["section_header"]))
    section_rule(story)

    val_text, val_color = valuation_label(upside)
    MODEL_NAMES = {
        "dcf": "DCF", "ddm": "DDM", "ev_ebitda": "EV/EBITDA",
        "pe_relative": "P/E Relative", "regression": "Regression"
    }

    # Weight pills row
    pill_data = [[
        Paragraph(
            f"<b>{MODEL_NAMES.get(k,'')}</b><br/>{int(v*100)}%",
            ParagraphStyle("pill", fontName="Helvetica", fontSize=8,
                           textColor=WHITE if v > 0 else SLATE,
                           alignment=TA_CENTER, leading=12)
        )
        for k, v in weights.items()
    ]]
    pill_tbl = Table(pill_data, colWidths=[1.5*inch]*5)
    pill_styles = []
    for i, (k, v) in enumerate(weights.items()):
        bg = NAVY if v > 0.2 else (NAVY_LIGHT if v > 0.1 else (SLATE if v > 0 else LIGHT_GRAY))
        pill_styles.append(("BACKGROUND", (i,0), (i,0), bg))
        pill_styles.append(("TEXTCOLOR",  (i,0), (i,0), GOLD_LIGHT if v > 0 else SLATE))
    pill_tbl.setStyle(TableStyle([
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("ROUNDEDCORNERS", [4,4,4,4]),
        *pill_styles,
    ]))
    story.append(pill_tbl)
    story.append(Spacer(1, 8))

    # Blended result
    result_data = [[
        Paragraph("Blended Intrinsic Value", ParagraphStyle("rl", fontName="Helvetica", fontSize=9, textColor=SLATE)),
        Paragraph(fmt(bv), ParagraphStyle("rv", fontName="Helvetica-Bold", fontSize=16, textColor=GOLD_LIGHT, alignment=TA_RIGHT)),
        Paragraph(fmt(price), ParagraphStyle("rp", fontName="Helvetica-Bold", fontSize=16, textColor=NAVY, alignment=TA_RIGHT)),
        Paragraph(f"{val_text}\n{fmt_pct(upside)}", ParagraphStyle("rs", fontName="Helvetica-Bold", fontSize=9, textColor=val_color, alignment=TA_RIGHT)),
    ]]
    result_hdr = [["", "Intrinsic Value", "Market Price", "Signal"]]
    result_tbl = Table(result_hdr + result_data, colWidths=[2.2*inch, 1.5*inch, 1.5*inch, 2.3*inch])
    result_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  NAVY),
        ("TEXTCOLOR",     (0,0), (-1,0),  GOLD_PALE),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,0),  7.5),
        ("ALIGN",         (1,0), (-1,-1), "RIGHT"),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("BACKGROUND",    (0,1), (-1,1),  OFF_WHITE),
        ("LINEBELOW",     (0,0), (-1,-1), 0.5, MID_GRAY),
    ]))
    story.append(result_tbl)
    story.append(Spacer(1, 10))


def build_sensitivity_table(story, styles, sens, price):
    """DCF sensitivity — WACC vs terminal growth rate."""
    if sens is None or sens.empty:
        return

    story.append(Paragraph("DCF SENSITIVITY ANALYSIS  (WACC × Terminal Growth)", styles["section_header"]))
    section_rule(story)
    story.append(Paragraph(
        "Intrinsic value estimates across WACC and terminal growth rate assumptions. "
        "Green = undervalued vs market price. Red = overvalued.",
        ParagraphStyle("note", fontName="Helvetica", fontSize=7.5, textColor=SLATE, spaceAfter=6)
    ))

    cols = list(sens.columns)
    idx  = list(sens.index)

    # Header row
    header = [Paragraph("WACC \\ g", ParagraphStyle("sh", fontName="Helvetica-Bold", fontSize=7.5,
                         textColor=WHITE, alignment=TA_CENTER))]
    for c in cols:
        header.append(Paragraph(f"g={c:.1%}", ParagraphStyle("sh2", fontName="Helvetica-Bold",
                                fontSize=7.5, textColor=WHITE, alignment=TA_CENTER)))
    rows = [header]

    for wacc_val in idx:
        row = [Paragraph(f"r={wacc_val:.1%}", ParagraphStyle("si", fontName="Helvetica-Bold",
                         fontSize=7.5, textColor=WHITE, alignment=TA_CENTER))]
        for g_val in cols:
            v = sens.loc[wacc_val, g_val]
            cell_color = GREEN if (v and price and v > price) else (RED if (v and price and v < price * 0.85) else ORANGE)
            row.append(Paragraph(
                fmt(v) if v else "—",
                ParagraphStyle("sv", fontName="Helvetica-Bold", fontSize=7.5,
                               textColor=cell_color, alignment=TA_CENTER)
            ))
        rows.append(row)

    n_cols = len(cols) + 1
    col_w = [0.85*inch] + [6.65*inch / len(cols)] * len(cols)
    tbl = Table(rows, colWidths=col_w)
    style_cmds = [
        ("BACKGROUND",    (0,0), (-1,0),  NAVY),
        ("BACKGROUND",    (0,0), (0,-1),  NAVY_LIGHT),
        ("TEXTCOLOR",     (0,0), (-1,0),  GOLD_PALE),
        ("TEXTCOLOR",     (0,0), (0,-1),  GOLD_PALE),
        ("ROWBACKGROUNDS",(1,1), (-1,-1), [OFF_WHITE, LIGHT_GRAY]),
        ("LINEBELOW",     (0,0), (-1,-1), 0.3, MID_GRAY),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 4),
        ("RIGHTPADDING",  (0,0), (-1,-1), 4),
    ]
    tbl.setStyle(TableStyle(style_cmds))
    story.append(tbl)
    story.append(Spacer(1, 10))


def build_disclaimer(story, styles, generated_at):
    """Footer disclaimer."""
    dark_rule(story)
    story.append(Paragraph(
        f"<b>IMPORTANT DISCLAIMER</b> — IntrinsiQ™ by McMillin Analytics is provided for "
        f"<b>educational and informational purposes only</b>. Nothing presented in this report "
        f"constitutes investment advice, a recommendation, or a solicitation to buy or sell any "
        f"securities. All valuations are based on simplified models using estimated inputs and "
        f"should not be relied upon for actual investment decisions. Past performance is not "
        f"indicative of future results. Investing involves risk, including the possible loss of "
        f"principal. Always consult a qualified financial advisor before making investment decisions. "
        f"McMillin Analytics is not a registered investment advisor, broker-dealer, or financial "
        f"planner. Report generated {generated_at}. © 2026 McMillin Analytics. All rights reserved.",
        styles["disclaimer"]
    ))


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_pdf_report(m: dict, mr: dict, weights: dict, bv, upside, price, sens=None) -> bytes:
    """
    Generate a branded PDF valuation report and return as bytes.

    Args:
        m:       metrics dict from data_fetcher
        mr:      model results dict (dcf, ddm, ev_ebitda, pe_relative, regression)
        weights: dict of model weights (0-1 scale)
        bv:      blended intrinsic value
        upside:  % difference (price - bv) / bv * 100
        price:   current market price
        sens:    sensitivity DataFrame (optional)

    Returns:
        PDF as bytes for st.download_button
    """
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=0.65*inch,
        rightMargin=0.65*inch,
        topMargin=0.6*inch,
        bottomMargin=0.6*inch,
    )
    styles = make_styles()
    story  = []
    now    = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    # Coerce all numeric inputs — guard against strings slipping through
    try: bv    = float(bv)    if bv    is not None else None
    except Exception: bv = None
    try: price = float(price) if price is not None else None
    except Exception: price = None
    try: upside = float(upside) if upside is not None else None
    except Exception: upside = None

    # Add market cap formatted string to m for convenience
    mc = m.get("market_cap")
    if mc:
        if mc >= 1e12:   m["market_cap_fmt"] = f"${mc/1e12:.2f}T"
        elif mc >= 1e9:  m["market_cap_fmt"] = f"${mc/1e9:.1f}B"
        else:            m["market_cap_fmt"] = f"${mc/1e6:.0f}M"
    else:
        m["market_cap_fmt"] = "—"

    build_letterhead(story, styles, now)
    story.append(Spacer(1, 14))
    build_company_header(story, styles, m, price, bv, upside)
    build_key_metrics(story, styles, m)
    build_valuation_bridge(story, styles, mr, weights, price)
    build_model_weights(story, styles, weights, bv, price, upside)
    build_sensitivity_table(story, styles, sens, price)
    build_disclaimer(story, styles, now)

    doc.build(story)
    buf.seek(0)
    return buf.read()

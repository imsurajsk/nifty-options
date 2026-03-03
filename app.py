#!/usr/bin/env python3
"""
Nifty Options Prediction System — Streamlit Web App
====================================================
Run:  bash run_web.sh
      OR:  .venv/bin/streamlit run app.py
Open: http://localhost:8501
"""

import csv
import logging
import os
import sys
from datetime import datetime, timezone, timedelta

# ── Make project root importable (same as main.py) ────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Suppress noisy third-party loggers ────────────────────────────────────
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s — %(message)s")
for _noisy in ("yfinance", "urllib3", "peewee", "asyncio"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

import streamlit as st

from config import DEFAULT_CAPITAL, HISTORICAL_DAYS, STRATEGY

IST = timezone(timedelta(hours=5, minutes=30))

# ── Page config — must be the FIRST Streamlit call ────────────────────────
st.set_page_config(
    page_title="Nifty Options Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# Helper colour/formatting functions
# ═══════════════════════════════════════════════════════════════════════════

def _qcol(q: int) -> str:
    if q >= 70: return "green"
    if q >= 55: return "blue"
    if q >= 40: return "orange"
    return "red"

def _dir_label(direction: str) -> str:
    return direction.replace("_", " ")

def _dte(expiry) -> int:
    if expiry is None:
        return 0
    try:
        return (expiry.date() - datetime.now(IST).date()).days
    except Exception:
        return 0

# ═══════════════════════════════════════════════════════════════════════════
# Core data pipeline — mirrors main.py exactly
# ═══════════════════════════════════════════════════════════════════════════

def run_analysis(capital: float, strategy: str) -> dict:
    """
    Execute the full 8-step pipeline.
    Imports are inside the function to match main.py's pattern and to avoid
    Streamlit re-run side effects on top-level imports.

    strategy: "SELL_PUTS" | "BUY_PUT" | "BUY_OPTIONS"
    """
    from src.data_fetcher  import DataFetcher
    from src.technical     import TechnicalAnalysis
    from src.options       import OptionsAnalysis
    from src.news_analyzer import NewsAnalyzer
    from src.signals       import SignalGenerator
    from src.recommender   import OptionRecommender

    import time

    results: dict = {}
    fetcher = DataFetcher()

    # 1 — Spot price
    spot = fetcher.get_nifty_spot()
    if spot <= 0:
        time.sleep(3)
        spot = fetcher.get_nifty_spot()
    results["spot"] = spot

    # 2 — India VIX
    results["vix"] = fetcher.get_india_vix()

    # 3 — Historical data + technical analysis
    hist_df   = fetcher.get_historical_data(days=HISTORICAL_DAYS)
    ta        = TechnicalAnalysis(hist_df)
    tech_data = ta.get_current_values()
    if spot <= 0:
        spot = tech_data.get("close", 0)
        results["spot"] = spot
    results["tech_data"]      = tech_data
    results["hist_n_days"]    = len(hist_df) if not hist_df.empty else 0
    results["recent_candles"] = ta.get_recent_candles(5)
    # Keep last 90 days of annotated OHLCV+indicators for the live chart
    results["hist_df"]        = ta.df.tail(90).copy() if not hist_df.empty else None

    # 4 — Options chain
    raw_opts     = fetcher.get_options_chain()
    options_df   = fetcher.parse_options_chain(raw_opts)
    expiry_dates = fetcher.get_expiry_dates(raw_opts)
    results["options_df"]    = options_df
    results["expiry_dates"]  = expiry_dates
    results["options_empty"] = options_df.empty

    # 5 — Options analysis
    oa              = OptionsAnalysis(options_df, spot)
    options_summary = oa.get_summary()
    # Inject days-to-nearest-expiry so SignalGenerator can scale max pain strength
    if expiry_dates:
        _dte = (expiry_dates[0].date() - datetime.now(IST).date()).days
        options_summary["days_to_expiry"] = max(0, _dte)
    else:
        options_summary["days_to_expiry"] = 7   # safe default (mid-week assumption)
    results["options_summary"] = options_summary

    # 6 — News & Global markets
    na        = NewsAnalyzer()
    na.run()
    news_data = na.get_summary()
    results["news_data"] = news_data

    # 7 — Signal generation
    sg          = SignalGenerator(tech_data, options_summary, results["vix"], news=news_data)
    signal_data = sg.get_full_analysis()
    results["signal_data"] = signal_data

    # 8 — Recommendation (quality score depends on strategy)
    q_score = (
        signal_data.get("sell_quality_score", 50)
        if strategy == "SELL_PUTS"
        else signal_data.get("quality_score", 50)
    )
    recommender = OptionRecommender(
        direction       = signal_data["direction"],
        confidence      = signal_data["confidence"],
        score           = signal_data["score"],
        quality_score   = q_score,
        spot_price      = spot,
        options_df      = options_df,
        options_summary = options_summary,
        expiry_dates    = expiry_dates,
        vix             = results["vix"],
        capital         = capital,
        strategy        = strategy,
    )
    results["recommendation"] = recommender.get_recommendation()
    results["strategy"]       = strategy
    results["timestamp"]      = datetime.now(IST).replace(tzinfo=None)

    # Log to CSV
    _save_log(results)

    return results


def _save_log(results: dict):
    """Append prediction to logs/predictions.csv (same as ReportGenerator.save_log)."""
    try:
        rec    = results.get("recommendation", {})
        signal = results.get("signal_data", {})
        opts   = results.get("options_summary", {})
        tech   = results.get("tech_data", {})
        news   = results.get("news_data", {})
        ts     = results.get("timestamp", datetime.now())
        path   = "logs/predictions.csv"
        os.makedirs("logs", exist_ok=True)
        row = {
            "timestamp":   ts.isoformat(),
            "spot":        results.get("spot", 0),
            "vix":         results.get("vix",  0),
            "direction":   signal.get("direction",  ""),
            "confidence":  signal.get("confidence", ""),
            "score":       signal.get("score",       0),
            "action":      rec.get("action",      ""),
            "strike":      rec.get("strike",       0),
            "option_type": rec.get("option_type",  "PE"),
            "expiry":      str(rec.get("expiry",   "")),
            "ltp":         rec.get("ltp",           0),
            "stop_loss":   rec.get("sl_buy_at",
                           rec.get("targets", {}).get("stop_loss", 0)),
            "target":      rec.get("target_buy_at",
                           rec.get("targets", {}).get("target_1",  0)),
            "pcr":         opts.get("pcr",      0),
            "max_pain":    opts.get("max_pain", 0),
            "rsi":         tech.get("rsi",       0),
            "event_score": news.get("event_score", 0),
            "major_event": news.get("has_major_event", False),
        }
        exists = os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if not exists:
                w.writeheader()
            w.writerow(row)
    except Exception as exc:
        logging.getLogger(__name__).warning("Log save failed: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════
# Render helpers
# ═══════════════════════════════════════════════════════════════════════════

def _render_sell_put(rec: dict, signal: dict):
    strike  = rec.get("strike", 0)
    expiry  = rec.get("expiry")
    ltp     = rec.get("ltp", 0)
    data_ok = rec.get("data_available", False)

    expiry_str = expiry.strftime("%d %b %Y") if expiry else "N/A"
    dte        = _dte(expiry)
    dte_str    = f"  ({dte} days to expiry)" if dte > 0 else ""

    st.success(f"⚡  SELL  NIFTY  {strike:,}  PE  —  {expiry_str}{dte_str}")

    if not data_ok:
        st.info(
            "Live prices unavailable (NSE market may be closed).  \n"
            "When market opens:  \n"
            "- **Sell** this PE at market price  \n"
            "- **Stop Loss** = buy back at 1.5× your sold price  \n"
            "- **Target** = buy back at 30% of your sold price (keep 70%)"
        )
        return

    sell_at    = rec.get("sell_at",          ltp)
    sl_buy_at  = rec.get("sl_buy_at",          0)
    target_buy = rec.get("target_buy_at",       0)
    spot_alert = rec.get("spot_alert",    strike)
    prem_coll  = rec.get("premium_collected",   0)
    max_profit = rec.get("max_profit",          0)
    max_loss   = rec.get("max_loss",            0)
    lots       = rec.get("recommended_lots",    1)
    margin     = rec.get("total_margin",        0)
    margin_lot = rec.get("margin_per_lot",      0)

    sl_pct  = round((sl_buy_at  - sell_at) / sell_at * 100) if sell_at else 0
    tgt_pct = round((sell_at - target_buy) / sell_at * 100) if sell_at else 0
    rr      = round(max_profit / max_loss, 1) if max_loss else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Sell At (collect premium)", f"₹ {sell_at:.1f}")
    with c2:
        st.metric("Stop Loss — Buy Back If", f"₹ {sl_buy_at:.1f}",
                  delta=f"+{sl_pct}% above sell price  EXIT if reaches here",
                  delta_color="inverse")
    with c3:
        st.metric("Target — Buy Back At", f"₹ {target_buy:.1f}",
                  delta=f"−{tgt_pct}% below sell price  TAKE PROFIT here")

    st.caption(
        f"⚠️  Spot Alert: If Nifty falls to **₹{spot_alert:,}** (your sold strike) — "
        "monitor closely and consider exiting early."
    )

    st.markdown("---")
    c4, c5, c6, c7 = st.columns(4)
    with c4:
        st.metric("Lots to Sell", f"{lots} lot{'s' if lots > 1 else ''}")
    with c5:
        st.metric("Premium Collected", f"₹ {prem_coll:,.0f}",
                  help="Cash received in your account when you sell")
    with c6:
        st.metric("Margin Required", f"₹ {margin:,.0f}",
                  help=f"≈ ₹{margin_lot:,.0f} per lot — keep this free in your account")
    with c7:
        st.metric("Risk : Reward", f"1 : {rr}")

    cp, cl = st.columns(2)
    with cp:
        st.metric("Max Profit", f"₹ {max_profit:,.0f}",
                  help="If you buy back at target (keep 70% of premium)")
    with cl:
        st.metric("Max Loss", f"₹ {max_loss:,.0f}",
                  delta="If stop loss is hit",
                  delta_color="inverse")

    exp_move     = rec.get("expected_move_pct", 0)
    straddle     = rec.get("straddle_price",    0)
    buffer_pts   = rec.get("buffer_pts",        0)
    buffer_ratio = rec.get("buffer_ratio",      0)
    buffer_safe  = rec.get("buffer_safe",       True)

    if exp_move > 0:
        st.caption(
            f"📊 Market expects ±{exp_move:.1f}% move by expiry "
            f"(ATM straddle: ₹{straddle:.0f})"
        )

    if buffer_pts > 0 and straddle > 0:
        if buffer_ratio >= 1.2:
            st.success(
                f"🛡️  **Strike Buffer: {buffer_pts:.0f} pts ({buffer_ratio:.1f}× expected move)** — "
                "Comfortable cushion. Market would need an unusual move to breach your strike."
            )
        elif buffer_ratio >= 0.85:
            st.warning(
                f"⚠️  **Strike Buffer: {buffer_pts:.0f} pts ({buffer_ratio:.1f}× expected move)** — "
                "Moderate cushion. Watch closely if Nifty starts falling."
            )
        else:
            st.error(
                f"🚨  **Strike Buffer: {buffer_pts:.0f} pts ({buffer_ratio:.1f}× expected move)** — "
                "Strike is within the implied move range. High breach risk."
            )
        if not buffer_safe:
            st.info("ℹ️  Strike was automatically widened one step for safety (buffer was < 85% of implied move).")


def _render_buy_option(rec: dict, signal: dict):
    otype    = rec.get("option_type", "CE")
    strike   = rec.get("strike", 0)
    expiry   = rec.get("expiry")
    ltp      = rec.get("ltp", 0)
    targets  = rec.get("targets", {})
    lot_info = rec.get("lot_info", {})
    e_range  = rec.get("entry_range", {})
    data_ok  = rec.get("data_available", False)

    expiry_str = expiry.strftime("%d %b %Y") if expiry else "N/A"
    dte        = _dte(expiry)
    dte_str    = f"  ({dte} days to expiry)" if dte > 0 else ""

    if otype == "CE":
        st.success(f"⚡  BUY  NIFTY  {strike:,}  CE  —  {expiry_str}{dte_str}")
    else:
        st.error(f"⚡  BUY  NIFTY  {strike:,}  PE  —  {expiry_str}{dte_str}")

    # Contrarian warning: signal is bullish but user chose BUY PUT mode
    if rec.get("contrarian") and otype == "PE":
        direction = rec.get("direction", "")
        score     = rec.get("score", 0)
        st.warning(
            f"⚠️  **Contrarian Trade** — Market signal is {direction} (score {score:+.0f}).  \n"
            "You are buying a PUT against the current trend. This works if you expect "
            "a reversal, breakdown, or high-impact event (RBI policy, global selloff, etc.)  \n"
            "**Use smaller position size and be very disciplined with the stop loss.**"
        )

    if not data_ok:
        st.info(
            "Live prices unavailable (NSE market may be closed).  \n"
            "When market opens:  \n"
            "- **Buy** at market price  \n"
            "- **Stop Loss** = 35% below entry  \n"
            "- **Target 1** = 50% above entry (book 50%)  \n"
            "- **Target 2** = 100% above entry (trail stop)"
        )
        return

    sl = targets.get("stop_loss", 0)
    t1 = targets.get("target_1",  0)
    t2 = targets.get("target_2",  0)
    el = e_range.get("low",  ltp)
    eh = e_range.get("high", ltp)

    lots      = lot_info.get("recommended_lots", 1)
    lot_size  = lot_info.get("lot_size", 25)
    tot_units = lots * lot_size
    deployed  = round(ltp * tot_units)
    max_loss  = round((ltp - sl) * tot_units)
    t1_profit = round((t1  - ltp) * tot_units * 0.5)
    t2_profit = round((t2  - ltp) * tot_units * 0.5)
    rr        = round((t1_profit + t2_profit) / max_loss, 1) if max_loss else 0

    sl_pct = round((ltp - sl) / ltp * 100) if ltp else 0
    t1_pct = round((t1 - ltp) / ltp * 100) if ltp else 0
    t2_pct = round((t2 - ltp) / ltp * 100) if ltp else 0

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Current Premium (LTP)", f"₹ {ltp:.1f}")
    with c2:
        st.metric("Entry Zone", f"₹ {el:.1f}  –  ₹ {eh:.1f}")

    c3, c4, c5 = st.columns(3)
    with c3:
        st.metric("Stop Loss", f"₹ {sl:.1f}",
                  delta=f"−{sl_pct}%  EXIT here",
                  delta_color="inverse")
    with c4:
        st.metric("Target 1", f"₹ {t1:.1f}",
                  delta=f"+{t1_pct}%  book 50%")
    with c5:
        st.metric("Target 2", f"₹ {t2:.1f}",
                  delta=f"+{t2_pct}%  trail stop")

    st.markdown("---")
    c6, c7, c8, c9 = st.columns(4)
    with c6:
        st.metric("Lots to Buy", f"{lots} lot{'s' if lots > 1 else ''}")
    with c7:
        st.metric("Capital to Use", f"₹ {deployed:,.0f}")
    with c8:
        st.metric("Max Loss", f"₹ {max_loss:,.0f}", delta_color="inverse")
    with c9:
        st.metric("Risk : Reward", f"1 : {rr}")

    cp, ct = st.columns(2)
    with cp:
        st.metric("Profit at Target 1", f"₹ {t1_profit:,.0f}",
                  help="On 50% of position")
    with ct:
        st.metric("Profit at Target 2", f"₹ {t2_profit:,.0f}",
                  help="On remaining 50%")

    exp_move = rec.get("expected_move_pct", 0)
    straddle = rec.get("straddle_price",    0)
    if exp_move > 0:
        st.caption(
            f"📊 Market expects ±{exp_move:.1f}% move by expiry "
            f"(ATM straddle: ₹{straddle:.0f})"
        )


def _render_why_this_trade(rec: dict, tech: dict, opts: dict, news: dict):
    is_sell = rec.get("action") == "SELL PUT"
    reasons = []

    # ADX
    adx = tech.get("adx", 0)
    if is_sell:
        if   adx < 15:  reasons.append(("green",  f"ADX {adx:.0f} — flat/ranging market — theta decays fast (ideal for selling)"))
        elif adx < 20:  reasons.append(("green",  f"ADX {adx:.0f} — mild ranging — good environment for selling"))
        elif adx < 25:  reasons.append(("orange", f"ADX {adx:.0f} — building trend — watch direction carefully"))
        elif tech.get("di_bullish"):
                        reasons.append(("green",  f"ADX {adx:.0f} — strong UPtrend — put seller is safe (market moving away)"))
        else:           reasons.append(("red",    f"ADX {adx:.0f} — strong DOWNtrend — DANGEROUS for put sellers"))
    else:
        if   adx >= 30: reasons.append(("green",  f"ADX {adx:.0f} — strong trend, ideal for directional options"))
        elif adx >= 25: reasons.append(("green",  f"ADX {adx:.0f} — good trend in place"))
        elif adx >= 20: reasons.append(("orange", f"ADX {adx:.0f} — moderate trend, watch for reversal"))
        else:           reasons.append(("red",    f"ADX {adx:.0f} — weak/no trend (ranging market)"))

    # IV Rank (critical for sell strategy)
    iv_rank = opts.get("iv_rank", "NORMAL")
    if is_sell:
        iv_msgs = {
            "VERY_HIGH": ("green",  "IV Rank VERY HIGH — collecting very rich premium, excellent for selling"),
            "HIGH":      ("green",  "IV Rank HIGH — premium is elevated, good time to sell"),
            "NORMAL":    ("orange", "IV Rank NORMAL — standard premium, OK to sell"),
            "VERY_LOW":  ("red",    "IV Rank VERY LOW — thin premium, margin risk may not be worth it"),
        }
        reasons.append(iv_msgs.get(iv_rank, ("orange", f"IV Rank {iv_rank}")))

    # Supertrend
    st_bull = tech.get("st_bullish")
    if is_sell:
        if st_bull is True:  reasons.append(("green",  "Supertrend BULLISH — uptrend confirmed, put sellers are safe"))
        elif st_bull is False: reasons.append(("red",  "Supertrend BEARISH — downtrend in place, puts may spike"))
    else:
        if st_bull is True:  reasons.append(("green",  "Supertrend BULLISH — trend-following signal confirms buy"))
        elif st_bull is False: reasons.append(("red",  "Supertrend BEARISH — trend-following signal says sell"))

    # EMA
    above_all = tech.get("above_ema200") and tech.get("above_ema50") and tech.get("above_ema20")
    below_all = not tech.get("above_ema200") and not tech.get("above_ema50") and not tech.get("above_ema20")
    if above_all:
        reasons.append(("green",  "Above EMA 20 / 50 / 200 — strong uptrend on all timeframes"))
    elif tech.get("above_ema200"):
        reasons.append(("green",  "Above EMA 200 — long-term uptrend intact"))
    elif below_all:
        reasons.append(("red",    "Below EMA 20 / 50 / 200 — downtrend on all timeframes"))
    else:
        reasons.append(("orange", "Mixed EMA signals — short-term and long-term disagree"))

    # RSI
    rsi = tech.get("rsi", 50)
    if   rsi < 30:  reasons.append(("green",  f"RSI {rsi:.0f} — oversold, bounce expected"))
    elif rsi < 45:  reasons.append(("green",  f"RSI {rsi:.0f} — plenty of upside room"))
    elif rsi < 60:  reasons.append(("orange", f"RSI {rsi:.0f} — neutral zone"))
    elif rsi < 70:  reasons.append(("orange", f"RSI {rsi:.0f} — approaching overbought"))
    else:           reasons.append(("red",    f"RSI {rsi:.0f} — overbought, momentum may fade"))

    # PCR
    pcr_sig = opts.get("pcr_signal", "NEUTRAL")
    pcr_val = opts.get("pcr", 0)
    if is_sell:
        if pcr_sig in ("STRONG_BULLISH", "BULLISH"):
            reasons.append(("green",  f"PCR {pcr_val:.2f} — high put OI = put premiums elevated, good for selling"))
        elif pcr_sig in ("STRONG_BEARISH", "BEARISH"):
            reasons.append(("red",    f"PCR {pcr_val:.2f} — market bearish, risky to sell puts"))
        else:
            reasons.append(("orange", f"PCR {pcr_val:.2f} — neutral market stance"))
    else:
        if pcr_sig in ("STRONG_BULLISH", "BULLISH"):
            reasons.append(("green",  f"PCR {pcr_val:.2f} — heavy put writing, contrarian bullish"))
        elif pcr_sig in ("STRONG_BEARISH", "BEARISH"):
            reasons.append(("red",    f"PCR {pcr_val:.2f} — heavy call writing, contrarian bearish"))
        else:
            reasons.append(("orange", f"PCR {pcr_val:.2f} — neutral"))

    # Max Pain
    mp     = opts.get("max_pain", 0)
    diff   = opts.get("max_pain_diff_pct", 0)
    strike = rec.get("strike", 0)
    if mp and is_sell and strike:
        if mp >= strike:
            reasons.append(("green",  f"Max Pain ₹{mp:,.0f} is above your sold strike ₹{strike:,} — gravity on your side"))
        else:
            reasons.append(("orange", f"Max Pain ₹{mp:,.0f} is near/below your sold strike — monitor carefully"))
    elif mp and diff != 0:
        col = "green" if diff > 0 else "red"
        reasons.append((col, f"Max Pain ₹{mp:,.0f} {'above' if diff > 0 else 'below'} spot — "
                        f"gravity {'pulls price up' if diff > 0 else 'pulls price down'} by expiry"))

    # Global markets
    es = news.get("event_score", 0)
    if   es >  15: reasons.append(("green", f"Global markets positive ({es:+.0f}) — tailwind for Nifty"))
    elif es < -15: reasons.append(("red",   f"Global markets negative ({es:+.0f}) — headwind for Nifty"))

    css = {"green": "#32cd32", "orange": "#ffd700", "red": "#ff4444"}
    for color, text in reasons[:8]:
        c = css.get(color, "#ffffff")
        st.markdown(f'<span style="color:{c}">● {text}</span>', unsafe_allow_html=True)


def _render_indicator_legend():
    st.markdown("""
**SCORE (−100 to +100)** — Weighted signal: 50% Technical + 30% Options + 20% News
- 🟢 +60 to +100 → **STRONG BULLISH**
- 🟢 +30 to +60 → **BULLISH**
- 🟡  −30 to +30 → **NEUTRAL**
- 🔴 −30 to −60 → **BEARISH**
- 🔴 −60 to −100 → **STRONG BEARISH**

---

**ADX (0–100)** — Trend *strength* only, not direction
- 🔴 < 15 → Flat/Sideways *(bad for buyers; **great for sellers** — theta burns fast)*
- 🟡 15–20 → Weak trend
- 🟡 20–25 → Building trend
- 🟢 ≥ 25 → Strong trend

---

**QUALITY (0–100)** — How good today's setup is **for your strategy**
*(Sell quality and buy quality use different scoring rules)*
- 🟢 ≥ 70 → **STRONG** (trade full size)
- 🔵 55–69 → **GOOD**
- 🟡 40–54 → **WEAK** (half size or skip)
- 🔴 < 40 → **AVOID** (system blocks the trade)

---

**VIX** — India Fear Index — how expensive options are
- ≤ 12 → Too calm / complacency
- 12–15 → 🟢 Ideal environment
- 15–20 → 🟡 Slightly elevated
- 20–25 → 🔴 High fear (reduce position size)
- > 25 → 🔴 Extreme fear (system halves lots)

---

**SUPERTREND** — Trend direction via ATR volatility bands (period 14, mult 3)
- 🟢 ↑ BULL → price above upper band (uptrend confirmed)
- 🔴 ↓ BEAR → price below lower band (downtrend confirmed)

---

**PCR (Put-Call Ratio)** — Total Put OI ÷ Total Call OI — *contrarian* indicator
- > 1.5 → Everyone panic-buying puts → **contrarian bullish** (market near bottom)
- < 0.5 → Everyone chasing calls → **contrarian bearish** (market near top)
- When the crowd buys puts in panic, the market tends to reverse UP *(and vice versa)*

---

**MAX PAIN** — Strike where option writers (sellers) lose the least money
- Market tends to pin near this level at expiry — useful gravity reference
- For put sellers: if max pain > your sold strike, gravity is on your side

---

**OBV (On Balance Volume)** — Volume flow confirmation
- 🟢 Rising → smart money accumulating (bullish)
- 🔴 Falling → distribution underway (bearish)

---

**IV RANK** — Where current Implied Volatility sits vs the past year
- VERY_LOW / NORMAL → Options cheap → 🟢 good for **buyers**
- HIGH / VERY_HIGH → Options expensive → 🟢 good for **sellers** (rich premium to collect)
""")


# ═══════════════════════════════════════════════════════════════════════════
# Live Nifty Chart
# ═══════════════════════════════════════════════════════════════════════════

def _render_nifty_chart(r: dict) -> None:
    """
    30-day daily Nifty 50 candlestick chart using the already-computed hist_df.
    No extra network call — data is always available (works on weekends/after hours).

    Indicators (all pre-computed by TechnicalAnalysis):
      • EMA 20 / 50 / 200  — short / medium / long-term trend
      • Bollinger Bands     — volatility envelope (shaded)
      • Supertrend          — green = bullish, red = bearish (flips at crossovers)
      • Volume bars         — green/red matching candle direction

    Horizontal levels:
      • Max Pain, OI Resistance, OI Support  — from live options chain
      • Strategy level: Sold Strike (SELL PUT) or Bought Strike (BUY PUT)
      • ATM reference line
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        st.caption("plotly not installed — run: pip install plotly")
        return

    hist_df = r.get("hist_df")
    if hist_df is None or hist_df.empty:
        st.caption("Chart data unavailable.")
        return

    df     = hist_df.tail(30).copy()
    opts   = r.get("options_summary", {})
    rec    = r.get("recommendation", {})
    action = rec.get("action", "AVOID")

    # ── Figure: price (75%) + volume (25%) ──────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.02,
        subplot_titles=["Nifty 50 — Daily  (Last 30 Sessions)", "Volume"],
    )

    # ── Candlesticks ────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color="#26a69a", increasing_fillcolor="#26a69a",
        decreasing_line_color="#ef5350", decreasing_fillcolor="#ef5350",
        name="Nifty 50", showlegend=False,
    ), row=1, col=1)

    # ── Bollinger Bands (shaded) ─────────────────────────────────────────────
    if "BB_Upper" in df.columns and "BB_Lower" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Upper"],
            line=dict(color="rgba(158,158,158,0.30)", width=1),
            name="BB Upper", showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Lower"],
            line=dict(color="rgba(158,158,158,0.30)", width=1),
            fill="tonexty", fillcolor="rgba(158,158,158,0.06)",
            name="Bollinger Bands", showlegend=True,
        ), row=1, col=1)

    # ── EMA lines ───────────────────────────────────────────────────────────
    for col_name, color, width, label in [
        ("EMA20",  "#FF9800", 1.5, "EMA 20"),
        ("EMA50",  "#2196F3", 1.5, "EMA 50"),
        ("EMA200", "#CE93D8", 2.0, "EMA 200"),
    ]:
        if col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col_name],
                line=dict(color=color, width=width),
                name=label, opacity=0.90,
            ), row=1, col=1)

    # ── Supertrend (green = bullish, red = bearish) ──────────────────────────
    if "Supertrend" in df.columns and "ST_Bullish" in df.columns:
        st_bull = df["Supertrend"].where(df["ST_Bullish"].astype(bool))
        st_bear = df["Supertrend"].where(~df["ST_Bullish"].astype(bool))
        fig.add_trace(go.Scatter(
            x=df.index, y=st_bull,
            line=dict(color="#00E676", width=2.5),
            name="Supertrend ↑", connectgaps=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=st_bear,
            line=dict(color="#FF5252", width=2.5),
            name="Supertrend ↓", connectgaps=False,
        ), row=1, col=1)

    # ── Volume bars ──────────────────────────────────────────────────────────
    if "Volume" in df.columns:
        bar_colors = [
            "#26a69a" if c >= o else "#ef5350"
            for o, c in zip(df["Open"], df["Close"])
        ]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"],
            marker_color=bar_colors,
            name="Volume", showlegend=False, opacity=0.65,
        ), row=2, col=1)

    # ── OI horizontal levels ─────────────────────────────────────────────────
    max_pain = opts.get("max_pain", 0)
    if max_pain:
        fig.add_hline(
            y=max_pain, line_dash="dot", line_color="#FFC107", line_width=1.5,
            annotation_text=f"Max Pain ₹{max_pain:,.0f}",
            annotation_position="bottom left",
            annotation_font=dict(color="#FFC107", size=11),
            row=1, col=1,
        )

    resistance_list = opts.get("top_call_resistance", [])
    if resistance_list:
        fig.add_hline(
            y=resistance_list[0], line_dash="dash", line_color="#EF5350", line_width=1.2,
            annotation_text=f"OI Resistance ₹{resistance_list[0]:,.0f}",
            annotation_position="top left",
            annotation_font=dict(color="#EF5350", size=11),
            row=1, col=1,
        )

    support_list = opts.get("top_put_support", [])
    if support_list:
        fig.add_hline(
            y=support_list[0], line_dash="dash", line_color="#66BB6A", line_width=1.2,
            annotation_text=f"OI Support ₹{support_list[0]:,.0f}",
            annotation_position="bottom left",
            annotation_font=dict(color="#66BB6A", size=11),
            row=1, col=1,
        )

    # ── Strategy-specific levels ─────────────────────────────────────────────
    if action in ("SELL PUT", "BUY PUT (PE)"):
        strike = rec.get("strike", 0)
        atm    = rec.get("atm_strike", 0)
        if action == "SELL PUT":
            s_color = "#FF9800"
            s_label = f"Sold Strike ₹{strike:,.0f} PE ⚠"
        else:
            s_color = "#EF5350"
            s_label = f"Bought Strike ₹{strike:,.0f} PE"
        if strike:
            fig.add_hline(
                y=strike, line_dash="solid", line_color=s_color, line_width=2,
                annotation_text=s_label,
                annotation_position="bottom right",
                annotation_font=dict(color=s_color, size=12),
                row=1, col=1,
            )
        if atm and atm != strike:
            fig.add_hline(
                y=atm, line_dash="dot", line_color="#9E9E9E", line_width=1,
                annotation_text=f"ATM ₹{atm:,.0f}",
                annotation_position="top right",
                annotation_font=dict(color="#9E9E9E", size=10),
                row=1, col=1,
            )

    # ── Layout ───────────────────────────────────────────────────────────────
    fig.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#fafafa", size=12),
        height=580,
        margin=dict(l=10, r=90, t=40, b=10),
        legend=dict(
            bgcolor="rgba(14,17,23,0.75)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            font=dict(size=11),
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="left", x=0,
        ),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        # Hide weekends — no trading data
        xaxis=dict(rangebreaks=[dict(bounds=["sat", "mon"])]),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1e2130", showline=True, linecolor="#333")
    fig.update_yaxes(
        showgrid=True, gridcolor="#1e2130", tickprefix="₹", tickformat=",.0f",
        showline=True, linecolor="#333", row=1, col=1,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="#1e2130", tickformat=".2s",
        row=2, col=1,
    )
    subplot_titles = {"Nifty 50 — Daily  (Last 30 Sessions)", "Volume"}
    for ann in fig.layout.annotations:
        if getattr(ann, "text", None) in subplot_titles:
            ann.font = dict(color="#9e9e9e", size=12)

    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# Main render — called after analysis completes
# ═══════════════════════════════════════════════════════════════════════════

def render_results(r: dict):
    signal = r.get("signal_data", {})
    rec    = r.get("recommendation", {})
    tech   = r.get("tech_data",      {})
    news   = r.get("news_data",      {})
    opts   = r.get("options_summary",{})
    spot   = r.get("spot",    0.0)
    vix    = r.get("vix",     0.0)
    ts     = r.get("timestamp")

    # ── Mode badge ────────────────────────────────────────────────────────
    strategy = r.get("strategy", STRATEGY)
    if strategy == "SELL_PUTS":
        st.info("**Mode: SELL PUT** — Selling OTM put to collect premium (neutral / bullish)")
    else:
        st.info("**Mode: BUY PUT** — Buying put option for a bearish directional trade")

    # ── Timestamp ─────────────────────────────────────────────────────────
    if ts:
        st.caption(f"Analysis run at {ts.strftime('%I:%M %p IST, %d %b %Y')}")

    # ── Market status notices ──────────────────────────────────────────────
    if datetime.now(IST).weekday() >= 5:
        st.warning("Today is a **weekend** — NSE is closed. Results based on last trading day's data. Useful for Monday prep.")
    if r.get("options_empty"):
        st.info("📋 Options chain unavailable (NSE may be closed or throttling). Options-based metrics show defaults — technical analysis still fully active.")

    # ── 4-metric snapshot row ──────────────────────────────────────────────
    score     = signal.get("score", 0.0)
    direction = signal.get("direction", "NEUTRAL")
    adx       = tech.get("adx", 0.0)
    adx_label = "Strong" if adx >= 25 else ("Moderate" if adx >= 20 else "Weak/Ranging")
    vix_label = "High" if vix > 20 else ("Normal" if vix <= 15 else "Elevated")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Nifty Spot", f"₹ {spot:,.2f}")
    with c2:
        st.metric("India VIX", f"{vix:.1f}", delta=vix_label,
                  delta_color="inverse" if vix > 20 else "normal")
    with c3:
        st.metric("ADX", f"{adx:.1f}", delta=adx_label)
    with c4:
        st.metric("Signal Score", f"{score:+.1f}", delta=_dir_label(direction))

    # ── Live Chart ─────────────────────────────────────────────────────────
    st.markdown("---")
    _render_nifty_chart(r)

    # ── Quality bar ────────────────────────────────────────────────────────
    is_sell   = rec.get("action") == "SELL PUT"
    is_buyput = rec.get("action") == "BUY PUT (PE)" and not is_sell
    quality   = signal.get("sell_quality_score") if is_sell else signal.get("quality_score", 0)
    qlabel    = signal.get("sell_quality_label")  if is_sell else signal.get("quality_label",  "")
    quality   = quality or 0

    st.markdown("---")
    qc1, qc2 = st.columns([3, 1])
    with qc1:
        st.markdown(f"**Trade Setup Quality:** `{qlabel}` &nbsp; {quality}/100")
        st.progress(quality / 100)
    with qc2:
        if is_sell:
            hint = "HIGH IV + ranging market = ideal for selling"
        elif is_buyput:
            hint = "Strong trend + cheap IV + bearish signal = ideal for buying put"
        else:
            hint = "Strong trend + cheap IV = ideal for buying"
        st.caption(hint)

    # ── Supertrend tag ─────────────────────────────────────────────────────
    st_bull = tech.get("st_bullish")
    if st_bull is True:
        st.success("Supertrend ↑ BULLISH")
    elif st_bull is False:
        st.error("Supertrend ↓ BEARISH")

    # ── Major event alert ──────────────────────────────────────────────────
    if news.get("has_major_event"):
        event_type = news.get("major_event_type", "NONE")
        top        = news.get("top_event") or {}
        headline   = top.get("headline", "Major market event detected")
        if event_type == "BEARISH":
            st.error(f"⚡ MAJOR BEARISH EVENT — {headline}  *(News weight: 50%, CRISIS MODE active)*")
        else:
            st.success(f"⚡ MAJOR BULLISH EVENT — {headline}  *(News weight: 50%, CRISIS MODE active)*")

    # ── Trade Recommendation ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Trade Recommendation")

    action = rec.get("action", "AVOID")

    if action == "AVOID":
        st.warning(f"**NO TRADE TODAY**\n\n{rec.get('reason', 'No clear directional bias.')}")

    elif action == "SELL PUT":
        _render_sell_put(rec, signal)

    else:
        _render_buy_option(rec, signal)

    # ── Global Markets ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Global Markets")
    gm = news.get("global_markets", {})
    if gm:
        rows = []
        for sym, data in gm.items():
            pct    = data.get("pct_change", 0)
            impact = data.get("score_contribution", 0)
            rows.append({
                "Market":        data.get("name", sym),
                "Price":         f"{data.get('price', 0):,.1f}",
                "Change":        f"{pct:+.2f}%",
                "Nifty Impact":  f"{impact:+.0f} pts",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        es = news.get("event_score", 0)
        st.caption(f"Combined global score: {es:+.1f} pts")
    else:
        st.caption("Global market data unavailable.")

    # ── Why This Trade ─────────────────────────────────────────────────────
    if action != "AVOID":
        with st.expander("Why This Trade", expanded=False):
            _render_why_this_trade(rec, tech, opts, news)

    # ── Recent Candles ─────────────────────────────────────────────────────
    recent = r.get("recent_candles")
    if recent is not None and not recent.empty:
        with st.expander("Recent Price History (last 5 sessions)", expanded=False):
            st.dataframe(recent, use_container_width=True)

    # ── Indicator Glossary ─────────────────────────────────────────────────
    with st.expander("Indicator Glossary", expanded=False):
        _render_indicator_legend()

    # ── Footer ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(f"Data: NSE India API + Yahoo Finance  •  Strategy: `{STRATEGY}`  •  Capital: ₹{r.get('capital', DEFAULT_CAPITAL):,.0f}")


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar + main panel
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("📈 Nifty Options")
    st.markdown("---")

    capital = st.number_input(
        "Trading Capital (INR)",
        min_value=10_000,
        max_value=10_000_000,
        value=int(DEFAULT_CAPITAL),
        step=10_000,
        format="%d",
        help="Your available trading capital in Indian Rupees.",
    )

    st.markdown("**Mode**")
    mode_choice = st.radio(
        "Mode",
        options=["SELL PUT", "BUY PUT"],
        index=0,
        horizontal=True,
        help=(
            "SELL PUT — Collect premium by selling an OTM put. "
            "Profit from time decay. Best in neutral-to-bullish, high-IV markets.\n\n"
            "BUY PUT — Buy a put option for a bearish directional bet. "
            "Limited risk, unlimited profit. Best when you expect Nifty to fall."
        ),
        label_visibility="collapsed",
    )
    # Map UI label → internal strategy string
    selected_strategy = "SELL_PUTS" if mode_choice == "SELL PUT" else "BUY_PUT"

    if mode_choice == "SELL PUT":
        st.caption("Collect premium · profit from time decay")
    else:
        st.caption("Bearish directional bet · limited risk")

    st.markdown("---")

    run_btn = st.button("▶  Run Analysis", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("Data: NSE API + Yahoo Finance")
    st.caption(f"Today: {datetime.now(IST).strftime('%d %b %Y')}")


# ── Main panel ─────────────────────────────────────────────────────────────
st.title("Nifty Options Prediction System")

# Initialise session state
if "results" not in st.session_state:
    st.session_state.results = None
if "error" not in st.session_state:
    st.session_state.error = None
if "last_strategy" not in st.session_state:
    st.session_state.last_strategy = selected_strategy

# Clear cached results when user switches mode
if st.session_state.last_strategy != selected_strategy:
    st.session_state.results       = None
    st.session_state.error         = None
    st.session_state.last_strategy = selected_strategy

if run_btn:
    st.session_state.results = None
    st.session_state.error   = None
    with st.spinner("Fetching market data and running analysis… (takes ~10–15 seconds)"):
        try:
            result = run_analysis(capital, selected_strategy)
            result["capital"] = capital
            st.session_state.results = result
        except Exception as exc:
            st.session_state.error = str(exc)
            logging.getLogger(__name__).exception("Analysis pipeline failed")

if st.session_state.error:
    st.error(f"Analysis failed: {st.session_state.error}")
    st.info(
        "Common causes:\n"
        "- NSE API offline or throttling (try again in 30 seconds)\n"
        "- No internet connection\n"
        "- NSE holiday (data still loads from Yahoo Finance)"
    )

elif st.session_state.results is not None:
    render_results(st.session_state.results)

else:
    st.info(
        "Configure your trading capital in the sidebar, then click **▶ Run Analysis** "
        "to fetch live data and get today's recommendation."
    )
    st.markdown("""
    **What you'll see:**
    - Live Nifty spot price, VIX, ADX, and signal score
    - Trade recommendation with exact strike, expiry, entry, stop loss, and target
    - Capital required and rupee profit/loss estimates
    - Global markets (US, Japan, China, Oil, Gold, USD/INR)
    - Full indicator glossary
    """)

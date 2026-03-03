#!/usr/bin/env python3
"""
Nifty SELL PUT Backtest — Realistic Weekly Strategy
====================================================
Simulates actual weekly put-selling workflow:
  • One trade per expiry week — enter on the FIRST day within the week
    where SellQ >= threshold.  If no day qualifies → AVOID week.
  • After entry, hold to expiry (Thursday). Do not re-enter until next week.
  • WIN  = Nifty never closed below the sold strike through expiry
  • LOSS = Nifty closed below the strike at any point before expiry

Runs over the last BACKTEST_WEEKS expiry cycles (~6 months).
Uses real historical Nifty + India VIX. Options data proxied via VIX-derived IV.
"""

import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from collections import defaultdict

from src.technical import TechnicalAnalysis
from src.signals   import SignalGenerator

# ── Config ────────────────────────────────────────────────────────────────────
STRIKE_GAP      = 50
OTM_STEPS       = 2        # ATM − 100 pts OTM
MIN_WARMUP      = 100      # bars for warm indicators
SELL_THRESHOLD  = 65       # SellQ threshold (raised to 65 for higher accuracy)
BACKTEST_WEEKS  = 26       # ≈ 6 months of weekly expiries
TODAY           = datetime.now().date()

# ── Data download ─────────────────────────────────────────────────────────────
print("Downloading Nifty 50 and India VIX history…")
nifty_raw = yf.download("^NSEI",     period="2y", interval="1d",
                        progress=False, auto_adjust=True)
vix_raw   = yf.download("^INDIAVIX", period="2y", interval="1d",
                        progress=False, auto_adjust=True)

for _df in [nifty_raw, vix_raw]:
    if isinstance(_df.columns, pd.MultiIndex):
        _df.columns = _df.columns.get_level_values(0)
nifty_raw.index = pd.to_datetime(nifty_raw.index).normalize()
vix_raw.index   = pd.to_datetime(vix_raw.index).normalize()

print(f"Loaded {len(nifty_raw)} Nifty sessions  |  {len(vix_raw)} VIX sessions\n")

# ── Helpers ───────────────────────────────────────────────────────────────────
def nearest_expiry(d: date, trading_set: set) -> date:
    dow        = d.weekday()
    days_ahead = (3 - dow) % 7
    if days_ahead == 0:
        days_ahead = 7
    candidate = d + timedelta(days=days_ahead)
    while candidate not in trading_set and candidate > d:
        candidate -= timedelta(days=1)
    return candidate

def vix_to_iv_rank(vix: float) -> str:
    if vix < 12:  return "VERY_LOW"
    if vix < 16:  return "NORMAL"
    if vix < 22:  return "HIGH"
    return "VERY_HIGH"

def expected_move_pct(close: float, vix: float, dte_cal: int) -> float:
    dte_trading = max(1, dte_cal * 252 / 365)
    return round((vix / 100) * (dte_trading / 252) ** 0.5 * 100, 2)

def compute_signal(hist: pd.DataFrame, vix: float, dte_cal: int) -> dict:
    ta    = TechnicalAnalysis(hist)
    tech  = ta.get_current_values()
    close = tech.get("close", 0)
    emp   = expected_move_pct(close, vix, dte_cal)
    opts  = {
        "pcr_signal":        "NEUTRAL",
        "max_pain_diff_pct": 0.5,
        "days_to_expiry":    max(1, dte_cal),
        "iv_rank":           vix_to_iv_rank(vix),
        "expected_move_pct": emp,
        "straddle_price":    round(close * emp / 100, 1),
        "total_new_put_oi":  0,
        "total_new_call_oi": 0,
    }
    sg = SignalGenerator(tech, opts, vix=vix)
    r  = sg.get_full_analysis()
    return {
        "score":      round(r["score"], 1),
        "sell_q":     r["sell_quality_score"],
        "rsi":        round(tech.get("rsi", 50), 1),
        "adx":        round(tech.get("adx", 20), 1),
        "st_bull":    tech.get("st_bullish"),
        "emp":        emp,
        "ema50_dist": round(tech.get("close_vs_ema50_pct", 0), 1),
        "ema50_slp":  round(tech.get("ema50_slope_pct",   0), 2),
        "mom_20d":    round(tech.get("momentum_20d_pct",  0), 1),
    }

# ── Build weekly expiry schedule ───────────────────────────────────────────────
all_ts        = nifty_raw.index.tolist()
trading_dates = {ts.date() for ts in all_ts}

# Map every trading date → its expiry
date_to_expiry: dict[date, date] = {}
for ts in all_ts:
    d = ts.date()
    date_to_expiry[d] = nearest_expiry(d, trading_dates)

# Group trading dates by expiry
expiry_groups: dict[date, list[date]] = defaultdict(list)
for d, exp in date_to_expiry.items():
    expiry_groups[exp].append(d)
for exp in expiry_groups:
    expiry_groups[exp].sort()

# Take the last BACKTEST_WEEKS expiry weeks that have enough warmup data
all_expiries = sorted(expiry_groups.keys())
# Warmup: need at least MIN_WARMUP bars before the first test date
warmup_cutoff = all_ts[MIN_WARMUP - 1].date()
valid_expiries = [e for e in all_expiries if expiry_groups[e][0] > warmup_cutoff]
test_expiries  = valid_expiries[-BACKTEST_WEEKS:]

print(f"Testing {len(test_expiries)} expiry weeks")
print(f"Window  : {expiry_groups[test_expiries[0]][0]}  →  {test_expiries[-1]}")
print(f"Entry   : first trading day within each week where SellQ >= {SELL_THRESHOLD}")
print(f"Strike  : ATM − {OTM_STEPS * STRIKE_GAP} pts\n")

# ── Backtest loop — one row per expiry week ───────────────────────────────────
rows = []

for exp_date in test_expiries:
    week_days = expiry_groups[exp_date]   # e.g. [Mon, Tue, Wed, Thu] or subset

    entry_date = None
    entry_sig  = None
    entry_close = None
    entry_vix   = None

    for trade_date in week_days:
        trade_ts = pd.Timestamp(trade_date)
        g_idx    = all_ts.index(trade_ts)

        hist_slice = nifty_raw.iloc[: g_idx + 1].copy()
        vix   = float(vix_raw.loc[trade_ts, "Close"]) if trade_ts in vix_raw.index else 15.0
        close = float(nifty_raw.loc[trade_ts, "Close"])
        dte   = (exp_date - trade_date).days

        try:
            sig = compute_signal(hist_slice, vix, dte)
        except Exception as e:
            continue

        if sig["sell_q"] >= SELL_THRESHOLD:
            entry_date  = trade_date
            entry_sig   = sig
            entry_close = close
            entry_vix   = vix
            break   # first qualifying day wins

    # ── Determine outcome ─────────────────────────────────────────────────────
    if entry_date is not None:
        # SELL PUT trade entered
        atm    = int(round(entry_close / STRIKE_GAP) * STRIKE_GAP)
        strike = atm - OTM_STEPS * STRIKE_GAP

        # Forward closes from entry+1 to expiry inclusive
        fwd_mask = (
            (nifty_raw.index > pd.Timestamp(entry_date)) &
            (nifty_raw.index <= pd.Timestamp(exp_date))
        )
        fwd_df = nifty_raw[fwd_mask]

        if TODAY > exp_date and not fwd_df.empty:
            min_fwd = float(fwd_df["Close"].min())
            breached = min_fwd < strike
            outcome  = "WIN ✅" if not breached else "LOSS ❌"
        elif TODAY <= exp_date:
            min_fwd = None
            outcome  = "PENDING ⏳"
        else:
            min_fwd = None
            outcome  = "PENDING ⏳"

        rows.append({
            "Week":     str(exp_date),
            "EntryDay": str(entry_date),
            "Close":    int(entry_close),
            "VIX":      round(entry_vix, 1),
            "RSI":      entry_sig["rsi"],
            "ADX":      entry_sig["adx"],
            "ST":       "↑" if entry_sig["st_bull"] else ("↓" if entry_sig["st_bull"] is False else "?"),
            "Score":    entry_sig["score"],
            "SellQ":    entry_sig["sell_q"],
            "EMA50%":   entry_sig["ema50_dist"],
            "Mom20d":   entry_sig["mom_20d"],
            "Strike":   strike,
            "MinClose": int(min_fwd) if min_fwd else "-",
            "Outcome":  outcome,
        })
    else:
        # No qualifying day this week → AVOID
        # Use the first day's close/signal for context
        first_day = week_days[0]
        first_ts  = pd.Timestamp(first_day)
        g_idx     = all_ts.index(first_ts)
        hist_sl   = nifty_raw.iloc[: g_idx + 1].copy()
        vix0  = float(vix_raw.loc[first_ts, "Close"]) if first_ts in vix_raw.index else 15.0
        close0 = float(nifty_raw.loc[first_ts, "Close"])
        dte0   = (exp_date - first_day).days
        try:
            sig0 = compute_signal(hist_sl, vix0, dte0)
        except Exception:
            sig0 = {"score": 0, "sell_q": 0, "rsi": 50, "adx": 15,
                    "st_bull": None, "emp": 1.5,
                    "ema50_dist": 0, "ema50_slp": 0, "mom_20d": 0}

        # Check what would have happened if traded
        atm0   = int(round(close0 / STRIKE_GAP) * STRIKE_GAP)
        strike0 = atm0 - OTM_STEPS * STRIKE_GAP
        fwd_mask0 = (
            (nifty_raw.index > first_ts) &
            (nifty_raw.index <= pd.Timestamp(exp_date))
        )
        fwd_df0 = nifty_raw[fwd_mask0]

        if TODAY > exp_date and not fwd_df0.empty:
            min_fwd0 = float(fwd_df0["Close"].min())
            breached0 = min_fwd0 < strike0
            av_outcome = "GOOD_AVOID ✓" if breached0 else "SAFE_AVOID ⚠"
        elif TODAY <= exp_date:
            min_fwd0  = None
            av_outcome = "PENDING ⏳"
        else:
            min_fwd0  = None
            av_outcome = "PENDING ⏳"

        rows.append({
            "Week":     str(exp_date),
            "EntryDay": f"({first_day})",
            "Close":    int(close0),
            "VIX":      round(vix0, 1),
            "RSI":      sig0["rsi"],
            "ADX":      sig0["adx"],
            "ST":       "↑" if sig0["st_bull"] else ("↓" if sig0["st_bull"] is False else "?"),
            "Score":    sig0["score"],
            "SellQ":    sig0["sell_q"],
            "EMA50%":   sig0["ema50_dist"],
            "Mom20d":   sig0["mom_20d"],
            "Strike":   f"({strike0}) AVOID",
            "MinClose": int(min_fwd0) if min_fwd0 else "-",
            "Outcome":  av_outcome,
        })

df = pd.DataFrame(rows)

# ── Cumulative accuracy ────────────────────────────────────────────────────────
cum_w, cum_t = 0, 0
cum_acc = []
for outcome in df["Outcome"]:
    if "WIN" in outcome:
        cum_w += 1; cum_t += 1; cum_acc.append(f"{cum_w/cum_t*100:.0f}%")
    elif "LOSS" in outcome:
        cum_t += 1; cum_acc.append(f"{cum_w/cum_t*100:.0f}%")
    else:
        cum_acc.append("—")
df["CumAcc"] = cum_acc

# ── Print table ────────────────────────────────────────────────────────────────
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 175)
pd.set_option("display.max_colwidth", 20)

print("=" * 155)
print(f"{'NIFTY SELL PUT BACKTEST — ONE TRADE PER EXPIRY WEEK  (~6 months)':^155}")
print("=" * 155)
print(df.to_string(index=False))
print("=" * 155)

# ── Summary stats ─────────────────────────────────────────────────────────────
sp_rows    = df[~df["Outcome"].str.contains("AVOID|PENDING")]
av_rows    = df[df["Outcome"].str.contains("AVOID")]
completed  = sp_rows[~sp_rows["Outcome"].str.contains("PENDING")]
wins       = completed["Outcome"].str.contains("WIN").sum()
losses     = completed["Outcome"].str.contains("LOSS").sum()
pending    = sp_rows["Outcome"].str.contains("PENDING").sum()
total_tr   = wins + losses
accuracy   = wins / total_tr * 100 if total_tr > 0 else 0

good_avoid = av_rows["Outcome"].str.contains("GOOD_AVOID").sum()
safe_avoid = av_rows["Outcome"].str.contains("SAFE_AVOID").sum()

# Naive: trade every week regardless of signal (using first day)
all_completed = df[~df["Outcome"].str.contains("PENDING")]
naive_wins    = all_completed["Outcome"].str.contains("WIN|SAFE_AVOID").sum()
naive_losses  = all_completed["Outcome"].str.contains("LOSS|GOOD_AVOID").sum()
naive_total   = naive_wins + naive_losses
naive_acc     = naive_wins / naive_total * 100 if naive_total > 0 else 0

print()
print("┌─────────────────────────────────────────────────────────────┐")
print("│                    BACKTEST SUMMARY                        │")
print("├─────────────────────────────────────────────────────────────┤")
print(f"│  Test period:        ~6 months ({len(test_expiries)} expiry weeks)         │")
print(f"│  Entry rule:         1st qualifying day per week (SellQ≥{SELL_THRESHOLD}) │")
print(f"│  Strike:             ATM − {OTM_STEPS*STRIKE_GAP} pts (2 steps OTM)             │")
print("├─────────────────────────────────────────────────────────────┤")
print(f"│  SELL PUT weeks:     {len(sp_rows):3d}  ({len(sp_rows)/len(df)*100:.0f}% of all weeks)             │")
print(f"│  AVOID weeks:        {len(av_rows):3d}  ({len(av_rows)/len(df)*100:.0f}% of all weeks)             │")
print("├─────────────────────────────────────────────────────────────┤")
print(f"│  Completed trades:   {total_tr:3d}                                   │")
print(f"│  ✅  Wins:           {wins:3d}                                   │")
print(f"│  ❌  Losses:         {losses:3d}                                   │")
print(f"│  ⏳  Pending:        {pending:3d}                                   │")
print(f"│                                                             │")
print(f"│  🎯  ACCURACY:       {accuracy:.1f}%  (on filtered SELL PUT weeks)   │")
print("├─────────────────────────────────────────────────────────────┤")
print(f"│  AVOID filter quality:                                      │")
print(f"│  ✓  Good avoids:     {good_avoid:3d}  (correctly avoided a LOSS)       │")
print(f"│  ⚠  Safe avoids:     {safe_avoid:3d}  (would have won, but skipped)   │")
if len(av_rows) > 0:
    fa = good_avoid / len(av_rows) * 100
    print(f"│  Filter precision:   {fa:.0f}%  of AVOID weeks were real danger   │")
print("├─────────────────────────────────────────────────────────────┤")
print(f"│  WITHOUT filter (all {len(df)} weeks): {naive_acc:.1f}% win rate              │")
print(f"│  WITH filter (our system):    {accuracy:.1f}% win rate              │")
upside = accuracy - naive_acc
sign = "+" if upside >= 0 else ""
print(f"│  Filter improvement:         {sign}{upside:.1f}pp                      │")
print("└─────────────────────────────────────────────────────────────┘")

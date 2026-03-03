"""
Option Recommender

Given a directional bias (from SignalGenerator), selects:
  • Option type  — CE (Call) for bullish, PE (Put) for bearish
  • Expiry date  — based on confidence level
  • Strike price — ATM, 1-OTM, or 2-OTM based on confidence
  • Entry range, Stop-loss, Targets, Lot-size

All risk parameters live in config.py.
"""

import logging
from datetime import datetime

import pandas as pd

from config import (
    NIFTY_STRIKE_GAP,
    NIFTY_LOT_SIZE,
    DEFAULT_CAPITAL,
    MAX_RISK_PCT,
    STOP_LOSS_PCT,
    TARGET_1_PCT,
    TARGET_2_PCT,
    MAX_LOTS,
    MIN_DAYS_TO_EXPIRY,
    PUT_SELL_OTM_STEPS,
    PUT_SELL_SL_MULT,
    PUT_SELL_TARGET_PCT,
)

logger = logging.getLogger(__name__)

# Minimum liquidity thresholds
MIN_OI     = 500    # open interest contracts
MIN_VOLUME = 200    # intraday contracts traded


class OptionRecommender:

    def __init__(
        self,
        direction:       str,
        confidence:      str,
        score:           float,
        spot_price:      float,
        options_df:      pd.DataFrame,
        options_summary: dict,
        expiry_dates:    list,
        vix:             float,
        capital:         float = DEFAULT_CAPITAL,
        quality_score:   int   = 50,
        strategy:        str   = "BUY_OPTIONS",
    ):
        self.direction    = direction
        self.confidence   = confidence
        self.score        = score
        self.spot         = spot_price
        self.df           = options_df.copy() if not options_df.empty else pd.DataFrame()
        self.opts_summary = options_summary or {}
        self.expiry_dates = expiry_dates or []
        self.vix          = vix
        self.capital      = capital
        self.quality      = quality_score
        self.strategy     = strategy

        self._rec: dict = {}
        self._build()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _atm(self) -> int:
        return int(round(self.spot / NIFTY_STRIKE_GAP) * NIFTY_STRIKE_GAP)

    def _select_expiry(self) -> datetime | None:
        today = datetime.now().date()
        valid = sorted(
            d for d in self.expiry_dates
            if (d.date() - today).days >= MIN_DAYS_TO_EXPIRY
        )
        if not valid:
            return self.expiry_dates[0] if self.expiry_dates else None

        # HIGH confidence → nearest expiry (max theta leverage)
        # MEDIUM          → 2nd expiry
        # LOW / VERY_LOW  → 3rd expiry (more time)
        idx_map = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "VERY_LOW": 2}
        idx = min(idx_map.get(self.confidence, 1), len(valid) - 1)
        return valid[idx]

    def _select_strike(self, opt_type: str) -> int:
        atm = self._atm()
        # Base OTM steps by confidence
        # HIGH → ATM  |  MEDIUM → 1-OTM  |  LOW → 2-OTM
        base = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "VERY_LOW": 2}.get(self.confidence, 1)

        # IV adjustment: when IV is elevated, options are expensive.
        # Going slightly further OTM reduces cost and manages IV-crush risk.
        iv_rank = self.opts_summary.get("iv_rank", "NORMAL")
        iv_adj  = {"VERY_LOW": 0, "NORMAL": 0, "HIGH": 1, "VERY_HIGH": 2}.get(iv_rank, 0)

        steps = base + iv_adj
        if opt_type == "CE":
            return atm + steps * NIFTY_STRIKE_GAP
        else:
            return atm - steps * NIFTY_STRIKE_GAP

    def _option_row(self, strike: int, opt_type: str, expiry: datetime | None) -> dict:
        if self.df.empty:
            return {}
        mask = self.df["strikePrice"] == strike
        if expiry is not None:
            mask_exp = self.df["expiryDate"] == expiry
            subset = self.df[mask & mask_exp]
            if subset.empty:
                subset = self.df[mask]
        else:
            subset = self.df[mask]
        if subset.empty:
            return {}
        r = subset.iloc[0]
        return {
            "ltp":       float(r.get(f"{opt_type}_LTP",      0)),
            "oi":        int(r.get(f"{opt_type}_OI",         0)),
            "volume":    int(r.get(f"{opt_type}_volume",     0)),
            "iv":        float(r.get(f"{opt_type}_IV",       0)),
            "bid":       float(r.get(f"{opt_type}_bid",      0)),
            "ask":       float(r.get(f"{opt_type}_ask",      0)),
            "change_oi": int(r.get(f"{opt_type}_changeOI",   0)),
        }

    def _liquidity_ok(self, row: dict) -> bool:
        return row.get("oi", 0) >= MIN_OI and row.get("volume", 0) >= MIN_VOLUME

    def _targets(self, ltp: float) -> dict:
        if ltp <= 0:
            return {}
        return {
            "entry":    round(ltp, 1),
            "stop_loss": round(ltp * (1 - STOP_LOSS_PCT), 1),
            "target_1":  round(ltp * (1 + TARGET_1_PCT),  1),
            "target_2":  round(ltp * (1 + TARGET_2_PCT),  1),
        }

    def _lot_info(self, ltp: float) -> dict:
        lot_cost     = ltp * NIFTY_LOT_SIZE if ltp > 0 else 0
        risk_per_lot = lot_cost * STOP_LOSS_PCT
        if risk_per_lot > 0:
            raw_lots = int((self.capital * MAX_RISK_PCT) / risk_per_lot)

            # VIX adjustment — reduce size in volatile environments
            if   self.vix > 25: raw_lots = max(1, raw_lots // 2)      # half size
            elif self.vix > 20: raw_lots = max(1, int(raw_lots * 0.7)) # 70%

            # Quality adjustment — weak signal = smaller bet
            if self.quality < 55:
                raw_lots = max(1, raw_lots // 2)

            max_lots = min(MAX_LOTS, max(1, raw_lots))
        else:
            max_lots = 1
        return {
            "lot_size":             NIFTY_LOT_SIZE,
            "recommended_lots":     max_lots,
            "cost_per_lot":         round(lot_cost, 0),
            "risk_per_lot":         round(risk_per_lot, 0),
            "total_capital_needed": round(lot_cost * max_lots, 0),
            "total_max_risk":       round(risk_per_lot * max_lots, 0),
        }

    # ── Main Build (dispatcher) ───────────────────────────────────────────────

    def _build(self):
        if self.strategy == "SELL_PUTS":
            self._build_sell_put()
        elif self.strategy == "BUY_PUT":
            self._build_buy_put()
        else:
            self._build_buy_option()

    # ── Buy Option Strategy ───────────────────────────────────────────────────

    def _build_buy_option(self):
        if self.direction == "NEUTRAL":
            self._rec = {
                "action": "AVOID",
                "reason": "No clear directional bias — market is neutral or ranging (ADX likely < 20).",
            }
            return

        # Quality gate: reject poor setups before doing any further work
        if self.quality < 40:
            self._rec = {
                "action": "AVOID",
                "reason": (
                    f"Trade quality score too low ({self.quality}/100). "
                    "Signals are conflicting, trend is weak, or options are too expensive. "
                    "Wait for a cleaner setup."
                ),
            }
            return

        opt_type = "CE" if "BULLISH" in self.direction else "PE"
        action   = "BUY CALL (CE)" if opt_type == "CE" else "BUY PUT (PE)"

        expiry   = self._select_expiry()
        strike   = self._select_strike(opt_type)
        row      = self._option_row(strike, opt_type, expiry)
        ltp      = row.get("ltp", 0)

        # If liquidity is thin, try ATM as fallback
        if ltp > 0 and not self._liquidity_ok(row):
            atm_row = self._option_row(self._atm(), opt_type, expiry)
            if atm_row.get("ltp", 0) > 0:
                logger.info("Low liquidity at %d, falling back to ATM.", strike)
                strike = self._atm()
                row    = atm_row
                ltp    = row.get("ltp", 0)

        targets  = self._targets(ltp)
        lot_info = self._lot_info(ltp)

        self._rec = {
            "action":        action,
            "direction":     self.direction,
            "confidence":    self.confidence,
            "score":         round(self.score, 1),
            "quality_score": self.quality,

            "option_type": opt_type,
            "strike":      strike,
            "atm_strike":  self._atm(),
            "expiry":      expiry,

            "ltp":    ltp,
            "iv":     row.get("iv",     0),
            "oi":     row.get("oi",     0),
            "volume": row.get("volume", 0),
            "bid":    row.get("bid",    0),
            "ask":    row.get("ask",    0),

            "entry_range": {
                "low":  round(ltp * 0.97, 1) if ltp else 0,
                "high": round(ltp * 1.03, 1) if ltp else 0,
            },

            "targets":  targets,
            "lot_info": lot_info,

            "liquidity_ok":   self._liquidity_ok(row),
            "data_available": ltp > 0,

            # Context from options chain
            "max_pain":    self.opts_summary.get("max_pain",            0),
            "pcr":         self.opts_summary.get("pcr",                 1),
            "pcr_signal":  self.opts_summary.get("pcr_signal",  "NEUTRAL"),
            "resistance":  self.opts_summary.get("top_call_resistance", []),
            "support":     self.opts_summary.get("top_put_support",     []),
            # IV / straddle info
            "straddle_price":    self.opts_summary.get("straddle_price",    0),
            "expected_move_pct": self.opts_summary.get("expected_move_pct", 0),
        }

    # ── Buy Put Strategy ──────────────────────────────────────────────────────

    def _build_buy_put(self):
        """
        Dedicated BUY PUT (PE) strategy: always buy a put regardless of direction.
        Best for: confirmed downtrend, breakdown below support, bearish divergence,
                  VIX expansion expected, event risk (budget, FOMC, earnings).

        Strike:  ATM for HIGH confidence, ATM-1 for MEDIUM, ATM-2 for LOW.
                 Goes 1 extra step OTM when IV is elevated (reduces IV-crush risk).
        Expiry:  Nearest for HIGH confidence; 2nd/3rd for lower confidence.
        SL:      Exit if PE loses 35% from entry (time decay eating the premium).
        Target:  T1 = +50% premium, T2 = +100% premium (booking in halves).

        Contrarian flag is set when market signal is bullish — a warning
        is shown in the UI but the trade is still recommended if quality >= 40.
        """
        # ── Quality gate ──────────────────────────────────────────────────────
        if self.quality < 40:
            self._rec = {
                "action": "AVOID",
                "reason": (
                    f"Buy quality score too low ({self.quality}/100). "
                    "Trend is too weak, IV is too expensive, or signals are conflicting. "
                    "Wait for a clean bearish setup before buying puts."
                ),
            }
            return

        # Hard AVOID if market is strongly bullish (score > 60) — put will bleed
        if self.score >= 60:
            self._rec = {
                "action": "AVOID",
                "reason": (
                    f"Signal is STRONGLY BULLISH (score {self.score:+.0f}). "
                    "Buying a put here is a very low-probability trade — "
                    "the market has strong upside momentum. Wait for a reversal signal."
                ),
            }
            return

        # ── Contrarian flag: warn but still trade if signal isn't strongly bullish ──
        # score 0..59 = market is neutral or mildly bullish = caution, not block
        contrarian = self.score >= 20   # bullish signal but user wants to buy put

        # ── Strike selection for PUT buying ───────────────────────────────────
        # ATM PE gives the best delta (~0.50) for a conviction bearish trade.
        # Go OTM (cheaper) when confidence is lower or IV is elevated.
        atm   = self._atm()
        steps = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "VERY_LOW": 2}.get(self.confidence, 1)

        # IV adjustment: when IV is elevated, go 1 step further OTM to cut cost
        iv_rank = self.opts_summary.get("iv_rank", "NORMAL")
        iv_adj  = 1 if iv_rank in ("HIGH", "VERY_HIGH") else 0
        steps   = min(3, steps + iv_adj)

        strike  = atm - steps * NIFTY_STRIKE_GAP
        expiry  = self._select_expiry()
        row     = self._option_row(strike, "PE", expiry)
        ltp     = row.get("ltp", 0)

        # Fallback to ATM if selected strike has no liquidity
        if ltp > 0 and not self._liquidity_ok(row):
            atm_row = self._option_row(atm, "PE", expiry)
            if atm_row.get("ltp", 0) > 0:
                logger.info("Low liquidity at %d PE, falling back to ATM %d PE.", strike, atm)
                strike = atm
                row    = atm_row
                ltp    = row.get("ltp", 0)

        targets  = self._targets(ltp)
        lot_info = self._lot_info(ltp)

        self._rec = {
            "action":        "BUY PUT (PE)",
            "direction":     self.direction,
            "confidence":    self.confidence,
            "score":         round(self.score, 1),
            "quality_score": self.quality,
            "contrarian":    contrarian,   # True = market is bullish, user buying put

            "option_type": "PE",
            "strike":      strike,
            "atm_strike":  atm,
            "expiry":      expiry,

            "ltp":    ltp,
            "iv":     row.get("iv",     0),
            "oi":     row.get("oi",     0),
            "volume": row.get("volume", 0),
            "bid":    row.get("bid",    0),
            "ask":    row.get("ask",    0),

            "entry_range": {
                "low":  round(ltp * 0.97, 1) if ltp else 0,
                "high": round(ltp * 1.03, 1) if ltp else 0,
            },

            "targets":  targets,
            "lot_info": lot_info,

            "liquidity_ok":   self._liquidity_ok(row),
            "data_available": ltp > 0,

            # Options chain context
            "max_pain":          self.opts_summary.get("max_pain",            0),
            "pcr":               self.opts_summary.get("pcr",                 1),
            "pcr_signal":        self.opts_summary.get("pcr_signal",  "NEUTRAL"),
            "resistance":        self.opts_summary.get("top_call_resistance", []),
            "support":           self.opts_summary.get("top_put_support",     []),
            "straddle_price":    self.opts_summary.get("straddle_price",    0),
            "expected_move_pct": self.opts_summary.get("expected_move_pct", 0),
        }

    # ── Sell Put Strategy ─────────────────────────────────────────────────────

    def _build_sell_put(self):
        """
        Short Put strategy: sell OTM PE, collect premium, profit from time decay.
        Ideal conditions: market neutral-to-bullish, high IV (rich premium), low ADX
        (ranging = theta burns fast), VIX 15-22.

        Entry:  Sell PE at ATM - (PUT_SELL_OTM_STEPS × 50 pts) below ATM
        SL:     Buy back if premium rises to PUT_SELL_SL_MULT × sold price
        Target: Buy back at PUT_SELL_TARGET_PCT × sold price (keep 70% of premium)
        """
        # ── Quality gate ──────────────────────────────────────────────────────
        if self.quality < 40:
            self._rec = {
                "action": "AVOID",
                "reason": (
                    f"Sell quality score too low ({self.quality}/100). "
                    "Market may be bearish or IV is too thin for meaningful premium. "
                    "Do NOT sell puts into a falling market — wait for stable/rising conditions."
                ),
            }
            return

        atm    = self._atm()
        strike = atm - PUT_SELL_OTM_STEPS * NIFTY_STRIKE_GAP
        expiry = self._select_expiry()
        row    = self._option_row(strike, "PE", expiry)
        ltp    = row.get("ltp", 0)

        # Fallback: if no data at target strike, try one step closer to ATM
        if not row or ltp == 0:
            fallback_strike = atm - NIFTY_STRIKE_GAP
            fb_row = self._option_row(fallback_strike, "PE", expiry)
            if fb_row.get("ltp", 0) > 0:
                logger.info("No data at %d PE, falling back to %d.", strike, fallback_strike)
                strike = fallback_strike
                row    = fb_row
                ltp    = row.get("ltp", 0)

        # ── Buffer adequacy check ─────────────────────────────────────────────
        # The straddle price = market's implied expected move by expiry.
        # If (ATM - sold_strike) < implied_move, the strike is WITHIN the expected
        # range — it will be breached ~50% of the time. Force one step wider OTM.
        straddle      = self.opts_summary.get("straddle_price", 0)
        buffer_pts    = atm - strike
        buffer_safe   = True
        if straddle > 0 and buffer_pts < straddle * 0.85:
            wider_strike = strike - NIFTY_STRIKE_GAP
            wider_row    = self._option_row(wider_strike, "PE", expiry)
            if wider_row.get("ltp", 0) > 0:
                logger.info(
                    "Buffer %d pts < 85%% of straddle %d — widening strike from %d to %d PE.",
                    buffer_pts, straddle, strike, wider_strike,
                )
                strike      = wider_strike
                row         = wider_row
                ltp         = row.get("ltp", 0)
                buffer_pts  = atm - strike
                buffer_safe = False   # flag: was widened for safety

        # ── Prices ────────────────────────────────────────────────────────────
        sl_price     = round(ltp * PUT_SELL_SL_MULT,    1) if ltp else 0
        target_price = round(ltp * PUT_SELL_TARGET_PCT, 1) if ltp else 0

        # ── Position sizing ───────────────────────────────────────────────────
        # Risk per lot = (SL price - sell price) × lot size
        risk_per_lot = round((sl_price - ltp) * NIFTY_LOT_SIZE, 0) if ltp else 0
        if risk_per_lot > 0:
            raw_lots = int((self.capital * MAX_RISK_PCT) / risk_per_lot)
            if   self.vix > 25: raw_lots = max(1, raw_lots // 2)
            elif self.vix > 20: raw_lots = max(1, int(raw_lots * 0.7))
            if self.quality < 55: raw_lots = max(1, raw_lots // 2)
            lots = min(MAX_LOTS, max(1, raw_lots))
        else:
            lots = 1

        # ── INR P&L ───────────────────────────────────────────────────────────
        # Premium collected when you sell
        premium_collected = round(ltp    * NIFTY_LOT_SIZE * lots, 0) if ltp else 0
        # Max profit = what you keep when you buy back at target
        max_profit        = round((ltp - target_price) * NIFTY_LOT_SIZE * lots, 0) if ltp else 0
        # Max loss = what you pay extra at SL
        max_loss          = round((sl_price - ltp) * NIFTY_LOT_SIZE * lots, 0)     if ltp else 0
        # Margin: ~12% of notional (strike × lot_size per lot) — approximate SPAN
        margin_per_lot    = round(strike * NIFTY_LOT_SIZE * 0.12, 0)
        total_margin      = round(margin_per_lot * lots, 0)

        # If Nifty spot falls to strike, puts explode — use as alert level
        spot_alert = strike  # "If Nifty falls to this level, watch closely"

        self._rec = {
            "action":        "SELL PUT",
            "direction":     self.direction,
            "confidence":    self.confidence,
            "score":         round(self.score, 1),
            "quality_score": self.quality,

            "option_type": "PE",
            "strike":      strike,
            "atm_strike":  atm,
            "expiry":      expiry,

            "ltp":    ltp,
            "iv":     row.get("iv",     0),
            "oi":     row.get("oi",     0),
            "volume": row.get("volume", 0),
            "bid":    row.get("bid",    0),
            "ask":    row.get("ask",    0),

            # Sell PUT specific fields
            "sell_at":            ltp,
            "sl_buy_at":          sl_price,
            "target_buy_at":      target_price,
            "spot_alert":         spot_alert,
            "premium_collected":  premium_collected,
            "max_profit":         max_profit,
            "max_loss":           max_loss,
            "recommended_lots":   lots,
            "margin_per_lot":     margin_per_lot,
            "total_margin":       total_margin,

            # Buffer safety vs implied expected move
            "buffer_pts":   buffer_pts,
            "buffer_ratio": round(buffer_pts / straddle, 2) if straddle > 0 else 0,
            "buffer_safe":  buffer_safe,   # False = strike was auto-widened for safety

            "liquidity_ok":   self._liquidity_ok(row),
            "data_available": ltp > 0,

            # Context from options chain
            "max_pain":    self.opts_summary.get("max_pain",            0),
            "pcr":         self.opts_summary.get("pcr",                 1),
            "pcr_signal":  self.opts_summary.get("pcr_signal",  "NEUTRAL"),
            "resistance":  self.opts_summary.get("top_call_resistance", []),
            "support":     self.opts_summary.get("top_put_support",     []),
            "straddle_price":    self.opts_summary.get("straddle_price",    0),
            "expected_move_pct": self.opts_summary.get("expected_move_pct", 0),
        }

    def get_recommendation(self) -> dict:
        return self._rec

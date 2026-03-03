"""
Signal Generator
================
Combines four data streams into a single score on [-100, +100]:

  +100 → extremely bullish
     0 → neutral
  -100 → extremely bearish

Weight breakdown (normal mode)
  50 % Technical analysis  (EMA, RSI, MACD, BB, Supertrend, ADX, OBV)
  30 % Options analysis    (PCR, Max Pain, OI change)
  20 % News & Global mkts  (events, S&P, Nikkei, crude, VIX, USD/INR)

Two multipliers applied to the raw score:
  • India VIX multiplier  — high VIX shrinks confidence (expensive options)
  • ADX multiplier        — low ADX (ranging market) shrinks score toward zero
                            This is the #1 filter: don't trade ranging markets!

Crisis mode: when a major event is detected (war, crash, ≥60-point keyword),
news weight jumps to 50%, overriding the technical picture.

Quality Score (0-100):
  >= 70 → STRONG  — high conviction, take the trade
  55-69 → GOOD    — solid setup, trade normal size
  40-54 → WEAK    — marginal setup, halve position or skip
   < 40 → AVOID   — conditions too poor, no trade today
"""

import logging

logger = logging.getLogger(__name__)

# ── Direction thresholds ──────────────────────────────────────────────────────
STRONG_BULLISH =  60
BULLISH        =  30
BEARISH        = -30
STRONG_BEARISH = -60


class SignalGenerator:

    def __init__(
        self,
        technical: dict,
        options:   dict,
        vix:       float,
        news:      dict | None = None,
    ):
        self.tech = technical or {}
        self.opts = options   or {}
        self.vix  = vix
        self.news = news      or {}

        self.score     = 0.0
        self.breakdown = {}

        self._compute()

    # ── Orchestrator ─────────────────────────────────────────────────────────

    def _compute(self):
        tech_score    = self._tech_score()     # also sets self.breakdown["tech_signals"]
        tech_sigs     = self.breakdown.get("tech_signals", {})
        options_score = self._options_score()  # also sets self.breakdown["options_signals"]
        opts_sigs     = self.breakdown.get("options_signals", {})
        news_score    = self._news_score()
        vix_mult      = self._vix_multiplier()
        adx_mult      = self._adx_multiplier()

        # Crisis mode boosts news to 50% so it dominates
        has_major = self.news.get("has_major_event", False)
        if has_major:
            news_weight = 0.50
            tech_weight = 0.30
            opts_weight = 0.20
        else:
            news_weight = 0.20
            tech_weight = 0.50
            opts_weight = 0.30

        raw = (tech_score    * tech_weight
               + options_score * opts_weight
               + news_score    * news_weight)

        # Apply both VIX and ADX multipliers
        self.score = max(-100.0, min(100.0, raw * vix_mult * adx_mult))

        self.breakdown = {
            "technical_score":   round(tech_score,    1),
            "options_score":     round(options_score, 1),
            "news_score":        round(news_score,    1),
            "vix_multiplier":    round(vix_mult,      2),
            "adx_multiplier":    round(adx_mult,      2),
            "final_score":       round(self.score,    1),
            "major_event":       has_major,
            "tech_signals":      tech_sigs,
            "options_signals":   opts_sigs,
            "weights": {
                "technical": tech_weight,
                "options":   opts_weight,
                "news":      news_weight,
            },
        }

    # ── Technical Signals ────────────────────────────────────────────────────

    def _tech_score(self) -> float:
        t = self.tech
        if not t:
            return 0.0

        signals = {}

        # 1. EMA Trend (max ±40 pts) — the backbone of trend direction
        ema = 0
        ema +=  20 if t.get("above_ema200")       else -20
        ema +=  10 if t.get("above_ema50")        else -10
        ema +=   7 if t.get("above_ema20")        else  -7
        ema +=   3 if t.get("ema20_above_ema50")  else  -3
        signals["EMA Trend"] = ema

        # 2. RSI (max ±25 pts) — momentum / overbought / oversold
        rsi = t.get("rsi", 50)
        if   rsi < 30:  rsi_s =  25   # oversold → bounce likely
        elif rsi < 40:  rsi_s =  15
        elif rsi < 45:  rsi_s =   8
        elif rsi < 55:  rsi_s =   0
        elif rsi < 60:  rsi_s =  -8
        elif rsi < 70:  rsi_s = -15
        else:           rsi_s = -25   # overbought → correction likely
        signals["RSI"] = rsi_s

        # 3. MACD (max ±25 pts) — trend + momentum crossover
        if   t.get("macd_crossover"):  macd_s =  25   # fresh buy signal
        elif t.get("macd_crossunder"): macd_s = -25   # fresh sell signal
        elif t.get("macd_bullish"):
            macd_s = 15 if t.get("macd_hist", 0) > 0 else 5
        else:
            macd_s = -15 if t.get("macd_hist", 0) < 0 else -5
        signals["MACD"] = macd_s

        # 4. Bollinger Band Position (max ±10 pts) — mean reversion signal
        bp = t.get("bb_position", 0.5)
        if   bp < 0.10: bb_s =  10   # near lower band, oversold
        elif bp < 0.20: bb_s =   5
        elif bp > 0.90: bb_s = -10   # near upper band, overbought
        elif bp > 0.80: bb_s =  -5
        else:           bb_s =   0
        signals["Bollinger"] = bb_s

        # 5. Supertrend (max ±15 pts) — definitive trend-following signal
        st_bull = t.get("st_bullish")
        if   st_bull is True:  signals["Supertrend"] =  15
        elif st_bull is False: signals["Supertrend"] = -15
        else:                  signals["Supertrend"] =   0

        # 6. ADX Directional (max ±10 pts) — which way is the strong trend going?
        if t.get("adx_trending"):   # ADX >= 20 (trend confirmed)
            signals["ADX_Dir"] = 10 if t.get("di_bullish") else -10
        else:
            signals["ADX_Dir"] = 0   # no trend → neutral

        # 7. OBV (max ±10 pts) — volume confirms the price move
        obv_up = t.get("obv_rising")
        if   obv_up is True:  signals["OBV"] =  10
        elif obv_up is False: signals["OBV"] = -10
        else:                 signals["OBV"] =   0

        # 8. Volume Surge (max ±15 pts) — high-volume moves are far more reliable
        # A big green candle on 2× average volume is much stronger than one on 0.5× volume.
        vol_ratio   = t.get("vol_ratio", 1.0)
        bullish_day = t.get("close", 0) > t.get("open", 0)
        if vol_ratio >= 2.0:
            signals["Volume"] = 15 if bullish_day else -15   # conviction breakout/breakdown
        elif vol_ratio >= 1.5:
            signals["Volume"] =  8 if bullish_day else  -8   # above-average confirmation
        elif vol_ratio < 0.5:
            signals["Volume"] = -5                           # suspiciously thin — unreliable
        else:
            signals["Volume"] =  0

        # 9. Consecutive Day Streak (max ±10 pts) — mean reversion from extended runs
        # 4+ up days in a row = overbought, mean reversion risk (negative signal).
        # 3+ down days in a row = oversold, bounce likely (positive signal).
        streak = t.get("streak", 0)
        if   streak >= 5:  signals["Streak"] = -10
        elif streak >= 4:  signals["Streak"] =  -6
        elif streak >= 3:  signals["Streak"] =  -3
        elif streak <= -4: signals["Streak"] =  10
        elif streak <= -3: signals["Streak"] =   6
        elif streak <= -2: signals["Streak"] =   3
        else:              signals["Streak"] =   0

        self.breakdown["tech_signals"] = signals
        return float(sum(signals.values()))

    # ── Options Signals ───────────────────────────────────────────────────────

    def _options_score(self) -> float:
        o = self.opts
        if not o:
            return 0.0

        signals = {}

        # 1. PCR contrarian (max ±50 pts)
        pcr_map = {
            "STRONG_BULLISH":  50,
            "BULLISH":         30,
            "NEUTRAL":          0,
            "BEARISH":        -30,
            "STRONG_BEARISH": -50,
        }
        signals["PCR"] = pcr_map.get(o.get("pcr_signal", "NEUTRAL"), 0)

        # 2. Max Pain vs Spot (max ±30 pts, scaled up to ±45 near expiry)
        # Max pain gravity is strongest on expiry day and weakens as DTE increases.
        diff = o.get("max_pain_diff_pct", 0)
        if   diff >  2.0: mp_s =  30
        elif diff >  1.0: mp_s =  15
        elif diff > -1.0: mp_s =   0
        elif diff > -2.0: mp_s = -15
        else:             mp_s = -30

        dte = o.get("days_to_expiry", 7)
        if   dte <= 0: mp_scale = 1.8   # expiry day: very strong pin
        elif dte <= 1: mp_scale = 1.5   # 1 day out: strong gravity
        elif dte <= 3: mp_scale = 1.2   # 2-3 days: meaningful pull
        elif dte <= 7: mp_scale = 1.0   # within the week: normal
        else:          mp_scale = 0.7   # far from expiry: weak signal

        signals["Max Pain"] = max(-45, min(45, round(mp_s * mp_scale)))

        # 3. OI change direction — volume-weighted (max ±20 pts)
        # The SIZE of the OI build matters far more than how many strikes moved.
        # Large put OI build = heavy hedging = contrarian bullish (like a high PCR).
        # Large call OI build = resistance being written = mildly bearish.
        total_new_put_oi  = o.get("total_new_put_oi",  0)
        total_new_call_oi = o.get("total_new_call_oi", 0)
        total_oi          = total_new_put_oi + total_new_call_oi
        if total_oi > 0:
            # ratio 1.0 = all puts, 0.0 = all calls, 0.5 = equal → maps to ±20
            ratio = total_new_put_oi / total_oi
            oi_s  = max(-20, min(20, round((ratio - 0.5) * 40)))
        else:
            # Fallback: count-based (when market is closed / data unavailable)
            np_ = len(o.get("new_put_oi_strikes",  []))
            nc_ = len(o.get("new_call_oi_strikes", []))
            oi_s = max(-20, min(20, (np_ - nc_) * 5))
        signals["OI Change"] = oi_s

        self.breakdown["options_signals"] = signals
        return float(sum(signals.values()))

    # ── News / Global Market Signals ─────────────────────────────────────────

    def _news_score(self) -> float:
        """Use the pre-computed event_score from NewsAnalyzer."""
        return float(self.news.get("event_score", 0.0))

    # ── ADX Multiplier ────────────────────────────────────────────────────────

    def _adx_multiplier(self) -> float:
        """
        THE most important filter: ADX below 20 = ranging market.
        In a ranging market, directional options decay without moving.
        Scale the score toward zero when there's no clear trend.
        """
        adx = self.tech.get("adx", 20)
        if   adx < 15:  return 0.20   # flat/sideways — almost force NEUTRAL
        elif adx < 20:  return 0.55   # weak trend — significant discount
        elif adx < 25:  return 0.80   # building trend
        else:           return 1.00   # strong trend — full signal

    # ── VIX Multiplier ────────────────────────────────────────────────────────

    def _vix_multiplier(self) -> float:
        v = self.vix
        if   v < 12:  return 0.85   # too calm = complacency
        elif v < 15:  return 1.00   # ideal environment
        elif v < 20:  return 0.90   # slightly elevated
        elif v < 25:  return 0.75   # high fear, reduce conviction
        else:         return 0.60   # extreme fear, low confidence

    # ── Quality Score ─────────────────────────────────────────────────────────

    def _quality_score(self) -> int:
        """
        Composite 0-100 score measuring HOW GOOD today's trade setup is.
        Considers: trend strength, signal strength, IV cost, VIX environment,
        and how many indicators agree with each other.

        >= 70 = STRONG  (high conviction — trade with full size)
        55-69 = GOOD    (solid setup — trade normal size)
        40-54 = WEAK    (marginal — reduce size or skip)
         < 40 = AVOID   (poor conditions — sit out today)
        """
        pts = 0
        t, o = self.tech, self.opts

        # ── 1. Trend Strength via ADX (0-30 pts) ─────────────────────────────
        # Strong trend = high quality. No trend = waste of premium.
        adx = t.get("adx", 15)
        if   adx >= 30: pts += 30
        elif adx >= 25: pts += 22
        elif adx >= 20: pts += 14
        elif adx >= 15: pts += 6
        # else: 0 (flat market, no quality)

        # ── 2. Signal Strength (0-25 pts) ────────────────────────────────────
        # After ADX and VIX multipliers, how strong is the final score?
        a = abs(self.score)
        if   a >= 60: pts += 25
        elif a >= 45: pts += 18
        elif a >= 30: pts += 11
        elif a >= 20: pts += 5
        # else: 0

        # ── 3. IV Cheapness — Are Options Cheap to Buy? (0-20 pts) ──────────
        # Buying options when IV is low = better value, more room to expand.
        # Buying when IV is high = expensive, risk of IV crush after event.
        iv_rank = o.get("iv_rank", "NORMAL")
        pts += {"VERY_LOW": 20, "NORMAL": 15, "HIGH": 5, "VERY_HIGH": 0}.get(iv_rank, 12)

        # ── 4. VIX Environment (0-15 pts) ────────────────────────────────────
        # Calm VIX = predictable market. Spiking VIX = uncertainty.
        v = self.vix
        if   v < 13:  pts += 13
        elif v < 15:  pts += 15   # sweet spot — calm but enough movement
        elif v < 18:  pts += 10
        elif v < 22:  pts += 4
        # else: 0 (too volatile)

        # ── 5. Indicator Confluence (0-10 pts) ───────────────────────────────
        # More indicators agreeing = higher quality signal.
        is_bull = self.score > 0
        bull_agree = [
            t.get("above_ema200",  False),
            t.get("macd_bullish",  False),
            t.get("st_bullish",    False),
            t.get("di_bullish",    False),
            o.get("pcr_signal", "") in ("BULLISH", "STRONG_BULLISH"),
        ]
        bear_agree = [
            not t.get("above_ema200",  True),
            not t.get("macd_bullish",  True),
            not t.get("st_bullish",    True),
            not t.get("di_bullish",    True),
            o.get("pcr_signal", "") in ("BEARISH", "STRONG_BEARISH"),
        ]
        agree_list  = bull_agree if is_bull else bear_agree
        agree_count = sum(1 for f in agree_list if f)
        pts += min(10, agree_count * 2)

        return min(100, max(0, pts))

    # ── Sell Quality Score (short put strategy) ──────────────────────────────

    def _sell_quality_score(self) -> int:
        """
        Quality score specifically for the SHORT PUT (sell PE) strategy.

        Key insight: SELL PUT loses money when Nifty FALLS sharply below the sold
        strike. So this score measures "how safe is it that Nifty will NOT fall?"
        — not just "is the market bullish?".

        New signals added:
          RSI Zone: RSI < 30 (falling market) = danger, even if technically "oversold".
          Straddle Adequacy: if market expects big moves (large straddle), selling
            is risky regardless of trend. This directly catches the "buffer too thin"
            problem that the old score ignored entirely.

        >= 70 → STRONG  (great setup to sell)
        55-69 → GOOD
        40-54 → WEAK  (thin premium or risky environment)
         < 40 → AVOID (market likely to fall — do NOT sell puts)
        """
        pts = 0
        t, o = self.tech, self.opts

        # ── 0. Hard block: bearish signal → do not sell puts ─────────────────
        if self.score <= -30:
            return 0   # caller will mark as AVOID

        # ── 1. Trend environment (0-25 pts) ───────────────────────────────────
        # Ranging market (low ADX) = premium decays without a big directional move.
        # Strong downtrend (high ADX + DI- > DI+) = puts will skyrocket = dangerous.
        adx = t.get("adx", 15)
        if adx < 15:
            pts += 25   # flat/sideways = perfect for selling
        elif adx < 20:
            pts += 20   # mild ranging = great
        elif adx < 25:
            pts += 12   # building trend = OK but watch out
        elif t.get("di_bullish", False):
            pts += 15   # strong uptrend = safe (market moving away from sold strike)
        else:
            pts += 0    # strong DOWNtrend = very dangerous for put sellers

        # ── 2. IV Richness (0-25 pts) — HIGH IV = more premium to collect ─────
        iv_rank = o.get("iv_rank", "NORMAL")
        pts += {"VERY_LOW": 0, "NORMAL": 12, "HIGH": 20, "VERY_HIGH": 25}.get(iv_rank, 12)

        # ── 3. VIX environment (0-15 pts) ─────────────────────────────────────
        v = self.vix
        if   15 <= v <= 22: pts += 15   # sweet spot: premium elevated, not panic
        elif v < 15:        pts += 10   # calm VIX = lower premium but stable
        elif v < 25:        pts += 8    # slightly elevated = caution
        else:               pts += 0    # extreme VIX = puts can explode = avoid

        # ── 4. Signal direction (0-15 pts) ────────────────────────────────────
        if   self.score >= 20:  pts += 15   # clearly bullish = very safe
        elif self.score >= 0:   pts += 10   # neutral-bullish = OK
        elif self.score >= -15: pts += 5    # mildly negative = risky

        # ── 5. PCR: high put OI = put premiums elevated (0-5 pts) ────────────
        pcr_sig = o.get("pcr_signal", "NEUTRAL")
        if pcr_sig in ("STRONG_BULLISH", "BULLISH"):
            pts += 5    # lots of puts = market has a floor + premiums are rich
        elif pcr_sig == "NEUTRAL":
            pts += 2

        # ── 6. RSI zone check (−25 to +10 pts) ── REVISED ──────────────────────
        # For put selling, RSI must be read BOTH ways:
        #   RSI < 30: market falling hard → puts will skyrocket → AVOID
        #   RSI > 70: overbought → correction is overdue → puts at risk
        #   RSI 40-65: neutral/mild bullish → ideal for selling premium
        # The worst inputs: RSI 90+ means blow-off top, corrections are violent.
        rsi = t.get("rsi", 50)
        if   40 <= rsi <= 60:  pts += 10   # neutral = ideal for selling
        elif 60 < rsi <= 65:   pts +=  5   # mildly bullish = OK
        elif 65 < rsi <= 70:   pts -=  5   # mildly overbought = caution
        elif 70 < rsi <= 80:   pts -= 15   # overbought = correction risk = danger
        elif rsi > 80:         pts -= 25   # extremely overbought = blow-off = AVOID
        elif 30 <= rsi < 40:   pts -=  5   # approaching oversold = caution
        else:                  pts -= 15   # RSI < 30 = market declining hard = danger

        # ── 7. Expected move adequacy (−20 to +10 pts) ── NEW ─────────────────
        # The ATM straddle tells us what the market EXPECTS Nifty to move by expiry.
        # If expected_move_pct is large, the sold strike (OTM) is at risk of being
        # breached. This was completely missing from the original score.
        #
        # Example: Nifty at 24000, straddle = 400 pts (1.67% move expected).
        # Selling 23800 PE (100 pts OTM) = strike is WITHIN the expected move range.
        # That's a ~50% probability of being breached — unacceptable for put selling.
        exp_move_pct = o.get("expected_move_pct", 1.5)
        if   exp_move_pct < 0.8: pts += 10   # tiny expected move = very safe
        elif exp_move_pct < 1.2: pts +=  5   # moderate = good
        elif exp_move_pct < 1.5: pts +=  0   # neutral
        elif exp_move_pct < 2.0: pts -= 10   # large expected move = risky
        else:                    pts -= 20   # very large move expected = avoid

        # ── 8. EMA50 position (−30 to +10 pts) — downtrend block + overextension ─
        # Price below EMA50 = in correction = dangerous for put selling.
        # Price TOO FAR above EMA50 (>2.5%) = overextended = correction due.
        # The sweet spot for selling puts: 0% to +2% above EMA50.
        close_vs_ema50 = t.get("close_vs_ema50_pct", 0.0)
        if   close_vs_ema50 >= 3.0:  pts -= 15   # overextended = mean reversion risk
        elif close_vs_ema50 >= 1.5:  pts +=  5   # comfortably above = safe
        elif close_vs_ema50 >= 0.0:  pts += 10   # sweet spot = ideal for selling
        elif close_vs_ema50 >= -0.5: pts -= 10   # just below EMA50 = caution
        elif close_vs_ema50 >= -1.5: pts -= 20   # in correction = risky
        else:                        pts -= 30   # deep correction = AVOID

        # ── 9. EMA50 slope (−10 to +8 pts) — is the medium-term trend rising? ──
        # Even if price is above EMA50, a declining EMA50 means the trend is
        # deteriorating. A rising EMA50 with price above it = ideal for selling puts.
        ema50_slope = t.get("ema50_slope_pct", 0.0)   # 5-day EMA50 change as %
        if   ema50_slope >= 0.15: pts +=  8   # clearly rising = uptrend intact
        elif ema50_slope >= 0.05: pts +=  4   # slightly rising = OK
        elif ema50_slope >= -0.05: pts += 0   # flat = neutral
        elif ema50_slope >= -0.15: pts -= 5   # declining = caution
        else:                      pts -= 10  # falling EMA50 = downtrend = danger

        # ── 10. 20-day momentum (−30 to +5 pts) — is market in multi-week decline? ─
        # If Nifty has fallen >2% over the past 20 days, it's in a correction.
        # The most dangerous period for put sellers is NOT a single-day crash
        # but a slow, sustained multi-week decline where every "bounce" gets sold.
        momentum_20d = t.get("momentum_20d_pct", 0.0)
        if   momentum_20d >= 3.0:  pts +=  5   # strong multi-week rally = very safe
        elif momentum_20d >= 0.0:  pts +=  0   # flat/slight rise = neutral
        elif momentum_20d >= -2.0: pts -= 10   # modest pullback = caution
        elif momentum_20d >= -5.0: pts -= 20   # correction mode = risky
        else:                      pts -= 30   # major correction / crash = AVOID

        return min(100, max(0, pts))

    # ── Public API ────────────────────────────────────────────────────────────

    def get_direction(self) -> str:
        s = self.score
        if   s >= STRONG_BULLISH: return "STRONG_BULLISH"
        elif s >= BULLISH:        return "BULLISH"
        elif s <= STRONG_BEARISH: return "STRONG_BEARISH"
        elif s <= BEARISH:        return "BEARISH"
        else:                     return "NEUTRAL"

    def get_confidence(self) -> str:
        a = abs(self.score)
        if   a >= 70: return "HIGH"
        elif a >= 50: return "MEDIUM"
        elif a >= 30: return "LOW"
        else:         return "VERY_LOW"

    def get_full_analysis(self) -> dict:
        quality = self._quality_score()
        if   quality >= 70: q_label = "STRONG"
        elif quality >= 55: q_label = "GOOD"
        elif quality >= 40: q_label = "WEAK"
        else:               q_label = "AVOID"

        sell_quality = self._sell_quality_score()
        if   sell_quality >= 70: sq_label = "STRONG"
        elif sell_quality >= 65: sq_label = "GOOD"    # raised from 55 → 65 for accuracy
        elif sell_quality >= 40: sq_label = "WEAK"
        else:                    sq_label = "AVOID"

        return {
            "score":              round(self.score, 1),
            "direction":          self.get_direction(),
            "confidence":         self.get_confidence(),
            "vix":                self.vix,
            "breakdown":          self.breakdown,
            "quality_score":      quality,
            "quality_label":      q_label,
            "sell_quality_score": sell_quality,
            "sell_quality_label": sq_label,
        }

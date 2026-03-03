"""
Technical Analysis Engine

Computes: EMA 20/50/200, RSI, MACD, Bollinger Bands, ATR,
          ADX + DI+/DI- (trend strength), Supertrend(14,3),
          OBV (volume trend), Pivot Points, Volume analysis.
All constants are self-contained so this module runs standalone.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
RSI_PERIOD         = 14
MACD_FAST          = 12
MACD_SLOW          = 26
MACD_SIGNAL        = 9
EMA_SHORT          = 20
EMA_MEDIUM         = 50
EMA_LONG           = 200
BB_PERIOD          = 20
BB_STD             = 2
ATR_PERIOD         = 14
ADX_PERIOD         = 14       # Wilder's ADX
SUPERTREND_PERIOD  = 14       # uses ATR_PERIOD ATR
SUPERTREND_MULT    = 3.0      # standard multiplier


class TechnicalAnalysis:
    """Compute and expose every technical indicator needed for signal scoring."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._pivot_points: dict = {}
        if not self.df.empty and len(self.df) >= 30:
            self._compute_all()

    # ── Private: compute everything ──────────────────────────────────────────

    def _compute_all(self):
        self._emas()
        self._rsi()
        self._macd()
        self._bollinger()
        self._atr()          # must be before supertrend + adx
        self._adx()
        self._supertrend()
        self._obv()
        self._volume()
        self._pivots()
        self._streak()

    def _emas(self):
        c = self.df["Close"]
        self.df["EMA20"]  = c.ewm(span=EMA_SHORT,  adjust=False).mean()
        self.df["EMA50"]  = c.ewm(span=EMA_MEDIUM, adjust=False).mean()
        self.df["EMA200"] = c.ewm(span=EMA_LONG,   adjust=False).mean()

    def _rsi(self):
        delta = self.df["Close"].diff()
        gain  = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
        loss  = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
        rs    = gain / loss.replace(0, np.nan)
        self.df["RSI"] = 100 - (100 / (1 + rs))

    def _macd(self):
        c = self.df["Close"]
        ema_f = c.ewm(span=MACD_FAST,   adjust=False).mean()
        ema_s = c.ewm(span=MACD_SLOW,   adjust=False).mean()
        self.df["MACD"]        = ema_f - ema_s
        self.df["MACD_Signal"] = self.df["MACD"].ewm(span=MACD_SIGNAL, adjust=False).mean()
        self.df["MACD_Hist"]   = self.df["MACD"] - self.df["MACD_Signal"]

    def _bollinger(self):
        c = self.df["Close"]
        sma  = c.rolling(BB_PERIOD).mean()
        std  = c.rolling(BB_PERIOD).std()
        self.df["BB_Upper"]  = sma + BB_STD * std
        self.df["BB_Middle"] = sma
        self.df["BB_Lower"]  = sma - BB_STD * std
        band_width = (self.df["BB_Upper"] - self.df["BB_Lower"]).replace(0, np.nan)
        self.df["BB_Position"] = (c - self.df["BB_Lower"]) / band_width
        self.df["BB_Width"]    = band_width / sma

    def _atr(self):
        h, l, c = self.df["High"], self.df["Low"], self.df["Close"]
        prev_c = c.shift(1)
        tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        self.df["ATR"] = tr.rolling(ATR_PERIOD).mean()

    # ── NEW: ADX + DI+ / DI- ────────────────────────────────────────────────

    def _adx(self):
        """
        Average Directional Index (ADX) using Wilder's smoothing.
        ADX >= 25 → strong trend (take trades)
        ADX 20–25 → moderate trend
        ADX < 20  → ranging / no trend (avoid directional options)
        DI+ > DI- → bullish trend direction
        DI- > DI+ → bearish trend direction
        """
        n   = ADX_PERIOD
        h, l, c = self.df["High"], self.df["Low"], self.df["Close"]
        prev_c  = c.shift(1)

        # True Range
        tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)

        # Directional Movement
        up   = h.diff()
        down = -l.diff()
        dm_p = up.where((up > down) & (up > 0), 0.0)
        dm_m = down.where((down > up) & (down > 0), 0.0)

        # Wilder's smoothing = EWM with alpha=1/n
        alpha = 1.0 / n
        tr_w  = tr.ewm(alpha=alpha,  adjust=False).mean()
        dmp_w = dm_p.ewm(alpha=alpha, adjust=False).mean()
        dmm_w = dm_m.ewm(alpha=alpha, adjust=False).mean()

        di_p = (100 * dmp_w / tr_w.replace(0, np.nan)).fillna(0)
        di_m = (100 * dmm_w / tr_w.replace(0, np.nan)).fillna(0)
        dx   = (100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)).fillna(0)

        self.df["ADX"]      = dx.ewm(alpha=alpha, adjust=False).mean().fillna(15)
        self.df["DI_Plus"]  = di_p
        self.df["DI_Minus"] = di_m

    # ── NEW: Supertrend ──────────────────────────────────────────────────────

    def _supertrend(self):
        """
        Supertrend indicator — clean trend-following buy/sell signal.
        Price above Supertrend = bullish (buy calls).
        Price below Supertrend = bearish (buy puts).
        """
        mult  = SUPERTREND_MULT
        mid   = (self.df["High"] + self.df["Low"]) / 2.0
        atr   = self.df["ATR"]

        bu_s = mid + mult * atr   # basic upper (Series)
        bl_s = mid - mult * atr   # basic lower (Series)

        bu = bu_s.values
        bl = bl_s.values
        c  = self.df["Close"].values
        n  = len(c)

        fu = np.full(n, np.nan)
        fl = np.full(n, np.nan)
        st = np.full(n, np.nan)

        # Find first index where ATR is valid
        valid_idx = np.where(~np.isnan(atr.values))[0]
        if len(valid_idx) == 0:
            self.df["Supertrend"] = np.nan
            self.df["ST_Bullish"] = False
            return

        s = valid_idx[0]
        fu[s] = bu[s]
        fl[s] = bl[s]
        st[s] = bu[s]   # start in "bearish" mode (price below upper)

        for i in range(s + 1, n):
            fu[i] = bu[i] if (bu[i] < fu[i-1] or c[i-1] > fu[i-1]) else fu[i-1]
            fl[i] = bl[i] if (bl[i] > fl[i-1] or c[i-1] < fl[i-1]) else fl[i-1]

            prev = st[i-1]
            if np.isnan(prev) or prev == fu[i-1]:
                st[i] = fu[i] if c[i] <= fu[i] else fl[i]
            else:
                st[i] = fl[i] if c[i] >= fl[i] else fu[i]

        self.df["Supertrend"] = st
        # True where price is above Supertrend (bullish)
        self.df["ST_Bullish"] = np.where(~np.isnan(st), c > st, False)

    # ── NEW: OBV (On Balance Volume) ────────────────────────────────────────

    def _obv(self):
        """
        On Balance Volume — confirms price moves with volume.
        Rising OBV with rising price = bullish (institutions accumulating).
        Falling OBV with rising price = bearish divergence (warning signal).
        """
        delta = self.df["Close"].diff()
        sign  = np.where(delta > 0, 1, np.where(delta < 0, -1, 0))
        self.df["OBV"]     = (self.df["Volume"] * sign).cumsum()
        self.df["OBV_SMA"] = self.df["OBV"].rolling(20).mean()

    def _volume(self):
        v = self.df["Volume"]
        self.df["Vol_SMA20"] = v.rolling(20).mean()
        self.df["Vol_Ratio"] = v / self.df["Vol_SMA20"].replace(0, np.nan)

    def _pivots(self):
        """Classic floor-pivot points from the previous session."""
        if len(self.df) < 2:
            return
        prev = self.df.iloc[-2]
        H, L, C = prev["High"], prev["Low"], prev["Close"]
        PP = (H + L + C) / 3
        self._pivot_points = {
            "PP": PP,
            "R1": 2 * PP - L,
            "R2": PP + (H - L),
            "R3": H + 2 * (PP - L),
            "S1": 2 * PP - H,
            "S2": PP - (H - L),
            "S3": L - 2 * (H - PP),
        }

    def _streak(self):
        """
        Count consecutive up/down days.
        +N = N consecutive days where close > previous close (overbought risk).
        -N = N consecutive days where close < previous close (oversold / bounce likely).
        A flat day (no change) resets the streak to 0.
        """
        c = self.df["Close"].values
        n = len(c)
        streak = np.zeros(n, dtype=int)
        for i in range(1, n):
            if c[i] > c[i - 1]:
                streak[i] = streak[i - 1] + 1 if streak[i - 1] > 0 else 1
            elif c[i] < c[i - 1]:
                streak[i] = streak[i - 1] - 1 if streak[i - 1] < 0 else -1
            # else: flat day → streak[i] stays 0
        self.df["Streak"] = streak

    # ── Public API ────────────────────────────────────────────────────────────

    def get_current_values(self) -> dict:
        """Return a flat dict of the latest indicator values."""
        if self.df.empty:
            return {}

        last = self.df.iloc[-1]
        prev = self.df.iloc[-2] if len(self.df) >= 2 else last

        close = float(last.get("Close", 0))

        # ADX helpers
        adx      = float(last.get("ADX",      15.0))
        di_plus  = float(last.get("DI_Plus",   0.0))
        di_minus = float(last.get("DI_Minus",  0.0))

        # Supertrend helpers
        st_val    = last.get("Supertrend", np.nan)
        st_valid  = not (isinstance(st_val, float) and np.isnan(st_val))
        st_bull   = bool(last.get("ST_Bullish", False))

        # OBV helpers
        obv      = float(last.get("OBV",     0.0))
        obv_sma  = float(last.get("OBV_SMA", 0.0))
        obv_rising = (obv > obv_sma) if obv_sma != 0 else None

        # ── EMA slope + distance metrics (for downtrend detection) ──────────────
        ema50_now    = float(last.get("EMA50",  close))
        ema50_5d_ago = float(self.df["EMA50"].iloc[-6])  if len(self.df) >= 6  else ema50_now
        ema50_slope_pct    = ((ema50_now - ema50_5d_ago) / ema50_5d_ago * 100) if ema50_5d_ago  > 0 else 0.0
        close_vs_ema50_pct = ((close - ema50_now)        / ema50_now     * 100) if ema50_now     > 0 else 0.0

        # 20-day price momentum
        close_20d_ago    = float(self.df["Close"].iloc[-21]) if len(self.df) >= 21 else close
        momentum_20d_pct = ((close - close_20d_ago) / close_20d_ago * 100)      if close_20d_ago > 0 else 0.0

        return {
            # Price
            "close":  close,
            "high":   float(last.get("High",   0)),
            "low":    float(last.get("Low",    0)),
            "open":   float(last.get("Open",   0)),
            "volume": float(last.get("Volume", 0)),

            # EMAs
            "ema20":  float(last.get("EMA20",  0)),
            "ema50":  ema50_now,
            "ema200": float(last.get("EMA200", 0)),

            # EMA slope + distance (medium-term trend health)
            "ema50_slope_pct":    round(ema50_slope_pct,    3),
            "close_vs_ema50_pct": round(close_vs_ema50_pct, 2),
            "momentum_20d_pct":   round(momentum_20d_pct,   2),

            # RSI
            "rsi": float(last.get("RSI", 50)),

            # MACD
            "macd":           float(last.get("MACD",        0)),
            "macd_signal":    float(last.get("MACD_Signal", 0)),
            "macd_hist":      float(last.get("MACD_Hist",   0)),
            "prev_macd_hist": float(prev.get("MACD_Hist",   0)),

            # Bollinger
            "bb_upper":    float(last.get("BB_Upper",    0)),
            "bb_middle":   float(last.get("BB_Middle",   0)),
            "bb_lower":    float(last.get("BB_Lower",    0)),
            "bb_position": float(last.get("BB_Position", 0.5)),
            "bb_width":    float(last.get("BB_Width",    0)),

            # ATR & Volume
            "atr":       float(last.get("ATR",       0)),
            "vol_ratio": float(last.get("Vol_Ratio", 1)),

            # ADX — trend strength + direction
            "adx":         adx,
            "di_plus":     di_plus,
            "di_minus":    di_minus,
            "adx_trending": adx >= 20,
            "di_bullish":   di_plus > di_minus,

            # Supertrend — clean buy/sell
            "st_bullish":  st_bull if st_valid else None,
            "supertrend":  float(st_val) if st_valid else 0.0,

            # OBV — volume confirmation
            "obv_rising": obv_rising,

            # Consecutive day streak — mean reversion signal
            # +N = N up days in a row (overbought risk), -N = N down days (oversold)
            "streak": int(last.get("Streak", 0)),

            # 20-day high/low (near-term support/resistance)
            "high_20d": float(self.df["High"].tail(20).max()) if len(self.df) >= 20 else close,
            "low_20d":  float(self.df["Low"].tail(20).min())  if len(self.df) >= 20 else close,

            # Pivot Points
            "pivots": self._pivot_points,

            # Derived booleans (used directly by SignalGenerator)
            "above_ema20":        close > float(last.get("EMA20",  0)),
            "above_ema50":        close > float(last.get("EMA50",  0)),
            "above_ema200":       close > float(last.get("EMA200", 0)),
            "ema20_above_ema50":  float(last.get("EMA20", 0)) > float(last.get("EMA50", 0)),
            "macd_bullish":       float(last.get("MACD", 0)) > float(last.get("MACD_Signal", 0)),
            "macd_crossover": (   # MACD hist just turned positive
                float(last.get("MACD_Hist", 0)) > 0
                and float(prev.get("MACD_Hist", 0)) <= 0
            ),
            "macd_crossunder": (  # MACD hist just turned negative
                float(last.get("MACD_Hist", 0)) < 0
                and float(prev.get("MACD_Hist", 0)) >= 0
            ),
        }

    def get_recent_candles(self, n: int = 5) -> pd.DataFrame:
        """Return the last n OHLCV rows for display."""
        return self.df.tail(n)[["Open", "High", "Low", "Close", "Volume"]]

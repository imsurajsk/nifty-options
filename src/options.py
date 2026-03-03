"""
Options Chain Analysis Engine

Computes:
  • Put-Call Ratio (PCR)            — sentiment gauge
  • Max Pain                         — strike where option writers lose least
  • OI concentration / resistance & support levels
  • IV snapshot
  • ATM option data
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STRIKE_GAP = 50   # Nifty option strikes are at 50-point intervals


class OptionsAnalysis:
    """Full options-chain analysis for one expiry session."""

    def __init__(self, options_df: pd.DataFrame, spot_price: float):
        self.df    = options_df.copy() if not options_df.empty else pd.DataFrame()
        self.spot  = spot_price
        self.res   = {}          # result bucket

        if not self.df.empty:
            self._run()

    # ── Orchestrator ─────────────────────────────────────────────────────────

    def _run(self):
        self._clean()
        if self.df.empty:
            return
        self._pcr()
        self._max_pain()
        self._oi_concentration()
        self._atm_info()

    # ── Step 1: Clean ────────────────────────────────────────────────────────

    def _clean(self):
        """Drop rows where both CE and PE OI are zero."""
        self.df = self.df[
            (self.df["CE_OI"] > 0) | (self.df["PE_OI"] > 0)
        ].copy()

    # ── Step 2: Put-Call Ratio ────────────────────────────────────────────────

    def _pcr(self):
        total_put_oi  = float(self.df["PE_OI"].sum())
        total_call_oi = float(self.df["CE_OI"].sum())

        pcr = (total_put_oi / total_call_oi) if total_call_oi > 0 else 1.0

        self.res["pcr"]           = round(pcr, 3)
        self.res["total_put_oi"]  = int(total_put_oi)
        self.res["total_call_oi"] = int(total_call_oi)

        # Contrarian interpretation
        if pcr > 1.50:
            signal = "STRONG_BULLISH"   # market overly hedged → contrarian buy
        elif pcr > 1.20:
            signal = "BULLISH"
        elif pcr > 0.80:
            signal = "NEUTRAL"
        elif pcr > 0.50:
            signal = "BEARISH"
        else:
            signal = "STRONG_BEARISH"   # market too complacent → contrarian sell

        self.res["pcr_signal"] = signal

    # ── Step 3: Max Pain ─────────────────────────────────────────────────────

    def _max_pain(self):
        """
        Max Pain = strike at which total option-writer loss is minimum.
        At expiry, prices are pulled towards this level (theory, not guarantee).
        """
        try:
            strikes = sorted(self.df["strikePrice"].unique())
            pain: dict[float, float] = {}

            for target in strikes:
                loss = 0.0
                for _, row in self.df.iterrows():
                    k = row["strikePrice"]
                    if target > k:
                        loss += (target - k) * row["CE_OI"]
                    if target < k:
                        loss += (k - target) * row["PE_OI"]
                pain[target] = loss

            max_pain_strike = min(pain, key=pain.get)
            self.res["max_pain"]          = max_pain_strike
            self.res["max_pain_diff_pct"] = (
                (max_pain_strike - self.spot) / self.spot * 100
                if self.spot else 0.0
            )
        except Exception as exc:
            logger.error("Max pain calc failed: %s", exc)
            self.res["max_pain"]          = self.spot
            self.res["max_pain_diff_pct"] = 0.0

    # ── Step 4: OI Concentration ─────────────────────────────────────────────

    def _oi_concentration(self):
        """Identify top resistance (call OI) and support (put OI) strikes."""
        try:
            top_calls = self.df.nlargest(3, "CE_OI")
            top_puts  = self.df.nlargest(3, "PE_OI")

            self.res["top_call_resistance"] = top_calls["strikePrice"].tolist()
            self.res["top_put_support"]     = top_puts["strikePrice"].tolist()

            # Strikes with biggest new OI build-up (change > 0)
            new_calls = self.df[self.df["CE_changeOI"] > 0].nlargest(3, "CE_changeOI")
            new_puts  = self.df[self.df["PE_changeOI"] > 0].nlargest(3, "PE_changeOI")

            self.res["new_call_oi_strikes"] = new_calls["strikePrice"].tolist()
            self.res["new_put_oi_strikes"]  = new_puts["strikePrice"].tolist()

            # Total OI change volumes — used for volume-weighted scoring in SignalGenerator
            # (counting strikes is weak; the SIZE of the build-up is what matters)
            self.res["total_new_call_oi"] = int(new_calls["CE_changeOI"].sum()) if not new_calls.empty else 0
            self.res["total_new_put_oi"]  = int(new_puts["PE_changeOI"].sum())  if not new_puts.empty else 0

        except Exception as exc:
            logger.error("OI concentration failed: %s", exc)

    # ── Step 5: ATM Info ─────────────────────────────────────────────────────

    def _atm_info(self):
        atm = round(self.spot / STRIKE_GAP) * STRIKE_GAP
        self.res["atm_strike"] = atm

        atm_rows = self.df[self.df["strikePrice"] == atm]
        if not atm_rows.empty:
            r = atm_rows.iloc[0]
            self.res["atm_ce_ltp"] = float(r.get("CE_LTP", 0))
            self.res["atm_pe_ltp"] = float(r.get("PE_LTP", 0))
            self.res["atm_ce_iv"]  = float(r.get("CE_IV",  0))
            self.res["atm_pe_iv"]  = float(r.get("PE_IV",  0))
            self.res["atm_ce_oi"]  = int(r.get("CE_OI",   0))
            self.res["atm_pe_oi"]  = int(r.get("PE_OI",   0))

        # IV Skew: how much more expensive puts are vs calls at ATM
        # Positive skew = market paying up for put protection = fear
        ce_iv = self.res.get("atm_ce_iv", 0)
        pe_iv = self.res.get("atm_pe_iv", 0)
        self.res["iv_skew"]     = round(pe_iv - ce_iv, 2)
        self.res["iv_skew_pct"] = round((pe_iv / ce_iv - 1) * 100, 1) if ce_iv > 0 else 0

        # Straddle price = ATM CE + ATM PE → market's expected move by expiry
        ce_ltp = self.res.get("atm_ce_ltp", 0)
        pe_ltp = self.res.get("atm_pe_ltp", 0)
        straddle = ce_ltp + pe_ltp
        self.res["straddle_price"]    = round(straddle, 1)
        self.res["expected_move_pct"] = (
            round(straddle / self.spot * 100, 2) if self.spot > 0 and straddle > 0 else 0
        )

    # ── Public Helpers ────────────────────────────────────────────────────────

    def get_option_row(self, strike: float, option_type: str,
                       expiry_date=None) -> dict:
        """Return data dict for a specific strike + option type."""
        df = self.df[self.df["strikePrice"] == strike]
        if expiry_date is not None:
            df2 = df[df["expiryDate"] == expiry_date]
            if not df2.empty:
                df = df2
        if df.empty:
            return {}
        row = df.iloc[0]
        prefix = option_type  # "CE" or "PE"
        return {
            "ltp":    float(row.get(f"{prefix}_LTP",      0)),
            "oi":     int(row.get(f"{prefix}_OI",         0)),
            "volume": int(row.get(f"{prefix}_volume",     0)),
            "iv":     float(row.get(f"{prefix}_IV",       0)),
            "bid":    float(row.get(f"{prefix}_bid",      0)),
            "ask":    float(row.get(f"{prefix}_ask",      0)),
            "change_oi": int(row.get(f"{prefix}_changeOI", 0)),
        }

    def get_iv_snapshot(self) -> dict:
        """Return median IV and a qualitative rank."""
        if self.df.empty or "CE_IV" not in self.df.columns:
            return {"current_iv": 15.0, "iv_rank": "NORMAL"}
        iv_vals = self.df["CE_IV"].replace(0, np.nan).dropna()
        current_iv = float(iv_vals.median()) if not iv_vals.empty else 15.0

        if current_iv < 12:
            rank = "VERY_LOW"
        elif current_iv < 18:
            rank = "NORMAL"
        elif current_iv < 25:
            rank = "HIGH"
        else:
            rank = "VERY_HIGH"

        return {"current_iv": round(current_iv, 2), "iv_rank": rank}

    def get_summary(self) -> dict:
        """Return the complete options analysis result dict."""
        iv = self.get_iv_snapshot()
        return {**self.res, **iv}

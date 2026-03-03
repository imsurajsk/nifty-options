"""
Data Fetcher — pulls live and historical market data.

Sources:
  • NSE India API  → options chain, spot price, India VIX
  • Yahoo Finance   → historical OHLCV (reliable fallback)
"""

import time
import json
import logging
from datetime import datetime, timedelta

import requests
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ─── NSE request headers (mimics a real browser) ─────────────────────────────
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept":          "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
    "Referer":         "https://www.nseindia.com/",
    "X-Requested-With": "XMLHttpRequest",
}

_NSE_HOME         = "https://www.nseindia.com"
_NSE_OPTION_CHAIN = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
_NSE_ALL_INDICES  = "https://www.nseindia.com/api/allIndices"


class DataFetcher:
    """Centralised data-fetching layer with automatic fallbacks."""

    def __init__(self):
        self.session = requests.Session()
        self._nse_ready = False
        self._init_nse_session()

    # ── Session Initialisation ────────────────────────────────────────────────

    def _init_nse_session(self):
        """Visit NSE pages to pick up required cookies."""
        seed_urls = [
            _NSE_HOME,
            "https://www.nseindia.com/option-chain",
        ]
        for url in seed_urls:
            try:
                self.session.get(url, headers=_HEADERS, timeout=12)
                time.sleep(1.0)
            except Exception as exc:
                logger.debug("NSE session seed failed for %s: %s", url, exc)
        self._nse_ready = True

    def _nse_get(self, url: str, retries: int = 2) -> dict:
        """GET an NSE API endpoint and return parsed JSON, or {}."""
        for attempt in range(retries):
            try:
                resp = self.session.get(url, headers=_HEADERS, timeout=15)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code in (401, 403):
                    # Session expired — reinitialise and retry
                    logger.debug("NSE session expired, reinitialising…")
                    self._init_nse_session()
            except Exception as exc:
                logger.debug("NSE GET attempt %d failed: %s", attempt + 1, exc)
            time.sleep(2)
        return {}

    # ── Spot Price ────────────────────────────────────────────────────────────

    def get_nifty_spot(self) -> float:
        """Return current Nifty 50 spot price."""
        # Primary: NSE allIndices
        data = self._nse_get(_NSE_ALL_INDICES)
        if data:
            for idx in data.get("data", []):
                if idx.get("indexSymbol") == "NIFTY 50":
                    try:
                        return float(idx["last"])
                    except (KeyError, ValueError):
                        pass

        # Fallback: yfinance
        try:
            ticker = yf.Ticker("^NSEI")
            price = ticker.fast_info.last_price
            if price and price > 0:
                return float(price)
        except Exception as exc:
            logger.warning("yfinance spot fallback failed: %s", exc)

        return 0.0

    # ── India VIX ─────────────────────────────────────────────────────────────

    def get_india_vix(self) -> float:
        """Return India VIX value."""
        data = self._nse_get(_NSE_ALL_INDICES)
        if data:
            for idx in data.get("data", []):
                sym = idx.get("indexSymbol", "")
                if "VIX" in sym.upper():
                    try:
                        return float(idx["last"])
                    except (KeyError, ValueError):
                        pass

        # Fallback: yfinance
        try:
            ticker = yf.Ticker("^INDIAVIX")
            price = ticker.fast_info.last_price
            if price and price > 0:
                return float(price)
        except Exception:
            pass

        logger.warning("VIX fetch failed — using default 15.0")
        return 15.0

    # ── Historical OHLCV ──────────────────────────────────────────────────────

    def get_historical_data(self, days: int = 150) -> pd.DataFrame:
        """Return daily OHLCV DataFrame for ^NSEI (last `days` trading sessions)."""
        # Fetch extra calendar days to account for weekends/holidays
        end_date   = datetime.now()
        start_date = end_date - timedelta(days=days * 2)

        try:
            ticker = yf.Ticker("^NSEI")
            df = ticker.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=True,
            )
            if df.empty:
                raise ValueError("Empty DataFrame returned by yfinance")

            df.index = pd.to_datetime(df.index)
            return df.tail(days).copy()

        except Exception as exc:
            logger.error("Historical data fetch failed: %s", exc)
            return pd.DataFrame()

    # ── Options Chain ─────────────────────────────────────────────────────────

    def get_options_chain(self) -> dict:
        """Fetch raw NSE options chain JSON for NIFTY."""
        return self._nse_get(_NSE_OPTION_CHAIN)

    def parse_options_chain(self, raw: dict) -> pd.DataFrame:
        """
        Convert raw NSE options chain JSON → clean DataFrame.

        Columns per row:
          strikePrice, expiryDate,
          CE_OI, CE_changeOI, CE_volume, CE_IV, CE_LTP, CE_change, CE_bid, CE_ask,
          PE_OI, PE_changeOI, PE_volume, PE_IV, PE_LTP, PE_change, PE_bid, PE_ask
        """
        if not raw:
            return pd.DataFrame()

        try:
            records = raw.get("records", {})
            rows = []

            for item in records.get("data", []):
                ce = item.get("CE", {})
                pe = item.get("PE", {})
                row = {
                    "strikePrice": item.get("strikePrice", 0),
                    "expiryDate":  item.get("expiryDate", ""),
                    # Call
                    "CE_OI":       ce.get("openInterest",       0),
                    "CE_changeOI": ce.get("changeinOpenInterest", 0),
                    "CE_volume":   ce.get("totalTradedVolume",   0),
                    "CE_IV":       ce.get("impliedVolatility",   0),
                    "CE_LTP":      ce.get("lastPrice",           0),
                    "CE_change":   ce.get("change",              0),
                    "CE_bid":      ce.get("bidprice",            0),
                    "CE_ask":      ce.get("askPrice",            0),
                    # Put
                    "PE_OI":       pe.get("openInterest",       0),
                    "PE_changeOI": pe.get("changeinOpenInterest", 0),
                    "PE_volume":   pe.get("totalTradedVolume",   0),
                    "PE_IV":       pe.get("impliedVolatility",   0),
                    "PE_LTP":      pe.get("lastPrice",           0),
                    "PE_change":   pe.get("change",              0),
                    "PE_bid":      pe.get("bidprice",            0),
                    "PE_ask":      pe.get("askPrice",            0),
                }
                rows.append(row)

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows)
            df["expiryDate"] = pd.to_datetime(
                df["expiryDate"], format="%d-%b-%Y", errors="coerce"
            )
            return df.sort_values("strikePrice").reset_index(drop=True)

        except Exception as exc:
            logger.error("Options chain parsing failed: %s", exc)
            return pd.DataFrame()

    def get_expiry_dates(self, raw: dict) -> list:
        """Return sorted list of upcoming expiry datetimes."""
        try:
            dates = raw.get("records", {}).get("expiryDates", [])
            return sorted(
                datetime.strptime(d, "%d-%b-%Y") for d in dates
            )
        except Exception:
            return []

"""
News & Global Market Analyzer

Two complementary lenses:

1. GLOBAL MARKETS — hard numbers, no interpretation needed
   Checks S&P 500, Dow, Nikkei, Hang Seng, Crude Oil, Gold, USD/INR, US VIX.
   Each % move maps to a numeric impact on Nifty.

2. NEWS HEADLINES — keyword-based event detection
   Fetches recent headlines via yfinance; scores them against an extensive
   dictionary of bullish / bearish event keywords.
   Covers wars, crises, rate decisions, geopolitical events, etc.

Combined output: event_score on [-100, +100] + list of detected events.
"""

import logging
import time
from datetime import datetime, timezone, timedelta

import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# ── Keyword dictionaries ──────────────────────────────────────────────────────
# Score magnitude indicates how strongly each event moves the market.
# Negative = bearish for Nifty / India, Positive = bullish.

BEARISH_EVENTS: dict[str, int] = {
    # ── War / Geopolitical (most severe) ─────────────────────────────────────
    "nuclear":              -90,
    "nuclear war":          -95,
    "world war":            -95,
    "war declared":         -85,
    "military invasion":    -85,
    "invasion":             -80,
    "war":                  -70,
    "military strike":      -70,
    "airstrike":            -65,
    "missile attack":       -65,
    "bombing":              -60,
    "terror attack":        -65,
    "terrorist attack":     -65,
    "coup":                 -60,
    "sanctions":            -45,
    "trade war":            -50,
    "geopolitical crisis":  -55,
    "geopolitical tension": -35,
    "conflict":             -35,
    "escalation":           -40,

    # ── Financial Crisis ──────────────────────────────────────────────────────
    "market crash":         -80,
    "circuit breaker":      -75,
    "bank failure":         -75,
    "banking crisis":       -70,
    "financial crisis":     -70,
    "debt crisis":          -65,
    "default":              -60,
    "sovereign default":    -70,
    "lehman":               -80,
    "collapse":             -55,
    "bankruptcy":           -45,
    "insolvency":           -40,

    # ── Monetary / Macro bearish ──────────────────────────────────────────────
    "emergency rate hike":  -60,
    "surprise rate hike":   -55,
    "rate hike":            -30,
    "hawkish":              -25,
    "inflation surge":      -40,
    "inflation spike":      -40,
    "stagflation":          -50,
    "recession":            -50,
    "slowdown":             -30,
    "gdp falls":            -40,
    "gdp contracts":        -45,
    "unemployment rises":   -30,
    "job losses":           -25,
    "trade deficit":        -20,
    "current account":      -15,
    "fiscal deficit":       -20,
    "downgrade":            -40,
    "rating cut":           -40,

    # ── India-specific bearish ────────────────────────────────────────────────
    "fii selling":          -40,
    "fii outflow":          -40,
    "rupee falls":          -35,
    "rupee crash":          -50,
    "crude surge":          -40,
    "oil spike":            -40,
    "monsoon failure":      -35,
    "drought":              -25,
    "lockdown":             -60,
    "pandemic":             -65,
    "outbreak":             -50,
    "sebi ban":             -30,
    "rbi intervention":     -20,

    # ── Sentiment / Market bearish ────────────────────────────────────────────
    "sell-off":             -35,
    "selloff":              -35,
    "plunge":               -45,
    "plummets":             -40,
    "tumbles":              -35,
    "crash":                -55,
    "bear market":          -40,
    "correction":           -25,
    "panic":                -45,
    "fear":                 -20,
}

BULLISH_EVENTS: dict[str, int] = {
    # ── Peace / Resolution ────────────────────────────────────────────────────
    "ceasefire":            +65,
    "peace deal":           +65,
    "peace agreement":      +70,
    "peace talks":          +40,
    "conflict resolved":    +60,
    "sanctions lifted":     +50,
    "trade deal":           +55,
    "trade agreement":      +50,
    "trade truce":          +45,

    # ── Monetary / Macro bullish ──────────────────────────────────────────────
    "rate cut":             +55,
    "emergency rate cut":   +65,
    "surprise rate cut":    +60,
    "dovish":               +30,
    "stimulus":             +50,
    "quantitative easing":  +45,
    "relief package":       +40,
    "bailout":              +35,
    "fiscal stimulus":      +45,
    "inflation cools":      +45,
    "inflation falls":      +40,
    "inflation eases":      +40,
    "gdp growth":           +40,
    "strong gdp":           +40,
    "record gdp":           +45,
    "jobs growth":          +30,
    "unemployment falls":   +30,
    "trade surplus":        +30,
    "current account surplus": +25,

    # ── India-specific bullish ────────────────────────────────────────────────
    "fii buying":           +50,
    "fii inflow":           +50,
    "record fii":           +60,
    "foreign investment":   +35,
    "capex boost":          +35,
    "infrastructure spend": +30,
    "rupee strengthens":    +30,
    "crude falls":          +35,
    "oil drops":            +35,
    "good monsoon":         +25,
    "rbi rate cut":         +55,
    "rbi policy":           +15,

    # ── Sentiment / Market bullish ────────────────────────────────────────────
    "rally":                +35,
    "surge":                +35,
    "record high":          +30,
    "all-time high":        +35,
    "bull market":          +40,
    "strong earnings":      +30,
    "earnings beat":        +25,
    "market recovery":      +30,
    "rebound":              +25,
    "bounce":               +20,
}

# ── Global market impact on Nifty (% change → score) ─────────────────────────
# Based on historical correlation studies

def _pct_to_score(pct: float, sensitivity: float) -> float:
    """Convert % change to a score contribution."""
    return max(-60, min(60, pct * sensitivity))


GLOBAL_INDEX_CONFIG = {
    # symbol: (display_name, sensitivity, direction_multiplier)
    # sensitivity: how many score points per 1% move
    # direction: +1 if positive correlation, -1 if negative
    "^GSPC":    ("S&P 500 (US)",         12.0,  +1),   # strongest correlation
    "^DJI":     ("Dow Jones (US)",        10.0,  +1),
    "^N225":    ("Nikkei 225 (Japan)",     7.0,  +1),
    "^HSI":     ("Hang Seng (HK)",         6.0,  +1),
    "^VIX":     ("US VIX (Fear Index)",    8.0,  -1),   # VIX up = markets fall
    "CL=F":     ("Crude Oil (WTI)",        5.0,  -1),   # crude up = bearish India
    "GC=F":     ("Gold",                   3.0,  -1),   # gold up = risk-off
    "USDINR=X": ("USD/INR",                6.0,  -1),   # rupee weakens = bearish
}


class NewsAnalyzer:
    """Fetches and scores news headlines + global market movements."""

    def __init__(self):
        self.headlines:      list[dict] = []
        self.global_markets: dict       = {}
        self.detected_events: list[dict] = []
        self.news_score:     float      = 0.0
        self.global_score:   float      = 0.0
        self.event_score:    float      = 0.0   # combined final score

    def run(self):
        """Fetch everything and compute scores."""
        self._fetch_headlines()
        self._score_headlines()
        self._fetch_global_markets()
        self._score_global_markets()
        self._combine_scores()

    # ── News Headlines ────────────────────────────────────────────────────────

    def _fetch_headlines(self):
        """Fetch recent headlines via yfinance for India-relevant symbols."""
        symbols = ["^NSEI", "^BSESN", "INFY.NS", "RELIANCE.NS"]
        seen: set[str] = set()

        for sym in symbols:
            try:
                ticker = yf.Ticker(sym)
                raw_news = ticker.news or []
                cutoff = time.time() - 48 * 3600   # last 48 hours

                for item in raw_news[:15]:
                    title = item.get("title", "").strip()
                    if not title or title in seen:
                        continue
                    ts = item.get("providerPublishTime", 0)
                    if ts < cutoff:
                        continue
                    seen.add(title)
                    self.headlines.append({
                        "title":  title,
                        "source": item.get("publisher", ""),
                        "time":   datetime.fromtimestamp(ts) if ts else None,
                    })
            except Exception as exc:
                logger.debug("News fetch for %s failed: %s", sym, exc)

    def _score_headlines(self):
        """Score all headlines using keyword matching."""
        total_score = 0.0

        for item in self.headlines:
            title_lower = item["title"].lower()
            headline_score = 0

            # Check bearish keywords
            for kw, score in BEARISH_EVENTS.items():
                if kw in title_lower:
                    headline_score = min(headline_score, score)   # take worst
                    self.detected_events.append({
                        "type":      "BEARISH",
                        "keyword":   kw,
                        "headline":  item["title"],
                        "score":     score,
                        "time":      item.get("time"),
                    })

            # Check bullish keywords
            for kw, score in BULLISH_EVENTS.items():
                if kw in title_lower:
                    headline_score = max(headline_score, score)   # take best
                    self.detected_events.append({
                        "type":      "BULLISH",
                        "keyword":   kw,
                        "headline":  item["title"],
                        "score":     score,
                        "time":      item.get("time"),
                    })

            item["score"] = headline_score
            total_score += headline_score

        # Average score, but cap at ±80
        n = max(1, len(self.headlines))
        self.news_score = max(-80, min(80, total_score / n))

    # ── Global Markets ────────────────────────────────────────────────────────

    def _fetch_global_markets(self):
        """Fetch latest % change for each global index/commodity."""
        for symbol, (name, sensitivity, direction) in GLOBAL_INDEX_CONFIG.items():
            try:
                ticker = yf.Ticker(symbol)
                info   = ticker.fast_info

                last_price = getattr(info, "last_price",       None)
                prev_close = getattr(info, "previous_close",   None)

                if last_price and prev_close and prev_close != 0:
                    pct_change = (last_price - prev_close) / prev_close * 100
                else:
                    # Try 2-day history as fallback
                    hist = ticker.history(period="2d", interval="1d")
                    if len(hist) >= 2:
                        prev_close = float(hist["Close"].iloc[-2])
                        last_price = float(hist["Close"].iloc[-1])
                        pct_change = (last_price - prev_close) / prev_close * 100
                    else:
                        pct_change = 0.0

                self.global_markets[symbol] = {
                    "name":        name,
                    "price":       round(float(last_price or 0), 2),
                    "pct_change":  round(pct_change, 2),
                    "sensitivity": sensitivity,
                    "direction":   direction,
                }
            except Exception as exc:
                logger.debug("Global market fetch failed for %s: %s", symbol, exc)

    def _score_global_markets(self):
        """Translate global market moves into a net score."""
        total = 0.0
        for sym, data in self.global_markets.items():
            pct    = data["pct_change"]
            sens   = data["sensitivity"]
            dirn   = data["direction"]
            contrib = _pct_to_score(pct * dirn, sens)
            data["score_contribution"] = round(contrib, 1)
            total += contrib

        # Normalise: average, then scale to ±100
        n = max(1, len(self.global_markets))
        self.global_score = max(-100, min(100, total / n))

    # ── Combine ───────────────────────────────────────────────────────────────

    def _combine_scores(self):
        """
        Final event_score:  60% global markets  +  40% news headlines
        Global markets are hard numbers → higher weight.
        """
        self.event_score = (self.global_score * 0.60
                            + self.news_score  * 0.40)
        self.event_score = max(-100, min(100, self.event_score))

    # ── Public API ────────────────────────────────────────────────────────────

    def get_summary(self) -> dict:
        """Return everything the signal generator and report need."""
        # Deduplicate detected events by keyword
        seen_kw: set[str] = set()
        unique_events = []
        for ev in sorted(self.detected_events, key=lambda x: abs(x["score"]), reverse=True):
            if ev["keyword"] not in seen_kw:
                seen_kw.add(ev["keyword"])
                unique_events.append(ev)

        # Identify the single most important event
        top_event = unique_events[0] if unique_events else None

        return {
            "event_score":       round(self.event_score, 1),
            "news_score":        round(self.news_score, 1),
            "global_score":      round(self.global_score, 1),
            "headline_count":    len(self.headlines),
            "detected_events":   unique_events[:8],   # top 8 for display
            "top_event":         top_event,
            "global_markets":    self.global_markets,
            "headlines":         self.headlines[:10],
            "has_major_event":   any(abs(ev["score"]) >= 60 for ev in unique_events),
            "major_event_type":  (
                "BEARISH" if any(ev["score"] <= -60 for ev in unique_events)
                else "BULLISH" if any(ev["score"] >= 60 for ev in unique_events)
                else "NONE"
            ),
        }

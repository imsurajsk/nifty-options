#!/usr/bin/env python3
"""
Nifty Options Prediction System
================================
Run every morning before 10:00 AM IST (or at 3:00 PM for next-day prep).

Usage:
    python main.py
    python main.py --capital 200000   # override default capital (INR)
"""

import sys
import os
import logging
import argparse
from datetime import datetime, timezone, timedelta

# ── make sure project root is on the import path ─────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── suppress noisy third-party loggers ───────────────────────────────────────
logging.basicConfig(level=logging.WARNING,
                    format="%(levelname)s %(name)s — %(message)s")
for noisy in ("yfinance", "urllib3", "peewee", "asyncio"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

console = Console()

IST = timezone(timedelta(hours=5, minutes=30))


def is_market_day() -> bool:
    """Return True if today is Mon–Fri (doesn't check NSE holidays)."""
    return datetime.now(IST).weekday() < 5


def parse_args():
    p = argparse.ArgumentParser(description="Nifty Options Prediction")
    p.add_argument("--capital", type=float, default=None,
                   help="Your trading capital in INR (overrides config default)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Imports ───────────────────────────────────────────────────────────────
    from src.data_fetcher  import DataFetcher
    from src.technical     import TechnicalAnalysis
    from src.options       import OptionsAnalysis
    from src.news_analyzer import NewsAnalyzer
    from src.signals       import SignalGenerator
    from src.recommender   import OptionRecommender
    from src.report        import ReportGenerator
    from config            import DEFAULT_CAPITAL, HISTORICAL_DAYS, STRATEGY

    capital = args.capital if args.capital else DEFAULT_CAPITAL

    # ── Market day check ─────────────────────────────────────────────────────
    if not is_market_day():
        console.print(Panel(
            "[yellow]Today is a weekend — NSE is closed.\n"
            "You can still run the analysis to prep for Monday.[/yellow]",
            border_style="yellow",
        ))

    console.print()
    console.print("[bold blue]  Nifty Options Prediction System  [/bold blue]")
    console.print(f"[dim]  Capital: ₹ {capital:,.0f}[/dim]\n")

    # ── Data Collection (with progress spinner) ───────────────────────────────
    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  console=console, transient=True) as prog:

        fetcher = DataFetcher()

        # ── Spot price ───────────────────────────────────────────────────────
        t = prog.add_task("Fetching Nifty spot price…")
        spot = fetcher.get_nifty_spot()
        if spot <= 0:
            import time; time.sleep(3)
            spot = fetcher.get_nifty_spot()
        prog.update(t, description=f"[green]✓ Nifty Spot: ₹ {spot:,.2f}[/green]")

        # ── India VIX ────────────────────────────────────────────────────────
        t2 = prog.add_task("Fetching India VIX…")
        vix = fetcher.get_india_vix()
        prog.update(t2, description=f"[green]✓ India VIX: {vix:.2f}[/green]")

        # ── Historical data ───────────────────────────────────────────────────
        t3 = prog.add_task("Loading historical price data…")
        hist_df = fetcher.get_historical_data(days=HISTORICAL_DAYS)
        n_days  = len(hist_df) if not hist_df.empty else 0
        prog.update(t3, description=f"[green]✓ Historical data: {n_days} sessions[/green]")

        # ── Technical analysis ────────────────────────────────────────────────
        t4 = prog.add_task("Computing technical indicators…")
        ta         = TechnicalAnalysis(hist_df)
        tech_data  = ta.get_current_values()

        # If NSE spot failed, fall back to last close from yfinance history
        if spot <= 0:
            spot = tech_data.get("close", 0)

        prog.update(t4, description="[green]✓ Technical indicators ready[/green]")

        # ── Options chain ─────────────────────────────────────────────────────
        t5 = prog.add_task("Fetching NSE options chain…")
        raw_opts    = fetcher.get_options_chain()
        options_df  = fetcher.parse_options_chain(raw_opts)
        expiry_dates = fetcher.get_expiry_dates(raw_opts)

        if options_df.empty:
            prog.update(t5, description="[yellow]⚠ Options chain unavailable (market may be closed)[/yellow]")
        else:
            prog.update(t5, description=f"[green]✓ Options chain: {len(options_df)} strike rows[/green]")

        # ── Options analysis ──────────────────────────────────────────────────
        t6 = prog.add_task("Analysing options chain…")
        oa              = OptionsAnalysis(options_df, spot)
        options_summary = oa.get_summary()
        # Inject days-to-nearest-expiry so SignalGenerator can scale max pain strength
        if expiry_dates:
            _dte = (expiry_dates[0].date() - datetime.now(IST).date()).days
            options_summary["days_to_expiry"] = max(0, _dte)
        else:
            options_summary["days_to_expiry"] = 7
        prog.update(t6, description="[green]✓ Options analysis complete[/green]")

        # ── News & Global Markets ─────────────────────────────────────────────
        t7 = prog.add_task("Fetching news & global markets…")
        na          = NewsAnalyzer()
        na.run()
        news_data   = na.get_summary()
        event_score = news_data.get("event_score", 0)
        major_flag  = " ⚡ MAJOR EVENT" if news_data.get("has_major_event") else ""
        prog.update(t7, description=f"[green]✓ News/Global: {event_score:+.0f} pts{major_flag}[/green]")

        # ── Signal generation ─────────────────────────────────────────────────
        t8 = prog.add_task("Generating signals…")
        sg          = SignalGenerator(tech_data, options_summary, vix, news=news_data)
        signal_data = sg.get_full_analysis()
        prog.update(t8, description=f"[green]✓ Signal: {signal_data['direction']} "
                                    f"({signal_data['score']:+.1f})[/green]")

        # ── Option recommendation ─────────────────────────────────────────────
        t9 = prog.add_task("Selecting best option…")
        # Use sell quality score when strategy is SELL_PUTS
        q_score = (
            signal_data.get("sell_quality_score", 50)
            if STRATEGY == "SELL_PUTS"
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
            vix             = vix,
            capital         = capital,
            strategy        = STRATEGY,
        )
        recommendation = recommender.get_recommendation()
        prog.update(t9, description="[green]✓ Recommendation ready[/green]")

    # ── Display Report ────────────────────────────────────────────────────────
    report = ReportGenerator(
        spot_price      = spot,
        vix             = vix,
        technical_data  = tech_data,
        options_summary = options_summary,
        signal_data     = signal_data,
        recommendation  = recommendation,
        news_data       = news_data,
        timestamp       = datetime.now(IST).replace(tzinfo=None),
    )
    report.display()
    report.save_log()


if __name__ == "__main__":
    main()

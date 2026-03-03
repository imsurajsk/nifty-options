"""
Report Generator

Renders a focused, actionable trading report — nothing but what matters.
Bold recommendation, exact rupee amounts, zero fluff.
"""

import csv
import logging
import os
from datetime import datetime

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

logger = logging.getLogger(__name__)
console = Console()


def _dir_col(direction: str) -> str:
    return {
        "STRONG_BULLISH": "bold bright_green",
        "BULLISH":        "green",
        "NEUTRAL":        "yellow",
        "BEARISH":        "red",
        "STRONG_BEARISH": "bold bright_red",
    }.get(direction, "white")


def _quality_col(q: int) -> str:
    if q >= 70: return "bright_green"
    if q >= 55: return "green"
    if q >= 40: return "yellow"
    return "red"


def _quality_bar(quality: int, label: str) -> str:
    filled = quality // 10
    bar    = "█" * filled + "░" * (10 - filled)
    col    = _quality_col(quality)
    return f"[{col}]{bar}  {quality}/100  {label}[/{col}]"


class ReportGenerator:

    def __init__(
        self,
        spot_price:      float,
        vix:             float,
        technical_data:  dict,
        options_summary: dict,
        signal_data:     dict,
        recommendation:  dict,
        news_data:       dict | None = None,
        timestamp:       datetime | None = None,
    ):
        self.spot   = spot_price
        self.vix    = vix
        self.tech   = technical_data  or {}
        self.opts   = options_summary or {}
        self.signal = signal_data     or {}
        self.rec    = recommendation  or {}
        self.news   = news_data       or {}
        self.ts     = timestamp or datetime.now()

    # ── Header ────────────────────────────────────────────────────────────────

    def _header(self):
        console.print()
        console.print(Rule(
            f"[bold blue]  NIFTY OPTIONS PREDICTION  •  "
            f"{self.ts.strftime('%A, %d %b %Y  •  %I:%M %p IST')}  [/bold blue]",
            style="blue",
        ))
        console.print()

    # ── Snapshot bar ─────────────────────────────────────────────────────────

    def _snapshot_bar(self):
        direction = self.signal.get("direction", "NEUTRAL")
        score     = self.signal.get("score", 0)
        # Show sell quality when action is SELL PUT
        is_sell   = self.rec.get("action") == "SELL PUT"
        quality   = self.signal.get("sell_quality_score", 0) if is_sell else self.signal.get("quality_score", 0)
        qlabel    = self.signal.get("sell_quality_label", "") if is_sell else self.signal.get("quality_label", "")
        adx       = self.tech.get("adx", 0)
        st_bull   = self.tech.get("st_bullish")

        dcol    = _dir_col(direction)
        vix_col = "red" if self.vix > 20 else ("yellow" if self.vix > 15 else "green")
        sc_col  = "bright_green" if score > 0 else ("bright_red" if score < 0 else "yellow")
        adx_col = "bright_green" if adx >= 25 else ("yellow" if adx >= 20 else "red")

        st_str = ""
        if   st_bull is True:  st_str = "   [green]Supertrend ↑ BULL[/green]"
        elif st_bull is False: st_str = "   [red]Supertrend ↓ BEAR[/red]"

        console.print(
            f"  [bold white]Nifty:[/bold white] [bold cyan]₹ {self.spot:,.2f}[/bold cyan]"
            f"   [bold white]VIX:[/bold white] [{vix_col}]{self.vix:.1f}[/{vix_col}]"
            f"   [bold white]ADX:[/bold white] [{adx_col}]{adx:.0f}[/{adx_col}]"
            f"   [bold white]Direction:[/bold white] [{dcol}]{direction.replace('_', ' ')}[/{dcol}]"
            f"   [bold white]Score:[/bold white] [{sc_col}]{score:+.1f}[/{sc_col}]"
            f"{st_str}"
        )
        console.print(
            f"  [bold white]Trade Quality:[/bold white] {_quality_bar(quality, qlabel)}"
        )
        console.print()

    # ── Major event alert ─────────────────────────────────────────────────────

    def _major_event_alert(self):
        n = self.news
        if not n.get("has_major_event"):
            return
        event_type = n.get("major_event_type", "NONE")
        col   = "bright_red"   if event_type == "BEARISH" else "bright_green"
        label = "⚡  BEARISH CRISIS — BUY PUTS" if event_type == "BEARISH" \
                else "⚡  MAJOR BULLISH EVENT — BUY CALLS"
        top      = n.get("top_event") or {}
        headline = top.get("headline", "")
        console.print(Panel(
            f"[bold {col}]{label}[/bold {col}]\n\n"
            f"[white]{headline}[/white]\n\n"
            f"[dim]Event score: {n.get('event_score', 0):+.0f}  —  "
            f"News weight: 50% (CRISIS MODE active)[/dim]",
            title=f"[bold {col}]⚠  MAJOR MARKET EVENT[/bold {col}]",
            border_style=col,
        ))
        console.print()

    # ── Global markets ────────────────────────────────────────────────────────

    def _global_markets(self):
        gm = self.news.get("global_markets", {})
        if not gm:
            return
        tbl = Table(
            box=box.SIMPLE_HEAVY, border_style="blue", show_header=True,
            title="[bold blue]Global Markets[/bold blue]",
        )
        tbl.add_column("Market",  style="bold", width=22)
        tbl.add_column("Price",               width=12)
        tbl.add_column("Change",              width=10)
        tbl.add_column("Nifty Impact",        width=20)

        for sym, data in gm.items():
            pct    = data.get("pct_change", 0)
            impact = data.get("score_contribution", 0)
            dirn   = data.get("direction", 1)
            eff    = pct * dirn
            pcol   = "green" if eff > 0.5 else ("red" if eff < -0.5 else "yellow")
            if   impact >  5: imp = f"[green]↑ bullish  +{impact:.0f} pts[/green]"
            elif impact < -5: imp = f"[red]↓ bearish  {impact:.0f} pts[/red]"
            else:             imp = f"[yellow]→ neutral[/yellow]"
            tbl.add_row(
                data.get("name", sym),
                f"{data.get('price', 0):,.1f}",
                f"[{pcol}]{pct:+.2f}%[/{pcol}]",
                imp,
            )
        console.print(tbl)
        console.print()

    # ── THE RECOMMENDATION ────────────────────────────────────────────────────

    def _trade_action(self):
        rec = self.rec
        if not rec:
            return

        action = rec.get("action", "AVOID")

        if action == "AVOID":
            console.print(Panel(
                Text.from_markup(
                    "[yellow bold]  NO TRADE TODAY[/yellow bold]\n\n"
                    f"[white]  {rec.get('reason', 'No clear directional bias.')}[/white]"
                ),
                title="[bold yellow]RECOMMENDATION[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            ))
            return

        if action == "SELL PUT":
            self._trade_action_sell_put()
        else:
            self._trade_action_buy_option()

    def _trade_action_sell_put(self):
        rec     = self.rec
        strike  = rec.get("strike", 0)
        expiry  = rec.get("expiry")
        ltp     = rec.get("ltp", 0)
        data_ok = rec.get("data_available", False)

        expiry_str = expiry.strftime("%d %b %Y") if expiry else "N/A"
        dte        = (expiry.date() - __import__("datetime").date.today()).days if expiry else 0
        dte_str    = f"  ({dte} days to expiry)" if dte > 0 else ""

        col = "dark_orange"   # orange for sell strategy

        txt = Text()

        # ── Quality banner ─────────────────────────────────────────────────
        quality = rec.get("quality_score", self.signal.get("sell_quality_score", 0))
        qlabel  = self.signal.get("sell_quality_label", "")
        qcol    = _quality_col(quality)
        qbar    = "█" * (quality // 10) + "░" * (10 - quality // 10)
        txt.append(f"  Sell Quality: {qbar}  {quality}/100  {qlabel}\n\n", style=f"bold {qcol}")

        # ── Big action line ─────────────────────────────────────────────────
        txt.append(
            f"  ⚡  SELL  NIFTY  {strike:,}  PE  —  {expiry_str}{dte_str}\n\n",
            style=f"bold {col}",
        )

        if data_ok:
            sell_at      = rec.get("sell_at",       ltp)
            sl_buy_at    = rec.get("sl_buy_at",      0)
            target_buy   = rec.get("target_buy_at",  0)
            spot_alert   = rec.get("spot_alert",     0)
            prem_coll    = rec.get("premium_collected", 0)
            max_profit   = rec.get("max_profit",     0)
            max_loss     = rec.get("max_loss",       0)
            lots         = rec.get("recommended_lots", 1)
            tot_units    = lots * 25
            margin_lot   = rec.get("margin_per_lot", 0)
            total_margin = rec.get("total_margin",   0)

            sl_pct     = round((sl_buy_at  - sell_at) / sell_at * 100) if sell_at else 0
            tgt_pct    = round((sell_at - target_buy) / sell_at * 100) if sell_at else 0
            rr         = round(max_profit / max_loss, 1) if max_loss else 0

            txt.append(f"  Sell (collect premium): ₹ {sell_at:.1f}  per unit\n\n", style="bold white")

            txt.append(f"  Stop Loss (buy back):   ₹ {sl_buy_at:.1f}  ", style="bold red")
            txt.append(f"(+{sl_pct}% — if premium rises here, BUY BACK immediately)\n", style="red")

            txt.append(f"  Target (buy back):      ₹ {target_buy:.1f}  ", style="bold green")
            txt.append(f"(−{tgt_pct}% — buy back here, KEEP the rest as profit)\n\n", style="green")

            txt.append(f"  Spot Alert Level:       ₹ {spot_alert:,.0f}  ", style="bold yellow")
            txt.append("(if Nifty falls to strike, monitor closely)\n\n", style="yellow")

            txt.append("  ─────────────────────────────────────────────────\n", style="dim")

            txt.append(f"  Lots to sell:       {lots} lot{'s' if lots > 1 else ''}  "
                       f"({tot_units} units × ₹{sell_at:.1f})\n", style="bold white")
            txt.append(f"  Premium collected:  ₹ {prem_coll:,.0f}  "
                       f"(cash received upfront)\n", style="bold yellow")
            txt.append(f"  Margin required:    ₹ {total_margin:,.0f}  "
                       f"(≈₹{margin_lot:,.0f}/lot — keep this in account)\n", style="white")
            txt.append(f"  Max Profit:         ₹ {max_profit:,.0f}  "
                       f"(if bought back at target)\n", style="green")
            txt.append(f"  Max Loss:           ₹ {max_loss:,.0f}  "
                       f"(if stop loss is hit)\n", style="red")
            txt.append(f"  Risk : Reward       1 : {rr}\n", style="bold white")

            exp_move = rec.get("expected_move_pct", 0)
            straddle = rec.get("straddle_price", 0)
            if exp_move > 0:
                txt.append(
                    f"\n  [dim]Market expects ±{exp_move:.1f}% move by expiry "
                    f"(ATM straddle: ₹{straddle:.0f})[/dim]\n",
                    style="",
                )

        else:
            txt.append(
                "  [yellow]Live price unavailable — NSE market may be closed.\n\n"
                "  When market opens, find this PE option and:\n"
                "    • Sell at market price (collect premium)\n"
                "    • Stop Loss  = buy back at 1.5× your sold price\n"
                "    • Target     = buy back at 30% of your sold price (keep 70%)[/yellow]\n",
                style="",
            )

        console.print(Panel(
            txt,
            title=f"[bold {col}]  ⚡  TRADE RECOMMENDATION — SHORT PUT  [/bold {col}]",
            border_style=col,
            padding=(1, 2),
        ))

    def _trade_action_buy_option(self):
        rec     = self.rec
        otype   = rec.get("option_type", "CE")
        strike  = rec.get("strike", 0)
        expiry  = rec.get("expiry")
        ltp     = rec.get("ltp", 0)
        targets = rec.get("targets", {})
        lot_info = rec.get("lot_info", {})
        e_range  = rec.get("entry_range", {})
        data_ok  = rec.get("data_available", False)

        col        = "bright_green" if otype == "CE" else "bright_red"
        expiry_str = expiry.strftime("%d %b %Y") if expiry else "N/A"
        dte        = (expiry.date() - __import__("datetime").date.today()).days if expiry else 0

        txt = Text()

        # ── Quality score banner ───────────────────────────────────────────
        quality = rec.get("quality_score", self.signal.get("quality_score", 0))
        qlabel  = self.signal.get("quality_label", "")
        qcol    = _quality_col(quality)
        qbar    = "█" * (quality // 10) + "░" * (10 - quality // 10)
        txt.append(f"  Quality: {qbar}  {quality}/100  {qlabel}\n\n", style=f"bold {qcol}")

        # ── Big action line ────────────────────────────────────────────────
        dte_str = f"  ({dte} days to expiry)" if dte > 0 else ""
        txt.append(
            f"  ⚡  BUY  NIFTY  {strike:,}  {otype}  —  {expiry_str}{dte_str}\n\n",
            style=f"bold {col}",
        )

        if data_ok:
            sl   = targets.get("stop_loss", 0)
            t1   = targets.get("target_1",  0)
            t2   = targets.get("target_2",  0)
            el   = e_range.get("low",  ltp)
            eh   = e_range.get("high", ltp)

            lots      = lot_info.get("recommended_lots", 1)
            lot_size  = lot_info.get("lot_size", 25)
            tot_units = lots * lot_size
            deployed  = round(ltp * tot_units)
            max_loss  = round((ltp - sl) * tot_units)
            t1_profit = round((t1  - ltp) * tot_units * 0.5)
            t2_profit = round((t2  - ltp) * tot_units * 0.5)
            total_pot = t1_profit + t2_profit
            rr        = round(total_pot / max_loss, 1) if max_loss else 0

            sl_pct = round((ltp - sl) / ltp * 100)
            t1_pct = round((t1  - ltp) / ltp * 100)
            t2_pct = round((t2  - ltp) / ltp * 100)

            txt.append(f"  Premium (LTP):    ₹ {ltp:.1f}\n", style="white")
            txt.append(f"  Entry zone:       ₹ {el:.1f}  –  ₹ {eh:.1f}\n\n", style="bold yellow")

            txt.append(f"  Stop Loss:        ₹ {sl:.1f}  ", style="bold red")
            txt.append(f"(−{sl_pct}%  — EXIT immediately if premium falls here)\n", style="red")

            txt.append(f"  Target 1:         ₹ {t1:.1f}  ", style="bold green")
            txt.append(f"(+{t1_pct}%  — book 50% of your position here)\n", style="green")

            txt.append(f"  Target 2:         ₹ {t2:.1f}  ", style="bold bright_green")
            txt.append(f"(+{t2_pct}%  — trail stop loss for the rest)\n\n", style="bright_green")

            txt.append("  ─────────────────────────────────────────────────\n", style="dim")

            txt.append(f"  Lots to buy:      {lots} lot{'s' if lots > 1 else ''}  "
                       f"({tot_units} units × ₹{ltp:.1f})\n", style="bold white")
            txt.append(f"  Capital to use:   ₹ {deployed:,.0f}\n", style="bold yellow")
            txt.append(f"  Max loss:         ₹ {max_loss:,.0f}  "
                       f"(if stop loss is hit)\n", style="red")
            txt.append(f"  Profit at T1:     ₹ {t1_profit:,.0f}  "
                       f"(booking 50% at Target 1)\n", style="green")
            txt.append(f"  Profit at T2:     ₹ {t2_profit:,.0f}  "
                       f"(on remaining 50% at Target 2)\n", style="bright_green")
            txt.append(f"  Risk : Reward     1 : {rr}\n", style="bold white")

            exp_move = rec.get("expected_move_pct", 0)
            straddle = rec.get("straddle_price", 0)
            if exp_move > 0:
                txt.append(
                    f"\n  [dim]Market expects ±{exp_move:.1f}% move by expiry "
                    f"(ATM straddle: ₹{straddle:.0f})[/dim]\n",
                    style="",
                )

        else:
            txt.append(
                "  [yellow]Live price unavailable — NSE market may be closed.\n\n"
                "  When market opens, look up this option and:\n"
                "    • Buy at market price\n"
                "    • Stop Loss  = 35% below your entry\n"
                "    • Target 1   = 50%  above entry  (book 50% here)\n"
                "    • Target 2   = 100% above entry  (trail stop)[/yellow]\n",
                style="",
            )

        console.print(Panel(
            txt,
            title=f"[bold {col}]  ⚡  TRADE RECOMMENDATION  [/bold {col}]",
            border_style=col,
            padding=(1, 2),
        ))

    # ── Why this trade ────────────────────────────────────────────────────────

    def _why_this_trade(self):
        rec = self.rec
        if not rec or rec.get("action") == "AVOID":
            return

        is_sell = rec.get("action") == "SELL PUT"
        reasons = []
        t = self.tech
        o = self.opts
        n = self.news

        # ADX — interpret differently for buyers vs sellers
        adx = t.get("adx", 0)
        if is_sell:
            if   adx < 15:  reasons.append(f"[green]✓  ADX {adx:.0f} — flat/ranging market — premium decays fast (ideal for selling)[/green]")
            elif adx < 20:  reasons.append(f"[green]✓  ADX {adx:.0f} — mild ranging — premium still good[/green]")
            elif adx < 25:  reasons.append(f"[yellow]→  ADX {adx:.0f} — building trend — watch direction carefully[/yellow]")
            elif t.get("di_bullish"):
                            reasons.append(f"[green]✓  ADX {adx:.0f} — strong UPtrend — put seller is safe (market moving away from strike)[/green]")
            else:           reasons.append(f"[red]✗  ADX {adx:.0f} — strong DOWNtrend — DANGEROUS for put sellers[/red]")
        else:
            if   adx >= 30: reasons.append(f"[green]✓  ADX {adx:.0f} — strong trend, ideal for directional options[/green]")
            elif adx >= 25: reasons.append(f"[green]✓  ADX {adx:.0f} — good trend in place[/green]")
            elif adx >= 20: reasons.append(f"[yellow]→  ADX {adx:.0f} — moderate trend, watch for reversal[/yellow]")
            else:           reasons.append(f"[red]✗  ADX {adx:.0f} — weak/no trend (ranging market)[/red]")

        # IV Rank — key for put sellers
        iv_rank = o.get("iv_rank", "NORMAL")
        if is_sell:
            iv_map = {
                "VERY_HIGH": f"[bright_green]✓  IV Rank VERY HIGH — collecting very rich premium, excellent for selling[/bright_green]",
                "HIGH":      f"[green]✓  IV Rank HIGH — premium is elevated, good time to sell[/green]",
                "NORMAL":    f"[yellow]→  IV Rank NORMAL — standard premium[/yellow]",
                "VERY_LOW":  f"[red]✗  IV Rank VERY LOW — thin premium, not worth the margin risk[/red]",
            }
            reasons.append(iv_map.get(iv_rank, f"[yellow]→  IV Rank {iv_rank}[/yellow]"))

        # Supertrend
        st_bull = t.get("st_bullish")
        if is_sell:
            if   st_bull is True:  reasons.append("[green]✓  Supertrend BULLISH — market is in uptrend, put sellers are safe[/green]")
            elif st_bull is False: reasons.append("[red]✗  Supertrend BEARISH — downtrend in place, puts may spike[/red]")
        else:
            if   st_bull is True:  reasons.append("[green]✓  Supertrend BULLISH — trend-following signal confirms buy[/green]")
            elif st_bull is False: reasons.append("[red]✗  Supertrend BEARISH — trend-following signal says sell[/red]")

        # EMA trend
        above_all = t.get("above_ema200") and t.get("above_ema50") and t.get("above_ema20")
        below_all = not t.get("above_ema200") and not t.get("above_ema50") and not t.get("above_ema20")
        if above_all:
            reasons.append("[green]✓  Above EMA 20 / 50 / 200 — strong uptrend on all timeframes[/green]")
        elif t.get("above_ema200"):
            reasons.append("[green]✓  Above EMA 200 — long-term uptrend intact[/green]")
        elif below_all:
            reasons.append("[red]✗  Below EMA 20 / 50 / 200 — downtrend on all timeframes[/red]")
        else:
            reasons.append("[yellow]→  Mixed EMA signals — short-term and long-term disagree[/yellow]")

        # RSI
        rsi = t.get("rsi", 50)
        if   rsi < 30:  reasons.append(f"[green]✓  RSI {rsi:.0f} — oversold, bounce expected[/green]")
        elif rsi < 45:  reasons.append(f"[green]✓  RSI {rsi:.0f} — plenty of room to move up[/green]")
        elif rsi < 60:  reasons.append(f"[yellow]→  RSI {rsi:.0f} — neutral zone[/yellow]")
        elif rsi < 70:  reasons.append(f"[yellow]→  RSI {rsi:.0f} — approaching overbought[/yellow]")
        else:           reasons.append(f"[red]✗  RSI {rsi:.0f} — overbought, momentum may fade[/red]")

        # PCR
        pcr_sig = o.get("pcr_signal", "NEUTRAL")
        pcr_val = o.get("pcr", 0)
        if is_sell:
            if pcr_sig in ("STRONG_BULLISH", "BULLISH"):
                reasons.append(f"[green]✓  PCR {pcr_val:.2f} — high put OI = put premiums elevated, good for selling[/green]")
            elif pcr_sig in ("STRONG_BEARISH", "BEARISH"):
                reasons.append(f"[red]✗  PCR {pcr_val:.2f} — market bearish, puts are expensive but risky to sell[/red]")
            else:
                reasons.append(f"[yellow]→  PCR {pcr_val:.2f} — neutral market stance[/yellow]")
        else:
            if   pcr_sig in ("STRONG_BULLISH", "BULLISH"):
                reasons.append(f"[green]✓  PCR {pcr_val:.2f} — heavy put writing, contrarian bullish[/green]")
            elif pcr_sig in ("STRONG_BEARISH", "BEARISH"):
                reasons.append(f"[red]✗  PCR {pcr_val:.2f} — heavy call writing, contrarian bearish[/red]")
            else:
                reasons.append(f"[yellow]→  PCR {pcr_val:.2f} — neutral[/yellow]")

        # Max Pain
        mp   = o.get("max_pain", 0)
        diff = o.get("max_pain_diff_pct", 0)
        strike = self.rec.get("strike", 0)
        if mp and is_sell and strike:
            if mp >= strike:
                reasons.append(
                    f"[green]✓  Max Pain ₹{mp:,.0f} is above your sold strike ₹{strike:,} — "
                    f"market gravity is on your side[/green]"
                )
            else:
                reasons.append(
                    f"[yellow]→  Max Pain ₹{mp:,.0f} is near/below your sold strike — "
                    f"monitor carefully[/yellow]"
                )
        elif mp and diff != 0:
            if diff > 0:
                reasons.append(f"[green]✓  Max Pain ₹{mp:,.0f} above spot — gravity pulls price up by expiry[/green]")
            else:
                reasons.append(f"[red]✗  Max Pain ₹{mp:,.0f} below spot — gravity pulls price down by expiry[/red]")

        # Global markets
        es = n.get("event_score", 0)
        if   es >  15: reasons.append(f"[green]✓  Global markets positive ({es:+.0f}) — tailwind for Nifty[/green]")
        elif es < -15: reasons.append(f"[red]✗  Global markets negative ({es:+.0f}) — headwind for Nifty[/red]")

        if not reasons:
            return

        txt = "\n".join(f"  {r}" for r in reasons[:8])
        console.print(Panel(
            txt,
            title="[bold white]WHY THIS TRADE[/bold white]",
            border_style="white",
            padding=(0, 1),
        ))

    # ── Indicator Legend ──────────────────────────────────────────────────────

    def _indicator_legend(self):
        lines = [
            "[bold cyan]SCORE  (−100 to +100)[/bold cyan]  Weighted combination of Technical + Options + News signals",
            "  [bright_green]+60 to +100[/bright_green] STRONG BULLISH   [green]+30 to +60[/green] BULLISH   [yellow]−30 to +30[/yellow] NEUTRAL",
            "  [red]−30 to −60[/red] BEARISH   [bright_red]−60 to −100[/bright_red] STRONG BEARISH",
            "",
            "[bold cyan]ADX  (0–100)[/bold cyan]  Trend STRENGTH only — not direction (from Average Directional Index)",
            "  [red]< 15[/red] Flat/Sideways (bad for buyers, great for sellers — theta burns fast)",
            "  [yellow]15–20[/yellow] Weak trend   [yellow]20–25[/yellow] Building trend   [bright_green]≥ 25[/bright_green] Strong trend",
            "",
            "[bold cyan]QUALITY  (0–100)[/bold cyan]  How good today's setup is for YOUR strategy (sell vs buy — different score)",
            "  [bright_green]≥ 70[/bright_green] STRONG (trade full size)   [green]55–69[/green] GOOD   [yellow]40–54[/yellow] WEAK   [red]< 40[/red] AVOID",
            "",
            "[bold cyan]VIX[/bold cyan]  India Fear Index — measures how expensive options are right now",
            "  [green]< 12[/green] Too calm/complacency   [bright_green]12–15[/bright_green] Ideal   [yellow]15–20[/yellow] Elevated",
            "  [red]20–25[/red] High fear (reduce size)   [bright_red]> 25[/bright_red] Extreme fear (half size)",
            "",
            "[bold cyan]SUPERTREND[/bold cyan]  Trend direction using ATR volatility bands (period 14, mult 3)",
            "  [green]↑ BULL[/green] = price above upper band (uptrend confirmed)",
            "  [red]↓ BEAR[/red] = price below lower band (downtrend confirmed)",
            "",
            "[bold cyan]PCR  (Put-Call Ratio)[/bold cyan]  Total Put OI ÷ Total Call OI  — CONTRARIAN indicator",
            "  [bright_green]> 1.5[/bright_green] Everyone buying puts → market near bottom (bullish signal)",
            "  [bright_red]< 0.5[/bright_red] Everyone buying calls → market near top (bearish signal)",
            "  When the crowd panic-buys puts, the market tends to reverse UP (and vice versa)",
            "",
            "[bold cyan]MAX PAIN[/bold cyan]  Strike where option writers (sellers) lose the least money",
            "  Market tends to pin near this level at expiry — useful gravity reference",
            "",
            "[bold cyan]OBV  (On Balance Volume)[/bold cyan]  Tracks whether volume is flowing INTO or OUT of the market",
            "  [green]Rising[/green] = smart money accumulating (bullish)   [red]Falling[/red] = distribution (bearish)",
            "",
            "[bold cyan]IV RANK[/bold cyan]  Where current Implied Volatility sits vs the past year's range",
            "  [green]VERY_LOW / NORMAL[/green] = options cheap — good for buyers",
            "  [bright_green]HIGH / VERY_HIGH[/bright_green] = options expensive — good for SELLERS (more premium to collect)",
        ]
        txt = "\n".join(f"  {line}" for line in lines)
        console.print(Panel(
            txt,
            title="[bold dim]  INDICATOR GLOSSARY  [/bold dim]",
            border_style="dim",
            padding=(0, 1),
        ))
        console.print()

    # ── Entry Point ───────────────────────────────────────────────────────────

    def display(self):
        self._header()
        self._snapshot_bar()
        self._major_event_alert()
        self._global_markets()
        self._trade_action()
        console.print()
        self._why_this_trade()
        console.print()
        self._indicator_legend()
        console.print(Rule(style="dim"))
        console.print()

    def save_log(self, path: str = "logs/predictions.csv"):
        """Append today's prediction to a CSV for performance tracking."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            rec = self.rec
            row = {
                "timestamp":    self.ts.isoformat(),
                "spot":         self.spot,
                "vix":          self.vix,
                "direction":    self.signal.get("direction",  ""),
                "confidence":   self.signal.get("confidence", ""),
                "score":        self.signal.get("score",       0),
                "action":       rec.get("action",      ""),
                "strike":       rec.get("strike",       0),
                "option_type":  rec.get("option_type",  ""),
                "expiry":       rec.get("expiry",       ""),
                "ltp":          rec.get("ltp",           0),
                "stop_loss":    rec.get("targets", {}).get("stop_loss", 0),
                "target_1":     rec.get("targets", {}).get("target_1",  0),
                "target_2":     rec.get("targets", {}).get("target_2",  0),
                "pcr":          self.opts.get("pcr",       0),
                "max_pain":     self.opts.get("max_pain",  0),
                "rsi":          self.tech.get("rsi",        0),
                "macd_bullish": self.tech.get("macd_bullish", False),
                "news_score":   self.news.get("news_score",   0),
                "event_score":  self.news.get("event_score",  0),
                "major_event":  self.news.get("has_major_event", False),
            }
            exists = os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=row.keys())
                if not exists:
                    w.writeheader()
                w.writerow(row)
            console.print(f"[dim]Prediction logged → {path}[/dim]")
        except Exception as exc:
            logger.error("Log save failed: %s", exc)

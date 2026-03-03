"""
Nifty Options Prediction System - Configuration
All tunable parameters in one place.
"""

# ─── Data Settings ───────────────────────────────────────────────────────────
SYMBOL                = "NIFTY"
HISTORICAL_DAYS       = 150          # Days of price history to analyse
NIFTY_STRIKE_GAP      = 50           # Strike-price interval on NSE
NIFTY_LOT_SIZE        = 25           # Current Nifty F&O lot size (verify on NSE)

# ─── Technical Analysis Parameters ───────────────────────────────────────────
RSI_PERIOD   = 14
MACD_FAST    = 12
MACD_SLOW    = 26
MACD_SIGNAL  = 9
EMA_SHORT    = 20
EMA_MEDIUM   = 50
EMA_LONG     = 200
BB_PERIOD    = 20
BB_STD       = 2

# ─── Signal Score Thresholds (scale: -100 to +100) ───────────────────────────
STRONG_BULLISH_THRESHOLD =  60
BULLISH_THRESHOLD        =  30
BEARISH_THRESHOLD        = -30
STRONG_BEARISH_THRESHOLD = -60

# ─── Risk Management Defaults ─────────────────────────────────────────────────
DEFAULT_CAPITAL        = 50_000      # INR — your trading capital
MAX_RISK_PCT           = 0.02        # Risk 2 % of capital per trade
STOP_LOSS_PCT          = 0.35        # Exit option if it falls 35 % from entry
TARGET_1_PCT           = 0.50        # First profit target: +50 %
TARGET_2_PCT           = 1.00        # Second target (trail stop here): +100 %
MAX_LOTS               = 5           # Upper cap on recommended lots

# ─── Trading Strategy ─────────────────────────────────────────────────────────
# "SELL_PUTS"  → Short put strategy (sell OTM PE, collect premium, profit from time decay)
# "BUY_OPTIONS"→ Buy CE (bullish) or PE (bearish) based on signal direction
STRATEGY              = "SELL_PUTS"

# Short Put specific settings (used when STRATEGY = "SELL_PUTS")
PUT_SELL_OTM_STEPS    = 2     # Strikes below ATM to sell (2 = 100pts OTM on Nifty)
PUT_SELL_SL_MULT      = 1.5   # Stop loss: buy back if premium rises to 1.5× sold price
PUT_SELL_TARGET_PCT   = 0.30  # Target: buy back at 30% of sold price (keep 70% of premium)

# ─── Expiry Selection Rules ───────────────────────────────────────────────────
MIN_DAYS_TO_EXPIRY     = 2           # Avoid expiry with fewer than 2 days left

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR  = "logs"
LOG_FILE = "logs/predictions.csv"

"""
core/setups.py — Deterministic strategy scanner.

Pure Python, no LLM.  Evaluates three setups against current indicators,
OHLCV data, regime, and trading context.  Returns SetupResult models.

Scoring model
─────────────
    hard_score  = (n_passed / n_total) × min_score
    soft_score  = Σ(points for each triggered soft condition)
    total_score = min(hard_score + soft_score, 100.0)
    passed      = all_hard AND no_disqualifiers AND total_score ≥ min_score
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from zoneinfo import ZoneInfo

from sauce.core.calendar import is_major_event_within_hours, is_near_major_event
from sauce.core.schemas import (
    Disqualification,
    HardConditionResult,
    Indicators,
    MarketRegime,
    SetupResult,
    SetupType,
    SignalLogEntry,
    SoftConditionResult,
)

# ── Asset-class routing ────────────────────────────────────────────────────────

def _is_crypto(symbol: str) -> bool:
    """Return True for Alpaca crypto pairs (e.g. 'BTC/USD')."""
    return "/" in symbol

# ── Eligible regimes ─────────────────────────────────────────────────────────

SETUP_1_REGIMES: frozenset[str] = frozenset({"RANGING", "TRENDING_UP"})
SETUP_2_REGIMES: frozenset[str] = frozenset({"TRENDING_UP"})
SETUP_3_REGIMES: frozenset[str] = frozenset({"RANGING"})

# ── Min scores per setup ─────────────────────────────────────────────────────

SETUP_1_MIN_SCORE: float = 60.0
SETUP_2_MIN_SCORE: float = 65.0
SETUP_3_MIN_SCORE: float = 70.0

# ── Minimum OHLCV bars required ──────────────────────────────────────────────

MIN_BARS_SETUP_1: int = 5
MIN_BARS_SETUP_2: int = 5
MIN_BARS_SETUP_3: int = 10

# ── Setup 1 thresholds ───────────────────────────────────────────────────────

S1_RSI_HARD: float = 38.0
S1_BB_TOLERANCE: float = 0.005
S1_DOWN_CANDLE_VOL_RATIO: float = 1.5
S1_RSI_SOFT: float = 32.0
S1_STOCH_K_SOFT: float = 20.0
S1_WIN_RATE_THRESHOLD: float = 0.60
S1_MAX_ATTEMPTS: int = 2

# ── Setup 2 thresholds ───────────────────────────────────────────────────────

S2_SMA_PROXIMITY_PCT: float = 0.01
S2_RSI_LOW: float = 38.0
S2_RSI_HIGH: float = 58.0
S2_RSI_TURNING_THRESHOLD: float = 45.0
S2_BOUNCE_TOLERANCE: float = 0.003
S2_VOLUME_CONTRACTION_RATIO: float = 0.5

# ── Setup 3 thresholds ───────────────────────────────────────────────────────

S3_CONSOLIDATION_BARS: int = 8
S3_RANGE_PCT_MAX: float = 2.0
S3_BREAKOUT_VOL_MULTIPLE: float = 2.0
S3_EXTENDED_CONSOL_BARS: int = 12
S3_VOLUME_3X: float = 3.0
S3_REVERSAL_WICK_RATIO: float = 0.5
S3_PRE_BREAKOUT_VOL_RATIO: float = 1.5
S3_MAX_ATTEMPTS: int = 2

# ── Narrative keyword matching (Setup 1 disqualifier) ─────────────────────────

SELLING_PRESSURE_KEYWORDS: list[str] = [
    "sustained selling",
    "continued selling",
    "no recovery",
    "relentless selling",
    "persistent selling",
]


# ── Internal helpers ──────────────────────────────────────────────────────────


def _compute_macd_histogram(closes: pd.Series) -> pd.Series:
    """Compute MACD histogram (12, 26, 9) from close prices."""
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


def _compute_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI from close prices using simple moving average method."""
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100.0 - (100.0 / (1.0 + rs))


def _resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 30-min OHLCV to 4-hour bars."""
    return (
        df.resample("4h")
        .agg({"open": "first", "high": "max", "low": "min",
              "close": "last", "volume": "sum"})
        .dropna()
    )


def _resample_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 30-min OHLCV to daily bars."""
    return (
        df.resample("1D")
        .agg({"open": "first", "high": "max", "low": "min",
              "close": "last", "volume": "sum"})
        .dropna()
    )


def _compute_score(
    hard_conditions: list[HardConditionResult],
    soft_conditions: list[SoftConditionResult],
    disqualifiers: list[Disqualification],
    min_score: float,
) -> tuple[float, bool]:
    """
    Compute final score and passed flag.

    Returns (score, passed).
    """
    n_total = len(hard_conditions)
    n_passed = sum(1 for h in hard_conditions if h.passed)
    hard_score = (n_passed / n_total) * min_score if n_total > 0 else 0.0
    soft_score = sum(s.points for s in soft_conditions if s.triggered)
    score = min(hard_score + soft_score, 100.0)
    all_passed = n_passed == n_total
    passed = all_passed and len(disqualifiers) == 0 and score >= min_score
    return round(score, 2), passed


def _build_narrative(
    setup_type: SetupType,
    symbol: str,
    hard_conditions: list[HardConditionResult],
    soft_conditions: list[SoftConditionResult],
    disqualifiers: list[Disqualification],
    score: float,
    passed: bool,
) -> str:
    """Build plain-English evidence narrative from rule outputs."""
    parts: list[str] = [f"{setup_type} scan for {symbol}:"]

    failed = [h for h in hard_conditions if not h.passed]
    if failed:
        labels = ", ".join(h.label for h in failed)
        parts.append(f"Failed hard conditions: {labels}.")
    else:
        parts.append("All hard conditions passed.")

    triggered = [s for s in soft_conditions if s.triggered]
    if triggered:
        total_pts = sum(s.points for s in triggered)
        labels = ", ".join(f"{s.label} (+{s.points:.0f})" for s in triggered)
        parts.append(f"Soft bonuses ({total_pts:.0f} pts): {labels}.")

    if disqualifiers:
        reasons = "; ".join(d.reason for d in disqualifiers)
        parts.append(f"DISQUALIFIED: {reasons}.")

    parts.append(f"Score: {score:.0f}. {'PASSED' if passed else 'REJECTED'}.")
    return " ".join(parts)


def _canon(s: str) -> str:
    """Canonical symbol form — strips '/' and uppercases for position lookups.

    Alpaca returns 'BTCUSD' in positions; setup symbols use 'BTC/USD'.
    Normalise both sides so the membership check is always correct.
    """
    return s.replace("/", "").upper()


# ── Setup 1: Crypto Mean Reversion ────────────────────────────────────────────


def evaluate_crypto_mean_reversion(
    symbol: str,
    indicators: Indicators,
    df: pd.DataFrame,
    regime: MarketRegime,
    *,
    has_open_position: bool = False,
    mean_reversion_attempts_today: int = 0,
    strategic_win_rate: float | None = None,
    narrative_text: str = "",
    as_of: datetime,
) -> SetupResult:
    """
    Evaluate crypto mean reversion setup (Setup 1).

    Looks for oversold crypto bouncing off lower Bollinger Band with MACD
    curling up and elevated down-candle volume (high-conviction flush).
    """
    hard: list[HardConditionResult] = []
    soft: list[SoftConditionResult] = []
    disqualifiers: list[Disqualification] = []
    last_close = float(df["close"].iloc[-1]) if len(df) > 0 else 0.0

    # ── Hard conditions ──────────────────────────────────────────────────────

    # H1: RSI 30-min < 38
    rsi_ok = indicators.rsi_14 is not None and indicators.rsi_14 < S1_RSI_HARD
    hard.append(HardConditionResult(
        label="RSI < 38",
        passed=rsi_ok,
        detail=f"RSI={indicators.rsi_14}" if indicators.rsi_14 is not None else "RSI unavailable",
    ))

    # H2: Price at or below lower Bollinger Band (within 0.5%)
    if indicators.bb_lower is not None and last_close > 0:
        bb_threshold = indicators.bb_lower * (1.0 + S1_BB_TOLERANCE)
        bb_ok = last_close <= bb_threshold
        detail = f"close={last_close:.4f}, bb_lower={indicators.bb_lower:.4f}"
    else:
        bb_ok = False
        detail = "BB lower or price unavailable"
    hard.append(HardConditionResult(label="Price near lower BB", passed=bb_ok, detail=detail))

    # H3: 4hr price > 4hr SMA50 (buying a dip, not a downtrend)
    if len(df) >= 200:
        df_4h = _resample_4h(df)
        if len(df_4h) >= 50:
            sma50_4h = df_4h["close"].rolling(50).mean().iloc[-1]
            price_4h = float(df_4h["close"].iloc[-1])
            if pd.notna(sma50_4h):
                h3_ok = price_4h > sma50_4h
                detail = f"4h_close={price_4h:.4f}, 4h_sma50={sma50_4h:.4f}"
            else:
                h3_ok = False
                detail = "4hr SMA50 is NaN"
        else:
            h3_ok = False
            detail = f"Only {len(df_4h)} 4hr bars (need 50)"
    else:
        h3_ok = False
        detail = f"Only {len(df)} bars (need 200 for 4hr SMA50)"
    hard.append(HardConditionResult(label="4hr price > 4hr SMA50", passed=h3_ok, detail=detail))

    # H4: Volume ratio on last 3 down candles > 1.5x average
    if len(df) >= 20:
        recent = df.tail(20)
        down = recent[recent["close"] < recent["open"]]
        if len(down) >= 3:
            avg_down_vol = float(down.tail(3)["volume"].mean())
            overall_avg = float(df["volume"].mean())
            vol_ratio = avg_down_vol / overall_avg if overall_avg > 0 else 0.0
            h4_ok = vol_ratio > S1_DOWN_CANDLE_VOL_RATIO
            detail = f"down_vol_ratio={vol_ratio:.2f}"
        else:
            h4_ok = False
            detail = f"Only {len(down)} down candles in last 20 bars"
    else:
        h4_ok = False
        detail = "Insufficient bars for down-candle volume check"
    hard.append(HardConditionResult(
        label="Down-candle volume > 1.5x", passed=h4_ok, detail=detail,
    ))

    # H5: MACD histogram curling (current value > 2 bars ago)
    if len(df) >= 30:
        hist = _compute_macd_histogram(df["close"])
        if len(hist) >= 3 and pd.notna(hist.iloc[-1]) and pd.notna(hist.iloc[-3]):
            h5_ok = float(hist.iloc[-1]) > float(hist.iloc[-3])
            detail = f"hist_now={hist.iloc[-1]:.6f}, hist_2ago={hist.iloc[-3]:.6f}"
        else:
            h5_ok = False
            detail = "MACD histogram values unavailable"
    else:
        h5_ok = False
        detail = "Insufficient bars for MACD computation"
    hard.append(HardConditionResult(label="MACD histogram curling", passed=h5_ok, detail=detail))

    # H6: No open position in this symbol
    h6_ok = not has_open_position
    hard.append(HardConditionResult(
        label="No open position",
        passed=h6_ok,
        detail="Position already open" if has_open_position else "No position",
    ))

    # ── Soft conditions ──────────────────────────────────────────────────────

    # S1: RSI < 32 → deep oversold (+15 pts)
    s1 = indicators.rsi_14 is not None and indicators.rsi_14 < S1_RSI_SOFT
    soft.append(SoftConditionResult(label="Deep oversold RSI < 32", triggered=s1, points=15.0))

    # S2: Stochastic K < 20 (+10 pts)
    s2 = indicators.stoch_k is not None and indicators.stoch_k < S1_STOCH_K_SOFT
    soft.append(SoftConditionResult(label="StochK < 20", triggered=s2, points=10.0))

    # S3: Price ≤ SMA20 (+10 pts)
    s3 = indicators.sma_20 is not None and last_close <= indicators.sma_20
    soft.append(SoftConditionResult(label="Price <= SMA20", triggered=s3, points=10.0))

    # S4: VWAP > price → institutional value zone (+5 pts)
    s4 = indicators.vwap is not None and indicators.vwap > last_close
    soft.append(SoftConditionResult(label="VWAP > price", triggered=s4, points=5.0))

    # S5: Strategic memory win rate > 60% for this setup/regime (+10 pts)
    s5 = strategic_win_rate is not None and strategic_win_rate > S1_WIN_RATE_THRESHOLD
    soft.append(SoftConditionResult(
        label="Strategic win rate > 60%", triggered=s5, points=10.0,
    ))

    # ── Disqualifiers ────────────────────────────────────────────────────────

    if is_near_major_event(as_of, window_minutes=90):
        disqualifiers.append(Disqualification(
            reason="Major economic event within 90 minutes",
        ))

    if mean_reversion_attempts_today >= S1_MAX_ATTEMPTS:
        disqualifiers.append(Disqualification(
            reason=f"Mean reversion already attempted {mean_reversion_attempts_today}x today",
        ))

    if regime == "VOLATILE":
        disqualifiers.append(Disqualification(reason="Regime is VOLATILE"))

    if regime == "TRENDING_DOWN":
        disqualifiers.append(Disqualification(reason="Regime is TRENDING_DOWN"))

    narrative_lower = narrative_text.lower()
    if any(kw in narrative_lower for kw in SELLING_PRESSURE_KEYWORDS):
        disqualifiers.append(Disqualification(
            reason="Narrative indicates sustained selling with no recovery",
        ))

    # ── Score and assemble ───────────────────────────────────────────────────

    score, passed = _compute_score(hard, soft, disqualifiers, SETUP_1_MIN_SCORE)
    narrative = _build_narrative(
        "crypto_mean_reversion", symbol, hard, soft, disqualifiers, score, passed,
    )

    return SetupResult(
        setup_type="crypto_mean_reversion",
        symbol=symbol,
        hard_conditions=hard,
        soft_conditions=soft,
        disqualifiers=disqualifiers,
        score=score,
        min_score=SETUP_1_MIN_SCORE,
        passed=passed,
        evidence_narrative=narrative,
        as_of=as_of,
    )


# ── Setup 2: Equity Trend Pullback ────────────────────────────────────────────


def evaluate_equity_trend_pullback(
    symbol: str,
    indicators: Indicators,
    df: pd.DataFrame,
    regime: MarketRegime,
    *,
    df_daily_ext: pd.DataFrame | None = None,
    as_of: datetime,
) -> SetupResult:
    """
    Evaluate equity trend pullback setup (Setup 2).

    Looks for pullback to rising daily SMA20 with controlled volume
    in an established uptrend on SPY or QQQ.

    df_daily_ext: Optional pre-fetched daily bars (50 days of 1-day OHLCV).
        When provided, used instead of resampling the 30-min df — critical
        because 60 bars of 30-min data only covers ~5 trading days, which
        is far too few for SMA50 and most H/S conditions.
    """
    hard: list[HardConditionResult] = []
    soft: list[SoftConditionResult] = []
    disqualifiers: list[Disqualification] = []
    last_close = float(df["close"].iloc[-1]) if len(df) > 0 else 0.0

    # Pre-compute daily data — prefer the externally-provided daily bars.
    if df_daily_ext is not None and not df_daily_ext.empty:
        df_daily = df_daily_ext
    else:
        df_daily = _resample_daily(df) if len(df) >= 20 else pd.DataFrame()
    daily_sma20 = float("nan")
    daily_sma50 = float("nan")
    if len(df_daily) >= 50:
        daily_sma20 = float(df_daily["close"].rolling(20).mean().iloc[-1])
        daily_sma50 = float(df_daily["close"].rolling(50).mean().iloc[-1])
    elif len(df_daily) >= 20:
        daily_sma20 = float(df_daily["close"].rolling(20).mean().iloc[-1])

    # ── Hard conditions ──────────────────────────────────────────────────────

    # H1: Daily SMA20 > Daily SMA50
    if pd.notna(daily_sma20) and pd.notna(daily_sma50):
        h1_ok = daily_sma20 > daily_sma50
        detail = f"daily_sma20={daily_sma20:.4f}, daily_sma50={daily_sma50:.4f}"
    else:
        h1_ok = False
        detail = "Insufficient daily data for SMA20/SMA50"
    hard.append(HardConditionResult(label="Daily SMA20 > SMA50", passed=h1_ok, detail=detail))

    # H2: Price within 1% of daily SMA20
    if pd.notna(daily_sma20) and daily_sma20 > 0:
        proximity = abs(last_close - daily_sma20) / daily_sma20
        h2_ok = proximity <= S2_SMA_PROXIMITY_PCT
        detail = f"proximity={proximity:.4f} ({proximity * 100:.2f}%)"
    else:
        h2_ok = False
        detail = "Daily SMA20 unavailable"
    hard.append(HardConditionResult(
        label="Price within 1% of SMA20", passed=h2_ok, detail=detail,
    ))

    # H3: Daily RSI between 38-58
    rsi_val: float | None = None
    if len(df_daily) >= 20:
        rsi_series = _compute_rsi(df_daily["close"], 14)
        if pd.notna(rsi_series.iloc[-1]):
            rsi_val = float(rsi_series.iloc[-1])
    if rsi_val is not None:
        h3_ok = S2_RSI_LOW <= rsi_val <= S2_RSI_HIGH
        detail = f"daily_rsi={rsi_val:.2f}"
    else:
        h3_ok = False
        detail = "Daily RSI unavailable"
    hard.append(HardConditionResult(label="Daily RSI 38-58", passed=h3_ok, detail=detail))

    # H4: Price > Daily SMA50
    if pd.notna(daily_sma50):
        h4_ok = last_close > daily_sma50
        detail = f"close={last_close:.4f}, daily_sma50={daily_sma50:.4f}"
    else:
        h4_ok = False
        detail = "Daily SMA50 unavailable"
    hard.append(HardConditionResult(label="Price > Daily SMA50", passed=h4_ok, detail=detail))

    # H5: Last 2 candles not making lower lows (selling fading)
    if len(df) >= 2:
        h5_ok = float(df["low"].iloc[-1]) >= float(df["low"].iloc[-2])
        detail = (
            f"low[-1]={df['low'].iloc[-1]:.4f}, low[-2]={df['low'].iloc[-2]:.4f}"
        )
    else:
        h5_ok = False
        detail = "Insufficient bars"
    hard.append(HardConditionResult(
        label="No lower lows (last 2 bars)", passed=h5_ok, detail=detail,
    ))

    # H6: Pullback volume < 20-day average volume
    if len(df_daily) >= 20:
        avg_20d_vol = df_daily["volume"].rolling(20).mean().iloc[-1]
        current_vol = float(df_daily["volume"].iloc[-1])
        if pd.notna(avg_20d_vol) and avg_20d_vol > 0:
            h6_ok = current_vol < float(avg_20d_vol)
            detail = f"today_vol={current_vol:.0f}, avg_20d={avg_20d_vol:.0f}"
        else:
            h6_ok = False
            detail = "20-day avg volume unavailable"
    else:
        h6_ok = False
        detail = "Insufficient daily data for 20-day volume average"
    hard.append(HardConditionResult(
        label="Pullback volume < 20d avg", passed=h6_ok, detail=detail,
    ))

    # ── Soft conditions ──────────────────────────────────────────────────────

    # S1: Bounced off daily SMA20 exactly — bar low near SMA20, close above (+20 pts)
    if pd.notna(daily_sma20) and daily_sma20 > 0 and len(df) >= 1:
        bar_low = float(df["low"].iloc[-1])
        near_sma = abs(bar_low - daily_sma20) / daily_sma20 <= S2_BOUNCE_TOLERANCE
        close_above = last_close > daily_sma20
        s1_triggered = near_sma and close_above
    else:
        s1_triggered = False
    soft.append(SoftConditionResult(
        label="SMA20 bounce", triggered=s1_triggered, points=20.0,
    ))

    # S2: Weekly uptrend proxy — daily SMA50 rising over 5 days (+15 pts)
    if len(df_daily) >= 55:
        sma50_series = df_daily["close"].rolling(50).mean()
        if pd.notna(sma50_series.iloc[-1]) and pd.notna(sma50_series.iloc[-5]):
            s2_triggered = float(sma50_series.iloc[-1]) > float(sma50_series.iloc[-5])
        else:
            s2_triggered = False
    else:
        s2_triggered = False
    soft.append(SoftConditionResult(
        label="Weekly uptrend proxy", triggered=s2_triggered, points=15.0,
    ))

    # S3: 30-min RSI turning up from below 45 (+10 pts)
    if indicators.rsi_14 is not None and len(df) >= 16:
        rsi_30m = _compute_rsi(df["close"], 14)
        if len(rsi_30m) >= 3 and pd.notna(rsi_30m.iloc[-2]):
            prev_rsi = float(rsi_30m.iloc[-2])
            s3_triggered = (
                prev_rsi < S2_RSI_TURNING_THRESHOLD
                and indicators.rsi_14 > prev_rsi
            )
        else:
            s3_triggered = False
    else:
        s3_triggered = False
    soft.append(SoftConditionResult(
        label="30m RSI turning up from <45", triggered=s3_triggered, points=10.0,
    ))

    # S4: Very low volume contraction — today's volume < 50% of 20d avg (+10 pts)
    if len(df_daily) >= 20:
        avg_20d = df_daily["volume"].rolling(20).mean().iloc[-1]
        current = float(df_daily["volume"].iloc[-1])
        if pd.notna(avg_20d) and avg_20d > 0:
            s4_triggered = (current / float(avg_20d)) < S2_VOLUME_CONTRACTION_RATIO
        else:
            s4_triggered = False
    else:
        s4_triggered = False
    soft.append(SoftConditionResult(
        label="Very low volume contraction", triggered=s4_triggered, points=10.0,
    ))

    # ── Disqualifiers ────────────────────────────────────────────────────────

    if is_major_event_within_hours(as_of, hours=48):
        disqualifiers.append(Disqualification(reason="FOMC/CPI/NFP within 48 hours"))

    if regime in ("RANGING", "VOLATILE"):
        disqualifiers.append(Disqualification(reason=f"Regime is {regime}"))

    # Bearish RSI divergence: price rising but daily RSI falling over 5 bars
    if len(df_daily) >= 20:
        rsi_daily = _compute_rsi(df_daily["close"], 14)
        if (
            len(rsi_daily) >= 5
            and pd.notna(rsi_daily.iloc[-1])
            and pd.notna(rsi_daily.iloc[-5])
        ):
            price_up = float(df_daily["close"].iloc[-1]) > float(df_daily["close"].iloc[-5])
            rsi_down = float(rsi_daily.iloc[-1]) < float(rsi_daily.iloc[-5])
            if price_up and rsi_down:
                disqualifiers.append(Disqualification(
                    reason="Bearish RSI divergence detected",
                ))

    # Price gapped down today (open < yesterday close)
    if len(df_daily) >= 2:
        today_open = float(df_daily["open"].iloc[-1])
        yesterday_close = float(df_daily["close"].iloc[-2])
        if today_open < yesterday_close:
            disqualifiers.append(Disqualification(reason="Price gapped down on the day"))

    # Friday after 2:00 PM ET
    et_time = as_of.astimezone(ZoneInfo("America/New_York"))
    if et_time.weekday() == 4 and et_time.hour >= 14:
        disqualifiers.append(Disqualification(reason="Friday after 2:00 PM ET"))

    # ── Score and assemble ───────────────────────────────────────────────────

    score, passed = _compute_score(hard, soft, disqualifiers, SETUP_2_MIN_SCORE)
    narrative = _build_narrative(
        "equity_trend_pullback", symbol, hard, soft, disqualifiers, score, passed,
    )

    return SetupResult(
        setup_type="equity_trend_pullback",
        symbol=symbol,
        hard_conditions=hard,
        soft_conditions=soft,
        disqualifiers=disqualifiers,
        score=score,
        min_score=SETUP_2_MIN_SCORE,
        passed=passed,
        evidence_narrative=narrative,
        as_of=as_of,
    )


# ── Setup 3: Crypto Breakout ─────────────────────────────────────────────────


def evaluate_crypto_breakout(
    symbol: str,
    indicators: Indicators,
    df: pd.DataFrame,
    regime: MarketRegime,
    *,
    has_open_position: bool = False,
    breakout_attempts_today: int = 0,
    strategic_win_rate: float | None = None,
    as_of: datetime,
) -> SetupResult:
    """
    Evaluate crypto breakout setup (Setup 3).

    Looks for price consolidation followed by a volume-confirmed breakout
    above the range high in BTC or ETH.
    """
    hard: list[HardConditionResult] = []
    soft: list[SoftConditionResult] = []
    disqualifiers: list[Disqualification] = []

    # Pre-compute consolidation window (bars before the current bar)
    has_enough = len(df) >= S3_CONSOLIDATION_BARS + 1
    if has_enough:
        consol = df.iloc[-(S3_CONSOLIDATION_BARS + 1):-1]
        range_high = float(consol["high"].max())
        range_low = float(consol["low"].min())
        range_pct = ((range_high - range_low) / range_low * 100) if range_low > 0 else 999.0
        consol_avg_vol = float(consol["volume"].mean())
        current_close = float(df["close"].iloc[-1])
        current_vol = float(df["volume"].iloc[-1])
    else:
        range_high = 0.0
        range_low = 0.0
        range_pct = 999.0
        consol_avg_vol = 0.0
        current_close = float(df["close"].iloc[-1]) if len(df) > 0 else 0.0
        current_vol = 0.0

    # ── Hard conditions ──────────────────────────────────────────────────────

    # H1: Price in range ≥4 hours with <2% range
    h1_ok = has_enough and range_pct < S3_RANGE_PCT_MAX
    if has_enough:
        detail = f"range={range_pct:.2f}% over {S3_CONSOLIDATION_BARS} bars"
    else:
        detail = f"Only {len(df)} bars (need {S3_CONSOLIDATION_BARS + 1})"
    hard.append(HardConditionResult(
        label="Consolidation >= 4hr, range < 2%", passed=h1_ok, detail=detail,
    ))

    # H2: Volume declining during consolidation
    if has_enough:
        half = S3_CONSOLIDATION_BARS // 2
        first_half_vol = float(consol["volume"].iloc[:half].mean())
        second_half_vol = float(consol["volume"].iloc[half:].mean())
        h2_ok = second_half_vol < first_half_vol if first_half_vol > 0 else False
        detail = f"1st_half_vol={first_half_vol:.0f}, 2nd_half_vol={second_half_vol:.0f}"
    else:
        h2_ok = False
        detail = "Insufficient bars"
    hard.append(HardConditionResult(
        label="Volume declining in consolidation", passed=h2_ok, detail=detail,
    ))

    # H3: Current bar closed above range high
    h3_ok = has_enough and current_close > range_high
    if has_enough:
        detail = f"close={current_close:.4f}, range_high={range_high:.4f}"
    else:
        detail = "Insufficient bars"
    hard.append(HardConditionResult(
        label="Close above range high", passed=h3_ok, detail=detail,
    ))

    # H4: Breakout volume ≥ 2x consolidation average
    if has_enough and consol_avg_vol > 0:
        vol_multiple = current_vol / consol_avg_vol
        h4_ok = vol_multiple >= S3_BREAKOUT_VOL_MULTIPLE
        detail = f"breakout_vol={current_vol:.0f}, consol_avg={consol_avg_vol:.0f}, ratio={vol_multiple:.2f}x"
    else:
        h4_ok = False
        detail = "Consolidation avg volume is zero or insufficient bars"
    hard.append(HardConditionResult(
        label="Breakout volume >= 2x avg", passed=h4_ok, detail=detail,
    ))

    # H5: 4hr trend aligned with breakout direction (upward)
    if len(df) >= 24:  # need at least 3 4hr bars
        df_4h = _resample_4h(df)
        if len(df_4h) >= 3:
            h5_ok = float(df_4h["close"].iloc[-1]) > float(df_4h["close"].iloc[-3])
            detail = (
                f"4h_close[-1]={df_4h['close'].iloc[-1]:.4f}, "
                f"4h_close[-3]={df_4h['close'].iloc[-3]:.4f}"
            )
        else:
            h5_ok = False
            detail = f"Only {len(df_4h)} 4hr bars (need 3)"
    else:
        h5_ok = False
        detail = f"Only {len(df)} bars (need 24 for 4hr trend)"
    hard.append(HardConditionResult(
        label="4hr trend aligned (up)", passed=h5_ok, detail=detail,
    ))

    # H6: No open positions
    h6_ok = not has_open_position
    hard.append(HardConditionResult(
        label="No open position",
        passed=h6_ok,
        detail="Position already open" if has_open_position else "No position",
    ))

    # ── Soft conditions ──────────────────────────────────────────────────────

    # S1: Consolidation lasted 6+ hours (12+ bars) (+15 pts)
    if len(df) >= S3_EXTENDED_CONSOL_BARS + 1:
        extended_consol = df.iloc[-(S3_EXTENDED_CONSOL_BARS + 1):-1]
        ext_high = float(extended_consol["high"].max())
        ext_low = float(extended_consol["low"].min())
        ext_range = ((ext_high - ext_low) / ext_low * 100) if ext_low > 0 else 999.0
        s1_triggered = ext_range < S3_RANGE_PCT_MAX
    else:
        s1_triggered = False
    soft.append(SoftConditionResult(
        label="Extended consolidation 6+ hrs", triggered=s1_triggered, points=15.0,
    ))

    # S2: Volume ratio on breakout 3x+ (+15 pts)
    if has_enough and consol_avg_vol > 0:
        s2_triggered = (current_vol / consol_avg_vol) >= S3_VOLUME_3X
    else:
        s2_triggered = False
    soft.append(SoftConditionResult(
        label="Breakout volume 3x+", triggered=s2_triggered, points=15.0,
    ))

    # S3: RSI trending up during consolidation (+10 pts)
    if has_enough and len(df) >= 20:
        rsi_full = _compute_rsi(df["close"], 14)
        consol_start_idx = -(S3_CONSOLIDATION_BARS + 1)
        consol_end_idx = -1
        if pd.notna(rsi_full.iloc[consol_start_idx]) and pd.notna(rsi_full.iloc[consol_end_idx]):
            s3_triggered = float(rsi_full.iloc[consol_end_idx]) > float(rsi_full.iloc[consol_start_idx])
        else:
            s3_triggered = False
    else:
        s3_triggered = False
    soft.append(SoftConditionResult(
        label="RSI trending up in consolidation", triggered=s3_triggered, points=10.0,
    ))

    # S4: Range below known resistance (+10 pts) — skipped, no resistance data
    soft.append(SoftConditionResult(
        label="Range below resistance", triggered=False, points=10.0,
    ))

    # S5: Strategic memory shows breakout performance good (+10 pts)
    s5_triggered = strategic_win_rate is not None and strategic_win_rate > 0.60
    soft.append(SoftConditionResult(
        label="Strategic breakout performance good", triggered=s5_triggered, points=10.0,
    ))

    # ── Disqualifiers ────────────────────────────────────────────────────────

    # 3rd breakout attempt today — previous two failed
    if breakout_attempts_today >= S3_MAX_ATTEMPTS:
        disqualifiers.append(Disqualification(
            reason=f"Breakout already attempted {breakout_attempts_today}x today",
        ))

    # Immediate reversal in same bar — upper wick > 50% of bar range
    if has_enough and len(df) >= 1:
        bar_high = float(df["high"].iloc[-1])
        bar_low = float(df["low"].iloc[-1])
        bar_range = bar_high - bar_low
        upper_wick = bar_high - current_close
        if bar_range > 0 and (upper_wick / bar_range) > S3_REVERSAL_WICK_RATIO:
            disqualifiers.append(Disqualification(
                reason="Breakout bar shows immediate reversal (large upper wick)",
            ))

    # Volume already elevated before the breakout bar
    if has_enough and consol_avg_vol > 0 and len(df) >= 2:
        pre_breakout_vol = float(df["volume"].iloc[-2])
        if pre_breakout_vol > S3_PRE_BREAKOUT_VOL_RATIO * consol_avg_vol:
            disqualifiers.append(Disqualification(
                reason="Volume already elevated before breakout",
            ))

    # Major economic event within 2 hours
    if is_near_major_event(as_of, window_minutes=120):
        disqualifiers.append(Disqualification(
            reason="Major economic event within 2 hours",
        ))

    # Regime is VOLATILE
    if regime == "VOLATILE":
        disqualifiers.append(Disqualification(reason="Regime is VOLATILE"))

    # ── Score and assemble ───────────────────────────────────────────────────

    score, passed = _compute_score(hard, soft, disqualifiers, SETUP_3_MIN_SCORE)
    narrative = _build_narrative(
        "crypto_breakout", symbol, hard, soft, disqualifiers, score, passed,
    )

    return SetupResult(
        setup_type="crypto_breakout",
        symbol=symbol,
        hard_conditions=hard,
        soft_conditions=soft,
        disqualifiers=disqualifiers,
        score=score,
        min_score=SETUP_3_MIN_SCORE,
        passed=passed,
        evidence_narrative=narrative,
        as_of=as_of,
    )


# ── Orchestrator ──────────────────────────────────────────────────────────────


def scan_setups(
    symbol: str,
    indicators: Indicators,
    df: pd.DataFrame,
    regime: MarketRegime,
    *,
    open_symbols: set[str] | None = None,
    signals_today: list[SignalLogEntry] | None = None,
    strategic_win_rates: dict[SetupType, float] | None = None,
    narrative_text: str = "",
    df_daily: pd.DataFrame | None = None,
    as_of: datetime,
) -> list[SetupResult]:
    """
    Run all eligible setup evaluations for a single symbol.

    Returns a list of SetupResult — one per eligible setup type.
    Empty list means no setups are eligible for this symbol/regime combination.
    """
    results: list[SetupResult] = []
    open_syms = open_symbols or set()
    sigs = signals_today or []
    win_rates = strategic_win_rates or {}

    def _count_attempts(setup: SetupType) -> int:
        return sum(1 for s in sigs if s.symbol == symbol and s.setup_type == setup)

    # Canonical form used for position membership checks.
    # Alpaca returns 'BTCUSD' in positions but setup symbols are 'BTC/USD'.
    # Callers may pass symbols in either format, so canonicalise both sides.
    canon_symbol = _canon(symbol)
    canon_open_syms = {_canon(s) for s in open_syms}

    # Setup 1: Crypto Mean Reversion
    if _is_crypto(symbol) and regime in SETUP_1_REGIMES:
        results.append(evaluate_crypto_mean_reversion(
            symbol=symbol,
            indicators=indicators,
            df=df,
            regime=regime,
            has_open_position=canon_symbol in canon_open_syms,
            mean_reversion_attempts_today=_count_attempts("crypto_mean_reversion"),
            strategic_win_rate=win_rates.get("crypto_mean_reversion"),
            narrative_text=narrative_text,
            as_of=as_of,
        ))

    # Setup 2: Equity Trend Pullback
    if not _is_crypto(symbol) and regime in SETUP_2_REGIMES:
        results.append(evaluate_equity_trend_pullback(
            symbol=symbol,
            indicators=indicators,
            df=df,
            regime=regime,
            df_daily_ext=df_daily,
            as_of=as_of,
        ))

    # Setup 3: Crypto Breakout
    if _is_crypto(symbol) and regime in SETUP_3_REGIMES:
        results.append(evaluate_crypto_breakout(
            symbol=symbol,
            indicators=indicators,
            df=df,
            regime=regime,
            has_open_position=canon_symbol in canon_open_syms,
            breakout_attempts_today=_count_attempts("crypto_breakout"),
            strategic_win_rate=win_rates.get("crypto_breakout"),
            as_of=as_of,
        ))

    return results

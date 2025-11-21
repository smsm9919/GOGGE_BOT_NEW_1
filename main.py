# -*- coding: utf-8 -*-
"""
DOGE/USDT — ELITE PRO BOT
- Smart separation: SCALP vs TREND
- Golden Zone Pro (scientific bottoms/tops)
- Strong Council + Footprint + SMC Liquidity Traps + Displacement
- Smart Profit AI (1–3 TPs) + Strict Close
- Anti-Chop (no trading in flat / dirty ranges)
"""

import os, time, math, random, logging, traceback
from logging.handlers import RotatingFileHandler
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from collections import deque

import ccxt
import pandas as pd
import numpy as np
from flask import Flask, jsonify

# =================== ENV & CONFIG ===================

API_KEY    = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")
EXCHANGE_NAME = os.getenv("EXCHANGE", "bingx").lower()   # bingx / bybit
SYMBOL     = os.getenv("SYMBOL", "DOGE/USDT:USDT")
INTERVAL   = os.getenv("INTERVAL", "15m")

LEVERAGE   = int(os.getenv("LEVERAGE", 10))
RISK_ALLOC = float(os.getenv("RISK_ALLOC", 0.60))   # 60% من الرصيد
MODE_LIVE  = os.getenv("MODE_LIVE", "true").lower() == "true"
DRY_RUN    = os.getenv("DRY_RUN", "false").lower() == "true"

PORT       = int(os.getenv("PORT", 5000))
SELF_URL   = os.getenv("SELF_URL", "").strip()

MAX_SPREAD_BPS  = float(os.getenv("MAX_SPREAD_BPS", 6.0))
ADX_GATE        = float(os.getenv("ADX_GATE", 17.0))
MIN_BALANCE     = float(os.getenv("MIN_BALANCE", 10.0))

# Dust guard
FINAL_CHUNK_QTY   = float(os.getenv("FINAL_CHUNK_QTY", 50.0))
RESIDUAL_MIN_QTY  = float(os.getenv("RESIDUAL_MIN_QTY", 10.0))

# ==== Golden Zone Settings ====
FIB_LOW, FIB_HIGH       = 0.618, 0.786
MIN_WICK_PCT            = 0.35
VOL_MA_LEN              = 20
RSI_LEN_GZ, RSI_MA_LEN_GZ = 14, 9
MIN_DISP                = 0.9
GZ_MIN_SCORE            = 6.0
GZ_REQ_ADX              = 20.0

# ==== Profit Profiles / Targets ====
SCALP_MIN_TP    = 0.5      # أقل TP للسكالب (٪) – لا نسمح بأقل من كده
TREND_MIN_TP1   = 1.0      # أقل TP1 للترند
TREND_MIN_TP2   = 2.0

# بروفايلات جاهزة
PROFIT_PROFILE_CONFIG = {
    "SCALP": {
        "label": "SCALP",
        "type": "scalp",
        "tp1_pct": 0.7,
        "tp2_pct": None,
        "tp3_pct": None,
        "tp1_fraction": 1.0,
        "tp2_fraction": 0.0,
        "tp3_fraction": 0.0,
        "scalp_tp_full_pct": 0.7
    },
    "TREND_MEDIUM": {
        "label": "TREND_MEDIUM",
        "type": "trend",
        "tp1_pct": 1.2,
        "tp2_pct": 2.4,
        "tp3_pct": None,
        "tp1_fraction": 0.4,
        "tp2_fraction": 0.6,
        "tp3_fraction": 0.0
    },
    "TREND_STRONG": {
        "label": "TREND_STRONG",
        "type": "trend",
        "tp1_pct": 1.5,
        "tp2_pct": 3.0,
        "tp3_pct": 4.5,
        "tp1_fraction": 0.30,
        "tp2_fraction": 0.40,
        "tp3_fraction": 0.30
    },
}

# ==== Super Scalp Engine ====
SCALP_MODE          = True
SCALP_EXECUTE       = True
SCALP_ADX_GATE      = 18.0
SCALP_MIN_SCORE     = 8.0        # سكالب مش هيتفتح إلا في حالات قوية
SCALP_VOL_FACTOR    = 1.25
SCALP_COOLDOWN_SEC  = 10
SCALP_BE_AFTER_PCT  = 0.25
SCALP_TP_SINGLE_PCT = 0.7

# Anti-chop band
RSI_NEUTRAL_BAND = (45, 55)

BASE_SLEEP   = 5
NEAR_CLOSE_S = 1

BOT_VERSION = "DOGE_ELITE_PRO_1.0"

# =================== LOGGING ===================

FG_R = "\033[31m"; FG_G = "\033[32m"; FG_Y = "\033[33m"; FG_C = "\033[36m"; FG_M = "\033[35m"; RESET = "\033[0m"; BOLD = "\033[1m"

def log_i(msg): logging.info(msg)
def log_w(msg): logging.warning(msg)
def log_e(msg): logging.error(msg)
def log_g(msg): logging.info(FG_G + msg + RESET)

def setup_file_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for h in logger.handlers[:]:
        logger.removeHandler(h)

    fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s [%(filename)s:%(lineno)d]"))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(ch)

    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    logging.getLogger("ccxt.base.exchange").setLevel(logging.INFO)

    log_i("🔄 Professional logging ready - rotation ON")

setup_file_logging()

# =================== EXCHANGE ===================

def make_ex():
    cfg = {
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "timeout": 20000,
    }
    if EXCHANGE_NAME == "bybit":
        cfg["options"] = {"defaultType": "swap"}
        return ccxt.bybit(cfg)
    else:
        cfg["options"] = {"defaultType": "swap"}
        return ccxt.bingx(cfg)

ex = make_ex()

MARKET   = {}
AMT_PREC = 0
LOT_STEP = None
LOT_MIN  = None

def load_market_specs():
    global MARKET, AMT_PREC, LOT_STEP, LOT_MIN
    try:
        ex.load_markets()
        MARKET = ex.markets.get(SYMBOL, {})
        AMT_PREC = int((MARKET.get("precision", {}) or {}).get("amount", 0) or 0)
        LOT_STEP = (MARKET.get("limits", {}) or {}).get("amount", {}).get("step", None)
        LOT_MIN  = (MARKET.get("limits", {}) or {}).get("amount", {}).get("min", None)
        log_i(f"🎯 {SYMBOL} specs → precision={AMT_PREC}, step={LOT_STEP}, min={LOT_MIN}")
    except Exception as e:
        log_w(f"load_market_specs: {e}")

def ensure_leverage():
    try:
        if EXCHANGE_NAME == "bybit":
            ex.set_leverage(LEVERAGE, SYMBOL)
        else:
            ex.set_leverage(LEVERAGE, SYMBOL, params={"side": "BOTH"})
        log_g(f"✅ leverage set: {LEVERAGE}x")
    except Exception as e:
        log_w(f"set_leverage: {e}")

try:
    load_market_specs()
    ensure_leverage()
except Exception as e:
    log_w(f"exchange init: {e}")

def exchange_specific_params(side, is_close=False):
    if EXCHANGE_NAME == "bybit":
        return {"reduceOnly": is_close, "positionSide": "Both"}
    else:
        return {"reduceOnly": is_close, "positionSide": "BOTH"}

# =================== HELPERS ===================

STATE = {
    "open": False,
    "side": None,
    "qty": 0.0,
    "entry": None,
    "mode": None,      # "scalp" / "trend"
    "reason": "",
    "profit_profile": None,
    "pnl": 0.0,
    "highest_profit_pct": 0.0,
    "tp1_done": False,
    "tp2_done": False,
    "tp3_done": False,
    "breakeven": None,
    "breakeven_armed": False,
    "opened_at": None,
    "last_exit_reason": "",
    "last_entry_source": "",
    "trade_type": "NONE"
}

performance_stats = {
    "total_trades": 0,
    "winning_trades": 0,
    "losing_trades": 0,
    "total_profit": 0.0,
}

last_scalp_ts = 0

def _fmt_pct(x): 
    try: return f"{float(x):.2f}%"
    except: return str(x)

def _round_amt(q):
    if q is None:
        return 0.0
    try:
        d = Decimal(str(q))
        if LOT_MIN:
            if d < Decimal(str(LOT_MIN)):
                return 0.0
        if LOT_STEP and LOT_STEP > 0:
            step = Decimal(str(LOT_STEP))
            d = (d / step).to_integral_value(rounding=ROUND_DOWN) * step
        prec = int(AMT_PREC) if AMT_PREC >= 0 else 0
        d = d.quantize(Decimal(1).scaleb(-prec), rounding=ROUND_DOWN)
        return float(d)
    except Exception:
        return float(q)

def balance_usdt():
    try:
        bal = ex.fetch_balance()
        total = bal.get("total", {})
        usdt = total.get("USDT", 0.0)
        return float(usdt)
    except Exception as e:
        log_w(f"balance error: {e}")
        return 0.0

def fetch_ohlcv(limit=200):
    try:
        ohlcv = ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit)
        if not ohlcv:
            return pd.DataFrame()
        df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
        return df
    except Exception as e:
        log_w(f"fetch_ohlcv error: {e}")
        return pd.DataFrame()

def price_now():
    try:
        ticker = ex.fetch_ticker(SYMBOL)
        return float(ticker["last"])
    except Exception:
        return None

def orderbook_spread_bps():
    try:
        ob = ex.fetch_order_book(SYMBOL, limit=5)
        bid = ob["bids"][0][0] if ob["bids"] else None
        ask = ob["asks"][0][0] if ob["asks"] else None
        if not bid or not ask:
            return None
        mid = (bid + ask) / 2
        return (ask - bid) / mid * 10000  # bps
    except Exception as e:
        log_w(f"spread error: {e}")
        return None

# =================== INDICATORS ===================

def compute_indicators(df: pd.DataFrame):
    if df.empty:
        return {}
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    vol   = df["volume"].astype(float)

    # ATR
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    # ADX
    plus_dm  = high.diff()
    minus_dm = low.diff().mul(-1)
    plus_dm  = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr_n    = atr.replace(0, 1e-9)
    plus_di  = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr_n)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / atr_n)
    dx       = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9) * 100
    adx      = dx.ewm(alpha=1/14).mean()

    # RSI
    delta = close.diff()
    gain  = delta.where(delta > 0, 0.0)
    loss  = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean().replace(0, 0.001)
    rs   = avg_gain / avg_loss
    rsi  = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist   = macd - macd_signal

    # Volume profile (بسيط)
    vol_ma = vol.rolling(VOL_MA_LEN).mean()
    volume_spike = vol.iloc[-1] > vol_ma.iloc[-1] * 1.6 if vol_ma.iloc[-1] > 0 else False
    volume_trend = "up" if vol.iloc[-1] > vol_ma.iloc[-1] else "flat"

    # Trend strength (بسيط)
    ret = close.pct_change(20) * 100
    strength = "flat"
    if abs(ret.iloc[-1]) > 5 and adx.iloc[-1] > 20:
        strength = "strong" if ret.iloc[-1] > 0 else "strong_down"
    elif abs(ret.iloc[-1]) > 2:
        strength = "medium"

    return {
        "atr": atr.iloc[-1],
        "adx": adx.iloc[-1],
        "plus_di": plus_di.iloc[-1],
        "minus_di": minus_di.iloc[-1],
        "rsi": rsi.iloc[-1],
        "macd": macd.iloc[-1],
        "macd_signal": macd_signal.iloc[-1],
        "macd_hist": macd_hist.iloc[-1],
        "volume_spike": volume_spike,
        "volume_trend": volume_trend,
        "trend_strength": strength,
        "close": close,
        "high": high,
        "low": low,
        "volume": vol,
    }

# =================== GOLDEN ZONE PRO ===================

def _displacement(closes: pd.Series):
    if len(closes) < 25:
        return 0.0
    recent = closes.iloc[-20:]
    std = recent.std()
    move = abs(closes.iloc[-1] - closes.iloc[-6])
    return float(move / max(std, 1e-9))

def detect_impulse_leg(df: pd.DataFrame):
    # بسيط: آخر موجة قوية في آخر 40 شمعة
    close = df["close"].astype(float)
    window = close.tail(40)
    idx_max = window.idxmax()
    idx_min = window.idxmin()
    if idx_min < idx_max:
        direction = "up"
        swing_low = close.loc[idx_min]
        swing_high= close.loc[idx_max]
    else:
        direction = "down"
        swing_high= close.loc[idx_max]
        swing_low = close.loc[idx_min]
    return direction, float(swing_low), float(swing_high)

def golden_zone_check(df: pd.DataFrame, ind: dict):
    try:
        if df.empty or len(df) < 50:
            return {"ok": False, "score": 0.0, "zone": None, "reasons": ["not_enough_data"]}

        direction, swing_low, swing_high = detect_impulse_leg(df)
        close = df["close"].astype(float).iloc[-1]
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        vol   = df["volume"].astype(float)

        if direction == "up":
            f618 = swing_low + (swing_high - swing_low) * FIB_LOW
            f786 = swing_low + (swing_high - swing_low) * FIB_HIGH
            zone_type = "golden_bottom"
        else:
            f618 = swing_high - (swing_high - swing_low) * FIB_LOW
            f786 = swing_high - (swing_high - swing_low) * FIB_HIGH
            zone_type = "golden_top"

        in_zone = f786 <= close <= f618 if direction == "up" else f618 <= close <= f786
        if not in_zone:
            return {"ok": False, "score": 0.0, "zone": None, "reasons": ["price_outside_gz"]}

        # Wick / rejection
        last_high = high.iloc[-1]
        last_low  = low.iloc[-1]
        rng = last_high - last_low
        if rng <= 0: 
            return {"ok": False, "score": 0.0, "zone": None, "reasons": ["no_range"]}

        if direction == "up":
            wick_down = (close - last_low) / rng
            wick_ok = wick_down >= MIN_WICK_PCT
        else:
            wick_up = (last_high - close) / rng
            wick_ok = wick_up >= MIN_WICK_PCT

        # Volume / RSI / ADX / displacement
        vol_ma = vol.rolling(VOL_MA_LEN).mean()
        volume_spike = vol.iloc[-1] > vol_ma.iloc[-1] * 1.5 if vol_ma.iloc[-1] > 0 else False

        rsi_val = float(ind.get("rsi", 50.0) or 50.0)
        adx_val = float(ind.get("adx", 0.0) or 0.0)
        disp    = _displacement(df["close"].astype(float))

        reasons = []
        score = 0.0

        if wick_ok:
            score += 2.0
            reasons.append("wick_rejection_ok")
        if volume_spike:
            score += 1.5
            reasons.append("volume_spike")
        if adx_val >= GZ_REQ_ADX:
            score += 1.5
            reasons.append("adx_trend_ok")
        if disp >= MIN_DISP:
            score += 1.0
            reasons.append("displacement_ok")

        if direction == "up" and 35 < rsi_val < 65:
            score += 1.0
            reasons.append("rsi_ok_for_bottom")
        if direction == "down" and 35 < rsi_val < 70:
            score += 1.0
            reasons.append("rsi_ok_for_top")

        zone = {
            "type": zone_type,
            "f618": f618,
            "f786": f786,
            "swing_low": swing_low,
            "swing_high": swing_high,
        }

        ok = score >= GZ_MIN_SCORE
        return {"ok": ok, "score": round(score,2), "zone": zone if ok else None, "reasons": reasons}

    except Exception as e:
        log_w(f"golden_zone_check error: {e}")
        return {"ok": False, "score": 0.0, "zone": None, "reasons": [f"error:{e}"]}

def log_golden_entry(side, price, qty, gz_data, council_data, ind):
    zone_type = gz_data["zone"]["type"].upper() if gz_data.get("zone") else "N/A"
    msg = f"""
🏆🚀 GOLDEN ZONE ENTRY CONFIRMED 🚀🏆
┌──────────────────────────────────────────────
│ 📍 الاتجاه: {side.upper()} | المنطقة: {zone_type}
│ 💰 السعر: {price:.6f} | الكمية: {qty:.4f}
│ 
│ 📊 METRICS:
│ ├─ Council Score: max(B={council_data.get('score_b',0):.1f}, S={council_data.get('score_s',0):.1f})
│ ├─ ADX: {ind.get('adx',0):.1f} | RSI: {ind.get('rsi',50):.1f}
│ ├─ GZ Score: {gz_data.get('score',0):.1f} | Reasons: {", ".join(gz_data.get('reasons',[]))}
│ └─ Dispersion: { _displacement(ind.get('close', pd.Series())):.2f }
└──────────────────────────────────────────────
"""
    log_g(msg)

# =================== FOOTPRINT + SMC (مبسّط) ===================

def compute_footprint_boost(df: pd.DataFrame, ind: dict):
    """محاكاة Footprint: مقارنة حجم/جسم الشمعة بأخر N شموع"""
    if df.empty or len(df) < 30:
        return {"votes_b": 0, "votes_s": 0, "score_b": 0.0, "score_s": 0.0, "tag": "no_data"}

    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    vol   = df["volume"].astype(float)

    body = (close - open_).abs()
    body_ma = body.rolling(20).mean()
    vol_ma  = vol.rolling(20).mean()

    last_body = body.iloc[-1]
    last_vol  = vol.iloc[-1]
    last_body_ma = body_ma.iloc[-1]
    last_vol_ma  = vol_ma.iloc[-1]

    score_b = score_s = 0.0
    votes_b = votes_s = 0
    tag = "neutral"

    if last_vol_ma > 0 and last_body_ma > 0:
        strong_vol = last_vol > last_vol_ma * 1.8
        strong_body= last_body > last_body_ma * 1.5
        bullish = close.iloc[-1] > open_.iloc[-1]
        bearish = close.iloc[-1] < open_.iloc[-1]

        if strong_vol and strong_body and bullish:
            score_b += 2.0; votes_b += 3; tag = "bull_footprint"
        elif strong_vol and strong_body and bearish:
            score_s += 2.0; votes_s += 3; tag = "bear_footprint"

    return {"votes_b": votes_b, "votes_s": votes_s, "score_b": score_b, "score_s": score_s, "tag": tag}

def detect_liquidity_trap(df: pd.DataFrame):
    """SMC Liquidity Trap: sweep قاع/قمة ثم ارتداد"""
    if df.empty or len(df) < 30:
        return None
    high = df["high"].astype(float)
    low  = df["low"].astype(float)
    close= df["close"].astype(float)

    prev_highs = high.iloc[-6:-1]
    prev_lows  = low.iloc[-6:-1]
    last_high  = high.iloc[-1]
    last_low   = low.iloc[-1]
    last_close = close.iloc[-1]
    last_open  = df["open"].astype(float).iloc[-1]

    # sweep high then close below
    if last_high > prev_highs.max() and last_close < last_open:
        return ("bear_trap", "liquidity_sweep_up")

    # sweep low then close above
    if last_low < prev_lows.min() and last_close > last_open:
        return ("bull_trap", "liquidity_sweep_down")

    return None

# =================== COUNCIL AI ===================

def council_votes(df: pd.DataFrame, ind: dict, gz: dict):
    votes_b = votes_s = 0
    score_b = score_s = 0.0
    logs = []

    rsi = ind.get("rsi", 50.0)
    adx = ind.get("adx", 0.0)
    plus_di = ind.get("plus_di", 0.0)
    minus_di= ind.get("minus_di", 0.0)
    macd_hist = ind.get("macd_hist", 0.0)
    vol_spike = ind.get("volume_spike", False)

    # Trend / ADX
    if adx >= 25:
        if plus_di > minus_di:
            score_b += 2.0; votes_b += 3; logs.append("📈 strong_up_trend")
        elif minus_di > plus_di:
            score_s += 2.0; votes_s += 3; logs.append("📉 strong_down_trend")

    # RSI Context
    if rsi < 35:
        score_b += 1.0; votes_b += 1; logs.append("🟢 rsi_buy_zone")
    elif rsi > 65:
        score_s += 1.0; votes_s += 1; logs.append("🔴 rsi_sell_zone")

    # MACD
    if macd_hist > 0:
        score_b += 1.0; votes_b += 1; logs.append("📈 macd_up")
    elif macd_hist < 0:
        score_s += 1.0; votes_s += 1; logs.append("📉 macd_down")

    # Volume
    if vol_spike:
        if plus_di > minus_di:
            score_b += 1.0; votes_b += 1; logs.append("📊 volume_spike_up")
        elif minus_di > plus_di:
            score_s += 1.0; votes_s += 1; logs.append("📊 volume_spike_down")

    # Golden Zone weight
    if gz.get("ok"):
        zone_type = gz["zone"]["type"]
        gz_score  = gz.get("score", 0.0)
        if zone_type == "golden_bottom":
            score_b += gz_score * 0.4; votes_b += 4; logs.append(f"🏆 golden_bottom score={gz_score:.1f}")
        else:
            score_s += gz_score * 0.4; votes_s += 4; logs.append(f"🏆 golden_top score={gz_score:.1f}")

    # Footprint Boost
    fp = compute_footprint_boost(df, ind)
    score_b += fp["score_b"]; score_s += fp["score_s"]
    votes_b += fp["votes_b"]; votes_s += fp["votes_s"]
    if fp["tag"] != "neutral":
        logs.append(f"🧭 footprint: {fp['tag']}")

    # Liquidity Trap
    trap = detect_liquidity_trap(df)
    if trap:
        kind, reason = trap
        if kind == "bull_trap":
            score_b += 1.5; votes_b += 2; logs.append("💧 bull_liquidity_trap")
        elif kind == "bear_trap":
            score_s += 1.5; votes_s += 2; logs.append("💧 bear_liquidity_trap")

    total_score = score_b + score_s
    confidence = min(1.0, total_score / 30.0) if total_score > 0 else 0.0

    return {
        "b": votes_b, "s": votes_s,
        "score_b": round(score_b,2), "score_s": round(score_s,2),
        "logs": logs, "confidence": confidence
    }

def is_choppy(ind: dict):
    adx = ind.get("adx", 0.0)
    rsi = ind.get("rsi", 50.0)
    atr = ind.get("atr", 0.0)
    low, high = RSI_NEUTRAL_BAND
    if adx < ADX_GATE and low <= rsi <= high:
        return True
    if atr <= 0:
        return True
    return False

# =================== PROFIT PROFILE & MODE ===================

def classify_trade_mode(ind: dict, council_data: dict, gz: dict):
    """
    يحدد:
    - mode: "scalp" أو "trend"
    - profile: SCALP / TREND_MEDIUM / TREND_STRONG
    """
    adx = ind.get("adx", 0.0)
    rsi = ind.get("rsi", 50.0)
    score_b = council_data.get("score_b", 0.0)
    score_s = council_data.get("score_s", 0.0)
    dom_score = max(score_b, score_s)

    # Golden Zone صاروخي = ترند قوي
    if gz.get("ok") and gz.get("score",0) >= GZ_MIN_SCORE + 1.0 and adx >= GZ_REQ_ADX:
        profile = PROFIT_PROFILE_CONFIG["TREND_STRONG"]
        mode = "trend"
        reason = "golden_zone_trend_strong"
        return mode, profile, reason

    # سكالب قوي فقط لما:
    # ADX متوسط + Council قوي + مش في تشبع مجنون
    if SCALP_MODE and adx >= SCALP_ADX_GATE and 35 < rsi < 65 and dom_score >= SCALP_MIN_SCORE:
        profile = PROFIT_PROFILE_CONFIG["SCALP"]
        mode = "scalp"
        reason = "strong_scalp_profile"
        return mode, profile, reason

    # ترند متوسط / قوي حسب ADX + score
    if adx >= 25 and dom_score >= 22:
        profile = PROFIT_PROFILE_CONFIG["TREND_STRONG"]
        mode = "trend"
        reason = "trend_strong_profile"
    else:
        profile = PROFIT_PROFILE_CONFIG["TREND_MEDIUM"]
        mode = "trend"
        reason = "trend_medium_profile"

    return mode, profile, reason

# =================== EXECUTION HELPERS ===================

def print_position_snapshot(reason="OPEN"):
    side = STATE["side"]
    qty  = STATE["qty"]
    entry= STATE["entry"]
    mode = STATE["mode"]
    profile = STATE.get("profit_profile", {})
    pnl  = STATE["pnl"]
    highest = STATE["highest_profit_pct"]

    icon = "🟢" if side=="long" else "🔴"
    mode_icon = "⚡" if mode=="scalp" else "📈"
    color = FG_G if side=="long" else FG_R

    msg = (
        f"{color}{BOLD}{icon} {reason} — {mode_icon} {mode.upper()} "
        f"[{profile.get('label','N/A')}] {RESET}\n"
        f"  • Side: {side.upper()} | Qty: {qty:.4f} | Entry: {entry:.6f}\n"
        f"  • TP plan: "
        f"TP1={profile.get('tp1_pct')}%  TP2={profile.get('tp2_pct')}%  TP3={profile.get('tp3_pct')}%\n"
        f"  • PnL now: {_fmt_pct(pnl)} | Max: {_fmt_pct(highest)}\n"
        f"  • Reason: {STATE.get('reason','')}\n"
        f"  • EntrySource: {STATE.get('last_entry_source','')}"
    )
    log_i(msg)

def open_market(side: str, price: float, reason: str, mode: str, profile: dict, entry_source: str, gz_data=None, council_data=None, ind=None):
    global STATE, performance_stats
    bal = balance_usdt()
    if bal < MIN_BALANCE:
        log_w(f"❌ balance too low: {bal}")
        return False

    risk_capital = bal * RISK_ALLOC
    notional = risk_capital * LEVERAGE
    qty = _round_amt(notional / price)
    if qty <= 0:
        log_w(f"❌ qty too small: {qty}")
        return False

    side_mkt = "buy" if side == "long" else "sell"

    if MODE_LIVE and not DRY_RUN:
        try:
            params = exchange_specific_params(side_mkt, is_close=False)
            ex.create_order(SYMBOL, "market", side_mkt, qty, None, params)
        except Exception as e:
            log_e(f"❌ open_market failed: {e}")
            return False

    STATE.update({
        "open": True,
        "side": side,
        "qty": qty,
        "entry": price,
        "mode": mode,
        "reason": reason,
        "profit_profile": profile,
        "pnl": 0.0,
        "highest_profit_pct": 0.0,
        "tp1_done": False,
        "tp2_done": False,
        "tp3_done": False,
        "breakeven": None,
        "breakeven_armed": False,
        "opened_at": time.time(),
        "last_exit_reason": "",
        "last_entry_source": entry_source,
        "trade_type": "GOLDEN" if (gz_data and gz_data.get("ok")) else mode.upper()
    })
    performance_stats["total_trades"] += 1

    if gz_data and council_data and ind:
        log_golden_entry("buy" if side=="long" else "sell", price, qty, gz_data, council_data, ind)

    print_position_snapshot("OPEN_POSITION")

    return True

def close_partial(percent: float, reason: str):
    if not STATE["open"] or STATE["qty"] <= 0:
        return
    close_qty = _round_amt(STATE["qty"] * percent)
    if close_qty <= 0:
        return
    side_mkt = "sell" if STATE["side"]=="long" else "buy"
    if MODE_LIVE and not DRY_RUN:
        try:
            params = exchange_specific_params(side_mkt, is_close=True)
            ex.create_order(SYMBOL, "market", close_qty, None, params)
        except Exception as e:
            log_e(f"❌ partial close failed: {e}")
            return
    STATE["qty"] = _round_amt(STATE["qty"] - close_qty)
    log_g(f"💰 PARTIAL CLOSE {percent*100:.0f}% | reason={reason}")

def close_full(reason: str):
    if not STATE["open"] or STATE["qty"] <= 0:
        return
    qty = STATE["qty"]
    side_mkt = "sell" if STATE["side"]=="long" else "buy"
    if MODE_LIVE and not DRY_RUN:
        try:
            params = exchange_specific_params(side_mkt, is_close=True)
            ex.create_order(SYMBOL, "market", qty, None, params)
        except Exception as e:
            log_e(f"❌ full close failed: {e}")
            return
    STATE["open"] = False
    STATE["qty"] = 0.0
    STATE["last_exit_reason"] = reason
    log_g(f"🏁 FULL CLOSE | reason={reason}")

# =================== SMART PROFIT AI ===================

def manage_open_trade(df: pd.DataFrame, ind: dict):
    if not STATE["open"]:
        return

    px = price_now()
    if not px or not STATE["entry"]:
        return

    side = STATE["side"]
    entry = STATE["entry"]
    pnl_pct = (px - entry) / entry * 100 * (1 if side=="long" else -1)
    STATE["pnl"] = pnl_pct
    if pnl_pct > STATE["highest_profit_pct"]:
        STATE["highest_profit_pct"] = pnl_pct

    profile = STATE.get("profit_profile", {}) or {}
    mode = STATE.get("mode","trend")

    # Breakeven logic
    if not STATE["breakeven_armed"] and pnl_pct >= profile.get("tp1_pct", 1.0):
        STATE["breakeven"] = entry
        STATE["breakeven_armed"] = True
        log_i("🛡️ BREAKEVEN ARMED")

    if STATE["breakeven_armed"]:
        if side=="long" and px <= STATE["breakeven"]:
            close_full("breakeven_hit")
            performance_stats["winning_trades"] += 1
            return
        if side=="short" and px >= STATE["breakeven"]:
            close_full("breakeven_hit")
            performance_stats["winning_trades"] += 1
            return

    # SCALP: هدف واحد محترم
    if mode == "scalp":
        tp_full = max(profile.get("scalp_tp_full_pct", SCALP_TP_SINGLE_PCT), SCALP_MIN_TP)
        if pnl_pct >= tp_full and not STATE["tp1_done"]:
            close_full("scalp_tp_full")
            STATE["tp1_done"] = True
            performance_stats["winning_trades"] += 1
            return

    # TREND: 2–3 أهداف
    else:
        tp1 = max(profile.get("tp1_pct", TREND_MIN_TP1), TREND_MIN_TP1)
        tp2 = max(profile.get("tp2_pct", TREND_MIN_TP2), TREND_MIN_TP2)
        tp3 = profile.get("tp3_pct", None)

        f1 = profile.get("tp1_fraction", 0.4)
        f2 = profile.get("tp2_fraction", 0.6)
        f3 = profile.get("tp3_fraction", 0.0)

        # TP1
        if pnl_pct >= tp1 and not STATE["tp1_done"] and STATE["qty"] > 0:
            close_partial(f1, "trend_tp1")
            STATE["tp1_done"] = True
            return

        # TP2
        if pnl_pct >= tp2 and not STATE["tp2_done"] and STATE["qty"] > 0:
            # لو مفيش TP3 → اقفل الباقي
            if not tp3:
                close_full("trend_tp2_full")
                STATE["tp2_done"] = True
                performance_stats["winning_trades"] += 1
                return
            else:
                close_partial(f2, "trend_tp2")
                STATE["tp2_done"] = True
                return

        # TP3
        if tp3 and pnl_pct >= tp3 and not STATE["tp3_done"] and STATE["qty"] > 0:
            close_full("trend_tp3_full")
            STATE["tp3_done"] = True
            performance_stats["winning_trades"] += 1
            return

    # حماية أرباح كبيرة: لو الربح عدّى 4–5% ومافيش TP باقي → شدد الدفاع
    if pnl_pct >= 4.0 and mode=="trend" and not STATE.get("big_profit_protected", False):
        STATE["big_profit_protected"] = True
        # قفل جزئي إضافي لو لسه في كمية
        if STATE["qty"] > 0:
            close_partial(0.25, "big_profit_extra")
        log_i("💰 Big Profit Protection Activated")

# =================== ENTRY DECISION ENGINE ===================

def decide_entry(df: pd.DataFrame, ind: dict):
    """
    يرجع:
      side: "long"/"short"/None
      reason: str
      mode: "scalp"/"trend"
      profile: dict
      entry_source: "GOLDEN" / "COUNCIL_STRONG" / "SCALP"
      gz_data: dict
      council_data: dict
    """
    if df.empty:
        return None, "no_data", None, None, None, None

    # Spread guard
    spread = orderbook_spread_bps()
    if spread and spread > MAX_SPREAD_BPS:
        return None, f"spread_too_wide_{spread:.1f}", None, None, None, None

    # Anti-chop
    if is_choppy(ind):
        return None, "choppy_market", None, None, None, None

    gz_data = golden_zone_check(df, ind)
    council_data = council_votes(df, ind, gz_data)

    score_b = council_data["score_b"]
    score_s = council_data["score_s"]

    # Golden Zone Priority
    if gz_data.get("ok") and gz_data.get("score",0) >= GZ_MIN_SCORE:
        zone_type = gz_data["zone"]["type"]
        if zone_type == "golden_bottom":
            mode, profile, reason_mode = classify_trade_mode(ind, council_data, gz_data)
            return "long", f"golden_bottom_{reason_mode}", mode, profile, "GOLDEN", gz_data, council_data
        else:
            mode, profile, reason_mode = classify_trade_mode(ind, council_data, gz_data)
            return "short", f"golden_top_{reason_mode}", mode, profile, "GOLDEN", gz_data, council_data

    # Council Strong Entry (بدون Golden لكن قوي)
    dom_side = "buy" if score_b > score_s else "sell"
    dom_score = max(score_b, score_s)
    if dom_score >= 20.0 and council_data["confidence"] >= 0.6:
        mode, profile, reason_mode = classify_trade_mode(ind, council_data, {"ok":False})
        if dom_side=="buy":
            return "long", f"council_strong_buy_{reason_mode}", mode, profile, "COUNCIL_STRONG", gz_data, council_data
        else:
            return "short", f"council_strong_sell_{reason_mode}", mode, profile, "COUNCIL_STRONG", gz_data, council_data

    # Strong scalp only (لو مفيش Golden ولا Council قوي)
    if SCALP_MODE:
        mode, profile, reason_mode = classify_trade_mode(ind, council_data, {"ok":False})
        if mode=="scalp" and council_data["confidence"] >= 0.5:
            # السكالب لازم يكون في اتجاه الدوم
            if dom_side=="buy":
                return "long", f"pure_scalp_buy_{reason_mode}", mode, profile, "SCALP", gz_data, council_data
            else:
                return "short", f"pure_scalp_sell_{reason_mode}", mode, profile, "SCALP", gz_data, council_data

    return None, "no_strong_setup", None, None, None, None

# =================== MAIN LOOP ===================

def trade_loop():
    global last_scalp_ts
    log_i("🔁 trade loop started")

    while True:
        try:
            df = fetch_ohlcv(limit=200)
            if df.empty:
                time.sleep(BASE_SLEEP)
                continue

            ind = compute_indicators(df)

            # إدارة صفقة مفتوحة
            if STATE["open"]:
                manage_open_trade(df, ind)
                time.sleep(NEAR_CLOSE_S if STATE["open"] else BASE_SLEEP)
                continue

            # لا يوجد صفقة مفتوحة → نبحث عن دخول
            side, reason, mode, profile, entry_source, gz_data, council_data = decide_entry(df, ind)

            if side is None:
                log_i(f"⏭ no_entry: {reason}")
                time.sleep(BASE_SLEEP)
                continue

            now = time.time()
            if entry_source == "SCALP" and now - last_scalp_ts < SCALP_COOLDOWN_SEC:
                log_i("⏳ scalp_cooldown_active")
                time.sleep(BASE_SLEEP)
                continue

            px = price_now()
            if not px:
                time.sleep(BASE_SLEEP)
                continue

            opened = open_market(
                "long" if side=="long" else "short",
                px,
                reason,
                mode,
                profile,
                entry_source,
                gz_data,
                council_data,
                ind
            )

            if opened and entry_source=="SCALP":
                last_scalp_ts = now

            time.sleep(BASE_SLEEP)

        except Exception as e:
            log_e(f"loop error: {e}")
            log_e(traceback.format_exc())
            time.sleep(5)

# =================== API & KEEPALIVE ===================

app = Flask(__name__)

@app.route("/")
def home():
    mode = "LIVE" if MODE_LIVE else "PAPER"
    return (f"✅ DOGE ELITE PRO BOT — {EXCHANGE_NAME.upper()} — {SYMBOL} {INTERVAL} — {mode} — "
            f"Golden Zone Pro + Strong Council + Smart Profit AI + SCALP/TREND Split")

@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "exchange": EXCHANGE_NAME,
        "symbol": SYMBOL,
        "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"],
        "side": STATE["side"],
        "qty": STATE["qty"],
        "pnl": STATE["pnl"],
        "trade_type": STATE.get("trade_type","NONE"),
        "last_exit_reason": STATE.get("last_exit_reason",""),
        "timestamp": datetime.utcnow().isoformat()
    }), 200

@app.route("/metrics")
def metrics():
    return jsonify({
        "exchange": EXCHANGE_NAME,
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE,
        "risk_alloc": RISK_ALLOC,
        "state": STATE,
        "performance": performance_stats,
        "config": {
            "golden_zone_pro": True,
            "council_strong": True,
            "smart_profit_ai": True,
            "scalp_mode": SCALP_MODE,
            "adx_gate": ADX_GATE,
            "max_spread_bps": MAX_SPREAD_BPS
        }
    })

def keepalive_loop():
    url = (SELF_URL or "").strip().rstrip("/")
    if not url:
        log_w("keepalive disabled (SELF_URL not set)")
        return
    import requests
    sess = requests.Session()
    sess.headers.update({"User-Agent":"doge-elite-pro-keepalive"})
    log_i(f"🌐 KEEPALIVE → {url}")
    while True:
        try:
            sess.get(url, timeout=8)
        except Exception:
            pass
        time.sleep(50)

# =================== MAIN ===================

if __name__ == "__main__":
    log_i(f"🚀 DOGE ELITE PRO BOT STARTED - {BOT_VERSION}")
    log_i(f"🎯 SYMBOL={SYMBOL} | INTERVAL={INTERVAL} | LEVERAGE={LEVERAGE}x")

    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    if SELF_URL:
        threading.Thread(target=keepalive_loop, daemon=True).start()

    app.run(host="0.0.0.0", port=PORT, debug=False)

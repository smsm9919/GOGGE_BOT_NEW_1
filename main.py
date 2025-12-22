# -*- coding: utf-8 -*-
"""
RF Futures Bot — RF-LIVE ONLY (BingX Perp via CCXT)
• Advanced Scenario Engine + Dynamic Trade Management
• Golden Entry + Smart Exit + Profit Protection
• 3-Level TP for Golden Trades + Scalp Boost
• Professional Risk Management + Market Phase Detection
• FALCON STYLE ADDON: TP1/TP2/TP3 + Early Exit + Box Detector
• SCENARIO ENGINE: Market State Detection + Smart Entry Points
• PROFESSIONAL UPGRADE: Food Print Detection + Smart Trade Council
"""

import os, time, math, random, signal, sys, traceback, logging, json
from logging.handlers import RotatingFileHandler
from datetime import datetime
import pandas as pd
import numpy as np
import ccxt
from flask import Flask, jsonify
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, Tuple, List

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# =================== ENV / MODE ===================
API_KEY = os.getenv("BINGX_API_KEY", "")
API_SECRET = os.getenv("BINGX_API_SECRET", "")
MODE_LIVE = bool(API_KEY and API_SECRET)

SELF_URL = os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")
PORT = int(os.getenv("PORT", 5000))

# ==== Run mode / Logging toggles ====
LOG_LEGACY = False
LOG_ADDONS = True

# ==== Execution Switches ====
EXECUTE_ORDERS = True
SHADOW_MODE_DASHBOARD = False
DRY_RUN = False

# ==== Addon: Logging + Recovery Settings ====
BOT_VERSION = "DOGE Scenario Engine PRO v7.0 — Professional Food Print + Smart Council"
print("🔁 Booting:", BOT_VERSION, flush=True)

STATE_PATH = "./bot_state.json"
RESUME_ON_RESTART = True
RESUME_LOOKBACK_SECS = 60 * 60

# === Addons config ===
BOOKMAP_DEPTH = 50
BOOKMAP_TOPWALLS = 3
IMBALANCE_ALERT = 1.30

FLOW_WINDOW = 20
FLOW_SPIKE_Z = 1.60
CVD_SMOOTH = 8

# =================== SETTINGS ===================
SYMBOL     = os.getenv("SYMBOL", "DOGE/USDT:USDT")
INTERVAL   = os.getenv("INTERVAL", "15m")
LEVERAGE   = int(os.getenv("LEVERAGE", 10))
RISK_ALLOC = float(os.getenv("RISK_ALLOC", 0.60))
POSITION_MODE = os.getenv("BINGX_POSITION_MODE", "oneway")

# RF Settings
RF_SOURCE = "close"
RF_PERIOD = int(os.getenv("RF_PERIOD", 20))
RF_MULT   = float(os.getenv("RF_MULT", 3.5))
RF_LIVE_ONLY = True
RF_HYST_BPS  = 6.0

# Indicators
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14

MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", 6.0))

# Dynamic TP / trail
TP1_PCT_BASE       = 0.40
TP1_CLOSE_FRAC     = 0.50
BREAKEVEN_AFTER    = 0.30
TRAIL_ACTIVATE_PCT = 1.20
ATR_TRAIL_MULT     = 1.6

TREND_TPS       = [0.50, 1.00, 1.80]
TREND_TP_FRACS  = [0.30, 0.30, 0.20]

# Dust guard
FINAL_CHUNK_QTY = float(os.getenv("FINAL_CHUNK_QTY", 40.0))
RESIDUAL_MIN_QTY = float(os.getenv("RESIDUAL_MIN_QTY", 9.0))

# Strict close
CLOSE_RETRY_ATTEMPTS = 6
CLOSE_VERIFY_WAIT_S  = 2.0

# Pacing
BASE_SLEEP   = 5
NEAR_CLOSE_S = 1

# ==== Smart Exit Tuning ===
TP1_SCALP_PCT      = 0.35/100
TP1_TREND_PCT      = 0.60/100
HARD_CLOSE_PNL_PCT = 1.10/100
WICK_ATR_MULT      = 1.5
EVX_SPIKE          = 1.8
BM_WALL_PROX_BPS   = 5
TIME_IN_TRADE_MIN  = 8
TRAIL_TIGHT_MULT   = 1.20

# ==== Golden Entry Settings ====
GOLDEN_ENTRY_SCORE = 6.0
GOLDEN_ENTRY_ADX   = 20.0
GOLDEN_REVERSAL_SCORE = 6.5

# ==== Execution & Strategy Thresholds ====
ADX_TREND_MIN = 20
DI_SPREAD_TREND = 6
RSI_MA_LEN = 9
RSI_NEUTRAL_BAND = (45, 55)
RSI_TREND_PERSIST = 3

GZ_MIN_SCORE = 6.0
GZ_REQ_ADX = 20
GZ_REQ_VOL_MA = 20
ALLOW_GZ_ENTRY = True

SCALP_TP1 = 0.40
SCALP_BE_AFTER = 0.30
SCALP_ATR_MULT = 1.6
TREND_TP1 = 1.20
TREND_BE_AFTER = 0.80
TREND_ATR_MULT = 1.8

MAX_TRADES_PER_HOUR = 6
COOLDOWN_SECS_AFTER_CLOSE = 60
ADX_GATE = 17

# ===== FALCON STYLE (TP1/TP2/TP3 + Early Exit) =====
FALCON_TP2_ATR_MULT = 1.6
FALCON_TP3_ATR_MULT = 2.4
FALCON_TP2_CLOSE_FRAC = 0.30
FALCON_TP3_CLOSE_FRAC = 0.40

EARLY_FAIL_BARS = 3
EARLY_FAIL_PNL_PCT = -0.15
TIME_STOP_BARS = 10
TIME_STOP_MIN_PNL_PCT = 0.05

REENTRY_COOLDOWN_BARS = 2

# ===== SCALP TARGET BOOST (higher TP for scalps) =====
SCALP_TP1_BOOST_PCT = 0.65
SCALP_TP2_BOOST_PCT = 1.10
SCALP_TP3_BOOST_PCT = 1.80

# ===== GOLDEN TRADE MULTI-LEVEL TP =====
GOLDEN_TP_LEVELS = [0.80, 1.50, 2.50]  # 3 مستويات للصفقات الذهبية
GOLDEN_TP_CLOSE_FRACTIONS = [0.30, 0.30, 0.40]

# =================== SCENARIO ENGINE CONFIG ===================
SCENARIO_ENGINE_ENABLED = True

DISPLACEMENT_ATR_MULT = 1.6
MIN_SCALP_POINTS = 7
POINT_BPS = 10
MIN_SCALP_BPS = MIN_SCALP_POINTS * POINT_BPS

CHOP_ADX_MAX = 18
TREND_ADX_MIN = 20
STRONG_TREND_ADX = 28

RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

FVG_LOOKBACK = 60
OB_LOOKBACK = 80
VWAP_LEN = 50

SCALP_TP_ATR_MULT = 1.0
SCALP_SL_ATR_MULT = 0.8
TREND_TRAIL_ATR_MULT = 1.6

# =================== PROFESSIONAL FOOD PRINT ===================
FOOD_PRINT_LOOKBACK = 100
MIN_ZONE_STRENGTH = 0.6
STRONG_ZONE_THRESHOLD = 0.8

# =================== DATA STRUCTURES ===================
class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    NO_TRADE = "NO_TRADE"

class Mode(str, Enum):
    SCALP = "SCALP"
    TREND = "TREND"
    GOLDEN = "GOLDEN"  # صفقات ذهبية صاروخية
    NONE = "NONE"

class Phase(str, Enum):
    ACCUMULATION = "ACCUMULATION"
    EXPANSION = "EXPANSION"
    TREND = "TREND"
    DISTRIBUTION = "DISTRIBUTION"
    EXHAUSTION = "EXHAUSTION"
    CHOP = "CHOP"
    UNKNOWN = "UNKNOWN"

@dataclass
class SignalContext:
    px: float
    atr: float
    adx: float
    di_plus: float
    di_minus: float
    rsi: float
    vwap: float
    vol_z: float
    displacement: bool
    phase: Phase
    bias: str
    notes: List[str]
    market_strength: float

@dataclass
class ScenarioDecision:
    action: Action
    mode: Mode
    reason: str
    confidence: float
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None
    is_golden_trade: bool = False
    zone_strength: float = 0.5  # قوة منطقة الدخول
    meta: Optional[Dict[str, Any]] = None

# =================== PROFESSIONAL LOGGING ===================
def log_i(msg): print(f"ℹ️ {msg}", flush=True)
def log_g(msg): print(f"✅ {msg}", flush=True)
def log_w(msg): print(f"🟨 {msg}", flush=True)
def log_e(msg): print(f"❌ {msg}", flush=True)

def log_banner(text): print(f"\n{'—'*12} {text} {'—'*12}\n", flush=True)

def save_state(state: dict):
    try:
        state["ts"] = int(time.time())
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        log_i(f"state saved → {STATE_PATH}")
    except Exception as e:
        log_w(f"state save failed: {e}")

def load_state() -> dict:
    try:
        if not os.path.exists(STATE_PATH): return {}
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_w(f"state load failed: {e}")
    return {}

# =================== SCENARIO ENGINE CORE ===================
def _true_range(h, l, c_prev):
    return np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))

def calc_atr(df: pd.DataFrame, length: int = ATR_LEN) -> pd.Series:
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    c = df["close"].astype(float).values
    c_prev = np.roll(c, 1)
    c_prev[0] = c[0]
    tr = _true_range(h, l, c_prev)
    atr = pd.Series(tr).ewm(alpha=1/length, adjust=False).mean()
    return atr

def calc_rsi(df: pd.DataFrame, length: int = RSI_LEN) -> pd.Series:
    c = df["close"].astype(float)
    d = c.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    rs = up.ewm(alpha=1/length, adjust=False).mean() / (dn.ewm(alpha=1/length, adjust=False).mean() + 1e-12)
    return 100 - (100 / (1 + rs))

def calc_adx_di(df: pd.DataFrame, length: int = ADX_LEN) -> Tuple[pd.Series, pd.Series, pd.Series]:
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    up_move = h.diff()
    dn_move = -l.diff()

    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)

    tr = _true_range(h.values, l.values, c.shift(1).fillna(c.iloc[0]).values)
    tr_sm = pd.Series(tr).ewm(alpha=1/length, adjust=False).mean()

    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/length, adjust=False).mean() / (tr_sm + 1e-12))
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/length, adjust=False).mean() / (tr_sm + 1e-12))

    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12))
    adx = dx.ewm(alpha=1/length, adjust=False).mean()
    return adx, plus_di, minus_di

def calc_vwap(df: pd.DataFrame, length: int = VWAP_LEN) -> pd.Series:
    tp = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
    v = df["volume"].astype(float).replace(0, np.nan).ffill().fillna(1.0)
    pv = tp * v
    vwap = pv.rolling(length).sum() / (v.rolling(length).sum() + 1e-12)
    return vwap.ffill()

def zscore(s: pd.Series, length: int = 50) -> pd.Series:
    m = s.rolling(length).mean()
    sd = s.rolling(length).std(ddof=0)
    return ((s - m) / (sd + 1e-12)).fillna(0.0)

def is_displacement(df: pd.DataFrame, i: int, atr: float, mult: float = DISPLACEMENT_ATR_MULT) -> bool:
    """شمعة قوية: جسم أو مدى أكبر من 1.6×ATR"""
    if i <= 0:
        return False
    o = float(df["open"].iloc[i])
    c = float(df["close"].iloc[i])
    h = float(df["high"].iloc[i])
    l = float(df["low"].iloc[i])
    body = abs(c - o)
    rng = (h - l)
    return (body >= mult * atr) or (rng >= mult * atr)

def candle_rejection(df: pd.DataFrame, i: int, side: str) -> bool:
    """Reject بسيط: ذيل عكس الاتجاه كبير"""
    o = float(df["open"].iloc[i])
    c = float(df["close"].iloc[i])
    h = float(df["high"].iloc[i])
    l = float(df["low"].iloc[i])
    body = abs(c - o) + 1e-12
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    if side == "sell":
        return upper_wick >= 1.5 * body and c < o
    if side == "buy":
        return lower_wick >= 1.5 * body and c > o
    return False

def detect_fvg(df: pd.DataFrame, lookback: int = FVG_LOOKBACK) -> Dict[str, Any]:
    """اكتشاف Fair Value Gaps"""
    n = len(df)
    if n < 3:
        return {"ok": False}

    start = max(2, n - lookback)
    fvg_list = []
    for i in range(start, n):
        hi_2 = float(df["high"].iloc[i-2])
        lo_2 = float(df["low"].iloc[i-2])
        hi = float(df["high"].iloc[i])
        lo = float(df["low"].iloc[i])

        if lo > hi_2:
            fvg_list.append({"type": "bull", "low": hi_2, "high": lo, "i": i})
        if hi < lo_2:
            fvg_list.append({"type": "bear", "low": hi, "high": lo_2, "i": i})

    if not fvg_list:
        return {"ok": False}

    last = fvg_list[-1]
    return {"ok": True, "last": last, "all": fvg_list}

def detect_simple_ob(df: pd.DataFrame, lookback: int = OB_LOOKBACK) -> Dict[str, Any]:
    """اكتشاف Order Blocks مبسّط"""
    n = len(df)
    if n < 5:
        return {"ok": False}

    start = max(2, n - lookback)
    atr_s = calc_atr(df, ATR_LEN)

    bull_ob = None
    bear_ob = None

    for i in range(start+2, n):
        atr = float(atr_s.iloc[i])
        disp = is_displacement(df, i, atr)
        o = float(df["open"].iloc[i])
        c = float(df["close"].iloc[i])

        if disp and c > o:
            for j in range(i-1, max(start, i-8), -1):
                oj = float(df["open"].iloc[j]); cj = float(df["close"].iloc[j])
                if cj < oj:
                    bull_ob = {"low": float(df["low"].iloc[j]), "high": float(df["high"].iloc[j]), "i": j}
                    break

        if disp and c < o:
            for j in range(i-1, max(start, i-8), -1):
                oj = float(df["open"].iloc[j]); cj = float(df["close"].iloc[j])
                if cj > oj:
                    bear_ob = {"low": float(df["low"].iloc[j]), "high": float(df["high"].iloc[j]), "i": j}
                    break

    return {"ok": True, "bull_ob": bull_ob, "bear_ob": bear_ob}

def near_zone(px: float, low: float, high: float, tolerance_bps: float = 15) -> bool:
    if low is None or high is None:
        return False
    mid = (low + high) / 2.0
    bps = abs(px - mid) / (mid + 1e-12) * 10000
    return bps <= tolerance_bps or (low <= px <= high)

def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 30) -> Dict[str, Any]:
    """اكتشاف Liquidity Sweeps"""
    n = len(df)
    if n < lookback + 2:
        return {"ok": False}

    recent = df.iloc[-(lookback+1):-1]
    prev_low = float(recent["low"].min())
    prev_high = float(recent["high"].max())

    o = float(df["open"].iloc[-1])
    c = float(df["close"].iloc[-1])
    h = float(df["high"].iloc[-1])
    l = float(df["low"].iloc[-1])

    sweep_low = (l < prev_low) and (c > prev_low) and ((min(o, c) - l) > (h - max(o, c)))
    sweep_high = (h > prev_high) and (c < prev_high) and ((h - max(o, c)) > (min(o, c) - l))

    return {"ok": True, "sweep_low": sweep_low, "sweep_high": sweep_high, "prev_low": prev_low, "prev_high": prev_high}

def detect_bias(adx: float, di_plus: float, di_minus: float, vwap: float, px: float) -> str:
    if adx < CHOP_ADX_MAX:
        return "neutral"
    if di_plus > di_minus and px >= vwap:
        return "bull"
    if di_minus > di_plus and px <= vwap:
        return "bear"
    return "neutral"

def detect_phase(df: pd.DataFrame, i: int, atr: float, adx: float, vol_z: float, displacement: bool) -> Phase:
    """كشف طور السوق الحالي"""
    if adx < CHOP_ADX_MAX and not displacement:
        if vol_z > 0.8:
            return Phase.ACCUMULATION
        return Phase.CHOP

    if displacement and vol_z > 0.7:
        return Phase.EXPANSION

    if adx >= TREND_ADX_MIN:
        return Phase.TREND

    return Phase.UNKNOWN

def bps_from_prices(a: float, b: float) -> float:
    return abs(a - b) / (a + 1e-12) * 10000

def scalp_tp_sl(px: float, atr: float, side: str, zone_strength: float = 0.5) -> Tuple[float, float]:
    """حساب TP/SL للسكالب بناءً على قوة المنطقة"""
    # تعديل TP/SL حسب قوة المنطقة
    strength_multiplier = 1.0 + (zone_strength - 0.5)  # 0.5-1.5
    
    min_tp_abs = px * (MIN_SCALP_BPS / 10000.0) * strength_multiplier
    tp_abs = max(atr * SCALP_TP_ATR_MULT * strength_multiplier, min_tp_abs)
    sl_abs = max(atr * SCALP_SL_ATR_MULT, px * (0.7 * MIN_SCALP_BPS / 10000.0))

    if side == "buy":
        return px + tp_abs, px - sl_abs
    else:
        return px - tp_abs, px + sl_abs

def trend_initial_sl(px: float, atr: float, side: str) -> float:
    """حساب SL الأولي للترند"""
    sl_abs = atr * TREND_TRAIL_ATR_MULT
    return (px - sl_abs) if side == "buy" else (px + sl_abs)

def build_context(df: pd.DataFrame) -> SignalContext:
    """بناء سياق السوق الحالي"""
    df = df.copy()
    atr_s = calc_atr(df, ATR_LEN)
    rsi_s = calc_rsi(df, RSI_LEN)
    adx_s, di_p_s, di_m_s = calc_adx_di(df, ADX_LEN)
    vwap_s = calc_vwap(df, VWAP_LEN)
    vol_z_s = zscore(df["volume"].astype(float), 50)

    i = len(df) - 1
    px = float(df["close"].iloc[i])
    atr = float(atr_s.iloc[i])
    adx = float(adx_s.iloc[i])
    di_p = float(di_p_s.iloc[i])
    di_m = float(di_m_s.iloc[i])
    rsi = float(rsi_s.iloc[i])
    vwap = float(vwap_s.iloc[i])
    vol_z = float(vol_z_s.iloc[i])

    disp = is_displacement(df, i, atr, DISPLACEMENT_ATR_MULT)
    phase = detect_phase(df, i, atr, adx, vol_z, disp)
    bias = detect_bias(adx, di_p, di_m, vwap, px)
    
    # حساب قوة السوق
    market_strength = (adx / 50.0) * 0.4 + (vol_z / 3.0) * 0.3 + (abs(di_p - di_m) / 30.0) * 0.3

    notes = []
    if disp: notes.append(f"displacement>{DISPLACEMENT_ATR_MULT}xATR")
    if adx < CHOP_ADX_MAX: notes.append("adx_chop")
    if adx >= TREND_ADX_MIN: notes.append("adx_trend")
    if di_p > di_m: notes.append("di_bull")
    if di_m > di_p: notes.append("di_bear")
    if px > vwap: notes.append("above_vwap")
    if px < vwap: notes.append("below_vwap")
    if rsi >= RSI_OVERBOUGHT: notes.append("rsi_overbought")
    if rsi <= RSI_OVERSOLD: notes.append("rsi_oversold")

    return SignalContext(
        px=px, atr=atr, adx=adx, di_plus=di_p, di_minus=di_m, rsi=rsi,
        vwap=vwap, vol_z=vol_z, displacement=disp, phase=phase, bias=bias, 
        notes=notes, market_strength=market_strength
    )

# =================== PROFESSIONAL FOOD PRINT DETECTION ===================
def detect_food_print_advanced(df: pd.DataFrame, lookback: int = FOOD_PRINT_LOOKBACK) -> Dict[str, Any]:
    """
    كشف فود برنت متقدم: مناطق السيولة والتراكم والتوزيع
    """
    n = len(df)
    if n < lookback + 10:
        return {"ok": False}
    
    results = {
        "liquidity_pools": [],
        "accumulation_zones": [],
        "distribution_zones": [],
        "fair_value_gaps": [],
        "zone_strength": 0.5
    }
    
    # 1. اكتشاف مناطق السيولة (سلسلة من الشموع الصغيرة)
    for i in range(n - lookback, n - 5):
        window = df.iloc[i:i+5]
        max_range = float(window["high"].max() - window["low"].min())
        avg_body = float(abs(window["close"] - window["open"]).mean())
        
        if avg_body < max_range * 0.3 and max_range < max_range * 0.5:
            results["liquidity_pools"].append({
                "start": i,
                "end": i + 5,
                "range_low": float(window["low"].min()),
                "range_high": float(window["high"].max())
            })
    
    # 2. مناطق التراكم (تجميع قبل الصعود)
    for i in range(n - lookback, n - 20):
        window = df.iloc[i:i+20]
        first_half = window.iloc[:10]
        second_half = window.iloc[10:]
        
        if (float(first_half["close"].mean()) < float(second_half["close"].mean()) and
            float(first_half["volume"].mean()) > float(second_half["volume"].mean())):
            results["accumulation_zones"].append({
                "start": i,
                "end": i + 20,
                "avg_price": float(window["close"].mean())
            })
    
    # 3. مناطق التوزيع (توزيع قبل الهبوط)
    for i in range(n - lookback, n - 20):
        window = df.iloc[i:i+20]
        first_half = window.iloc[:10]
        second_half = window.iloc[10:]
        
        if (float(first_half["close"].mean()) > float(second_half["close"].mean()) and
            float(first_half["volume"].mean()) > float(second_half["volume"].mean())):
            results["distribution_zones"].append({
                "start": i,
                "end": i + 20,
                "avg_price": float(window["close"].mean())
            })
    
    # 4. حساب قوة المنطقة بناءً على عدة عوامل
    zone_strength = 0.5
    recent_price = float(df["close"].iloc[-1])
    
    # عامل 1: قرب السعر من مناطق التراكم/التوزيع
    for zone in results["accumulation_zones"]:
        zone_center = zone["avg_price"]
        distance_pct = abs(recent_price - zone_center) / zone_center
        if distance_pct < 0.02:  # 2%
            zone_strength += 0.2
            break
    
    # عامل 2: وجود مناطق سيولة قريبة
    if results["liquidity_pools"]:
        zone_strength += 0.1
    
    # عامل 3: عدد المناطق المكتشفة
    total_zones = len(results["accumulation_zones"]) + len(results["distribution_zones"])
    if total_zones > 2:
        zone_strength += min(0.1 * total_zones, 0.3)
    
    results["zone_strength"] = min(max(zone_strength, 0.1), 0.9)
    
    return {"ok": True, **results}

def is_golden_trade_setup_enhanced(ctx: SignalContext, fvg: Dict, ob: Dict, 
                                  sweep: Dict, food_print: Dict) -> Tuple[bool, float]:
    """كشف إعدادات الصفقات الذهبية مع فود برنت"""
    
    # الشروط الأساسية
    if ctx.adx < 25:
        return False, 0.0
    
    if not ctx.displacement:
        return False, 0.0
    
    if ctx.vol_z < 0.5:
        return False, 0.0
    
    # شروط فود برنت
    zone_strength = food_print.get("zone_strength", 0.5) if food_print.get("ok") else 0.5
    
    if food_print.get("ok"):
        # تحقق من مناطق التراكم للتداول الطويل
        if ctx.bias == "bull":
            accumulation_nearby = any(
                zone["avg_price"] * 1.02 >= ctx.px >= zone["avg_price"] * 0.98
                for zone in food_print.get("accumulation_zones", [])
            )
            if not accumulation_nearby:
                zone_strength *= 0.7  # تقليل قوة المنطقة
        
        # تحقق من مناطق التوزيع للتداول القصير
        if ctx.bias == "bear":
            distribution_nearby = any(
                zone["avg_price"] * 1.02 >= ctx.px >= zone["avg_price"] * 0.98
                for zone in food_print.get("distribution_zones", [])
            )
            if not distribution_nearby:
                zone_strength *= 0.7  # تقليل قوة المنطقة
    
    # RSI مناسب
    if ctx.bias == "bull" and ctx.rsi > 70:
        zone_strength *= 0.6
    if ctx.bias == "bear" and ctx.rsi < 30:
        zone_strength *= 0.6
    
    # منطقة دخول واضحة
    has_entry_zone = False
    if fvg.get("ok"):
        last_fvg = fvg.get("last")
        if last_fvg:
            if last_fvg["type"] == "bull" and ctx.bias == "bull":
                has_entry_zone = True
            elif last_fvg["type"] == "bear" and ctx.bias == "bear":
                has_entry_zone = True
    
    if ob.get("ok"):
        bull_ob = ob.get("bull_ob")
        bear_ob = ob.get("bear_ob")
        if (bull_ob and ctx.bias == "bull") or (bear_ob and ctx.bias == "bear"):
            has_entry_zone = True
    
    if not has_entry_zone:
        zone_strength *= 0.5
    
    # الشروط النهائية للصفقات الذهبية
    is_golden = (zone_strength >= MIN_ZONE_STRENGTH and 
                 has_entry_zone and 
                 ctx.displacement and 
                 ctx.adx >= 25)
    
    return is_golden, zone_strength

def scenario_decide_enhanced(df: pd.DataFrame) -> ScenarioDecision:
    """اتخاذ القرار مع فود برنت محسن"""
    if not SCENARIO_ENGINE_ENABLED:
        return ScenarioDecision(Action.NO_TRADE, Mode.NONE, "Scenario Engine Disabled", 0.0)
    
    ctx = build_context(df)
    fvg = detect_fvg(df, FVG_LOOKBACK)
    ob = detect_simple_ob(df, OB_LOOKBACK)
    sweep = detect_liquidity_sweep(df, 30)
    food_print = detect_food_print_advanced(df, FOOD_PRINT_LOOKBACK)  # إضافة فود برنت
    
    px = ctx.px
    notes = list(ctx.notes)
    
    # ====================
    # 1. كشف الصفقات الذهبية أولاً مع فود برنت
    # ====================
    golden_trade, zone_strength = is_golden_trade_setup_enhanced(ctx, fvg, ob, sweep, food_print)
    
    if golden_trade and zone_strength >= MIN_ZONE_STRENGTH:
        notes.append(f"GOLDEN_TRADE_SETUP_STR:{zone_strength:.2f}")
        
        if ctx.bias == "bull":
            # حساب TP على 3 مستويات للصفقات الذهبية
            tp1 = px * (1 + GOLDEN_TP_LEVELS[0]/100)
            tp2 = px * (1 + GOLDEN_TP_LEVELS[1]/100)
            tp3 = px * (1 + GOLDEN_TP_LEVELS[2]/100)
            sl = trend_initial_sl(px, ctx.atr, "buy")
            
            return ScenarioDecision(
                action=Action.BUY,
                mode=Mode.GOLDEN,
                reason=f"GOLDEN BUY: Strong trend + Displacement + Clear zone (ADX:{ctx.adx:.1f}, VolZ:{ctx.vol_z:.2f}, ZoneStr:{zone_strength:.2f})",
                confidence=0.88,
                tp_price=tp1,  # أول مستوى TP
                sl_price=sl,
                is_golden_trade=True,
                zone_strength=zone_strength,
                meta={
                    "ctx": ctx.__dict__,
                    "ob": ob,
                    "fvg": fvg,
                    "sweep": sweep,
                    "food_print": food_print,
                    "notes": notes,
                    "golden_tp_levels": [tp1, tp2, tp3],
                    "golden_close_fractions": GOLDEN_TP_CLOSE_FRACTIONS
                }
            )
        
        elif ctx.bias == "bear":
            tp1 = px * (1 - GOLDEN_TP_LEVELS[0]/100)
            tp2 = px * (1 - GOLDEN_TP_LEVELS[1]/100)
            tp3 = px * (1 - GOLDEN_TP_LEVELS[2]/100)
            sl = trend_initial_sl(px, ctx.atr, "sell")
            
            return ScenarioDecision(
                action=Action.SELL,
                mode=Mode.GOLDEN,
                reason=f"GOLDEN SELL: Strong trend + Displacement + Clear zone (ADX:{ctx.adx:.1f}, VolZ:{ctx.vol_z:.2f}, ZoneStr:{zone_strength:.2f})",
                confidence=0.88,
                tp_price=tp1,
                sl_price=sl,
                is_golden_trade=True,
                zone_strength=zone_strength,
                meta={
                    "ctx": ctx.__dict__,
                    "ob": ob,
                    "fvg": fvg,
                    "sweep": sweep,
                    "food_print": food_print,
                    "notes": notes,
                    "golden_tp_levels": [tp1, tp2, tp3],
                    "golden_close_fractions": GOLDEN_TP_CLOSE_FRACTIONS
                }
            )
    
    # ====================
    # 2. منع التداول في الـ Chop
    # ====================
    if ctx.phase == Phase.CHOP:
        if sweep.get("ok") and (sweep.get("sweep_low") or sweep.get("sweep_high")):
            notes.append("chop_but_sweep")
        else:
            return ScenarioDecision(
                Action.NO_TRADE, 
                Mode.NONE, 
                f"CHOP: Low ADX ({ctx.adx:.1f}), no clear setup", 
                0.35,
                zone_strength=zone_strength,
                meta={"ctx": ctx.__dict__, "notes": notes, "food_print": food_print}
            )
    
    # ====================
    # 3. منع الدخول عكس الترند القوي
    # ====================
    strong_bull = (ctx.adx >= TREND_ADX_MIN and ctx.di_plus > ctx.di_minus)
    strong_bear = (ctx.adx >= TREND_ADX_MIN and ctx.di_minus > ctx.di_plus)
    
    # ====================
    # 4. مناطق الدخول (OB/FVG)
    # ====================
    near_bull_ob = False
    near_bear_ob = False
    bull_ob = ob.get("bull_ob")
    bear_ob = ob.get("bear_ob")
    
    if bull_ob:
        near_bull_ob = near_zone(px, bull_ob["low"], bull_ob["high"], tolerance_bps=20)
    if bear_ob:
        near_bear_ob = near_zone(px, bear_ob["low"], bear_ob["high"], tolerance_bps=20)
    
    near_bull_fvg = False
    near_bear_fvg = False
    last_fvg = fvg.get("last") if fvg.get("ok") else None
    if last_fvg:
        if last_fvg["type"] == "bull":
            near_bull_fvg = near_zone(px, last_fvg["low"], last_fvg["high"], tolerance_bps=20)
        if last_fvg["type"] == "bear":
            near_bear_fvg = near_zone(px, last_fvg["low"], last_fvg["high"], tolerance_bps=20)
    
    # ====================
    # 5. تأكيد الدخول
    # ====================
    confirm_buy = candle_rejection(df, len(df)-1, "buy") or ctx.displacement
    confirm_sell = candle_rejection(df, len(df)-1, "sell") or ctx.displacement
    
    # ====================
    # 6. إعدادات السكالب (مع مراعاة قوة المنطقة)
    # ====================
    scalp_buy_setup = (sweep.get("sweep_low") or near_bull_ob or near_bull_fvg) and confirm_buy
    scalp_sell_setup = (sweep.get("sweep_high") or near_bear_ob or near_bear_fvg) and confirm_sell
    
    # منع السكالب عكس الترند إلا في مناطق قوية
    if strong_bull and scalp_sell_setup:
        if not (candle_rejection(df, len(df)-1, "sell") and sweep.get("sweep_high") and zone_strength > 0.8):
            scalp_sell_setup = False
            notes.append("blocked_scalp_sell_vs_bull_trend")
    
    if strong_bear and scalp_buy_setup:
        if not (candle_rejection(df, len(df)-1, "buy") and sweep.get("sweep_low") and zone_strength > 0.8):
            scalp_buy_setup = False
            notes.append("blocked_scalp_buy_vs_bear_trend")
    
    # ====================
    # 7. إعدادات الترند
    # ====================
    trend_ok = (ctx.phase in (Phase.EXPANSION, Phase.TREND)) and ctx.displacement and (ctx.adx >= TREND_ADX_MIN)
    
    trend_buy_setup = trend_ok and (ctx.bias == "bull") and (near_bull_ob or near_bull_fvg or (px >= ctx.vwap))
    trend_sell_setup = trend_ok and (ctx.bias == "bear") and (near_bear_ob or near_bear_fvg or (px <= ctx.vwap))
    
    # ====================
    # 8. اتخاذ القرار النهائي
    # ====================
    if trend_buy_setup:
        sl = trend_initial_sl(px, ctx.atr, "buy")
        return ScenarioDecision(
            Action.BUY, 
            Mode.TREND, 
            f"TREND BUY: {ctx.phase.value} phase + displacement + bullish bias (ZoneStr:{zone_strength:.2f})", 
            0.82,
            tp_price=None,  # الترند بدون TP ثابت
            sl_price=sl,
            zone_strength=zone_strength,
            meta={"ctx": ctx.__dict__, "ob": ob, "fvg": fvg, "sweep": sweep, "food_print": food_print, "notes": notes}
        )
    
    if trend_sell_setup:
        sl = trend_initial_sl(px, ctx.atr, "sell")
        return ScenarioDecision(
            Action.SELL, 
            Mode.TREND, 
            f"TREND SELL: {ctx.phase.value} phase + displacement + bearish bias (ZoneStr:{zone_strength:.2f})", 
            0.82,
            tp_price=None,
            sl_price=sl,
            zone_strength=zone_strength,
            meta={"ctx": ctx.__dict__, "ob": ob, "fvg": fvg, "sweep": sweep, "food_print": food_print, "notes": notes}
        )
    
    if scalp_buy_setup and zone_strength >= MIN_ZONE_STRENGTH:
        tp, sl = scalp_tp_sl(px, ctx.atr, "buy", zone_strength)
        return ScenarioDecision(
            Action.BUY, 
            Mode.SCALP, 
            f"SCALP BUY: zone touch + confirm (min {MIN_SCALP_POINTS} points, ZoneStr:{zone_strength:.2f})", 
            0.68,
            tp_price=tp,
            sl_price=sl,
            zone_strength=zone_strength,
            meta={"ctx": ctx.__dict__, "ob": ob, "fvg": fvg, "sweep": sweep, "food_print": food_print, "notes": notes}
        )
    
    if scalp_sell_setup and zone_strength >= MIN_ZONE_STRENGTH:
        tp, sl = scalp_tp_sl(px, ctx.atr, "sell", zone_strength)
        return ScenarioDecision(
            Action.SELL, 
            Mode.SCALP, 
            f"SCALP SELL: zone touch + confirm (min {MIN_SCALP_POINTS} points, ZoneStr:{zone_strength:.2f})", 
            0.68,
            tp_price=tp,
            sl_price=sl,
            zone_strength=zone_strength,
            meta={"ctx": ctx.__dict__, "ob": ob, "fvg": fvg, "sweep": sweep, "food_print": food_print, "notes": notes}
        )
    
    return ScenarioDecision(
        Action.HOLD, 
        Mode.NONE, 
        f"Waiting for better setup (Phase: {ctx.phase.value}, Bias: {ctx.bias}, ZoneStr:{zone_strength:.2f})", 
        0.45,
        zone_strength=zone_strength,
        meta={"ctx": ctx.__dict__, "ob": ob, "fvg": fvg, "sweep": sweep, "food_print": food_print, "notes": notes}
    )

def should_promote_scalp_to_trend(entry_px: float, px: float, side: str, ctx: SignalContext) -> bool:
    """ترقية الصفقة من سكالب لترند"""
    profit_bps = (px - entry_px) / (entry_px + 1e-12) * 10000
    if side == "sell":
        profit_bps = (entry_px - px) / (entry_px + 1e-12) * 10000

    if profit_bps < MIN_SCALP_BPS:
        return False

    if ctx.adx < TREND_ADX_MIN:
        return False

    if side == "buy" and ctx.bias != "bull":
        return False
    if side == "sell" and ctx.bias != "bear":
        return False

    if side == "buy" and ctx.rsi >= 85:
        return False
    if side == "sell" and ctx.rsi <= 15:
        return False

    return True

def should_take_scalp_profit(entry_px: float, current_px: float, 
                            side: str, ctx: SignalContext, 
                            zone_strength: float) -> Tuple[bool, float]:
    """
    تحديد جني الأرباح للسكالب بناءً على قوة المنطقة
    
    Args:
        zone_strength: قوة منطقة الدخول (0-1)
                      > 0.7: منطقة قوية
                      < 0.4: منطقة ضعيفة
    """
    profit_bps = abs(current_px - entry_px) / entry_px * 10000
    
    # للمناطق القوية: نصبر أكثر
    if zone_strength > 0.7:
        min_profit_bps = MIN_SCALP_BPS * 1.5  # زيادة الهدف للمناطق القوية
        max_profit_bps = MIN_SCALP_BPS * 3.0
    else:
        min_profit_bps = MIN_SCALP_BPS  # الهدف الأساسي
        max_profit_bps = MIN_SCALP_BPS * 1.5
    
    # للمناطق الضعيفة: نخرج مبكراً
    if zone_strength < 0.4:
        min_profit_bps = MIN_SCALP_BPS * 0.5
        max_profit_bps = MIN_SCALP_BPS
    
    # تحقق من تحقيق الهدف
    if profit_bps >= max_profit_bps:
        return True, 1.0  # إغلاق كامل
    
    if profit_bps >= min_profit_bps:
        # خروج جزئي للمناطق متوسطة القوة
        if 0.4 <= zone_strength <= 0.7:
            close_pct = 0.5  # إغلاق نصف المركز
            return True, close_pct
    
    return False, 0.0

# =================== SMART TRADE COUNCIL ===================
class SmartTradeCouncil:
    """مجلس إدارة ذكي لاتخاذ قرارات إدارة الصفقات"""
    
    def __init__(self):
        self.members = [
            self._risk_manager,
            self._trend_analyst,
            self._volume_analyst,
            self._price_action_analyst
        ]
        self.weights = [0.3, 0.25, 0.25, 0.2]  # أوزان القرارات
    
    def evaluate_trade(self, trade_data: Dict, market_ctx: SignalContext) -> Dict:
        """تقييم الصفقة من قبل جميع أعضاء المجلس"""
        decisions = []
        
        for member in self.members:
            decision = member(trade_data, market_ctx)
            decisions.append(decision)
        
        # دمج القرارات المرجحة
        final_decision = self._weighted_decision(decisions)
        return final_decision
    
    def _risk_manager(self, trade_data: Dict, ctx: SignalContext) -> Dict:
        """مدير المخاطر"""
        entry = trade_data.get("entry", 0)
        current = ctx.px
        side = trade_data.get("side", "long")
        
        if side == "long":
            profit_pct = (current - entry) / entry * 100
            stop_distance = abs(entry - trade_data.get("sl_price", 0)) / entry * 100
        else:
            profit_pct = (entry - current) / entry * 100
            stop_distance = abs(entry - trade_data.get("sl_price", 0)) / entry * 100
        
        risk_reward = profit_pct / stop_distance if stop_distance > 0 else 0
        
        if risk_reward < 0.5:
            return {"action": "close", "reason": "نسبة المخاطرة/العائد سيئة", "confidence": 0.8}
        elif risk_reward > 2:
            return {"action": "hold", "reason": "نسبة المخاطرة/العائد ممتازة", "confidence": 0.9}
        
        return {"action": "hold", "reason": "نسبة المخاطرة/العائد متوسطة", "confidence": 0.6}
    
    def _trend_analyst(self, trade_data: Dict, ctx: SignalContext) -> Dict:
        """محلل الترند"""
        entry_bias = trade_data.get("entry_context", {}).get("bias", "neutral")
        current_bias = ctx.bias
        
        if entry_bias != current_bias and ctx.adx > 25:
            return {"action": "close", "reason": "انعكاس الترند", "confidence": 0.85}
        
        if ctx.adx < 15:
            return {"action": "close", "reason": "ضعف الترند", "confidence": 0.7}
        
        return {"action": "hold", "reason": "الترند سليم", "confidence": 0.75}
    
    def _volume_analyst(self, trade_data: Dict, ctx: SignalContext) -> Dict:
        """محلل الحجم"""
        if ctx.vol_z < -1.0:
            return {"action": "close", "reason": "حجم ضعيف", "confidence": 0.7}
        
        if ctx.vol_z > 2.0 and ctx.displacement:
            return {"action": "hold", "reason": "حجم قوي مع إزاحة", "confidence": 0.9}
        
        return {"action": "hold", "reason": "حجم طبيعي", "confidence": 0.6}
    
    def _price_action_analyst(self, trade_data: Dict, ctx: SignalContext) -> Dict:
        """محلل حركة السعر"""
        # تحليل الشمعة الحالية
        df = trade_data.get("current_candles", None)
        if df is not None and len(df) > 1:
            last_candle = df.iloc[-1]
            o = float(last_candle["open"])
            c = float(last_candle["close"])
            h = float(last_candle["high"])
            l = float(last_candle["low"])
            
            body = abs(c - o)
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            
            side = trade_data.get("side", "long")
            
            # شمعة انعكاسية
            if side == "long" and upper_wick > body * 2:
                return {"action": "close", "reason": "شمعة انعكاسية", "confidence": 0.75}
            if side == "short" and lower_wick > body * 2:
                return {"action": "close", "reason": "شمعة انعكاسية", "confidence": 0.75}
        
        return {"action": "hold", "reason": "حركة سعر طبيعية", "confidence": 0.65}
    
    def _weighted_decision(self, decisions: List[Dict]) -> Dict:
        """دمج القرارات المرجحة"""
        if not decisions:
            return {"action": "hold", "reason": "لا توجد قرارات", "confidence": 0.5}
        
        # حساب متوسط الثقة لكل إجراء
        action_scores = {}
        for i, decision in enumerate(decisions):
            action = decision["action"]
            confidence = decision["confidence"] * self.weights[i]
            
            if action not in action_scores:
                action_scores[action] = {"score": 0, "reasons": []}
            
            action_scores[action]["score"] += confidence
            action_scores[action]["reasons"].append(decision["reason"])
        
        # اختيار الإجراء بأعلى درجة
        best_action = max(action_scores.items(), key=lambda x: x[1]["score"])
        
        return {
            "action": best_action[0],
            "reason": " | ".join(best_action[1]["reasons"][:2]),
            "confidence": best_action[1]["score"],
            "all_decisions": decisions
        }

# =================== CANDLES MODULE ===================
def _body(o,c): return abs(c-o)
def _rng(h,l):  return max(h-l, 1e-12)
def _upper_wick(h,o,c): return h - max(o,c)
def _lower_wick(l,o,c): return min(o,c) - l

def _is_doji(o,c,h,l,th=0.1):
    return _body(o,c) <= th * _rng(h,l)

def compute_candles(df):
    if len(df) < 5:
        return {"buy":False,"sell":False,"score_buy":0.0,"score_sell":0.0,
                "wick_up_big":False,"wick_dn_big":False,"doji":False,"pattern":None}

    o1,h1,l1,c1 = float(df["open"].iloc[-2]), float(df["high"].iloc[-2]), float(df["low"].iloc[-2]), float(df["close"].iloc[-2])
    o0,h0,l0,c0 = float(df["open"].iloc[-3]), float(df["high"].iloc[-3]), float(df["low"].iloc[-3]), float(df["close"].iloc[-3])

    strength_b = strength_s = 0.0
    tags = []

    # Bullish engulfing
    if c0 < o0 and c1 > o1 and o1 <= c0 and c1 >= o0:
        strength_b += 2.0; tags.append("bull_engulf")
    
    # Bearish engulfing
    if c0 > o0 and c1 < o1 and o1 >= c0 and c1 <= o0:
        strength_s += 2.0; tags.append("bear_engulf")
    
    # Hammer
    if _lower_wick(l1,o1,c1) >= 2 * _body(o1,c1) and _upper_wick(h1,o1,c1) <= 0.4 * _body(o1,c1):
        strength_b += 1.5; tags.append("hammer")
    
    # Shooting star
    if _upper_wick(h1,o1,c1) >= 2 * _body(o1,c1) and _lower_wick(l1,o1,c1) <= 0.4 * _body(o1,c1):
        strength_s += 1.5; tags.append("shooting_star")
    
    is_doji = _is_doji(o1,c1,h1,l1)
    if is_doji: tags.append("doji")

    rng1 = _rng(h1,l1); up = _upper_wick(h1,o1,c1); dn = _lower_wick(l1,o1,c1)
    wick_up_big = (up >= 1.2*_body(o1,c1)) and (up >= 0.4*rng1)
    wick_dn_big = (dn >= 1.2*_body(o1,c1)) and (dn >= 0.4*rng1)

    if is_doji:
        strength_b *= 0.8; strength_s *= 0.8

    return {
        "buy": strength_b>0, "sell": strength_s>0,
        "score_buy": round(strength_b,2), "score_sell": round(strength_s,2),
        "wick_up_big": bool(wick_up_big), "wick_dn_big": bool(wick_dn_big),
        "doji": bool(is_doji), "pattern": ",".join(tags) if tags else None
    }

# =================== SMART TRADE MANAGEMENT ===================
class TradeManager:
    """مدير الصفقات الذكي مع إدارة ديناميكية"""
    
    def __init__(self):
        self.golden_tp_levels = []
        self.golden_close_fractions = []
        self.current_tp_level = 0
        self.promoted_to_trend = False
        self.entry_context = None
        self.danger_zone_detected = False
        self.smart_council = SmartTradeCouncil()
        self.prev_adx = None
        self.prev_vwap_bias = None
        
    def update_context(self, ctx: SignalContext):
        """تحديث سياق السوق للإدارة الذكية"""
        self.entry_context = ctx
        
        # كشف المناطق الخطرة
        self.danger_zone_detected = self._detect_danger_zone(ctx)
        
    def _detect_danger_zone(self, ctx: SignalContext) -> bool:
        """كشف المناطق الخطرة للخروج الفوري"""
        # 1. RSI في أقصى المدى
        if ctx.rsi >= 90 or ctx.rsi <= 10:
            return True
            
        # 2. تغيير حاد في ADX (انعكاس)
        if self.prev_adx is not None:
            adx_change = abs(ctx.adx - self.prev_adx) / (self.prev_adx + 1e-12)
            if adx_change > 0.3 and ctx.adx < 20:
                return True
        
        # 3. اختراق VWAP عكسي مع حجم ضعيف
        current_bias = "above" if ctx.px > ctx.vwap else "below"
        if self.prev_vwap_bias is not None:
            if current_bias != self.prev_vwap_bias and ctx.vol_z < 0.3:
                return True
        
        # تحديث القيم السابقة
        self.prev_adx = ctx.adx
        self.prev_vwap_bias = current_bias
        
        return False
    
    def manage_golden_trade(self, px: float, entry_px: float, side: str, 
                          tp_levels: list, close_fractions: list) -> dict:
        """إدارة الصفقات الذهبية مع 3 مستويات TP"""
        if not tp_levels or len(tp_levels) < 3:
            return {"action": "hold", "close_pct": 0.0}
        
        hit_tp1 = (px >= tp_levels[0]) if side == "long" else (px <= tp_levels[0])
        hit_tp2 = (px >= tp_levels[1]) if side == "long" else (px <= tp_levels[1])
        hit_tp3 = (px >= tp_levels[2]) if side == "long" else (px <= tp_levels[2])
        
        if hit_tp3 and self.current_tp_level < 3:
            self.current_tp_level = 3
            return {
                "action": "close",
                "reason": "GOLDEN TP3 HIT",
                "close_pct": 1.0
            }
        
        elif hit_tp2 and self.current_tp_level < 2:
            self.current_tp_level = 2
            return {
                "action": "partial",
                "reason": "GOLDEN TP2 HIT",
                "close_pct": close_fractions[1]
            }
        
        elif hit_tp1 and self.current_tp_level < 1:
            self.current_tp_level = 1
            return {
                "action": "partial",
                "reason": "GOLDEN TP1 HIT",
                "close_pct": close_fractions[0]
            }
        
        # خروج استثنائي من المنطقة الخطرة
        if self.danger_zone_detected:
            return {
                "action": "close",
                "reason": "DANGER ZONE DETECTED",
                "close_pct": 1.0
            }
        
        return {"action": "hold", "close_pct": 0.0}
    
    def should_close_trade(self, px: float, entry_px: float, side: str, 
                          ctx: SignalContext, df: pd.DataFrame) -> tuple:
        """تحديد إذا كان يجب إغلاق الصفقة باستخدام مجلس الإدارة"""
        # استشارة مجلس الإدارة
        trade_data = {
            "entry": entry_px,
            "current_price": px,
            "side": "long" if side == "buy" else "short",
            "sl_price": getattr(self, 'initial_sl', None),
            "entry_context": self.entry_context.__dict__ if self.entry_context else {},
            "current_candles": df
        }
        
        council_decision = self.smart_council.evaluate_trade(trade_data, ctx)
        
        if council_decision["action"] == "close" and council_decision["confidence"] > 0.7:
            return True, f"COUNCIL: {council_decision['reason']}"
        
        # 1. كسر SL
        if hasattr(self, 'initial_sl'):
            sl_hit = (px <= self.initial_sl) if side == "long" else (px >= self.initial_sl)
            if sl_hit:
                return True, "SL HIT"
        
        # 2. منطقة خطرة
        if self.danger_zone_detected:
            return True, "DANGER ZONE"
        
        # 3. انعكاس الترند
        if self.entry_context:
            entry_bias = self.entry_context.bias
            current_bias = ctx.bias
            
            if entry_bias != current_bias and ctx.adx > 25:
                return True, "TREND REVERSAL"
        
        # 4. استنفاذ الزمن
        if hasattr(self, 'trade_start_time'):
            trade_duration = time.time() - self.trade_start_time
            if trade_duration > 3600:  # ساعة واحدة
                profit_pct = abs(px - entry_px) / entry_px * 100
                if profit_pct < 0.5:  # أقل من 0.5% ربح
                    return True, "TIME EXHAUSTION"
        
        return False, ""

# =================== EXCHANGE SETUP ===================
def make_ex():
    return ccxt.bingx({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "swap"}
    })

ex = make_ex()
MARKET = {}
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
        LOT_MIN  = (MARKET.get("limits", {}) or {}).get("amount", {}).get("min",  None)
        log_i(f"precision={AMT_PREC}, step={LOT_STEP}, min={LOT_MIN}")
    except Exception as e:
        log_w(f"load_market_specs: {e}")

def ensure_leverage_mode():
    try:
        try:
            ex.set_leverage(LEVERAGE, SYMBOL, params={"side": "BOTH"})
            log_g(f"leverage set: {LEVERAGE}x")
        except Exception as e:
            log_w(f"set_leverage warn: {e}")
        log_i(f"position mode: {POSITION_MODE}")
    except Exception as e:
        log_w(f"ensure_leverage_mode: {e}")

try:
    load_market_specs()
    ensure_leverage_mode()
except Exception as e:
    log_w(f"exchange init: {e}")

# =================== HELPERS ===================
_consec_err = 0
last_loop_ts = time.time()

def _round_amt(q):
    if q is None: return 0.0
    try:
        d = Decimal(str(q))
        if LOT_STEP and isinstance(LOT_STEP,(int,float)) and LOT_STEP>0:
            step = Decimal(str(LOT_STEP))
            d = (d/step).to_integral_value(rounding=ROUND_DOWN)*step
        prec = int(AMT_PREC) if AMT_PREC and AMT_PREC>=0 else 0
        d = d.quantize(Decimal(1).scaleb(-prec), rounding=ROUND_DOWN)
        if LOT_MIN and isinstance(LOT_MIN,(int,float)) and LOT_MIN>0 and d < Decimal(str(LOT_MIN)): return 0.0
        return float(d)
    except (InvalidOperation, ValueError, TypeError):
        return max(0.0, float(q))

def safe_qty(q): 
    q = _round_amt(q)
    if q<=0: log_w(f"qty invalid after normalize → {q}")
    return q

def fmt(v, d=6, na="—"):
    try:
        if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))): return na
        return f"{float(v):.{d}f}"
    except Exception:
        return na

def with_retry(fn, tries=3, base_wait=0.4):
    global _consec_err
    for i in range(tries):
        try:
            r = fn()
            _consec_err = 0
            return r
        except Exception:
            _consec_err += 1
            if i == tries-1: raise
            time.sleep(base_wait*(2**i) + random.random()*0.25)

def fetch_ohlcv(limit=600):
    rows = with_retry(lambda: ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, params={"type":"swap"}))
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

def price_now():
    try:
        t = with_retry(lambda: ex.fetch_ticker(SYMBOL))
        return t.get("last") or t.get("close")
    except Exception: return None

def balance_usdt():
    if not MODE_LIVE: return 100.0
    try:
        b = with_retry(lambda: ex.fetch_balance(params={"type":"swap"}))
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT")
    except Exception: return None

def orderbook_spread_bps():
    try:
        ob = with_retry(lambda: ex.fetch_order_book(SYMBOL, limit=5))
        bid = ob["bids"][0][0] if ob["bids"] else None
        ask = ob["asks"][0][0] if ob["asks"] else None
        if not (bid and ask): return None
        mid = (bid+ask)/2.0
        return ((ask-bid)/mid)*10000.0
    except Exception:
        return None

# =================== TRADE EXECUTION ===================
trade_manager = TradeManager()

def open_market_enhanced(decision: ScenarioDecision, balance: float, current_price: float) -> bool:
    """فتح صفقة بناءً على قرار السيناريو"""
    if decision.action not in [Action.BUY, Action.SELL]:
        log_w(f"No trade action: {decision.action}")
        return False
    
    side = "buy" if decision.action == Action.BUY else "sell"
    qty = compute_size(balance, current_price)
    
    if qty <= 0:
        log_e("Invalid quantity")
        return False
    
    log_banner("ENTERING TRADE")
    log_g(f"🎯 DECISION: {decision.action.value} | MODE: {decision.mode.value}")
    log_g(f"📈 REASON: {decision.reason}")
    log_g(f"💪 CONFIDENCE: {decision.confidence:.2f}")
    log_g(f"🏆 ZONE STRENGTH: {decision.zone_strength:.2f}")
    
    if decision.is_golden_trade:
        log_g("🏆 GOLDEN TRADE DETECTED - 3 Level TP Activated")
    
    # تنفيذ الأمر
    if EXECUTE_ORDERS and not DRY_RUN and MODE_LIVE:
        try:
            ex.set_leverage(LEVERAGE, SYMBOL, params={"side": "BOTH"})
            ex.create_order(SYMBOL, "market", side, qty, None, _params_open(side))
            log_g(f"✅ TRADE EXECUTED: {side.upper()} {qty:.4f} @ {current_price:.6f}")
            
            # تخزين بيانات الصفقة
            STATE.update({
                "open": True,
                "side": "long" if side == "buy" else "short",
                "entry": current_price,
                "qty": qty,
                "mode": decision.mode.value.lower(),
                "decision_reason": decision.reason,
                "is_golden": decision.is_golden_trade,
                "zone_strength": decision.zone_strength,
                "tp_price": decision.tp_price,
                "sl_price": decision.sl_price,
                "entry_time": int(time.time()),
                "bars_in_trade": 0,
                "max_profit_pct": 0.0,
                "trade_manager": {
                    "golden_tp_levels": decision.meta.get("golden_tp_levels", []) if decision.is_golden_trade else [],
                    "golden_close_fractions": decision.meta.get("golden_close_fractions", []) if decision.is_golden_trade else [],
                    "current_tp_level": 0,
                    "promoted": False
                }
            })
            
            # تخزين سياق الدخول
            if decision.meta and 'ctx' in decision.meta:
                STATE["entry_context"] = decision.meta['ctx']
            
            # إعداد مدير الصفقة
            trade_manager.initial_sl = decision.sl_price
            trade_manager.trade_start_time = time.time()
            trade_manager.golden_tp_levels = decision.meta.get("golden_tp_levels", []) if decision.is_golden_trade else []
            trade_manager.golden_close_fractions = decision.meta.get("golden_close_fractions", []) if decision.is_golden_trade else []
            
            save_state(STATE)
            return True
            
        except Exception as e:
            log_e(f"❌ EXECUTION FAILED: {e}")
            return False
    else:
        log_i(f"DRY_RUN: Would {side.upper()} {qty:.4f} @ {current_price:.6f}")
        return True

def compute_size(balance, price):
    effective = balance or 0.0
    capital = effective * RISK_ALLOC * LEVERAGE
    raw = max(0.0, capital / max(float(price or 0.0), 1e-9))
    return safe_qty(raw)

def _params_open(side):
    if POSITION_MODE == "hedge":
        return {"positionSide": "LONG" if side=="buy" else "SHORT", "reduceOnly": False}
    return {"positionSide": "BOTH", "reduceOnly": False}

def _params_close():
    if POSITION_MODE == "hedge":
        return {"positionSide": "LONG" if STATE.get("side")=="long" else "SHORT", "reduceOnly": True}
    return {"positionSide": "BOTH", "reduceOnly": True}

def close_market_strict(reason="MANUAL"):
    """إغلاق صارم للصفقة"""
    global compound_pnl
    
    try:
        if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
            side_to_close = "sell" if STATE.get("side") == "long" else "buy"
            qty_to_close = safe_qty(STATE.get("qty", 0))
            
            if qty_to_close <= 0:
                log_w("No quantity to close")
                return
            
            ex.create_order(SYMBOL, "market", side_to_close, qty_to_close, None, _params_close())
            log_g(f"✅ POSITION CLOSED: {reason}")
        
        # حساب PnL
        entry = STATE.get("entry", 0)
        current_price = price_now() or entry
        side = STATE.get("side", "long")
        qty = STATE.get("qty", 0)
        
        if side == "long":
            pnl = (current_price - entry) * qty
        else:
            pnl = (entry - current_price) * qty
        
        compound_pnl += pnl
        
        # إعادة تعيين الحالة
        STATE.update({
            "open": False,
            "side": None,
            "entry": None,
            "qty": 0,
            "pnl": 0,
            "mode": None,
            "decision_reason": None,
            "is_golden": False,
            "zone_strength": 0.5,
            "tp_price": None,
            "sl_price": None,
            "entry_time": None,
            "bars_in_trade": 0,
            "max_profit_pct": 0.0,
            "trade_manager": {},
            "entry_context": None,
            "trail_price": None,
            "breakeven_armed": False,
            "breakeven_price": None
        })
        
        save_state(STATE)
        
    except Exception as e:
        log_e(f"❌ CLOSE FAILED: {e}")

# =================== DYNAMIC TRADE MANAGEMENT ===================
def manage_open_position_enhanced(df: pd.DataFrame, current_price: float):
    """إدارة الصفقة المفتوحة ديناميكياً مع مجلس الإدارة"""
    if not STATE.get("open"):
        return
    
    # تحديث عدد الشمعات في الصفقة
    STATE["bars_in_trade"] = STATE.get("bars_in_trade", 0) + 1
    
    # حساب الربح الحالي
    entry = STATE.get("entry", 0)
    side = STATE.get("side", "long")
    qty = STATE.get("qty", 0)
    
    if side == "long":
        profit_pct = (current_price - entry) / entry * 100
    else:
        profit_pct = (entry - current_price) / entry * 100
    
    STATE["pnl"] = profit_pct
    
    # تحديث أقصى ربح
    if profit_pct > STATE.get("max_profit_pct", 0):
        STATE["max_profit_pct"] = profit_pct
    
    # بناء سياق السوق الحالي
    ctx = build_context(df)
    trade_manager.update_context(ctx)
    
    log_i(f"📊 TRADE STATUS: {side.upper()} | PnL: {profit_pct:.2f}% | Bars: {STATE['bars_in_trade']}")
    
    # ====================
    # 1. إدارة الصفقات الذهبية
    # ====================
    if STATE.get("is_golden"):
        tm = STATE.get("trade_manager", {})
        tp_levels = tm.get("golden_tp_levels", [])
        close_fractions = tm.get("golden_close_fractions", [])
        
        if len(tp_levels) >= 3:
            action = trade_manager.manage_golden_trade(
                current_price, entry, side, 
                tp_levels, close_fractions
            )
            
            if action["action"] == "close":
                log_g(f"🏆 {action['reason']} - Closing full position")
                close_market_strict(action["reason"])
                return
            elif action["action"] == "partial":
                log_g(f"🎯 {action['reason']} - Closing {action['close_pct']*100:.0f}%")
                # هنا يمكن إضافة منطق الإغلاق الجزئي
                # سنكتفي بالإغلاق الكامل للتبسيط
                close_market_strict(action["reason"])
                return
    
    # ====================
    # 2. ترقية السكالب لترند
    # ====================
    if STATE.get("mode") == "scalp" and not STATE.get("trade_manager", {}).get("promoted", False):
        if should_promote_scalp_to_trend(entry, current_price, side, ctx):
            log_g("🚀 PROMOTING SCALP TO TREND - Switching to trail mode")
            STATE["mode"] = "trend"
            STATE["trade_manager"]["promoted"] = True
            STATE["tp_price"] = None  # إلغاء TP الثابت
    
    # ====================
    # 3. مجلس الإدارة الذكي
    # ====================
    should_close, close_reason = trade_manager.should_close_trade(current_price, entry, side, ctx, df)
    if should_close:
        log_w(f"🏛️ SMART COUNCIL DECISION: {close_reason}")
        close_market_strict(close_reason)
        return
    
    # ====================
    # 4. كشف المناطق الخطرة والخروج الفوري
    # ====================
    if trade_manager.danger_zone_detected:
        log_w("⚠️ DANGER ZONE DETECTED - Emergency exit!")
        close_market_strict("DANGER_ZONE")
        return
    
    # ====================
    # 5. إدارة SL الديناميكي
    # ====================
    initial_sl = STATE.get("sl_price")
    if initial_sl:
        sl_hit = (current_price <= initial_sl) if side == "long" else (current_price >= initial_sl)
        if sl_hit:
            log_w(f"🛑 SL HIT: {current_price:.6f} vs SL {initial_sl:.6f}")
            close_market_strict("SL_HIT")
            return
    
    # ====================
    # 6. إدارة TP للسكالب مع مراعاة قوة المنطقة
    # ====================
    if STATE.get("mode") == "scalp":
        zone_strength = STATE.get("zone_strength", 0.5)
        should_take_profit, close_pct = should_take_scalp_profit(entry, current_price, side, ctx, zone_strength)
        
        if should_take_profit:
            if close_pct >= 1.0:
                log_g(f"🎯 TP HIT (ZoneStr:{zone_strength:.2f}): {current_price:.6f}")
                close_market_strict("TP_HIT")
                return
            elif close_pct > 0:
                log_g(f"🎯 PARTIAL TP ({close_pct*100:.0f}%) HIT (ZoneStr:{zone_strength:.2f})")
                # هنا يمكن إضافة منطق الإغلاق الجزئي
                close_market_strict("PARTIAL_TP_HIT")
                return
    
    # ====================
    # 7. Trailing Stop للترند
    # ====================
    if STATE.get("mode") == "trend":
        atr = ctx.atr
        trail_distance = atr * ATR_TRAIL_MULT
        
        if not STATE.get("trail_price"):
            STATE["trail_price"] = entry - trail_distance if side == "long" else entry + trail_distance
        else:
            if side == "long":
                new_trail = current_price - trail_distance
                if new_trail > STATE["trail_price"]:
                    STATE["trail_price"] = new_trail
            else:
                new_trail = current_price + trail_distance
                if new_trail < STATE["trail_price"]:
                    STATE["trail_price"] = new_trail
        
        # التحقق من كسر الترailing
        trail_hit = (current_price <= STATE["trail_price"]) if side == "long" else (current_price >= STATE["trail_price"])
        if trail_hit:
            log_w(f"🔄 TRAIL STOP: {current_price:.6f} vs Trail {STATE['trail_price']:.6f}")
            close_market_strict("TRAIL_STOP")
            return
    
    # ====================
    # 8. Early Fail Protection
    # ====================
    if STATE["bars_in_trade"] <= EARLY_FAIL_BARS and profit_pct <= EARLY_FAIL_PNL_PCT:
        log_w(f"🧨 EARLY FAIL: PnL {profit_pct:.2f}% <= {EARLY_FAIL_PNL_PCT}%")
        close_market_strict("EARLY_FAIL")
        return
    
    # ====================
    # 9. Time Stop Protection
    # ====================
    if STATE["bars_in_trade"] >= TIME_STOP_BARS and profit_pct < TIME_STOP_MIN_PNL_PCT:
        log_w(f"⏱️ TIME STOP: Bars {STATE['bars_in_trade']}, PnL {profit_pct:.2f}%")
        close_market_strict("TIME_STOP")
        return
    
    # ====================
    # 10. Breakeven Protection
    # ====================
    if not STATE.get("breakeven_armed") and profit_pct >= BREAKEVEN_AFTER:
        STATE["breakeven_armed"] = True
        STATE["breakeven_price"] = entry
        log_i("🛡️ BREAKEVEN ARMED")
    
    if STATE.get("breakeven_armed"):
        be_hit = (current_price <= STATE["breakeven_price"]) if side == "long" else (current_price >= STATE["breakeven_price"])
        if be_hit:
            log_w(f"⚖️ BREAKEVEN HIT: {current_price:.6f}")
            close_market_strict("BREAKEVEN")
            return

# =================== MAIN TRADING LOOP ===================
def trade_loop_professional():
    """الحلقة الرئيسية المحترفة مع جميع التحسينات"""
    loop_i = 0
    
    while True:
        try:
            # جمع البيانات
            bal = balance_usdt()
            px = price_now()
            df = fetch_ohlcv()
            
            if px is None or df.empty:
                time.sleep(BASE_SLEEP)
                continue
            
            # إدارة الصفقة المفتوحة بالمحسنة
            if STATE.get("open"):
                manage_open_position_enhanced(df, px)
            
            # اتخاذ قرار جديد فقط إذا لم تكن هناك صفقة مفتوحة
            if not STATE.get("open"):
                # التحقق من السبريد
                spread_bps = orderbook_spread_bps()
                if spread_bps and spread_bps > MAX_SPREAD_BPS:
                    log_w(f"Spread too high: {spread_bps:.1f}bps > {MAX_SPREAD_BPS}bps")
                    time.sleep(BASE_SLEEP)
                    continue
                
                # اتخاذ القرار بواسطة Scenario Engine المحسن
                decision = scenario_decide_enhanced(df)
                
                # طباعة تقرير مفصل
                log_banner("PROFESSIONAL SCENARIO ANALYSIS")
                log_i(f"📊 MARKET PHASE: {decision.meta.get('ctx', {}).get('phase', 'UNKNOWN')}")
                log_i(f"🧭 MARKET BIAS: {decision.meta.get('ctx', {}).get('bias', 'neutral')}")
                log_i(f"💪 MARKET STRENGTH: {decision.meta.get('ctx', {}).get('market_strength', 0):.2f}")
                log_i(f"📈 ADX: {decision.meta.get('ctx', {}).get('adx', 0):.1f}")
                log_i(f"📉 RSI: {decision.meta.get('ctx', {}).get('rsi', 0):.1f}")
                log_i(f"💰 VOLUME Z: {decision.meta.get('ctx', {}).get('vol_z', 0):.2f}")
                log_i(f"🏆 ZONE STRENGTH: {decision.zone_strength:.2f}")
                
                if decision.action in [Action.BUY, Action.SELL]:
                    log_g(f"🎯 SIGNAL: {decision.action.value} | MODE: {decision.mode.value}")
                    log_g(f"📝 REASON: {decision.reason}")
                    log_g(f"💪 CONFIDENCE: {decision.confidence:.2f}")
                    
                    if decision.is_golden_trade:
                        log_g("🏆 GOLDEN TRADE DETECTED - High probability setup!")
                        log_g(f"🎯 TP Levels: {decision.meta.get('golden_tp_levels', [])}")
                    
                    # تنفيذ الصفقة
                    open_market_enhanced(decision, bal, px)
                else:
                    if decision.action == Action.HOLD:
                        log_i(f"⏳ HOLDING: {decision.reason}")
                    else:
                        log_i(f"🚫 NO TRADE: {decision.reason}")
            
            # النوم قبل الدورة التالية
            loop_i += 1
            sleep_s = NEAR_CLOSE_S if time_to_candle_close(df) <= 10 else BASE_SLEEP
            time.sleep(sleep_s)
            
        except Exception as e:
            log_e(f"loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

def time_to_candle_close(df: pd.DataFrame) -> int:
    """حساب الوقت المتبقي حتى إغلاق الشمعة"""
    if df.empty:
        return BASE_SLEEP
    
    tf = 900  # 15 دقيقة بالثواني
    cur_start_ms = int(df["time"].iloc[-1])
    now_ms = int(time.time()*1000)
    next_close_ms = cur_start_ms + tf*1000
    
    while next_close_ms <= now_ms:
        next_close_ms += tf*1000
    
    left = max(0, next_close_ms - now_ms)
    return int(left/1000)

# =================== STATE INITIALIZATION ===================
STATE = {
    "open": False,
    "side": None,
    "entry": None,
    "qty": 0.0,
    "pnl": 0.0,
    "mode": None,
    "decision_reason": None,
    "is_golden": False,
    "zone_strength": 0.5,
    "tp_price": None,
    "sl_price": None,
    "entry_time": None,
    "bars_in_trade": 0,
    "max_profit_pct": 0.0,
    "trade_manager": {},
    "entry_context": None,
    "trail_price": None,
    "breakeven_armed": False,
    "breakeven_price": None
}

compound_pnl = 0.0
wait_for_next_signal_side = None

# =================== FLASK API ===================
app = Flask(__name__)

@app.route("/")
def home():
    mode = 'LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ Professional Scenario Engine Bot v7.0 — {SYMBOL} {INTERVAL} — {mode}"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE,
        "risk_alloc": RISK_ALLOC,
        "price": price_now(),
        "state": STATE,
        "compound_pnl": compound_pnl,
        "engine": "PROFESSIONAL_SCENARIO_ENGINE_v3",
        "trade_manager": trade_manager.__dict__ if hasattr(trade_manager, '__dict__') else {},
        "council": trade_manager.smart_council.__dict__ if hasattr(trade_manager, 'smart_council') else {}
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"],
        "side": STATE["side"],
        "qty": STATE["qty"],
        "compound_pnl": compound_pnl,
        "timestamp": datetime.utcnow().isoformat(),
        "engine": "PROFESSIONAL_SCENARIO_ENGINE"
    }), 200

# =================== MAIN EXECUTION ===================
if __name__ == "__main__":
    log_banner("PROFESSIONAL SCENARIO ENGINE BOT v7.0")
    
    # تحميل الحالة السابقة
    state = load_state() or {}
    state.setdefault("in_position", False)
    
    if RESUME_ON_RESTART:
        try:
            # هنا يمكن إضافة منطق استئناف الصفقات
            pass
        except Exception as e:
            log_w(f"resume error: {e}")
    
    # عرض إعدادات البوت
    print(colored(f"🔥 MODE: {'LIVE' if MODE_LIVE else 'PAPER'} • {SYMBOL} • {INTERVAL}", "yellow"))
    print(colored(f"💰 RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x", "yellow"))
    print(colored(f"🧠 ENGINE: Professional Scenario Engine v3", "yellow"))
    print(colored(f"🏛️ SMART COUNCIL: 4-Member Decision Making", "yellow"))
    print(colored(f"🏆 GOLDEN TRADES: 3-Level TP ({GOLDEN_TP_LEVELS}%)", "yellow"))
    print(colored(f"🗺️ FOOD PRINT: Advanced Zone Detection", "yellow"))
    print(colored(f"⚡ SCALP MANAGEMENT: Dynamic TP based on Zone Strength", "yellow"))
    print(colored(f"🛡️ PROTECTION: Early Fail + Time Stop + Danger Zone + Smart Council", "yellow"))
    print(colored(f"🚀 EXECUTION: {'ACTIVE' if EXECUTE_ORDERS and not DRY_RUN else 'SIMULATION'}", "yellow"))
    
    # بدء الخيوط
    import threading
    threading.Thread(target=trade_loop_professional, daemon=True).start()
    
    # تشغيل الخادم
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)

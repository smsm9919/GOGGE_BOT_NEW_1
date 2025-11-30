# -*- coding: utf-8 -*-
"""
RF Futures Bot — RF-LIVE ONLY (BingX Perp via CCXT)
• Council PRO Unified Decision System with Candles & Golden Entry
• Golden Entry + Golden Reversal + Wick Exhaustion + Smart Profit AI
• Dynamic TP ladder + Breakeven + ATR-trailing
• Smart Exit Management + Wait-for-next-signal
• Professional Logging & Dashboard
• Enhanced with Footprint, SMC Candles, Liquidity Traps + VWAP Strategy
• Box Engine + FVG Detection + RF B&S + Stop Hunt Zones
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
from typing import List, Dict, Optional

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
BOT_VERSION = "DOGE Council PRO v5.0 — Smart Profit AI + Golden Zone Pro + VWAP Strategy + Box Engine + FVG"
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

# =================== ENHANCED SETTINGS ===================
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

ENTRY_RF_ONLY = False
MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", 6.0))

# Enhanced Dynamic TP / trail
TP1_PCT_BASE       = 0.40
TP2_PCT_BASE       = 1.00
TP3_PCT_BASE       = 1.80
TP1_CLOSE_FRAC     = 0.40
TP2_CLOSE_FRAC     = 0.40
TP3_CLOSE_FRAC     = 0.20

BREAKEVEN_AFTER    = 0.30
TRAIL_ACTIVATE_PCT = 1.20
ATR_TRAIL_MULT     = 1.6

# Enhanced Trend TPs for 3-phase profit taking
TREND_TPS       = [0.50, 1.00, 1.80]
TREND_TP_FRACS  = [0.30, 0.30, 0.20]

SCALP_TPS       = [0.40]
SCALP_TP_FRACS  = [0.60]

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

# ==== Enhanced Golden Entry Settings ====
GOLDEN_ENTRY_SCORE = 7.0
GOLDEN_ENTRY_ADX   = 22.0
GOLDEN_REVERSAL_SCORE = 7.5
GOLDEN_ZONE_CONFIRMATION_BARS = 3

# ==== Enhanced Execution & Strategy Thresholds ====
ADX_TREND_MIN = 22
DI_SPREAD_TREND = 7
RSI_MA_LEN = 9
RSI_NEUTRAL_BAND = (40, 60)
RSI_TREND_PERSIST = 3

GZ_MIN_SCORE = 7.0
GZ_REQ_ADX = 22
GZ_REQ_VOL_MA = 20
ALLOW_GZ_ENTRY = True

# Enhanced Strategy Config
SCALP_TP1 = 0.40
SCALP_BE_AFTER = 0.30
SCALP_ATR_MULT = 1.6
TREND_TP1 = 1.20
TREND_BE_AFTER = 0.80
TREND_ATR_MULT = 1.8

MAX_TRADES_PER_HOUR = 6
COOLDOWN_SECS_AFTER_CLOSE = 60
ADX_GATE = 18

# ==== New: Footprint & SMC Settings ====
FOOTPRINT_WINDOW = 10
VOLUME_SPIKE_THRESHOLD = 2.0
LIQUIDITY_TRAP_DETECTION = True
DISPLACEMENT_THRESHOLD = 0.002  # 0.2%

# ==== VWAP Settings ====
VWAP_ENABLED = True
VWAP_SCALP_BAND_BPS = 8.0     # قرب من VWAP = سكالب
VWAP_TREND_BAND_BPS = 20.0    # بعيد عن VWAP = ترند قوي

# =================== BOX ENGINE + FVG + RF-B&S SETTINGS ===================
BOX_ENGINE_ENABLED      = True
BOX_LOOKBACK_BARS       = 120          # عدد الشموع اللي نحلل عليها الصناديق
BOX_MIN_TOUCHES         = 2            # أقل عدد لمسات علشان نعتبره Box محترم
BOX_MAX_HEIGHT_BPS      = 120          # أقصى ارتفاع للبوكس (bps) علشان ما يكونش واسع قوي
BOX_MIN_IMPULSE_BPS     = 80           # أقل حركة بعد البوكس علشان نعتبره قوي
BOX_VOL_RATIO_STRONG    = 1.4          # حجم قوي داخل/خارج البوكس

# Box Rejection / Stop-hunt Zones
BOX_REJECTION_WICK_PCT  = 0.35         # نسبة جسم الشمعة إلى الذيل عند رفض البوكس
BOX_RE_ENTRY_TOL_BPS    = 25           # هامش مسموح لرجوع السعر داخل البوكس
STOP_HUNT_ZONE_BPS      = 30           # منطقة ضرب استوبات فوق/تحت البوكس

# FVG Engine Config
FVG_LOOKBACK            = 80
FVG_MIN_SIZE_BPS        = 40           # أقل حجم فجوة علشان نعتبرها FVG
FVG_REAL_MIN_HOLD_BARS  = 6            # FVG حقيقي = ما يتقفلش بسرعة
FVG_FAKE_MAX_FILL_BARS  = 3            # FVG فيك = يتقفل بسرعة
FVG_STOP_HUNT_BPS       = 25           # منطقة stop hunt جوه/قرب الفجوة

# Box+FVG Profit Profile Config
BOX_SCALP_WEAK_TP1      = 0.45         # % تقريبية
BOX_SCALP_WEAK_TP2      = 0.0          # مفيش TP2 في الضعيف
BOX_SCALP_WEAK_TP3      = 0.0

BOX_MID_TP1             = 0.60
BOX_MID_TP2             = 1.10
BOX_MID_TP3             = 0.0

BOX_STRONG_TP1          = 0.80
BOX_STRONG_TP2          = 1.60
BOX_STRONG_TP3          = 2.40

BOX_WEAK_TRAIL_START    = 0.70
BOX_MID_TRAIL_START     = 1.20
BOX_STRONG_TRAIL_START  = 1.80

BOX_PARTIAL_1           = 0.35        # نسبة الكمية في TP1
BOX_PARTIAL_2           = 0.35        # نسبة الكمية في TP2
BOX_PARTIAL_3           = 0.30        # المتبقي في الترند

# RF B&S كعنصر مساعد
RF_BS_PERIOD            = 20
RF_BS_MULT              = 3.5
RF_BS_ENABLED           = True

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

# =================== BOX ENGINE CORE ===================
@dataclass
class SRBox:
    kind: str       # "demand" أو "supply"
    low: float
    high: float
    start_idx: int
    last_touch_idx: int
    touches: int
    breaks: int
    volume_sum: float = 0.0
    strength: float = 0.0

    def height_bps(self, price: float) -> float:
        if price <= 0:
            return 0.0
        return (self.high - self.low) / price * 10_000

def _detect_swings(df, window: int = 5):
    highs = df["high"].astype(float).values
    lows  = df["low"].astype(float).values
    swing_highs = []
    swing_lows  = []

    for i in range(window, len(df) - window):
        hi = highs[i]
        lo = lows[i]
        if all(hi >= highs[i - j] and hi > highs[i + j] for j in range(1, window + 1)):
            swing_highs.append(i)
        if all(lo <= lows[i - j] and lo < lows[i + j] for j in range(1, window + 1)):
            swing_lows.append(i)
    return swing_highs, swing_lows

def build_sr_boxes(df):
    """
    يبني مناطق عرض/طلب (Supply/Demand) بسيطة من الـ swings
    """
    if len(df) < 10:
        return []

    swing_highs, swing_lows = _detect_swings(df, window=5)
    boxes: List[SRBox] = []

    highs = df["high"].astype(float).values
    lows  = df["low"].astype(float).values
    vols  = df["volume"].astype(float).values

    # Demand boxes من swing lows
    for idx in swing_lows:
        low = lows[idx]
        high = lows[idx] + (highs[idx] - lows[idx]) * 0.3  # منطقة صغيرة فوق القاع
        
        # حساب اللمسات والحجم
        touches = 1
        volume_sum = vols[idx]
        for j in range(idx + 1, min(len(df), idx + 20)):
            if low <= lows[j] <= high or low <= highs[j] <= high:
                touches += 1
                volume_sum += vols[j]
        
        if touches >= BOX_MIN_TOUCHES:
            boxes.append(SRBox(
                kind="demand",
                low=low, high=high,
                start_idx=idx, last_touch_idx=idx,
                touches=touches, breaks=0, volume_sum=volume_sum
            ))

    # Supply boxes من swing highs
    for idx in swing_highs:
        high = highs[idx]
        low  = highs[idx] - (highs[idx] - lows[idx]) * 0.3
        
        # حساب اللمسات والحجم
        touches = 1
        volume_sum = vols[idx]
        for j in range(idx + 1, min(len(df), idx + 20)):
            if low <= lows[j] <= high or low <= highs[j] <= high:
                touches += 1
                volume_sum += vols[j]
        
        if touches >= BOX_MIN_TOUCHES:
            boxes.append(SRBox(
                kind="supply",
                low=low, high=high,
                start_idx=idx, last_touch_idx=idx,
                touches=touches, breaks=0, volume_sum=volume_sum
            ))

    # تنظيف البوكسات الواسعة أو الضعيفة
    price = float(df["close"].iloc[-1])
    pruned = []
    for b in boxes:
        if b.height_bps(price) <= BOX_MAX_HEIGHT_BPS:
            pruned.append(b)
    return pruned

def _box_volume_context(df, box: SRBox):
    """
    تحليل الفوليوم داخل البوكس: قوي / عادي / ضعيف
    """
    if box is None:
        return {"label": "none", "rejects": 0}

    box_df = df.iloc[box.start_idx:]
    vols = box_df["volume"].astype(float).values
    if len(vols) < 5:
        return {"label": "normal", "rejects": 0}

    avg_vol = vols.mean()
    last_vol = vols[-1] if len(vols) > 0 else avg_vol
    label = "normal"
    if last_vol > avg_vol * BOX_VOL_RATIO_STRONG:
        label = "strong"
    elif last_vol < avg_vol * 0.7:
        label = "weak"

    # عدد المرات اللي السعر لمس فيها حواف البوكس وارتد
    closes = box_df["close"].astype(float).values
    highs  = box_df["high"].astype(float).values
    lows   = box_df["low"].astype(float).values
    rejects = 0
    for i in range(len(closes)):
        h, l, c = highs[i], lows[i], closes[i]
        if box.kind == "supply":
            # لمس قريب من الحد العلوي ثم إغلاق في النص/تحت
            if h >= box.high * 0.999 and c < box.high:
                rejects += 1
        else:
            # لمس قريب من الحد السفلي ثم إغلاق في النص/فوق
            if l <= box.low * 1.001 and c > box.low:
                rejects += 1

    return {"label": label, "rejects": rejects}

def analyze_box_strength(box: SRBox, df) -> Dict:
    """يحسب قوة البوكس بناء على الرفض والحجم"""
    volume_ctx = _box_volume_context(df, box)
    rejection_count = volume_ctx["rejects"]
    
    strength = 0.0
    strength += rejection_count * 0.8
    strength += box.touches * 0.5
    if volume_ctx["label"] == "strong":
        strength += 2.0
    
    # تقييم ارتفاع البوكس
    price_ref = float(df["close"].iloc[-1])
    height_bps = box.height_bps(price_ref)
    if height_bps <= BOX_MAX_HEIGHT_BPS * 0.6:
        strength += 1.0  # بوكس ضيق = أقوى
    
    tier = "strong" if strength >= 5.0 else "mid" if strength >= 2.5 else "weak"
    
    return {
        "strength": strength,
        "tier": tier,
        "rejections": rejection_count,
        "volume_ctx": volume_ctx,
        "height_bps": height_bps
    }

def analyze_box_context(df, boxes):
    """
    يرجّع Box Context واحد relevant بالنسبة للسعر الحالي:
    - ctx: نوع السيناريو (none / strong_reversal_long / strong_reversal_short / retest / داخل البوكس...)
    - dir: "buy" أو "sell" أو "none"
    - tier: weak/mid/strong
    - score: رقم عام لقوة المنطقة
    - rr: تقدير Risk/Reward النسبي
    """
    if not boxes or len(df) < 5:
        return {"ctx": "none", "dir": "none", "tier": "weak", "score": 0.0, "rr": 0.0, "debug": "", "box": None}

    price = float(df["close"].iloc[-1])
    atr = float(df["high"].astype(float).rolling(14).max().iloc[-1] -
                df["low"].astype(float).rolling(14).min().iloc[-1]) / 14.0

    best = None
    best_score = -1e9
    debug = ""

    for b in boxes:
        # تحقق إذا كان السعر قريب من البوكس
        if not (b.low <= price <= b.high or 
                abs(price - b.high) <= 2*atr or 
                abs(price - b.low) <= 2*atr):
            continue

        strength_info = analyze_box_strength(b, df)
        strength = strength_info["strength"]
        tier = strength_info["tier"]
        vol_ctx = strength_info["volume_ctx"]
        rejects = strength_info["rejections"]

        # اتجاه منطقي
        if b.kind == "demand":
            rr = (b.high - price) / (price - b.low + 1e-9)  # reward/risk
            dir_ = "buy"
        else:
            rr = (price - b.low) / (b.high - price + 1e-9)  # reward/risk
            dir_ = "sell"

        # base score
        score = strength
        if rr >= 2.0: score += 2.0
        elif rr >= 1.5: score += 1.0

        ctx = "inside_box"
        if b.kind == "demand" and price >= b.high:
            ctx = "breakout_long"
        elif b.kind == "supply" and price <= b.low:
            ctx = "breakdown_short"
        elif rejects >= 2 and rr >= 1.5:
            ctx = "strong_reversal_long" if dir_ == "buy" else "strong_reversal_short"

        if score > best_score:
            best_score = score
            best = {
                "ctx": ctx,
                "dir": dir_,
                "tier": tier,
                "score": float(score),
                "rr": float(rr),
                "box": b,
                "box_vol": vol_ctx,
                "debug": f"kind={b.kind},height={strength_info['height_bps']:.1f}bp,rr={rr:.2f},vol={vol_ctx['label']},rejects={rejects},strength={strength:.1f}"
            }

    if best is None:
        return {"ctx": "none", "dir": "none", "tier": "weak", "score": 0.0, "rr": 0.0, "debug": "", "box": None}

    return best

# =================== FVG ENGINE ===================
def detect_fvg(df, lookback=FVG_LOOKBACK):
    """
    يكتشف FVG بسيط (ثلاث شموع) على آخر lookback شمعات.
    يرجّع قائمة gaps: {type, upper, lower, start_idx, filled_bars}
    """
    if len(df) < 5:
        return []

    sub = df.tail(lookback).reset_index(drop=True)
    gaps = []

    for i in range(2, len(sub)):
        h2 = float(sub["high"].iloc[i-1])
        l2 = float(sub["low"].iloc[i-1])

        h1 = float(sub["high"].iloc[i-2])
        l1 = float(sub["low"].iloc[i-2])

        h3 = float(sub["high"].iloc[i])
        l3 = float(sub["low"].iloc[i])

        # Bullish FVG: low3 > high1
        if l3 > h1:
            mid = (l3 + h1) / 2.0
            size_bps = (l3 - h1) / mid * 10000.0
            if size_bps >= FVG_MIN_SIZE_BPS:
                gaps.append({
                    "type": "bull",
                    "upper": l3,
                    "lower": h1,
                    "start_idx": i,
                    "size_bps": size_bps,
                    "filled_bars": 0
                })

        # Bearish FVG: high3 < low1
        if h3 < l1:
            mid = (h1 + l3) / 2.0 if (h1 + l3) else h1
            size_bps = (h1 - l3) / mid * 10000.0
            if size_bps >= FVG_MIN_SIZE_BPS:
                gaps.append({
                    "type": "bear",
                    "upper": h1,
                    "lower": l3,
                    "start_idx": i,
                    "size_bps": size_bps,
                    "filled_bars": 0
                })

    return gaps

def classify_fvg_context(df, gaps):
    """
    يحدد هل في FVG حقيقي / فيك / stop-hunt حوالين السعر الحالي.
    """
    if not gaps or len(df) == 0:
        return {"ctx": "none", "dir": "none", "zone": None, "real_fake": "neutral", "stop_hunt": False}

    sub = df.tail(FVG_LOOKBACK).reset_index(drop=True)
    price = float(sub["close"].iloc[-1])
    last_idx = len(sub) - 1

    best_gap = None
    best_dist = 1e9

    for g in gaps:
        upper = g["upper"]
        lower = g["lower"]
        mid = (upper + lower) / 2.0
        dist = abs(price - mid)
        if dist < best_dist:
            best_dist = dist
            best_gap = g

    if not best_gap:
        return {"ctx": "none", "dir": "none", "zone": None, "real_fake": "neutral", "stop_hunt": False}

    g = best_gap
    upper = g["upper"]
    lower = g["lower"]
    mid = (upper + lower) / 2.0
    bars_passed = max(0, last_idx - g["start_idx"])

    # Real vs Fake
    if bars_passed >= FVG_REAL_MIN_HOLD_BARS:
        real_fake = "real"
    elif bars_passed <= FVG_FAKE_MAX_FILL_BARS:
        real_fake = "fake"
    else:
        real_fake = "neutral"

    ctx = "outside"
    if lower <= price <= upper:
        ctx = "inside"
    elif price < lower:
        ctx = "below"
    elif price > upper:
        ctx = "above"

    # stop-hunt zone = قريب من حافة الفجوة
    stop_hunt = False
    edge_dist_bps = min(
        abs(price - lower) / mid * 10000.0 if mid else 0,
        abs(price - upper) / mid * 10000.0 if mid else 0
    )
    if edge_dist_bps <= FVG_STOP_HUNT_BPS:
        stop_hunt = True

    dir_ = "buy" if g["type"] == "bull" else "sell"

    return {
        "ctx": ctx,
        "dir": dir_,
        "zone": g,
        "real_fake": real_fake,
        "stop_hunt": stop_hunt,
        "edge_dist_bps": edge_dist_bps,
        "bars_passed": bars_passed,
        "price": price
    }

# =================== STOP HUNT DETECTION ===================
def detect_stop_hunt_zones(df, boxes):
    """
    يكتشف مناطق ضرب الاستوبات باستخدام ATR والبوکسات
    """
    if len(df) < 10:
        return {"ok": False, "side": None, "reason": "short_df"}

    current_price = float(df["close"].iloc[-1])
    atr = calculate_atr(df, 14)
    
    last = df.iloc[-1]
    o = float(last["open"])
    c = float(last["close"])
    h = float(last["high"])
    l = float(last["low"])
    body = abs(c - o)
    rng = max(h - l, 1e-9)

    up_wick = h - max(o, c)
    down_wick = min(o, c) - l

    # البحث عن استوبات فوق القمم
    stop_zones = []
    for box in boxes:
        if box.kind == "supply":
            # استوب فوق البوكس
            stop_level = box.high + atr * 0.3
            if abs(current_price - stop_level) / current_price * 10000 <= STOP_HUNT_ZONE_BPS:
                stop_zones.append({
                    'side': 'short',
                    'level': stop_level,
                    'type': 'liquidity_above_supply',
                    'box': box
                })
        else:  # demand box
            # استوب تحت البوكس
            stop_level = box.low - atr * 0.3
            if abs(current_price - stop_level) / current_price * 10000 <= STOP_HUNT_ZONE_BPS:
                stop_zones.append({
                    'side': 'long', 
                    'level': stop_level,
                    'type': 'liquidity_below_demand',
                    'box': box
                })

    return {
        "ok": len(stop_zones) > 0,
        "zones": stop_zones,
        "reason": f"found {len(stop_zones)} stop zones"
    }

def calculate_atr(df, period=14):
    """حساب ATR"""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return float(atr.iloc[-1]) if len(atr) > 0 else 0.0

# =================== RF B&S SIGNALS ===================
def compute_rf_bs(df, period=RF_BS_PERIOD, mult=RF_BS_MULT):
    """
    محاكاة مؤشر Range Filter - B&S Signals (الإصدار القديم)
    نفس منطق Pine قدر الإمكان على الداتا المتاحة
    """
    if len(df) < period + 3:
        return {
            "filt": None,
            "up": False,
            "down": False,
            "long": False,
            "short": False
        }

    close = df["close"].astype(float).values
    n = period
    qty = mult
    wper = (n * 2) - 1

    # حساب avrng = EMA(abs(x - x[1]), n)
    avrng = [0.0] * len(close)
    alpha_n = 2 / (n + 1)
    for i in range(1, len(close)):
        diff = abs(close[i] - close[i-1])
        avrng[i] = avrng[i-1] + alpha_n * (diff - avrng[i-1])

    # AC = ema(avrng, wper) * qty
    AC = [0.0] * len(close)
    alpha_w = 2 / (wper + 1)
    for i in range(1, len(close)):
        AC[i] = AC[i-1] + alpha_w * (avrng[i] - AC[i-1])
    rng_size = [x * qty for x in AC]

    # rng_filt logic
    rfilt = [close[0]] * len(close)
    for i in range(1, len(close)):
        r = rng_size[i]
        prev = rfilt[i-1]
        candidate = prev
        if close[i] - r > prev:
            candidate = close[i] - r
        if close[i] + r < prev:
            candidate = close[i] + r
        rfilt[i] = candidate

    # hi_band / lo_band
    hi_band = [rfilt[i] + rng_size[i] for i in range(len(close))]
    lo_band = [rfilt[i] - rng_size[i] for i in range(len(close))]

    # fdir
    fdir = [0] * len(close)
    for i in range(1, len(close)):
        if rfilt[i] > rfilt[i-1]:
            fdir[i] = 1
        elif rfilt[i] < rfilt[i-1]:
            fdir[i] = -1
        else:
            fdir[i] = fdir[i-1]

    upward   = [1 if d == 1 else 0 for d in fdir]
    downward = [1 if d == -1 else 0 for d in fdir]

    # trading conditions (آخر بار فقط)
    i = len(close) - 1
    longCond  = (
        (close[i] > rfilt[i] and close[i] > close[i-1] and upward[i] > 0) or
        (close[i] > rfilt[i] and close[i] < close[i-1] and upward[i] > 0)
    )
    shortCond = (
        (close[i] < rfilt[i] and close[i] < close[i-1] and downward[i] > 0) or
        (close[i] < rfilt[i] and close[i] > close[i-1] and downward[i] > 0)
    )

    # CondIni محلية بسيطة (ما فيش history كاملة، فا نشتغل بإشارة حالية فقط)
    long_sig  = longCond
    short_sig = shortCond

    return {
        "filt": rfilt[-1],
        "up": upward[-1] > 0,
        "down": downward[-1] > 0,
        "long": bool(long_sig),
        "short": bool(short_sig),
        "hi_band": hi_band[-1],
        "lo_band": lo_band[-1]
    }

# =================== BOX+FVG PROFIT PROFILE ===================
def classify_box_fvg_profit_profile(state, box_ctx, fvg_ctx, mode):
    """
    يحدد بروفايل جني الأرباح بناءً على:
    - قوة البوكس (strength + type + ctx)
    - نوع الـ FVG (real/fake + stop_hunt)
    - نمط الصفقة (mode: scalp / trend)
    يرجّع label + نسب TP + trail_start
    """

    label = "DEFAULT"
    tp1 = state.get("tp1_pct", BOX_SCALP_WEAK_TP1/100.0)
    tp2 = state.get("tp2_pct", 0.0)
    tp3 = state.get("tp3_pct", 0.0)
    trail_start = state.get("trail_activate_pct", BOX_WEAK_TRAIL_START/100.0)

    # لو مفيش Box/FVG → نسيب البروفايل الأصلي
    if not box_ctx or box_ctx.get("ctx") == "none":
        return {
            "label": label,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "trail_start": trail_start
        }

    strength = box_ctx.get("score", 0.0)
    box_dir  = box_ctx.get("dir", "none")
    fvg_ctx  = fvg_ctx or {"ctx": "none", "real_fake": "neutral", "stop_hunt": False}

    fvg_real_fake = fvg_ctx.get("real_fake", "neutral")
    stop_hunt     = fvg_ctx.get("stop_hunt", False)

    # ===== PROF1: SCALP_WEAK =====
    weak_box = strength < 2.5
    weak_fvg = (fvg_real_fake == "fake")

    if weak_box or weak_fvg:
        label = "SCALP_WEAK"
        tp1 = BOX_SCALP_WEAK_TP1/100.0
        tp2 = BOX_SCALP_WEAK_TP2/100.0
        tp3 = BOX_SCALP_WEAK_TP3/100.0
        trail_start = BOX_WEAK_TRAIL_START/100.0
        return {"label": label, "tp1": tp1, "tp2": tp2, "tp3": tp3, "trail_start": trail_start}

    # ===== PROF2: MID_SWING =====
    mid_box = 2.5 <= strength < 5.0
    neutral_fvg = (fvg_real_fake == "neutral" and not stop_hunt)

    if mid_box or neutral_fvg:
        label = "MID_SWING"
        tp1 = BOX_MID_TP1/100.0
        tp2 = BOX_MID_TP2/100.0
        tp3 = BOX_MID_TP3/100.0
        trail_start = BOX_MID_TRAIL_START/100.0
        return {"label": label, "tp1": tp1, "tp2": tp2, "tp3": tp3, "trail_start": trail_start}

    # ===== PROF3: STRONG_TREND / STOP-HUNT REVERSAL =====
    strong_box = strength >= 5.0
    strong_fvg = (fvg_real_fake == "real" and stop_hunt)

    if strong_box or strong_fvg or mode == "trend":
        label = "TREND_STRONG"
        tp1 = BOX_STRONG_TP1/100.0
        tp2 = BOX_STRONG_TP2/100.0
        tp3 = BOX_STRONG_TP3/100.0
        trail_start = BOX_STRONG_TRAIL_START/100.0

    return {"label": label, "tp1": tp1, "tp2": tp2, "tp3": tp3, "trail_start": trail_start}

# =================== ENHANCED CANDLES MODULE WITH SMC ===================
def _body(o,c): return abs(c-o)
def _rng(h,l):  return max(h-l, 1e-12)
def _upper_wick(h,o,c): return h - max(o,c)
def _lower_wick(l,o,c): return min(o,c) - l

def _is_doji(o,c,h,l,th=0.1):
    return _body(o,c) <= th * _rng(h,l)

def _engulfing(po,pc,o,c, min_ratio=1.05):
    bull = (c>o) and (pc<po) and _body(po,pc)>0 and _body(o,c)>=min_ratio*_body(po,pc) and (o<=pc and c>=po)
    bear = (c<o) and (pc>po) and _body(po,pc)>0 and _body(o,c)>=min_ratio*_body(po,pc) and (o>=pc and c<=po)
    return bull, bear

def _hammer_like(o,c,h,l, body_max=0.35, wick_ratio=2.0):
    rng, body = _rng(h,l), _body(o,c)
    lower, upper = _lower_wick(l,o,c), _upper_wick(h,o,c)
    hammer  = (body/rng<=body_max) and (lower>=wick_ratio*body) and (upper<=0.4*body)
    inv_ham = (body/rng<=body_max) and (upper>=wick_ratio*body) and (lower<=0.4*body)
    return hammer, inv_ham

def _shooting_star(o,c,h,l, body_max=0.35, wick_ratio=2.0):
    rng, body = _rng(h,l), _body(o,c)
    return (body/rng<=body_max) and (_upper_wick(h,o,c)>=wick_ratio*body) and (_lower_wick(l,o,c)<=0.4*body)

def _marubozu(o,c,h,l, min_body=0.9): return _body(o,c)/_rng(h,l) >= min_body
def _piercing(po,pc,o,c, min_pen=0.5): return (pc<po) and (c>o) and (c>(po - min_pen*(po-pc))) and (o<pc)
def _dark_cloud(po,pc,o,c, min_pen=0.5): return (pc>po) and (c<o) and (c<(po + min_pen*(pc-po))) and (o>pc)

def _tweezer(ph,pl,h,l, tol=0.15):
    top = abs(h-ph) <= tol*max(h,ph)
    bot = abs(l-pl) <= tol*max(l,pl)
    return top, bot

# ========= SMC CANDLES PATTERNS =========
def _smc_breakaway(o,c,h,l,po,pc,ph,pl):
    """Breakaway pattern - strong continuation"""
    bull_break = (c>o) and (po>pc) and (c>ph) and (l>pl)
    bear_break = (c<o) and (po<pc) and (c<pl) and (h<ph)
    return bull_break, bear_break

def _smc_absorption(po,pc,o,c,h,l,v,pv):
    """Absorption pattern - smart money accumulation/distribution"""
    bull_abs = (pc<po) and (c>o) and (c>po) and (v>pv*1.5)
    bear_abs = (pc>po) and (c<o) and (c<po) and (v>pv*1.5)
    return bull_abs, bear_abs

def _liquidity_grab(o,c,h,l,po,pc,pl,ph):
    """Liquidity grab pattern - stop hunting"""
    bull_grab = (c<o) and (l<pl) and (c>pc)  # false breakdown
    bear_grab = (c>o) and (h>ph) and (c<pc)  # false breakout
    return bull_grab, bear_grab

def compute_enhanced_candles(df):
    """
    إرجاع: إشارات شراء/بيع معقّدة + أنماط SMC + فخاخ سيولة
    """
    if len(df) < 6:
        return {"buy":False,"sell":False,"score_buy":0.0,"score_sell":0.0,
                "wick_up_big":False,"wick_dn_big":False,"doji":False,
                "pattern":None, "smc_pattern":None, "liquidity_trap":False}

    # Current and previous candles
    o1,h1,l1,c1,v1 = float(df["open"].iloc[-2]), float(df["high"].iloc[-2]), float(df["low"].iloc[-2]), float(df["close"].iloc[-2]), float(df["volume"].iloc[-2])
    o0,h0,l0,c0,v0 = float(df["open"].iloc[-3]), float(df["high"].iloc[-3]), float(df["low"].iloc[-3]), float(df["close"].iloc[-3]), float(df["volume"].iloc[-3])
    o2,h2,l2,c2,v2 = float(df["open"].iloc[-4]), float(df["high"].iloc[-4]), float(df["low"].iloc[-4]), float(df["close"].iloc[-4]), float(df["volume"].iloc[-4])

    strength_b = strength_s = 0.0
    tags = []
    smc_tags = []
    liquidity_trap = False

    # Basic candle patterns
    bull_eng, bear_eng = _engulfing(o0,c0,o1,c1)
    if bull_eng: strength_b += 2.0; tags.append("bull_engulf")
    if bear_eng: strength_s += 2.0; tags.append("bear_engulf")

    ham, inv = _hammer_like(o1,c1,h1,l1)
    if ham: strength_b += 1.5; tags.append("hammer")
    if inv: strength_s += 1.5; tags.append("inverted_hammer")

    if _shooting_star(o1,c1,h1,l1): strength_s += 1.5; tags.append("shooting_star")
    if _piercing(o0,c0,o1,c1):      strength_b += 1.2; tags.append("piercing")
    if _dark_cloud(o0,c0,o1,c1):    strength_s += 1.2; tags.append("dark_cloud")

    is_doji = _is_doji(o1,c1,h1,l1)
    if is_doji: tags.append("doji")

    tw_top, tw_bot = _tweezer(h0,l0,h1,l1)
    if tw_bot: strength_b += 1.0; tags.append("tweezer_bottom")
    if tw_top: strength_s += 1.0; tags.append("tweezer_top")

    if _marubozu(o1,c1,h1,l1):
        if c1>o1: strength_b += 1.0; tags.append("marubozu_bull")
        else:     strength_s += 1.0; tags.append("marubozu_bear")

    # SMC Patterns
    bull_break, bear_break = _smc_breakaway(o1,c1,h1,l1,o0,c0,h0,l0)
    if bull_break: strength_b += 2.5; smc_tags.append("breakaway_bull")
    if bear_break: strength_s += 2.5; smc_tags.append("breakaway_bear")

    bull_abs, bear_abs = _smc_absorption(o0,c0,o1,c1,h1,l1,v1,v0)
    if bull_abs: strength_b += 3.0; smc_tags.append("absorption_bull")
    if bear_abs: strength_s += 3.0; smc_tags.append("absorption_bear")

    bull_grab, bear_grab = _liquidity_grab(o1,c1,h1,l1,o0,c0,l0,h0)
    if bull_grab: 
        strength_b += 2.0; 
        smc_tags.append("liquidity_grab_bull")
        liquidity_trap = True
    if bear_grab: 
        strength_s += 2.0; 
        smc_tags.append("liquidity_grab_bear")
        liquidity_trap = True

    # فتائل كبيرة = إرهاق
    rng1 = _rng(h1,l1); up = _upper_wick(h1,o1,c1); dn = _lower_wick(l1,o1,c1)
    wick_up_big = (up >= 1.2*_body(o1,c1)) and (up >= 0.4*rng1)
    wick_dn_big = (dn >= 1.2*_body(o1,c1)) and (dn >= 0.4*rng1)

    if is_doji:  # تخفيف ثقة
        strength_b *= 0.8; strength_s *= 0.8

    # دمج أنماط SMC
    all_tags = tags + [f"SMC:{t}" for t in smc_tags]
    pattern_str = ",".join(all_tags) if all_tags else None

    return {
        "buy": strength_b>0, "sell": strength_s>0,
        "score_buy": round(strength_b,2), "score_sell": round(strength_s,2),
        "wick_up_big": bool(wick_up_big), "wick_dn_big": bool(wick_dn_big),
        "doji": bool(is_doji), "pattern": pattern_str,
        "smc_pattern": ",".join(smc_tags) if smc_tags else None,
        "liquidity_trap": liquidity_trap
    }

# =================== FOOTPRINT & VOLUME ANALYSIS ===================
def compute_footprint_metrics(df):
    """
    تحليل تدفق الأوامر والقدم (Footprint)
    """
    if len(df) < FOOTPRINT_WINDOW + 2:
        return {"ok": False, "why": "short_df"}
    
    try:
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        
        # حجم عند السعر (Price Volume)
        price_changes = close.diff()
        up_volume = volume.where(price_changes > 0, 0)
        down_volume = volume.where(price_changes < 0, 0)
        
        # delta = حجم الشراء - حجم البيع
        delta = up_volume - down_volume
        cumulative_delta = delta.cumsum()
        
        # حجم غير عادي
        vol_ma = volume.rolling(FOOTPRINT_WINDOW).mean()
        volume_spike = (volume / vol_ma) > VOLUME_SPIKE_THRESHOLD
        
        # مناطق الامتصاص
        absorption_bull = (delta > 0) & (close < close.shift(1))  # شراء على هبوط
        absorption_bear = (delta < 0) & (close > close.shift(1))  # بيع على صعود
        
        return {
            "ok": True,
            "delta": float(delta.iloc[-1]),
            "cumulative_delta": float(cumulative_delta.iloc[-1]),
            "volume_spike": bool(volume_spike.iloc[-1]),
            "absorption_bull": bool(absorption_bull.iloc[-1]),
            "absorption_bear": bool(absorption_bear.iloc[-1]),
            "delta_trend": "bull" if cumulative_delta.iloc[-1] > cumulative_delta.iloc[-2] else "bear"
        }
    except Exception as e:
        return {"ok": False, "why": str(e)}

# =================== LIQUIDITY TRAP DETECTION ===================
def detect_liquidity_traps(df, current_price):
    """
    كشف فخاخ السيولة ونقاط الوقف (Liquidity Pools)
    """
    if len(df) < 20:
        return {"ok": False, "traps": []}
    
    try:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        
        # نقاط السيولة (Highs/Lows السابقة)
        recent_highs = high.rolling(10).max().dropna()
        recent_lows = low.rolling(10).min().dropna()
        
        traps = []
        
        # فخاخ السيولة فوق (نقاط وقف الشراء)
        for level in recent_highs.unique():
            if abs(current_price - level) / current_price <= DISPLACEMENT_THRESHOLD:
                traps.append({"type": "stop_hunt_bull", "level": level, "distance_pct": abs(current_price - level) / current_price * 100})
        
        # فخاخ السيولة تحت (نقاط وقف البيع)
        for level in recent_lows.unique():
            if abs(current_price - level) / current_price <= DISPLACEMENT_THRESHOLD:
                traps.append({"type": "stop_hunt_bear", "level": level, "distance_pct": abs(current_price - level) / current_price * 100})
        
        return {"ok": True, "traps": traps}
    
    except Exception as e:
        return {"ok": False, "why": str(e), "traps": []}

# =================== EXECUTION VERIFICATION ===================
def verify_execution_environment():
    """التحقق من بيئة التنفيذ عند الإقلاع"""
    print(f"⚙️ EXECUTION ENVIRONMENT", flush=True)
    print(f"🔧 EXECUTE_ORDERS: {EXECUTE_ORDERS} | SHADOW_MODE: {SHADOW_MODE_DASHBOARD} | DRY_RUN: {DRY_RUN}", flush=True)
    print(f"🎯 GOLDEN ENTRY PRO: score={GOLDEN_ENTRY_SCORE} | ADX={GOLDEN_ENTRY_ADX}", flush=True)
    print(f"📈 ENHANCED CANDLES: SMC Patterns + Liquidity Traps", flush=True)
    print(f"👣 FOOTPRINT ANALYSIS: Volume spikes + Absorption", flush=True)
    print(f"📊 VWAP STRATEGY: SCALP(near {VWAP_SCALP_BAND_BPS}bps) | TREND(far {VWAP_TREND_BAND_BPS}bps)", flush=True)
    print(f"📦 BOX ENGINE: Supply/Demand Zones + Strength Scoring", flush=True)
    print(f"🕳️ FVG DETECTION: Real vs Fake Gaps + Stop Hunt Zones", flush=True)
    print(f"⚡ RF B&S: Scalp Assistant Signals", flush=True)
    
    if not EXECUTE_ORDERS:
        print("🟡 WARNING: EXECUTE_ORDERS=False - البوت في وضع التحليل فقط!", flush=True)
    if DRY_RUN:
        print("🟡 WARNING: DRY_RUN=True - البوت في وضع المحاكاة!", flush=True)

# =================== ENHANCED INDICATORS ===================
def sma(series, n: int):
    return series.rolling(n, min_periods=1).mean()

def compute_rsi(close, n: int = 14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(span=n, adjust=False).mean()
    roll_down = down.ewm(span=n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, 1e-12)
    rsi = 100 - (100/(1+rs))
    return rsi.fillna(50)

def rsi_ma_context(df):
    if len(df) < max(RSI_MA_LEN, 14):
        return {"rsi": 50, "rsi_ma": 50, "cross": "none", "trendZ": "none", "in_chop": True}
    
    rsi = compute_rsi(df['close'].astype(float), 14)
    rsi_ma = sma(rsi, RSI_MA_LEN)
    
    cross = "none"
    if len(rsi) >= 2:
        if (rsi.iloc[-2] <= rsi_ma.iloc[-2]) and (rsi.iloc[-1] > rsi_ma.iloc[-1]):
            cross = "bull"
        elif (rsi.iloc[-2] >= rsi_ma.iloc[-2]) and (rsi.iloc[-1] < rsi_ma.iloc[-1]):
            cross = "bear"
    
    above = (rsi > rsi_ma)
    below = (rsi < rsi_ma)
    persist_bull = above.tail(RSI_TREND_PERSIST).all() if len(above) >= RSI_TREND_PERSIST else False
    persist_bear = below.tail(RSI_TREND_PERSIST).all() if len(below) >= RSI_TREND_PERSIST else False
    
    current_rsi = float(rsi.iloc[-1])
    in_chop = RSI_NEUTRAL_BAND[0] <= current_rsi <= RSI_NEUTRAL_BAND[1]
    
    return {
        "rsi": current_rsi,
        "rsi_ma": float(rsi_ma.iloc[-1]),
        "cross": cross,
        "trendZ": "bull" if persist_bull else ("bear" if persist_bear else "none"),
        "in_chop": in_chop
    }

# =================== GOLDEN ZONE PRO ANALYSIS ===================
def golden_zone_pro_analysis(df, current_price):
    """
    تحليل متقدم للمناطق الذهبية مع تأكيدات متعددة
    """
    if len(df) < 30:
        return {"ok": False, "score": 0.0, "zone": None, "reasons": ["short_df"], "confirmed": False}
    
    try:
        h = df['high'].astype(float)
        l = df['low'].astype(float)
        c = df['close'].astype(float)
        v = df['volume'].astype(float)
        
        # مستويات فيبوناتشي المتقدمة
        swing_hi = h.rolling(15).max().iloc[-1]
        swing_lo = l.rolling(15).min().iloc[-1]
        
        if swing_hi <= swing_lo:
            return {"ok": False, "score": 0.0, "zone": None, "reasons": ["flat_market"], "confirmed": False}
        
        # مستويات فيبوناتشي موسعة
        fib_levels = {
            'f0382': swing_lo + 0.382 * (swing_hi - swing_lo),
            'f0500': swing_lo + 0.500 * (swing_hi - swing_lo),
            'f0618': swing_lo + 0.618 * (swing_hi - swing_lo),
            'f0786': swing_lo + 0.786 * (swing_hi - swing_lo),
            'f0886': swing_lo + 0.886 * (swing_hi - swing_lo)
        }
        
        last_close = float(c.iloc[-1])
        
        # نحسب قرب السعر من القاع والقمة
        dist_to_low  = abs(last_close - swing_lo)
        dist_to_high = abs(last_close - swing_hi)

        # تحليل الحجم
        vol_ma20 = v.rolling(20).mean().iloc[-1]
        vol_ok = float(v.iloc[-1]) >= vol_ma20 * 0.8
        volume_spike = float(v.iloc[-1]) > vol_ma20 * 1.5
        
        # تحليل الشمعة الحالية
        current_open = float(df['open'].iloc[-1])
        current_high = float(h.iloc[-1])
        current_low = float(l.iloc[-1])
        
        body = abs(last_close - current_open)
        wick_up = current_high - max(last_close, current_open)
        wick_down = min(last_close, current_open) - current_low
        
        bull_candle = (wick_down > (body * 1.2) and last_close > current_open) or (body > 0 and last_close > current_open and wick_down > wick_up)
        bear_candle = (wick_up > (body * 1.2) and last_close < current_open) or (body > 0 and last_close < current_open and wick_up > wick_down)
        
        # Footprint analysis
        footprint = compute_footprint_metrics(df)
        
        score = 0.0
        zone_type = None
        reasons = []
        confirmed = False
        
        # المنطقة الذهبية السفلية (شراء) — السعر داخل 0.618–0.786 وأقرب للقاع
        if fib_levels['f0618'] <= last_close <= fib_levels['f0786'] and dist_to_low <= dist_to_high:
            score += 3.0
            reasons.append("منطقة_ذهبية_سفلية")
            
            if bull_candle:
                score += 2.0
                reasons.append("شمعة_صاعدة")
            
            if volume_spike:
                score += 1.5
                reasons.append("حجم_مرتفع")
            
            if footprint.get('ok') and footprint.get('absorption_bull'):
                score += 2.0
                reasons.append("امتصاص_شرائي")
            
            # تأكيد من الشموع السابقة داخل نفس المنطقة
            confirmation_bars = 0
            for i in range(2, min(6, len(df))):
                prev_close = float(df['close'].iloc[-i])
                if fib_levels['f0618'] <= prev_close <= fib_levels['f0786']:
                    confirmation_bars += 1
            
            if confirmation_bars >= 2:
                score += 1.5
                reasons.append(f"تأكيد_{confirmation_bars}_شمعة")
                confirmed = True
            
            if score >= GOLDEN_ENTRY_SCORE:
                zone_type = "golden_bottom"
        
        # المنطقة الذهبية العلوية (بيع) — السعر داخل 0.618–0.786 وأقرب للقمة
        elif fib_levels['f0618'] <= last_close <= fib_levels['f0786'] and dist_to_high < dist_to_low:
            score += 3.0
            reasons.append("منطقة_ذهبية_علوية")
            
            if bear_candle:
                score += 2.0
                reasons.append("شمعة_هابطة")
            
            if volume_spike:
                score += 1.5
                reasons.append("حجم_مرتفع")
            
            if footprint.get('ok') and footprint.get('absorption_bear'):
                score += 2.0
                reasons.append("امتصاص_بيعي")
            
            # تأكيد من الشموع السابقة داخل نفس المنطقة
            confirmation_bars = 0
            for i in range(2, min(6, len(df))):
                prev_close = float(df['close'].iloc[-i])
                if fib_levels['f0618'] <= prev_close <= fib_levels['f0786']:
                    confirmation_bars += 1
            
            if confirmation_bars >= 2:
                score += 1.5
                reasons.append(f"تأكيد_{confirmation_bars}_شمعة")
                confirmed = True
            
            if score >= GOLDEN_ENTRY_SCORE:
                zone_type = "golden_top"
        
        ok = zone_type is not None and ALLOW_GZ_ENTRY
        return {
            "ok": ok,
            "score": score,
            "zone": {"type": zone_type, "levels": fib_levels} if zone_type else None,
            "reasons": reasons,
            "confirmed": confirmed
        }
        
    except Exception as e:
        return {"ok": False, "score": 0.0, "zone": None, "reasons": [f"error: {e}"], "confirmed": False}

def decide_strategy_mode_enhanced(df, adx=None, di_plus=None, di_minus=None, rsi_ctx=None, footprint=None):
    """تحديد نمط التداول المحسن: SCALP أم TREND مع VWAP + Footprint"""
    ind = compute_indicators(df)

    if adx is None or di_plus is None or di_minus is None:
        adx = ind.get('adx', 0)
        di_plus = ind.get('plus_di', 0)
        di_minus = ind.get('minus_di', 0)

    if rsi_ctx is None:
        rsi_ctx = rsi_ma_context(df)

    if footprint is None:
        footprint = compute_footprint_metrics(df)

    di_spread = abs(di_plus - di_minus)

    # VWAP context
    vwap = ind.get("vwap")
    price = float(df["close"].iloc[-1])
    if vwap and VWAP_ENABLED:
        vwap_diff_bps = abs(price - vwap) / vwap * 10000.0
        near_vwap = vwap_diff_bps <= VWAP_SCALP_BAND_BPS
        far_from_vwap = vwap_diff_bps >= VWAP_TREND_BAND_BPS
    else:
        near_vwap = False
        far_from_vwap = False

    # ترند قوي
    strong_trend = (
        (adx >= ADX_TREND_MIN and di_spread >= DI_SPREAD_TREND) or
        (rsi_ctx["trendZ"] in ("bull", "bear") and not rsi_ctx["in_chop"])
    )

    # Footprint confirmation
    footprint_confirmation = False
    if footprint.get('ok'):
        trend_dir = 'bull' if di_plus > di_minus else 'bear'
        if strong_trend and footprint.get('delta_trend') == trend_dir:
            footprint_confirmation = True

    # منطق اختيار النمط:
    # - ترند: ترند قوي + بعيد عن VWAP
    # - سكالب: عادي أو قرب من VWAP
    if strong_trend and footprint_confirmation and (far_from_vwap or not VWAP_ENABLED):
        mode = "trend"
        why = "strong_trend+footprint+far_from_vwap"
    else:
        mode = "scalp"
        why_parts = ["scalp_default"]
        if VWAP_ENABLED and near_vwap:
            why_parts.append("near_vwap")
        why = "+".join(why_parts)

    return {"mode": mode, "why": why, "footprint_ok": footprint_confirmation}

# =================== SMART PROFIT AI ===================
def smart_profit_ai_decision(state, df, ind, mode, side, entry_price, current_price, info=None):
    """
    ذكاء اصطناعي لجني الأرباح بشكل ذكي حسب قوة الصفقة
    """
    info = info or {}
    box_ctx = info.get("box_ctx")
    fvg_ctx = info.get("fvg_ctx")
    
    pnl_pct = (current_price - entry_price) / entry_price * 100 * (1 if side == "long" else -1)
    
    # استخدام Box+FVG Profile إذا كان متاحًا
    if box_ctx and fvg_ctx:
        profile = classify_box_fvg_profit_profile(state, box_ctx, fvg_ctx, mode)
        if profile["label"] != "DEFAULT":
            tp1_pct = profile["tp1"] * 100  # تحويل لـ percentage
            tp2_pct = profile["tp2"] * 100
            tp3_pct = profile["tp3"] * 100
            
            achieved_targets = state.get("profit_targets_achieved", 0)
            
            if achieved_targets == 0 and pnl_pct >= tp1_pct:
                return {
                    "action": "take_profit", 
                    "target": 1, 
                    "target_pct": tp1_pct,
                    "fraction": BOX_PARTIAL_1,
                    "reason": f"BOX+FVG TP1 ({tp1_pct:.2f}%)"
                }
            elif achieved_targets == 1 and pnl_pct >= tp2_pct:
                return {
                    "action": "take_profit", 
                    "target": 2, 
                    "target_pct": tp2_pct,
                    "fraction": BOX_PARTIAL_2, 
                    "reason": f"BOX+FVG TP2 ({tp2_pct:.2f}%)"
                }
            elif achieved_targets == 2 and pnl_pct >= tp3_pct:
                return {
                    "action": "take_profit", 
                    "target": 3, 
                    "target_pct": tp3_pct,
                    "fraction": BOX_PARTIAL_3,
                    "reason": f"BOX+FVG TP3 ({tp3_pct:.2f}%)"
                }
    
    # المنطق الأصلي إذا لم يكن هناك Box+FVG profile
    if mode == "scalp":
        # جني الأرباح على مرحلة واحدة للسكالب
        tp_levels = SCALP_TPS
        tp_fractions = SCALP_TP_FRACS
        max_tp = max(tp_levels)
    else:
        # جني الأرباح على 3 مراحل للترند
        tp_levels = TREND_TPS
        tp_fractions = TREND_TP_FRACS
        max_tp = max(tp_levels)
    
    achieved_targets = state.get("profit_targets_achieved", 0)
    next_target_index = achieved_targets
    
    if next_target_index >= len(tp_levels):
        return {"action": "hold", "target": None, "reason": "كل الأهداف محققة"}
    
    next_target_pct = tp_levels[next_target_index]
    next_target_fraction = tp_fractions[next_target_index]
    
    # تحسين القرار بناء على قوة الإشارة
    signal_strength = calculate_signal_strength(df, ind, side)
    
    # تعديل الأهداف حسب قوة الإشارة
    if signal_strength >= 8.0:  # إشارة قوية جداً
        next_target_pct *= 1.2  # زيادة الهدف 20%
    elif signal_strength >= 6.0:  # إشارة قوية
        next_target_pct *= 1.1  # زيادة الهدف 10%
    elif signal_strength < 4.0:  # إشارة ضعيفة
        next_target_pct *= 0.8  # تقليل الهدف 20%
    
    if pnl_pct >= next_target_pct:
        return {
            "action": "take_profit",
            "target": next_target_index + 1,
            "target_pct": next_target_pct,
            "fraction": next_target_fraction,
            "reason": f"تحقيق الهدف {next_target_index + 1} ({next_target_pct:.2f}%)"
        }
    
    return {"action": "hold", "target": next_target_index + 1, "reason": "لم يحقق الهدف بعد"}

def calculate_signal_strength(df, ind, side):
    """حساب قوة الإشارة للتداول"""
    strength = 0.0
    
    # قوة ADX
    adx = ind.get('adx', 0)
    if adx > 25:
        strength += 3.0
    elif adx > 20:
        strength += 2.0
    elif adx > 15:
        strength += 1.0
    
    # قوة RSI
    rsi = ind.get('rsi', 50)
    if (side == "long" and rsi < 70) or (side == "short" and rsi > 30):
        strength += 2.0
    
    # قوة DI Spread
    di_spread = ind.get('di_spread', 0)
    if di_spread > 8:
        strength += 2.0
    elif di_spread > 5:
        strength += 1.0
    
    # Footprint confirmation
    footprint = ind.get('footprint', {})
    if footprint.get('ok'):
        if (side == "long" and footprint.get('delta_trend') == 'bull') or \
           (side == "short" and footprint.get('delta_trend') == 'bear'):
            strength += 2.0
    
    # Golden Zone confirmation
    gz = ind.get('gz', {})
    if gz.get('ok') and gz.get('confirmed'):
        strength += 3.0
    elif gz.get('ok'):
        strength += 1.5
    
    return min(10.0, strength)

# =================== ENHANCED COUNCIL VOTING ===================
def council_votes_pro_enhanced(df):
    """مجلس تصويت محسّن مع Footprint + SMC + Golden Zone Pro + VWAP + Box Engine"""
    try:
        ind = compute_indicators(df)
        rsi_ctx = rsi_ma_context(df)
        current_price = float(df['close'].iloc[-1])
        gz = golden_zone_pro_analysis(df, current_price)
        
        # Footprint analysis
        footprint = compute_footprint_metrics(df)
        
        # Enhanced candles with SMC
        cd = compute_enhanced_candles(df)
        
        # Liquidity traps
        liquidity_traps = detect_liquidity_traps(df, current_price)

        # Box Engine Analysis
        boxes = build_sr_boxes(df) if BOX_ENGINE_ENABLED else []
        box_ctx = analyze_box_context(df, boxes) if boxes else {"ctx": "none", "dir": "none", "score": 0.0}
        
        # FVG Analysis
        fvg_gaps = detect_fvg(df) if BOX_ENGINE_ENABLED else []
        fvg_ctx = classify_fvg_context(df, fvg_gaps) if fvg_gaps else {"ctx": "none", "dir": "none", "real_fake": "neutral"}
        
        # RF B&S Signals
        rf_bs = compute_rf_bs(df) if RF_BS_ENABLED else {"long": False, "short": False}

        votes_b = 0; votes_s = 0
        score_b = 0.0; score_s = 0.0
        logs = []

        adx = ind.get('adx', 0)
        plus_di = ind.get('plus_di', 0)
        minus_di = ind.get('minus_di', 0)
        di_spread = ind.get('di_spread', abs(plus_di - minus_di))

        # ==== VWAP CONTEXT ====
        vwap = ind.get("vwap")
        vwap_diff_bps = None
        near_vwap = False
        far_from_vwap = False
        if VWAP_ENABLED and vwap:
            vwap_diff_bps = abs(current_price - vwap) / vwap * 10000.0
            near_vwap = vwap_diff_bps <= VWAP_SCALP_BAND_BPS
            far_from_vwap = vwap_diff_bps >= VWAP_TREND_BAND_BPS
            above_vwap = current_price > vwap
            logs.append(f"VWAP ctx: px={current_price:.6f} vwap={vwap:.6f} Δ={vwap_diff_bps:.1f}bps")

        # ==== BOX ENGINE BOOST ====
        if box_ctx["ctx"] != "none":
            if box_ctx["dir"] == "buy":
                votes_b += 2
                score_b += box_ctx["score"] * 0.3
                logs.append(f"📦 BOX BUY boost: score={box_ctx['score']:.1f} tier={box_ctx['tier']}")
            elif box_ctx["dir"] == "sell":
                votes_s += 2
                score_s += box_ctx["score"] * 0.3
                logs.append(f"📦 BOX SELL boost: score={box_ctx['score']:.1f} tier={box_ctx['tier']}")

        # ==== FVG CONTEXT ====
        if fvg_ctx["ctx"] != "none":
            if fvg_ctx["real_fake"] == "real":
                if fvg_ctx["dir"] == "buy":
                    votes_b += 2
                    score_b += 1.5
                    logs.append("🕳️ FVG REAL BULLISH")
                else:
                    votes_s += 2
                    score_s += 1.5
                    logs.append("🕳️ FVG REAL BEARISH")
            elif fvg_ctx["real_fake"] == "fake":
                # FVG مزيف = اتجاه معاكس
                if fvg_ctx["dir"] == "buy":
                    votes_s += 2
                    score_s += 1.5
                    logs.append("🕳️ FVG FAKE BULLISH → SELL")
                else:
                    votes_b += 2
                    score_b += 1.5
                    logs.append("🕳️ FVG FAKE BEARISH → BUY")

        # ==== RF B&S SIGNALS ====
        if rf_bs.get("long"):
            votes_b += 1
            score_b += 0.8
            logs.append("⚡ RF B&S BUY")
        if rf_bs.get("short"):
            votes_s += 1
            score_s += 0.8
            logs.append("⚡ RF B&S SELL")

        # --- تصويت VWAP للسكالب (قرب من VWAP) ---
        if VWAP_ENABLED and near_vwap and cd:
            if cd.get("buy"):
                votes_b += 2; score_b += 1.5
                logs.append("⚡ VWAP SCALP BUY zone")
            if cd.get("sell"):
                votes_s += 2; score_s += 1.5
                logs.append("⚡ VWAP SCALP SELL zone")

        # --- بوست للترند بعيد عن VWAP ---
        if VWAP_ENABLED and far_from_vwap and adx >= ADX_TREND_MIN:
            if plus_di > minus_di and current_price > (vwap or current_price):
                votes_b += 1; score_b += 1.0
                logs.append("📈 VWAP TREND BOOST BUY")
            elif minus_di > plus_di and current_price < (vwap or current_price):
                votes_s += 1; score_s += 1.0
                logs.append("📉 VWAP TREND BOOST SELL")

        # --- ترند ADX/DI مع Footprint تأكيد
        if adx > ADX_TREND_MIN:
            if plus_di > minus_di and di_spread > DI_SPREAD_TREND:
                if footprint.get('ok') and footprint.get('delta_trend') == 'bull':
                    votes_b += 3; score_b += 2.0; logs.append("📈 ترند صاعد قوي + Footprint تأكيد")
                else:
                    votes_b += 2; score_b += 1.5; logs.append("📈 ترند صاعد قوي")
            elif minus_di > plus_di and di_spread > DI_SPREAD_TREND:
                if footprint.get('ok') and footprint.get('delta_trend') == 'bear':
                    votes_s += 3; score_s += 2.0; logs.append("📉 ترند هابط قوي + Footprint تأكيد")
                else:
                    votes_s += 2; score_s += 1.5; logs.append("📉 ترند هابط قوي")

        # --- RSI-MA cross / Trend-Z
        if rsi_ctx["cross"] == "bull" and rsi_ctx["rsi"] < 70:
            votes_b += 2; score_b += 1.0; logs.append("🟢 RSI-MA إيجابي")
        elif rsi_ctx["cross"] == "bear" and rsi_ctx["rsi"] > 30:
            votes_s += 2; score_s += 1.0; logs.append("🔴 RSI-MA سلبي")

        if rsi_ctx["trendZ"] == "bull":
            votes_b += 3; score_b += 1.5; logs.append("🚀 RSI ترند صاعد مستمر")
        elif rsi_ctx["trendZ"] == "bear":
            votes_s += 3; score_s += 1.5; logs.append("💥 RSI ترند هابط مستمر")

        # --- Golden Zones Pro
        if gz and gz.get("ok"):
            if gz['zone']['type'] == 'golden_bottom':
                votes_b += 4 if gz['confirmed'] else 3
                score_b += 2.0 if gz['confirmed'] else 1.5
                conf_text = "مؤكد" if gz['confirmed'] else "محتمل"
                logs.append(f"🏆 قاع ذهبي {conf_text} (قوة: {gz['score']:.1f})")
            elif gz['zone']['type'] == 'golden_top':
                votes_s += 4 if gz['confirmed'] else 3
                score_s += 2.0 if gz['confirmed'] else 1.5
                conf_text = "مؤكد" if gz['confirmed'] else "محتمل"
                logs.append(f"🏆 قمة ذهبية {conf_text} (قوة: {gz['score']:.1f})")

        # --- Footprint Boost
        if footprint.get('ok'):
            if footprint.get('absorption_bull'):
                votes_b += 2; score_b += 1.5; logs.append("👣 Footprint امتصاص شرائي")
            if footprint.get('absorption_bear'):
                votes_s += 2; score_s += 1.5; logs.append("👣 Footprint امتصاص بيعي")
            
            if footprint.get('volume_spike'):
                if footprint.get('delta') > 0:
                    votes_b += 1; score_b += 1.0; logs.append("📊 حجم شرائي عالي")
                else:
                    votes_s += 1; score_s += 1.0; logs.append("📊 حجم بيعي عالي")

        # --- SMC Candles
        if cd["score_buy"]>0:
            score_b += min(3.0, cd["score_buy"])
            logs.append(f"🕯️ SMC BUY ({cd['smc_pattern']}) +{cd['score_buy']:.1f}")
        if cd["score_sell"]>0:
            score_s += min(3.0, cd["score_sell"])
            logs.append(f"🕯️ SMC SELL ({cd['smc_pattern']}) +{cd['score_sell']:.1f}")

        # --- Liquidity Trap Awareness
        if liquidity_traps.get('ok') and liquidity_traps.get('traps'):
            for trap in liquidity_traps['traps']:
                if trap['type'] == 'stop_hunt_bull' and score_b > score_s:
                    score_b *= 1.1  # تعزيز الثقة في فخ الصعود
                    logs.append(f"🪤 فخ سيولة صاعد قريب ({trap['distance_pct']:.2f}%)")
                elif trap['type'] == 'stop_hunt_bear' and score_s > score_s:
                    score_s *= 1.1  # تعزيز الثقة في فخ الهبوط
                    logs.append(f"🪤 فخ سيولة هابط قريب ({trap['distance_pct']:.2f}%)")

        # تخفيف النطاق المحايد
        if rsi_ctx["in_chop"]:
            score_b *= 0.7; score_s *= 0.7; logs.append("⚖️ RSI محايد — تخفيض ثقة")

        # حارس ADX عام
        if adx < ADX_GATE:
            score_b *= 0.8; score_s *= 0.8; logs.append(f"🛡️ ADX Gate ({adx:.1f} < {ADX_GATE})")

        # منع الفلات والرينج
        if di_spread < 3 and adx < 15:
            score_b *= 0.6; score_s *= 0.6; logs.append("🚫 سوق مسطح - تجنب الدخول")

        # ضمّ إشارات المحسنة لباقي المنظومة
        ind.update({
            "rsi_ma": rsi_ctx["rsi_ma"],
            "rsi_trendz": rsi_ctx["trendZ"],
            "di_spread": di_spread,
            "gz": gz,
            "footprint": footprint,
            "candle_buy_score": cd["score_buy"],
            "candle_sell_score": cd["score_sell"],
            "wick_up_big": cd["wick_up_big"],
            "wick_dn_big": cd["wick_dn_big"],
            "candle_tags": cd["pattern"],
            "smc_pattern": cd["smc_pattern"],
            "liquidity_trap": cd["liquidity_trap"],
            "box_ctx": box_ctx,
            "fvg_ctx": fvg_ctx,
            "rf_bs": rf_bs
        })

        return {
            "b": votes_b, "s": votes_s,
            "score_b": score_b, "score_s": score_s,
            "logs": logs, "ind": ind, "gz": gz, 
            "footprint": footprint, "candles": cd,
            "liquidity_traps": liquidity_traps,
            "box_ctx": box_ctx,
            "fvg_ctx": fvg_ctx,
            "rf_bs": rf_bs
        }
    except Exception as e:
        log_w(f"council_votes_pro_enhanced error: {e}")
        return {"b":0,"s":0,"score_b":0.0,"score_s":0.0,"logs":[],"ind":{},"gz":None,"candles":{}}

# ... (بقية الدوال تبقى كما هي مع التعديلات البسيطة)

# =================== ENHANCED TRADE MANAGEMENT ===================
def manage_after_entry_enhanced(df, ind, info):
    """إدارة محسنة للمركز مع Smart Profit AI + Smart Exit Guard + Box Engine"""
    if not STATE["open"] or STATE["qty"] <= 0:
        return

    px   = info["price"]
    entry = STATE["entry"]
    side  = STATE["side"]
    qty   = STATE["qty"]
    mode  = STATE.get("mode", "trend")
    
    # PnL % (كـ نسبة مئوية)
    pnl_pct = (px - entry) / entry * 100.0 * (1 if side == "long" else -1)
    STATE["pnl"] = pnl_pct

    if pnl_pct > STATE.get("highest_profit_pct", 0.0):
        STATE["highest_profit_pct"] = pnl_pct

    # ========= Smart Profit AI (سلم جني الأرباح الذكي) =========
    profit_decision = smart_profit_ai_decision(STATE, df, ind, mode, side, entry, px, {
        "box_ctx": STATE.get("box_ctx"),
        "fvg_ctx": STATE.get("fvg_ctx")
    })
    
    if profit_decision["action"] == "take_profit":
        target_num = profit_decision["target"]
        fraction   = profit_decision["fraction"]
        close_qty  = safe_qty(qty * fraction)
        
        if close_qty > 0:
            close_side = "sell" if side == "long" else "buy"
            if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                try:
                    ex.create_order(SYMBOL, "market", close_side, close_qty, None, _params_close())
                    log_g(f"✅ SMART TP{target_num}: closed {fraction*100:.0f}% | {profit_decision['reason']}")
                    STATE["qty"] = safe_qty(qty - close_qty)
                    STATE["profit_targets_achieved"] = target_num
                    
                    # لو دي آخر مرحلة جني أرباح في السلم: اقفل الصفقة بالكامل
                    if target_num >= (len(SCALP_TPS) if mode == "scalp" else len(TREND_TPS)):
                        close_market_strict("all_targets_achieved")
                        return
                except Exception as e:
                    log_e(f"❌ Smart TP failed: {e}")
            else:
                log_i(f"DRY_RUN: Smart TP{target_num} close {close_qty:.4f}")

    # ... (بقية دوال الإدارة تبقى كما هي)

# =================== ENHANCED TRADE LOOP ===================
def trade_loop_enhanced():
    """حلقة تداول محسنة مع Golden Zone Pro وSmart Profit AI وVWAP وBox Engine"""
    global wait_for_next_signal_side
    loop_i = 0
    
    while True:
        try:
            # جمع البيانات الأساسية
            bal = balance_usdt()
            px = price_now()
            df = fetch_ohlcv()
            info = rf_signal_live(df)
            ind = compute_indicators(df)
            spread_bps = orderbook_spread_bps()
            
            # Enhanced Snapshots مع Box Engine
            snap = emit_snapshots(ex, SYMBOL, df,
                                balance_fn=lambda: float(bal) if bal else None,
                                pnl_fn=lambda: float(compound_pnl))
            
            # Box Engine Analysis
            boxes = build_sr_boxes(df) if BOX_ENGINE_ENABLED else []
            box_ctx = analyze_box_context(df, boxes) if boxes else {"ctx": "none", "dir": "none", "score": 0.0}
            
            # FVG Analysis
            fvg_gaps = detect_fvg(df) if BOX_ENGINE_ENABLED else []
            fvg_ctx = classify_fvg_context(df, fvg_gaps) if fvg_gaps else {"ctx": "none", "dir": "none", "real_fake": "neutral"}
            
            # Stop Hunt Zones
            stop_hunt = detect_stop_hunt_zones(df, boxes) if BOX_ENGINE_ENABLED else {"ok": False, "zones": []}
            
            # تحديث حالة الربح/الخسارة
            if STATE["open"] and px:
                STATE["pnl"] = (px-STATE["entry"])*STATE["qty"] if STATE["side"]=="long" else (STATE["entry"]-px)*STATE["qty"]
            
            # إدارة الصفقة المفتوحة مع Smart Profit AI
            if STATE["open"]:
                manage_after_entry_enhanced(df, ind, {
                    "price": px or info["price"], 
                    "bm": snap["bm"],
                    "flow": snap["flow"],
                    "box_ctx": box_ctx,
                    "fvg_ctx": fvg_ctx,
                    **info
                })
            
            # قرار الدخول المحسن
            reason = None
            if spread_bps is not None and spread_bps > MAX_SPREAD_BPS:
                reason = f"spread too high ({fmt(spread_bps,2)}bps > {MAX_SPREAD_BPS})"
            
            council_data = council_votes_pro_enhanced(df)
            gz = council_data.get("gz")
            footprint = council_data.get("footprint", {})
            sig = None

            # --- Enhanced Golden Entry Pro ---
            golden_entry = False
            if (gz and gz.get("ok") and gz.get("confirmed")):
                if gz["zone"]["type"]=="golden_bottom" and gz["score"]>=GOLDEN_ENTRY_SCORE:
                    if footprint.get('ok') and footprint.get('absorption_bull'):
                        sig = "buy"
                        golden_entry = True
                        log_i(f"🎯 GOLDEN ENTRY PRO: BUY | score={gz['score']:.1f} | منطقة ذهبية مؤكدة + Footprint")
                elif gz["zone"]["type"]=="golden_top" and gz["score"]>=GOLDEN_ENTRY_SCORE:
                    if footprint.get('ok') and footprint.get('absorption_bear'):
                        sig = "sell"
                        golden_entry = True
                        log_i(f"🎯 GOLDEN ENTRY PRO: SELL | score={gz['score']:.1f} | منطقة ذهبية مؤكدة + Footprint")

            # Box Engine Entry (إذا لم يكن هناك دخول ذهبي)
            if not golden_entry and box_ctx["ctx"] != "none":
                if box_ctx["dir"] == "buy" and box_ctx["score"] >= 4.0:
                    sig = "buy"
                    log_i(f"📦 BOX ENGINE ENTRY: BUY | score={box_ctx['score']:.1f} | {box_ctx['debug']}")
                elif box_ctx["dir"] == "sell" and box_ctx["score"] >= 4.0:
                    sig = "sell" 
                    log_i(f"📦 BOX ENGINE ENTRY: SELL | score={box_ctx['score']:.1f} | {box_ctx['debug']}")

            # Council Strong Entry (إذا لم يكن هناك دخول ذهبي أو بوكس)
            if not golden_entry and sig is None:
                if council_data["score_b"] > council_data["score_s"] and council_data["score_b"] >= 8.0:
                    sig = "buy"
                elif council_data["score_s"] > council_data["score_b"] and council_data["score_s"] >= 8.0:
                    sig = "sell"
            
            if not STATE["open"] and sig and reason is None:
                allow_wait, wait_reason = wait_gate_allow(df, info)
                if not allow_wait:
                    reason = wait_reason
                else:
                    qty = compute_size(bal, px or info["price"])
                    if qty > 0:
                        ok = open_market_enhanced(sig, qty, px or info["price"])
                        if ok:
                            wait_for_next_signal_side = None
                            # حفظ سياق البوكس والإشارات
                            STATE["box_ctx"] = box_ctx
                            STATE["fvg_ctx"] = fvg_ctx
                            # تسجيل قرار المجلس المحسن
                            log_i(f"🎯 ENHANCED COUNCIL DECISION: {sig.upper()} | "
                                  f"Score B/S: {council_data['score_b']:.1f}/{council_data['score_s']:.1f} | "
                                  f"Signal Strength: {STATE.get('signal_strength', 0):.1f}")
                            for log_msg in council_data.get("logs", []):
                                log_i(f"   - {log_msg}")
                    else:
                        reason = "qty<=0"
            
            loop_i += 1
            sleep_s = NEAR_CLOSE_S if time_to_candle_close(df) <= 10 else BASE_SLEEP
            time.sleep(sleep_s)
            
        except Exception as e:
            log_e(f"loop error: {e}\n{traceback.format_exc()}")
            logging.error(f"trade_loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

# ... (بقية الكود يبقى كما هو مع إضافة Box Engine information في اللوغات والسنابشوتات)

# =================== BOOT ===================
if __name__ == "__main__":
    log_banner("INIT")
    state = load_state() or {}
    state.setdefault("in_position", False)

    if RESUME_ON_RESTART:
        try:
            state = resume_open_position(ex, SYMBOL, state)
        except Exception as e:
            log_w(f"resume error: {e}\n{traceback.format_exc()}")

    verify_execution_environment()

    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'}  •  {SYMBOL}  •  {INTERVAL}", "yellow"))
    print(colored(f"RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x  •  COUNCIL_PRO=ENHANCED", "yellow"))
    print(colored(f"GOLDEN ENTRY PRO: score≥{GOLDEN_ENTRY_SCORE} | ADX≥{GOLDEN_ENTRY_ADX}", "yellow"))
    print(colored(f"ENHANCED CANDLES: SMC Patterns + Wick exhaustion + Golden reversal", "yellow"))
    print(colored(f"FOOTPRINT ANALYSIS: Volume spikes + Absorption detection", "yellow"))
    print(colored(f"SMART PROFIT AI: Dynamic profit taking + Signal strength", "yellow"))
    print(colored(f"VWAP STRATEGY: SCALP(near {VWAP_SCALP_BAND_BPS}bps) | TREND(far {VWAP_TREND_BAND_BPS}bps)", "yellow"))
    print(colored(f"BOX ENGINE: Supply/Demand Zones + Strength Scoring + FVG Detection", "yellow"))
    print(colored(f"EXECUTION: {'ACTIVE' if EXECUTE_ORDERS and not DRY_RUN else 'SIMULATION'}", "yellow"))
    
    logging.info("enhanced service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    
    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)

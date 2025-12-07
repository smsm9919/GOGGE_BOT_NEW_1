# -*- coding: utf-8 -*-
"""
RF Futures Bot — RF-LIVE ONLY (BingX Perp via CCXT)
• Council PRO Unified Decision System with Candles & Golden Entry
• Golden Entry + Golden Reversal + Wick Exhaustion + Smart Profit AI
• Dynamic TP ladder + Breakeven + ATR-trailing
• Smart Exit Management + Wait-for-next-signal
• Professional Logging & Dashboard
• Enhanced with Footprint, SMC Candles, Liquidity Traps + VWAP Strategy
• OTC Hidden Flow Detection & Protection System
• EMA Crossover Strength Engine (Strong/Weak Trend Detection)
• ENHANCED WITH: Hard Stop Loss, Post-Big-Win Guard, Auto-Recovery
"""

import os, time, math, random, signal, sys, traceback, logging, json
from logging.handlers import RotatingFileHandler
from datetime import datetime
import pandas as pd
import numpy as np
import ccxt
from flask import Flask, jsonify
from decimal import Decimal, ROUND_DOWN, InvalidOperation

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
BOT_VERSION = "DOGE Council PRO v5.0 — Smart Profit AI + Golden Zone Pro + VWAP Strategy + OTC Detection + EMA Crossover Engine + Hard Stop + Post-Big-Win Guard + Auto-Recovery"
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

# ================== RISK & GUARDS CONFIG ==================
MAX_LOSS_PCT      = -0.005   # -0.50% حدّ خسارة صارم للصفقة الواحدة
BIG_WIN_PCT       =  0.020   # +2.0% أو أكثر تعتبر صفقة Big Win
POST_BIG_WIN_BARS =  2       # عدد الشموع بعد Big Win يتم فيها تفعيل الحذر (بعدها يرجع يشتغل عادي)

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
# كل قد إيه نطبع حالة الصفقة وهي مفتوحة (بالثواني)
POSITION_STATUS_LOG_INTERVAL = 30

# ==== Smart Exit Tuning ===
TP1_SCALP_PCT      = 0.35/100
TP1_TREND_PCT      = 0.60/100
HARD_CLOSE_PNL_PCT = 1.10/100
WICK_ATR_MULT      = 1.5
EVX_SPIKE          = 1.8
BM_WALL_PROX_BPS   = 5
TIME_IN_TRADE_MIN  = 8
TRAIL_TIGHT_MULT   = 1.20

# ==== OTC Hidden Flow Detection Settings ====
OTC_WINDOW_BARS          = 5        # عدد الشموع اللي نحلّل عليها حركة السعر + الفلو
OTC_MIN_MOVE_BPS         = 60.0     # أقل حركة سعر (بالـ bps) نعتبرها Pump/Dump (0.6%)
OTC_MAX_VISIBLE_FLOW_PCT = 0.25     # لو الفلو الظاهر أقل من 25% من الفوليوم -> نعتبرها سيولة مخفية
OTC_STRENGTH_SCALE       = 0.1      # مقياس لتحويل الحركة + ضعف الفلو لقوة (score)

# ==== OTC Exit Tuning (بعد TP1) ====
OTC_EXIT_MIN_STRENGTH = 2.0      # أقل قوة OTC نعتبرها تهديد حقيقي
OTC_EXIT_MIN_PNL_PCT  = 0.60/100 # أقل ربح (0.6%) نسمح فيه بإغلاق بسبب OTC عكسي

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

# =================== OTC HIDDEN FLOW DETECTION ===================
def detect_otc_flows(df, window: int = OTC_WINDOW_BARS):
    """
    كشف سيولة OTC (شراء/بيع مخفي) من خلال:
    - حركة سعر قوية في آخر N شمعة
    - مع فلو ظاهر (delta) ضعيف مقارنة بالفوليوم
    """
    try:
        if len(df) < window + 2:
            return {
                "otc_buy": False,
                "otc_sell": False,
                "strength": 0.0,
                "reason": "",
            }

        close  = df["close"].astype(float)
        volume = df["volume"].astype(float)

        # حركة السعر في آخر window شمعة
        start_price = float(close.iloc[-window])
        last_price  = float(close.iloc[-1])
        if start_price <= 0:
            return {
                "otc_buy": False,
                "otc_sell": False,
                "strength": 0.0,
                "reason": "",
            }

        ret = (last_price - start_price) / start_price  # نسبة الحركة
        move_bps = abs(ret) * 10000.0

        # لو الحركة أصلاً ضعيفة ما نضيعش وقت
        if move_bps < OTC_MIN_MOVE_BPS:
            return {
                "otc_buy": False,
                "otc_sell": False,
                "strength": 0.0,
                "reason": "",
            }

        # تقريب delta من الفوليوم + اتجاه الشمعة
        sub_close  = close.iloc[-window:]
        sub_vol    = volume.iloc[-window:]
        price_diff = sub_close.diff()

        buy_vol  = sub_vol.where(price_diff > 0, 0.0)
        sell_vol = sub_vol.where(price_diff < 0, 0.0)
        delta_series = buy_vol - sell_vol

        cum_delta_window = float(delta_series.sum())
        total_vol        = float(sub_vol.sum() or 1.0)

        visible_flow_ratio = abs(cum_delta_window) / total_vol  # نسبة الفلو الظاهر للفوليوم

        # لو الفلو الظاهر ضعيف جدًا مقارنة بالفوليوم مع حركة سعر قوية -> OTC
        otc_buy = False
        otc_sell = False
        strength = 0.0
        reason = ""

        # Pump بلا فلو شرائي واضح -> OTC BUY
        if ret > 0:
            if cum_delta_window <= 0 or visible_flow_ratio < OTC_MAX_VISIBLE_FLOW_PCT:
                otc_buy = True
                # قوة الـ OTC = حركة السعر / الفلو الظاهر
                strength = (move_bps * OTC_STRENGTH_SCALE) / max(visible_flow_ratio, 0.05)
                reason = "price_up_without_visible_buy_flow"

        # Dump بلا فلو بيعي واضح -> OTC SELL
        elif ret < 0:
            if cum_delta_window >= 0 or visible_flow_ratio < OTC_MAX_VISIBLE_FLOW_PCT:
                otc_sell = True
                strength = (move_bps * OTC_STRENGTH_SCALE) / max(visible_flow_ratio, 0.05)
                reason = "price_down_without_visible_sell_flow"

        return {
            "otc_buy": bool(otc_buy),
            "otc_sell": bool(otc_sell),
            "strength": float(strength),
            "reason": reason,
            "move_bps": move_bps,
            "visible_flow_ratio": visible_flow_ratio,
        }
    except Exception as e:
        # في حالة أي خطأ ما نكسرش المجلس
        return {
            "otc_buy": False,
            "otc_sell": False,
            "strength": 0.0,
            "reason": f"error:{e}",
        }

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
    print(f"💰 OTC DETECTION: Hidden flow detection + Protection", flush=True)
    print(f"📊 VWAP STRATEGY: SCALP(near {VWAP_SCALP_BAND_BPS}bps) | TREND(far {VWAP_TREND_BAND_BPS}bps)", flush=True)
    print(f"📈 EMA CROSSOVER ENGINE: Strong/Weak Trend Detection", flush=True)
    print(f"🛡️ RISK GUARDS: Hard Stop Loss (-{abs(MAX_LOSS_PCT)*100}%) | Post-Big-Win Guard (+{BIG_WIN_PCT*100}%)", flush=True)
    
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

# =================== EMA CROSSOVER STRENGTH ENGINE ===================
def ema_crossover_strength(df, ind=None, fast=9, mid=21, slow=50):
    """
    تصنيف تقاطع EMA:
    - strong_bull / weak_bull
    - strong_bear / weak_bear
    - none

    يرجّع:
    {
        "side": "bull"/"bear"/"flat",
        "label": "strong_bull"/"weak_bull"/"strong_bear"/"weak_bear"/"none",
        "score": float
    }
    """
    try:
        if len(df) < slow + 5:
            return {"side": "flat", "label": "none", "score": 0.0}

        close = df["close"].astype(float)
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_mid  = close.ewm(span=mid, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        f_now = float(ema_fast.iloc[-1]); f_prev = float(ema_fast.iloc[-2])
        m_now = float(ema_mid.iloc[-1]);  m_prev = float(ema_mid.iloc[-2])
        s_now = float(ema_slow.iloc[-1])
        px    = float(close.iloc[-1])

        cross_up   = (f_prev <= m_prev) and (f_now > m_now)
        cross_down = (f_prev >= m_prev) and (f_now < m_now)

        adx = 0.0
        if ind:
            adx = float(ind.get("adx", 0.0))

        side  = "flat"
        label = "none"
        score = 0.0

        # ===== Bullish context =====
        if f_now > m_now >= s_now:
            side = "bull"
            # strong bull: ema fast > mid > slow + price فوق fast + ADX محترم + cross قريب
            if cross_up and adx >= 22 and px > f_now:
                label = "strong_bull"
                score = 3.0
            else:
                label = "weak_bull"
                score = 1.5

        # ===== Bearish context =====
        elif f_now < m_now <= s_now:
            side = "bear"
            if cross_down and adx >= 22 and px < f_now:
                label = "strong_bear"
                score = 3.0
            else:
                label = "weak_bear"
                score = 1.5

        return {"side": side, "label": label, "score": score}
    except Exception:
        return {"side": "flat", "label": "none", "score": 0.0}

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

def decide_strategy_mode_enhanced(df, adx=None, di_plus=None, di_minus=None, rsi_ctx=None, footprint=None, ema_ctx=None):
    """تحديد نمط التداول المحسن: SCALP أم TREND مع VWAP + Footprint + EMA"""
    ind = compute_indicators(df)

    if adx is None or di_plus is None or di_minus is None:
        adx = ind.get('adx', 0)
        di_plus = ind.get('plus_di', 0)
        di_minus = ind.get('minus_di', 0)

    if rsi_ctx is None:
        rsi_ctx = rsi_ma_context(df)

    if footprint is None:
        footprint = compute_footprint_metrics(df)

    if ema_ctx is None:
        ema_ctx = ema_crossover_strength(df, ind)

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

    # تقاطع EMA قوي يفرض نمط الترند
    ema_label = ema_ctx.get("label", "none")
    if ema_label in ("strong_bull", "strong_bear"):
        return {"mode": "trend", "why": "ema_strong_cross"}

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

# =================== SCALP DIRECTION GUARD ===================
def enforce_scalp_trend_alignment(df, ind, planned_side: str, mode_ctx: dict):
    """
    حارس سكالب:
    - لو mode = scalp لازم نمشي مع اتجاه الترند العام.
    - Bull ⇒ سكالب BUY (long) فقط
    - Bear ⇒ سكالب SELL (short) فقط
    - Flat / Chop ⇒ نلغي الصفقة السكالب.
    يرجّع:
      side, skip
    """
    try:
        if mode_ctx.get("mode") != "scalp":
            return planned_side, False  # مش سكالب → ما نغيّرش حاجة

        # اتجاه EMA العام
        ema_ctx = ema_crossover_strength(df, ind)
        trend_side = ema_ctx.get("side", "flat")   # bull / bear / flat
        ema_label = ema_ctx.get("label", "none")

        # سياق RSI
        rsi_ctx = rsi_ma_context(df)
        rsi_trend = rsi_ctx.get("trendZ")          # bull / bear / chop

        # Footprint / OrderFlow
        footprint = compute_footprint_metrics(df)
        flow_trend = footprint.get("delta_trend") if footprint.get("ok") else None  # bull/bear

        # بناء bias نهائي
        bias = "flat"

        # Flow قوي ياخد أولوية لو موجود
        if flow_trend in ("bull", "bear"):
            bias = flow_trend
        else:
            # بعده EMA
            if trend_side in ("bull", "bear"):
                bias = trend_side

        # لو RSI واضح ومش متعارض مع الـ bias، نعزّزه
        if rsi_trend in ("bull", "bear"):
            if bias == "flat":
                bias = rsi_trend
            elif bias != rsi_trend:
                # تعارض قوي ⇒ نعتبر السوق فوضوي للسكالب
                log_i(f"⏸️ SCALP: RSI vs EMA/Flow conflict (bias={bias}, rsi={rsi_trend}) → skip")
                return planned_side, True

        if bias == "flat":
            # مفيش اتجاه واضح → نلغي السكالب
            log_i("⏸️ SCALP: no clear trend bias (flat) → skip scalp entry")
            return planned_side, True

        forced_side = "long" if bias == "bull" else "short"

        if planned_side != forced_side:
            log_w(
                f"🔄 SCALP DIRECTION OVERRIDE | planned={planned_side} → forced={forced_side} "
                f"| bias={bias} | ema={ema_label}"
            )
            return forced_side, False

        # الاتجاه planned متماشي مع الترند
        return planned_side, False

    except Exception as e:
        log_w(f"SCALP direction guard error: {e}")
        return planned_side, False

# =================== SMART PROFIT AI ===================
def smart_profit_ai_decision(state, df, ind, mode, side, entry_price, current_price):
    """
    ذكاء اصطناعي لجني الأرباح:
      - يعتمد على trade_profile المخزّن في STATE:
        * STRONG_TREND  → سلم 3 مراحل (TREND_TPS / TREND_TP_FRACS)
        * MID_TREND     → جني أرباح مرة واحدة محترمة (باستخدام SCALP_TPS لكن بحجم محترم)
        * SCALP_LIGHT   → سكالب خفيف هدف واحد وسريع
    """
    pnl_pct = (current_price - entry_price) / entry_price * 100 * (1 if side == "long" else -1)
    
    trade_profile = state.get("trade_profile") or "MID_TREND"
    signal_strength = state.get("signal_strength", 0.0)
    
    # --- اختيار سلم الأهداف حسب الـ Profile ---
    if trade_profile == "STRONG_TREND":
        # ترند قوي → 3 مراحل TP (سلم كامل)
        tp_levels = TREND_TPS[:]          # [0.50, 1.00, 1.80] مبدئياً
        tp_fractions = TREND_TP_FRACS[:]  # [0.30, 0.30, 0.20]
    elif trade_profile == "SCALP_LIGHT":
        # سكالب خفيف → هدف واحد سريع
        tp_levels = SCALP_TPS[:]          # [0.40]
        tp_fractions = SCALP_TP_FRACS[:]  # [0.60]
    else:
        # MID_TREND (نص ترند / باي عادية) → جني أرباح مرة واحدة محترمة
        # نستخدم نفس SCALP_TPS لكن نسمح بسلوك "صفقة محترمة" (Signal boost)
        tp_levels = SCALP_TPS[:]
        tp_fractions = SCALP_TP_FRACS[:]
    
    achieved_targets = state.get("profit_targets_achieved", 0)
    next_target_index = achieved_targets
    
    if next_target_index >= len(tp_levels):
        return {"action": "hold", "target": None, "reason": "كل الأهداف محققة"}
    
    next_target_pct = tp_levels[next_target_index]
    next_target_fraction = tp_fractions[next_target_index]
    
    # لو ما فيش signal_strength في الـ STATE لأي سبب، نحسبه احتياطيًا
    if signal_strength <= 0.0:
        signal_strength = calculate_signal_strength(df, ind, side)
    
    # تعديل الأهداف حسب قوة الإشارة (Fine Tuning جوه الـ Profile)
    if signal_strength >= 8.0:
        # إشارة قوية جداً → نمدّي الهدف شوية
        next_target_pct *= 1.25
    elif signal_strength >= 6.0:
        # إشارة قوية → نمدّي الهدف بسيط
        next_target_pct *= 1.10
    elif signal_strength < 3.0:
        # إشارة ضعيفة → نقلّل الهدف عشان ما نطمعش في سكالب ضعيف
        next_target_pct *= 0.80
    
    # قرار التنفيذ
    if pnl_pct >= next_target_pct:
        return {
            "action": "take_profit",
            "target": next_target_index + 1,
            "target_pct": next_target_pct,
            "fraction": next_target_fraction,
            "reason": (
                f"TP{next_target_index + 1} hit "
                f"({next_target_pct:.2f}%) | profile={trade_profile} "
                f"| strength={signal_strength:.1f}"
            ),
        }
    
    return {
        "action": "hold",
        "target": next_target_index + 1,
        "reason": (
            f"لم يحقق الهدف بعد | "
            f"profile={trade_profile} | "
            f"target={next_target_pct:.2f}% | strength={signal_strength:.1f}"
        ),
    }

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
    
    # EMA Crossover impact
    ema_ctx = ema_crossover_strength(df, ind)
    ema_label = ema_ctx.get("label", "none")

    if side == "long":
        if ema_label == "strong_bull":
            strength += 2.0
        elif ema_label == "weak_bull":
            strength += 1.0
        elif ema_label in ("strong_bear", "weak_bear"):
            # إشارة معاكسة تقلل الثقة
            strength -= 1.0
    elif side == "short":
        if ema_label == "strong_bear":
            strength += 2.0
        elif ema_label == "weak_bear":
            strength += 1.0
        elif ema_label in ("strong_bull", "weak_bull"):
            strength -= 1.0

    strength = max(0.0, strength)
    return min(10.0, strength)

# =================== TRADE PROFILE CLASSIFICATION ===================
def classify_trade_profile(signal_strength: float, mode: str) -> str:
    """
    تحويل قوة الإشارة + نمط الصفقة (scalp/trend)
    إلى Profile واضح لإدارة جني الأرباح:
      - STRONG_TREND  → ترند قوي (3 مراحل TP)
      - MID_TREND     → نص ترند / صفقة محترمة (TP مرة واحدة محترمة)
      - SCALP_LIGHT   → سكالب خفيف (TP سريع واحد)
    """
    # أولاً نحدد bucket للقوة
    # 0–3  : ضعيف/خفيف
    # 3–7  : متوسّط (نص ترند)
    # 7–10 : قوي/سوبر ترند
    if signal_strength >= 7.0:
        strength_bucket = "strong"
    elif signal_strength >= 3.0:
        strength_bucket = "mid"
    else:
        strength_bucket = "weak"

    # لو النمط Trend و الإشارة قوية → ترند قوي حقيقي
    if mode == "trend" and strength_bucket == "strong":
        return "STRONG_TREND"
    
    # لو سكالب أو قوة متوسطة → نص ترند / باي عادية
    if strength_bucket == "mid":
        return "MID_TREND"
    
    # أي شيء أضعف → سكالب خفيف
    return "SCALP_LIGHT"

# =================== ENHANCED COUNCIL VOTING ===================
def council_votes_pro_enhanced(df):
    """مجلس تصويت محسّن مع Footprint + SMC + Golden Zone Pro + VWAP + OTC + EMA"""
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

        votes_b = 0; votes_s = 0
        score_b = 0.0; score_s = 0.0
        logs = []

        adx = ind.get('adx', 0)
        plus_di = ind.get('plus_di', 0)
        minus_di = ind.get('minus_di', 0)
        di_spread = ind.get('di_spread', abs(plus_di - minus_di))

        # ==== EMA CROSSOVER CONTEXT =====
        ema_ctx = ema_crossover_strength(df, ind)
        ema_label = ema_ctx.get("label", "none")

        if ema_label == "strong_bull":
            votes_b += 3; score_b += 2.0
            logs.append("📈 EMA STRONG BULL TREND")
        elif ema_label == "weak_bull":
            votes_b += 1; score_b += 0.8
            logs.append("🟢 EMA WEAK BULL BIAS")
        elif ema_label == "strong_bear":
            votes_s += 3; score_s += 2.0
            logs.append("📉 EMA STRONG BEAR TREND")
        elif ema_label == "weak_bear":
            votes_s += 1; score_s += 0.8
            logs.append("🔴 EMA WEAK BEAR BIAS")

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

        # --- OTC Hidden Flow Detection ---
        otc = detect_otc_flows(df)
        if otc.get("otc_buy"):
            # سيولة شراء مخفية تدعم الشراء
            boost = min(3.0, otc.get("strength", 1.5))
            votes_b += 2
            score_b += boost
            logs.append(
                f"💰 OTC BUY ({otc.get('reason','')}) "
                f"move={otc.get('move_bps',0):.1f}bps flow={otc.get('visible_flow_ratio',0)*100:.1f}% s={boost:.1f}"
            )
        elif otc.get("otc_sell"):
            # سيولة بيع مخفية تدعم البيع
            boost = min(3.0, otc.get("strength", 1.5))
            votes_s += 2
            score_s += boost
            logs.append(
                f"💰 OTC SELL ({otc.get('reason','')}) "
                f"move={otc.get('move_bps',0):.1f}bps flow={otc.get('visible_flow_ratio',0)*100:.1f}% s={boost:.1f}"
            )
        else:
            otc = {"otc_buy": False, "otc_sell": False, "strength": 0.0}

        # --- Liquidity Trap Awareness
        if liquidity_traps.get('ok') and liquidity_traps.get('traps'):
            for trap in liquidity_traps['traps']:
                if trap['type'] == 'stop_hunt_bull' and score_b > score_s:
                    score_b *= 1.1  # تعزيز الثقة في فخ الصعود
                    logs.append(f"🪤 فخ سيولة صاعد قريب ({trap['distance_pct']:.2f}%)")
                elif trap['type'] == 'stop_hunt_bear' and score_s > score_b:
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
            "ema_ctx": ema_ctx,
        })

        return {
            "b": votes_b, "s": votes_s,
            "score_b": score_b, "score_s": score_s,
            "logs": logs, "ind": ind, "gz": gz, 
            "footprint": footprint, "candles": cd,
            "liquidity_traps": liquidity_traps,
            "otc": otc,
            "ema": ema_ctx,
        }
    except Exception as e:
        log_w(f"council_votes_pro_enhanced error: {e}")
        return {
            "b": 0, "s": 0,
            "score_b": 0.0, "score_s": 0.0,
            "logs": [], "ind": {}, "gz": None,
            "candles": {}, "footprint": {}, "liquidity_traps": {}, "otc": {}, "ema": {}
        }

# =================== POSITION RECOVERY ===================
def _normalize_side(pos):
    side = pos.get("side") or pos.get("positionSide") or ""
    if side: return side.upper()
    qty = float(pos.get("contracts") or pos.get("positionAmt") or pos.get("size") or 0)
    return "LONG" if qty > 0 else ("SHORT" if qty < 0 else "")

def fetch_live_position(exchange, symbol: str):
    try:
        if hasattr(exchange, "fetch_positions"):
            arr = exchange.fetch_positions([symbol])
            for p in arr or []:
                sym = p.get("symbol") or p.get("info", {}).get("symbol")
                if sym and symbol.replace(":","") in sym.replace(":",""):
                    side = _normalize_side(p)
                    qty = abs(float(p.get("contracts") or p.get("positionAmt") or p.get("info",{}).get("size",0) or 0))
                    if qty > 0:
                        entry = float(p.get("entryPrice") or p.get("info",{}).get("entryPrice") or 0.0)
                        lev = float(p.get("leverage") or p.get("info",{}).get("leverage") or 0.0)
                        unr = float(p.get("unrealizedPnl") or 0.0)
                        return {"ok": True, "side": side, "qty": qty, "entry": entry, "unrealized": unr, "leverage": lev, "raw": p}
        if hasattr(exchange, "fetch_position"):
            p = exchange.fetch_position(symbol)
            side = _normalize_side(p); qty = abs(float(p.get("size") or 0))
            if qty > 0:
                entry = float(p.get("entryPrice") or 0.0)
                lev   = float(p.get("leverage") or 0.0)
                unr   = float(p.get("unrealizedPnl") or 0.0)
                return {"ok": True, "side": side, "qty": qty, "entry": entry, "unrealized": unr, "leverage": lev, "raw": p}
    except Exception as e:
        log_w(f"fetch_live_position error: {e}")
    return {"ok": False, "why": "no_open_position"}

# ================== RISK HELPERS ==================

def compute_unrealized_pnl_pct(state, current_price: float) -> float:
    """
    حساب % PnL غير المحقَّق للصفقة الحالية بناءً على الـ STATE.
    """
    if not state.get("open"):
        return 0.0

    side = state.get("side")
    entry = float(state.get("entry") or 0.0)
    if not entry:
        return 0.0

    if side == "long":
        return (current_price / entry) - 1.0
    elif side == "short":
        return (entry / current_price) - 1.0
    return 0.0


def risk_on_new_trade(state, side: str, entry_price: float):
    """
    يُستدعى مباشرة بعد فتح صفقة جديدة:
    - يحسب max_loss_price كـ % ثابت من الـ entry
    - يخفّض وضع Post Big Win لو انتهى
    """
    state["open"] = True
    state["side"] = side
    state["entry"] = float(entry_price)

    # ستوب على أساس % من سعر الدخول
    if side == "long":
        state["max_loss_price"] = entry_price * (1.0 + MAX_LOSS_PCT)
        log_i(f"🛡️ Hard Stop Loss SET for LONG: {state['max_loss_price']:.6f} (-{abs(MAX_LOSS_PCT)*100}%)")
    else:  # short
        state["max_loss_price"] = entry_price * (1.0 - MAX_LOSS_PCT)
        log_i(f"🛡️ Hard Stop Loss SET for SHORT: {state['max_loss_price']:.6f} (-{abs(MAX_LOSS_PCT)*100}%)")

    # لما نفتح صفقة جديدة، نكمّل عدّاد post_big_win_mode لو لسه شغّال
    if state.get("post_big_win_mode") and state.get("post_big_win_bars_left", 0) <= 0:
        state["post_big_win_mode"] = False
        state["post_big_win_bars_left"] = 0


def risk_on_trade_closed(state, realized_pnl_pct: float):
    """
    يُستدعى بعد إغلاق الصفقة بالكامل:
    - يخزن آخر % PnL
    - يفعّل وضع Post Big Win لو الربح كبير
    """
    state["last_closed_pnl_pct"] = float(realized_pnl_pct or 0.0)
    state["open"] = False
    state["side"] = None
    state["entry"] = None
    state["max_loss_price"] = None

    if realized_pnl_pct >= BIG_WIN_PCT:
        state["post_big_win_mode"] = True
        state["post_big_win_bars_left"] = POST_BIG_WIN_BARS
        log_g(f"🎉 BIG WIN DETECTED! {realized_pnl_pct*100:.2f}% ≥ {BIG_WIN_PCT*100}% - Post-Big-Win Mode ACTIVATED for {POST_BIG_WIN_BARS} bars")
    else:
        # مفيش Big Win → نطفي المود ده
        state["post_big_win_mode"] = False
        state["post_big_win_bars_left"] = 0


def tick_post_big_win_decay(state):
    """
    تُستدعى مع كل شمعة/لوب:
    تقلل عدّاد Post Big Win لو شغّال.
    """
    if state.get("post_big_win_mode"):
        left = int(state.get("post_big_win_bars_left", 0))
        if left > 0:
            state["post_big_win_bars_left"] = left - 1
            if state["post_big_win_bars_left"] <= 0:
                state["post_big_win_mode"] = False
                state["post_big_win_bars_left"] = 0
                log_i("🛡️ Post-Big-Win Mode EXPIRED - Returning to normal trading")

# =================== AUTO RECOVERY & SYNC ===================
def sync_open_position_from_exchange(exchange, symbol: str, state):
    """
    محاولة مزامنة أي صفقة مفتوحة على المنصة مع STATE بعد الريستارت.
    """
    try:
        positions = exchange.fetch_positions([symbol])
    except Exception as e:
        log_w(f"sync_open_position error: {e}")
        return state

    live_pos = None
    for p in positions:
        sym = p.get("symbol") or p.get("info", {}).get("symbol") or ""
        if symbol.replace(":", "") in sym.replace(":", ""):
            qty = abs(float(p.get("contracts") or p.get("positionAmt") or p.get("info", {}).get("size", 0) or 0))
            if qty > 0:
                live_pos = p
                break

    if not live_pos:
        # مفيش صفقة حية على البورصة → صفّر الحالة لو عندك in_position
        if state.get("open"):
            log_w("🔄 No live position on exchange, resetting local STATE.")
            state["open"] = False
            state["side"] = None
            state["entry"] = None
            state["max_loss_price"] = None
            state["post_big_win_mode"] = False
            state["post_big_win_bars_left"] = 0
        return state

    # استخراج بيانات الصفقة الحية
    side_raw = (live_pos.get("side") or live_pos.get("positionSide") or "").lower()
    side = "long" if "long" in side_raw or float(live_pos.get("cost", 0)) > 0 else "short"
    entry_price = float(live_pos.get("entryPrice") or live_pos.get("info", {}).get("avgEntryPrice") or 0.0)
    qty = abs(float(live_pos.get("contracts") or live_pos.get("positionAmt") or 0))
    
    if qty <= 0:
        return state

    log_i(f"🔄 Syncing live position from exchange: {side.upper()} {qty:.4f} @ {entry_price:.6f}")
    
    # تحديث STATE بالصفقة الحية
    state.update({
        "open": True,
        "side": side,
        "entry": entry_price,
        "qty": qty,
        "in_position": True,
        "max_loss_price": entry_price * (1.0 + MAX_LOSS_PCT) if side == "long" else entry_price * (1.0 - MAX_LOSS_PCT),
    })
    
    # تحديث Post-Big-Win Mode من آخر صفقة مغلقة
    last_closed_pnl = state.get("last_closed_pnl_pct", 0.0)
    if last_closed_pnl >= BIG_WIN_PCT:
        state["post_big_win_mode"] = True
        state["post_big_win_bars_left"] = POST_BIG_WIN_BARS
        log_i(f"🔄 Post-Big-Win Mode RESTORED from last trade: {last_closed_pnl*100:.2f}%")
    
    return state


def enhanced_resume_open_position(exchange, symbol: str, state: dict) -> dict:
    """
    دالة استئناف محسنة مع Auto-Recovery الكامل
    """
    if not RESUME_ON_RESTART:
        log_i("🔄 Resume disabled")
        return state

    log_i("🔄 Starting enhanced position recovery...")
    
    # 1. أولاً: Sync مع المنصة
    state = sync_open_position_from_exchange(exchange, symbol, state)
    
    # 2. ثانياً: جلب البيانات الحية (فال باك أب)
    live = fetch_live_position(exchange, symbol)
    
    # 3. الدمج الذكي بين STATE والبيانات الحية
    if live.get("ok") and state.get("open"):
        # تأكيد البيانات من المصدرين
        ts = int(time.time())
        prev = load_state()
        
        if prev.get("ts") and (ts - int(prev["ts"])) > RESUME_LOOKBACK_SECS:
            log_w("🔄 Found old local state — overriding with exchange live snapshot")
        
        # تحديث STATE بالبيانات الأحدث
        state.update({
            "in_position": True,
            "side": live["side"],
            "entry_price": live["entry"],
            "position_qty": live["qty"],
            "leverage": live.get("leverage") or state.get("leverage") or LEVERAGE,
            "partial_taken": prev.get("partial_taken", False),
            "breakeven_armed": prev.get("breakeven_armed", False),
            "trail_active": prev.get("trail_active", False),
            "trail_tightened": prev.get("trail_tightened", False),
            "mode": prev.get("mode", "trend"),
            "gz_snapshot": prev.get("gz_snapshot", {}),
            "cv_snapshot": prev.get("cv_snapshot", {}),
            "footprint_snapshot": prev.get("footprint_snapshot", {}),
            "ema_snapshot": prev.get("ema_snapshot", {}),
            "opened_at": prev.get("opened_at", ts),
            
            # تحديث Stop Loss
            "max_loss_price": live["entry"] * (1.0 + MAX_LOSS_PCT) if live["side"] == "LONG" else live["entry"] * (1.0 - MAX_LOSS_PCT),
        })
        
        # حفظ الحالة المحدثة
        save_state(state)
        log_g(f"🔄 ENHANCED RESUME: {state['side']} qty={state['position_qty']} @ {state['entry_price']:.6f} "
              f"lev={state['leverage']}x | Stop Loss: {state.get('max_loss_price', 'N/A'):.6f}")
        
        # تسجيل حالة Post-Big-Win
        if state.get("post_big_win_mode"):
            log_i(f"🛡️ Post-Big-Win Mode ACTIVE: {state['post_big_win_bars_left']} bars remaining")
    
    return state

def resume_open_position(exchange, symbol: str, state: dict) -> dict:
    if not RESUME_ON_RESTART:
        log_i("resume disabled"); return state

    live = fetch_live_position(exchange, symbol)
    if not live.get("ok"):
        log_i("no live position to resume"); return state

    ts = int(time.time())
    prev = load_state()
    if prev.get("ts") and (ts - int(prev["ts"])) > RESUME_LOOKBACK_SECS:
        log_w("found old local state — will override with exchange live snapshot")

    state.update({
        "in_position": True,
        "side": live["side"],
        "entry_price": live["entry"],
        "position_qty": live["qty"],
        "leverage": live.get("leverage") or state.get("leverage") or 10,
        "partial_taken": prev.get("partial_taken", False),
        "breakeven_armed": prev.get("breakeven_armed", False),
        "trail_active": prev.get("trail_active", False),
        "trail_tightened": prev.get("trail_tightened", False),
        "mode": prev.get("mode", "trend"),
        "gz_snapshot": prev.get("gz_snapshot", {}),
        "cv_snapshot": prev.get("cv_snapshot", {}),
        "footprint_snapshot": prev.get("footprint_snapshot", {}),
        "ema_snapshot": prev.get("ema_snapshot", {}),
        "opened_at": prev.get("opened_at", ts),
    })
    save_state(state)
    log_g(f"RESUME: {state['side']} qty={state['position_qty']} @ {state['entry_price']:.6f} lev={state['leverage']}x")
    return state

# =================== LOGGING SETUP ===================
def setup_file_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "").endswith("bot.log")
               for h in logger.handlers):
        fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=7, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    log_i("log rotation ready")

setup_file_logging()

# =================== EXCHANGE ===================
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

def _interval_seconds(iv: str) -> int:
    iv=(iv or "").lower().strip()
    if iv.endswith("m"): return int(float(iv[:-1]))*60
    if iv.endswith("h"): return int(float(iv[:-1]))*3600
    if iv.endswith("d"): return int(float(iv[:-1]))*86400
    return 15*60

def time_to_candle_close(df: pd.DataFrame) -> int:
    tf = _interval_seconds(INTERVAL)
    if len(df) == 0: return tf
    cur_start_ms = int(df["time"].iloc[-1])
    now_ms = int(time.time()*1000)
    next_close_ms = cur_start_ms + tf*1000
    while next_close_ms <= now_ms:
        next_close_ms += tf*1000
    left = max(0, next_close_ms - now_ms)
    return int(left/1000)

# ========= Professional logging helpers =========
def fmt_walls(walls):
    return ", ".join([f"{p:.6f}@{q:.0f}" for p, q in walls]) if walls else "-"

# ========= Bookmap snapshot =========
def bookmap_snapshot(exchange, symbol, depth=BOOKMAP_DEPTH):
    try:
        ob = exchange.fetch_order_book(symbol, depth)
        bids = ob.get("bids", [])[:depth]; asks = ob.get("asks", [])[:depth]
        if not bids or not asks:
            return {"ok": False, "why": "empty"}
        b_sizes = np.array([b[1] for b in bids]); b_prices = np.array([b[0] for b in bids])
        a_sizes = np.array([a[1] for a in asks]); a_prices = np.array([a[0] for a in asks])
        b_idx = b_sizes.argsort()[::-1][:BOOKMAP_TOPWALLS]
        a_idx = a_sizes.argsort()[::-1][:BOOKMAP_TOPWALLS]
        buy_walls = [(float(b_prices[i]), float(b_sizes[i])) for i in b_idx]
        sell_walls = [(float(a_prices[i]), float(a_sizes[i])) for i in a_idx]
        imb = b_sizes.sum() / max(a_sizes.sum(), 1e-12)
        return {"ok": True, "buy_walls": buy_walls, "sell_walls": sell_walls, "imbalance": float(imb)}
    except Exception as e:
        return {"ok": False, "why": f"{e}"}

# ========= Volume flow / Delta & CVD =========
def compute_flow_metrics(df):
    try:
        if len(df) < max(30, FLOW_WINDOW+2):
            return {"ok": False, "why": "short_df"}
        close = df["close"].astype(float).copy()
        vol = df["volume"].astype(float).copy()
        up_mask = close.diff().fillna(0) > 0
        up_vol = (vol * up_mask).astype(float)
        dn_vol = (vol * (~up_mask)).astype(float)
        delta = up_vol - dn_vol
        cvd = delta.cumsum()
        cvd_ma = cvd.rolling(CVD_SMOOTH).mean()
        wnd = delta.tail(FLOW_WINDOW)
        mu = float(wnd.mean()); sd = float(wnd.std() or 1e-12)
        z = float((wnd.iloc[-1] - mu) / sd)
        trend = "up" if (cvd_ma.iloc[-1] - cvd_ma.iloc[-min(CVD_SMOOTH, len(cvd_ma))]) >= 0 else "down"
        return {"ok": True, "delta_last": float(delta.iloc[-1]), "delta_mean": mu, "delta_z": z,
                "cvd_last": float(cvd.iloc[-1]), "cvd_trend": trend, "spike": abs(z) >= FLOW_SPIKE_Z}
    except Exception as e:
        return {"ok": False, "why": str(e)}

# ========= Unified snapshot emitter =========
def emit_snapshots(exchange, symbol, df, balance_fn=None, pnl_fn=None):
    """
    يطبع Snapshot موحّد: Bookmap + Flow + Council + Strategy + Balance/PnL + VWAP + OTC + EMA
    """
    try:
        bm = bookmap_snapshot(exchange, symbol)
        flow = compute_flow_metrics(df)
        cv = council_votes_pro_enhanced(df)
        mode = decide_strategy_mode_enhanced(df, 
                                            adx=cv["ind"].get("adx"),
                                            di_plus=cv["ind"].get("plus_di"),
                                            di_minus=cv["ind"].get("minus_di"),
                                            rsi_ctx=rsi_ma_context(df),
                                            footprint=cv.get("footprint", {}),
                                            ema_ctx=cv.get("ema", {}))
        current_price = float(df['close'].iloc[-1])
        gz = golden_zone_pro_analysis(df, current_price)

        bal = None; cpnl = None
        if callable(balance_fn):
            try: bal = balance_fn()
            except: bal = None
        if callable(pnl_fn):
            try: cpnl = pnl_fn()
            except: cpnl = None

        if bm.get("ok"):
            imb_tag = "🟢" if bm["imbalance"]>=IMBALANCE_ALERT else ("🔴" if bm["imbalance"]<=1/IMBALANCE_ALERT else "⚖️")
            bm_note = f"Bookmap: {imb_tag} Imb={bm['imbalance']:.2f} | Buy[{fmt_walls(bm['buy_walls'])}] | Sell[{fmt_walls(bm['sell_walls'])}]"
        else:
            bm_note = f"Bookmap: N/A ({bm.get('why')})"

        if flow.get("ok"):
            dtag = "🟢Buy" if flow["delta_last"]>0 else ("🔴Sell" if flow["delta_last"]<0 else "⚖️Flat")
            spk = " ⚡Spike" if flow["spike"] else ""
            fl_note = f"Flow: {dtag} Δ={flow['delta_last']:.0f} z={flow['delta_z']:.2f}{spk} | CVD {'↗️' if flow['cvd_trend']=='up' else '↘️'} {flow['cvd_last']:.0f}"
        else:
            fl_note = f"Flow: N/A ({flow.get('why')})"

        side_hint = "BUY" if cv["b"]>=cv["s"] else "SELL"
        dash = (f"DASH → hint-{side_hint} | Council BUY({cv['b']},{cv['score_b']:.1f}) "
                f"SELL({cv['s']},{cv['score_s']:.1f}) | "
                f"RSI={cv['ind'].get('rsi',0):.1f} ADX={cv['ind'].get('adx',0):.1f} "
                f"DI={cv['ind'].get('di_spread',0):.1f}")

        strat_icon = "⚡" if mode["mode"]=="scalp" else "📈" if mode["mode"]=="trend" else "ℹ️"
        strat = f"Strategy: {strat_icon} {mode['mode'].upper()} ({mode['why']})"

        bal_note = f"Balance={bal:.2f}" if bal is not None else ""
        pnl_note = f"CompoundPnL={cpnl:.6f}" if cpnl is not None else ""
        wallet = (" | ".join(x for x in [bal_note, pnl_note] if x)) or ""

        gz_note = ""
        if gz and gz.get("ok"):
            gz_note = f" | 🟡 {gz['zone']['type']} s={gz['score']:.1f}"

        # OTC info
        otc_note = ""
        if cv.get("otc", {}).get("otc_buy"):
            otc_note = f" | 💰 OTC BUY s={cv['otc'].get('strength',0):.1f}"
        elif cv.get("otc", {}).get("otc_sell"):
            otc_note = f" | 💰 OTC SELL s={cv['otc'].get('strength',0):.1f}"
            
        # EMA info
        ema_note = ""
        ema_ctx = cv.get("ema", {})
        if ema_ctx.get("label") != "none":
            ema_label = ema_ctx.get("label", "")
            ema_score = ema_ctx.get("score", 0.0)
            ema_note = f" | 📈 EMA:{ema_label}({ema_score:.1f})"

        if LOG_ADDONS:
            print(f"🧱 {bm_note}", flush=True)
            print(f"📦 {fl_note}", flush=True)
            print(f"📊 {dash}{gz_note}{otc_note}{ema_note}", flush=True)
            print(f"{strat}{(' | ' + wallet) if wallet else ''}", flush=True)
            
            gz_snap_note = ""
            if gz and gz.get("ok"):
                zone_type = gz["zone"]["type"]
                zone_score = gz["score"]
                gz_snap_note = f" | 🟡{zone_type} s={zone_score:.1f}"
            
            flow_z = flow['delta_z'] if flow and flow.get('ok') else 0.0
            bm_imb = bm['imbalance'] if bm and bm.get('ok') else 1.0
            
            # إضافة معلومات VWAP للسنابشوت
            vwap_info = ""
            if VWAP_ENABLED and cv['ind'].get('vwap'):
                vwap_val = cv['ind']['vwap']
                current_price = float(df['close'].iloc[-1])
                vwap_diff_bps = abs(current_price - vwap_val) / vwap_val * 10000.0
                vwap_status = "NEAR" if vwap_diff_bps <= VWAP_SCALP_BAND_BPS else "FAR" if vwap_diff_bps >= VWAP_TREND_BAND_BPS else "MID"
                vwap_info = f" | VWAP:{vwap_status}({vwap_diff_bps:.1f}bps)"
            
            # OTC info for snapshot
            otc_snap = ""
            if cv.get("otc", {}).get("otc_buy"):
                otc_snap = f" | 💰OTC-BUY({cv['otc'].get('strength',0):.1f})"
            elif cv.get("otc", {}).get("otc_sell"):
                otc_snap = f" | 💰OTC-SELL({cv['otc'].get('strength',0):.1f})"
            
            # EMA info for snapshot
            ema_snap = ""
            if ema_ctx.get("label") != "none":
                ema_snap = f" | 📈EMA:{ema_ctx.get('label','')}({ema_ctx.get('score',0):.1f})"
            
            print(f"🧠 SNAP | {side_hint} | votes={cv['b']}/{cv['s']} score={cv['score_b']:.1f}/{cv['score_s']:.1f} "
                  f"| ADX={cv['ind'].get('adx',0):.1f} DI={cv['ind'].get('di_spread',0):.1f} | "
                  f"z={flow_z:.2f} | imb={bm_imb:.2f}{gz_snap_note}{vwap_info}{otc_snap}{ema_snap}", 
                  flush=True)
            
            # إضافة معلومات Footprint وSMC
            if cv.get('footprint', {}).get('ok'):
                fp = cv['footprint']
                print(f"👣 FOOTPRINT | Delta={fp['delta']:.0f} | CVD={fp['cumulative_delta']:.0f} | "
                      f"Spike={fp['volume_spike']} | AbsBull={fp['absorption_bull']} | AbsBear={fp['absorption_bear']}", flush=True)
            
            if cv.get('candles', {}).get('smc_pattern'):
                print(f"🕯️ SMC | {cv['candles']['smc_pattern']} | Trap={cv['candles']['liquidity_trap']}", flush=True)
            
            # OTC detailed info
            if cv.get('otc', {}).get('otc_buy') or cv.get('otc', {}).get('otc_sell'):
                otc = cv['otc']
                print(f"💰 OTC | {'BUY' if otc.get('otc_buy') else 'SELL'} | strength={otc.get('strength',0):.1f} | "
                      f"move={otc.get('move_bps',0):.1f}bps | flow={otc.get('visible_flow_ratio',0)*100:.1f}% | "
                      f"reason={otc.get('reason','')}", flush=True)
            
            # EMA detailed info
            if ema_ctx.get("label") != "none":
                print(f"📈 EMA CROSSOVER | {ema_ctx.get('label')} | score={ema_ctx.get('score',0):.1f} | side={ema_ctx.get('side','flat')}", flush=True)
            
            print("✅ ENHANCED ADDONS LIVE", flush=True)

        return {"bm": bm, "flow": flow, "cv": cv, "mode": mode, "gz": gz, "wallet": wallet}
    except Exception as e:
        print(f"🟨 AddonLog error: {e}", flush=True)
        return {"bm": None, "flow": None, "cv": {"b":0,"s":0,"score_b":0.0,"score_s":0.0,"ind":{}},
                "mode": {"mode":"n/a"}, "gz": None, "wallet": ""}

# =================== EXECUTION MANAGER ===================
def execute_trade_decision(side, price, qty, mode, council_data, gz_data):
    """تنفيذ قرار التداول مع التسجيل الواضح"""
    if not EXECUTE_ORDERS or DRY_RUN:
        log_i(f"DRY_RUN: {side} {qty:.4f} @ {price:.6f} | mode={mode}")
        return True
    
    if qty <= 0:
        log_e("❌ كمية غير صالحة للتنفيذ")
        return False

    gz_note = ""
    if gz_data and gz_data.get("ok"):
        gz_note = f" | 🟡 {gz_data['zone']['type']} s={gz_data['score']:.1f}"
    
    # OTC note
    otc_note = ""
    if council_data.get("otc", {}).get("otc_buy"):
        otc_note = f" | 💰 OTC BUY s={council_data['otc'].get('strength',0):.1f}"
    elif council_data.get("otc", {}).get("otc_sell"):
        otc_note = f" | 💰 OTC SELL s={council_data['otc'].get('strength',0):.1f}"
    
    # EMA note
    ema_note = ""
    ema_ctx = council_data.get("ema", {})
    if ema_ctx.get("label") != "none":
        ema_note = f" | 📈 EMA:{ema_ctx.get('label','')}"
    
    votes = council_data
    print(f"🎯 EXECUTE: {side.upper()} {qty:.4f} @ {price:.6f} | "
          f"mode={mode} | votes={votes['b']}/{votes['s']} score={votes['score_b']:.1f}/{votes['score_s']:.1f}"
          f"{gz_note}{otc_note}{ema_note}", flush=True)

    try:
        if MODE_LIVE:
            ex.set_leverage(LEVERAGE, SYMBOL, params={"side": "BOTH"})
            ex.create_order(SYMBOL, "market", side, qty, None, _params_open(side))
        
        log_g(f"✅ EXECUTED: {side.upper()} {qty:.4f} @ {price:.6f}")
        return True
    except Exception as e:
        log_e(f"❌ EXECUTION FAILED: {e}")
        return False

def setup_trade_management(mode):
    """تهيئة إدارة الصفقة حسب النمط"""
    if mode == "scalp":
        return {
            "tp1_pct": SCALP_TP1 / 100.0,
            "be_activate_pct": SCALP_BE_AFTER / 100.0,
            "trail_activate_pct": 0.8 / 100.0,
            "atr_trail_mult": SCALP_ATR_MULT,
            "close_aggression": "high"
        }
    else:
        return {
            "tp1_pct": TREND_TP1 / 100.0,
            "be_activate_pct": TREND_BE_AFTER / 100.0,
            "trail_activate_pct": 1.2 / 100.0,
            "atr_trail_mult": TREND_ATR_MULT,
            "close_aggression": "medium"
        }

# =================== ENHANCED TRADE EXECUTION ===================
def open_market_enhanced(side, qty, price):
    if qty <= 0: 
        log_e("skip open (qty<=0)")
        return False
    
    df = fetch_ohlcv()
    current_price = price or float(df['close'].iloc[-1])
    
    # Enhanced analysis
    snap = emit_snapshots(ex, SYMBOL, df)
    votes = snap["cv"]
    footprint = votes.get("footprint", {})
    ema_ctx = votes.get("ema", {})
    
    mode_data = decide_strategy_mode_enhanced(df, 
                                   adx=votes["ind"].get("adx"),
                                   di_plus=votes["ind"].get("plus_di"),
                                   di_minus=votes["ind"].get("minus_di"),
                                   rsi_ctx=rsi_ma_context(df),
                                   footprint=footprint,
                                   ema_ctx=ema_ctx)
    
    mode = mode_data["mode"]
    gz = snap["gz"]
    
    # Enhanced management config
    management_config = setup_trade_management(mode)
    
    success = execute_trade_decision(side, price, qty, mode, votes, gz)
    
    if success:
        # نحسب قوة الإشارة
        side_label = "long" if side == "buy" else "short"
        signal_strength = calculate_signal_strength(df, votes["ind"], side_label)
        
        # نحدّد Profile جني الأرباح (سكالب خفيف / نص ترند / ترند قوي)
        trade_profile = classify_trade_profile(signal_strength, mode)
        
        # تحديث Risk Management
        risk_on_new_trade(STATE, side_label, price)
        
        STATE.update({
            "open": True, 
            "side": side_label, 
            "entry": price,
            "qty": qty, 
            "pnl": 0.0, 
            "bars": 0, 
            "trail": None, 
            "breakeven": None,
            "tp1_done": False, 
            "highest_profit_pct": 0.0, 
            "profit_targets_achieved": 0,
            "mode": mode,
            "management": management_config,
            "signal_strength": signal_strength,
            "trade_profile": trade_profile,
        })
        
        save_state({
            "in_position": True,
            "side": "LONG" if side.upper().startswith("B") else "SHORT",
            "entry_price": price,
            "position_qty": qty,
            "leverage": LEVERAGE,
            "mode": mode,
            "management": management_config,
            "signal_strength": signal_strength,
            "trade_profile": trade_profile,
            "gz_snapshot": gz if isinstance(gz, dict) else {},
            "cv_snapshot": votes if isinstance(votes, dict) else {},
            "footprint_snapshot": footprint if isinstance(footprint, dict) else {},
            "ema_snapshot": ema_ctx if isinstance(ema_ctx, dict) else {},
            "opened_at": int(time.time()),
            "partial_taken": False,
            "breakeven_armed": False,
            "trail_active": False,
            "trail_tightened": False,
            "last_status_log_ts": 0.0,
        })

        # === لوج افتتاح الصفقة + خطة جني الأرباح ===
        mgmt = management_config or {}
        tp1_pct           = mgmt.get("tp1_pct", TP1_PCT_BASE / 100.0) * 100.0
        be_activate_pct   = mgmt.get("be_activate_pct", BREAKEVEN_AFTER / 100.0) * 100.0
        trail_activate_pct= mgmt.get("trail_activate_pct", TRAIL_ACTIVATE_PCT / 100.0) * 100.0
        atr_trail_mult    = mgmt.get("atr_trail_mult", ATR_TRAIL_MULT)

        side_txt = "BUY" if side == "buy" else "SELL"

        log_banner("NEW POSITION OPENED")
        log_g(
            f"📌 OPEN {side_txt} | mode={mode} | qty={qty:.4f} | entry={price:.6f} | "
            f"signal_strength={signal_strength:.1f}"
        )
        log_i(
            f"🎯 PROFIT PLAN | TP1≈{tp1_pct:.2f}% | BE@{be_activate_pct:.2f}% | "
            f"TRAIL@{trail_activate_pct:.2f}% ATR×{atr_trail_mult:.2f}"
        )
        
        log_g(f"✅ ENHANCED POSITION OPENED: {side.upper()} | mode={mode} | signal_strength={signal_strength:.1f}")
        return True
    
    return False

# =================== INDICATORS ===================
def wilder_ema(s: pd.Series, n: int): 
    return s.ewm(alpha=1/n, adjust=False).mean()

def compute_indicators(df: pd.DataFrame):
    if len(df) < max(ATR_LEN, RSI_LEN, ADX_LEN) + 2:
        return {
            "rsi": 50.0, "plus_di": 0.0, "minus_di": 0.0,
            "dx": 0.0, "adx": 0.0, "atr": 0.0,
            "di_spread": 0.0, "vwap": None
        }

    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["volume"].astype(float)

    # ATR
    tr = pd.concat([(h - l).abs(),
                    (h - c.shift(1)).abs(),
                    (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = wilder_ema(tr, ATR_LEN)

    # RSI
    delta = c.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    rs = wilder_ema(up, RSI_LEN) / wilder_ema(dn, RSI_LEN).replace(0, 1e-12)
    rsi = 100 - (100 / (1 + rs))

    # ADX / DI
    up_move = h.diff()
    down_move = l.shift(1) - l
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    plus_di = 100 * (wilder_ema(plus_dm, ADX_LEN) / atr.replace(0, 1e-12))
    minus_di = 100 * (wilder_ema(minus_dm, ADX_LEN) / atr.replace(0, 1e-12))
    dx = (100 * (plus_di - minus_di).abs() /
          (plus_di + minus_di).replace(0, 1e-12)).fillna(0.0)
    adx = wilder_ema(dx, ADX_LEN)

    # VWAP (session-style على كامل الـ df)
    typical_price = (h + l + c) / 3.0
    pv = typical_price * v
    cum_pv = pv.cumsum()
    cum_vol = v.cumsum().replace(0, 1e-12)
    vwap_series = cum_pv / cum_vol

    i = len(df) - 1
    di_spread = float(abs(plus_di.iloc[i] - minus_di.iloc[i]))
    vwap_val = float(vwap_series.iloc[i]) if not vwap_series.empty else None

    return {
        "rsi": float(rsi.iloc[i]),
        "plus_di": float(plus_di.iloc[i]),
        "minus_di": float(minus_di.iloc[i]),
        "dx": float(dx.iloc[i]),
        "adx": float(adx.iloc[i]),
        "atr": float(atr.iloc[i]),
        "di_spread": di_spread,
        "vwap": vwap_val,
    }

# =================== RANGE FILTER ===================
def _rng_size(src: pd.Series, qty: float, n: int) -> pd.Series:
    avrng = _ema((src - src.shift(1)).abs(), n); wper = (n*2)-1
    return _ema(avrng, wper) * qty

def _rng_filter(src: pd.Series, rsize: pd.Series):
    rf=[float(src.iloc[0])]
    for i in range(1,len(src)):
        prev=rf[-1]; x=float(src.iloc[i]); r=float(rsize.iloc[i]); cur=prev
        if x - r > prev: cur = x - r
        if x + r < prev: cur = x + r
        rf.append(cur)
    filt=pd.Series(rf, index=src.index, dtype="float64")
    return filt + rsize, filt - rsize, filt

def _ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rf_signal_live(df: pd.DataFrame):
    if len(df) < RF_PERIOD + 3:
        i = -1
        price = float(df["close"].iloc[i]) if len(df) else None
        return {"time": int(df["time"].iloc[i]) if len(df) else int(time.time()*1000),
                "price": price or 0.0, "long": False, "short": False,
                "filter": price or 0.0, "hi": price or 0.0, "lo": price or 0.0}
    src = df[RF_SOURCE].astype(float)
    hi, lo, filt = _rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))
    def _bps(a,b):
        try: return abs((a-b)/b)*10000.0
        except Exception: return 0.0
    p_now = float(src.iloc[-1]); p_prev = float(src.iloc[-2])
    f_now = float(filt.iloc[-1]); f_prev = float(filt.iloc[-2])
    long_flip  = (p_prev <= f_prev and p_now > f_now and _bps(p_now, f_now) >= RF_HYST_BPS)
    short_flip = (p_prev >= f_prev and p_now < f_now and _bps(p_now, f_now) >= RF_HYST_BPS)
    return {
        "time": int(df["time"].iloc[-1]), "price": p_now,
        "long": bool(long_flip), "short": bool(short_flip),
        "filter": f_now, "hi": float(hi.iloc[-1]), "lo": float(lo.iloc[-1])
    }

# =================== STATE ===================
STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
    "tp1_done": False, "highest_profit_pct": 0.0,
    "profit_targets_achieved": 0,
    
    # نمط الصفقة (لجني الأرباح): SCALP_LIGHT / MID_TREND / STRONG_TREND
    "trade_profile": None,
    "signal_strength": 0.0,
    
    # --- Risk / Guards ---
    "max_loss_price": None,           # سعر ستوب الصفقة الحالية
    "last_closed_pnl_pct": 0.0,       # نسبة ربح/خسارة آخر صفقة مغلقة
    "post_big_win_mode": False,       # هل إحنا في وضع حماية بعد مكسب كبير؟
    "post_big_win_bars_left": 0,      # عدد الشموع المتبقية في وضع Post Big Win
}
compound_pnl = 0.0
wait_for_next_signal_side = None

# =================== WAIT FOR NEXT SIGNAL ===================
def _arm_wait_after_close(prev_side):
    """تفعيل انتظار الإشارة التالية بعد الإغلاق"""
    global wait_for_next_signal_side
    wait_for_next_signal_side = "sell" if prev_side=="long" else ("buy" if prev_side=="short" else None)
    log_i(f"🛑 WAIT FOR NEXT SIGNAL: {wait_for_next_signal_side}")

def wait_gate_allow(df, info):
    """التحقق من بوابة الانتظار"""
    if wait_for_next_signal_side is None: 
        return True, ""
    
    bar_ts = int(info.get("time") or 0)
    need = (wait_for_next_signal_side=="buy" and info.get("long")) or (wait_for_next_signal_side=="sell" and info.get("short"))
    
    if need:
        return True, ""
    return False, f"wait-for-next-RF({wait_for_next_signal_side})"

# =================== ORDERS ===================
def _params_open(side):
    if POSITION_MODE == "hedge":
        return {"positionSide": "LONG" if side=="buy" else "SHORT", "reduceOnly": False}
    return {"positionSide": "BOTH", "reduceOnly": False}

def _params_close():
    if POSITION_MODE == "hedge":
        return {"positionSide": "LONG" if STATE.get("side")=="long" else "SHORT", "reduceOnly": True}
    return {"positionSide": "BOTH", "reduceOnly": True}

def _read_position():
    try:
        poss = ex.fetch_positions(params={"type":"swap"})
        for p in poss:
            sym = (p.get("symbol") or p.get("info",{}).get("symbol") or "")
            if SYMBOL.split(":")[0] not in sym: continue
            qty = abs(float(p.get("contracts") or p.get("info",{}).get("positionAmt") or 0))
            if qty <= 0: return 0.0, None, None
            entry = float(p.get("entryPrice") or p.get("info",{}).get("avgEntryPrice") or 0)
            side_raw = (p.get("side") or p.get("info",{}).get("positionSide") or "").lower()
            side = "long" if ("long" in side_raw or float(p.get("cost",0))>0) else "short"
            return qty, side, entry
    except Exception as e:
        logging.error(f"_read_position error: {e}")
    return 0.0, None, None

def compute_size(balance, price):
    effective = balance or 0.0
    capital = effective * RISK_ALLOC * LEVERAGE
    raw = max(0.0, capital / max(float(price or 0.0), 1e-9))
    return safe_qty(raw)

def close_market_strict(reason="STRICT"):
    global compound_pnl, wait_for_next_signal_side
    exch_qty, exch_side, exch_entry = _read_position()
    if exch_qty <= 0:
        if STATE.get("open"):
            _reset_after_close(reason)
        return
    side_to_close = "sell" if (exch_side=="long") else "buy"
    qty_to_close  = safe_qty(exch_qty)
    attempts=0; last_error=None
    while attempts < CLOSE_RETRY_ATTEMPTS:
        try:
            if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                params = _params_close(); params["reduceOnly"]=True
                ex.create_order(SYMBOL,"market",side_to_close,qty_to_close,None,params)
            time.sleep(CLOSE_VERIFY_WAIT_S)
            left_qty, _, _ = _read_position()
            if left_qty <= 0:
                px = price_now() or STATE.get("entry")
                entry_px = STATE.get("entry") or exch_entry or px
                side = STATE.get("side") or exch_side or ("long" if side_to_close=="sell" else "short")
                qty  = exch_qty
                pnl  = (px - entry_px) * qty * (1 if side=="long" else -1)
                compound_pnl += pnl
                
                # تحديث Risk Management
                realized_pnl_pct = compute_unrealized_pnl_pct(STATE, px)
                risk_on_trade_closed(STATE, realized_pnl_pct)
                
                log_i(f"STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}")
                logging.info(f"STRICT_CLOSE {side} pnl={pnl} total={compound_pnl}")
                _reset_after_close(reason, prev_side=side)
                return
            qty_to_close = safe_qty(left_qty)
            attempts += 1
            log_w(f"strict close retry {attempts}/{CLOSE_RETRY_ATTEMPTS} — residual={fmt(left_qty,4)}")
            time.sleep(CLOSE_VERIFY_WAIT_S)
        except Exception as e:
            last_error = e; logging.error(f"close_market_strict attempt {attempts+1}: {e}"); attempts += 1; time.sleep(CLOSE_VERIFY_WAIT_S)
    log_e(f"STRICT CLOSE FAILED after {CLOSE_RETRY_ATTEMPTS} attempts — last error: {last_error}")
    logging.critical(f"STRICT CLOSE FAILED — last_error={last_error}")

def _reset_after_close(reason, prev_side=None):
    """إعادة تعيين الحالة بعد الإغلاق"""
    global wait_for_next_signal_side
    
    prev_side = prev_side or STATE.get("side")
    
    # تحديث Risk Management لو كان هناك صفقة مفتوحة
    if STATE["open"] and STATE["entry"]:
        current_price = price_now() or STATE["entry"]
        realized_pnl_pct = compute_unrealized_pnl_pct(STATE, current_price)
        risk_on_trade_closed(STATE, realized_pnl_pct)
    
    STATE.update({
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
        "tp1_done": False, "highest_profit_pct": 0.0, "profit_targets_achieved": 0,
        "trail_tightened": False, "partial_taken": False,
        "max_loss_price": None,  # إعادة تعيين Stop Loss
        "trade_profile": None,
        "signal_strength": 0.0,
        "last_status_log_ts": 0.0,
    })
    save_state({"in_position": False, "position_qty": 0})
    
    # تفعيل انتظار الإشارة التالية
    _arm_wait_after_close(prev_side)
    logging.info(f"AFTER_CLOSE waiting_for={wait_for_next_signal_side}")

# =================== SMART EXIT GUARD ===================
def smart_exit_guard(state, df, ind, flow, bm, now_price, pnl_pct, mode, side, entry_price, gz=None):
    """يقرر: Partial / Tighten / Strict Close مع لوج واضح."""
    atr = ind.get('atr', 0.0)
    adx = ind.get('adx', 0.0)
    rsi = ind.get('rsi', 50.0)
    rsi_ma = ind.get('rsi_ma', 50.0)
    
    if len(df) >= 3:
        adx_slope = adx - ind.get('adx_prev', adx)
    else:
        adx_slope = 0.0

    # حساب الفتائل
    wick_signal = False
    if len(df) > 0:
        c = df.iloc[-1]
        wick_up = float(c['high']) - max(float(c['close']), float(c['open']))
        wick_down = min(float(c['close']), float(c['open'])) - float(c['low'])
        wick_signal = (wick_up >= WICK_ATR_MULT * atr) if side == "long" else (wick_down >= WICK_ATR_MULT * atr)

    rsi_cross_down = (rsi < rsi_ma) if side == "long" else (rsi > rsi_ma)
    adx_falling = (adx_slope < 0)
    cvd_down = (flow and flow.get('ok') and flow.get('cvd_trend') == 'down')
    evx_spike = False  # يمكن إضافة حساب EVX لاحقًا
    
    bm_wall_close = False
    if bm and bm.get('ok'):
        if side == "long":
            sell_walls = bm.get('sell_walls', [])
            if sell_walls:
                best_ask = min([p for p, _ in sell_walls])
                bps = abs((best_ask - now_price) / now_price) * 10000.0
                bm_wall_close = (bps <= BM_WALL_PROX_BPS)
        else:
            buy_walls = bm.get('buy_walls', [])
            if buy_walls:
                best_bid = max([p for p, _ in buy_walls])
                bps = abs((best_bid - now_price) / now_price) * 10000.0
                bm_wall_close = (bps <= BM_WALL_PROX_BPS)

    # --- Golden Reversal بعد TP1 ---
    if state.get('tp1_done') and (gz and gz.get('ok')):
        # إغلاق صارم لو تقاطع Golden عكس اتجاهي بعد TP1
        opp = (gz['zone']['type']=='golden_top' and side=='long') or (gz['zone']['type']=='golden_bottom' and side=='short')
        if opp and gz.get('score',0) >= GOLDEN_REVERSAL_SCORE:
            return {
                "action": "close", 
                "why": "golden_reversal",
                "log": f"🔴 CLOSE STRONG | golden reversal after TP1 | score={gz['score']:.1f}"
            }

    # --- OTC Reversal بعد TP1 (سيولة مخفية عكسية) ---
    # ملاحظة: هنا pnl_pct = كسر (0.01 = 1%)
    if state.get('tp1_done') and pnl_pct >= OTC_EXIT_MIN_PNL_PCT:
        try:
            otc = detect_otc_flows(df)
        except Exception as e:
            otc = {"otc_buy": False, "otc_sell": False, "strength": 0.0, "reason": f"error:{e}"}

        opp_otc = False
        if side == "long" and otc.get("otc_sell"):
            opp_otc = True
        elif side == "short" and otc.get("otc_buy"):
            opp_otc = True

        if opp_otc and otc.get("strength", 0.0) >= OTC_EXIT_MIN_STRENGTH:
            return {
                "action": "close",
                "why": "otc_reversal",
                "log": (
                    f"🔴 CLOSE STRONG | OTC reversal after TP1 | "
                    f"side={side} pnl={pnl_pct*100:.2f}% "
                    f"strength={otc.get('strength',0):.1f} "
                    f"move={otc.get('move_bps',0):.1f}bps "
                    f"flow={otc.get('visible_flow_ratio',0)*100:.1f}% "
                    f"reason={otc.get('reason','')}"
                )
            }

    tp1_target = TP1_SCALP_PCT if mode == 'scalp' else TP1_TREND_PCT
    if pnl_pct >= tp1_target and not state.get('tp1_done'):
        qty_pct = 0.35 if mode == 'scalp' else 0.25
        return {
            "action": "partial", 
            "why": f"TP1 hit {tp1_target*100:.2f}%",
            "qty_pct": qty_pct,
            "log": f"💰 TP1 جزئي {tp1_target*100:.2f}% | pnl={pnl_pct*100:.2f}% | mode={mode}"
        }

    # --- Wick exhaustion + Tighten عند إجهاد/تدفق/جدار ---
    if pnl_pct > 0:
        if wick_signal or evx_spike or bm_wall_close or cvd_down:
            return {
                "action": "tighten", 
                "why": "exhaustion/flow/wall",
                "trail_mult": TRAIL_TIGHT_MULT,
                "log": f"🛡️ Tighten | wick={int(bool(wick_signal))} evx={int(bool(evx_spike))} wall={bm_wall_close} cvd_down={cvd_down}"
            }

    bearish_signals = [rsi_cross_down, adx_falling, cvd_down, evx_spike, bm_wall_close]
    bearish_count = sum(bearish_signals)
    
    if pnl_pct >= HARD_CLOSE_PNL_PCT and bearish_count >= 2:
        reasons = []
        if rsi_cross_down: reasons.append("rsi↓")
        if adx_falling: reasons.append("adx↓")
        if cvd_down: reasons.append("cvd↓")
        if evx_spike: reasons.append("evx")
        if bm_wall_close: reasons.append("wall")
        
        return {
            "action": "close", 
            "why": "hard_close_signal",
            "log": f"🔴 CLOSE STRONG | pnl={pnl_pct*100:.2f}% | {', '.join(reasons)}"
        }

    return {
        "action": "hold", 
        "why": "keep_riding", 
        "log": None
    }

# =================== ENHANCED TRADE MANAGEMENT ===================
def manage_after_entry_enhanced(df, ind, info):
    """إدارة محسنة للمركز مع Smart Profit AI + Smart Exit Guard + Hard Stop"""
    if not STATE["open"] or STATE["qty"] <= 0:
        return

    px   = info["price"]
    entry = STATE["entry"]
    side  = STATE["side"]
    qty   = STATE["qty"]
    mode  = STATE.get("mode", "trend")
    
    # ========= HARD MAX LOSS GUARD (ستوب صارم) =========
    max_loss_price = STATE.get("max_loss_price")
    if max_loss_price:
        if side == "long" and px <= max_loss_price:
            pnl_pct = compute_unrealized_pnl_pct(STATE, px)
            log_w(f"🛑 MAX-LOSS GUARD HIT (LONG) @ {px:.6f} | Loss: {pnl_pct*100:.2f}%")
            close_market_strict("max_loss_guard_long")
            risk_on_trade_closed(STATE, pnl_pct)
            return
        elif side == "short" and px >= max_loss_price:
            pnl_pct = compute_unrealized_pnl_pct(STATE, px)
            log_w(f"🛑 MAX-LOSS GUARD HIT (SHORT) @ {px:.6f} | Loss: {pnl_pct*100:.2f}%")
            close_market_strict("max_loss_guard_short")
            risk_on_trade_closed(STATE, pnl_pct)
            return
    
    # PnL % (كـ نسبة مئوية)
    pnl_pct = (px - entry) / entry * 100.0 * (1 if side == "long" else -1)
    STATE["pnl"] = pnl_pct

    if pnl_pct > STATE.get("highest_profit_pct", 0.0):
        STATE["highest_profit_pct"] = pnl_pct

    # ========= Smart Profit AI (سلم جني الأرباح الذكي) =========
    profit_decision = smart_profit_ai_decision(STATE, df, ind, mode, side, entry, px)
    
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

    # === لوج حالة الصفقة كل 30 ثانية ===
    now_ts = time.time()
    last_ts = STATE.get("last_status_log_ts", 0.0)

    if now_ts - last_ts >= POSITION_STATUS_LOG_INTERVAL:
        STATE["last_status_log_ts"] = now_ts

        management = STATE.get("management", {}) or {}
        tp1_pct            = management.get("tp1_pct", TP1_PCT_BASE / 100.0) * 100.0
        be_activate_pct    = management.get("be_activate_pct", BREAKEVEN_AFTER / 100.0) * 100.0
        trail_activate_pct = management.get("trail_activate_pct", TRAIL_ACTIVATE_PCT / 100.0) * 100.0
        atr_trail_mult     = management.get("atr_trail_mult", ATR_TRAIL_MULT)

        mode = STATE.get("mode", "scalp")
        max_pnl = STATE.get("highest_profit_pct", 0.0)
        targets_done = STATE.get("profit_targets_achieved", 0)
        side_txt = "BUY" if side == "long" else "SELL"

        log_i(
            f"📊 POSITION STATUS | {side_txt} | mode={mode} | "
            f"price={px:.6f} | entry={entry:.6f} | qty={qty:.4f}"
        )
        log_i(
            f"   PnL={pnl_pct:.2f}% (max={max_pnl:.2f}%) | TP_done={targets_done} | "
            f"plan: TP1≈{tp1_pct:.2f}% / BE@{be_activate_pct:.2f}% / "
            f"TRAIL@{trail_activate_pct:.2f}% ATR×{atr_trail_mult:.2f}"
        )

    # ========= الإدارة الكلاسيك (TP1 + BE + Trail + Dust) =========
    current_atr      = ind.get("atr", 0.0)
    management       = STATE.get("management", {})
    
    tp1_pct          = management.get("tp1_pct", TP1_PCT_BASE/100.0)
    be_activate_pct  = management.get("be_activate_pct", BREAKEVEN_AFTER/100.0)
    trail_activate_pct = management.get("trail_activate_pct", TRAIL_ACTIVATE_PCT/100.0)
    atr_trail_mult   = management.get("atr_trail_mult", ATR_TRAIL_MULT)

    # نحول PnL من % إلى كسور عشان نستخدمه مع الحراس اللي شغّالين بالـ fraction
    pnl_frac = pnl_pct / 100.0

    # TP1 جزئي (مرة واحدة فقط)
    if not STATE.get("tp1_done") and pnl_frac >= tp1_pct:
        close_fraction = TP1_CLOSE_FRAC
        close_qty = safe_qty(STATE["qty"] * close_fraction)
        if close_qty > 0:
            close_side = "sell" if STATE["side"] == "long" else "buy"
            if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                try:
                    ex.create_order(SYMBOL, "market", close_side, close_qty, None, _params_close())
                    log_g(f"✅ TP1 HIT: closed {close_fraction*100:.0f}%")
                except Exception as e:
                    log_e(f"❌ TP1 close failed: {e}")
            STATE["qty"] = safe_qty(STATE["qty"] - close_qty)
            STATE["tp1_done"] = True
            STATE["profit_targets_achieved"] += 1

    # تفعيل Breakeven
    if not STATE.get("breakeven_armed") and pnl_frac >= be_activate_pct:
        STATE["breakeven_armed"] = True
        STATE["breakeven"] = entry
        log_i("BREAKEVEN ARMED")

    # تفعيل التريل
    if not STATE.get("trail_active") and pnl_frac >= trail_activate_pct:
        STATE["trail_active"] = True
        log_i("TRAIL ACTIVATED")

    # تحديث مستوى التريل
    if STATE.get("trail_active"):
        trail_mult = TRAIL_TIGHT_MULT if STATE.get("trail_tightened") else atr_trail_mult
        if side == "long":
            new_trail = px - (current_atr * trail_mult)
            if STATE.get("trail") is None or new_trail > STATE["trail"]:
                STATE["trail"] = new_trail
        else:
            new_trail = px + (current_atr * trail_mult)
            if STATE.get("trail") is None or new_trail < STATE["trail"]:
                STATE["trail"] = new_trail

    # تنفيذ وقف التريل
    if STATE.get("trail"):
        if (side == "long" and px <= STATE["trail"]) or (side == "short" and px >= STATE["trail"]):
            log_w(f"TRAIL STOP: {px} vs trail {STATE['trail']}")
            close_market_strict("trail_stop")
            return

    # تنفيذ Breakeven الصارم
    if STATE.get("breakeven"):
        if (side == "long" and px <= STATE["breakeven"]) or (side == "short" and px >= STATE["breakeven"]):
            log_w(f"BREAKEVEN STOP: {px} vs breakeven {STATE['breakeven']}")
            close_market_strict("breakeven_stop")
            return

    # Dust guard: لو الكمية بقت فتات اقفل وخلاص
    if STATE["qty"] <= FINAL_CHUNK_QTY:
        log_w(f"DUST GUARD: qty {STATE['qty']} <= {FINAL_CHUNK_QTY}, closing...")
        close_market_strict("dust_guard")
        return

    # ========= Smart Exit Guard (Golden Reversal + Wick/Flow/Wall + OTC) =========
    try:
        guard = smart_exit_guard(
            STATE,
            df,
            ind,
            info.get("flow"),
            info.get("bm"),
            px,
            pnl_frac,           # هنا بنمررها كـ fraction (0.01 = 1%)
            mode,
            side,
            entry,
            gz=STATE.get("gz_snapshot", {})
        )
    except Exception as e:
        log_w(f"smart_exit_guard error: {e}")
        guard = None

    if guard and guard.get("action") != "hold":
        if guard.get("log"):
            log_w(guard["log"])
        act = guard["action"]

        # إحكام التريل عند إجهاد / جدار / تدفق معاكس
        if act == "tighten":
            STATE["trail_tightened"] = True

        # إغلاق صارم عند Golden Reversal أو Hard Close أو OTC Reversal
        elif act == "close":
            close_market_strict(guard.get("why", "smart_exit_guard"))
            return
        # متعمدين نتجاهل "partial" هنا عشان ما نعملش TP1 مزدوج (Smart AI + Guard)

# =================== ENHANCED TRADE LOOP ===================
def trade_loop_enhanced():
    """حلقة تداول محسنة مع Golden Zone Pro وSmart Profit AI وVWAP وOTC Detection وEMA Cross + Risk Guards"""
    global wait_for_next_signal_side
    loop_i = 0
    
    while True:
        try:
            # تحديث عدّاد Post Big Win كل لوب
            tick_post_big_win_decay(STATE)
            
            # جمع البيانات الأساسية
            bal = balance_usdt()
            px = price_now()
            df = fetch_ohlcv()
            info = rf_signal_live(df)
            ind = compute_indicators(df)
            spread_bps = orderbook_spread_bps()
            
            # Enhanced Snapshots
            snap = emit_snapshots(ex, SYMBOL, df,
                                balance_fn=lambda: float(bal) if bal else None,
                                pnl_fn=lambda: float(compound_pnl))
            
            # تحديث حالة الربح/الخسارة
            if STATE["open"] and px:
                STATE["pnl"] = (px-STATE["entry"])*STATE["qty"] if STATE["side"]=="long" else (STATE["entry"]-px)*STATE["qty"]
            
            # إدارة الصفقة المفتوحة مع Smart Profit AI + Hard Stop
            if STATE["open"]:
                manage_after_entry_enhanced(df, ind, {
                    "price": px or info["price"], 
                    "bm": snap["bm"],
                    "flow": snap["flow"],
                    **info
                })
            
            # قرار الدخول المحسن
            reason = None
            if spread_bps is not None and spread_bps > MAX_SPREAD_BPS:
                reason = f"spread too high ({fmt(spread_bps,2)}bps > {MAX_SPREAD_BPS})"
            
            council_data = council_votes_pro_enhanced(df)
            gz = council_data.get("gz")
            footprint = council_data.get("footprint", {})
            otc = council_data.get("otc", {})
            ema_ctx = council_data.get("ema", {})
            sig = None

            # ===== POST-BIG-WIN FILTER =====
            if STATE.get("post_big_win_mode") and not STATE["open"]:
                # فلترة الإشارات في وضع Post-Big-Win
                if STATE["post_big_win_bars_left"] > 0:
                    log_i(f"🛡️ POST-BIG-WIN MODE ACTIVE: {STATE['post_big_win_bars_left']} bars remaining - Applying strict filters")

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

            # --- OTC Enhanced Entry ---
            if not golden_entry and otc:
                if otc.get("otc_buy") and otc.get("strength", 0) >= 2.0:
                    sig = "buy"
                    log_i(f"💰 OTC ENTRY: BUY | strength={otc.get('strength',0):.1f} | سيولة شراء مخفية قوية")
                elif otc.get("otc_sell") and otc.get("strength", 0) >= 2.0:
                    sig = "sell"
                    log_i(f"💰 OTC ENTRY: SELL | strength={otc.get('strength',0):.1f} | سيولة بيع مخفية قوية")

            # --- Council Strong Entry مع تخفيف احترافي للتقاطع القوي ---
            if not golden_entry and not sig:
                ema_label = ema_ctx.get("label", "none")
                strong_bull = (ema_label == "strong_bull")
                strong_bear = (ema_label == "strong_bear")

                buy_score  = council_data["score_b"]
                sell_score = council_data["score_s"]

                buy_ok = False
                sell_ok = False

                # الأساس: 8.0 زي ما هو
                if buy_score >= 8.0 and buy_score > sell_score:
                    buy_ok = True
                elif sell_score >= 8.0 and sell_score > buy_score:
                    sell_ok = True

                # ===== POST-BIG-WIN FILTER =====
                if STATE.get("post_big_win_mode"):
                    # في وضع Post-Big-Win نرفع السقف
                    if buy_score >= 8.5 and buy_score > sell_score and strong_bull:
                        buy_ok = True
                        log_i(f"✅ BUY by Council(8.5+) + EMA STRONG BULL ({buy_score:.1f}) - POST-BIG-WIN MODE")
                    elif sell_score >= 8.5 and sell_score > buy_score and strong_bear:
                        sell_ok = True
                        log_i(f"✅ SELL by Council(8.5+) + EMA STRONG BEAR ({sell_score:.1f}) - POST-BIG-WIN MODE")
                    else:
                        log_i(f"🛑 POST-BIG-WIN MODE: Skipping weak signal (Buy: {buy_score:.1f}, Sell: {sell_score:.1f})")
                else:
                    # تخفيف احترافي: 7.0 مسموح فقط لو EMA strong في نفس الاتجاه
                    if not buy_ok and buy_score >= 7.0 and buy_score > sell_score and strong_bull:
                        buy_ok = True
                        log_i(f"✅ BUY by Council(7.x) + EMA STRONG BULL ({buy_score:.1f})")
                    if not sell_ok and sell_score >= 7.0 and sell_score > buy_score and strong_bear:
                        sell_ok = True
                        log_i(f"✅ SELL by Council(7.x) + EMA STRONG BEAR ({sell_score:.1f})")

                if buy_ok:
                    sig = "buy"
                elif sell_ok:
                    sig = "sell"
            
            if not STATE["open"] and sig and reason is None:
                # تحديد نمط الصفقة (scalp / trend)
                mode_ctx = decide_strategy_mode_enhanced(
                    df,
                    ind=council_data["ind"],
                    rsi_ctx=rsi_ma_context(df),
                    footprint=footprint,
                    ema_ctx=ema_ctx,
                )
                trade_mode = mode_ctx.get("mode", "scalp")
                
                # ✅ حارس اتجاه السكالب
                side_label = "long" if sig == "buy" else "short"
                side_label, skip_scalp = enforce_scalp_trend_alignment(
                    df,
                    council_data["ind"],
                    side_label,
                    mode_ctx,
                )
                if skip_scalp:
                    # سكالب في سوق مش واضح → نسيبه
                    log_i("⏸️ Skipping scalp due to unclear trend alignment")
                    continue
                
                # تحديث السيجنال بعد التعديل
                sig = "buy" if side_label == "long" else "sell"
                
                allow_wait, wait_reason = wait_gate_allow(df, info)
                if not allow_wait:
                    reason = wait_reason
                else:
                    qty = compute_size(bal, px or info["price"])
                    if qty > 0:
                        ok = open_market_enhanced(sig, qty, px or info["price"])
                        if ok:
                            wait_for_next_signal_side = None
                            # تسجيل قرار المجلس المحسن
                            entry_type = "GOLDEN" if golden_entry else "OTC" if otc.get("otc_buy") or otc.get("otc_sell") else "COUNCIL"
                            if ema_label in ("strong_bull", "strong_bear"):
                                entry_type = f"EMA_STRONG_{entry_type}"
                            log_i(f"🎯 ENHANCED {entry_type} DECISION: {sig.upper()} | "
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

# استبدال الدوال الأساسية بالمحسنة
compute_candles = compute_enhanced_candles
council_votes_pro = council_votes_pro_enhanced
manage_after_entry = manage_after_entry_enhanced
open_market = open_market_enhanced
trade_loop = trade_loop_enhanced
decide_strategy_mode = decide_strategy_mode_enhanced
golden_zone_check = golden_zone_pro_analysis

# =================== LOOP / LOG ===================
def pretty_snapshot(bal, info, ind, spread_bps, reason=None, df=None):
    if LOG_LEGACY:
        left_s = time_to_candle_close(df) if df is not None else 0
        print(colored("─"*100,"cyan"))
        print(colored(f"📊 {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
        print(colored("─"*100,"cyan"))
        print("📈 INDICATORS & RF")
        print(f"   💲 Price {fmt(info.get('price'))} | RF filt={fmt(info.get('filter'))}  hi={fmt(info.get('hi'))} lo={fmt(info.get('lo'))}")
        print(f"   🧮 RSI={fmt(ind.get('rsi'))}  +DI={fmt(ind.get('plus_di'))}  -DI={fmt(ind.get('minus_di'))}  ADX={fmt(ind.get('adx'))}  ATR={fmt(ind.get('atr'))}")
        print(f"   🎯 ENTRY: COUNCIL PRO + GOLDEN ENTRY + VWAP STRATEGY + OTC DETECTION + EMA CROSSOVER ENGINE  |  spread_bps={fmt(spread_bps,2)}")
        print(f"   ⏱️ closes_in ≈ {left_s}s")
        
        # عرض معلومات الـ Guards
        print("\n🛡️ RISK GUARDS")
        if STATE["open"]:
            stop_price = STATE.get("max_loss_price")
            current_price = info.get("price") or 0
            if stop_price:
                stop_distance_pct = abs(stop_price - current_price) / current_price * 100
                stop_side = "BELOW" if STATE['side'] == 'long' else "ABOVE"
                print(f"   🔴 HARD STOP: {stop_side} {fmt(stop_price)} ({stop_distance_pct:.2f}%)")
        
        if STATE.get("post_big_win_mode"):
            print(f"   🛡️ POST-BIG-WIN: ACTIVE ({STATE.get('post_big_win_bars_left', 0)} bars left)")
        
        print("\n🧭 POSITION")
        bal_line = f"Balance={fmt(bal,2)}  Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x  CompoundPnL={fmt(compound_pnl)}  Eq~{fmt((bal or 0)+compound_pnl,2)}"
        print(colored(f"   {bal_line}", "yellow"))
        
        if STATE["open"]:
            lamp='🟩 LONG' if STATE['side']=='long' else '🟥 SHORT'
            print(f"   {lamp}  Entry={fmt(STATE['entry'])}  Qty={fmt(STATE['qty'],4)}  Bars={STATE['bars']}  Trail={fmt(STATE['trail'])}  BE={fmt(STATE['breakeven'])}")
            print(f"   🎯 TP_done={STATE['profit_targets_achieved']}  HP={fmt(STATE['highest_profit_pct'],2)}%")
        else:
            print("   ⚪ FLAT")
            if wait_for_next_signal_side:
                print(colored(f"   ⏳ Waiting for opposite RF: {wait_for_next_signal_side.upper()}", "cyan"))
        if reason: print(colored(f"   ℹ️ reason: {reason}", "white"))
        print(colored("─"*100,"cyan"))

# =================== API / KEEPALIVE ===================
app = Flask(__name__)
@app.route("/")
def home():
    mode='LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ Council PRO Bot — {SYMBOL} {INTERVAL} — {mode} — Enhanced Candles + Golden Zone Pro + Smart Profit AI + VWAP Strategy + OTC Detection + EMA Crossover Engine + Hard Stop Loss + Post-Big-Win Guard + Auto-Recovery"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "state": STATE, "compound_pnl": compound_pnl,
        "entry_mode": "COUNCIL_PRO_GOLDEN_ENHANCED_VWAP_OTC_EMA", "wait_for_next_signal": wait_for_next_signal_side,
        "risk_guards": {
            "hard_stop_pct": abs(MAX_LOSS_PCT)*100,
            "big_win_pct": BIG_WIN_PCT*100,
            "post_big_win_bars": POST_BIG_WIN_BARS,
            "max_spread_bps": MAX_SPREAD_BPS,
            "final_chunk_qty": FINAL_CHUNK_QTY
        },
        "vwap_strategy": {
            "enabled": VWAP_ENABLED,
            "scalp_band_bps": VWAP_SCALP_BAND_BPS,
            "trend_band_bps": VWAP_TREND_BAND_BPS
        },
        "otc_detection": {
            "enabled": True,
            "window_bars": OTC_WINDOW_BARS,
            "min_move_bps": OTC_MIN_MOVE_BPS,
            "exit_min_strength": OTC_EXIT_MIN_STRENGTH
        },
        "ema_crossover": {
            "enabled": True,
            "fast_period": 9,
            "mid_period": 21,
            "slow_period": 50
        },
        "current_guards": {
            "post_big_win_active": STATE.get("post_big_win_mode", False),
            "post_big_win_bars_left": STATE.get("post_big_win_bars_left", 0),
            "hard_stop_price": STATE.get("max_loss_price"),
            "last_closed_pnl_pct": STATE.get("last_closed_pnl_pct", 0.0)
        }
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True, "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"], "side": STATE["side"], "qty": STATE["qty"],
        "compound_pnl": compound_pnl, "timestamp": datetime.utcnow().isoformat(),
        "entry_mode": "COUNCIL_PRO_GOLDEN_ENHANCED_VWAP_OTC_EMA", "wait_for_next_signal": wait_for_next_signal_side,
        "risk_guards": {
            "hard_stop_active": STATE.get("max_loss_price") is not None,
            "post_big_win_active": STATE.get("post_big_win_mode", False)
        }
    }), 200

def keepalive_loop():
    url=(SELF_URL or "").strip().rstrip("/")
    if not url:
        log_w("keepalive disabled (SELF_URL not set)")
        return
    import requests
    sess=requests.Session(); sess.headers.update({"User-Agent":"rf-live-bot/keepalive"})
    log_i(f"KEEPALIVE every 50s → {url}")
    while True:
        try: sess.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

# =================== BOOT ===================
if __name__ == "__main__":
    log_banner("INIT")
    state = load_state() or {}
    state.setdefault("in_position", False)

    if RESUME_ON_RESTART:
        try:
            # استخدام الدالة المحسنة للاستئناف
            state = enhanced_resume_open_position(ex, SYMBOL, state)
        except Exception as e:
            log_w(f"🔄 Enhanced resume error: {e}\n{traceback.format_exc()}")

    verify_execution_environment()

    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'}  •  {SYMBOL}  •  {INTERVAL}", "yellow"))
    print(colored(f"RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x  •  COUNCIL_PRO=ENHANCED", "yellow"))
    print(colored(f"GOLDEN ENTRY PRO: score≥{GOLDEN_ENTRY_SCORE} | ADX≥{GOLDEN_ENTRY_ADX}", "yellow"))
    print(colored(f"ENHANCED CANDLES: SMC Patterns + Wick exhaustion + Golden reversal", "yellow"))
    print(colored(f"FOOTPRINT ANALYSIS: Volume spikes + Absorption detection", "yellow"))
    print(colored(f"OTC DETECTION: Hidden flow detection + Protection system", "yellow"))
    print(colored(f"SMART PROFIT AI: Dynamic profit taking + Signal strength", "yellow"))
    print(colored(f"VWAP STRATEGY: SCALP(near {VWAP_SCALP_BAND_BPS}bps) | TREND(far {VWAP_TREND_BAND_BPS}bps)", "yellow"))
    print(colored(f"EMA CROSSOVER ENGINE: Strong/Weak Trend Detection (9/21/50)", "yellow"))
    print(colored(f"🛡️ RISK GUARDS: Hard Stop Loss (-{abs(MAX_LOSS_PCT)*100}%) | Post-Big-Win Guard (+{BIG_WIN_PCT*100}%)", "red"))
    print(colored(f"🔄 AUTO RECOVERY: Enhanced position sync on restart", "green"))
    print(colored(f"EXECUTION: {'ACTIVE' if EXECUTE_ORDERS and not DRY_RUN else 'SIMULATION'}", "yellow"))
    
    logging.info("enhanced service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    
    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)

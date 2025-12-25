# -*- coding: utf-8 -*-
"""
RF Futures Bot — RF-LIVE ONLY (BingX Perp via CCXT)
• Council ELITE Unified Decision System with Smart Management
• Golden Entry + SMC/ICT + Smart Exit Management
• Dynamic TP ladder + Breakeven + ATR-trailing
• Professional Logging & Dashboard
• ENHANCED VERSION - More Trades & Faster Execution
• TREND BIRTH ENGINE v1 - اصطياد بدايات الترند
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
BOT_VERSION = "DOGE Council ELITE v6.0 — Enhanced Fast Trading + Trend Birth Engine"
print("🔁 Booting:", BOT_VERSION, flush=True)

STATE_PATH = "./bot_state.json"
RESUME_ON_RESTART = True
SAFE_RECONCILE = True
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

ENTRY_RF_ONLY = False

# =================== COUNCIL ELITE SETTINGS - ENHANCED ===================
# Council Weights & Gates - RELAXED FOR MORE TRADES
ADX_GATE = 12.0           # ⬇️ كان 17.0 - تخفيض 29%
ADX_TREND_MIN = 15.0      # ⬇️ كان 22.0 - تخفيض 32%
DI_SPREAD_TREND = 5.0     # ⬇️ كان 6.0 - تخفيض 17%
RSI_MA_LEN = 9

RSI_TREND_PERSIST = 5
RSI_NEUTRAL_BAND = (45, 55)

# Golden Zones - RELAXED
GZ_FIB_LOW = 0.618
GZ_FIB_HIGH = 0.786
GZ_MIN_SCORE = 3.0        # ⬇️ كان 6.0 - تخفيض 50%
GZ_ADX_MIN = 14.0         # ⬇️ كان 20.0 - تخفيض 30%
GOLDEN_ENTRY_SCORE = 3.0  # ⬇️ كان 6.0 - تخفيض 50%
GOLDEN_ENTRY_ADX = 14.0   # ⬇️ كان 20.0 - تخفيض 30%
GOLDEN_REVERSAL_SCORE = 4.0  # ⬇️ كان 6.5 - تخفيض 38%

# FVG/SMC - RELAXED
FVG_MIN_BPS = 6.0         # ⬇️ كان 8.0 - تخفيض 25%
BOS_MIN_PCT = 0.25        # ⬇️ كان 0.35 - تخفيض 29%
SWEEP_WICK_X_ATR = 1.0    # ⬇️ كان 1.2 - تخفيض 17%
OB_LOOKBACK = 35          # ⬇️ كان 40 - تخفيض 13%

# Flow/Bookmap
DELTA_Z_BULL = 0.40       # ⬇️ كان 0.50 - تخفيض 20%
DELTA_Z_BEAR = -0.40      # ⬇️ كان -0.50 - تخفيض 20%
IMB_ALERT = 1.15          # ⬇️ كان 1.20 - تخفيض 4%

# Management profiles
TP1_PCT_SCALP = 0.0040   # 0.40%
TP1_PCT_TREND = 0.0060   # 0.60%
BE_AFTER_SCALP = 0.0030  # 0.30%
BE_AFTER_TREND = 0.0040  # 0.40%
TRAIL_ACT_SCALP = 0.0080 # 0.80%
TRAIL_ACT_TREND = 0.0120 # 1.20%
ATR_TRAIL_MULT = 1.6
TRAIL_TIGHT_MULT = 1.2

# Decision thresholds - RELAXED FOR MORE TRADES
COUNCIL_STRONG_TH = 5.0   # ⬇️ كان 8.0 - تخفيض 37%
COUNCIL_OK_TH = 4.0       # ⬇️ كان 7.0 - تخفيض 43%

# Smart Exit Tuning
TP1_SCALP_PCT = 0.0035
TP1_TREND_PCT = 0.0060
HARD_CLOSE_PNL_PCT = 0.0110
WICK_ATR_MULT = 1.5
EVX_SPIKE = 1.8
BM_WALL_PROX_BPS = 5
TIME_IN_TRADE_MIN = 8

# Dust guard
FINAL_CHUNK_QTY = float(os.getenv("FINAL_CHUNK_QTY", 40.0))
RESIDUAL_MIN_QTY = float(os.getenv("RESIDUAL_MIN_QTY", 9.0))

# Strict close
CLOSE_RETRY_ATTEMPTS = 6
CLOSE_VERIFY_WAIT_S = 2.0

# Pacing - FASTER FOR MORE TRADES
BASE_SLEEP = 2           # ⬇️ كان 5 ثواني - تخفيض 60%
NEAR_CLOSE_S = 0.5       # ⬇️ كان 1 ثانية - تخفيض 50%

# Spread - RELAXED FOR MORE TRADES
MAX_SPREAD_BPS = 15.0    # ⬆️ كان 6.0 - زيادة 150%

# =================== TREND BIRTH ENGINE SETTINGS ===================
TBE_ENABLED = True  # تفعيل نظام اصطياد بدايات الترند
TBE_SWEEP_ATR_MULT = 0.15  # نسبة ATR للكشف عن Sweep (0.15 = 15%)
TBE_BOS_MIN_PCT = 0.12     # نسبة كسر الهيكل
TBE_ZONE_TTL_BARS = 20     # صلاحية المنطقة (عدد الشمعات)
TBE_FAILFAST_BARS = 3      # عدد الشمعات للكشف عن الدخول الخاطئ
TBE_REHUNT_COOLDOWN_BARS = 6  # تبريد بعد إغلاق خاطئ

# شروط الدخول
TBE_ENTRY_SCORE_MIN = 7.0   # الحد الأدنى لدرجة الدخول
TBE_RSI_CROSS_WEIGHT = 2.0  # وزن تقاطع RSI
TBE_FLOW_WEIGHT = 1.5       # وزن مؤشرات التدفق
TBE_HTF_WEIGHT = 2.0        # وزن الإطار الزمني الأعلى

# =================== FAST TRADING SETTINGS ===================
FAST_TRADE_ENABLED = True
FAST_MIN_SCORE = 3.0
FAST_MAX_HOLD_BARS = 3

# =================== PROFESSIONAL LOGGING ===================
def log_i(msg): print(f"ℹ️ {msg}", flush=True)
def log_g(msg): print(f"✅ {msg}", flush=True)
def log_w(msg): print(f"🟨 {msg}", flush=True)
def log_e(msg): print(f"❌ {msg}", flush=True)

def log_banner(text): print(f"\n{'—'*12} {text} {'—'*12}\n", flush=True)

# =============== TRADE OPEN LOG (BUY=🟢 / SELL=🔴) ===============
def log_trade_open(*, side:str, price:float, qty:float, leverage:int,
                   source:str, mode:str, risk_alloc:float,
                   council:dict=None, gz:dict=None, mgmt:dict=None,
                   tbe_data:dict=None):
    lamp = "🟢 BUY" if side.lower().startswith("b") else "🔴 SELL"
    p = f"{float(price):.6f}"
    q = f"{float(qty):.4f}"
    lev = f"{int(leverage)}x"
    ra = f"{int(risk_alloc*100)}%"

    c_part = ""
    if council:
        c_part = f" | Council B/S={council.get('score_b',0):.1f}/{council.get('score_s',0):.1f} votes={council.get('b',0)}/{council.get('s',0)}"

    gz_part = ""
    if gz and gz.get("ok"):
        gz_part = f" | GZ={gz['zone']['type']} s={gz.get('score',0):.1f}"

    mg = mgmt or {}
    tp1 = mg.get("tp1_pct"); bea = mg.get("be_activate_pct"); tra = mg.get("trail_activate_pct"); atrx = mg.get("atr_trail_mult")
    mg_part = ""
    if any(v is not None for v in (tp1, bea, tra)):
        mg_part = " | MGMT:" \
                  + (f" TP1={tp1*100:.2f}%" if tp1 is not None else "") \
                  + (f" BE≥{bea*100:.2f}%" if bea is not None else "") \
                  + (f" Trail≥{tra*100:.2f}%" if tra is not None else "") \
                  + (f" ATRx={atrx}" if atrx is not None else "")

    tbe_part = ""
    if tbe_data and tbe_data.get("stage"):
        stage = tbe_data["stage"]
        score = tbe_data.get("score", 0)
        tbe_part = f" | TBE={stage} s={score:.1f}"

    msg = f"{lamp} • {source} • {mode.upper()} | Price={p} Qty={q} Lev={lev} Risk={ra}{c_part}{gz_part}{mg_part}{tbe_part}"

    try:
        (log_g if side.lower().startswith("b") else log_w)(msg)
    except NameError:
        print(msg, flush=True)

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

# =================== TREND BIRTH ENGINE STATE MACHINE ===================
TBE_STATE = {
    "state": "IDLE",  # IDLE, SWEPT, BROKEN, WAIT_RETEST, REHUNT
    "dir": None,  # BUY or SELL
    "sweep_level": None,
    "break_level": None,
    "zone": None,  # {"type": "OB"/"FVG", "low": float, "high": float, "created_at": int (bar_index)}
    "created_at": None,  # timestamp of state entry
    "ttl_bars": 0,  # time to live in bars
    "blacklisted_zones": []  # list of zones that failed
}

def tbe_reset(reason="RESET"):
    """إعادة تعيين محرك بداية الترند"""
    global TBE_STATE
    log_i(f"TBE Reset: {reason}")
    TBE_STATE = {
        "state": "IDLE",
        "dir": None,
        "sweep_level": None,
        "break_level": None,
        "zone": None,
        "created_at": None,
        "ttl_bars": 0,
        "blacklisted_zones": []
    }

def tbe_blacklist_zone(zone):
    """إضافة منطقة للقائمة السوداء"""
    global TBE_STATE
    if zone not in TBE_STATE["blacklisted_zones"]:
        TBE_STATE["blacklisted_zones"].append(zone)
        # نحتفظ فقط بآخر 5 مناطق
        if len(TBE_STATE["blacklisted_zones"]) > 5:
            TBE_STATE["blacklisted_zones"].pop(0)

# =================== TREND BIRTH ENGINE DETECTORS ===================
def detect_sweep_tbe(df, atr, dir_filter=None):
    """
    كشف Sweep السيولة - مرحلة 1
    يعيد: {"ok": True, "dir": "BUY"/"SELL", "level": float} أو {"ok": False}
    """
    if len(df) < 30:
        return {"ok": False}
    
    # احصل على آخر swing low/high (آخر 20 شمعة)
    highs = df['high'].astype(float).iloc[-30:-5]
    lows = df['low'].astype(float).iloc[-30:-5]
    
    swing_low = lows.min()
    swing_low_idx = lows.idxmin()
    swing_high = highs.max()
    swing_high_idx = highs.idxmax()
    
    # الشمعة الحالية
    current = df.iloc[-1]
    current_high = float(current['high'])
    current_low = float(current['low'])
    current_close = float(current['close'])
    
    # BUY Sweep: سحب سيولة تحت swing low ثم رجوع
    if (not dir_filter or dir_filter == "BUY") and current_low < swing_low - (TBE_SWEEP_ATR_MULT * atr):
        # ثم أغلق فوق swing low (رفض)
        if current_close > swing_low:
            log_i(f"🎯 TBE SWEEP BUY: {current_low:.6f} < {swing_low:.6f} (ATR={atr:.6f})")
            return {"ok": True, "dir": "BUY", "level": swing_low}
    
    # SELL Sweep: سحب سيولة فوق swing high ثم رجوع
    if (not dir_filter or dir_filter == "SELL") and current_high > swing_high + (TBE_SWEEP_ATR_MULT * atr):
        if current_close < swing_high:
            log_i(f"🎯 TBE SWEEP SELL: {current_high:.6f} > {swing_high:.6f} (ATR={atr:.6f})")
            return {"ok": True, "dir": "SELL", "level": swing_high}
    
    return {"ok": False}

def detect_bos_choc_tbe(df, dir_filter=None):
    """
    كسر الهيكل (BOS) أو تغيير الهيكل (CHoCH) - مرحلة 2
    يعيد: {"ok": True, "dir": "BUY"/"SELL", "level": float} أو {"ok": False}
    """
    if len(df) < 40:
        return {"ok": False}
    
    high_series = df['high'].astype(float)
    low_series = df['low'].astype(float)
    close_series = df['close'].astype(float)
    
    # أحدث swing high و swing low (باستثناء آخر 5 شمعات)
    swing_high = high_series.iloc[-30:-5].max()
    swing_low = low_series.iloc[-30:-5].min()
    
    current_close = close_series.iloc[-1]
    prev_close = close_series.iloc[-2]
    
    # BUY BOS: كسر أعلى swing high
    if (not dir_filter or dir_filter == "BUY") and current_close > swing_high:
        # تحقق من أن الحركة قوية (نسبة مئوية)
        change_pct = ((current_close - swing_high) / swing_high) * 100
        if change_pct >= TBE_BOS_MIN_PCT:
            log_i(f"🎯 TBE BOS BUY: {current_close:.6f} > {swing_high:.6f} (+{change_pct:.2f}%)")
            return {"ok": True, "dir": "BUY", "level": swing_high, "change_pct": change_pct}
    
    # SELL BOS: كسر أقل swing low
    if (not dir_filter or dir_filter == "SELL") and current_close < swing_low:
        change_pct = ((swing_low - current_close) / swing_low) * 100
        if change_pct >= TBE_BOS_MIN_PCT:
            log_i(f"🎯 TBE BOS SELL: {current_close:.6f} < {swing_low:.6f} (-{change_pct:.2f}%)")
            return {"ok": True, "dir": "SELL", "level": swing_low, "change_pct": change_pct}
    
    return {"ok": False}

def momentum_flip_tbe(df, indicators, dir_filter=None):
    """
    تأكيد انعكاس الزخم - مرحلة 3
    يعيد: {"ok": True, "dir": "BUY"/"SELL", "score": float} أو {"ok": False}
    """
    rsi = indicators.get('rsi', 50)
    rsi_ma = indicators.get('rsi_ma', 50)
    rsi_trendz = indicators.get('rsi_trendz', 'none')
    
    # BUY Momentum: RSI > 50 و RSI فوق RSI_MA
    if (not dir_filter or dir_filter == "BUY") and rsi > 50 and rsi > rsi_ma:
        score = 1.0
        if rsi_trendz == 'bull':
            score += 0.5
        log_i(f"🎯 TBE MOMENTUM BUY: RSI={rsi:.1f} > RSI_MA={rsi_ma:.1f}, score={score:.1f}")
        return {"ok": True, "dir": "BUY", "score": score}
    
    # SELL Momentum: RSI < 50 و RSI تحت RSI_MA
    if (not dir_filter or dir_filter == "SELL") and rsi < 50 and rsi < rsi_ma:
        score = 1.0
        if rsi_trendz == 'bear':
            score += 0.5
        log_i(f"🎯 TBE MOMENTUM SELL: RSI={rsi:.1f} < RSI_MA={rsi_ma:.1f}, score={score:.1f}")
        return {"ok": True, "dir": "SELL", "score": score}
    
    return {"ok": False}

def build_ob_or_fvg_tbe(df, dir_filter=None):
    """
    بناء منطقة الدخول (Order Block أو FVG) - مرحلة 4
    يعيد: {"ok": True, "type": "OB"/"FVG", "low": float, "high": float} أو {"ok": False}
    """
    if len(df) < 10:
        return {"ok": False}
    
    # FVG Bullish: low[i] > high[i-2]
    # FVG Bearish: high[i] < low[i-2]
    for i in range(-1, -5, -1):
        if len(df) >= abs(i)+2:
            if dir_filter in [None, "BUY"]:
                low_current = float(df['low'].iloc[i])
                high_prev2 = float(df['high'].iloc[i-2])
                if low_current > high_prev2:
                    zone = {"type": "FVG", "low": high_prev2, "high": low_current}
                    log_i(f"🎯 TBE ZONE FVG BUY: {high_prev2:.6f} - {low_current:.6f}")
                    return {"ok": True, **zone}
            if dir_filter in [None, "SELL"]:
                high_current = float(df['high'].iloc[i])
                low_prev2 = float(df['low'].iloc[i-2])
                if high_current < low_prev2:
                    zone = {"type": "FVG", "low": high_current, "high": low_prev2}
                    log_i(f"🎯 TBE ZONE FVG SELL: {high_current:.6f} - {low_prev2:.6f}")
                    return {"ok": True, **zone}
    
    # إذا لم نجد FVG نبحث عن OB
    # نبحث عن آخر شمعة معاكسة للاتجاه الحالي
    last_close = float(df['close'].iloc[-1])
    last_open = float(df['open'].iloc[-1])
    last_dir = "BUY" if last_close > last_open else "SELL"
    
    # نبحث عن شمعة معاكسة للاتجاه الأخير
    for i in range(-2, -10, -1):
        if i+1 < 0:
            open_price = float(df['open'].iloc[i])
            close_price = float(df['close'].iloc[i])
            if last_dir == "BUY" and close_price < open_price:
                # وجدنا شمعة هابطة (معاكسة) للاتجاه الصاعد
                low = float(df['low'].iloc[i])
                high = float(df['high'].iloc[i])
                zone = {"type": "OB", "low": low, "high": high}
                log_i(f"🎯 TBE ZONE OB BUY: {low:.6f} - {high:.6f}")
                return {"ok": True, **zone}
            elif last_dir == "SELL" and close_price > open_price:
                # وجدنا شمعة صاعدة (معاكسة) للاتجاه الهابط
                low = float(df['low'].iloc[i])
                high = float(df['high'].iloc[i])
                zone = {"type": "OB", "low": low, "high": high}
                log_i(f"🎯 TBE ZONE OB SELL: {low:.6f} - {high:.6f}")
                return {"ok": True, **zone}
    
    return {"ok": False}

def price_in_zone(price, zone):
    """تحقق إذا كان السعر داخل المنطقة"""
    if not zone:
        return False
    return zone["low"] <= price <= zone["high"]

def rejection_ok(df, dir):
    """
    تحقق إذا كان هناك رفض (rejection) عند المنطقة - مرحلة 5
    """
    if len(df) < 2:
        return False
    
    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    current_open = float(current['open'])
    current_close = float(current['close'])
    current_high = float(current['high'])
    current_low = float(current['low'])
    
    if dir == "BUY":
        # شمعة رفض للهبوط (مثل hammer, engulfing صاعد)
        candle_bull = current_close > current_open
        body = abs(current_close - current_open)
        lower_wick = min(current_open, current_close) - current_low
        # ذيل سفلي طويل (إشارة رفض للهبوط)
        if lower_wick > body * 1.5 and lower_wick > (current_high - current_low) * 0.3:
            return True
        # engulfing صاعد مقارنة بالشمعة السابقة
        if current_close > float(prev['high']) and current_open < float(prev['low']):
            return True
    else:  # SELL
        candle_bear = current_close < current_open
        upper_wick = current_high - max(current_open, current_close)
        body = abs(current_close - current_open)
        # ذيل علوي طويل (إشارة رفض للصعود)
        if upper_wick > body * 1.5 and upper_wick > (current_high - current_low) * 0.3:
            return True
        # engulfing هابط مقارنة بالشمعة السابقة
        if current_close < float(prev['low']) and current_open > float(prev['high']):
            return True
    
    return False

def tbe_score_entry(htf_ctx, daily_open_ctx, indicators, zone, chop_penalty):
    """
    حساب درجة الدخول بناءً على السياق الكلي
    """
    try:
        score = 5.0  # درجة أساسية
        
        # HTF Context
        if htf_ctx.get("trend") == "bull":
            if zone.get("type") in ["FVG", "OB"]:  # BUY zones
                score += TBE_HTF_WEIGHT
            else:
                score -= 1.0
        elif htf_ctx.get("trend") == "bear":
            if zone.get("type") in ["FVG", "OB"]:  # SELL zones
                score += TBE_HTF_WEIGHT
            else:
                score -= 1.0
        
        # Daily Open Bias
        current_price = indicators.get('price', 0)
        daily_open = daily_open_ctx.get('open', current_price)
        
        if current_price > daily_open:
            score += 1.0  # انحياز للشراء
        elif current_price < daily_open:
            score += 1.0  # انحياز للبيع
        
        # مؤشر RSI
        rsi = indicators.get('rsi', 50)
        if rsi > 55:
            score += 0.5
        elif rsi < 45:
            score += 0.5
        
        # مؤشر التدفق (إذا كان متاحاً)
        flow = indicators.get('flow', {})
        if flow.get('ok'):
            delta_z = flow.get('delta_z', 0)
            if delta_z > 0.3:
                score += TBE_FLOW_WEIGHT
            elif delta_z < -0.3:
                score += TBE_FLOW_WEIGHT
        
        # عقوبة التداول في نطاق جانبي على HTF
        if chop_penalty and htf_ctx.get("chop", False):
            score -= 2.0
        
        return max(0, min(score, 10))  # التأكد من أن الدرجة بين 0 و 10
    except Exception as e:
        log_w(f"TBE score error: {e}")
        return 5.0  # درجة متوسطة في حالة الخطأ

def tbe_failfast_check(df, zone, side, entry_price, bars_in_trade):
    """
    فحص إذا كان الدخول خاطئاً خلال الشمعات الأولى (Fail-Fast System)
    """
    if bars_in_trade > TBE_FAILFAST_BARS:
        return False  # تجاوزنا فترة الفشل السريع
    
    current = df.iloc[-1]
    current_close = float(current['close'])
    
    if side == "long":
        # إذا أغلق السعر تحت zone_low
        if current_close < zone["low"]:
            return True
    else:  # short
        if current_close > zone["high"]:
            return True
    
    return False

def tbe_update(df, htf_ctx, daily_open_ctx, indicators):
    """
    تحديث محرك بداية الترند - State Machine
    يعيد: {"enter": True/False, "side": "BUY"/"SELL", "zone": {...}, "score": float, "reason": str, "stage": str}
    """
    global TBE_STATE
    
    # تهيئة القاموس الافتراضي
    default_result = {"enter": False, "reason": "NO_TRIGGER", "stage": TBE_STATE["state"]}
    
    try:
        # إذا كان في حالة REHUNT، تحقق من انتهاء الوقت
        if TBE_STATE["state"] == "REHUNT":
            if TBE_STATE["ttl_bars"] <= 0:
                TBE_STATE["state"] = "IDLE"
                log_i("TBE: REHUNT cooldown finished")
            else:
                TBE_STATE["ttl_bars"] -= 1
                return {"enter": False, "reason": "REHUNT_COOLDOWN", "stage": "REHUNT"}
        
        # 1) حالة IDLE: ابحث عن Sweep
        if TBE_STATE["state"] == "IDLE":
            sweep = detect_sweep_tbe(df, indicators.get('atr', 0))
            if sweep.get("ok"):
                TBE_STATE["state"] = "SWEPT"
                TBE_STATE["dir"] = sweep.get("dir")
                TBE_STATE["sweep_level"] = sweep.get("level")
                TBE_STATE["created_at"] = len(df)
                TBE_STATE["ttl_bars"] = 12
                log_g(f"🎯 TBE SWEPT: {sweep.get('dir')} at {sweep.get('level', 0):.6f}")
                return {"enter": False, "reason": "SWEPT", "stage": "SWEPT", "dir": sweep.get("dir")}
        
        # 2) حالة SWEPT: ابحث عن كسر الهيكل (BOS/CHoCH)
        elif TBE_STATE["state"] == "SWEPT":
            if TBE_STATE["ttl_bars"] <= 0:
                tbe_reset("EXPIRED_AFTER_SWEEP")
                return {"enter": False, "reason": "EXPIRED_AFTER_SWEEP"}
            
            bos = detect_bos_choc_tbe(df, dir_filter=TBE_STATE["dir"])
            if bos.get("ok") and bos.get("dir") == TBE_STATE["dir"]:
                TBE_STATE["state"] = "BROKEN"
                TBE_STATE["break_level"] = bos.get("level")
                TBE_STATE["ttl_bars"] = 12
                log_g(f"🎯 TBE BROKEN: {bos.get('dir')} at {bos.get('level', 0):.6f} ({bos.get('change_pct', 0):.2f}%)")
                return {"enter": False, "reason": "STRUCTURE_SHIFT", "stage": "BROKEN", "dir": bos.get("dir")}
            
            TBE_STATE["ttl_bars"] -= 1
        
        # 3) حالة BROKEN: ابحث عن انعكاس الزخم ثم بناء المنطقة
        elif TBE_STATE["state"] == "BROKEN":
            if TBE_STATE["ttl_bars"] <= 0:
                tbe_reset("EXPIRED_AFTER_BOS")
                return {"enter": False, "reason": "EXPIRED_AFTER_BOS"}
            
            mom = momentum_flip_tbe(df, indicators, dir_filter=TBE_STATE["dir"])
            if mom.get("ok") and mom.get("dir") == TBE_STATE["dir"]:
                zone = build_ob_or_fvg_tbe(df, dir_filter=TBE_STATE["dir"])
                if zone.get("ok"):
                    TBE_STATE["state"] = "WAIT_RETEST"
                    TBE_STATE["zone"] = zone
                    TBE_STATE["ttl_bars"] = TBE_ZONE_TTL_BARS
                    log_g(f"🎯 TBE ZONE: {zone.get('type')} at {zone.get('low', 0):.6f}-{zone.get('high', 0):.6f}")
                    return {"enter": False, "reason": "ZONE_CREATED", "stage": "WAIT_RETEST", "dir": TBE_STATE["dir"], "zone": zone}
            
            TBE_STATE["ttl_bars"] -= 1
        
        # 4) حالة WAIT_RETEST: انتظار عودة السعر للمنطقة مع رفض
        elif TBE_STATE["state"] == "WAIT_RETEST":
            if TBE_STATE["ttl_bars"] <= 0:
                tbe_reset("EXPIRED_WAIT_RETEST")
                return {"enter": False, "reason": "EXPIRED_WAIT_RETEST"}
            
            current_price = indicators.get('price', float(df['close'].iloc[-1]))
            zone = TBE_STATE.get("zone")
            
            if not zone:
                tbe_reset("NO_ZONE")
                return {"enter": False, "reason": "NO_ZONE"}
            
            # تحقق إذا كانت المنطقة في القائمة السوداء
            if zone in TBE_STATE["blacklisted_zones"]:
                tbe_reset("ZONE_BLACKLISTED")
                return {"enter": False, "reason": "ZONE_BLACKLISTED"}
            
            if price_in_zone(current_price, zone) and rejection_ok(df, TBE_STATE["dir"]):
                chop_penalty = htf_ctx.get("chop", False)
                score = tbe_score_entry(htf_ctx, daily_open_ctx, indicators, zone, chop_penalty)
                
                if score >= TBE_ENTRY_SCORE_MIN:
                    enter_side = TBE_STATE["dir"]
                    enter_zone = zone
                    reason = "TBE_ENTRY"
                    log_g(f"🎯 TBE ENTRY TRIGGERED: {enter_side} score={score:.1f} zone={zone.get('type')}")
                    tbe_reset("ENTERED")
                    return {"enter": True, "side": enter_side, "zone": enter_zone, 
                            "score": score, "reason": reason, "stage": "ENTRY"}
            
            TBE_STATE["ttl_bars"] -= 1
        
        return default_result
        
    except Exception as e:
        log_w(f"TBE update error: {e}")
        tbe_reset(f"ERROR: {str(e)[:50]}")
        return default_result

# =================== HTF & DAILY OPEN CONTEXT ===================
last_htf_update = 0
htf_ctx_cache = {"trend": "none", "chop": False, "ema200": 0, "ma_slope": 0}
last_daily_open_update = 0
daily_open_cache = None

def fetch_htf_context(exchange, symbol, interval="1h", limit=100):
    """
    جلب سياق الإطار الزمني الأعلى (1H أو 4H)
    """
    try:
        df_htf = fetch_ohlcv_generic(exchange, symbol, interval, limit)
        if len(df_htf) < 50:
            return {"trend": "none", "chop": True, "ema200": 0, "ma_slope": 0}
        
        closes = df_htf['close'].astype(float)
        highs = df_htf['high'].astype(float)
        lows = df_htf['low'].astype(float)
        
        # حساب EMA200 و MA50
        ema200 = closes.ewm(span=200, adjust=False).mean().iloc[-1]
        ma50 = closes.rolling(50).mean().iloc[-1]
        ma20 = closes.rolling(20).mean().iloc[-1]
        
        current_price = closes.iloc[-1]
        
        # تحديد الاتجاه
        trend = "none"
        if current_price > ema200 and ma50 > ema200 and ma20 > ma50:
            trend = "bull"
        elif current_price < ema200 and ma50 < ema200 and ma20 < ma50:
            trend = "bear"
        
        # حساب ميل المتوسطات
        ma_slope = (ma20 - ma50) / ma50 * 100
        
        # تحديد إذا كان السوق في نطاق جانبي (ATR منخفض)
        atr = (highs - lows).rolling(14).mean().iloc[-1]
        atr_percent = atr / current_price * 100
        is_chop = atr_percent < 0.4  # أقل من 0.4% يعتبر نطاق جانبي
        
        return {
            "trend": trend, 
            "chop": is_chop,
            "ema200": ema200,
            "ma_slope": ma_slope,
            "price": current_price,
            "atr_percent": atr_percent
        }
    except Exception as e:
        log_w(f"HTF context error: {e}")
        return {"trend": "none", "chop": False, "ema200": 0, "ma_slope": 0}

def fetch_daily_open_context(exchange, symbol):
    """
    جلب سعر افتتاح اليوم
    """
    try:
        # جلب بيانات 1D للحصول على افتتاح اليوم
        df_daily = fetch_ohlcv_generic(exchange, symbol, "1d", 2)
        if len(df_daily) < 1:
            return None
        
        daily_open = float(df_daily['open'].iloc[-1])
        daily_high = float(df_daily['high'].iloc[-1])
        daily_low = float(df_daily['low'].iloc[-1])
        
        return {
            "open": daily_open,
            "high": daily_high,
            "low": daily_low,
            "range": daily_high - daily_low
        }
    except Exception as e:
        log_w(f"Daily open error: {e}")
        return None

def fetch_ohlcv_generic(exchange, symbol, timeframe, limit):
    """دالة عامة لجلب بيانات OHLCV"""
    try:
        rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, params={"type":"swap"})
        return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])
    except Exception as e:
        log_w(f"fetch_ohlcv_generic error: {e}")
        return pd.DataFrame()

def get_htf_ctx():
    """الحصول على سياق HTF مع التخزين المؤقت ومعالجة الأخطاء"""
    global last_htf_update, htf_ctx_cache
    try:
        now = time.time()
        if now - last_htf_update > 900:  # تحديث كل 15 دقيقة
            htf_ctx_cache = fetch_htf_context(ex, SYMBOL, "1h", 200)
            last_htf_update = now
        return htf_ctx_cache or {"trend": "none", "chop": False, "ema200": 0, "ma_slope": 0}
    except Exception as e:
        log_w(f"get_htf_ctx error: {e}")
        return {"trend": "none", "chop": False, "ema200": 0, "ma_slope": 0}

def get_daily_open_ctx():
    """الحصول على افتتاح اليوم مع التخزين المؤقت ومعالجة الأخطاء"""
    global last_daily_open_update, daily_open_cache
    try:
        now = time.time()
        if now - last_daily_open_update > 3600:  # تحديث كل ساعة
            daily_open_cache = fetch_daily_open_context(ex, SYMBOL)
            last_daily_open_update = now
        return daily_open_cache or {"open": 0, "high": 0, "low": 0, "range": 0}
    except Exception as e:
        log_w(f"get_daily_open_ctx error: {e}")
        return {"open": 0, "high": 0, "low": 0, "range": 0}

# =================== SMC/ICT TOOLS ===================
def _fib_zone(last_impulse_low, last_impulse_high):
    rng = last_impulse_high - last_impulse_low
    return last_impulse_low + GZ_FIB_LOW * rng, last_impulse_low + GZ_FIB_HIGH * rng

def detect_bos(df):
    """Break of Structure detection"""
    if len(df) < 30: 
        return {"ok": False}
    
    high_series = df['high'].astype(float)
    low_series = df['low'].astype(float)
    close_series = df['close'].astype(float)
    
    swing_high = high_series.iloc[-20:-5].max()
    swing_low = low_series.iloc[-20:-5].min()
    close = close_series.iloc[-1]
    prev = close_series.iloc[-2]
    
    up_bos = (close > swing_high) and ((close - prev) / prev * 100 >= BOS_MIN_PCT)
    down_bos = (close < swing_low) and ((prev - close) / prev * 100 >= BOS_MIN_PCT)
    
    if up_bos:   
        return {"ok": True, "dir": "bull", "ref": swing_high}
    if down_bos: 
        return {"ok": True, "dir": "bear", "ref": swing_low}
    
    return {"ok": False}

def detect_sweep(df, atr):
    """Liquidity Sweep detection"""
    if len(df) < 5: 
        return {"ok": False}
    
    c = df.iloc[-1]
    current_high = float(c['high'])
    current_low = float(c['low'])
    current_close = float(c['close'])
    current_open = float(c['open'])
    
    wick_up = current_high - max(current_close, current_open)
    wick_dn = min(current_close, current_open) - current_low
    
    bull = wick_dn >= SWEEP_WICK_X_ATR * atr
    bear = wick_up >= SWEEP_WICK_X_ATR * atr
    
    if bull: 
        return {"ok": True, "dir": "bull"}
    if bear: 
        return {"ok": True, "dir": "bear"}
    
    return {"ok": False}

def detect_fvg(df, min_bps=FVG_MIN_BPS):
    """Fair Value Gap detection"""
    if len(df) < 5: 
        return {"ok": False}
    
    h1 = float(df['high'].iloc[-2])
    l1 = float(df['low'].iloc[-2])
    h0 = float(df['high'].iloc[-1])
    l0 = float(df['low'].iloc[-1])
    
    up = (l0 - h1) / ((h1 + l0) / 2) * 10000.0
    down = (l1 - h0) / ((h0 + l1) / 2) * 10000.0
    
    if up >= min_bps:   
        return {"ok": True, "dir": "bull", "bps": up}
    if down >= min_bps: 
        return {"ok": True, "dir": "bear", "bps": down}
    
    return {"ok": False}

def detect_order_block(df, bullish=True, lookback=OB_LOOKBACK):
    """Order Block detection"""
    try:
        if len(df) < lookback:
            return {"ok": False}
            
        window = df.iloc[-lookback:]
        high_series = window['high'].astype(float)
        low_series = window['low'].astype(float)
        
        if bullish:
            lowest_indices = low_series.nsmallest(3).index
            base = window.loc[lowest_indices]
            ob_low = base['low'].astype(float).min()
            ob_high = base['high'].astype(float).max()
        else:
            highest_indices = high_series.nlargest(3).index
            base = window.loc[highest_indices]
            ob_low = base['low'].astype(float).min()
            ob_high = base['high'].astype(float).max()
            
        return {"ok": True, "low": ob_low, "high": ob_high}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def golden_zone_check_pro(df, ind):
    """Enhanced Golden Zone detection with Fibonacci levels"""
    if len(df) < 40:
        return {"ok": False}
        
    closes = df['close'].astype(float).values
    recent = closes[-30:]
    hi = recent.max()
    lo = recent.min()
    
    # Simple trend detection
    trend_up = hi == recent[-1]
    trend_dn = lo == recent[-1]
    
    fib_lo, fib_hi = _fib_zone(lo, hi)
    last = closes[-1]
    score = 0.0
    ztype = None

    if fib_lo <= last <= fib_hi:
        # Inside golden zone
        if ind.get('adx', 0) >= GZ_ADX_MIN:
            score += 2.0
        if ind.get('rsi', 50) < 45 and ind.get('rsi', 50) > ind.get('rsi_ma', 50):
            score += 1.0
        if ind.get('evx', 1.0) < 1.2:
            score += 0.5
            
        if trend_up:
            ztype = 'golden_top'
        elif trend_dn:
            ztype = 'golden_bottom'
        else:
            zone_mid = (fib_lo + fib_hi) / 2
            ztype = 'golden_top' if last > zone_mid else 'golden_bottom'
            
        return {
            "ok": True, 
            "score": score + 3.0, 
            "zone": {
                "type": ztype, 
                "lo": fib_lo, 
                "hi": fib_hi
            }
        }
        
    return {"ok": False}

# =================== CANDLES MODULE ===================
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

def compute_candles(df):
    if len(df) < 5:
        return {"buy":False,"sell":False,"score_buy":0.0,"score_sell":0.0,
                "wick_up_big":False,"wick_dn_big":False,"doji":False,"pattern":None}

    o1,h1,l1,c1 = float(df["open"].iloc[-2]), float(df["high"].iloc[-2]), float(df["low"].iloc[-2]), float(df["close"].iloc[-2])
    o0,h0,l0,c0 = float(df["open"].iloc[-3]), float(df["high"].iloc[-3]), float(df["low"].iloc[-3]), float(df["close"].iloc[-3])

    strength_b = strength_s = 0.0
    tags = []

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

# =================== EXECUTION VERIFICATION ===================
def verify_execution_environment():
    """التحقق من بيئة التنفيذ عند الإقلاع"""
    print(f"⚙️ EXECUTION ENVIRONMENT", flush=True)
    print(f"🔧 EXECUTE_ORDERS: {EXECUTE_ORDERS} | SHADOW_MODE: {SHADOW_MODE_DASHBOARD} | DRY_RUN: {DRY_RUN}", flush=True)
    print(f"🎯 COUNCIL ELITE ENHANCED: Smart Entry + Fast Trading", flush=True)
    print(f"📈 SMC/ICT: Golden Zones + FVG + BOS + Sweeps", flush=True)
    print(f"🚀 TREND BIRTH ENGINE: اصطياد بدايات الترند - {'مفعّل' if TBE_ENABLED else 'معطّل'}", flush=True)
    
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

def decide_strategy_mode(df, adx=None, di_plus=None, di_minus=None, rsi_ctx=None):
    """تحديد نمط التداول: SCALP أم TREND"""
    if adx is None or di_plus is None or di_minus is None:
        ind = compute_indicators(df)
        adx = ind.get('adx', 0)
        di_plus = ind.get('plus_di', 0)
        di_minus = ind.get('minus_di', 0)
    
    if rsi_ctx is None:
        rsi_ctx = rsi_ma_context(df)
    
    di_spread = abs(di_plus - di_minus)
    
    strong_trend = (
        (adx >= ADX_TREND_MIN and di_spread >= DI_SPREAD_TREND) or
        (rsi_ctx["trendZ"] in ("bull", "bear") and not rsi_ctx["in_chop"])
    )
    
    mode = "trend" if strong_trend else "scalp"
    why = "adx/di_trend" if adx >= ADX_TREND_MIN else ("rsi_trendZ" if rsi_ctx["trendZ"] != "none" else "scalp_default")
    
    return {"mode": mode, "why": why}

# =================== COUNCIL ELITE VOTING - ENHANCED ===================
COUNCIL_BUSY = False
LAST_COUNCIL = {"b": 0, "s": 0, "score_b": 0.0, "score_s": 0.0, "logs": [], "ind": {}}

def council_votes_enhanced(df):
    """نسخة محسنة من Council بشروط أسهل للمزيد من الصفقات"""
    global COUNCIL_BUSY, LAST_COUNCIL
    if COUNCIL_BUSY:
        return LAST_COUNCIL
        
    COUNCIL_BUSY = True
    try:
        ind = compute_indicators(df)
        rsi_ctx = rsi_ma_context(df)
        atr = ind.get('atr', 0.0)

        # SMC/ICT Detection
        bos = detect_bos(df)
        fvg = detect_fvg(df)
        sweep = detect_sweep(df, atr)
        ob_bull = detect_order_block(df, bullish=True)
        ob_bear = detect_order_block(df, bullish=False)

        # Enhanced Golden Zones
        gz = golden_zone_check_pro(df, ind)

        votes_b = votes_s = 0
        score_b = score_s = 0.0
        logs = []

        adx = ind.get('adx', 0.0)
        plus_di = ind.get('plus_di', 0.0)
        minus_di = ind.get('minus_di', 0.0)
        di_spread = abs(plus_di - minus_di)

        # Strong Trend (ADX/DI) - شروط أسهل
        if adx >= 14:  # ⬇️ كان ADX_TREND_MIN
            if plus_di > minus_di and di_spread > 4.0:  # ⬇️ كان DI_SPREAD_TREND
                votes_b += 2
                score_b += 1.2  # ⬇️ كان 1.5
                logs.append("📈 ترند صاعد (ADX/DI)")
            elif minus_di > plus_di and di_spread > 4.0:
                votes_s += 2
                score_s += 1.2
                logs.append("📉 ترند هابط (ADX/DI)")

        # RSI+MA Cross & Trend - شروط أسهل
        if rsi_ctx["cross"] == "bull" and rsi_ctx["rsi"] < 65:  # ⬆️ كان 70
            votes_b += 2
            score_b += 1.0
            logs.append("🟢 RSI-MA إيجابي")
        elif rsi_ctx["cross"] == "bear" and rsi_ctx["rsi"] > 35:  # ⬇️ كان 30
            votes_s += 2
            score_s += 1.0
            logs.append("🔴 RSI-MA سلبي")

        if rsi_ctx["trendZ"] == "bull":
            votes_b += 2  # ⬇️ كان 3
            score_b += 1.2  # ⬇️ كان 1.5
            logs.append("🚀 RSI ترند صاعد")
        elif rsi_ctx["trendZ"] == "bear":
            votes_s += 2  # ⬇️ كان 3
            score_s += 1.2  # ⬇️ كان 1.5
            logs.append("💥 RSI ترند هابط")

        # FVG (Fair Value Gap) - شروط أسهل
        if fvg.get("ok"):
            if fvg["dir"] == "bull":
                votes_b += 1  # ⬇️ كان 2
                score_b += 0.8  # ⬇️ كان 1.0
                logs.append(f"🟢 FVG bull {fvg['bps']:.1f}bps")
            else:
                votes_s += 1  # ⬇️ كان 2
                score_s += 0.8  # ⬇️ كان 1.0
                logs.append(f"🔴 FVG bear {fvg['bps']:.1f}bps")

        # BOS (Break of Structure) - شروط أسهل
        if bos.get("ok"):
            if bos["dir"] == "bull":
                votes_b += 1  # ⬇️ كان 2
                score_b += 0.8  # ⬇️ كان 1.0
                logs.append("🟩 BOS ↑")
            else:
                votes_s += 1  # ⬇️ كان 2
                score_s += 0.8  # ⬇️ كان 1.0
                logs.append("🟥 BOS ↓")

        # Liquidity Sweeps
        if sweep.get("ok"):
            if sweep["dir"] == "bull":
                votes_b += 1
                score_b += 0.5
                logs.append("💧 Liquidity Sweep (bull)")
            else:
                votes_s += 1
                score_s += 0.5
                logs.append("💧 Liquidity Sweep (bear)")

        # Order Blocks
        if ob_bull.get("ok"):
            votes_b += 1  # ⬆️ كان مجرد لوج
            score_b += 0.5
            logs.append("🟢 OB Demand")
        if ob_bear.get("ok"):
            votes_s += 1  # ⬆️ كان مجرد لوج
            score_s += 0.5
            logs.append("🔴 OB Supply")

        # Golden Zones - شروط أسهل
        if gz and gz.get("ok") and adx >= 14:  # ⬇️ كان GZ_ADX_MIN
            if gz['zone']['type'] == 'golden_bottom':
                votes_b += 2  # ⬇️ كان 3
                score_b += 1.2  # ⬇️ كان 1.5
                logs.append(f"🏆 قاع ذهبي s={gz['score']:.1f}")
            elif gz['zone']['type'] == 'golden_top':
                votes_s += 2  # ⬇️ كان 3
                score_s += 1.2  # ⬇️ كان 1.5
                logs.append(f"🏆 قمة ذهبية s={gz['score']:.1f}")

        # Flow/Bookmap Integration
        flow = compute_flow_metrics(df)
        bm = bookmap_snapshot(ex, SYMBOL)
        
        if flow.get("ok"):
            dz = flow.get("delta_z", 0)
            if dz >= 0.3:  # ⬇️ كان DELTA_Z_BULL
                votes_b += 1  # ⬇️ كان 2
                score_b += 0.8  # ⬇️ كان 1.0
                logs.append("📊 Flow ضغط شراء")
            if dz <= -0.3:  # ⬆️ كان DELTA_Z_BEAR
                votes_s += 1  # ⬇️ كان 2
                score_s += 0.8  # ⬇️ كان 1.0
                logs.append("📊 Flow ضغط بيع")
                
        if bm.get("ok"):
            imb = bm.get("imbalance", 1.0)
            if imb >= 1.1:  # ⬇️ كان IMB_ALERT
                logs.append(f"🧱 Bookmap imb={imb:.2f}")

        # Neutral/Chop Reduction - أقل عقوبة
        if rsi_ctx["in_chop"]:
            score_b *= 0.90  # ⬆️ كان 0.85
            score_s *= 0.90  # ⬆️ كان 0.85
            logs.append("⚖️ نطاق حيادي (RSI 45–55)")

        # ADX Gate - أقل عقوبة
        if adx < 12:  # ⬇️ كان ADX_GATE
            score_b *= 0.95  # ⬆️ كان 0.9
            score_s *= 0.95  # ⬆️ كان 0.9
            logs.append(f"🛡️ ADX Gate {adx:.1f}<12")

        # Update indicators with new data
        ind.update({
            "rsi": rsi_ctx["rsi"],
            "rsi_ma": rsi_ctx["rsi_ma"], 
            "rsi_trendz": rsi_ctx["trendZ"],
            "di_spread": di_spread,
            "fvg": fvg,
            "bos": bos, 
            "sweep": sweep,
            "gz": gz,
            "flow": flow,
            "bm": bm
        })

        result = {
            "b": votes_b,
            "s": votes_s, 
            "score_b": round(score_b, 2),
            "score_s": round(score_s, 2),
            "logs": logs,
            "ind": ind
        }
        
        LAST_COUNCIL = result
        return result
        
    except Exception as e:
        log_w(f"council_votes_enhanced error: {e}")
        return LAST_COUNCIL
    finally:
        COUNCIL_BUSY = False

council_votes_pro = council_votes_enhanced

# =================== FAST TRADING SYSTEM ===================
def detect_fast_opportunity(df, council_data):
    """كشف فرص التداول السريع"""
    if not FAST_TRADE_ENABLED:
        return None
        
    ind = council_data["ind"]
    score_b = council_data["score_b"]
    score_s = council_data["score_s"]
    
    # شروط أسهل للدخول السريع
    fast_buy = (
        score_b >= FAST_MIN_SCORE and 
        ind.get('rsi', 50) < 65 and  # ⬆️ كان 70
        ind.get('adx', 0) > 10 and   # ⬇️ كان 12
        council_data["b"] > council_data["s"]
    )
    
    fast_sell = (
        score_s >= FAST_MIN_SCORE and 
        ind.get('rsi', 50) > 35 and  # ⬇️ كان 30
        ind.get('adx', 0) > 10 and   # ⬇️ كان 12
        council_data["s"] > council_data["b"]
    )
    
    if fast_buy:
        return {"action": "fast_buy", "reason": f"فرصة سريعة - score:{score_b:.1f}"}
    elif fast_sell:
        return {"action": "fast_sell", "reason": f"فرصة سريعة - score:{score_s:.1f}"}
    
    return None

# =================== SMART TRADE MANAGEMENT ===================
def setup_trade_management(mode):
    """تهيئة إدارة الصفقة حسب النمط"""
    if mode == "scalp":
        return {
            "tp1_pct": TP1_PCT_SCALP,
            "be_activate_pct": BE_AFTER_SCALP,
            "trail_activate_pct": TRAIL_ACT_SCALP,
            "atr_trail_mult": ATR_TRAIL_MULT,
            "close_aggression": "high"
        }
    else:
        return {
            "tp1_pct": TP1_PCT_TREND,
            "be_activate_pct": BE_AFTER_TREND,
            "trail_activate_pct": TRAIL_ACT_TREND,
            "atr_trail_mult": ATR_TRAIL_MULT,
            "close_aggression": "medium"
        }

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
    evx_spike = False

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

    # Golden Reversal بعد TP1
    if state.get('tp1_done') and (gz and gz.get('ok')):
        opp = (gz['zone']['type']=='golden_top' and side=='long') or (gz['zone']['type']=='golden_bottom' and side=='short')
        if opp and gz.get('score',0) >= GOLDEN_REVERSAL_SCORE:
            return {
                "action": "close", 
                "why": "golden_reversal",
                "log": f"🔴 CLOSE STRONG | golden reversal after TP1 | score={gz['score']:.1f}"
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

    # Wick exhaustion + Tighten عند إجهاد/تدفق/جدار
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

def resume_open_position_enhanced(exchange, symbol: str, state: dict) -> dict:
    """استئناف محسن للمركز مع مصالحة آمنة"""
    if not RESUME_ON_RESTART:
        log_i("resume disabled")
        return state

    prev = load_state() or {}
    
    # 1) المحاولة الأولى: جلب المركز من المنصة
    live = fetch_live_position(exchange, symbol)
    if live.get("ok"):
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
            "opened_at": prev.get("opened_at", int(time.time())),
        })
        save_state(state)
        log_g(f"✅ RESUME via EXCHANGE: {state['side']} qty={state['position_qty']} @ {state['entry_price']:.6f}")
        return state
    
    # 2) Fallback: استخدام STATE.json إذا كان حديثاً
    if SAFE_RECONCILE and prev.get("in_position") and prev.get("position_qty", 0) > 0:
        ts = int(time.time())
        state_ts = prev.get("ts", 0)
        
        # التحقق من حداثة الحالة (أقل من ساعة)
        if ts - state_ts < 3600:
            state.update(prev)
            save_state(state)
            log_w(f"⚠️ RESUME via STATE.json (exchange unavailable): {state['side']} qty={state['position_qty']}")
            return state
    
    log_i("No position to resume — starting fresh")
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
    يطبع Snapshot موحّد: Bookmap + Flow + Council + Strategy + Balance/PnL
    """
    try:
        bm = bookmap_snapshot(exchange, symbol)
        flow = compute_flow_metrics(df)
        cv = council_votes_pro(df)
        mode = decide_strategy_mode(df)
        gz = cv["ind"].get("gz", {})

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
        strat = f"Strategy: {strat_icon} {mode['mode'].upper()}"

        bal_note = f"Balance={bal:.2f}" if bal is not None else ""
        pnl_note = f"CompoundPnL={cpnl:.6f}" if cpnl is not None else ""
        wallet = (" | ".join(x for x in [bal_note, pnl_note] if x)) or ""

        gz_note = ""
        if gz and gz.get("ok"):
            gz_note = f" | 🟡 {gz['zone']['type']} s={gz['score']:.1f}"

        if LOG_ADDONS:
            print(f"🧱 {bm_note}", flush=True)
            print(f"📦 {fl_note}", flush=True)
            print(f"📊 {dash}{gz_note}", flush=True)
            print(f"{strat}{(' | ' + wallet) if wallet else ''}", flush=True)
            
            gz_snap_note = ""
            if gz and gz.get("ok"):
                zone_type = gz["zone"]["type"]
                zone_score = gz["score"]
                gz_snap_note = f" | 🟡{zone_type} s={zone_score:.1f}"
            
            flow_z = flow['delta_z'] if flow and flow.get('ok') else 0.0
            bm_imb = bm['imbalance'] if bm and bm.get('ok') else 1.0
            
            print(f"🧠 SNAP | {side_hint} | votes={cv['b']}/{cv['s']} score={cv['score_b']:.1f}/{cv['score_s']:.1f} "
                  f"| ADX={cv['ind'].get('adx',0):.1f} DI={cv['ind'].get('di_spread',0):.1f} | "
                  f"z={flow_z:.2f} | imb={bm_imb:.2f}{gz_snap_note}", 
                  flush=True)
            
            print("✅ ADDONS LIVE", flush=True)

        return {"bm": bm, "flow": flow, "cv": cv, "mode": mode, "gz": gz, "wallet": wallet}
    except Exception as e:
        print(f"🟨 AddonLog error: {e}", flush=True)
        return {"bm": None, "flow": None, "cv": {"b":0,"s":0,"score_b":0.0,"score_s":0.0,"ind":{}},
                "mode": {"mode":"n/a"}, "gz": None, "wallet": ""}

# =================== EXECUTION MANAGER ===================
def execute_trade_decision(side, price, qty, mode, council_data, gz_data, tbe_data=None):
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
    
    tbe_note = ""
    if tbe_data and tbe_data.get("stage"):
        tbe_note = f" | TBE={tbe_data['stage']}"
    
    votes = council_data
    print(f"🎯 EXECUTE: {side.upper()} {qty:.4f} @ {price:.6f} | "
          f"mode={mode} | votes={votes['b']}/{votes['s']} score={votes['score_b']:.1f}/{votes['score_s']:.1f}"
          f"{gz_note}{tbe_note}", flush=True)

    try:
        if MODE_LIVE:
            ex.set_leverage(LEVERAGE, SYMBOL, params={"side": "BOTH"})
            ex.create_order(SYMBOL, "market", side, qty, None, _params_open(side))
        
        log_g(f"✅ EXECUTED: {side.upper()} {qty:.4f} @ {price:.6f}")
        return True
    except Exception as e:
        log_e(f"❌ EXECUTION FAILED: {e}")
        return False

# =================== ENHANCED TRADE EXECUTION ===================
def open_market_enhanced(side, qty, price, zone=None, tbe_data=None):
    if qty <= 0: 
        log_e("skip open (qty<=0)")
        return False
    
    df = fetch_ohlcv()
    snap = emit_snapshots(ex, SYMBOL, df)
    
    votes = snap["cv"]
    mode_data = decide_strategy_mode(df, 
                                   adx=votes["ind"].get("adx"),
                                   di_plus=votes["ind"].get("plus_di"),
                                   di_minus=votes["ind"].get("minus_di"),
                                   rsi_ctx=rsi_ma_context(df))
    
    mode = mode_data["mode"]
    gz = snap["gz"]
    
    management_config = setup_trade_management(mode)
    
    success = execute_trade_decision(side, price, qty, mode, votes, gz, tbe_data)
    
    if success:
        STATE.update({
            "open": True, 
            "side": "long" if side=="buy" else "short", 
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
            "tbe_zone": zone,  # حفظ المنطقة إذا كانت من TBE
            "tbe_entry_time": int(time.time()),
            "bars_in_trade": 0
        })
        
        save_state({
            "in_position": True,
            "side": "LONG" if side.upper().startswith("B") else "SHORT",
            "entry_price": price,
            "position_qty": qty,
            "leverage": LEVERAGE,
            "mode": mode,
            "management": management_config,
            "gz_snapshot": gz if isinstance(gz, dict) else {},
            "cv_snapshot": votes if isinstance(votes, dict) else {},
            "tbe_zone": zone if isinstance(zone, dict) else {},
            "opened_at": int(time.time()),
            "partial_taken": False,
            "breakeven_armed": False,
            "trail_active": False,
            "trail_tightened": False,
        })
        
        log_trade_open(
            side=side, price=price, qty=qty, leverage=LEVERAGE,
            source="Trend Birth Engine" if zone else "Council ELITE ENHANCED",
            mode=mode,
            risk_alloc=RISK_ALLOC,
            council=votes,
            gz=gz,
            mgmt=management_config,
            tbe_data=tbe_data
        )
        
        log_g(f"✅ POSITION OPENED: {side.upper()} | mode={mode} | {'TBE' if zone else 'Council'}")
        return True
    
    return False

open_market = open_market_enhanced

# =================== INDICATORS ===================
def wilder_ema(s: pd.Series, n: int): 
    return s.ewm(alpha=1/n, adjust=False).mean()

def compute_indicators(df: pd.DataFrame):
    if len(df) < max(ATR_LEN, RSI_LEN, ADX_LEN) + 2:
        return {"rsi":50.0,"plus_di":0.0,"minus_di":0.0,"dx":0.0,"adx":0.0,"atr":0.0}
    c,h,l = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float)
    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = wilder_ema(tr, ATR_LEN)

    delta=c.diff(); up=delta.clip(lower=0.0); dn=(-delta).clip(lower=0.0)
    rs = wilder_ema(up, RSI_LEN) / wilder_ema(dn, RSI_LEN).replace(0,1e-12)
    rsi = 100 - (100/(1+rs))

    up_move=h.diff(); down_move=l.shift(1)-l
    plus_dm=up_move.where((up_move>down_move)&(up_move>0),0.0)
    minus_dm=down_move.where((down_move>up_move)&(down_move>0),0.0)
    plus_di=100*(wilder_ema(plus_dm, ADX_LEN)/atr.replace(0,1e-12))
    minus_di=100*(wilder_ema(minus_dm, ADX_LEN)/atr.replace(0,1e-12))
    dx=(100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,1e-12)).fillna(0.0)
    adx=wilder_ema(dx, ADX_LEN)

    i=len(df)-1
    return {
        "rsi": float(rsi.iloc[i]), "plus_di": float(plus_di.iloc[i]),
        "minus_di": float(minus_di.iloc[i]), "dx": float(dx.iloc[i]),
        "adx": float(adx.iloc[i]), "atr": float(atr.iloc[i])
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
    "tbe_zone": None, "tbe_entry_time": 0, "bars_in_trade": 0
}
compound_pnl = 0.0
wait_for_next_signal_side = None

# =================== WAIT FOR NEXT SIGNAL - ENHANCED ===================
def _arm_wait_after_close(prev_side):
    """NO WAITING - جاهز فورًا لصفقة جديدة"""
    global wait_for_next_signal_side
    wait_for_next_signal_side = None  # ⬅️ لا انتظار للإشارة المعاكسة
    log_i("🔄 نظام الانتظار معطل - جاهز لصفقة جديدة فورًا")

def wait_gate_allow(df, info):
    """التحقق من بوابة الانتظار - دائماً مسموح"""
    return True, ""  # ⬅️ دائماً يسمح بالدخول

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
    STATE.update({
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
        "tp1_done": False, "highest_profit_pct": 0.0, "profit_targets_achieved": 0,
        "trail_tightened": False, "partial_taken": False,
        "tbe_zone": None, "tbe_entry_time": 0, "bars_in_trade": 0
    })
    save_state({"in_position": False, "position_qty": 0})
    
    # NO WAITING - جاهز فورًا
    _arm_wait_after_close(prev_side)
    logging.info(f"AFTER_CLOSE ready for next trade immediately")

# =================== ENHANCED TRADE MANAGEMENT ===================
def manage_after_entry_enhanced(df, ind, info):
    """إدارة محسنة للمركز مع خروج ذكي حسب النمط"""
    if not STATE["open"] or STATE["qty"] <= 0:
        return

    px = info["price"]
    entry = STATE["entry"]
    side = STATE["side"]
    qty = STATE["qty"]
    mode = STATE.get("mode", "trend")
    management = STATE.get("management", {})
    
    pnl_pct = (px - entry) / entry * 100 * (1 if side == "long" else -1)
    STATE["pnl"] = pnl_pct
    
    if pnl_pct > STATE["highest_profit_pct"]:
        STATE["highest_profit_pct"] = pnl_pct

    snap = emit_snapshots(ex, SYMBOL, df)
    gz = snap["gz"]
    
    # فحص Fail-Fast إذا كانت الصفقة من TBE
    if STATE.get("tbe_zone") and STATE.get("bars_in_trade", 0) <= TBE_FAILFAST_BARS:
        if tbe_failfast_check(df, STATE["tbe_zone"], side, entry, STATE["bars_in_trade"]):
            log_w(f"🚨 TBE FAIL-FAST: Zone broken, closing immediately")
            close_market_strict("TBE_FAILFAST")
            # إضافة المنطقة للقائمة السوداء
            tbe_blacklist_zone(STATE["tbe_zone"])
            # الدخول في وضع REHUNT
            TBE_STATE["state"] = "REHUNT"
            TBE_STATE["ttl_bars"] = TBE_REHUNT_COOLDOWN_BARS
            return
    
    exit_signal = smart_exit_guard(STATE, df, ind, snap["flow"], snap["bm"], 
                                 px, pnl_pct/100, mode, side, entry, gz)
    
    if exit_signal["log"]:
        print(f"🔔 {exit_signal['log']}", flush=True)

    if exit_signal["action"] == "partial" and not STATE.get("partial_taken"):
        partial_qty = safe_qty(qty * exit_signal.get("qty_pct", 0.3))
        if partial_qty > 0:
            close_side = "sell" if side == "long" else "buy"
            if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                try:
                    ex.create_order(SYMBOL, "market", close_side, partial_qty, None, _params_close())
                    log_g(f"✅ PARTIAL CLOSE: {partial_qty:.4f} | {exit_signal['why']}")
                    STATE["partial_taken"] = True
                    STATE["qty"] = safe_qty(qty - partial_qty)
                except Exception as e:
                    log_e(f"❌ Partial close failed: {e}")
            else:
                log_i(f"DRY_RUN: Partial close {partial_qty:.4f}")
    
    elif exit_signal["action"] == "tighten" and not STATE.get("trail_tightened"):
        STATE["trail_tightened"] = True
        STATE["trail"] = None
        log_i(f"🔄 TRAIL TIGHTENED: {exit_signal['why']}")
    
    elif exit_signal["action"] == "close":
        log_w(f"🚨 SMART EXIT: {exit_signal['why']}")
        close_market_strict(f"smart_exit_{exit_signal['why']}")
        return

    current_atr = ind.get("atr", 0.0)
    tp1_pct = management.get("tp1_pct", TP1_PCT_SCALP)
    be_activate_pct = management.get("be_activate_pct", BE_AFTER_SCALP)
    trail_activate_pct = management.get("trail_activate_pct", TRAIL_ACT_SCALP)
    atr_trail_mult = management.get("atr_trail_mult", ATR_TRAIL_MULT)

    if not STATE.get("tp1_done") and pnl_pct/100 >= tp1_pct:
        close_fraction = 0.5  # Close 50% at TP1
        close_qty = safe_qty(STATE["qty"] * close_fraction)
        if close_qty > 0:
            close_side = "sell" if STATE["side"] == "long" else "buy"
            if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                try:
                    ex.create_order(SYMBOL, "market", close_side, close_qty, None, _params_close())
                    log_g(f"✅ TP1 HIT: closed {close_fraction*100}%")
                except Exception as e:
                    log_e(f"❌ TP1 close failed: {e}")
            STATE["qty"] = safe_qty(STATE["qty"] - close_qty)
            STATE["tp1_done"] = True
            STATE["profit_targets_achieved"] += 1

    if not STATE.get("breakeven_armed") and pnl_pct/100 >= be_activate_pct:
        STATE["breakeven_armed"] = True
        STATE["breakeven"] = entry
        log_i("BREAKEVEN ARMED")

    if not STATE.get("trail_active") and pnl_pct/100 >= trail_activate_pct:
        STATE["trail_active"] = True
        log_i("TRAIL ACTIVATED")

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

    if STATE.get("trail"):
        if (side == "long" and px <= STATE["trail"]) or (side == "short" and px >= STATE["trail"]):
            log_w(f"TRAIL STOP: {px} vs trail {STATE['trail']}")
            close_market_strict("trail_stop")

    if STATE.get("breakeven"):
        if (side == "long" and px <= STATE["breakeven"]) or (side == "short" and px >= STATE["breakeven"]):
            log_w(f"BREAKEVEN STOP: {px} vs breakeven {STATE['breakeven']}")
            close_market_strict("breakeven_stop")

    if STATE["qty"] <= FINAL_CHUNK_QTY:
        log_w(f"DUST GUARD: qty {STATE['qty']} <= {FINAL_CHUNK_QTY}, closing...")
        close_market_strict("dust_guard")

manage_after_entry = manage_after_entry_enhanced

# =================== ENHANCED TRADE LOOP - TREND BIRTH ENGINE ===================
def trade_loop_enhanced():
    """حلقة تداول محسنة مع Trend Birth Engine و Council Elite"""
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
            
            # تحديث الـ Snapshots
            snap = emit_snapshots(ex, SYMBOL, df,
                                balance_fn=lambda: float(bal) if bal else None,
                                pnl_fn=lambda: float(compound_pnl))
            
            # تحديث حالة الربح/الخسارة
            if STATE["open"] and px:
                STATE["pnl"] = (px-STATE["entry"])*STATE["qty"] if STATE["side"]=="long" else (STATE["entry"]-px)*STATE["qty"]
            
            # إدارة الصفقة المفتوحة
            if STATE["open"]:
                manage_after_entry(df, ind, {
                    "price": px or info["price"], 
                    "bm": snap["bm"],
                    "flow": snap["flow"],
                    **info
                })
                
                # تحديث عدد الشمعات في الصفقة
                STATE["bars_in_trade"] = STATE.get("bars_in_trade", 0) + 1
            
            # 🔍 تشخيص مفصل
            council_data = council_votes_pro(df)
            
            # الحصول على سياق HTF و Daily Open
            htf_ctx = get_htf_ctx()
            daily_open_ctx = get_daily_open_ctx()
            
            # قرار الدخول باستخدام نظام محسن
            reason = None
            if spread_bps is not None and spread_bps > MAX_SPREAD_BPS:
                reason = f"spread too high ({fmt(spread_bps,2)}bps > {MAX_SPREAD_BPS})"

            sig = None
            trade_decision = None
            tbe_data = None
            zone = None

            # ⚡ فحص الفرص السريعة أولاً
            fast_opp = detect_fast_opportunity(df, council_data)
            if fast_opp and not STATE["open"] and reason is None:
                action = fast_opp["action"]
                if action == "fast_buy":
                    sig = "buy"
                    trade_decision = {"enter": True, "side": "BUY", "reason": fast_opp["reason"], "source": "FAST"}
                else:
                    sig = "sell"
                    trade_decision = {"enter": True, "side": "SELL", "reason": fast_opp["reason"], "source": "FAST"}
            
            # Trend Birth Engine - إذا لم تكن هناك فرصة سريعة
            elif TBE_ENABLED and not STATE["open"] and reason is None:
                tbe_decision = tbe_update(df, htf_ctx, daily_open_ctx, ind)
                if tbe_decision.get("enter"):
                    sig = tbe_decision["side"].lower()
                    trade_decision = {**tbe_decision, "source": "TBE"}
                    zone = tbe_decision.get("zone")
                    tbe_data = {
                        "stage": tbe_decision.get("stage", "ENTRY"),
                        "score": tbe_decision.get("score", 0),
                        "reason": tbe_decision.get("reason", "")
                    }
                    log_g(f"🎯 TBE Decision: {tbe_decision['side']} score={tbe_decision.get('score',0):.1f}")
            
            # Council Elite - كخيار احتياطي
            elif not STATE["open"] and reason is None and (not trade_decision or not trade_decision.get("enter")):
                if council_data["score_b"] >= COUNCIL_STRONG_TH and council_data["b"] > council_data["s"]:
                    sig = "buy"
                    trade_decision = {"enter": True, "side": "BUY", "reason": "COUNCIL_BUY", "source": "COUNCIL"}
                elif council_data["score_s"] >= COUNCIL_STRONG_TH and council_data["s"] > council_data["b"]:
                    sig = "sell"
                    trade_decision = {"enter": True, "side": "SELL", "reason": "COUNCIL_SELL", "source": "COUNCIL"}
            
            # تنفيذ الدخول إذا كان هناك إشارة
            if sig and trade_decision and trade_decision.get("enter"):
                qty = compute_size(bal, px or info["price"])
                if qty > 0:
                    # إذا كان القرار من TBE، نمرر المنطقة
                    if trade_decision.get("source") == "TBE":
                        ok = open_market(sig, qty, px or info["price"], zone, tbe_data)
                        if ok:
                            log_i(f"✅ TBE entry: {sig.upper()} - score={tbe_data.get('score',0):.1f}")
                    else:
                        ok = open_market(sig, qty, px or info["price"])
                        if ok:
                            log_i(f"✅ {'FAST' if trade_decision.get('source') == 'FAST' else 'Council'} entry: {sig.upper()}")
                else:
                    reason = "qty<=0"
                    log_w(f"⚠️ Cannot enter: quantity is zero or negative")
            
            # 🔍 لوج التشخيص إذا لم يتم الدخول
            if not STATE["open"] and not sig:
                reason_str = reason or (trade_decision.get('reason') if trade_decision else "No signal")
                print(f"🔍 لا توجد صفقة | السبب: {reason_str} | الانتشار: {spread_bps}", flush=True)

            # ⚡ نوم أقصر بين الدورات
            sleep_time = 0.5 if time_to_candle_close(df) <= 30 else BASE_SLEEP
            time.sleep(sleep_time)
            
        except Exception as e:
            log_e(f"loop error: {e}\n{traceback.format_exc()}")
            logging.error(f"trade_loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

trade_loop = trade_loop_enhanced

# =================== LOOP / LOG ===================
def pretty_snapshot(bal, info, ind, spread_bps, reason=None, df=None):
    if LOG_LEGACY:
        left_s = time_to_candle_close(df) if df is not None else 0
        print(colored("─"*100,"cyan"))
        print(colored(f"📊 {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-d %H:%M:%S')} UTC","cyan"))
        print(colored("─"*100,"cyan"))
        print("📈 INDICATORS & RF")
        print(f"   💲 Price {fmt(info.get('price'))} | RF filt={fmt(info.get('filter'))}  hi={fmt(info.get('hi'))} lo={fmt(info.get('lo'))}")
        print(f"   🧮 RSI={fmt(ind.get('rsi'))}  +DI={fmt(ind.get('plus_di'))}  -DI={fmt(ind.get('minus_di'))}  ADX={fmt(ind.get('adx'))}  ATR={fmt(ind.get('atr'))}")
        print(f"   🎯 ENTRY: COUNCIL ELITE ENHANCED + TREND BIRTH ENGINE  |  spread_bps={fmt(spread_bps,2)}")
        print(f"   ⏱️ closes_in ≈ {left_s}s")
        print("\n🧭 POSITION")
        bal_line = f"Balance={fmt(bal,2)}  Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x  CompoundPnL={fmt(compound_pnl)}  Eq~{fmt((bal or 0)+compound_pnl,2)}"
        print(colored(f"   {bal_line}", "yellow"))
        if STATE["open"]:
            lamp='🟩 LONG' if STATE['side']=='long' else '🟥 SHORT'
            print(f"   {lamp}  Entry={fmt(STATE['entry'])}  Qty={fmt(STATE['qty'],4)}  Bars={STATE['bars']}  Trail={fmt(STATE['trail'])}  BE={fmt(STATE['breakeven'])}")
            print(f"   🎯 TP_done={STATE['profit_targets_achieved']}  HP={fmt(STATE['highest_profit_pct'],2)}%")
            if STATE.get('tbe_zone'):
                zone = STATE['tbe_zone']
                print(f"   🚀 TBE Zone: {zone['type']} [{fmt(zone['low'])}-{fmt(zone['high'])}]  Bars in trade: {STATE.get('bars_in_trade',0)}")
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
    return f"✅ Council ELITE Bot ENHANCED — {SYMBOL} {INTERVAL} — {mode} — Trend Birth Engine v1"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "state": STATE, "compound_pnl": compound_pnl,
        "entry_mode": "TREND_BIRTH_ENGINE_v1", "wait_for_next_signal": wait_for_next_signal_side,
        "guards": {"max_spread_bps": MAX_SPREAD_BPS, "final_chunk_qty": FINAL_CHUNK_QTY},
        "fast_trading": FAST_TRADE_ENABLED,
        "trend_birth_engine": {
            "enabled": TBE_ENABLED,
            "state": TBE_STATE["state"],
            "direction": TBE_STATE["dir"],
            "blacklisted_zones": len(TBE_STATE["blacklisted_zones"])
        }
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True, "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"], "side": STATE["side"], "qty": STATE["qty"],
        "compound_pnl": compound_pnl, "timestamp": datetime.utcnow().isoformat(),
        "entry_mode": "TREND_BIRTH_ENGINE_v1", "wait_for_next_signal": wait_for_next_signal_side,
        "fast_trading": FAST_TRADE_ENABLED,
        "trend_birth_engine": TBE_ENABLED
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
    log_banner("COUNCIL ELITE ENHANCED + TREND BIRTH ENGINE v1 INIT")
    state = load_state() or {}
    state.setdefault("in_position", False)

    if RESUME_ON_RESTART:
        try:
            state = resume_open_position_enhanced(ex, SYMBOL, state)
        except Exception as e:
            log_w(f"resume error: {e}\n{traceback.format_exc()}")

    verify_execution_environment()

    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'}  •  {SYMBOL}  •  {INTERVAL}", "yellow"))
    print(colored(f"RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x  •  COUNCIL_ELITE_ENHANCED=ENABLED", "yellow"))
    print(colored(f"SMC/ICT: Golden Zones + FVG + BOS + Sweeps + Order Blocks", "yellow"))
    print(colored(f"MANAGEMENT: Smart TP + Smart Exit + Trail Adaptation", "yellow"))
    print(colored(f"FAST TRADING: {'ENABLED' if FAST_TRADE_ENABLED else 'DISABLED'}", "yellow"))
    print(colored(f"TREND BIRTH ENGINE: {'ENABLED' if TBE_ENABLED else 'DISABLED'}", "yellow"))
    print(colored(f"EXECUTION: {'ACTIVE' if EXECUTE_ORDERS and not DRY_RUN else 'SIMULATION'}", "yellow"))
    
    logging.info("Council ELITE ENHANCED + Trend Birth Engine v1 service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    
    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)

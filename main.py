# -*- coding: utf-8 -*-
"""
RF Futures Bot — RF-LIVE ONLY (BingX Perp via CCXT)
• Council ELITE Unified Decision System with Smart Management
• Golden Entry + SMC/ICT + Smart Exit Management
• Dynamic TP ladder + Breakeven + ATR-trailing
• Professional Logging & Dashboard
• ENHANCED VERSION - Quality Focused Scalp System
• QUALITY SCALP PROTECTION - High Quality Trades Only
• TREND COOLDOWN SYSTEM - Protection after strong trends
• STRATEGY AVOID MODE - Avoid weak market conditions
• SMART CHOP DETECTION - Avoid choppy markets
• INTELLIGENT ENTRY SYSTEM - Smart trend and signal detection
• FORBIDDEN ZONES DETECTION - Prevent scalp in dangerous areas
• VOLUME & RSI CROSS VALIDATION - Mandatory for scalp trades
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
BOT_VERSION = "DOGE Council ELITE v10.0 — Ultimate Quality Focused Scalp System"
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

# Management profiles - UPDATED FOR STRONGER SCALP
TP1_PCT_SCALP = 0.0050   # ⬆️ من 0.40% إلى 0.50%
TP1_PCT_TREND = 0.0060   # 0.60%
BE_AFTER_SCALP = 0.0040  # ⬆️ من 0.30% إلى 0.40%
BE_AFTER_TREND = 0.0040  # 0.40%
TRAIL_ACT_SCALP = 0.0100 # ⬆️ من 0.80% إلى 1.00%
TRAIL_ACT_TREND = 0.0120 # 1.20%
ATR_TRAIL_MULT = 1.6
TRAIL_TIGHT_MULT = 1.2

# =================== QUALITY FOCUSED SCALP PROTECTION SETTINGS ===================
SCALP_MIN_SCORE = 5.0  # ⬆️ زيادة من 3.0 إلى 5.0
SCALP_MIN_VOTES = 4    # ⬆️ زيادة من 2 إلى 4 أصوات
SCALP_ADX_RANGE = (16, 25)  # نطاق ADX مثالي للسكالب
SCALP_RSI_RANGE = (35, 65)  # نطاق RSI آمن للسكالب
SCALP_MIN_FLOW_Z = 0.6      # عتبة تدفق أعلى
SCALP_COOLDOWN_MINUTES = 40   # ⬇️ تقليل من 90 إلى 40 دقيقة (مرونة أكثر)
SCALP_QUALITY_THRESHOLD = 6.0 # عتبة جودة دنيا

# Trend Cooldown System - تقليل المدة
TREND_COOLDOWN_HOURS = 2  # ⬇️ تقليل من 4 إلى 2 ساعة

# Smart Entry System - NEW
STRONG_TREND_ADX = 20     # عتبة الترند القوي
VERY_STRONG_TREND_ADX = 25 # عتبة الترند القوي جداً
MIN_COUNCIL_SCORE = 2.5   # ⬇️ عتبة أقل للمجلس
STRONG_FLOW_Z = 0.8       # عتبة التدفق القوي

# Decision thresholds - RELAXED FOR MORE TRADES
COUNCIL_STRONG_TH = 5.0   # ⬇️ كان 8.0 - تخفيض 37%
COUNCIL_OK_TH = 4.0       # ⬇️ كان 7.0 - تخفيض 43%

# Smart Exit Tuning
TP1_SCALP_PCT = 0.0050    # ⬆️ تحديث مع TP1_PCT_SCALP
TP1_TREND_PCT = 0.0060
HARD_CLOSE_PNL_PCT = 0.0110
WICK_ATR_MULT = 1.5
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
                   council:dict=None, gz:dict=None, mgmt:dict=None):
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

    msg = f"{lamp} • {source} • {mode.upper()} | Price={p} Qty={q} Lev={lev} Risk={ra}{c_part}{gz_part}{mg_part}"

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

# =================== QUALITY FOCUSED SCALP PROTECTION SYSTEM ===================
# تتبع صفقات السكالب
last_scalp_time = 0

def update_scalp_trade_timestamp():
    """تحديث وقت آخر صفقة سكالب"""
    global last_scalp_time
    last_scalp_time = time.time()

def is_in_scalp_cooldown():
    """التحقق من فترة التبريد بين صفقات السكالب"""
    if last_scalp_time == 0:
        return False, ""
    
    cooldown_end = last_scalp_time + (SCALP_COOLDOWN_MINUTES * 60)
    remaining = cooldown_end - time.time()
    
    if remaining > 0:
        mins_left = remaining / 60
        return True, f"تبديد سكالب - متبقي {mins_left:.1f} دقيقة"
    
    return False, ""

# =================== FORBIDDEN ZONES DETECTION ===================
def detect_forbidden_zones(df, council_data, current_price):
    """
    كشف المناطق المحظورة للسكالب بناء على 4 شروط رئيسية
    """
    ind = council_data["ind"]
    forbidden_reasons = []
    
    # 1) نطاق ضيق (ATR منخفض + ADX منخفض)
    atr = ind.get('atr', 0.0)
    atr_pct = (atr / current_price) * 100 if current_price > 0 else 0
    adx = ind.get('adx', 0)
    
    if atr_pct < 0.15 and adx < 12:
        forbidden_reasons.append("نطاق سعري ضيق جداً (ATR منخفض + ADX منخفض)")
    
    # 2) ضد الاتجاه الرئيسي للـ15m
    if len(df) >= 100:  # تحتاج بيانات كافية لتحليل الاتجاه
        # تحليل الاتجاه على الإطار 15m (باستخدام 20 شمعة سابقة)
        closes_15m = df['close'].astype(float).tail(20)
        trend_15m = "up" if closes_15m.iloc[-1] > closes_15m.iloc[0] else "down"
        
        # تحليل الاتجاه الحالي على الإطار الحالي
        current_trend = "up" if current_price > closes_15m.iloc[-5] else "down"
        
        if current_trend != trend_15m:
            forbidden_reasons.append(f"ضد الاتجاه الرئيسي للـ15m ({trend_15m.upper()})")
    
    # 3) ذيول سيولة كبيرة (Liquidity Sweep)
    sweep = ind.get('sweep', {})
    if sweep.get('ok'):
        forbidden_reasons.append("وجود Liquidity Sweep كبير")
    
    # 4) شمعة ذات ذيول كبيرة جداً
    if len(df) >= 2:
        current_candle = df.iloc[-1]
        high = float(current_candle['high'])
        low = float(current_candle['low'])
        open_price = float(current_candle['open'])
        close_price = float(current_candle['close'])
        
        body_size = abs(close_price - open_price)
        total_range = high - low
        upper_wick = high - max(open_price, close_price)
        lower_wick = min(open_price, close_price) - low
        
        # إذا كانت الذيول أكبر من الجسم بثلاثة أضعاف
        if upper_wick > body_size * 3 or lower_wick > body_size * 3:
            forbidden_reasons.append("شمعة ذات ذيول كبيرة جداً (فخ سيولة)")
    
    return forbidden_reasons

# =================== VOLUME & RSI CROSS VALIDATION ===================
def validate_volume_and_rsi(df, council_data, current_price):
    """
    التحقق من شرطي الفوليوم وتقاطع RSI للسكالب
    """
    ind = council_data["ind"]
    validation_errors = []
    
    # 1) فحص الفوليوم: Volume الحالي يجب أن يكون أعلى من MA20
    if len(df) >= 20:
        current_volume = float(df['volume'].iloc[-1])
        volume_ma_20 = df['volume'].tail(20).astype(float).mean()
        
        if current_volume <= volume_ma_20:
            validation_errors.append(f"الفوليوم ضعيف ({current_volume:.0f} ≤ {volume_ma_20:.0f})")
    
    # 2) فحص تقاطع RSI
    rsi = ind.get('rsi', 50)
    rsi_ma = ind.get('rsi_ma', 50)
    rsi_cross = ind.get('rsi_cross', 'none')
    
    # شرط التقاطع الإلزامي للسكالب
    if rsi_cross == 'none':
        validation_errors.append("لا يوجد تقاطع RSI")
    else:
        # شروط إضافية للتقاطع
        if rsi_cross == 'bull' and rsi >= 70:
            validation_errors.append("RSI في ذروة شراء رغم التقاطع الصاعد")
        elif rsi_cross == 'bear' and rsi <= 30:
            validation_errors.append("RSI في ذروة بيع رغم التقاطع الهابط")
    
    return validation_errors

def is_scalp_allowed(df, council_data, current_price):
    """
    يقرر إذا كان مسموحاً بدخول صفقة سكالب
    """
    # 1) فحص المناطق المحظورة
    forbidden_zones = detect_forbidden_zones(df, council_data, current_price)
    if forbidden_zones:
        return False, f"منطقة محظورة: {forbidden_zones[0]}"
    
    # 2) فحص الفوليوم وتقاطع RSI
    volume_rsi_errors = validate_volume_and_rsi(df, council_data, current_price)
    if volume_rsi_errors:
        return False, f"تحقق فني: {volume_rsi_errors[0]}"
    
    # 3) فحص الظروف الأساسية للسكالب
    ind = council_data["ind"]
    
    # شروط السماح بالسكالب
    required_conditions = [
        ind.get('adx', 0) >= 16,           # ADX كافي للاتجاه
        ind.get('atr', 0) > 0,             # ATR غير معدوم
        council_data.get('b', 0) >= 2 or council_data.get('s', 0) >= 2,  # تصويت كافي
        ind.get('rsi', 50) < 70,           # RSI ليس في ذروة شراء
        ind.get('rsi', 50) > 30,           # RSI ليس في ذروة بيع
    ]
    
    if not all(required_conditions):
        return False, "ظروف السوق غير مناسبة للسكالب"
    
    return True, "مسموح بالسكالب"

# =================== ENHANCED QUALITY SCALP CHECK ===================
def enhanced_high_quality_scalp(df, council_data, current_price):
    """
    نسخة محسنة من فحص جودة السكالب مع المناطق المحظورة + الفوليوم + RSI
    """
    # أولاً: فحص السماح الأساسي بالسكالب
    scalp_allowed, allow_reason = is_scalp_allowed(df, council_data, current_price)
    if not scalp_allowed:
        return False, allow_reason
    
    # ثانياً: فحص الجودة المتقدم
    ind = council_data["ind"]
    score_b = council_data["score_b"]
    score_s = council_data["score_s"]
    votes_b = council_data["b"]
    votes_s = council_data["s"]
    
    quality_score = 0
    max_quality_score = 10
    reasons = []
    
    # 1) قوة المجلس (3 نقاط)
    council_strength = max(score_b, score_s)
    if council_strength >= SCALP_MIN_SCORE:
        quality_score += 3
        reasons.append(f"مجلس قوي ({council_strength:.1f})")
    elif council_strength >= 4.0:
        quality_score += 2
        reasons.append(f"مجلس جيد ({council_strength:.1f})")
    else:
        return False, "مجلس ضعيف للسكالب"
    
    # 2) المؤشرات الفنية (4 نقاط) - ⬆️ زيادة الأهمية
    adx = ind.get('adx', 0)
    rsi = ind.get('rsi', 50)
    di_spread = ind.get('di_spread', 0)
    atr = ind.get('atr', 0)
    atr_pct = (atr / current_price) * 100 if current_price > 0 else 0
    rsi_cross = ind.get('rsi_cross', 'none')
    
    tech_points = 0
    
    # ✅ شرط RSI Cross الإلزامي (نقطة إضافية)
    if rsi_cross != 'none':
        tech_points += 1
        reasons.append(f"تقاطع RSI {rsi_cross}")
    
    if SCALP_ADX_RANGE[0] <= adx <= SCALP_ADX_RANGE[1]:
        tech_points += 1
    if SCALP_RSI_RANGE[0] <= rsi <= SCALP_RSI_RANGE[1]:
        tech_points += 1  
    if di_spread >= 4.0:
        tech_points += 1
    if 0.3 <= atr_pct <= 1.0:  # ATR معقول (ليس كبير جداً ولا صغير جداً)
        tech_points += 1
        
    if tech_points >= 4:  # ⬆️ زيادة العتبة
        quality_score += 4
        reasons.append(f"مؤشرات قوية (ADX:{adx:.1f}, RSI:{rsi:.1f}, ATR:{atr_pct:.2f}%)")
    elif tech_points >= 3:
        quality_score += 3
        reasons.append(f"مؤشرات جيدة (ADX:{adx:.1f}, RSI:{rsi:.1f})")
    else:
        return False, "مؤشرات تقنية ضعيفة للسكالب"
    
    # 3) الفوليوم القوي (2 نقطة) - ⬆️ زيادة الأهمية
    if len(df) >= 20:
        current_volume = float(df['volume'].iloc[-1])
        volume_ma_20 = df['volume'].tail(20).astype(float).mean()
        volume_ratio = current_volume / volume_ma_20 if volume_ma_20 > 0 else 1.0
        
        if volume_ratio >= 1.2:
            quality_score += 2
            reasons.append(f"فوليوم قوي (x{volume_ratio:.1f})")
        elif volume_ratio >= 1.0:
            quality_score += 1
            reasons.append(f"فوليوم جيد (x{volume_ratio:.1f})")
        else:
            # الفوليوم الضعيف يخفض الجودة
            quality_score -= 1
            reasons.append(f"فوليوم ضعيف (x{volume_ratio:.1f})")
    
    # 4) التدفق والكتاب (2 نقطة)
    flow = ind.get('flow', {})
    bm = ind.get('bm', {})
    
    flow_points = 0
    if flow.get('ok') and abs(flow.get('delta_z', 0)) >= SCALP_MIN_FLOW_Z:
        flow_points += 1
    if bm.get('ok') and (bm.get('imbalance', 1.0) >= 1.2 or bm.get('imbalance', 1.0) <= 0.8):
        flow_points += 1
        
    if flow_points >= 1:
        quality_score += 2
        reasons.append("تدفق/كتاب قوي")
    
    # 5) SMC/ICT إضافية (2 نقطة)
    smc_points = 0
    fvg = ind.get('fvg', {})
    gz = ind.get('gz', {})
    bos = ind.get('bos', {})
    
    if fvg.get('ok') and fvg.get('bps', 0) >= 8.0:
        smc_points += 1
    if gz.get('ok') and gz.get('score', 0) >= 4.0:
        smc_points += 1
    if bos.get('ok'):
        smc_points += 1
        
    if smc_points >= 1:
        quality_score += 2
        reasons.append("إشارات SMC/ICT")
    
    # 6) نسبة المخاطرة/العائد (نقطة إضافية)
    expected_profit = TP1_PCT_SCALP
    stop_loss_pct = (atr * 2.0) / current_price
    rr_ratio = expected_profit / stop_loss_pct
    
    if rr_ratio >= 1.8:  # نسبة مخاطرة/عائد ممتازة
        quality_score += 1
        reasons.append(f"R/R ممتاز ({rr_ratio:.2f})")
    elif rr_ratio >= 1.5:
        quality_score += 0.5
        reasons.append(f"R/R جيد ({rr_ratio:.2f})")
    
    # القرار النهائي بناء على الجودة
    if quality_score >= 8.0:
        return True, f"سكالب استثنائي ({quality_score:.1f}/10): {', '.join(reasons)}"
    elif quality_score >= SCALP_QUALITY_THRESHOLD:
        return True, f"سكالب عالي الجودة ({quality_score:.1f}/10): {', '.join(reasons)}"
    else:
        return False, f"جودة غير كافية للسكالب ({quality_score:.1f}/10)"

def log_quality_decision(decision, details, council_data, quality_score):
    """تسجيل قرار الجودة مع التفاصيل"""
    score_b = council_data["score_b"]
    score_s = council_data["score_s"]
    votes_b = council_data["b"]
    votes_s = council_data["s"]
    
    if decision:
        if quality_score >= 8.0:
            log_g(f"🏆 [سكالب استثنائي] {details} | نقاط: {max(score_b, score_s):.1f}")
        elif quality_score >= 6.0:
            log_g(f"✅ [سكالب عالي الجودة] {details} | نقاط: {max(score_b, score_s):.1f}")
        else:
            log_g(f"🟢 [سكالب جيد] {details} | نقاط: {max(score_b, score_s):.1f}")
    else:
        log_w(f"⏳ [سكالب مؤجل] {details} | نقاط: {max(score_b, score_s):.1f}")

# =================== MARKET CHOP DETECTION SYSTEM ===================
def detect_market_chop(df, council_data, current_price):
    """
    كشف حالات التذبذب وعدم الاتجاه في السوق
    """
    ind = council_data["ind"]
    
    chop_signals = []
    
    # 1) ADX مرتفع لكن DI متقارب (ترند ضعيف)
    adx = ind.get('adx', 0)
    di_plus = ind.get('plus_di', 0)
    di_minus = ind.get('minus_di', 0)
    di_spread = abs(di_plus - di_minus)
    
    if adx > 20 and di_spread < 5:
        chop_signals.append(f"ADX مرتفع ({adx:.1f}) لكن DI متقارب ({di_spread:.1f})")
    
    # 2) RSI في منتصف الطريق (40-60) + ADX منخفض
    rsi = ind.get('rsi', 50)
    if 40 <= rsi <= 60 and adx < 18:
        chop_signals.append(f"RSI محايد ({rsi:.1f}) مع ADX منخفض")
    
    # 3) Bookmap متوازن جداً
    bm = council_data.get("ind", {}).get("bm", {})
    if bm.get("ok"):
        imb = bm.get("imbalance", 1.0)
        if 0.9 <= imb <= 1.1:  # توازن تام
            chop_signals.append(f"Bookmap متوازن (imb={imb:.2f})")
    
    # 4) تدفق ضعيف ومتذبذب
    flow = council_data.get("ind", {}).get("flow", {})
    if flow.get("ok"):
        delta_z = flow.get("delta_z", 0)
        if abs(delta_z) < 0.5:  # تدفق ضعيف
            chop_signals.append(f"تدفق ضعيف (z={delta_z:.2f})")
    
    # 5) قرار مجلس ضعيف ومتضارب
    score_b = council_data.get('score_b', 0)
    score_s = council_data.get('score_s', 0)
    if max(score_b, score_s) < 3.0:
        chop_signals.append(f"قرار مجلس ضعيف (B:{score_b:.1f}/S:{score_s:.1f})")
    
    # 6) ATR منخفض (تذبذب سعري قليل)
    atr = ind.get('atr', 0)
    atr_pct = (atr / current_price) * 100 if current_price > 0 else 0
    if atr_pct < 0.2:  # ATR أقل من 0.2%
        chop_signals.append(f"تذبذب سعري منخفض (ATR={atr_pct:.2f}%)")
    
    return chop_signals

def should_avoid_chop_market(chop_signals):
    """
    يقرر إذا كان يجب تجنب السوق المتذبذب
    """
    if len(chop_signals) >= 3:  # إذا توفرت 3 إشارات تذبذب أو أكثر
        return True, chop_signals
    elif len(chop_signals) == 2 and any("ADX مرتفع" in s for s in chop_signals):
        return True, chop_signals
    return False, chop_signals

# =================== SMART ENTRY DECISION SYSTEM ===================
def enhanced_smart_entry_decision(df, council_data, strategy_mode, snap, current_price):
    """
    قرار دخول ذكي محسّن مع مراعاة المناطق المحظورة + الفوليوم + RSI
    """
    # أولاً: فحص تبريد الترند القوي (يمنع كل شيء)
    in_trend_cooldown, trend_cooldown_reason = is_in_trend_cooldown()
    if in_trend_cooldown:
        return None, f"🛑 {trend_cooldown_reason}"
    
    # ثانياً: كشف التذبذب
    chop_signals = detect_market_chop(df, council_data, current_price)
    avoid_chop, chop_details = should_avoid_chop_market(chop_signals)
    
    if avoid_chop:
        return None, f"سوق متذبذب: {', '.join(chop_details[:3])}"
    
    # ثالثاً: فحص المناطق المحظورة للسكالب
    forbidden_zones = detect_forbidden_zones(df, council_data, current_price)
    if forbidden_zones:
        return None, f"منطقة محظورة: {forbidden_zones[0]}"
    
    # رابعاً: فحص تبريد السكالب (يمكن تجاوزه للجودة العالية)
    in_scalp_cooldown, scalp_cooldown_reason = is_in_scalp_cooldown()
    
    # خامساً: تحليل قوة الإشارة
    ind = council_data["ind"]
    score_b = council_data["score_b"]
    score_s = council_data["score_s"]
    votes_b = council_data["b"]
    votes_s = council_data["s"]
    
    # قرار الدخول النهائي
    buy_advantage = (score_b > score_s and votes_b > votes_s)
    sell_advantage = (score_s > score_b and votes_s > votes_b)
    
    # 🔥 التركيز على الجودة بغض النظر عن النمط
    is_quality_trade, quality_reason = enhanced_high_quality_scalp(df, council_data, current_price)
    
    if is_quality_trade and (buy_advantage or sell_advantage):
        # إذا كانت الجودة عالية، يمكن تجاوز تبريد السكالب
        if in_scalp_cooldown:
            # تحقق إذا كانت الجودة استثنائية
            quality_score = float(quality_reason.split('(')[1].split('/')[0])
            if quality_score >= 8.0:  # جودة استثنائية
                log_g(f"🔥 تجاوز تبريد السكالب لجودة استثنائية ({quality_score}/10)")
            else:
                return None, f"⏳ {scalp_cooldown_reason} - الجودة: {quality_score:.1f}/10"
        
        if buy_advantage:
            update_scalp_trade_timestamp()
            quality_score = float(quality_reason.split('(')[1].split('/')[0])
            log_quality_decision(True, quality_reason, council_data, quality_score)
            return "buy", f"🎯 {quality_reason}"
        elif sell_advantage:
            update_scalp_trade_timestamp()
            quality_score = float(quality_reason.split('(')[1].split('/')[0])
            log_quality_decision(True, quality_reason, council_data, quality_score)
            return "sell", f"🎯 {quality_reason}"
    
    return None, f"جودة غير كافية: {quality_reason}"

# =================== PROTECTION SYSTEMS ===================
# Trend Cooldown System
last_strong_trend_time = 0
last_strong_trend_profit = 0.0

def update_strong_trend_timestamp(profit_pct, bars_count):
    """تحديث وقت آخر ترند قوي"""
    global last_strong_trend_time, last_strong_trend_profit
    
    if profit_pct >= 0.008 or bars_count >= 8:
        last_strong_trend_time = time.time()
        last_strong_trend_profit = profit_pct
        log_i(f"🔄 تبديد ترند قوي: ربح {profit_pct*100:.2f}% لمدة {TREND_COOLDOWN_HOURS} ساعات")

def is_in_trend_cooldown():
    """التحقق من فترة التبريد بعد الترند القوي"""
    if last_strong_trend_time == 0:
        return False, ""
    
    cooldown_end = last_strong_trend_time + (TREND_COOLDOWN_HOURS * 3600)
    remaining = cooldown_end - time.time()
    
    if remaining > 0:
        hours_left = remaining / 3600
        return True, f"تبديد ترند قوي سابق ({last_strong_trend_profit*100:.2f}%) - متبقي {hours_left:.1f} ساعة"
    
    return False, ""

# Strict Weak Scalp Protection
def detect_weak_scalp(df, council_data, expected_profit_pct):
    """
    كشف السكالب الضعيف بناء على 3 شروط
    إذا اتجمع شرطين ⇒ رفض الصفقة
    """
    ind = council_data["ind"]
    conditions_met = 0
    reasons = []
    
    # الشرط 1: الهدف الصغير (< 0.4%)
    if expected_profit_pct < 0.004:
        conditions_met += 1
        reasons.append(f"هدف صغير ({expected_profit_pct*100:.2f}%)")
    
    # الشرط 2: RR تعبان (< 1.3)
    atr = ind.get('atr', 0.0)
    current_price = float(df['close'].iloc[-1]) if len(df) > 0 else 0
    if atr > 0 and current_price > 0:
        stop_loss_pct = (atr * 1.8) / current_price
        rr_ratio = expected_profit_pct / stop_loss_pct
        if rr_ratio < 1.3:
            conditions_met += 1
            reasons.append(f"RR ضعيف ({rr_ratio:.2f})")
    
    # الشرط 3: مفيش ترند (ADX < 15 + RSI في المنتصف)
    adx = ind.get('adx', 0)
    rsi = ind.get('rsi', 50)
    if adx < 15 and (40 <= rsi <= 60):
        conditions_met += 1
        reasons.append("لا يوجد ترند واضح")
    
    # القرار النهائي
    if conditions_met >= 2:
        return True, f"سكالب ضعيف: {', '.join(reasons)}"
    
    return False, "سكالب مقبول"

def log_protection_event(event_type, details):
    """تسجيل أحداث نظام الحماية"""
    icons = {
        "weak_scalp": "⚠️",
        "trend_cooldown": "🛑", 
        "strategy_avoid": "🚫",
        "protection_pass": "✅"
    }
    
    icon = icons.get(event_type, "🔔")
    print(f"{icon} [نظام الحماية] {details}", flush=True)

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
    print(f"🎯 COUNCIL ELITE ENHANCED: Quality Focused Scalp System", flush=True)
    print(f"📈 SMC/ICT: Golden Zones + FVG + BOS + Sweeps", flush=True)
    print(f"🛡️ QUALITY FOCUSED SCALP: ACTIVE (Min {SCALP_QUALITY_THRESHOLD}/10)", flush=True)
    print(f"🔄 TREND COOLDOWN SYSTEM: {TREND_COOLDOWN_HOURS} hours", flush=True)
    print(f"🎯 STRATEGY AVOID MODE: ACTIVE", flush=True)
    print(f"🔄 SMART CHOP DETECTION: ACTIVE", flush=True)
    print(f"🚫 FORBIDDEN ZONES DETECTION: ACTIVE", flush=True)
    print(f"📊 VOLUME & RSI CROSS VALIDATION: ACTIVE", flush=True)
    
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

def decide_strategy_mode_enhanced(df, adx=None, di_plus=None, di_minus=None, rsi_ctx=None):
    """نسخة محسنة مع وضع avoid"""
    if adx is None or di_plus is None or di_minus is None:
        ind = compute_indicators(df)
        adx = ind.get('adx', 0)
        di_plus = ind.get('plus_di', 0)
        di_minus = ind.get('minus_di', 0)
    
    if rsi_ctx is None:
        rsi_ctx = rsi_ma_context(df)
    
    di_spread = abs(di_plus - di_minus)
    
    # شروط avoid (تجنب التداول)
    avoid_conditions = [
        adx < 10,                           # ADX منخفض جداً
        di_spread < 2.0,                    # مؤشرات الاتجاه متقاربة
        rsi_ctx["in_chop"] and adx < 12,    # سوق متذبذب بلا اتجاه
        rsi_ctx["rsi"] > 75 or rsi_ctx["rsi"] < 25,  # RSI في مناطق متطرفة
    ]
    
    if any(avoid_conditions):
        return {"mode": "avoid", "why": "سوق ضعيف/متذبذب"}
    
    # شروط trend
    strong_trend = (
        (adx >= 18 and di_spread >= 6.0) or
        (rsi_ctx["trendZ"] in ("bull", "bear") and not rsi_ctx["in_chop"])
    )
    
    mode = "trend" if strong_trend else "scalp"
    why = "adx/di_trend" if adx >= 18 else ("rsi_trendZ" if rsi_ctx["trendZ"] != "none" else "scalp_default")
    
    return {"mode": mode, "why": why}

# =================== COUNCIL ELITE VOTING - ULTIMATE ===================
COUNCIL_BUSY = False
LAST_COUNCIL = {"b": 0, "s": 0, "score_b": 0.0, "score_s": 0.0, "logs": [], "ind": {}}

def council_votes_ultimate(df):
    """
    النسخة النهائية من Council مع جميع أنظمة الحماية
    """
    global COUNCIL_BUSY, LAST_COUNCIL
    if COUNCIL_BUSY:
        return LAST_COUNCIL
        
    COUNCIL_BUSY = True
    try:
        ind = compute_indicators(df)
        rsi_ctx = rsi_ma_context(df)
        atr = ind.get('atr', 0.0)
        current_price = float(df['close'].iloc[-1]) if len(df) > 0 else 0

        # فحص المناطق المحظورة
        council_data_temp = {"ind": ind, "b": 0, "s": 0, "score_b": 0, "score_s": 0}
        forbidden_zones = detect_forbidden_zones(df, council_data_temp, current_price)
        
        # فحص الفوليوم وتقاطع RSI
        volume_rsi_errors = validate_volume_and_rsi(df, council_data_temp, current_price)
        
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

        # 🔒 تطبيق عقوبة المناطق المحظورة والفوليوم
        penalty_multiplier = 1.0
        
        if forbidden_zones:
            penalty_multiplier *= 0.3  # عقوبة شديدة
            logs.append(f"🛑 مناطق محظورة: {forbidden_zones[0]}")
        
        if volume_rsi_errors:
            penalty_multiplier *= 0.5  # عقوبة متوسطة
            logs.append(f"📉 {volume_rsi_errors[0]}")

        # Strong Trend (ADX/DI)
        if adx >= 14:
            if plus_di > minus_di and di_spread > 4.0:
                votes_b += 2
                score_b += 1.2 * penalty_multiplier
                logs.append("📈 ترند صاعد (ADX/DI)")
            elif minus_di > plus_di and di_spread > 4.0:
                votes_s += 2
                score_s += 1.2 * penalty_multiplier
                logs.append("📉 ترند هابط (ADX/DI)")

        # RSI+MA Cross & Trend - ⬆️ زيادة الأهمية
        if rsi_ctx["cross"] == "bull" and rsi_ctx["rsi"] < 65:
            votes_b += 2
            score_b += 1.0 * penalty_multiplier
            logs.append("🟢 RSI-MA إيجابي")
        elif rsi_ctx["cross"] == "bear" and rsi_ctx["rsi"] > 35:
            votes_s += 2
            score_s += 1.0 * penalty_multiplier
            logs.append("🔴 RSI-MA سلبي")

        if rsi_ctx["trendZ"] == "bull":
            votes_b += 2
            score_b += 1.2 * penalty_multiplier
            logs.append("🚀 RSI ترند صاعد")
        elif rsi_ctx["trendZ"] == "bear":
            votes_s += 2
            score_s += 1.2 * penalty_multiplier
            logs.append("💥 RSI ترند هابط")

        # FVG (Fair Value Gap)
        if fvg.get("ok"):
            if fvg["dir"] == "bull":
                votes_b += 1
                score_b += 0.8 * penalty_multiplier
                logs.append(f"🟢 FVG bull {fvg['bps']:.1f}bps")
            else:
                votes_s += 1
                score_s += 0.8 * penalty_multiplier
                logs.append(f"🔴 FVG bear {fvg['bps']:.1f}bps")

        # BOS (Break of Structure)
        if bos.get("ok"):
            if bos["dir"] == "bull":
                votes_b += 1
                score_b += 0.8 * penalty_multiplier
                logs.append("🟩 BOS ↑")
            else:
                votes_s += 1
                score_s += 0.8 * penalty_multiplier
                logs.append("🟥 BOS ↓")

        # Liquidity Sweeps
        if sweep.get("ok"):
            if sweep["dir"] == "bull":
                votes_b += 1
                score_b += 0.5 * penalty_multiplier
                logs.append("💧 Liquidity Sweep (bull)")
            else:
                votes_s += 1
                score_s += 0.5 * penalty_multiplier
                logs.append("💧 Liquidity Sweep (bear)")

        # Order Blocks
        if ob_bull.get("ok"):
            votes_b += 1
            score_b += 0.5 * penalty_multiplier
            logs.append("🟢 OB Demand")
        if ob_bear.get("ok"):
            votes_s += 1
            score_s += 0.5 * penalty_multiplier
            logs.append("🔴 OB Supply")

        # Golden Zones
        if gz and gz.get("ok") and adx >= 14:
            if gz['zone']['type'] == 'golden_bottom':
                votes_b += 2
                score_b += 1.2 * penalty_multiplier
                logs.append(f"🏆 قاع ذهبي s={gz['score']:.1f}")
            elif gz['zone']['type'] == 'golden_top':
                votes_s += 2
                score_s += 1.2 * penalty_multiplier
                logs.append(f"🏆 قمة ذهبية s={gz['score']:.1f}")

        # Flow/Bookmap Integration
        flow = compute_flow_metrics(df)
        bm = bookmap_snapshot(ex, SYMBOL)
        
        if flow.get("ok"):
            dz = flow.get("delta_z", 0)
            if dz >= 0.3:
                votes_b += 1
                score_b += 0.8 * penalty_multiplier
                logs.append("📊 Flow ضغط شراء")
            if dz <= -0.3:
                votes_s += 1
                score_s += 0.8 * penalty_multiplier
                logs.append("📊 Flow ضغط بيع")
                
        if bm.get("ok"):
            imb = bm.get("imbalance", 1.0)
            if imb >= 1.1:
                logs.append(f"🧱 Bookmap imb={imb:.2f}")

        # Neutral/Chop Reduction
        if rsi_ctx["in_chop"]:
            score_b *= 0.90
            score_s *= 0.90
            logs.append("⚖️ نطاق حيادي (RSI 45–55)")

        # ADX Gate
        if adx < 12:
            score_b *= 0.95
            score_s *= 0.95
            logs.append(f"🛡️ ADX Gate {adx:.1f}<12")

        # 🔒 STRICTER WEAK SCALP PROTECTION
        mode_data = decide_strategy_mode_enhanced(df, adx=adx, di_plus=plus_di, di_minus=minus_di, rsi_ctx=rsi_ctx)
        
        if mode_data["mode"] == "scalp":
            # تحقق إذا كان مسموحاً بالسكالب من الأساس
            allow_scalp, scalp_reason = is_scalp_allowed(df, {
                "b": votes_b, "s": votes_s, 
                "score_b": score_b, "score_s": score_s,
                "logs": logs, "ind": ind
            })
            if not allow_scalp:
                # منع السكالب تماماً
                score_b *= 0.1  # ⬇️ تخفيض شديد
                score_s *= 0.1
                votes_b = 0
                votes_s = 0
                logs.append(f"🛑 ممنوع السكالب: {scalp_reason}")
            else:
                # تحقق من السكالب الضعيف
                expected_profit = TP1_PCT_SCALP
                is_weak, weak_reason = detect_weak_scalp(df, {
                    "b": votes_b, "s": votes_s, 
                    "score_b": score_b, "score_s": score_s,
                    "logs": logs, "ind": ind
                }, expected_profit)
                if is_weak:
                    score_b *= 0.2  # ⬇️ تخفيض كبير
                    score_s *= 0.2
                    votes_b = max(0, votes_b - 2)
                    votes_s = max(0, votes_s - 2)
                    logs.append(f"🛑 سكالب ضعيف: {weak_reason}")

        # Update indicators with new data
        ind.update({
            "rsi": rsi_ctx["rsi"],
            "rsi_ma": rsi_ctx["rsi_ma"], 
            "rsi_trendz": rsi_ctx["trendZ"],
            "rsi_cross": rsi_ctx["cross"],  # ⬅️ إضافة التقاطع
            "di_spread": di_spread,
            "fvg": fvg,
            "bos": bos, 
            "sweep": sweep,
            "gz": gz,
            "flow": flow,
            "bm": bm,
            "forbidden_zones": forbidden_zones,
            "volume_rsi_errors": volume_rsi_errors
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
        log_w(f"council_votes_ultimate error: {e}")
        return LAST_COUNCIL
    finally:
        COUNCIL_BUSY = False

council_votes_pro = council_votes_ultimate

# =================== FAST TRADING SYSTEM ===================
def detect_fast_opportunity(df, council_data):
    """كشف فرص التداول السريع مع شروط أقوى للسكالب"""
    if not FAST_TRADE_ENABLED:
        return None
        
    ind = council_data["ind"]
    score_b = council_data["score_b"]
    score_s = council_data["score_s"]
    current_price = float(df['close'].iloc[-1]) if len(df) > 0 else 0
    
    # للسكالب: شروط أقسى
    fast_buy = (
        score_b >= SCALP_MIN_SCORE and 
        council_data["b"] >= SCALP_MIN_VOTES and
        ind.get('rsi', 50) < 65 and
        ind.get('adx', 0) > 16 and   # ⬆️ زيادة من 10 إلى 16
        council_data["b"] > council_data["s"]
    )
    
    fast_sell = (
        score_s >= SCALP_MIN_SCORE and 
        council_data["s"] >= SCALP_MIN_VOTES and
        ind.get('rsi', 50) > 35 and
        ind.get('adx', 0) > 16 and   # ⬆️ زيادة من 10 إلى 16
        council_data["s"] > council_data["b"]
    )
    
    # فحص الجودة الإضافي
    if fast_buy or fast_sell:
        is_quality, quality_reason = enhanced_high_quality_scalp(df, council_data, current_price)
        if is_quality:
            if fast_buy:
                update_scalp_trade_timestamp()
                quality_score = float(quality_reason.split('(')[1].split('/')[0])
                log_quality_decision(True, quality_reason, council_data, quality_score)
                return {"action": "fast_buy", "reason": f"🔥 سكالب عالي الجودة - {quality_reason}"}
            elif fast_sell:
                update_scalp_trade_timestamp()
                quality_score = float(quality_reason.split('(')[1].split('/')[0])
                log_quality_decision(True, quality_reason, council_data, quality_score)
                return {"action": "fast_sell", "reason": f"🔥 سكالب عالي الجودة - {quality_reason}"}
    
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
        mode = decide_strategy_mode_enhanced(df)
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

        strat_icon = "⚡" if mode["mode"]=="scalp" else "📈" if mode["mode"]=="trend" else "🚫" if mode["mode"]=="avoid" else "ℹ️"
        strat = f"Strategy: {strat_icon} {mode['mode'].upper()} ({mode['why']})"

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
    
    votes = council_data
    print(f"🎯 EXECUTE: {side.upper()} {qty:.4f} @ {price:.6f} | "
          f"mode={mode} | votes={votes['b']}/{votes['s']} score={votes['score_b']:.1f}/{votes['score_s']:.1f}"
          f"{gz_note}", flush=True)

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
def open_market_enhanced(side, qty, price):
    if qty <= 0: 
        log_e("skip open (qty<=0)")
        return False
    
    df = fetch_ohlcv()
    snap = emit_snapshots(ex, SYMBOL, df)
    
    votes = snap["cv"]
    mode_data = snap["mode"]
    
    mode = mode_data["mode"]
    gz = snap["gz"]
    
    management_config = setup_trade_management(mode)
    
    success = execute_trade_decision(side, price, qty, mode, votes, gz)
    
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
            "management": management_config
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
            "opened_at": int(time.time()),
            "partial_taken": False,
            "breakeven_armed": False,
            "trail_active": False,
            "trail_tightened": False,
        })
        
        log_trade_open(
            side=side, price=price, qty=qty, leverage=LEVERAGE,
            source="ULTIMATE QUALITY FOCUSED SCALP SYSTEM",
            mode=mode,
            risk_alloc=RISK_ALLOC,
            council=votes,
            gz=gz,
            mgmt=management_config
        )
        
        log_g(f"✅ POSITION OPENED: {side.upper()} | mode={mode}")
        return True
    
    return False

open_market = open_market_enhanced

# =================== INDICATORS ===================
def wilder_ema(s: pd.Series, n: int): 
    return s.ewm(alpha=1/n, adjust=False).mean()

def compute_indicators(df: pd.DataFrame):
    if len(df) < max(ATR_LEN, RSI_LEN, ADX_LEN) + 2:
        return {"rsi":50.0,"plus_di":0.0,"minus_di":0.0,"dx":0.0,"adx":0.0,"atr":0.0}
    c,h,l,v = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float), df["volume"].astype(float)
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

    # حساب Volume MA20
    volume_ma_20 = v.rolling(20).mean() if len(v) >= 20 else pd.Series([v.mean()]*len(v))

    i=len(df)-1
    return {
        "rsi": float(rsi.iloc[i]), 
        "plus_di": float(plus_di.iloc[i]),
        "minus_di": float(minus_di.iloc[i]), 
        "dx": float(dx.iloc[i]),
        "adx": float(adx.iloc[i]), 
        "atr": float(atr.iloc[i]),
        "volume_ma_20": float(volume_ma_20.iloc[i]) if len(volume_ma_20) > i else float(v.iloc[i])
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
        "trail_tightened": False, "partial_taken": False
    })
    save_state({"in_position": False, "position_qty": 0})
    
    # NO WAITING - جاهز فورًا
    _arm_wait_after_close(prev_side)
    logging.info(f"AFTER_CLOSE ready for next trade immediately")

# =================== ENHANCED TRADE MANAGEMENT ===================
def manage_after_entry_enhanced_pro(df, ind, info):
    """إدارة محسنة للمركز مع خروج ذكي حسب النمط + تحديث نظام التبريد"""
    if not STATE["open"] or STATE["qty"] <= 0:
        return

    px = info["price"]
    entry = STATE["entry"]
    side = STATE["side"]
    qty = STATE["qty"]
    mode = STATE.get("mode", "trend")
    management = STATE.get("management", {})
    
    # تحديث عدد البارات
    STATE["bars"] = STATE.get("bars", 0) + 1
    
    pnl_pct = (px - entry) / entry * 100 * (1 if side == "long" else -1)
    STATE["pnl"] = pnl_pct
    
    if pnl_pct > STATE["highest_profit_pct"]:
        STATE["highest_profit_pct"] = pnl_pct

    snap = emit_snapshots(ex, SYMBOL, df)
    gz = snap["gz"]
    
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
        
        # تحديث نظام التبريد إذا كانت صفقة ترند قوية
        if mode == "trend" and (pnl_pct/100 >= 0.008 or STATE["bars"] >= 8):
            update_strong_trend_timestamp(pnl_pct/100, STATE["bars"])
        
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

manage_after_entry = manage_after_entry_enhanced_pro

# =================== SMART TRADE LOOP ===================
def trade_loop_smart_system():
    """نظام تداول ذكي يتعامل مع جميع حالات السوق"""
    global wait_for_next_signal_side, last_strong_trend_time
    
    while True:
        try:
            # جمع البيانات الأساسية
            bal = balance_usdt()
            px = price_now()
            df = fetch_ohlcv()
            info = rf_signal_live(df)
            ind = compute_indicators(df)
            spread_bps = orderbook_spread_bps()
            
            if not px:
                time.sleep(BASE_SLEEP)
                continue
                
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
            
            # 🔍 تشخيص مفصل
            council_data = council_votes_pro(df)
            strategy_mode = snap["mode"]
            
            print(f"🔍 التشخيص | B: {council_data['b']}/{council_data['score_b']:.1f} | "
                  f"S: {council_data['s']}/{council_data['score_s']:.1f} | "
                  f"الاستراتيجية: {strategy_mode['mode']} ({strategy_mode['why']})")
            
            # 🛡️ فحص أنظمة الحماية الأساسية
            protection_checks = []
            
            # 1. فحص وضع avoid
            if strategy_mode["mode"] == "avoid":
                protection_checks.append(("🛑", f"تجنب التداول: {strategy_mode['why']}"))
            
            # 2. فحص الانتشار
            if spread_bps is not None and spread_bps > MAX_SPREAD_BPS:
                protection_checks.append(("🛑", f"انتشار عالي: {fmt(spread_bps,2)}bps"))
            
            # 3. فحص التذبذب (النظام الجديد)
            chop_signals = detect_market_chop(df, council_data, px)
            avoid_chop, chop_details = should_avoid_chop_market(chop_signals)
            if avoid_chop:
                protection_checks.append(("🔄", f"سوق متذبذب: {chop_details[0]}"))
            
            # 4. فحص المناطق المحظورة
            forbidden_zones = detect_forbidden_zones(df, council_data, px)
            if forbidden_zones:
                protection_checks.append(("🚫", f"منطقة محظورة: {forbidden_zones[0]}"))
            
            # 5. فحص الفوليوم وتقاطع RSI
            volume_rsi_errors = validate_volume_and_rsi(df, council_data, px)
            if volume_rsi_errors:
                protection_checks.append(("📉", f"تحقق فني: {volume_rsi_errors[0]}"))
            
            # إذا فيه أي حماية نشطة، منع التداول
            if protection_checks and not STATE["open"]:
                for icon, reason in protection_checks:
                    print(f"{icon} {reason}", flush=True)
                if avoid_chop and len(chop_details) > 1:
                    for i, signal in enumerate(chop_details[1:3], 1):
                        print(f"   ↳ {signal}", flush=True)
                print("🔒 منع التداول بسبب أنظمة الحماية", flush=True)
                time.sleep(BASE_SLEEP)
                continue
            
            # 🎯 قرار الدخول الذكي
            sig = None
            reason = None

            if not STATE["open"]:
                # استخدام النظام الذكي الجديد
                sig, reason = enhanced_smart_entry_decision(df, council_data, strategy_mode, snap, px)
                
                # فحص إضافي للسكالب الضعيف
                if sig and strategy_mode["mode"] == "scalp":
                    expected_profit = TP1_PCT_SCALP
                    is_weak, weak_reason = detect_weak_scalp(df, council_data, expected_profit)
                    if is_weak:
                        log_w(f"⚠️ رفض سكالب ضعيف: {weak_reason}")
                        sig = None
                        reason = weak_reason

            # تنفيذ الصفقة
            if sig and not protection_checks:
                qty = compute_size(bal, px)
                if qty > 0:
                    ok = open_market(sig, qty, px)
                    if ok:
                        log_g(f"✅ {reason}")
                else:
                    reason = "qty<=0"
            
            # 🔍 لوج التشخيص
            if not STATE["open"] and not sig:
                if chop_signals:
                    print(f"🔄 سوق متذبذب | إشارات: {len(chop_signals)} | {chop_signals[0]}", flush=True)
                elif forbidden_zones:
                    print(f"🚫 مناطق محظورة | {forbidden_zones[0]}", flush=True)
                elif volume_rsi_errors:
                    print(f"📉 مشاكل فنية | {volume_rsi_errors[0]}", flush=True)
                else:
                    print(f"🔍 لا توجد صفقة | السبب: {reason or 'شروط غير متحققة'}", flush=True)

            time.sleep(BASE_SLEEP)
            
        except Exception as e:
            log_e(f"loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

trade_loop = trade_loop_smart_system

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
        print(f"   🎯 ENTRY: ULTIMATE QUALITY FOCUSED SCALP SYSTEM  |  spread_bps={fmt(spread_bps,2)}")
        print(f"   ⏱️ closes_in ≈ {left_s}s")
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
    return f"✅ ULTIMATE QUALITY FOCUSED SCALP BOT — {SYMBOL} {INTERVAL} — {mode} — Ultimate High Quality Trades Only"

@app.route("/metrics")
def metrics():
    in_cooldown, cooldown_reason = is_in_trend_cooldown()
    in_scalp_cooldown, scalp_cooldown_reason = is_in_scalp_cooldown()
    
    return jsonify({
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "state": STATE, "compound_pnl": compound_pnl,
        "entry_mode": "ULTIMATE_QUALITY_FOCUSED_SCALP",
        "protection_system": {
            "quality_scalp_protection": True,
            "scalp_cooldown": {
                "active": in_scalp_cooldown,
                "reason": scalp_cooldown_reason,
            },
            "trend_cooldown": {
                "active": in_cooldown,
                "reason": cooldown_reason,
            },
            "market_chop_detection": True,
            "strategy_avoid": True,
            "forbidden_zones_detection": True,
            "volume_rsi_validation": True,
        }
    })

@app.route("/health")
def health():
    in_cooldown, cooldown_reason = is_in_trend_cooldown()
    in_scalp_cooldown, scalp_cooldown_reason = is_in_scalp_cooldown()
    
    return jsonify({
        "ok": True, "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"], "side": STATE["side"], "qty": STATE["qty"],
        "compound_pnl": compound_pnl, "timestamp": datetime.utcnow().isoformat(),
        "protection_active": {
            "quality_scalp": True,
            "scalp_cooldown": in_scalp_cooldown,
            "trend_cooldown": in_cooldown,
            "market_chop_detection": True,
            "strategy_avoid": True,
            "forbidden_zones": True,
            "volume_rsi": True,
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
    log_banner("ULTIMATE QUALITY FOCUSED SCALP SYSTEM")
    state = load_state() or {}
    state.setdefault("in_position", False)

    if RESUME_ON_RESTART:
        try:
            state = resume_open_position_enhanced(ex, SYMBOL, state)
        except Exception as e:
            log_w(f"resume error: {e}\n{traceback.format_exc()}")

    verify_execution_environment()

    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'}  •  {SYMBOL}  •  {INTERVAL}", "yellow"))
    print(colored(f"RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x  •  ULTIMATE_QUALITY_FOCUSED_SYSTEM=ENABLED", "yellow"))
    print(colored(f"SMC/ICT: Golden Zones + FVG + BOS + Sweeps + Order Blocks", "yellow"))
    print(colored(f"MANAGEMENT: Smart TP + Smart Exit + Trail Adaptation", "yellow"))
    print(colored(f"🛡️  ULTIMATE QUALITY FOCUSED SCALP: ACTIVATED (Min {SCALP_QUALITY_THRESHOLD}/10)", "green"))
    print(colored(f"🔄 TREND COOLDOWN SYSTEM: {TREND_COOLDOWN_HOURS} hours", "green")) 
    print(colored(f"🎯 STRATEGY AVOID MODE: ACTIVATED", "green"))
    print(colored(f"🔄 SMART CHOP DETECTION: ACTIVATED", "green"))
    print(colored(f"🚫 FORBIDDEN ZONES DETECTION: ACTIVATED", "green"))
    print(colored(f"📊 VOLUME & RSI CROSS VALIDATION: ACTIVATED", "green"))
    print(colored(f"🎯 INTELLIGENT ENTRY SYSTEM: ACTIVATED", "green"))
    print(colored(f"EXECUTION: {'ACTIVE' if EXECUTE_ORDERS and not DRY_RUN else 'SIMULATION'}", "yellow"))
    
    logging.info("ULTIMATE QUALITY FOCUSED SCALP service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    
    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)

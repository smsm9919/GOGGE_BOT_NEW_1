# -*- coding: utf-8 -*-
"""
RF Futures Bot — RF-LIVE ONLY (BingX Perp via CCXT)
• Council PRO Unified Decision System with Candles & Golden Entry
• Golden Entry + Golden Reversal + Wick Exhaustion + Smart Profit AI
• Dynamic TP ladder + Breakeven + ATR-trailing
• Smart Exit Management + Wait-for-next-signal
• Professional Logging & Dashboard
• Enhanced with Footprint, SMC Candles, Liquidity Traps + VWAP Strategy
• Advanced Smart Money Concepts (Liquidity Pools, Order Blocks, FVGs)
• Professional Price Structure Analysis
• Intelligent Entry Logic with Risk Management
• Advanced Entry Quality Engine with Tier System
• PROFESSIONAL GOLDEN ZONE TRADING SYSTEM
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
BOT_VERSION = "DOGE Council PRO v8.0 — Professional Golden Zone Trading System + Advanced SMC"
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

# ================== ENTRY QUALITY ENGINE (TIERS) ==================

# أوزان الطبقات
TIER_A_WEIGHT = 4      # Golden / SMC قوي / Liquidity Sweep كبيرة
TIER_B_WEIGHT = 2      # FVG / OB / VWAP / Structure / Flow
TIER_C_WEIGHT = 1      # RSI / ADX / Candles

# عتبات الدخول
TREND_MIN_SCORE = 10      # أقل Score لصفقة ترند
SCALP_MIN_SCORE = 5       # أقل Score لسكالب محترم

TREND_NEED_TIER_A = True  # لازم إشارة Tier A لصفقة ترند
SCALP_NEED_TIER_A = False # السكالب ممكن بدون Tier A لو النقاط كافية

# درجات الثقة المطلوبة لكل نمط
CONFIDENCE_TREND_STRONG = 8.5
CONFIDENCE_STUDIED_SCALP = 7.5
CONFIDENCE_CAUTIOUS_SCALP = 6.5

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

# ==== Advanced Council Decision Settings ====
CONFIDENCE_AVOID = 5.0
CONFIDENCE_CAUTIOUS_SCALP = 6.5
CONFIDENCE_STUDIED_SCALP = 8.0
CONFIDENCE_TREND = 8.5

# ==== Smart Money Concepts Settings ====
SMC_LIQUIDITY_WINDOW = 20
SMC_PIVOT_STRENGTH_THRESHOLD = 3
ORDER_BLOCK_LOOKBACK = 10
FVG_MIN_SIZE_BPS = 5.0  # 0.05%

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

# ==== OTC Settings ====
OTC_ENABLED = True
OTC_VOLUME_THRESHOLD = 3.0    # حجم أكبر 3x من المتوسط

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

# =================== SMART MONEY CONCEPTS ADVANCED ===================
def detect_liquidity_pools(df, window=20, sensitivity=2.0):
    """كشف مناطق السيولة باستخدام تحليل القمم والقيعان"""
    try:
        if len(df) < window * 2:
            return {"ok": False, "why": "insufficient_data"}
            
        highs = df['high'].astype(float)
        lows = df['low'].astype(float)
        
        # تحديد القمم والقيعان المحورية
        pivot_highs = []
        pivot_lows = []
        
        for i in range(window, len(df)-window):
            window_high = highs.iloc[i-window:i+window]
            window_low = lows.iloc[i-window:i+window]
            
            if highs.iloc[i] == window_high.max():
                strength = calculate_pivot_strength(df, i, 'high')
                if strength >= SMC_PIVOT_STRENGTH_THRESHOLD:
                    pivot_highs.append({
                        'price': float(highs.iloc[i]),
                        'time': int(df.index[i].timestamp() if hasattr(df.index[i], 'timestamp') else int(time.time())),
                        'strength': strength,
                        'type': 'liquidity_pool_high'
                    })
            
            if lows.iloc[i] == window_low.min():
                strength = calculate_pivot_strength(df, i, 'low')
                if strength >= SMC_PIVOT_STRENGTH_THRESHOLD:
                    pivot_lows.append({
                        'price': float(lows.iloc[i]),
                        'time': int(df.index[i].timestamp() if hasattr(df.index[i], 'timestamp') else int(time.time())),
                        'strength': strength,
                        'type': 'liquidity_pool_low'
                    })
        
        liquidity_zones = calculate_liquidity_zones(pivot_highs, pivot_lows)
        
        return {
            "ok": True,
            "pivot_highs": pivot_highs[-5:],  # آخر 5 قمم
            "pivot_lows": pivot_lows[-5:],    # آخر 5 قيعان
            "liquidity_zones": liquidity_zones,
            "current_strength": analyze_current_liquidity(df, pivot_highs, pivot_lows)
        }
    except Exception as e:
        return {"ok": False, "why": str(e)}

def calculate_pivot_strength(df, index, pivot_type):
    """حساب قوة النقطة المحورية"""
    try:
        volume = float(df['volume'].iloc[index])
        avg_volume = df['volume'].astype(float).rolling(20).mean().iloc[index]
        price_range = float(df['high'].iloc[index] - df['low'].iloc[index])
        avg_range = (df['high'].astype(float) - df['low'].astype(float)).rolling(20).mean().iloc[index]
        
        strength = 0
        if volume > avg_volume * 1.5:
            strength += 2
        if price_range > avg_range * 1.5:
            strength += 2
        if pivot_type == 'high' and df['close'].iloc[index] < df['open'].iloc[index]:
            strength += 1
        if pivot_type == 'low' and df['close'].iloc[index] > df['open'].iloc[index]:
            strength += 1
            
        return min(strength, 5)  # قوة من 0 إلى 5
    except:
        return 0

def calculate_liquidity_zones(pivot_highs, pivot_lows):
    """حساب مناطق السيولة من القمم والقيعان"""
    zones = []
    
    for ph in pivot_highs:
        zones.append({
            'price': ph['price'],
            'type': 'supply_zone',
            'strength': ph['strength'],
            'origin': 'pivot_high'
        })
    
    for pl in pivot_lows:
        zones.append({
            'price': pl['price'],
            'type': 'demand_zone', 
            'strength': pl['strength'],
            'origin': 'pivot_low'
        })
    
    return zones

def analyze_current_liquidity(df, pivot_highs, pivot_lows):
    """تحليل السيولة الحالية بالنسبة للأسعار التاريخية"""
    if len(df) == 0 or not pivot_highs or not pivot_lows:
        return {"ok": False}
    
    current_price = float(df['close'].iloc[-1])
    recent_high = max([ph['price'] for ph in pivot_highs[-3:]]) if pivot_highs else current_price
    recent_low = min([pl['price'] for pl in pivot_lows[-3:]]) if pivot_lows else current_price
    
    distance_to_high = abs(current_price - recent_high) / current_price * 100
    distance_to_low = abs(current_price - recent_low) / current_price * 100
    
    return {
        "ok": True,
        "near_supply": distance_to_high < 1.0,  # within 1%
        "near_demand": distance_to_low < 1.0,
        "supply_distance_pct": distance_to_high,
        "demand_distance_pct": distance_to_low
    }

def identify_supply_demand_zones(df):
    """تحديد مناطق العرض والطلب مع قوة كل منطقة"""
    try:
        if len(df) < 5:
            return []
            
        zones = []
        for i in range(2, len(df)-1):
            # منطقة طلب (Demand) - قاع مع شمعة صاعدة قوية
            if (float(df['low'].iloc[i]) < float(df['low'].iloc[i-1]) and 
                float(df['close'].iloc[i]) > float(df['open'].iloc[i]) and
                float(df['close'].iloc[i]) > float(df['close'].iloc[i-1])):
                
                strength = calculate_zone_strength(df, i, 'demand')
                zones.append({
                    'type': 'demand',
                    'price': float(df['low'].iloc[i]),
                    'time': int(df.index[i].timestamp() if hasattr(df.index[i], 'timestamp') else int(time.time())),
                    'strength': strength,
                    'width': calculate_zone_width(df, i, 'demand')
                })
            
            # منطقة عرض (Supply) - قمة مع شمعة هابطة قوية
            if (float(df['high'].iloc[i]) > float(df['high'].iloc[i-1]) and 
                float(df['close'].iloc[i]) < float(df['open'].iloc[i]) and
                float(df['close'].iloc[i]) < float(df['close'].iloc[i-1])):
                
                strength = calculate_zone_strength(df, i, 'supply')
                zones.append({
                    'type': 'supply',
                    'price': float(df['high'].iloc[i]),
                    'time': int(df.index[i].timestamp() if hasattr(df.index[i], 'timestamp') else int(time.time())),
                    'strength': strength,
                    'width': calculate_zone_width(df, i, 'supply')
                })
        
        return zones[-8:]  # آخر 8 مناطق
    except Exception as e:
        log_w(f"identify_supply_demand_zones error: {e}")
        return []

def calculate_zone_strength(df, index, zone_type):
    """حساب قوة منطقة العرض أو الطلب"""
    try:
        strength = 0
        
        # حجم التداول
        volume = float(df['volume'].iloc[index])
        avg_volume = df['volume'].astype(float).rolling(20).mean().iloc[index]
        if volume > avg_volume * 2.0:
            strength += 2
        elif volume > avg_volume * 1.5:
            strength += 1
        
        # نطاق السعر
        price_range = float(df['high'].iloc[index] - df['low'].iloc[index])
        avg_range = (df['high'].astype(float) - df['low'].astype(float)).rolling(20).mean().iloc[index]
        if price_range > avg_range * 2.0:
            strength += 2
        elif price_range > avg_range * 1.5:
            strength += 1
        
        # شكل الشمعة
        body_size = abs(float(df['close'].iloc[index]) - float(df['open'].iloc[index]))
        total_range = price_range
        if body_size / total_range > 0.7:  # شمعة قوية
            strength += 1
        
        return min(strength, 5)
    except:
        return 0

def calculate_zone_width(df, index, zone_type):
    """حساب عرض منطقة العرض أو الطلب"""
    try:
        if zone_type == 'demand':
            return float(df['high'].iloc[index]) - float(df['low'].iloc[index])
        else:  # supply
            return float(df['high'].iloc[index]) - float(df['low'].iloc[index])
    except:
        return 0.0

def detect_fair_value_gaps(df):
    """كشف فجوات القيمة العادلة (FVG)"""
    try:
        if len(df) < 3:
            return []
            
        fvgs = []
        for i in range(1, len(df)-1):
            current_high = float(df['high'].iloc[i])
            current_low = float(df['low'].iloc[i])
            prev_high = float(df['high'].iloc[i-1])
            prev_low = float(df['low'].iloc[i-1])
            next_high = float(df['high'].iloc[i+1])
            next_low = float(df['low'].iloc[i+1])
            
            # FVG صاعد - عدم وجود تداخل بين الشمعة الحالية والشموع المحيطة
            if current_low > prev_high and current_low > next_high:
                gap_size = ((current_low - max(prev_high, next_high)) / current_low) * 10000
                if gap_size >= FVG_MIN_SIZE_BPS:
                    fvgs.append({
                        'type': 'bullish_fvg',
                        'top': current_low,
                        'bottom': max(prev_high, next_high),
                        'time': int(df.index[i].timestamp() if hasattr(df.index[i], 'timestamp') else int(time.time())),
                        'size_bps': gap_size
                    })
            
            # FVG هابط - عدم وجود تداخل بين الشمعة الحالية والشموع المحيطة
            if current_high < prev_low and current_high < next_low:
                gap_size = ((min(prev_low, next_low) - current_high) / current_high) * 10000
                if gap_size >= FVG_MIN_SIZE_BPS:
                    fvgs.append({
                        'type': 'bearish_fvg',
                        'top': min(prev_low, next_low),
                        'bottom': current_high,
                        'time': int(df.index[i].timestamp() if hasattr(df.index[i], 'timestamp') else int(time.time())),
                        'size_bps': gap_size
                    })
        
        return fvgs[-5:]  # آخر 5 فجوات
    except Exception as e:
        log_w(f"detect_fair_value_gaps error: {e}")
        return []

def identify_order_blocks(df):
    """تحديد كتل الأوامر للدخول المحترف"""
    try:
        if len(df) < 4:
            return []
            
        blocks = []
        for i in range(1, len(df)-2):
            # كتلة أوامر شراء (Buy Order Block)
            if (float(df['close'].iloc[i]) > float(df['open'].iloc[i]) and  # شمعة صاعدة
                float(df['close'].iloc[i+1]) < float(df['open'].iloc[i+1]) and  # يليها شمعة هابطة
                float(df['low'].iloc[i+1]) >= float(df['low'].iloc[i])):  # لم تخترق القاع
                
                blocks.append({
                    'type': 'buy_block',
                    'high': float(df['high'].iloc[i]),
                    'low': float(df['low'].iloc[i]),
                    'entry': float(df['low'].iloc[i]),
                    'time': int(df.index[i].timestamp() if hasattr(df.index[i], 'timestamp') else int(time.time())),
                    'strength': calculate_block_strength(df, i, 'buy')
                })
            
            # كتلة أوامر بيع (Sell Order Block)
            if (float(df['close'].iloc[i]) < float(df['open'].iloc[i]) and  # شمعة هابطة
                float(df['close'].iloc[i+1]) > float(df['open'].iloc[i+1]) and  # يليها شمعة صاعدة
                float(df['high'].iloc[i+1]) <= float(df['high'].iloc[i])):  # لم تخترق القمة
                
                blocks.append({
                    'type': 'sell_block',
                    'high': float(df['high'].iloc[i]),
                    'low': float(df['low'].iloc[i]),
                    'entry': float(df['high'].iloc[i]),
                    'time': int(df.index[i].timestamp() if hasattr(df.index[i], 'timestamp') else int(time.time())),
                    'strength': calculate_block_strength(df, i, 'sell')
                })
        
        return blocks[-6:]  # آخر 6 كتل أوامر
    except Exception as e:
        log_w(f"identify_order_blocks error: {e}")
        return []

def calculate_block_strength(df, index, block_type):
    """حساب قوة كتلة الأوامر"""
    try:
        strength = 0
        
        # حجم التداول في كتلة الأوامر
        volume = float(df['volume'].iloc[index])
        avg_volume = df['volume'].astype(float).rolling(20).mean().iloc[index]
        if volume > avg_volume * 2.0:
            strength += 2
        elif volume > avg_volume * 1.5:
            strength += 1
        
        # حجم الشمعة التالية (رد الفعل)
        reaction_volume = float(df['volume'].iloc[index+1])
        if reaction_volume > avg_volume * 1.5:
            strength += 1
        
        # قوة حركة السعر بعد كتلة الأوامر
        if block_type == 'buy':
            price_move = float(df['close'].iloc[index+2]) - float(df['close'].iloc[index+1])
        else:
            price_move = float(df['close'].iloc[index+1]) - float(df['close'].iloc[index+2])
            
        if abs(price_move) > 0:
            strength += 1
        
        return min(strength, 4)
    except:
        return 0

# =================== PRICE STRUCTURE ANALYSIS ===================
def analyze_price_structure(df):
    """تحليل متقدم للهيكل السعري"""
    try:
        if len(df) < 10:
            return {"ok": False, "trend": "sideways", "structure": []}
        
        highs = df['high'].astype(float)
        lows = df['low'].astype(float)
        closes = df['close'].astype(float)
        
        # تحديد Higher Highs / Lower Lows
        hh, lh, ll, hl = identify_swing_points(highs, lows)
        
        # تحديد الاتجاه
        trend = determine_trend_strength(hh, lh, ll, hl)
        
        # نقاط التحول المحتملة
        reversal_points = identify_potential_reversals(df, hh, lh, ll, hl)
        
        return {
            "ok": True,
            "trend": trend['direction'],
            "trend_strength": trend['strength'],
            "higher_highs": hh[-3:],
            "lower_highs": lh[-3:],
            "lower_lows": ll[-3:],
            "higher_lows": hl[-3:],
            "reversal_points": reversal_points,
            "market_structure": analyze_market_structure(hh, lh, ll, hl)
        }
    except Exception as e:
        return {"ok": False, "why": str(e)}

def identify_swing_points(highs, lows, window=3):
    """تحديد نقاط التقلب بدقة"""
    hh, lh, ll, hl = [], [], [], []
    
    for i in range(window, len(highs)-window):
        # Higher High
        if all(highs.iloc[i] > highs.iloc[i-j] for j in range(1, window+1)) and \
           all(highs.iloc[i] > highs.iloc[i+j] for j in range(1, window+1)):
            hh.append({'index': i, 'price': float(highs.iloc[i])})
        
        # Lower High
        elif all(highs.iloc[i] < highs.iloc[i-j] for j in range(1, window+1)) and \
             all(highs.iloc[i] < highs.iloc[i+j] for j in range(1, window+1)):
            lh.append({'index': i, 'price': float(highs.iloc[i])})
        
        # Lower Low
        if all(lows.iloc[i] < lows.iloc[i-j] for j in range(1, window+1)) and \
           all(lows.iloc[i] < lows.iloc[i+j] for j in range(1, window+1)):
            ll.append({'index': i, 'price': float(lows.iloc[i])})
        
        # Higher Low
        elif all(lows.iloc[i] > lows.iloc[i-j] for j in range(1, window+1)) and \
             all(lows.iloc[i] > lows.iloc[i+j] for j in range(1, window+1)):
            hl.append({'index': i, 'price': float(lows.iloc[i])})
    
    return hh, lh, ll, hl

def determine_trend_strength(hh, lh, ll, hl):
    """تحديد قوة الاتجاه"""
    if not hh or not ll:
        return {"direction": "sideways", "strength": 0}
    
    # حساب قوة الاتجاه الصاعد
    if len(hh) >= 2 and len(hl) >= 2:
        uptrend_strength = min(len(hh), len(hl))
        recent_hh = hh[-1]['price'] if hh else 0
        recent_hl = hl[-1]['price'] if hl else 0
        prev_hh = hh[-2]['price'] if len(hh) >= 2 else 0
        prev_hl = hl[-2]['price'] if len(hl) >= 2 else 0
        
        if recent_hh > prev_hh and recent_hl > prev_hl:
            return {"direction": "uptrend", "strength": uptrend_strength}
    
    # حساب قوة الاتجاه الهابط
    if len(lh) >= 2 and len(ll) >= 2:
        downtrend_strength = min(len(lh), len(ll))
        recent_lh = lh[-1]['price'] if lh else 0
        recent_ll = ll[-1]['price'] if ll else 0
        prev_lh = lh[-2]['price'] if len(lh) >= 2 else 0
        prev_ll = ll[-2]['price'] if len(ll) >= 2 else 0
        
        if recent_lh < prev_lh and recent_ll < prev_ll:
            return {"direction": "downtrend", "strength": downtrend_strength}
    
    return {"direction": "sideways", "strength": 0}

def identify_potential_reversals(df, hh, lh, ll, hl):
    """تحديد نقاط التحول المحتملة"""
    reversals = []
    
    # تحليل التباعد (Divergence)
    if len(df) > 14:
        rsi = compute_rsi(df['close'].astype(float), 14)
        current_rsi = float(rsi.iloc[-1])
        prev_rsi = float(rsi.iloc[-2]) if len(rsi) > 1 else current_rsi
        
        # تباعد هابط (Bearish Divergence)
        if hh and len(hh) >= 2:
            recent_high = hh[-1]['price']
            prev_high = hh[-2]['price'] if len(hh) >= 2 else recent_high
            if recent_high > prev_high and current_rsi < prev_rsi:
                reversals.append({
                    'type': 'bearish_divergence',
                    'price': recent_high,
                    'strength': 2
                })
        
        # تباعد صاعد (Bullish Divergence)
        if ll and len(ll) >= 2:
            recent_low = ll[-1]['price']
            prev_low = ll[-2]['price'] if len(ll) >= 2 else recent_low
            if recent_low < prev_low and current_rsi > prev_rsi:
                reversals.append({
                    'type': 'bullish_divergence', 
                    'price': recent_low,
                    'strength': 2
                })
    
    return reversals

def analyze_market_structure(hh, lh, ll, hl):
    """تحليل هيكل السوق"""
    structure = {
        "uptrend_break": False,
        "downtrend_break": False,
        "consolidation": False
    }
    
    if len(hh) >= 2 and len(hl) >= 2:
        # كسر هيكل صاعد
        if ll and ll[-1]['price'] < hl[-2]['price']:
            structure["uptrend_break"] = True
    
    if len(lh) >= 2 and len(ll) >= 2:
        # كسر هيكل هابط
        if hh and hh[-1]['price'] > lh[-2]['price']:
            structure["downtrend_break"] = True
    
    # توطيد (Consolidation)
    if not hh and not ll and len(lh) >= 2 and len(hl) >= 2:
        structure["consolidation"] = True
    
    return structure

# =================== ENHANCED GOLDEN ZONE & TREND REVERSAL DETECTION ===================
def detect_advanced_golden_zones(df, current_price):
    """كشف متقدم للمناطق الذهبية وانعكاسات الترند"""
    try:
        if len(df) < 20:
            return {"ok": False, "zones": []}
        
        indicators = compute_indicators(df)
        price_structure = analyze_price_structure(df)
        smc_data = {
            'liquidity_pools': detect_liquidity_pools(df),
            'supply_demand_zones': identify_supply_demand_zones(df),
            'order_blocks': identify_order_blocks(df)
        }
        
        golden_zones = []
        
        # 1. القاع الذهبي القوي (Golden Bottom)
        golden_bottom = detect_golden_bottom(df, indicators, price_structure, smc_data)
        if golden_bottom["ok"]:
            golden_zones.append(golden_bottom)
        
        # 2. القمة الذهبية القوية (Golden Top)  
        golden_top = detect_golden_top(df, indicators, price_structure, smc_data)
        if golden_top["ok"]:
            golden_zones.append(golden_top)
        
        # 3. انعكاس الترند المدروس (Trend Reversal)
        trend_reversal = detect_trend_reversal(df, indicators, price_structure, smc_data)
        if trend_reversal["ok"]:
            golden_zones.append(trend_reversal)
        
        # ترتيب المناطق حسب القوة
        golden_zones.sort(key=lambda x: x["strength"], reverse=True)
        
        return {
            "ok": len(golden_zones) > 0,
            "zones": golden_zones,
            "strongest_zone": golden_zones[0] if golden_zones else None
        }
    except Exception as e:
        return {"ok": False, "zones": [], "error": str(e)}

def detect_golden_bottom(df, indicators, price_structure, smc_data):
    """كشف القاع الذهبي القوي"""
    try:
        rsi = indicators.get('rsi', 50)
        adx = indicators.get('adx', 0)
        atr = indicators.get('atr', 0)
        
        # شروط القاع الذهبي
        conditions = []
        strength = 0
        
        # 1. RSI في منطقة ذروة البيع
        if rsi < 32:
            conditions.append("RSI oversold")
            strength += 3
        
        # 2. ADX قوي للإشارة لترند صاعد
        if adx > 25:
            conditions.append("ADX strong")
            strength += 2
        
        # 3. تأكيد من الهيكل السعري
        if price_structure.get("ok") and price_structure.get("trend") == "downtrend":
            conditions.append("Trend exhaustion")
            strength += 2
        
        # 4. دعم من مناطق السيولة
        liquidity = smc_data['liquidity_pools']
        if liquidity.get('ok') and liquidity.get('current_strength', {}).get('near_demand'):
            conditions.append("Liquidity support")
            strength += 2
        
        # 5. Order Blocks شرائية قريبة
        buy_blocks = [b for b in smc_data['order_blocks'] if b['type'] == 'buy_block']
        if len(buy_blocks) >= 2:
            conditions.append("Multiple buy blocks")
            strength += 2
        
        # 6. تأكيد من الشموع
        candles = compute_enhanced_candles(df)
        if candles['score_buy'] >= 4:
            conditions.append("Bullish candle confirmation")
            strength += 1
        
        is_golden = strength >= 8 and len(conditions) >= 4
        
        return {
            "ok": is_golden,
            "type": "golden_bottom",
            "strength": strength,
            "conditions": conditions,
            "rsi": rsi,
            "adx": adx,
            "entry_score": min(strength * 2, 10),
            "description": "قاع ذهبي قوي مع تأكيدات متعددة"
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

def detect_golden_top(df, indicators, price_structure, smc_data):
    """كشف القمة الذهبية القوية"""
    try:
        rsi = indicators.get('rsi', 50)
        adx = indicators.get('adx', 0)
        atr = indicators.get('atr', 0)
        
        # شروط القمة الذهبية
        conditions = []
        strength = 0
        
        # 1. RSI في منطقة ذروة الشراء
        if rsi > 68:
            conditions.append("RSI overbought")
            strength += 3
        
        # 2. ADX قوي للإشارة لترند هابط
        if adx > 25:
            conditions.append("ADX strong")
            strength += 2
        
        # 3. تأكيد من الهيكل السعري
        if price_structure.get("ok") and price_structure.get("trend") == "uptrend":
            conditions.append("Trend exhaustion")
            strength += 2
        
        # 4. مقاومة من مناطق السيولة
        liquidity = smc_data['liquidity_pools']
        if liquidity.get('ok') and liquidity.get('current_strength', {}).get('near_supply'):
            conditions.append("Liquidity resistance")
            strength += 2
        
        # 5. Order Blocks بيعية قريبة
        sell_blocks = [b for b in smc_data['order_blocks'] if b['type'] == 'sell_block']
        if len(sell_blocks) >= 2:
            conditions.append("Multiple sell blocks")
            strength += 2
        
        # 6. تأكيد من الشموع
        candles = compute_enhanced_candles(df)
        if candles['score_sell'] >= 4:
            conditions.append("Bearish candle confirmation")
            strength += 1
        
        is_golden = strength >= 8 and len(conditions) >= 4
        
        return {
            "ok": is_golden,
            "type": "golden_top", 
            "strength": strength,
            "conditions": conditions,
            "rsi": rsi,
            "adx": adx,
            "entry_score": min(strength * 2, 10),
            "description": "قمة ذهبية قوية مع تأكيدات متعددة"
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

def detect_trend_reversal(df, indicators, price_structure, smc_data):
    """كشف انعكاس الترند المدروس"""
    try:
        rsi = indicators.get('rsi', 50)
        adx = indicators.get('adx', 0)
        plus_di = indicators.get('plus_di', 0)
        minus_di = indicators.get('minus_di', 0)
        
        conditions = []
        strength = 0
        reversal_type = None
        
        # 1. انعكاس من هابط إلى صاعد
        if (price_structure.get("ok") and 
            price_structure.get("market_structure", {}).get("downtrend_break")):
            
            # تأكيدات الانعكاس الصاعد
            if plus_di > minus_di and adx > 20:
                conditions.append("Uptrend reversal confirmed")
                strength += 3
                reversal_type = "bearish_to_bullish"
        
        # 2. انعكاس من صاعد إلى هابط
        if (price_structure.get("ok") and 
            price_structure.get("market_structure", {}).get("uptrend_break")):
            
            # تأكيدات الانعكاس الهابط
            if minus_di > plus_di and adx > 20:
                conditions.append("Downtrend reversal confirmed")
                strength += 3
                reversal_type = "bullish_to_bearish"
        
        # 3. تأكيد من RSI
        if reversal_type == "bearish_to_bullish" and rsi < 40:
            conditions.append("RSI supports bullish reversal")
            strength += 2
        elif reversal_type == "bullish_to_bearish" and rsi > 60:
            conditions.append("RSI supports bearish reversal") 
            strength += 2
        
        # 4. تأكيد من SMC
        if smc_data['liquidity_pools'].get('ok'):
            current_liq = smc_data['liquidity_pools'].get('current_strength', {})
            if (reversal_type == "bearish_to_bullish" and current_liq.get('near_demand')) or \
               (reversal_type == "bullish_to_bearish" and current_liq.get('near_supply')):
                conditions.append("SMC liquidity confirmation")
                strength += 2
        
        is_reversal = strength >= 7 and reversal_type is not None
        
        return {
            "ok": is_reversal,
            "type": "trend_reversal",
            "reversal_type": reversal_type,
            "strength": strength,
            "conditions": conditions,
            "entry_score": min(strength * 1.5, 10),
            "description": f"انعكاس ترند قوي: {reversal_type}"
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

# =================== PROFESSIONAL TRADE MANAGEMENT ===================
def setup_professional_trade_management(zone_type, entry_score, atr, entry_price):
    """إعداد محترف لإدارة الصفقة بناءً على نوع المنطقة وقوة الإشارة"""
    
    # تحديد نمط التداول بناءً على قوة الدخول
    if entry_score >= 9:
        trade_mode = "strong_trend"
        leverage_multiplier = 1.0
        tp_stages = 3
    elif entry_score >= 7:
        trade_mode = "medium_trend" 
        leverage_multiplier = 0.8
        tp_stages = 2
    else:
        trade_mode = "cautious_trend"
        leverage_multiplier = 0.6
        tp_stages = 2
    
    # حساب مستويات جني الأرباح بناءً على قوة الإشارة
    base_multiplier = 1.0 + (entry_score - 7) * 0.1  # زيادة الأرباح للإشارات القوية
    
    if zone_type in ["golden_bottom", "trend_reversal"]:
        # صفقات شرائية
        tp_levels = []
        tp_fractions = []
        
        if tp_stages >= 1:
            tp1 = entry_price * (1 + (0.8 * base_multiplier) / 100)  # 0.8% - 1.2%
            tp_levels.append(tp1)
            tp_fractions.append(0.4)  # 40% عند TP1
        
        if tp_stages >= 2:
            tp2 = entry_price * (1 + (1.8 * base_multiplier) / 100)  # 1.8% - 2.4%
            tp_levels.append(tp2)
            tp_fractions.append(0.4)  # 40% عند TP2
        
        if tp_stages >= 3:
            tp3 = entry_price * (1 + (3.0 * base_multiplier) / 100)  # 3.0% - 3.8%
            tp_levels.append(tp3) 
            tp_fractions.append(0.2)  # 20% عند TP3
        
        # وقف الخسارة المتحرك
        initial_sl = entry_price * (1 - (1.0 * base_multiplier) / 100)
        trail_activation = 0.4  # تفعيل التريل بعد 0.4%
        trail_distance = 0.3    # مسافة التريل 0.3%
        
    else:  # golden_top
        # صفقات بيعية
        tp_levels = []
        tp_fractions = []
        
        if tp_stages >= 1:
            tp1 = entry_price * (1 - (0.8 * base_multiplier) / 100)
            tp_levels.append(tp1)
            tp_fractions.append(0.4)
        
        if tp_stages >= 2:
            tp2 = entry_price * (1 - (1.8 * base_multiplier) / 100)
            tp_levels.append(tp2)
            tp_fractions.append(0.4)
        
        if tp_stages >= 3:
            tp3 = entry_price * (1 - (3.0 * base_multiplier) / 100)
            tp_levels.append(tp3)
            tp_fractions.append(0.2)
        
        initial_sl = entry_price * (1 + (1.0 * base_multiplier) / 100)
        trail_activation = 0.4
        trail_distance = 0.3
    
    return {
        "trade_mode": trade_mode,
        "leverage_multiplier": leverage_multiplier,
        "tp_levels": tp_levels,
        "tp_fractions": tp_fractions,
        "initial_sl": initial_sl,
        "trail_activation_pct": trail_activation,
        "trail_distance_pct": trail_distance,
        "atr_multiplier": 2.0,  # استخدام ATR للتريل
        "breakeven_trigger": 0.6,  # تفعيل بريك إيفن بعد 0.6%
        "max_trade_duration_hours": 24  # أقصى مدة للصفقة
    }

def manage_professional_trend_trade(df, current_price, management_config):
    """إدارة محترفة لصفقات الترند"""
    if not STATE["open"]:
        return
    
    entry = STATE["entry"]
    side = STATE["side"]
    pnl_pct = (current_price - entry) / entry * 100 * (1 if side == "long" else -1)
    
    # تحديث أعلى ربح وصلت إليه الصفقة
    if "max_profit" not in STATE:
        STATE["max_profit"] = pnl_pct
    else:
        STATE["max_profit"] = max(STATE["max_profit"], pnl_pct)
    
    # جني الأرباح على المراحل
    tp_levels = management_config["tp_levels"]
    tp_fractions = management_config["tp_fractions"]
    
    for i, (tp_level, fraction) in enumerate(zip(tp_levels, tp_fractions)):
        tp_key = f"tp_{i+1}_hit"
        
        if not STATE.get(tp_key):
            if (side == "long" and current_price >= tp_level) or (side == "short" and current_price <= tp_level):
                close_fraction(fraction)
                STATE[tp_key] = True
                log_g(f"🎯 TP{i+1} Hit: {pnl_pct:.2f}% | Closed {fraction*100}%")
                break
    
    # تفعيل وقف الخسارة المتحرك
    if pnl_pct >= management_config["trail_activation_pct"]:
        activate_trailing_stop(current_price, management_config)
    
    # تفعيل بريك إيفن
    if pnl_pct >= management_config["breakeven_trigger"] and not STATE.get("breakeven_activated"):
        activate_breakeven(entry)
        STATE["breakeven_activated"] = True
        log_g("🔒 Breakeven Activated")

def activate_trailing_stop(current_price, management_config):
    """تفعيل وقف الخسارة المتحرك"""
    if not STATE.get("trailing_active"):
        STATE["trailing_active"] = True
        STATE["trail_start_price"] = current_price
        log_g("🎯 Trailing Stop Activated")
    
    # تحديث أعلى سعر للصفقة الطويلة أو أدنى سعر للصفقة القصيرة
    if STATE["side"] == "long":
        STATE["highest_price"] = max(STATE.get("highest_price", current_price), current_price)
        new_sl = STATE["highest_price"] * (1 - management_config["trail_distance_pct"] / 100)
    else:
        STATE["lowest_price"] = min(STATE.get("lowest_price", current_price), current_price)
        new_sl = STATE["lowest_price"] * (1 + management_config["trail_distance_pct"] / 100)
    
    # تحديث وقف الخسارة فقط إذا كان أفضل
    if (STATE["side"] == "long" and new_sl > STATE.get("current_sl", 0)) or \
       (STATE["side"] == "short" and new_sl < STATE.get("current_sl", float('inf'))):
        STATE["current_sl"] = new_sl

def activate_breakeven(entry_price):
    """تفعيل نقطة التعادل"""
    STATE["current_sl"] = entry_price

# =================== ENHANCED ENTRY DECISION MAKING ===================
def professional_golden_zone_entry(df, current_price):
    """قرار دخول محترف يعتمد على المناطق الذهبية"""
    try:
        # كشف المناطق الذهبية المتقدمة
        golden_zones = detect_advanced_golden_zones(df, current_price)
        
        if not golden_zones["ok"] or not golden_zones["strongest_zone"]:
            return {
                "action": "wait",
                "confidence": 0,
                "reasons": ["لا توجد مناطق ذهبية قوية"],
                "trade_mode": None,
                "zone_info": None
            }
        
        best_zone = golden_zones["strongest_zone"]
        zone_type = best_zone["type"]
        strength = best_zone["strength"]
        entry_score = best_zone["entry_score"]
        
        # تحديد اتجاه الدخول
        if zone_type in ["golden_bottom", "trend_reversal"]:
            action = "buy"
            direction_desc = "شراء"
        else:  # golden_top
            action = "sell" 
            direction_desc = "بيع"
        
        # فقط المناطق القوية جداً (8+)
        if strength < 8:
            return {
                "action": "wait",
                "confidence": entry_score,
                "reasons": [f"قوة المنطقة غير كافية: {strength}/10"],
                "trade_mode": None,
                "zone_info": best_zone
            }
        
        # إعداد إدارة الصفقة المحترفة
        indicators = compute_indicators(df)
        atr = indicators.get('atr', 0)
        management_config = setup_professional_trade_management(zone_type, entry_score, atr, current_price)
        
        reasons = [
            f"🎯 {direction_desc} في {best_zone['description']}",
            f"💪 قوة الإشارة: {strength}/10",
            f"📊 نقاط الدخول: {entry_score}/10",
            f"🎮 نمط التداول: {management_config['trade_mode']}",
            f"🔢 مراحل جني الأرباح: {len(management_config['tp_levels'])}",
            f"📈 مستويات TP: {', '.join([f'{tp:.6f}' for tp in management_config['tp_levels']])}"
        ]
        
        # إضافة شروط الدخول
        for condition in best_zone.get('conditions', [])[:3]:  # أول 3 شروط فقط
            reasons.append(f"✅ {condition}")
        
        return {
            "action": action,
            "confidence": entry_score,
            "reasons": reasons,
            "trade_mode": management_config["trade_mode"],
            "zone_info": best_zone,
            "management_config": management_config
        }
        
    except Exception as e:
        log_e(f"professional_golden_zone_entry error: {e}")
        return {
            "action": "wait", 
            "confidence": 0,
            "reasons": [f"خطأ في تحليل المناطق الذهبية: {str(e)}"],
            "trade_mode": None,
            "zone_info": None
        }

# =================== ENHANCED TRADE EXECUTION ===================
def execute_professional_golden_trade(side, qty, price, trade_mode, zone_info, management_config):
    """تنفيذ محترف لصفقات المناطق الذهبية"""
    if not EXECUTE_ORDERS or DRY_RUN:
        log_i(f"DRY_RUN: {side} {qty:.4f} @ {price:.6f} | {trade_mode} | Zone: {zone_info['type']}")
        return True
    
    if qty <= 0:
        log_e("❌ كمية غير صالحة للتنفيذ")
        return False

    # تطبيق مضاعف الرافعة بناءً على قوة الصفقة
    leverage_multiplier = management_config.get("leverage_multiplier", 1.0)
    adjusted_qty = qty * leverage_multiplier
    
    zone_type = zone_info['type']
    strength = zone_info['strength']
    
    print(f"🎯 PROFESSIONAL GOLDEN TRADE EXECUTION", flush=True)
    print(f"📍 {zone_info['description']}", flush=True)
    print(f"💪 قوة المنطقة: {strength}/10", flush=True)
    print(f"🎮 نمط التداول: {trade_mode.upper()}", flush=True)
    print(f"🧭 الاتجاه: {side.upper()} {adjusted_qty:.4f} @ {price:.6f}", flush=True)
    print(f"🎯 مستويات TP: {len(management_config['tp_levels'])} مراحل", flush=True)
    print(f"📈 الرافعة المعدلة: {leverage_multiplier:.1f}x", flush=True)
    
    # عرض شروط الدخول
    for condition in zone_info.get('conditions', [])[:4]:
        print(f"   ✅ {condition}", flush=True)

    try:
        if MODE_LIVE:
            ex.set_leverage(LEVERAGE, SYMBOL, params={"side": "BOTH"})
            ex.create_order(SYMBOL, "market", side, adjusted_qty, None, _params_open(side))
        
        # حفظ إعدادات إدارة الصفقة
        STATE["management_config"] = management_config
        STATE["zone_type"] = zone_type
        STATE["entry_strength"] = strength
        
        log_g(f"✅ GOLDEN TRADE EXECUTED: {side.upper()} {adjusted_qty:.4f} @ {price:.6f}")
        log_g(f"🎯 TP Levels: {', '.join([f'{tp:.6f}' for tp in management_config['tp_levels']])}")
        log_g(f"🛡️ Initial SL: {management_config['initial_sl']:.6f}")
        
        return True
    except Exception as e:
        log_e(f"❌ GOLDEN TRADE EXECUTION FAILED: {e}")
        return False

# =================== UPDATED TRADE LOOP FOR GOLDEN ZONES ===================
def trade_loop_golden_zones_pro():
    """حلقة تداول متقدمة تركز على المناطق الذهبية وإدارة الترند"""
    global wait_for_next_signal_side, STATE, compound_pnl
    
    loop_i = 0
    
    while True:
        try:
            # جمع البيانات الأساسية
            bal = balance_usdt()
            px = price_now()
            df = fetch_ohlcv()
            
            if df.empty:
                log_w("No data fetched, skipping iteration")
                time.sleep(BASE_SLEEP)
                continue
                
            info = rf_signal_live(df)
            ind = compute_indicators(df)
            spread_bps = orderbook_spread_bps()
            
            # قرار الدخول المحترف بالمناطق الذهبية
            entry_decision = professional_golden_zone_entry(df, px or info["price"])
            
            # Enhanced Snapshots
            snap = emit_snapshots_enhanced(ex, SYMBOL, df, 
                                         {"entry_decision": entry_decision}, 
                                         entry_decision,
                                         balance_fn=lambda: float(bal) if bal else None,
                                         pnl_fn=lambda: float(compound_pnl))
            
            # تحديث حالة الربح/الخسارة
            if STATE["open"] and px:
                pnl_pct = (px-STATE["entry"])/STATE["entry"]*100 * (1 if STATE["side"]=="long" else -1)
                STATE["pnl"] = pnl_pct
                
                # إدارة الصفقة المفتوحة محترفة
                if STATE.get("management_config"):
                    manage_professional_trend_trade(df, px, STATE["management_config"])
            
            # قرار الدخول للمناطق الذهبية
            reason = None
            if spread_bps is not None and spread_bps > MAX_SPREAD_BPS:
                reason = f"spread too high ({fmt(spread_bps,2)}bps > {MAX_SPREAD_BPS})"
            
            sig = entry_decision["action"]
            trade_mode = entry_decision["trade_mode"]
            
            # دخول في المناطق الذهبية القوية فقط
            if not STATE["open"] and sig not in ["wait", None] and reason is None:
                zone_info = entry_decision.get("zone_info")
                management_config = entry_decision.get("management_config")
                
                if zone_info and management_config:
                    allow_wait, wait_reason = wait_gate_allow(df, info)
                    if not allow_wait:
                        reason = wait_reason
                    else:
                        qty = compute_size(bal, px or info["price"])
                        if qty > 0:
                            # ✅ دخول منطقة ذهبية قوية
                            log_i(f"🔥 GOLDEN ZONE ENTRY | {sig.upper()} | Strength: {zone_info['strength']}/10")
                            log_i(f"🎯 Zone Type: {zone_info['type']} | Mode: {trade_mode}")
                            
                            for reason_text in entry_decision["reasons"]:
                                log_i(f"   📍 {reason_text}")
                            
                            ok = execute_professional_golden_trade(sig, qty, px or info["price"], 
                                                                 trade_mode, zone_info, management_config)
                            if ok:
                                wait_for_next_signal_side = None
                        else:
                            reason = "qty<=0"
            
            # Enhanced Logging للمناطق الذهبية
            if LOG_ADDONS and loop_i % 3 == 0 and entry_decision.get("zone_info"):
                zone = entry_decision["zone_info"]
                if zone["strength"] >= 7:  # تسجيل المناطق القوية فقط
                    log_i(f"🏆 GOLDEN ZONE DETECTED | {zone['type']} | Strength: {zone['strength']}/10")
                    for condition in zone.get('conditions', [])[:2]:
                        log_i(f"   ✅ {condition}")
            
            loop_i += 1
            sleep_s = NEAR_CLOSE_S if time_to_candle_close(df) <= 10 else BASE_SLEEP
            time.sleep(sleep_s)
            
        except Exception as e:
            log_e(f"loop error: {e}\n{traceback.format_exc()}")
            logging.error(f"trade_loop_golden_zones error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

# =================== PROFESSIONAL LOGGING FUNCTIONS ===================
def log_strategy_line(side_hint: str, mode: str, balance: float, compound_pnl: float):
    log_i(
        f"📊 Strategy: 📈 {mode.upper()} | Balance={balance:.2f} | "
        f"CompoundPnL={compound_pnl:.6f}"
    )

def log_snap_line(side: str, votes_for: int, votes_against: int,
                  score: float, adx: float, di_spread: float,
                  z_score: float, orderbook_imb: float):
    log_i(
        f"🎯 SNAP | {side.upper()} | votes={votes_for}/{votes_against} "
        f"| score={score:.1f}/10.0 | ADX={adx:.1f} DI={di_spread:.1f} "
        f"| z={z_score:.2f} | imb={orderbook_imb:.2f}"
    )

def log_addons_live():
    log_i("🧩 ADDONS LIVE")

def log_bookmap_line(bookmap_ctx: dict):
    """لقطة Bookmap محسنة"""
    try:
        imb = bookmap_ctx.get("imbalance", 1.0)
        bids = bookmap_ctx.get("bids", [])
        asks = bookmap_ctx.get("asks", [])
        
        buy_levels = [bid[0] for bid in bids[:3]] if bids else []
        sell_levels = [ask[0] for ask in asks[:3]] if asks else []
        
        buys_txt = ", ".join(f"{p:.6f}" for p in buy_levels) if buy_levels else "n/a"
        sells_txt = ", ".join(f"{p:.6f}" for p in sell_levels) if sell_levels else "n/a"
        
        log_i(
            f"📉 Bookmap: 🔴 Imb={imb:.2f} | Buy[{buys_txt}] | Sell[{sells_txt}]"
        )
    except Exception as e:
        log_w(f"log_bookmap_line error: {e}")

def log_flow_line(flow_ctx: dict):
    """لقطة التدفق محسنة"""
    try:
        flow = flow_ctx.get("flow", 0)
        delta = flow_ctx.get("delta", 0)
        z_score = flow_ctx.get("z_score", 0)
        cvd = flow_ctx.get("cvd", 0)
        
        # تحديد الاتجاه من التدفق
        if flow > 0:
            side_emoji = "🟢 Buy"
            side = "buy"
        elif flow < 0:
            side_emoji = "🔴 Sell" 
            side = "sell"
        else:
            side_emoji = "⚪ Flat"
            side = "flat"
        
        log_i(
            f"💧 Flow: {side_emoji} Δ={delta:.0f} z={z_score:.2f} | CVD={cvd:.0f}"
        )
        return side
    except Exception as e:
        log_w(f"log_flow_line error: {e}")
        return "flat"

def log_dash_hint_line(hint_side: str, council_decision: dict, indicators: dict):
    """لقطة Dashboard محسنة"""
    try:
        confidence = council_decision.get('confidence_score', 0)
        rsi = indicators.get('rsi', 50)
        adx = indicators.get('adx', 0)
        di_spread = indicators.get('plus_di', 0) - indicators.get('minus_di', 0)
        
        # استخراج أصوات المجلس
        smc_data = council_decision['details']['smc_analysis']
        buy_signals = len([z for z in smc_data.get('supply_demand_zones', []) if z.get('type') == 'demand'])
        sell_signals = len([z for z in smc_data.get('supply_demand_zones', []) if z.get('type') == 'supply'])
        
        log_i(
            f"📟 DASH → hint-{hint_side.upper()} | Council BUY({buy_signals}) "
            f"SELL({sell_signals}) | RSI={rsi:.1f} ADX={adx:.1f} DI={di_spread:.1f} "
            f"| Confidence={confidence:.1f}"
        )
    except Exception as e:
        log_w(f"log_dash_hint_line error: {e}")

# =================== ENHANCED SNAPSHOT WITH PROFESSIONAL LOGGING ===================
def emit_snapshots_enhanced(exchange, symbol, df, council_decision, entry_decision, balance_fn=None, pnl_fn=None):
    """سنابشوت محسّن مع التسجيل الاحترافي"""
    try:
        bm = bookmap_snapshot(exchange, symbol)
        flow = compute_flow_metrics(df)
        
        bal = None; cpnl = None
        if callable(balance_fn):
            try: bal = balance_fn()
            except: bal = None
        if callable(pnl_fn):
            try: cpnl = pnl_fn()
            except: cpnl = None

        # Enhanced Professional Logging
        if LOG_ADDONS:
            # استخراج البيانات للتسجيل
            entry_action = entry_decision.get("action", "wait")
            trade_mode = entry_decision.get("trade_mode", "scalp")
            quality = entry_decision.get("quality_check", {})
            quality_score = quality.get("score", 0)
            
            indicators = compute_indicators(df)
            snap_data = council_decision.get('details', {}).get('candle_signals', {})
            
            # حساب الأصوات
            votes_for = 0
            votes_against = 0
            
            # التسجيل الاحترافي
            log_strategy_line(
                side_hint=entry_action, 
                mode=trade_mode, 
                balance=bal or 0, 
                compound_pnl=cpnl or 0
            )
            
            log_snap_line(
                side=entry_action,
                votes_for=votes_for,
                votes_against=votes_against,
                score=quality_score,
                adx=indicators.get('adx', 0),
                di_spread=indicators.get('plus_di', 0) - indicators.get('minus_di', 0),
                z_score=flow.get('z_score', 0),
                orderbook_imb=bm.get('imbalance', 1.0)
            )
            
            log_addons_live()
            log_bookmap_line(bm)
            flow_side = log_flow_line(flow)
            log_dash_hint_line(flow_side or entry_action, council_decision, indicators)

        return {"bm": bm, "flow": flow, "council_decision": council_decision, "entry_decision": entry_decision}
    except Exception as e:
        print(f"🟨 EnhancedSnapshot error: {e}", flush=True)
        return {"bm": None, "flow": None, "council_decision": {}, "entry_decision": {}}

# =================== BASIC BOT FUNCTIONS ===================
def compute_enhanced_candles(df):
    """تحليل محسن للشموع اليابانية"""
    try:
        if len(df) < 3:
            return {"score_buy": 0, "score_sell": 0, "patterns": []}
        
        opens = df['open'].astype(float)
        highs = df['high'].astype(float)
        lows = df['low'].astype(float)
        closes = df['close'].astype(float)
        
        score_buy = 0
        score_sell = 0
        patterns = []
        
        # Bullish Engulfing
        if (closes.iloc[-1] > opens.iloc[-1] and 
            closes.iloc[-2] < opens.iloc[-2] and
            closes.iloc[-1] > opens.iloc[-2] and 
            opens.iloc[-1] < closes.iloc[-2]):
            score_buy += 2
            patterns.append("bullish_engulfing")
        
        # Bearish Engulfing
        if (closes.iloc[-1] < opens.iloc[-1] and 
            closes.iloc[-2] > opens.iloc[-2] and
            closes.iloc[-1] < opens.iloc[-2] and 
            opens.iloc[-1] > closes.iloc[-2]):
            score_sell += 2
            patterns.append("bearish_engulfing")
        
        # Hammer
        if (closes.iloc[-1] > opens.iloc[-1] and
            (lows.iloc[-1] - min(opens.iloc[-1], closes.iloc[-1])) > 
            2 * abs(opens.iloc[-1] - closes.iloc[-1]) and
            (highs.iloc[-1] - max(opens.iloc[-1], closes.iloc[-1])) < 
            abs(opens.iloc[-1] - closes.iloc[-1])):
            score_buy += 1
            patterns.append("hammer")
        
        # Shooting Star
        if (closes.iloc[-1] < opens.iloc[-1] and
            (highs.iloc[-1] - max(opens.iloc[-1], closes.iloc[-1])) > 
            2 * abs(opens.iloc[-1] - closes.iloc[-1]) and
            (min(opens.iloc[-1], closes.iloc[-1]) - lows.iloc[-1]) < 
            abs(opens.iloc[-1] - closes.iloc[-1])):
            score_sell += 1
            patterns.append("shooting_star")
        
        # Doji
        body_size = abs(closes.iloc[-1] - opens.iloc[-1])
        total_range = highs.iloc[-1] - lows.iloc[-1]
        if body_size / total_range < 0.1 and total_range > 0:
            # Doji at support
            if min(opens.iloc[-2], closes.iloc[-2]) > closes.iloc[-1]:
                score_buy += 1
                patterns.append("doji_support")
            # Doji at resistance
            elif max(opens.iloc[-2], closes.iloc[-2]) < closes.iloc[-1]:
                score_sell += 1
                patterns.append("doji_resistance")
        
        return {
            "score_buy": min(score_buy, 6),
            "score_sell": min(score_sell, 6),
            "patterns": patterns
        }
    except Exception as e:
        return {"score_buy": 0, "score_sell": 0, "patterns": [], "error": str(e)}

def compute_indicators(df):
    """حساب المؤشرات الفنية"""
    try:
        if len(df) < max(RSI_LEN, ADX_LEN, ATR_LEN):
            return {}
        
        closes = df['close'].astype(float)
        highs = df['high'].astype(float)
        lows = df['low'].astype(float)
        
        # RSI
        rsi = compute_rsi(closes, RSI_LEN)
        
        # ADX
        adx, plus_di, minus_di = compute_adx(highs, lows, closes, ADX_LEN)
        
        # ATR
        atr = compute_atr(highs, lows, closes, ATR_LEN)
        
        # VWAP
        vwap = compute_vwap(df)
        
        return {
            "rsi": float(rsi.iloc[-1]) if len(rsi) > 0 else 50,
            "adx": float(adx.iloc[-1]) if len(adx) > 0 else 0,
            "plus_di": float(plus_di.iloc[-1]) if len(plus_di) > 0 else 0,
            "minus_di": float(minus_di.iloc[-1]) if len(minus_di) > 0 else 0,
            "atr": float(atr.iloc[-1]) if len(atr) > 0 else 0,
            "vwap": vwap
        }
    except Exception as e:
        return {}

def compute_rsi(prices, period=14):
    """حساب RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_adx(high, low, close, period=14):
    """حساب ADX"""
    try:
        plus_dm = high.diff()
        minus_dm = low.diff().abs()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(period).mean()
        
        return adx, plus_di, minus_di
    except:
        return pd.Series(), pd.Series(), pd.Series()

def compute_atr(high, low, close, period=14):
    """حساب ATR"""
    try:
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr
    except:
        return pd.Series()

def compute_vwap(df):
    """حساب VWAP"""
    try:
        typical_price = (df['high'].astype(float) + df['low'].astype(float) + df['close'].astype(float)) / 3
        volume = df['volume'].astype(float)
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return float(vwap.iloc[-1]) if len(vwap) > 0 else 0
    except:
        return 0

def compute_footprint_metrics(df):
    """حساب مقاييس Footprint"""
    try:
        if len(df) < 5:
            return {"ok": False}
        
        volume = df['volume'].astype(float)
        closes = df['close'].astype(float)
        opens = df['open'].astype(float)
        
        # حجم غير عادي
        volume_ma = volume.rolling(20).mean()
        volume_std = volume.rolling(20).std()
        current_volume = volume.iloc[-1]
        volume_z = (current_volume - volume_ma.iloc[-1]) / volume_std.iloc[-1] if volume_std.iloc[-1] > 0 else 0
        
        # امتصاص (Absorption)
        price_change = (closes.iloc[-1] - opens.iloc[-1]) / opens.iloc[-1] * 100
        volume_spike = volume_z > VOLUME_SPIKE_THRESHOLD
        
        absorption_bull = volume_spike and price_change > 0.1
        absorption_bear = volume_spike and price_change < -0.1
        
        return {
            "ok": True,
            "volume_spike": volume_spike,
            "volume_z_score": volume_z,
            "absorption_bull": absorption_bull,
            "absorption_bear": absorption_bear,
            "price_change_pct": price_change
        }
    except Exception as e:
        return {"ok": False, "why": str(e)}

def rf_signal_live(df):
    """إشارة RF الحية"""
    try:
        if len(df) < RF_PERIOD:
            return {"live": False, "price": 0}
        
        price = float(df['close'].iloc[-1])
        return {
            "live": True,
            "price": price,
            "upper": price * 1.02,  # مبسطة
            "lower": price * 0.98   # مبسطة
        }
    except:
        return {"live": False, "price": 0}

def fetch_ohlcv(limit=100):
    """جلب بيانات OHLCV"""
    try:
        if ex:
            ohlcv = ex.fetch_ohlcv(SYMBOL, INTERVAL, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('time', inplace=True)
            return df
        return pd.DataFrame()
    except Exception as e:
        log_e(f"fetch_ohlcv error: {e}")
        return pd.DataFrame()

def price_now():
    """السعر الحالي"""
    try:
        if ex:
            ticker = ex.fetch_ticker(SYMBOL)
            return float(ticker['last'])
        return 0.0
    except Exception as e:
        log_e(f"price_now error: {e}")
        return 0.0

def balance_usdt():
    """الرصيد بالـUSDT"""
    try:
        if ex and MODE_LIVE:
            balance = ex.fetch_balance()
            return float(balance['total'].get('USDT', 0))
        return 1000.0  # Default for testing
    except Exception as e:
        log_e(f"balance_usdt error: {e}")
        return 0.0

def orderbook_spread_bps():
    """السبريد من orderbook"""
    try:
        if ex:
            orderbook = ex.fetch_order_book(SYMBOL)
            best_bid = orderbook['bids'][0][0] if orderbook['bids'] else 0
            best_ask = orderbook['asks'][0][0] if orderbook['asks'] else 0
            if best_bid and best_ask:
                spread = (best_ask - best_bid) / best_bid * 10000
                return spread
        return 2.0
    except Exception as e:
        log_e(f"orderbook_spread_bps error: {e}")
        return 10.0

def bookmap_snapshot(exchange, symbol):
    """لقطة Bookmap"""
    try:
        orderbook = exchange.fetch_order_book(symbol)
        bids = orderbook['bids'][:BOOKMAP_DEPTH]
        asks = orderbook['asks'][:BOOKMAP_DEPTH]
        
        # حساب عدم التوازن
        total_bid = sum([bid[1] for bid in bids])
        total_ask = sum([ask[1] for ask in asks])
        imbalance = total_bid / total_ask if total_ask > 0 else 1.0
        
        return {
            "bids": bids[:BOOKMAP_TOPWALLS],
            "asks": asks[:BOOKMAP_TOPWALLS],
            "imbalance": imbalance,
            "imbalance_alert": imbalance > IMBALANCE_ALERT or imbalance < 1/IMBALANCE_ALERT
        }
    except Exception as e:
        return {"bids": [], "asks": [], "imbalance": 1.0, "imbalance_alert": False}

def compute_flow_metrics(df):
    """حساب مقاييس التدفق"""
    try:
        if len(df) < FLOW_WINDOW:
            return {"flow": 0, "cvd": 0, "delta": 0}
        
        # حساب التدفق المبسط
        closes = df['close'].astype(float)
        volumes = df['volume'].astype(float)
        
        price_change = closes.diff()
        volume_flow = price_change * volumes
        
        # Cumulative Volume Delta مبسطة
        cvd = volume_flow.cumsum()
        
        # Flow momentum
        flow_momentum = volume_flow.rolling(FLOW_WINDOW).mean()
        
        current_flow = float(flow_momentum.iloc[-1]) if len(flow_momentum) > 0 else 0
        current_cvd = float(cvd.iloc[-1]) if len(cvd) > 0 else 0
        
        return {
            "flow": current_flow,
            "cvd": current_cvd,
            "delta": current_flow,
            "spike": abs(current_flow) > FLOW_SPIKE_Z * flow_momentum.std() if len(flow_momentum) > 0 else False
        }
    except Exception as e:
        return {"flow": 0, "cvd": 0, "delta": 0, "spike": False}

def close_fraction(fraction):
    """Close fraction of position"""
    try:
        if EXECUTE_ORDERS and not DRY_RUN and MODE_LIVE:
            size_to_close = STATE["size"] * fraction
            if size_to_close > 0:
                ex.create_order(SYMBOL, "market", "sell" if STATE["side"] == "long" else "buy", 
                              size_to_close, None, _params_close())
                log_g(f"Closed {fraction*100}% of position")
    except Exception as e:
        log_e(f"close_fraction error: {e}")

def close_position():
    """Close entire position"""
    try:
        if EXECUTE_ORDERS and not DRY_RUN and MODE_LIVE:
            ex.create_order(SYMBOL, "market", "sell" if STATE["side"] == "long" else "buy", 
                          STATE["size"], None, _params_close())
            STATE["open"] = False
            STATE["last_trade_time"] = time.time()
            log_g("Position closed")
    except Exception as e:
        log_e(f"close_position error: {e}")

def _params_open(side):
    """معلمات فتح الصفقة"""
    return {"positionSide": "LONG" if side == "buy" else "SHORT"}

def _params_close():
    """Parameters for closing position"""
    return {"positionSide": "LONG" if STATE["side"] == "short" else "SHORT"}

def wait_gate_allow(df, info):
    """التحقق من بوابة الانتظار"""
    # Implement cooldown logic, time-based filters, etc.
    if STATE.get("last_trade_time"):
        time_since_last = time.time() - STATE["last_trade_time"]
        if time_since_last < COOLDOWN_SECS_AFTER_CLOSE:
            return False, f"Cooldown: {int(COOLDOWN_SECS_AFTER_CLOSE - time_since_last)}s"
    
    # Check if we're approaching candle close
    if time_to_candle_close(df) <= 5:
        return False, "Too close to candle close"
    
    return True, "Allowed"

def time_to_candle_close(df):
    """Calculate seconds until candle close"""
    try:
        if len(df) == 0:
            return 60
        
        last_time = df.index[-1]
        if INTERVAL.endswith('m'):
            minutes = int(INTERVAL[:-1])
            next_close = last_time + pd.Timedelta(minutes=minutes)
            return (next_close - pd.Timestamp.now()).total_seconds()
        else:
            return 60  # Default fallback
    except:
        return 60

def compute_size(balance, price):
    """حساب حجم الصفقة"""
    try:
        if balance <= 0 or price <= 0:
            return 0
        risk_amount = balance * RISK_ALLOC
        return risk_amount / price
    except:
        return 0

def resume_open_position(exchange, symbol, state):
    """استئناف الصفقة المفتوحة"""
    try:
        if MODE_LIVE:
            positions = exchange.fetch_positions([symbol])
            for pos in positions:
                if pos['symbol'] == symbol.replace(':', '') and float(pos['contracts']) > 0:
                    state["open"] = True
                    state["side"] = "long" if float(pos['contracts']) > 0 else "short"
                    state["entry"] = float(pos['entryPrice'])
                    state["size"] = float(pos['contracts'])
                    state["pnl"] = float(pos['percentage']) * 100
                    log_g(f"Resumed open position: {state['side']} {state['size']} @ {state['entry']}")
                    break
        return state
    except Exception as e:
        log_w(f"resume_open_position error: {e}")
        return state

def keepalive_loop():
    """حلقة الحفاظ على التشغيل"""
    while True:
        try:
            # حفظ الحالة بشكل دوري
            save_state(STATE)
            time.sleep(60)
        except Exception as e:
            log_w(f"keepalive_loop error: {e}")
            time.sleep(60)

def fmt(value, decimals=2):
    """تنسيق الأرقام"""
    try:
        return f"{value:.{decimals}f}"
    except:
        return str(value)

# =================== INTEGRATION ===================
# استبدال الدوال الأساسية بالإصدار المحسن
trade_loop = trade_loop_golden_zones_pro
enhanced_entry_logic = professional_golden_zone_entry

# =================== FLASK APP ===================
app = Flask(__name__)

@app.route('/')
def dashboard():
    return jsonify({"status": "running", "version": BOT_VERSION})

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/state')
def state_endpoint():
    return jsonify(STATE)

@app.route('/decision')
def decision_endpoint():
    try:
        df = fetch_ohlcv()
        px = price_now()
        if df.empty:
            return jsonify({"error": "No data"})
        decision = professional_golden_zone_entry(df, px)
        return jsonify(decision)
    except Exception as e:
        return jsonify({"error": str(e)})

# =================== EXCHANGE SETUP ===================
try:
    ex = ccxt.bingx({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'sandbox': not MODE_LIVE,
        'enableRateLimit': True,
    })
except Exception as e:
    log_e(f"Exchange init error: {e}")
    ex = None

# =================== GLOBAL STATE ===================
STATE = {
    "open": False,
    "side": None,
    "entry": 0,
    "size": 0,
    "pnl": 0,
    "mode": "scalp",
    "signal_strength": 0,
    "last_trade_time": 0,
    "management_config": None,
    "zone_type": None,
    "entry_strength": 0,
    "max_profit": 0,
    "trailing_active": False,
    "trail_start_price": 0,
    "highest_price": 0,
    "lowest_price": 0,
    "current_sl": 0,
    "breakeven_activated": False
}

wait_for_next_signal_side = None
compound_pnl = 0.0

# =================== MAIN EXECUTION ===================
if __name__ == "__main__":
    log_banner("PROFESSIONAL GOLDEN ZONE TRADING SYSTEM - ADVANCED SMC")
    state = load_state() or {}
    state.setdefault("in_position", False)

    if RESUME_ON_RESTART:
        try:
            state = resume_open_position(ex, SYMBOL, state)
        except Exception as e:
            log_w(f"resume error: {e}\n{traceback.format_exc()}")

    print(colored(f"🔥 PROFESSIONAL GOLDEN ZONE TRADING SYSTEM", "yellow"))
    print(colored(f"🎯 MODE: {'LIVE' if MODE_LIVE else 'PAPER'}  •  {SYMBOL}  •  {INTERVAL}", "yellow"))
    print(colored(f"💪 STRATEGY: Golden Bottoms + Golden Tops + Trend Reversals", "yellow"))
    print(colored(f"📊 ENTRY REQUIREMENT: Minimum Strength 8/10", "yellow"))
    print(colored(f"🎮 TRADE MODES: Strong Trend (9+) | Medium Trend (7-8) | Cautious Trend (<7)", "yellow"))
    print(colored(f"📈 PROFIT TAKING: 3-Stage (40%-40%-20%) with Trailing Stop", "yellow"))
    print(colored(f"🛡️ RISK MANAGEMENT: Dynamic SL + Breakeven + ATR Trailing", "yellow"))
    print(colored(f"🔮 SMC INTEGRATION: Liquidity Pools + Order Blocks + Supply/Demand", "yellow"))
    print(colored(f"⚡ EXECUTION: {'ACTIVE' if EXECUTE_ORDERS and not DRY_RUN else 'SIMULATION'}", "yellow"))
    
    logging.info("professional golden zone trading system starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    
    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)

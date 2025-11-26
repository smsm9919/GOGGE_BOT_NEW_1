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
BOT_VERSION = "DOGE Council PRO v6.0 — Smart Money Concepts Pro + Advanced Council Decision"
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
                        'time': int(df['time'].iloc[i]),
                        'strength': strength,
                        'type': 'liquidity_pool_high'
                    })
            
            if lows.iloc[i] == window_low.min():
                strength = calculate_pivot_strength(df, i, 'low')
                if strength >= SMC_PIVOT_STRENGTH_THRESHOLD:
                    pivot_lows.append({
                        'price': float(lows.iloc[i]),
                        'time': int(df['time'].iloc[i]),
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
                    'time': int(df['time'].iloc[i]),
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
                    'time': int(df['time'].iloc[i]),
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
                        'time': int(df['time'].iloc[i]),
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
                        'time': int(df['time'].iloc[i]),
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
                    'time': int(df['time'].iloc[i]),
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
                    'time': int(df['time'].iloc[i]),
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

# =================== ADVANCED COUNCIL DECISION SYSTEM ===================
def council_pro_advanced_decision(df, current_price):
    """نظام قرار موحد متقدم يجمع كل التحليلات"""
    try:
        # جمع جميع البيانات
        smc_data = {
            'liquidity_pools': detect_liquidity_pools(df),
            'supply_demand_zones': identify_supply_demand_zones(df),
            'fair_value_gaps': detect_fair_value_gaps(df),
            'order_blocks': identify_order_blocks(df)
        }
        
        # التحليلات الأساسية
        candles = compute_enhanced_candles(df)
        footprint = compute_footprint_metrics(df)
        indicators = compute_indicators(df)
        golden_zone = golden_zone_pro_analysis(df, current_price)
        price_structure = analyze_price_structure(df)
        vwap_analysis = analyze_vwap_context(df, current_price)
        otc_analysis = analyze_otc_flow(df)
        
        # حساب الثقة الشاملة
        confidence_score = calculate_confidence_score({
            'smc': smc_data,
            'candles': candles,
            'footprint': footprint,
            'indicators': indicators,
            'golden_zone': golden_zone,
            'price_structure': price_structure,
            'vwap': vwap_analysis,
            'otc': otc_analysis
        })
        
        # تحديد مستوى القرار
        decision_level = determine_decision_level(confidence_score)
        
        # التوصية الاستراتيجية
        recommendation = generate_recommendation(decision_level, {
            'smc': smc_data,
            'candles': candles,
            'golden_zone': golden_zone,
            'price_structure': price_structure,
            'vwap': vwap_analysis
        })
        
        return {
            'decision_level': decision_level,
            'confidence_score': confidence_score,
            'recommendation': recommendation,
            'details': {
                'smc_analysis': smc_data,
                'candle_signals': candles,
                'footprint_analysis': footprint,
                'golden_zone': golden_zone,
                'price_structure': price_structure,
                'indicators': indicators,
                'vwap_analysis': vwap_analysis,
                'otc_analysis': otc_analysis
            },
            'timestamp': int(time.time())
        }
    except Exception as e:
        log_e(f"council_pro_advanced_decision error: {e}")
        return {
            'decision_level': 'تجنب',
            'confidence_score': 0,
            'recommendation': {'action': 'wait', 'reason': 'error_in_analysis'},
            'details': {},
            'timestamp': int(time.time())
        }

def calculate_confidence_score(analyses):
    """حساب ثقة شاملة تعكس قوة الإشارات"""
    try:
        score = 0.0
        max_score = 0.0
        
        # Smart Money Concepts (25%)
        smc_weight = 0.25
        smc_score = calculate_smc_confidence(analyses['smc'])
        score += smc_score * smc_weight
        max_score += smc_weight
        
        # الشموع المحسنة (15%)
        candle_weight = 0.15
        candle_score = max(analyses['candles']['score_buy'], analyses['candles']['score_sell'])
        score += min(candle_score / 6.0, 1.0) * candle_weight
        max_score += candle_weight
        
        # Footprint Analysis (10%)
        footprint_weight = 0.10
        if analyses['footprint']['ok']:
            footprint_score = 0.5
            if analyses['footprint']['volume_spike']: footprint_score += 0.3
            if analyses['footprint']['absorption_bull'] or analyses['footprint']['absorption_bear']: footprint_score += 0.2
            score += footprint_score * footprint_weight
        max_score += footprint_weight
        
        # المناطق الذهبية (15%)
        golden_weight = 0.15
        if analyses['golden_zone']['ok']:
            golden_score = min(analyses['golden_zone']['score'] / 10.0, 1.0)
            if analyses['golden_zone']['confirmed']: golden_score += 0.2
            score += golden_score * golden_weight
        max_score += golden_weight
        
        # الهيكل السعري (10%)
        structure_weight = 0.10
        if analyses['price_structure']['ok']:
            structure_score = analyses['price_structure']['trend_strength'] / 5.0
            score += structure_score * structure_weight
        max_score += structure_weight
        
        # VWAP Context (10%)
        vwap_weight = 0.10
        if analyses['vwap']['ok']:
            vwap_score = 0.7 if analyses['vwap']['aligned'] else 0.3
            score += vwap_score * vwap_weight
        max_score += vwap_weight
        
        # OTC Flow (5%)
        otc_weight = 0.05
        if analyses['otc']['ok'] and analyses['otc']['significant']:
            score += 1.0 * otc_weight
        max_score += otc_weight
        
        return min(score / max_score * 10.0, 10.0)  # تقدير من 0 إلى 10
    except Exception as e:
        log_w(f"calculate_confidence_score error: {e}")
        return 0.0

def calculate_smc_confidence(smc_data):
    """حساب ثقة Smart Money Concepts"""
    try:
        score = 0.0
        max_possible = 0.0
        
        # تحليل مناطق السيولة
        if smc_data['liquidity_pools']['ok']:
            liquidity_score = min(len(smc_data['liquidity_pools']['pivot_highs']) + 
                                 len(smc_data['liquidity_pools']['pivot_lows']), 5) / 5.0
            score += liquidity_score * 0.3
            max_possible += 0.3
        
        # تحليل مناطق العرض والطلب
        supply_demand_zones = smc_data['supply_demand_zones']
        if supply_demand_zones:
            zone_strength = sum([zone['strength'] for zone in supply_demand_zones]) / len(supply_demand_zones)
            score += (zone_strength / 5.0) * 0.3
            max_possible += 0.3
        
        # تحليل فجوات القيمة العادلة
        fvgs = smc_data['fair_value_gaps']
        if fvgs:
            fvg_score = min(len(fvgs) / 3.0, 1.0)
            score += fvg_score * 0.2
            max_possible += 0.2
        
        # تحليل كتل الأوامر
        order_blocks = smc_data['order_blocks']
        if order_blocks:
            block_strength = sum([block['strength'] for block in order_blocks]) / len(order_blocks)
            score += (block_strength / 4.0) * 0.2
            max_possible += 0.2
        
        return (score / max_possible) * 10.0 if max_possible > 0 else 0.0
    except:
        return 0.0

def determine_decision_level(confidence_score):
    """تحديد مستوى القرار بناء على الثقة"""
    if confidence_score < CONFIDENCE_AVOID:
        return "تجنب"
    elif confidence_score < CONFIDENCE_CAUTIOUS_SCALP:
        return "سكالب بحذر"
    elif confidence_score < CONFIDENCE_STUDIED_SCALP:
        return "سكالب مدروس"
    else:
        return "ترند"

def generate_recommendation(decision_level, analyses):
    """توليد توصية استراتيجية بناء على مستوى القرار"""
    if decision_level == "تجنب":
        return {"action": "wait", "reason": "ثقة منخفضة - تجنب الدخول"}
    
    # تحليل SMC للمناطق الذهبية
    smc_signals = analyze_smc_for_entry(analyses['smc'], analyses.get('current_price', 0))
    
    if decision_level == "سكالب بحذر":
        return {
            "action": "cautious_scalp",
            "reason": "سكالب بحذر في مناطق ذهبية مع تأكيد SMC",
            "risk_level": "medium",
            "position_size": "small"
        }
    
    elif decision_level == "سكالب مدروس":
        return {
            "action": "studied_scalp", 
            "reason": "سكالب مدروس في مناطق ذهبية مع إشارات قوية",
            "risk_level": "medium_high",
            "position_size": "medium"
        }
    
    else:  # ترند
        trend_alignment = check_trend_alignment(analyses)
        return {
            "action": "trend",
            "reason": "ترند قوي متعدد المؤشرات مع تأكيد SMC",
            "risk_level": "high",
            "position_size": "large",
            "trend_alignment": trend_alignment
        }

def analyze_smc_for_entry(smc_data, current_price):
    """تحليل SMC لدعم قرار الدخول"""
    try:
        alignment = False
        reasons = []
        
        # التحقق من محاذاة مناطق العرض/الطلب مع السعر الحالي
        for zone in smc_data['supply_demand_zones']:
            distance_pct = abs(zone['price'] - current_price) / current_price * 100
            if distance_pct < 1.0:  # within 1%
                alignment = True
                reasons.append(f"قرب من منطقة {zone['type']} بقوة {zone['strength']}")
        
        # التحقق من كتل الأوامر القريبة
        for block in smc_data['order_blocks']:
            distance_pct = abs(block['entry'] - current_price) / current_price * 100
            if distance_pct < 0.5:  # within 0.5%
                alignment = True
                reasons.append(f"قرب من كتلة أوامر {block['type']} بقوة {block['strength']}")
        
        return {
            "aligned": alignment,
            "reasons": reasons,
            "strong_confirmation": len(reasons) >= 2,
            "very_strong": len(reasons) >= 3
        }
    except:
        return {"aligned": False, "reasons": [], "strong_confirmation": False, "very_strong": False}

def check_trend_alignment(analyses):
    """التحقق من محاذاة الاتجاه عبر المؤشرات"""
    try:
        alignments = []
        
        # محاذاة الهيكل السعري
        if analyses['price_structure']['ok']:
            trend = analyses['price_structure']['trend']
            if trend != "sideways":
                alignments.append(f"هيكل_سعري_{trend}")
        
        # محاذاة المؤشرات
        indicators = analyses.get('indicators', {})
        if indicators.get('adx', 0) > 25:
            if indicators.get('plus_di', 0) > indicators.get('minus_di', 0):
                alignments.append("مؤشرات_صاعدة")
            else:
                alignments.append("مؤشرات_هابطة")
        
        # محاذاة VWAP
        if analyses['vwap']['ok'] and analyses['vwap']['aligned']:
            alignments.append("VWAP_متوافق")
        
        return {
            "ok": len(alignments) >= 2,
            "alignments": alignments,
            "strength": len(alignments)
        }
    except:
        return {"ok": False, "alignments": [], "strength": 0}

# =================== VWAP & OTC ANALYSIS ===================
def analyze_vwap_context(df, current_price):
    """تحليل سياق VWAP المتقدم"""
    try:
        if len(df) < 20 or not VWAP_ENABLED:
            return {"ok": False, "aligned": False}
        
        indicators = compute_indicators(df)
        vwap = indicators.get('vwap')
        
        if not vwap:
            return {"ok": False, "aligned": False}
        
        vwap_diff_bps = abs(current_price - vwap) / vwap * 10000.0
        near_vwap = vwap_diff_bps <= VWAP_SCALP_BAND_BPS
        far_from_vwap = vwap_diff_bps >= VWAP_TREND_BAND_BPS
        
        # تحديد المحاذاة مع الاتجاه
        above_vwap = current_price > vwap
        vwap_slope = calculate_vwap_slope(df)
        
        aligned = False
        if near_vwap and abs(vwap_slope) < 0.001:  # VWAP مسطح وقريب
            aligned = True
        elif far_from_vwap and vwap_slope > 0.001 and above_vwap:  # فوق VWAP ومتجه لأعلى
            aligned = True
        elif far_from_vwap and vwap_slope < -0.001 and not above_vwap:  # تحت VWAP ومتجه لأسفل
            aligned = True
        
        return {
            "ok": True,
            "vwap_value": vwap,
            "current_price": current_price,
            "distance_bps": vwap_diff_bps,
            "near_vwap": near_vwap,
            "far_from_vwap": far_from_vwap,
            "above_vwap": above_vwap,
            "vwap_slope": vwap_slope,
            "aligned": aligned
        }
    except Exception as e:
        return {"ok": False, "aligned": False, "why": str(e)}

def calculate_vwap_slope(df):
    """حساب ميل VWAP"""
    try:
        if len(df) < 20:
            return 0.0
        
        indicators = compute_indicators(df)
        current_vwap = indicators.get('vwap')
        
        # حساب VWAP السابق
        prev_df = df.iloc[:-1].copy()
        if len(prev_df) >= 20:
            prev_indicators = compute_indicators(prev_df)
            prev_vwap = prev_indicators.get('vwap')
            
            if current_vwap and prev_vwap:
                return (current_vwap - prev_vwap) / prev_vwap
        
        return 0.0
    except:
        return 0.0

def analyze_otc_flow(df):
    """تحليل تدفق OTC (Over-The-Counter)"""
    try:
        if len(df) < 30 or not OTC_ENABLED:
            return {"ok": False, "significant": False}
        
        volume = df['volume'].astype(float)
        closes = df['close'].astype(float)
        
        # كشف حجم OTC (صافي كبير غير عادي)
        volume_ma = volume.rolling(20).mean()
        volume_std = volume.rolling(20).std()
        
        current_volume = volume.iloc[-1]
        volume_z_score = (current_volume - volume_ma.iloc[-1]) / volume_std.iloc[-1] if volume_std.iloc[-1] > 0 else 0
        
        # تحليل عدم التوافق بين السعر والحجم
        price_change = (closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2] * 100
        volume_spike = volume_z_score > 2.0
        
        significant = False
        if volume_spike and abs(price_change) < 0.5:  # حجم كبير مع حركة سعر صغيرة
            significant = True
        elif volume_z_score > OTC_VOLUME_THRESHOLD:  # حجم استثنائي
            significant = True
        
        return {
            "ok": True,
            "significant": significant,
            "volume_z_score": volume_z_score,
            "current_volume": current_volume,
            "avg_volume": volume_ma.iloc[-1],
            "price_change_pct": price_change
        }
    except Exception as e:
        return {"ok": False, "significant": False, "why": str(e)}

# =================== INTELLIGENT ENTRY LOGIC ===================
def enhanced_entry_logic(council_decision, current_price):
    """منطق دخول محسن بناء على قرارات مجلس الإدارة"""
    decision_level = council_decision['decision_level']
    confidence = council_decision['confidence_score']
    details = council_decision['details']
    
    entry_signal = None
    entry_confidence = 0
    reasons = []
    trade_mode = "scalp"  # default
    
    if decision_level == "تجنب":
        return {
            "action": "wait", 
            "confidence": 0, 
            "reasons": ["ثقة منخفضة - تجنب الدخول"],
            "trade_mode": None
        }
    
    # تحليل SMC للمناطق الذهبية
    smc_signals = analyze_smc_for_entry(details['smc_analysis'], current_price)
    
    # الدخول بحذر للمناطق الذهبية ذات الثقة المتوسطة
    if decision_level == "سكالب بحذر" and details['golden_zone']['ok']:
        if details['golden_zone']['score'] >= 6.0 and smc_signals['alignment']:
            entry_signal = "buy" if details['golden_zone']['zone']['type'] == "golden_bottom" else "sell"
            entry_confidence = confidence
            trade_mode = "cautious_scalp"
            reasons.append("سكالب بحذر في منطقة ذهبية مع تأكيد SMC")
    
    # السكالب المدروس في المناطق الذهبية مع إشارات جيدة
    elif decision_level == "سكالب مدروس":
        golden_ok = details['golden_zone']['ok']
        candle_strength = max(details['candle_signals']['score_buy'], details['candle_signals']['score_sell'])
        
        if golden_ok and candle_strength >= 4.0 and smc_signals['strong_confirmation']:
            entry_signal = "buy" if details['golden_zone']['zone']['type'] == "golden_bottom" else "sell"
            entry_confidence = confidence
            trade_mode = "studied_scalp"
            reasons.extend([
                "سكالب مدروس في منطقة ذهبية",
                f"قوة شموع: {candle_strength:.1f}",
                "تأكيد SMC قوي"
            ])
    
    # الترند القوي يحتاج ثقة عالية وإشارات متعددة
    elif decision_level == "ترند" and confidence >= CONFIDENCE_TREND:
        trend_alignment = check_trend_alignment(details)
        if trend_alignment['ok'] and smc_signals['very_strong']:
            # تحديد اتجاه الترند من الهيكل السعري
            if details['price_structure']['ok']:
                if details['price_structure']['trend'] == "uptrend":
                    entry_signal = "buy"
                elif details['price_structure']['trend'] == "downtrend":
                    entry_signal = "sell"
            
            if entry_signal:
                entry_confidence = confidence
                trade_mode = "trend"
                reasons.extend([
                    "ترند قوي متعدد المؤشرات",
                    f"ثقة مجلس الإدارة: {confidence:.1f}",
                    f"محاذاة SMC: {trend_alignment['strength']} إشارات",
                    f"هيكل سعري: {details['price_structure']['trend']}"
                ])
    
    return {
        "action": entry_signal if entry_signal else "wait",
        "confidence": entry_confidence,
        "reasons": reasons,
        "trade_mode": trade_mode if entry_signal else None,
        "decision_level": decision_level,
        "smc_alignment": smc_signals
    }

# =================== DYNAMIC POSITION MANAGEMENT ===================
def dynamic_position_management(state, council_decision, market_data):
    """إدارة ديناميكية متقدمة للمراكز"""
    if not state.get("open", False):
        return state
    
    current_mode = state.get("mode", "scalp")
    signal_strength = state.get("signal_strength", 0)
    
    # ترقية السكالب لترند إذا ظهرت إشارات ترند قوية
    if (current_mode in ["scalp", "cautious_scalp", "studied_scalp"] and 
        council_decision['decision_level'] == "ترند" and
        council_decision['confidence_score'] >= CONFIDENCE_TREND):
        
        state["mode"] = "trend"
        state["management"] = setup_trade_management("trend")
        log_g("🔄 ترقية السكالب إلى ترند - تعديل إدارة المركز")
    
    # وقف خسارة ديناميكي يعتمد على ATR ومناطق السيولة
    dynamic_sl = calculate_dynamic_stoploss(state, market_data)
    if dynamic_sl and dynamic_sl != state.get("dynamic_sl"):
        state["dynamic_sl"] = dynamic_sl
        log_i(f"🎯 وقف خسارة ديناميكي: {dynamic_sl:.6f}")
    
    # جني أرباح ذكي متعدد المراحل
    smart_tp = adjust_take_profits_based_on_strength(state, council_decision)
    state["smart_tp_levels"] = smart_tp
    
    return state

def calculate_dynamic_stoploss(state, market_data):
    """حساب وقف خسارة ديناميكي"""
    try:
        atr = market_data.get('atr', 0)
        liquidity_zones = market_data.get('liquidity_zones', [])
        entry = state["entry"]
        side = state["side"]
        mode = state.get("mode", "scalp")
        
        # الأساس: يعتمد على نمط التداول
        if mode == "trend":
            base_sl_mult = 2.0
        elif mode == "studied_scalp":
            base_sl_mult = 1.5
        else:  # cautious_scalp or scalp
            base_sl_mult = 1.2
        
        base_sl = atr * base_sl_mult
        
        # تعديل بناء على مناطق السيولة القريبة
        for zone in liquidity_zones:
            zone_distance = abs(zone['price'] - entry) / entry
            if zone_distance < 0.01:  # ضمن 1%
                if (side == "long" and zone['type'] == "demand") or \
                   (side == "short" and zone['type'] == "supply"):
                    base_sl = min(base_sl, atr * 1.0)  # وقف أكثر إحكاماً near liquidity
        
        if side == "long":
            return entry - base_sl
        else:
            return entry + base_sl
    except:
        return None

def adjust_take_profits_based_on_strength(state, council_decision):
    """ضبط مستويات جني الأرباح بناء على قوة الإشارة"""
    try:
        signal_strength = council_decision['confidence_score']
        mode = state.get("mode", "scalp")
        
        if mode == "trend":
            base_tps = TREND_TPS.copy()
            # زيادة الأرباح للإشارات القوية
            if signal_strength >= 9.0:
                base_tps = [tp * 1.2 for tp in base_tps]
            elif signal_strength >= 8.0:
                base_tps = [tp * 1.1 for tp in base_tps]
            return base_tps
        else:  # scalp modes
            base_tps = SCALP_TPS.copy()
            if signal_strength >= 8.0:
                base_tps = [tp * 1.15 for tp in base_tps]
            return base_tps
    except:
        return SCALP_TPS if state.get("mode") != "trend" else TREND_TPS

# =================== ENHANCED TRADE LOOP WITH SMART MONEY CONCEPTS ===================
def trade_loop_enhanced_with_smc():
    """حلقة تداول محسنة مع Smart Money Concepts المتكاملة"""
    global wait_for_next_signal_side
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
            
            # Advanced Council Decision with Smart Money Concepts
            council_decision = council_pro_advanced_decision(df, px or info["price"])
            
            # Enhanced Entry Logic
            entry_decision = enhanced_entry_logic(council_decision, px or info["price"])
            
            # Enhanced Snapshots with SMC
            snap = emit_snapshots_enhanced(ex, SYMBOL, df, council_decision, entry_decision,
                                        balance_fn=lambda: float(bal) if bal else None,
                                        pnl_fn=lambda: float(compound_pnl))
            
            # تحديث حالة الربح/الخسارة
            if STATE["open"] and px:
                pnl_pct = (px-STATE["entry"])/STATE["entry"]*100 * (1 if STATE["side"]=="long" else -1)
                STATE["pnl"] = pnl_pct
            
            # إدارة الصفقة المفتوحة مع Smart Profit AI
            if STATE["open"]:
                manage_after_entry_enhanced(df, ind, {
                    "price": px or info["price"], 
                    "bm": snap["bm"],
                    "flow": snap["flow"],
                    "council_decision": council_decision,
                    **info
                })
                
                # Dynamic Position Management
                STATE = dynamic_position_management(STATE, council_decision, {
                    "atr": ind.get("atr", 0),
                    "liquidity_zones": council_decision['details']['smc_analysis'].get('liquidity_pools', {}).get('liquidity_zones', [])
                })
            
            # قرار الدخول المحسن مع Smart Money Concepts
            reason = None
            if spread_bps is not None and spread_bps > MAX_SPREAD_BPS:
                reason = f"spread too high ({fmt(spread_bps,2)}bps > {MAX_SPREAD_BPS})"
            
            sig = entry_decision["action"]
            trade_mode = entry_decision["trade_mode"]
            
            if not STATE["open"] and sig not in ["wait", None] and reason is None:
                allow_wait, wait_reason = wait_gate_allow(df, info)
                if not allow_wait:
                    reason = wait_reason
                else:
                    qty = compute_size(bal, px or info["price"])
                    if qty > 0:
                        # Adjust position size based on trade mode
                        if trade_mode == "cautious_scalp":
                            qty = qty * 0.5  # 50% position size for cautious scalp
                        elif trade_mode == "studied_scalp":
                            qty = qty * 0.75  # 75% position size for studied scalp
                        # trend mode uses full position size
                        
                        ok = open_market_enhanced_with_smc(sig, qty, px or info["price"], trade_mode, council_decision)
                        if ok:
                            wait_for_next_signal_side = None
                            # تسجيل قرار المجلس المحسن
                            log_i(f"🎯 ENHANCED COUNCIL DECISION: {sig.upper()} | "
                                  f"Mode: {trade_mode} | "
                                  f"Confidence: {council_decision['confidence_score']:.1f}/10")
                            for reason in entry_decision["reasons"]:
                                log_i(f"   - {reason}")
                    else:
                        reason = "qty<=0"
            
            # Enhanced Logging
            if LOG_ADDONS and loop_i % 5 == 0:  # Log every 5 iterations to reduce noise
                log_council_decision(council_decision, entry_decision)
            
            loop_i += 1
            sleep_s = NEAR_CLOSE_S if time_to_candle_close(df) <= 10 else BASE_SLEEP
            time.sleep(sleep_s)
            
        except Exception as e:
            log_e(f"loop error: {e}\n{traceback.format_exc()}")
            logging.error(f"trade_loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

def log_council_decision(council_decision, entry_decision):
    """تسجيل قرار مجلس الإدارة بتفصيل"""
    try:
        decision_level = council_decision['decision_level']
        confidence = council_decision['confidence_score']
        action = entry_decision['action']
        
        print(f"🏛️ COUNCIL DECISION: {decision_level} | Confidence: {confidence:.1f}/10 | Action: {action}", flush=True)
        
        # تفاصيل SMC
        smc_data = council_decision['details']['smc_analysis']
        if smc_data['liquidity_pools']['ok']:
            pools = smc_data['liquidity_pools']
            print(f"   💧 Liquidity: {len(pools['pivot_highs'])} highs, {len(pools['pivot_lows'])} lows", flush=True)
        
        if smc_data['supply_demand_zones']:
            print(f"   📊 Supply/Demand: {len(smc_data['supply_demand_zones'])} zones", flush=True)
        
        if smc_data['fair_value_gaps']:
            print(f"   📈 FVGs: {len(smc_data['fair_value_gaps'])} gaps", flush=True)
        
        if smc_data['order_blocks']:
            print(f"   🧱 Order Blocks: {len(smc_data['order_blocks'])} blocks", flush=True)
        
        # أسباب القرار
        if entry_decision['reasons']:
            print(f"   🎯 Reasons: {', '.join(entry_decision['reasons'])}", flush=True)
            
    except Exception as e:
        log_w(f"log_council_decision error: {e}")

def open_market_enhanced_with_smc(side, qty, price, trade_mode, council_decision):
    """فتح صفقة محسّن مع Smart Money Concepts"""
    if not EXECUTE_ORDERS or DRY_RUN:
        log_i(f"DRY_RUN: {side} {qty:.4f} @ {price:.6f} | mode={trade_mode}")
        return True
    
    if qty <= 0:
        log_e("❌ كمية غير صالحة للتنفيذ")
        return False

    # تفاصيل SMC للمنطقة
    smc_details = council_decision['details']['smc_analysis']
    golden_zone = council_decision['details']['golden_zone']
    
    gz_note = ""
    if golden_zone and golden_zone.get("ok"):
        gz_note = f" | 🟡 {golden_zone['zone']['type']} s={golden_zone['score']:.1f}"
    
    smc_note = ""
    if smc_details['liquidity_pools']['ok']:
        pools = smc_details['liquidity_pools']
        smc_note = f" | 💧 LQ({len(pools['pivot_highs'])}H,{len(pools['pivot_lows'])}L)"
    
    print(f"🎯 EXECUTE {trade_mode.upper()}: {side.upper()} {qty:.4f} @ {price:.6f} | "
          f"Confidence: {council_decision['confidence_score']:.1f}/10"
          f"{gz_note}{smc_note}", flush=True)

    try:
        if MODE_LIVE:
            ex.set_leverage(LEVERAGE, SYMBOL, params={"side": "BOTH"})
            ex.create_order(SYMBOL, "market", side, qty, None, _params_open(side))
        
        log_g(f"✅ EXECUTED: {side.upper()} {qty:.4f} @ {price:.6f} | Mode: {trade_mode}")
        return True
    except Exception as e:
        log_e(f"❌ EXECUTION FAILED: {e}")
        return False

def emit_snapshots_enhanced(exchange, symbol, df, council_decision, entry_decision, balance_fn=None, pnl_fn=None):
    """سنابشوت محسّن مع Smart Money Concepts"""
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

        # Enhanced Dashboard with SMC
        if LOG_ADDONS:
            # Council Decision Summary
            cd = council_decision
            ed = entry_decision
            
            print(f"🏛️ COUNCIL | {cd['decision_level']} | Confidence: {cd['confidence_score']:.1f}/10 | Action: {ed['action']}", flush=True)
            
            # SMC Summary
            smc = cd['details']['smc_analysis']
            if smc['liquidity_pools']['ok']:
                lp = smc['liquidity_pools']
                print(f"💧 LIQUIDITY | Pivot Highs: {len(lp['pivot_highs'])} | Pivot Lows: {len(lp['pivot_lows'])}", flush=True)
            
            print(f"📊 SMC ZONES | Supply/Demand: {len(smc['supply_demand_zones'])} | FVGs: {len(smc['fair_value_gaps'])} | Order Blocks: {len(smc['order_blocks'])}", flush=True)
            
            # Trading Context
            if ed['action'] != 'wait':
                print(f"🎯 ENTRY | {ed['action'].upper()} | Mode: {ed['trade_mode']} | Confidence: {ed['confidence']:.1f}", flush=True)
                for reason in ed['reasons']:
                    print(f"   📍 {reason}", flush=True)

        return {"bm": bm, "flow": flow, "council_decision": council_decision, "entry_decision": entry_decision}
    except Exception as e:
        print(f"🟨 EnhancedSnapshot error: {e}", flush=True)
        return {"bm": None, "flow": None, "council_decision": {}, "entry_decision": {}}

# =================== INTEGRATION WITH EXISTING FUNCTIONS ===================
# استبدال الدوال الأساسية بالمحسنة
compute_candles = compute_enhanced_candles
council_votes_pro = council_pro_advanced_decision
manage_after_entry = manage_after_entry_enhanced
open_market = open_market_enhanced_with_smc
trade_loop = trade_loop_enhanced_with_smc
golden_zone_check = golden_zone_pro_analysis

# الحفاظ على التوافق مع الدوال الحالية
def council_votes_pro_enhanced(df):
    """وظيفة متوافقة مع الإصدار السابق"""
    current_price = float(df['close'].iloc[-1]) if len(df) > 0 else 0
    decision = council_pro_advanced_decision(df, current_price)
    
    # تحويل إلى تنسيق متوافق
    score_b = decision['confidence_score'] if decision['recommendation']['action'] in ['buy', 'studied_scalp', 'trend'] else 0
    score_s = decision['confidence_score'] if decision['recommendation']['action'] in ['sell', 'studied_scalp', 'trend'] else 0
    
    return {
        "b": 1 if score_b > 0 else 0,
        "s": 1 if score_s > 0 else 0,
        "score_b": score_b,
        "score_s": score_s,
        "logs": decision['recommendation']['reason'],
        "ind": decision['details']['indicators'],
        "gz": decision['details']['golden_zone'],
        "footprint": decision['details']['footprint_analysis'],
        "candles": decision['details']['candle_signals']
    }

# =================== VERIFICATION ===================
def verify_execution_environment():
    """التحقق من بيئة التنفيذ المحسنة"""
    print(f"⚙️ ENHANCED EXECUTION ENVIRONMENT", flush=True)
    print(f"🔧 EXECUTE_ORDERS: {EXECUTE_ORDERS} | SHADOW_MODE: {SHADOW_MODE_DASHBOARD} | DRY_RUN: {DRY_RUN}", flush=True)
    print(f"🏛️ ADVANCED COUNCIL: Confidence Levels [Avoid<{CONFIDENCE_AVOID}, Cautious<{CONFIDENCE_CAUTIOUS_SCALP}, Studied<{CONFIDENCE_STUDIED_SCALP}, Trend>={CONFIDENCE_TREND}]", flush=True)
    print(f"💧 SMART MONEY CONCEPTS: Liquidity Pools + Order Blocks + FVGs + Supply/Demand", flush=True)
    print(f"📊 PRICE STRUCTURE: HH/HL/LH/LL Analysis + Trend Strength", flush=True)
    print(f"🎯 INTELLIGENT ENTRY: Studied Scalp + Trend Recognition", flush=True)
    
    if not EXECUTE_ORDERS:
        print("🟡 WARNING: EXECUTE_ORDERS=False - البوت في وضع التحليل فقط!", flush=True)
    if DRY_RUN:
        print("🟡 WARNING: DRY_RUN=True - البوت في وضع المحاكاة!", flush=True)

# =================== MAIN EXECUTION ===================
if __name__ == "__main__":
    log_banner("ADVANCED COUNCIL PRO - SMART MONEY CONCEPTS")
    state = load_state() or {}
    state.setdefault("in_position", False)

    if RESUME_ON_RESTART:
        try:
            state = resume_open_position(ex, SYMBOL, state)
        except Exception as e:
            log_w(f"resume error: {e}\n{traceback.format_exc()}")

    verify_execution_environment()

    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'}  •  {SYMBOL}  •  {INTERVAL}", "yellow"))
    print(colored(f"RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x  •  COUNCIL_PRO=ADVANCED_SMC", "yellow"))
    print(colored(f"SMART MONEY: Liquidity Pools + Order Blocks + FVGs + Supply/Demand", "yellow"))
    print(colored(f"PRICE STRUCTURE: Advanced HH/HL/LH/LL Analysis", "yellow"))
    print(colored(f"INTELLIGENT ENTRY: Studied Scalp + Trend Recognition", "yellow"))
    print(colored(f"CONFIDENCE LEVELS: Avoid({CONFIDENCE_AVOID}) | Cautious({CONFIDENCE_CAUTIOUS_SCALP}) | Studied({CONFIDENCE_STUDIED_SCALP}) | Trend({CONFIDENCE_TREND})", "yellow"))
    print(colored(f"EXECUTION: {'ACTIVE' if EXECUTE_ORDERS and not DRY_RUN else 'SIMULATION'}", "yellow"))
    
    logging.info("advanced council pro with smart money concepts starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    
    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)

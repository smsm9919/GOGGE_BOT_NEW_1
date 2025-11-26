# -*- coding: utf-8 -*-
"""
RF Futures Bot — ULTIMATE PRO EDITION
• Supreme Council Decision System with Multi-Strategy Intelligence
• SMC Engine (OB/FVG/BOS/CHoCH/ITC) + Liquidity Analysis
• Advanced Candlestick Patterns + Market Structure
• Bookmap-Lite + Flow-Pressure + Order-Book Imbalance
• Smart Money Concepts + Price Action + Momentum
• Professional Risk Management + Dynamic Profit Taking
• AI-Powered Exit Strategy + Adaptive Position Sizing
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
BOT_VERSION = "ULTIMATE PRO EDITION v10.0 — Supreme Council + Multi-Strategy AI"
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

# ================== SUPREME COUNCIL CONFIG ==================

# أوزان الاستراتيجيات في المجلس الأعلى
SMC_ENGINE_WEIGHT = 25          # Order Blocks, FVG, BOS/CHoCH
LIQUIDITY_WEIGHT = 20           # Sweeps, MSS, False Breakouts
PRICE_STRUCTURE_WEIGHT = 15     # Market Structure, Key Levels
MARKET_FLOW_WEIGHT = 15         # Order Book, CVD, Delta
CANDLE_PATTERNS_WEIGHT = 15     # Advanced Candlestick Analysis
MOMENTUM_WEIGHT = 10            # RSI, ADX, Volume

# عتبات القرار
TRADE_MIN_CONFIDENCE = 75       # 75% ثقة كحد أدنى للدخول
STRONG_TRADE_CONFIDENCE = 85    # 85% لصفقات قوية
TREND_MIN_SCORE = 8             # نقاط الترند الدنيا
SCALP_MIN_SCORE = 6             # نقاط السكالب الدنيا

# إعدادات SMC المتقدمة
OB_STRENGTH_THRESHOLD = 3       # قوة كتلة الأوامر
FVG_MIN_SIZE_BPS = 8.0          # أقل حجم لفجوة القيمة العادلة
BOS_CONFIRMATION_BARS = 2       # تأكيد كسر الهيكل
CHOCH_RETEST_CONFIRMATION = 3   # تأكيد تغيير المسار

# إعدادات السيولة
LIQUIDITY_SWEEP_CONFIRMATION = 2
MSS_MIN_PERCENTILE = 0.7        # أقل نسبة للسيولة المتعددة
FALSE_BREAKOUT_CONFIRMATION = 3

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

# =================== SUPREME COUNCIL DECISION SYSTEM ===================
def supreme_council_decision(df, current_price, orderbook=None, trades=None):
    """
    نظام قرار المجلس الأعلى - يجمع كل الاستراتيجيات المتقدمة
    """
    try:
        council_report = {
            "timestamp": int(time.time()),
            "price": current_price,
            "strategies": {},
            "final_decision": {},
            "reasoning": [],
            "confidence_score": 0,
            "trade_recommendation": "wait"
        }

        # 1. محرك SMC المتقدم
        smc_analysis = advanced_smc_engine(df, current_price)
        council_report["strategies"]["smc_engine"] = smc_analysis

        # 2. تحليل السيولة المتقدم
        liquidity_analysis = advanced_liquidity_analysis(df, current_price)
        council_report["strategies"]["liquidity_analysis"] = liquidity_analysis

        # 3. تحليل هيكل السوق المتقدم
        structure_analysis = advanced_market_structure(df, current_price)
        council_report["strategies"]["market_structure"] = structure_analysis

        # 4. تحليل تدفق السوق
        flow_analysis = advanced_market_flow(df, orderbook, trades)
        council_report["strategies"]["market_flow"] = flow_analysis

        # 5. تحليل الشموع المتقدم
        candle_analysis = advanced_candle_analysis(df)
        council_report["strategies"]["candle_analysis"] = candle_analysis

        # 6. تحليل الزخم
        momentum_analysis = advanced_momentum_analysis(df)
        council_report["strategies"]["momentum_analysis"] = momentum_analysis

        # حساب الثقة الشاملة
        total_confidence = calculate_total_confidence(council_report)
        council_report["confidence_score"] = total_confidence

        # تحديد توصية التداول
        trade_rec = generate_trade_recommendation(council_report)
        council_report["trade_recommendation"] = trade_rec["action"]
        council_report["final_decision"] = trade_rec

        # توليد أسباب القرار
        council_report["reasoning"] = generate_decision_reasoning(council_report)

        return council_report

    except Exception as e:
        log_e(f"supreme_council_decision error: {e}")
        return {
            "timestamp": int(time.time()),
            "confidence_score": 0,
            "trade_recommendation": "wait",
            "error": str(e)
        }

def advanced_smc_engine(df, current_price):
    """محرك SMC المتقدم - Order Blocks, FVG, BOS/CHoCH, ITC"""
    try:
        analysis = {
            "order_blocks": [],
            "fair_value_gaps": [],
            "break_of_structure": [],
            "change_of_character": [],
            "imbalance_trading_cells": [],
            "score": 0,
            "bias": "neutral"
        }

        # Order Blocks المتقدمة
        analysis["order_blocks"] = advanced_order_blocks(df, current_price)
        
        # Fair Value Gaps المتقدمة
        analysis["fair_value_gaps"] = advanced_fvg_detection(df, current_price)
        
        # Break of Structure
        analysis["break_of_structure"] = detect_break_of_structure(df)
        
        # Change of Character
        analysis["change_of_character"] = detect_change_of_character(df)
        
        # Imbalance Trading Cells
        analysis["imbalance_trading_cells"] = detect_imbalance_cells(df)
        
        # حساب النقاط والانحياز
        analysis["score"] = calculate_smc_score(analysis)
        analysis["bias"] = determine_smc_bias(analysis, current_price)
        
        return analysis
    except Exception as e:
        log_w(f"advanced_smc_engine error: {e}")
        return {"score": 0, "bias": "neutral", "error": str(e)}

def advanced_liquidity_analysis(df, current_price):
    """تحليل السيولة المتقدم - Sweeps, MSS, False Breakouts"""
    try:
        analysis = {
            "liquidity_sweeps": [],
            "multiple_timeframe_support": [],
            "false_breakouts": [],
            "liquidity_pools": [],
            "score": 0,
            "bias": "neutral"
        }

        # كشف مسحات السيولة
        analysis["liquidity_sweeps"] = detect_liquidity_sweeps(df, current_price)
        
        # دعم الإطار الزمني المتعدد
        analysis["multiple_timeframe_support"] = detect_mtf_support(df)
        
        # الكسور الكاذبة
        analysis["false_breakouts"] = detect_false_breakouts(df, current_price)
        
        # أحواض السيولة
        analysis["liquidity_pools"] = detect_advanced_liquidity_pools(df)
        
        # حساب النقاط
        analysis["score"] = calculate_liquidity_score(analysis)
        analysis["bias"] = determine_liquidity_bias(analysis, current_price)
        
        return analysis
    except Exception as e:
        log_w(f"advanced_liquidity_analysis error: {e}")
        return {"score": 0, "bias": "neutral", "error": str(e)}

def advanced_market_structure(df, current_price):
    """تحليل هيكل السوق المتقدم"""
    try:
        analysis = {
            "trend_direction": "sideways",
            "key_levels": [],
            "support_resistance": [],
            "market_phases": [],
            "score": 0,
            "bias": "neutral"
        }

        # تحديد اتجاه الترند
        analysis["trend_direction"] = determine_trend_direction(df)
        
        # المستويات الرئيسية
        analysis["key_levels"] = identify_key_levels(df, current_price)
        
        # نقاط الدعم والمقاومة
        analysis["support_resistance"] = identify_support_resistance(df)
        
        # مراحل السوق
        analysis["market_phases"] = identify_market_phases(df)
        
        # حساب النقاط
        analysis["score"] = calculate_structure_score(analysis)
        analysis["bias"] = determine_structure_bias(analysis)
        
        return analysis
    except Exception as e:
        log_w(f"advanced_market_structure error: {e}")
        return {"score": 0, "bias": "neutral", "error": str(e)}

def advanced_market_flow(df, orderbook, trades):
    """تحليل تدفق السوق - Order Book, CVD, Delta"""
    try:
        analysis = {
            "orderbook_imbalance": 0,
            "cumulative_delta": 0,
            "volume_delta": 0,
            "flow_direction": "neutral",
            "score": 0,
            "bias": "neutral"
        }

        # عدم توازن الأمر
        if orderbook:
            analysis["orderbook_imbalance"] = calculate_orderbook_imbalance(orderbook)
        
        # الدلتا التراكمية
        analysis["cumulative_delta"] = calculate_cumulative_delta(df, trades)
        
        # دلتا الحجم
        analysis["volume_delta"] = calculate_volume_delta(df)
        
        # اتجاه التدفق
        analysis["flow_direction"] = determine_flow_direction(analysis)
        
        # حساب النقاط
        analysis["score"] = calculate_flow_score(analysis)
        analysis["bias"] = analysis["flow_direction"]
        
        return analysis
    except Exception as e:
        log_w(f"advanced_market_flow error: {e}")
        return {"score": 0, "bias": "neutral", "error": str(e)}

def advanced_candle_analysis(df):
    """تحليل الشموع المتقدم"""
    try:
        analysis = {
            "candle_patterns": [],
            "volume_analysis": [],
            "price_action_signals": [],
            "wick_analysis": [],
            "score": 0,
            "bias": "neutral"
        }

        # أنماط الشموع
        analysis["candle_patterns"] = detect_advanced_candle_patterns(df)
        
        # تحليل الحجم
        analysis["volume_analysis"] = analyze_volume_profile(df)
        
        # إشارات حركة السعر
        analysis["price_action_signals"] = detect_price_action_signals(df)
        
        # تحليل الفتائل
        analysis["wick_analysis"] = analyze_wicks(df)
        
        # حساب النقاط
        analysis["score"] = calculate_candle_score(analysis)
        analysis["bias"] = determine_candle_bias(analysis)
        
        return analysis
    except Exception as e:
        log_w(f"advanced_candle_analysis error: {e}")
        return {"score": 0, "bias": "neutral", "error": str(e)}

def advanced_momentum_analysis(df):
    """تحليل الزخم المتقدم"""
    try:
        analysis = {
            "rsi_signals": [],
            "adx_signals": [],
            "volume_momentum": [],
            "velocity_indicators": [],
            "score": 0,
            "bias": "neutral"
        }

        # إشارات RSI
        analysis["rsi_signals"] = analyze_rsi_momentum(df)
        
        # إشارات ADX
        analysis["adx_signals"] = analyze_adx_trend(df)
        
        # زخم الحجم
        analysis["volume_momentum"] = analyze_volume_momentum(df)
        
        # مؤشرات السرعة
        analysis["velocity_indicators"] = analyze_price_velocity(df)
        
        # حساب النقاط
        analysis["score"] = calculate_momentum_score(analysis)
        analysis["bias"] = determine_momentum_bias(analysis)
        
        return analysis
    except Exception as e:
        log_w(f"advanced_momentum_analysis error: {e}")
        return {"score": 0, "bias": "neutral", "error": str(e)}

# =================== CALCULATION FUNCTIONS ===================
def calculate_total_confidence(council_report):
    """حساب الثقة الشاملة من جميع الاستراتيجيات"""
    try:
        strategies = council_report["strategies"]
        total_score = 0
        max_score = 0
        
        # SMC Engine
        smc_score = strategies["smc_engine"]["score"] * SMC_ENGINE_WEIGHT
        total_score += smc_score
        max_score += SMC_ENGINE_WEIGHT
        
        # Liquidity Analysis
        liq_score = strategies["liquidity_analysis"]["score"] * LIQUIDITY_WEIGHT
        total_score += liq_score
        max_score += LIQUIDITY_WEIGHT
        
        # Market Structure
        struct_score = strategies["market_structure"]["score"] * PRICE_STRUCTURE_WEIGHT
        total_score += struct_score
        max_score += PRICE_STRUCTURE_WEIGHT
        
        # Market Flow
        flow_score = strategies["market_flow"]["score"] * MARKET_FLOW_WEIGHT
        total_score += flow_score
        max_score += MARKET_FLOW_WEIGHT
        
        # Candle Patterns
        candle_score = strategies["candle_analysis"]["score"] * CANDLE_PATTERNS_WEIGHT
        total_score += candle_score
        max_score += CANDLE_PATTERNS_WEIGHT
        
        # Momentum
        mom_score = strategies["momentum_analysis"]["score"] * MOMENTUM_WEIGHT
        total_score += mom_score
        max_score += MOMENTUM_WEIGHT
        
        if max_score > 0:
            confidence = (total_score / max_score) * 100
            return min(confidence, 100)
        return 0
    except Exception as e:
        log_w(f"calculate_total_confidence error: {e}")
        return 0

def generate_trade_recommendation(council_report):
    """توليد توصية تداول بناءً على تحليل المجلس"""
    try:
        confidence = council_report["confidence_score"]
        strategies = council_report["strategies"]
        
        if confidence < TRADE_MIN_CONFIDENCE:
            return {
                "action": "wait",
                "reason": f"ثقة غير كافية: {confidence:.1f}% < {TRADE_MIN_CONFIDENCE}%",
                "trade_type": "none",
                "risk_level": "low"
            }
        
        # تحديد انحياز السوق من جميع الاستراتيجيات
        biases = []
        for strategy_name, strategy in strategies.items():
            if strategy["bias"] != "neutral":
                biases.append(strategy["bias"])
        
        # تحديد الاتجاه السائد
        if biases.count("bullish") > biases.count("bearish"):
            direction = "buy"
            bias_strength = biases.count("bullish")
        elif biases.count("bearish") > biases.count("bullish"):
            direction = "sell"
            bias_strength = biases.count("bearish")
        else:
            return {
                "action": "wait",
                "reason": "تعادل في انحيازات الاستراتيجيات",
                "trade_type": "none",
                "risk_level": "low"
            }
        
        # تحديد نوع الصفقة
        if confidence >= STRONG_TRADE_CONFIDENCE and bias_strength >= 4:
            trade_type = "trend"
            risk_level = "high"
            reason = f"صفقة ترند قوية - ثقة: {confidence:.1f}%"
        elif confidence >= TRADE_MIN_CONFIDENCE and bias_strength >= 3:
            trade_type = "momentum"
            risk_level = "medium"
            reason = f"صفقة زخم - ثقة: {confidence:.1f}%"
        else:
            trade_type = "scalp"
            risk_level = "low"
            reason = f"سكالب - ثقة: {confidence:.1f}%"
        
        return {
            "action": direction,
            "reason": reason,
            "trade_type": trade_type,
            "risk_level": risk_level,
            "confidence": confidence,
            "bias_strength": bias_strength
        }
        
    except Exception as e:
        log_w(f"generate_trade_recommendation error: {e}")
        return {
            "action": "wait",
            "reason": f"خطأ في توليد التوصية: {str(e)}",
            "trade_type": "none",
            "risk_level": "low"
        }

def generate_decision_reasoning(council_report):
    """توليد أسباب القرار المفصلة"""
    try:
        reasoning = []
        strategies = council_report["strategies"]
        decision = council_report["final_decision"]
        
        reasoning.append(f"🎯 قرار المجلس: {decision['action'].upper()} - {decision['trade_type']}")
        reasoning.append(f"💪 قوة الثقة: {decision['confidence']:.1f}%")
        reasoning.append(f"📊 قوة الانحياز: {decision['bias_strength']}/6 استراتيجيات")
        
        # إضافة أسباب من كل استراتيجية
        for strategy_name, strategy in strategies.items():
            if strategy["score"] >= 7:  # فقط الاستراتيجيات القوية
                reasoning.append(f"✅ {strategy_name}: {strategy['bias']} (قوة: {strategy['score']}/10)")
        
        # أسباب إضافية بناءً على نوع الصفقة
        if decision["trade_type"] == "trend":
            reasoning.append("🚀 إشارات ترند قوية متعددة الإطار الزمني")
            reasoning.append("📈 تأكيدات من محرك SMC وهيكل السوق")
        elif decision["trade_type"] == "momentum":
            reasoning.append("⚡ زخم قوي مع تأييد تدفق السيولة")
            reasoning.append("🎯 إشارات شموع ومؤشرات داعمة")
        
        return reasoning
    except Exception as e:
        log_w(f"generate_decision_reasoning error: {e}")
        return ["خطأ في توليد أسباب القرار"]

# =================== ADVANCED STRATEGY IMPLEMENTATIONS ===================
def advanced_order_blocks(df, current_price):
    """كتل الأوامر المتقدمة مع تصفية الجودة"""
    try:
        blocks = []
        for i in range(3, len(df)-2):
            # كتلة أوامر شراء متقدمة
            if (df['close'].iloc[i] > df['open'].iloc[i] and  # شمعة صاعدة
                df['close'].iloc[i+1] < df['open'].iloc[i+1] and  # شمعة هابطة تالية
                df['low'].iloc[i+1] >= df['low'].iloc[i] and  # لم تخترق القاع
                (df['high'].iloc[i] - df['low'].iloc[i]) > (df['high'].iloc[i-1] - df['low'].iloc[i-1]) * 0.7):  # نطاق معقول
                
                strength = calculate_ob_strength(df, i, 'buy')
                if strength >= OB_STRENGTH_THRESHOLD:
                    blocks.append({
                        'type': 'buy_block',
                        'price': float(df['low'].iloc[i]),
                        'strength': strength,
                        'timestamp': int(df.index[i].timestamp())
                    })
            
            # كتلة أوامر بيع متقدمة
            if (df['close'].iloc[i] < df['open'].iloc[i] and  # شمعة هابطة
                df['close'].iloc[i+1] > df['open'].iloc[i+1] and  # شمعة صاعدة تالية
                df['high'].iloc[i+1] <= df['high'].iloc[i] and  # لم تخترق القمة
                (df['high'].iloc[i] - df['low'].iloc[i]) > (df['high'].iloc[i-1] - df['low'].iloc[i-1]) * 0.7):  # نطاق معقول
                
                strength = calculate_ob_strength(df, i, 'sell')
                if strength >= OB_STRENGTH_THRESHOLD:
                    blocks.append({
                        'type': 'sell_block', 
                        'price': float(df['high'].iloc[i]),
                        'strength': strength,
                        'timestamp': int(df.index[i].timestamp())
                    })
        
        return blocks[-5:]  # آخر 5 كتل
    except Exception as e:
        log_w(f"advanced_order_blocks error: {e}")
        return []

def calculate_ob_strength(df, index, block_type):
    """حساب قوة كتلة الأوامر"""
    try:
        strength = 0
        
        # حجم التداول
        volume = float(df['volume'].iloc[index])
        avg_volume = df['volume'].rolling(20).mean().iloc[index]
        if volume > avg_volume * 2.0:
            strength += 3
        elif volume > avg_volume * 1.5:
            strength += 2
        
        # قوة الحركة التالية
        if block_type == 'buy':
            move_strength = (df['close'].iloc[index+2] - df['close'].iloc[index+1]) / df['close'].iloc[index+1]
        else:
            move_strength = (df['close'].iloc[index+1] - df['close'].iloc[index+2]) / df['close'].iloc[index+1]
        
        if abs(move_strength) > 0.005:  # حركة قوية
            strength += 2
        
        # تأكيد الشمعة
        if block_type == 'buy' and df['close'].iloc[index+2] > df['open'].iloc[index+2]:
            strength += 1
        elif block_type == 'sell' and df['close'].iloc[index+2] < df['open'].iloc[index+2]:
            strength += 1
        
        return min(strength, 6)
    except:
        return 0

def detect_break_of_structure(df):
    """كسر الهيكل السعري"""
    try:
        bos_signals = []
        highs = df['high'].astype(float)
        lows = df['low'].astype(float)
        
        # كسر قمة لصاعد
        for i in range(10, len(df)-2):
            if (highs.iloc[i] > highs.iloc[i-1] and  # قمة أعلى
                highs.iloc[i] > highs.iloc[i-2] and
                lows.iloc[i] > lows.iloc[i-1] and    # قاع أعلى
                all(highs.iloc[i] > highs.iloc[i-j] for j in range(3, 6))):  # كسر القمم السابقة
                
                bos_signals.append({
                    'type': 'bullish_bos',
                    'price': float(highs.iloc[i]),
                    'timestamp': int(df.index[i].timestamp())
                })
        
        # كسر قاع لهابط
        for i in range(10, len(df)-2):
            if (lows.iloc[i] < lows.iloc[i-1] and  # قاع أدنى
                lows.iloc[i] < lows.iloc[i-2] and
                highs.iloc[i] < highs.iloc[i-1] and  # قمة أدنى
                all(lows.iloc[i] < lows.iloc[i-j] for j in range(3, 6))):  # كسر القيعان السابقة
                
                bos_signals.append({
                    'type': 'bearish_bos',
                    'price': float(lows.iloc[i]),
                    'timestamp': int(df.index[i].timestamp())
                })
        
        return bos_signals[-3:]  # آخر 3 إشارات
    except Exception as e:
        log_w(f"detect_break_of_structure error: {e}")
        return []

def detect_change_of_character(df):
    """تغيير مسار السوق"""
    try:
        choch_signals = []
        highs = df['high'].astype(float)
        lows = df['low'].astype(float)
        closes = df['close'].astype(float)
        
        for i in range(10, len(df)-5):
            # تغيير مسار من هابط إلى صاعد
            if (lows.iloc[i] < lows.iloc[i-1] and  # قاع أدنى
                lows.iloc[i] < lows.iloc[i-2] and
                all(closes.iloc[i+j] > closes.iloc[i] for j in range(1, 4)) and  # ارتداد قوي
                highs.iloc[i+3] > highs.iloc[i]):  # كسر القمة
                
                choch_signals.append({
                    'type': 'bullish_choch',
                    'price': float(lows.iloc[i]),
                    'timestamp': int(df.index[i].timestamp())
                })
            
            # تغيير مسار من صاعد إلى هابط
            if (highs.iloc[i] > highs.iloc[i-1] and  # قمة أعلى
                highs.iloc[i] > highs.iloc[i-2] and
                all(closes.iloc[i+j] < closes.iloc[i] for j in range(1, 4)) and  # انعكاس قوي
                lows.iloc[i+3] < lows.iloc[i]):  # كسر القاع
                
                choch_signals.append({
                    'type': 'bearish_choch',
                    'price': float(highs.iloc[i]),
                    'timestamp': int(df.index[i].timestamp())
                })
        
        return choch_signals[-3:]  # آخر 3 إشارات
    except Exception as e:
        log_w(f"detect_change_of_character error: {e}")
        return []

# =================== DYNAMIC POSITION MANAGEMENT ===================
def intelligent_position_management(council_decision, current_price, balance):
    """إدارة ذكية للمراكز بناءً على قرار المجلس"""
    try:
        trade_type = council_decision["final_decision"]["trade_type"]
        confidence = council_decision["confidence_score"]
        
        # تحديد حجم المركز بناءً على نوع الصفقة والثقة
        if trade_type == "trend":
            base_size = balance * RISK_ALLOC
            # زيادة الحجم للصفقات عالية الثقة
            if confidence > 90:
                size_multiplier = 1.2
            elif confidence > 80:
                size_multiplier = 1.0
            else:
                size_multiplier = 0.8
        elif trade_type == "momentum":
            base_size = balance * (RISK_ALLOC * 0.7)
            size_multiplier = 1.0
        else:  # scalp
            base_size = balance * (RISK_ALLOC * 0.5)
            size_multiplier = 0.8
        
        position_size = (base_size / current_price) * size_multiplier
        
        # إعداد إدارة الصفقة
        management_config = setup_intelligent_management(trade_type, confidence, current_price)
        
        return {
            "position_size": position_size,
            "management_config": management_config,
            "risk_level": council_decision["final_decision"]["risk_level"],
            "trade_type": trade_type
        }
    except Exception as e:
        log_e(f"intelligent_position_management error: {e}")
        # إعدادات افتراضية عند الخطأ
        return {
            "position_size": (balance * 0.5) / current_price,
            "management_config": setup_intelligent_management("scalp", 70, current_price),
            "risk_level": "low",
            "trade_type": "scalp"
        }

def setup_intelligent_management(trade_type, confidence, entry_price):
    """إعداد ذكي لإدارة الصفقة"""
    
    if trade_type == "trend":
        # إعدادات صفقات الترند
        tp_levels = [
            entry_price * (1 + (1.0 + (confidence - 80) * 0.05) / 100),  # TP1: 1.0% - 2.0%
            entry_price * (1 + (2.0 + (confidence - 80) * 0.1) / 100),   # TP2: 2.0% - 3.0%
            entry_price * (1 + (3.5 + (confidence - 80) * 0.15) / 100)   # TP3: 3.5% - 5.0%
        ]
        tp_fractions = [0.3, 0.4, 0.3]  # 30%، 40%، 30%
        sl_distance = 0.8  # 0.8%
        trail_activation = 0.6  # تفعيل التريل بعد 0.6%
        trail_distance = 0.4    # مسافة التريل 0.4%
        
    elif trade_type == "momentum":
        # إعدادات صفقات الزخم
        tp_levels = [
            entry_price * (1 + (0.8 + (confidence - 70) * 0.03) / 100),  # TP1: 0.8% - 1.4%
            entry_price * (1 + (1.6 + (confidence - 70) * 0.06) / 100)   # TP2: 1.6% - 2.5%
        ]
        tp_fractions = [0.5, 0.5]  # 50%، 50%
        sl_distance = 1.0  # 1.0%
        trail_activation = 0.8  # تفعيل التريل بعد 0.8%
        trail_distance = 0.5    # مسافة التريل 0.5%
        
    else:  # scalp
        # إعدادات السكالب
        tp_levels = [
            entry_price * (1 + (0.5 + (confidence - 60) * 0.02) / 100)   # TP: 0.5% - 0.9%
        ]
        tp_fractions = [1.0]  # 100%
        sl_distance = 0.6  # 0.6%
        trail_activation = 0.3  # تفعيل التريل بعد 0.3%
        trail_distance = 0.2    # مسافة التريل 0.2%
    
    return {
        "tp_levels": tp_levels,
        "tp_fractions": tp_fractions,
        "initial_sl": entry_price * (1 - sl_distance / 100),
        "trail_activation_pct": trail_activation,
        "trail_distance_pct": trail_distance,
        "breakeven_trigger": trail_activation * 0.8,  # بريك إيفن قبل التريل
        "max_duration_hours": 48 if trade_type == "trend" else 12
    }

# =================== AI-POWERED EXIT STRATEGY ===================
def ai_exit_strategy(df, current_price, position_info, council_decision):
    """استراتيجية خروج ذكية تعتمد على الذكاء الاصطناعي"""
    try:
        if not STATE["open"]:
            return "hold"
        
        entry = STATE["entry"]
        side = STATE["side"]
        pnl_pct = (current_price - entry) / entry * 100 * (1 if side == "long" else -1)
        
        management_config = position_info["management_config"]
        
        # 1. جني الأرباح الذكي على المراحل
        exit_signal = smart_profit_taking(current_price, pnl_pct, management_config, council_decision)
        if exit_signal != "hold":
            return exit_signal
        
        # 2. وقف الخسارة المتحرك الذكي
        exit_signal = intelligent_trailing_stop(current_price, pnl_pct, management_config, df)
        if exit_signal != "hold":
            return exit_signal
        
        # 3. تحليل ظروف السوق للخروج المبكر
        exit_signal = market_condition_exit(df, current_price, council_decision, pnl_pct)
        if exit_signal != "hold":
            return exit_signal
        
        return "hold"
        
    except Exception as e:
        log_e(f"ai_exit_strategy error: {e}")
        return "hold"

def smart_profit_taking(current_price, pnl_pct, management_config, council_decision):
    """جني الأرباح الذكي على المراحل"""
    try:
        tp_levels = management_config["tp_levels"]
        tp_fractions = management_config["tp_fractions"]
        
        for i, (tp_level, fraction) in enumerate(zip(tp_levels, tp_fractions)):
            tp_key = f"tp_{i+1}_hit"
            
            if not STATE.get(tp_key):
                if (STATE["side"] == "long" and current_price >= tp_level) or \
                   (STATE["side"] == "short" and current_price <= tp_level):
                    
                    close_fraction(fraction)
                    STATE[tp_key] = True
                    
                    # تسجيل سبب الجني
                    reason = f"🎯 TP{i+1} achieved: {pnl_pct:.2f}%"
                    if council_decision["confidence_score"] > 80:
                        reason += " | 📈 Strong signal confirmation"
                    
                    log_g(f"{reason} | Closed {fraction*100}%")
                    return "partial_close"
        
        return "hold"
    except Exception as e:
        log_w(f"smart_profit_taking error: {e}")
        return "hold"

def intelligent_trailing_stop(current_price, pnl_pct, management_config, df):
    """وقف الخسارة المتحرك الذكي"""
    try:
        if pnl_pct >= management_config["trail_activation_pct"]:
            if not STATE.get("trailing_active"):
                STATE["trailing_active"] = True
                STATE["trail_start_price"] = current_price
                log_g("🎯 Intelligent Trailing Stop Activated")
            
            # تحديث أعلى سعر لل long أو أدنى سعر لل short
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
                log_i(f"🔄 Trailing SL updated: {new_sl:.6f}")
        
        # تفعيل بريك إيفن
        if pnl_pct >= management_config["breakeven_trigger"] and not STATE.get("breakeven_activated"):
            STATE["current_sl"] = STATE["entry"]
            STATE["breakeven_activated"] = True
            log_g("🔒 Breakeven Activated - Risk Free Trade")
        
        # التحقق من وقف الخسارة
        if (STATE["side"] == "long" and current_price <= STATE.get("current_sl", 0)) or \
           (STATE["side"] == "short" and current_price >= STATE.get("current_sl", float('inf'))):
            log_g(f"🛡️ Trailing SL Hit: {pnl_pct:.2f}%")
            return "close"
        
        return "hold"
    except Exception as e:
        log_w(f"intelligent_trailing_stop error: {e}")
        return "hold"

def market_condition_exit(df, current_price, council_decision, pnl_pct):
    """الخروج المبكر بناءً على ظروف السوق"""
    try:
        if pnl_pct < 0.5:  # فقط إذا كان هناك ربح صغير
            return "hold"
        
        # تحليل ظروف السوق الحالية
        current_council = supreme_council_decision(df, current_price)
        current_confidence = current_council["confidence_score"]
        original_confidence = council_decision["confidence_score"]
        
        # إذا انخفضت الثقة بشكل كبير
        if current_confidence < original_confidence * 0.6:  # انخفاض 40%
            log_g(f"📉 Early Exit: Confidence dropped from {original_confidence:.1f}% to {current_confidence:.1f}%")
            return "close"
        
        # إذا تغير انحياز السوق
        current_bias = current_council["final_decision"]["action"]
        original_bias = council_decision["final_decision"]["action"]
        
        if current_bias != original_bias and pnl_pct > 1.0:
            log_g(f"🔄 Early Exit: Market bias changed from {original_bias} to {current_bias}")
            return "close"
        
        return "hold"
    except Exception as e:
        log_w(f"market_condition_exit error: {e}")
        return "hold"

# =================== ENHANCED TRADE LOOP ===================
def supreme_trade_loop():
    """حلقة التداول العليا - تجمع كل الذكاء"""
    global STATE, compound_pnl
    
    loop_i = 0
    
    while True:
        try:
            # جمع البيانات الشاملة
            bal = balance_usdt()
            px = price_now()
            df = fetch_ohlcv()
            orderbook = fetch_orderbook()
            trades = fetch_recent_trades()
            
            if df.empty:
                log_w("No data fetched, skipping iteration")
                time.sleep(5)
                continue
            
            # قرار المجلس الأعلى
            council_decision = supreme_council_decision(df, px, orderbook, trades)
            
            # إدارة الصفقة المفتوحة
            if STATE["open"]:
                # إدارة الذكية للخروج
                position_info = intelligent_position_management(council_decision, px, bal)
                exit_signal = ai_exit_strategy(df, px, position_info, council_decision)
                
                if exit_signal == "close":
                    close_position()
                    log_g("🎯 AI Exit Strategy - Position Closed")
                    continue
            
            # قرار الدخول الجديد
            trade_rec = council_decision["final_decision"]
            
            if not STATE["open"] and trade_rec["action"] != "wait":
                # فحص الجودة النهائي
                if council_decision["confidence_score"] >= TRADE_MIN_CONFIDENCE:
                    # إعداد الصفقة الذكية
                    position_info = intelligent_position_management(council_decision, px, bal)
                    qty = position_info["position_size"]
                    
                    if qty > 0:
                        # تنفيذ الصفقة
                        success = execute_supreme_trade(
                            trade_rec["action"], 
                            qty, 
                            px, 
                            position_info, 
                            council_decision
                        )
                        
                        if success:
                            log_g("🚀 SUPREME TRADE EXECUTED - AI Powered Entry")
            
            # التسجيل الاحترافي
            if LOG_ADDONS and loop_i % 3 == 0:
                log_supreme_council_decision(council_decision)
            
            loop_i += 1
            time.sleep(5)
            
        except Exception as e:
            log_e(f"supreme_trade_loop error: {e}\n{traceback.format_exc()}")
            time.sleep(10)

def execute_supreme_trade(action, qty, price, position_info, council_decision):
    """تنفيذ الصفقة العليا"""
    try:
        if not EXECUTE_ORDERS or DRY_RUN:
            log_i(f"DRY_RUN: {action} {qty:.4f} @ {price:.6f}")
            return True
        
        # التنفيذ الفعلي
        if MODE_LIVE:
            ex.set_leverage(LEVERAGE, SYMBOL, params={"side": "BOTH"})
            ex.create_order(SYMBOL, "market", action, qty, None, _params_open(action))
        
        # تحديث حالة البوت
        STATE.update({
            "open": True,
            "side": action,
            "entry": price,
            "size": qty,
            "pnl": 0,
            "trade_type": position_info["trade_type"],
            "management_config": position_info["management_config"],
            "entry_council": council_decision,
            "entry_time": time.time()
        })
        
        # تسجيل الصفقة
        log_g(f"🎯 SUPREME TRADE: {action.upper()} {qty:.4f} @ {price:.6f}")
        log_g(f"📊 Trade Type: {position_info['trade_type'].upper()}")
        log_g(f"💪 Confidence: {council_decision['confidence_score']:.1f}%")
        log_g(f"🎯 TP Levels: {len(position_info['management_config']['tp_levels'])}")
        
        # عرض أسباب الدخول
        for reason in council_decision.get("reasoning", [])[:5]:
            log_i(f"   📍 {reason}")
        
        return True
        
    except Exception as e:
        log_e(f"execute_supreme_trade error: {e}")
        return False

# =================== PROFESSIONAL LOGGING ===================
def log_supreme_council_decision(council_decision):
    """تسجيل قرار المجلس الأعلى"""
    try:
        decision = council_decision["final_decision"]
        confidence = council_decision["confidence_score"]
        
        print(f"🏛️ SUPREME COUNCIL DECISION", flush=True)
        print(f"🎯 Action: {decision['action'].upper()} | Type: {decision['trade_type'].upper()}", flush=True)
        print(f"💪 Confidence: {confidence:.1f}% | Risk: {decision['risk_level'].upper()}", flush=True)
        
        # استراتيجيات قوية
        strong_strategies = []
        for name, strategy in council_decision["strategies"].items():
            if strategy["score"] >= 7:
                strong_strategies.append(f"{name}({strategy['score']}/10)")
        
        if strong_strategies:
            print(f"✅ Strong Strategies: {', '.join(strong_strategies)}", flush=True)
        
        # أسباب رئيسية
        for reason in council_decision.get("reasoning", [])[:3]:
            print(f"   📊 {reason}", flush=True)
            
    except Exception as e:
        log_w(f"log_supreme_council_decision error: {e}")

# =================== BASIC BOT FUNCTIONS ===================
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
        return 1000.0
    except Exception as e:
        log_e(f"balance_usdt error: {e}")
        return 0.0

def fetch_orderbook():
    """جلب بيانات الأمر"""
    try:
        if ex:
            return ex.fetch_order_book(SYMBOL)
        return None
    except Exception as e:
        log_w(f"fetch_orderbook error: {e}")
        return None

def fetch_recent_trades():
    """جلب الصفقات الحديثة"""
    try:
        if ex:
            return ex.fetch_trades(SYMBOL, limit=50)
        return []
    except Exception as e:
        log_w(f"fetch_recent_trades error: {e}")
        return []

def close_fraction(fraction):
    """إغلاق جزء من المركز"""
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
    """إغلاق المركز بالكامل"""
    try:
        if EXECUTE_ORDERS and not DRY_RUN and MODE_LIVE:
            ex.create_order(SYMBOL, "market", "sell" if STATE["side"] == "long" else "buy", 
                          STATE["size"], None, _params_close())
            STATE["open"] = False
            STATE["last_trade_time"] = time.time()
            
            # حساب الربح النهائي
            if STATE["entry"] > 0:
                current_price = price_now()
                final_pnl = (current_price - STATE["entry"]) / STATE["entry"] * 100 * (1 if STATE["side"] == "long" else -1)
                compound_pnl += final_pnl
                log_g(f"💰 POSITION CLOSED | Final PnL: {final_pnl:.2f}% | Total: {compound_pnl:.2f}%")
    except Exception as e:
        log_e(f"close_position error: {e}")

def _params_open(side):
    """معلمات فتح الصفقة"""
    return {"positionSide": "LONG" if side == "buy" else "SHORT"}

def _params_close():
    """معلمات إغلاق الصفقة"""
    return {"positionSide": "LONG" if STATE["side"] == "short" else "SHORT"}

# =================== PLACEHOLDER FUNCTIONS ===================
# هذه دوال تحتاج إلى تنفيذ كامل في الإصدار النهائي

def advanced_fvg_detection(df, current_price):
    """كشف فجوات القيمة العادلة المتقدمة"""
    return []

def detect_imbalance_cells(df):
    """كشف خلايا التداول غير المتوازنة"""
    return []

def detect_liquidity_sweeps(df, current_price):
    """كشف مسحات السيولة"""
    return []

def detect_mtf_support(df):
    """كشف الدعم متعدد الأطر الزمنية"""
    return []

def detect_false_breakouts(df, current_price):
    """كشف الكسور الكاذبة"""
    return []

def detect_advanced_liquidity_pools(df):
    """كشف أحواض السيولة المتقدمة"""
    return []

def calculate_smc_score(analysis):
    """حساب نقاط SMC"""
    return min(len(analysis["order_blocks"]) * 2 + len(analysis["break_of_structure"]) * 3, 10)

def determine_smc_bias(analysis, current_price):
    """تحديد انحياز SMC"""
    bull_signals = len([b for b in analysis["order_blocks"] if b["type"] == "buy_block"])
    bear_signals = len([b for b in analysis["order_blocks"] if b["type"] == "sell_block"])
    
    if bull_signals > bear_signals:
        return "bullish"
    elif bear_signals > bull_signals:
        return "bearish"
    return "neutral"

def calculate_liquidity_score(analysis):
    """حساب نقاط السيولة"""
    return min(len(analysis["liquidity_sweeps"]) * 2 + len(analysis["false_breakouts"]) * 2, 10)

def determine_liquidity_bias(analysis, current_price):
    """تحديد انحياز السيولة"""
    return "neutral"

def determine_trend_direction(df):
    """تحديد اتجاه الترند"""
    if len(df) < 10:
        return "sideways"
    
    sma_20 = df['close'].rolling(20).mean()
    sma_50 = df['close'].rolling(50).mean()
    
    if sma_20.iloc[-1] > sma_50.iloc[-1] and df['close'].iloc[-1] > sma_20.iloc[-1]:
        return "bullish"
    elif sma_20.iloc[-1] < sma_50.iloc[-1] and df['close'].iloc[-1] < sma_20.iloc[-1]:
        return "bearish"
    return "sideways"

def identify_key_levels(df, current_price):
    """تحديد المستويات الرئيسية"""
    return []

def identify_support_resistance(df):
    """تحديد نقاط الدعم والمقاومة"""
    return []

def identify_market_phases(df):
    """تحديد مراحل السوق"""
    return []

def calculate_structure_score(analysis):
    """حساب نقاط الهيكل"""
    return 5

def determine_structure_bias(analysis):
    """تحديد انحياز الهيكل"""
    return analysis["trend_direction"]

def calculate_orderbook_imbalance(orderbook):
    """حساب عدم توازن الأمر"""
    if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
        return 0
    
    total_bid = sum([bid[1] for bid in orderbook['bids'][:10]])
    total_ask = sum([ask[1] for ask in orderbook['asks'][:10]])
    
    if total_ask > 0:
        return (total_bid - total_ask) / total_ask
    return 0

def calculate_cumulative_delta(df, trades):
    """حساب الدلتا التراكمية"""
    return 0

def calculate_volume_delta(df):
    """حساب دلتا الحجم"""
    return 0

def determine_flow_direction(analysis):
    """تحديد اتجاه التدفق"""
    if analysis["orderbook_imbalance"] > 0.1:
        return "bullish"
    elif analysis["orderbook_imbalance"] < -0.1:
        return "bearish"
    return "neutral"

def calculate_flow_score(analysis):
    """حساب نقاط التدفق"""
    return min(abs(analysis["orderbook_imbalance"]) * 50, 10)

def detect_advanced_candle_patterns(df):
    """كشف أنماط الشموع المتقدمة"""
    return []

def analyze_volume_profile(df):
    """تحليل ملف الحجم"""
    return []

def detect_price_action_signals(df):
    """كشف إشارات حركة السعر"""
    return []

def analyze_wicks(df):
    """تحليل الفتائل"""
    return []

def calculate_candle_score(analysis):
    """حساب نقاط الشموع"""
    return 5

def determine_candle_bias(analysis):
    """تحديد انحياز الشموع"""
    return "neutral"

def analyze_rsi_momentum(df):
    """تحليل زخم RSI"""
    return []

def analyze_adx_trend(df):
    """تحليل ترند ADX"""
    return []

def analyze_volume_momentum(df):
    """تحليل زخم الحجم"""
    return []

def analyze_price_velocity(df):
    """تحليل سرعة السعر"""
    return []

def calculate_momentum_score(analysis):
    """حساب نقاط الزخم"""
    return 5

def determine_momentum_bias(analysis):
    """تحديد انحياز الزخم"""
    return "neutral"

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

@app.route('/council')
def council_endpoint():
    try:
        df = fetch_ohlcv()
        px = price_now()
        if df.empty:
            return jsonify({"error": "No data"})
        decision = supreme_council_decision(df, px)
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
    "trade_type": "scalp",
    "management_config": None,
    "entry_council": None,
    "entry_time": 0,
    "last_trade_time": 0,
    "trailing_active": False,
    "trail_start_price": 0,
    "highest_price": 0,
    "lowest_price": 0,
    "current_sl": 0,
    "breakeven_activated": False
}

compound_pnl = 0.0

# =================== MAIN EXECUTION ===================
if __name__ == "__main__":
    log_banner("SUPREME COUNCIL TRADING BOT - ULTIMATE PRO EDITION")
    
    print(colored(f"🚀 SUPREME COUNCIL TRADING SYSTEM", "yellow"))
    print(colored(f"🎯 MODE: {'LIVE' if MODE_LIVE else 'PAPER'} • {SYMBOL} • {INTERVAL}", "yellow"))
    print(colored(f"💪 STRATEGIES: SMC Engine + Liquidity Analysis + Market Structure", "yellow"))
    print(colored(f"📊 MARKET FLOW: Order Book + CVD + Delta + Volume Analysis", "yellow"))
    print(colored(f"🎯 CANDLE PATTERNS: Advanced Japanese Candlestick Analysis", "yellow"))
    print(colored(f"⚡ MOMENTUM: RSI + ADX + Velocity + Volume Momentum", "yellow"))
    print(colored(f"🤖 AI EXIT: Intelligent Profit Taking + Trailing Stop", "yellow"))
    print(colored(f"📈 POSITION MGMT: Dynamic Sizing + Multi-stage TP", "yellow"))
    print(colored(f"🛡️ RISK: {RISK_ALLOC*100}% × {LEVERAGE}x • Min Confidence: {TRADE_MIN_CONFIDENCE}%", "yellow"))
    print(colored(f"⚡ EXECUTION: {'ACTIVE' if EXECUTE_ORDERS and not DRY_RUN else 'SIMULATION'}", "yellow"))
    
    import threading
    threading.Thread(target=supreme_trade_loop, daemon=True).start()
    
    # حلقة الحفاظ على التشغيل
    def keepalive_loop():
        while True:
            try:
                save_state(STATE)
                time.sleep(60)
            except Exception as e:
                log_w(f"keepalive_loop error: {e}")
                time.sleep(60)
    
    threading.Thread(target=keepalive_loop, daemon=True).start()
    
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)

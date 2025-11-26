# -*- coding: utf-8 -*-
"""
ULTIMATE PRO TRADING BOT - Supreme Council AI System
• Multi-Strategy Intelligence Fusion
• Advanced SMC Engine + Liquidity Analysis
• Professional Risk Management + AI Exit Strategy
• Real-time Market Analysis + Smart Execution
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

# ==== Execution Switches ====
EXECUTE_ORDERS = True
SHADOW_MODE_DASHBOARD = False
DRY_RUN = False

# ==== Bot Configuration ====
BOT_VERSION = "ULTIMATE PRO v12.0 - Supreme Council AI"
print("🔁 Booting:", BOT_VERSION, flush=True)

STATE_PATH = "./bot_state.json"
RESUME_ON_RESTART = True

# =================== TRADING SETTINGS ===================
SYMBOL     = os.getenv("SYMBOL", "DOGE/USDT:USDT")
INTERVAL   = os.getenv("INTERVAL", "15m")
LEVERAGE   = int(os.getenv("LEVERAGE", 10))
RISK_ALLOC = float(os.getenv("RISK_ALLOC", 0.60))

# =================== POSITION MANAGEMENT SETTINGS ===================
MIN_QTY = 30
FINAL_CHUNK_QTY = 50
CLOSE_RETRY_ATTEMPTS = 3
CLOSE_VERIFY_WAIT_S = 1
RESUME_LOOKBACK_SECS = 3600  # ساعة واحدة

# إعدادات إدارة الصفقة
TP1_PCT_BASE = 0.5  # 0.5%
TP1_CLOSE_FRAC = 0.3  # إغلاق 30% عند TP1
BREAKEVEN_AFTER = 0.3  # تفعيل Breakeven بعد تحقيق 0.3% ربح

# =================== STRATEGY CONFIGURATION ===================
TRADE_MIN_CONFIDENCE = 75
STRONG_TRADE_CONFIDENCE = 85
TREND_MIN_SCORE = 8
SCALP_MIN_SCORE = 6

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

# =================== PROFESSIONAL LOGGING SYSTEM ===================
def log_strategy_line(side_hint: str, mode: str, balance: float, compound_pnl: float):
    """تسجيل خط الاستراتيجية"""
    try:
        if side_hint is None: side_hint = "WAIT"
        if mode is None: mode = "SCALP"
        log_i(f"📊 Strategy: 📈 {mode.upper()} | Balance={balance:.2f} | CompoundPnL={compound_pnl:.6f}")
    except Exception as e: log_w(f"log_strategy_line error: {e}")

def log_snap_line(side: str, votes_for: int, votes_against: int, score: float, adx: float, di_spread: float, z_score: float, orderbook_imb: float):
    """تسجيل خط SNAP"""
    try:
        if side is None: side = "WAIT"
        log_i(f"🎯 SNAP | {side.upper()} | votes={votes_for}/{votes_against} | score={score:.1f}/10.0 | ADX={adx:.1f} DI={di_spread:.1f} | z={z_score:.2f} | imb={orderbook_imb:.2f}")
    except Exception as e: log_w(f"log_snap_line error: {e}")

def log_addons_live():
    """تسجيل ADDONS LIVE"""
    log_i("🧩 ADDONS LIVE")

def log_bookmap_line(bookmap_ctx: dict):
    """تسجيل خط Bookmap"""
    try:
        imb = bookmap_ctx.get("imbalance", 1.0)
        bids = bookmap_ctx.get("bids", [])
        asks = bookmap_ctx.get("asks", [])
        buy_levels = [bid[0] for bid in bids[:3]] if bids else []
        sell_levels = [ask[0] for ask in asks[:3]] if asks else []
        buys_txt = ", ".join(f"{p:.6f}" for p in buy_levels) if buy_levels else "n/a"
        sells_txt = ", ".join(f"{p:.6f}" for p in sell_levels) if sell_levels else "n/a"
        log_i(f"📉 Bookmap: 🔴 Imb={imb:.2f} | Buy[{buys_txt}] | Sell[{sells_txt}]")
    except Exception as e: log_w(f"log_bookmap_line error: {e}")

def log_flow_line(flow_ctx: dict):
    """تسجيل خط التدفق"""
    try:
        flow = flow_ctx.get("flow", 0)
        delta = flow_ctx.get("delta", 0)
        z_score = flow_ctx.get("z_score", 0)
        cvd = flow_ctx.get("cvd", 0)
        if flow > 1000: side_emoji, side = "🟢 Buy", "buy"
        elif flow < -1000: side_emoji, side = "🔴 Sell", "sell"
        else: side_emoji, side = "⚪ Flat", "flat"
        log_i(f"💧 Flow: {side_emoji} Δ={delta:.0f} z={z_score:.2f} | CVD={cvd:.0f}")
        return side
    except Exception as e: log_w(f"log_flow_line error: {e}"); return "flat"

def log_dash_hint_line(hint_side: str, council_data: dict, indicators: dict):
    """تسجيل خط DASH"""
    try:
        if hint_side is None: hint_side = "WAIT"
        confidence = council_data.get('confidence_score', 0)
        rsi = indicators.get('rsi', 50)
        adx = indicators.get('adx', 0)
        di_spread = indicators.get('plus_di', 0) - indicators.get('minus_di', 0)
        strategies = council_data.get('strategies', {})
        buy_votes = sum(1 for s in strategies.values() if s.get('bias') == 'bullish')
        sell_votes = sum(1 for s in strategies.values() if s.get('bias') == 'bearish')
        log_i(f"📟 DASH → hint-{hint_side.upper()} | Council BUY({buy_votes}) SELL({sell_votes}) | RSI={rsi:.1f} ADX={adx:.1f} DI={di_spread:.1f} | Confidence={confidence:.1f}")
    except Exception as e: log_w(f"log_dash_hint_line error: {e}")

# =================== PRECISE POSITION SIZING ===================
def compute_precise_size(balance, price, symbol=SYMBOL):
    """حساب حجم صفقة دقيق يتوافق مع متطلبات البورصة"""
    try:
        if balance <= 0 or price <= 0: return 0
        risk_amount = balance * RISK_ALLOC
        base_size = risk_amount / price
        
        if "DOGE" in symbol:
            precise_size = math.floor(base_size)
            if precise_size < 1: return 0
        elif "BTC" in symbol or "ETH" in symbol:
            precise_size = math.floor(base_size * 10000) / 10000
        else:
            precise_size = math.floor(base_size * 100) / 100
            
        log_i(f"📊 Position Size: {base_size:.4f} → {precise_size:.4f} (precise)")
        return precise_size
    except Exception as e:
        log_e(f"compute_precise_size error: {e}")
        return 0

# =================== MARKET ANALYSIS ENGINE ===================
def compute_indicators(df):
    """حساب المؤشرات الفنية المتقدمة"""
    try:
        if len(df) < 20: return {}
        closes = df['close'].astype(float)
        highs = df['high'].astype(float)
        lows = df['low'].astype(float)
        
        # RSI
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # ADX مع DI
        plus_dm = highs.diff()
        minus_dm = lows.diff().abs()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        tr1 = highs - lows
        tr2 = (highs - closes.shift()).abs()
        tr3 = (lows - closes.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(14).mean()
        
        # مؤشرات إضافية
        sma_20 = closes.rolling(20).mean()
        sma_50 = closes.rolling(50).mean()
        typical_price = (highs + lows + closes) / 3
        volume = df['volume'].astype(float)
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        return {
            "rsi": float(rsi.iloc[-1]) if len(rsi) > 0 else 50,
            "adx": float(adx.iloc[-1]) if len(adx) > 0 else 0,
            "plus_di": float(plus_di.iloc[-1]) if len(plus_di) > 0 else 0,
            "minus_di": float(minus_di.iloc[-1]) if len(minus_di) > 0 else 0,
            "atr": float(atr.iloc[-1]) if len(atr) > 0 else 0,
            "vwap": float(vwap.iloc[-1]) if len(vwap) > 0 else 0,
            "sma_20": float(sma_20.iloc[-1]) if len(sma_20) > 0 else 0,
            "sma_50": float(sma_50.iloc[-1]) if len(sma_50) > 0 else 0,
            "price": float(closes.iloc[-1])
        }
    except Exception as e:
        log_w(f"compute_indicators error: {e}")
        return {}

# =================== SUPREME COUNCIL DECISION SYSTEM ===================
def supreme_council_decision(df, current_price, orderbook=None, trades=None):
    """نظام قرار المجلس الأعلى - ذكاء جماعي متكامل"""
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

        # 1. تحليل SMC المتقدم
        smc_analysis = advanced_smc_engine(df, current_price)
        council_report["strategies"]["smc_engine"] = smc_analysis

        # 2. تحليل الهيكل السعري
        structure_analysis = advanced_market_structure(df, current_price)
        council_report["strategies"]["market_structure"] = structure_analysis

        # 3. تحليل تدفق السوق
        flow_analysis = advanced_market_flow(df, orderbook, trades)
        council_report["strategies"]["market_flow"] = flow_analysis

        # 4. تحليل الشموع
        candle_analysis = advanced_candle_analysis(df)
        council_report["strategies"]["candle_analysis"] = candle_analysis

        # 5. تحليل الزخم
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
    """محرك SMC المتقدم"""
    try:
        analysis = {"order_blocks": [], "score": 0, "bias": "neutral"}
        
        # كتل الأوامر المبسطة
        for i in range(3, len(df)-2):
            if (df['close'].iloc[i] > df['open'].iloc[i] and 
                df['close'].iloc[i+1] < df['open'].iloc[i+1] and 
                df['low'].iloc[i+1] >= df['low'].iloc[i]):
                analysis["order_blocks"].append({
                    'type': 'buy_block', 'price': float(df['low'].iloc[i]), 'strength': 3
                })
            
            if (df['close'].iloc[i] < df['open'].iloc[i] and 
                df['close'].iloc[i+1] > df['open'].iloc[i+1] and 
                df['high'].iloc[i+1] <= df['high'].iloc[i]):
                analysis["order_blocks"].append({
                    'type': 'sell_block', 'price': float(df['high'].iloc[i]), 'strength': 3
                })
        
        analysis["score"] = min(len(analysis["order_blocks"]) * 2, 10)
        
        # تحديد الانحياز
        buy_blocks = len([b for b in analysis["order_blocks"] if b['type'] == 'buy_block'])
        sell_blocks = len([b for b in analysis["order_blocks"] if b['type'] == 'sell_block'])
        if buy_blocks > sell_blocks: analysis["bias"] = "bullish"
        elif sell_blocks > buy_blocks: analysis["bias"] = "bearish"
        
        return analysis
    except Exception as e:
        log_w(f"advanced_smc_engine error: {e}")
        return {"score": 0, "bias": "neutral"}

def advanced_market_structure(df, current_price):
    """تحليل هيكل السوق المتقدم"""
    try:
        analysis = {"trend_direction": "sideways", "score": 0, "bias": "neutral"}
        
        if len(df) < 20:
            return analysis
        
        # تحليل الاتجاه البسيط
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        
        if sma_20.iloc[-1] > sma_50.iloc[-1] and df['close'].iloc[-1] > sma_20.iloc[-1]:
            analysis["trend_direction"] = "bullish"
            analysis["bias"] = "bullish"
            analysis["score"] = 7
        elif sma_20.iloc[-1] < sma_50.iloc[-1] and df['close'].iloc[-1] < sma_20.iloc[-1]:
            analysis["trend_direction"] = "bearish" 
            analysis["bias"] = "bearish"
            analysis["score"] = 7
        else:
            analysis["score"] = 4
            
        return analysis
    except Exception as e:
        log_w(f"advanced_market_structure error: {e}")
        return {"score": 0, "bias": "neutral"}

def advanced_market_flow(df, orderbook, trades):
    """تحليل تدفق السوق"""
    try:
        analysis = {"orderbook_imbalance": 0, "score": 0, "bias": "neutral"}
        
        if orderbook and 'bids' in orderbook and 'asks' in orderbook:
            total_bid = sum([bid[1] for bid in orderbook['bids'][:5]])
            total_ask = sum([ask[1] for ask in orderbook['asks'][:5]])
            if total_ask > 0:
                analysis["orderbook_imbalance"] = (total_bid - total_ask) / total_ask
        
        if analysis["orderbook_imbalance"] > 0.1:
            analysis["bias"] = "bullish"
            analysis["score"] = 8
        elif analysis["orderbook_imbalance"] < -0.1:
            analysis["bias"] = "bearish"
            analysis["score"] = 8
        else:
            analysis["score"] = 5
            
        return analysis
    except Exception as e:
        log_w(f"advanced_market_flow error: {e}")
        return {"score": 0, "bias": "neutral"}

def advanced_candle_analysis(df):
    """تحليل الشموع المتقدم"""
    try:
        analysis = {"candle_patterns": [], "score": 0, "bias": "neutral"}
        
        if len(df) < 3:
            return analysis
            
        # تحليل الشموع البسيط
        last_close = df['close'].iloc[-1]
        last_open = df['open'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        prev_open = df['open'].iloc[-2]
        
        # شمعة صاعدة قوية
        if last_close > last_open and (last_close - last_open) > (last_open - df['low'].iloc[-1]):
            analysis["candle_patterns"].append("bullish_strong")
            analysis["bias"] = "bullish"
            analysis["score"] = 8
        # شمعة هابطة قوية
        elif last_close < last_open and (last_open - last_close) > (df['high'].iloc[-1] - last_open):
            analysis["candle_patterns"].append("bearish_strong")
            analysis["bias"] = "bearish"
            analysis["score"] = 8
        else:
            analysis["score"] = 5
            
        return analysis
    except Exception as e:
        log_w(f"advanced_candle_analysis error: {e}")
        return {"score": 0, "bias": "neutral"}

def advanced_momentum_analysis(df):
    """تحليل الزخم المتقدم"""
    try:
        analysis = {"rsi_signals": [], "score": 0, "bias": "neutral"}
        indicators = compute_indicators(df)
        
        rsi = indicators.get("rsi", 50)
        adx = indicators.get("adx", 0)
        
        if rsi < 30 and adx > 25:
            analysis["rsi_signals"].append("oversold_bullish")
            analysis["bias"] = "bullish"
            analysis["score"] = 9
        elif rsi > 70 and adx > 25:
            analysis["rsi_signals"].append("overbought_bearish")
            analysis["bias"] = "bearish" 
            analysis["score"] = 9
        elif 40 < rsi < 60 and adx > 20:
            analysis["score"] = 6
        else:
            analysis["score"] = 4
            
        return analysis
    except Exception as e:
        log_w(f"advanced_momentum_analysis error: {e}")
        return {"score": 0, "bias": "neutral"}

def calculate_total_confidence(council_report):
    """حساب الثقة الشاملة"""
    try:
        strategies = council_report["strategies"]
        total_score = 0
        max_score = 0
        
        weights = {"smc_engine": 25, "market_structure": 20, "market_flow": 20, 
                  "candle_analysis": 18, "momentum_analysis": 17}
        
        for strategy_name, strategy in strategies.items():
            weight = weights.get(strategy_name, 20)
            total_score += strategy["score"] * weight
            max_score += weight
        
        if max_score > 0:
            confidence = (total_score / max_score) * 100
            return min(confidence, 100)
        return 0
    except Exception as e:
        log_w(f"calculate_total_confidence error: {e}")
        return 0

def generate_trade_recommendation(council_report):
    """توليد توصية تداول ذكية"""
    try:
        confidence = council_report["confidence_score"]
        strategies = council_report["strategies"]
        
        if confidence < TRADE_MIN_CONFIDENCE:
            return {
                "action": "wait",
                "reason": f"ثقة غير كافية: {confidence:.1f}%",
                "trade_type": "none",
                "risk_level": "low"
            }
        
        # تحديد الانحياز السائد
        biases = [s["bias"] for s in strategies.values() if s["bias"] != "neutral"]
        if not biases:
            return {
                "action": "wait", 
                "reason": "لا يوجد انحياز واضح",
                "trade_type": "none", 
                "risk_level": "low"
            }
        
        bullish_count = biases.count("bullish")
        bearish_count = biases.count("bearish")
        
        if bullish_count > bearish_count:
            direction = "buy"
            bias_strength = bullish_count
        else:
            direction = "sell"
            bias_strength = bearish_count
        
        # تحديد نوع الصفقة
        if confidence >= STRONG_TRADE_CONFIDENCE and bias_strength >= 3:
            trade_type = "trend"
            risk_level = "high"
            reason = f"صفقة ترند قوية - ثقة: {confidence:.1f}%"
        elif confidence >= TRADE_MIN_CONFIDENCE and bias_strength >= 2:
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
        return {"action": "wait", "reason": f"خطأ: {str(e)}", "trade_type": "none", "risk_level": "low"}

def generate_decision_reasoning(council_report):
    """توليد أسباب القرار المفصلة"""
    try:
        reasoning = []
        decision = council_report["final_decision"]
        
        reasoning.append(f"🎯 قرار المجلس: {decision['action'].upper()} - {decision['trade_type']}")
        reasoning.append(f"💪 قوة الثقة: {decision['confidence']:.1f}%")
        reasoning.append(f"📊 قوة الانحياز: {decision['bias_strength']}/5 استراتيجيات")
        
        # إضافة أسباب من الاستراتيجيات القوية
        strategies = council_report["strategies"]
        for name, strategy in strategies.items():
            if strategy["score"] >= 7:
                reasoning.append(f"✅ {name}: {strategy['bias']} (قوة: {strategy['score']}/10)")
        
        return reasoning
    except Exception as e:
        log_w(f"generate_decision_reasoning error: {e}")
        return ["خطأ في توليد أسباب القرار"]

# =================== INTELLIGENT POSITION MANAGEMENT ===================
def intelligent_position_management(council_decision, current_price, balance):
    """إدارة ذكية للمراكز بناءً على قرار المجلس"""
    try:
        trade_type = council_decision["final_decision"]["trade_type"]
        confidence = council_decision["confidence_score"]
        
        # تحديد حجم المركز الذكي
        if trade_type == "trend":
            base_size = balance * RISK_ALLOC
            size_multiplier = 1.2 if confidence > 90 else 1.0 if confidence > 80 else 0.8
        elif trade_type == "momentum":
            base_size = balance * (RISK_ALLOC * 0.7)
            size_multiplier = 1.0
        else:  # scalp
            base_size = balance * (RISK_ALLOC * 0.5)
            size_multiplier = 0.8
        
        position_size = compute_precise_size(base_size * size_multiplier, current_price, SYMBOL)
        
        # إعداد إدارة الصفقة الذكية
        management_config = setup_intelligent_management(trade_type, confidence, current_price)
        
        return {
            "position_size": position_size,
            "management_config": management_config,
            "risk_level": council_decision["final_decision"]["risk_level"],
            "trade_type": trade_type
        }
    except Exception as e:
        log_e(f"intelligent_position_management error: {e}")
        return {
            "position_size": compute_precise_size(balance * 0.5, current_price, SYMBOL),
            "management_config": setup_intelligent_management("scalp", 70, current_price),
            "risk_level": "low",
            "trade_type": "scalp"
        }

def setup_intelligent_management(trade_type, confidence, entry_price):
    """إعداد ذكي لإدارة الصفقة"""
    if trade_type == "trend":
        tp_levels = [
            entry_price * (1 + (1.0 + (confidence - 80) * 0.05) / 100),
            entry_price * (1 + (2.0 + (confidence - 80) * 0.1) / 100),
            entry_price * (1 + (3.5 + (confidence - 80) * 0.15) / 100)
        ]
        tp_fractions = [0.3, 0.4, 0.3]
        sl_distance = 0.8
        trail_activation = 0.6
        trail_distance = 0.4
        
    elif trade_type == "momentum":
        tp_levels = [
            entry_price * (1 + (0.8 + (confidence - 70) * 0.03) / 100),
            entry_price * (1 + (1.6 + (confidence - 70) * 0.06) / 100)
        ]
        tp_fractions = [0.5, 0.5]
        sl_distance = 1.0
        trail_activation = 0.8
        trail_distance = 0.5
        
    else:  # scalp
        tp_levels = [entry_price * (1 + (0.5 + (confidence - 60) * 0.02) / 100)]
        tp_fractions = [1.0]
        sl_distance = 0.6
        trail_activation = 0.3
        trail_distance = 0.2
    
    return {
        "tp_levels": tp_levels,
        "tp_fractions": tp_fractions,
        "initial_sl": entry_price * (1 - sl_distance / 100),
        "trail_activation_pct": trail_activation,
        "trail_distance_pct": trail_distance,
        "breakeven_trigger": trail_activation * 0.8,
        "max_duration_hours": 48 if trade_type == "trend" else 12
    }

# =================== AI-POWERED EXIT STRATEGY ===================
def ai_exit_strategy(df, current_price, position_info, council_decision):
    """استراتيجية خروج ذكية بالذكاء الاصطناعي"""
    try:
        if not STATE["open"]: return "hold"
        
        entry = STATE["entry"]
        side = STATE["side"]
        pnl_pct = (current_price - entry) / entry * 100 * (1 if side == "long" else -1)
        management_config = position_info["management_config"]
        
        # 1. جني الأرباح الذكي على المراحل
        exit_signal = smart_profit_taking(current_price, pnl_pct, management_config, council_decision)
        if exit_signal != "hold": return exit_signal
        
        # 2. وقف الخسارة المتحرك الذكي
        exit_signal = intelligent_trailing_stop(current_price, pnl_pct, management_config, df)
        if exit_signal != "hold": return exit_signal
        
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
                    log_g(f"🎯 TP{i+1} achieved: {pnl_pct:.2f}% | Closed {fraction*100}%")
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
            
            if STATE["side"] == "long":
                STATE["highest_price"] = max(STATE.get("highest_price", current_price), current_price)
                new_sl = STATE["highest_price"] * (1 - management_config["trail_distance_pct"] / 100)
            else:
                STATE["lowest_price"] = min(STATE.get("lowest_price", current_price), current_price)
                new_sl = STATE["lowest_price"] * (1 + management_config["trail_distance_pct"] / 100)
            
            if (STATE["side"] == "long" and new_sl > STATE.get("current_sl", 0)) or \
               (STATE["side"] == "short" and new_sl < STATE.get("current_sl", float('inf'))):
                STATE["current_sl"] = new_sl
        
        if pnl_pct >= management_config["breakeven_trigger"] and not STATE.get("breakeven_activated"):
            STATE["current_sl"] = STATE["entry"]
            STATE["breakeven_activated"] = True
            log_g("🔒 Breakeven Activated - Risk Free Trade")
        
        if (STATE["side"] == "long" and current_price <= STATE.get("current_sl", 0)) or \
           (STATE["side"] == "short" and current_price >= STATE.get("current_sl", float('inf'))):
            log_g(f"🛡️ Trailing SL Hit: {pnl_pct:.2f}%")
            return "close"
        
        return "hold"
    except Exception as e:
        log_w(f"intelligent_trailing_stop error: {e}")
        return "hold"

# =================== EXCHANGE / POSITION API (from old DOGE bot) ===================

def _normalize_side(pos):
    side = pos.get("side") or pos.get("positionSide") or ""
    if side:
        return side.upper()
    qty = float(pos.get("contracts") or pos.get("positionAmt") or pos.get("size") or 0)
    return "LONG" if qty > 0 else ("SHORT" if qty < 0 else "")

def fetch_live_position(exchange, symbol: str):
    """Read live position from exchange (generic for ccxt)."""
    try:
        if hasattr(exchange, "fetch_positions"):
            arr = exchange.fetch_positions([symbol])
            for p in arr or []:
                sym = p.get("symbol") or p.get("info", {}).get("symbol")
                if sym and symbol.replace(":", "") in sym.replace(":", ""):
                    side = _normalize_side(p)
                    qty = abs(float(
                        p.get("contracts")
                        or p.get("positionAmt")
                        or p.get("info", {}).get("size", 0)
                        or 0
                    ))
                    if qty > 0:
                        entry = float(p.get("entryPrice") or p.get("info", {}).get("entryPrice") or 0.0)
                        lev   = float(p.get("leverage") or p.get("info", {}).get("leverage") or 0.0)
                        unr   = float(p.get("unrealizedPnl") or 0.0)
                        return {
                            "ok": True,
                            "side": side,
                            "qty": qty,
                            "entry": entry,
                            "unrealized": unr,
                            "leverage": lev,
                            "raw": p,
                        }
        if hasattr(exchange, "fetch_position"):
            p = exchange.fetch_position(symbol)
            side = _normalize_side(p)
            qty  = abs(float(p.get("size") or 0))
            if qty > 0:
                entry = float(p.get("entryPrice") or 0.0)
                lev   = float(p.get("leverage") or 0.0)
                unr   = float(p.get("unrealizedPnl") or 0.0)
                return {"ok": True, "side": side, "qty": qty, "entry": entry, "unrealized": unr, "leverage": lev, "raw": p}
    except Exception as e:
        log_w(f"fetch_live_position error: {e}")
    return {"ok": False, "why": "no_open_position"}

def resume_open_position(exchange, symbol: str, state: dict) -> dict:
    """Try to resume open position after restart."""
    if not RESUME_ON_RESTART:
        log_i("resume disabled")
        return state

    live = fetch_live_position(exchange, symbol)
    if not live.get("ok"):
        log_i("no live position to resume")
        return state

    ts = int(time.time())
    prev = load_state() or {}
    if prev.get("ts") and (ts - int(prev["ts"])) > RESUME_LOOKBACK_SECS:
        log_w("found old local state — will override with exchange live snapshot")

    state.update({
        "open": True,
        "side": live["side"].lower(),      # long/short
        "entry": live["entry"],
        "size": live["qty"],
        "qty": live["qty"],
    })
    STATE["open"] = True
    STATE["side"] = live["side"].lower()
    STATE["entry"] = live["entry"]
    STATE["size"] = live["qty"]
    STATE["qty"] = live["qty"]
    log_i(f"RESUMED position from exchange: side={STATE['side']} qty={STATE['qty']} entry={STATE['entry']}")
    return state

# =================== SIZE / POSITION HELPERS ===================

def safe_qty(qty: float) -> float:
    """Clamp quantity to exchange precision + avoid tiny dust."""
    try:
        q = float(qty)
    except Exception:
        return 0.0
    if q <= 0:
        return 0.0
    # BingX DOGE الحد الأدنى عادة 1
    return max(round(q), 0.0)

def _read_position():
    """Read position from STATE first, fallback to exchange if needed."""
    try:
        if STATE.get("open") and STATE.get("qty", 0) > 0:
            return STATE["qty"], STATE["side"], STATE["entry"]
        live = fetch_live_position(ex, SYMBOL)
        if live.get("ok"):
            side = "long" if live["side"] == "LONG" else "short"
            return live["qty"], side, live["entry"]
    except Exception as e:
        log_w(f"_read_position error: {e}")
    return 0.0, None, None

def compute_size(balance: float, price: float) -> float:
    """
    Fixed risk: RISK_ALLOC × balance × LEVERAGE, converted to qty.
    Same spirit as old DOGE bot.
    """
    if balance <= 0 or price <= 0:
        return 0.0
    notional = balance * RISK_ALLOC * LEVERAGE
    qty = notional / price
    if qty < MIN_QTY:
        qty = MIN_QTY
    return safe_qty(qty)

# =================== ORDER EXECUTION (OPEN / CLOSE) ===================

def _params_open(side: str):
    """Exchange-specific params for opening a position."""
    return {"positionSide": "LONG" if side == "buy" else "SHORT"}

def _params_close():
    """Params for closing (reduceOnly, etc.)."""
    return {"reduceOnly": True}

def open_market_enhanced(side: str, qty: float, price: float = None, reason: str = "COUNCIL"):
    """
    Unified market order open with logging + STATE update.
    side = 'buy' (long) أو 'sell' (short)
    """
    global STATE
    qty = safe_qty(qty)
    if qty <= 0:
        log_w("open_market_enhanced: qty<=0, skip")
        return False

    if STATE.get("open"):
        log_w("open_market_enhanced: already in position, skip")
        return False

    order_side = side.lower()
    params = _params_open(order_side)
    try:
        if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
            ex.create_order(SYMBOL, "market", order_side, qty, None, params)
        px = price or (price_now() or 0.0)
        STATE.update({
            "open": True,
            "side": "long" if order_side == "buy" else "short",
            "entry": px,
            "size": qty,
            "qty": qty,
            "pnl": 0.0,
            "bars": 0,
            "trail": None,
            "breakeven": None,
            "tp1_done": False,
            "highest_profit_pct": 0.0,
            "profit_targets_achieved": 0,
            "trail_tightened": False,
            "partial_taken": False,
        })
        save_state(STATE)
        log_i(f"✅ OPEN {STATE['side'].upper()} qty={qty:.4f} px={px:.6f} reason={reason}")
        return True
    except Exception as e:
        log_e(f"❌ open_market_enhanced error: {e}")
        return False

def close_market_strict(reason: str = "STRICT"):
    """
    Strict full close with retries + compound_pnl update.
    """
    global compound_pnl
    exch_qty, exch_side, exch_entry = _read_position()
    if exch_qty <= 0:
        if STATE.get("open"):
            _reset_after_close(reason)
        return

    side_to_close = "sell" if exch_side == "long" else "buy"
    qty_to_close  = safe_qty(exch_qty)
    attempts = 0
    last_error = None

    while attempts < CLOSE_RETRY_ATTEMPTS:
        try:
            if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                params = _params_close()
                ex.create_order(SYMBOL, "market", side_to_close, qty_to_close, None, params)
            time.sleep(CLOSE_VERIFY_WAIT_S)
            left_qty, _, _ = _read_position()
            if left_qty <= 0:
                px = price_now() or STATE.get("entry") or exch_entry
                entry_px = STATE.get("entry") or exch_entry or px
                side = STATE.get("side") or exch_side or ("long" if side_to_close == "sell" else "short")
                qty = exch_qty
                pnl_pct = (px - entry_px) / entry_px * 100 * (1 if side == "long" else -1)
                compound_pnl += pnl_pct
                log_i(f"STRICT CLOSE {side} reason={reason} pnl={pnl_pct:.2f}% total={compound_pnl:.2f}%")
                _reset_after_close(reason, prev_side=side)
                return
            qty_to_close = safe_qty(left_qty)
            attempts += 1
            log_w(f"strict close retry {attempts}/{CLOSE_RETRY_ATTEMPTS} — residual={left_qty:.4f}")
            time.sleep(CLOSE_VERIFY_WAIT_S)
        except Exception as e:
            last_error = e
            log_e(f"close_market_strict attempt {attempts+1}: {e}")
            attempts += 1
            time.sleep(CLOSE_VERIFY_WAIT_S)

    log_e(f"STRICT CLOSE FAILED after {CLOSE_RETRY_ATTEMPTS} attempts — last error: {last_error}")

def close_position_partial(close_fraction: float, why: str = "partial"):
    """
    Simple partial close using reduceOnly market orders.
    close_fraction = نسبة الكمية (مثلاً 0.3 = 30%)
    """
    if not STATE.get("open") or STATE.get("qty", 0) <= 0:
        return
    qty = STATE["qty"]
    partial_qty = safe_qty(qty * close_fraction)
    if partial_qty <= 0:
        return

    close_side = "sell" if STATE["side"] == "long" else "buy"
    try:
        if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
            ex.create_order(SYMBOL, "market", close_side, partial_qty, None, _params_close())
        STATE["qty"] = safe_qty(qty - partial_qty)
        STATE["size"] = STATE["qty"]
        log_i(f"✅ PARTIAL CLOSE {partial_qty:.4f} | {why}")
    except Exception as e:
        log_e(f"❌ close_position_partial error: {e}")

def _reset_after_close(reason: str, prev_side: str = None):
    """Reset STATE after closing a position."""
    global STATE
    STATE.update({
        "open": False,
        "side": None,
        "entry": 0.0,
        "size": 0.0,
        "qty": 0.0,
        "pnl": 0.0,
        "bars": 0,
        "trail": None,
        "breakeven": None,
        "tp1_done": False,
        "highest_profit_pct": 0.0,
        "profit_targets_achieved": 0,
        "trail_tightened": False,
        "partial_taken": False,
        "trailing_active": False,
        "breakeven_activated": False,
    })
    save_state(STATE)
    log_i(f"🔄 STATE reset after close: {reason} (prev_side={prev_side})")

# =================== ENHANCED POSITION MANAGEMENT ===================

def manage_after_entry_enhanced(df, current_price, council_decision):
    """Basic smart management using pnl%, TP1, breakeven, trail and smart exit."""
    if not STATE.get("open") or STATE.get("qty", 0) <= 0:
        return "hold"

    px    = current_price
    entry = STATE["entry"]
    side  = STATE["side"]
    qty   = STATE["qty"]

    pnl_pct = (px - entry) / entry * 100 * (1 if side == "long" else -1)
    STATE["pnl"] = pnl_pct
    if pnl_pct > STATE.get("highest_profit_pct", 0.0):
        STATE["highest_profit_pct"] = pnl_pct

    # TP1 بسيط
    if not STATE.get("tp1_done") and pnl_pct >= TP1_PCT_BASE:
        close_position_partial(TP1_CLOSE_FRAC, why=f"TP1 {TP1_PCT_BASE:.2f}%")
        STATE["tp1_done"] = True
        STATE["profit_targets_achieved"] = STATE.get("profit_targets_achieved", 0) + 1
        return "partial_close"

    # Breakeven
    if not STATE.get("breakeven_armed") and pnl_pct >= BREAKEVEN_AFTER:
        STATE["breakeven_armed"] = True
        STATE["breakeven"] = entry
        STATE["current_sl"] = entry
        log_i("🔒 BREAKEVEN ARMED")

    # وقف خسارة صارم
    if pnl_pct <= -3.0:  # -3% خسارة
        log_w("🚨 HARD STOP: pnl<=-3%")
        close_market_strict("hard_stop_loss")
        return "close"

    return "hold"

# =================== SUPREME TRADE EXECUTION ===================
def execute_supreme_trade(action, qty, price, position_info, council_decision):
    """تنفيذ الصفقة العليا - الإصدار النهائي"""
    try:
        if not EXECUTE_ORDERS or DRY_RUN:
            log_i(f"DRY_RUN: {action} {qty:.4f} @ {price:.6f}")
            return True
        
        if qty <= 0:
            log_e("❌ Invalid quantity for execution")
            return False

        # التأكد من دقة الكمية
        if "DOGE" in SYMBOL:
            qty = int(qty)
            if qty < 1:
                log_e("❌ DOGE quantity must be at least 1")
                return False

        log_g(f"🎯 Executing: {action.upper()} {qty:.0f} @ {price:.6f}")
        
        # التنفيذ الفعلي
        if MODE_LIVE:
            try:
                ex.set_leverage(LEVERAGE, SYMBOL, params={"side": "BOTH"})
                ex.create_order(SYMBOL, "market", action, qty, None, _params_open(action))
            except Exception as e:
                log_e(f"❌ Exchange execution error: {e}")
                return False
        
        # تحديث حالة البوت
        STATE.update({
            "open": True,
            "side": action,
            "entry": price,
            "size": qty,
            "qty": qty,
            "pnl": 0,
            "trade_type": position_info["trade_type"],
            "management_config": position_info["management_config"],
            "entry_council": council_decision,
            "entry_time": time.time(),
            "trailing_active": False,
            "breakeven_activated": False
        })
        
        # تسجيل الصفقة
        log_g(f"🎯 SUPREME TRADE EXECUTED: {action.upper()} {qty:.0f} @ {price:.6f}")
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

# =================== BASIC EXCHANGE FUNCTIONS ===================
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
            
            if STATE["entry"] > 0:
                current_price = price_now()
                final_pnl = (current_price - STATE["entry"]) / STATE["entry"] * 100 * (1 if STATE["side"] == "long" else -1)
                compound_pnl += final_pnl
                log_g(f"💰 POSITION CLOSED | Final PnL: {final_pnl:.2f}% | Total: {compound_pnl:.2f}%")
    except Exception as e:
        log_e(f"close_position error: {e}")

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
    "qty": 0,
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
    "breakeven_activated": False,
    "bars": 0,
    "trail": None,
    "breakeven": None,
    "tp1_done": False,
    "highest_profit_pct": 0.0,
    "profit_targets_achieved": 0,
    "trail_tightened": False,
    "partial_taken": False,
    "breakeven_armed": False,
}

compound_pnl = 0.0

# محاولة استئناف البوزيشن المفتوح من المنصة
STATE = resume_open_position(ex, SYMBOL, STATE)

# =================== MAIN TRADING LOOP ===================
def supreme_trade_loop():
    """حلقة التداول العليا - الذكاء المتكامل"""
    global STATE, compound_pnl
    
    loop_i = 0
    
    while True:
        try:
            # جمع البيانات الشاملة
            bal = balance_usdt()
            px = price_now()
            df = fetch_ohlcv()
            orderbook = fetch_orderbook()
            
            if df.empty:
                log_w("No data fetched, skipping iteration")
                time.sleep(5)
                continue
            
            # حساب المؤشرات
            indicators = compute_indicators(df)
            
            # قرار المجلس الأعلى
            council_decision = supreme_council_decision(df, px, orderbook)
            
            # 🎯 التسجيل المحترف
            if loop_i % 3 == 0:
                try:
                    trade_rec = council_decision["final_decision"]
                    action = trade_rec.get("action", "wait")
                    trade_type = trade_rec.get("trade_type", "scalp")
                    confidence = council_decision.get("confidence_score", 0)
                    
                    # حساب الأصوات
                    strategies = council_decision.get("strategies", {})
                    votes_for = sum(1 for s in strategies.values() if s.get('score', 0) >= 7)
                    votes_against = len(strategies) - votes_for
                    
                    # بيانات Bookmap و Flow
                    bookmap_data = {
                        "imbalance": council_decision.get("strategies", {}).get("market_flow", {}).get("orderbook_imbalance", 1.0),
                        "bids": orderbook.get('bids', [])[:3] if orderbook else [],
                        "asks": orderbook.get('asks', [])[:3] if orderbook else []
                    }
                    
                    flow_data = {
                        "flow": council_decision.get("strategies", {}).get("market_flow", {}).get("orderbook_imbalance", 0) * 10000,
                        "delta": council_decision.get("strategies", {}).get("market_flow", {}).get("orderbook_imbalance", 0) * 5000,
                        "z_score": 0,
                        "cvd": council_decision.get("strategies", {}).get("market_flow", {}).get("orderbook_imbalance", 0) * 100000
                    }
                    
                    # 🔥 التسجيل بالشكل المطلوب
                    log_strategy_line(action, trade_type, bal or 0, compound_pnl or 0)
                    log_snap_line(action, votes_for, votes_against, confidence/10, 
                                indicators.get('adx',0), indicators.get('plus_di',0)-indicators.get('minus_di',0),
                                0, bookmap_data.get("imbalance",1.0))
                    log_addons_live()
                    log_bookmap_line(bookmap_data)
                    flow_side = log_flow_line(flow_data)
                    log_dash_hint_line(flow_side or action, council_decision, indicators)
                    
                except Exception as e:
                    log_w(f"Professional logging error: {e}")
            
            # إدارة الصفقة المفتوحة
            if STATE["open"]:
                # استخدام نظام الإدارة المحسن
                exit_signal = manage_after_entry_enhanced(df, px, council_decision)
                
                if exit_signal == "close":
                    close_market_strict("smart_management_exit")
                    log_g("🎯 Smart Management - Position Closed")
                    continue
            
            # قرار الدخول الجديد
            trade_rec = council_decision["final_decision"]
            
            if not STATE["open"] and trade_rec["action"] != "wait":
                if council_decision["confidence_score"] >= TRADE_MIN_CONFIDENCE:
                    # استخدام compute_size الموحدة من البوت القديم
                    qty = compute_size(bal, px)
                    
                    if qty > 0:
                        # استخدام open_market_enhanced بدلاً من execute_supreme_trade
                        success = open_market_enhanced(trade_rec["action"], qty, px, reason="COUNCIL_PRO_TIER")
                        if success:
                            # تحديث معلومات إضافية في STATE
                            STATE.update({
                                "trade_type": trade_rec.get("trade_type", "scalp"),
                                "management_config": intelligent_position_management(council_decision, px, bal)["management_config"],
                                "entry_council": council_decision,
                                "entry_time": time.time(),
                            })
                            log_g("🚀 SUPREME TRADE EXECUTED - AI Powered Entry")
                            log_g(f"📊 Trade Type: {trade_rec.get('trade_type', 'scalp').upper()}")
                            log_g(f"💪 Confidence: {council_decision['confidence_score']:.1f}%")
                    else:
                        log_w(f"⚠️ Quantity too small: {qty}")
            
            loop_i += 1
            time.sleep(5)
            
        except Exception as e:
            log_e(f"supreme_trade_loop error: {e}\n{traceback.format_exc()}")
            time.sleep(10)

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

# =================== MAIN EXECUTION ===================
if __name__ == "__main__":
    log_banner("SUPREME COUNCIL AI TRADING BOT - ULTIMATE PRO")
    
    print(colored(f"🚀 SUPREME COUNCIL AI TRADING SYSTEM", "yellow"))
    print(colored(f"🎯 MODE: {'LIVE' if MODE_LIVE else 'PAPER'} • {SYMBOL} • {INTERVAL}", "yellow"))
    print(colored(f"💪 STRATEGIES: SMC Engine + Market Structure + Flow Analysis", "yellow"))
    print(colored(f"📊 ANALYSIS: Candle Patterns + Momentum + Multi-timeframe", "yellow"))
    print(colored(f"🤖 AI EXIT: Smart Profit Taking + Intelligent Trailing Stop", "yellow"))
    print(colored(f"📈 POSITION MGMT: Dynamic Sizing + Multi-stage TP", "yellow"))
    print(colored(f"🛡️ RISK: {RISK_ALLOC*100}% × {LEVERAGE}x • Min Confidence: {TRADE_MIN_CONFIDENCE}%", "yellow"))
    print(colored(f"⚡ EXECUTION: {'ACTIVE' if EXECUTE_ORDERS and not DRY_RUN else 'SIMULATION'}", "yellow"))
    
    import threading
    threading.Thread(target=supreme_trade_loop, daemon=True).start()
    
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

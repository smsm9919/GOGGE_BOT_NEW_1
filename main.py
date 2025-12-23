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
• ADVANCED FEATURES: Danger Zone Detection + Professional Exit Management
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
BOT_VERSION = "DOGE Scenario Engine PRO v8.0 — Professional Food Print + Smart Council + Danger Zone Detection"
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

# =================== DANGER ZONE DETECTION ===================
DANGER_ZONE_ADX_CHANGE = 0.3  # تغيير ADX بنسبة 30%
DANGER_ZONE_VOLUME_DROP = -1.5  # انخفاض الحجم
DANGER_ZONE_VWAP_BREAK = True   # اختراق VWAP مع حجم ضعيف
DANGER_ZONE_RSI_EXTREME = 90    # RSI فوق 90 أو تحت 10

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
    danger_zone: bool = False  # إضافة كشف المنطقة الخطرة

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
    danger_zone_detected: bool = False  # كشف المنطقة الخطرة
    meta: Optional[Dict[str, Any]] = None

# =================== PROFESSIONAL LOGGING ===================
def log_i(msg): print(f"ℹ️ {msg}", flush=True)
def log_g(msg): print(f"✅ {msg}", flush=True)
def log_w(msg): print(f"🟨 {msg}", flush=True)
def log_e(msg): print(f"❌ {msg}", flush=True)
def log_d(msg): print(f"⚠️ [DANGER] {msg}", flush=True)  # تسجيل المناطق الخطرة

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

# =================== ENHANCED FOOD PRINT DETECTION ===================
def detect_food_print_advanced_v2(df: pd.DataFrame, lookback: int = FOOD_PRINT_LOOKBACK) -> Dict[str, Any]:
    """
    كشف فود برنت متقدم محسّن: مناطق السيولة والتراكم والتوزيع مع تحليل عميق
    """
    n = len(df)
    if n < lookback + 10:
        return {"ok": False}
    
    results = {
        "liquidity_pools": [],
        "accumulation_zones": [],
        "distribution_zones": [],
        "fair_value_gaps": [],
        "danger_zones": [],
        "zone_strength": 0.5,
        "zone_quality": "UNKNOWN"
    }
    
    # 1. اكتشاف مناطق السيولة المتقدمة
    window_size = 7
    for i in range(n - lookback, n - window_size):
        window = df.iloc[i:i+window_size]
        max_range = float(window["high"].max() - window["low"].min())
        avg_body = float(abs(window["close"] - window["open"]).mean())
        volume_avg = float(window["volume"].mean())
        
        # منطقة سيولة حقيقية: شموع صغيرة + حجم مرتفع
        if avg_body < max_range * 0.2 and volume_avg > df["volume"].iloc[i-20:i].mean() * 1.5:
            results["liquidity_pools"].append({
                "start": i,
                "end": i + window_size,
                "range_low": float(window["low"].min()),
                "range_high": float(window["high"].max()),
                "volume_factor": volume_avg / max(df["volume"].mean(), 1e-12)
            })
    
    # 2. مناطق التراكم المتقدمة (سعر يتراكم والحجم يرتفع)
    for i in range(n - lookback, n - 25):
        window = df.iloc[i:i+25]
        first_third = window.iloc[:8]
        second_third = window.iloc[8:16]
        last_third = window.iloc[16:]
        
        price_trend = (float(last_third["close"].mean()) > float(first_third["close"].mean()))
        volume_trend = (float(first_third["volume"].mean()) > float(last_third["volume"].mean()) * 1.2)
        
        if price_trend and volume_trend:
            results["accumulation_zones"].append({
                "start": i,
                "end": i + 25,
                "avg_price": float(window["close"].mean()),
                "volume_ratio": float(first_third["volume"].mean()) / float(last_third["volume"].mean())
            })
    
    # 3. مناطق التوزيع المتقدمة
    for i in range(n - lookback, n - 25):
        window = df.iloc[i:i+25]
        first_third = window.iloc[:8]
        second_third = window.iloc[8:16]
        last_third = window.iloc[16:]
        
        price_trend = (float(last_third["close"].mean()) < float(first_third["close"].mean()))
        volume_trend = (float(first_third["volume"].mean()) > float(last_third["volume"].mean()) * 1.2)
        
        if price_trend and volume_trend:
            results["distribution_zones"].append({
                "start": i,
                "end": i + 25,
                "avg_price": float(window["close"].mean()),
                "volume_ratio": float(first_third["volume"].mean()) / float(last_third["volume"].mean())
            })
    
    # 4. كشف المناطق الخطرة
    for i in range(n - 10, n - 3):
        window = df.iloc[i:i+3]
        if len(window) < 3:
            continue
            
        # انخفاض حاد في الحجم + اختراق سعري
        vol_drop = (float(window["volume"].iloc[-1]) < float(window["volume"].iloc[0]) * 0.5)
        price_spike = (abs(float(window["close"].iloc[-1]) - float(window["close"].iloc[0])) / 
                      float(window["close"].iloc[0]) > 0.02)
        
        if vol_drop and price_spike:
            results["danger_zones"].append({
                "start": i,
                "end": i + 3,
                "reason": "Volume drop with price spike",
                "risk_level": "HIGH"
            })
    
    # 5. حساب قوة وجودة المنطقة
    recent_price = float(df["close"].iloc[-1])
    zone_strength = 0.5
    zone_quality = "NEUTRAL"
    
    # عامل 1: قرب السعر من مناطق التراكم/التوزيع
    accumulation_distance = None
    for zone in results["accumulation_zones"]:
        distance = abs(recent_price - zone["avg_price"]) / zone["avg_price"]
        if accumulation_distance is None or distance < accumulation_distance:
            accumulation_distance = distance
    
    if accumulation_distance and accumulation_distance < 0.015:
        zone_strength += 0.3
        zone_quality = "ACCUMULATION_ZONE"
    
    # عامل 2: وجود مناطق سيولة قريبة
    liquidity_nearby = False
    for pool in results["liquidity_pools"]:
        if pool["range_low"] <= recent_price <= pool["range_high"]:
            liquidity_nearby = True
            break
    
    if liquidity_nearby:
        zone_strength += 0.15
        zone_quality = "LIQUIDITY_ZONE" if zone_quality == "NEUTRAL" else zone_quality
    
    # عامل 3: المناطق الخطرة
    if results["danger_zones"]:
        zone_strength -= 0.25
        zone_quality = "DANGER_ZONE"
    
    # عامل 4: عدد المناطق المكتشفة
    total_zones = len(results["accumulation_zones"]) + len(results["distribution_zones"])
    if total_zones > 3:
        zone_strength += min(0.15 * total_zones, 0.3)
    
    results["zone_strength"] = min(max(zone_strength, 0.1), 0.95)
    results["zone_quality"] = zone_quality
    
    return {"ok": True, **results}

# =================== DANGER ZONE DETECTION ===================
def detect_danger_zone_advanced(df: pd.DataFrame, ctx: SignalContext) -> Dict[str, Any]:
    """كشف المنطقة الخطرة المتقدم"""
    danger_reasons = []
    risk_level = "LOW"
    
    # 1. RSI في أقصى المدى
    if ctx.rsi >= DANGER_ZONE_RSI_EXTREME or ctx.rsi <= (100 - DANGER_ZONE_RSI_EXTREME):
        danger_reasons.append(f"RSI_EXTREME({ctx.rsi:.1f})")
        risk_level = "HIGH"
    
    # 2. تغيير حاد في ADX
    if len(df) > ADX_LEN:
        adx_series = calc_adx_di(df, ADX_LEN)[0]
        if len(adx_series) > 1:
            adx_change = abs(float(adx_series.iloc[-1]) - float(adx_series.iloc[-2])) / float(adx_series.iloc[-2])
            if adx_change > DANGER_ZONE_ADX_CHANGE and ctx.adx < 20:
                danger_reasons.append(f"ADX_CHANGE({adx_change:.2f})")
                risk_level = "MEDIUM"
    
    # 3. انخفاض حاد في الحجم
    if ctx.vol_z < DANGER_ZONE_VOLUME_DROP:
        danger_reasons.append(f"VOLUME_DROP(z={ctx.vol_z:.2f})")
        risk_level = "MEDIUM"
    
    # 4. اختراق VWAP مع حجم ضعيف
    vwap_distance = abs(ctx.px - ctx.vwap) / ctx.vwap
    if DANGER_ZONE_VWAP_BREAK and vwap_distance > 0.005 and ctx.vol_z < 0:
        danger_reasons.append(f"VWAP_BREAK_WEAK_VOLUME")
        risk_level = "MEDIUM"
    
    # 5. شمعة إشارة خطيرة
    candles = compute_candles(df)
    if candles["wick_up_big"] and ctx.rsi > 70:
        danger_reasons.append("UPPER_WICK_OVERSOLD")
        risk_level = "HIGH"
    if candles["wick_dn_big"] and ctx.rsi < 30:
        danger_reasons.append("LOWER_WICK_OVERSOLD")
        risk_level = "HIGH"
    
    # 6. انعكاس مفاجئ في الترند
    if ctx.adx > 25 and abs(ctx.di_plus - ctx.di_minus) < 5:
        danger_reasons.append("TREND_WEAKENING")
        risk_level = "MEDIUM"
    
    return {
        "is_danger_zone": len(danger_reasons) > 0,
        "risk_level": risk_level,
        "reasons": danger_reasons,
        "adx": ctx.adx,
        "rsi": ctx.rsi,
        "vol_z": ctx.vol_z
    }

# =================== SMART TRADE COUNCIL ENHANCED ===================
class SmartTradeCouncilEnhanced:
    """مجلس إدارة ذكي محسّن مع كاشف فود برنت"""
    
    def __init__(self):
        self.members = [
            self._risk_manager,
            self._trend_analyst,
            self._volume_analyst,
            self._price_action_analyst,
            self._footprint_analyst  # عضو جديد: محلل فود برنت
        ]
        self.weights = [0.25, 0.20, 0.20, 0.15, 0.20]  # أوزان القرارات
    
    def evaluate_trade(self, trade_data: Dict, market_ctx: SignalContext, 
                      footprint_data: Dict) -> Dict:
        """تقييم الصفقة من قبل جميع أعضاء المجلس مع فود برنت"""
        decisions = []
        
        for member in self.members:
            if member.__name__ == "_footprint_analyst":
                decision = member(trade_data, market_ctx, footprint_data)
            else:
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
        
        if risk_reward < 0.5 and profit_pct > 0.3:
            return {"action": "close", "reason": "نسبة المخاطرة/العائد سيئة", "confidence": 0.8}
        elif risk_reward > 2 and profit_pct > 1.0:
            return {"action": "take_profit", "reason": "نسبة المخاطرة/العائد ممتازة", "confidence": 0.9}
        
        return {"action": "hold", "reason": "نسبة المخاطرة/العائد متوسطة", "confidence": 0.6}
    
    def _trend_analyst(self, trade_data: Dict, ctx: SignalContext) -> Dict:
        """محلل الترند"""
        entry_bias = trade_data.get("entry_context", {}).get("bias", "neutral")
        current_bias = ctx.bias
        
        if entry_bias != current_bias and ctx.adx > 25:
            return {"action": "close", "reason": "انعكاس الترند", "confidence": 0.85}
        
        if ctx.adx < 15 and ctx.px < trade_data.get("entry", 0) * 0.99:
            return {"action": "close", "reason": "ضعف الترند مع خسارة", "confidence": 0.7}
        
        return {"action": "hold", "reason": "الترند سليم", "confidence": 0.75}
    
    def _volume_analyst(self, trade_data: Dict, ctx: SignalContext) -> Dict:
        """محلل الحجم"""
        if ctx.vol_z < -1.5:
            return {"action": "close", "reason": "حجم ضعيف جداً", "confidence": 0.8}
        
        if ctx.vol_z > 2.0 and ctx.displacement:
            return {"action": "hold", "reason": "حجم قوي مع إزاحة", "confidence": 0.9}
        
        return {"action": "hold", "reason": "حجم طبيعي", "confidence": 0.6}
    
    def _price_action_analyst(self, trade_data: Dict, ctx: SignalContext) -> Dict:
        """محلل حركة السعر"""
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
            
            # شمعة انعكاسية قوية
            if side == "long" and upper_wick > body * 2.5 and c < o:
                return {"action": "close", "reason": "شمعة انعكاسية قوية", "confidence": 0.85}
            if side == "short" and lower_wick > body * 2.5 and c > o:
                return {"action": "close", "reason": "شمعة انعكاسية قوية", "confidence": 0.85}
        
        return {"action": "hold", "reason": "حركة سعر طبيعية", "confidence": 0.65}
    
    def _footprint_analyst(self, trade_data: Dict, ctx: SignalContext, 
                          footprint_data: Dict) -> Dict:
        """محلل فود برنت"""
        if not footprint_data.get("ok", False):
            return {"action": "hold", "reason": "لا بيانات فود برنت", "confidence": 0.5}
        
        zone_strength = footprint_data.get("zone_strength", 0.5)
        zone_quality = footprint_data.get("zone_quality", "UNKNOWN")
        danger_zones = footprint_data.get("danger_zones", [])
        
        # إذا كانت المنطقة خطرة
        if danger_zones:
            recent_danger = any(dz["end"] >= len(trade_data.get("current_candles", [])) - 5 
                              for dz in danger_zones)
            if recent_danger:
                return {"action": "close", "reason": "منطقة خطرة في فود برنت", "confidence": 0.9}
        
        # إذا كانت قوة المنطقة ضعيفة
        if zone_strength < 0.4:
            return {"action": "close", "reason": f"قوة منطقة ضعيفة ({zone_strength:.2f})", "confidence": 0.7}
        
        # إذا كانت المنطقة عالية الجودة
        if zone_quality in ["ACCUMULATION_ZONE", "LIQUIDITY_ZONE"] and zone_strength > 0.7:
            return {"action": "hold", "reason": f"منطقة عالية الجودة: {zone_quality}", "confidence": 0.85}
        
        return {"action": "hold", "reason": f"جودة المنطقة: {zone_quality}", "confidence": 0.6}
    
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

# =================== ENHANCED SCENARIO DECISION ===================
def scenario_decide_professional(df: pd.DataFrame) -> ScenarioDecision:
    """اتخاذ القرار المحترف مع فود برنت محسن وكشف المناطق الخطرة"""
    if not SCENARIO_ENGINE_ENABLED:
        return ScenarioDecision(Action.NO_TRADE, Mode.NONE, "Scenario Engine Disabled", 0.0)
    
    ctx = build_context(df)
    fvg = detect_fvg(df, FVG_LOOKBACK)
    ob = detect_simple_ob(df, OB_LOOKBACK)
    sweep = detect_liquidity_sweep(df, 30)
    food_print = detect_food_print_advanced_v2(df, FOOD_PRINT_LOOKBACK)  # إصدار محسن
    danger_zone = detect_danger_zone_advanced(df, ctx)  # كشف المنطقة الخطرة
    
    # تحديث السياق بخطر المنطقة
    ctx.danger_zone = danger_zone["is_danger_zone"]
    
    px = ctx.px
    notes = list(ctx.notes)
    
    # إضافة معلومات المنطقة الخطرة
    if danger_zone["is_danger_zone"]:
        notes.append(f"DANGER_ZONE({danger_zone['risk_level']})")
        for reason in danger_zone["reasons"]:
            notes.append(reason)
    
    # ====================
    # 1. منع الدخول في المناطق الخطرة
    # ====================
    if danger_zone["is_danger_zone"] and danger_zone["risk_level"] in ["HIGH", "MEDIUM"]:
        return ScenarioDecision(
            Action.NO_TRADE,
            Mode.NONE,
            f"🚫 DANGER ZONE DETECTED: {', '.join(danger_zone['reasons'][:2])}",
            0.9,
            danger_zone_detected=True,
            meta={
                "ctx": ctx.__dict__,
                "danger_zone": danger_zone,
                "food_print": food_print,
                "notes": notes
            }
        )
    
    # ====================
    # 2. كشف الصفقات الذهبية أولاً مع فود برنت محسن
    # ====================
    golden_trade, zone_strength = is_golden_trade_setup_enhanced(ctx, fvg, ob, sweep, food_print)
    
    if golden_trade and zone_strength >= STRONG_ZONE_THRESHOLD:
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
                reason=f"🏆 GOLDEN BUY: Strong trend + Displacement + Premium zone (ADX:{ctx.adx:.1f}, VolZ:{ctx.vol_z:.2f}, ZoneStr:{zone_strength:.2f})",
                confidence=min(0.95, 0.7 + zone_strength * 0.25),
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
                    "danger_zone": danger_zone,
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
                reason=f"🏆 GOLDEN SELL: Strong trend + Displacement + Premium zone (ADX:{ctx.adx:.1f}, VolZ:{ctx.vol_z:.2f}, ZoneStr:{zone_strength:.2f})",
                confidence=min(0.95, 0.7 + zone_strength * 0.25),
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
                    "danger_zone": danger_zone,
                    "notes": notes,
                    "golden_tp_levels": [tp1, tp2, tp3],
                    "golden_close_fractions": GOLDEN_TP_CLOSE_FRACTIONS
                }
            )
    
    # ====================
    # 3. منع التداول في الـ Chop
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
                danger_zone_detected=danger_zone["is_danger_zone"],
                meta={
                    "ctx": ctx.__dict__, 
                    "notes": notes, 
                    "food_print": food_print,
                    "danger_zone": danger_zone
                }
            )
    
    # ====================
    # 4. الدخول في المناطق القوية فقط (للسكالب)
    # ====================
    if zone_strength < MIN_ZONE_STRENGTH:
        return ScenarioDecision(
            Action.NO_TRADE,
            Mode.NONE,
            f"Zone too weak for scalp ({zone_strength:.2f} < {MIN_ZONE_STRENGTH})",
            0.4,
            zone_strength=zone_strength,
            danger_zone_detected=danger_zone["is_danger_zone"],
            meta={
                "ctx": ctx.__dict__,
                "food_print": food_print,
                "danger_zone": danger_zone,
                "notes": notes
            }
        )
    
    # ... باقي الكود كما هو مع إضافة danger_zone_detected ...
    
    return ScenarioDecision(
        Action.HOLD, 
        Mode.NONE, 
        f"Waiting for premium setup (Phase: {ctx.phase.value}, Bias: {ctx.bias}, ZoneStr:{zone_strength:.2f})", 
        0.45,
        zone_strength=zone_strength,
        danger_zone_detected=danger_zone["is_danger_zone"],
        meta={
            "ctx": ctx.__dict__, 
            "ob": ob, 
            "fvg": fvg, 
            "sweep": sweep, 
            "food_print": food_print,
            "danger_zone": danger_zone,
            "notes": notes
        }
    )

# =================== ENHANCED TRADE MANAGER ===================
class TradeManagerProfessional:
    """مدير الصفقات المحترف مع إدارة ديناميكية متقدمة"""
    
    def __init__(self):
        self.golden_tp_levels = []
        self.golden_close_fractions = []
        self.current_tp_level = 0
        self.promoted_to_trend = False
        self.entry_context = None
        self.danger_zone_detected = False
        self.smart_council = SmartTradeCouncilEnhanced()  # المجلس المحسّن
        self.prev_adx = None
        self.prev_vwap_bias = None
        self.consecutive_warnings = 0
        self.last_profit_pct = 0
        self.best_profit_pct = 0
        
    def update_context(self, ctx: SignalContext, footprint_data: Dict):
        """تحديث سياق السوق للإدارة الذكية مع فود برنت"""
        self.entry_context = ctx
        
        # كشف المناطق الخطرة المتقدم
        self.danger_zone_detected = self._detect_danger_zone_advanced(ctx, footprint_data)
        
        if self.danger_zone_detected:
            self.consecutive_warnings += 1
        else:
            self.consecutive_warnings = max(0, self.consecutive_warnings - 1)
    
    def _detect_danger_zone_advanced(self, ctx: SignalContext, footprint_data: Dict) -> bool:
        """كشف المناطق الخطرة المتقدم"""
        danger_signals = []
        
        # 1. تحذيرات فود برنت
        if footprint_data.get("ok", False):
            danger_zones = footprint_data.get("danger_zones", [])
            if danger_zones and any(dz["risk_level"] == "HIGH" for dz in danger_zones):
                danger_signals.append("FOOTPRINT_HIGH_RISK")
        
        # 2. RSI في أقصى المدى
        if ctx.rsi >= 90 or ctx.rsi <= 10:
            danger_signals.append("RSI_EXTREME")
            
        # 3. تغيير حاد في ADX
        if self.prev_adx is not None:
            adx_change = abs(ctx.adx - self.prev_adx) / (self.prev_adx + 1e-12)
            if adx_change > 0.4 and ctx.adx < 18:
                danger_signals.append("ADX_CRASH")
        
        # 4. اختراق VWAP مع حجم ضعيف
        current_bias = "above" if ctx.px > ctx.vwap else "below"
        if self.prev_vwap_bias is not None:
            if current_bias != self.prev_vwap_bias and ctx.vol_z < -0.8:
                danger_signals.append("VWAP_BREAK_WEAK_VOL")
        
        # 5. تحذيرات متتالية
        if self.consecutive_warnings >= 3:
            danger_signals.append("CONSECUTIVE_WARNINGS")
        
        # تحديث القيم السابقة
        self.prev_adx = ctx.adx
        self.prev_vwap_bias = current_bias
        
        return len(danger_signals) >= 2  # إشارتين خطيرتين أو أكثر
    
    def manage_golden_trade_advanced(self, px: float, entry_px: float, side: str, 
                                   tp_levels: list, close_fractions: list,
                                   ctx: SignalContext) -> dict:
        """إدارة الصفقات الذهبية مع 3 مستويات TP متقدم"""
        if not tp_levels or len(tp_levels) < 3:
            return {"action": "hold", "close_pct": 0.0, "reason": "No TP levels"}
        
        hit_tp1 = (px >= tp_levels[0]) if side == "long" else (px <= tp_levels[0])
        hit_tp2 = (px >= tp_levels[1]) if side == "long" else (px <= tp_levels[1])
        hit_tp3 = (px >= tp_levels[2]) if side == "long" else (px <= tp_levels[2])
        
        current_profit_pct = abs(px - entry_px) / entry_px * 100
        
        # تحديث أفضل ربح
        if current_profit_pct > self.best_profit_pct:
            self.best_profit_pct = current_profit_pct
        
        # TP3: إغلاق كامل
        if hit_tp3 and self.current_tp_level < 3:
            self.current_tp_level = 3
            return {
                "action": "close",
                "reason": "🏆 GOLDEN TP3 HIT - Maximum profit achieved",
                "close_pct": 1.0,
                "profit_pct": current_profit_pct
            }
        
        # TP2: إغلاق جزئي
        elif hit_tp2 and self.current_tp_level < 2:
            self.current_tp_level = 2
            return {
                "action": "partial",
                "reason": "🎯 GOLDEN TP2 HIT - Taking partial profit",
                "close_pct": close_fractions[1],
                "profit_pct": current_profit_pct
            }
        
        # TP1: إغلاق جزئي
        elif hit_tp1 and self.current_tp_level < 1:
            self.current_tp_level = 1
            return {
                "action": "partial",
                "reason": "✅ GOLDEN TP1 HIT - First profit taken",
                "close_pct": close_fractions[0],
                "profit_pct": current_profit_pct
            }
        
        # خروج استثنائي: فقدان الربح بعد الوصول للذروة
        if self.best_profit_pct > 1.5 and current_profit_pct < self.best_profit_pct * 0.5:
            return {
                "action": "close",
                "reason": "📉 Profit retracement - Protecting gains",
                "close_pct": 1.0,
                "profit_pct": current_profit_pct
            }
        
        # خروج استثنائي من المنطقة الخطرة
        if self.danger_zone_detected:
            return {
                "action": "close",
                "reason": "⚠️ DANGER ZONE DETECTED - Emergency exit",
                "close_pct": 1.0,
                "profit_pct": current_profit_pct
            }
        
        return {"action": "hold", "close_pct": 0.0, "reason": "Holding for next TP"}
    
    def manage_scalp_trade_professional(self, px: float, entry_px: float, side: str,
                                      ctx: SignalContext, zone_strength: float) -> dict:
        """إدارة الصفقات السكالب بشكل محترف بناءً على قوة المنطقة"""
        profit_pct = abs(px - entry_px) / entry_px * 100
        
        # تحديث أفضل ربح
        if profit_pct > self.best_profit_pct:
            self.best_profit_pct = profit_pct
        
        # المناطق القوية: نصبر أكثر
        if zone_strength > 0.7:
            min_target = MIN_SCALP_BPS * 1.8 / 10000 * 100  # 1.8x للقوة العالية
            max_target = MIN_SCALP_BPS * 3.0 / 10000 * 100
        # المناطق المتوسطة: هدف عادي
        elif zone_strength > 0.5:
            min_target = MIN_SCALP_BPS * 1.2 / 10000 * 100
            max_target = MIN_SCALP_BPS * 2.0 / 10000 * 100
        # المناطق الضعيفة: نخرج مبكراً
        else:
            min_target = MIN_SCALP_BPS * 0.8 / 10000 * 100
            max_target = MIN_SCALP_BPS * 1.2 / 10000 * 100
        
        # إغلاق عند تحقيق الهدف الأقصى
        if profit_pct >= max_target:
            return {
                "action": "close",
                "reason": f"🎯 MAX SCALP TARGET HIT ({profit_pct:.2f}%) - ZoneStr: {zone_strength:.2f}",
                "close_pct": 1.0,
                "profit_pct": profit_pct
            }
        
        # إغلاق جزئي عند تحقيق الهدف الأدنى (للمناطق المتوسطة والقوية)
        if profit_pct >= min_target and zone_strength > 0.5:
            close_pct = 0.5 if zone_strength > 0.7 else 0.3
            return {
                "action": "partial",
                "reason": f"✅ SCALP TARGET HIT ({profit_pct:.2f}%) - Partial exit",
                "close_pct": close_pct,
                "profit_pct": profit_pct
            }
        
        # خروج سريع إذا كانت المنطقة ضعيفة وفقدنا الربح
        if zone_strength < 0.5 and profit_pct > 0.3 and profit_pct < self.best_profit_pct * 0.7:
            return {
                "action": "close",
                "reason": "📉 Weak zone profit retracement",
                "close_pct": 1.0,
                "profit_pct": profit_pct
            }
        
        return {"action": "hold", "close_pct": 0.0, "reason": "Holding scalp position"}

# =================== ENHANCED POSITION MANAGEMENT ===================
def manage_open_position_professional(df: pd.DataFrame, current_price: float):
    """إدارة الصفقة المفتوحة ديناميكياً مع مجلس الإدارة المحسّن"""
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
    food_print = detect_food_print_advanced_v2(df, FOOD_PRINT_LOOKBACK)
    trade_manager.update_context(ctx, food_print)
    
    log_i(f"📊 TRADE STATUS: {side.upper()} | PnL: {profit_pct:.2f}% | Bars: {STATE['bars_in_trade']} | MaxPnL: {STATE['max_profit_pct']:.2f}%")
    
    # ====================
    # 1. إدارة الصفقات الذهبية (3 مستويات TP)
    # ====================
    if STATE.get("is_golden"):
        tm = STATE.get("trade_manager", {})
        tp_levels = tm.get("golden_tp_levels", [])
        close_fractions = tm.get("golden_close_fractions", [])
        
        if len(tp_levels) >= 3:
            action = trade_manager.manage_golden_trade_advanced(
                current_price, entry, side, 
                tp_levels, close_fractions, ctx
            )
            
            if action["action"] == "close":
                log_g(f"🏆 {action['reason']} - Profit: {action['profit_pct']:.2f}%")
                close_market_strict(action["reason"])
                return
            elif action["action"] == "partial":
                log_g(f"🎯 {action['reason']} - Closing {action['close_pct']*100:.0f}% at {action['profit_pct']:.2f}% profit")
                # هنا يمكن إضافة منطق الإغلاق الجزئي
                close_market_strict(action["reason"])
                return
    
    # ====================
    # 2. إدارة الصفقات السكالب المحترفة
    # ====================
    if STATE.get("mode") == "scalp":
        zone_strength = STATE.get("zone_strength", 0.5)
        action = trade_manager.manage_scalp_trade_professional(
            current_price, entry, side, ctx, zone_strength
        )
        
        if action["action"] == "close":
            log_g(f"🎯 {action['reason']}")
            close_market_strict(action["reason"])
            return
        elif action["action"] == "partial":
            log_g(f"✅ {action['reason']}")
            close_market_strict(action["reason"])
            return
    
    # ====================
    # 3. مجلس الإدارة الذكي المحسّن مع فود برنت
    # ====================
    trade_data = {
        "entry": entry,
        "current_price": current_price,
        "side": "long" if side == "buy" else "short",
        "sl_price": STATE.get("sl_price"),
        "entry_context": STATE.get("entry_context", {}),
        "current_candles": df
    }
    
    council_decision = trade_manager.smart_council.evaluate_trade(
        trade_data, ctx, food_print
    )
    
    if council_decision["action"] == "close" and council_decision["confidence"] > 0.75:
        log_w(f"🏛️ SMART COUNCIL DECISION: {council_decision['reason']}")
        close_market_strict(f"COUNCIL: {council_decision['reason']}")
        return
    elif council_decision["action"] == "take_profit" and profit_pct > 0.5:
        log_g(f"🏛️ SMART COUNCIL: Taking profit at {profit_pct:.2f}%")
        close_market_strict(f"COUNCIL_TAKE_PROFIT: {profit_pct:.2f}%")
        return
    
    # ====================
    # 4. كشف المناطق الخطرة والخروج الفوري
    # ====================
    if trade_manager.danger_zone_detected:
        log_d("⚠️ DANGER ZONE DETECTED - Emergency exit!")
        close_market_strict("DANGER_ZONE_EMERGENCY")
        return
    
    # ====================
    # 5. حماية الربح بعد الوصول للذروة
    # ====================
    if STATE["max_profit_pct"] > 1.0 and profit_pct < STATE["max_profit_pct"] * 0.6:
        log_w(f"📉 Profit retracement: {profit_pct:.2f}% from {STATE['max_profit_pct']:.2f}% max")
        close_market_strict("PROFIT_PROTECTION")
        return
    
    # ... باقي إدارة الصفقات (SL, Trailing, Breakeven, Time Stop) ...

# =================== MAIN EXECUTION ===================
if __name__ == "__main__":
    log_banner("PROFESSIONAL SCENARIO ENGINE BOT v8.0")
    
    # تحميل الحالة السابقة
    state = load_state() or {}
    state.setdefault("in_position", False)
    
    if RESUME_ON_RESTART:
        try:
            pass
        except Exception as e:
            log_w(f"resume error: {e}")
    
    # عرض إعدادات البوت
    print(colored(f"🔥 MODE: {'LIVE' if MODE_LIVE else 'PAPER'} • {SYMBOL} • {INTERVAL}", "yellow"))
    print(colored(f"💰 RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x", "yellow"))
    print(colored(f"🧠 ENGINE: Professional Scenario Engine v8.0", "yellow"))
    print(colored(f"🏛️ SMART COUNCIL: 5-Member with Footprint Analyst", "yellow"))
    print(colored(f"🏆 GOLDEN TRADES: 3-Level TP ({GOLDEN_TP_LEVELS}%)", "yellow"))
    print(colored(f"🗺️ FOOD PRINT: Advanced V2 Detection", "yellow"))
    print(colored(f"⚠️ DANGER ZONE: Advanced Detection System", "yellow"))
    print(colored(f"⚡ SCALP MANAGEMENT: Professional Zone-Based TP", "yellow"))
    print(colored(f"🛡️ PROTECTION: Danger Zone + Profit Protection + Smart Council", "yellow"))
    print(colored(f"🚀 EXECUTION: {'ACTIVE' if EXECUTE_ORDERS and not DRY_RUN else 'SIMULATION'}", "yellow"))
    
    # بدء الخيوط
    import threading
    threading.Thread(target=trade_loop_professional, daemon=True).start()
    
    # تشغيل الخادم
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)

# -*- coding: utf-8 -*-
"""
SUI ULTRA PRO AI BOT - النسخة المحترفة الذكية المتقدمة
• نظام الصياد المحترف مع محرك السيناريوهات المتقدم
• مجلس الإدارة المدعوم بمنظومة Footprint/Order-Flow المتقدمة
• إدارة صفقات ذكية ديناميكية مع كشف مبكر للمناطق الخطأ
• نظام جني أرباح ذكي (3 مستويات للذهب / مستوى للسكالب)
• مناطق دخول ذهبية محسوبة علمياً
• نظام الخروج الذكي من المناطق الخطأ
• متكامل مع BingX & Bybit
"""

import os, time, math, random, signal, sys, traceback, logging, json
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List, Union
import pandas as pd
import numpy as np
import ccxt
from flask import Flask, jsonify
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from collections import deque, defaultdict
import statistics

# ============================================
# محرك السيناريوهات المتقدم (Decision Layer)
# ============================================

class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    NO_TRADE = "NO_TRADE"
    CLOSE = "CLOSE"

class Mode(str, Enum):
    SCALP = "SCALP"
    TREND = "TREND"
    SWING = "SWING"
    NONE = "NONE"

class Phase(str, Enum):
    ACCUMULATION = "ACCUMULATION"
    EXPANSION = "EXPANSION"
    TREND = "TREND"
    DISTRIBUTION = "DISTRIBUTION"
    EXHAUSTION = "EXHAUSTION"
    CHOP = "CHOP"
    LIQUIDITY_SWEEP = "LIQUIDITY_SWEEP"
    STOP_HUNT = "STOP_HUNT"
    UNKNOWN = "UNKNOWN"

class ZoneType(str, Enum):
    GOLDEN_BOTTOM = "GOLDEN_BOTTOM"
    GOLDEN_TOP = "GOLDEN_TOP"
    ORDER_BLOCK = "ORDER_BLOCK"
    FVG = "FAIR_VALUE_GAP"
    LIQUIDITY_ZONE = "LIQUIDITY_ZONE"
    BREAKOUT_RETEST = "BREAKOUT_RETEST"
    SMC_LEVEL = "SMC_LEVEL"
    NONE = "NONE"

@dataclass
class ZoneInfo:
    type: ZoneType
    price_start: float
    price_end: float
    strength: float  # 1-10
    confidence: float  # 0-1
    reasons: List[str]
    timestamp: int

@dataclass
class MarketContext:
    phase: Phase
    bias: str  # BULLISH, BEARISH, NEUTRAL
    strength: float  # 0-10
    volatility: float  # ATR نسبي
    volume_profile: str  # ACCUMULATION, DISTRIBUTION, NEUTRAL
    key_levels: List[float]
    displacement_detected: bool
    zones: List[ZoneInfo]

@dataclass
class TradeDecision:
    action: Action
    mode: Mode
    entry_zone: ZoneInfo
    confidence: float  # 0-1
    reasons: List[str]
    tp_levels: List[float]  # مستويات جني الأرباح كنسبة مئوية
    tp_weights: List[float]  # أوزان إغلاق كل مستوى
    sl_pct: float  # نسبة وقف الخسارة
    min_target_pips: int  # الحد الأدنى للنقاط المستهدفة
    trail_config: Dict[str, float]
    risk_factor: float  # عامل المخاطرة (0.5-2.0)

class AdvancedScenarioEngine:
    """محرك السيناريوهات المتقدم - صياد محترف"""
    
    def __init__(self):
        self.historical_data = deque(maxlen=500)
        self.zone_memory = deque(maxlen=20)
        self.trade_log = deque(maxlen=100)
        self.performance_stats = {
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'consecutive_wins': 0,
            'consecutive_losses': 0
        }
        
    def analyze_market_structure(self, df: pd.DataFrame) -> MarketContext:
        """تحليل متقدم لهيكل السوق"""
        try:
            if len(df) < 100:
                return MarketContext(
                    phase=Phase.UNKNOWN,
                    bias="NEUTRAL",
                    strength=0.0,
                    volatility=0.0,
                    volume_profile="NEUTRAL",
                    key_levels=[],
                    displacement_detected=False,
                    zones=[]
                )
            
            close = df['close'].astype(float)
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            volume = df['volume'].astype(float)
            
            # 1. تحديد المرحلة (Phase)
            phase = self._detect_market_phase(df)
            
            # 2. تحليل الاتجاه والقوة
            bias, strength = self._analyze_trend_bias(df)
            
            # 3. تحليل التقلب
            volatility = self._calculate_volatility(df)
            
            # 4. تحليل الحجم
            volume_profile = self._analyze_volume_profile(df)
            
            # 5. تحديد المستويات الرئيسية
            key_levels = self._find_key_levels(df)
            
            # 6. كشف Displacement (1.6×ATR)
            displacement_detected = self._detect_displacement(df)
            
            # 7. تحديد المناطق الذكية
            zones = self._scan_smart_zones(df)
            
            return MarketContext(
                phase=phase,
                bias=bias,
                strength=strength,
                volatility=volatility,
                volume_profile=volume_profile,
                key_levels=key_levels,
                displacement_detected=displacement_detected,
                zones=zones
            )
            
        except Exception as e:
            print(f"❌ خطأ في تحليل هيكل السوق: {e}")
            return MarketContext(
                phase=Phase.UNKNOWN,
                bias="NEUTRAL",
                strength=0.0,
                volatility=0.0,
                volume_profile="NEUTRAL",
                key_levels=[],
                displacement_detected=False,
                zones=[]
            )
    
    def _detect_market_phase(self, df: pd.DataFrame) -> Phase:
        """كشف مرحلة السوق بدقة"""
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        volume = df['volume'].astype(float)
        
        # حساب المؤشرات
        atr = self._calculate_atr(df, 14)
        adx = self._calculate_adx(df, 14)
        
        # 1. التراكم (Accumulation)
        if self._is_accumulation(df):
            return Phase.ACCUMULATION
        
        # 2. التوسع (Expansion)
        if self._is_expansion(df):
            return Phase.EXPANSION
        
        # 3. الاتجاه (Trend)
        if adx > 25:
            return Phase.TREND
        
        # 4. التوزيع (Distribution)
        if self._is_distribution(df):
            return Phase.DISTRIBUTION
        
        # 5. الإرهاق (Exhaustion)
        if self._is_exhaustion(df):
            return Phase.EXHAUSTION
        
        # 6. التذبذب (Chop)
        if adx < 18:
            return Phase.CHOP
        
        # 7. كنس السيولة (Liquidity Sweep)
        if self._is_liquidity_sweep(df):
            return Phase.LIQUIDITY_SWEEP
        
        # 8. صيد الوقفات (Stop Hunt)
        if self._is_stop_hunt(df):
            return Phase.STOP_HUNT
        
        return Phase.UNKNOWN
    
    def _is_accumulation(self, df: pd.DataFrame) -> bool:
        """كشف مرحلة التراكم"""
        if len(df) < 30:
            return False
        
        recent = df.tail(20)
        close = recent['close'].astype(float)
        
        # نطاق تداول ضيق مع تقلص في الحجم
        price_range = (close.max() - close.min()) / close.mean()
        volume_declining = recent['volume'].astype(float).pct_change().mean() < -0.1
        
        return price_range < 0.02 and volume_declining
    
    def _is_expansion(self, df: pd.DataFrame) -> bool:
        """كشف مرحلة التوسع"""
        if len(df) < 30:
            return False
        
        recent = df.tail(10)
        close = recent['close'].astype(float)
        volume = recent['volume'].astype(float)
        
        # حركة سعر قوية مع زيادة في الحجم
        price_change = abs((close.iloc[-1] - close.iloc[0]) / close.iloc[0])
        volume_spike = volume.iloc[-1] > volume.mean() * 1.5
        
        return price_change > 0.03 and volume_spike
    
    def _analyze_trend_bias(self, df: pd.DataFrame) -> Tuple[str, float]:
        """تحليل تحيز وقوة الاتجاه"""
        if len(df) < 50:
            return "NEUTRAL", 0.0
        
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        
        # المتوسطات المتحركة
        ema_20 = close.ewm(span=20).mean()
        ema_50 = close.ewm(span=50).mean()
        ema_200 = close.ewm(span=200).mean()
        
        # ADX و DI
        adx, plus_di, minus_di = self._calculate_dmi(df, 14)
        
        # RSI
        rsi = self._calculate_rsi(close, 14)
        
        # تحليل متعدد الأبعاد
        bullish_signals = 0
        bearish_signals = 0
        
        # إشارات صاعدة
        if ema_20.iloc[-1] > ema_50.iloc[-1] > ema_200.iloc[-1]:
            bullish_signals += 2
        if plus_di.iloc[-1] > minus_di.iloc[-1] and adx.iloc[-1] > 20:
            bullish_signals += 2
        if rsi.iloc[-1] > 50 and rsi.iloc[-1] < 70:
            bullish_signals += 1
        if close.iloc[-1] > ema_20.iloc[-1]:
            bullish_signals += 1
        
        # إشارات هابطة
        if ema_20.iloc[-1] < ema_50.iloc[-1] < ema_200.iloc[-1]:
            bearish_signals += 2
        if minus_di.iloc[-1] > plus_di.iloc[-1] and adx.iloc[-1] > 20:
            bearish_signals += 2
        if rsi.iloc[-1] < 50 and rsi.iloc[-1] > 30:
            bearish_signals += 1
        if close.iloc[-1] < ema_20.iloc[-1]:
            bearish_signals += 1
        
        # تحديد التحيز والقوة
        if bullish_signals - bearish_signals >= 3:
            bias = "BULLISH"
            strength = min(10.0, (bullish_signals / 8.0) * 10)
        elif bearish_signals - bullish_signals >= 3:
            bias = "BEARISH"
            strength = min(10.0, (bearish_signals / 8.0) * 10)
        else:
            bias = "NEUTRAL"
            strength = max(bullish_signals, bearish_signals) / 8.0 * 10
        
        return bias, strength
    
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """حساب التقلب النسبي"""
        if len(df) < 20:
            return 0.0
        
        atr = self._calculate_atr(df, 14)
        current_atr = atr.iloc[-1]
        avg_atr = atr.mean()
        
        if avg_atr == 0:
            return 0.0
        
        return min(10.0, (current_atr / avg_atr) * 5)
    
    def _analyze_volume_profile(self, df: pd.DataFrame) -> str:
        """تحليل توزيع الحجم"""
        if len(df) < 50:
            return "NEUTRAL"
        
        volume = df['volume'].astype(float)
        close = df['close'].astype(float)
        
        # تحليل VWAP
        typical_price = (df['high'].astype(float) + df['low'].astype(float) + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        current_price = close.iloc[-1]
        current_vwap = vwap.iloc[-1]
        
        # نسبة الحجم فوق/تحت السعر
        recent_volume = volume.tail(20)
        price_up = close.tail(20).diff() > 0
        up_volume = recent_volume[price_up].sum()
        down_volume = recent_volume[~price_up].sum()
        
        if current_price > current_vwap and up_volume > down_volume * 1.5:
            return "ACCUMULATION"
        elif current_price < current_vwap and down_volume > up_volume * 1.5:
            return "DISTRIBUTION"
        else:
            return "NEUTRAL"
    
    def _find_key_levels(self, df: pd.DataFrame) -> List[float]:
        """إيجاد مستويات الدعم والمقاومة الرئيسية"""
        if len(df) < 100:
            return []
        
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        
        # استخراج القمم والقيعان
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(df) - 2):
            # قمة
            if (high.iloc[i] > high.iloc[i-1] and 
                high.iloc[i] > high.iloc[i-2] and
                high.iloc[i] > high.iloc[i+1] and
                high.iloc[i] > high.iloc[i+2]):
                swing_highs.append(high.iloc[i])
            
            # قاع
            if (low.iloc[i] < low.iloc[i-1] and 
                low.iloc[i] < low.iloc[i-2] and
                low.iloc[i] < low.iloc[i+1] and
                low.iloc[i] < low.iloc[i+2]):
                swing_lows.append(low.iloc[i])
        
        # دمج المستويات القريبة
        merged_highs = self._merge_nearby_levels(swing_highs, tolerance=0.002)
        merged_lows = self._merge_nearby_levels(swing_lows, tolerance=0.002)
        
        key_levels = merged_highs + merged_lows
        return sorted(key_levels)[-10:]  # آخر 10 مستويات
    
    def _merge_nearby_levels(self, levels: List[float], tolerance: float = 0.002) -> List[float]:
        """دمج المستويات القريبة من بعضها"""
        if not levels:
            return []
        
        sorted_levels = sorted(levels)
        merged = []
        current_group = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            if abs(level - current_group[-1]) / current_group[-1] <= tolerance:
                current_group.append(level)
            else:
                merged.append(sum(current_group) / len(current_group))
                current_group = [level]
        
        if current_group:
            merged.append(sum(current_group) / len(current_group))
        
        return merged
    
    def _detect_displacement(self, df: pd.DataFrame) -> bool:
        """كشف Displacement (1.6×ATR)"""
        if len(df) < 5:
            return False
        
        atr = self._calculate_atr(df, 14)
        current_atr = atr.iloc[-1]
        
        # تحليل آخر 3 شمعات
        for i in range(-3, 0):
            idx = len(df) + i
            if idx < 0:
                continue
            
            candle = df.iloc[idx]
            high = float(candle['high'])
            low = float(candle['low'])
            candle_range = high - low
            
            if candle_range >= 1.6 * current_atr:
                return True
        
        return False
    
    def _scan_smart_zones(self, df: pd.DataFrame) -> List[ZoneInfo]:
        """مسح المناطق الذكية في السوق"""
        zones = []
        
        # 1. المناطق الذهبية (Golden Zones)
        golden_zones = self._find_golden_zones(df)
        zones.extend(golden_zones)
        
        # 2. Order Blocks
        order_blocks = self._find_order_blocks(df)
        zones.extend(order_blocks)
        
        # 3. Fair Value Gaps
        fvg_zones = self._find_fvg_zones(df)
        zones.extend(fvg_zones)
        
        # 4. مناطق السيولة
        liquidity_zones = self._find_liquidity_zones(df)
        zones.extend(liquidity_zones)
        
        # 5. مناطق SMC
        smc_zones = self._find_smc_zones(df)
        zones.extend(smc_zones)
        
        # 6. مناطق إعادة الاختبار للاختراق
        retest_zones = self._find_breakout_retest_zones(df)
        zones.extend(retest_zones)
        
        return sorted(zones, key=lambda x: x.strength, reverse=True)[:5]  # أفضل 5 مناطق
    
    def _find_golden_zones(self, df: pd.DataFrame) -> List[ZoneInfo]:
        """إيجاد المناطق الذهبية باستخدام فيبوناتشي"""
        zones = []
        
        if len(df) < 50:
            return zones
        
        # البحث عن آخر حركة كبيرة
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        
        # أقرب قمة وقاع رئيسيين
        recent_data = df.tail(50)
        swing_high = recent_data['high'].max()
        swing_low = recent_data['low'].min()
        
        # مستويات فيبوناتشي الذهبية
        fib_levels = {
            0.618: "GOLDEN_RETRACEMENT",
            0.786: "GOLDEN_DEEP"
        }
        
        current_price = df['close'].iloc[-1]
        price_range = swing_high - swing_low
        
        for fib_level, zone_type in fib_levels.items():
            # للحركة الصاعدة
            retracement_level = swing_high - (price_range * fib_level)
            if abs(current_price - retracement_level) / current_price < 0.005:  # 0.5%
                zones.append(ZoneInfo(
                    type=ZoneType.GOLDEN_BOTTOM,
                    price_start=retracement_level * 0.995,
                    price_end=retracement_level * 1.005,
                    strength=8.5,
                    confidence=0.7,
                    reasons=[f"Fibonacci {fib_level} retracement from recent swing"],
                    timestamp=int(time.time())
                ))
            
            # للحركة الهابطة
            retracement_level = swing_low + (price_range * fib_level)
            if abs(current_price - retracement_level) / current_price < 0.005:
                zones.append(ZoneInfo(
                    type=ZoneType.GOLDEN_TOP,
                    price_start=retracement_level * 0.995,
                    price_end=retracement_level * 1.005,
                    strength=8.5,
                    confidence=0.7,
                    reasons=[f"Fibonacci {fib_level} retracement from recent swing"],
                    timestamp=int(time.time())
                ))
        
        return zones
    
    def _find_order_blocks(self, df: pd.DataFrame) -> List[ZoneInfo]:
        """إيجاد Order Blocks"""
        zones = []
        
        if len(df) < 10:
            return zones
        
        # البحث عن شموع الرفض مع حجم مرتفع
        for i in range(len(df) - 5, len(df) - 1):
            candle = df.iloc[i]
            next_candle = df.iloc[i + 1]
            
            open_price = float(candle['open'])
            close_price = float(candle['close'])
            volume = float(candle['volume'])
            avg_volume = df['volume'].astype(float).tail(20).mean()
            
            # Bullish Order Block: شمعة هابطة تليها شمعة صاعدة
            if (close_price < open_price and 
                float(next_candle['close']) > float(next_candle['open'])):
                
                zones.append(ZoneInfo(
                    type=ZoneType.ORDER_BLOCK,
                    price_start=min(open_price, close_price),
                    price_end=max(open_price, close_price),
                    strength=7.0 if volume > avg_volume else 5.0,
                    confidence=0.6 if volume > avg_volume else 0.4,
                    reasons=["Bullish Order Block detected"],
                    timestamp=int(time.time())
                ))
            
            # Bearish Order Block: شمعة صاعدة تليها شمعة هابطة
            elif (close_price > open_price and 
                  float(next_candle['close']) < float(next_candle['open'])):
                
                zones.append(ZoneInfo(
                    type=ZoneType.ORDER_BLOCK,
                    price_start=min(open_price, close_price),
                    price_end=max(open_price, close_price),
                    strength=7.0 if volume > avg_volume else 5.0,
                    confidence=0.6 if volume > avg_volume else 0.4,
                    reasons=["Bearish Order Block detected"],
                    timestamp=int(time.time())
                ))
        
        return zones
    
    def _find_fvg_zones(self, df: pd.DataFrame) -> List[ZoneInfo]:
        """إيجاد Fair Value Gaps"""
        zones = []
        
        if len(df) < 4:
            return zones
        
        # تحليل آخر 3 شمعات لـ FVG
        for i in range(len(df) - 4, len(df) - 1):
            if i + 3 >= len(df):
                continue
            
            candle1 = df.iloc[i]
            candle2 = df.iloc[i + 1]
            candle3 = df.iloc[i + 2]
            
            high1 = float(candle1['high'])
            low1 = float(candle1['low'])
            low2 = float(candle2['low'])
            high2 = float(candle2['high'])
            low3 = float(candle3['low'])
            
            # Bullish FVG
            if low2 > high1:
                zones.append(ZoneInfo(
                    type=ZoneType.FVG,
                    price_start=high1,
                    price_end=low2,
                    strength=6.5,
                    confidence=0.65,
                    reasons=["Bullish Fair Value Gap detected"],
                    timestamp=int(time.time())
                ))
            
            # Bearish FVG
            if high2 < low1:
                zones.append(ZoneInfo(
                    type=ZoneType.FVG,
                    price_start=high2,
                    price_end=low1,
                    strength=6.5,
                    confidence=0.65,
                    reasons=["Bearish Fair Value Gap detected"],
                    timestamp=int(time.time())
                ))
        
        return zones
    
    def _find_liquidity_zones(self, df: pd.DataFrame) -> List[ZoneInfo]:
        """إيجاد مناطق السيولة"""
        zones = []
        
        if len(df) < 30:
            return zones
        
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        
        # البحث عن المستويات التي تم اختراقها حديثاً
        recent_highs = high.tail(20)
        recent_lows = low.tail(20)
        
        current_price = df['close'].iloc[-1]
        
        # مناطق السيولة فوق السعر (للبيع)
        for level in recent_highs:
            if level > current_price * 1.01:  # فوق السعر بـ 1%
                zones.append(ZoneInfo(
                    type=ZoneType.LIQUIDITY_ZONE,
                    price_start=level * 0.995,
                    price_end=level * 1.005,
                    strength=6.0,
                    confidence=0.6,
                    reasons=["Liquidity zone above price"],
                    timestamp=int(time.time())
                ))
        
        # مناطق السيولة تحت السعر (للشراء)
        for level in recent_lows:
            if level < current_price * 0.99:  # تحت السعر بـ 1%
                zones.append(ZoneInfo(
                    type=ZoneType.LIQUIDITY_ZONE,
                    price_start=level * 0.995,
                    price_end=level * 1.005,
                    strength=6.0,
                    confidence=0.6,
                    reasons=["Liquidity zone below price"],
                    timestamp=int(time.time())
                ))
        
        return zones
    
    def _find_smc_zones(self, df: pd.DataFrame) -> List[ZoneInfo]:
        """إيجاد مناطق SMC (Smart Money Concepts)"""
        zones = []
        
        if len(df) < 40:
            return zones
        
        # تحليل الاختراقات الكاذبة (False Breakouts)
        recent = df.tail(30)
        highs = recent['high'].astype(float)
        lows = recent['low'].astype(float)
        
        # قمة وقاع مدروسين
        significant_high = highs.max()
        significant_low = lows.min()
        
        current_price = df['close'].iloc[-1]
        
        # SMC Buy Zone: اختراق كاذب للقاع
        if current_price > significant_low and lows.iloc[-1] < significant_low:
            zones.append(ZoneInfo(
                type=ZoneType.SMC_LEVEL,
                price_start=significant_low * 0.995,
                price_end=significant_low * 1.01,
                strength=8.0,
                confidence=0.75,
                reasons=["SMC Buy Zone: False breakdown"],
                timestamp=int(time.time())
            ))
        
        # SMC Sell Zone: اختراق كاذب للقمة
        if current_price < significant_high and highs.iloc[-1] > significant_high:
            zones.append(ZoneInfo(
                type=ZoneType.SMC_LEVEL,
                price_start=significant_high * 0.99,
                price_end=significant_high * 1.005,
                strength=8.0,
                confidence=0.75,
                reasons=["SMC Sell Zone: False breakout"],
                timestamp=int(time.time())
            ))
        
        return zones
    
    def _find_breakout_retest_zones(self, df: pd.DataFrame) -> List[ZoneInfo]:
        """إيجاد مناطق إعادة الاختبار للاختراقات"""
        zones = []
        
        if len(df) < 40:
            return zones
        
        # البحث عن اختراقات حديثة
        recent = df.tail(30)
        earlier = df.iloc[-60:-30] if len(df) > 60 else df.iloc[:30]
        
        recent_high = recent['high'].max()
        recent_low = recent['low'].min()
        earlier_high = earlier['high'].max()
        earlier_low = earlier['low'].min()
        
        current_price = df['close'].iloc[-1]
        
        # اختراق صاعد وإعادة اختبار
        if recent_high > earlier_high * 1.01:
            retest_zone = earlier_high
            if abs(current_price - retest_zone) / current_price < 0.01:
                zones.append(ZoneInfo(
                    type=ZoneType.BREAKOUT_RETEST,
                    price_start=retest_zone * 0.995,
                    price_end=retest_zone * 1.01,
                    strength=7.5,
                    confidence=0.7,
                    reasons=["Breakout retest zone"],
                    timestamp=int(time.time())
                ))
        
        # اختراق هابط وإعادة اختبار
        if recent_low < earlier_low * 0.99:
            retest_zone = earlier_low
            if abs(current_price - retest_zone) / current_price < 0.01:
                zones.append(ZoneInfo(
                    type=ZoneType.BREAKOUT_RETEST,
                    price_start=retest_zone * 0.99,
                    price_end=retest_zone * 1.005,
                    strength=7.5,
                    confidence=0.7,
                    reasons=["Breakdown retest zone"],
                    timestamp=int(time.time())
                ))
        
        return zones
    
    def _is_liquidity_sweep(self, df: pd.DataFrame) -> bool:
        """كشف كنس السيولة"""
        if len(df) < 10:
            return False
        
        recent = df.tail(5)
        highs = recent['high'].astype(float)
        lows = recent['low'].astype(float)
        
        # البحث عن ذيول طويلة تم محوها
        for i in range(len(recent)):
            candle = recent.iloc[i]
            high = float(candle['high'])
            low = float(candle['low'])
            close = float(candle['close'])
            
            upper_wick = high - max(float(candle['open']), close)
            lower_wick = min(float(candle['open']), close) - low
            
            # ذيل علوي طويل تم محوه في الشمعة التالية
            if upper_wick > (high - low) * 0.4 and i < len(recent) - 1:
                next_candle = recent.iloc[i + 1]
                if float(next_candle['close']) < high * 0.995:
                    return True
            
            # ذيل سفلي طويل تم محوه في الشمعة التالية
            if lower_wick > (high - low) * 0.4 and i < len(recent) - 1:
                next_candle = recent.iloc[i + 1]
                if float(next_candle['close']) > low * 1.005:
                    return True
        
        return False
    
    def _is_stop_hunt(self, df: pd.DataFrame) -> bool:
        """كشف صيد الوقفات"""
        if len(df) < 15:
            return False
        
        recent = df.tail(10)
        highs = recent['high'].astype(float)
        lows = recent['low'].astype(float)
        
        # قمم وقيعان واضحة
        swing_high = highs.max()
        swing_low = lows.min()
        
        # كسور سريعة مع عودة
        for i in range(len(recent) - 2):
            candle = recent.iloc[i]
            next_candle = recent.iloc[i + 1]
            
            high = float(candle['high'])
            low = float(candle['low'])
            
            # كسر قمة مع عودة سريعة
            if high > swing_high * 0.999:
                if float(next_candle['close']) < swing_high * 0.995:
                    return True
            
            # كسر قاع مع عودة سريعة
            if low < swing_low * 1.001:
                if float(next_candle['close']) > swing_low * 1.005:
                    return True
        
        return False
    
    def _is_distribution(self, df: pd.DataFrame) -> bool:
        """كشف مرحلة التوزيع"""
        if len(df) < 30:
            return False
        
        recent = df.tail(20)
        close = recent['close'].astype(float)
        
        # نطاق تداول عريض مع تقلب في الحجم
        price_range = (close.max() - close.min()) / close.mean()
        volume_volatile = recent['volume'].astype(float).pct_change().std() > 0.3
        
        return price_range > 0.025 and volume_volatile
    
    def _is_exhaustion(self, df: pd.DataFrame) -> bool:
        """كشف مرحلة الإرهاق"""
        if len(df) < 20:
            return False
        
        recent = df.tail(10)
        
        # حركة سعرية قوية مع حجم ضعيف
        price_change = abs((float(recent['close'].iloc[-1]) - float(recent['close'].iloc[0])) / float(recent['close'].iloc[0]))
        volume_ratio = float(recent['volume'].iloc[-1]) / recent['volume'].astype(float).mean()
        
        return price_change > 0.05 and volume_ratio < 0.8
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """حساب Average True Range"""
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'].astype(float)
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """حساب ADX"""
        if len(df) < period * 2:
            return 0.0
        
        try:
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            close = df['close'].astype(float)
            
            # حساب +DM و -DM
            up_move = high.diff()
            down_move = low.diff().abs()
            
            plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
            minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
            
            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # المتوسطات المتحركة
            atr = tr.rolling(period).mean()
            plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
            
            # حساب DX و ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(period).mean()
            
            return float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0
        except:
            return 0.0
    
    def _calculate_dmi(self, df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """حساب DMI (+DI, -DI, ADX)"""
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'].astype(float)
        
        # حساب +DM و -DM
        up_move = high.diff()
        down_move = low.diff().abs()
        
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # المتوسطات المتحركة
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        # حساب DX و ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx, plus_di, minus_di
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """حساب RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / (avg_loss + 1e-12)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def make_decision(self, df: pd.DataFrame, context: MarketContext, 
                     current_position: Optional[Dict] = None) -> TradeDecision:
        """صنع قرار تداول ذكي بناءً على السياق"""
        
        current_price = float(df['close'].iloc[-1])
        
        # 1. إذا كان هناك صفقة مفتوحة: إدارة الصفقة
        if current_position and current_position.get('open', False):
            return self._manage_open_trade(df, context, current_position)
        
        # 2. البحث عن فرص دخول جديدة
        return self._find_entry_opportunity(df, context)
    
    def _manage_open_trade(self, df: pd.DataFrame, context: MarketContext, 
                          position: Dict) -> TradeDecision:
        """إدارة صفقة مفتوحة بذكاء"""
        
        current_price = float(df['close'].iloc[-1])
        entry_price = position.get('entry_price', current_price)
        position_side = position.get('side', 'BUY')
        
        # حساب الربح/الخسارة
        if position_side == 'BUY':
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
        
        position['current_pnl'] = pnl_pct
        
        # 1. كشف المنطقة الخطأ والخروج المبكر
        if self._is_bad_entry_zone(df, context, position):
            return TradeDecision(
                action=Action.CLOSE,
                mode=Mode.NONE,
                entry_zone=ZoneInfo(type=ZoneType.NONE, price_start=0, price_end=0, 
                                  strength=0, confidence=0, reasons=[], timestamp=0),
                confidence=0.85,
                reasons=["Bad entry zone detected - early exit"],
                tp_levels=[],
                tp_weights=[],
                sl_pct=0,
                min_target_pips=0,
                trail_config={},
                risk_factor=1.0
            )
        
        # 2. تحليل قوة الصفقة الحالية
        trade_strength = self._analyze_trade_strength(df, context, position)
        
        # 3. ضبط أهداف الربح ديناميكياً
        adjusted_tp_levels, adjusted_weights = self._adjust_profit_targets(
            position, trade_strength, pnl_pct
        )
        
        # 4. قرار البقاء أو الخروج
        if self._should_close_trade(df, context, position, trade_strength):
            return TradeDecision(
                action=Action.CLOSE,
                mode=Mode.NONE,
                entry_zone=ZoneInfo(type=ZoneType.NONE, price_start=0, price_end=0, 
                                  strength=0, confidence=0, reasons=[], timestamp=0),
                confidence=0.8,
                reasons=["Trade management: target achieved or conditions changed"],
                tp_levels=[],
                tp_weights=[],
                sl_pct=0,
                min_target_pips=0,
                trail_config={},
                risk_factor=1.0
            )
        
        # 5. اقتراح تعديلات على الإدارة
        trail_config = self._suggest_trailing_adjustment(position, trade_strength)
        
        return TradeDecision(
            action=Action.HOLD,
            mode=Mode(position.get('mode', 'SCALP')),
            entry_zone=ZoneInfo(type=ZoneType.NONE, price_start=0, price_end=0, 
                              strength=0, confidence=0, reasons=[], timestamp=0),
            confidence=trade_strength.get('confidence', 0.6),
            reasons=["Trade is healthy - continue managing"],
            tp_levels=adjusted_tp_levels,
            tp_weights=adjusted_weights,
            sl_pct=position.get('sl_pct', 2.0),
            min_target_pips=position.get('min_target_pips', 6),
            trail_config=trail_config,
            risk_factor=trade_strength.get('risk_factor', 1.0)
        )
    
    def _is_bad_entry_zone(self, df: pd.DataFrame, context: MarketContext, 
                          position: Dict) -> bool:
        """كشف إذا كانت المنطقة دخول خاطئة"""
        
        current_price = float(df['close'].iloc[-1])
        entry_price = position.get('entry_price', current_price)
        position_side = position.get('side', 'BUY')
        
        # 1. خسارة سريعة وكبيرة (أكثر من 1.5% في وقت قصير)
        if position_side == 'BUY':
            loss_pct = ((entry_price - current_price) / entry_price) * 100
        else:
            loss_pct = ((current_price - entry_price) / entry_price) * 100
        
        time_in_trade = time.time() - position.get('entry_time', time.time())
        
        if loss_pct > 1.5 and time_in_trade < 300:  # 5 دقائق
            return True
        
        # 2. تحول سريع في تحيز السوق
        if position_side == 'BUY' and context.bias == "BEARISH":
            if context.strength > 6.0:
                return True
        
        if position_side == 'SELL' and context.bias == "BULLISH":
            if context.strength > 6.0:
                return True
        
        # 3. كسر مناطق دعم/مقاومة رئيسية
        key_levels = context.key_levels
        if position_side == 'BUY':
            support_broken = any(level > current_price for level in key_levels 
                               if level < entry_price * 0.99)
            if support_broken:
                return True
        
        if position_side == 'SELL':
            resistance_broken = any(level < current_price for level in key_levels 
                                  if level > entry_price * 1.01)
            if resistance_broken:
                return True
        
        # 4. حجم ضعيف يدعم الاتجاه
        recent_volume = df['volume'].astype(float).tail(5).mean()
        avg_volume = df['volume'].astype(float).tail(20).mean()
        
        if recent_volume < avg_volume * 0.7:
            # اتجاه بدون حجم يدعمه = ضعيف
            return True
        
        return False
    
    def _analyze_trade_strength(self, df: pd.DataFrame, context: MarketContext, 
                               position: Dict) -> Dict[str, float]:
        """تحليل قوة الصفقة الحالية"""
        
        current_price = float(df['close'].iloc[-1])
        entry_price = position.get('entry_price', current_price)
        position_side = position.get('side', 'BUY')
        
        strength_score = 0.0
        factors = []
        
        # 1. اتجاه السوق مع الصفقة
        market_alignment = 0.0
        if (position_side == 'BUY' and context.bias == "BULLISH") or \
           (position_side == 'SELL' and context.bias == "BEARISH"):
            market_alignment = 1.0
            factors.append("Market alignment: Strong")
        
        # 2. حجم التداول
        recent_volume = df['volume'].astype(float).tail(5).mean()
        avg_volume = df['volume'].astype(float).tail(20).mean()
        volume_ratio = recent_volume / avg_volume
        
        if volume_ratio > 1.2:
            volume_factor = 0.8
            factors.append(f"Volume support: {volume_ratio:.2f}x")
        elif volume_ratio > 0.8:
            volume_factor = 0.5
            factors.append(f"Volume: Normal")
        else:
            volume_factor = 0.2
            factors.append(f"Weak volume: {volume_ratio:.2f}x")
        
        # 3. قوة الاتجاه
        trend_strength = min(1.0, context.strength / 10.0)
        
        # 4. السيولة والنطاق
        atr = self._calculate_atr(df, 14).iloc[-1]
        price_range = (df['high'].astype(float).max() - df['low'].astype(float).min()) / current_price
        
        if price_range < 0.02:  # نطاق ضيق
            range_factor = 0.3
            factors.append("Narrow range")
        elif price_range > 0.05:  # نطاق واسع
            range_factor = 0.8
            factors.append("Wide range - good liquidity")
        else:
            range_factor = 0.6
        
        # 5. الربح الحالي
        if position_side == 'BUY':
            current_profit = ((current_price - entry_price) / entry_price) * 100
        else:
            current_profit = ((entry_price - current_price) / entry_price) * 100
        
        profit_factor = min(1.0, current_profit / 5.0)  # كل 5% ربح = عامل 1.0
        
        # حساب النتيجة النهائية
        strength_score = (
            market_alignment * 0.3 +
            volume_factor * 0.2 +
            trend_strength * 0.2 +
            range_factor * 0.15 +
            profit_factor * 0.15
        )
        
        # عامل المخاطرة بناءً على القوة
        risk_factor = 1.5 if strength_score > 0.7 else 1.0
        
        return {
            'score': strength_score,
            'confidence': strength_score,
            'risk_factor': risk_factor,
            'factors': factors
        }
    
    def _adjust_profit_targets(self, position: Dict, trade_strength: Dict, 
                              current_profit: float) -> Tuple[List[float], List[float]]:
        """ضبط أهداف الربح ديناميكياً"""
        
        original_tp_levels = position.get('tp_levels', [])
        original_weights = position.get('tp_weights', [])
        
        if not original_tp_levels:
            # إذا كانت صفقة سكالب: هدف واحد
            if position.get('mode') == 'SCALP':
                min_pips = position.get('min_target_pips', 6)
                tp_level = [min_pips / 10000 * 100]  # تحويل لنسبة مئوية
                weights = [1.0]
                return tp_level, weights
            
            # إذا كانت صفقة ترند: 3 مستويات
            base_target = 0.8  # 0.8%
            tp_levels = [base_target, base_target * 2, base_target * 3]
            weights = [0.3, 0.3, 0.4]
            return tp_levels, weights
        
        # تعديل بناءً على قوة الصفقة
        strength_score = trade_strength.get('score', 0.5)
        adjustment_factor = 0.5 + strength_score  # 0.5-1.5
        
        adjusted_levels = [level * adjustment_factor for level in original_tp_levels]
        
        # إذا كانت القوة عالية، نقلل من الوزن الأولي ونزيد الأوزان اللاحقة
        if strength_score > 0.7:
            if len(original_weights) >= 3:
                adjusted_weights = [0.2, 0.3, 0.5]
            else:
                adjusted_weights = original_weights
        else:
            adjusted_weights = original_weights
        
        return adjusted_levels, adjusted_weights
    
    def _should_close_trade(self, df: pd.DataFrame, context: MarketContext, 
                           position: Dict, trade_strength: Dict) -> bool:
        """تحديد إذا كان يجب إغلاق الصفقة"""
        
        current_price = float(df['close'].iloc[-1])
        entry_price = position.get('entry_price', current_price)
        position_side = position.get('side', 'BUY')
        tp_levels = position.get('tp_levels', [])
        
        # حساب الربح الحالي
        if position_side == 'BUY':
            current_profit = ((current_price - entry_price) / entry_price) * 100
        else:
            current_profit = ((entry_price - current_price) / entry_price) * 100
        
        # 1. تحقيق هدف الربح
        if tp_levels:
            for tp_level in tp_levels:
                if current_profit >= tp_level * 0.9:  # 90% من الهدف
                    return True
        
        # 2. تغير ظروف السوق بشكل كبير
        if position_side == 'BUY' and context.bias == "BEARISH":
            if context.strength > 7.0 and current_profit > 0.5:
                return True
        
        if position_side == 'SELL' and context.bias == "BULLISH":
            if context.strength > 7.0 and current_profit > 0.5:
                return True
        
        # 3. انخفاض قوة الصفقة
        if trade_strength.get('score', 0) < 0.3 and current_profit > 0.3:
            return True
        
        # 4. تحذيرات فنية
        rsi = self._calculate_rsi(df['close'].astype(float), 14).iloc[-1]
        
        if position_side == 'BUY' and rsi > 80:
            return True
        
        if position_side == 'SELL' and rsi < 20:
            return True
        
        return False
    
    def _suggest_trailing_adjustment(self, position: Dict, 
                                     trade_strength: Dict) -> Dict[str, float]:
        """اقتراح تعديلات على الوقف المتحرك"""
        
        strength_score = trade_strength.get('score', 0.5)
        position_side = position.get('side', 'BUY')
        
        # قاعدة الوقف المتحرك
        if strength_score > 0.7:
            # قوة عالية: وقف متحرك مريح
            trail_distance = 0.015  # 1.5%
            activation = 0.008  # تفعيل بعد 0.8%
        elif strength_score > 0.4:
            # قوة متوسطة: وقف متحرك متوسط
            trail_distance = 0.01  # 1.0%
            activation = 0.005  # تفعيل بعد 0.5%
        else:
            # قوة ضعيفة: وقف متحرك محكم
            trail_distance = 0.006  # 0.6%
            activation = 0.003  # تفعيل بعد 0.3%
        
        return {
            'trail_distance_pct': trail_distance,
            'activation_pct': activation,
            'adjustment_factor': strength_score
        }
    
    def _find_entry_opportunity(self, df: pd.DataFrame, 
                               context: MarketContext) -> TradeDecision:
        """البحث عن فرص دخول جديدة"""
        
        current_price = float(df['close'].iloc[-1])
        
        # 1. فلترة المناطق القوية فقط
        strong_zones = [
            zone for zone in context.zones 
            if zone.strength >= 7.0 and zone.confidence >= 0.65
        ]
        
        if not strong_zones:
            return TradeDecision(
                action=Action.NO_TRADE,
                mode=Mode.NONE,
                entry_zone=ZoneInfo(type=ZoneType.NONE, price_start=0, price_end=0, 
                                  strength=0, confidence=0, reasons=[], timestamp=0),
                confidence=0.3,
                reasons=["No strong entry zones detected"],
                tp_levels=[],
                tp_weights=[],
                sl_pct=0,
                min_target_pips=0,
                trail_config={},
                risk_factor=1.0
            )
        
        # 2. اختيار أفضل منطقة
        best_zone = max(strong_zones, key=lambda z: z.strength * z.confidence)
        
        # 3. تحديد نوع التداول (سكالب/ترند/سوينج)
        trade_mode = self._determine_trade_mode(df, context, best_zone)
        
        # 4. حساب أهداف الربح ونقاط الوقف
        tp_levels, tp_weights, sl_pct, min_pips = self._calculate_targets(
            trade_mode, best_zone, context
        )
        
        # 5. تحديد الإجراء
        action = self._determine_action(best_zone, current_price, context)
        
        # 6. حساب الثقة النهائية
        confidence = min(0.95, best_zone.confidence * 1.2)
        
        return TradeDecision(
            action=action,
            mode=trade_mode,
            entry_zone=best_zone,
            confidence=confidence,
            reasons=best_zone.reasons,
            tp_levels=tp_levels,
            tp_weights=tp_weights,
            sl_pct=sl_pct,
            min_target_pips=min_pips,
            trail_config=self._get_trail_config(trade_mode, best_zone.strength),
            risk_factor=self._calculate_risk_factor(best_zone, context)
        )
    
    def _determine_trade_mode(self, df: pd.DataFrame, context: MarketContext, 
                             zone: ZoneInfo) -> Mode:
        """تحديد نوع التداول بناءً على المنطقة والسياق"""
        
        # المناطق الذهبية والقوية = ترند
        if zone.type in [ZoneType.GOLDEN_BOTTOM, ZoneType.GOLDEN_TOP, ZoneType.SMC_LEVEL]:
            if context.phase in [Phase.TREND, Phase.EXPANSION]:
                return Mode.TREND
            else:
                return Mode.SWING
        
        # Order Blocks و FVG = سكالب محسن
        elif zone.type in [ZoneType.ORDER_BLOCK, ZoneType.FVG]:
            if context.phase == Phase.CHOP:
                return Mode.SCALP
            else:
                return Mode.SWING
        
        # مناطق أخرى = سكالب
        else:
            return Mode.SCALP
    
    def _calculate_targets(self, mode: Mode, zone: ZoneInfo, 
                          context: MarketContext) -> Tuple[List[float], List[float], float, int]:
        """حساب أهداف الربح ونقاط الوقف"""
        
        # قاعدة: كلما كانت المنطقة أقوى، كلما كانت الأهداف أكبر
        strength_multiplier = 1.0 + (zone.strength - 7.0) / 10.0
        
        if mode == Mode.TREND:
            # 3 مستويات للصفقات الذهبية
            base_target = 0.8 * strength_multiplier  # 0.8% مع تعديل القوة
            tp_levels = [base_target, base_target * 2, base_target * 3]
            tp_weights = [0.3, 0.3, 0.4]
            sl_pct = 1.5  # وقف خسارة 1.5%
            min_pips = 8  # 8 نقاط كحد أدنى
            
        elif mode == Mode.SWING:
            # مستويين للصفقات المتوسطة
            base_target = 0.6 * strength_multiplier
            tp_levels = [base_target, base_target * 1.8]
            tp_weights = [0.5, 0.5]
            sl_pct = 1.2
            min_pips = 6
            
        else:  # SCALP
            # مستوى واحد للصفقات السريعة
            base_target = 0.4 * strength_multiplier
            tp_levels = [base_target]
            tp_weights = [1.0]
            sl_pct = 0.8
            min_pips = 4
        
        # تعديل بناءً على تقلب السوق
        volatility_factor = 1.0 + (context.volatility - 5.0) / 10.0
        tp_levels = [level * volatility_factor for level in tp_levels]
        
        return tp_levels, tp_weights, sl_pct, min_pips
    
    def _determine_action(self, zone: ZoneInfo, current_price: float, 
                         context: MarketContext) -> Action:
        """تحديد الإجراء بناءً على المنطقة"""
        
        # التحقق من توافق السعر مع المنطقة
        in_zone = zone.price_start <= current_price <= zone.price_end
        
        if not in_zone:
            # إذا كان السعر قريباً جداً (ضمن 0.3%)
            price_diff_pct = abs(current_price - (zone.price_start + zone.price_end) / 2) / current_price
            if price_diff_pct < 0.003:
                in_zone = True
        
        if not in_zone:
            return Action.NO_TRADE
        
        # تحديد اتجاه الدخول بناءً على نوع المنطقة
        if zone.type in [ZoneType.GOLDEN_BOTTOM, ZoneType.SMC_LEVEL]:
            return Action.BUY
        
        elif zone.type in [ZoneType.GOLDEN_TOP]:
            return Action.SELL
        
        elif zone.type == ZoneType.ORDER_BLOCK:
            # تحليل Order Block للاتجاه
            zone_mid = (zone.price_start + zone.price_end) / 2
            if current_price > zone_mid:
                return Action.BUY
            else:
                return Action.SELL
        
        elif zone.type == ZoneType.FVG:
            # Fair Value Gap يشير لاتجاه
            if zone.reasons and "Bullish" in zone.reasons[0]:
                return Action.BUY
            elif zone.reasons and "Bearish" in zone.reasons[0]:
                return Action.SELL
        
        # التحقق من تحيز السوق
        if context.bias == "BULLISH" and zone.strength >= 7.0:
            return Action.BUY
        elif context.bias == "BEARISH" and zone.strength >= 7.0:
            return Action.SELL
        
        return Action.NO_TRADE
    
    def _get_trail_config(self, mode: Mode, zone_strength: float) -> Dict[str, float]:
        """الحصول على إعدادات الوقف المتحرك"""
        
        if mode == Mode.TREND:
            return {
                'enabled': True,
                'activation_pct': 0.8,
                'distance_pct': 0.015,
                'tighten_above': 2.0,
                'min_distance': 0.005
            }
        elif mode == Mode.SWING:
            return {
                'enabled': True,
                'activation_pct': 0.5,
                'distance_pct': 0.01,
                'tighten_above': 1.5,
                'min_distance': 0.003
            }
        else:  # SCALP
            return {
                'enabled': zone_strength > 7.5,
                'activation_pct': 0.3,
                'distance_pct': 0.006,
                'tighten_above': 1.0,
                'min_distance': 0.002
            }
    
    def _calculate_risk_factor(self, zone: ZoneInfo, context: MarketContext) -> float:
        """حساب عامل المخاطرة"""
        
        risk_factor = 1.0
        
        # زيادة المخاطرة للمناطق الذهبية القوية
        if zone.type in [ZoneType.GOLDEN_BOTTOM, ZoneType.GOLDEN_TOP, ZoneType.SMC_LEVEL]:
            risk_factor = 1.5
        
        # تقليل المخاطرة في مراحل التذبذب
        if context.phase == Phase.CHOP:
            risk_factor *= 0.7
        
        # زيادة المخاطرة مع قوة المنطقة
        risk_factor *= (0.8 + (zone.strength / 10.0))
        
        # تقليل المخاطرة مع ارتفاع التقلب
        if context.volatility > 7.0:
            risk_factor *= 0.8
        
        return min(2.0, max(0.5, risk_factor))  # تحديد نطاق 0.5-2.0

# ============================================
# تعزيز نظام Footprint/Order-Flow
# ============================================

class AdvancedFootprintAnalyzer:
    """محلل Footprint/Order-Flow المتقدم"""
    
    def __init__(self, depth_levels=10):
        self.depth_levels = depth_levels
        self.order_flow_history = deque(maxlen=100)
        self.volume_profile = defaultdict(float)
        self.delta_history = deque(maxlen=50)
        self.cvd_history = deque(maxlen=100)
        
    def analyze_orderbook(self, orderbook: Dict) -> Dict[str, Any]:
        """تحليل متقدم للكتيب الطلبات"""
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if not bids or not asks:
                return {"ok": False, "error": "Empty orderbook"}
            
            # حساب توزيع السيولة
            bid_liquidity = sum(bid[1] for bid in bids[:self.depth_levels])
            ask_liquidity = sum(ask[1] for ask in asks[:self.depth_levels])
            total_liquidity = bid_liquidity + ask_liquidity
            
            # نسبة السيولة
            liquidity_ratio = bid_liquidity / ask_liquidity if ask_liquidity > 0 else 1.0
            
            # البحث عن جدران السيولة
            bid_walls = self._find_liquidity_walls(bids, is_bid=True)
            ask_walls = self._find_liquidity_walls(asks, is_bid=False)
            
            # تحليل امتصاص السيولة
            absorption_analysis = self._analyze_absorption(bids, asks)
            
            # حساب CVD مبسط
            current_cvd = self._calculate_cvd(bids, asks)
            self.cvd_history.append(current_cvd)
            
            # اتجاه CVD
            cvd_trend = self._analyze_cvd_trend()
            
            return {
                "ok": True,
                "bid_liquidity": bid_liquidity,
                "ask_liquidity": ask_liquidity,
                "liquidity_ratio": liquidity_ratio,
                "bid_walls": bid_walls,
                "ask_walls": ask_walls,
                "absorption": absorption_analysis,
                "cvd": current_cvd,
                "cvd_trend": cvd_trend,
                "imbalance": self._calculate_imbalance(bids, asks)
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}
    
    def _find_liquidity_walls(self, orders: List, is_bid: bool = True, 
                             threshold_multiplier: float = 3.0) -> List[Dict]:
        """إيجاد جدران السيولة الكبيرة"""
        if not orders:
            return []
        
        # حساب متوسط الحجم
        volumes = [order[1] for order in orders[:self.depth_levels]]
        avg_volume = sum(volumes) / len(volumes)
        
        walls = []
        for price, volume in orders[:self.depth_levels]:
            if volume > avg_volume * threshold_multiplier:
                walls.append({
                    "price": price,
                    "volume": volume,
                    "size_vs_avg": volume / avg_volume,
                    "side": "bid" if is_bid else "ask"
                })
        
        return sorted(walls, key=lambda x: x['volume'], reverse=True)[:3]
    
    def _analyze_absorption(self, bids: List, asks: List) -> Dict[str, Any]:
        """تحليل امتصاص السيولة"""
        # تحليل امتصاص البيع (عند الشراء)
        bid_absorption = 0
        for price, volume in bids[:5]:
            if volume > sum(bid[1] for bid in bids[5:10]):
                bid_absorption += 1
        
        # تحليل امتصاص الشراء (عند البيع)
        ask_absorption = 0
        for price, volume in asks[:5]:
            if volume > sum(ask[1] for ask in asks[5:10]):
                ask_absorption += 1
        
        return {
            "bid_absorption": bid_absorption,
            "ask_absorption": ask_absorption,
            "strong_absorption": bid_absorption >= 3 or ask_absorption >= 3,
            "imbalance_direction": "buy" if bid_absorption > ask_absorption else "sell"
        }
    
    def _calculate_cvd(self, bids: List, asks: List) -> float:
        """حساب Cumulative Volume Delta مبسط"""
        bid_volume = sum(bid[1] for bid in bids[:5])
        ask_volume = sum(ask[1] for ask in asks[:5])
        
        return bid_volume - ask_volume
    
    def _analyze_cvd_trend(self) -> str:
        """تحليل اتجاه CVD"""
        if len(self.cvd_history) < 3:
            return "neutral"
        
        recent = list(self.cvd_history)[-3:]
        if all(recent[i] > recent[i-1] for i in range(1, 3)):
            return "rising"
        elif all(recent[i] < recent[i-1] for i in range(1, 3)):
            return "falling"
        else:
            return "neutral"
    
    def _calculate_imbalance(self, bids: List, asks: List) -> float:
        """حساب اختلال التوازن في السيولة"""
        top_bid_volume = bids[0][1] if bids else 0
        top_ask_volume = asks[0][1] if asks else 0
        
        if top_ask_volume == 0:
            return 1.0
        
        return top_bid_volume / top_ask_volume
    
    def analyze_tick_data(self, tick_data: List[Dict]) -> Dict[str, Any]:
        """تحليل بيانات التيك المتقدمة"""
        try:
            if len(tick_data) < 10:
                return {"ok": False, "error": "Insufficient tick data"}
            
            # تحليل تدفق الصفقات
            buy_volume = 0
            sell_volume = 0
            price_levels = defaultdict(float)
            
            for tick in tick_data[-20:]:  # آخر 20 تيك
                price = tick.get('price', 0)
                volume = tick.get('volume', 0)
                side = tick.get('side', '').lower()
                
                if side == 'buy':
                    buy_volume += volume
                elif side == 'sell':
                    sell_volume += volume
                
                price_levels[price] += volume
            
            # حساب الدلتا
            delta = buy_volume - sell_volume
            total_volume = buy_volume + sell_volume
            delta_ratio = delta / total_volume if total_volume > 0 else 0
            
            # تحليل حجم التيك
            large_trades = sum(1 for tick in tick_data[-20:] 
                             if tick.get('volume', 0) > 1000)  # عتبة كبيرة
            
            # توزيع الأحجام
            volume_distribution = {
                'buy_pct': buy_volume / total_volume if total_volume > 0 else 0,
                'sell_pct': sell_volume / total_volume if total_volume > 0 else 0,
                'delta_ratio': delta_ratio,
                'large_trades': large_trades,
                'avg_trade_size': total_volume / len(tick_data[-20:]) if tick_data[-20:] else 0
            }
            
            # اكتشاف تدفق كبير
            flow_spike = abs(delta_ratio) > 0.7
            
            return {
                "ok": True,
                "buy_volume": buy_volume,
                "sell_volume": sell_volume,
                "delta": delta,
                "delta_ratio": delta_ratio,
                "volume_distribution": volume_distribution,
                "flow_spike": flow_spike,
                "price_levels": dict(sorted(price_levels.items())[-5:])
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

# ============================================
# تعزيز مجلس الإدارة بـ Footprint
# ============================================

class EnhancedCouncilWithFootprint:
    """مجلس إدارة معزز بـ Footprint/Order-Flow"""
    
    def __init__(self):
        self.footprint_analyzer = AdvancedFootprintAnalyzer()
        self.scenario_engine = AdvancedScenarioEngine()
        self.historical_decisions = deque(maxlen=50)
        self.performance_tracker = {
            'total_decisions': 0,
            'correct_decisions': 0,
            'total_profit': 0.0,
            'recent_performance': deque(maxlen=20)
        }
    
    def analyze_market_with_footprint(self, df: pd.DataFrame, 
                                      orderbook: Dict = None,
                                      tick_data: List[Dict] = None) -> Dict[str, Any]:
        """تحليل السوق مع Footprint المتقدم"""
        
        # 1. تحليل السيناريو الأساسي
        market_context = self.scenario_engine.analyze_market_structure(df)
        
        # 2. تحليل Footprint إذا توفرت البيانات
        footprint_analysis = {"ok": False}
        if orderbook:
            footprint_analysis = self.footprint_analyzer.analyze_orderbook(orderbook)
        
        tick_analysis = {"ok": False}
        if tick_data:
            tick_analysis = self.footprint_analyzer.analyze_tick_data(tick_data)
        
        # 3. صنع القرار المتكامل
        integrated_decision = self._make_integrated_decision(
            df, market_context, footprint_analysis, tick_analysis
        )
        
        # 4. تسجيل القرار للتتبع
        self._record_decision(integrated_decision)
        
        return {
            "market_context": market_context,
            "footprint_analysis": footprint_analysis,
            "tick_analysis": tick_analysis,
            "decision": integrated_decision,
            "council_confidence": self._calculate_council_confidence(integrated_decision),
            "performance_stats": dict(self.performance_tracker)
        }
    
    def _make_integrated_decision(self, df: pd.DataFrame, 
                                  market_context: MarketContext,
                                  footprint_analysis: Dict,
                                  tick_analysis: Dict) -> TradeDecision:
        """صنع قرار متكامل يجمع كل التحليلات"""
        
        # القرار الأساسي من محرك السيناريوهات
        base_decision = self.scenario_engine.make_decision(df, market_context)
        
        # إذا كان القرار NO_TRADE أو HOLD، نرجع كما هو
        if base_decision.action in [Action.NO_TRADE, Action.HOLD]:
            return base_decision
        
        # تعزيز القرار بـ Footprint
        enhanced_decision = self._enhance_with_footprint(
            base_decision, footprint_analysis, tick_analysis
        )
        
        return enhanced_decision
    
    def _enhance_with_footprint(self, decision: TradeDecision,
                                footprint_analysis: Dict,
                                tick_analysis: Dict) -> TradeDecision:
        """تعزيز القرار بتحليل Footprint"""
        
        if not footprint_analysis.get("ok", False):
            return decision
        
        confidence_boost = 0.0
        additional_reasons = []
        
        # 1. تحليل سيولة Footprint
        liquidity_ratio = footprint_analysis.get("liquidity_ratio", 1.0)
        
        if decision.action == Action.BUY:
            if liquidity_ratio > 1.5:
                confidence_boost += 0.15
                additional_reasons.append("Strong bid liquidity in orderbook")
            elif liquidity_ratio < 0.7:
                confidence_boost -= 0.2
                additional_reasons.append("Weak bid liquidity - caution")
        
        elif decision.action == Action.SELL:
            if liquidity_ratio < 0.67:  # 1/1.5
                confidence_boost += 0.15
                additional_reasons.append("Strong ask liquidity in orderbook")
            elif liquidity_ratio > 1.3:
                confidence_boost -= 0.2
                additional_reasons.append("Weak ask liquidity - caution")
        
        # 2. تحليل جدران السيولة
        bid_walls = footprint_analysis.get("bid_walls", [])
        ask_walls = footprint_analysis.get("ask_walls", [])
        
        if decision.action == Action.BUY and bid_walls:
            # جدران شراء قريبة تدعم القرار
            confidence_boost += 0.1
            additional_reasons.append(f"Bid walls detected: {len(bid_walls)}")
        
        if decision.action == Action.SELL and ask_walls:
            confidence_boost += 0.1
            additional_reasons.append(f"Ask walls detected: {len(ask_walls)}")
        
        # 3. تحليل امتصاص السيولة
        absorption = footprint_analysis.get("absorption", {})
        if absorption.get("strong_absorption", False):
            if (decision.action == Action.BUY and 
                absorption.get("imbalance_direction") == "buy"):
                confidence_boost += 0.2
                additional_reasons.append("Strong absorption supporting buy")
            
            elif (decision.action == Action.SELL and 
                  absorption.get("imbalance_direction") == "sell"):
                confidence_boost += 0.2
                additional_reasons.append("Strong absorption supporting sell")
        
        # 4. تحليل CVD
        cvd_trend = footprint_analysis.get("cvd_trend", "neutral")
        if (decision.action == Action.BUY and cvd_trend == "rising") or \
           (decision.action == Action.SELL and cvd_trend == "falling"):
            confidence_boost += 0.1
            additional_reasons.append(f"CVD trend confirms direction: {cvd_trend}")
        
        # 5. تحليل بيانات التيك
        if tick_analysis.get("ok", False):
            delta_ratio = tick_analysis.get("volume_distribution", {}).get("delta_ratio", 0)
            flow_spike = tick_analysis.get("flow_spike", False)
            
            if flow_spike:
                if (decision.action == Action.BUY and delta_ratio > 0) or \
                   (decision.action == Action.SELL and delta_ratio < 0):
                    confidence_boost += 0.15
                    additional_reasons.append("Tick flow spike confirms direction")
        
        # تحديث القرار
        new_confidence = min(0.95, decision.confidence + confidence_boost)
        new_reasons = decision.reasons + additional_reasons
        
        # تعديل عامل المخاطرة بناءً على Footprint
        risk_factor = decision.risk_factor
        if confidence_boost > 0.1:
            risk_factor *= 1.2  # زيادة المخاطرة مع ثقة أعلى
        elif confidence_boost < -0.1:
            risk_factor *= 0.8  # تقليل المخاطرة مع ثقة أقل
        
        return TradeDecision(
            action=decision.action,
            mode=decision.mode,
            entry_zone=decision.entry_zone,
            confidence=new_confidence,
            reasons=new_reasons,
            tp_levels=decision.tp_levels,
            tp_weights=decision.tp_weights,
            sl_pct=decision.sl_pct,
            min_target_pips=decision.min_target_pips,
            trail_config=decision.trail_config,
            risk_factor=min(2.0, max(0.5, risk_factor))
        )
    
    def _record_decision(self, decision: TradeDecision):
        """تسجيل القرار للتتبع"""
        self.performance_tracker['total_decisions'] += 1
        self.historical_decisions.append({
            'timestamp': time.time(),
            'decision': decision.action.value,
            'confidence': decision.confidence,
            'mode': decision.mode.value
        })
    
    def _calculate_council_confidence(self, decision: TradeDecision) -> float:
        """حساب ثقة المجلس الإجمالية"""
        base_confidence = decision.confidence
        
        # تحسين الثقة بناءً على أداء القرارات السابقة
        if len(self.performance_tracker['recent_performance']) >= 5:
            recent_success_rate = sum(self.performance_tracker['recent_performance']) / \
                                 len(self.performance_tracker['recent_performance'])
            
            # تعديل الثقة بناءً على الأداء الأخير
            if recent_success_rate > 0.7:
                base_confidence *= 1.1
            elif recent_success_rate < 0.3:
                base_confidence *= 0.9
        
        return min(0.95, base_confidence)
    
    def update_performance(self, trade_result: Dict):
        """تحديث أداء القرارات"""
        if trade_result.get('success', False):
            self.performance_tracker['correct_decisions'] += 1
            self.performance_tracker['total_profit'] += trade_result.get('profit', 0)
            self.performance_tracker['recent_performance'].append(1)
        else:
            self.performance_tracker['recent_performance'].append(0)
        
        # حساب معدل النجاح
        if self.performance_tracker['total_decisions'] > 0:
            success_rate = self.performance_tracker['correct_decisions'] / \
                          self.performance_tracker['total_decisions']
            self.performance_tracker['success_rate'] = success_rate

# ============================================
# نظام إدارة الصفقات المحسن
# ============================================

class ProfessionalTradeManager:
    """مدير صفقات محترف مع إدارة ديناميكية"""
    
    def __init__(self, exchange, symbol):
        self.exchange = exchange
        self.symbol = symbol
        self.active_trades = {}
        self.trade_history = deque(maxlen=100)
        self.scenario_engine = AdvancedScenarioEngine()
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'current_streak': 0,
            'streak_type': 'none'
        }
        
        # إعدادات إدارة المخاطر
        self.risk_settings = {
            'max_daily_loss_pct': 5.0,
            'max_position_size_pct': 60.0,
            'min_confidence_threshold': 0.65,
            'max_trades_per_hour': 4,
            'cooldown_after_loss': 300,  # 5 دقائق
            'trailing_stop_enabled': True,
            'breakeven_activation_pct': 0.5,
            'partial_profit_enabled': True
        }
        
        # تتبع الحالة
        self.daily_stats = {
            'date': datetime.now().date(),
            'trades_today': 0,
            'profit_today': 0.0,
            'loss_today': 0.0,
            'net_today': 0.0
        }
        
        self.last_trade_time = 0
        self.consecutive_losses = 0
    
    def open_trade(self, decision: TradeDecision, current_price: float, 
                   balance: float) -> Dict[str, Any]:
        """فتح صفقة جديدة باحترافية"""
        
        # 1. التحقق من شروط المخاطرة
        risk_check = self._check_risk_conditions(decision)
        if not risk_check['allowed']:
            return {
                'success': False,
                'reason': risk_check['reason'],
                'action': 'NO_TRADE'
            }
        
        # 2. حساب حجم المركز
        position_size = self._calculate_position_size(decision, balance, current_price)
        if position_size <= 0:
            return {
                'success': False,
                'reason': 'Invalid position size',
                'action': 'NO_TRADE'
            }
        
        # 3. فتح الصفقة فعلياً
        trade_id = f"trade_{int(time.time())}_{random.randint(1000, 9999)}"
        
        trade_details = {
            'id': trade_id,
            'symbol': self.symbol,
            'action': decision.action.value,
            'mode': decision.mode.value,
            'entry_price': current_price,
            'position_size': position_size,
            'entry_time': time.time(),
            'zone_type': decision.entry_zone.type.value,
            'zone_strength': decision.entry_zone.strength,
            'decision_confidence': decision.confidence,
            'tp_levels': decision.tp_levels,
            'tp_weights': decision.tp_weights,
            'sl_pct': decision.sl_pct,
            'sl_price': self._calculate_sl_price(
                current_price, decision.action, decision.sl_pct
            ),
            'min_target_pips': decision.min_target_pips,
            'trail_config': decision.trail_config,
            'risk_factor': decision.risk_factor,
            'reasons': decision.reasons,
            'status': 'OPEN',
            'current_pnl': 0.0,
            'current_pnl_pct': 0.0,
            'highest_pnl': 0.0,
            'lowest_pnl': 0.0,
            'tp_hits': [],
            'management_actions': [],
            'monitoring_log': []
        }
        
        # 4. تنفيذ الأمر (في الوضع الحي)
        try:
            # هنا يتم تنفيذ الأمر الفعلي على المنصة
            # نستخدم وضع محاكاة للتوضيح
            executed = self._execute_order(
                action=decision.action.value,
                quantity=position_size,
                price=current_price
            )
            
            if not executed:
                return {
                    'success': False,
                    'reason': 'Order execution failed',
                    'action': 'NO_TRADE'
                }
            
        except Exception as e:
            return {
                'success': False,
                'reason': f'Execution error: {str(e)}',
                'action': 'NO_TRADE'
            }
        
        # 5. حفظ وتتبع الصفقة
        self.active_trades[trade_id] = trade_details
        self.last_trade_time = time.time()
        
        # تحديث الإحصائيات
        self._update_daily_stats('open_trade')
        
        return {
            'success': True,
            'trade_id': trade_id,
            'details': trade_details,
            'message': f'Trade opened: {decision.action.value} {position_size:.4f} @ {current_price:.6f}'
        }
    
    def manage_trades(self, df: pd.DataFrame, current_price: float) -> List[Dict]:
        """إدارة الصفقات المفتوحة ديناميكياً"""
        
        management_results = []
        
        for trade_id, trade in list(self.active_trades.items()):
            if trade['status'] != 'OPEN':
                continue
            
            # 1. تحليل حالة الصفقة
            trade_analysis = self._analyze_trade_status(trade, df, current_price)
            
            # 2. تطبيق إدارة الصفقة
            management_action = self._apply_trade_management(trade, trade_analysis, current_price)
            
            # 3. تنفيذ الإجراءات إذا لزم الأمر
            if management_action['action'] != 'HOLD':
                execution_result = self._execute_management_action(
                    trade_id, trade, management_action, current_price
                )
                
                management_results.append(execution_result)
                
                # تحديث حالة الصفقة
                if management_action['action'] in ['CLOSE', 'PARTIAL_CLOSE']:
                    self._update_trade_after_action(trade_id, management_action)
        
        return management_results
    
    def _analyze_trade_status(self, trade: Dict, df: pd.DataFrame, 
                             current_price: float) -> Dict[str, Any]:
        """تحليل متقدم لحالة الصفقة"""
        
        entry_price = trade['entry_price']
        action = trade['action']
        
        # حساب الربح/الخسارة الحالي
        if action == 'BUY':
            pnl = current_price - entry_price
            pnl_pct = (pnl / entry_price) * 100
        else:  # SELL
            pnl = entry_price - current_price
            pnl_pct = (pnl / entry_price) * 100
        
        trade['current_pnl'] = pnl
        trade['current_pnl_pct'] = pnl_pct
        
        # تحديث أعلى/أقل ربح
        if pnl_pct > trade['highest_pnl']:
            trade['highest_pnl'] = pnl_pct
        
        if pnl_pct < trade['lowest_pnl']:
            trade['lowest_pnl'] = pnl_pct
        
        # تحليل السياق الحالي
        market_context = self.scenario_engine.analyze_market_structure(df)
        
        # كشف إذا كانت المنطقة دخول خاطئة
        is_bad_zone = self._detect_bad_entry_zone(trade, market_context, current_price)
        
        # تحليل قوة الصفقة الحالية
        trade_strength = self._assess_trade_strength(trade, market_context, df)
        
        # تحقق من تحقيق أهداف الربح
        tp_achievements = self._check_tp_achievements(trade, pnl_pct)
        
        # تحقق من وقف الخسارة
        sl_triggered = self._check_sl_triggered(trade, pnl_pct)
        
        return {
            'pnl_pct': pnl_pct,
            'is_bad_zone': is_bad_zone,
            'trade_strength': trade_strength,
            'tp_achievements': tp_achievements,
            'sl_triggered': sl_triggered,
            'market_context': market_context,
            'time_in_trade': time.time() - trade['entry_time']
        }
    
    def _detect_bad_entry_zone(self, trade: Dict, market_context: MarketContext,
                              current_price: float) -> bool:
        """كشف إذا كانت منطقة الدخول خاطئة"""
        
        # 1. خسارة سريعة وكبيرة
        time_in_trade = time.time() - trade['entry_time']
        pnl_pct = trade['current_pnl_pct']
        
        if pnl_pct < -1.5 and time_in_trade < 180:  # 3 دقائق
            return True
        
        # 2. تحول في تحيز السوق
        if trade['action'] == 'BUY' and market_context.bias == "BEARISH":
            if market_context.strength > 6.0:
                return True
        
        if trade['action'] == 'SELL' and market_context.bias == "BULLISH":
            if market_context.strength > 6.0:
                return True
        
        # 3. كسر مناطق دعم/مقاومة رئيسية
        entry_price = trade['entry_price']
        key_levels = market_context.key_levels
        
        if trade['action'] == 'BUY':
            # تحقق من كسر الدعم
            support_levels = [level for level in key_levels if level < entry_price]
            if support_levels:
                strongest_support = max(support_levels)
                if current_price < strongest_support * 0.995:
                    return True
        
        if trade['action'] == 'SELL':
            # تحقق من كسر المقاومة
            resistance_levels = [level for level in key_levels if level > entry_price]
            if resistance_levels:
                strongest_resistance = min(resistance_levels)
                if current_price > strongest_resistance * 1.005:
                    return True
        
        # 4. انخفاض الثقة في المنطقة
        zone_strength = trade.get('zone_strength', 5.0)
        if zone_strength < 4.0 and pnl_pct < -0.5:
            return True
        
        return False
    
    def _assess_trade_strength(self, trade: Dict, market_context: MarketContext,
                              df: pd.DataFrame) -> Dict[str, float]:
        """تقييم قوة الصفقة الحالية"""
        
        pnl_pct = trade['current_pnl_pct']
        action = trade['action']
        
        strength_score = 0.0
        factors = []
        
        # 1. محاذاة مع اتجاه السوق
        if (action == 'BUY' and market_context.bias == "BULLISH") or \
           (action == 'SELL' and market_context.bias == "BEARISH"):
            strength_score += 0.3
            factors.append("Market alignment")
        
        # 2. قوة المنطقة الأصلية
        zone_strength = trade.get('zone_strength', 5.0)
        strength_score += (zone_strength / 10.0) * 0.2
        
        # 3. الربح الحالي
        if pnl_pct > 0:
            profit_factor = min(0.3, (pnl_pct / 5.0) * 0.3)
            strength_score += profit_factor
            factors.append(f"Profit: {pnl_pct:.2f}%")
        
        # 4. حجم التداول الداعم
        recent_volume = df['volume'].astype(float).tail(5).mean()
        avg_volume = df['volume'].astype(float).tail(20).mean()
        
        if recent_volume > avg_volume * 1.2:
            strength_score += 0.1
            factors.append("Volume support")
        
        # 5. زمن الصفقة
        time_factor = min(0.1, (time.time() - trade['entry_time']) / 600)  # 10 دقائق
        strength_score += time_factor
        
        return {
            'score': min(1.0, strength_score),
            'confidence': strength_score,
            'factors': factors,
            'recommendation': 'HOLD' if strength_score > 0.5 else 'REVIEW'
        }
    
    def _check_tp_achievements(self, trade: Dict, pnl_pct: float) -> List[Dict]:
        """التحقق من تحقيق أهداف الربح"""
        
        tp_levels = trade.get('tp_levels', [])
        tp_weights = trade.get('tp_weights', [])
        tp_hits = trade.get('tp_hits', [])
        
        achievements = []
        
        for i, (tp_level, tp_weight) in enumerate(zip(tp_levels, tp_weights)):
            if i >= len(tp_hits) and pnl_pct >= tp_level:
                achievements.append({
                    'level': i + 1,
                    'tp_pct': tp_level,
                    'weight': tp_weight,
                    'pnl_at_hit': pnl_pct
                })
        
        return achievements
    
    def _check_sl_triggered(self, trade: Dict, pnl_pct: float) -> bool:
        """التحقق من تنفيذ وقف الخسارة"""
        sl_pct = trade.get('sl_pct', 2.0)
        return pnl_pct <= -sl_pct
    
    def _apply_trade_management(self, trade: Dict, analysis: Dict,
                               current_price: float) -> Dict[str, Any]:
        """تطبيق إدارة الصفقة الذكية"""
        
        pnl_pct = analysis['pnl_pct']
        is_bad_zone = analysis['is_bad_zone']
        tp_achievements = analysis['tp_achievements']
        sl_triggered = analysis['sl_triggered']
        trade_strength = analysis['trade_strength']
        
        # 1. وقف الخسارة
        if sl_triggered:
            return {
                'action': 'CLOSE',
                'reason': 'Stop loss triggered',
                'details': f'SL hit: {pnl_pct:.2f}%'
            }
        
        # 2. كشف المنطقة الخاطئة والخروج المبكر
        if is_bad_zone:
            return {
                'action': 'CLOSE',
                'reason': 'Bad entry zone detected',
                'details': 'Early exit to minimize loss'
            }
        
        # 3. تحقيق أهداف الربح
        if tp_achievements:
            achievement = tp_achievements[0]
            
            # إذا كانت الصفقة سكالب: إغلاق كامل عند أول هدف
            if trade['mode'] == 'SCALP':
                return {
                    'action': 'CLOSE',
                    'reason': 'Scalp target achieved',
                    'details': f'TP{achievement["level"]} hit: {pnl_pct:.2f}%'
                }
            
            # إذا كانت صفقة ترند: إغلاق جزئي
            else:
                close_percentage = achievement['weight']
                return {
                    'action': 'PARTIAL_CLOSE',
                    'reason': f'TP{achievement["level"]} achieved',
                    'details': f'Close {close_percentage*100:.0f}% at {pnl_pct:.2f}%',
                    'close_percentage': close_percentage
                }
        
        # 4. إدارة الوقف المتحرك
        trail_config = trade.get('trail_config', {})
        if trail_config.get('enabled', False) and pnl_pct >= trail_config.get('activation_pct', 0):
            trail_action = self._manage_trailing_stop(trade, pnl_pct, current_price)
            if trail_action:
                return trail_action
        
        # 5. تحويل السكالب لترند إذا كان الأداء جيداً
        if trade['mode'] == 'SCALP' and pnl_pct >= 0.5:
            if self._should_promote_to_trend(trade, analysis):
                return {
                    'action': 'PROMOTE_TO_TREND',
                    'reason': 'Promoting scalp to trend',
                    'details': f'Good performance: {pnl_pct:.2f}%'
                }
        
        # 6. اقتراح تعديلات على الإدارة
        if trade_strength['score'] < 0.3 and pnl_pct > 0.3:
            # قوة ضعيفة مع ربح: خروج جزئي
            return {
                'action': 'PARTIAL_CLOSE',
                'reason': 'Weak trade strength with profit',
                'details': f'Close 50% at {pnl_pct:.2f}%',
                'close_percentage': 0.5
            }
        
        # 7. البقاء في الصفقة
        return {
            'action': 'HOLD',
            'reason': 'Continue managing trade',
            'details': f'Current PnL: {pnl_pct:.2f}%'
        }
    
    def _manage_trailing_stop(self, trade: Dict, pnl_pct: float, 
                             current_price: float) -> Optional[Dict]:
        """إدارة الوقف المتحرك"""
        
        trail_config = trade.get('trail_config', {})
        activation_pct = trail_config.get('activation_pct', 0)
        distance_pct = trail_config.get('distance_pct', 0.01)
        
        if pnl_pct < activation_pct:
            return None
        
        # حساب الوقف المتحرك الجديد
        if trade['action'] == 'BUY':
            new_trail_price = current_price * (1 - distance_pct / 100)
            current_trail = trade.get('trail_price', 0)
            
            if new_trail_price > current_trail:
                # تحديث الوقف المتحرك
                return {
                    'action': 'UPDATE_TRAIL',
                    'reason': 'Trailing stop updated',
                    'details': f'New trail: {new_trail_price:.6f}',
                    'new_trail_price': new_trail_price
                }
        
        else:  # SELL
            new_trail_price = current_price * (1 + distance_pct / 100)
            current_trail = trade.get('trail_price', float('inf'))
            
            if new_trail_price < current_trail:
                return {
                    'action': 'UPDATE_TRAIL',
                    'reason': 'Trailing stop updated',
                    'details': f'New trail: {new_trail_price:.6f}',
                    'new_trail_price': new_trail_price
                }
        
        # التحقق من تنفيذ الوقف المتحرك
        if trade.get('trail_price'):
            if (trade['action'] == 'BUY' and current_price <= trade['trail_price']) or \
               (trade['action'] == 'SELL' and current_price >= trade['trail_price']):
                return {
                    'action': 'CLOSE',
                    'reason': 'Trailing stop hit',
                    'details': f'Trail executed at {current_price:.6f}'
                }
        
        return None
    
    def _should_promote_to_trend(self, trade: Dict, analysis: Dict) -> bool:
        """تحديد إذا كان يجب تحويل السكالب لترند"""
        
        pnl_pct = analysis['pnl_pct']
        market_context = analysis['market_context']
        time_in_trade = analysis['time_in_trade']
        
        # شروط الترقية
        conditions = [
            pnl_pct >= 0.5,  # ربح 0.5% على الأقل
            time_in_trade >= 60,  # مدة كافية
            market_context.bias == ("BULLISH" if trade['action'] == 'BUY' else "BEARISH"),
            market_context.strength >= 5.0,
            trade.get('zone_strength', 0) >= 6.0
        ]
        
        return all(conditions)
    
    def _execute_management_action(self, trade_id: str, trade: Dict,
                                  action: Dict, current_price: float) -> Dict:
        """تنفيذ إجراء إدارة الصفقة"""
        
        try:
            if action['action'] == 'CLOSE':
                # إغلاق كامل
                success = self._execute_order(
                    action='SELL' if trade['action'] == 'BUY' else 'BUY',
                    quantity=trade['position_size'],
                    price=current_price
                )
                
                if success:
                    profit = trade['current_pnl'] * trade['position_size']
                    
                    return {
                        'trade_id': trade_id,
                        'action': 'CLOSED',
                        'profit': profit,
                        'pnl_pct': trade['current_pnl_pct'],
                        'reason': action['reason'],
                        'details': action['details']
                    }
            
            elif action['action'] == 'PARTIAL_CLOSE':
                # إغلاق جزئي
                close_percentage = action.get('close_percentage', 0.5)
                close_quantity = trade['position_size'] * close_percentage
                
                success = self._execute_order(
                    action='SELL' if trade['action'] == 'BUY' else 'BUY',
                    quantity=close_quantity,
                    price=current_price
                )
                
                if success:
                    profit = trade['current_pnl'] * close_quantity
                    
                    return {
                        'trade_id': trade_id,
                        'action': 'PARTIAL_CLOSE',
                        'closed_quantity': close_quantity,
                        'remaining_quantity': trade['position_size'] - close_quantity,
                        'profit': profit,
                        'pnl_pct': trade['current_pnl_pct'],
                        'reason': action['reason'],
                        'details': action['details']
                    }
            
            elif action['action'] == 'UPDATE_TRAIL':
                # تحديث الوقف المتحرك
                trade['trail_price'] = action.get('new_trail_price')
                
                return {
                    'trade_id': trade_id,
                    'action': 'TRAIL_UPDATED',
                    'new_trail': action.get('new_trail_price'),
                    'reason': action['reason'],
                    'details': action['details']
                }
            
            elif action['action'] == 'PROMOTE_TO_TREND':
                # ترقية السكالب لترند
                trade['mode'] = 'TREND'
                # تعديل أهداف الربح لتصبح 3 مستويات
                trade['tp_levels'] = [0.8, 1.6, 2.5]
                trade['tp_weights'] = [0.3, 0.3, 0.4]
                
                return {
                    'trade_id': trade_id,
                    'action': 'PROMOTED',
                    'new_mode': 'TREND',
                    'reason': action['reason'],
                    'details': action['details']
                }
        
        except Exception as e:
            return {
                'trade_id': trade_id,
                'action': 'ERROR',
                'reason': f'Execution failed: {str(e)}',
                'details': action
            }
        
        return {
            'trade_id': trade_id,
            'action': 'NO_ACTION',
            'reason': 'Unknown action',
            'details': action
        }
    
    def _update_trade_after_action(self, trade_id: str, action: Dict):
        """تحديث بيانات الصفقة بعد الإجراء"""
        
        if trade_id not in self.active_trades:
            return
        
        trade = self.active_trades[trade_id]
        
        if action['action'] == 'CLOSE':
            # إغلاق الصفقة
            trade['status'] = 'CLOSED'
            trade['close_time'] = time.time()
            trade['close_price'] = trade.get('current_price', 0)
            
            # نقل للتاريخ
            self.trade_history.append(trade.copy())
            del self.active_trades[trade_id]
            
            # تحديث الإحصائيات
            self._update_performance_metrics(trade)
            
        elif action['action'] == 'PARTIAL_CLOSE':
            # تحديث الكمية المتبقية
            close_percentage = action.get('close_percentage', 0.5)
            trade['position_size'] *= (1 - close_percentage)
            
            # تسجيل الإغلاق الجزئي
            if 'partial_closes' not in trade:
                trade['partial_closes'] = []
            
            trade['partial_closes'].append({
                'time': time.time(),
                'percentage': close_percentage,
                'price': trade.get('current_price', 0)
            })
        
        elif action['action'] == 'UPDATE_TRAIL':
            # تحديث سعر الوقف المتحرك
            trade['trail_price'] = action.get('new_trail_price')
        
        elif action['action'] == 'PROMOTE_TO_TREND':
            # تحديث النمط
            trade['mode'] = 'TREND'
            trade['tp_levels'] = [0.8, 1.6, 2.5]
            trade['tp_weights'] = [0.3, 0.3, 0.4]
    
    def _check_risk_conditions(self, decision: TradeDecision) -> Dict[str, Any]:
        """التحقق من شروط المخاطرة"""
        
        # 1. التحقق من تاريخ اليوم
        today = datetime.now().date()
        if self.daily_stats['date'] != today:
            self._reset_daily_stats()
        
        # 2. الحد الأقصى للصفقات اليومية
        if self.daily_stats['trades_today'] >= 20:
            return {
                'allowed': False,
                'reason': 'Maximum daily trades reached'
            }
        
        # 3. الحد الأقصى للخسارة اليومية
        if self.daily_stats['loss_today'] >= self.risk_settings['max_daily_loss_pct']:
            return {
                'allowed': False,
                'reason': 'Maximum daily loss reached'
            }
        
        # 4. التبريد بعد الخسائر المتتالية
        if self.consecutive_losses >= 2:
            time_since_last_loss = time.time() - self.last_trade_time
            if time_since_last_loss < self.risk_settings['cooldown_after_loss']:
                return {
                    'allowed': False,
                    'reason': f'Cooling down after {self.consecutive_losses} consecutive losses'
                }
        
        # 5. الحد الأقصى للصفقات في الساعة
        recent_trades = [t for t in self.trade_history 
                        if time.time() - t.get('entry_time', 0) < 3600]
        if len(recent_trades) >= self.risk_settings['max_trades_per_hour']:
            return {
                'allowed': False,
                'reason': 'Maximum trades per hour reached'
            }
        
        # 6. ثقة القرار
        if decision.confidence < self.risk_settings['min_confidence_threshold']:
            return {
                'allowed': False,
                'reason': f'Decision confidence too low: {decision.confidence:.2f}'
            }
        
        return {'allowed': True, 'reason': 'Risk conditions satisfied'}
    
    def _calculate_position_size(self, decision: TradeDecision, 
                                balance: float, current_price: float) -> float:
        """حساب حجم المركز"""
        
        # الحجم الأساسي: 60% من الرصيد
        base_position = balance * 0.6
        
        # تعديل بناءً على عامل المخاطرة
        adjusted_position = base_position * decision.risk_factor
        
        # الحد الأقصى
        max_position = balance * (self.risk_settings['max_position_size_pct'] / 100)
        
        position = min(adjusted_position, max_position)
        
        # تحويل للكمية
        quantity = position / current_price
        
        # التقريب حسب منصة التداول
        quantity = self._round_quantity(quantity)
        
        return quantity
    
    def _round_quantity(self, quantity: float) -> float:
        """تقريب الكمية حسب مواصفات المنصة"""
        # يمكن تعديل هذا حسب كل منصة
        step = 0.001  # خطوة 0.001
        return round(quantity / step) * step
    
    def _calculate_sl_price(self, entry_price: float, action: str, sl_pct: float) -> float:
        """حساب سعر وقف الخسارة"""
        if action == 'BUY':
            return entry_price * (1 - sl_pct / 100)
        else:  # SELL
            return entry_price * (1 + sl_pct / 100)
    
    def _execute_order(self, action: str, quantity: float, price: float) -> bool:
        """تنفيذ الأمر (محاكاة أو حقيقي)"""
        # في النسخة الحقيقية، هنا يتم استدعاء API المنصة
        # نستخدم محاكاة للتوضيح
        try:
            # محاكاة التنفيذ
            time.sleep(0.1)  # محاكاة زمن التنفيذ
            
            # تسجيل التنفيذ
            print(f"[EXECUTE] {action} {quantity:.4f} @ {price:.6f}")
            
            return True
        except Exception as e:
            print(f"[EXECUTE ERROR] {str(e)}")
            return False
    
    def _update_daily_stats(self, action: str):
        """تحديث إحصائيات اليوم"""
        today = datetime.now().date()
        
        if self.daily_stats['date'] != today:
            self._reset_daily_stats()
        
        if action == 'open_trade':
            self.daily_stats['trades_today'] += 1
        
        # تحديث صافي الربح/الخسارة يتم في _update_performance_metrics
    
    def _reset_daily_stats(self):
        """إعادة تعيين إحصائيات اليوم"""
        self.daily_stats = {
            'date': datetime.now().date(),
            'trades_today': 0,
            'profit_today': 0.0,
            'loss_today': 0.0,
            'net_today': 0.0
        }
    
    def _update_performance_metrics(self, trade: Dict):
        """تحديث مقاييس الأداء"""
        
        profit = trade.get('current_pnl', 0) * trade.get('position_size', 0)
        pnl_pct = trade.get('current_pnl_pct', 0)
        
        # تحديث إحصائيات اليوم
        if profit > 0:
            self.daily_stats['profit_today'] += profit
            self.daily_stats['net_today'] += profit
            self.consecutive_losses = 0
            
            # تحديث تسلسل الفوز
            if self.performance_metrics['streak_type'] == 'win':
                self.performance_metrics['current_streak'] += 1
            else:
                self.performance_metrics['current_streak'] = 1
                self.performance_metrics['streak_type'] = 'win'
        else:
            self.daily_stats['loss_today'] += abs(profit)
            self.daily_stats['net_today'] -= abs(profit)
            self.consecutive_losses += 1
            
            # تحديث تسلسل الخسارة
            if self.performance_metrics['streak_type'] == 'loss':
                self.performance_metrics['current_streak'] += 1
            else:
                self.performance_metrics['current_streak'] = 1
                self.performance_metrics['streak_type'] = 'loss'
        
        # تحديث الإحصائيات العامة
        self.performance_metrics['total_trades'] += 1
        
        if profit > 0:
            self.performance_metrics['winning_trades'] += 1
            self.performance_metrics['total_profit'] += profit
            
            # أكبر فوز
            if profit > self.performance_metrics['largest_win']:
                self.performance_metrics['largest_win'] = profit
            
            # متوسط الفوز
            if self.performance_metrics['winning_trades'] > 0:
                self.performance_metrics['avg_win'] = (
                    self.performance_metrics['total_profit'] / 
                    self.performance_metrics['winning_trades']
                )
        else:
            # أكبر خسارة
            if abs(profit) > self.performance_metrics['largest_loss']:
                self.performance_metrics['largest_loss'] = abs(profit)
            
            # متوسط الخسارة
            losing_trades = self.performance_metrics['total_trades'] - self.performance_metrics['winning_trades']
            total_loss = abs(profit)  # سيتم تحديثه في التطبيق الكامل
            
            if losing_trades > 0:
                self.performance_metrics['avg_loss'] = total_loss / losing_trades
        
        # معدل الفوز
        if self.performance_metrics['total_trades'] > 0:
            self.performance_metrics['win_rate'] = (
                self.performance_metrics['winning_trades'] / 
                self.performance_metrics['total_trades']
            ) * 100
        
        # عامل الربح (Profit Factor)
        total_wins = self.performance_metrics['total_profit']
        total_losses = abs(self.performance_metrics['total_trades'] - 
                          self.performance_metrics['winning_trades']) * self.performance_metrics['avg_loss']
        
        if total_losses > 0:
            self.performance_metrics['profit_factor'] = total_wins / total_losses
        
        # أكبر تسلسل فوز
        if self.performance_metrics['streak_type'] == 'win':
            if self.performance_metrics['current_streak'] > self.performance_metrics['max_consecutive_wins']:
                self.performance_metrics['max_consecutive_wins'] = self.performance_metrics['current_streak']
        
        # أكبر تسلسل خسارة
        if self.performance_metrics['streak_type'] == 'loss':
            if self.performance_metrics['current_streak'] > self.performance_metrics['max_consecutive_losses']:
                self.performance_metrics['max_consecutive_losses'] = self.performance_metrics['current_streak']
    
    def get_performance_report(self) -> Dict[str, Any]:
        """الحصول على تقرير الأداء"""
        return {
            'daily_stats': self.daily_stats,
            'performance_metrics': self.performance_metrics,
            'active_trades': len(self.active_trades),
            'trade_history_count': len(self.trade_history),
            'consecutive_losses': self.consecutive_losses,
            'time_since_last_trade': time.time() - self.last_trade_time if self.last_trade_time > 0 else 0
        }

# ============================================
# دمج كل المكونات في البوت الرئيسي
# ============================================

# ... (الكود الحالي للبوت يبقى كما هو مع إضافة ما يلي)

# إضافة المتغيرات العالمية الجديدة
advanced_council = EnhancedCouncilWithFootprint()
professional_manager = None  # سيتم تهيئته لاحقاً
scenario_engine = AdvancedScenarioEngine()

# دمج النظام الجديد في دورة التداول
def enhanced_trade_loop():
    """دورة تداول محسنة مع النظام الجديد"""
    global professional_manager
    
    # تهيئة مدير الصفقات
    if professional_manager is None:
        professional_manager = ProfessionalTradeManager(ex, SYMBOL)
    
    while True:
        try:
            # جلب البيانات
            df = fetch_ohlcv(limit=200)
            current_price = price_now()
            balance = balance_usdt()
            
            if df.empty or current_price is None:
                time.sleep(BASE_SLEEP)
                continue
            
            # 1. تحليل السوق مع Footprint المتقدم
            try:
                orderbook = ex.fetch_order_book(SYMBOL, limit=20)
                council_analysis = advanced_council.analyze_market_with_footprint(
                    df=df,
                    orderbook=orderbook
                )
            except:
                council_analysis = advanced_council.analyze_market_with_footprint(df=df)
            
            # 2. صنع القرار
            decision = council_analysis['decision']
            
            # 3. التحقق من وجود صفقات مفتوحة
            active_trades = professional_manager.active_trades
            
            if active_trades:
                # إدارة الصفقات المفتوحة
                management_results = professional_manager.manage_trades(df, current_price)
                
                for result in management_results:
                    if result.get('action') in ['CLOSED', 'PARTIAL_CLOSE']:
                        print(f"💰 {result['action']}: {result.get('profit', 0):.2f} | {result['reason']}")
            
            # 4. البحث عن فرص دخول جديدة
            elif decision.action in [Action.BUY, Action.SELL]:
                # فتح صفقة جديدة
                trade_result = professional_manager.open_trade(decision, current_price, balance)
                
                if trade_result['success']:
                    print(f"🎯 {trade_result['message']}")
                    print(f"   Mode: {decision.mode.value} | Confidence: {decision.confidence:.2f}")
                    print(f"   TP Levels: {decision.tp_levels} | SL: {decision.sl_pct}%")
            
            # 5. عرض الإحصائيات
            if random.random() < 0.1:  # كل 10 دورات تقريباً
                report = professional_manager.get_performance_report()
                print(f"\n📊 Performance Report:")
                print(f"   Trades Today: {report['daily_stats']['trades_today']}")
                print(f"   Net Today: {report['daily_stats']['net_today']:.2f}")
                print(f"   Win Rate: {report['performance_metrics']['win_rate']:.1f}%")
                print(f"   Active Trades: {report['active_trades']}")
                print(f"   Consecutive Losses: {report['consecutive_losses']}")
                print()
            
            # 6. النوم حتى الدورة التالية
            time.sleep(max(2, time_to_candle_close(df)))
            
        except Exception as e:
            print(f"❌ Error in enhanced trade loop: {str(e)}")
            print(traceback.format_exc())
            time.sleep(BASE_SLEEP)

# استبدال دورة التداول القديمة بالجديدة
trade_loop = enhanced_trade_loop

# إضافة نقاط نهاية API جديدة
@app.route('/advanced_stats')
def get_advanced_stats():
    """الحصول على إحصائيات النظام المتقدم"""
    if professional_manager:
        report = professional_manager.get_performance_report()
        
        return jsonify({
            'success': True,
            'professional_manager': report,
            'council_confidence': advanced_council.performance_tracker.get('success_rate', 0),
            'active_trades': len(professional_manager.active_trades),
            'scenario_engine_active': True,
            'footprint_integrated': True
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Professional manager not initialized'
        })

@app.route('/active_trades')
def get_active_trades():
    """الحصول على الصفقات النشطة"""
    if professional_manager:
        return jsonify({
            'success': True,
            'active_trades': professional_manager.active_trades,
            'count': len(professional_manager.active_trades)
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Professional manager not initialized'
        })

@app.route('/trade_history')
def get_trade_history():
    """الحصول على سجل الصفقات"""
    if professional_manager:
        return jsonify({
            'success': True,
            'trade_history': list(professional_manager.trade_history)[-20:],
            'total_trades': len(professional_manager.trade_history)
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Professional manager not initialized'
        })

@app.route('/market_analysis')
def get_market_analysis():
    """الحصول على تحليل السوق الحالي"""
    try:
        df = fetch_ohlcv(limit=200)
        
        # تحليل السيناريو
        market_context = scenario_engine.analyze_market_structure(df)
        
        # تحليل مجلس الإدارة
        council_analysis = advanced_council.analyze_market_with_footprint(df=df)
        
        return jsonify({
            'success': True,
            'market_context': {
                'phase': market_context.phase.value,
                'bias': market_context.bias,
                'strength': market_context.strength,
                'volatility': market_context.volatility,
                'volume_profile': market_context.volume_profile,
                'displacement_detected': market_context.displacement_detected,
                'zones_count': len(market_context.zones)
            },
            'council_analysis': {
                'decision': council_analysis['decision'].action.value if council_analysis.get('decision') else 'NO_TRADE',
                'confidence': council_analysis.get('council_confidence', 0),
                'footprint_ok': council_analysis.get('footprint_analysis', {}).get('ok', False)
            },
            'current_price': price_now(),
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# تحديث نقطة النهاية الرئيسية
@app.route('/')
def home():
    mode = 'LIVE' if MODE_LIVE else 'PAPER'
    return f"""
    <html>
        <head><title>SUI ULTRA PRO AI BOT - Professional Hunter</title></head>
        <body>
            <h1>🚀 SUI ULTRA PRO AI BOT - Professional Hunter</h1>
            <p><strong>Exchange:</strong> {EXCHANGE_NAME.upper()} | <strong>Symbol:</strong> {SYMBOL} | <strong>Interval:</strong> {INTERVAL}</p>
            <p><strong>Mode:</strong> {mode} | <strong>Advanced Features:</strong> ENABLED</p>
            <hr>
            <h2>🎯 Advanced Systems:</h2>
            <ul>
                <li>✅ Professional Scenario Engine</li>
                <li>✅ Enhanced Footprint/Order-Flow</li>
                <li>✅ Dynamic Trade Management</li>
                <li>✅ Smart Profit Taking (3 levels for Gold, 1 for Scalp)</li>
                <li>✅ Bad Zone Detection & Early Exit</li>
                <li>✅ Professional Risk Management</li>
            </ul>
            <hr>
            <h2>📊 Endpoints:</h2>
            <ul>
                <li><a href="/advanced_stats">/advanced_stats</a> - Advanced performance stats</li>
                <li><a href="/active_trades">/active_trades</a> - Active trades</li>
                <li><a href="/trade_history">/trade_history</a> - Trade history</li>
                <li><a href="/market_analysis">/market_analysis</a> - Current market analysis</li>
                <li><a href="/health">/health</a> - Health check</li>
                <li><a href="/metrics">/metrics</a> - Basic metrics</li>
            </ul>
        </body>
    </html>
    """

# تحديث التحقق من البيئة
def verify_execution_environment():
    print(f"⚙️ PROFESSIONAL EXECUTION ENVIRONMENT", flush=True)
    print(f"🔧 EXCHANGE: {EXCHANGE_NAME.upper()} | SYMBOL: {SYMBOL}", flush=True)
    print(f"🔧 PROFESSIONAL HUNTER MODE: ENABLED", flush=True)
    print(f"🎯 ADVANCED SYSTEMS:", flush=True)
    print(f"   • Scenario Engine: ENABLED", flush=True)
    print(f"   • Enhanced Footprint: ENABLED", flush=True)
    print(f"   • Dynamic Trade Management: ENABLED", flush=True)
    print(f"   • Smart Profit Taking: 3 levels (Gold) / 1 level (Scalp)", flush=True)
    print(f"   • Bad Zone Detection: ENABLED", flush=True)
    print(f"   • Professional Risk Management: ENABLED", flush=True)
    print(f"🚀 BOT READY - PROFESSIONAL HUNTER MODE ACTIVATED", flush=True)

# تشغيل البوت
if __name__ == "__main__":
    verify_execution_environment()
    
    import threading
    threading.Thread(target=keepalive_loop, daemon=True).start()
    threading.Thread(target=trade_loop, daemon=True).start()
    
    log_i(f"🚀 SUI ULTRA PRO AI BOT STARTED - PROFESSIONAL HUNTER MODE")
    log_i(f"🎯 SYMBOL: {SYMBOL} | INTERVAL: {INTERVAL} | LEVERAGE: {LEVERAGE}x")
    log_i(f"💡 ADVANCED SYSTEMS ACTIVATED: Scenario Engine + Footprint + Dynamic Management")
    log_i(f"🎯 PROFIT SYSTEM: 3 levels for Golden Trades, 1 level for Scalp")
    log_i(f"🛡️ RISK MANAGEMENT: Professional with bad zone detection")
    
    app.run(host="0.0.0.0", port=PORT, debug=False)

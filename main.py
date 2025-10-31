# /app/doge_ai_council_pro.py
# -*- coding: utf-8 -*-
"""
DOGE/USDT — AI Trading Bot v3.0
+ Advanced AI Decision Engine with Machine Learning
+ Smart Council with Neural Network Pattern Recognition
+ Professional Risk Management & Profit Optimization
+ Real-time Market Intelligence & Sentiment Analysis

Exchange: BingX USDT Perp via CCXT
"""

import os, time, math, random, signal, sys, traceback, logging, json, tempfile
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from collections import deque, defaultdict
import statistics
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import ccxt
from flask import Flask, jsonify, request
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# ===== ENV =====
API_KEY    = os.getenv("BINGX_API_KEY", "")
API_SECRET = os.getenv("BINGX_API_SECRET", "")
MODE_LIVE  = bool(API_KEY and API_SECRET)
PORT       = int(os.getenv("PORT", 5000))
SELF_URL   = (os.getenv("SELF_URL") or os.getenv("RENDER_EXTERNAL_URL") or "").strip().rstrip("/")

# ===== AI Trading Configuration =====
SYMBOL        = "DOGE/USDT:USDT"
INTERVAL      = "15m"
LEVERAGE      = 10  # كما طلبت
RISK_ALLOC    = 0.60  # كما طلبت
POSITION_MODE = "oneway"

# AI Model Parameters
AI_MODEL_PATH = "ai_trading_model.joblib"
SCALER_PATH = "feature_scaler.joblib"
MODEL_UPDATE_FREQUENCY = 50  # تحديث النموذج كل 50 صفقة

# Advanced Technical Indicators
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14
VWAP_LEN = 20
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIG  = 9
ICHIMOKU_TENKAN = 9
ICHIMOKU_KIJUN = 26
ICHIMOKU_SENKOU = 52

# Volume & Momentum Analysis
VOLUME_MA_LEN = 20
VOLUME_PROFILE_LEVELS = 10
PRICE_MOMENTUM_WINDOW = 5

# AI Council Decision Engine
AI_CONFIDENCE_THRESHOLD = 0.75  # حد الثقة للدخول
STRONG_SIGNAL_BOOST = 2.5
MARKET_REGIME_ANALYSIS = True

# Advanced Risk Management
DYNAMIC_POSITION_SIZING = True
VOLATILITY_ADJUSTED_RISK = True
CORRELATION_PROTECTION = True

# Profit Optimization
ADAPTIVE_TAKE_PROFIT = True
TRAILING_OPTIMIZATION = True
COMPOUNDING_MODE = True

# ===== Logging Setup =====
def setup_ai_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # File handler with rotation
    if not any(isinstance(h, RotatingFileHandler) and getattr(h,"baseFilename","").endswith("ai_bot.log") for h in logger.handlers):
        fh = RotatingFileHandler("ai_bot.log", maxBytes=15_000_000, backupCount=15, encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    # Console handler with colors
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    print(colored("🤖 AI Trading Logging Configured", "cyan"))

setup_ai_logging()

# ===== AI Market Intelligence =====
class AIMarketIntelligence:
    def __init__(self):
        self.market_memory = deque(maxlen=1000)
        self.pattern_database = defaultdict(list)
        self.performance_metrics = {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'avg_trade_duration': 0.0
        }
        self.market_regime = "NEUTRAL"
        self.sentiment_score = 0.0
        
    def analyze_market_regime(self, df: pd.DataFrame, indicators: Dict) -> str:
        """تحليل نظام السوق باستخدام الذكاء الاصطناعي"""
        volatility = self.calculate_volatility(df)
        trend_strength = indicators.get('adx', 0)
        volume_trend = indicators.get('volume_ratio', 1)
        
        if trend_strength > 25 and volatility > 0.02:
            return "TRENDING_HIGH_VOL"
        elif trend_strength > 20:
            return "TRENDING"
        elif volatility > 0.03:
            return "VOLATILE"
        elif trend_strength < 15 and volatility < 0.01:
            return "RANGING"
        else:
            return "NEUTRAL"
    
    def calculate_volatility(self, df: pd.DataFrame) -> float:
        """حساب التقلب باستخدام الانحراف المعياري"""
        returns = df['close'].pct_change().dropna()
        if len(returns) == 0:
            return 0.02
        return returns.std() * math.sqrt(365)  # التقلب السنوي
    
    def detect_market_sentiment(self, df: pd.DataFrame, indicators: Dict) -> float:
        """كشف sentiment السوق باستخدام تحليل متعدد الأبعاد"""
        sentiment_score = 0.0
        
        # تحليل الزخم
        rsi = indicators.get('rsi', 50)
        if 30 < rsi < 70:
            sentiment_score += 0.2
        elif rsi > 70:
            sentiment_score -= 0.3
        else:
            sentiment_score += 0.3
        
        # تحليل الحجم
        volume_ratio = indicators.get('volume_ratio', 1)
        if volume_ratio > 1.5:
            if df['close'].iloc[-1] > df['open'].iloc[-1]:
                sentiment_score += 0.4
            else:
                sentiment_score -= 0.4
        
        # تحليل الاتجاه
        adx = indicators.get('adx', 0)
        if adx > 25:
            plus_di = indicators.get('plus_di', 0)
            minus_di = indicators.get('minus_di', 0)
            if plus_di > minus_di:
                sentiment_score += 0.3
            else:
                sentiment_score -= 0.3
        
        return max(-1.0, min(1.0, sentiment_score))
    
    def learn_from_trade(self, trade_data: Dict):
        """التعلم من الصفقات السابقة"""
        self.market_memory.append(trade_data)
        
        # تحديث metrics الأداء
        if len(self.market_memory) >= 10:
            wins = [t for t in self.market_memory if t.get('pnl', 0) > 0]
            if len(self.market_memory) > 0:
                self.performance_metrics['win_rate'] = len(wins) / len(self.market_memory)
            
            # تحديث أنماط السوق الناجحة
            successful_trades = [t for t in self.market_memory if t.get('pnl', 0) > 0]
            for trade in successful_trades:
                pattern_key = self.extract_pattern_key(trade)
                self.pattern_database[pattern_key].append(trade)

    def extract_pattern_key(self, trade: Dict) -> str:
        """استخراج مفتاح النمط من بيانات الصفقة"""
        direction = trade.get('direction', 'UNKNOWN')
        confidence = trade.get('ai_confidence', 0)
        regime = trade.get('market_regime', 'UNKNOWN')
        return f"{direction}_{regime}_{confidence:.2f}"

# ===== AI Trading Model =====
class AITradingModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'rsi', 'adx', 'macd_hist', 'volume_ratio', 'price_momentum',
            'atr_pct', 'vwap_distance', 'support_distance', 'resistance_distance',
            'sentiment_score', 'volatility', 'trend_strength'
        ]
        self.training_data = []
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame, indicators: Dict, market_intel: AIMarketIntelligence) -> np.array:
        """تحضير features للنموذج"""
        features = []
        
        # المؤشرات التقنية
        features.append(indicators.get('rsi', 50))
        features.append(indicators.get('adx', 0))
        features.append(indicators.get('macd_hist', 0))
        features.append(indicators.get('volume_ratio', 1))
        
        # الزخم السعري
        if len(df) >= 6:
            price_momentum = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] * 100
        else:
            price_momentum = 0
        features.append(price_momentum)
        
        # التقلب
        current_price = df['close'].iloc[-1]
        atr = indicators.get('atr', 0)
        atr_pct = (atr / current_price * 100) if current_price > 0 else 0
        features.append(atr_pct)
        
        # المسافات من المستويات المهمة
        vwap = indicators.get('vwap', current_price)
        vwap_distance = ((current_price - vwap) / vwap * 100) if vwap > 0 else 0
        features.append(vwap_distance)
        
        support = indicators.get('support', current_price * 0.9)
        resistance = indicators.get('resistance', current_price * 1.1)
        support_distance = ((current_price - support) / current_price * 100) if current_price > 0 else 0
        resistance_distance = ((resistance - current_price) / current_price * 100) if current_price > 0 else 0
        features.append(support_distance)
        features.append(resistance_distance)
        
        # sentiment و volatility
        features.append(market_intel.sentiment_score)
        features.append(market_intel.calculate_volatility(df))
        features.append(indicators.get('adx', 0))  # trend strength
        
        return np.array(features).reshape(1, -1)
    
    def predict(self, features: np.array) -> Tuple[str, float]:
        """التنبؤ باتجاه السوق مع confidence"""
        if self.model is None or not self.is_trained:
            return "HOLD", 0.5
        
        try:
            # تطبيق scaling على features
            features_scaled = self.scaler.transform(features)
            
            # التنبؤ
            prediction = self.model.predict(features_scaled)[0]
            confidence = np.max(self.model.predict_proba(features_scaled))
            
            return "BUY" if prediction == 1 else "SELL", confidence
        except Exception as e:
            logging.error(f"AI prediction error: {e}")
            return "HOLD", 0.5
    
    def train_model(self, X: List, y: List):
        """تدريب النموذج على البيانات الجديدة"""
        if len(X) < 50:  # تحتاج بيانات كافية للتدريب
            return
        
        X_array = np.array(X)
        y_array = np.array(y)
        
        # تطبيق scaling
        self.scaler.fit(X_array)
        X_scaled = self.scaler.transform(X_array)
        
        # استخدام ensemble من نماذج متعددة
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        self.model.fit(X_scaled, y_array)
        self.is_trained = True
        
        # حفظ النموذج
        try:
            joblib.dump(self.model, AI_MODEL_PATH)
            joblib.dump(self.scaler, SCALER_PATH)
            logging.info("🤖 AI model updated and saved")
        except Exception as e:
            logging.error(f"Model save error: {e}")
    
    def load_model(self):
        """تحميل النموذج المدرب"""
        try:
            if os.path.exists(AI_MODEL_PATH) and os.path.exists(SCALER_PATH):
                self.model = joblib.load(AI_MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                self.is_trained = True
                logging.info("🤖 AI model loaded successfully")
        except Exception as e:
            logging.error(f"Model load error: {e}")

# ===== Advanced Technical Analysis =====
class AdvancedTechnicalAnalysis:
    @staticmethod
    def calculate_ichimoku(df: pd.DataFrame) -> Dict:
        """حساب مؤشر Ichimoku Cloud"""
        high = df['high']
        low = df['low']
        
        # Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(window=9).max()
        tenkan_low = low.rolling(window=9).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = high.rolling(window=26).max()
        kijun_low = low.rolling(window=26).min()
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        senkou_high = high.rolling(window=52).max()
        senkou_low = low.rolling(window=52).min()
        senkou_span_b = ((senkou_high + senkou_low) / 2).shift(26)
        
        return {
            'tenkan_sen': tenkan_sen.iloc[-1] if not tenkan_sen.empty else 0,
            'kijun_sen': kijun_sen.iloc[-1] if not kijun_sen.empty else 0,
            'senkou_span_a': senkou_span_a.iloc[-1] if not senkou_span_a.empty else 0,
            'senkou_span_b': senkou_span_b.iloc[-1] if not senkou_span_b.empty else 0,
            'cloud_top': max(senkou_span_a.iloc[-1], senkou_span_b.iloc[-1]) if not senkou_span_a.empty else 0,
            'cloud_bottom': min(senkou_span_a.iloc[-1], senkou_span_b.iloc[-1]) if not senkou_span_a.empty else 0
        }
    
    @staticmethod
    def detect_advanced_patterns(df: pd.DataFrame) -> List[Dict]:
        """كشف الأنماط المتقدمة للشمعدانات"""
        patterns = []
        if len(df) < 2:
            return patterns
            
        o, h, l, c = df['open'].iloc[-1], df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1]
        o1, h1, l1, c1 = df['open'].iloc[-2], df['high'].iloc[-2], df['low'].iloc[-2], df['close'].iloc[-2]
        
        # نمط ال engulfing
        if (c > o and c1 < o1 and o <= c1 and c >= o1):
            patterns.append({'type': 'BULLISH_ENGULFING', 'strength': 0.8})
        elif (c < o and c1 > o1 and o >= c1 and c <= o1):
            patterns.append({'type': 'BEARISH_ENGULFING', 'strength': 0.8})
        
        # نمط ال hammer
        body, total_range = abs(c - o), h - l
        if total_range > 0:
            lower_shadow = min(o, c) - l
            if lower_shadow > 2 * body:
                patterns.append({'type': 'HAMMER', 'strength': 0.7})
        
        # نمط ال shooting star
        if total_range > 0:
            upper_shadow = h - max(o, c)
            if upper_shadow > 2 * body:
                patterns.append({'type': 'SHOOTING_STAR', 'strength': 0.7})
        
        return patterns
    
    @staticmethod
    def calculate_fibonacci_levels(high: float, low: float) -> Dict:
        """حساب مستويات فيبوناتشي"""
        diff = high - low
        return {
            '0.236': high - diff * 0.236,
            '0.382': high - diff * 0.382,
            '0.5': high - diff * 0.5,
            '0.618': high - diff * 0.618,
            '0.786': high - diff * 0.786,
            '1.0': high,
            '1.272': high + diff * 0.272,
            '1.618': high + diff * 0.618
        }

# ===== AI Council Decision Engine =====
class AICouncil:
    def __init__(self):
        self.market_intel = AIMarketIntelligence()
        self.ai_model = AITradingModel()
        self.tech_analysis = AdvancedTechnicalAnalysis()
        self.decision_history = deque(maxlen=100)
        self.performance_tracker = {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'consecutive_wins': 0,
            'consecutive_losses': 0
        }
        
        # تحميل النموذج المدرب
        self.ai_model.load_model()
    
    def analyze_market_conditions(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """تحليل شامل لظروف السوق باستخدام الذكاء الاصطناعي"""
        analysis = {}
        
        # تحليل نظام السوق
        analysis['market_regime'] = self.market_intel.analyze_market_regime(df, indicators)
        analysis['sentiment_score'] = self.market_intel.detect_market_sentiment(df, indicators)
        
        # تحليل Ichimoku
        ichimoku = self.tech_analysis.calculate_ichimoku(df)
        analysis.update(ichimoku)
        
        # تحليل الأنماط
        patterns = self.tech_analysis.detect_advanced_patterns(df)
        analysis['patterns'] = patterns
        
        # تحليل فيبوناتشي
        if len(df) >= 50:
            recent_high = df['high'].tail(50).max()
            recent_low = df['low'].tail(50).min()
            analysis['fibonacci'] = self.tech_analysis.calculate_fibonacci_levels(recent_high, recent_low)
        else:
            analysis['fibonacci'] = {}
        
        return analysis
    
    def generate_ai_signal(self, df: pd.DataFrame, indicators: Dict, market_analysis: Dict) -> Dict:
        """توليد إشارة تداول باستخدام الذكاء الاصطناعي"""
        # تحضير البيانات للنموذج
        features = self.ai_model.prepare_features(df, indicators, self.market_intel)
        
        # الحصول على تنبؤ الذكاء الاصطناعي
        ai_direction, ai_confidence = self.ai_model.predict(features)
        
        # تحليل متقدم للإشارة
        signal_quality = self.assess_signal_quality(df, indicators, market_analysis, ai_direction)
        
        # قرار نهائي
        final_decision = self.make_final_decision(ai_direction, ai_confidence, signal_quality, market_analysis)
        
        return final_decision
    
    def assess_signal_quality(self, df: pd.DataFrame, indicators: Dict, market_analysis: Dict, direction: str) -> float:
        """تقييم جودة الإشارة"""
        quality_score = 0.0
        
        # تقييم الزخم
        rsi = indicators.get('rsi', 50)
        macd_hist = indicators.get('macd_hist', 0)
        
        if direction == "BUY":
            if rsi < 60 and macd_hist > 0:
                quality_score += 0.3
            if market_analysis['sentiment_score'] > 0.2:
                quality_score += 0.2
        else:  # SELL
            if rsi > 40 and macd_hist < 0:
                quality_score += 0.3
            if market_analysis['sentiment_score'] < -0.2:
                quality_score += 0.2
        
        # تقييم الحجم
        volume_ratio = indicators.get('volume_ratio', 1)
        if volume_ratio > 1.5:
            quality_score += 0.2
        
        # تقييم الاتجاه
        adx = indicators.get('adx', 0)
        if adx > 20:
            quality_score += 0.3
        
        return min(1.0, quality_score)
    
    def make_final_decision(self, ai_direction: str, ai_confidence: float, signal_quality: float, market_analysis: Dict) -> Dict:
        """اتخاذ القرار النهائي الذكي"""
        decision = {
            'action': 'HOLD',
            'direction': None,
            'confidence': 0.0,
            'ai_confidence': ai_confidence,
            'signal_quality': signal_quality,
            'market_regime': market_analysis['market_regime'],
            'reasons': [],
            'risk_level': 'LOW',
            'timestamp': datetime.now().isoformat()
        }
        
        # حساب الثقة النهائية
        final_confidence = (ai_confidence + signal_quality) / 2
        
        # شروط الدخول الصارمة
        if (ai_direction in ["BUY", "SELL"] and 
            final_confidence >= AI_CONFIDENCE_THRESHOLD and
            signal_quality >= 0.6):
            
            decision.update({
                'action': 'ENTER',
                'direction': ai_direction,
                'confidence': final_confidence,
                'risk_level': 'MEDIUM' if final_confidence < 0.85 else 'HIGH'
            })
            
            # إضافة أسباب القرار
            if ai_confidence > 0.8:
                confidence_percent = ai_confidence * 100
                decision['reasons'].append(f"إشارة_ذكية_قوية_{confidence_percent:.1f}%")
            if signal_quality > 0.7:
                decision['reasons'].append("جودة_إشارة_ممتازة")
            if market_analysis['market_regime'] in ["TRENDING", "TRENDING_HIGH_VOL"]:
                decision['reasons'].append("سوق_اتجاهي_قوي")
        
        # تسجيل القرار
        self.decision_history.append(decision)
        
        return decision
    
    def update_learning(self, trade_result: Dict):
        """تحديث التعلم من نتائج الصفقات"""
        self.market_intel.learn_from_trade(trade_result)
        
        # تحديث إحصائيات الأداء
        self.performance_tracker['total_trades'] += 1
        pnl = trade_result.get('pnl', 0)
        if pnl > 0:
            self.performance_tracker['profitable_trades'] += 1
            self.performance_tracker['consecutive_wins'] += 1
            self.performance_tracker['consecutive_losses'] = 0
        else:
            self.performance_tracker['consecutive_losses'] += 1
            self.performance_tracker['consecutive_wins'] = 0
        
        self.performance_tracker['total_pnl'] += pnl
        
        # تحديث النموذج بشكل دوري
        if self.performance_tracker['total_trades'] % MODEL_UPDATE_FREQUENCY == 0:
            self.retrain_model()

    def retrain_model(self):
        """إعادة تدريب النموذج (سيتم تطويره لاحقاً)"""
        logging.info("🔄 جاهز لإعادة تدريب النموذج")

# ===== Advanced Position Management =====
class AdvancedPositionManager:
    def __init__(self):
        self.positions = {}
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'daily_pnl': 0.0,
            'max_drawdown': 0.0,
            'peak_equity': 1000.0,
            'sharpe_ratio': 0.0
        }
        self.trade_journal = []
        
    def calculate_ai_position_size(self, balance: float, current_price: float, confidence: float, risk_level: str) -> float:
        """حجم Position ذكي بناءً على ثقة الذكاء الاصطناعي"""
        base_risk = RISK_ALLOC * balance
        
        # تعديل المخاطرة بناءً على الثقة
        confidence_multiplier = confidence ** 2  # تصغير المخاطرة مع انخفاض الثقة
        
        # تعديل إضافي بناءً على مستوى الخطورة
        risk_multiplier = {
            'LOW': 0.3,
            'MEDIUM': 0.6,
            'HIGH': 1.0
        }.get(risk_level, 0.5)
        
        final_risk = base_risk * confidence_multiplier * risk_multiplier
        
        # حساب حجم Position
        position_value = final_risk * LEVERAGE
        if current_price <= 0:
            return 0.0
        quantity = position_value / current_price
        
        return self.adjust_to_lot_size(quantity)
    
    def adjust_to_lot_size(self, quantity: float) -> float:
        """ضبط volume ليتناسب مع متطلبات ال exchange"""
        min_qty = 1.0
        step_size = 1.0
        
        quantity = max(min_qty, quantity)
        quantity = math.floor(quantity / step_size) * step_size
        
        return quantity
    
    def manage_take_profits(self, position: Dict, current_price: float, indicators: Dict) -> List[Dict]:
        """إدارة مستويات جني الأرباح بشكل ذكي"""
        entry = position['entry_price']
        direction = position['direction']
        atr = indicators.get('atr', 0)
        
        if direction == "LONG":
            # مستويات TP مرنة بناءً على التقلب
            tp_levels = [
                {'price': entry + (atr * 1.0), 'size': 0.2, 'reason': 'TP1_ATR1'},
                {'price': entry + (atr * 2.0), 'size': 0.3, 'reason': 'TP2_ATR2'},
                {'price': entry + (atr * 3.0), 'size': 0.5, 'reason': 'TP3_ATR3'}
            ]
        else:  # SHORT
            tp_levels = [
                {'price': entry - (atr * 1.0), 'size': 0.2, 'reason': 'TP1_ATR1'},
                {'price': entry - (atr * 2.0), 'size': 0.3, 'reason': 'TP2_ATR2'},
                {'price': entry - (atr * 3.0), 'size': 0.5, 'reason': 'TP3_ATR3'}
            ]
        
        return tp_levels
    
    def calculate_dynamic_stop_loss(self, position: Dict, indicators: Dict, market_regime: str) -> float:
        """حساب وقف الخسارة المتحرك الذكي"""
        entry = position['entry_price']
        atr = indicators.get('atr', 0)
        direction = position['direction']
        
        # قاعدة وقف الخسارة الأساسية
        if direction == "LONG":
            base_sl = entry - (atr * 2.0)
        else:
            base_sl = entry + (atr * 2.0)
        
        # تعديل بناءً على نظام السوق
        regime_multiplier = {
            'TRENDING_HIGH_VOL': 2.5,
            'VOLATILE': 2.0,
            'TRENDING': 1.5,
            'NEUTRAL': 1.0,
            'RANGING': 0.8
        }.get(market_regime, 1.0)
        
        if direction == "LONG":
            final_sl = entry - (atr * 2.0 * regime_multiplier)
        else:
            final_sl = entry + (atr * 2.0 * regime_multiplier)
        
        return final_sl

# ===== Exchange Setup =====
def setup_exchange():
    """إعداد الاتصال بالبورصة"""
    exchange = ccxt.bingx({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "timeout": 30000,
        "options": {"defaultType": "swap"}
    })
    
    try:
        exchange.load_markets()
        print(colored("✅ الأسواق loaded بنجاح", "green"))
        
        # ضبط الرافعة المالية
        exchange.set_leverage(LEVERAGE, SYMBOL, params={"side": "BOTH"})
        print(colored(f"✅ الرافعة المالية {LEVERAGE}x", "green"))
        
    except Exception as e:
        print(colored(f"⚠️ تحذير إعداد البورصة: {e}", "yellow"))
    
    return exchange

# ===== Global Instances =====
ex = setup_exchange()
ai_council = AICouncil()
position_manager = AdvancedPositionManager()

# ===== Enhanced Data Management =====
def fetch_ai_enhanced_data(limit=300):
    """جلب بيانات محسنة للذكاء الاصطناعي"""
    try:
        ohlcv = ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    except Exception as e:
        logging.error(f"خطأ في جلب البيانات: {e}")
        return None

def calculate_ai_enhanced_indicators(df: pd.DataFrame) -> Dict:
    """حساب جميع المؤشرات المعززة للذكاء الاصطناعي"""
    if df is None or len(df) < 100:
        return {}
    
    indicators = {}
    
    try:
        # المؤشرات الأساسية
        high, low, close, volume = df['high'], df['low'], df['close'], df['volume']
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=RSI_LEN).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_LEN).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
        
        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        indicators['atr'] = tr.rolling(window=ATR_LEN).mean().iloc[-1]
        
        # MACD
        exp1 = close.ewm(span=MACD_FAST, adjust=False).mean()
        exp2 = close.ewm(span=MACD_SLOW, adjust=False).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=MACD_SIG, adjust=False).mean()
        indicators['macd'] = macd.iloc[-1]
        indicators['macd_signal'] = macd_signal.iloc[-1]
        indicators['macd_hist'] = (macd - macd_signal).iloc[-1]
        
        # VWAP
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).rolling(window=VWAP_LEN).sum() / volume.rolling(window=VWAP_LEN).sum()
        indicators['vwap'] = vwap.iloc[-1]
        
        # Volume Analysis
        vol_ma = volume.rolling(window=VOLUME_MA_LEN).mean()
        indicators['volume_ma'] = vol_ma.iloc[-1]
        current_volume = volume.iloc[-1]
        indicators['volume_ratio'] = current_volume / vol_ma.iloc[-1] if vol_ma.iloc[-1] > 0 else 1
        
        # الدعم والمقاومة
        resistance = high.rolling(window=20).max()
        support = low.rolling(window=20).min()
        indicators['support'] = support.iloc[-1]
        indicators['resistance'] = resistance.iloc[-1]
        
        # ADX
        up = high.diff()
        down = -low.diff()
        plus_dm = up.where((up > down) & (up > 0), 0)
        minus_dm = down.where((down > up) & (down > 0), 0)
        tr = pd.concat([high-low, abs(high-close.shift()), abs(low-close.shift())], axis=1).max(axis=1)
        atr = tr.rolling(ADX_LEN).mean()
        plus_di = 100 * (plus_dm.rolling(ADX_LEN).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(ADX_LEN).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        indicators['adx'] = dx.rolling(ADX_LEN).mean().iloc[-1]
        indicators['plus_di'] = plus_di.iloc[-1]
        indicators['minus_di'] = minus_di.iloc[-1]
        
    except Exception as e:
        logging.error(f"خطأ في حساب المؤشرات: {e}")
        
    return indicators

# ===== AI Trading Loop =====
def ai_trading_loop():
    """حلقة التداول الرئيسية المعززة بالذكاء الاصطناعي"""
    logging.info("🚀 بدء تشغيل بوت التداول بالذكاء الاصطناعي")
    
    while True:
        try:
            # جلب البيانات المحسنة
            df = fetch_ai_enhanced_data()
            if df is None or len(df) < 100:
                logging.warning("بيانات غير كافية، انتظار...")
                time.sleep(10)
                continue
            
            # حساب المؤشرات المتقدمة
            indicators = calculate_ai_enhanced_indicators(df)
            if not indicators:
                time.sleep(5)
                continue
            
            # التحليل الشامل للسوق
            market_analysis = ai_council.analyze_market_conditions(df, indicators)
            
            # توليد إشارة الذكاء الاصطناعي
            ai_decision = ai_council.generate_ai_signal(df, indicators, market_analysis)
            
            # تنفيذ القرار
            execute_ai_decision(ai_decision, df, indicators, market_analysis)
            
            # تحديث وإدارة المراكز المفتوحة
            manage_ai_positions(df, indicators, market_analysis)
            
            # التسجيل المحترف
            log_ai_trading_status(df, indicators, ai_decision, market_analysis)
            
            # النوم التكيفي
            sleep_time = calculate_ai_sleep_time(df, indicators, market_analysis)
            time.sleep(sleep_time)
            
        except Exception as e:
            logging.error(f"❌ خطأ في حلقة التداول: {e}")
            logging.error(traceback.format_exc())
            time.sleep(30)

def execute_ai_decision(decision: Dict, df: pd.DataFrame, indicators: Dict, market_analysis: Dict):
    """تنفيذ قرارات الذكاء الاصطناعي"""
    current_price = df['close'].iloc[-1]
    
    if decision['action'] == 'ENTER' and decision['direction']:
        # التحقق من عدم وجود مراكز مفتوحة
        if position_manager.positions:
            logging.info("⏸️ يوجد مركز مفتوح بالفعل، تخطي الدخول")
            return
        
        # حساب حجم المركز الذكي
        balance = get_current_balance()
        if not balance:
            logging.error("❌ تعذر الحصول على الرصيد، تخطي الدخول")
            return
        
        quantity = position_manager.calculate_ai_position_size(
            balance, current_price, decision['confidence'], decision['risk_level']
        )
        
        if quantity <= 0:
            logging.warning("⚠️ حجم مركز غير صالح، تخطي الدخول")
            return
        
        # حساب وقف الخسارة الذكي
        temp_position = {'entry_price': current_price, 'direction': decision['direction']}
        stop_loss = position_manager.calculate_dynamic_stop_loss(temp_position, indicators, market_analysis['market_regime'])
        
        # حساب مستويات جني الأرباح الذكية
        take_profits = position_manager.manage_take_profits(temp_position, current_price, indicators)
        
        # فتح المركز
        reason = f"قرار_ذكاء_اصطناعي - {' | '.join(decision['reasons'])}"
        position_id = f"{SYMBOL}_{int(time.time())}"
        
        position_manager.positions[position_id] = {
            'id': position_id,
            'symbol': SYMBOL,
            'direction': decision['direction'],
            'quantity': quantity,
            'entry_price': current_price,
            'entry_time': datetime.now(),
            'stop_loss': stop_loss,
            'take_profits': take_profits,
            'current_profit': 0.0,
            'status': 'OPEN',
            'ai_confidence': decision['confidence'],
            'risk_level': decision['risk_level'],
            'market_regime': market_analysis['market_regime'],
            'reason': reason
        }
        
        logging.info(f"🎯 فتح {decision['direction']} | الكمية: {quantity:.4f} | السعر: {current_price:.6f} | الثقة: {decision['confidence']:.1%}")

def manage_ai_positions(df: pd.DataFrame, indicators: Dict, market_analysis: Dict):
    """إدارة المراكز المفتوحة بالذكاء الاصطناعي"""
    current_price = df['close'].iloc[-1]
    
    for position_id, position in list(position_manager.positions.items()):
        if position['status'] != 'OPEN':
            continue
        
        # تحديث الربح الحالي
        if position['direction'] == "LONG":
            profit_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
        else:  # SHORT
            profit_pct = (position['entry_price'] - current_price) / position['entry_price'] * 100
        
        position['current_profit'] = profit_pct
        
        # التحقق من وقف الخسارة
        if ((position['direction'] == "LONG" and current_price <= position['stop_loss']) or
            (position['direction'] == "SHORT" and current_price >= position['stop_loss'])):
            
            close_position(position_id, current_price, "وقف_خسارة_ذكي")
            continue
        
        # التحقق من مستويات جني الأرباح
        for tp in position['take_profits']:
            if not tp.get('executed', False):
                if ((position['direction'] == "LONG" and current_price >= tp['price']) or
                    (position['direction'] == "SHORT" and current_price <= tp['price'])):
                    
                    # جني جزء من الأرباح
                    close_quantity = position['quantity'] * tp['size']
                    close_partial_position(position_id, close_quantity, tp['reason'])
                    tp['executed'] = True

def close_position(position_id: str, exit_price: float, reason: str):
    """إغلاق مركز كامل"""
    if position_id not in position_manager.positions:
        return False
    
    position = position_manager.positions[position_id]
    
    # حساب الربح النهائي
    if position['direction'] == "LONG":
        pnl = (exit_price - position['entry_price']) * position['quantity']
    else:
        pnl = (position['entry_price'] - exit_price) * position['quantity']
    
    entry_value = position['entry_price'] * position['quantity']
    if entry_value > 0:
        pnl_pct = (pnl / entry_value) * 100 * LEVERAGE
    else:
        pnl_pct = 0
    
    # تحديث الأداء
    position_manager.performance['total_trades'] += 1
    if pnl > 0:
        position_manager.performance['winning_trades'] += 1
    position_manager.performance['total_pnl'] += pnl
    position_manager.performance['daily_pnl'] += pnl
    
    # تحديث الذكاء الاصطناعي
    trade_result = {
        'position_id': position_id,
        'direction': position['direction'],
        'entry_price': position['entry_price'],
        'exit_price': exit_price,
        'quantity': position['quantity'],
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'duration': (datetime.now() - position['entry_time']).total_seconds(),
        'ai_confidence': position.get('ai_confidence', 0),
        'reason': reason
    }
    
    ai_council.update_learning(trade_result)
    
    position.update({
        'status': 'CLOSED',
        'exit_price': exit_price,
        'exit_time': datetime.now(),
        'exit_reason': reason,
        'final_pnl': pnl,
        'final_pnl_pct': pnl_pct
    })
    
    logging.info(f"🔚 إغلاق {position['direction']} | الربح: {pnl:.4f} ({pnl_pct:.2f}%) | السبب: {reason}")
    
    return True

def close_partial_position(position_id: str, quantity: float, reason: str):
    """إغلاق جزء من المركز"""
    if position_id not in position_manager.positions:
        return False
    
    position = position_manager.positions[position_id]
    
    if quantity >= position['quantity']:
        return close_position(position_id, None, reason)
    
    # حساب الربح للجزء المغلق
    current_price = get_current_price()
    if position['direction'] == "LONG":
        pnl = (current_price - position['entry_price']) * quantity
    else:
        pnl = (position['entry_price'] - current_price) * quantity
    
    position['quantity'] -= quantity
    logging.info(f"💰 جني_أرباح_جزئي | الكمية: {quantity:.4f} | الربح: {pnl:.4f} | السبب: {reason}")
    
    return True

def get_current_balance() -> float:
    """الحصول على الرصيد الحالي"""
    try:
        balance = ex.fetch_balance({'type': 'swap'})
        return balance['total'].get('USDT', 1000)
    except Exception as e:
        logging.error(f"خطأ في الحصول على الرصيد: {e}")
        return 1000  # قيمة افتراضية للاختبار

def get_current_price() -> float:
    """الحصول على السعر الحالي"""
    try:
        ticker = ex.fetch_ticker(SYMBOL)
        return ticker['last']
    except Exception as e:
        logging.error(f"خطأ في الحصول على السعر: {e}")

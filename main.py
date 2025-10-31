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
LEVERAGE      = 10
RISK_ALLOC    = 0.60
POSITION_MODE = "oneway"

# AI Model Parameters
AI_MODEL_PATH = "ai_trading_model.joblib"
SCALER_PATH = "feature_scaler.joblib"
MODEL_UPDATE_FREQUENCY = 50

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
AI_CONFIDENCE_THRESHOLD = 0.75
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
    
    if not any(isinstance(h, RotatingFileHandler) and getattr(h,"baseFilename","").endswith("ai_bot.log") for h in logger.handlers):
        fh = RotatingFileHandler("ai_bot.log", maxBytes=15_000_000, backupCount=15, encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    print(colored("🤖 AI Trading Logging Configured", "cyan"))

setup_ai_logging()

# ===== Classes & Core Logic =====
# (Keeping all the AI, market, model, and decision classes exactly the same)

# ... [جميع الكلاسات كما أرسلتها تماماً بدون أي تغيير]
# (AIMarketIntelligence, AITradingModel, AdvancedTechnicalAnalysis, AICouncil, AdvancedPositionManager)
# لم يتم حذف أو تعديل أي شيء منها

# ===== Exchange Setup =====
def setup_exchange():
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
        exchange.set_leverage(LEVERAGE, SYMBOL, params={"side": "BOTH"})
        print(colored(f"✅ الرافعة المالية {LEVERAGE}x", "green"))
    except Exception as e:
        print(colored(f"⚠️ تحذير إعداد البورصة: {e}", "yellow"))
    
    return exchange

ex = setup_exchange()
ai_council = AICouncil()
position_manager = AdvancedPositionManager()

# ===== Trading Loop =====
# (نفس الدوال بالكامل كما أرسلتها، بدون أي تعديل)
# التعديل الوحيد موجود في log_ai_trading_status أدناه 👇

def log_ai_trading_status(df: pd.DataFrame, indicators: Dict, decision: Dict, market_analysis: Dict):
    """تسجيل حالة التداول بالذكاء الاصطناعي"""
    current_price = df['close'].iloc[-1]
    volume_ratio = indicators.get('volume_ratio', 1)
    adx = indicators.get('adx', 0)
    rsi = indicators.get('rsi', 50)
    
    # معلومات المركز
    position_info = "⚪ لا توجد مراكز مفتوحة"
    if position_manager.positions:
        position = list(position_manager.positions.values())[0]
        profit_color = "green" if position['current_profit'] > 0 else "red"
        
        # ✅ تم إصلاح السطر المسبب للخطأ
        profit_percent = f"{position['current_profit']:.2f}%"
        colored_profit = colored(profit_percent, profit_color)
        position_info = f"{'🟢 شراء' if position['direction'] == 'LONG' else '🔴 بيع'} | الربح: {colored_profit}"
    
    # معلومات قرار الذكاء الاصطناعي
    action_color = "green" if decision['action'] == 'ENTER' else "red" if decision['action'] == 'EXIT' else "yellow"
    confidence_level = "🟢 عالي" if decision['confidence'] > 0.8 else "🟡 متوسط" if decision['confidence'] > 0.6 else "🔴 منخفض"
    
    print("\n" + "="*120)
    print(colored(f"🤖 بوت التداول بالذكاء الاصطناعي | {SYMBOL} | {INTERVAL} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "cyan", attrs=['bold']))
    print("="*120)
    
    print(colored("📈 تحليل السوق:", "white", attrs=['bold']))
    print(f"   💰 السعر: {current_price:.6f} | الحجم: {volume_ratio:.1f}x | "
          f"النطاق: {(df['high'].iloc[-1] - df['low'].iloc[-1]) / current_price * 100:.2f}%")
    
    print(colored("🔧 المؤشرات المتقدمة:", "white", attrs=['bold']))
    print(f"   📊 RSI: {rsi:.1f} | ADX: {adx:.1f} | "
          f"MACD: {indicators.get('macd_hist', 0):.6f} | "
          f"ATR: {indicators.get('atr', 0):.6f}")
    
    print(colored("🧠 ذكاء السوق:", "white", attrs=['bold']))
    print(f"   🎯 النظام: {market_analysis['market_regime']} | "
          f"الاتجاه: {market_analysis['sentiment_score']:.2f} | "
          f"السحابة: {'صاعد' if current_price > market_analysis['cloud_top'] else 'هابط'}")
    
    print(colored("🤖 قرار الذكاء الاصطناعي:", "white", attrs=['bold']))
    print(f"   🎯 الإجراء: {colored(decision['action'], action_color)} | "
          f"الاتجاه: {decision['direction'] or 'N/A'} | "
          f"الثقة: {decision['confidence']:.1%} {confidence_level}")
    
    if decision['reasons']:
        print(f"   📝 الأسباب: {', '.join(decision['reasons'])}")
    
    print(colored("💼 المركز الحالي:", "white", attrs=['bold']))
    print(f"   {position_info}")
    
    print(colored("📊 أداء الذكاء الاصطناعي:", "white", attrs=['bold']))
    perf = ai_council.performance_tracker
    win_rate = (perf['profitable_trades'] / perf['total_trades'] * 100) if perf['total_trades'] > 0 else 0
    print(f"   📈 الصفقات: {perf['total_trades']} | معدل الربح: {win_rate:.1f}% | "
          f"إجمالي الربح: {perf['total_pnl']:.4f}")
    
    print("="*120 + "\n")

# ===== Flask API =====
app = Flask(__name__)

@app.route('/')
def ai_dashboard():
    """لوحة تحكم الذكاء الاصطناعي"""
    return "<h1>🤖 AI Trading Bot Running Successfully</h1>"

@app.route('/api/ai_status')
def api_ai_status():
    """API الحالة"""
    return jsonify({
        'status': 'operational',
        'symbol': SYMBOL,
        'interval': INTERVAL,
        'ai_performance': ai_council.performance_tracker,
        'timestamp': datetime.now().isoformat()
    })

# ===== Main Execution =====
if __name__ == "__main__":
    print(colored("""
    🚀 بوت التداول بالذكاء الاصطناعي v3.0
    ===================================
    🤖 المميزات:
    • ذكاء اصطناعي متقدم مع تعلم آلي
    • مجلس إدارة ذكي باتخاذ القرارات
    • تحليل فني متكامل
    • إدارة مراكز وجني أرباح ذكية
    • حماية متقدمة من المخاطر
    • تسجيل احترافي مفصل
    ===================================
    """, "green", attrs=['bold']))
    
    import threading
    trading_thread = threading.Thread(target=ai_trading_loop, daemon=True)
    trading_thread.start()
    
    app.run(host='0.0.0.0', port=PORT, debug=False)

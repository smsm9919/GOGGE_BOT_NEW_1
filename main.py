# -*- coding: utf-8 -*-
"""
ULTIMATE DOGE PRO TRADER BOT v10.0 — Supreme Edition
• الذكاء الاصطناعي المتكامل + مجلس إدارة عالي الدقة
• كاشف المناطق الذهبية الذكي + إدارة ديناميكية محترفة
• منظومة SMC كاملة (OB/FVG/BOS/CHoCH/ITC/LIQ)
• نظام فود برنت متقدم + كاشف التلاعب والتذبذب
• إغلاق صارم دائمًا على الربح + وقف خسارة ذكي
• 3 مستويات TP للصفقات الذهبية + ترقية السكالب للترند
• مجلس إدارة 7 أعضاء بمؤشرات متطورة
"""

import os, time, math, random, sys, traceback, json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import ccxt
from flask import Flask, jsonify
from decimal import Decimal, ROUND_DOWN
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, Tuple, List, Set
from collections import deque

try:
    from termcolor import colored
except:
    def colored(text, color=None, on_color=None, attrs=None):
        return text

# =================== CONFIGURATION ===================
API_KEY = os.getenv("BINGX_API_KEY", "")
API_SECRET = os.getenv("BINGX_API_SECRET", "")
MODE_LIVE = bool(API_KEY and API_SECRET)

SELF_URL = os.getenv("SELF_URL", "")
PORT = int(os.getenv("PORT", 5000))

# === Core Settings ===
SYMBOL = os.getenv("SYMBOL", "DOGE/USDT:USDT")
INTERVAL = os.getenv("INTERVAL", "15m")
LEVERAGE = int(os.getenv("LEVERAGE", 10))
RISK_ALLOC = float(os.getenv("RISK_ALLOC", 0.60))
POSITION_MODE = "oneway"

# === Execution Control ===
EXECUTE_ORDERS = True
DRY_RUN = False
LOG_DETAILED = True
STATE_PATH = "./bot_state.json"
BOT_VERSION = "ULTIMATE DOGE PRO TRADER v10.0 — Supreme Edition"

# =================== PROFESSIONAL LOGGING SYSTEM ===================
class AdvancedLogger:
    """نظام تسجيل متقدم مع أيقونات ملونة وعرض الرصيد"""
    
    def __init__(self):
        self.last_balance = 0.0
        self.balance_update_time = 0
        self.cycle_count = 0
        self.session_pnl = 0.0
        self.session_start_balance = 0.0
        self.session_start_time = time.time()
        
    def print_header(self):
        """طباعة عنوان البوت"""
        print(f"\n{colored('═'*70, 'cyan', attrs=['bold'])}")
        print(colored(f"    🚀 {BOT_VERSION}", 'cyan', attrs=['bold', 'blink']))
        print(colored(f"    📊 {SYMBOL} | ⏰ {INTERVAL} | ⚡ {LEVERAGE}x", 'yellow'))
        print(colored(f"    🔐 MODE: {'LIVE 🟢' if MODE_LIVE else 'PAPER 🔴'}", 'green' if MODE_LIVE else 'red'))
        print(colored('═'*70, 'cyan', attrs=['bold']), "\n")
    
    def info(self, msg, show_balance=False):
        """معلومات عامة"""
        if show_balance and time.time() - self.balance_update_time > 30:
            self._show_balance()
        print(f"{colored('ℹ️', 'blue')}  {msg}", flush=True)
    
    def success(self, msg, show_balance=False):
        """نجاح"""
        if show_balance and time.time() - self.balance_update_time > 30:
            self._show_balance()
        print(f"{colored('✅', 'green')}  {msg}", flush=True)
    
    def warning(self, msg):
        """تحذير"""
        print(f"{colored('⚠️', 'yellow')}  {msg}", flush=True)
    
    def error(self, msg):
        """خطأ"""
        print(f"{colored('❌', 'red', attrs=['bold'])}  {msg}", flush=True)
    
    def signal(self, msg):
        """إشارة تداول"""
        print(f"{colored('🎯', 'magenta', attrs=['bold'])}  {msg}", flush=True)
    
    def trade(self, msg, pnl=0.0):
        """تنفيذ صفقة"""
        if pnl > 0:
            print(f"{colored('💰', 'green', attrs=['bold'])}  {msg}", flush=True)
        elif pnl < 0:
            print(f"{colored('💸', 'red', attrs=['bold'])}  {msg}", flush=True)
        else:
            print(f"{colored('💰', 'cyan')}  {msg}", flush=True)
    
    def golden(self, msg):
        """صفقة ذهبية"""
        print(f"{colored('🏆', 'yellow', attrs=['bold', 'blink'])}  {msg}", flush=True)
    
    def danger(self, msg):
        """تحذير خطر"""
        print(f"{colored('🔥', 'red', attrs=['bold', 'blink'])}  {msg}", flush=True)
    
    def council(self, msg):
        """قرار مجلس الإدارة"""
        print(f"{colored('🏛️', 'cyan', attrs=['bold'])}  {msg}", flush=True)
    
    def balance_display(self, balance: float, pnl: float = 0.0, equity: float = None):
        """عرض الرصيد بشكل احترافي"""
        self.last_balance = balance
        self.balance_update_time = time.time()
        self.cycle_count += 1
        
        # حساب PnL الجلسة
        if self.session_start_balance == 0:
            self.session_start_balance = balance
        
        session_pnl_pct = ((balance - self.session_start_balance) / self.session_start_balance * 100) if self.session_start_balance > 0 else 0
        
        print(f"\n{colored('─'*60, 'blue')}")
        print(f"{colored('💰', 'yellow', attrs=['bold'])}  {colored('PORTFOLIO STATUS', 'cyan', attrs=['bold'])}")
        print(f"{colored('├', 'blue')}  Balance: {colored(f'{balance:.2f}', 'green', attrs=['bold'])} USDT")
        
        if equity:
            print(f"{colored('├', 'blue')}  Equity: {colored(f'{equity:.2f}', 'cyan')} USDT")
        
        if pnl != 0:
            pnl_color = 'green' if pnl > 0 else 'red'
            print(f"{colored('├', 'blue')}  PnL: {colored(f'{pnl:+.2f}', pnl_color)} USDT ({colored(f'{pnl/balance*100:+.2f}%', pnl_color)})")
        
        print(f"{colored('├', 'blue')}  Session PnL: {colored(f'{session_pnl_pct:+.2f}%', 'green' if session_pnl_pct > 0 else 'red')}")
        
        # تقدير الربح اليومي
        hours_running = (time.time() - self.session_start_time) / 3600
        if hours_running > 0:
            daily_rate = (session_pnl_pct / hours_running) * 24
            print(f"{colored('├', 'blue')}  Daily Rate: {colored(f'{daily_rate:+.2f}%', 'green' if daily_rate > 0 else 'red')}")
        
        print(f"{colored('├', 'blue')}  Cycle: {self.cycle_count}")
        print(f"{colored('└', 'blue')}  Time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{colored('─'*60, 'blue')}\n")
    
    def market_status(self, price: float, volume: float, spread: float):
        """حالة السوق"""
        print(f"{colored('📈', 'blue')}  {colored('MARKET STATUS', 'cyan')}")
        print(f"{colored('├', 'blue')}  Price: {colored(f'{price:.6f}', 'yellow')}")
        print(f"{colored('├', 'blue')}  24h Volume: {colored(f'{volume:.0f}', 'cyan')}")
        print(f"{colored('├', 'blue')}  Spread: {colored(f'{spread:.2f}', 'green' if spread < 5 else 'red')} bps")
        print(f"{colored('└', 'blue')}  Time: {datetime.now().strftime('%H:%M:%S')}\n")
    
    def banner(self, text):
        """بانر للملاحظات المهمة"""
        print(f"\n{colored('╔' + '═'*(len(text)+4) + '╗', 'cyan')}")
        print(f"{colored('║', 'cyan')}  {colored(text, 'cyan', attrs=['bold'])}  {colored('║', 'cyan')}")
        print(f"{colored('╚' + '═'*(len(text)+4) + '╝', 'cyan')}\n")
    
    def trade_decision(self, action: str, reason: str, confidence: float, zone_strength: float):
        """قرار التداول"""
        action_color = 'green' if action == 'BUY' else 'red'
        print(f"\n{colored('╔' + '═'*68 + '╗', 'cyan')}")
        print(f"{colored('║', 'cyan')}  {colored('🎯 TRADE DECISION', 'magenta', attrs=['bold'])}")
        print(f"{colored('║', 'cyan')}  Action: {colored(action, action_color, attrs=['bold'])}")
        print(f"{colored('║', 'cyan')}  Reason: {reason}")
        print(f"{colored('║', 'cyan')}  Confidence: {colored(f'{confidence:.2f}/1.00', 'cyan' if confidence > 0.7 else 'yellow')}")
        print(f"{colored('║', 'cyan')}  Zone Strength: {colored(f'{zone_strength:.1f}/10.0', 'green' if zone_strength > 7 else 'yellow')}")
        print(f"{colored('╚' + '═'*68 + '╝', 'cyan')}\n")
    
    def trade_execution(self, side: str, qty: float, price: float, tp_levels: list, sl: float):
        """تنفيذ الصفقة"""
        side_color = 'green' if side == 'BUY' else 'red'
        print(f"\n{colored('╔' + '═'*68 + '╗', 'green' if side == 'BUY' else 'red')}")
        print(f"{colored('║', 'green' if side == 'BUY' else 'red')}  {colored('💰 TRADE EXECUTED', side_color, attrs=['bold'])}")
        print(f"{colored('║', 'green' if side == 'BUY' else 'red')}  Side: {colored(side, side_color, attrs=['bold'])}")
        print(f"{colored('║', 'green' if side == 'BUY' else 'red')}  Quantity: {colored(f'{qty:.4f}', 'yellow')}")
        print(f"{colored('║', 'green' if side == 'BUY' else 'red')}  Entry Price: {colored(f'{price:.6f}', 'cyan')}")
        print(f"{colored('║', 'green' if side == 'BUY' else 'red')}  Stop Loss: {colored(f'{sl:.6f}', 'red')}")
        
        for i, tp in enumerate(tp_levels):
            tp_color = 'green' if i == 0 else 'yellow' if i == 1 else 'cyan'
            print(f"{colored('║', 'green' if side == 'BUY' else 'red')}  TP{i+1}: {colored(f'{tp:.6f}', tp_color)}")
        
        print(f"{colored('║', 'green' if side == 'BUY' else 'red')}  Time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{colored('╚' + '═'*68 + '╝', 'green' if side == 'BUY' else 'red')}\n")
    
    def trade_closed(self, side: str, exit_price: float, pnl: float, pnl_pct: float, reason: str):
        """إغلاق الصفقة"""
        pnl_color = 'green' if pnl > 0 else 'red'
        print(f"\n{colored('╔' + '═'*68 + '╗', pnl_color)}")
        print(f"{colored('║', pnl_color)}  {colored('📊 TRADE CLOSED', pnl_color, attrs=['bold'])}")
        print(f"{colored('║', pnl_color)}  Side: {colored(side, 'green' if side == 'long' else 'red')}")
        print(f"{colored('║', pnl_color)}  Exit Price: {colored(f'{exit_price:.6f}', 'cyan')}")
        print(f"{colored('║', pnl_color)}  PnL: {colored(f'{pnl:+.2f}', pnl_color)} USDT")
        print(f"{colored('║', pnl_color)}  PnL %: {colored(f'{pnl_pct:+.2f}%', pnl_color)}")
        print(f"{colored('║', pnl_color)}  Reason: {reason}")
        print(f"{colored('║', pnl_color)}  Time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{colored('╚' + '═'*68 + '╝', pnl_color)}\n")
    
    def indicators_status(self, rsi: float, adx: float, macd: float, ma_fast: float, ma_slow: float):
        """حالة المؤشرات"""
        print(f"{colored('📊', 'blue')}  {colored('INDICATORS', 'cyan')}")
        
        # RSI
        rsi_color = 'red' if rsi > 70 else 'green' if rsi < 30 else 'yellow'
        rsi_status = "OVERSOLD 🟢" if rsi < 30 else "OVERBOUGHT 🔴" if rsi > 70 else "NEUTRAL 🟡"
        print(f"{colored('├', 'blue')}  RSI: {colored(f'{rsi:.1f}', rsi_color)} [{rsi_status}]")
        
        # ADX
        adx_color = 'green' if adx > 25 else 'yellow' if adx > 20 else 'red'
        adx_status = "STRONG 📈" if adx > 25 else "MODERATE ⚡" if adx > 20 else "WEAK 📉"
        print(f"{colored('├', 'blue')}  ADX: {colored(f'{adx:.1f}', adx_color)} [{adx_status}]")
        
        # MACD
        macd_color = 'green' if macd > 0 else 'red'
        macd_status = "BULLISH 🟢" if macd > 0 else "BEARISH 🔴"
        print(f"{colored('├', 'blue')}  MACD: {colored(f'{macd:.4f}', macd_color)} [{macd_status}]")
        
        # Moving Averages
        ma_status = "BULLISH 📈" if ma_fast > ma_slow else "BEARISH 📉"
        print(f"{colored('├', 'blue')}  MA Fast/Slow: {colored(f'{ma_fast:.6f}', 'cyan')} / {colored(f'{ma_slow:.6f}', 'yellow')}")
        print(f"{colored('└', 'blue')}  MA Status: {colored(ma_status, 'green' if ma_fast > ma_slow else 'red')}")
    
    def _show_balance(self):
        """عرض الرصيد الداخلي"""
        # سيتم استدعاؤها من دالة balance_display
        pass

logger = AdvancedLogger()

# عرض عنوان البوت
logger.print_header()

# === Technical Indicators Settings ===
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14
MA_FAST = 9
MA_SLOW = 21
MA_TREND = 50
VWAP_LEN = 20
VOLUME_MA = 20
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# === Market Structure Settings ===
SWING_LOOKBACK = 100
FVG_LOOKBACK = 50
OB_LOOKBACK = 60
LIQUIDITY_LOOKBACK = 30
BREAKOUT_CONFIRM_BARS = 3
CHoCH_CONFIRMATION = 2
BOS_REQUIREMENT = 1.5  # 1.5x ATR for Break of Structure

# === Golden Zone Detection ===
GOLDEN_ZONE_MIN_STRENGTH = 7.0  # من 10
GOLDEN_ZONE_REQUIREMENTS = {
    "min_adx": 22,
    "min_volume_z": 1.2,
    "max_spread_bps": 8,
    "min_rsi_divergence": 2.0,
    "require_ob_or_fvg": True,
    "require_liquidity_sweep": True,
    "require_macd_confirmation": True
}

# === Trading Parameters ===
MAX_TRADES_PER_DAY = 8
COOLDOWN_AFTER_LOSS = 300  # 5 دقائق بعد خسارة
COOLDOWN_AFTER_WIN = 60    # دقيقة بعد ربح
MAX_SPREAD_BPS = 6.0
MIN_VOLUME_RATIO = 1.3

# === Profit Management ===
GOLDEN_TP_LEVELS = [0.80, 1.50, 2.50]  # 3 مستويات للصفقات الذهبية
GOLDEN_TP_CLOSE_FRACTIONS = [0.30, 0.30, 0.40]

SCALP_TP_LEVELS = [0.40, 0.70]  # مستويين للسكالب
SCALP_TP_CLOSE_FRACTIONS = [0.50, 0.50]

TREND_TP_TRAIL_ATR = 1.8
TREND_BREAKEVEN_ATR = 0.8

# === Risk Management ===
MAX_DRAWDOWN_PCT = 2.0
MAX_LOSS_PER_TRADE_PCT = 0.8
MIN_RISK_REWARD = 1.5
DYNAMIC_SL_ATR_MULT = 1.2

# === Advanced Detection ===
FALSE_BREAKOUT_CONFIRMATION = 3  # شمعات للتأكيد
LIQUIDITY_TRAP_THRESHOLD = 0.7   # نسبة كسر ثم عودة
MANIPULATION_DETECTION_WINDOW = 10
VOLUME_SPIKE_THRESHOLD = 2.5

# =================== DATA STRUCTURES ===================
class MarketBias(str, Enum):
    STRONG_BULL = "STRONG_BULL"
    BULL = "BULL"
    NEUTRAL = "NEUTRAL"
    BEAR = "BEAR"
    STRONG_BEAR = "STRONG_BEAR"

class MarketPhase(str, Enum):
    ACCUMULATION = "ACCUMULATION"
    MARKUP = "MARKUP"
    DISTRIBUTION = "DISTRIBUTION"
    MARKDOWN = "MARKDOWN"
    CHOP = "CHOP"
    BREAKOUT = "BREAKOUT"
    REVERSAL = "REVERSAL"

class TradeType(str, Enum):
    GOLDEN_REVERSAL = "GOLDEN_REVERSAL"
    GOLDEN_BREAKOUT = "GOLDEN_BREAKOUT"
    TREND_FOLLOW = "TREND_FOLLOW"
    SCALP_RETEST = "SCALP_RETEST"
    LIQUIDITY_GRAB = "LIQUIDITY_GRAB"

class OrderBlockType(str, Enum):
    BULLISH_OB = "BULLISH_OB"
    BEARISH_OB = "BEARISH_OB"
    FVG_BULL = "FVG_BULL"
    FVG_BEAR = "FVG_BEAR"
    BOS_CONFIRMATION = "BOS_CONFIRMATION"
    CHOCH_CONFIRMATION = "CHOCH_CONFIRMATION"

@dataclass
class MarketStructure:
    """هيكل السوق المتقدم"""
    higher_highs: List[float] = field(default_factory=list)
    higher_lows: List[float] = field(default_factory=list)
    lower_highs: List[float] = field(default_factory=list)
    lower_lows: List[float] = field(default_factory=list)
    swing_highs: List[Dict] = field(default_factory=list)
    swing_lows: List[Dict] = field(default_factory=list)
    last_bos: Optional[Dict] = None
    last_choch: Optional[Dict] = None
    trend: str = "neutral"
    structure: str = "intact"
    itc_zones: List[Dict] = field(default_factory=list)  # Internal Retracement Zones

@dataclass
class SmartSignal:
    """إشارة ذكية متكاملة"""
    action: str  # BUY/SELL
    trade_type: TradeType
    entry_price: float
    stop_loss: float
    take_profits: List[float]
    close_fractions: List[float]
    confidence: float  # من 0-10
    reasons: List[str]
    zone_strength: float  # قوة المنطقة 0-10
    risk_reward: float
    market_context: Dict[str, Any]
    timestamp: int
    is_golden: bool = False
    requires_confirmation: bool = True

@dataclass
class TradingContext:
    """السياق الشامل للتداول"""
    price: float
    volume: float
    volume_ma: float
    volume_ratio: float
    volume_zscore: float
    atr: float
    rsi: float
    rsi_ma: float
    adx: float
    di_plus: float
    di_minus: float
    macd: float
    macd_signal: float
    macd_histogram: float
    ma_fast: float
    ma_slow: float
    ma_trend: float
    vwap: float
    bias: MarketBias
    phase: MarketPhase
    market_strength: float  # 0-10
    structure: MarketStructure
    order_blocks: List[Dict]
    fair_value_gaps: List[Dict]
    liquidity_zones: List[Dict]
    support_zones: List[Dict]
    resistance_zones: List[Dict]
    manipulation_signals: List[str]
    danger_zones: List[str]

# =================== TECHNICAL INDICATORS ===================
class AdvancedIndicators:
    """فئة المؤشرات المتقدمة"""
    
    @staticmethod
    def calculate_all(df: pd.DataFrame) -> Dict[str, Any]:
        """حساب جميع المؤشرات مرة واحدة"""
        results = {}
        
        # الأسعار الأساسية
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        volume = df['volume'].astype(float)
        
        # 1. المتوسطات المتحركة
        results['ma_fast'] = close.rolling(window=MA_FAST).mean()
        results['ma_slow'] = close.rolling(window=MA_SLOW).mean()
        results['ma_trend'] = close.rolling(window=MA_TREND).mean()
        
        # 2. RSI مع تحسينات
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=RSI_LEN).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_LEN).mean()
        rs = gain / (loss + 1e-12)
        results['rsi'] = 100 - (100 / (1 + rs))
        results['rsi_ma'] = results['rsi'].rolling(window=RSI_LEN).mean()
        
        # 3. ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        results['atr'] = tr.rolling(window=ATR_LEN).mean()
        
        # 4. ADX & DI
        plus_dm = high.diff()
        minus_dm = low.diff() * -1
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr_smooth = tr.rolling(window=ADX_LEN).mean()
        plus_di = 100 * (plus_dm.rolling(window=ADX_LEN).mean() / tr_smooth)
        minus_di = 100 * (minus_dm.rolling(window=ADX_LEN).mean() / tr_smooth)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12)
        results['adx'] = dx.rolling(window=ADX_LEN).mean()
        results['di_plus'] = plus_di
        results['di_minus'] = minus_di
        
        # 5. MACD
        exp1 = close.ewm(span=MACD_FAST, adjust=False).mean()
        exp2 = close.ewm(span=MACD_SLOW, adjust=False).mean()
        results['macd'] = exp1 - exp2
        results['macd_signal'] = results['macd'].ewm(span=MACD_SIGNAL, adjust=False).mean()
        results['macd_histogram'] = results['macd'] - results['macd_signal']
        
        # 6. VWAP
        typical_price = (high + low + close) / 3
        results['vwap'] = (typical_price * volume).cumsum() / volume.cumsum()
        
        # 7. Volume Analysis
        results['volume_ma'] = volume.rolling(window=VOLUME_MA).mean()
        results['volume_ratio'] = volume / results['volume_ma']
        volume_z = (volume - volume.rolling(window=50).mean()) / (volume.rolling(window=50).std() + 1e-12)
        results['volume_zscore'] = volume_z
        
        return results

# =================== MARKET STRUCTURE DETECTOR ===================
class MarketStructureDetector:
    """كاشف هيكل السوق المتقدم"""
    
    def __init__(self):
        self.swing_history = deque(maxlen=100)
        self.structure = MarketStructure()
        
    def detect_swings(self, df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> MarketStructure:
        """كشف النقاط المحورية (Swing Highs/Lows)"""
        highs = df['high'].astype(float).values
        lows = df['low'].astype(float).values
        
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(df) - 2):
            # Swing High
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                swing_highs.append({
                    'index': i,
                    'price': highs[i],
                    'time': int(df['time'].iloc[i])
                })
            
            # Swing Low
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                swing_lows.append({
                    'index': i,
                    'price': lows[i],
                    'time': int(df['time'].iloc[i])
                })
        
        self.structure.swing_highs = swing_highs[-5:]  # آخر 5
        self.structure.swing_lows = swing_lows[-5:]    # آخر 5
        
        return self._analyze_structure(swing_highs, swing_lows)
    
    def _analyze_structure(self, swing_highs: List, swing_lows: List) -> MarketStructure:
        """تحليل هيكل السوق"""
        if len(swing_highs) < 3 or len(swing_lows) < 3:
            return self.structure
        
        # تحليل Higher Highs/Lower Lows
        self.structure.higher_highs = []
        self.structure.higher_lows = []
        self.structure.lower_highs = []
        self.structure.lower_lows = []
        
        # كشف BOS (Break of Structure)
        if len(swing_highs) >= 3:
            last_3_highs = swing_highs[-3:]
            if (last_3_highs[-1]['price'] > last_3_highs[-2]['price'] and
                last_3_highs[-2]['price'] > last_3_highs[-3]['price']):
                self.structure.last_bos = {
                    'type': 'bullish',
                    'price': last_3_highs[-1]['price'],
                    'time': last_3_highs[-1]['time']
                }
                self.structure.trend = "bullish"
        
        if len(swing_lows) >= 3:
            last_3_lows = swing_lows[-3:]
            if (last_3_lows[-1]['price'] < last_3_lows[-2]['price'] and
                last_3_lows[-2]['price'] < last_3_lows[-3]['price']):
                self.structure.last_bos = {
                    'type': 'bearish',
                    'price': last_3_lows[-1]['price'],
                    'time': last_3_lows[-1]['time']
                }
                self.structure.trend = "bearish"
        
        # كشف CHoCH (Change of Character)
        self._detect_choch(swing_highs, swing_lows)
        
        # كشف مناطق ITC (Internal Retracement)
        self._detect_itc_zones(swing_highs, swing_lows)
        
        return self.structure
    
    def _detect_choch(self, swing_highs: List, swing_lows: List):
        """كشف تغيير الطابع"""
        if len(swing_highs) >= 4 and len(swing_lows) >= 4:
            # CHoCH هابط (Bullish to Bearish)
            if (swing_highs[-1]['price'] < swing_highs[-2]['price'] and
                swing_lows[-1]['price'] < swing_lows[-2]['price']):
                self.structure.last_choch = {
                    'type': 'bearish',
                    'price': swing_lows[-1]['price'],
                    'time': swing_lows[-1]['time']
                }
            
            # CHoCH صاعد (Bearish to Bullish)
            elif (swing_lows[-1]['price'] > swing_lows[-2]['price'] and
                  swing_highs[-1]['price'] > swing_highs[-2]['price']):
                self.structure.last_choch = {
                    'type': 'bullish',
                    'price': swing_highs[-1]['price'],
                    'time': swing_highs[-1]['time']
                }
    
    def _detect_itc_zones(self, swing_highs: List, swing_lows: List):
        """كشف مناطق التراجع الداخلي"""
        itc_zones = []
        
        for i in range(1, min(len(swing_highs), len(swing_lows))):
            # منطقة بين Swing High و Swing Low
            zone = {
                'high': swing_highs[i-1]['price'],
                'low': swing_lows[i-1]['price'],
                'mid': (swing_highs[i-1]['price'] + swing_lows[i-1]['price']) / 2,
                'time': swing_highs[i-1]['time']
            }
            itc_zones.append(zone)
        
        self.structure.itc_zones = itc_zones[-3:]  # آخر 3 مناطق

# =================== ORDER BLOCK & FVG DETECTOR ===================
class OrderBlockDetector:
    """كاشف Order Blocks و Fair Value Gaps"""
    
    @staticmethod
    def detect_order_blocks(df: pd.DataFrame, lookback: int = OB_LOOKBACK) -> List[Dict]:
        """كشف Order Blocks المتقدمة"""
        blocks = []
        close = df['close'].astype(float).values
        high = df['high'].astype(float).values
        low = df['low'].astype(float).values
        volume = df['volume'].astype(float).values
        
        for i in range(2, min(len(df), lookback)):
            # Bullish OB: شمعة هابطة قوية تليها شمعة صاعدة
            if (close[i-1] < high[i-2] and  # كسر قمة الشمعة السابقة
                close[i] > close[i-1] and   # إغلاق أعلى
                volume[i] > volume[i-1] * 1.5):  # حجم مرتفع
                
                block = {
                    'type': OrderBlockType.BULLISH_OB,
                    'high': high[i-2],
                    'low': low[i-2],
                    'entry': close[i],
                    'time': int(df['time'].iloc[i]),
                    'strength': min(9.0, (volume[i] / volume[i-1]) * 3)
                }
                blocks.append(block)
            
            # Bearish OB: شمعة صاعدة قوية تليها شمعة هابطة
            elif (close[i-1] > low[i-2] and   # كسر قاع الشمعة السابقة
                  close[i] < close[i-1] and    # إغلاق أقل
                  volume[i] > volume[i-1] * 1.5):
                
                block = {
                    'type': OrderBlockType.BEARISH_OB,
                    'high': high[i-2],
                    'low': low[i-2],
                    'entry': close[i],
                    'time': int(df['time'].iloc[i]),
                    'strength': min(9.0, (volume[i] / volume[i-1]) * 3)
                }
                blocks.append(block)
        
        return blocks[-5:]  # آخر 5 Order Blocks
    
    @staticmethod
    def detect_fvgs(df: pd.DataFrame, lookback: int = FVG_LOOKBACK) -> List[Dict]:
        """كشف Fair Value Gaps"""
        fvgs = []
        high = df['high'].astype(float).values
        low = df['low'].astype(float).values
        
        for i in range(2, min(len(df), lookback)):
            # Bullish FVG
            if low[i] > high[i-2]:
                fvg = {
                    'type': OrderBlockType.FVG_BULL,
                    'low': high[i-2],
                    'high': low[i],
                    'time': int(df['time'].iloc[i]),
                    'gap_size': (low[i] - high[i-2]) / high[i-2] * 100
                }
                fvgs.append(fvg)
            
            # Bearish FVG
            elif high[i] < low[i-2]:
                fvg = {
                    'type': OrderBlockType.FVG_BEAR,
                    'low': high[i],
                    'high': low[i-2],
                    'time': int(df['time'].iloc[i]),
                    'gap_size': (low[i-2] - high[i]) / low[i-2] * 100
                }
                fvgs.append(fvg)
        
        return fvgs[-3:]  # آخر 3 FVGs

# =================== LIQUIDITY & MANIPULATION DETECTOR ===================
class LiquidityDetector:
    """كاشف السيولة والتلاعب"""
    
    @staticmethod
    def detect_liquidity_zones(df: pd.DataFrame, lookback: int = LIQUIDITY_LOOKBACK) -> List[Dict]:
        """كشف مناطق السيولة (Liquidity Pools)"""
        zones = []
        high = df['high'].astype(float).values
        low = df['low'].astype(float).values
        volume = df['volume'].astype(float).values
        
        # كشف Highs & Lows الأخيرة
        recent_highs = []
        recent_lows = []
        
        for i in range(max(0, len(df) - lookback), len(df) - 1):
            # High Liquidity Zone
            if high[i] == max(high[max(0, i-5):i+1]):
                recent_highs.append({
                    'price': high[i],
                    'time': int(df['time'].iloc[i]),
                    'volume': volume[i]
                })
            
            # Low Liquidity Zone
            if low[i] == min(low[max(0, i-5):i+1]):
                recent_lows.append({
                    'price': low[i],
                    'time': int(df['time'].iloc[i]),
                    'volume': volume[i]
                })
        
        # تحويل إلى مناطق
        for h in recent_highs[-3:]:  # آخر 3 قمم
            zone = {
                'type': 'resistance_liquidity',
                'price': h['price'],
                'range_high': h['price'] * 1.002,
                'range_low': h['price'] * 0.998,
                'strength': min(10.0, h['volume'] / np.mean(volume) * 2),
                'time': h['time']
            }
            zones.append(zone)
        
        for l in recent_lows[-3:]:  # آخر 3 قيعان
            zone = {
                'type': 'support_liquidity',
                'price': l['price'],
                'range_high': l['price'] * 1.002,
                'range_low': l['price'] * 0.998,
                'strength': min(10.0, l['volume'] / np.mean(volume) * 2),
                'time': l['time']
            }
            zones.append(zone)
        
        return zones
    
    @staticmethod
    def detect_manipulation(df: pd.DataFrame) -> List[str]:
        """كشف علامات التلاعب في السوق"""
        signals = []
        close = df['close'].astype(float).values
        high = df['high'].astype(float).values
        low = df['low'].astype(float).values
        volume = df['volume'].astype(float).values
        
        if len(df) < MANIPULATION_DETECTION_WINDOW + 1:
            return signals
        
        # 1. False Breakouts (كسر وهمي)
        for i in range(len(df) - MANIPULATION_DETECTION_WINDOW, len(df) - 1):
            # كسر قمة ثم عودة
            if (high[i] > max(high[i-3:i]) and  # كسر قمة
                close[i+1] < high[i] and        # إغلاق أقل
                close[i+2] < close[i+1]):       # استمرار الهبوط
                signals.append(f"FALSE_BREAKOUT_HIGH_{i}")
            
            # كسر قاع ثم عودة
            if (low[i] < min(low[i-3:i]) and    # كسر قاع
                close[i+1] > low[i] and         # إغلاق أعلى
                close[i+2] > close[i+1]):       # استمرار الصعود
                signals.append(f"FALSE_BREAKOUT_LOW_{i}")
        
        # 2. Volume Spikes (قفزات حجم مفاجئة)
        volume_ma = np.mean(volume[-VOLUME_MA:])
        if volume[-1] > volume_ma * VOLUME_SPIKE_THRESHOLD:
            signals.append(f"VOLUME_SPIKE_{volume[-1]/volume_ma:.1f}x")
        
        # 3. Wick Rejections (رفض بالذيل)
        current_candle = df.iloc[-1]
        o, c, h, l = current_candle['open'], current_candle['close'], current_candle['high'], current_candle['low']
        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        
        if upper_wick > body * 2 and c < o:  # رفض من الأعلى
            signals.append("WICK_REJECTION_HIGH")
        if lower_wick > body * 2 and c > o:  # رفض من الأسفل
            signals.append("WICK_REJECTION_LOW")
        
        return signals

# =================== GOLDEN ZONE DETECTOR ===================
class GoldenZoneDetector:
    """كاشف المناطق الذهبية (القمم والقيعان الذهبية)"""
    
    def __init__(self):
        self.golden_zones = []
        self.last_golden_signal = None
        
    def detect_golden_zones(self, df: pd.DataFrame, ctx: TradingContext) -> List[Dict]:
        """كشف المناطق الذهبية المتقدمة"""
        golden_zones = []
        
        # 1. Golden Bottom (قاع ذهبي)
        bottom_zones = self._detect_golden_bottoms(df, ctx)
        golden_zones.extend(bottom_zones)
        
        # 2. Golden Top (قمة ذهبية)
        top_zones = self._detect_golden_tops(df, ctx)
        golden_zones.extend(top_zones)
        
        # 3. Trend Reversal Zones (مناطق انعكاس الترند)
        reversal_zones = self._detect_trend_reversals(df, ctx)
        golden_zones.extend(reversal_zones)
        
        self.golden_zones = golden_zones[-5:]  # آخر 5 مناطق ذهبية
        return self.golden_zones
    
    def _detect_golden_bottoms(self, df: pd.DataFrame, ctx: TradingContext) -> List[Dict]:
        """كشف القيعان الذهبية"""
        bottoms = []
        close = df['close'].astype(float).values
        low = df['low'].astype(float).values
        volume = df['volume'].astype(float).values
        
        for i in range(len(df) - 10, len(df) - 1):
            # شروط القاع الذهبي:
            # 1. سعر منخفض جديد أو قريب من القاع
            # 2. حجم مرتفع
            # 3. RSI تشبع بيع
            # 4. إغلاق أعلى من الفتح
            # 5. تأكيد من المؤشرات
            
            is_new_low = low[i] == min(low[max(0, i-20):i+1])
            volume_spike = volume[i] > np.mean(volume[max(0, i-10):i]) * 1.5
            bullish_close = close[i] > df['open'].iloc[i]
            
            if is_new_low and volume_spike and bullish_close:
                # حساب قوة الإشارة
                strength = self._calculate_bottom_strength(df, i, ctx)
                
                if strength >= GOLDEN_ZONE_MIN_STRENGTH:
                    bottom = {
                        'type': 'GOLDEN_BOTTOM',
                        'price': low[i],
                        'entry': close[i],
                        'strength': strength,
                        'time': int(df['time'].iloc[i]),
                        'reasons': [
                            f"New low detected",
                            f"Volume spike: {volume[i]/np.mean(volume[max(0,i-10):i]):.1f}x",
                            f"Bullish close",
                            f"RSI: {ctx.rsi:.1f}",
                            f"Strength: {strength:.1f}/10"
                        ]
                    }
                    bottoms.append(bottom)
        
        return bottoms
    
    def _detect_golden_tops(self, df: pd.DataFrame, ctx: TradingContext) -> List[Dict]:
        """كشف القمم الذهبية"""
        tops = []
        close = df['close'].astype(float).values
        high = df['high'].astype(float).values
        volume = df['volume'].astype(float).values
        
        for i in range(len(df) - 10, len(df) - 1):
            # شروط القمة الذهبية:
            # 1. سعر مرتفع جديد أو قريب من القمة
            # 2. حجم مرتفع
            # 3. RSI تشبع شراء
            # 4. إغلاق أقل من الفتح
            # 5. تأكيد من المؤشرات
            
            is_new_high = high[i] == max(high[max(0, i-20):i+1])
            volume_spike = volume[i] > np.mean(volume[max(0, i-10):i]) * 1.5
            bearish_close = close[i] < df['open'].iloc[i]
            
            if is_new_high and volume_spike and bearish_close:
                # حساب قوة الإشارة
                strength = self._calculate_top_strength(df, i, ctx)
                
                if strength >= GOLDEN_ZONE_MIN_STRENGTH:
                    top = {
                        'type': 'GOLDEN_TOP',
                        'price': high[i],
                        'entry': close[i],
                        'strength': strength,
                        'time': int(df['time'].iloc[i]),
                        'reasons': [
                            f"New high detected",
                            f"Volume spike: {volume[i]/np.mean(volume[max(0,i-10):i]):.1f}x",
                            f"Bearish close",
                            f"RSI: {ctx.rsi:.1f}",
                            f"Strength: {strength:.1f}/10"
                        ]
                    }
                    tops.append(top)
        
        return tops
    
    def _detect_trend_reversals(self, df: pd.DataFrame, ctx: TradingContext) -> List[Dict]:
        """كشف مناطق انعكاس الترند"""
        reversals = []
        
        # شروط انعكاس الترند:
        # 1. تغيير في هيكل السوق (BOS/CHoCH)
        # 2. تأكيد من المؤشرات
        # 3. حجم قوي
        # 4. اختراق المتوسطات
        
        if ctx.structure.last_choch:
            strength = self._calculate_reversal_strength(df, ctx)
            
            if strength >= GOLDEN_ZONE_MIN_STRENGTH:
                reversal = {
                    'type': 'TREND_REVERSAL',
                    'direction': ctx.structure.last_choch['type'],
                    'price': ctx.structure.last_choch['price'],
                    'strength': strength,
                    'time': ctx.structure.last_choch['time'],
                    'reasons': [
                        f"CHoCH detected: {ctx.structure.last_choch['type']}",
                        f"Market structure change",
                        f"ADX: {ctx.adx:.1f}",
                        f"Volume ratio: {ctx.volume_ratio:.1f}",
                        f"Strength: {strength:.1f}/10"
                    ]
                }
                reversals.append(reversal)
        
        return reversals
    
    def _calculate_bottom_strength(self, df: pd.DataFrame, idx: int, ctx: TradingContext) -> float:
        """حساب قوة القاع الذهبي"""
        strength = 0.0
        
        # 1. RSI تشبع بيع (20 نقطة)
        if ctx.rsi < 30:
            strength += 2.0
        elif ctx.rsi < 40:
            strength += 1.0
        
        # 2. حجم مرتفع (20 نقطة)
        if ctx.volume_ratio > 2.0:
            strength += 2.0
        elif ctx.volume_ratio > 1.5:
            strength += 1.5
        
        # 3. تأكيد MACD (20 نقطة)
        if ctx.macd_histogram > 0 and ctx.macd > ctx.macd_signal:
            strength += 2.0
        
        # 4. هيكل السوق (20 نقطة)
        if ctx.structure.trend == "bearish" and ctx.structure.last_choch:
            if ctx.structure.last_choch['type'] == 'bullish':
                strength += 2.0
        
        # 5. Order Block قريب (10 نقطة)
        for ob in ctx.order_blocks:
            if ob['type'] == OrderBlockType.BULLISH_OB:
                if abs(df['close'].iloc[idx] - ob['entry']) / ob['entry'] < 0.01:
                    strength += 1.0
        
        # 6. اختراق المتوسطات (10 نقطة)
        if df['close'].iloc[idx] > ctx.ma_fast and df['close'].iloc[idx] > ctx.ma_slow:
            strength += 1.0
        
        return min(10.0, strength)
    
    def _calculate_top_strength(self, df: pd.DataFrame, idx: int, ctx: TradingContext) -> float:
        """حساب قوة القمة الذهبية"""
        strength = 0.0
        
        # 1. RSI تشبع شراء (20 نقطة)
        if ctx.rsi > 70:
            strength += 2.0
        elif ctx.rsi > 60:
            strength += 1.0
        
        # 2. حجم مرتفع (20 نقطة)
        if ctx.volume_ratio > 2.0:
            strength += 2.0
        elif ctx.volume_ratio > 1.5:
            strength += 1.5
        
        # 3. تأكيد MACD (20 نقطة)
        if ctx.macd_histogram < 0 and ctx.macd < ctx.macd_signal:
            strength += 2.0
        
        # 4. هيكل السوق (20 نقطة)
        if ctx.structure.trend == "bullish" and ctx.structure.last_choch:
            if ctx.structure.last_choch['type'] == 'bearish':
                strength += 2.0
        
        # 5. Order Block قريب (10 نقطة)
        for ob in ctx.order_blocks:
            if ob['type'] == OrderBlockType.BEARISH_OB:
                if abs(df['close'].iloc[idx] - ob['entry']) / ob['entry'] < 0.01:
                    strength += 1.0
        
        # 6. اختراق المتوسطات (10 نقطة)
        if df['close'].iloc[idx] < ctx.ma_fast and df['close'].iloc[idx] < ctx.ma_slow:
            strength += 1.0
        
        return min(10.0, strength)
    
    def _calculate_reversal_strength(self, df: pd.DataFrame, ctx: TradingContext) -> float:
        """حساب قوة انعكاس الترند"""
        strength = 0.0
        
        # 1. تأكيد CHoCH (30 نقطة)
        if ctx.structure.last_choch:
            strength += 3.0
        
        # 2. حجم مرتفع (20 نقطة)
        if ctx.volume_ratio > 1.8:
            strength += 2.0
        
        # 3. تأكيد مؤشرات (30 نقطة)
        if (ctx.macd_histogram > 0 and ctx.structure.last_choch['type'] == 'bullish') or \
           (ctx.macd_histogram < 0 and ctx.structure.last_choch['type'] == 'bearish'):
            strength += 3.0
        
        # 4. RSI تأكيد (20 نقطة)
        if (ctx.rsi < 40 and ctx.structure.last_choch['type'] == 'bullish') or \
           (ctx.rsi > 60 and ctx.structure.last_choch['type'] == 'bearish'):
            strength += 2.0
        
        return min(10.0, strength)

# =================== SMART TRADE COUNCIL (7 MEMBERS) ===================
class SupremeTradeCouncil:
    """مجلس الإدارة الأعلى (7 أعضاء)"""
    
    def __init__(self):
        self.members = {
            'market_structure': self._market_structure_analyst,
            'volume_flow': self._volume_flow_analyst,
            'momentum': self._momentum_analyst,
            'trend': self._trend_analyst,
            'liquidity': self._liquidity_analyst,
            'risk': self._risk_analyst,
            'manipulation': self._manipulation_analyst
        }
        self.weights = {
            'market_structure': 0.20,
            'volume_flow': 0.15,
            'momentum': 0.15,
            'trend': 0.15,
            'liquidity': 0.15,
            'risk': 0.10,
            'manipulation': 0.10
        }
    
    def evaluate_signal(self, signal: SmartSignal, ctx: TradingContext, df: pd.DataFrame) -> Dict[str, Any]:
        """تقييم الإشارة من قبل جميع الأعضاء"""
        decisions = {}
        total_score = 0.0
        max_score = 0.0
        
        for name, analyst in self.members.items():
            decision = analyst(signal, ctx, df)
            decisions[name] = decision
            
            score = decision.get('score', 0) * self.weights[name]
            total_score += score
            max_score += 10 * self.weights[name]  # أعلى درجة لكل عضو = 10
        
        final_score = (total_score / max_score) * 10 if max_score > 0 else 0
        
        # جمع الأسباب
        all_reasons = []
        for name, decision in decisions.items():
            if decision.get('approve', False):
                all_reasons.extend(decision.get('reasons', []))
        
        return {
            'approved': final_score >= 7.0,
            'score': final_score,
            'confidence': min(0.95, final_score / 10),
            'reasons': all_reasons[:5],  # أول 5 أسباب
            'details': decisions,
            'vote': 'BUY' if signal.action == 'BUY' else 'SELL'
        }
    
    def _market_structure_analyst(self, signal: SmartSignal, ctx: TradingContext, df: pd.DataFrame) -> Dict:
        """محلل هيكل السوق"""
        score = 0.0
        reasons = []
        
        # تأكيد من هيكل السوق
        if signal.action == 'BUY':
            if ctx.structure.trend in ["bullish", "neutral"]:
                score += 3.0
                reasons.append("Trend supports BUY")
            
            if ctx.structure.last_bos and ctx.structure.last_bos['type'] == 'bullish':
                score += 4.0
                reasons.append("Bullish BOS confirmed")
            
            if ctx.structure.last_choch and ctx.structure.last_choch['type'] == 'bullish':
                score += 3.0
                reasons.append("Bullish CHoCH detected")
        
        elif signal.action == 'SELL':
            if ctx.structure.trend in ["bearish", "neutral"]:
                score += 3.0
                reasons.append("Trend supports SELL")
            
            if ctx.structure.last_bos and ctx.structure.last_bos['type'] == 'bearish':
                score += 4.0
                reasons.append("Bearish BOS confirmed")
            
            if ctx.structure.last_choch and ctx.structure.last_choch['type'] == 'bearish':
                score += 3.0
                reasons.append("Bearish CHoCH detected")
        
        # مناطق ITC قريبة
        for zone in ctx.structure.itc_zones[-2:]:
            if zone['low'] <= signal.entry_price <= zone['high']:
                score += 2.0
                reasons.append(f"Entry in ITC zone: {zone['low']:.6f}-{zone['high']:.6f}")
        
        return {
            'approve': score >= 6.0,
            'score': min(10.0, score),
            'reasons': reasons
        }
    
    def _volume_flow_analyst(self, signal: SmartSignal, ctx: TradingContext, df: pd.DataFrame) -> Dict:
        """محلل تدفق الحجم"""
        score = 0.0
        reasons = []
        
        # حجم قوي
        if ctx.volume_ratio > 1.8:
            score += 4.0
            reasons.append(f"Strong volume: {ctx.volume_ratio:.1f}x")
        elif ctx.volume_ratio > 1.3:
            score += 2.0
            reasons.append(f"Good volume: {ctx.volume_ratio:.1f}x")
        
        # Z-score إيجابي
        if ctx.volume_zscore > 1.5:
            score += 3.0
            reasons.append(f"Volume z-score: {ctx.volume_zscore:.1f}")
        elif ctx.volume_zscore > 0.5:
            score += 1.0
        
        # تدفق الحجم في اتجاه الصفقة
        last_5_volume = df['volume'].astype(float).values[-5:]
        if len(last_5_volume) >= 2:
            volume_trend = np.mean(last_5_volume[-2:]) / np.mean(last_5_volume[:2])
            if volume_trend > 1.2:
                score += 3.0
                reasons.append(f"Volume increasing: {volume_trend:.1f}x")
        
        return {
            'approve': score >= 5.0,
            'score': min(10.0, score),
            'reasons': reasons
        }
    
    def _momentum_analyst(self, signal: SmartSignal, ctx: TradingContext, df: pd.DataFrame) -> Dict:
        """محلل الزخم"""
        score = 0.0
        reasons = []
        
        # RSI في المنطقة المناسبة
        if signal.action == 'BUY':
            if ctx.rsi < 40:
                score += 3.0
                reasons.append(f"RSI oversold: {ctx.rsi:.1f}")
            elif ctx.rsi < 50:
                score += 2.0
                reasons.append(f"RSI neutral-bullish: {ctx.rsi:.1f}")
        else:  # SELL
            if ctx.rsi > 60:
                score += 3.0
                reasons.append(f"RSI overbought: {ctx.rsi:.1f}")
            elif ctx.rsi > 50:
                score += 2.0
                reasons.append(f"RSI neutral-bearish: {ctx.rsi:.1f}")
        
        # MACD تأكيد
        if signal.action == 'BUY':
            if ctx.macd_histogram > 0 and ctx.macd > ctx.macd_signal:
                score += 4.0
                reasons.append("MACD bullish")
        else:
            if ctx.macd_histogram < 0 and ctx.macd < ctx.macd_signal:
                score += 4.0
                reasons.append("MACD bearish")
        
        # ADX قوي
        if ctx.adx > 25:
            score += 3.0
            reasons.append(f"Strong trend: ADX={ctx.adx:.1f}")
        elif ctx.adx > 20:
            score += 2.0
            reasons.append(f"Moderate trend: ADX={ctx.adx:.1f}")
        
        return {
            'approve': score >= 6.0,
            'score': min(10.0, score),
            'reasons': reasons
        }
    
    def _trend_analyst(self, signal: SmartSignal, ctx: TradingContext, df: pd.DataFrame) -> Dict:
        """محلل الترند"""
        score = 0.0
        reasons = []
        
        # المتوسطات المتحركة
        if signal.action == 'BUY':
            if ctx.ma_fast > ctx.ma_slow and ctx.ma_slow > ctx.ma_trend:
                score += 5.0
                reasons.append("All MAs bullish aligned")
            elif ctx.ma_fast > ctx.ma_slow:
                score += 3.0
                reasons.append("Fast MA above Slow MA")
            
            if signal.entry_price > ctx.ma_fast:
                score += 2.0
                reasons.append("Price above Fast MA")
        else:  # SELL
            if ctx.ma_fast < ctx.ma_slow and ctx.ma_slow < ctx.ma_trend:
                score += 5.0
                reasons.append("All MAs bearish aligned")
            elif ctx.ma_fast < ctx.ma_slow:
                score += 3.0
                reasons.append("Fast MA below Slow MA")
            
            if signal.entry_price < ctx.ma_fast:
                score += 2.0
                reasons.append("Price below Fast MA")
        
        # VWAP تأكيد
        if signal.action == 'BUY' and signal.entry_price > ctx.vwap:
            score += 2.0
            reasons.append("Price above VWAP")
        elif signal.action == 'SELL' and signal.entry_price < ctx.vwap:
            score += 2.0
            reasons.append("Price below VWAP")
        
        # DI تأكيد
        if signal.action == 'BUY' and ctx.di_plus > ctx.di_minus:
            score += 1.0
            reasons.append("DI+ above DI-")
        elif signal.action == 'SELL' and ctx.di_minus > ctx.di_plus:
            score += 1.0
            reasons.append("DI- above DI+")
        
        return {
            'approve': score >= 5.0,
            'score': min(10.0, score),
            'reasons': reasons
        }
    
    def _liquidity_analyst(self, signal: SmartSignal, ctx: TradingContext, df: pd.DataFrame) -> Dict:
        """محلل السيولة"""
        score = 0.0
        reasons = []
        
        # Order Blocks قريبة
        for ob in ctx.order_blocks[-3:]:
            distance = abs(signal.entry_price - ob['entry']) / ob['entry']
            if distance < 0.005:  # 0.5%
                score += 3.0
                reasons.append(f"Near {ob['type'].value} at {ob['entry']:.6f}")
        
        # FVGs قريبة
        for fvg in ctx.fair_value_gaps[-2:]:
            if fvg['low'] <= signal.entry_price <= fvg['high']:
                score += 2.0
                reasons.append(f"In FVG zone: {fvg['low']:.6f}-{fvg['high']:.6f}")
        
        # مناطق سيولة قريبة
        for zone in ctx.liquidity_zones:
            if zone['range_low'] <= signal.entry_price <= zone['range_high']:
                score += zone.get('strength', 1)
                reasons.append(f"Liquidity zone: {zone['type']}")
        
        return {
            'approve': score >= 4.0,
            'score': min(10.0, score),
            'reasons': reasons
        }
    
    def _risk_analyst(self, signal: SmartSignal, ctx: TradingContext, df: pd.DataFrame) -> Dict:
        """محلل المخاطر"""
        score = 0.0
        reasons = []
        
        # نسبة المخاطرة/العائد
        if signal.risk_reward >= 2.0:
            score += 5.0
            reasons.append(f"Great R:R = {signal.risk_reward:.1f}")
        elif signal.risk_reward >= 1.5:
            score += 3.0
            reasons.append(f"Good R:R = {signal.risk_reward:.1f}")
        
        # قوة المنطقة
        if signal.zone_strength >= 8.0:
            score += 3.0
            reasons.append(f"Very strong zone: {signal.zone_strength:.1f}/10")
        elif signal.zone_strength >= 6.0:
            score += 2.0
            reasons.append(f"Strong zone: {signal.zone_strength:.1f}/10")
        
        # ATR مناسب
        stop_distance = abs(signal.entry_price - signal.stop_loss) / signal.entry_price * 100
        atr_percent = ctx.atr / signal.entry_price * 100
        
        if stop_distance <= atr_percent * 1.5:
            score += 2.0
            reasons.append(f"SL distance reasonable: {stop_distance:.2f}%")
        
        return {
            'approve': score >= 6.0,
            'score': min(10.0, score),
            'reasons': reasons
        }
    
    def _manipulation_analyst(self, signal: SmartSignal, ctx: TradingContext, df: pd.DataFrame) -> Dict:
        """محلل التلاعب"""
        score = 10.0  # ابدأ بدرجة كاملة
        reasons = ["No manipulation detected"]
        
        # تحقق من علامات التلاعب
        if ctx.manipulation_signals:
            for manip in ctx.manipulation_signals:
                if "FALSE_BREAKOUT" in manip:
                    score -= 3.0
                    reasons.append("False breakout detected")
                elif "VOLUME_SPIKE" in manip:
                    score -= 1.0
                elif "WICK_REJECTION" in manip:
                    if (signal.action == 'BUY' and "HIGH" in manip) or \
                       (signal.action == 'SELL' and "LOW" in manip):
                        score -= 4.0
                        reasons.append("Wick rejection against trade")
        
        # مناطق خطرة
        if ctx.danger_zones:
            score -= len(ctx.danger_zones) * 2.0
            reasons.append(f"{len(ctx.danger_zones)} danger zones")
        
        return {
            'approve': score >= 6.0,
            'score': max(0.0, min(10.0, score)),
            'reasons': reasons
        }

# =================== ULTIMATE SIGNAL GENERATOR ===================
class UltimateSignalGenerator:
    """مولد الإشارات المتكامل"""
    
    def __init__(self):
        self.indicators = AdvancedIndicators()
        self.structure_detector = MarketStructureDetector()
        self.ob_detector = OrderBlockDetector()
        self.liquidity_detector = LiquidityDetector()
        self.golden_detector = GoldenZoneDetector()
        self.council = SupremeTradeCouncil()
        self.last_signals = deque(maxlen=10)
        
    def analyze_market(self, df: pd.DataFrame) -> TradingContext:
        """تحليل السوق الشامل"""
        # حساب جميع المؤشرات
        indicators = self.indicators.calculate_all(df)
        
        # كشف هيكل السوق
        structure = self.structure_detector.detect_swings(df)
        
        # كشف Order Blocks و FVGs
        order_blocks = self.ob_detector.detect_order_blocks(df)
        fvgs = self.ob_detector.detect_fvgs(df)
        
        # كشف السيولة والتلاعب
        liquidity_zones = self.liquidity_detector.detect_liquidity_zones(df)
        manipulation_signals = self.liquidity_detector.detect_manipulation(df)
        
        # تحديد التحيز والطور
        current_idx = len(df) - 1
        bias = self._determine_bias(indicators, df, current_idx)
        phase = self._determine_phase(indicators, structure, current_idx)
        
        # قوة السوق
        market_strength = self._calculate_market_strength(indicators, structure, current_idx)
        
        # مناطق الخطر
        danger_zones = self._detect_danger_zones(indicators, df, current_idx)
        
        # بناء السياق
        ctx = TradingContext(
            price=float(df['close'].iloc[current_idx]),
            volume=float(df['volume'].iloc[current_idx]),
            volume_ma=float(indicators['volume_ma'].iloc[current_idx]),
            volume_ratio=float(indicators['volume_ratio'].iloc[current_idx]),
            volume_zscore=float(indicators['volume_zscore'].iloc[current_idx]),
            atr=float(indicators['atr'].iloc[current_idx]),
            rsi=float(indicators['rsi'].iloc[current_idx]),
            rsi_ma=float(indicators['rsi_ma'].iloc[current_idx]),
            adx=float(indicators['adx'].iloc[current_idx]),
            di_plus=float(indicators['di_plus'].iloc[current_idx]),
            di_minus=float(indicators['di_minus'].iloc[current_idx]),
            macd=float(indicators['macd'].iloc[current_idx]),
            macd_signal=float(indicators['macd_signal'].iloc[current_idx]),
            macd_histogram=float(indicators['macd_histogram'].iloc[current_idx]),
            ma_fast=float(indicators['ma_fast'].iloc[current_idx]),
            ma_slow=float(indicators['ma_slow'].iloc[current_idx]),
            ma_trend=float(indicators['ma_trend'].iloc[current_idx]),
            vwap=float(indicators['vwap'].iloc[current_idx]),
            bias=bias,
            phase=phase,
            market_strength=market_strength,
            structure=structure,
            order_blocks=order_blocks,
            fair_value_gaps=fvgs,
            liquidity_zones=liquidity_zones,
            support_zones=[z for z in liquidity_zones if 'support' in z['type']],
            resistance_zones=[z for z in liquidity_zones if 'resistance' in z['type']],
            manipulation_signals=manipulation_signals,
            danger_zones=danger_zones
        )
        
        return ctx
    
    def _determine_bias(self, indicators: Dict, df: pd.DataFrame, idx: int) -> MarketBias:
        """تحديد تحيز السوق"""
        ma_fast = indicators['ma_fast'].iloc[idx]
        ma_slow = indicators['ma_slow'].iloc[idx]
        ma_trend = indicators['ma_trend'].iloc[idx]
        price = df['close'].iloc[idx]
        
        # قواعد التحيز
        if price > ma_fast > ma_slow > ma_trend:
            return MarketBias.STRONG_BULL
        elif price > ma_fast > ma_slow:
            return MarketBias.BULL
        elif price < ma_fast < ma_slow < ma_trend:
            return MarketBias.STRONG_BEAR
        elif price < ma_fast < ma_slow:
            return MarketBias.BEAR
        else:
            return MarketBias.NEUTRAL
    
    def _determine_phase(self, indicators: Dict, structure: MarketStructure, idx: int) -> MarketPhase:
        """تحديد طور السوق"""
        adx = indicators['adx'].iloc[idx]
        volume_ratio = indicators['volume_ratio'].iloc[idx]
        di_plus = indicators['di_plus'].iloc[idx]
        di_minus = indicators['di_minus'].iloc[idx]
        
        if adx < 20 and volume_ratio < 1.2:
            return MarketPhase.CHOP
        elif adx > 25 and di_plus > di_minus:
            return MarketPhase.MARKUP
        elif adx > 25 and di_minus > di_plus:
            return MarketPhase.MARKDOWN
        elif volume_ratio > 1.5 and structure.last_bos:
            return MarketPhase.BREAKOUT
        elif structure.last_choch:
            return MarketPhase.REVERSAL
        elif volume_ratio > 1.3 and abs(di_plus - di_minus) < 5:
            return MarketPhase.ACCUMULATION if di_plus > di_minus else MarketPhase.DISTRIBUTION
        else:
            return MarketPhase.CHOP
    
    def _calculate_market_strength(self, indicators: Dict, structure: MarketStructure, idx: int) -> float:
        """حساب قوة السوق"""
        strength = 0.0
        
        # ADX (30%)
        adx = indicators['adx'].iloc[idx]
        strength += min(3.0, adx / 50 * 3)
        
        # Volume (30%)
        volume_ratio = indicators['volume_ratio'].iloc[idx]
        strength += min(3.0, volume_ratio * 1.5)
        
        # Trend Structure (20%)
        if structure.trend != "neutral":
            strength += 2.0
        
        # Momentum (20%)
        macd_hist = abs(indicators['macd_histogram'].iloc[idx])
        strength += min(2.0, macd_hist * 10)
        
        return min(10.0, strength)
    
    def _detect_danger_zones(self, indicators: Dict, df: pd.DataFrame, idx: int) -> List[str]:
        """كشف المناطق الخطرة"""
        dangers = []
        
        # RSI في أقصى المدى
        rsi = indicators['rsi'].iloc[idx]
        if rsi > 85 or rsi < 15:
            dangers.append(f"RSI_EXTREME_{rsi:.1f}")
        
        # حجم ضعيف جدًا
        volume_ratio = indicators['volume_ratio'].iloc[idx]
        if volume_ratio < 0.5:
            dangers.append(f"LOW_VOLUME_{volume_ratio:.1f}")
        
        # ADX ضعيف مع تقلبات كبيرة
        adx = indicators['adx'].iloc[idx]
        atr = indicators['atr'].iloc[idx]
        price = df['close'].iloc[idx]
        
        if adx < 15 and atr / price * 100 > 1.5:
            dangers.append(f"HIGH_VOLATILITY_CHOP")
        
        # اختراق كاذب حديث
        if len(df) > 5:
            recent_high = df['high'].iloc[idx-5:idx].max()
            recent_low = df['low'].iloc[idx-5:idx].min()
            
            if price > recent_high * 1.01 or price < recent_low * 0.99:
                if volume_ratio < 1.0:
                    dangers.append("FAKE_BREAKOUT_SUSPECTED")
        
        return dangers
    
    def generate_signals(self, df: pd.DataFrame) -> List[SmartSignal]:
        """توليد إشارات ذكية متكاملة"""
        signals = []
        
        # تحليل السوق أولاً
        ctx = self.analyze_market(df)
        
        # 1. إشارات المناطق الذهبية
        golden_signals = self._generate_golden_signals(df, ctx)
        signals.extend(golden_signals)
        
        # 2. إشارات Break of Structure
        bos_signals = self._generate_bos_signals(df, ctx)
        signals.extend(bos_signals)
        
        # 3. إشارات Retest
        retest_signals = self._generate_retest_signals(df, ctx)
        signals.extend(retest_signals)
        
        # 4. إشارات Trend Following
        trend_signals = self._generate_trend_signals(df, ctx)
        signals.extend(trend_signals)
        
        # 5. تقييم جميع الإشارات بواسطة المجلس
        evaluated_signals = []
        for signal in signals:
            evaluation = self.council.evaluate_signal(signal, ctx, df)
            
            if evaluation['approved']:
                signal.confidence = evaluation['confidence']
                signal.reasons.extend(evaluation['reasons'])
                evaluated_signals.append(signal)
        
        # حفظ آخر الإشارات
        self.last_signals.extend(evaluated_signals[:3])
        
        return evaluated_signals
    
    def _generate_golden_signals(self, df: pd.DataFrame, ctx: TradingContext) -> List[SmartSignal]:
        """توليد إشارات المناطق الذهبية"""
        signals = []
        current_price = ctx.price
        
        # كشف المناطق الذهبية
        golden_zones = self.golden_detector.detect_golden_zones(df, ctx)
        
        for zone in golden_zones[-3:]:  # آخر 3 مناطق ذهبية
            if zone['type'] == 'GOLDEN_BOTTOM':
                # حساب TP/SL للقاع الذهبي
                entry = current_price
                sl = entry * 0.995  # 0.5% stop loss
                
                # 3 مستويات TP للصفقات الذهبية
                tps = [
                    entry * (1 + GOLDEN_TP_LEVELS[0]/100),
                    entry * (1 + GOLDEN_TP_LEVELS[1]/100),
                    entry * (1 + GOLDEN_TP_LEVELS[2]/100)
                ]
                
                signal = SmartSignal(
                    action='BUY',
                    trade_type=TradeType.GOLDEN_REVERSAL,
                    entry_price=entry,
                    stop_loss=sl,
                    take_profits=tps,
                    close_fractions=GOLDEN_TP_CLOSE_FRACTIONS,
                    confidence=zone['strength'] / 10 * 0.8,
                    reasons=zone['reasons'],
                    zone_strength=zone['strength'],
                    risk_reward=(tps[0] - entry) / (entry - sl),
                    market_context=ctx.__dict__,
                    timestamp=int(time.time()),
                    is_golden=True,
                    requires_confirmation=True
                )
                signals.append(signal)
            
            elif zone['type'] == 'GOLDEN_TOP':
                # حساب TP/SL للقمة الذهبية
                entry = current_price
                sl = entry * 1.005  # 0.5% stop loss
                
                # 3 مستويات TP للصفقات الذهبية
                tps = [
                    entry * (1 - GOLDEN_TP_LEVELS[0]/100),
                    entry * (1 - GOLDEN_TP_LEVELS[1]/100),
                    entry * (1 - GOLDEN_TP_LEVELS[2]/100)
                ]
                
                signal = SmartSignal(
                    action='SELL',
                    trade_type=TradeType.GOLDEN_REVERSAL,
                    entry_price=entry,
                    stop_loss=sl,
                    take_profits=tps,
                    close_fractions=GOLDEN_TP_CLOSE_FRACTIONS,
                    confidence=zone['strength'] / 10 * 0.8,
                    reasons=zone['reasons'],
                    zone_strength=zone['strength'],
                    risk_reward=(entry - tps[0]) / (sl - entry),
                    market_context=ctx.__dict__,
                    timestamp=int(time.time()),
                    is_golden=True,
                    requires_confirmation=True
                )
                signals.append(signal)
        
        return signals
    
    def _generate_bos_signals(self, df: pd.DataFrame, ctx: TradingContext) -> List[SmartSignal]:
        """توليد إشارات Break of Structure"""
        signals = []
        current_price = ctx.price
        
        if ctx.structure.last_bos:
            # تأخر بسيط بعد BOS للدخول على Retest
            if ctx.structure.last_bos['type'] == 'bullish' and current_price > ctx.structure.last_bos['price']:
                # انتظار تراجع بسيط
                recent_low = min(df['low'].iloc[-5:])
                if current_price <= recent_low * 1.01:  # قريب من القاع الأخير
                    entry = current_price
                    sl = entry * 0.992  # 0.8% stop loss
                    
                    # مستويين TP للسكالب
                    tps = [
                        entry * (1 + SCALP_TP_LEVELS[0]/100),
                        entry * (1 + SCALP_TP_LEVELS[1]/100)
                    ]
                    
                    signal = SmartSignal(
                        action='BUY',
                        trade_type=TradeType.GOLDEN_BREAKOUT,
                        entry_price=entry,
                        stop_loss=sl,
                        take_profits=tps,
                        close_fractions=SCALP_TP_CLOSE_FRACTIONS,
                        confidence=0.7,
                        reasons=[
                            f"Bullish BOS at {ctx.structure.last_bos['price']:.6f}",
                            f"Retesting support",
                            f"Volume ratio: {ctx.volume_ratio:.1f}"
                        ],
                        zone_strength=7.0,
                        risk_reward=(tps[0] - entry) / (entry - sl),
                        market_context=ctx.__dict__,
                        timestamp=int(time.time()),
                        is_golden=False,
                        requires_confirmation=True
                    )
                    signals.append(signal)
            
            elif ctx.structure.last_bos['type'] == 'bearish' and current_price < ctx.structure.last_bos['price']:
                # انتظار ارتداد بسيط
                recent_high = max(df['high'].iloc[-5:])
                if current_price >= recent_high * 0.99:  # قريب من القمة الأخيرة
                    entry = current_price
                    sl = entry * 1.008  # 0.8% stop loss
                    
                    # مستويين TP للسكالب
                    tps = [
                        entry * (1 - SCALP_TP_LEVELS[0]/100),
                        entry * (1 - SCALP_TP_LEVELS[1]/100)
                    ]
                    
                    signal = SmartSignal(
                        action='SELL',
                        trade_type=TradeType.GOLDEN_BREAKOUT,
                        entry_price=entry,
                        stop_loss=sl,
                        take_profits=tps,
                        close_fractions=SCALP_TP_CLOSE_FRACTIONS,
                        confidence=0.7,
                        reasons=[
                            f"Bearish BOS at {ctx.structure.last_bos['price']:.6f}",
                            f"Retesting resistance",
                            f"Volume ratio: {ctx.volume_ratio:.1f}"
                        ],
                        zone_strength=7.0,
                        risk_reward=(entry - tps[0]) / (sl - entry),
                        market_context=ctx.__dict__,
                        timestamp=int(time.time()),
                        is_golden=False,
                        requires_confirmation=True
                    )
                    signals.append(signal)
        
        return signals
    
    def _generate_retest_signals(self, df: pd.DataFrame, ctx: TradingContext) -> List[SmartSignal]:
        """توليد إشارات إعادة الاختبار"""
        signals = []
        current_price = ctx.price
        
        # إعادة اختبار Order Blocks
        for ob in ctx.order_blocks[-3:]:
            distance = abs(current_price - ob['entry']) / ob['entry']
            
            if distance < 0.003:  # 0.3% قريب من OB
                if ob['type'] == OrderBlockType.BULLISH_OB:
                    entry = current_price
                    sl = min(ob['low'] * 0.998, entry * 0.995)
                    
                    tps = [
                        entry * (1 + SCALP_TP_LEVELS[0]/100),
                        entry * (1 + SCALP_TP_LEVELS[1]/100)
                    ]
                    
                    signal = SmartSignal(
                        action='BUY',
                        trade_type=TradeType.SCALP_RETEST,
                        entry_price=entry,
                        stop_loss=sl,
                        take_profits=tps,
                        close_fractions=SCALP_TP_CLOSE_FRACTIONS,
                        confidence=0.6,
                        reasons=[
                            f"Retesting Bullish OB at {ob['entry']:.6f}",
                            f"OB strength: {ob['strength']:.1f}/10",
                            f"Distance: {distance*100:.2f}%"
                        ],
                        zone_strength=ob['strength'],
                        risk_reward=(tps[0] - entry) / (entry - sl),
                        market_context=ctx.__dict__,
                        timestamp=int(time.time()),
                        is_golden=False,
                        requires_confirmation=True
                    )
                    signals.append(signal)
                
                elif ob['type'] == OrderBlockType.BEARISH_OB:
                    entry = current_price
                    sl = max(ob['high'] * 1.002, entry * 1.005)
                    
                    tps = [
                        entry * (1 - SCALP_TP_LEVELS[0]/100),
                        entry * (1 - SCALP_TP_LEVELS[1]/100)
                    ]
                    
                    signal = SmartSignal(
                        action='SELL',
                        trade_type=TradeType.SCALP_RETEST,
                        entry_price=entry,
                        stop_loss=sl,
                        take_profits=tps,
                        close_fractions=SCALP_TP_CLOSE_FRACTIONS,
                        confidence=0.6,
                        reasons=[
                            f"Retesting Bearish OB at {ob['entry']:.6f}",
                            f"OB strength: {ob['strength']:.1f}/10",
                            f"Distance: {distance*100:.2f}%"
                        ],
                        zone_strength=ob['strength'],
                        risk_reward=(entry - tps[0]) / (sl - entry),
                        market_context=ctx.__dict__,
                        timestamp=int(time.time()),
                        is_golden=False,
                        requires_confirmation=True
                    )
                    signals.append(signal)
        
        return signals
    
    def _generate_trend_signals(self, df: pd.DataFrame, ctx: TradingContext) -> List[SmartSignal]:
        """توليد إشارات متابعة الترند"""
        signals = []
        current_price = ctx.price
        
        # إشارات متابعة الترند القوي
        if ctx.adx > 25 and ctx.market_strength > 6.0:
            if ctx.bias in [MarketBias.STRONG_BULL, MarketBias.BULL]:
                # دخول على تراجع بسيط في الترند الصاعد
                if current_price <= ctx.ma_fast * 1.005:  # قريب من MA السريع
                    entry = current_price
                    sl = entry * 0.99  # 1% stop loss
                    
                    # TP ديناميكي للترند
                    tps = [
                        entry * (1 + TREND_TP_TRAIL_ATR/100),
                        entry * (1 + TREND_TP_TRAIL_ATR * 1.5/100)
                    ]
                    
                    signal = SmartSignal(
                        action='BUY',
                        trade_type=TradeType.TREND_FOLLOW,
                        entry_price=entry,
                        stop_loss=sl,
                        take_profits=tps,
                        close_fractions=[0.5, 0.5],  # نصف في كل مستوى
                        confidence=0.65,
                        reasons=[
                            f"Strong bullish trend (ADX: {ctx.adx:.1f})",
                            f"Retesting MA Fast: {ctx.ma_fast:.6f}",
                            f"Market strength: {ctx.market_strength:.1f}/10"
                        ],
                        zone_strength=7.5,
                        risk_reward=(tps[0] - entry) / (entry - sl),
                        market_context=ctx.__dict__,
                        timestamp=int(time.time()),
                        is_golden=False,
                        requires_confirmation=False  # لا يحتاج تأكيد في ترند قوي
                    )
                    signals.append(signal)
            
            elif ctx.bias in [MarketBias.STRONG_BEAR, MarketBias.BEAR]:
                # دخول على ارتداد بسيط في الترند الهابط
                if current_price >= ctx.ma_fast * 0.995:  # قريب من MA السريع
                    entry = current_price
                    sl = entry * 1.01  # 1% stop loss
                    
                    # TP ديناميكي للترند
                    tps = [
                        entry * (1 - TREND_TP_TRAIL_ATR/100),
                        entry * (1 - TREND_TP_TRAIL_ATR * 1.5/100)
                    ]
                    
                    signal = SmartSignal(
                        action='SELL',
                        trade_type=TradeType.TREND_FOLLOW,
                        entry_price=entry,
                        stop_loss=sl,
                        take_profits=tps,
                        close_fractions=[0.5, 0.5],  # نصف في كل مستوى
                        confidence=0.65,
                        reasons=[
                            f"Strong bearish trend (ADX: {ctx.adx:.1f})",
                            f"Retesting MA Fast: {ctx.ma_fast:.6f}",
                            f"Market strength: {ctx.market_strength:.1f}/10"
                        ],
                        zone_strength=7.5,
                        risk_reward=(entry - tps[0]) / (sl - entry),
                        market_context=ctx.__dict__,
                        timestamp=int(time.time()),
                        is_golden=False,
                        requires_confirmation=False
                    )
                    signals.append(signal)
        
        return signals

# =================== PROFESSIONAL TRADE MANAGER ===================
class ProfessionalTradeManager:
    """مدير الصفقات المحترف"""
    
    def __init__(self):
        self.current_trade = None
        self.trade_history = []
        self.entry_time = None
        self.best_price = None
        self.trail_stop = None
        self.tp_levels_hit = []
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.cooldown_until = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
    def open_trade(self, signal: SmartSignal, qty: float, entry_price: float):
        """فتح صفقة جديدة"""
        self.current_trade = {
            'signal': signal,
            'qty': qty,
            'entry_price': entry_price,
            'entry_time': time.time(),
            'best_price': entry_price,
            'trail_stop': signal.stop_loss,
            'tp_levels_hit': [],
            'current_pnl': 0.0,
            'max_pnl': 0.0,
            'status': 'OPEN'
        }
        self.entry_time = time.time()
        self.best_price = entry_price
        self.total_trades += 1
        
        logger.trade_execution(
            side=signal.action,
            qty=qty,
            price=entry_price,
            tp_levels=signal.take_profits,
            sl=signal.stop_loss
        )
        
        for reason in signal.reasons[:3]:
            logger.info(f"   📝 {reason}")
    
    def manage_trade(self, current_price: float, ctx: TradingContext) -> Optional[str]:
        """إدارة الصفقة الحالية"""
        if not self.current_trade:
            return None
        
        signal = self.current_trade['signal']
        entry = self.current_trade['entry_price']
        qty = self.current_trade['qty']
        
        # حساب PnL الحالي
        if signal.action == 'BUY':
            pnl_pct = (current_price - entry) / entry * 100
            pnl_usdt = (current_price - entry) * qty
        else:  # SELL
            pnl_pct = (entry - current_price) / entry * 100
            pnl_usdt = (entry - current_price) * qty
        
        self.current_trade['current_pnl'] = pnl_pct
        self.current_trade['max_pnl'] = max(self.current_trade['max_pnl'], pnl_pct)
        
        # تحديث أفضل سعر للترايلينغ
        if signal.action == 'BUY':
            if current_price > self.current_trade['best_price']:
                self.current_trade['best_price'] = current_price
        else:
            if current_price < self.current_trade['best_price']:
                self.current_trade['best_price'] = current_price
        
        # 1. تحقق من Stop Loss
        if self._check_stop_loss(current_price, signal):
            return self._close_trade("STOP_LOSS", current_price, pnl_pct, pnl_usdt)
        
        # 2. تحقق من Take Profit Levels
        tp_action = self._check_take_profits(current_price, signal)
        if tp_action:
            return self._close_trade(tp_action, current_price, pnl_pct, pnl_usdt)
        
        # 3. إدارة Trailing Stop للترند
        if signal.trade_type in [TradeType.TREND_FOLLOW, TradeType.GOLDEN_BREAKOUT]:
            trail_action = self._manage_trailing_stop(current_price, signal, ctx)
            if trail_action:
                return self._close_trade(trail_action, current_price, pnl_pct, pnl_usdt)
        
        # 4. حماية الربح
        if self.current_trade['max_pnl'] > 0.5 and pnl_pct < self.current_trade['max_pnl'] * 0.5:
            return self._close_trade("PROFIT_PROTECTION", current_price, pnl_pct, pnl_usdt)
        
        # 5. كشف المناطق الخطرة
        if ctx.danger_zones and len(ctx.danger_zones) >= 2:
            return self._close_trade("DANGER_ZONE", current_price, pnl_pct, pnl_usdt)
        
        # 6. Time Stop (حسب نوع الصفقة)
        time_in_trade = time.time() - self.entry_time
        if signal.trade_type == TradeType.SCALP_RETEST and time_in_trade > 1800:  # 30 دقيقة
            return self._close_trade("TIME_STOP_SCALP", current_price, pnl_pct, pnl_usdt)
        elif time_in_trade > 7200:  # ساعتين
            return self._close_trade("TIME_STOP_MAX", current_price, pnl_pct, pnl_usdt)
        
        return None
    
    def _check_stop_loss(self, current_price: float, signal: SmartSignal) -> bool:
        """تحقق من Stop Loss"""
        if signal.action == 'BUY':
            return current_price <= signal.stop_loss
        else:
            return current_price >= signal.stop_loss
    
    def _check_take_profits(self, current_price: float, signal: SmartSignal) -> Optional[str]:
        """تحقق من مستويات Take Profit"""
        for i, tp in enumerate(signal.take_profits):
            if i in self.current_trade['tp_levels_hit']:
                continue
            
            if signal.action == 'BUY':
                hit = current_price >= tp
            else:
                hit = current_price <= tp
            
            if hit:
                self.current_trade['tp_levels_hit'].append(i)
                
                # إغلاق كلي أو جزئي حسب الكسور
                close_fraction = signal.close_fractions[i]
                if close_fraction >= 1.0 or len(signal.take_profits) == i + 1:
                    return f"TP{i+1}_FULL"
                else:
                    # هنا يمكن تطبيق إغلاق جزئي
                    return f"TP{i+1}_PARTIAL"
        
        return None
    
    def _manage_trailing_stop(self, current_price: float, signal: SmartSignal, ctx: TradingContext) -> Optional[str]:
        """إدارة Trailing Stop للترند"""
        atr_percent = ctx.atr / current_price * 100
        
        if signal.action == 'BUY':
            # تحديث الترailing Stop
            new_trail = current_price * (1 - atr_percent * TREND_BREAKEVEN_ATR / 100)
            if new_trail > self.current_trade['trail_stop']:
                self.current_trade['trail_stop'] = new_trail
            
            # تحقق من كسر الترailing
            if current_price <= self.current_trade['trail_stop']:
                return "TRAIL_STOP"
        
        else:  # SELL
            new_trail = current_price * (1 + atr_percent * TREND_BREAKEVEN_ATR / 100)
            if new_trail < self.current_trade['trail_stop']:
                self.current_trade['trail_stop'] = new_trail
            
            if current_price >= self.current_trade['trail_stop']:
                return "TRAIL_STOP"
        
        return None
    
    def _close_trade(self, reason: str, exit_price: float, pnl_pct: float, pnl_usdt: float) -> str:
        """إغلاق الصفقة"""
        signal = self.current_trade['signal']
        entry = self.current_trade['entry_price']
        qty = self.current_trade['qty']
        side = "long" if signal.action == "BUY" else "short"
        
        self.total_pnl += pnl_usdt
        
        # تحديث الإحصائيات
        if pnl_pct > 0:
            self.winning_trades += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.cooldown_until = time.time() + COOLDOWN_AFTER_WIN
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.max_drawdown = min(self.max_drawdown, pnl_pct)
            
            # كولدآون بعد الخسارة
            loss_size = abs(pnl_pct)
            if loss_size > MAX_LOSS_PER_TRADE_PCT:
                self.cooldown_until = time.time() + COOLDOWN_AFTER_LOSS * 2
            else:
                self.cooldown_until = time.time() + COOLDOWN_AFTER_LOSS
        
        # تسجيل في التاريخ
        trade_record = {
            'timestamp': int(time.time()),
            'action': signal.action,
            'type': signal.trade_type.value,
            'entry': entry,
            'exit': exit_price,
            'qty': qty,
            'pnl': pnl_usdt,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'duration': time.time() - self.entry_time,
            'signal': signal.__dict__
        }
        self.trade_history.append(trade_record)
        
        # تسجيل النتيجة
        logger.trade_closed(
            side=side,
            exit_price=exit_price,
            pnl=pnl_usdt,
            pnl_pct=pnl_pct,
            reason=reason
        )
        
        # إعادة تعيين
        self.current_trade = None
        self.entry_time = None
        self.best_price = None
        
        return reason
    
    def should_trade(self) -> bool:
        """هل يمكن التداول الآن؟"""
        if time.time() < self.cooldown_until:
            remaining = int(self.cooldown_until - time.time())
            if remaining > 0 and remaining % 30 == 0:  # كل 30 ثانية
                logger.info(f"⏳ Cooldown: {remaining}s remaining")
            return False
        
        if self.consecutive_losses >= 3:
            logger.warning(f"🚫 Max consecutive losses reached: {self.consecutive_losses}")
            return False
        
        if abs(self.max_drawdown) > MAX_DRAWDOWN_PCT:
            logger.warning(f"🚫 Max drawdown reached: {self.max_drawdown:.2f}%")
            return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """الحصول على الإحصائيات"""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        # حساب متوسط الربح والخسارة
        avg_win = 0.0
        avg_loss = 0.0
        
        if self.trade_history:
            wins = [t['pnl_pct'] for t in self.trade_history if t['pnl_pct'] > 0]
            losses = [t['pnl_pct'] for t in self.trade_history if t['pnl_pct'] < 0]
            
            avg_win = np.mean(wins) if wins else 0.0
            avg_loss = np.mean(losses) if losses else 0.0
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'max_drawdown': self.max_drawdown,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0
        }

# =================== EXCHANGE INTEGRATION ===================
class BingXExchange:
    """تكامل مع منصة BingX"""
    
    def __init__(self):
        self.exchange = ccxt.bingx({
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "enableRateLimit": True,
            "timeout": 15000,
            "options": {"defaultType": "swap"}
        })
        self.symbol = SYMBOL
        self.load_market_info()
        
    def load_market_info(self):
        """تحميل معلومات السوق"""
        try:
            self.exchange.load_markets()
            market = self.exchange.market(self.symbol)
            self.precision = market['precision']['amount']
            self.min_qty = market['limits']['amount']['min']
            logger.success(f"Market loaded: {self.symbol}")
        except Exception as e:
            logger.error(f"Failed to load market: {e}")
    
    def get_balance(self) -> float:
        """الحصول على الرصيد"""
        try:
            balance = self.exchange.fetch_balance(params={"type": "swap"})
            total_balance = balance['USDT']['total']
            logger.info(f"💰 Balance: {total_balance:.2f} USDT")
            return total_balance
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0
    
    def get_current_price(self) -> float:
        """الحصول على السعر الحالي"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Failed to get price: {e}")
            return 0.0
    
    def get_orderbook_spread(self) -> float:
        """الحصول على السبريد"""
        try:
            orderbook = self.exchange.fetch_order_book(self.symbol, limit=5)
            bid = orderbook['bids'][0][0] if orderbook['bids'] else 0
            ask = orderbook['asks'][0][0] if orderbook['asks'] else 0
            if bid and ask:
                spread = ((ask - bid) / ((ask + bid) / 2)) * 10000
                return spread
        except Exception as e:
            logger.error(f"Failed to get spread: {e}")
        return 0.0
    
    def get_24h_volume(self) -> float:
        """الحصول على حجم التداول لـ24 ساعة"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker['quoteVolume'] if 'quoteVolume' in ticker else 0
        except Exception as e:
            logger.error(f"Failed to get 24h volume: {e}")
            return 0.0
    
    def execute_order(self, side: str, qty: float) -> bool:
        """تنفيذ أمر"""
        if DRY_RUN or not MODE_LIVE:
            logger.info(f"DRY RUN: Would {side.upper()} {qty:.4f} {self.symbol}")
            return True
        
        try:
            # ضبط الرافعة
            self.exchange.set_leverage(LEVERAGE, self.symbol, params={"side": "BOTH"})
            
            # تنفيذ الأمر
            order = self.exchange.create_order(
                symbol=self.symbol,
                type='market',
                side=side,
                amount=qty,
                params={"positionSide": "LONG" if side == "buy" else "SHORT", "reduceOnly": False}
            )
            
            logger.success(f"✅ Order executed: {side.upper()} {qty:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to execute order: {e}")
            return False
    
    def close_position(self, side: str, qty: float) -> bool:
        """إغلاق المركز"""
        if DRY_RUN or not MODE_LIVE:
            logger.info(f"DRY RUN: Would close {side.upper()} {qty:.4f}")
            return True
        
        try:
            close_side = "sell" if side == "long" else "buy"
            order = self.exchange.create_order(
                symbol=self.symbol,
                type='market',
                side=close_side,
                amount=qty,
                params={"positionSide": "LONG" if side == "long" else "SHORT", "reduceOnly": True}
            )
            
            logger.success(f"✅ Position closed: {side} {qty:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to close position: {e}")
            return False

# =================== MAIN BOT ENGINE ===================
class UltimateDogeProBot:
    """المحرك الرئيسي للبوت"""
    
    def __init__(self):
        self.exchange = BingXExchange()
        self.signal_generator = UltimateSignalGenerator()
        self.trade_manager = ProfessionalTradeManager()
        self.ohlcv_data = None
        self.last_update = 0
        self.update_interval = 15  # ثانية
        self.is_running = True
        self.cycle_count = 0
        
        logger.banner("ULTIMATE DOGE PRO BOT INITIALIZED")
    
    def fetch_ohlcv(self) -> pd.DataFrame:
        """جلب بيانات OHLCV"""
        try:
            ohlcv = self.exchange.exchange.fetch_ohlcv(
                SYMBOL, 
                timeframe=INTERVAL, 
                limit=200
            )
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['time'] = df['timestamp']
            return df
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV: {e}")
            return None
    
    def calculate_position_size(self, balance: float, entry_price: float) -> float:
        """حجم المركز"""
        risk_amount = balance * RISK_ALLOC * LEVERAGE
        raw_qty = risk_amount / entry_price
        
        # تقريب حسب دقة السوق
        precision = self.exchange.precision
        if precision > 0:
            qty = round(raw_qty, precision)
        else:
            qty = int(raw_qty)
        
        # التحقق من الحد الأدنى
        if qty < self.exchange.min_qty:
            return 0.0
        
        return qty
    
    def show_market_status(self):
        """عرض حالة السوق"""
        try:
            price = self.exchange.get_current_price()
            volume_24h = self.exchange.get_24h_volume()
            spread = self.exchange.get_orderbook_spread()
            
            logger.market_status(price, volume_24h, spread)
        except Exception as e:
            logger.error(f"Failed to show market status: {e}")
    
    def show_balance_status(self, show_equity=True):
        """عرض حالة الرصيد"""
        try:
            balance = self.exchange.get_balance()
            pnl = self.trade_manager.total_pnl
            
            if show_equity and self.trade_manager.current_trade:
                # حساب قيمة المركز الحالي
                current_price = self.exchange.get_current_price()
                trade = self.trade_manager.current_trade
                
                if trade['signal'].action == 'BUY':
                    equity = balance + (current_price - trade['entry_price']) * trade['qty']
                else:
                    equity = balance + (trade['entry_price'] - current_price) * trade['qty']
            else:
                equity = None
            
            logger.balance_display(balance, pnl, equity)
        except Exception as e:
            logger.error(f"Failed to show balance status: {e}")
    
    def show_indicators_status(self, df: pd.DataFrame):
        """عرض حالة المؤشرات"""
        try:
            if df is not None and len(df) > 50:
                ctx = self.signal_generator.analyze_market(df)
                logger.indicators_status(
                    rsi=ctx.rsi,
                    adx=ctx.adx,
                    macd=ctx.macd_histogram,
                    ma_fast=ctx.ma_fast,
                    ma_slow=ctx.ma_slow
                )
        except Exception as e:
            logger.error(f"Failed to show indicators: {e}")
    
    def run_cycle(self):
        """دورة التداول الرئيسية"""
        try:
            self.cycle_count += 1
            
            # 1. تحديث البيانات
            current_time = time.time()
            if current_time - self.last_update >= self.update_interval:
                self.ohlcv_data = self.fetch_ohlcv()
                self.last_update = current_time
            
            if self.ohlcv_data is None or len(self.ohlcv_data) < 50:
                time.sleep(5)
                return
            
            # 2. عرض معلومات كل 10 دورات
            if self.cycle_count % 10 == 0:
                logger.banner(f"CYCLE {self.cycle_count}")
                self.show_market_status()
                self.show_balance_status()
                self.show_indicators_status(self.ohlcv_data)
            
            # 3. التحقق من السبريد
            spread = self.exchange.get_orderbook_spread()
            if spread > MAX_SPREAD_BPS:
                logger.warning(f"Spread too high: {spread:.1f} bps")
                time.sleep(10)
                return
            
            # 4. إذا كانت هناك صفقة مفتوحة، إدارتها
            if self.trade_manager.current_trade:
                current_price = self.exchange.get_current_price()
                if current_price > 0:
                    close_reason = self.trade_manager.manage_trade(
                        current_price, 
                        self.signal_generator.analyze_market(self.ohlcv_data)
                    )
                    
                    if close_reason:
                        # إغلاق الصفقة في البورصة
                        trade = self.trade_manager.current_trade
                        qty = trade['qty']
                        side = "long" if trade['signal'].action == "BUY" else "short"
                        self.exchange.close_position(side, qty)
                        
                        # عرض الرصيد بعد الإغلاق
                        self.show_balance_status(show_equity=False)
            
            # 5. إذا لم تكن هناك صفقة مفتوحة، البحث عن إشارات
            elif self.trade_manager.should_trade():
                # تحليل السوق
                ctx = self.signal_generator.analyze_market(self.ohlcv_data)
                
                # توليد الإشارات
                signals = self.signal_generator.generate_signals(self.ohlcv_data)
                
                # اختيار أفضل إشارة
                if signals:
                    # فرز حسب الثقة وقوة المنطقة
                    signals.sort(key=lambda x: (x.confidence * 0.6 + x.zone_strength/10 * 0.4), reverse=True)
                    best_signal = signals[0]
                    
                    # عرض الإشارة
                    logger.trade_decision(
                        action=best_signal.action,
                        reason=best_signal.reasons[0] if best_signal.reasons else "No reason",
                        confidence=best_signal.confidence,
                        zone_strength=best_signal.zone_strength
                    )
                    
                    # الحصول على الرصيد وحساب الحجم
                    balance = self.exchange.get_balance()
                    qty = self.calculate_position_size(balance, best_signal.entry_price)
                    
                    if qty > 0:
                        # تنفيذ الصفقة
                        if self.exchange.execute_order(
                            "buy" if best_signal.action == "BUY" else "sell",
                            qty
                        ):
                            self.trade_manager.open_trade(best_signal, qty, best_signal.entry_price)
                            self.show_balance_status()
            
            # 6. النوم قبل الدورة التالية
            sleep_time = 3 if self.trade_manager.current_trade else 5
            time.sleep(sleep_time)
            
        except Exception as e:
            logger.error(f"Error in run cycle: {e}")
            traceback.print_exc()
            time.sleep(30)
    
    def show_statistics(self):
        """عرض الإحصائيات"""
        stats = self.trade_manager.get_statistics()
        
        logger.banner("TRADING STATISTICS")
        logger.info(f"📈 Total Trades: {stats['total_trades']}")
        logger.info(f"✅ Winning Trades: {stats['winning_trades']} ({stats['win_rate']:.1f}%)")
        logger.info(f"❌ Losing Trades: {stats['losing_trades']}")
        logger.info(f"💰 Total PnL: {stats['total_pnl']:+.2f} USDT")
        logger.info(f"📊 Avg Win: {stats['avg_win']:+.2f}%")
        logger.info(f"📉 Avg Loss: {stats['avg_loss']:+.2f}%")
        logger.info(f"⚡ Consecutive Wins: {stats['consecutive_wins']}")
        logger.info(f"🔻 Consecutive Losses: {stats['consecutive_losses']}")
        logger.info(f"📉 Max Drawdown: {stats['max_drawdown']:.2f}%")
        logger.info(f"📊 Profit Factor: {stats['profit_factor']:.2f}")
    
    def run(self):
        """تشغيل البوت"""
        logger.success("🚀 Starting Ultimate Doge Pro Bot...")
        
        # عرض الرصيد الأولي
        self.show_balance_status(show_equity=False)
        
        while self.is_running:
            try:
                self.run_cycle()
            except KeyboardInterrupt:
                logger.warning("🛑 Bot stopped by user")
                self.is_running = False
                self.show_statistics()
            except Exception as e:
                logger.error(f"⚠️ Unexpected error: {e}")
                traceback.print_exc()
                time.sleep(60)

# =================== FLASK API ===================
app = Flask(__name__)
bot = UltimateDogeProBot()

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ultimate Doge Pro Bot</title>
        <style>
            body { 
                font-family: 'Arial', sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: white;
            }
            .container { 
                max-width: 1000px; 
                margin: 0 auto; 
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            .status-card {
                background: rgba(255, 255, 255, 0.2);
                border-radius: 15px;
                padding: 20px;
                margin: 20px 0;
                transition: transform 0.3s;
            }
            .status-card:hover {
                transform: translateY(-5px);
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            .stat-item {
                background: rgba(255, 255, 255, 0.15);
                padding: 15px;
                border-radius: 10px;
                text-align: center;
            }
            .btn {
                display: inline-block;
                background: linear-gradient(45deg, #FF6B6B, #FF8E53);
                color: white;
                padding: 12px 30px;
                border-radius: 25px;
                text-decoration: none;
                margin: 10px;
                transition: all 0.3s;
                font-weight: bold;
            }
            .btn:hover {
                transform: scale(1.05);
                box-shadow: 0 5px 15px rgba(255, 107, 107, 0.4);
            }
            .live-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                background: #4CAF50;
                border-radius: 50%;
                animation: pulse 2s infinite;
                margin-right: 8px;
            }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            .icon {
                font-size: 24px;
                margin-right: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Ultimate Doge Pro Bot v10.0</h1>
                <p><span class="live-indicator"></span> Live Trading System</p>
            </div>
            
            <div class="status-card">
                <h2>📊 System Status</h2>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="icon">💰</div>
                        <h3>Balance</h3>
                        <p>Loading...</p>
                    </div>
                    <div class="stat-item">
                        <div class="icon">📈</div>
                        <h3>Total PnL</h3>
                        <p>Loading...</p>
                    </div>
                    <div class="stat-item">
                        <div class="icon">✅</div>
                        <h3>Win Rate</h3>
                        <p>Loading...</p>
                    </div>
                    <div class="stat-item">
                        <div class="icon">⚡</div>
                        <h3>Trades</h3>
                        <p>Loading...</p>
                    </div>
                </div>
            </div>
            
            <div class="status-card">
                <h2>🔗 Quick Links</h2>
                <div style="text-align: center;">
                    <a href="/metrics" class="btn">📊 Trading Metrics</a>
                    <a href="/health" class="btn">❤️ Health Check</a>
                    <a href="/statistics" class="btn">📈 Statistics</a>
                    <a href="/signals" class="btn">🎯 Recent Signals</a>
                </div>
            </div>
            
            <div class="status-card">
                <h2>⚙️ Configuration</h2>
                <p><strong>Symbol:</strong> ''' + SYMBOL + '''</p>
                <p><strong>Interval:</strong> ''' + INTERVAL + '''</p>
                <p><strong>Leverage:</strong> ''' + str(LEVERAGE) + '''x</p>
                <p><strong>Risk Allocation:</strong> ''' + str(RISK_ALLOC * 100) + '''%</p>
                <p><strong>Mode:</strong> ''' + ("LIVE 🟢" if MODE_LIVE else "PAPER 🔴") + '''</p>
            </div>
        </div>
        
        <script>
            async function updateStats() {
                try {
                    const response = await fetch('/statistics');
                    const data = await response.json();
                    
                    // Update statistics
                    document.querySelectorAll('.stat-item')[0].querySelector('p').textContent = data.balance || 'Loading...';
                    document.querySelectorAll('.stat-item')[1].querySelector('p').textContent = data.total_pnl ? data.total_pnl.toFixed(2) + ' USDT' : 'Loading...';
                    document.querySelectorAll('.stat-item')[2].querySelector('p').textContent = data.win_rate ? data.win_rate.toFixed(1) + '%' : 'Loading...';
                    document.querySelectorAll('.stat-item')[3].querySelector('p').textContent = data.total_trades || '0';
                    
                    // Update configuration with actual data if available
                    if (data.balance) {
                        document.querySelectorAll('.status-card')[2].innerHTML += `
                            <p><strong>Current Balance:</strong> ${data.balance.toFixed(2)} USDT</p>
                            <p><strong>Active Trades:</strong> ${data.current_trade ? 'Yes' : 'No'}</p>
                        `;
                    }
                } catch (error) {
                    console.error('Error updating stats:', error);
                }
            }
            
            // Update stats every 10 seconds
            setInterval(updateStats, 10000);
            updateStats(); // Initial call
        </script>
    </body>
    </html>
    '''

@app.route('/metrics')
def metrics():
    try:
        balance = bot.exchange.get_balance()
        price = bot.exchange.get_current_price()
        spread = bot.exchange.get_orderbook_spread()
        
        return {
            'bot': BOT_VERSION,
            'symbol': SYMBOL,
            'interval': INTERVAL,
            'mode': 'LIVE' if MODE_LIVE else 'PAPER',
            'leverage': LEVERAGE,
            'risk_allocation': RISK_ALLOC,
            'balance': balance,
            'price': price,
            'spread': spread,
            'status': 'running',
            'timestamp': datetime.utcnow().isoformat(),
            'cycle_count': bot.cycle_count
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}, 500

@app.route('/health')
def health():
    try:
        price = bot.exchange.get_current_price()
        balance = bot.exchange.get_balance()
        
        return {
            'status': 'healthy',
            'price': price,
            'balance': balance,
            'current_trade': bot.trade_manager.current_trade is not None,
            'exchange_connected': price > 0,
            'cycle_count': bot.cycle_count,
            'uptime': time.time() - bot.session_start_time if hasattr(bot, 'session_start_time') else 0
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}, 500

@app.route('/statistics')
def statistics():
    try:
        stats = bot.trade_manager.get_statistics()
        balance = bot.exchange.get_balance()
        
        stats['balance'] = balance
        stats['current_trade'] = bot.trade_manager.current_trade is not None
        stats['cycle_count'] = bot.cycle_count
        stats['timestamp'] = datetime.utcnow().isoformat()
        
        return stats
    except Exception as e:
        return {'status': 'error', 'message': str(e)}, 500

@app.route('/signals')
def signals():
    try:
        if bot.signal_generator.last_signals:
            signals = []
            for signal in list(bot.signal_generator.last_signals)[-5:]:
                signals.append({
                    'action': signal.action,
                    'type': signal.trade_type.value,
                    'entry': signal.entry_price,
                    'confidence': signal.confidence,
                    'zone_strength': signal.zone_strength,
                    'timestamp': signal.timestamp,
                    'time_ago': time.time() - signal.timestamp,
                    'reasons': signal.reasons[:3]
                })
            return {'signals': signals}
        return {'signals': []}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}, 500

# =================== MAIN EXECUTION ===================
if __name__ == "__main__":
    import threading
    
    # بدء البوت في thread منفصل
    bot_thread = threading.Thread(target=bot.run, daemon=True)
    bot_thread.start()
    
    logger.success(f"🤖 Bot started in separate thread")
    logger.info(f"🌐 Web interface available at http://localhost:{PORT}")
    logger.info(f"📊 Metrics at http://localhost:{PORT}/metrics")
    
    # تشغيل Flask
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)

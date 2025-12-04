# -*- coding: utf-8 -*-
"""
RF Futures Bot — RF-LIVE ONLY (BingX Perp via CCXT)
• Council PRO Unified Decision System with Candles & Golden Entry
• Golden Entry + Golden Reversal + Wick Exhaustion + Smart Profit AI
• Dynamic TP ladder + Breakeven + ATR-trailing
• Smart Exit Management + Wait-for-next-signal
• Professional Logging & Dashboard
• Enhanced with Footprint, SMC Candles, Liquidity Traps + VWAP Strategy
• Advanced SMC Integration: Supply/Demand, Liquidity Sweep, Breaker Blocks, FVG, Elliott Waves, Stop Hunt
• STOP-HUNT FILTERS: ADX + ATR Advanced Filtering System
• Ultra Market Structure Engine (BOS/CHoCH, FVG, Premium/Discount, Liquidity Grabs)
• Real Range Filter (RF) Pine Exact + VWAP Session
• EDGE ALGO ENGINE: RR Zones + Setup Quality + Dynamic Profit Profiles
• CVD DIVERGENCE OSCILLATOR: TradingFinder-style divergence detection
• ENHANCED FILTERS: CVD ADX≥20 + Score≥8 | SMC Confidence≥0.8 | Signal Strength≥6.0
• SMART SIGNAL & SMART PROFIT SYSTEM: RF Master + Edge Zones + HOLD-TP Logic
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
BOT_VERSION = "DOGE Council PRO v8.0 — Ultra Market Structure + Real RF + VWAP + SMC Advanced + Elliott Waves + Stop Hunt AI + ADX+ATR Filters + EdgeAlgo + CVD Divergence + ENHANCED FILTERS + SMART SIGNAL/PROFIT"
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

# ==== Smart Signal / Smart Profit Settings ====
# أقل قوة إشارة مسموح بيها عشان ندخل من الأساس
MIN_SIGNAL_FOR_ENTRY        = 6.0   # أقل من كده = سكالب تافه → نرميه
MIN_SIGNAL_FOR_SCALP_ENTRY  = 8.0   # في بيئة سكالب/عرضية لازم الإشارة تكون قوية جدًا

# منطق HOLD-TP بعد TP1
HOLD_AFTER_TP1_MIN_STRENGTH = 7.0   # قوة الإشارة المطلوبة بعد TP1
HOLD_AFTER_TP1_MIN_ADX      = 22.0  # ADX محترم
HOLD_AFTER_TP1_EXTRA_BOOST  = 1.30  # نرفع الهدف 30%

# لو الإشارة خرافية جدًا
VERY_STRONG_SIGNAL          = 8.5
VERY_STRONG_TP_BOOST        = 1.50  # نرفع الهدف 50%

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

# ==== Advanced SMC Settings ====
SMC_ENABLED = True
OB_STRENGTH_THRESHOLD = 0.7
FVG_VALIDITY_THRESHOLD = 0.7
STOP_HUNT_CONFIRMATION_BARS = 3
LIQUIDITY_SWEEP_CONFIRMATION = True
ELLIOTT_WAVE_ENABLED = True

# =================== ENHANCED STOP-HUNT FILTERS WITH ADX+ATR ===================
STOP_HUNT_ADX_MIN = 20          # أدنى ADX لاعتبار حركة Stop-Hunt
STOP_HUNT_ADX_MAX = 45          # أقصى ADX لدخول عكسي آمن
STOP_HUNT_ATR_MULT_MIN = 1.3    # الحد الأدنى لحجم الشمعة (أكبر من المتوسط)
STOP_HUNT_WICK_RATIO = 0.6      # نسبة الذيل/فتيلة في الشمعة
STOP_HUNT_DISTANCE_ATR = 0.5    # المسافة الدنيا فوق/تحت مستوى السيولة
STOP_HUNT_SL_ATR_MULT = 0.7     # مضاعف ATR لوضع Stop Loss

# =================== ENHANCED ENTRY FILTERS ===================
CVD_ADX_MIN = 20                # الحد الأدنى لـ ADX لدخول CVD
CVD_SCORE_MIN = 8.0             # الحد الأدنى لـ Council Score لدخول CVD
SMC_CONFIDENCE_MIN = 0.8        # الحد الأدنى لثقة SMC (كان 0.7)
SIGNAL_STRENGTH_MIN = 6.0       # الحد الأدنى لقوة الإشارة لأي دخول

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
            json.dump(state, f, ensure_asci=False, indent=2)
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

# =========================
# RANGE FILTER REAL (RF) — PINE EXACT
# =========================

def compute_range_filter(df: pd.DataFrame, period: int = 20, qty: float = 3.5) -> dict:
    """
    تحويل سكريبت Pine Range Filter (DW) إلى Python
    يرجّع:
      - rf_filt, rf_dir
      - rf_buy_signal, rf_sell_signal
      - hi_band, lo_band
    ويضيف الأعمدة دي في df أيضًا.
    """
    src = df["close"].astype(float).copy()

    if len(src) < period + 2:
        # df صغير → رجّع قيم افتراضية
        df["rf_filt"] = src
        df["rf_hi"] = src
        df["rf_lo"] = src
        df["rf_dir"] = 0
        df["rf_buy_signal"] = False
        df["rf_sell_signal"] = False
        return {
            "filt": float(src.iloc[-1]),
            "hi_band": float(src.iloc[-1]),
            "lo_band": float(src.iloc[-1]),
            "dir": 0,
            "buy_signal": False,
            "sell_signal": False,
        }

    # ===== rng_size من Pine =====
    diff = (src - src.shift(1)).abs()
    avrng = diff.ewm(span=period, adjust=False).mean()
    wper = (period * 2) - 1
    ac = avrng.ewm(span=wper, adjust=False).mean() * qty  # AC في Pine

    # ===== rng_filt array logic =====
    filt_vals = []
    hi_vals = []
    lo_vals = []

    # أول قيمة
    first_x = float(src.iloc[0])
    first_r = float(ac.iloc[0])
    cur_filt = first_x
    filt_vals.append(cur_filt)
    hi_vals.append(cur_filt + first_r)
    lo_vals.append(cur_filt - first_r)

    for i in range(1, len(src)):
        x = float(src.iloc[i])
        r = float(ac.iloc[i])
        prev = cur_filt

        # نفس منطق:
        # if x - r > rfilt[1] → rfilt[0] = x - r
        if x - r > prev:
            cur_filt = x - r
        # if x + r < rfilt[1] → rfilt[0] = x + r
        elif x + r < prev:
            cur_filt = x + r
        # else يبقى كما هو

        filt_vals.append(cur_filt)
        hi_vals.append(cur_filt + r)
        lo_vals.append(cur_filt - r)

    rf_filt = pd.Series(filt_vals, index=df.index)
    hi_band = pd.Series(hi_vals, index=df.index)
    lo_band = pd.Series(lo_vals, index=df.index)

    # ===== Direction + Signals من Pine =====
    fdir = [0] * len(src)
    cond_ini = [0] * len(src)
    long_sig = [False] * len(src)
    short_sig = [False] * len(src)

    for i in range(1, len(src)):
        # fdir := filt > filt[1] ? 1 : filt < filt[1] ? -1 : fdir
        if rf_filt.iloc[i] > rf_filt.iloc[i - 1]:
            fdir[i] = 1
        elif rf_filt.iloc[i] < rf_filt.iloc[i - 1]:
            fdir[i] = -1
        else:
            fdir[i] = fdir[i - 1]

        upward = fdir[i] == 1
        downward = fdir[i] == -1

        # longCond / shortCond من Pine بالظبط
        longCond = (
            (src.iloc[i] > rf_filt.iloc[i] and src.iloc[i] > src.iloc[i - 1] and upward)
            or (src.iloc[i] > rf_filt.iloc[i] and src.iloc[i] < src.iloc[i - 1] and upward)
        )
        shortCond = (
            (src.iloc[i] < rf_filt.iloc[i] and src.iloc[i] < src.iloc[i - 1] and downward)
            or (src.iloc[i] < rf_filt.iloc[i] and src.iloc[i] > src.iloc[i - 1] and downward)
        )

        # CondIni := long ? 1 : short ? -1 : CondIni[1]
        if longCond:
            cond_ini[i] = 1
        elif shortCond:
            cond_ini[i] = -1
        else:
            cond_ini[i] = cond_ini[i - 1]

        # longCondition = longCond and CondIni[1] == -1
        if longCond and cond_ini[i - 1] == -1:
            long_sig[i] = True
        # shortCondition = shortCond and CondIni[1] == 1
        if shortCond and cond_ini[i - 1] == 1:
            short_sig[i] = True

    rf_dir = pd.Series(fdir, index=df.index)
    buy_series = pd.Series(long_sig, index=df.index)
    sell_series = pd.Series(short_sig, index=df.index)

    # الحق الأعمدة في df لاستخدامها لاحقاً لو حبّينا
    df["rf_filt"] = rf_filt
    df["rf_hi"] = hi_band
    df["rf_lo"] = lo_band
    df["rf_dir"] = rf_dir
    df["rf_buy_signal"] = buy_series
    df["rf_sell_signal"] = sell_series

    return {
        "filt": float(rf_filt.iloc[-1]),
        "hi_band": float(hi_band.iloc[-1]),
        "lo_band": float(lo_band.iloc[-1]),
        "dir": int(rf_dir.iloc[-1]),
        "buy_signal": bool(buy_series.iloc[-1]),
        "sell_signal": bool(sell_series.iloc[-1]),
    }

# =========================
# VWAP ENGINE (SESSION VWAP)
# =========================

def compute_vwap(df: pd.DataFrame) -> float:
    """
    VWAP الكلاسيكي:
    sum(price * volume) / sum(volume) من بداية البيانات حتى آخر شمعة.
    """
    if "close" not in df.columns or "volume" not in df.columns or len(df) == 0:
        return 0.0

    close = df["close"].astype(float)
    vol = df["volume"].astype(float)

    pv = close * vol
    cum_pv = pv.cumsum()
    cum_vol = vol.cumsum().replace(0, np.nan)

    vwap = cum_pv / cum_vol
    df["vwap"] = vwap

    return float(vwap.iloc[-1])

# =========================
# ULTRA MARKET STRUCTURE ENGINE
# =========================

class UltraMarketStructureEngine:
    """
    تبسيط علمي لمؤشر Ultra Market Structure:
    - Internal / External structure (آخر قمم وقيعان + BOS / CHoCH)
    - FVG (Bull / Bear) + فلتر حجم gap
    - Premium / Discount zones بناءً على SMA200 + انحراف
    - Liquidity Grab (كسرة وهمية فوق قمة أو تحت قاع)
    """

    def __init__(
        self,
        int_lookback: int = 20,
        ext_lookback: int = 200,
        fvg_threshold_mult: float = 1.0,
        premium_mult_inner: float = 2.0,
        premium_mult_outer: float = 3.0,
    ):
        self.int_lookback = int_lookback
        self.ext_lookback = ext_lookback
        self.fvg_threshold_mult = fvg_threshold_mult
        self.prem_inner = premium_mult_inner
        self.prem_outer = premium_mult_outer

    def _detect_swings(self, df: pd.DataFrame, window: int = 3):
        """
        اكتشاف swing highs/lows البسيطة (internal).
        """
        h = df["high"].astype(float)
        l = df["low"].astype(float)

        swing_high_idx = []
        swing_low_idx = []

        for i in range(window, len(df) - window):
            hi = h.iloc[i]
            lo = l.iloc[i]

            if hi == h.iloc[i - window : i + window + 1].max():
                swing_high_idx.append(i)

            if lo == l.iloc[i - window : i + window + 1].min():
                swing_low_idx.append(i)

        return swing_high_idx, swing_low_idx

    def _last_swing_levels(self, df: pd.DataFrame, lookback: int):
        """
        استخراج آخر قمة وآخر قاع خلال نطاق lookback.
        """
        sub = df.iloc[-lookback:]
        high = sub["high"].astype(float)
        low = sub["low"].astype(float)

        last_high_idx = high.idxmax()
        last_low_idx = low.idxmin()

        return (
            float(df.loc[last_high_idx, "high"]),
            int(df.index.get_loc(last_high_idx)),
            float(df.loc[last_low_idx, "low"]),
            int(df.index.get_loc(last_low_idx)),
        )

    def _detect_bos_choch(self, df: pd.DataFrame, lookback: int = 50):
        """
        BOS / CHoCH بسيط:
        - BOS UP: إغلاق فوق آخر قمة مهمة.
        - BOS DOWN: إغلاق تحت آخر قاع مهم.
        """
        if len(df) < lookback + 5:
            return None, None

        close = df["close"].astype(float)
        last_high, last_high_pos, last_low, last_low_pos = self._last_swing_levels(df, lookback)

        bos = None
        choch = None

        # BOS UP
        if close.iloc[-1] > last_high and close.iloc[-2] <= last_high:
            bos = "up"
        # BOS DOWN
        if close.iloc[-1] < last_low and close.iloc[-2] >= last_low:
            bos = "down"

        # CHoCH = BOS عكس الاتجاه السابق البسيط
        if bos is not None:
            choch = bos

        return bos, choch

    def _detect_fvg(self, df: pd.DataFrame, max_lookback: int = 40):
        """
        كشف أقرب FVG بسيط خلال آخر max_lookback شمعة.
        """
        if len(df) < 5:
            return None

        h = df["high"].astype(float)
        l = df["low"].astype(float)
        c = df["close"].astype(float)

        # ATR بسيط للفلتر
        tr1 = (h - l).abs()
        tr2 = (h - c.shift(1)).abs()
        tr3 = (l - c.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=5).mean()
        atr_val = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 0.0
        if atr_val <= 0:
            atr_val = (h.iloc[-1] - l.iloc[-1]) or 1e-6

        start_idx = max(2, len(df) - max_lookback)
        bull_fvg = None
        bear_fvg = None

        for i in range(start_idx, len(df)):
            # Bullish FVG: المنطقة بين high[i-2] و low[i]
            if l.iloc[i] > h.iloc[i - 2]:
                gap = l.iloc[i] - h.iloc[i - 2]
                if gap >= self.fvg_threshold_mult * (0.5 * atr_val):
                    bull_fvg = {
                        "type": "bull",
                        "index": int(i),
                        "upper": float(l.iloc[i]),
                        "lower": float(h.iloc[i - 2]),
                        "size": float(gap),
                    }

            # Bearish FVG: المنطقة بين low[i-2] و high[i]
            if h.iloc[i] < l.iloc[i - 2]:
                gap = l.iloc[i - 2] - h.iloc[i]
                if gap >= self.fvg_threshold_mult * (0.5 * atr_val):
                    bear_fvg = {
                        "type": "bear",
                        "index": int(i),
                        "upper": float(l.iloc[i - 2]),
                        "lower": float(h.iloc[i]),
                        "size": float(gap),
                    }

        current_price = float(df["close"].iloc[-1])
        fvg_ctx = {
            "bull_near": False,
            "bear_near": False,
            "bull": bull_fvg,
            "bear": bear_fvg,
        }

        if bull_fvg is not None:
            mid = 0.5 * (bull_fvg["upper"] + bull_fvg["lower"])
            if abs(current_price - mid) <= atr_val:
                fvg_ctx["bull_near"] = True

        if bear_fvg is not None:
            mid = 0.5 * (bear_fvg["upper"] + bear_fvg["lower"])
            if abs(current_price - mid) <= atr_val:
                fvg_ctx["bear_near"] = True

        return fvg_ctx

    def _premium_discount(self, df: pd.DataFrame):
        """
        Premium / Discount بناءً على SMA200 + انحراف قياسي.
        """
        c = df["close"].astype(float)
        if len(c) < 210:
            return {
                "zone": "mid",
                "basis": float(c.iloc[-1]),
                "upper": float(c.iloc[-1]),
                "lower": float(c.iloc[-1]),
            }

        basis = c.rolling(window=200).mean()
        std = c.rolling(window=200).std()

        b = float(basis.iloc[-1])
        s = float(std.iloc[-1])
        if np.isnan(b) or np.isnan(s) or s == 0:
            b = float(c.iloc[-1])
            s = (c.max() - c.min()) / 10 or 1e-6

        upper_outer = b + self.prem_outer * s
        lower_outer = b - self.prem_outer * s

        price = float(c.iloc[-1])

        zone = "mid"
        if price > upper_outer:
            zone = "ultra_premium"
        elif price > b + self.prem_inner * s:
            zone = "premium"
        elif price < lower_outer:
            zone = "ultra_discount"
        elif price < b - self.prem_inner * s:
            zone = "discount"

        return {
            "zone": zone,
            "basis": b,
            "upper": upper_outer,
            "lower": lower_outer,
        }

    def _detect_liquidity_grab(self, df: pd.DataFrame, lookback: int = 20):
        """
        Liquidity Grab بسيط:
        - شمعة عملت ذيل فوق آخر قمة ثم أغلقت تحتها → grab up.
        - أو تحت آخر قاع ثم أغلقت فوقه → grab down.
        """
        if len(df) < lookback + 3:
            return {"grab_up": False, "grab_down": False}

        sub = df.iloc[-lookback:]
        high = sub["high"].astype(float)
        low = sub["low"].astype(float)
        close = sub["close"].astype(float)

        last_high = float(high.max())
        last_low = float(low.min())

        # آخر شمعة
        h_last = float(df["high"].iloc[-1])
        l_last = float(df["low"].iloc[-1])
        c_last = float(df["close"].iloc[-1])

        grab_up = h_last > last_high and c_last < last_high
        grab_down = l_last < last_low and c_last > last_low

        return {
            "grab_up": bool(grab_up),
            "grab_down": bool(grab_down),
        }

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        يرجّع سياق كامل لـ Ultra Market Structure:
        - bias (bull/bear/neutral)
        - bos / choch
        - fvg context
        - premium/discount zone
        - liquidity grab flags
        """
        if df is None or len(df) < 30:
            return {
                "bias": "neutral",
                "bos": None,
                "choch": None,
                "fvg": None,
                "premium_discount": None,
                "liq_grab": {"grab_up": False, "grab_down": False},
            }

        bos_int, choch_int = self._detect_bos_choch(df, lookback=self.int_lookback)
        fvg_ctx = self._detect_fvg(df, max_lookback=40)
        prem_ctx = self._premium_discount(df)
        liq_ctx = self._detect_liquidity_grab(df, lookback=self.int_lookback)

        # bias بسيط:
        bias = "neutral"
        if bos_int == "up":
            bias = "bull"
        elif bos_int == "down":
            bias = "bear"

        return {
            "bias": bias,
            "bos": bos_int,
            "choch": choch_int,
            "fvg": fvg_ctx,
            "premium_discount": prem_ctx,
            "liq_grab": liq_ctx,
        }

# ============================================
#  EDGE ALGO ENGINE — RR ZONES + SETUP QUALITY
# ============================================

class EdgeAlgoEngine:
    """
    محرك Edge Algo:
    - يحسب Setup كامل للصفقة (ENTRY/SL/TP1/TP2/TP3)
    - يحسب RR تقريبي (1R, 2R, 3R)
    - يقيم قوة الصفقة: strong / mid / weak
    - يطلع tp_profile: TREND_3TP / MID_2TP / SCALP_STRICT
    """

    def __init__(self):
        self.atr_sl_mult = 1.2
        self.min_candles = 30

    def _calc_basic_atr(self, df, period=14):
        if len(df) < period + 2:
            return None
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        close = df["close"].astype(float)

        tr1 = (high - low).abs()
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low  - close.shift(1)).abs()
        tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        return float(atr) if not math.isnan(atr) else None

    def compute_setup(self, df: pd.DataFrame, side: str, trend_info: dict, smc_ctx: dict = None):
        """
        side: "BUY" / "SELL"
        trend_info: من TrendAnalyzer أو من مؤشرات ADX/DI في البوت
        smc_ctx: ممكن تديها SMC/Golden/Liquidity لو حابب
        """
        if df is None or len(df) < self.min_candles:
            return {"valid": False, "reason": "not_enough_data"}

        side = side.upper()
        close = df["close"].astype(float)
        price = float(close.iloc[-1])

        atr = trend_info.get("atr") or self._calc_basic_atr(df)
        if not atr or atr <= 0:
            return {"valid": False, "reason": "atr_invalid"}

        # ------------------------------------------------
        # بناء SL على أساس ATR
        # ------------------------------------------------
        if side == "BUY":
            sl = price - self.atr_sl_mult * atr
            direction = 1
        else:
            sl = price + self.atr_sl_mult * atr
            direction = -1

        risk = abs(price - sl)
        if risk <= 0:
            return {"valid": False, "reason": "risk_zero"}

        # أهداف 1R / 2R / 3R
        tp1 = price + direction * risk
        tp2 = price + direction * risk * 2.0
        tp3 = price + direction * risk * 3.0

        # ------------------------------------------------
        # بناء Edge Score من الترند + SMC + Golden
        # ------------------------------------------------
        adx       = float(trend_info.get("adx", 0.0) or 0.0)
        strength  = float(trend_info.get("strength", 0.0) or 0.0)
        is_strong = bool(trend_info.get("is_strong", False))

        edge_score = 0.0
        tags = []

        # ترند + ADX
        if is_strong and adx >= 25:
            edge_score += 2.0
            tags.append("strong_trend")
        elif adx >= 20:
            edge_score += 1.0
            tags.append("trend_adx")

        if strength >= 0.7:
            edge_score += 1.0
            tags.append("strong_ma_momentum")
        elif strength >= 0.4:
            edge_score += 0.5
            tags.append("mid_ma_momentum")

        # لو عندك SMC/GZ تقدر تزود هنا
        if smc_ctx:
            if smc_ctx.get("demand_box") and side == "BUY":
                edge_score += 1.0
                tags.append("demand_box")
            if smc_ctx.get("supply_box") and side == "SELL":
                edge_score += 1.0
                tags.append("supply_box")
            if smc_ctx.get("liquidity_sweep"):
                edge_score += 0.5
                tags.append("liq_sweep")

        # ------------------------------------------------
        # تصنيف الصفقة: strong / mid / weak
        # ------------------------------------------------
        if edge_score >= 3.5 and adx >= 25:
            grade = "strong"
            tp_profile = "TREND_3TP"
        elif edge_score >= 2.0:
            grade = "mid"
            tp_profile = "MID_2TP"
        else:
            grade = "weak"
            tp_profile = "SCALP_STRICT"

        rr1 = abs(tp1 - price) / risk
        rr2 = abs(tp2 - price) / risk
        rr3 = abs(tp3 - price) / risk

        return {
            "valid": True,
            "side": side,
            "entry_price": price,
            "sl_price": float(sl),
            "tp1": float(tp1),
            "tp2": float(tp2),
            "tp3": float(tp3),
            "rr1": float(rr1),
            "rr2": float(rr2),
            "rr3": float(rr3),
            "edge_score": float(edge_score),
            "grade": grade,          # strong / mid / weak
            "tp_profile": tp_profile,  # TREND_3TP / MID_2TP / SCALP_STRICT
            "tags": tags,
            "adx": adx,
            "trend_strength": strength,
            "is_strong_trend": is_strong,
        }

# إنشاء نسخة عالمية من EdgeAlgoEngine
EDGE_ENGINE = EdgeAlgoEngine()

# =================== ADVANCED SMC ENGINE WITH ADX+ATR STOP-HUNT FILTERS ===================

class SMCCoreEngine:
    def __init__(self):
        self.demand_zones = []
        self.supply_zones = []
        self.liquidity_pools = []
        self.valid_fvgs = []
        self.order_blocks = []
        self.breaker_blocks = []
        self.spring_patterns = []
        
    def analyze_smc_structure(self, df):
        """تحليل هيكل السوق باستخدام SMC مع فلتر ADX+ATR لـ Stop-Hunt"""
        analysis = {
            'market_structure': self._analyze_market_structure(df),
            'liquidity_levels': self._find_liquidity_pools(df),
            'supply_demand_zones': self._identify_supply_demand_zones(df),
            'order_blocks': self._find_order_blocks(df),
            'fair_value_gaps': self._analyze_fvg(df),
            'breaker_spring': self._analyze_breaker_spring(df),
            'trading_opportunities': []
        }
        
        # توليد فرص التداول مع تطبيق فلتر ADX+ATR
        analysis['trading_opportunities'] = self._generate_trading_opportunities(analysis, df)
        
        return analysis
    
    def _analyze_market_structure(self, df):
        """تحليل هيكل السوق: BOS, CHoCH, Displacement"""
        structure = {
            'bos': self._find_break_of_structure(df),
            'choch': self._find_change_of_character(df),
            'displacement': self._find_displacement_moves(df),
            'trend': self._determine_trend_structure(df)
        }
        return structure
    
    def _find_liquidity_pools(self, df):
        """اكتشاف تجمعات السيولة مع فلتر ADX+ATR"""
        liquidity_pools = []
        
        # Equal Highs / Equal Lows
        equal_highs = self._find_equal_highs(df)
        equal_lows = self._find_equal_lows(df)
        
        # Engineering Liquidity (مصائد سيولة مصممة)
        engineered_liquidity = self._find_engineered_liquidity(df)
        
        # Stop Hunt Zones مع فلتر ADX+ATR
        stop_hunt_zones = self._detect_stop_hunt_zones_with_filters(df)
        
        liquidity_pools.extend(equal_highs)
        liquidity_pools.extend(equal_lows)
        liquidity_pools.extend(engineered_liquidity)
        liquidity_pools.extend(stop_hunt_zones)
        
        return liquidity_pools
    
    def _find_equal_highs(self, df, tolerance=0.001):
        """إيجاد القمم المتساوية"""
        equal_highs = []
        highs = df['high'].astype(float)
        
        for i in range(10, len(highs)-5):
            current_high = highs.iloc[i]
            # البحث عن قمم متساوية في النطاق
            lookback_highs = highs.iloc[i-10:i]
            similar_highs = lookback_highs[abs(lookback_highs - current_high) / current_high <= tolerance]
            
            if len(similar_highs) >= 2:
                equal_highs.append({
                    'price': current_high,
                    'time': df['time'].iloc[i],
                    'count': len(similar_highs),
                    'type': 'equal_highs'
                })
        
        return equal_highs
    
    def _find_equal_lows(self, df, tolerance=0.001):
        """إيجاد القيعان المتساوية"""
        equal_lows = []
        lows = df['low'].astype(float)
        
        for i in range(10, len(lows)-5):
            current_low = lows.iloc[i]
            # البحث عن قيعان متساوية في النطاق
            lookback_lows = lows.iloc[i-10:i]
            similar_lows = lookback_lows[abs(lookback_lows - current_low) / current_low <= tolerance]
            
            if len(similar_lows) >= 2:
                equal_lows.append({
                    'price': current_low,
                    'time': df['time'].iloc[i],
                    'count': len(similar_lows),
                    'type': 'equal_lows'
                })
        
        return equal_lows
    
    def _identify_supply_demand_zones(self, df):
        """تحديد مناطق الطلب والعرض"""
        zones = []
        
        # البحث عن القمم والقيعان المحورية
        pivot_highs = self._find_pivot_highs(df, left_bars=3, right_bars=3)
        pivot_lows = self._find_pivot_lows(df, left_bars=3, right_bars=3)
        
        # مناطق العرض عند القمم المحورية
        for pivot in pivot_highs:
            zone = {
                'type': 'supply',
                'price_level': pivot['price'],
                'time': pivot['time'],
                'strength': self._calculate_zone_strength(df, pivot, 'supply'),
                'reason': 'Pivot High Supply Zone'
            }
            zones.append(zone)
        
        # مناطق الطلب عند القيعان المحورية
        for pivot in pivot_lows:
            zone = {
                'type': 'demand',
                'price_level': pivot['price'],
                'time': pivot['time'],
                'strength': self._calculate_zone_strength(df, pivot, 'demand'),
                'reason': 'Pivot Low Demand Zone'
            }
            zones.append(zone)
        
        return zones
    
    def _find_pivot_highs(self, df, left_bars=3, right_bars=3):
        """إيجاد القمم المحورية"""
        pivot_highs = []
        highs = df['high'].astype(float)
        
        for i in range(left_bars, len(highs)-right_bars):
            current_high = highs.iloc[i]
            left_range = highs.iloc[i-left_bars:i]
            right_range = highs.iloc[i+1:i+right_bars+1]
            
            if (current_high > left_range.max() and 
                current_high > right_range.max()):
                pivot_highs.append({
                    'price': current_high,
                    'time': df['time'].iloc[i],
                    'strength': len(left_range) + len(right_range)
                })
        
        return pivot_highs
    
    def _find_pivot_lows(self, df, left_bars=3, right_bars=3):
        """إيجاد القيعان المحورية"""
        pivot_lows = []
        lows = df['low'].astype(float)
        
        for i in range(left_bars, len(lows)-right_bars):
            current_low = lows.iloc[i]
            left_range = lows.iloc[i-left_bars:i]
            right_range = lows.iloc[i+1:i+right_bars+1]
            
            if (current_low < left_range.min() and 
                current_low < right_range.min()):
                pivot_lows.append({
                    'price': current_low,
                    'time': df['time'].iloc[i],
                    'strength': len(left_range) + len(right_range)
                })
        
        return pivot_lows
    
    def _calculate_zone_strength(self, df, pivot, zone_type):
        """حساب قوة المنطقة"""
        strength = 0.0
        
        # قوة الحركة من المنطقة
        move_strength = self._measure_move_strength(df, pivot, zone_type)
        strength += move_strength * 0.4
        
        # عدد مرات احترام المنطقة
        respect_count = self._count_zone_respect(df, pivot, zone_type)
        strength += respect_count * 0.3
        
        # الوقت منذ تشكل المنطقة
        time_factor = self._calculate_time_factor(df, pivot)
        strength += time_factor * 0.3
        
        return min(1.0, strength)
    
    def _find_order_blocks(self, df):
        """إيجاد Order Blocks"""
        order_blocks = []
        
        for i in range(2, len(df)-1):
            # Order Block Bullish: شمعة هابطة كبيرة يليها شمعة صاعدة
            if (self._is_bearish_candle(df, i) and 
                self._is_bullish_candle(df, i+1)):
                ob = {
                    'type': 'bullish_ob',
                    'high': max(float(df['high'].iloc[i]), float(df['high'].iloc[i+1])),
                    'low': min(float(df['low'].iloc[i]), float(df['low'].iloc[i+1])),
                    'time': df['time'].iloc[i],
                    'strength': self._calculate_ob_strength(df, i)
                }
                order_blocks.append(ob)
            
            # Order Block Bearish: شمعة صاعدة كبيرة يليها شمعة هابطة  
            if (self._is_bullish_candle(df, i) and 
                self._is_bearish_candle(df, i+1)):
                ob = {
                    'type': 'bearish_ob',
                    'high': max(float(df['high'].iloc[i]), float(df['high'].iloc[i+1])),
                    'low': min(float(df['low'].iloc[i]), float(df['low'].iloc[i+1])),
                    'time': df['time'].iloc[i],
                    'strength': self._calculate_ob_strength(df, i)
                }
                order_blocks.append(ob)
        
        return order_blocks
    
    def _analyze_fvg(self, df):
        """تحليل Fair Value Gaps"""
        fvgs = []
        
        for i in range(2, len(df)-2):
            fvg = self._identify_fvg(df, i)
            if fvg and fvg['valid']:
                fvg['probability'] = self._calculate_fvg_probability(fvg, df, i)
                fvgs.append(fvg)
        
        return fvgs
    
    def _identify_fvg(self, df, index):
        """تحديد FVG"""
        if index < 1 or index >= len(df)-1:
            return None
        
        prev_candle = df.iloc[index-1]
        curr_candle = df.iloc[index]
        next_candle = df.iloc[index+1]
        
        # FVG هابطة
        if (float(curr_candle['low']) > float(prev_candle['high']) and
            float(next_candle['high']) < float(curr_candle['low'])):
            
            return {
                'type': 'fvg_bearish',
                'high': float(curr_candle['low']),
                'low': float(prev_candle['high']),
                'time': curr_candle['time'],
                'valid': self._is_valid_fvg(df, index, 'bearish')
            }
        
        # FVG صاعدة
        if (float(curr_candle['high']) < float(prev_candle['low']) and
            float(next_candle['low']) > float(curr_candle['high'])):
            
            return {
                'type': 'fvg_bullish',
                'high': float(prev_candle['low']),
                'low': float(curr_candle['high']),
                'time': curr_candle['time'],
                'valid': self._is_valid_fvg(df, index, 'bullish')
            }
        
        return None
    
    def _is_valid_fvg(self, df, index, fvg_type):
        """التحقق من صحة FVG"""
        curr_candle = df.iloc[index]
        next_candle = df.iloc[index+1]
        
        if fvg_type == 'bearish':
            # لا يجب أن تلمس فتائل الشمعة التالية المنطقة
            return (float(next_candle['high']) < float(curr_candle['low']) and
                    float(next_candle['low']) > float(df.iloc[index-1]['high']))
        else:
            # لا يجب أن تلمس فتائل الشمعة التالية المنطقة
            return (float(next_candle['low']) > float(curr_candle['high']) and
                    float(next_candle['high']) < float(df.iloc[index-1]['low']))
    
    def _analyze_breaker_spring(self, df):
        """تحليل نماذج Breaker Blocks و Spring"""
        patterns = {
            'breaker_blocks': self._find_breaker_blocks(df),
            'spring_patterns': self._find_spring_patterns(df)
        }
        return patterns
    
    def _find_breaker_blocks(self, df):
        """إيجاد Breaker Blocks"""
        breakers = []
        
        for i in range(3, len(df)-2):
            breaker = self._identify_breaker_block(df, i)
            if breaker:
                breakers.append(breaker)
        
        return breakers
    
    def _identify_breaker_block(self, df, index):
        """تحديد Breaker Block"""
        if index < 2 or index >= len(df)-1:
            return None
        
        # Breaker Block هابط: كسر قمة فاشل
        if (float(df['high'].iloc[index]) > float(df['high'].iloc[index-2]) and  # كسر قمة
            float(df['close'].iloc[index]) < float(df['open'].iloc[index]) and   # شمعة هابطة
            float(df['close'].iloc[index+1]) < float(df['close'].iloc[index])):  # تأكيد هبوط
            
            return {
                'type': 'breaker_bearish',
                'level': float(df['high'].iloc[index]),
                'time': df['time'].iloc[index],
                'strength': 0.8
            }
        
        # Breaker Block صاعد: كسر قاع فاشل
        if (float(df['low'].iloc[index]) < float(df['low'].iloc[index-2]) and    # كسر قاع
            float(df['close'].iloc[index]) > float(df['open'].iloc[index]) and   # شمعة صاعدة
            float(df['close'].iloc[index+1]) > float(df['close'].iloc[index])):  # تأكيد صعود
            
            return {
                'type': 'breaker_bullish',
                'level': float(df['low'].iloc[index]),
                'time': df['time'].iloc[index],
                'strength': 0.8
            }
        
        return None
    
    def _find_spring_patterns(self, df):
        """كشف نماذج Spring (الكسر الوهمي)"""
        springs = []
        
        for i in range(5, len(df)-3):
            spring = self._identify_spring(df, i)
            if spring:
                springs.append(spring)
        
        return springs
    
    def _identify_spring(self, df, index):
        """تحديد نموذج Spring"""
        if index < 3 or index >= len(df)-2:
            return None
        
        # البحث عن قاع سابق
        prev_low = float(df['low'].iloc[index-3:index].min())
        current_low = float(df['low'].iloc[index])
        
        # شرط 1: كسر القاع
        if current_low >= prev_low * 0.998:  # ليس كسر حقيقي
            return None
        
        # شرط 2: العودة السريعة
        next_close = float(df['close'].iloc[index+1])
        if next_close <= prev_low:
            return None
        
        # شرط 3: شمعة تأكيد
        confirm_candle = self._is_bullish_confirmation(df, index+1)
        if not confirm_candle:
            return None
        
        return {
            'type': 'spring_bullish',
            'break_level': current_low,
            'return_level': prev_low,
            'time': df['time'].iloc[index],
            'strength': 0.85
        }
    
    def _generate_trading_opportunities(self, analysis, df):
        """توليد فرص التداول بناءً على تحليل SMC مع فلتر ADX+ATR"""
        opportunities = []
        current_price = float(df['close'].iloc[-1])
        
        # فرص من مناطق الطلب
        for zone in analysis['supply_demand_zones']:
            if zone['type'] == 'demand' and self._is_zone_active(zone, current_price):
                opportunities.append({
                    'type': 'demand_zone_entry',
                    'direction': 'long',
                    'zone': zone,
                    'confidence': zone['strength']
                })
        
        # فرص من مناطق العرض
        for zone in analysis['supply_demand_zones']:
            if zone['type'] == 'supply' and self._is_zone_active(zone, current_price):
                opportunities.append({
                    'type': 'supply_zone_entry',
                    'direction': 'short',
                    'zone': zone,
                    'confidence': zone['strength']
                })
        
        # فرص من Order Blocks
        for ob in analysis['order_blocks']:
            if (ob['low'] <= current_price <= ob['high'] and 
                ob['strength'] >= OB_STRENGTH_THRESHOLD):
                direction = 'long' if ob['type'] == 'bullish_ob' else 'short'
                opportunities.append({
                    'type': 'order_block_entry',
                    'direction': direction,
                    'ob': ob,
                    'confidence': ob['strength']
                })
        
        # فرص من Stop-Hunt مع فلتر ADX+ATR
        for trap in analysis.get('liquidity_levels', []):
            if trap.get('type') in ['stop_hunt_bull', 'stop_hunt_bear']:
                # تطبيق فلتر ADX+ATR المتقدم
                if self._is_valid_stop_hunt_with_filters(trap, df):
                    if trap['type'] == 'stop_hunt_bull':
                        opportunities.append({
                            'type': 'stop_hunt_reversal',
                            'direction': 'short',  # دخول عكسي بعد سحب استوبات الشراء
                            'trap': trap,
                            'confidence': trap.get('strength', 0.5) * 1.5,  # تعزيز الثقة
                            'stop_loss': self._calculate_stop_loss_for_trap(trap, 'short', df),
                            'filters_applied': True
                        })
                    else:  # stop_hunt_bear
                        opportunities.append({
                            'type': 'stop_hunt_reversal',
                            'direction': 'long',  # دخول عكسي بعد سحب استوبات البيع
                            'trap': trap,
                            'confidence': trap.get('strength', 0.5) * 1.5,  # تعزيز الثقة
                            'stop_loss': self._calculate_stop_loss_for_trap(trap, 'long', df),
                            'filters_applied': True
                        })
    
        return opportunities
    
    # ========== STOP-HUNT FILTERS WITH ADX+ATR ==========
    
    def _detect_stop_hunt_zones_with_filters(self, df):
        """كشف مناطق صيد الوقفيات مع فلتر ADX+ATR"""
        zones = []
        
        if len(df) < 20:
            return zones
        
        # حساب مؤشرات ADX و ATR
        ind = compute_indicators(df)
        current_adx = ind.get('adx', 0)
        current_atr = ind.get('atr', 0)
        
        # حساب متوسط ATR (20 شمعة)
        if len(df) >= 20:
            highs = df['high'].astype(float).tail(20)
            lows = df['low'].astype(float).tail(20)
            atr_base = (highs.max() - lows.min()) / 20
        else:
            atr_base = current_atr
        
        atr_mult = current_atr / atr_base if atr_base > 0 else 1.0
        
        for i in range(5, len(df) - 3):
            # فحص حركة Stop-Hunt لأعلى (سحب استوبات الشراء)
            if self._is_bull_stop_hunt(df, i):
                # فلتر ADX: ممنوع في ترند قوي جداً
                if current_adx > STOP_HUNT_ADX_MAX:
                    continue
                    
                # فلتر ADX: ممنوع في ترند ضعيف جداً
                if current_adx < STOP_HUNT_ADX_MIN:
                    continue
                    
                # فلتر ATR: الحركة يجب أن تكون أكبر من المعتاد
                if atr_mult < STOP_HUNT_ATR_MULT_MIN:
                    continue
                    
                # فلتر ATR: لا تكون الحركة عنيفة جداً
                if atr_mult > 2.5:
                    continue
                    
                zones.append({
                    'type': 'stop_hunt_bull',
                    'level': float(df['high'].iloc[i]),
                    'time': df['time'].iloc[i],
                    'adx': current_adx,
                    'atr_mult': atr_mult,
                    'strength': self._calculate_stop_hunt_strength(df, i, 'bull')
                })
            
            # فحص حركة Stop-Hunt لأسفل (سحب استوبات البيع)
            if self._is_bear_stop_hunt(df, i):
                # فلتر ADX: ممنوع في ترند قوي جداً
                if current_adx > STOP_HUNT_ADX_MAX:
                    continue
                    
                # فلتر ADX: ممنوع في ترند ضعيف جداً
                if current_adx < STOP_HUNT_ADX_MIN:
                    continue
                    
                # فلتر ATR: الحركة يجب أن تكون أكبر من المعتاد
                if atr_mult < STOP_HUNT_ATR_MULT_MIN:
                    continue
                    
                # فلتر ATR: لا تكون الحركة عنيفة جداً
                if atr_mult > 2.5:
                    continue
                    
                zones.append({
                    'type': 'stop_hunt_bear',
                    'level': float(df['low'].iloc[i]),
                    'time': df['time'].iloc[i],
                    'adx': current_adx,
                    'atr_mult': atr_mult,
                    'strength': self._calculate_stop_hunt_strength(df, i, 'bear')
                })
        
        return zones
    
    def _is_bull_stop_hunt(self, df, index):
        """تحديد حركة Stop-Hunt لأعلى (سحب استوبات الشراء)"""
        if index < 2 or index >= len(df) - 2:
            return False
        
        # الشمعة الحالية
        h = float(df['high'].iloc[index])
        l = float(df['low'].iloc[index])
        c = float(df['close'].iloc[index])
        o = float(df['open'].iloc[index])
        
        # البحث عن قمة سابقة في آخر 10 شموع
        lookback_start = max(0, index - 10)
        lookback = df['high'].iloc[lookback_start:index]
        if len(lookback) < 3:
            return False
        
        prev_high = float(lookback.max())
        
        # شرط 1: كسر القمة السابقة
        if h <= prev_high:
            return False
        
        # شرط 2: العودة السريعة تحت القمة
        next_close = float(df['close'].iloc[index + 1])
        if next_close >= prev_high:
            return False
        
        # شرط 3: شمعة ذات ذيل علوي كبير (فتيلة)
        candle_range = h - l
        if candle_range == 0:
            return False
            
        upper_wick = h - max(o, c)
        if upper_wick < STOP_HUNT_WICK_RATIO * candle_range:
            return False
        
        # شرط 4: المسافة فوق القمة السابقة لا تقل عن 0.5 * ATR
        current_atr = abs(h - l)  # تقدير ATR
        if (h - prev_high) < STOP_HUNT_DISTANCE_ATR * current_atr:
            return False
        
        return True
    
    def _is_bear_stop_hunt(self, df, index):
        """تحديد حركة Stop-Hunt لأسفل (سحب استوبات البيع)"""
        if index < 2 or index >= len(df) - 2:
            return False
        
        # الشمعة الحالية
        h = float(df['high'].iloc[index])
        l = float(df['low'].iloc[index])
        c = float(df['close'].iloc[index])
        o = float(df['open'].iloc[index])
        
        # البحث عن قاع سابق في آخر 10 شموع
        lookback_start = max(0, index - 10)
        lookback = df['low'].iloc[lookback_start:index]
        if len(lookback) < 3:
            return False
        
        prev_low = float(lookback.min())
        
        # شرط 1: كسر القاع السابق
        if l >= prev_low:
            return False
        
        # شرط 2: العودة السريعة فوق القاع
        next_close = float(df['close'].iloc[index + 1])
        if next_close <= prev_low:
            return False
        
        # شرط 3: شمعة ذات ذيل سفلي كبير (فتيلة)
        candle_range = h - l
        if candle_range == 0:
            return False
            
        lower_wick = min(o, c) - l
        if lower_wick < STOP_HUNT_WICK_RATIO * candle_range:
            return False
        
        # شرط 4: المسافة تحت القاع السابق لا تقل عن 0.5 * ATR
        current_atr = abs(h - l)  # تقدير ATR
        if (prev_low - l) < STOP_HUNT_DISTANCE_ATR * current_atr:
            return False
        
        return True
    
    def _calculate_stop_hunt_strength(self, df, index, hunt_type):
        """حساب قوة حركة Stop-Hunt مع فلتر ADX+ATR"""
        strength = 0.0
        
        # حساب المؤشرات
        ind = compute_indicators(df)
        adx = ind.get('adx', 0)
        
        # قوة ADX
        if STOP_HUNT_ADX_MIN <= adx <= STOP_HUNT_ADX_MAX:
            strength += 0.4  # ADX في النطاق المثالي
        elif adx < STOP_HUNT_ADX_MIN:
            strength += 0.2  # ADX منخفض (سوق فلات)
        else:
            strength += 0.1  # ADX مرتفع جداً (خطير)
        
        # قوة ATR
        current_atr = ind.get('atr', 0)
        if len(df) >= 20:
            highs = df['high'].astype(float).tail(20)
            lows = df['low'].astype(float).tail(20)
            atr_base = (highs.max() - lows.min()) / 20
        else:
            atr_base = current_atr
        
        atr_mult = current_atr / atr_base if atr_base > 0 else 1.0
        if STOP_HUNT_ATR_MULT_MIN <= atr_mult <= 2.5:
            strength += 0.3
        elif atr_mult > 2.5:
            strength += 0.1  # حركة عنيفة جداً
        else:
            strength += 0.0  # حركة ضعيفة
        
        # قوة الحركة (الذيل/الفتيلة)
        if hunt_type == 'bull':
            h = float(df['high'].iloc[index])
            l = float(df['low'].iloc[index])
            c = float(df['close'].iloc[index])
            o = float(df['open'].iloc[index])
            
            candle_range = h - l
            if candle_range > 0:
                upper_wick = h - max(o, c)
                wick_ratio = upper_wick / candle_range
                if wick_ratio >= STOP_HUNT_WICK_RATIO:
                    strength += 0.3
                elif wick_ratio >= 0.4:
                    strength += 0.2
                else:
                    strength += 0.1
        else:  # bear
            h = float(df['high'].iloc[index])
            l = float(df['low'].iloc[index])
            c = float(df['close'].iloc[index])
            o = float(df['open'].iloc[index])
            
            candle_range = h - l
            if candle_range > 0:
                lower_wick = min(o, c) - l
                wick_ratio = lower_wick / candle_range
                if wick_ratio >= STOP_HUNT_WICK_RATIO:
                    strength += 0.3
                elif wick_ratio >= 0.4:
                    strength += 0.2
                else:
                    strength += 0.1
        
        return min(1.0, strength)
    
    def _is_valid_stop_hunt_with_filters(self, trap, df):
        """التحقق من صحة Stop-Hunt مع فلتر ADX+ATR"""
        # فحص ADX
        ind = compute_indicators(df)
        adx = ind.get('adx', 0)
        
        # شرط 1: ADX لا يكون في منطقة الترند القوي المجنون
        if adx > STOP_HUNT_ADX_MAX:
            return False
        
        # شرط 2: ADX لا يكون منخفض جداً (سوق فلات بلا اتجاه)
        if adx < STOP_HUNT_ADX_MIN:
            return False
        
        # شرط 3: فحص ميل ADX (ADX Slope)
        adx_slope = self._calculate_adx_slope(df)
        
        # إذا كان ADX في ارتفاع مستمر → ترند قوي → تجنب الدخول العكسي
        if adx_slope > 0.5:  # ميل إيجابي قوي
            return False
        
        # شرط 4: فحص ATR
        current_atr = ind.get('atr', 0)
        if len(df) >= 20:
            highs = df['high'].astype(float).tail(20)
            lows = df['low'].astype(float).tail(20)
            atr_base = (highs.max() - lows.min()) / 20
        else:
            atr_base = current_atr
        
        atr_mult = current_atr / atr_base if atr_base > 0 else 1.0
        
        # ATR يجب أن يكون فوق الحد الأدنى
        if atr_mult < STOP_HUNT_ATR_MULT_MIN:
            return False
        
        # ATR لا يكون كبير جداً (حركة عنيفة)
        if atr_mult > 2.5:
            return False
        
        return True
    
    def _calculate_adx_slope(self, df):
        """حساب ميل ADX"""
        if len(df) < 10:
            return 0.0
        
        # حساب ADX للشموع الأخيرة
        adx_values = []
        for i in range(min(5, len(df))):
            if i == 0:
                df_slice = df
            else:
                df_slice = df.iloc[:-(i)]
            ind = compute_indicators(df_slice)
            adx_values.append(ind.get('adx', 0))
        
        if len(adx_values) < 2:
            return 0.0
        
        # حساب الميل (التغير في ADX)
        slope = adx_values[0] - adx_values[-1]
        return slope
    
    def _calculate_stop_loss_for_trap(self, trap, direction, df):
        """حساب Stop Loss بناءً على ATR لـ Stop-Hunt"""
        ind = compute_indicators(df)
        atr = ind.get('atr', 0)
        
        if direction == 'short':
            # للصفقات البيعية: SL فوق ذيل الضربة
            sl_price = trap['level'] + (STOP_HUNT_SL_ATR_MULT * atr)
        else:
            # للصفقات الشرائية: SL تحت ذيل الضربة
            sl_price = trap['level'] - (STOP_HUNT_SL_ATR_MULT * atr)
        
        return sl_price
    
    # ========== Helper Methods ==========
    
    def _is_bearish_candle(self, df, index):
        o, c = float(df['open'].iloc[index]), float(df['close'].iloc[index])
        return c < o
    
    def _is_bullish_candle(self, df, index):
        o, c = float(df['open'].iloc[index]), float(df['close'].iloc[index])
        return c > o
    
    def _is_bullish_confirmation(self, df, index):
        if index >= len(df):
            return False
        return self._is_bullish_candle(df, index)
    
    def _calculate_ob_strength(self, df, index):
        """حساب قوة Order Block"""
        strength = 0.0
        
        # حجم الشمعة
        candle_size = abs(float(df['close'].iloc[index]) - float(df['open'].iloc[index]))
        avg_size = abs(df['close'].astype(float) - df['open'].astype(float)).tail(20).mean()
        
        if candle_size > avg_size * 1.5:
            strength += 0.4
        
        # حجم التداول
        volume = float(df['volume'].iloc[index])
        avg_volume = df['volume'].astype(float).tail(20).mean()
        
        if volume > avg_volume * 1.2:
            strength += 0.3
        
        # قوة الحركة التالية
        if index < len(df) - 2:
            next_move = abs(float(df['close'].iloc[index+2]) - float(df['close'].iloc[index+1]))
            if next_move > candle_size:
                strength += 0.3
        
        return strength
    
    def _calculate_fvg_probability(self, fvg, df, index):
        """حساب احتمالية FVG"""
        probability = 0.0
        
        # نقاء الفجوة
        if fvg['valid']:
            probability += 0.4
        
        # وجود Displacement
        if self._has_displacement(df, index, fvg['type']):
            probability += 0.3
        
        # حجم التداول
        if self._has_volume_confirmation(df, index):
            probability += 0.2
        
        # توافق مع الترند
        if self._is_trend_aligned(df, index, fvg['type']):
            probability += 0.1
        
        return probability
    
    def _has_displacement(self, df, index, fvg_type):
        """التحقق من وجود Displacement"""
        if index < 2:
            return False
        
        if fvg_type == 'bearish':
            # حركة هابطة قوية قبل الفجوة
            move = float(df['close'].iloc[index-1]) - float(df['open'].iloc[index-1])
            return move < 0 and abs(move) > df['close'].astype(float).diff().abs().tail(10).mean()
        else:
            # حركة صاعدة قوية قبل الفجوة
            move = float(df['close'].iloc[index-1]) - float(df['open'].iloc[index-1])
            return move > 0 and abs(move) > df['close'].astype(float).diff().abs().tail(10).mean()
    
    def _has_volume_confirmation(self, df, index):
        """التحقق من تأكيد الحجم"""
        if index >= len(df):
            return False
        volume = float(df['volume'].iloc[index])
        avg_volume = df['volume'].astype(float).tail(20).mean()
        return volume > avg_volume * 1.2
    
    def _is_trend_aligned(self, df, index, fvg_type):
        """التحقق من توافق FVG مع الترند"""
        if index < 5:
            return False
        
        # اتجاه بسيط بناءً على المتوسط المتحرك
        short_ma = df['close'].astype(float).tail(5).mean()
        long_ma = df['close'].astype(float).tail(20).mean()
        
        if fvg_type == 'bullish':
            return short_ma > long_ma  # ترند صاعد
        else:
            return short_ma < long_ma  # ترند هابط
    
    def _is_zone_active(self, zone, current_price):
        """التحقق إذا كانت المنطقة نشطة"""
        zone_price = zone['price_level']
        tolerance = zone_price * 0.002  # 0.2% tolerance
        return abs(current_price - zone_price) <= tolerance
    
    def _find_break_of_structure(self, df):
        """إيجاد Break of Structure"""
        return {"status": "analyzing", "direction": None}
    
    def _find_change_of_character(self, df):
        """إيجاد Change of Character"""
        return {"status": "analyzing", "direction": None}
    
    def _find_displacement_moves(self, df):
        """إيجاد حركات Displacement"""
        return []
    
    def _determine_trend_structure(self, df):
        """تحديد هيكل الترند"""
        return "analyzing"
    
    def _find_engineered_liquidity(self, df):
        """إيجاد السيولة المصممة"""
        return []
    
    def _measure_move_strength(self, df, pivot, zone_type):
        """قياس قوة الحركة من المنطقة"""
        return 0.5
    
    def _count_zone_respect(self, df, pivot, zone_type):
        """عد مرات احترام المنطقة"""
        return 0.3
    
    def _calculate_time_factor(self, df, pivot):
        """حساب عامل الوقت"""
        return 0.2

# =================== ENHANCED CANDLES MODULE WITH SMC ===================
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

# ========= SMC CANDLES PATTERNS =========
def _smc_breakaway(o,c,h,l,po,pc,ph,pl):
    """Breakaway pattern - strong continuation"""
    bull_break = (c>o) and (po>pc) and (c>ph) and (l>pl)
    bear_break = (c<o) and (po<pc) and (c<pl) and (h<ph)
    return bull_break, bear_break

def _smc_absorption(po,pc,o,c,h,l,v,pv):
    """Absorption pattern - smart money accumulation/distribution"""
    bull_abs = (pc<po) and (c>o) and (c>po) and (v>pv*1.5)
    bear_abs = (pc>po) and (c<o) and (c<po) and (v>pv*1.5)
    return bull_abs, bear_abs

def _liquidity_grab(o,c,h,l,po,pc,pl,ph):
    """Liquidity grab pattern - stop hunting"""
    bull_grab = (c<o) and (l<pl) and (c>pc)  # false breakdown
    bear_grab = (c>o) and (h>ph) and (c<pc)  # false breakout
    return bull_grab, bear_grab

def compute_enhanced_candles(df):
    """
    إرجاع: إشارات شراء/بيع معقّدة + أنماط SMC + فخاخ سيولة
    """
    if len(df) < 6:
        return {"buy":False,"sell":False,"score_buy":0.0,"score_sell":0.0,
                "wick_up_big":False,"wick_dn_big":False,"doji":False,
                "pattern":None, "smc_pattern":None, "liquidity_trap":False}

    # Current and previous candles
    o1,h1,l1,c1,v1 = float(df["open"].iloc[-2]), float(df["high"].iloc[-2]), float(df["low"].iloc[-2]), float(df["close"].iloc[-2]), float(df["volume"].iloc[-2])
    o0,h0,l0,c0,v0 = float(df["open"].iloc[-3]), float(df["high"].iloc[-3]), float(df["low"].iloc[-3]), float(df["close"].iloc[-3]), float(df["volume"].iloc[-3])
    o2,h2,l2,c2,v2 = float(df["open"].iloc[-4]), float(df["high"].iloc[-4]), float(df["low"].iloc[-4]), float(df["close"].iloc[-4]), float(df["volume"].iloc[-4])

    strength_b = strength_s = 0.0
    tags = []
    smc_tags = []
    liquidity_trap = False

    # Basic candle patterns
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

    # SMC Patterns
    bull_break, bear_break = _smc_breakaway(o1,c1,h1,l1,o0,c0,h0,l0)
    if bull_break: strength_b += 2.5; smc_tags.append("breakaway_bull")
    if bear_break: strength_s += 2.5; smc_tags.append("breakaway_bear")

    bull_abs, bear_abs = _smc_absorption(o0,c0,o1,c1,h1,l1,v1,v0)
    if bull_abs: strength_b += 3.0; smc_tags.append("absorption_bull")
    if bear_abs: strength_s += 3.0; smc_tags.append("absorption_bear")

    bull_grab, bear_grab = _liquidity_grab(o1,c1,h1,l1,o0,c0,l0,h0)
    if bull_grab: 
        strength_b += 2.0; 
        smc_tags.append("liquidity_grab_bull")
        liquidity_trap = True
    if bear_grab: 
        strength_s += 2.0; 
        smc_tags.append("liquidity_grab_bear")
        liquidity_trap = True

    # فتائل كبيرة = إرهاق
    rng1 = _rng(h1,l1); up = _upper_wick(h1,o1,c1); dn = _lower_wick(l1,o1,c1)
    wick_up_big = (up >= 1.2*_body(o1,c1)) and (up >= 0.4*rng1)
    wick_dn_big = (dn >= 1.2*_body(o1,c1)) and (dn >= 0.4*rng1)

    if is_doji:  # تخفيف ثقة
        strength_b *= 0.8; strength_s *= 0.8

    # دمج أنماط SMC
    all_tags = tags + [f"SMC:{t}" for t in smc_tags]
    pattern_str = ",".join(all_tags) if all_tags else None

    return {
        "buy": strength_b>0, "sell": strength_s>0,
        "score_buy": round(strength_b,2), "score_sell": round(strength_s,2),
        "wick_up_big": bool(wick_up_big), "wick_dn_big": bool(wick_dn_big),
        "doji": bool(is_doji), "pattern": pattern_str,
        "smc_pattern": ",".join(smc_tags) if smc_tags else None,
        "liquidity_trap": liquidity_trap
    }

# =================== FOOTPRINT & VOLUME ANALYSIS ===================
def compute_footprint_metrics(df):
    """
    تحليل تدفق الأوامر والقدم (Footprint)
    """
    if len(df) < FOOTPRINT_WINDOW + 2:
        return {"ok": False, "why": "short_df"}
    
    try:
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        
        # حجم عند السعر (Price Volume)
        price_changes = close.diff()
        up_volume = volume.where(price_changes > 0, 0)
        down_volume = volume.where(price_changes < 0, 0)
        
        # delta = حجم الشراء - حجم البيع
        delta = up_volume - down_volume
        cumulative_delta = delta.cumsum()
        
        # حجم غير عادي
        vol_ma = volume.rolling(FOOTPRINT_WINDOW).mean()
        volume_spike = (volume / vol_ma) > VOLUME_SPIKE_THRESHOLD
        
        # مناطق الامتصاص
        absorption_bull = (delta > 0) & (close < close.shift(1))  # شراء على هبوط
        absorption_bear = (delta < 0) & (close > close.shift(1))  # بيع على صعود
        
        return {
            "ok": True,
            "delta": float(delta.iloc[-1]),
            "cumulative_delta": float(cumulative_delta.iloc[-1]),
            "volume_spike": bool(volume_spike.iloc[-1]),
            "absorption_bull": bool(absorption_bull.iloc[-1]),
            "absorption_bear": bool(absorption_bear.iloc[-1]),
            "delta_trend": "bull" if cumulative_delta.iloc[-1] > cumulative_delta.iloc[-2] else "bear"
        }
    except Exception as e:
        return {"ok": False, "why": str(e)}

# =================== LIQUIDITY TRAP DETECTION ===================
def detect_liquidity_traps(df, current_price):
    """
    كشف فخاخ السيولة ونقاط الوقف (Liquidity Pools)
    """
    if len(df) < 20:
        return {"ok": False, "traps": []}
    
    try:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        
        # نقاط السيولة (Highs/Lows السابقة)
        recent_highs = high.rolling(10).max().dropna()
        recent_lows = low.rolling(10).min().dropna()
        
        traps = []
        
        # فخاخ السيولة فوق (نقاط وقف الشراء)
        for level in recent_highs.unique():
            if abs(current_price - level) / current_price <= DISPLACEMENT_THRESHOLD:
                traps.append({"type": "stop_hunt_bull", "level": level, "distance_pct": abs(current_price - level) / current_price * 100})
        
        # فخاخ السيولة تحت (نقاط وقف البيع)
        for level in recent_lows.unique():
            if abs(current_price - level) / current_price <= DISPLACEMENT_THRESHOLD:
                traps.append({"type": "stop_hunt_bear", "level": level, "distance_pct": abs(current_price - level) / current_price * 100})
        
        return {"ok": True, "traps": traps}
    
    except Exception as e:
        return {"ok": False, "why": str(e), "traps": []}

# ========== CVD DIVERGENCE OSCILLATOR (TradingFinder style) ==========

def _compute_cvd_hist(df: pd.DataFrame, period: int = 21, mode: str = "periodic") -> pd.Series:
    """
    CVD Histogram حسب منطق TradingFinder:
    Buying  = Volume * ((close - low) / (high - low))
    Selling = Volume * ((high - close) / (high - low))
    delta   = Buying - Selling
    Hist    = sum(delta, period) أو EMA(delta, period)
    """
    if len(df) < period + 5:
        return pd.Series([0.0] * len(df), index=df.index)

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    vol = df["volume"].astype(float)

    rng = (high - low).replace(0, np.nan)
    # Buying / Selling بنفس المعادلة الأصلية
    buying = vol * ((close - low) / rng)
    selling = vol * ((high - close) / rng)
    delta = (buying - selling).fillna(0.0)

    if mode.lower().startswith("ema"):
        hist = delta.ewm(span=period, adjust=False).mean()
    else:
        hist = delta.rolling(period).sum()

    return hist.fillna(0.0)


def _pivot_high(series: pd.Series, n: int) -> pd.Series:
    """Pivot High: قمة محلية بمقدار n شموع يمين/شمال."""
    vals = series.values
    res = [False] * len(series)
    for i in range(n, len(series) - n):
        window = vals[i-n:i+n+1]
        if vals[i] == np.max(window):
            res[i] = True
    return pd.Series(res, index=series.index)


def _pivot_low(series: pd.Series, n: int) -> pd.Series:
    """Pivot Low: قاع محلي بمقدار n شموع يمين/شمال."""
    vals = series.values
    res = [False] * len(series)
    for i in range(n, len(series) - n):
        window = vals[i-n:i+n+1]
        if vals[i] == np.min(window):
            res[i] = True
    return pd.Series(res, index=series.index)


def detect_cvd_divergence(
    df: pd.DataFrame,
    n: int = 2,
    period: int = 21,
    mode: str = "periodic",
) -> dict:
    """
    كشف Bullish / Bearish Divergence بين السعر و CVD Hist:
    - Bearish: السعر Higher High و CVD Hist Lower High.
    - Bullish: السعر Lower Low  و CVD Hist Higher Low.
    يرجّع:
      {
        'bull_div': bool,
        'bear_div': bool,
        'last_price_pivots': {...},
        'last_cvd_pivots': {...},
        'hist_last': float
      }
    """
    if len(df) < max(period + 5, n * 4 + 5):
        return {
            "bull_div": False,
            "bear_div": False,
            "hist_last": 0.0,
            "last_price_pivots": None,
            "last_cvd_pivots": None,
        }

    close = df["close"].astype(float)
    hist = _compute_cvd_hist(df, period=period, mode=mode)
    ema50 = close.ewm(span=50, adjust=False).mean()

    # نفس فكرة up_trend / down_trend في Pine
    up_trend = close.shift(n) > ema50
    down_trend = close.shift(n) < ema50

    # Pivots في السعر والهستوجرام
    ph = _pivot_high(close, n) & up_trend
    pl = _pivot_low(close, n) & down_trend
    ch = _pivot_high(hist, n)
    cl = _pivot_low(hist, n)

    # ناخد آخر اتنين pivots
    price_high_idx = list(close[ph].index)
    price_low_idx = list(close[pl].index)
    cvd_high_idx = list(hist[ch].index)
    cvd_low_idx = list(hist[cl].index)

    bear_div = False
    bull_div = False

    # Bearish: HH في السعر + LH في CVD
    if len(price_high_idx) >= 2 and len(cvd_high_idx) >= 2:
        p1, p2 = price_high_idx[-2], price_high_idx[-1]
        h1, h2 = close.loc[p1], close.loc[p2]
        c1, c2 = hist.loc[cvd_high_idx[-2]], hist.loc[cvd_high_idx[-1]]
        if h2 > h1 and c2 < c1:
            bear_div = True

    # Bullish: LL في السعر + HL في CVD
    if len(price_low_idx) >= 2 and len(cvd_low_idx) >= 2:
        p1, p2 = price_low_idx[-2], price_low_idx[-1]
        l1, l2 = close.loc[p1], close.loc[p2]
        c1, c2 = hist.loc[cvd_low_idx[-2]], hist.loc[cvd_low_idx[-1]]
        if l2 < l1 and c2 > c1:
            bull_div = True

    last_price_pivots = {
        "highs": [(i, float(close.loc[i])) for i in price_high_idx[-2:]],
        "lows": [(i, float(close.loc[i])) for i in price_low_idx[-2:]],
    }
    last_cvd_pivots = {
        "highs": [(i, float(hist.loc[i])) for i in cvd_high_idx[-2:]],
        "lows": [(i, float(hist.loc[i])) for i in cvd_low_idx[-2:]],
    }

    return {
        "bull_div": bool(bull_div),
        "bear_div": bool(bear_div),
        "hist_last": float(hist.iloc[-1]),
        "last_price_pivots": last_price_pivots,
        "last_cvd_pivots": last_cvd_pivots,
    }

# =================== EXECUTION VERIFICATION ===================
def verify_execution_environment():
    """التحقق من بيئة التنفيذ عند الإقلاع"""
    print(f"⚙️ EXECUTION ENVIRONMENT", flush=True)
    print(f"🔧 EXECUTE_ORDERS: {EXECUTE_ORDERS} | SHADOW_MODE: {SHADOW_MODE_DASHBOARD} | DRY_RUN: {DRY_RUN}", flush=True)
    print(f"🎯 GOLDEN ENTRY PRO: score={GOLDEN_ENTRY_SCORE} | ADX={GOLDEN_ENTRY_ADX}", flush=True)
    print(f"🛡️ STOP-HUNT FILTERS: ADX={STOP_HUNT_ADX_MIN}-{STOP_HUNT_ADX_MAX} | ATR_MULT≥{STOP_HUNT_ATR_MULT_MIN} | Wick≥{STOP_HUNT_WICK_RATIO*100}%", flush=True)
    print(f"📈 ENHANCED CANDLES: SMC Patterns + Liquidity Traps", flush=True)
    print(f"👣 FOOTPRINT ANALYSIS: Volume spikes + Absorption", flush=True)
    print(f"📊 VWAP STRATEGY: SCALP(near {VWAP_SCALP_BAND_BPS}bps) | TREND(far {VWAP_TREND_BAND_BPS}bps)", flush=True)
    print(f"🧠 ADVANCED SMC: Supply/Demand + Liquidity Sweep + FVG + Elliott Waves + ADX+ATR Filters", flush=True)
    print(f"🏛 ULTRA MARKET STRUCTURE: BOS/CHoCH + FVG + Premium/Discount + Liquidity Grabs", flush=True)
    print(f"📗 REAL RF FILTER: Pine Exact + Live Signals", flush=True)
    print(f"🧠 EDGE ALGO ENGINE: RR Zones + Setup Quality + Dynamic Profit Profiles", flush=True)
    print(f"📊 CVD DIVERGENCE OSCILLATOR: TradingFinder-style divergence detection", flush=True)
    print(f"🔒 ENHANCED ENTRY FILTERS: CVD ADX≥{CVD_ADX_MIN} + Score≥{CVD_SCORE_MIN} | SMC Confidence≥{SMC_CONFIDENCE_MIN} | Signal Strength≥{SIGNAL_STRENGTH_MIN}", flush=True)
    print(f"🧠 SMART SIGNAL/PROFIT SYSTEM: RF Master + HOLD-TP Logic + Edge Zones", flush=True)
    
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

# =================== GOLDEN ZONE PRO ANALYSIS ===================
def golden_zone_pro_analysis(df, current_price):
    """
    تحليل متقدم للمناطق الذهبية مع تأكيدات متعددة
    """
    if len(df) < 30:
        return {"ok": False, "score": 0.0, "zone": None, "reasons": ["short_df"], "confirmed": False}
    
    try:
        h = df['high'].astype(float)
        l = df['low'].astype(float)
        c = df['close'].astype(float)
        v = df['volume'].astype(float)
        
        # مستويات فيبوناتشي المتقدمة
        swing_hi = h.rolling(15).max().iloc[-1]
        swing_lo = l.rolling(15).min().iloc[-1]
        
        if swing_hi <= swing_lo:
            return {"ok": False, "score": 0.0, "zone": None, "reasons": ["flat_market"], "confirmed": False}
        
        # مستويات فيبوناتشي موسعة
        fib_levels = {
            'f0382': swing_lo + 0.382 * (swing_hi - swing_lo),
            'f0500': swing_lo + 0.500 * (swing_hi - swing_lo),
            'f0618': swing_lo + 0.618 * (swing_hi - swing_lo),
            'f0786': swing_lo + 0.786 * (swing_hi - swing_lo),
            'f0886': swing_lo + 0.886 * (swing_hi - swing_lo)
        }
        
        last_close = float(c.iloc[-1])
        
        # نحسب قرب السعر من القاع والقمة
        dist_to_low  = abs(last_close - swing_lo)
        dist_to_high = abs(last_close - swing_hi)

        # تحليل الحجم
        vol_ma20 = v.rolling(20).mean().iloc[-1]
        vol_ok = float(v.iloc[-1]) >= vol_ma20 * 0.8
        volume_spike = float(v.iloc[-1]) > vol_ma20 * 1.5
        
        # تحليل الشمعة الحالية
        current_open = float(df['open'].iloc[-1])
        current_high = float(h.iloc[-1])
        current_low = float(l.iloc[-1])
        
        body = abs(last_close - current_open)
        wick_up = current_high - max(last_close, current_open)
        wick_down = min(last_close, current_open) - current_low
        
        bull_candle = (wick_down > (body * 1.2) and last_close > current_open) or (body > 0 and last_close > current_open and wick_down > wick_up)
        bear_candle = (wick_up > (body * 1.2) and last_close < current_open) or (body > 0 and last_close < current_open and wick_up > wick_down)
        
        # Footprint analysis
        footprint = compute_footprint_metrics(df)
        
        score = 0.0
        zone_type = None
        reasons = []
        confirmed = False
        
        # المنطقة الذهبية السفلية (شراء) — السعر داخل 0.618–0.786 وأقرب للقاع
        if fib_levels['f0618'] <= last_close <= fib_levels['f0786'] and dist_to_low <= dist_to_high:
            score += 3.0
            reasons.append("منطقة_ذهبية_سفلية")
            
            if bull_candle:
                score += 2.0
                reasons.append("شمعة_صاعدة")
            
            if volume_spike:
                score += 1.5
                reasons.append("حجم_مرتفع")
            
            if footprint.get('ok') and footprint.get('absorption_bull'):
                score += 2.0
                reasons.append("امتصاص_شرائي")
            
            # تأكيد من الشموع السابقة داخل نفس المنطقة
            confirmation_bars = 0
            for i in range(2, min(6, len(df))):
                prev_close = float(df['close'].iloc[-i])
                if fib_levels['f0618'] <= prev_close <= fib_levels['f0786']:
                    confirmation_bars += 1
            
            if confirmation_bars >= 2:
                score += 1.5
                reasons.append(f"تأكيد_{confirmation_bars}_شمعة")
                confirmed = True
            
            if score >= GOLDEN_ENTRY_SCORE:
                zone_type = "golden_bottom"
        
        # المنطقة الذهبية العلوية (بيع) — السعر داخل 0.618–0.786 وأقرب للقمة
        elif fib_levels['f0618'] <= last_close <= fib_levels['f0786'] and dist_to_high < dist_to_low:
            score += 3.0
            reasons.append("منطقة_ذهبية_علوية")
            
            if bear_candle:
                score += 2.0
                reasons.append("شمعة_هابطة")
            
            if volume_spike:
                score += 1.5
                reasons.append("حجم_مرتفع")
            
            if footprint.get('ok') and footprint.get('absorption_bear'):
                score += 2.0
                reasons.append("امتصاص_بيعي")
            
            # تأكيد من الشموع السابقة داخل نفس المنطقة
            confirmation_bars = 0
            for i in range(2, min(6, len(df))):
                prev_close = float(df['close'].iloc[-i])
                if fib_levels['f0618'] <= prev_close <= fib_levels['f0786']:
                    confirmation_bars += 1
            
            if confirmation_bars >= 2:
                score += 1.5
                reasons.append(f"تأكيد_{confirmation_bars}_شمعة")
                confirmed = True
            
            if score >= GOLDEN_ENTRY_SCORE:
                zone_type = "golden_top"
        
        ok = zone_type is not None and ALLOW_GZ_ENTRY
        return {
            "ok": ok,
            "score": score,
            "zone": {"type": zone_type, "levels": fib_levels} if zone_type else None,
            "reasons": reasons,
            "confirmed": confirmed
        }
        
    except Exception as e:
        return {"ok": False, "score": 0.0, "zone": None, "reasons": [f"error: {e}"], "confirmed": False}

def decide_strategy_mode_enhanced(df, adx=None, di_plus=None, di_minus=None, rsi_ctx=None, footprint=None):
    """تحديد نمط التداول المحسن: SCALP أم TREND مع VWAP + Footprint"""
    ind = compute_indicators(df)

    if adx is None or di_plus is None or di_minus is None:
        adx = ind.get('adx', 0)
        di_plus = ind.get('plus_di', 0)
        di_minus = ind.get('minus_di', 0)

    if rsi_ctx is None:
        rsi_ctx = rsi_ma_context(df)

    if footprint is None:
        footprint = compute_footprint_metrics(df)

    di_spread = abs(di_plus - di_minus)

    # VWAP context
    vwap = ind.get("vwap")
    price = float(df["close"].iloc[-1])
    if vwap and VWAP_ENABLED:
        vwap_diff_bps = abs(price - vwap) / vwap * 10000.0
        near_vwap = vwap_diff_bps <= VWAP_SCALP_BAND_BPS
        far_from_vwap = vwap_diff_bps >= VWAP_TREND_BAND_BPS
    else:
        near_vwap = False
        far_from_vwap = False

    # ترند قوي
    strong_trend = (
        (adx >= ADX_TREND_MIN and di_spread >= DI_SPREAD_TREND) or
        (rsi_ctx["trendZ"] in ("bull", "bear") and not rsi_ctx["in_chop"])
    )

    # Footprint confirmation
    footprint_confirmation = False
    if footprint.get('ok'):
        trend_dir = 'bull' if di_plus > di_minus else 'bear'
        if strong_trend and footprint.get('delta_trend') == trend_dir:
            footprint_confirmation = True

    # منطق اختيار النمط:
    # - ترند: ترند قوي + بعيد عن VWAP
    # - سكالب: عادي أو قرب من VWAP
    if strong_trend and footprint_confirmation and (far_from_vwap or not VWAP_ENABLED):
        mode = "trend"
        why = "strong_trend+footprint+far_from_vwap"
    else:
        mode = "scalp"
        why_parts = ["scalp_default"]
        if VWAP_ENABLED and near_vwap:
            why_parts.append("near_vwap")
        why = "+".join(why_parts)

    return {"mode": mode, "why": why, "footprint_ok": footprint_confirmation}

# =================== SMART PROFIT AI ===================
def smart_profit_ai_decision(state, df, ind, mode, side, entry_price, current_price):
    """
    ذكاء اصطناعي لجني الأرباح:
    - يعتمد على profit_profile من الحالة (SCALP_STRICT / MID_2TP / TREND_3TP)
    - يفرق بين TP1 وما بعدها
    - يطبق HOLD-TP ورفع الأهداف لما الصفقة تكون قوية
    """
    # PnL % كنسبة مئوية
    pnl_pct = (current_price - entry_price) / entry_price * 100.0 * (1 if side == "long" else -1)

    profit_profile      = state.get("profit_profile", "SCALP_STRICT")
    achieved_targets    = state.get("profit_targets_achieved", 0)
    highest_profit_pct  = state.get("highest_profit_pct", 0.0)

    # اختيار سلم الـ TP حسب البروفايل
    if profit_profile == "TREND_3TP":
        tp_levels    = TREND_TPS[:]         # مثال: [0.50, 1.00, 1.80]
        tp_fractions = TREND_TP_FRACS[:]    # [0.30, 0.30, 0.20]
    elif profit_profile == "MID_2TP":
        tp_levels    = TREND_TPS[:2]        # نستخدم أول هدفين كصفقة mid
        tp_fractions = [0.50, 0.50]
    else:  # SCALP_STRICT
        tp_levels    = SCALP_TPS[:]
        tp_fractions = SCALP_TP_FRACS[:]

    next_target_index = achieved_targets
    if next_target_index >= len(tp_levels):
        # كل أهداف السلم اتحققت بالفعل
        return {"action": "hold", "target": None, "reason": "all_targets_reached"}

    base_target_pct      = tp_levels[next_target_index]
    next_target_fraction = tp_fractions[next_target_index]

    # قوة الإشارة
    signal_strength = calculate_signal_strength(df, ind, side)
    adx             = float(ind.get("adx", 0.0))

    boosted_target_pct = base_target_pct

    # ===== بعد TP1 في صفقات MID/TREND: HOLD-TP ورفع الأهداف =====
    if profit_profile in ("TREND_3TP", "MID_2TP") and achieved_targets >= 1:
        if (
            signal_strength >= HOLD_AFTER_TP1_MIN_STRENGTH
            and adx >= HOLD_AFTER_TP1_MIN_ADX
            and highest_profit_pct >= base_target_pct
        ):
            boosted_target_pct = base_target_pct * HOLD_AFTER_TP1_EXTRA_BOOST
            if signal_strength >= VERY_STRONG_SIGNAL:
                boosted_target_pct = base_target_pct * VERY_STRONG_TP_BOOST
    else:
        # قبل TP1 أو سكالب: Boost بسيط حسب القوة
        if signal_strength >= 8.0:
            boosted_target_pct = base_target_pct * 1.20
        elif signal_strength >= 6.0:
            boosted_target_pct = base_target_pct * 1.10
        elif signal_strength < 4.0:
            boosted_target_pct = base_target_pct * 0.80

    target_pct = boosted_target_pct

    if pnl_pct >= target_pct:
        return {
            "action": "take_profit",
            "target": next_target_index + 1,
            "target_pct": target_pct,
            "fraction": next_target_fraction,
            "reason": (
                f"TP{next_target_index + 1} hit "
                f"({base_target_pct:.2f}%→{target_pct:.2f}%) "
                f"| strength={signal_strength:.1f}"
            ),
        }

    return {
        "action": "hold",
        "target": next_target_index + 1,
        "reason": (
            f"waiting TP{next_target_index + 1} "
            f"({pnl_pct:.2f}% / {target_pct:.2f}%) "
            f"| strength={signal_strength:.1f}"
        ),
    }

def calculate_signal_strength(df, ind, side):
    """حساب قوة الإشارة للتداول"""
    strength = 0.0
    
    # قوة ADX
    adx = ind.get('adx', 0)
    if adx > 25:
        strength += 3.0
    elif adx > 20:
        strength += 2.0
    elif adx > 15:
        strength += 1.0
    
    # قوة RSI
    rsi = ind.get('rsi', 50)
    if (side == "long" and rsi < 70) or (side == "short" and rsi > 30):
        strength += 2.0
    
    # قوة DI Spread
    di_spread = ind.get('di_spread', 0)
    if di_spread > 8:
        strength += 2.0
    elif di_spread > 5:
        strength += 1.0
    
    # Footprint confirmation
    footprint = ind.get('footprint', {})
    if footprint.get('ok'):
        if (side == "long" and footprint.get('delta_trend') == 'bull') or \
           (side == "short" and footprint.get('delta_trend') == 'bear'):
            strength += 2.0
    
    # Golden Zone confirmation
    gz = ind.get('gz', {})
    if gz.get('ok') and gz.get('confirmed'):
        strength += 3.0
    elif gz.get('ok'):
        strength += 1.5
    
    return min(10.0, strength)

# =================== ADVANCED COUNCIL VOTING WITH ALL ENGINES + EDGE ALGO + CVD DIVERGENCE ===================
def evaluate_rf_signal_context(council_data, rf_side: str) -> dict:
    """
    تقييم إشارة RF باستخدام:
      SMC + Golden Zone + Flow + UltraMS + Edge + VWAP + ADX/DI + CVD Divergence
    يخرج:
      grade: "weak" / "mid" / "strong"
      score: رقم القوة
      area_type: نوع المنطقة (golden_bottom / supply / demand / liquidity / ...)
      tp_profile: "SCALP_STRICT" / "MID_2TP" / "TREND_3TP"
    """
    score = 0.0
    reasons = []

    ind      = council_data.get("ind", {})
    smc      = council_data.get("smc_analysis", {})
    gz       = council_data.get("gz", {})
    flow     = council_data.get("flow", {})
    ultra_ms = council_data.get("ultra_ms", {})
    edge     = council_data.get("edge_setup", {})
    vwap_ctx = council_data.get("vwap", {})
    cvd_div  = council_data.get("ind", {}).get("cvd_div", {})

    is_buy  = (rf_side == "buy")
    is_sell = (rf_side == "sell")

    adx       = float(ind.get("adx", 0.0))
    di_spread = float(ind.get("di_spread", 0.0))

    # ===== 1) SMC / مناطق عرض وطلب / مصائد سيولة =====
    zone_type = smc.get("zone_type") or smc.get("zone_type_primary")
    smc_bias  = smc.get("bias")
    is_sweep  = smc.get("liquidity_sweep", False) or smc.get("stop_hunt", False)
    is_ob     = smc.get("is_ob", False)
    is_sd     = smc.get("is_sd", False)
    is_retest = smc.get("retest", False)

    area_type = zone_type or "unknown"

    if is_buy and smc_bias == "bull":
        score += 4.0
        reasons.append("📦 SMC: bullish demand/OB")
    if is_sell and smc_bias == "bear":
        score += 4.0
        reasons.append("📦 SMC: bearish supply/OB")

    if is_ob:
        score += 1.5; reasons.append("OB confluence")
    if is_sd:
        score += 1.0; reasons.append("Supply/Demand zone")
    if is_retest:
        score += 1.0; reasons.append("clean retest")

    if is_sweep:
        score += 2.0; reasons.append("💧 Liquidity sweep / stop-hunt")

    # ===== 2) Golden Zone =====
    gz_bias = gz.get("bias")
    gz_conf = gz.get("confirmed", False)
    if gz_conf:
        if is_buy and gz_bias == "bullish":
            score += 3.0
            area_type = area_type or "golden_bottom"
            reasons.append("⭐ Golden Bottom confirmed")
        if is_sell and gz_bias == "bearish":
            score += 3.0
            area_type = area_type or "golden_top"
            reasons.append("⭐ Golden Top confirmed")

    # ===== 3) Ultra Market Structure =====
    ms_bias = ultra_ms.get("bias", "neutral")
    if is_buy and ms_bias == "bull":
        score += 2.0; reasons.append("📈 UltraMS bullish BOS/structure")
    if is_sell and ms_bias == "bear":
        score += 2.0; reasons.append("📉 UltraMS bearish BOS/structure")

    if ultra_ms.get("bos"):
        score += 1.5; reasons.append("BOS in RF direction")
    if ultra_ms.get("choch"):
        score += 1.0; reasons.append("CHoCH context")

    # ===== 4) Flow / Footprint =====
    flow_side = flow.get("side")
    if flow_side == rf_side:
        score += 2.0; reasons.append("🧱 Flow pressure with RF side")

    if flow.get("big_buyers") and is_buy:
        score += 1.5; reasons.append("big buyers at zone")
    if flow.get("big_sellers") and is_sell:
        score += 1.5; reasons.append("big sellers at zone")

    # ===== 5) ADX / DI =====
    if adx >= 18 and di_spread >= 4:
        score += 2.0; reasons.append(f"ADX {adx:.1f} / DI spread {di_spread:.1f}")
    if adx >= 25 and di_spread >= 7:
        score += 2.0; reasons.append("strong trend strength")

    # ===== 6) Edge Algo grade =====
    grade_raw = (edge or {}).get("grade", "weak")
    if grade_raw == "mid":
        score += 2.0; reasons.append("EdgeAlgo mid setup")
    elif grade_raw == "strong":
        score += 4.0; reasons.append("EdgeAlgo strong setup")

    # ===== 7) VWAP distance =====
    try:
        z = float(vwap_ctx.get("zscore", 0.0))
    except Exception:
        z = 0.0
    if abs(z) >= 0.7:
        score += 1.0; reasons.append(f"VWAP distance |z|={z:.2f}")
    else:
        reasons.append("VWAP mid-zone (neutral)")

    # ===== 8) CVD Divergence =====
    div_type = cvd_div.get("type")
    if div_type == "regular_bullish" and is_buy:
        score += 2.5; reasons.append("CVD Regular Bullish Divergence")
    if div_type == "regular_bearish" and is_sell:
        score += 2.5; reasons.append("CVD Regular Bearish Divergence")
    if div_type == "hidden_bullish" and is_buy:
        score += 1.5; reasons.append("Hidden Bullish Divergence")
    if div_type == "hidden_bearish" and is_sell:
        score += 1.5; reasons.append("Hidden Bearish Divergence")

    # ===== Final Grade → TP profile =====
    if score >= 12.0:
        grade = "strong"
        tp_profile = "TREND_3TP"
    elif score >= 7.0:
        grade = "mid"
        tp_profile = "MID_2TP"
    else:
        grade = "weak"
        tp_profile = "SCALP_STRICT"

    return {
        "grade": grade,
        "score": score,
        "area_type": area_type,
        "tp_profile": tp_profile,
        "reasons": reasons,
    }

def council_votes_pro_enhanced(df):
    """مجلس تصويت محسّن مع جميع المحركات المتكاملة + Edge Algo + CVD Divergence"""
    try:
        # حساب جميع المؤشرات الأساسية
        ind = compute_indicators(df)
        rsi_ctx = rsi_ma_context(df)
        current_price = float(df['close'].iloc[-1])
        gz = golden_zone_pro_analysis(df, current_price)
        
        # Footprint analysis
        footprint = compute_footprint_metrics(df)
        
        # Enhanced candles with SMC
        cd = compute_enhanced_candles(df)
        
        # Liquidity traps
        liquidity_traps = detect_liquidity_traps(df, current_price)
        
        # Advanced SMC Analysis
        smc_engine = SMCCoreEngine()
        smc_analysis = smc_engine.analyze_smc_structure(df) if SMC_ENABLED else {}

        # REAL RF FILTER
        rf_ctx = compute_range_filter(df, period=RF_PERIOD, qty=RF_MULT)
        
        # REAL VWAP
        vwap_value = compute_vwap(df)
        
        # ULTRA MARKET STRUCTURE
        ultra_ms = UltraMarketStructureEngine()
        ultra_ms_ctx = ultra_ms.analyze(df)

        # ===== CVD DIVERGENCE OSCILLATOR (TradingFinder style) =====
        cvd_div = detect_cvd_divergence(
            df,
            n=2,
            period=21,
            mode="periodic",  # أو "ema" لو حابب
        )

        votes_b = 0; votes_s = 0
        score_b = 0.0; score_s = 0.0
        logs = []

        adx = ind.get('adx', 0)
        plus_di = ind.get('plus_di', 0)
        minus_di = ind.get('minus_di', 0)
        di_spread = ind.get('di_spread', abs(plus_di - minus_di))

        # ==== CVD DIVERGENCE VOTING =====
        if cvd_div.get("bull_div"):
            votes_b += 2
            score_b += 1.5
            logs.append("📈 CVD Bullish Divergence (TradingFinder)")

        if cvd_div.get("bear_div"):
            votes_s += 2
            score_s += 1.5
            logs.append("📉 CVD Bearish Divergence (TradingFinder)")

        # ==== EDGE ALGO VOTE =====
        # نحاول نحدد اتجاه الفائز مبدئيًا من باقي المؤشرات:
        side_hint = None
        if score_b > score_s and score_b >= 6.0:
            side_hint = "BUY"
        elif score_s > score_b and score_s >= 6.0:
            side_hint = "SELL"

        edge_setup = None
        if side_hint:
            # نبني Trend Info من المؤشرات الموجودة
            trend_info = {
                "direction": "up" if plus_di > minus_di else "down",
                "adx": float(adx or 0.0),
                "strength": float(di_spread / 100.0 if di_spread > 0 else 0.0),
                "is_strong": bool(adx or 0.0) >= 28,
                "atr": float(ind.get('atr', 0.0) or 0.0),
            }

            smc_ctx = smc_analysis or {}
            try:
                edge_setup = EDGE_ENGINE.compute_setup(df, side_hint, trend_info, smc_ctx)
            except Exception as e:
                log_w(f"EDGE_ENGINE error: {e}")
                edge_setup = None

            if edge_setup and edge_setup.get("valid"):
                grade = edge_setup.get("grade", "weak")
                # strong / mid / weak
                if grade == "strong":
                    if side_hint == "BUY":
                        score_b += 3.0; votes_b += 3
                    else:
                        score_s += 3.0; votes_s += 3
                    logs.append(f"🧠 EDGE STRONG {side_hint}")
                elif grade == "mid":
                    if side_hint == "BUY":
                        score_b += 2.0; votes_b += 2
                    else:
                        score_s += 2.0; votes_s += 2
                    logs.append(f"🧠 EDGE MID {side_hint}")
                else:
                    # weak → نقلل الثقة شوية بدل ما نزوّدها
                    if side_hint == "BUY":
                        score_b *= 0.9
                    else:
                        score_s *= 0.9
                    logs.append(f"⚠️ EDGE WEAK {side_hint} — تخفيف ثقة")
            else:
                logs.append("EDGE: no_valid_setup")

        # ==== REAL RF CONTRIBUTION =====
        if rf_ctx.get("buy_signal") and current_price > rf_ctx.get("filt", current_price):
            score_b += 1.5
            votes_b += 2
            logs.append("📗 RF BUY Signal (Real)")
        
        if rf_ctx.get("sell_signal") and current_price < rf_ctx.get("filt", current_price):
            score_s += 1.5
            votes_s += 2
            logs.append("📕 RF SELL Signal (Real)")

        # ==== REAL VWAP CONTRIBUTION =====
        if vwap_value:
            dist = (current_price - vwap_value) / vwap_value  # انحراف عن الـ VWAP

            # مع الترند وفي اتجاه الـ VWAP → تقوية القرار
            if dist > 0 and plus_di > minus_di:
                score_b += 1.0
                votes_b += 1
                logs.append("⚖️ Above VWAP in Uptrend")
            elif dist < 0 and minus_di > plus_di:
                score_s += 1.0
                votes_s += 1
                logs.append("⚖️ Below VWAP in Downtrend")

            # لو انحراف كبير عن VWAP (> 1%) وضد الاتجاه → حذر
            if abs(dist) > 0.01:
                if dist > 0 and score_b < score_s:
                    # السعر فوق VWAP بس سكور البيع أعلى → خفّف البيع شوية
                    score_s *= 0.9
                    logs.append("⚠️ SELL far above VWAP (risk)")
                elif dist < 0 and score_s < score_b:
                    # السعر تحت VWAP بس سكور الشراء أعلى → خفّف الشراء شوية
                    score_b *= 0.9
                    logs.append("⚠️ BUY far below VWAP (risk)")

        # ==== ULTRA MARKET STRUCTURE CONTRIBUTION =====
        ms_bias = ultra_ms_ctx.get("bias", "neutral")
        ms_fvg = ultra_ms_ctx.get("fvg") or {}
        ms_prem = ultra_ms_ctx.get("premium_discount") or {}
        liq_ctx = ultra_ms_ctx.get("liq_grab") or {}

        # Bias عام من BOS / CHoCH
        if ms_bias == "bull":
            score_b += 2.0
            votes_b += 2
            logs.append("🏛 UltraMS Bull BOS")
        elif ms_bias == "bear":
            score_s += 2.0
            votes_s += 2
            logs.append("🏛 UltraMS Bear BOS")

        # FVG قريب
        if ms_fvg:
            if ms_fvg.get("bull_near"):
                score_b += 1.5
                votes_b += 1
                logs.append("🟩 Bull FVG Near")
            if ms_fvg.get("bear_near"):
                score_s += 1.5
                votes_s += 1
                logs.append("🟥 Bear FVG Near")

        # Premium / Discount zones
        zone = ms_prem.get("zone", "mid")
        if zone in ("discount", "ultra_discount") and ms_bias == "bull":
            score_b += 1.0
            votes_b += 1
            logs.append("💚 Discount + Bull Bias")
        if zone in ("premium", "ultra_premium") and ms_bias == "bear":
            score_s += 1.0
            votes_s += 1
            logs.append("❤️ Premium + Bear Bias")

        # Liquidity Grabs
        if liq_ctx.get("grab_up"):
            # كسرة وهمية فوق → تميل للهبوط
            score_s += 1.0
            votes_s += 1
            logs.append("💦 Liquidity Grab UP")
        if liq_ctx.get("grab_down"):
            score_b += 1.0
            votes_b += 1
            logs.append("💦 Liquidity Grab DOWN")

        # ==== ADVANCED SMC VOTING مع فلتر ADX+ATR ====
        if SMC_ENABLED:
            # 1. تصويت مناطق الطلب والعرض القوية
            for zone in smc_analysis.get('supply_demand_zones', []):
                if smc_engine._is_zone_active(zone, current_price):
                    if zone['type'] == 'demand' and zone['strength'] >= 0.7:
                        votes_b += 3
                        score_b += 2.0
                        logs.append(f"📈 SMC Demand Zone (قوة: {zone['strength']:.1f})")
                    elif zone['type'] == 'supply' and zone['strength'] >= 0.7:
                        votes_s += 3
                        score_s += 2.0
                        logs.append(f"📉 SMC Supply Zone (قوة: {zone['strength']:.1f})")

            # 2. تصويت Order Blocks القوية
            for ob in smc_analysis.get('order_blocks', []):
                if (ob['low'] <= current_price <= ob['high'] and 
                    ob['strength'] >= OB_STRENGTH_THRESHOLD):
                    if ob['type'] == 'bullish_ob':
                        votes_b += 2
                        score_b += 1.5
                        logs.append(f"🟢 Bullish OB (قوة: {ob['strength']:.1f})")
                    else:
                        votes_s += 2
                        score_s += 1.5
                        logs.append(f"🔴 Bearish OB (قوة: {ob['strength']:.1f})")

            # 3. تصويت FVG عالية الاحتمالية
            for fvg in smc_analysis.get('fair_value_gaps', []):
                if (fvg['low'] <= current_price <= fvg['high'] and 
                    fvg.get('probability', 0) >= FVG_VALIDITY_THRESHOLD):
                    if fvg['type'] == 'fvg_bullish':
                        votes_b += 2
                        score_b += 1.5
                        logs.append(f"⚡ Bullish FVG (احتمال: {fvg['probability']:.1f})")
                    else:
                        votes_s += 2
                        score_s += 1.5
                        logs.append(f"⚡ Bearish FVG (احتمال: {fvg['probability']:.1f})")

            # 4. تصويت Stop-Hunt مع فلتر ADX+ATR
            for trap in smc_analysis.get('liquidity_levels', []):
                if trap.get('type') in ['stop_hunt_bull', 'stop_hunt_bear']:
                    # تطبيق فلتر ADX+ATر
                    if smc_engine._is_valid_stop_hunt_with_filters(trap, df):
                        if trap['type'] == 'stop_hunt_bull':
                            votes_s += 3  # دخول عكسي (بيع)
                            score_s += 2.5
                            logs.append(f"🎯 STOP-HUNT BULL FILTERED (ADX={trap.get('adx',0):.1f}, ATRx={trap.get('atr_mult',0):.1f})")
                        else:  # stop_hunt_bear
                            votes_b += 3  # دخول عكسي (شراء)
                            score_b += 2.5
                            logs.append(f"🎯 STOP-HUNT BEAR FILTERED (ADX={trap.get('adx',0):.1f}, ATRx={trap.get('atr_mult',0):.1f})")

        # --- VWAP SCALP Strategy ---
        if VWAP_ENABLED and vwap_value:
            vwap_diff_bps = abs(current_price - vwap_value) / vwap_value * 10000.0
            near_vwap = vwap_diff_bps <= VWAP_SCALP_BAND_BPS
            
            if near_vwap and cd:
                if cd.get("buy"):
                    votes_b += 2; score_b += 1.5
                    logs.append("⚡ VWAP SCALP BUY zone")
                if cd.get("sell"):
                    votes_s += 2; score_s += 1.5
                    logs.append("⚡ VWAP SCALP SELL zone")

        # --- VWAP TREND Boost ---
        if VWAP_ENABLED and vwap_value:
            vwap_diff_bps = abs(current_price - vwap_value) / vwap_value * 10000.0
            far_from_vwap = vwap_diff_bps >= VWAP_TREND_BAND_BPS
            
            if far_from_vwap and adx >= ADX_TREND_MIN:
                if plus_di > minus_di and current_price > vwap_value:
                    votes_b += 1; score_b += 1.0
                    logs.append("📈 VWAP TREND BOOST BUY")
                elif minus_di > plus_di and current_price < vwap_value:
                    votes_s += 1; score_s += 1.0
                    logs.append("📉 VWAP TREND BOOST SELL")

        # --- ترند ADX/DI مع Footprint تأكيد
        if adx > ADX_TREND_MIN:
            if plus_di > minus_di and di_spread > DI_SPREAD_TREND:
                if footprint.get('ok') and footprint.get('delta_trend') == 'bull':
                    votes_b += 3; score_b += 2.0; logs.append("📈 ترند صاعد قوي + Footprint تأكيد")
                else:
                    votes_b += 2; score_b += 1.5; logs.append("📈 ترند صاعد قوي")
            elif minus_di > plus_di and di_spread > DI_SPREAD_TREND:
                if footprint.get('ok') and footprint.get('delta_trend') == 'bear':
                    votes_s += 3; score_s += 2.0; logs.append("📉 ترند هابط قوي + Footprint تأكيد")
                else:
                    votes_s += 2; score_s += 1.5; logs.append("📉 ترند هابط قوي")

        # --- RSI-MA cross / Trend-Z
        if rsi_ctx["cross"] == "bull" and rsi_ctx["rsi"] < 70:
            votes_b += 2; score_b += 1.0; logs.append("🟢 RSI-MA إيجابي")
        elif rsi_ctx["cross"] == "bear" and rsi_ctx["rsi"] > 30:
            votes_s += 2; score_s += 1.0; logs.append("🔴 RSI-MA سلبي")

        if rsi_ctx["trendZ"] == "bull":
            votes_b += 3; score_b += 1.5; logs.append("🚀 RSI ترند صاعد مستمر")
        elif rsi_ctx["trendZ"] == "bear":
            votes_s += 3; score_s += 1.5; logs.append("💥 RSI ترند هابط مستمر")

        # --- Golden Zones Pro
        if gz and gz.get("ok"):
            if gz['zone']['type'] == 'golden_bottom':
                votes_b += 4 if gz['confirmed'] else 3
                score_b += 2.0 if gz['confirmed'] else 1.5
                conf_text = "مؤكد" if gz['confirmed'] else "محتمل"
                logs.append(f"🏆 قاع ذهبي {conf_text} (قوة: {gz['score']:.1f})")
            elif gz['zone']['type'] == 'golden_top':
                votes_s += 4 if gz['confirmed'] else 3
                score_s += 2.0 if gz['confirmed'] else 1.5
                conf_text = "مؤكد" if gz['confirmed'] else "محتمل"
                logs.append(f"🏆 قمة ذهبية {conf_text} (قوة: {gz['score']:.1f})")

        # --- Footprint Boost
        if footprint.get('ok'):
            if footprint.get('absorption_bull'):
                votes_b += 2; score_b += 1.5; logs.append("👣 Footprint امتصاص شرائي")
            if footprint.get('absorption_bear'):
                votes_s += 2; score_s += 1.5; logs.append("👣 Footprint امتصاص بيعي")
            
            if footprint.get('volume_spike'):
                if footprint.get('delta') > 0:
                    votes_b += 1; score_b += 1.0; logs.append("📊 حجم شرائي عالي")
                else:
                    votes_s += 1; score_s += 1.0; logs.append("📊 حجم بيعي عالي")

        # --- SMC Candles
        if cd["score_buy"]>0:
            score_b += min(3.0, cd["score_buy"])
            logs.append(f"🕯️ SMC BUY ({cd['smc_pattern']}) +{cd['score_buy']:.1f}")
        if cd["score_sell"]>0:
            score_s += min(3.0, cd["score_sell"])
            logs.append(f"🕯️ SMC SELL ({cd['smc_pattern']}) +{cd['score_sell']:.1f}")

        # --- Liquidity Trap Awareness
        if liquidity_traps.get('ok') and liquidity_traps.get('traps'):
            for trap in liquidity_traps['traps']:
                if trap['type'] == 'stop_hunt_bull' and score_b > score_s:
                    score_b *= 1.1  # تعزيز الثقة في فخ الصعود
                    logs.append(f"🪤 فخ سيولة صاعد قريب ({trap['distance_pct']:.2f}%)")
                elif trap['type'] == 'stop_hunt_bear' and score_s > score_b:
                    score_s *= 1.1  # تعزيز الثقة في فخ الهبوط
                    logs.append(f"🪤 فخ سيولة هابط قريب ({trap['distance_pct']:.2f}%)")

        # تخفيف النطاق المحايد
        if rsi_ctx["in_chop"]:
            score_b *= 0.7; score_s *= 0.7; logs.append("⚖️ RSI محايد — تخفيض ثقة")

        # حارس ADX عام
        if adx < ADX_GATE:
            score_b *= 0.8; score_s *= 0.8; logs.append(f"🛡️ ADX Gate ({adx:.1f} < {ADX_GATE})")

        # منع الفلات والرينج
        if di_spread < 3 and adx < 15:
            score_b *= 0.6; score_s *= 0.6; logs.append("🚫 سوق مسطح - تجنب الدخول")

        # ضمّ إشارات المحسنة لباقي المنظومة
        ind.update({
            "rsi_ma": rsi_ctx["rsi_ma"],
            "rsi_trendz": rsi_ctx["trendZ"],
            "di_spread": di_spread,
            "gz": gz,
            "footprint": footprint,
            "candle_buy_score": cd["score_buy"],
            "candle_sell_score": cd["score_sell"],
            "wick_up_big": cd["wick_up_big"],
            "wick_dn_big": cd["wick_dn_big"],
            "candle_tags": cd["pattern"],
            "smc_pattern": cd["smc_pattern"],
            "liquidity_trap": cd["liquidity_trap"],
            "vwap": vwap_value,
            "rf": rf_ctx,
            "cvd_div": cvd_div,  # 🔹 إضافة المؤشر الجديد
        })

        return {
            "b": votes_b, "s": votes_s,
            "score_b": score_b, "score_s": score_s,
            "logs": logs, "ind": ind, "gz": gz, 
            "footprint": footprint, "candles": cd,
            "liquidity_traps": liquidity_traps,
            "smc_analysis": smc_analysis if SMC_ENABLED else {},
            "rf": rf_ctx,
            "vwap": vwap_value,
            "ultra_ms": ultra_ms_ctx,
            "edge_setup": edge_setup  # 🔹 تمت إضافة Edge Setup
        }
    except Exception as e:
        log_w(f"council_votes_pro_enhanced error: {e}")
        return {"b":0,"s":0,"score_b":0.0,"score_s":0.0,"logs":[],"ind":{},"gz":None,"candles":{}}

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

def resume_open_position(exchange, symbol: str, state: dict) -> dict:
    if not RESUME_ON_RESTART:
        log_i("resume disabled"); return state

    live = fetch_live_position(exchange, symbol)
    if not live.get("ok"):
        log_i("no live position to resume"); return state

    ts = int(time.time())
    prev = load_state()
    if prev.get("ts") and (ts - int(prev["ts"])) > RESUME_LOOKBACK_SECS:
        log_w("found old local state — will override with exchange live snapshot")

    state.update({
        "in_position": True,
        "side": live["side"],
        "entry_price": live["entry"],
        "position_qty": live["qty"],
        "leverage": live.get("leverage") or state.get("leverage") or 10,
        "partial_taken": prev.get("partial_taken", False),
        "breakeven_armed": prev.get("breakeven_armed", False),
        "trail_active": prev.get("trail_active", False),
        "trail_tightened": prev.get("trail_tightened", False),
        "mode": prev.get("mode", "trend"),
        "gz_snapshot": prev.get("gz_snapshot", {}),
        "cv_snapshot": prev.get("cv_snapshot", {}),
        "footprint_snapshot": prev.get("footprint_snapshot", {}),
        "opened_at": prev.get("opened_at", ts),
    })
    save_state(state)
    log_g(f"RESUME: {state['side']} qty={state['position_qty']} @ {state['entry_price']:.6f} lev={state['leverage']}x")
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
    يطبع Snapshot موحّد: Bookmap + Flow + Council + Strategy + Balance/PnL + VWAP + SMC + ADX+ATR Filters + Edge Algo + CVD Divergence
    """
    try:
        bm = bookmap_snapshot(exchange, symbol)
        flow = compute_flow_metrics(df)
        cv = council_votes_pro_enhanced(df)
        mode = decide_strategy_mode_enhanced(df)
        current_price = float(df['close'].iloc[-1])
        gz = golden_zone_pro_analysis(df, current_price)

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

        strat_icon = "⚡" if mode["mode"]=="scalp" else "📈" if mode["mode"]=="trend" else "ℹ️"
        strat = f"Strategy: {strat_icon} {mode['mode'].upper()}"

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
            
            # عرض معلومات RF + VWAP + UltraMS + Edge Algo + CVD Divergence
            rf_info = cv.get('rf', {})
            vwap_val = cv.get('vwap', 0)
            ultra_ms = cv.get('ultra_ms', {})
            edge_setup = cv.get('edge_setup', {})
            cvd_div = cv.get('ind', {}).get('cvd_div', {})
            
            if rf_info:
                print(f"📗 REAL RF: dir={rf_info.get('dir', 0)} | filt={rf_info.get('filt', 0):.6f} | "
                      f"BUY={rf_info.get('buy_signal', False)} SELL={rf_info.get('sell_signal', False)}", flush=True)
            
            if vwap_val:
                vwap_diff = abs(current_price - vwap_val) / vwap_val * 10000.0
                print(f"⚖️ VWAP: {vwap_val:.6f} | Δ={vwap_diff:.1f}bps | "
                      f"{'NEAR' if vwap_diff <= VWAP_SCALP_BAND_BPS else 'FAR' if vwap_diff >= VWAP_TREND_BAND_BPS else 'MID'}", flush=True)
            
            if ultra_ms:
                print(f"🏛 UltraMS: bias={ultra_ms.get('bias', 'neutral')} | zone={(ultra_ms.get('premium_discount') or {}).get('zone', 'mid')} | "
                      f"BOS={ultra_ms.get('bos')} | FVG={ultra_ms.get('fvg', {}).get('bull_near', False) or ultra_ms.get('fvg', {}).get('bear_near', False)}", flush=True)
            
            # Edge Algo Logging
            if edge_setup and edge_setup.get('valid'):
                print(f"🧠 EDGE ALGO: {edge_setup['side']} grade={edge_setup['grade']} | "
                      f"RR≈{edge_setup['rr1']:.2f}/{edge_setup['rr2']:.2f}/{edge_setup['rr3']:.2f} | "
                      f"profile={edge_setup['tp_profile']} | score={edge_setup['edge_score']:.1f}", flush=True)
            
            # CVD Divergence Logging
            if cvd_div:
                if cvd_div.get('bull_div'):
                    print(f"📈 CVD DIVERGENCE: Bullish (Hist={cvd_div.get('hist_last', 0):.0f})", flush=True)
                elif cvd_div.get('bear_div'):
                    print(f"📉 CVD DIVERGENCE: Bearish (Hist={cvd_div.get('hist_last', 0):.0f})", flush=True)
            
            # Advanced SMC Logging with ADX+ATR Filters
            if SMC_ENABLED and cv.get('smc_analysis'):
                smc = cv['smc_analysis']
                
                # عرض Stop-Hunt المصفاة
                if smc.get('liquidity_levels'):
                    stop_hunts = [t for t in smc['liquidity_levels'] if 'stop_hunt' in t.get('type', '')]
                    filtered_stop_hunts = [h for h in stop_hunts 
                                          if STOP_HUNT_ADX_MIN <= h.get('adx', 0) <= STOP_HUNT_ADX_MAX
                                          and h.get('atr_mult', 0) >= STOP_HUNT_ATR_MULT_MIN]
                    
                    if filtered_stop_hunts:
                        best_hunt = max(filtered_stop_hunts, key=lambda x: x.get('strength', 0))
                        print(f"🎯 STOP-HUNT FILTERED: {best_hunt['type']} | ADX={best_hunt.get('adx',0):.1f} | "
                              f"ATRx={best_hunt.get('atr_mult',0):.1f} | قوة={best_hunt.get('strength',0):.1f}", flush=True)
                
                if smc.get('trading_opportunities'):
                    opps = smc['trading_opportunities']
                    if opps:
                        # عرض أفضل فرص Stop-Hunt المصفاة
                        stop_hunt_opps = [o for o in opps if 'stop_hunt' in o['type']]
                        if stop_hunt_opps:
                            best_stop_hunt = max(stop_hunt_opps, key=lambda x: x.get('confidence', 0))
                            if best_stop_hunt.get('filters_applied'):
                                print(f"🧠 SMC STOP-HUNT: {best_stop_hunt['type']} {best_stop_hunt['direction'].upper()} "
                                      f"(ثقة: {best_stop_hunt['confidence']:.1f}) | ADX+ATR FILTERED ✓", flush=True)
                        
                        # عرض أفضل فرص بشكل عام
                        best_opp = max(opps, key=lambda x: x.get('confidence', 0))
                        print(f"🧠 SMC BEST: {best_opp['type']} {best_opp['direction'].upper()} (ثقة: {best_opp['confidence']:.1f})", flush=True)
            
            flow_z = flow['delta_z'] if flow and flow.get('ok') else 0.0
            bm_imb = bm['imbalance'] if bm and bm.get('ok') else 1.0
            
            rf_dir = rf_info.get('dir', 0) if rf_info else 0
            rf_buy = rf_info.get('buy_signal', False) if rf_info else False
            rf_sell = rf_info.get('sell_signal', False) if rf_info else False
            
            print(f"🧠 SNAP | {side_hint} | votes={cv['b']}/{cv['s']} score={cv['score_b']:.1f}/{cv['score_s']:.1f} "
                  f"| ADX={cv['ind'].get('adx',0):.1f} DI={cv['ind'].get('di_spread',0):.1f} | "
                  f"RF dir={rf_dir} B={rf_buy} S={rf_sell} | z={flow_z:.2f} | imb={bm_imb:.2f}", 
                  flush=True)
            
            # إضافة معلومات Footprint وSMC وEdge وCVD Divergence
            if cv.get('footprint', {}).get('ok'):
                fp = cv['footprint']
                print(f"👣 FOOTPRINT | Delta={fp['delta']:.0f} | CVD={fp['cumulative_delta']:.0f} | "
                      f"Spike={fp['volume_spike']} | AbsBull={fp['absorption_bull']} | AbsBear={fp['absorption_bear']}", flush=True)
            
            if cv.get('candles', {}).get('smc_pattern'):
                print(f"🕯️ SMC | {cv['candles']['smc_pattern']} | Trap={cv['candles']['liquidity_trap']}", flush=True)
            
            if cvd_div:
                if cvd_div.get('bull_div'):
                    print(f"📈 CVD DIVERGENCE: Bullish | Hist={cvd_div.get('hist_last', 0):.0f}", flush=True)
                elif cvd_div.get('bear_div'):
                    print(f"📉 CVD DIVERGENCE: Bearish | Hist={cvd_div.get('hist_last', 0):.0f}", flush=True)
            
            print("✅ ULTRA MARKET STRUCTURE + REAL RF + VWAP + SMC ADDONS LIVE + ADX+ATR FILTERS + EDGE ALGO + CVD DIVERGENCE ACTIVE", flush=True)

        return {"bm": bm, "flow": flow, "cv": cv, "mode": mode, "gz": gz, "wallet": wallet}
    except Exception as e:
        print(f"🟨 AddonLog error: {e}", flush=True)
        return {"bm": None, "flow": None, "cv": {"b":0,"s":0,"score_b":0.0,"score_s":0.0,"ind":{}},
                "mode": {"mode":"n/a"}, "gz": None, "wallet": ""}

# =================== EXECUTION MANAGER ===================
def execute_trade_decision(side, price, qty, mode, council_data, gz_data, smc_data=None, edge_setup=None):
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
    
    smc_note = ""
    stop_hunt_info = ""
    if smc_data and smc_data.get('trading_opportunities'):
        best_opp = max(smc_data['trading_opportunities'], key=lambda x: x.get('confidence', 0))
        smc_note = f" | 🧠 SMC:{best_opp['type']}({best_opp['confidence']:.1f})"
        
        # إضافة معلومات خاصة بـ Stop-Hunt
        if 'stop_hunt' in best_opp['type']:
            stop_hunt_info = f" | ADX+ATR FILTERED"
            if best_opp.get('stop_loss'):
                stop_hunt_info += f" | SL={best_opp['stop_loss']:.6f}"
    
    edge_note = ""
    if edge_setup and edge_setup.get('valid'):
        edge_note = f" | 🧠 EDGE:{edge_setup['grade']} RR≈{edge_setup['rr1']:.2f}/{edge_setup['rr2']:.2f}/{edge_setup['rr3']:.2f}"
    
    votes = council_data
    print(f"🎯 EXECUTE: {side.upper()} {qty:.4f} @ {price:.6f} | "
          f"mode={mode} | votes={votes['b']}/{votes['s']} score={votes['score_b']:.1f}/{votes['score_s']:.1f}"
          f"{gz_note}{smc_note}{stop_hunt_info}{edge_note}", flush=True)

    try:
        if MODE_LIVE:
            ex.set_leverage(LEVERAGE, SYMBOL, params={"side": "BOTH"})
            ex.create_order(SYMBOL, "market", side, qty, None, _params_open(side))
        
        log_g(f"✅ EXECUTED: {side.upper()} {qty:.4f} @ {price:.6f}")
        return True
    except Exception as e:
        log_e(f"❌ EXECUTION FAILED: {e}")
        return False

def setup_trade_management(mode, edge_setup=None):
    """تهيئة إدارة الصفقة حسب النمط و Edge Setup"""
    if edge_setup and edge_setup.get('valid'):
        # استخدام Edge Algo Profile
        tp_profile = edge_setup.get('tp_profile', 'SCALP_STRICT')
        
        if tp_profile == "TREND_3TP":
            return {
                "tp1_pct": TREND_TP1 / 100.0,
                "be_activate_pct": TREND_BE_AFTER / 100.0,
                "trail_activate_pct": 1.2 / 100.0,
                "atr_trail_mult": TREND_ATR_MULT,
                "close_aggression": "low",
                "max_tp_steps": 3,
                "tp_profile": tp_profile,
                "edge_setup": edge_setup
            }
        elif tp_profile == "MID_2TP":
            return {
                "tp1_pct": 0.8 / 100.0,
                "be_activate_pct": 0.5 / 100.0,
                "trail_activate_pct": 0.8 / 100.0,
                "atr_trail_mult": 1.6,
                "close_aggression": "medium",
                "max_tp_steps": 2,
                "tp_profile": tp_profile,
                "edge_setup": edge_setup
            }
        else:  # SCALP_STRICT
            return {
                "tp1_pct": SCALP_TP1 / 100.0,
                "be_activate_pct": SCALP_BE_AFTER / 100.0,
                "trail_activate_pct": 0.5 / 100.0,
                "atr_trail_mult": SCALP_ATR_MULT,
                "close_aggression": "high",
                "max_tp_steps": 1,
                "tp_profile": tp_profile,
                "edge_setup": edge_setup
            }
    else:
        # الوضع القديم
        if mode == "scalp":
            return {
                "tp1_pct": SCALP_TP1 / 100.0,
                "be_activate_pct": SCALP_BE_AFTER / 100.0,
                "trail_activate_pct": 0.8 / 100.0,
                "atr_trail_mult": SCALP_ATR_MULT,
                "close_aggression": "high",
                "max_tp_steps": 1,
                "tp_profile": "SCALP_STRICT",
                "edge_setup": None
            }
        else:
            return {
                "tp1_pct": TREND_TP1 / 100.0,
                "be_activate_pct": TREND_BE_AFTER / 100.0,
                "trail_activate_pct": 1.2 / 100.0,
                "atr_trail_mult": TREND_ATR_MULT,
                "close_aggression": "medium",
                "max_tp_steps": 3,
                "tp_profile": "TREND_3TP",
                "edge_setup": None
            }

# =================== STATE ===================
STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
    "tp1_done": False, "highest_profit_pct": 0.0,
    "profit_targets_achieved": 0,
    "edge_setup": None,
    "profit_profile": "SCALP_STRICT"
}
compound_pnl = 0.0
wait_for_next_signal_side = None

# RF-driven profit profile (يتم ضبطه وقت الدخول بناءً على تقييم إشارة RF)
pending_profit_profile = None

# =================== WAIT FOR NEXT SIGNAL ===================
def _arm_wait_after_close(prev_side):
    """تفعيل انتظار الإشارة التالية بعد الإغلاق"""
    global wait_for_next_signal_side
    wait_for_next_signal_side = "sell" if prev_side=="long" else ("buy" if prev_side=="short" else None)
    log_i(f"🛑 WAIT FOR NEXT SIGNAL: {wait_for_next_signal_side}")

def wait_gate_allow(df, info):
    """التحقق من بوابة الانتظار"""
    if wait_for_next_signal_side is None: 
        return True, ""
    
    bar_ts = int(info.get("time") or 0)
    need = (wait_for_next_signal_side=="buy" and info.get("long")) or (wait_for_next_signal_side=="sell" and info.get("short"))
    
    if need:
        return True, ""
    return False, f"wait-for-next-RF({wait_for_next_signal_side})"

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
    global wait_for_next_signal_side, pending_profit_profile
    prev_side = prev_side or STATE.get("side")
    STATE.update({
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
        "tp1_done": False, "highest_profit_pct": 0.0, "profit_targets_achieved": 0,
        "trail_tightened": False, "partial_taken": False,
        "edge_setup": None,
        "profit_profile": "SCALP_STRICT"
    })
    pending_profit_profile = None
    save_state({"in_position": False, "position_qty": 0})
    
    # تفعيل انتظار الإشارة التالية
    _arm_wait_after_close(prev_side)
    logging.info(f"AFTER_CLOSE waiting_for={wait_for_next_signal_side}")

# =================== SMART EXIT GUARD ===================
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
    evx_spike = False  # يمكن إضافة حساب EVX لاحقًا
    
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

    # --- Golden Reversal بعد TP1 ---
    if state.get('tp1_done') and (gz and gz.get('ok')):
        # إغلاق صارم لو تقاطع Golden عكس اتجاهي بعد TP1
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

    # --- Wick exhaustion + Tighten عند إجهاد/تدفق/جدار ---
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

# =================== ENHANCED TRADE EXECUTION WITH EDGE ALGO ===================
def open_market_enhanced(side, qty, price):
    """فتح صفقة محسنة مع Edge Algo + RF Master Profile"""
    global pending_profit_profile
    if qty <= 0: 
        log_e("skip open (qty<=0)")
        return False
    
    df = fetch_ohlcv()
    current_price = price or float(df['close'].iloc[-1])
    
    # Enhanced analysis
    snap = emit_snapshots(ex, SYMBOL, df)
    votes = snap["cv"]
    footprint = votes.get("footprint", {})
    smc_analysis = votes.get("smc_analysis", {})
    rf_ctx = votes.get("rf", {})
    ultra_ms = votes.get("ultra_ms", {})
    edge_setup = votes.get("edge_setup", {})
    cvd_div = votes.get("ind", {}).get("cvd_div", {})
    
    mode_data = decide_strategy_mode_enhanced(df, 
                                   adx=votes["ind"].get("adx"),
                                   di_plus=votes["ind"].get("plus_di"),
                                   di_minus=votes["ind"].get("minus_di"),
                                   rsi_ctx=rsi_ma_context(df),
                                   footprint=footprint)
    
    mode = mode_data["mode"]
    gz = snap["gz"]
    
    # فلتر قوة الإشارة قبل تنفيذ أي صفقة
    signal_strength_pre = calculate_signal_strength(df, votes["ind"], "long" if side == "buy" else "short")
    if signal_strength_pre < MIN_SIGNAL_FOR_ENTRY:
        log_i(f"⛔ SKIP ENTRY {side.upper()} | signal_strength={signal_strength_pre:.1f} < {MIN_SIGNAL_FOR_ENTRY}")
        return False
    
    # Enhanced management config with Edge Algo
    management_config = setup_trade_management(mode, edge_setup)
    
    success = execute_trade_decision(side, price, qty, mode, votes, gz, smc_analysis, edge_setup)
    
    if success:
        signal_strength = calculate_signal_strength(df, votes["ind"], "long" if side=="buy" else "short")
        
        # تحديد profit_profile:
        # 1) لو RF حدد بروفايل → نستخدمه
        # 2) لو Edge عنده tp_profile → نستخدمه
        # 3) غير كده: mode → SCALP / TREND
        if pending_profit_profile:
            profit_profile = pending_profit_profile
        else:
            profit_profile = "SCALP_STRICT"
            if edge_setup and edge_setup.get("valid"):
                profit_profile = edge_setup.get("tp_profile", "SCALP_STRICT")
            else:
                if mode == "trend":
                    profit_profile = "TREND_3TP"
                else:
                    profit_profile = "SCALP_STRICT"
        
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
            "management": management_config,
            "signal_strength": signal_strength,
            "edge_setup": edge_setup,
            "profit_profile": profit_profile
        })
        
        save_state({
            "in_position": True,
            "side": "LONG" if side.upper().startswith("B") else "SHORT",
            "entry_price": price,
            "position_qty": qty,
            "leverage": LEVERAGE,
            "mode": mode,
            "management": management_config,
            "signal_strength": signal_strength,
            "edge_setup": edge_setup,
            "profit_profile": profit_profile,
            "gz_snapshot": gz if isinstance(gz, dict) else {},
            "cv_snapshot": votes if isinstance(votes, dict) else {},
            "footprint_snapshot": footprint if isinstance(footprint, dict) else {},
            "smc_snapshot": smc_analysis if isinstance(smc_analysis, dict) else {},
            "rf_snapshot": rf_ctx if isinstance(rf_ctx, dict) else {},
            "ultra_ms_snapshot": ultra_ms if isinstance(ultra_ms, dict) else {},
            "cvd_div_snapshot": cvd_div if isinstance(cvd_div, dict) else {},
            "opened_at": int(time.time()),
            "partial_taken": False,
            "breakeven_armed": False,
            "trail_active": False,
            "trail_tightened": False,
        })
        
        pending_profit_profile = None  # تصفير بعد الاستخدام
        
        log_g(f"✅ ULTRA ENHANCED POSITION OPENED: {side.upper()} | mode={mode} | signal_strength={signal_strength:.1f} | edge_profile={profit_profile}")
        return True
    
    return False

# =================== INDICATORS ===================
def wilder_ema(s: pd.Series, n: int): 
    return s.ewm(alpha=1/n, adjust=False).mean()

def compute_indicators(df: pd.DataFrame):
    if len(df) < max(ATR_LEN, RSI_LEN, ADX_LEN) + 2:
        return {
            "rsi": 50.0, "plus_di": 0.0, "minus_di": 0.0,
            "dx": 0.0, "adx": 0.0, "atr": 0.0,
            "di_spread": 0.0, "vwap": None
        }

    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["volume"].astype(float)

    # ATR
    tr = pd.concat([(h - l).abs(),
                    (h - c.shift(1)).abs(),
                    (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = wilder_ema(tr, ATR_LEN)

    # RSI
    delta = c.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    rs = wilder_ema(up, RSI_LEN) / wilder_ema(dn, RSI_LEN).replace(0, 1e-12)
    rsi = 100 - (100 / (1 + rs))

    # ADX / DI
    up_move = h.diff()
    down_move = l.shift(1) - l
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    plus_di = 100 * (wilder_ema(plus_dm, ADX_LEN) / atr.replace(0, 1e-12))
    minus_di = 100 * (wilder_ema(minus_dm, ADX_LEN) / atr.replace(0, 1e-12))
    dx = (100 * (plus_di - minus_di).abs() /
          (plus_di + minus_di).replace(0, 1e-12)).fillna(0.0)
    adx = wilder_ema(dx, ADX_LEN)

    # VWAP (session-style على كامل الـ df)
    typical_price = (h + l + c) / 3.0
    pv = typical_price * v
    cum_pv = pv.cumsum()
    cum_vol = v.cumsum().replace(0, 1e-12)
    vwap_series = cum_pv / cum_vol

    i = len(df) - 1
    di_spread = float(abs(plus_di.iloc[i] - minus_di.iloc[i]))
    vwap_val = float(vwap_series.iloc[i]) if not vwap_series.empty else None

    return {
        "rsi": float(rsi.iloc[i]),
        "plus_di": float(plus_di.iloc[i]),
        "minus_di": float(minus_di.iloc[i]),
        "dx": float(dx.iloc[i]),
        "adx": float(adx.iloc[i]),
        "atr": float(atr.iloc[i]),
        "di_spread": di_spread,
        "vwap": vwap_val,
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

# =================== ENHANCED TRADE MANAGEMENT WITH EDGE ALGO ===================
def manage_after_entry_enhanced(df, ind, info):
    """إدارة محسنة للمركز مع Smart Profit AI + Smart Exit Guard + Edge Algo"""
    if not STATE["open"] or STATE["qty"] <= 0:
        return

    px   = info["price"]
    entry = STATE["entry"]
    side  = STATE["side"]
    qty   = STATE["qty"]
    mode  = STATE.get("mode", "trend")
    edge_setup = STATE.get("edge_setup", {})
    profit_profile = STATE.get("profit_profile", "SCALP_STRICT")
    
    # PnL % (كـ نسبة مئوية)
    pnl_pct = (px - entry) / entry * 100.0 * (1 if side == "long" else -1)
    STATE["pnl"] = pnl_pct

    if pnl_pct > STATE.get("highest_profit_pct", 0.0):
        STATE["highest_profit_pct"] = pnl_pct

    # ========= Smart Profit AI (سلم جني الأرباح الذكي) =========
    profit_decision = smart_profit_ai_decision(STATE, df, ind, mode, side, entry, px)
    
    if profit_decision["action"] == "take_profit":
        target_num = profit_decision["target"]
        fraction   = profit_decision["fraction"]
        close_qty  = safe_qty(qty * fraction)
        
        if close_qty > 0:
            close_side = "sell" if side == "long" else "buy"
            if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                try:
                    ex.create_order(SYMBOL, "market", close_side, close_qty, None, _params_close())
                    log_g(f"✅ SMART TP{target_num}: closed {fraction*100:.0f}% | {profit_decision['reason']}")
                    STATE["qty"] = safe_qty(qty - close_qty)
                    STATE["profit_targets_achieved"] = target_num
                    
                    # تحديد عدد الأهداف حسب البروفايل الفعلي
                    if profit_profile == "SCALP_STRICT":
                        max_targets = len(SCALP_TPS)
                    elif profit_profile == "MID_2TP":
                        max_targets = 2
                    else:  # TREND_3TP
                        max_targets = len(TREND_TPS)

                    if target_num >= max_targets:
                        close_market_strict("all_targets_achieved")
                        return
                except Exception as e:
                    log_e(f"❌ Smart TP failed: {e}")
            else:
                log_i(f"DRY_RUN: Smart TP{target_num} close {close_qty:.4f}")

    # ========= الإدارة الكلاسيك (TP1 + BE + Trail + Dust) مع Edge Algo =========
    current_atr      = ind.get("atr", 0.0)
    management       = STATE.get("management", {})
    
    # تحديد إعدادات بناءً على Edge Profile
    if profit_profile == "TREND_3TP":
        tp1_pct          = 1.2 / 100.0  # 1.2% لـ TREND
        be_activate_pct  = 0.8 / 100.0
        trail_activate_pct = 1.2 / 100.0
        atr_trail_mult   = 2.0
        max_tp_steps     = 3
    elif profit_profile == "MID_2TP":
        tp1_pct          = 0.8 / 100.0  # 0.8% لـ MID
        be_activate_pct  = 0.5 / 100.0
        trail_activate_pct = 0.8 / 100.0
        atr_trail_mult   = 1.6
        max_tp_steps     = 2
    else:  # SCALP_STRICT
        tp1_pct          = management.get("tp1_pct", SCALP_TP1/100.0)
        be_activate_pct  = management.get("be_activate_pct", SCALP_BE_AFTER/100.0)
        trail_activate_pct = management.get("trail_activate_pct", 0.5/100.0)
        atr_trail_mult   = management.get("atr_trail_mult", SCALP_ATR_MULT)
        max_tp_steps     = 1

    # نحول PnL من % إلى كسور عشان نستخدمه مع الحراس اللي شغّالين بالـ fraction
    pnl_frac = pnl_pct / 100.0

    # TP1 جزئي (مرة واحدة فقط)
    if not STATE.get("tp1_done") and pnl_frac >= tp1_pct:
        # تحديد كمية الإغلاق بناءً على البروفايل
        if profit_profile == "TREND_3TP":
            close_fraction = 0.25  # 25% للترند القوي
        elif profit_profile == "MID_2TP":
            close_fraction = 0.35  # 35% للترند المتوسط
        else:
            close_fraction = TP1_CLOSE_FRAC  # 40% للسكالب
        
        close_qty = safe_qty(STATE["qty"] * close_fraction)
        if close_qty > 0:
            close_side = "sell" if STATE["side"] == "long" else "buy"
            if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                try:
                    ex.create_order(SYMBOL, "market", close_side, close_qty, None, _params_close())
                    log_g(f"✅ TP1 HIT: closed {close_fraction*100:.0f}% | profile={profit_profile}")
                except Exception as e:
                    log_e(f"❌ TP1 close failed: {e}")
            STATE["qty"] = safe_qty(STATE["qty"] - close_qty)
            STATE["tp1_done"] = True
            STATE["profit_targets_achieved"] += 1

    # تفعيل Breakeven
    if not STATE.get("breakeven_armed") and pnl_frac >= be_activate_pct:
        STATE["breakeven_armed"] = True
        STATE["breakeven"] = entry
        log_i(f"BREAKEVEN ARMED | profile={profit_profile}")

    # تفعيل التريل
    if not STATE.get("trail_active") and pnl_frac >= trail_activate_pct:
        STATE["trail_active"] = True
        log_i(f"TRAIL ACTIVATED | profile={profit_profile}")

    # تحديث مستوى التريل
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

    # تنفيذ وقف التريل
    if STATE.get("trail"):
        if (side == "long" and px <= STATE["trail"]) or (side == "short" and px >= STATE["trail"]):
            log_w(f"TRAIL STOP: {px} vs trail {STATE['trail']} | profile={profit_profile}")
            close_market_strict("trail_stop")
            return

    # تنفيذ Breakeven الصارم
    if STATE.get("breakeven"):
        if (side == "long" and px <= STATE["breakeven"]) or (side == "short" and px >= STATE["breakeven"]):
            log_w(f"BREAKEVEN STOP: {px} vs breakeven {STATE['breakeven']} | profile={profit_profile}")
            close_market_strict("breakeven_stop")
            return

    # Dust guard: لو الكمية بقت فتات اقفل وخلاص
    if STATE["qty"] <= FINAL_CHUNK_QTY:
        log_w(f"DUST GUARD: qty {STATE['qty']} <= {FINAL_CHUNK_QTY}, closing... | profile={profit_profile}")
        close_market_strict("dust_guard")
        return

    # ========= Smart Exit Guard (Golden Reversal + Wick/Flow/Wall) =========
    try:
        guard = smart_exit_guard(
            STATE,
            df,
            ind,
            info.get("flow"),
            info.get("bm"),
            px,
            pnl_frac,           # هنا بنمررها كـ fraction (0.01 = 1%)
            mode,
            side,
            entry,
            gz=STATE.get("gz_snapshot", {})
        )
    except Exception as e:
        log_w(f"smart_exit_guard error: {e}")
        guard = None

    if guard and guard.get("action") != "hold":
        if guard.get("log"):
            log_w(guard["log"])
        act = guard["action"]

        # إحكام التريل عند إجهاد / جدار / تدفق معاكس
        if act == "tighten":
            STATE["trail_tightened"] = True

        # إغلاق صارم عند Golden Reversal أو Hard Close
        elif act == "close":
            close_market_strict(guard.get("why", "smart_exit_guard"))
            return
        # متعمدين نتجاهل "partial" هنا عشان ما نعملش TP1 مزدوج (Smart AI + Guard)

# =================== ULTRA ENHANCED TRADE LOOP WITH EDGE ALGO + CVD DIVERGENCE + ENHANCED FILTERS ===================
def trade_loop_enhanced():
    """حلقة تداول محسنة مع جميع المحركات المتكاملة + Edge Algo + CVD Divergence + Enhanced Filters"""
    global wait_for_next_signal_side, pending_profit_profile
    loop_i = 0
    
    while True:
        try:
            # جمع البيانات الأساسية
            bal = balance_usdt()
            px = price_now()
            df = fetch_ohlcv()
            info = rf_signal_live(df)
            ind = compute_indicators(df)
            spread_bps = orderbook_spread_bps()
            
            # Enhanced Snapshots مع جميع المحركات
            snap = emit_snapshots(ex, SYMBOL, df,
                                balance_fn=lambda: float(bal) if bal else None,
                                pnl_fn=lambda: float(compound_pnl))
            
            # تحديث حالة الربح/الخسارة
            if STATE["open"] and px:
                STATE["pnl"] = (px-STATE["entry"])*STATE["qty"] if STATE["side"]=="long" else (STATE["entry"]-px)*STATE["qty"]
            
            # إدارة الصفقة المفتوحة مع Smart Profit AI
            if STATE["open"]:
                manage_after_entry_enhanced(df, ind, {
                    "price": px or info["price"], 
                    "bm": snap["bm"],
                    "flow": snap["flow"],
                    **info
                })
            
            # قرار الدخول المحسن مع جميع المحركات + Edge Algo + CVD Divergence + Enhanced Filters
            reason = None
            if spread_bps is not None and spread_bps > MAX_SPREAD_BPS:
                reason = f"spread too high ({fmt(spread_bps,2)}bps > {MAX_SPREAD_BPS})"
            
            council_data = council_votes_pro_enhanced(df)
            gz = council_data.get("gz")
            footprint = council_data.get("footprint", {})
            smc_analysis = council_data.get("smc_analysis", {})
            rf_ctx = council_data.get("rf", {})
            ultra_ms = council_data.get("ultra_ms", {})
            edge_setup = council_data.get("edge_setup", {})
            cvd_div = council_data.get("ind", {}).get("cvd_div", {})
            
            sig = None
            entry_type = None
            pending_profit_profile = None

            # ========= RF MASTER ENTRY =========
            rf_side = None
            if rf_ctx.get("buy_signal"):
                rf_side = "buy"
            elif rf_ctx.get("sell_signal"):
                rf_side = "sell"

            if rf_side:
                rf_eval = evaluate_rf_signal_context(council_data, rf_side)
                grade      = rf_eval["grade"]
                score      = rf_eval["score"]
                tp_profile = rf_eval["tp_profile"]
                area_type  = rf_eval["area_type"]
                reasons    = rf_eval["reasons"]

                log_i(
                    f"🧮 RF MASTER EVAL: side={rf_side.upper()} | "
                    f"grade={grade} | score={score:.1f} | area={area_type} | "
                    f"reasons: " + " | ".join(reasons)
                )

                if grade == "weak":
                    log_i("🚫 RF entry skipped (weak setup)")
                else:
                    sig = rf_side
                    entry_type = f"RF_{grade.upper()}"
                    pending_profit_profile = tp_profile
                    log_g(f"✅ RF entry accepted: {rf_side.upper()} | tp_profile={tp_profile}")

            # من هنا فصاعدًا، باقي الاستراتيجيات تشتغل فقط لو RF ما فتحش صفقة
            # --- CVD Divergence Entry ---
            cvd_entry = False
            if sig is None and cvd_div and cvd_div.get("active") and cvd_div.get("strength",0)>=1:
                if cvd_div.get("bull_div"):
                    if council_data.get("ind", {}).get("adx", 0) >= CVD_ADX_MIN and council_data["score_b"] >= CVD_SCORE_MIN:
                        sig = "buy"
                        cvd_entry = True
                        log_i(
                            f"📈 CVD DIVERGENCE ENTRY: BUY | Bullish Divergence detected "
                            f"(ADX={council_data.get('ind', {}).get('adx', 0):.1f} ≥ {CVD_ADX_MIN} • score_b={council_data['score_b']:.1f} ≥ {CVD_SCORE_MIN})"
                        )
                    else:
                        log_i(
                            f"⚠️ CVD BUY skipped | weak context "
                            f"(ADX={council_data.get('ind', {}).get('adx', 0):.1f} • score_b={council_data['score_b']:.1f})"
                        )
                elif cvd_div.get("bear_div"):
                    if council_data.get("ind", {}).get("adx", 0) >= CVD_ADX_MIN and council_data["score_s"] >= CVD_SCORE_MIN:
                        sig = "sell"
                        cvd_entry = True
                        log_i(
                            f"📉 CVD DIVERGENCE ENTRY: SELL | Bearish Divergence detected "
                            f"(ADX={council_data.get('ind', {}).get('adx', 0):.1f} ≥ {CVD_ADX_MIN} • score_s={council_data['score_s']:.1f} ≥ {CVD_SCORE_MIN})"
                        )
                    else:
                        log_i(
                            f"⚠️ CVD SELL skipped | weak context "
                            f"(ADX={council_data.get('ind', {}).get('adx', 0):.1f} • score_s={council_data['score_s']:.1f})"
                        )

            # --- Advanced SMC Entry Pro مع ADX+ATR Filters وفلتر الثقة ≥ 0.8 ---
            smc_entry = False
            if sig is None and SMC_ENABLED and smc_analysis.get('trading_opportunities'):
                # فلترة الفرص بناءً على ADX+ATR وفلتر الثقة
                filtered_opportunities = []
                for opp in smc_analysis['trading_opportunities']:
                    if 'stop_hunt' in opp['type']:
                        # تطبيق فلتر ADX+ATR لفرص Stop-Hunt
                        if opp.get('filters_applied'):
                            filtered_opportunities.append(opp)
                    else:
                        # الفرص الأخرى (Supply/Demand, Order Blocks, etc.)
                        filtered_opportunities.append(opp)
                
                if filtered_opportunities:
                    best_opportunity = max(filtered_opportunities, 
                                         key=lambda x: x.get('confidence', 0))
                    
                    # تطبيق فلتر الثقة الجديد ≥ 0.8 (كان 0.7)
                    if best_opportunity['confidence'] >= SMC_CONFIDENCE_MIN:
                        if best_opportunity['direction'] == 'long':
                            sig = "buy"
                            smc_entry = True
                            entry_type = "SMC_STOP_HUNT" if 'stop_hunt' in best_opportunity['type'] else "SMC"
                            log_i(f"🎯 {entry_type} ENTRY: BUY | {best_opportunity['type']} (ثقة: {best_opportunity['confidence']:.1f} ≥ {SMC_CONFIDENCE_MIN})")
                        elif best_opportunity['direction'] == 'short':
                            sig = "sell" 
                            smc_entry = True
                            entry_type = "SMC_STOP_HUNT" if 'stop_hunt' in best_opportunity['type'] else "SMC"
                            log_i(f"🎯 {entry_type} ENTRY: SELL | {best_opportunity['type']} (ثقة: {best_opportunity['confidence']:.1f} ≥ {SMC_CONFIDENCE_MIN})")
                    else:
                        log_i(f"⚠️ SMC opportunity skipped | low confidence {best_opportunity['confidence']:.1f} < {SMC_CONFIDENCE_MIN}")

            # --- Enhanced Golden Entry Pro ---
            golden_entry = False
            if sig is None and (gz and gz.get("ok") and gz.get("confirmed")):
                if gz["zone"]["type"]=="golden_bottom" and gz["score"]>=GOLDEN_ENTRY_SCORE:
                    if footprint.get('ok') and footprint.get('absorption_bull'):
                        sig = "buy"
                        golden_entry = True
                        log_i(f"🎯 GOLDEN ENTRY PRO: BUY | score={gz['score']:.1f} | منطقة ذهبية مؤكدة + Footprint")
                elif gz["zone"]["type"]=="golden_top" and gz["score"]>=GOLDEN_ENTRY_SCORE:
                    if footprint.get('ok') and footprint.get('absorption_bear'):
                        sig = "sell"
                        golden_entry = True
                        log_i(f"🎯 GOLDEN ENTRY PRO: SELL | score={gz['score']:.1f} | منطقة ذهبية مؤكدة + Footprint")

            # --- Ultra Market Structure Entry ---
            ultra_entry = False
            if sig is None and ultra_ms:
                ms_bias = ultra_ms.get("bias", "neutral")
                fvg = ultra_ms.get("fvg", {})
                
                if ms_bias == "bull" and fvg.get("bull_near"):
                    if rf_ctx.get("buy_signal") and council_data["score_b"] > council_data["score_s"]:
                        sig = "buy"
                        ultra_entry = True
                        log_i(f"🏛 ULTRA MS ENTRY: BUY | Bull BOS + FVG + RF confirm")
                elif ms_bias == "bear" and fvg.get("bear_near"):
                    if rf_ctx.get("sell_signal") and council_data["score_s"] > council_data["score_b"]:
                        sig = "sell"
                        ultra_entry = True
                        log_i(f"🏛 ULTRA MS ENTRY: SELL | Bear BOS + FVG + RF confirm")

            # --- EDGE ALGO ENTRY ---
            edge_entry = False
            if sig is None and edge_setup and edge_setup.get("valid"):
                grade = edge_setup.get("grade", "weak")
                if grade in ["strong", "mid"]:
                    if edge_setup["side"] == "BUY":
                        sig = "buy"
                        edge_entry = True
                        log_i(f"🧠 EDGE ALGO ENTRY: BUY | grade={grade} | RR≈{edge_setup['rr1']:.2f}/{edge_setup['rr2']:.2f}/{edge_setup['rr3']:.2f}")
                    elif edge_setup["side"] == "SELL":
                        sig = "sell"
                        edge_entry = True
                        log_i(f"🧠 EDGE ALGO ENTRY: SELL | grade={grade} | RR≈{edge_setup['rr1']:.2f}/{edge_setup['rr2']:.2f}/{edge_setup['rr3']:.2f}")

            # Council Strong Entry (لو مفيش أي entry من اللي فوق)
            if sig is None and not cvd_entry and not smc_entry and not golden_entry and not ultra_entry and not edge_entry:
                if council_data["score_b"] > council_data["score_s"] and council_data["score_b"] >= 8.0:
                    sig = "buy"
                    entry_type = "COUNCIL_STRONG_BUY"
                elif council_data["score_s"] > council_data["score_b"] and council_data["score_s"] >= 8.0:
                    sig = "sell"
                    entry_type = "COUNCIL_STRONG_SELL"

            # ========== فلتر قوة الإشارة FINAL قبل أي دخول ==========
            if not STATE["open"] and sig and reason is None:
                # فلتر قوة الإشارة قبل أي دخول
                sig_side = "long" if sig == "buy" else "short"
                signal_strength = calculate_signal_strength(df, ind, sig_side)
                
                if signal_strength < SIGNAL_STRENGTH_MIN:
                    reason = f"weak_signal_strength({signal_strength:.1f}<{SIGNAL_STRENGTH_MIN})"
                    log_i(f"🚫 SKIP ENTRY | {sig.upper()} | ضعيف جداً | strength={signal_strength:.1f} < {SIGNAL_STRENGTH_MIN}")
                else:
                    allow_wait, wait_reason = wait_gate_allow(df, info)
                    if not allow_wait:
                        reason = wait_reason
                    else:
                        qty = compute_size(bal, px or info["price"])
                        if qty > 0:
                            ok = open_market_enhanced(sig, qty, px or info["price"])
                            if ok:
                                wait_for_next_signal_side = None
                                # تسجيل قرار المجلس المحسن مع جميع المعلومات
                                entry_type = entry_type or "GENERIC"
                                
                                adx_info = f" | ADX={ind.get('adx',0):.1f}" if ind.get('adx') else ""
                                atr_info = f" | ATR={ind.get('atr',0):.6f}" if ind.get('atr') else ""
                                rf_info = f" | RF dir={rf_ctx.get('dir', 0)}" if rf_ctx else ""
                                edge_info = f" | Edge grade={edge_setup.get('grade', 'N/A')}" if edge_setup else ""
                                
                                log_i(f"🎯 {entry_type} ULTRA ENHANCED DECISION: {sig.upper()} | "
                                      f"Score B/S: {council_data['score_b']:.1f}/{council_data['score_s']:.1f} | "
                                      f"Signal Strength: {signal_strength:.1f}{adx_info}{atr_info}{rf_info}{edge_info}")
                                for log_msg in council_data.get("logs", []):
                                    log_i(f"   - {log_msg}")
                        else:
                            reason = "qty<=0"
            
            loop_i += 1
            sleep_s = NEAR_CLOSE_S if time_to_candle_close(df) <= 10 else BASE_SLEEP
            time.sleep(sleep_s)
            
        except Exception as e:
            log_e(f"loop error: {e}\n{traceback.format_exc()}")
            logging.error(f"trade_loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

# استبدال الدوال الأساسية بالمحسنة
compute_candles = compute_enhanced_candles
council_votes_pro = council_votes_pro_enhanced
manage_after_entry = manage_after_entry_enhanced
open_market = open_market_enhanced
trade_loop = trade_loop_enhanced
decide_strategy_mode = decide_strategy_mode_enhanced
golden_zone_check = golden_zone_pro_analysis

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
        print(f"   🎯 ENTRY: ULTRA MARKET STRUCTURE + REAL RF + VWAP + SMC ADVANCED + COUNCIL PRO + GOLDEN ENTRY + ADX+ATR STOP-HUNT FILTERS + EDGE ALGO + CVD DIVERGENCE + ENHANCED FILTERS |  spread_bps={fmt(spread_bps,2)}")
        print(f"   ⏱️ closes_in ≈ {left_s}s")
        print("\n🧭 POSITION")
        bal_line = f"Balance={fmt(bal,2)}  Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x  CompoundPnL={fmt(compound_pnl)}  Eq~{fmt((bal or 0)+compound_pnl,2)}"
        print(colored(f"   {bal_line}", "yellow"))
        if STATE["open"]:
            lamp='🟩 LONG' if STATE['side']=='long' else '🟥 SHORT'
            print(f"   {lamp}  Entry={fmt(STATE['entry'])}  Qty={fmt(STATE['qty'],4)}  Bars={STATE['bars']}  Trail={fmt(STATE['trail'])}  BE={fmt(STATE['breakeven'])}")
            print(f"   🎯 TP_done={STATE['profit_targets_achieved']}  HP={fmt(STATE['highest_profit_pct'],2)}%  Profile={STATE.get('profit_profile', 'N/A')}")
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
    return f"✅ ULTRA Council PRO Bot v8.0 — {SYMBOL} {INTERVAL} — {mode} — Ultra Market Structure + Real RF + VWAP + SMC Advanced + Elliott Waves + Stop Hunt AI + ADX+ATR Filters + Edge Algo + CVD Divergence + ENHANCED FILTERS + SMART SIGNAL/PROFIT SYSTEM"

@app.route("/metrics")
def metrics():
    edge_setup = STATE.get("edge_setup", {})
    cvd_div = STATE.get("cvd_div_snapshot", {})
    return jsonify({
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "state": STATE, "compound_pnl": compound_pnl,
        "entry_mode": "ULTRA_MARKET_STRUCTURE+REAL_RF+VWAP+SMC_ADVANCED_COUNCIL_PRO_GOLDEN_ENHANCED_ADX_ATR_FILTERS+EDGE_ALGO+CVD_DIVERGENCE+ENHANCED_FILTERS+SMART_SIGNAL/PROFIT", 
        "wait_for_next_signal": wait_for_next_signal_side,
        "guards": {"max_spread_bps": MAX_SPREAD_BPS, "final_chunk_qty": FINAL_CHUNK_QTY},
        "vwap_strategy": {
            "enabled": VWAP_ENABLED,
            "scalp_band_bps": VWAP_SCALP_BAND_BPS,
            "trend_band_bps": VWAP_TREND_BAND_BPS
        },
        "smc_enabled": SMC_ENABLED,
        "elliott_enabled": ELLIOTT_WAVE_ENABLED,
        "stop_hunt_filters": {
            "adx_min": STOP_HUNT_ADX_MIN,
            "adx_max": STOP_HUNT_ADX_MAX,
            "atr_mult_min": STOP_HUNT_ATR_MULT_MIN,
            "wick_ratio": STOP_HUNT_WICK_RATIO,
            "distance_atr": STOP_HUNT_DISTANCE_ATR,
            "sl_atr_mult": STOP_HUNT_SL_ATR_MULT
        },
        "ultra_market_structure": {
            "enabled": True,
            "description": "BOS/CHoCH + FVG + Premium/Discount + Liquidity Grabs"
        },
        "real_rf": {
            "enabled": True,
            "period": RF_PERIOD,
            "mult": RF_MULT,
            "description": "Pine Exact Range Filter"
        },
        "edge_algo": {
            "enabled": True,
            "current_setup": edge_setup if edge_setup else None,
            "profile": STATE.get("profit_profile", "SCALP_STRICT")
        },
        "cvd_divergence": {
            "enabled": True,
            "current": cvd_div if cvd_div else None,
            "description": "TradingFinder-style CVD Divergence Oscillator"
        },
        "enhanced_filters": {
            "cvd_adx_min": CVD_ADX_MIN,
            "cvd_score_min": CVD_SCORE_MIN,
            "smc_confidence_min": SMC_CONFIDENCE_MIN,
            "signal_strength_min": SIGNAL_STRENGTH_MIN,
            "description": "Enhanced entry filters for higher quality trades"
        },
        "smart_signal_profit": {
            "min_signal_for_entry": MIN_SIGNAL_FOR_ENTRY,
            "hold_after_tp1_min_strength": HOLD_AFTER_TP1_MIN_STRENGTH,
            "hold_after_tp1_min_adx": HOLD_AFTER_TP1_MIN_ADX,
            "hold_after_tp1_extra_boost": HOLD_AFTER_TP1_EXTRA_BOOST,
            "very_strong_tp_boost": VERY_STRONG_TP_BOOST,
            "description": "RF Master + Smart Profit AI + HOLD-TP Logic"
        }
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True, "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"], "side": STATE["side"], "qty": STATE["qty"],
        "compound_pnl": compound_pnl, "timestamp": datetime.utcnow().isoformat(),
        "entry_mode": "ULTRA_MARKET_STRUCTURE+REAL_RF+VWAP+SMC_ADVANCED_COUNCIL_PRO_GOLDEN_ENHANCED_ADX_ATR_FILTERS+EDGE_ALGO+CVD_DIVERGENCE+ENHANCED_FILTERS+SMART_SIGNAL/PROFIT", 
        "wait_for_next_signal": wait_for_next_signal_side,
        "edge_profile": STATE.get("profit_profile", "SCALP_STRICT"),
        "enhanced_filters_active": True,
        "smart_signal_active": True
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
    log_banner("ULTRA ENHANCED BOT INIT WITH EDGE ALGO + CVD DIVERGENCE + ENHANCED FILTERS + SMART SIGNAL/PROFIT")
    state = load_state() or {}
    state.setdefault("in_position", False)

    if RESUME_ON_RESTART:
        try:
            state = resume_open_position(ex, SYMBOL, state)
        except Exception as e:
            log_w(f"resume error: {e}\n{traceback.format_exc()}")

    verify_execution_environment()

    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'}  •  {SYMBOL}  •  {INTERVAL}", "yellow"))
    print(colored(f"RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x  •  ULTRA_COUNCIL_PRO=ALL_ENGINES", "yellow"))
    print(colored(f"ULTRA MARKET STRUCTURE: BOS/CHoCH + FVG + Premium/Discount + Liquidity Grabs", "yellow"))
    print(colored(f"REAL RF FILTER: Pine Exact + Live Signals + {RF_PERIOD} period + {RF_MULT} mult", "yellow"))
    print(colored(f"REAL VWAP: Session-based Fair Value Axis", "yellow"))
    print(colored(f"GOLDEN ENTRY PRO: score≥{GOLDEN_ENTRY_SCORE} | ADX≥{GOLDEN_ENTRY_ADX}", "yellow"))
    print(colored(f"ADVANCED SMC: Supply/Demand + Order Blocks + FVG + Breaker Blocks + Spring Patterns", "yellow"))
    print(colored(f"STOP-HUNT FILTERS: ADX={STOP_HUNT_ADX_MIN}-{STOP_HUNT_ADX_MAX} | ATR_MULT≥{STOP_HUNT_ATR_MULT_MIN} | WICK≥{STOP_HUNT_WICK_RATIO*100}%", "yellow"))
    print(colored(f"ENHANCED CANDLES: SMC Patterns + Wick exhaustion + Golden reversal", "yellow"))
    print(colored(f"FOOTPRINT ANALYSIS: Volume spikes + Absorption detection", "yellow"))
    print(colored(f"SMART PROFIT AI: Dynamic profit taking + Signal strength + HOLD-TP Logic", "yellow"))
    print(colored(f"VWAP STRATEGY: SCALP(near {VWAP_SCALP_BAND_BPS}bps) | TREND(far {VWAP_TREND_BAND_BPS}bps)", "yellow"))
    print(colored(f"🧠 EDGE ALGO ENGINE: RR Zones + Setup Quality + Dynamic Profit Profiles (TREND_3TP/MID_2TP/SCALP_STRICT)", "yellow"))
    print(colored(f"📊 CVD DIVERGENCE OSCILLATOR: TradingFinder-style divergence detection", "yellow"))
    print(colored(f"🔒 ENHANCED ENTRY FILTERS:", "yellow"))
    print(colored(f"   • CVD Divergence: ADX≥{CVD_ADX_MIN} + Council Score≥{CVD_SCORE_MIN}", "yellow"))
    print(colored(f"   • SMC Confidence: ≥{SMC_CONFIDENCE_MIN} (was 0.7)", "yellow"))
    print(colored(f"   • Signal Strength: ≥{SIGNAL_STRENGTH_MIN} for any entry", "yellow"))
    print(colored(f"🧠 SMART SIGNAL/PROFIT SYSTEM:", "yellow"))
    print(colored(f"   • RF Master Entry: تقييم الإشارة من كل المحركات", "yellow"))
    print(colored(f"   • HOLD-TP Logic: رفع الأهداف بعد TP1 للصفقات القوية", "yellow"))
    print(colored(f"   • Signal Filter: MIN_SIGNAL_FOR_ENTRY={MIN_SIGNAL_FOR_ENTRY}", "yellow"))
    print(colored(f"EXECUTION: {'ACTIVE' if EXECUTE_ORDERS and not DRY_RUN else 'SIMULATION'}", "yellow"))
    
    logging.info("ULTRA enhanced SMC service starting with all engines + Edge Algo + CVD Divergence + Enhanced Filters + Smart Signal/Profit…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    
    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)

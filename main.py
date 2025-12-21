# -*- coding: utf-8 -*-
"""
RF Futures Bot — RF-LIVE ONLY (BingX Perp via CCXT)
• Council PRO Unified Decision System with Candles & Golden Entry
• Golden Entry + Golden Reversal + Wick Exhaustion
• Dynamic TP ladder + Breakeven + ATR-trailing
• Smart Exit Management + Wait-for-next-signal
• Professional Logging & Dashboard
• FALCON STYLE ADDON: TP1/TP2/TP3 + Early Exit + Box Detector
• TREND BIRTH ENGINE: Liquidity Sweep → BOS/CHoCH → Momentum Flip → Displacement → OB/FVG → Retest → Entry
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
BOT_VERSION = "DOGE Council PRO v5.0 — Trend Birth Engine + Box-First + FALCON STYLE"
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

ENTRY_RF_ONLY = False  # Now using Council decision
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
EARLY_FAIL_PNL_PCT = -0.15   # اقفل بدري لو -0.15% خلال أول 3 شمعات
TIME_STOP_BARS = 10          # لو الصفقة ما اتحركتش خلال 10 شمعات
TIME_STOP_MIN_PNL_PCT = 0.05 # لو لسه أقل من +0.05% بعد TIME_STOP_BARS → اقفل

REENTRY_COOLDOWN_BARS = 2

# ===== SCALP TARGET BOOST (higher TP for scalps) =====
SCALP_TP1_BOOST_PCT = 0.65   # بدل 0.40 مثلاً
SCALP_TP2_BOOST_PCT = 1.10
SCALP_TP3_BOOST_PCT = 1.80

# =================== BOX-FIRST ENTRY (Falcon-style) ===================
BOX_ENTRY_ENABLED = True

BOX_LOOKBACK = 80              # عدد الشموع اللي بنستخرج منها نطاق البوكس
BOX_BAND_PCT = 0.12            # عرض البوكس كنسبة من الرينج (12%)
BOX_TOUCH_BPS = 6              # لازم السعر يلمس/يقرب من البوكس (بالـ bps)

CONFIRM_MIN_BODY_ATR = 0.35    # شمعة تأكيد: جسم >= 0.35 * ATR
CONFIRM_MIN_VOL_MULT = 1.10    # أو حجم >= 1.10 * MA20

# Council Guard: يمنع فقط الحالات "الخطيرة"
CHOP_ADX_MAX = 15.0            # لو ADX أقل من كده غالبًا Chop
CHOP_RSI_BAND = 6.0            # RSI قريب من 50 (+/- 6) = تذبذب ممل

DI_STRONG_GAP = 6.0            # فرق DI قوي ضد اتجاهك = منع

# هدفك 3-5 سكالب يوميًا (من غير سبام)
MAX_SCALP_TRADES_PER_DAY = 5
MIN_SCALP_TRADES_HINT = 3      # "Hint" فقط (مش إجبار)

# =================== TREND BIRTH ENGINE v1 (FINAL WITH DISPLACEMENT) ===================
TBE_ENABLED = True
TBE_SWEEP_LOOKBACK = 25
TBE_SWEEP_WICK_MIN = 0.50
TBE_SHIFT_SWING_W = 3
TBE_ADX_MIN = 18.0
TBE_DI_GAP_MIN = 4.0
TBE_RSI_MID = 50.0
TBE_RSI_FLIP_MARGIN = 2.0
TBE_OB_DISPLACE_ATR = 1.2
TBE_RETEST_PAD_PCT = 0.15
TBE_DISPLACE_ATR_MULT = 1.6      # ✅ بوابة الاندفاع
TBE_DISPLACE_LOOKBACK = 6        # ✅ نبحث عن شمعة اندفاع خلال آخر 6 شموع
# =====================================================================================

# =================== FLOW PRESSURE MODULE (CCXT) ===================
FLOW_ENABLED = True
FLOW_TRADES_LIMIT = 200          # عدد الصفقات الأخيرة
FLOW_ORDERBOOK_DEPTH = 20        # عمق الأوامر
FLOW_WALL_MULT = 3.0             # wall = أكبر من متوسط * 3
FLOW_DELTA_MIN = 0.12            # حد أدنى delta_ratio
FLOW_OBI_MIN = 0.10              # حد أدنى imbalance
FLOW_CACHE_S = 2                 # كاش ثانيتين لتخفيف ضغط API

# =================== SCALP PRO (HUNT ENTRY) ===================
SCALP_PRO_ENABLED = True

SCALP_BOX_LOOKBACK = 90
SCALP_TOUCH_BPS = 10

SCALP_REQUIRE_SWEEP = True
SCALP_SWEEP_LOOKBACK = 20
SCALP_SWEEP_WICK_MIN = 0.45

SCALP_DISPLACE_ATR_MULT = 1.2   # أصغر من Trend Birth (1.6)
SCALP_DISPLACE_LOOKBACK = 6

SCALP_RETEST_PAD_PCT = 0.20
SCALP_MIN_ROOM_BPS = 45         # لازم مساحة للربح لحد البوكس المعاكس

SCALP_CHOP_ADX_MAX = 15.0
SCALP_CHOP_RSI_BAND = 5.0
# ==============================================================

# =================== SMART SCALP SYSTEM v2 ===================
DOGE_TICK = 0.0001

# معايير الدخول
SCALP_MIN_POINTS = 6
SCALP_MIN_MOVE_PCT_FIXED = 0.50
SCALP_ATR_EXPECT_MULT = 0.8

# Zone-First
ZONE_FIRST_ENABLED = True
ZONE_BOX_LOOKBACK = 90
ZONE_TOUCH_BPS = 10

# ترقية سكالب -> ترند
UPGRADE_ENABLE = True
UPGRADE_MIN_MOVE_PCT_FIXED = 0.50
UPGRADE_ATR_MULT = 0.9
UPGRADE_ADX_MIN = 20.0
UPGRADE_DI_DOM_MIN = 5.0
UPGRADE_IMB_LONG = 1.12
UPGRADE_IMB_SHORT = 0.90

# إدارة المخاطر
BE_BUFFER_PCT = 0.05
TRAIL_ATR_MULT = 1.0
TRAIL_RATCHET = True
MIN_BOX_EXIT_SCALP = 0.35
MIN_BOX_EXIT_TREND = 0.90

# فلتر الجدوى
SCALP_FEASIBILITY_ENABLED = True
SCALP_MIN_EXPECTED_POINTS = 6
SCALP_MIN_ROOM_TO_BOX = 8
SCALP_REQUIRED_DISPLACEMENT = True
SCALP_MAX_SLIPPAGE_POINTS = 2

# TP ديناميكي
SCALP_TP_DYNAMIC = True
SCALP_TP_MIN_PCT = 0.50
SCALP_TP_MAX_PCT = 1.20

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

# =================== CANDLES MODULE ===================
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

def compute_candles(df):
    """
    يرجّع: buy/sell + score لكل اتجاه + فتائل كبيرة (exhaustion) + tags
    يعمل على آخر شمعة مغلقة df.iloc[-2]
    """
    if len(df) < 5:
        return {"buy":False,"sell":False,"score_buy":0.0,"score_sell":0.0,
                "wick_up_big":False,"wick_dn_big":False,"doji":False,"pattern":None}

    o1,h1,l1,c1 = float(df["open"].iloc[-2]), float(df["high"].iloc[-2]), float(df["low"].iloc[-2]), float(df["close"].iloc[-2])
    o0,h0,l0,c0 = float(df["open"].iloc[-3]), float(df["high"].iloc[-3]), float(df["low"].iloc[-3]), float(df["close"].iloc[-3])

    strength_b = strength_s = 0.0
    tags = []

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

    # فتائل كبيرة = إرهاق
    rng1 = _rng(h1,l1); up = _upper_wick(h1,o1,c1); dn = _lower_wick(l1,o1,c1)
    wick_up_big = (up >= 1.2*_body(o1,c1)) and (up >= 0.4*rng1)
    wick_dn_big = (dn >= 1.2*_body(o1,c1)) and (dn >= 0.4*rng1)

    if is_doji:  # تخفيف ثقة
        strength_b *= 0.8; strength_s *= 0.8

    return {
        "buy": strength_b>0, "sell": strength_s>0,
        "score_buy": round(strength_b,2), "score_sell": round(strength_s,2),
        "wick_up_big": bool(wick_up_big), "wick_dn_big": bool(wick_dn_big),
        "doji": bool(is_doji), "pattern": ",".join(tags) if tags else None
    }

# =================== FALCON: BOX DETECTOR ===================
def detect_simple_boxes(df, lookback=60):
    """
    Demand/Supply boxes بسيطة (مش مثالية بس عملية):
    - Demand: أقل لو + bounce واضح
    - Supply: أعلى هاي + رفض واضح
    """
    if df is None or len(df) < lookback:
        return {"ok": False}

    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    window = min(lookback, len(df))
    hi = float(h.tail(window).max())
    lo = float(l.tail(window).min())
    px = float(c.iloc[-1])

    # عرض صندوق صغير كنسبة من الرينج
    rng = max(hi - lo, 1e-9)
    band = rng * BOX_BAND_PCT  # 12% من الرينج (قابل للضبط)

    demand = {"low": lo, "high": lo + band}
    supply = {"low": hi - band, "high": hi}

    # هل السعر داخل/قريب من الصندوق؟
    in_demand = demand["low"] <= px <= demand["high"]
    in_supply = supply["low"] <= px <= supply["high"]

    return {
        "ok": True,
        "px": px,
        "demand": demand,
        "supply": supply,
        "in_demand": in_demand,
        "in_supply": in_supply,
    }

# =================== BOX-FIRST ENTRY HELPER FUNCTIONS ===================
def _day_key_utc():
    return datetime.utcnow().strftime("%Y-%m-%d")

def reset_daily_scalp_counter():
    k = _day_key_utc()
    if STATE.get("day_key") != k:
        STATE["day_key"] = k
        STATE["scalp_trades_today"] = 0

def _near_bps(px, level):
    if not level:
        return 999999
    return abs(float(px) - float(level)) / float(level) * 10000.0

def candle_confirmation(df, ind, direction: str) -> (bool, str):
    """
    Confirmation بسيط وفعال:
    - جسم شمعة >= نسبة من ATR
    - أو حجم >= MA20 * multiplier
    - ومع اتجاه إغلاق مناسب (bull/bear)
    """
    if df is None or len(df) < 25:
        return False, "no_df"

    last = df.iloc[-1]
    body = abs(float(last["close"]) - float(last["open"]))
    atr = float(ind.get("atr", 0.0) or 0.0)
    if atr <= 0:
        return False, "no_atr"

    vol = float(last.get("volume", 0.0) or 0.0)
    vma = float(df["volume"].astype(float).rolling(20).mean().iloc[-1] or 0.0)

    bull_close = float(last["close"]) > float(last["open"])
    bear_close = float(last["close"]) < float(last["open"])

    body_ok = (body >= CONFIRM_MIN_BODY_ATR * atr)
    vol_ok = (vma > 0 and vol >= CONFIRM_MIN_VOL_MULT * vma)

    if direction == "buy":
        if (bull_close and (body_ok or vol_ok)):
            return True, f"bull_confirm(body_ok={body_ok}, vol_ok={vol_ok})"
        return False, f"no_bull_confirm(body_ok={body_ok}, vol_ok={vol_ok})"

    if direction == "sell":
        if (bear_close and (body_ok or vol_ok)):
            return True, f"bear_confirm(body_ok={body_ok}, vol_ok={vol_ok})"
        return False, f"no_bear_confirm(body_ok={body_ok}, vol_ok={vol_ok})"

    return False, "bad_dir"

def council_guard_for_box_entry(ind: dict, direction: str) -> (bool, str):
    """
    Council هنا مش شرط دخول خانق.
    هو فقط يمنع الحالات اللي بتعمل خسارة:
    - Chop واضح
    - DI قوي ضدك (ترند عكسي قوي)
    """
    adx = float(ind.get("adx", 0.0) or 0.0)
    rsi = float(ind.get("rsi", 50.0) or 50.0)
    di_p = float(ind.get("plus_di", 0.0) or 0.0)
    di_m = float(ind.get("minus_di", 0.0) or 0.0)

    # Chop filter
    if adx <= CHOP_ADX_MAX and abs(rsi - 50.0) <= CHOP_RSI_BAND:
        return False, f"chop(adx={adx:.1f}, rsi={rsi:.1f})"

    # Strong DI against direction
    if direction == "buy" and (di_m - di_p) >= DI_STRONG_GAP and adx >= 18:
        return False, f"di_against_buy(di- {di_m:.1f} > di+ {di_p:.1f})"
    if direction == "sell" and (di_p - di_m) >= DI_STRONG_GAP and adx >= 18:
        return False, f"di_against_sell(di+ {di_p:.1f} > di- {di_m:.1f})"

    return True, "ok"

def box_first_signal(df, ind, px: float) -> (str, str):
    """
    Box-First:
    - Buy من demand + confirmation
    - Sell من supply + confirmation
    """
    if not BOX_ENTRY_ENABLED:
        return None, "box_entry_disabled"

    boxes = detect_simple_boxes(df, lookback=BOX_LOOKBACK)
    if not boxes.get("ok"):
        return None, "no_boxes"

    demand = boxes.get("demand") or {}
    supply = boxes.get("supply") or {}

    d_low, d_high = float(demand.get("low", 0)), float(demand.get("high", 0))
    s_low, s_high = float(supply.get("low", 0)), float(supply.get("high", 0))

    # قرب السعر من حدود البوكس (Touch)
    near_d = (_near_bps(px, d_high) <= BOX_TOUCH_BPS) or (d_low <= px <= d_high)
    near_s = (_near_bps(px, s_low)  <= BOX_TOUCH_BPS) or (s_low <= px <= s_high)

    # Buy from demand
    if near_d:
        ok, why = candle_confirmation(df, ind, "buy")
        if not ok:
            return None, f"demand_touch_no_confirm({why})"
        allow, guard = council_guard_for_box_entry(ind, "buy")
        if not allow:
            return None, f"demand_blocked({guard})"
        return "buy", "BOX_DEMAND+CONFIRM"

    # Sell from supply
    if near_s:
        ok, why = candle_confirmation(df, ind, "sell")
        if not ok:
            return None, f"supply_touch_no_confirm({why})"
        allow, guard = council_guard_for_box_entry(ind, "sell")
        if not allow:
            return None, f"supply_blocked({guard})"
        return "sell", "BOX_SUPPLY+CONFIRM"

    return None, "no_box_touch"

# =================== TREND BIRTH ENGINE v1 FUNCTIONS ===================
def tbe_wick_ratios(o,h,l,c):
    rng  = max(h-l, 1e-9)
    upper = h - max(o,c)
    lower = min(o,c) - l
    body  = abs(c-o)
    return upper/rng, lower/rng, body/rng

def tbe_find_swings(df, w=3):
    highs = df["high"].astype(float).values
    lows  = df["low"].astype(float).values
    sh, sl = [], []
    for i in range(w, len(df)-w):
        if highs[i] == max(highs[i-w:i+w+1]):
            sh.append((i, highs[i]))
        if lows[i] == min(lows[i-w:i+w+1]):
            sl.append((i, lows[i]))
    return sh, sl

def tbe_in_zone(px, zone, pad_pct=0.15):
    if not zone:
        return False
    lo, hi = float(zone["low"]), float(zone["high"])
    pad = (hi - lo) * pad_pct
    return (lo - pad) <= float(px) <= (hi + pad)

def tbe_detect_sweep_low(df, lookback=25, wick_min=0.50):
    if len(df) < lookback + 2:
        return False, {}
    last = df.iloc[-1]
    prev_low = float(df["low"].astype(float).iloc[-lookback-1:-1].min())
    o,h,l,c = map(float, (last.open, last.high, last.low, last.close))
    uw,lw,br = tbe_wick_ratios(o,h,l,c)
    ok = (l < prev_low) and (c > prev_low) and (lw >= wick_min)
    return ok, {"prev_low": prev_low, "lw": lw}

def tbe_detect_sweep_high(df, lookback=25, wick_min=0.50):
    if len(df) < lookback + 2:
        return False, {}
    last = df.iloc[-1]
    prev_high = float(df["high"].astype(float).iloc[-lookback-1:-1].max())
    o,h,l,c = map(float, (last.open, last.high, last.low, last.close))
    uw,lw,br = tbe_wick_ratios(o,h,l,c)
    ok = (h > prev_high) and (c < prev_high) and (uw >= wick_min)
    return ok, {"prev_high": prev_high, "uw": uw}

def tbe_detect_shift_up(df, w=3):
    sh, sl = tbe_find_swings(df, w=w)
    if not sh:
        return False, {}
    last_swing_high = float(sh[-1][1])
    c = float(df["close"].astype(float).iloc[-1])
    return (c > last_swing_high), {"swing_high": last_swing_high}

def tbe_detect_shift_down(df, w=3):
    sh, sl = tbe_find_swings(df, w=w)
    if not sl:
        return False, {}
    last_swing_low = float(sl[-1][1])
    c = float(df["close"].astype(float).iloc[-1])
    return (c < last_swing_low), {"swing_low": last_swing_low}

def tbe_momentum_flip_buy(ind, flow=None):
    adx  = float(ind.get("adx",0) or 0)
    di_p = float(ind.get("plus_di",0) or 0)
    di_m = float(ind.get("minus_di",0) or 0)
    rsi  = float(ind.get("rsi",50) or 50)
    if adx < TBE_ADX_MIN: return False, f"adx<{TBE_ADX_MIN}"
    if (di_p - di_m) < TBE_DI_GAP_MIN: return False, f"di_gap<{TBE_DI_GAP_MIN}"
    if rsi < (TBE_RSI_MID + TBE_RSI_FLIP_MARGIN): return False, "rsi_not_bullish_flip"
    
    # ✅ FLOW PRESSURE CHECK
    if flow and flow.get("ok"):
        dr = float(flow["delta"]["delta_ratio"])
        obi = float(flow["obi"]["obi"])
        if dr < FLOW_DELTA_MIN or obi < FLOW_OBI_MIN:
            return False, f"flow_weak(delta_ratio={dr:.2f}, obi={obi:.2f})"
    return True, "mom_flip_ok"

def tbe_momentum_flip_sell(ind, flow=None):
    adx  = float(ind.get("adx",0) or 0)
    di_p = float(ind.get("plus_di",0) or 0)
    di_m = float(ind.get("minus_di",0) or 0)
    rsi  = float(ind.get("rsi",50) or 50)
    if adx < TBE_ADX_MIN: return False, f"adx<{TBE_ADX_MIN}"
    if (di_m - di_p) < TBE_DI_GAP_MIN: return False, f"di_gap<{TBE_DI_GAP_MIN}"
    if rsi > (TBE_RSI_MID - TBE_RSI_FLIP_MARGIN): return False, "rsi_not_bearish_flip"
    
    # ✅ FLOW PRESSURE CHECK
    if flow and flow.get("ok"):
        dr = float(flow["delta"]["delta_ratio"])
        obi = float(flow["obi"]["obi"])
        # للـ sell نريد delta سلبي و OBI سلبي
        if dr > -FLOW_DELTA_MIN or obi > -FLOW_OBI_MIN:
            return False, f"flow_weak(delta_ratio={dr:.2f}, obi={obi:.2f})"
    return True, "mom_flip_ok"

def tbe_detect_bullish_fvg(df):
    if len(df) < 4: return False, {}
    a = df.iloc[-3]
    c = df.iloc[-1]
    a_low  = float(a.low)
    c_high = float(c.high)
    ok = a_low > c_high
    if not ok: return False, {}
    return True, {"low": c_high, "high": a_low}

def tbe_detect_bearish_fvg(df):
    if len(df) < 4: return False, {}
    a = df.iloc[-3]
    c = df.iloc[-1]
    a_high = float(a.high)
    c_low  = float(c.low)
    ok = a_high < c_low
    if not ok: return False, {}
    return True, {"low": a_high, "high": c_low}

def tbe_detect_bullish_ob(df, ind):
    if len(df) < 10: return False, {}
    atr = float(ind.get("atr",0) or 0)
    if atr <= 0: return False, {}
    # آخر شمعة حمراء قبل اندفاع صاعد قوي
    for k in range(2, 8):
        b   = df.iloc[-k]
        nxt = df.iloc[-k+1]
        if float(b.close) < float(b.open):  # bearish candle
            body = abs(float(nxt.close) - float(nxt.open))
            if float(nxt.close) > float(nxt.open) and body >= (TBE_OB_DISPLACE_ATR * atr):
                lo = min(float(b.open), float(b.close))
                hi = max(float(b.open), float(b.close))
                return True, {"low": lo, "high": hi, "k": k}
    return False, {}

def tbe_detect_bearish_ob(df, ind):
    if len(df) < 10: return False, {}
    atr = float(ind.get("atr",0) or 0)
    if atr <= 0: return False, {}
    # آخر شمعة خضراء قبل اندفاع هابط قوي
    for k in range(2, 8):
        b   = df.iloc[-k]
        nxt = df.iloc[-k+1]
        if float(b.close) > float(b.open):  # bullish candle
            body = abs(float(nxt.close) - float(nxt.open))
            if float(nxt.close) < float(nxt.open) and body >= (TBE_OB_DISPLACE_ATR * atr):
                lo = min(float(b.open), float(b.close))
                hi = max(float(b.open), float(b.close))
                return True, {"low": lo, "high": hi, "k": k}
    return False, {}

def tbe_displacement_ok(df, ind, side, lookback=6, mult=1.6):
    """
    Displacement Gate: شمعة اندفاع قوية (جسم كبير مقارنة بالـ ATR)
    - للـ BUY: شمعة صاعدة قوية (جسم >= mult * ATR)
    - للـ SELL: شمعة هابطة قوية (جسم >= mult * ATR)
    """
    atr = float(ind.get("atr", 0) or 0)
    if atr <= 0 or len(df) < lookback + 2:
        return False, {"why": "no_atr_or_not_enough_bars"}

    tail = df.tail(lookback).copy()
    for idx, row in tail.iterrows():
        o = float(row.open); c = float(row.close)
        body = abs(c - o)
        if body < (mult * atr):
            continue

        # اتجاه الاندفاع
        if side == "buy" and c > o:
            return True, {"body": body, "atr": atr, "mult": mult, "dir": "bull", "idx": idx}
        if side == "sell" and c < o:
            return True, {"body": body, "atr": atr, "mult": mult, "dir": "bear", "idx": idx}

    return False, {"why": "no_displacement", "atr": atr, "mult": mult}

def trend_birth_buy_signal(df, ind, px, flow=None):
    """✅ الترتيب العلمي: Sweep → Shift → Momentum → Displacement → OB/FVG → Retest → Entry"""
    if not TBE_ENABLED: 
        return None, "tbe_disabled"
    
    # 1) Liquidity Sweep (Low)
    ok_sweep, sweep_data = tbe_detect_sweep_low(df, TBE_SWEEP_LOOKBACK, TBE_SWEEP_WICK_MIN)
    if not ok_sweep: 
        return None, "no_sweep_low"
    
    # 2) Structure Shift (BOS/CHoCH)
    ok_shift, shift_data = tbe_detect_shift_up(df, TBE_SHIFT_SWING_W)
    if not ok_shift: 
        return None, "no_structure_shift_up"
    
    # 3) Momentum Flip
    ok_mom, why_mom = tbe_momentum_flip_buy(ind, flow)
    if not ok_mom: 
        return None, f"no_momentum_flip({why_mom})"
    
    # ✅ 4) DISPLACEMENT GATE (شمعة اندفاع مؤسسي) — هنا بالضبط
    ok_disp, disp_data = tbe_displacement_ok(df, ind, "buy", 
                                           lookback=TBE_DISPLACE_LOOKBACK, 
                                           mult=TBE_DISPLACE_ATR_MULT)
    if not ok_disp:
        return None, f"no_displacement({disp_data.get('why','')})"
    
    # 5) OB/FVG (Order Block / Fair Value Gap)
    ok_ob, ob = tbe_detect_bullish_ob(df, ind)
    ok_fvg, fvg = tbe_detect_bullish_fvg(df)
    
    zone, ztag = (ob, "OB") if ok_ob else ((fvg, "FVG") if ok_fvg else (None, None))
    if not zone: 
        return None, "no_ob_or_fvg"
    
    # 6) Retest (مع مساحة صغيرة)
    if not tbe_in_zone(px, zone, TBE_RETEST_PAD_PCT): 
        return None, f"waiting_retest_{ztag}"
    
    # 7) Confirmation Candle
    ok_conf, cwhy = candle_confirmation(df, ind, "buy")
    if not ok_conf: 
        return None, f"retest_no_confirm({cwhy})"
    
    # ✅ الإشارة جاهزة
    return "buy", f"TBE_BUY sweep→shift↑→mom→DISP({disp_data.get('dir')}@{disp_data.get('body'):.3f})→retest_{ztag}→confirm"

def trend_birth_sell_signal(df, ind, px, flow=None):
    """✅ نفس الترتيب للـ SELL"""
    if not TBE_ENABLED: 
        return None, "tbe_disabled"
    
    # 1) Liquidity Sweep (High)
    ok_sweep, sweep_data = tbe_detect_sweep_high(df, TBE_SWEEP_LOOKBACK, TBE_SWEEP_WICK_MIN)
    if not ok_sweep: 
        return None, "no_sweep_high"
    
    # 2) Structure Shift Down
    ok_shift, shift_data = tbe_detect_shift_down(df, TBE_SHIFT_SWING_W)
    if not ok_shift: 
        return None, "no_structure_shift_down"
    
    # 3) Momentum Flip
    ok_mom, why_mom = tbe_momentum_flip_sell(ind, flow)
    if not ok_mom: 
        return None, f"no_momentum_flip({why_mom})"
    
    # ✅ 4) DISPLACEMENT GATE
    ok_disp, disp_data = tbe_displacement_ok(df, ind, "sell", 
                                           lookback=TBE_DISPLACE_LOOKBACK, 
                                           mult=TBE_DISPLACE_ATR_MULT)
    if not ok_disp:
        return None, f"no_displacement({disp_data.get('why','')})"
    
    # 5) OB/FVG
    ok_ob, ob = tbe_detect_bearish_ob(df, ind)
    ok_fvg, fvg = tbe_detect_bearish_fvg(df)
    
    zone, ztag = (ob, "OB") if ok_ob else ((fvg, "FVG") if ok_fvg else (None, None))
    if not zone: 
        return None, "no_ob_or_fvg"
    
    # 6) Retest
    if not tbe_in_zone(px, zone, TBE_RETEST_PAD_PCT): 
        return None, f"waiting_retest_{ztag}"
    
    # 7) Confirmation
    ok_conf, cwhy = candle_confirmation(df, ind, "sell")
    if not ok_conf: 
        return None, f"retest_no_confirm({cwhy})"
    
    return "sell", f"TBE_SELL sweep→shift↓→mom→DISP({disp_data.get('dir')}@{disp_data.get('body'):.3f})→retest_{ztag}→confirm"

# =================== FALCON: TP PLAN ===================
def falcon_plan(entry_px: float, side: str, mode: str, atr: float):
    atr = float(atr or 0.0)
    if atr <= 0:
        atr = entry_px * 0.002  # fallback 0.2%

    if mode == "scalp":
        # ✅ استخدام أهداف أعلى للسكالب
        if side == "long":
            tp1 = entry_px * (1 + SCALP_TP1_BOOST_PCT/100.0)
            tp2 = entry_px * (1 + SCALP_TP2_BOOST_PCT/100.0)
            tp3 = entry_px * (1 + SCALP_TP3_BOOST_PCT/100.0)
        else:
            tp1 = entry_px * (1 - SCALP_TP1_BOOST_PCT/100.0)
            tp2 = entry_px * (1 - SCALP_TP2_BOOST_PCT/100.0)
            tp3 = entry_px * (1 - SCALP_TP3_BOOST_PCT/100.0)
        return {"tp1": tp1, "tp2": tp2, "tp3": tp3}

    # trend
    tp1 = entry_px * (1 + TREND_TP1/100.0) if side == "long" else entry_px * (1 - TREND_TP1/100.0)
    tp2 = entry_px + (FALCON_TP2_ATR_MULT * atr) if side == "long" else entry_px - (FALCON_TP2_ATR_MULT * atr)
    tp3 = entry_px + (FALCON_TP3_ATR_MULT * atr) if side == "long" else entry_px - (FALCON_TP3_ATR_MULT * atr)
    return {"tp1": tp1, "tp2": tp2, "tp3": tp3}

# =================== EXECUTION VERIFICATION ===================
def verify_execution_environment():
    """التحقق من بيئة التنفيذ عند الإقلاع"""
    print(f"⚙️ EXECUTION ENVIRONMENT", flush=True)
    print(f"🔧 EXECUTE_ORDERS: {EXECUTE_ORDERS} | SHADOW_MODE: {SHADOW_MODE_DASHBOARD} | DRY_RUN: {DRY_RUN}", flush=True)
    print(f"🎯 GOLDEN ENTRY: score={GOLDEN_ENTRY_SCORE} | ADX={GOLDEN_ENTRY_ADX}", flush=True)
    print(f"📈 CANDLES: Full patterns + Wick exhaustion + Golden reversal", flush=True)
    print(f"🦅 FALCON STYLE: TP1/2/3 + Box Exit + Early Fail", flush=True)
    print(f"🚀 TREND BIRTH ENGINE: Liquidity Sweep → BOS/CHoCH → Momentum → Displacement → OB/FVG → Retest", flush=True)
    
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

def golden_zone_check(df, ind=None, side_hint=None):
    """اكتشاف المناطق الذهبية (فيبو 0.618-0.786) مع تأكيدات"""
    if len(df) < 30:
        return {"ok": False, "score": 0.0, "zone": None, "reasons": ["short_df"]}
    
    try:
        h = df['high'].astype(float)
        l = df['low'].astype(float)
        c = df['close'].astype(float)
        v = df['volume'].astype(float)
        
        swing_hi = h.rolling(10).max().iloc[-1]
        swing_lo = l.rolling(10).min().iloc[-1]
        
        if swing_hi <= swing_lo:
            return {"ok": False, "score": 0.0, "zone": None, "reasons": ["flat_market"]}
        
        f618 = swing_lo + 0.618 * (swing_hi - swing_lo)
        f786 = swing_lo + 0.786 * (swing_hi - swing_lo)
        last_close = float(c.iloc[-1])
        
        vol_ma20 = v.rolling(20).mean().iloc[-1]
        vol_ok = float(v.iloc[-1]) >= vol_ma20 * 0.8
        
        current_open = float(df['open'].iloc[-1])
        current_high = float(h.iloc[-1])
        current_low = float(l.iloc[-1])
        
        body = abs(last_close - current_open)
        wick_up = current_high - max(last_close, current_open)
        wick_down = min(last_close, current_open) - current_low
        
        bull_candle = wick_down > (body * 1.2) and last_close > current_open
        bear_candle = wick_up > (body * 1.2) and last_close < current_open
        
        adx = ind.get('adx', 0) if ind else 0
        rsi_ctx = rsi_ma_context(df)
        
        score = 0.0
        zone_type = None
        reasons = []
        
        if f618 <= last_close <= f786 and bull_candle:
            score += 4.0
            reasons.append("فيبو_قاع+شمعة_صاعدة")
            if adx >= GZ_REQ_ADX:
                score += 2.0
                reasons.append("ADX_قوي")
            if rsi_ctx["cross"] == "bull" or rsi_ctx["trendZ"] == "bull":
                score += 1.5
                reasons.append("RSI_إيجابي")
            if vol_ok:
                score += 0.5
                reasons.append("حجم_مرتفع")
            
            if score >= GZ_MIN_SCORE:
                zone_type = "golden_bottom"
        
        elif f618 <= last_close <= f786 and bear_candle:
            score += 4.0
            reasons.append("فيبو_قمة+شمعة_هابطة")
            if adx >= GZ_REQ_ADX:
                score += 2.0
                reasons.append("ADX_قوي")
            if rsi_ctx["cross"] == "bear" or rsi_ctx["trendZ"] == "bear":
                score += 1.5
                reasons.append("RSI_سلبي")
            if vol_ok:
                score += 0.5
                reasons.append("حجم_مرتفع")
            
            if score >= GZ_MIN_SCORE:
                zone_type = "golden_top"
        
        ok = zone_type is not None and ALLOW_GZ_ENTRY
        return {
            "ok": ok,
            "score": score,
            "zone": {"type": zone_type, "f618": f618, "f786": f786} if zone_type else None,
            "reasons": reasons
        }
        
    except Exception as e:
        return {"ok": False, "score": 0.0, "zone": None, "reasons": [f"error: {e}"]}

def decide_strategy_mode(df, adx=None, di_plus=None, di_minus=None, rsi_ctx=None):
    """تحديد نمط التداول: SCALP أم TREND"""
    if adx is None or di_plus is None or di_minus is None:
        ind = compute_indicators(df)
        adx = ind.get('adx', 0)
        di_plus = ind.get('plus_di', 0)
        di_minus = ind.get('minus_di', 0)
    
    if rsi_ctx is None:
        rsi_ctx = rsi_ma_context(df)
    
    di_spread = abs(di_plus - di_minus)
    
    strong_trend = (
        (adx >= ADX_TREND_MIN and di_spread >= DI_SPREAD_TREND) or
        (rsi_ctx["trendZ"] in ("bull", "bear") and not rsi_ctx["in_chop"])
    )
    
    mode = "trend" if strong_trend else "scalp"
    why = "adx/di_trend" if adx >= ADX_TREND_MIN else ("rsi_trendZ" if rsi_ctx["trendZ"] != "none" else "scalp_default")
    
    return {"mode": mode, "why": why}

# =================== ENHANCED COUNCIL VOTING ===================
def council_votes_pro_enhanced(df):
    """مجلس تصويت محسّن مع RSI+MA والمناطق الذهبية + الشموع"""
    try:
        ind = compute_indicators(df)
        rsi_ctx = rsi_ma_context(df)
        gz = golden_zone_check(df, ind)

        # جديد: حساب الشموع
        cd = compute_candles(df)

        votes_b = 0; votes_s = 0
        score_b = 0.0; score_s = 0.0
        logs = []

        adx = ind.get('adx', 0)
        plus_di = ind.get('plus_di', 0)
        minus_di = ind.get('minus_di', 0)
        di_spread = abs(plus_di - minus_di)

        # --- ترند ADX/DI
        if adx > ADX_TREND_MIN:
            if plus_di > minus_di and di_spread > DI_SPREAD_TREND:
                votes_b += 2; score_b += 1.5; logs.append("📈 ترند صاعد قوي")
            elif minus_di > plus_di and di_spread > DI_SPREAD_TREND:
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

        # --- Golden Zones
        if gz and gz.get("ok"):
            if gz['zone']['type'] == 'golden_bottom':
                votes_b += 3; score_b += 1.5; logs.append(f"🏆 قاع ذهبي (قوة: {gz['score']:.1f})")
            elif gz['zone']['type'] == 'golden_top':
                votes_s += 3; score_s += 1.5; logs.append(f"🏆 قمة ذهبية (قوة: {gz['score']:.1f})")

        # جديد: الشموع
        if cd["score_buy"]>0:
            score_b += min(2.5, cd["score_buy"]); logs.append(f"🕯️ شموع BUY ({cd['pattern']}) +{cd['score_buy']:.1f}")
        if cd["score_sell"]>0:
            score_s += min(2.5, cd["score_sell"]); logs.append(f"🕯️ شموع SELL ({cd['pattern']}) +{cd['score_sell']:.1f}")

        # تخفيف النطاق المحايد
        if rsi_ctx["in_chop"]:
            score_b *= 0.8; score_s *= 0.8; logs.append("⚖️ RSI محايد — تخفيض ثقة")

        # حارس ADX عام
        if adx < ADX_GATE:
            score_b *= 0.85; score_s *= 0.85; logs.append(f"🛡️ ADX Gate ({adx:.1f} < {ADX_GATE})")

        # ضمّ إشارات الشموع ليتوفّر لباقي المنظومة (إدارة/خروج)
        ind.update({
            "rsi_ma": rsi_ctx["rsi_ma"],
            "rsi_trendz": rsi_ctx["trendZ"],
            "di_spread": di_spread,
            "gz": gz,
            "candle_buy_score": cd["score_buy"],
            "candle_sell_score": cd["score_sell"],
            "wick_up_big": cd["wick_up_big"],
            "wick_dn_big": cd["wick_dn_big"],
            "candle_tags": cd["pattern"]
        })

        return {
            "b": votes_b, "s": votes_s,
            "score_b": score_b, "score_s": score_s,
            "logs": logs, "ind": ind, "gz": gz, "candles": cd
        }
    except Exception as e:
        log_w(f"council_votes_pro_enhanced error: {e}")
        return {"b":0,"s":0,"score_b":0.0,"score_s":0.0,"logs":[],"ind":{},"gz":None,"candles":{}}

council_votes_pro = council_votes_pro_enhanced

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

# =================== BINGX POSITION PATCH ===================
_bingx_pos_debug_printed = False

def _try_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def _extract_positions_from_balance(bal):
    info = (bal or {}).get("info") or {}
    candidates = []
    if isinstance(info, dict):
        candidates.append(info)
        if isinstance(info.get("data"), dict):
            candidates.append(info["data"])
        if isinstance(info.get("result"), dict):
            candidates.append(info["result"])
        if isinstance(info.get("data"), list):
            candidates.append({"positions": info["data"]})

    for node in candidates:
        if not isinstance(node, dict):
            continue
        for k in ("positions", "position", "openPositions", "openPosition", "pos", "data"):
            v = node.get(k)
            if isinstance(v, list) and v:
                return v
    return []

def _read_position_bingx():
    global _bingx_pos_debug_printed

    # 1) حاول fetch_positions
    try:
        poss = ex.fetch_positions(params={"type": "swap"})
        if isinstance(poss, list) and poss:
            for p in poss:
                sym = (p.get("symbol") or p.get("info", {}).get("symbol") or "")
                if SYMBOL.split(":")[0] not in sym:
                    continue
                qty = abs(_try_float(p.get("contracts") or p.get("info", {}).get("positionAmt") or 0))
                if qty <= 0:
                    continue
                entry = _try_float(p.get("entryPrice") or p.get("info", {}).get("avgEntryPrice") or 0)
                side_raw = (p.get("side") or p.get("info", {}).get("positionSide") or p.get("info", {}).get("side") or "").lower()
                side = "long" if (side_raw in ("long","buy") or "long" in side_raw) else "short"
                return qty, side, entry
    except Exception as e:
        log_w(f"bingx fetch_positions not available: {e}")

    # 2) fallback: fetch_balance(type=swap)
    try:
        bal = ex.fetch_balance(params={"type": "swap"})
        if not _bingx_pos_debug_printed:
            _bingx_pos_debug_printed = True
            info = (bal or {}).get("info")
            if isinstance(info, dict):
                log_i(f"[BINGX DEBUG] balance.info keys: {list(info.keys())[:25]}")
                if isinstance(info.get("data"), dict):
                    log_i(f"[BINGX DEBUG] balance.info.data keys: {list(info['data'].keys())[:25]}")

        poss = _extract_positions_from_balance(bal)
        for p in poss:
            if not isinstance(p, dict):
                continue
            sym = str(p.get("symbol") or p.get("s") or p.get("instrumentId") or "")
            if SYMBOL.split(":")[0] not in sym:
                continue
            qty = abs(_try_float(p.get("positionAmt") or p.get("qty") or p.get("position") or p.get("size") or 0))
            if qty <= 0:
                continue
            entry = _try_float(p.get("avgEntryPrice") or p.get("entryPrice") or p.get("avgPrice") or 0)
            side_raw = str(p.get("side") or p.get("positionSide") or p.get("direction") or "").lower()
            side = "long" if (side_raw in ("long","buy") or "long" in side_raw) else "short"
            return qty, side, entry
    except Exception as e:
        log_w(f"bingx fetch_balance positions fallback failed: {e}")

    return 0.0, None, None

def _read_position():
    return _read_position_bingx()

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

# =================== FLOW PRESSURE MODULE ===================
_FLOW_CACHE = {"ts": 0, "data": None}

def _now():
    return time.time()

def safe_fetch_trades(symbol, limit=200):
    try:
        return ex.fetch_trades(symbol, limit=limit)
    except Exception as e:
        logging.warning(f"fetch_trades failed: {e}")
        return []

def safe_fetch_order_book(symbol, limit=20):
    try:
        return ex.fetch_order_book(symbol, limit=limit)
    except Exception as e:
        logging.warning(f"fetch_order_book failed: {e}")
        return {"bids": [], "asks": []}

def compute_delta_from_trades(trades):
    """
    delta = buy_qty - sell_qty
    delta_ratio = delta / total_qty
    """
    buy_qty = 0.0
    sell_qty = 0.0
    for t in trades or []:
        amt = float(t.get("amount") or 0.0)
        side = (t.get("side") or "").lower()
        if side == "buy":
            buy_qty += amt
        elif side == "sell":
            sell_qty += amt
    total = buy_qty + sell_qty
    delta = buy_qty - sell_qty
    delta_ratio = (delta / total) if total > 0 else 0.0
    return {
        "buy_qty": buy_qty,
        "sell_qty": sell_qty,
        "delta": delta,
        "delta_ratio": delta_ratio,
        "total_qty": total
    }

def compute_obi_and_walls(ob, depth=20):
    bids = (ob.get("bids") or [])[:depth]
    asks = (ob.get("asks") or [])[:depth]

    bid_qty = sum(float(x[1]) for x in bids) if bids else 0.0
    ask_qty = sum(float(x[1]) for x in asks) if asks else 0.0
    denom = (bid_qty + ask_qty)
    obi = ((bid_qty - ask_qty) / denom) if denom > 0 else 0.0

    # Walls: أكبر مستوى حجم مقارنة بالمتوسط
    bid_sizes = [float(x[1]) for x in bids] or [0.0]
    ask_sizes = [float(x[1]) for x in asks] or [0.0]
    bid_avg = (sum(bid_sizes) / len(bid_sizes)) if bid_sizes else 0.0
    ask_avg = (sum(ask_sizes) / len(ask_sizes)) if ask_sizes else 0.0
    bid_max = max(bid_sizes) if bid_sizes else 0.0
    ask_max = max(ask_sizes) if ask_sizes else 0.0

    bid_wall = (bid_avg > 0 and bid_max >= FLOW_WALL_MULT * bid_avg)
    ask_wall = (ask_avg > 0 and ask_max >= FLOW_WALL_MULT * ask_avg)

    return {
        "bid_qty": bid_qty,
        "ask_qty": ask_qty,
        "obi": obi,
        "bid_wall": bid_wall,
        "ask_wall": ask_wall,
        "bid_max": bid_max,
        "ask_max": ask_max,
        "bid_avg": bid_avg,
        "ask_avg": ask_avg,
    }

def flow_pressure_snapshot(symbol):
    if not FLOW_ENABLED:
        return {"ok": False, "why": "disabled"}

    # cache لتقليل ضربات API
    if _FLOW_CACHE["data"] and (_now() - _FLOW_CACHE["ts"] < FLOW_CACHE_S):
        return _FLOW_CACHE["data"]

    trades = safe_fetch_trades(symbol, limit=FLOW_TRADES_LIMIT)
    ob = safe_fetch_order_book(symbol, limit=FLOW_ORDERBOOK_DEPTH)

    delta = compute_delta_from_trades(trades)
    obi = compute_obi_and_walls(ob, depth=FLOW_ORDERBOOK_DEPTH)

    data = {
        "ok": True,
        "delta": delta,
        "obi": obi
    }
    _FLOW_CACHE["ts"] = _now()
    _FLOW_CACHE["data"] = data
    return data

# =================== SCALP PRO ENGINE FUNCTIONS ===================
def scalp_detect_micro_sweep(df, direction, lookback=20, wick_min=0.45):
    if len(df) < lookback + 2:
        return False, "insufficient_data"
    
    if direction == "buy":
        prev_low = float(df["low"].astype(float).iloc[-lookback-1:-1].min())
        last = df.iloc[-1]
        o,h,l,c = map(float, (last.open, last.high, last.low, last.close))
        uw,lw,br = tbe_wick_ratios(o,h,l,c)
        ok = (l < prev_low) and (c > prev_low) and (lw >= wick_min)
        return ok, f"micro_sweep_low(prev_low={prev_low}, lw={lw:.2f})"
    else:  # sell
        prev_high = float(df["high"].astype(float).iloc[-lookback-1:-1].max())
        last = df.iloc[-1]
        o,h,l,c = map(float, (last.open, last.high, last.low, last.close))
        uw,lw,br = tbe_wick_ratios(o,h,l,c)
        ok = (h > prev_high) and (c < prev_high) and (uw >= wick_min)
        return ok, f"micro_sweep_high(prev_high={prev_high}, uw={uw:.2f})"

def scalp_displacement_ok(df, ind, direction, lookback=6, mult=1.2):
    atr = float(ind.get("atr", 0) or 0)
    if atr <= 0 or len(df) < lookback + 2:
        return False, "no_atr"

    tail = df.tail(lookback)
    for _, row in tail.iterrows():
        o = float(row.open); c = float(row.close)
        body = abs(c - o)
        if body < mult * atr:
            continue
        if direction == "buy" and c > o:
            return True, f"disp_ok(body={body:.6f} atr={atr:.6f})"
        if direction == "sell" and c < o:
            return True, f"disp_ok(body={body:.6f} atr={atr:.6f})"
    return False, "no_displacement"

def scalp_room_ok(px, boxes, direction, min_room_bps=45):
    if not boxes.get("ok"):
        return False, "no_boxes"
    
    px_f = float(px)
    if direction == "buy":
        supply_high = float(boxes.get("supply", {}).get("high", 0))
        if supply_high <= 0: return True, "no_supply_defined"
        room_bps = ((supply_high - px_f) / px_f) * 10000.0
        if room_bps >= min_room_bps:
            return True, f"room_to_supply={room_bps:.0f}bps"
        return False, f"room_to_supply={room_bps:.0f}bps"
    else:  # sell
        demand_low = float(boxes.get("demand", {}).get("low", 0))
        if demand_low <= 0: return True, "no_demand_defined"
        room_bps = ((px_f - demand_low) / px_f) * 10000.0
        if room_bps >= min_room_bps:
            return True, f"room_to_demand={room_bps:.0f}bps"
        return False, f"room_to_demand={room_bps:.0f}bps"

def scalp_chop_guard(ind):
    adx = float(ind.get("adx", 0.0) or 0.0)
    rsi = float(ind.get("rsi", 50.0) or 50.0)
    
    if adx <= SCALP_CHOP_ADX_MAX and abs(rsi - 50.0) <= SCALP_CHOP_RSI_BAND:
        return False, f"chop(adx={adx:.1f}, rsi={rsi:.1f})"
    return True, "ok"

def scalp_pro_signal(df, ind, px):
    if not SCALP_PRO_ENABLED:
        return None, "disabled"

    boxes = detect_simple_boxes(df, lookback=SCALP_BOX_LOOKBACK)
    if not boxes.get("ok"):
        return None, "no_boxes"

    # 1) chop guard
    ok_chop, why_chop = scalp_chop_guard(ind)
    if not ok_chop:
        return None, f"chop_block({why_chop})"

    # 2) تحديد اتجاه السكالب حسب لمس البوكس
    side = None
    if boxes.get("in_demand"):
        side = "buy"
    elif boxes.get("in_supply"):
        side = "sell"
    else:
        return None, "no_touch_box"

    # 3) لازم room كفاية للبوكس المعاكس
    ok_room, why_room = scalp_room_ok(float(px), boxes, side, SCALP_MIN_ROOM_BPS)
    if not ok_room:
        return None, f"no_room({why_room})"

    # 4) micro sweep (إلزامي هنا)
    if SCALP_REQUIRE_SWEEP:
        ok_sw, why_sw = scalp_detect_micro_sweep(df, side, SCALP_SWEEP_LOOKBACK, SCALP_SWEEP_WICK_MIN)
        if not ok_sw:
            return None, f"no_sweep({why_sw})"

    # 5) displacement صغير (إلزامي)
    ok_disp, why_disp = scalp_displacement_ok(df, ind, side, SCALP_DISPLACE_LOOKBACK, SCALP_DISPLACE_ATR_MULT)
    if not ok_disp:
        return None, f"no_displacement({why_disp})"

    # 6) retest zone (OB/FVG)
    zone = None; ztag = None
    if side == "buy":
        ok_ob, ob = tbe_detect_bullish_ob(df, ind)
        ok_fvg, fvg = tbe_detect_bullish_fvg(df)
        if ok_ob: zone, ztag = ob, "OB"
        elif ok_fvg: zone, ztag = fvg, "FVG"
    else:
        ok_ob, ob = tbe_detect_bearish_ob(df, ind)
        ok_fvg, fvg = tbe_detect_bearish_fvg(df)
        if ok_ob: zone, ztag = ob, "OB"
        elif ok_fvg: zone, ztag = fvg, "FVG"
    
    if zone:
        if not tbe_in_zone(float(px), zone, pad_pct=SCALP_RETEST_PAD_PCT):
            return None, f"waiting_retest_{ztag}"
    # لو مفيش OB/FVG نكتفي ببوكس نفسه (لأنه touch already)

    # 7) confirmation candle (إلزامي)
    ok_conf, why_conf = candle_confirmation(df, ind, side)
    if not ok_conf:
        return None, f"no_confirm({why_conf})"

    return side, f"SCALP_PRO {side} | room={why_room} | {why_disp} | retest={ztag or 'BOX'} | conf={why_conf}"

# ========= Unified snapshot emitter =========
def emit_snapshots(exchange, symbol, df, balance_fn=None, pnl_fn=None):
    """
    يطبع Snapshot موحّد: Bookmap + Flow + Council + Strategy + Balance/PnL
    """
    try:
        bm = bookmap_snapshot(exchange, symbol)
        flow = compute_flow_metrics(df)
        cv = council_votes_pro(df)
        mode = decide_strategy_mode(df)
        gz = golden_zone_check(df, {"adx": cv["ind"]["adx"]}, "buy" if cv["b"]>=cv["s"] else "sell")
        
        # ✅ FLOW PRESSURE ADDON
        pressure = flow_pressure_snapshot(symbol)

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
            
        # ✅ FLOW PRESSURE LOGGING
        pressure_note = ""
        if pressure and pressure.get("ok"):
            dr = pressure["delta"]["delta_ratio"]
            obi = pressure["obi"]["obi"]
            ask_wall = pressure["obi"]["ask_wall"]
            bid_wall = pressure["obi"]["bid_wall"]
            pressure_note = f" | 🎯 DeltaR={dr:.2f} OBI={obi:.2f} {'🔴AskWall' if ask_wall else ''}{'🟢BidWall' if bid_wall else ''}"

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
            print(f"📊 {dash}{gz_note}{pressure_note}", flush=True)
            print(f"{strat}{(' | ' + wallet) if wallet else ''}", flush=True)
            
            gz_snap_note = ""
            if gz and gz.get("ok"):
                zone_type = gz["zone"]["type"]
                zone_score = gz["score"]
                gz_snap_note = f" | 🟡{zone_type} s={zone_score:.1f}"
            
            flow_z = flow['delta_z'] if flow and flow.get('ok') else 0.0
            bm_imb = bm['imbalance'] if bm and bm.get('ok') else 1.0
            
            print(f"🧠 SNAP | {side_hint} | votes={cv['b']}/{cv['s']} score={cv['score_b']:.1f}/{cv['score_s']:.1f} "
                  f"| ADX={cv['ind'].get('adx',0):.1f} DI={cv['ind'].get('di_spread',0):.1f} | "
                  f"z={flow_z:.2f} | imb={bm_imb:.2f}{gz_snap_note}", 
                  flush=True)
            
            print("✅ ADDONS LIVE", flush=True)

        return {"bm": bm, "flow": flow, "cv": cv, "mode": mode, "gz": gz, "wallet": wallet, "pressure": pressure}
    except Exception as e:
        print(f"🟨 AddonLog error: {e}", flush=True)
        return {"bm": None, "flow": None, "cv": {"b":0,"s":0,"score_b":0.0,"score_s":0.0,"ind":{}},
                "mode": {"mode":"n/a"}, "gz": None, "wallet": "", "pressure": None}

# =================== EXECUTION MANAGER ===================
def execute_trade_decision(side, price, qty, mode, council_data, gz_data):
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
    
    votes = council_data
    print(f"🎯 EXECUTE: {side.upper()} {qty:.4f} @ {price:.6f} | "
          f"mode={mode} | votes={votes['b']}/{votes['s']} score={votes['score_b']:.1f}/{votes['score_s']:.1f}"
          f"{gz_note}", flush=True)

    try:
        if MODE_LIVE:
            ex.set_leverage(LEVERAGE, SYMBOL, params={"side": "BOTH"})
            ex.create_order(SYMBOL, "market", side, qty, None, _params_open(side))
        
        log_g(f"✅ EXECUTED: {side.upper()} {qty:.4f} @ {price:.6f}")
        return True
    except Exception as e:
        log_e(f"❌ EXECUTION FAILED: {e}")
        return False

def setup_trade_management(mode):
    """تهيئة إدارة الصفقة حسب النمط"""
    if mode == "scalp":
        return {
            "tp1_pct": SCALP_TP1 / 100.0,
            "be_activate_pct": SCALP_BE_AFTER / 100.0,
            "trail_activate_pct": 0.8 / 100.0,
            "atr_trail_mult": SCALP_ATR_MULT,
            "close_aggression": "high"
        }
    else:
        return {
            "tp1_pct": TREND_TP1 / 100.0,
            "be_activate_pct": TREND_BE_AFTER / 100.0,
            "trail_activate_pct": 1.2 / 100.0,
            "atr_trail_mult": TREND_ATR_MULT,
            "close_aggression": "medium"
        }

# =================== ENHANCED TRADE EXECUTION ===================
def open_market_enhanced(side, qty, price, mode_override=None, reason_override=None, engine_tag=None):
    """✅ PATCH A - تعديل open_market_enhanced لدعم Mode Override + Reason"""
    if qty <= 0: 
        log_e("skip open (qty<=0)")
        return False
    
    df = fetch_ohlcv()
    snap = emit_snapshots(ex, SYMBOL, df)
    
    votes = snap["cv"]
    mode_data = decide_strategy_mode(df, 
                                   adx=votes["ind"].get("adx"),
                                   di_plus=votes["ind"].get("plus_di"),
                                   di_minus=votes["ind"].get("minus_di"),
                                   rsi_ctx=rsi_ma_context(df))
    
    # ===== MODE OVERRIDE (Trend Birth / Box Scalper) =====
    mode = mode_data["mode"]
    if mode_override in ("trend", "mid", "scalp"):
        mode = mode_override

    # ===== ENTRY META (for logs / management rules) =====
    if reason_override:
        STATE["entry_reason"] = str(reason_override)
    if engine_tag:
        STATE["entry_engine"] = str(engine_tag)
    
    gz = snap["gz"]
    
    management_config = setup_trade_management(mode)
    
    success = execute_trade_decision(side, price, qty, mode, votes, gz)
    
    if success:
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
            "entry_reason": STATE.get("entry_reason"),
            "entry_engine": STATE.get("entry_engine")
        })
        
        # ===== FALCON PLAN SNAPSHOT =====
        ind = votes.get("ind", {}) or {}
        atr = float(ind.get("atr", 0.0))
        side_norm = "long" if side == "buy" else "short"
        plan = falcon_plan(entry_px=float(price), side=side_norm, mode=mode, atr=atr)

        STATE["falcon_tp1"] = plan["tp1"]
        STATE["falcon_tp2"] = plan["tp2"]
        STATE["falcon_tp3"] = plan["tp3"]
        STATE["falcon_tp2_done"] = False
        STATE["falcon_tp3_done"] = False
        STATE["cooldown_until_ts"] = 0

        # ===== PRINT LIKE THE IMAGES =====
        log_g(f"🚀 {('BUY' if side_norm=='long' else 'SELL')} | entry={price:.6f} | mode={mode} "
              f"| TP1={STATE['falcon_tp1']:.6f}")
        if STATE['falcon_tp2'] is not None:
            log_g(f"   ↳ TP2={STATE['falcon_tp2']:.6f}")
        if STATE['falcon_tp3'] is not None:
            log_g(f"   ↳ TP3={STATE['falcon_tp3']:.6f}")
        
        save_state({
            "in_position": True,
            "side": "LONG" if side.upper().startswith("B") else "SHORT",
            "entry_price": price,
            "position_qty": qty,
            "leverage": LEVERAGE,
            "mode": mode,
            "management": management_config,
            "gz_snapshot": gz if isinstance(gz, dict) else {},
            "cv_snapshot": votes if isinstance(votes, dict) else {},
            "opened_at": int(time.time()),
            "partial_taken": False,
            "breakeven_armed": False,
            "trail_active": False,
            "trail_tightened": False,
        })
        
        # ===== Scalp count (Box-first entries تعتبر scalp افتراضيًا) =====
        try:
            # لو وضعك الحالي بيخزن mode في STATE، استخدمه. لو لا: اعتبرها scalp
            mode_now = STATE.get("falcon_mode") or STATE.get("mode") or "scalp"
            if mode_now == "scalp":
                STATE["scalp_trades_today"] = int(STATE.get("scalp_trades_today", 0)) + 1
                log_i(f"📌 SCALP COUNT TODAY = {STATE['scalp_trades_today']}/{MAX_SCALP_TRADES_PER_DAY}")
        except Exception:
            pass
        
        log_g(f"✅ POSITION OPENED: {side.upper()} | mode={mode}")
        return True
    
    return False

open_market = open_market_enhanced

# =================== INDICATORS ===================
def wilder_ema(s: pd.Series, n: int): 
    return s.ewm(alpha=1/n, adjust=False).mean()

def compute_indicators(df: pd.DataFrame):
    if len(df) < max(ATR_LEN, RSI_LEN, ADX_LEN) + 2:
        return {"rsi":50.0,"plus_di":0.0,"minus_di":0.0,"dx":0.0,"adx":0.0,"atr":0.0}
    c,h,l = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float)
    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = wilder_ema(tr, ATR_LEN)

    delta=c.diff(); up=delta.clip(lower=0.0); dn=(-delta).clip(lower=0.0)
    rs = wilder_ema(up, RSI_LEN) / wilder_ema(dn, RSI_LEN).replace(0,1e-12)
    rsi = 100 - (100/(1+rs))

    up_move=h.diff(); down_move=l.shift(1)-l
    plus_dm=up_move.where((up_move>down_move)&(up_move>0),0.0)
    minus_dm=down_move.where((down_move>up_move)&(down_move>0),0.0)
    plus_di=100*(wilder_ema(plus_dm, ADX_LEN)/atr.replace(0,1e-12))
    minus_di=100*(wilder_ema(minus_dm, ADX_LEN)/atr.replace(0,1e-12))
    dx=(100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,1e-12)).fillna(0.0)
    adx=wilder_ema(dx, ADX_LEN)

    i=len(df)-1
    return {
        "rsi": float(rsi.iloc[i]), "plus_di": float(plus_di.iloc[i]),
        "minus_di": float(minus_di.iloc[i]), "dx": float(dx.iloc[i]),
        "adx": float(adx.iloc[i]), "atr": float(atr.iloc[i])
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

# =================== STATE ===================
STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
    "tp1_done": False, "highest_profit_pct": 0.0,
    "profit_targets_achieved": 0,
    "cooldown_until_ts": 0,
    "day_key": "",
    "scalp_trades_today": 0,
    "entry_reason": None,
    "entry_engine": None
}
compound_pnl = 0.0
wait_for_next_signal_side = None

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

def wait_gate_allow_side(desired_side: str):
    """بوابة الانتظار المعدلة للسماح بالدخول بالاتجاه"""
    global wait_for_next_signal_side
    if wait_for_next_signal_side is None:
        return True, ""
    if desired_side == wait_for_next_signal_side:
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
    global wait_for_next_signal_side
    prev_side = prev_side or STATE.get("side")
    
    # تفعيل كولداون بعد إغلاق فاشل
    if "early_fail" in reason or "time_stop" in reason:
        STATE["cooldown_until_ts"] = int(time.time()) + (REENTRY_COOLDOWN_BARS * 60)
        log_i(f"⏸️ COOLDOWN ACTIVATED for {REENTRY_COOLDOWN_BARS} minutes")
    
    STATE.update({
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
        "tp1_done": False, "highest_profit_pct": 0.0, "profit_targets_achieved": 0,
        "trail_tightened": False, "partial_taken": False,
        "falcon_tp1": None, "falcon_tp2": None, "falcon_tp3": None,
        "falcon_tp2_done": False, "falcon_tp3_done": False,
        "entry_reason": None,
        "entry_engine": None,
        "be_armed": False,
        "trend_upgraded": False,
        "scalp_tp_data": {},
        "scalp_bars_in_trade": 0,
    })
    save_state({"in_position": False, "position_qty": 0})
    
    # تفعيل انتظار الإشارة التالية
    _arm_wait_after_close(prev_side)
    logging.info(f"AFTER_CLOSE waiting_for={wait_for_next_signal_side} reason={reason}")

# =================== SMART SCALP HELPER FUNCTIONS ===================
def min_expected_move_pct(ind, px):
    """أقل حركة متوقعة للسكالب"""
    atr = float(ind.get("atr", 0) or 0)
    if atr <= 0 or px <= 0:
        return SCALP_MIN_MOVE_PCT_FIXED
    atr_pct = (atr / px) * 100.0
    return max(SCALP_MIN_MOVE_PCT_FIXED, atr_pct * SCALP_ATR_EXPECT_MULT)

def expected_room_pct_for_boxes(px, boxes, sig):
    """حساب المساحة المتوقعة للبوكس المعاكس"""
    if not boxes or not boxes.get("ok") or px <= 0:
        return 0.0, "no_boxes"
    d = boxes.get("demand") or {}
    s = boxes.get("supply") or {}
    d_high = float(d.get("high", 0) or 0)
    s_low  = float(s.get("low", 0) or 0)

    if sig == "buy":
        if s_low <= 0: 
            return 0.0, "no_supply"
        return ((s_low - px) / px) * 100.0, "to_supply"
    else:
        if d_high <= 0:
            return 0.0, "no_demand"
        return ((px - d_high) / px) * 100.0, "to_demand"

def zone_touch_ok(px, box, sig):
    """التحقق من لمس المنطقة"""
    if not box:
        return False
    low = float(box.get("low", 0) or 0)
    high = float(box.get("high", 0) or 0)
    if low <= 0 or high <= 0:
        return False
    if low <= px <= high:
        return True
    ref = high if sig == "buy" else low
    if ref <= 0: 
        return False
    bps = abs(px - ref) / ref * 10000.0
    return bps <= ZONE_TOUCH_BPS

def zone_first_signal(df, ind, px):
    """إشارة Zone-First"""
    boxes = detect_simple_boxes(df, lookback=ZONE_BOX_LOOKBACK)
    if not boxes.get("ok"):
        return None, "no_boxes"

    if boxes.get("in_demand") or zone_touch_ok(px, boxes.get("demand"), "buy"):
        ok, why = candle_confirmation(df, ind, "buy")
        if ok:
            return "buy", f"ZONE_FIRST demand+confirm({why})"
        return None, f"wait_confirm_buy({why})"

    if boxes.get("in_supply") or zone_touch_ok(px, boxes.get("supply"), "sell"):
        ok, why = candle_confirmation(df, ind, "sell")
        if ok:
            return "sell", f"ZONE_FIRST supply+confirm({why})"
        return None, f"wait_confirm_sell({why})"

    return None, "no_touch"

def calculate_scalp_feasibility(df, ind, entry_price, direction, boxes_data):
    """حساب جدوى صفقة السكالب"""
    if not SCALP_FEASIBILITY_ENABLED:
        return {"feasible": True, "score": 10, "reason": "feasibility_disabled"}
    
    score = 10
    reasons = []
    warnings = []
    
    atr = float(ind.get("atr", 0.0) or 0.0)
    current_price = float(df["close"].iloc[-1]) if len(df) > 0 else entry_price
    
    # حساب النقاط المتوقعة
    if direction == "buy":
        supply_high = float(boxes_data.get("supply", {}).get("high", 0))
        if supply_high > 0:
            expected_points = ((supply_high - entry_price) / DOGE_TICK)
        else:
            expected_points = (atr / DOGE_TICK) * 0.8
    else:
        demand_low = float(boxes_data.get("demand", {}).get("low", 0))
        if demand_low > 0:
            expected_points = ((entry_price - demand_low) / DOGE_TICK)
        else:
            expected_points = (atr / DOGE_TICK) * 0.8
    
    # التحقق من الحد الأدنى للنقاط
    if expected_points < SCALP_MIN_EXPECTED_POINTS:
        score -= 4
        reasons.append(f"نقاط متوقعة منخفضة: {expected_points:.1f} < {SCALP_MIN_EXPECTED_POINTS}")
    
    # Displacement شرط
    if SCALP_REQUIRED_DISPLACEMENT:
        displacement_ok, disp_reason = scalp_displacement_ok(df, ind, direction)
        if not displacement_ok:
            score -= 3
            reasons.append(f"لا يوجد اندفاع: {disp_reason}")
    
    # نسبة المخاطرة/العائد
    if atr > 0:
        sl_points = (atr * 0.5) / DOGE_TICK
        if expected_points > 0:
            rr_ratio = expected_points / sl_points
            if rr_ratio < 1.5:
                score -= 2
                warnings.append(f"نسبة R:R ضعيفة: {rr_ratio:.1f}:1")
    
    # حالة الترند المصغر
    micro_trend = analyze_micro_trend(df, 5)
    if direction == "buy" and micro_trend != "up":
        score -= 1
        warnings.append("الاتجاه المصغر ليس صاعد")
    elif direction == "sell" and micro_trend != "down":
        score -= 1
        warnings.append("الاتجاه المصغر ليس هابط")
    
    feasible = score >= 7
    
    return {
        "feasible": feasible,
        "score": score,
        "expected_points": expected_points,
        "reasons": reasons,
        "warnings": warnings,
        "atr_points": atr / DOGE_TICK if atr > 0 else 0
    }

def analyze_micro_trend(df, period=5):
    """تحليل الاتجاه المصغر"""
    if len(df) < period:
        return "neutral"
    
    closes = df["close"].astype(float).tail(period).values
    
    # حساب الانحدار
    x = np.arange(len(closes))
    try:
        slope, intercept = np.polyfit(x, closes, 1)
    except:
        return "neutral"
    
    if slope > 0.0005:
        return "up"
    elif slope < -0.0005:
        return "down"
    else:
        return "neutral"

def calculate_dynamic_scalp_tp(entry_price, direction, feasibility_data, ind):
    """حساب TP ديناميكي للسكالب"""
    expected_points = feasibility_data.get("expected_points", 6)
    atr_points = feasibility_data.get("atr_points", 10)
    
    base_points = max(SCALP_MIN_EXPECTED_POINTS, min(expected_points, SCALP_TP_MAX_PCT * 100))
    
    # تعديل بناءً على الزخم
    momentum = check_momentum_strength(ind, direction)
    momentum_boost = 1.0 + (momentum - 0.5) * 0.4
    
    final_points = base_points * momentum_boost
    
    if direction == "buy":
        tp_price = entry_price + (final_points * DOGE_TICK)
    else:
        tp_price = entry_price - (final_points * DOGE_TICK)
    
    return {
        "tp_price": tp_price,
        "points": final_points,
        "base_points": base_points,
        "momentum_boost": momentum_boost
    }

def check_momentum_strength(ind, direction):
    """فحص قوة الزخم"""
    rsi = float(ind.get("rsi", 50))
    adx = float(ind.get("adx", 0))
    
    momentum_score = 0.0
    
    if direction == "buy":
        if rsi > 40 and rsi < 70:
            momentum_score += 0.4
        elif rsi <= 40:
            momentum_score += 0.6
    else:
        if rsi > 30 and rsi < 60:
            momentum_score += 0.4
        elif rsi >= 60:
            momentum_score += 0.6
    
    if adx > 20:
        momentum_score += 0.3
    
    return min(momentum_score, 1.0)

# =================== ENHANCED TRADE MANAGEMENT ===================
def manage_after_entry_enhanced(df, ind, info):
    """إدارة محسنة للمركز مع خروج ذكي حسب النمط + FALCON STYLE"""
    if not STATE["open"] or STATE["qty"] <= 0:
        return

    px = info["price"]
    entry = STATE["entry"]
    side = STATE["side"]
    qty = STATE["qty"]
    mode = STATE.get("mode", "trend")
    management = STATE.get("management", {})
    
    pnl_pct = (px - entry) / entry * 100 * (1 if side == "long" else -1)
    STATE["pnl"] = pnl_pct
    
    if pnl_pct > STATE["highest_profit_pct"]:
        STATE["highest_profit_pct"] = pnl_pct

    snap = emit_snapshots(ex, SYMBOL, df)
    gz = snap["gz"]
    
    # ===== DYNAMIC BE / UPGRADE / TRAIL =====
    try:
        side = STATE.get("side")
        px_now = float(px)
        entry = float(STATE.get("entry", px_now))
        
        # 1) عتبة التحويل ديناميكيًا
        atr = float(ind.get("atr", 0) or 0)
        px_ref = max(entry, 1e-9)
        atr_pct = (atr / px_ref) * 100.0 if atr > 0 else 0.0
        upgrade_thr = max(UPGRADE_MIN_MOVE_PCT_FIXED, atr_pct * UPGRADE_ATR_MULT)
        
        # 2) ARM breakeven عند الوصول للحد الأدنى
        if pnl_pct >= upgrade_thr and not STATE.get("be_armed", False):
            be = entry * (1.0 + (BE_BUFFER_PCT/100.0) * (1 if side=="long" else -1))
            STATE["breakeven"] = be
            STATE["be_armed"] = True
            log_g(f"🔒 BE ARMED @ {be:.6f} | pnl={pnl_pct:.2f}% thr={upgrade_thr:.2f}%")
        
        # 3) تقييم قوة للترقية
        if UPGRADE_ENABLE and STATE.get("mode") == "scalp" and not STATE.get("trend_upgraded", False):
            adx = float(ind.get("adx", 0) or 0)
            di_p = float(ind.get("plus_di", 0) or 0)
            di_m = float(ind.get("minus_di", 0) or 0)
            
            # imbalance
            imb = 1.0
            try:
                bm = snap.get("bm") or {}
                imb = float(bm.get("imb", bm.get("imbalance", 1.0)) or 1.0)
            except Exception:
                pass
            
            if side == "long":
                di_dom_ok = (di_p - di_m) >= UPGRADE_DI_DOM_MIN
                flow_ok = imb >= UPGRADE_IMB_LONG
            else:
                di_dom_ok = (di_m - di_p) >= UPGRADE_DI_DOM_MIN
                flow_ok = imb <= UPGRADE_IMB_SHORT
            
            if pnl_pct >= upgrade_thr and adx >= UPGRADE_ADX_MIN and di_dom_ok and flow_ok:
                STATE["trend_upgraded"] = True
                STATE["mode"] = "trend"
                log_g(f"🧠 UPGRADE → TREND | pnl={pnl_pct:.2f}% adx={adx:.1f} imb={imb:.2f}")
        
        # 4) Trailing Stop ديناميكي بعد الترقية
        if STATE.get("trend_upgraded", False):
            if atr > 0:
                dist = atr * TRAIL_ATR_MULT
            else:
                dist = entry * 0.003
            
            trail = (px_now - dist) if side=="long" else (px_now + dist)
            
            # ratchet (لا يرجع للخلف)
            if TRAIL_RATCHET and STATE.get("trail") is not None:
                prev = float(STATE.get("trail") or 0)
                if side=="long":
                    trail = max(prev, trail)
                else:
                    trail = min(prev, trail)
            
            STATE["trail"] = trail
    except Exception as e:
        log_w(f"Dynamic management error: {e}")

    # ===== SCALP MANAGEMENT ENHANCEMENT =====
    if STATE.get("mode") == "scalp" and not STATE.get("trend_upgraded", False):
        # 1) استخدام TP الديناميكي
        tp_data = STATE.get("scalp_tp_data", {})
        if tp_data and not STATE.get("tp1_done", False):
            tp_price = tp_data.get("tp_price")
            if tp_price:
                if (side == "long" and px >= tp_price) or (side == "short" and px <= tp_price):
                    log_g(f"✅ SCALP TP HIT | {tp_data.get('points', 0):.1f} نقاط")
                    close_market_strict("scalp_tp_hit")
                    return
        
        # 2) Early Exit محسن
        STATE["scalp_bars_in_trade"] = STATE.get("scalp_bars_in_trade", 0) + 1
        
        # خروج إذا لم يتحقق الربح بعد 5 شمعات
        if STATE["scalp_bars_in_trade"] >= 5 and abs(pnl_pct) < 0.2:
            log_w(f"⏱️ SCALP TIME EXIT | {STATE['scalp_bars_in_trade']} شمعات بدون حركة")
            close_market_strict("scalp_time_exit")
            return
    
    # ===== Box invalidation (super cut) =====
    try:
        boxes = detect_simple_boxes(df, lookback=BOX_LOOKBACK)
        if boxes.get("ok"):
            d = boxes.get("demand") or {}
            s = boxes.get("supply") or {}
            d_low = float(d.get("low", 0))
            s_high = float(s.get("high", 0))

            # لو LONG وكسر قاع الديماند بشكل واضح → خروج فوري
            if side == "long" and d_low and px < d_low:
                log_w("🧱 INVALIDATION | broke DEMAND low → EXIT BUY")
                close_market_strict("box_invalidation_long")
                return

            # لو SHORT وكسر سقف السابلای → خروج فوري
            if side == "short" and s_high and px > s_high:
                log_w("🧱 INVALIDATION | broke SUPPLY high → EXIT SELL")
                close_market_strict("box_invalidation_short")
                return
    except Exception:
        pass
    
    # =================== FALCON BOXES (for EXIT like images) ===================
    boxes = detect_simple_boxes(df, lookback=60)
    if boxes.get("ok"):
        # Exit عند الصندوق المعاكس + ضعف بسيط
        if side == "long" and boxes.get("in_supply") and pnl_pct > 0.10:
            log_w("🟥 EXIT BUY | hit SUPPLY box")
            close_market_strict("exit_buy_supply_box")
            return
        if side == "short" and boxes.get("in_demand") and pnl_pct > 0.10:
            log_w("🟩 EXIT SELL | hit DEMAND box")
            close_market_strict("exit_sell_demand_box")
            return

    # =================== TBE Fast Fail (avoid big loss) =====
    if STATE.get("entry_engine") == "TBE":
        if STATE.get("bars", 0) <= 3 and pnl_pct <= -0.20:
            log_w("🧯 TBE FAIL FAST → EXIT")
            close_market_strict("tbe_fail_fast")
            return

    # =================== EARLY FAIL (cut loss fast) ===================
    STATE["bars"] = int(STATE.get("bars", 0)) + 1
    if STATE["bars"] <= EARLY_FAIL_BARS and pnl_pct <= EARLY_FAIL_PNL_PCT:
        log_w(f"🧨 EARLY FAIL | pnl={pnl_pct:.2f}% <= {EARLY_FAIL_PNL_PCT:.2f}% → EXIT")
        close_market_strict("early_fail")
        return

    # =================== TIME STOP (if dead trade) ===================
    if STATE["bars"] >= TIME_STOP_BARS and pnl_pct < TIME_STOP_MIN_PNL_PCT:
        log_w(f"⏱️ TIME STOP | bars={STATE['bars']} pnl={pnl_pct:.2f}% → EXIT")
        close_market_strict("time_stop")
        return

    exit_signal = smart_exit_guard(STATE, df, ind, snap["flow"], snap["bm"], 
                                 px, pnl_pct/100, mode, side, entry, gz)
    
    if exit_signal["log"]:
        print(f"🔔 {exit_signal['log']}", flush=True)

    if exit_signal["action"] == "partial" and not STATE.get("partial_taken"):
        partial_qty = safe_qty(qty * exit_signal.get("qty_pct", 0.3))
        if partial_qty > 0:
            close_side = "sell" if side == "long" else "buy"
            if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                try:
                    ex.create_order(SYMBOL, "market", close_side, partial_qty, None, _params_close())
                    log_g(f"✅ PARTIAL CLOSE: {partial_qty:.4f} | {exit_signal['why']}")
                    STATE["partial_taken"] = True
                    STATE["qty"] = safe_qty(qty - partial_qty)
                except Exception as e:
                    log_e(f"❌ Partial close failed: {e}")
            else:
                log_i(f"DRY_RUN: Partial close {partial_qty:.4f}")
    
    elif exit_signal["action"] == "tighten" and not STATE.get("trail_tightened"):
        STATE["trail_tightened"] = True
        STATE["trail"] = None
        log_i(f"🔄 TRAIL TIGHTENED: {exit_signal['why']}")
    
    elif exit_signal["action"] == "close":
        log_w(f"🚨 SMART EXIT: {exit_signal['why']}")
        close_market_strict(f"smart_exit_{exit_signal['why']}")
        return

    current_atr = ind.get("atr", 0.0)
    tp1_pct = management.get("tp1_pct", TP1_PCT_BASE/100.0)
    be_activate_pct = management.get("be_activate_pct", BREAKEVEN_AFTER/100.0)
    trail_activate_pct = management.get("trail_activate_pct", TRAIL_ACTIVATE_PCT/100.0)
    atr_trail_mult = management.get("atr_trail_mult", ATR_TRAIL_MULT)

    if not STATE.get("tp1_done") and pnl_pct/100 >= tp1_pct:
        close_fraction = TP1_CLOSE_FRAC
        close_qty = safe_qty(STATE["qty"] * close_fraction)
        if close_qty > 0:
            close_side = "sell" if STATE["side"] == "long" else "buy"
            if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                try:
                    ex.create_order(SYMBOL, "market", close_side, close_qty, None, _params_close())
                    log_g(f"✅ TP1 HIT: closed {close_fraction*100}%")
                except Exception as e:
                    log_e(f"❌ TP1 close failed: {e}")
            STATE["qty"] = safe_qty(STATE["qty"] - close_qty)
            STATE["tp1_done"] = True
            STATE["profit_targets_achieved"] += 1

    # =================== FALCON TP2 / TP3 (TREND ONLY) ===================
    if mode != "scalp":
        tp2 = STATE.get("falcon_tp2")
        tp3 = STATE.get("falcon_tp3")

        # TP2
        if tp2 and not STATE.get("falcon_tp2_done"):
            hit_tp2 = (px >= tp2) if side == "long" else (px <= tp2)
            if hit_tp2 and STATE["qty"] > 0:
                close_qty = safe_qty(STATE["qty"] * FALCON_TP2_CLOSE_FRAC)
                if close_qty > 0:
                    close_side = "sell" if side == "long" else "buy"
                    if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                        ex.create_order(SYMBOL, "market", close_side, close_qty, None, _params_close())
                    STATE["qty"] = safe_qty(STATE["qty"] - close_qty)
                STATE["falcon_tp2_done"] = True
                log_g("🎯 TP2 HIT | partial close")

        # TP3
        if tp3 and not STATE.get("falcon_tp3_done"):
            hit_tp3 = (px >= tp3) if side == "long" else (px <= tp3)
            if hit_tp3:
                log_g("🏁 TP3 HIT | EXIT")
                close_market_strict("tp3_exit")
                return

    if not STATE.get("breakeven_armed") and pnl_pct/100 >= be_activate_pct:
        STATE["breakeven_armed"] = True
        STATE["breakeven"] = entry
        log_i("BREAKEVEN ARMED")

    if not STATE.get("trail_active") and pnl_pct/100 >= trail_activate_pct:
        STATE["trail_active"] = True
        log_i("TRAIL ACTIVATED")

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

    if STATE.get("trail"):
        if (side == "long" and px <= STATE["trail"]) or (side == "short" and px >= STATE["trail"]):
            log_w(f"TRAIL STOP: {px} vs trail {STATE['trail']}")
            close_market_strict("trail_stop")

    if STATE.get("breakeven"):
        if (side == "long" and px <= STATE["breakeven"]) or (side == "short" and px >= STATE["breakeven"]):
            log_w(f"BREAKEVEN STOP: {px} vs breakeven {STATE['breakeven']}")
            close_market_strict("breakeven_stop")

    if STATE["qty"] <= FINAL_CHUNK_QTY:
        log_w(f"DUST GUARD: qty {STATE['qty']} <= {FINAL_CHUNK_QTY}, closing...")
        close_market_strict("dust_guard")

manage_after_entry = manage_after_entry_enhanced

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

# =================== ENHANCED TRADE LOOP ===================
LOG_THROTTLE_S = 7
_last_pretty_ts = 0

def trade_loop_enhanced():
    """حلقة تداول محسنة مع Golden Entry ومجلس الإدارة + FALCON STYLE + Trend Birth Engine"""
    global wait_for_next_signal_side, _last_pretty_ts
    loop_i = 0
    
    while True:
        try:
            # التحقق من الكولداون
            if int(time.time()) < int(STATE.get("cooldown_until_ts", 0)):
                time_left = STATE["cooldown_until_ts"] - int(time.time())
                if time_left > 0 and time_left % 30 == 0:  # طباعة كل 30 ثانية
                    log_i(f"⏸️ COOLDOWN active → {time_left//60}m:{time_left%60}s left")
                time.sleep(2)
                continue
            
            # جمع البيانات الأساسية
            bal = balance_usdt()
            px = price_now()
            df = fetch_ohlcv()
            info = rf_signal_live(df)
            ind = compute_indicators(df)
            spread_bps = orderbook_spread_bps()
            
            # تحديث الـ Snapshots
            snap = emit_snapshots(ex, SYMBOL, df,
                                balance_fn=lambda: float(bal) if bal else None,
                                pnl_fn=lambda: float(compound_pnl))
            
            # تحديث حالة الربح/الخسارة
            if STATE["open"] and px:
                STATE["pnl"] = (px-STATE["entry"])*STATE["qty"] if STATE["side"]=="long" else (STATE["entry"]-px)*STATE["qty"]
            
            # إدارة الصفقة المفتوحة
            if STATE["open"]:
                manage_after_entry(df, ind, {
                    "price": px or info["price"], 
                    "bm": snap["bm"],
                    "flow": snap["flow"],
                    **info
                })
            
            # قرار الدخول
            reason = None
            if spread_bps is not None and spread_bps > MAX_SPREAD_BPS:
                reason = f"spread too high ({fmt(spread_bps,2)}bps > {MAX_SPREAD_BPS})"
            
            council_data = council_votes_pro_enhanced(df)
            gz = council_data.get("gz")
            sig = None

            # ===== Daily reset for scalp count =====
            reset_daily_scalp_counter()

            # ===== TREND BIRTH ENGINE (Priority #1) =====
            if TBE_ENABLED and not STATE["open"]:
                pressure = snap.get("pressure")
                sig_b, why_b = trend_birth_buy_signal(df, ind, float(px), pressure)
                sig_s, why_s = trend_birth_sell_signal(df, ind, float(px), pressure)
                
                if sig_b == "buy":
                    qty = compute_size(bal, float(px))
                    if qty > 0:
                        log_g(f"🚀 TBE BUY: {why_b}")
                        if open_market("buy", qty, float(px), mode_override="trend", 
                                      reason_override=why_b, engine_tag="TBE"):
                            continue
                
                if sig_s == "sell":
                    qty = compute_size(bal, float(px))
                    if qty > 0:
                        log_g(f"🚀 TBE SELL: {why_s}")
                        if open_market("sell", qty, float(px), mode_override="trend",
                                      reason_override=why_s, engine_tag="TBE"):
                            continue

            # ===== ENTRY PRIORITY v2 =====
            sig = None
            reason = None

            # 1) Zone-First أولوية أعلى
            if ZONE_FIRST_ENABLED and not STATE["open"]:
                zs, zw = zone_first_signal(df, ind, float(px))
                if zs:
                    if STATE.get("scalp_trades_today", 0) < MAX_SCALP_TRADES_PER_DAY:
                        sig = zs
                        reason = zw

            # 2) Box-First
            sig_box, why_box = box_first_signal(df, ind, float(px or info.get("price") or 0))
            if sig_box and not sig:
                if STATE.get("scalp_trades_today", 0) >= MAX_SCALP_TRADES_PER_DAY:
                    reason = f"daily_scalp_cap({STATE['scalp_trades_today']}/{MAX_SCALP_TRADES_PER_DAY})"
                else:
                    sig = sig_box
                    reason = f"BOX_FIRST:{why_box}"

            # 3) Golden override
            if (gz and gz.get("ok") and ind.get("adx",0) >= GOLDEN_ENTRY_ADX):
                if gz["zone"]["type"]=="golden_bottom" and gz["score"]>=GOLDEN_ENTRY_SCORE:
                    sig = "buy"
                    reason = f"GOLDEN_BUY score={gz['score']:.1f}"
                elif gz["zone"]["type"]=="golden_top" and gz["score"]>=GOLDEN_ENTRY_SCORE:
                    sig = "sell" 
                    reason = f"GOLDEN_SELL score={gz['score']:.1f}"

            # 4) Council strong
            if sig is None:
                if council_data["score_b"] > council_data["score_s"] and council_data["score_b"] >= 8.0:
                    sig = "buy"
                    reason = "COUNCIL_STRONG_BUY"
                elif council_data["score_s"] > council_data["score_b"] and council_data["score_s"] >= 8.0:
                    sig = "sell"
                    reason = "COUNCIL_STRONG_SELL"

            # ===== SCALP PRO ENTRY (مع فلتر الجدوى) =====
            if not STATE.get("open", False):
                reset_daily_scalp_counter()
                
                # الحصول على إشارة السكالب
                sig_scalp, why_scalp = scalp_pro_signal(df, ind, float(px))
                
                if sig_scalp in ("buy", "sell"):
                    # فحص الجدوى قبل الدخول
                    boxes = detect_simple_boxes(df, lookback=SCALP_BOX_LOOKBACK)
                    feasibility = calculate_scalp_feasibility(df, ind, float(px), sig_scalp, boxes)
                    
                    if feasibility["feasible"]:
                        # حساب حجم المركز
                        qty = compute_size(bal, float(px))
                        
                        # حساب TP ديناميكي
                        tp_data = calculate_dynamic_scalp_tp(float(px), sig_scalp, feasibility, ind)
                        
                        log_g(f"🎯 SCALP PRO {sig_scalp.upper()} | نقاط متوقعة: {tp_data['points']:.1f} | "
                              f"جدوى: {feasibility['score']}/10 | {why_scalp}")
                        
                        # تخزين بيانات TP
                        STATE["scalp_tp_data"] = tp_data
                        
                        # فتح الصفقة
                        if open_market(sig_scalp, qty, float(px), mode_override="scalp", 
                                      reason_override=why_scalp, engine_tag="SCALP_PRO"):
                            # تحديث عداد السكالب اليومي
                            STATE["scalp_trades_today"] = int(STATE.get("scalp_trades_today", 0)) + 1
                            log_i(f"📊 SCALP COUNT: {STATE['scalp_trades_today']}/{MAX_SCALP_TRADES_PER_DAY}")
                            continue
                    else:
                        log_w(f"⛔ SCALP REJECTED | جدوى: {feasibility['score']}/10 | "
                              f"أسباب: {', '.join(feasibility['reasons'])}")
                        if feasibility.get("warnings"):
                            log_w(f"تحذيرات: {', '.join(feasibility['warnings'])}")

            # منطق الدخول النهائي المعدل:
            if not STATE["open"] and sig:
                allow_wait, wait_reason = wait_gate_allow_side(sig)
                if not allow_wait:
                    reason = wait_reason
                else:
                    px_now = float(px or info["price"])
                    boxes = detect_simple_boxes(df, lookback=ZONE_BOX_LOOKBACK)
                    
                    # فلتر الجدوى للسكالب
                    is_scalp_entry = (reason or "").startswith(("BOX_FIRST", "ZONE_FIRST"))
                    
                    if is_scalp_entry:
                        # حساب الجدوى
                        feasibility = calculate_scalp_feasibility(df, ind, px_now, sig, boxes)
                        
                        if feasibility["feasible"]:
                            qty = compute_size(bal, px_now)
                            if qty > 0:
                                # حساب TP ديناميكي
                                tp_data = calculate_dynamic_scalp_tp(px_now, sig, feasibility, ind)
                                STATE["scalp_tp_data"] = tp_data
                                
                                ok = open_market(sig, qty, px_now)
                                if ok:
                                    wait_for_next_signal_side = None
                                    # تحديث عداد السكالب
                                    STATE["scalp_trades_today"] = int(STATE.get("scalp_trades_today", 0)) + 1
                                    log_i(f"📊 SCALP COUNT: {STATE['scalp_trades_today']}/{MAX_SCALP_TRADES_PER_DAY}")
                            else:
                                reason = "qty<=0"
                        else:
                            reason = f"scalp_not_feasible(score={feasibility['score']}/10)"
                            log_w(f"⛔ SCALP REJECTED | {reason}")
                    else:
                        # دخول ترند (Council/Golden)
                        qty = compute_size(bal, px_now)
                        if qty > 0:
                            ok = open_market(sig, qty, px_now)
                            if ok:
                                wait_for_next_signal_side = None
                        else:
                            reason = "qty<=0"
            
            # اللوج الاحترافي
            if LOG_LEGACY:
                now = time.time()
                if now - _last_pretty_ts >= LOG_THROTTLE_S:
                    _last_pretty_ts = now
                    pretty_snapshot(bal, {"price": px or info["price"], **info}, ind, spread_bps, reason, df)
            
            loop_i += 1
            sleep_s = NEAR_CLOSE_S if time_to_candle_close(df) <= 10 else BASE_SLEEP
            time.sleep(sleep_s)
            
        except Exception as e:
            log_e(f"loop error: {e}\n{traceback.format_exc()}")
            logging.error(f"trade_loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

# استبدال حلقة التداول الأصلية بالمحسنة
trade_loop = trade_loop_enhanced

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
        print(f"   🎯 ENTRY: TREND BIRTH ENGINE + SCALP PRO + GOLDEN ENTRY + FALCON  |  spread_bps={fmt(spread_bps,2)}")
        print(f"   ⏱️ closes_in ≈ {left_s}s")
        print("\n🧭 POSITION")
        bal_line = f"Balance={fmt(bal,2)}  Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x  CompoundPnL={fmt(compound_pnl)}  Eq~{fmt((bal or 0)+compound_pnl,2)}"
        print(colored(f"   {bal_line}", "yellow"))
        if STATE["open"]:
            lamp='🟩 LONG' if STATE['side']=='long' else '🟥 SHORT'
            print(f"   {lamp}  Entry={fmt(STATE['entry'])}  Qty={fmt(STATE['qty'],4)}  Bars={STATE['bars']}  Trail={fmt(STATE['trail'])}  BE={fmt(STATE['breakeven'])}")
            print(f"   🎯 TP_done={STATE['profit_targets_achieved']}  HP={fmt(STATE['highest_profit_pct'],2)}%")
            if STATE.get("falcon_tp1"):
                print(f"   🦅 FALCON TP1={fmt(STATE['falcon_tp1'])}")
            if STATE.get("falcon_tp2"):
                print(f"   🦅 FALCON TP2={fmt(STATE['falcon_tp2'])} {'✓' if STATE.get('falcon_tp2_done') else ''}")
            if STATE.get("falcon_tp3"):
                print(f"   🦅 FALCON TP3={fmt(STATE['falcon_tp3'])} {'✓' if STATE.get('falcon_tp3_done') else ''}")
            if STATE.get("entry_engine"):
                print(f"   🔧 Engine: {STATE['entry_engine']} | Reason: {STATE.get('entry_reason', 'N/A')}")
        else:
            print("   ⚪ FLAT")
            if wait_for_next_signal_side:
                print(colored(f"   ⏳ Waiting for opposite RF: {wait_for_next_signal_side.upper()}", "cyan"))
        if reason: print(colored(f"   ℹ️ reason: {reason}", "white"))
        if STATE.get("cooldown_until_ts", 0) > time.time():
            time_left = int(STATE["cooldown_until_ts"] - time.time())
            print(colored(f"   ⏸️ COOLDOWN: {time_left//60}m:{time_left%60}s left", "yellow"))
        print(colored("─"*100,"cyan"))

# =================== API / KEEPALIVE ===================
app = Flask(__name__)
@app.route("/")
def home():
    mode='LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ Council PRO Bot v5.0 — {SYMBOL} {INTERVAL} — {mode} — Trend Birth Engine + Scalp Pro + FALCON STYLE"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "state": STATE, "compound_pnl": compound_pnl,
        "entry_mode": "TREND_BIRTH_ENGINE+SCALP_PRO+GOLDEN", "wait_for_next_signal": wait_for_next_signal_side,
        "guards": {"max_spread_bps": MAX_SPREAD_BPS, "final_chunk_qty": FINAL_CHUNK_QTY},
        "falcon": {"tp1": STATE.get("falcon_tp1"), "tp2": STATE.get("falcon_tp2"), "tp3": STATE.get("falcon_tp3")},
        "engines": {"tbe_enabled": TBE_ENABLED, "scalp_pro_enabled": SCALP_PRO_ENABLED}
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True, "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"], "side": STATE["side"], "qty": STATE["qty"],
        "compound_pnl": compound_pnl, "timestamp": datetime.utcnow().isoformat(),
        "entry_mode": "TREND_BIRTH_ENGINE+SCALP_PRO+GOLDEN", "wait_for_next_signal": wait_for_next_signal_side,
        "cooldown_active": STATE.get("cooldown_until_ts", 0) > time.time(),
        "entry_engine": STATE.get("entry_engine")
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
    log_banner("INIT")
    state = load_state() or {}
    state.setdefault("in_position", False)

    if RESUME_ON_RESTART:
        try:
            state = resume_open_position(ex, SYMBOL, state)
        except Exception as e:
            log_w(f"resume error: {e}\n{traceback.format_exc()}")

    verify_execution_environment()

    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'}  •  {SYMBOL}  •  {INTERVAL}", "yellow"))
    print(colored(f"RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x  •  TREND BIRTH ENGINE=ENABLED", "yellow"))
    print(colored(f"SCALP PRO: {MAX_SCALP_TRADES_PER_DAY} scalps/day • TP Boost: {SCALP_TP1_BOOST_PCT}/{SCALP_TP2_BOOST_PCT}/{SCALP_TP3_BOOST_PCT}%", "yellow"))
    print(colored(f"TBE: Sweep→Shift→Momentum→Displacement({TBE_DISPLACE_ATR_MULT}×ATR)→OB/FVG→Retest→Confirm", "yellow"))
    print(colored(f"SCALP PRO: Box→Sweep→Displacement→Retest→Confirm", "yellow"))
    print(colored(f"FLOW PRESSURE: Delta+OBI+Walls detection", "yellow"))
    print(colored(f"GOLDEN ENTRY: score≥{GOLDEN_ENTRY_SCORE} | ADX≥{GOLDEN_ENTRY_ADX}", "yellow"))
    print(colored(f"FALCON STYLE: TP1/2/3 + Box Exit + Early Fail + Cooldown", "yellow"))
    print(colored(f"EXECUTION: {'ACTIVE' if EXECUTE_ORDERS and not DRY_RUN else 'SIMULATION'}", "yellow"))
    
    logging.info("service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    
    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)

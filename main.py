# -*- coding: utf-8 -*-
"""
RF Futures Bot — RF-LIVE ONLY (BingX Perp via CCXT)
• Council PRO Unified Decision System with Candles & Golden Entry
• Golden Entry + Golden Reversal + Wick Exhaustion + Smart Profit AI
• Dynamic TP ladder + Breakeven + ATR-trailing
• Smart Exit Management + Wait-for-next-signal
• Professional Logging & Dashboard
• Enhanced with Footprint, SMC Candles, Liquidity Traps + VWAP Strategy
• OTC Hidden Flow Detection & Protection System
• EMA Crossover Strength Engine (Strong/Weak Trend Detection)
• ENHANCED WITH: Hard Stop Loss, Post-Big-Win Guard, Auto-Recovery
• CHART PRIME: High-Volume Boxes + Liquidity Cycle Monitor
• HUNTER PATCH: Unified Entry, ADX/ATR Smart Monitoring, 8-Tick Iron SL
• MARKET STATE ENGINE: Accumulation / Liquidity / Breakout / Shock Detection
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
BOT_VERSION = "DOGE Council PRO v6.0 — Smart Profit AI + Golden Zone Pro + VWAP Strategy + OTC Detection + EMA Crossover Engine + Hard Stop + Post-Big-Win Guard + Auto-Recovery + ChartPrime Liquidity Monitor + HUNTER PATCH + Market State Engine"
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

# ================== RISK & GUARDS CONFIG ==================
MAX_LOSS_PCT      = -0.005   # -0.50% حدّ خسارة صارم للصفقة الواحدة
BIG_WIN_PCT       =  0.020   # +2.0% أو أكثر تعتبر صفقة Big Win
POST_BIG_WIN_BARS =  2       # عدد الشموع بعد Big Win يتم فيها تفعيل الحذر

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

# EMA Engine
EMA_FAST_LEN = int(os.getenv("EMA_FAST_LEN", 21))
EMA_SLOW_LEN = int(os.getenv("EMA_SLOW_LEN", 55))

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
BASE_SLEEP   = 15
NEAR_CLOSE_S = 5
POSITION_STATUS_LOG_INTERVAL = 15

# ==== Smart Exit Tuning ===
TP1_SCALP_PCT      = 0.35/100
TP1_TREND_PCT      = 0.60/100
HARD_CLOSE_PNL_PCT = 1.10/100
WICK_ATR_MULT      = 1.5
EVX_SPIKE          = 1.8
BM_WALL_PROX_BPS   = 5
TIME_IN_TRADE_MIN  = 8
TRAIL_TIGHT_MULT   = 1.20

# ==== OTC Hidden Flow Detection Settings ====
OTC_WINDOW_BARS          = 5
OTC_MIN_MOVE_BPS         = 60.0
OTC_MAX_VISIBLE_FLOW_PCT = 0.25
OTC_STRENGTH_SCALE       = 0.1

# ==== OTC Exit Tuning (بعد TP1) ====
OTC_EXIT_MIN_STRENGTH = 2.0
OTC_EXIT_MIN_PNL_PCT  = 0.60/100

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
DISPLACEMENT_THRESHOLD = 0.002

# ==== VWAP Settings ====
VWAP_ENABLED = True
VWAP_SCALP_BAND_BPS = 8.0
VWAP_TREND_BAND_BPS = 20.0

# ===================== HUNTER PATCH CONFIG =====================
USE_IRON_SL_TICKS = True
HARD_SL_TICKS = 8        # 8 نقاط Bybit
ADX_ACCUM_MAX = 15       # تحتها = تجميع
ADX_TREND_MIN = 22       # بداية ترند
ADX_EXIT_WEAK = 18       # ضعف ترند للخروج
DI_EDGE = 2.0

EXIT_WEAKNESS_VOTES = 2  # 2 إشارات ضعف = خروج
MIN_PROFIT_TO_EXIT = 0.20  # % ربح أدنى قبل الخروج الذكي
# ===============================================================

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

# =================== GATE TRACE (WHY NO ENTRY) ===================
GATE_TRACE_ENABLED = True

def gate_block(gate_name: str, gate_value=None, missing: str = "", extra: dict = None):
    """
    Unified gate block logger:
    - gate_name: اسم الجيت
    - gate_value: قيمة الجيت (رقم/tuple/str)
    - missing: إيه اللي ناقص عشان يسمح
    - extra: معلومات إضافية (adx/di/atr/score/...)
    """
    if not GATE_TRACE_ENABLED:
        return
    extra = extra or {}
    gv = gate_value
    try:
        if isinstance(gv, float):
            gv = f"{gv:.4f}"
    except:
        pass

    msg = f"🚫 GATE_BLOCK | {gate_name}={gv} | missing={missing}"
    log_w(msg)

    # اطبع سياق مختصر مفيد
    if extra:
        # خليك مختصر عشان اللوج مايبقاش spam
        keys = ["px","spread_bps","adx","plus_di","minus_di","atr","rsi","buy_score","sell_score","regime","event","strength"]
        ctx = {k: extra.get(k) for k in keys if k in extra}
        log_w(f"   ↳ ctx: {ctx}")

# ===================== MARKET STATE ENGINE (ACCUM / LIQ / BREAK / SHOCK) =====================

def _tf(x, d=0.0):
    try:
        return float(x)
    except:
        return d

def _body_ratio(o,h,l,c):
    rng = max(1e-12, float(h)-float(l))
    return abs(float(c)-float(o))/rng

def detect_sr_pivots(df, lookback=80):
    """SR بسيط وعملي (Fallback)"""
    if df is None or len(df) < max(lookback, 30):
        return {"ok": False}
    d = df.tail(lookback)
    sup = float(d["low"].astype(float).min())
    res = float(d["high"].astype(float).max())
    return {"ok": True, "support": sup, "resistance": res, "src": f"pivot{lookback}"}

def atr_regime(ind, px):
    atr = _tf(ind.get("atr"), 0.0)
    if px <= 0 or atr <= 0:
        return {"regime":"unknown","atr":atr,"atr_pct":0.0,"buf_atr":0.40}
    atr_pct = atr/px
    # نظام بسيط، مضبوط للـ 15m
    if atr_pct >= 0.020:
        return {"regime":"explosion","atr":atr,"atr_pct":atr_pct,"buf_atr":0.50}
    if atr_pct >= 0.012:
        return {"regime":"high","atr":atr,"atr_pct":atr_pct,"buf_atr":0.40}
    if atr_pct >= 0.007:
        return {"regime":"normal","atr":atr,"atr_pct":atr_pct,"buf_atr":0.35}
    return {"regime":"quiet","atr":atr,"atr_pct":atr_pct,"buf_atr":0.30}

def breakout_confirmed(df, ind, bias, level):
    """
    bias='buy' => close فوق المقاومة
    bias='sell' => close تحت الدعم
    شروط: Close + Body محترم + ADX/DI يؤيد
    """
    if df is None or len(df) < 3 or level is None:
        return {"ok": False, "why": "no_df_or_level"}

    o = float(df["open"].iloc[-1]); c = float(df["close"].iloc[-1])
    h = float(df["high"].iloc[-1]); l = float(df["low"].iloc[-1])
    body_r = _body_ratio(o,h,l,c)

    adx = _tf(ind.get("adx"), 0.0)
    pdi = _tf(ind.get("plus_di"), 0.0)
    mdi = _tf(ind.get("minus_di"), 0.0)
    di_spread = abs(pdi-mdi)

    BODY_MIN = 0.55
    ADX_MIN = 18.0
    DI_MIN  = 8.0

    cond_body = body_r >= BODY_MIN
    cond_adx  = adx >= ADX_MIN
    cond_di   = di_spread >= DI_MIN

    if bias == "buy":
        cond_lvl = c > float(level)
        cond_dir = pdi >= mdi
        ok = cond_lvl and cond_body and cond_adx and cond_di and cond_dir
        why = f"c>{level}:{cond_lvl} body={cond_body}({body_r:.2f}) adx={cond_adx}({adx:.1f}) di={cond_di}({di_spread:.1f}) dir={cond_dir}"
        return {"ok": ok, "why": why}

    cond_lvl = c < float(level)
    cond_dir = mdi >= pdi
    ok = cond_lvl and cond_body and cond_adx and cond_di and cond_dir
    why = f"c<{level}:{cond_lvl} body={cond_body}({body_r:.2f}) adx={cond_adx}({adx:.1f}) di={cond_di}({di_spread:.1f}) dir={cond_dir}"
    return {"ok": ok, "why": why}

def liquidity_grab(df, sr, px, atr):
    """
    كشف سحب السيولة (Fakeout): wick فوق مقاومة/تحت دعم + رجوع بالإغلاق.
    """
    if df is None or len(df) < 3 or not sr.get("ok") or atr <= 0:
        return {"ok": False}

    o = float(df["open"].iloc[-1]); c = float(df["close"].iloc[-1])
    h = float(df["high"].iloc[-1]); l = float(df["low"].iloc[-1])
    sup = float(sr["support"]); res = float(sr["resistance"])
    buf = 0.20 * atr  # صغير عشان يمسك السحب الحقيقي

    fake_up = (h > res + buf) and (c < res - buf*0.25)   # طلع فوق وبعدين قفل جوه
    fake_dn = (l < sup - buf) and (c > sup + buf*0.25)   # نزل تحت وبعدين قفل جوه

    return {"ok": True, "fakeout_up": bool(fake_up), "fakeout_dn": bool(fake_dn), "support": sup, "resistance": res}

def shock_detector(df, ind, px):
    """
    انفجار/انهيار: ATR% عالي + جسم كبير + حجم أعلى من MA20
    """
    if df is None or len(df) < 25 or px <= 0:
        return {"shock": False}

    reg = atr_regime(ind, px)
    atr = reg["atr"]
    if atr <= 0:
        return {"shock": False}

    vol = df["volume"].astype(float)
    vma = float(vol.tail(20).mean() or 1e-9)
    v1 = float(vol.iloc[-1])

    o = float(df["open"].iloc[-1]); c = float(df["close"].iloc[-1])
    h = float(df["high"].iloc[-1]); l = float(df["low"].iloc[-1])
    body = abs(c-o)
    rng  = h-l

    atr_pct_ok = reg["atr_pct"] >= 0.012
    body_ok = body >= 1.2*atr
    vol_ok  = v1 >= 1.5*vma

    shock = atr_pct_ok and body_ok and vol_ok
    direction = "pump" if c>o else "dump"
    strength = int(min(100, (40 if atr_pct_ok else 0) + (35 if body_ok else 0) + (25 if vol_ok else 0)))

    return {"shock": bool(shock), "dir": direction, "strength": strength, "atr_pct": reg["atr_pct"], "body_atr": body/max(1e-12,atr), "vol_mult": v1/max(1e-12,vma), "regime": reg["regime"]}

def market_state_engine(df, ind, px, extra_zones=None):
    """
    يطلع حالة سوق موحّدة:
    phase: ACCUM / PREP / TREND / EXPANSION
    bias : bull / bear / neutral
    plus: breakout/breakdown confirmed, liquidity grabs, shock
    """
    adx = _tf(ind.get("adx", 0.0))
    pdi = _tf(ind.get("plus_di", 0.0))
    mdi = _tf(ind.get("minus_di", 0.0))

    bias = "neutral"
    if pdi > mdi: bias = "bull"
    elif mdi > pdi: bias = "bear"

    reg = atr_regime(ind, px)
    sr = detect_sr_pivots(df, lookback=80)

    # phase logic
    if adx < 15 and reg["regime"] in ["quiet","normal"]:
        phase = "ACCUMULATION"
    elif adx < 22:
        phase = "PREP"
    elif adx < 30:
        phase = "TREND"
    else:
        phase = "TREND_STRONG"

    # Confirmed breakouts
    brk = {"ok": False}; bdn = {"ok": False}
    if sr.get("ok"):
        brk = breakout_confirmed(df, ind, "buy", sr["resistance"])
        bdn = breakout_confirmed(df, ind, "sell", sr["support"])

        if brk["ok"] or bdn["ok"]:
            phase = "EXPANSION"

    # Liquidity grab
    liq = liquidity_grab(df, sr, px, reg["atr"])

    # Shock
    shock = shock_detector(df, ind, px)

    return {
        "ok": True,
        "phase": phase,
        "bias": bias,
        "sr": sr,
        "atr_regime": reg,
        "breakout_up": brk,
        "breakout_dn": bdn,
        "liq": liq,
        "shock": shock
    }

def momentum_gate_log(ms):
    """Gate لوج واضح: ليه Momentum اتفعل أو اترفض"""
    if not ms or not ms.get("ok"):
        return

    sh = ms["shock"]
    brk_up = ms["breakout_up"]
    brk_dn = ms["breakout_dn"]
    liq = ms["liq"]
    reg = ms["atr_regime"]

    # Momentum is allowed if shock OR confirmed breakout/breakdown
    allow = bool(sh.get("shock") or brk_up.get("ok") or brk_dn.get("ok"))
    reason = "clear"
    if sh.get("shock"):
        reason = f"SHOCK {sh.get('dir')} str={sh.get('strength')}"
    elif brk_up.get("ok"):
        reason = f"BREAKOUT_OK {brk_up.get('why')}"
    elif brk_dn.get("ok"):
        reason = f"BREAKDOWN_OK {brk_dn.get('why')}"
    else:
        reason = "need shock OR breakout_confirmed"

    # Print as Gate trace
    try:
        log_i(
            f"{'✅' if allow else '⛔'} MOMENTUM_GATE | allow={allow} | {reason} | "
            f"phase={ms.get('phase')} bias={ms.get('bias')} "
            f"adx={_tf(getattr(ms,'adx', None),0.0) if False else ''}"
            f" atr%={reg.get('atr_pct',0)*100:.2f}% reg={reg.get('regime')} "
            f"liq(up/dn)={liq.get('fakeout_up')}/{liq.get('fakeout_dn')}"
        )
    except:
        pass

# =================== HUNTER PATCH UTILITIES ===================
def get_tick_size(exchange, symbol):
    try:
        m = exchange.market(symbol)
        p = m.get("precision", {}).get("price", None)
        if isinstance(p, int):
            return 10 ** (-p)
        info = m.get("info", {}) or {}
        for k in ("tickSize", "tick_size"):
            if k in info and float(info[k]) > 0:
                return float(info[k])
    except Exception:
        pass
    return 0.0001  # fallback

def compute_iron_sl(entry_price, side, tick_size, ticks):
    dist = tick_size * ticks
    return entry_price - dist if side == "long" else entry_price + dist

def adx_atr_watcher(ind, state):
    adx = float(ind.get("adx", 0))
    atr = float(ind.get("atr", 0))
    dip = float(ind.get("plus_di", 0))
    dim = float(ind.get("minus_di", 0))

    hist_adx = state.setdefault("hist_adx", [])
    hist_atr = state.setdefault("hist_atr", [])
    hist_adx.append(adx); hist_atr.append(atr)
    if len(hist_adx) > 20: hist_adx.pop(0)
    if len(hist_atr) > 20: hist_atr.pop(0)

    adx_slope = hist_adx[-1] - hist_adx[-3] if len(hist_adx) >= 3 else 0
    atr_ma = sum(hist_atr) / len(hist_atr)
    atr_reg = "expand" if atr > atr_ma * 1.05 else "contract" if atr < atr_ma * 0.95 else "stable"

    if adx <= ADX_ACCUM_MAX:
        regime = "ACCUMULATION"
    elif adx >= ADX_TREND_MIN:
        regime = "TREND"
    else:
        regime = "PREP"

    side = "up" if dip > dim + DI_EDGE else "down" if dim > dip + DI_EDGE else "flat"

    return {
        "adx": adx,
        "atr": atr,
        "adx_slope": adx_slope,
        "atr_reg": atr_reg,
        "regime": regime,
        "side": side
    }

def accumulation_launch_signal(ind, state):
    adx = ind["adx"]
    dip = ind["plus_di"]
    dim = ind["minus_di"]

    w = state.get("watcher", {})
    atr_reg = w.get("atr_reg")

    hist_adx = state.get("hist_adx", [])
    adx_rising = len(hist_adx) >= 3 and hist_adx[-1] > hist_adx[-2] > hist_adx[-3]

    compression = adx <= ADX_ACCUM_MAX and atr_reg == "contract"

    launch = compression and adx_rising and atr_reg in ("stable", "expand")

    side = None
    if launch:
        if dip > dim + DI_EDGE:
            side = "buy"
        elif dim > dip + DI_EDGE:
            side = "sell"

    return {
        "compression": compression,
        "launch": bool(launch and side),
        "side": side
    }

# =================== DYNAMIC ATR REGIME ===================
def atr_regime_ind(ind: dict):
    """
    Determines ATR regime to adjust buffers dynamically.
    Returns: ("quiet" | "normal" | "expansion", buffer_mult)
    """
    atr = float(ind.get("atr", 0.0) or 0.0)
    atr_ma = float(ind.get("atr_ma", 0.0) or atr)  # لو عندك ATR MA، وإلا fallback
    if atr <= 0:
        return "normal", 0.40

    ratio = atr / max(atr_ma, 1e-9)

    if ratio < 0.85:
        return "quiet", 0.50     # سوق هادي → وسّع الحظر
    elif ratio > 1.20:
        return "expansion", 0.30 # انفجار → ضيّق الحظر
    else:
        return "normal", 0.40

# =================== SMART ENTRY GATE ===================
def smart_entry_gate(sig, df, ind, gz=None):
    """
    Unified Gate:
    - SR rule (no BUY under RES / no SELL above SUP)
    - Golden Zone Validator override
    - Dynamic ATR buffer by regime
    """
    # استخدام analyze_structure_liquidity إذا كانت موجودة، وإلا استخدام القيم الأساسية
    try:
        sl = analyze_structure_liquidity(df, ind)
        sup = sl.get("support")
        res = sl.get("resistance")
        event = sl.get("event", "none")
        strength = sl.get("strength", 0)
    except:
        # إذا لم تكن الدالة موجودة، استخدام قيم افتراضية
        sup = ind.get("support") or 0
        res = ind.get("resistance") or float('inf')
        event = "none"
        strength = 0

    atr = float(ind.get("atr", 0.0) or 0.0)
    adx = float(ind.get("adx", 0.0) or 0.0)
    pdi = float(ind.get("plus_di", 0.0) or 0.0)
    mdi = float(ind.get("minus_di", 0.0) or 0.0)
    px  = float(ind.get("price", df["close"].iloc[-1]))

    if atr <= 0 or sup is None or res is None:
        return True, "no_sr_context"

    # === Dynamic ATR buffer ===
    regime, buf_mult = atr_regime_ind(ind)
    buf = atr * buf_mult

    # === Golden Zone Validation ===
    gz_ok = False
    gz_score = 0.0
    if gz and gz.get("ok"):
        gz_ok = True
        gz_score = float(gz.get("score", 0.0))

    # ================= BUY =================
    if sig == "buy":
        near_res = abs(px - res) <= buf

        # Golden Bottom override (even under resistance)
        if gz_ok and gz.get("zone", {}).get("type") == "golden_bottom" and gz_score >= 6.0:
            return True, f"BUY Golden validated (score={gz_score:.1f}, regime={regime})"

        if near_res:
            breakout_ok = (
                event == "breakout" and
                strength >= 80 and
                adx >= 18 and
                pdi > mdi
            )
            if breakout_ok:
                return True, f"BUY breakout confirmed (regime={regime})"
            return False, f"BLOCK BUY near RES | regime={regime}"

    # ================= SELL =================
    if sig == "sell":
        near_sup = abs(px - sup) <= buf

        # Golden Top override
        if gz_ok and gz.get("zone", {}).get("type") == "golden_top" and gz_score >= 6.0:
            return True, f"SELL Golden validated (score={gz_score:.1f}, regime={regime})"

        if near_sup:
            breakdown_ok = (
                event == "breakdown" and
                strength >= 80 and
                adx >= 18 and
                mdi > pdi
            )
            if breakdown_ok:
                return True, f"SELL breakdown confirmed (regime={regime})"
            return False, f"BLOCK SELL near SUP | regime={regime}"

    return True, f"gate_pass (regime={regime})"

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

# =================== OTC HIDDEN FLOW DETECTION ===================
def detect_otc_flows(df, window: int = OTC_WINDOW_BARS):
    """
    كشف سيولة OTC (شراء/بيع مخفي) من خلال:
    - حركة سعر قوية في آخر N شمعة
    - مع فلو ظاهر (delta) ضعيف مقارنة بالفوليوم
    """
    try:
        if len(df) < window + 2:
            return {
                "otc_buy": False,
                "otc_sell": False,
                "strength": 0.0,
                "reason": "",
            }

        close  = df["close"].astype(float)
        volume = df["volume"].astype(float)

        # حركة السعر في آخر window شمعة
        start_price = float(close.iloc[-window])
        last_price  = float(close.iloc[-1])
        if start_price <= 0:
            return {
                "otc_buy": False,
                "otc_sell": False,
                "strength": 0.0,
                "reason": "",
            }

        ret = (last_price - start_price) / start_price  # نسبة الحركة
        move_bps = abs(ret) * 10000.0

        # لو الحركة أصلاً ضعيفة ما نضيعش وقت
        if move_bps < OTC_MIN_MOVE_BPS:
            return {
                "otc_buy": False,
                "otc_sell": False,
                "strength": 0.0,
                "reason": "",
            }

        # تقريب delta من الفوليوم + اتجاه الشمعة
        sub_close  = close.iloc[-window:]
        sub_vol    = volume.iloc[-window:]
        price_diff = sub_close.diff()

        buy_vol  = sub_vol.where(price_diff > 0, 0.0)
        sell_vol = sub_vol.where(price_diff < 0, 0.0)
        delta_series = buy_vol - sell_vol

        cum_delta_window = float(delta_series.sum())
        total_vol        = float(sub_vol.sum() or 1.0)

        visible_flow_ratio = abs(cum_delta_window) / total_vol  # نسبة الفلو الظاهر للفوليوم

        # لو الفلو الظاهر ضعيف جدًا مقارنة بالفوليوم مع حركة سعر قوية -> OTC
        otc_buy = False
        otc_sell = False
        strength = 0.0
        reason = ""

        # Pump بلا فلو شرائي واضح -> OTC BUY
        if ret > 0:
            if cum_delta_window <= 0 or visible_flow_ratio < OTC_MAX_VISIBLE_FLOW_PCT:
                otc_buy = True
                # قوة الـ OTC = حركة السعر / الفلو الظاهر
                strength = (move_bps * OTC_STRENGTH_SCALE) / max(visible_flow_ratio, 0.05)
                reason = "price_up_without_visible_buy_flow"

        # Dump بلا فلو بيعي واضح -> OTC SELL
        elif ret < 0:
            if cum_delta_window >= 0 or visible_flow_ratio < OTC_MAX_VISIBLE_FLOW_PCT:
                otc_sell = True
                strength = (move_bps * OTC_STRENGTH_SCALE) / max(visible_flow_ratio, 0.05)
                reason = "price_down_without_visible_sell_flow"

        return {
            "otc_buy": bool(otc_buy),
            "otc_sell": bool(otc_sell),
            "strength": float(strength),
            "reason": reason,
            "move_bps": move_bps,
            "visible_flow_ratio": visible_flow_ratio,
        }
    except Exception as e:
        # في حالة أي خطأ ما نكسرش المجلس
        return {
            "otc_buy": False,
            "otc_sell": False,
            "strength": 0.0,
            "reason": f"error:{e}",
        }

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

# =================== EXECUTION VERIFICATION ===================
def verify_execution_environment():
    """التحقق من بيئة التنفيذ عند الإقلاع"""
    print(f"⚙️ EXECUTION ENVIRONMENT", flush=True)
    print(f"🔧 EXECUTE_ORDERS: {EXECUTE_ORDERS} | SHADOW_MODE: {SHADOW_MODE_DASHBOARD} | DRY_RUN: {DRY_RUN}", flush=True)
    print(f"🎯 GOLDEN ENTRY PRO: score={GOLDEN_ENTRY_SCORE} | ADX={GOLDEN_ENTRY_ADX}", flush=True)
    print(f"📈 ENHANCED CANDLES: SMC Patterns + Liquidity Traps", flush=True)
    print(f"👣 FOOTPRINT ANALYSIS: Volume spikes + Absorption", flush=True)
    print(f"💰 OTC DETECTION: Hidden flow detection + Protection", flush=True)
    print(f"📊 VWAP STRATEGY: SCALP(near {VWAP_SCALP_BAND_BPS}bps) | TREND(far {VWAP_TREND_BAND_BPS}bps)", flush=True)
    print(f"📈 EMA CROSSOVER ENGINE: Strong/Weak Trend Detection", flush=True)
    print(f"🛡️ RISK GUARDS: Hard Stop Loss (-{abs(MAX_LOSS_PCT)*100}%) | Post-Big-Win Guard (+{BIG_WIN_PCT*100}%)", flush=True)
    print(f"📦 CHART PRIME: High-Volume Boxes + Liquidity Cycle Monitor", flush=True)
    print(f"🎯 HUNTER PATCH: Unified Entry | ADX/ATR Smart Monitor | 8-Tick Iron SL", flush=True)
    print(f"🚀 MARKET STATE ENGINE: Accumulation / Liquidity / Breakout / Shock Detection", flush=True)
    
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

# =================== EMA CROSSOVER STRENGTH ENGINE ===================
def ema_crossover_strength(df, ind=None, fast=9, mid=21, slow=50):
    """
    تصنيف تقاطع EMA:
    - strong_bull / weak_bull
    - strong_bear / weak_bear
    - none

    يرجّع:
    {
        "side": "bull"/"bear"/"flat",
        "label": "strong_bull"/"weak_bull"/"strong_bear"/"weak_bear"/"none",
        "score": float
    }
    """
    try:
        if len(df) < slow + 5:
            return {"side": "flat", "label": "none", "score": 0.0}

        close = df["close"].astype(float)
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_mid  = close.ewm(span=mid, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        f_now = float(ema_fast.iloc[-1]); f_prev = float(ema_fast.iloc[-2])
        m_now = float(ema_mid.iloc[-1]);  m_prev = float(ema_mid.iloc[-2])
        s_now = float(ema_slow.iloc[-1])
        px    = float(close.iloc[-1])

        cross_up   = (f_prev <= m_prev) and (f_now > m_now)
        cross_down = (f_prev >= m_prev) and (f_now < m_now)

        adx = 0.0
        if ind:
            adx = float(ind.get("adx", 0.0))

        side  = "flat"
        label = "none"
        score = 0.0

        # ===== Bullish context =====
        if f_now > m_now >= s_now:
            side = "bull"
            # strong bull: ema fast > mid > slow + price فوق fast + ADX محترم + cross قريب
            if cross_up and adx >= 22 and px > f_now:
                label = "strong_bull"
                score = 3.0
            else:
                label = "weak_bull"
                score = 1.5

        # ===== Bearish context =====
        elif f_now < m_now <= s_now:
            side = "bear"
            if cross_down and adx >= 22 and px < f_now:
                label = "strong_bear"
                score = 3.0
            else:
                label = "weak_bear"
                score = 1.5

        return {"side": side, "label": label, "score": score}
    except Exception:
        return {"side": "flat", "label": "none", "score": 0.0}

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

def decide_strategy_mode_enhanced(df, adx=None, di_plus=None, di_minus=None, rsi_ctx=None, footprint=None, ema_ctx=None):
    """تحديد نمط التداول المحسن: SCALP أم TREND مع VWAP + Footprint + EMA"""
    ind = compute_indicators(df)

    if adx is None or di_plus is None or di_minus is None:
        adx = ind.get('adx', 0)
        di_plus = ind.get('plus_di', 0)
        di_minus = ind.get('minus_di', 0)

    if rsi_ctx is None:
        rsi_ctx = rsi_ma_context(df)

    if footprint is None:
        footprint = compute_footprint_metrics(df)

    if ema_ctx is None:
        ema_ctx = ema_crossover_strength(df, ind)

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

    # تقاطع EMA قوي يفرض نمط الترند
    ema_label = ema_ctx.get("label", "none")
    if ema_label in ("strong_bull", "strong_bear"):
        return {"mode": "trend", "why": "ema_strong_cross"}

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

# =================== SCALP DIRECTION GUARD ===================
def enforce_scalp_trend_alignment(df, ind, planned_side: str, mode_ctx: dict):
    """
    حارس سكالب:
    - لو mode = scalp لازم نمشي مع اتجاه الترند العام.
    - Bull ⇒ سكالب BUY (long) فقط
    - Bear ⇒ سكالب SELL (short) فقط
    - Flat / Chop ⇒ نلغي الصفقة السكالب.
    يرجّع:
      side, skip
    """
    try:
        if mode_ctx.get("mode") != "scalp":
            return planned_side, False  # مش سكالب → ما نغيّرش حاجة

        # اتجاه EMA العام
        ema_ctx = ema_crossover_strength(df, ind)
        trend_side = ema_ctx.get("side", "flat")   # bull / bear / flat
        ema_label = ema_ctx.get("label", "none")

        # سياق RSI
        rsi_ctx = rsi_ma_context(df)
        rsi_trend = rsi_ctx.get("trendZ")          # bull / bear / chop

        # Footprint / OrderFlow
        footprint = compute_footprint_metrics(df)
        flow_trend = footprint.get("delta_trend") if footprint.get("ok") else None  # bull/bear

        # بناء bias نهائي
        bias = "flat"

        # Flow قوي ياخد أولوية لو موجود
        if flow_trend in ("bull", "bear"):
            bias = flow_trend
        else:
            # بعده EMA
            if trend_side in ("bull", "bear"):
                bias = trend_side

        # لو RSI واضح ومش متعارض مع الـ bias، نعزّزه
        if rsi_trend in ("bull", "bear"):
            if bias == "flat":
                bias = rsi_trend
            elif bias != rsi_trend:
                # تعارض قوي ⇒ نعتبر السوق فوضوي للسكالب
                log_i(f"⏸️ SCALP: RSI vs EMA/Flow conflict (bias={bias}, rsi={rsi_trend}) → skip")
                return planned_side, True

        if bias == "flat":
            # مفيش اتجاه واضح → نلغي السكالب
            log_i("⏸️ SCALP: no clear trend bias (flat) → skip scalp entry")
            return planned_side, True

        forced_side = "long" if bias == "bull" else "short"

        if planned_side != forced_side:
            log_w(
                f"🔄 SCALP DIRECTION OVERRIDE | planned={planned_side} → forced={forced_side} "
                f"| bias={bias} | ema={ema_label}"
            )
            return forced_side, False

        # الاتجاه planned متماشي مع الترند
        return planned_side, False

    except Exception as e:
        log_w(f"SCALP direction guard error: {e}")
        return planned_side, False

# =================== SMART PROFIT AI ===================
def smart_profit_ai_decision(state, df, ind, mode, side, entry_price, current_price):
    """
    ذكاء اصطناعي لجني الأرباح:
      - يعتمد على trade_profile المخزّن في STATE:
        * STRONG_TREND  → سلم 3 مراحل (TREND_TPS / TREND_TP_FRACS)
        * MID_TREND     → جني أرباح مرة واحدة محترمة (باستخدام SCALP_TPS لكن بحجم محترم)
        * SCALP_LIGHT   → سكالب خفيف هدف واحد وسريع
    """
    pnl_pct = (current_price - entry_price) / entry_price * 100 * (1 if side == "long" else -1)
    
    trade_profile = state.get("trade_profile") or "MID_TREND"
    signal_strength = state.get("signal_strength", 0.0)
    
    # --- اختيار سلم الأهداف حسب الـ Profile ---
    if trade_profile == "STRONG_TREND":
        # ترند قوي → 3 مراحل TP (سلم كامل)
        tp_levels = TREND_TPS[:]          # [0.50, 1.00, 1.80] مبدئياً
        tp_fractions = TREND_TP_FRACS[:]  # [0.30, 0.30, 0.20]
    elif trade_profile == "SCALP_LIGHT":
        # سكالب خفيف → هدف واحد سريع
        tp_levels = SCALP_TPS[:]          # [0.40]
        tp_fractions = SCALP_TP_FRACS[:]  # [0.60]
    else:
        # MID_TREND (نص ترند / باي عادية) → جني أرباح مرة واحدة محترمة
        # نستخدم نفس SCALP_TPS لكن نسمح بسلوك "صفقة محترمة" (Signal boost)
        tp_levels = SCALP_TPS[:]
        tp_fractions = SCALP_TP_FRACS[:]
    
    achieved_targets = state.get("profit_targets_achieved", 0)
    next_target_index = achieved_targets
    
    if next_target_index >= len(tp_levels):
        return {"action": "hold", "target": None, "reason": "كل الأهداف محققة"}
    
    next_target_pct = tp_levels[next_target_index]
    next_target_fraction = tp_fractions[next_target_index]
    
    # لو ما فيش signal_strength في الـ STATE لأي سبب، نحسبه احتياطيًا
    if signal_strength <= 0.0:
        signal_strength = calculate_signal_strength(df, ind, side)
    
    # تعديل الأهداف حسب قوة الإشارة (Fine Tuning جوه الـ Profile)
    if signal_strength >= 8.0:
        # إشارة قوية جداً → نمدّي الهدف شوية
        next_target_pct *= 1.25
    elif signal_strength >= 6.0:
        # إشارة قوية → نمدّي الهدف بسيط
        next_target_pct *= 1.10
    elif signal_strength < 3.0:
        # إشارة ضعيفة → نقلّل الهدف عشان ما نطمعش في سكالب ضعيف
        next_target_pct *= 0.80
    
    # قرار التنفيذ
    if pnl_pct >= next_target_pct:
        return {
            "action": "take_profit",
            "target": next_target_index + 1,
            "target_pct": next_target_pct,
            "fraction": next_target_fraction,
            "reason": (
                f"TP{next_target_index + 1} hit "
                f"({next_target_pct:.2f}%) | profile={trade_profile} "
                f"| strength={signal_strength:.1f}"
            ),
        }
    
    return {
        "action": "hold",
        "target": next_target_index + 1,
        "reason": (
            f"لم يحقق الهدف بعد | "
            f"profile={trade_profile} | "
            f"target={next_target_pct:.2f}% | strength={signal_strength:.1f}"
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
    if gz and gz.get('ok') and gz.get('confirmed'):
        strength += 3.0
    elif gz and gz.get('ok'):
        strength += 1.5
    
    # EMA Crossover impact
    ema_ctx = ema_crossover_strength(df, ind)
    ema_label = ema_ctx.get("label", "none")

    if side == "long":
        if ema_label == "strong_bull":
            strength += 2.0
        elif ema_label == "weak_bull":
            strength += 1.0
        elif ema_label in ("strong_bear", "weak_bear"):
            # إشارة معاكسة تقلل الثقة
            strength -= 1.0
    elif side == "short":
        if ema_label == "strong_bear":
            strength += 2.0
        elif ema_label == "weak_bear":
            strength += 1.0
        elif ema_label in ("strong_bull", "weak_bull"):
            strength -= 1.0

    strength = max(0.0, strength)
    return min(10.0, strength)

# =================== TRADE PROFILE CLASSIFICATION ===================
def classify_trade_profile(signal_strength: float, mode: str) -> str:
    """
    تحويل قوة الإشارة + نمط الصفقة (scalp/trend)
    إلى Profile واضح لإدارة جني الأرباح:
      - STRONG_TREND  → ترند قوي (3 مراحل TP)
      - MID_TREND     → نص ترند / صفقة محترمة (TP مرة واحدة محترمة)
      - SCALP_LIGHT   → سكالب خفيف (TP سريع واحد)
    """
    # أولاً نحدد bucket للقوة
    # 0–3  : ضعيف/خفيف
    # 3–7  : متوسّط (نص ترند)
    # 7–10 : قوي/سوبر ترند
    if signal_strength >= 7.0:
        strength_bucket = "strong"
    elif signal_strength >= 3.0:
        strength_bucket = "mid"
    else:
        strength_bucket = "weak"

    # لو النمط Trend و الإشارة قوية → ترند قوي حقيقي
    if mode == "trend" and strength_bucket == "strong":
        return "STRONG_TREND"
    
    # لو سكالب أو قوة متوسطة → نص ترند / باي عادية
    if strength_bucket == "mid":
        return "MID_TREND"
    
    # أي شيء أضعف → سكالب خفيف
    return "SCALP_LIGHT"

# =================== TRUE ADX/DI CONTEXT + ART CYCLE ===================
def compute_adx_di_context(df: pd.DataFrame, length: int = ADX_LEN):
    """
    حساب ADX / +DI / -DI بأسلوب Wilder (نفس منطق TradingView تقريبًا)
    ويرجع آخر القيم + السلاسل.
    """
    if len(df) < length + 5:
        return {
            "adx": 0.0, "adx_prev": 0.0,
            "plus_di": 0.0, "minus_di": 0.0,
            "plus_di_prev": 0.0, "minus_di_prev": 0.0,
            "series": {}
        }

    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    prev_c = c.shift(1)
    tr1 = (h - l).abs()
    tr2 = (h - prev_c).abs()
    tr3 = (l - prev_c).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_smooth = wilder_ema(tr, length)

    up_move   = h.diff()
    down_move = l.shift(1) - l

    plus_dm  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    plus_dm_s  = wilder_ema(plus_dm, length)
    minus_dm_s = wilder_ema(minus_dm, length)

    plus_di  = 100.0 * (plus_dm_s / atr_smooth.replace(0, 1e-12))
    minus_di = 100.0 * (minus_dm_s / atr_smooth.replace(0, 1e-12))

    di_sum  = (plus_di + minus_di).replace(0, 1e-12)
    di_diff = (plus_di - minus_di).abs()
    dx = 100.0 * (di_diff / di_sum)

    adx = wilder_ema(dx, length)

    return {
        "adx": float(adx.iloc[-1]),
        "adx_prev": float(adx.iloc[-2]) if len(adx) > 1 else float(adx.iloc[-1]),
        "plus_di": float(plus_di.iloc[-1]),
        "minus_di": float(minus_di.iloc[-1]),
        "plus_di_prev": float(plus_di.iloc[-2]) if len(plus_di) > 1 else float(plus_di.iloc[-1]),
        "minus_di_prev": float(minus_di.iloc[-2]) if len(minus_di) > 1 else float(minus_di.iloc[-1]),
        "series": {
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "dx": dx,
            "atr": atr_smooth,
        },
    }


def analyze_adx_cycle(adx_ctx: dict):
    """
    يحوّل أرقام ADX/DI إلى حالة مفهومة:
      phase: flat/accumulation/expansion/trend/exhaustion/neutral
      trend_side: up/down/chop
      di_cross_up / di_cross_down
    """
    adx       = float(adx_ctx.get("adx", 0.0))
    adx_prev  = float(adx_ctx.get("adx_prev", 0.0))
    plus_di   = float(adx_ctx.get("plus_di", 0.0))
    minus_di  = float(adx_ctx.get("minus_di", 0.0))
    plus_prev = float(adx_ctx.get("plus_di_prev", 0.0))
    minus_prev= float(adx_ctx.get("minus_di_prev", 0.0))

    rising  = adx > adx_prev + 0.5
    falling = adx < adx_prev - 0.5

    if adx < 15:
        phase = "flat"
    elif adx < 18:
        phase = "accumulation"
    elif adx >= 18 and adx < 25 and rising:
        phase = "expansion"
    elif adx >= 25 and not falling:
        phase = "trend"
    elif adx >= 30 and falling:
        phase = "exhaustion"
    else:
        phase = "neutral"

    if plus_di > minus_di + 3:
        trend_side = "up"
    elif minus_di > plus_di + 3:
        trend_side = "down"
    else:
        trend_side = "chop"

    di_cross_up   = (plus_prev <= minus_prev) and (plus_di > minus_di)
    di_cross_down = (minus_prev <= plus_prev) and (minus_di > plus_di)

    strong_trend = (phase == "trend" and adx >= 25 and abs(plus_di - minus_di) >= 5)

    return {
        "phase": phase,
        "adx": adx,
        "adx_prev": adx_prev,
        "rising": rising,
        "falling": falling,
        "trend_side": trend_side,
        "di_cross_up": di_cross_up,
        "di_cross_down": di_cross_down,
        "strong_trend": strong_trend,
        "plus_di": plus_di,
        "minus_di": minus_di,
    }


def compute_atr_series(df: pd.DataFrame, length: int = ATR_LEN) -> pd.Series:
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    prev_c = c.shift(1)
    tr1 = (h - l).abs()
    tr2 = (h - prev_c).abs()
    tr3 = (l - prev_c).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return wilder_ema(tr, length)


def analyze_art_cycle(df: pd.DataFrame, atr_len: int = ATR_LEN, window: int = 20):
    """
    ART Cycle:
      accumulation: ATR/Vol/Range منخفضة
      impulse: ATR/Vol/Range عالية
      correction: وسط
      distribution: حركة متذبذبة
    """
    if len(df) < max(atr_len, window) + 5:
        return {"phase": "unknown", "atr": 0.0, "atr_rel": 1.0, "vol_rel": 1.0, "rng_rel": 1.0}

    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    vol   = df["volume"].astype(float)

    atr_series = compute_atr_series(df, atr_len)
    atr_now = float(atr_series.iloc[-1])
    atr_med = float(atr_series.tail(window).median() or atr_now)

    vol_now = float(vol.iloc[-1])
    vol_med = float(vol.tail(window).median() or vol_now)

    rng = (high - low).astype(float)
    rng_now = float(rng.iloc[-1])
    rng_med = float(rng.tail(window).median() or rng_now)

    atr_rel = atr_now / (atr_med or 1e-9)
    vol_rel = vol_now / (vol_med or 1e-9)
    rng_rel = rng_now / (rng_med or 1e-9)

    if atr_rel < 0.8 and vol_rel < 0.8 and rng_rel < 0.8:
        phase = "accumulation"
    elif atr_rel > 1.2 and vol_rel > 1.2 and rng_rel > 1.2:
        phase = "impulse"
    elif 0.8 <= atr_rel <= 1.2 and 0.8 <= vol_rel <= 1.2:
        phase = "correction"
    else:
        phase = "distribution"

    return {
        "phase": phase,
        "atr": atr_now,
        "atr_rel": atr_rel,
        "vol_rel": vol_rel,
        "rng_rel": rng_rel,
    }

# =================== EMA ENGINE (CROSS & TREND SCORE) ===================
def compute_ema_engine(df: pd.DataFrame,
                       fast_len: int = EMA_FAST_LEN,
                       slow_len: int = EMA_SLOW_LEN) -> dict:
    """
    EMA Engine يعطي:
      - label: strong_bull / bull / chop / bear / strong_bear
      - score: رقم من 0 إلى ~3 يمثل قوة الكروس والترند
      - cross_up / cross_down
      - fast_above_slow, dist_rel, slope_fast
    """
    try:
        if len(df) < max(fast_len, slow_len) + 3:
            return {
                "ok": False,
                "label": "unknown",
                "score": 0.0,
                "cross_up": False,
                "cross_down": False,
                "fast_above_slow": False,
                "dist_rel": 0.0,
                "slope_fast": 0.0,
            }

        close = df["close"].astype(float)
        ema_fast = _ema(close, fast_len)
        ema_slow = _ema(close, slow_len)

        fast_now = float(ema_fast.iloc[-1])
        slow_now = float(ema_slow.iloc[-1])
        fast_prev = float(ema_fast.iloc[-2])
        slow_prev = float(ema_slow.iloc[-2])

        price_now = float(close.iloc[-1])

        spread = fast_now - slow_now
        spread_prev = fast_prev - slow_prev
        dist_rel = abs(spread) / (price_now or 1e-9) * 100.0

        # ميل المتوسط السريع (سلوب بسيط)
        slope_fast = fast_now - float(ema_fast.iloc[-4])

        fast_above_slow = spread > 0

        cross_up = spread_prev <= 0 and spread > 0
        cross_down = spread_prev >= 0 and spread < 0

        # تصنيف بسيط للترند
        label = "chop"
        score = 0.0

        if fast_above_slow and slope_fast > 0:
            # ترند صاعد
            if dist_rel >= 0.7 and slope_fast > 0:
                label = "strong_bull"
                score = 3.0
            elif dist_rel >= 0.3:
                label = "bull"
                score = 2.3
            else:
                label = "bull"
                score = 1.5
        elif (not fast_above_slow) and slope_fast < 0:
            # ترند هابط
            if dist_rel >= 0.7 and slope_fast < 0:
                label = "strong_bear"
                score = 3.0
            elif dist_rel >= 0.3:
                label = "bear"
                score = 2.3
            else:
                label = "bear"
                score = 1.5
        else:
            label = "chop"
            score = 0.8 if dist_rel >= 0.3 else 0.3

        # Bonus بسيط عند الكروس نفسه
        if cross_up and score < 2.5:
            score += 0.5
        if cross_down and score < 2.5:
            score += 0.5

        return {
            "ok": True,
            "label": label,
            "score": float(score),
            "cross_up": bool(cross_up),
            "cross_down": bool(cross_down),
            "fast_above_slow": bool(fast_above_slow),
            "dist_rel": float(dist_rel),
            "slope_fast": float(slope_fast),
            "ema_fast": fast_now,
            "ema_slow": slow_now,
        }
    except Exception as e:
        log_w(f"compute_ema_engine error: {e}")
        return {
            "ok": False,
            "label": "error",
            "score": 0.0,
            "cross_up": False,
            "cross_down": False,
            "fast_above_slow": False,
            "dist_rel": 0.0,
            "slope_fast": 0.0,
        }

# =================== CHART PRIME HIGH VOLUME BOXES ===================
def compute_delta_volume(df, period=2):
    """
    حساب حجم الدلتا: الفرق بين حجم الشراء وحجم البيع.
    """
    close = df['close'].astype(float)
    volume = df['volume'].astype(float)
    price_change = close.diff()
    buy_volume = volume.where(price_change > 0, 0)
    sell_volume = volume.where(price_change < 0, 0)
    delta_volume = buy_volume - sell_volume
    return delta_volume

def _pivot_low(series, left, right):
    """
    تحديد قيعان محورية.
    """
    lows = []
    for i in range(left, len(series) - right):
        if series[i] == min(series[i-left:i+right+1]):
            lows.append(i)
    return lows

def _pivot_high(series, left, right):
    """
    تحديد قمم محورية.
    """
    highs = []
    for i in range(left, len(series) - right):
        if series[i] == max(series[i-left:i+right+1]):
            highs.append(i)
    return highs

def detect_sr_high_volume_boxes(df, lookback_period=20, vol_len=2, box_width_mult=1.0, atr_len=200):
    """
    كشف صناديق الدعم/المقاومة عالية الحجم (ChartPrime).
    """
    if len(df) < lookback_period:
        return {
            "support_level": None,
            "support_level_1": None,
            "resistance_level": None,
            "resistance_level_1": None,
            "side": None,
            "box_score": 0.0
        }

    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].astype(float)

    # حساب ATR
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    atr = tr.rolling(atr_len).mean().iloc[-1]

    # حساب حجم الدلتا
    delta_vol = compute_delta_volume(df, vol_len)

    # تحديد النقاط المحورية للقيعان والقمم
    pivot_lows = _pivot_low(low, 3, 3)
    pivot_highs = _pivot_high(high, 3, 3)

    # جمع أحجام الدلتا عند النقاط المحورية
    low_volumes = []
    for idx in pivot_lows:
        if idx < len(delta_vol):
            low_volumes.append((low.iloc[idx], delta_vol.iloc[idx]))

    high_volumes = []
    for idx in pivot_highs:
        if idx < len(delta_vol):
            high_volumes.append((high.iloc[idx], delta_vol.iloc[idx]))

    # تحديد مستويات الدعم (القيعان ذات الدلتا الموجبة الكبيرة)
    support_levels = []
    for price, vol in low_volumes:
        if vol > 0:
            support_levels.append((price, vol))

    # تحديد مستويات المقاومة (القمم ذات الدلتا السالبة الكبيرة)
    resistance_levels = []
    for price, vol in high_volumes:
        if vol < 0:
            resistance_levels.append((price, vol))

    # ترتيب المستويات حسب الحجم
    support_levels.sort(key=lambda x: x[1], reverse=True)
    resistance_levels.sort(key=lambda x: x[1], reverse=False)  # لأن الدلتا سالبة، الأكثر سلبية أولاً

    # أخذ أقوى مستويين للدعم والمقاومة
    support1 = support_levels[0][0] if support_levels else None
    support2 = support_levels[1][0] if len(support_levels) > 1 else None
    resistance1 = resistance_levels[0][0] if resistance_levels else None
    resistance2 = resistance_levels[1][0] if len(resistance_levels) > 1 else None

    # تحديد الجانب الحالي (هل السعر قريب من دعم أم مقاومة؟)
    current_price = close.iloc[-1]
    side = None
    box_score = 0.0

    if support1 and resistance1:
        # حساب المسافة النسبية
        dist_to_support = abs(current_price - support1) / current_price
        dist_to_resistance = abs(current_price - resistance1) / current_price
        if dist_to_support < dist_to_resistance:
            side = "support"
            # حساب score للدعم: يعتمد على قوة الحجم والقرب
            vol_score = min(10, abs(support_levels[0][1]) / (volume.mean() + 1e-9) * 2)
            proximity_score = max(0, 10 - (dist_to_support * 1000))
            box_score = (vol_score + proximity_score) / 2
        else:
            side = "resistance"
            vol_score = min(10, abs(resistance_levels[0][1]) / (volume.mean() + 1e-9) * 2)
            proximity_score = max(0, 10 - (dist_to_resistance * 1000))
            box_score = (vol_score + proximity_score) / 2

    return {
        "support_level": support1,
        "support_level_1": support2,
        "resistance_level": resistance1,
        "resistance_level_1": resistance2,
        "side": side,
        "box_score": box_score
    }

# =================== ADX/ATR + BOXES LIQUIDITY MONITOR ===================
def build_liquidity_cycle_context(
    df: pd.DataFrame,
    adx_cycle: dict,
    art_cycle: dict,
    cp_boxes: dict,
    flow: dict,
) -> dict:
    """
    مراقبة حالة السيولة/التجميع من:
      - ADX cycle (trend/flat/exhaustion)
      - ART/ATR cycle (accumulation/impulse/correction/distribution)
      - High-Volume Boxes (support/resistance + score)
      - Flow (delta_z + cvd_trend)
    """
    close = df["close"].astype(float)
    price_now = float(close.iloc[-1])

    trend_side = adx_cycle.get("trend_side", "chop")
    phase      = adx_cycle.get("phase", "neutral")
    adx_val    = float(adx_cycle.get("adx", 0.0) or 0.0)
    rising     = bool(adx_cycle.get("rising", False))
    falling    = bool(adx_cycle.get("falling", False))

    art_phase  = art_cycle.get("phase", "unknown")
    atr_rel    = float(art_cycle.get("atr_rel", 1.0) or 1.0)

    cb_side    = cp_boxes.get("side")
    cb_score   = float(cp_boxes.get("box_score", 0.0) or 0.0)
    sup        = cp_boxes.get("support_level")
    res        = cp_boxes.get("resistance_level")

    delta_z   = float(flow.get("delta_z", 0.0) or 0.0)
    cvd_trend = (flow.get("cvd_trend") or "").lower()

    # قرب السعر من الصندوق
    def _distance(level: float | None) -> float:
        if level is None:
            return 1e9
        return abs(price_now - float(level)) / max(price_now, 1e-9)

    dist_sup = _distance(sup)
    dist_res = _distance(res)

    near_support    = sup is not None and dist_sup <= 0.01  # 1% من السعر
    near_resistance = res is not None and dist_res <= 0.01

    # Flow bias بسيط
    if delta_z >= 0.7 and cvd_trend == "up":
        flow_bias = "bullish"
    elif delta_z <= -0.7 and cvd_trend == "down":
        flow_bias = "bearish"
    else:
        flow_bias = "neutral"

    regime = "chop"
    continuation_bias = "none"
    notes = []

    # 1) ترند صاعد قوي + تصحيح عند دعم بصندوق قوي + Flow شراء
    if trend_side == "up" and phase in ("expansion", "trend") and art_phase in ("correction", "accumulation"):
        if near_support and cb_side == "support" and cb_score >= 6.0 and flow_bias in ("bullish", "neutral"):
            regime = "bull_accumulation"
            continuation_bias = "up"
            notes.append("Trend up + correction into strong support box + bullish/neutral flow")

    # 2) ترند هابط قوي + تصحيح عند مقاومة بصندوق قوي + Flow بيع
    if regime == "chop" and trend_side == "down" and phase in ("expansion", "trend") and art_phase in ("correction", "accumulation"):
        if near_resistance and cb_side == "resistance" and cb_score >= 6.0 and flow_bias in ("bearish", "neutral"):
            regime = "bear_accumulation"
            continuation_bias = "down"
            notes.append("Trend down + correction into strong resistance box + bearish/neutral flow")

    # 3) ADX كان عالي وبدأ ينزل عند مقاومة → Distribution / Liquidity Sweep top
    if regime == "chop" and near_resistance and cb_side == "resistance" and cb_score >= 6.5:
        if phase in ("trend", "exhaustion") and falling and adx_val >= 25 and art_phase in ("distribution", "impulse"):
            # Flow عكسي → سحب سيولة
            if flow_bias == "bearish":
                regime = "bear_liquidity_sweep"   # سحب سيولة من المشترين
                continuation_bias = "down"
                notes.append("ADX roll-over at resistance with bearish flow → liquidity sweep top")
            else:
                regime = "distribution_top"
                continuation_bias = "none"
                notes.append("ADX falling at resistance → potential distribution")

    # 4) ADX كان عالي وبدأ ينزل عند دعم → accumulation bottom أو sweep bottom
    if regime == "chop" and near_support and cb_side == "support" and cb_score >= 6.5:
        if phase in ("trend", "exhaustion") and falling and adx_val >= 25 and art_phase in ("accumulation", "impulse"):
            if flow_bias == "bullish":
                regime = "bull_accumulation"
                continuation_bias = "up"
                notes.append("ADX roll-over at support with bullish flow → accumulation bottom")
            else:
                regime = "bull_liquidity_sweep"  # شمعات حمراء قوية لكن بدون Flow شراء واضح
                continuation_bias = "up"
                notes.append("ADX falling at support with mixed flow → possible liquidity sweep bottom")

    # 5) لو لسه ما حدّدناش,
    if regime == "chop":
        if trend_side == "up" and adx_val >= 18 and atr_rel >= 1.0:
            regime = "trend_up"
            continuation_bias = "up"
            notes.append("Generic uptrend regime")
        elif trend_side == "down" and adx_val >= 18 and atr_rel >= 1.0:
            regime = "trend_down"
            continuation_bias = "down"
            notes.append("Generic downtrend regime")
        else:
            regime = "chop"
            continuation_bias = "none"
            notes.append("Choppy / low conviction regime")

    return {
        "regime": regime,
        "continuation_bias": continuation_bias,
        "trend_side": trend_side,
        "adx": adx_val,
        "phase": phase,
        "art_phase": art_phase,
        "atr_rel": atr_rel,
        "near_support": near_support,
        "near_resistance": near_resistance,
        "cp_box_side": cb_side,
        "cp_box_score": cb_score,
        "flow_bias": flow_bias,
        "delta_z": delta_z,
        "notes": notes,
    }

# =================== PROFESSIONAL ENTRY ENGINE HELPERS ===================
def classify_liquidity_regime_from_footprint(footprint: dict) -> str:
    """
    تحويل footprint إلى تصنيف سيولة بسيط:
      buy_absorption / sell_absorption / distribution / unknown
    """
    if not isinstance(footprint, dict) or not footprint.get("ok"):
        return "unknown"
    if footprint.get("absorption_bull"):
        return "buy_absorption"
    if footprint.get("absorption_bear"):
        return "sell_absorption"
    return "distribution"


def compute_zone_grade_from_gz(gz: dict) -> str:
    """
    تقييم قوة المنطقة الذهبية بناءً على score:
      strong / mid / weak / none
    """
    if not gz or not isinstance(gz, dict) or not gz.get("zone"):
        return "none"
    score = float(gz.get("score", 0.0))
    if score >= 7.5:
        return "strong"
    if score >= 5.0:
        return "mid"
    return "weak"


def detect_bullish_rejection_candle(df: pd.DataFrame) -> dict:
    """
    Hammer / Bullish Engulfing كإشارة BUY محترمة
    """
    if len(df) < 3:
        return {"match": False, "pattern": None}

    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    o0, h0, l0, c0 = o.iloc[-1], h.iloc[-1], l.iloc[-1], c.iloc[-1]
    o1, h1, l1, c1 = o.iloc[-2], h.iloc[-2], l.iloc[-2], c.iloc[-2]

    body0 = abs(c0 - o0)
    rng0  = max(h0 - l0, 1e-9)
    lower_wick0 = min(o0, c0) - l0
    upper_wick0 = h0 - max(o0, c0)

    hammer_cond = (
        c0 > o0 and
        lower_wick0 >= body0 * 2.0 and
        upper_wick0 <= body0 * 0.5 and
        rng0 > 0
    )
    if hammer_cond:
        return {"match": True, "pattern": "hammer"}

    prev_bear = c1 < o1
    curr_bull = c0 > o0
    engulf_cond = (
        prev_bear and curr_bull and
        o0 <= c1 and
        c0 >= o1
    )
    if engulf_cond:
        return {"match": True, "pattern": "bullish_engulfing"}

    return {"match": False, "pattern": None}


def detect_bearish_rejection_candle(df: pd.DataFrame) -> dict:
    """
    Shooting Star / Bearish Engulfing كإشارة SELL محترمة
    """
    if len(df) < 3:
        return {"match": False, "pattern": None}

    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    o0, h0, l0, c0 = o.iloc[-1], h.iloc[-1], l.iloc[-1], c.iloc[-1]
    o1, h1, l1, c1 = o.iloc[-2], h.iloc[-2], l.iloc[-2], c.iloc[-2]

    body0 = abs(c0 - o0)
    rng0  = max(h0 - l0, 1e-9)
    upper_wick0 = h0 - max(o0, c0)
    lower_wick0 = min(o0, c0) - l0

    star_cond = (
        c0 < o0 and
        upper_wick0 >= body0 * 2.0 and
        lower_wick0 <= body0 * 0.5 and
        rng0 > 0
    )
    if star_cond:
        return {"match": True, "pattern": "shooting_star"}

    prev_bull = c1 > o1
    curr_bear = c0 < o0
    engulf_cond = (
        prev_bull and curr_bear and
        o0 >= c1 and
        c0 <= o1
    )
    if engulf_cond:
        return {"match": True, "pattern": "bearish_engulfing"}

    return {"match": False, "pattern": None}


def professional_entry_engine(df: pd.DataFrame, side: str, council_data: dict):
    """
    محرك دخول احترافي مع فلاتر السيولة والتراكم:
      - side: "buy"/"sell"
      - يستخدم: Golden Zone + Footprint + ADX/ART Cycle + EMA Cross + ChartPrime Boxes + Liquidity Cycle
      - يرجع (ok, info_dict)
    """
    side = (side or "").lower()
    if side not in ("buy", "sell"):
        return False, {"reason": "no_side"}

    gz         = council_data.get("gz")
    footprint  = council_data.get("footprint", {}) or {}
    ema_ctx    = council_data.get("ema", {}) or {}
    adx_cycle  = council_data.get("adx_cycle") or analyze_adx_cycle(compute_adx_di_context(df, ADX_LEN))
    art_cycle  = council_data.get("art_cycle") or analyze_art_cycle(df, ATR_LEN, 20)
    cp_boxes   = council_data.get("cp_boxes", {}) or {}
    liq_cycle  = council_data.get("liquidity_cycle", {}) or {}

    zone_grade = compute_zone_grade_from_gz(gz)
    liq_regime = liq_cycle.get("regime", "chop")
    liq_bias   = liq_cycle.get("continuation_bias", "none")
    liq_regime_fp = classify_liquidity_regime_from_footprint(footprint)

    # لو مفيش EMA Engine حالياً، نعتبر الكروس قوي افتراضيًا (ما نبلوكش صفقات كويسة)
    if ema_ctx:
        cross_strength = float(ema_ctx.get("score", 0.0))
    else:
        cross_strength = 3.0

    phase      = adx_cycle["phase"]
    trend_side = adx_cycle["trend_side"]
    art_phase  = art_cycle["phase"]

    # ========== 1) فلتر السيولة / التجميع ==========
    # 1.1 منع شراء في قمم سحب سيولة/توزيع
    if side == "buy" and liq_regime in ("bear_liquidity_sweep", "distribution_top"):
        return False, {
            "reason": f"liquidity_regime_block_buy({liq_regime})",
            "zone_grade": zone_grade,
            "ema_score": cross_strength,
            "flow_bias": liq_cycle.get("flow_bias", "neutral"),
            "adx": adx_cycle.get("adx", 0),
        }

    # 1.2 منع بيع في قيعان تجميع/سحب سيولة لصعود
    if side == "sell" and liq_regime in ("bull_accumulation", "bull_liquidity_sweep"):
        return False, {
            "reason": f"liquidity_regime_block_sell({liq_regime})",
            "zone_grade": zone_grade,
            "ema_score": cross_strength,
            "flow_bias": liq_cycle.get("flow_bias", "neutral"),
            "adx": adx_cycle.get("adx", 0),
        }

    # 2) Zone Strength: لو في Golden Zone ضعيفة → بلوك
    if gz and isinstance(gz, dict) and gz.get("ok"):
        if zone_grade == "weak":
            return False, {"reason": "weak_golden_zone", "zone_grade": zone_grade}
    # لو مفيش GZ، ما نبلوكش، لكن هنشد في باقي الشروط

    # 3) Liquidity Absorption
    if side == "buy" and liq_regime_fp == "sell_absorption":
        return False, {"reason": "liquidity_sell_absorption_above", "liq": liq_regime_fp}
    if side == "sell" and liq_regime_fp == "buy_absorption":
        return False, {"reason": "liquidity_buy_absorption_below", "liq": liq_regime_fp}

    # 4) ADX Cycle: منع الدخول في سوق نايم/تجميعي لو مفيش قاع/قمة ذهبية
    if phase in ("flat", "accumulation") and not (gz and isinstance(gz, dict) and gz.get("ok")):
        return False, {"reason": f"adx_phase_{phase}_no_gz"}

    # 5) منع الدخول ضد ترند قوي إلا لو عندنا Golden Top/Bottom
    gz_type = None
    if gz and isinstance(gz, dict):
        zone = gz.get("zone")
        if zone and isinstance(zone, dict):
            gz_type = zone.get("type")
    
    if side == "buy" and trend_side == "down" and gz_type != "golden_bottom":
        return False, {"reason": "counter_trend_buy_without_golden_bottom", "gz_type": gz_type}
    if side == "sell" and trend_side == "up" and gz_type != "golden_top":
        return False, {"reason": "counter_trend_sell_without_golden_top", "gz_type": gz_type}

    # 6) Cross Strength من EMA Engine (لو موجود)
    if gz and isinstance(gz, dict) and gz.get("ok"):
        if zone_grade == "strong" and cross_strength < 2.0:
            return False, {"reason": f"cross_weak_for_strong_zone({cross_strength:.1f})"}
        if zone_grade == "mid" and cross_strength < 2.5:
            return False, {"reason": f"cross_weak_for_mid_zone({cross_strength:.1f})"}
    else:
        # بدون Zone، عايزين cross أقوى
        if cross_strength < 2.5:
            return False, {"reason": f"cross_weak_no_zone({cross_strength:.1f})"}

    # 7) شمعة رفض/ابتلاع في نفس الاتجاه
    if side == "buy":
        candle = detect_bullish_rejection_candle(df)
    else:
        candle = detect_bearish_rejection_candle(df)

    if not candle.get("match"):
        # استثناء: لو Golden Zone مؤكدة + Cross قوي
        strong_gz = gz and isinstance(gz, dict) and gz.get("ok") and gz.get("confirmed") and zone_grade in ("strong", "mid") and cross_strength >= 2.5
        if not strong_gz:
            return False, {"reason": "no_rejection_candle", "zone_grade": zone_grade}

    # 8) ART Phase: لو Accumulation ومفيش Zone → تجنّب
    if art_phase == "accumulation" and not (gz and isinstance(gz, dict) and gz.get("ok")):
        return False, {"reason": "art_accumulation_no_zone"}

    info = {
        "zone_grade": zone_grade,
        "liquidity": liq_regime_fp,
        "adx_phase": phase,
        "adx_trend_side": trend_side,
        "art_phase": art_phase,
        "cross_strength": cross_strength,
        "candle_pattern": candle.get("pattern"),
        "gz_type": gz_type,
        "liquidity_regime": liq_regime,
        "liquidity_bias": liq_bias,
        "cp_box_score": cp_boxes.get("box_score", 0),
        "cp_box_side": cp_boxes.get("side"),
    }
    return True, info

# =================== ENHANCED COUNCIL VOTING ===================
def council_votes_pro_enhanced(df):
    """مجلس تصويت محسّن مع Footprint + SMC + Golden Zone Pro + VWAP + OTC + EMA + ChartPrime"""
    try:
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
        
        # OTC Hidden Flow Detection
        otc = detect_otc_flows(df)
        
        # ADX / ART CYCLES (TradingView-style Context)
        adx_ctx   = compute_adx_di_context(df, ADX_LEN)
        adx_cycle = analyze_adx_cycle(adx_ctx)
        art_cycle = analyze_art_cycle(df, ATR_LEN, 20)
        
        # EMA Engine Context
        ema_ctx = compute_ema_engine(df, EMA_FAST_LEN, EMA_SLOW_LEN)

        # ChartPrime High-Volume Boxes
        cp_boxes = detect_sr_high_volume_boxes(
            df,
            lookback_period=20,
            vol_len=2,
            box_width_mult=1.0,
            atr_len=200,
        )

        # Liquidity Cycle Context
        flow = compute_flow_metrics(df)
        liquidity_cycle = build_liquidity_cycle_context(df, adx_cycle, art_cycle, cp_boxes, flow)

        votes_b = 0; votes_s = 0
        score_b = 0.0; score_s = 0.0
        logs = []

        adx = ind.get('adx', 0)
        plus_di = ind.get('plus_di', 0)
        minus_di = ind.get('minus_di', 0)
        di_spread = ind.get('di_spread', abs(plus_di - minus_di))

        # ==== EMA Engine تأثير على التصويت =====
        if ema_ctx.get("ok"):
            ema_label = ema_ctx.get("label")
            ema_score = float(ema_ctx.get("score", 0.0))

            if ema_label == "strong_bull":
                votes_b += 2
                score_b += 1.5
                logs.append(f"📈 EMA STRONG BULL (score={ema_score:.1f})")
            elif ema_label == "bull":
                votes_b += 1
                score_b += 0.8
                logs.append(f"📈 EMA BULL (score={ema_score:.1f})")
            elif ema_label == "strong_bear":
                votes_s += 2
                score_s += 1.5
                logs.append(f"📉 EMA STRONG BEAR (score={ema_score:.1f})")
            elif ema_label == "bear":
                votes_s += 1
                score_s += 0.8
                logs.append(f"📉 EMA BEAR (score={ema_score:.1f})")
            else:
                # chop ⇒ تخفيف بسيط للثقة العامة
                score_b *= 0.9
                score_s *= 0.9
                logs.append(f"🔁 EMA CHOP MODE (score={ema_score:.1f})")

        # ==== VWAP CONTEXT ====
        vwap = ind.get("vwap")
        vwap_diff_bps = None
        near_vwap = False
        far_from_vwap = False
        if VWAP_ENABLED and vwap:
            vwap_diff_bps = abs(current_price - vwap) / vwap * 10000.0
            near_vwap = vwap_diff_bps <= VWAP_SCALP_BAND_BPS
            far_from_vwap = vwap_diff_bps >= VWAP_TREND_BAND_BPS
            above_vwap = current_price > vwap
            logs.append(f"VWAP ctx: px={current_price:.6f} vwap={vwap:.6f} Δ={vwap_diff_bps:.1f}bps")

        # --- ChartPrime Boxes Boost ---
        near_support = liquidity_cycle.get("near_support", False)
        near_resistance = liquidity_cycle.get("near_resistance", False)
        cb_side = cp_boxes.get("side")
        
        if cb_side == "support" and cp_boxes.get("box_score", 0) >= 7.0:
            if near_support and current_price <= cp_boxes.get("support_level"):
                votes_b += 2
                score_b += 2.0
                logs.append(f"📦 STRONG SUPPORT BOX (score={cp_boxes['box_score']:.1f})")
        
        if cb_side == "resistance" and cp_boxes.get("box_score", 0) >= 7.0:
            if near_resistance and current_price >= cp_boxes.get("resistance_level"):
                votes_s += 2
                score_s += 2.0
                logs.append(f"📦 STRONG RESISTANCE BOX (score={cp_boxes['box_score']:.1f})")

        # --- Liquidity Cycle Boost ---
        if liquidity_cycle.get("regime") == "bull_accumulation":
            votes_b += 3
            score_b += 2.5
            logs.append(f"💰 BULL ACCUMULATION REGIME | {liquidity_cycle.get('notes', [''])[0]}")
        
        if liquidity_cycle.get("regime") == "bear_accumulation":
            votes_s += 3
            score_s += 2.5
            logs.append(f"💰 BEAR ACCUMULATION REGIME | {liquidity_cycle.get('notes', [''])[0]}")

        # --- تصويت VWAP للسكالب (قرب من VWAP) ---
        if VWAP_ENABLED and near_vwap and cd:
            if cd.get("buy"):
                votes_b += 2; score_b += 1.5
                logs.append("⚡ VWAP SCALP BUY zone")
            if cd.get("sell"):
                votes_s += 2; score_s += 1.5
                logs.append("⚡ VWAP SCALP SELL zone")

        # --- بوست للترند بعيد عن VWAP ---
        if VWAP_ENABLED and far_from_vwap and adx >= ADX_TREND_MIN:
            if plus_di > minus_di and current_price > (vwap or current_price):
                votes_b += 1; score_b += 1.0
                logs.append("📈 VWAP TREND BOOST BUY")
            elif minus_di > plus_di and current_price < (vwap or current_price):
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

        # --- OTC Hidden Flow Detection ---
        if otc.get("otc_buy"):
            # سيولة شراء مخفية تدعم الشراء
            boost = min(3.0, otc.get("strength", 1.5))
            votes_b += 2
            score_b += boost
            logs.append(
                f"💰 OTC BUY ({otc.get('reason','')}) "
                f"move={otc.get('move_bps',0):.1f}bps flow={otc.get('visible_flow_ratio',0)*100:.1f}% s={boost:.1f}"
            )
        elif otc.get("otc_sell"):
            # سيولة بيع مخفية تدعم البيع
            boost = min(3.0, otc.get("strength", 1.5))
            votes_s += 2
            score_s += boost
            logs.append(
                f"💰 OTC SELL ({otc.get('reason','')}) "
                f"move={otc.get('move_bps',0):.1f}bps flow={otc.get('visible_flow_ratio',0)*100:.1f}% s={boost:.1f}"
            )
        else:
            otc = {"otc_buy": False, "otc_sell": False, "strength": 0.0}

        # --- Liquidity Trap Awareness
        if liquidity_traps.get('ok') and liquidity_traps.get('traps'):
            for trap in liquidity_traps['traps']:
                if trap['type'] == 'stop_hunt_bull' and score_b > score_s:
                    score_b *= 1.1  # تعزيز الثقة في فخ الصعود
                    logs.append(f"🪤 فخ سيولة صاعد قريب ({trap['distance_pct']:.2f}%)")
                elif trap['type'] == 'stop_hunt_bear' and score_s > score_b:
                    score_s *= 1.1  # تعزيز الثقة في فخ الهبوط
                    logs.append(f"🪤 فخ سيولة هابط قريب ({trap['distance_pct']:.2f}%)")
        
        # ADX/ART Cycle Context — Boost/De-boost
        phase        = adx_cycle["phase"]
        trend_side   = adx_cycle["trend_side"]
        strong_trend = adx_cycle["strong_trend"]
        art_phase    = art_cycle["phase"]

        # ترند صاعد قوي + ART اندفاع/تصحيح ⇒ Boost BUY
        if trend_side == "up" and strong_trend and art_phase in ("impulse", "correction"):
            votes_b += 2
            score_b += 1.5
            logs.append(f"🚀 ADX/ART ترند صاعد قوي (phase={phase}, art={art_phase}, adx={adx_cycle['adx']:.1f})")

        # ترند هابط قوي + ART اندفاع/تصحيح ⇒ Boost SELL
        if trend_side == "down" and strong_trend and art_phase in ("impulse", "correction"):
            votes_s += 2
            score_s += 1.5
            logs.append(f"💥 ADX/ART ترند هابط قوي (phase={phase}, art={art_phase}, adx={adx_cycle['adx']:.1f})")

        # سوق تجميعي ضعيف ⇒ تقليل شهية الدخول
        if phase in ("flat", "accumulation") and art_phase == "accumulation":
            score_b *= 0.7
            score_s *= 0.7
            logs.append("🔄 ADX/ART تجميع وضعف حركة — تخفيض ثقة")

        # Exhaustion / Distribution ⇒ خفض إضافي
        if phase == "exhaustion" or art_phase == "distribution":
            score_b *= 0.85
            score_s *= 0.85
            logs.append(f"⚠️ ADX Exhaustion / ART Distribution (phase={phase}, art={art_phase}) — تخفيف المخاطرة")

        # Cross في DI كإشارة ميل مبكر
        if adx_cycle["di_cross_up"]:
            votes_b += 1
            score_b += 0.5
            logs.append("📈 DI+ Cross UP → ميل صاعد محتمل")
        if adx_cycle["di_cross_down"]:
            votes_s += 1
            score_s += 0.5
            logs.append("📉 DI- Cross DOWN → ميل هابط محتمل")

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
            "vwap": vwap,
            "adx_cycle_phase": adx_cycle["phase"],
            "adx_trend_side": adx_cycle["trend_side"],
            "art_phase": art_cycle["phase"],
            "art_atr_rel": art_cycle["atr_rel"],
            "art_vol_rel": art_cycle["vol_rel"],
            "ema_label": ema_ctx.get("label"),
            "ema_score": ema_ctx.get("score"),
            "ema_fast": ema_ctx.get("ema_fast"),
            "ema_slow": ema_ctx.get("ema_slow"),
            "cp_support": cp_boxes["support_level"],
            "cp_support_1": cp_boxes["support_level_1"],
            "cp_resistance": cp_boxes["resistance_level"],
            "cp_resistance_1": cp_boxes["resistance_level_1"],
            "cp_box_side": cp_boxes["side"],
            "cp_box_score": cp_boxes["box_score"],
            "liquidity_cycle": liquidity_cycle,
        })

        return {
            "b": votes_b, "s": votes_s,
            "score_b": score_b, "score_s": score_s,
            "logs": logs, "ind": ind, "gz": gz, 
            "footprint": footprint, "candles": cd,
            "liquidity_traps": liquidity_traps,
            "otc": otc,
            "adx_cycle": adx_cycle,
            "art_cycle": art_cycle,
            "ema": ema_ctx,
            "cp_boxes": cp_boxes,
            "liquidity_cycle": liquidity_cycle,
        }
    except Exception as e:
        log_w(f"council_votes_pro_enhanced error: {e}")
        return {
            "b": 0, "s": 0,
            "score_b": 0.0, "score_s": 0.0,
            "logs": [], "ind": {}, "gz": None,
            "candles": {}, "footprint": {}, "liquidity_traps": {}, "otc": {}, 
            "adx_cycle": {}, "art_cycle": {}, "ema": {},
            "cp_boxes": {}, "liquidity_cycle": {}
        }

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

# ================== RISK HELPERS ==================

def compute_unrealized_pnl_pct(state, current_price: float) -> float:
    """
    حساب % PnL غير المحقَّق للصفقة الحالية بناءً على الـ STATE.
    """
    if not state.get("open"):
        return 0.0

    side = state.get("side")
    entry = float(state.get("entry") or 0.0)
    if not entry:
        return 0.0

    if side == "long":
        return (current_price / entry) - 1.0
    elif side == "short":
        return (entry / current_price) - 1.0
    return 0.0


def risk_on_new_trade(state, side: str, entry_price: float):
    """
    يُستدعى مباشرة بعد فتح صفقة جديدة:
    - يحسب max_loss_price كـ % ثابت من الـ entry
    - يخفّض وضع Post Big Win لو انتهى
    """
    state["open"] = True
    state["side"] = side
    state["entry"] = float(entry_price)

    # ستوب على أساس % من سعر الدخول
    if side == "long":
        state["max_loss_price"] = entry_price * (1.0 + MAX_LOSS_PCT)
        log_i(f"🛡️ Hard Stop Loss SET for LONG: {state['max_loss_price']:.6f} (-{abs(MAX_LOSS_PCT)*100}%)")
    else:  # short
        state["max_loss_price"] = entry_price * (1.0 - MAX_LOSS_PCT)
        log_i(f"🛡️ Hard Stop Loss SET for SHORT: {state['max_loss_price']:.6f} (-{abs(MAX_LOSS_PCT)*100}%)")

    # لما نفتح صفقة جديدة، نكمّل عدّاد post_big_win_mode لو لسه شغّال
    if state.get("post_big_win_mode") and state.get("post_big_win_bars_left", 0) <= 0:
        state["post_big_win_mode"] = False
        state["post_big_win_bars_left"] = 0


def risk_on_trade_closed(state, realized_pnl_pct: float):
    """
    يُستدعى بعد إغلاق الصفقة بالكامل:
    - يخزن آخر % PnL
    - يفعّل وضع Post Big Win لو الربح كبير
    """
    state["last_closed_pnl_pct"] = float(realized_pnl_pct or 0.0)
    state["open"] = False
    state["side"] = None
    state["entry"] = None
    state["max_loss_price"] = None

    if realized_pnl_pct >= BIG_WIN_PCT:
        state["post_big_win_mode"] = True
        state["post_big_win_bars_left"] = POST_BIG_WIN_BARS
        log_g(f"🎉 BIG WIN DETECTED! {realized_pnl_pct*100:.2f}% ≥ {BIG_WIN_PCT*100}% - Post-Big-Win Mode ACTIVATED for {POST_BIG_WIN_BARS} bars")
    else:
        # مفيش Big Win → نطفي المود ده
        state["post_big_win_mode"] = False
        state["post_big_win_bars_left"] = 0


def tick_post_big_win_decay(state):
    """
    تُستدعى مع كل شمعة/لوب:
    تقلل عدّاد Post Big Win لو شغّال.
    """
    if state.get("post_big_win_mode"):
        left = int(state.get("post_big_win_bars_left", 0))
        if left > 0:
            state["post_big_win_bars_left"] = left - 1
            if state["post_big_win_bars_left"] <= 0:
                state["post_big_win_mode"] = False
                state["post_big_win_bars_left"] = 0
                log_i("🛡️ Post-Big-Win Mode EXPIRED - Returning to normal trading")

# =================== AUTO RECOVERY & SYNC ===================
def sync_open_position_from_exchange(exchange, symbol: str, state):
    """
    محاولة مزامنة أي صفقة مفتوحة على المنصة مع STATE بعد الريستارت.
    """
    try:
        positions = exchange.fetch_positions([symbol])
    except Exception as e:
        log_w(f"sync_open_position error: {e}")
        return state

    live_pos = None
    for p in positions:
        sym = p.get("symbol") or p.get("info", {}).get("symbol") or ""
        if symbol.replace(":", "") in sym.replace(":", ""):
            qty = abs(float(p.get("contracts") or p.get("positionAmt") or p.get("info", {}).get("size", 0) or 0))
            if qty > 0:
                live_pos = p
                break

    if not live_pos:
        # مفيش صفقة حية على البورصة → صفّر الحالة لو عندك in_position
        if state.get("open"):
            log_w("🔄 No live position on exchange, resetting local STATE.")
            state["open"] = False
            state["side"] = None
            state["entry"] = None
            state["max_loss_price"] = None
            state["post_big_win_mode"] = False
            state["post_big_win_bars_left"] = 0
        return state

    # استخراج بيانات الصفقة الحية
    side_raw = (live_pos.get("side") or live_pos.get("positionSide") or "").lower()
    side = "long" if "long" in side_raw or float(live_pos.get("cost", 0)) > 0 else "short"
    entry_price = float(live_pos.get("entryPrice") or live_pos.get("info", {}).get("avgEntryPrice") or 0.0)
    qty = abs(float(live_pos.get("contracts") or live_pos.get("positionAmt") or 0))
    
    if qty <= 0:
        return state

    log_i(f"🔄 Syncing live position from exchange: {side.upper()} {qty:.4f} @ {entry_price:.6f}")
    
    # تحديث STATE بالصفقة الحية
    state.update({
        "open": True,
        "side": side,
        "entry": entry_price,
        "qty": qty,
        "in_position": True,
        "max_loss_price": entry_price * (1.0 + MAX_LOSS_PCT) if side == "long" else entry_price * (1.0 - MAX_LOSS_PCT),
    })
    
    # تحديث Post-Big-Win Mode من آخر صفقة مغلقة
    last_closed_pnl = state.get("last_closed_pnl_pct", 0.0)
    if last_closed_pnl >= BIG_WIN_PCT:
        state["post_big_win_mode"] = True
        state["post_big_win_bars_left"] = POST_BIG_WIN_BARS
        log_i(f"🔄 Post-Big-Win Mode RESTORED from last trade: {last_closed_pnl*100:.2f}%")
    
    return state


def enhanced_resume_open_position(exchange, symbol: str, state: dict) -> dict:
    """
    دالة استئناف محسنة مع Auto-Recovery الكامل
    """
    if not RESUME_ON_RESTART:
        log_i("🔄 Resume disabled")
        return state

    log_i("🔄 Starting enhanced position recovery...")
    
    # 1. أولاً: Sync مع المنصة
    state = sync_open_position_from_exchange(exchange, symbol, state)
    
    # 2. ثانياً: جلب البيانات الحية (فال باك أب)
    live = fetch_live_position(exchange, symbol)
    
    # 3. الدمج الذكي بين STATE والبيانات الحية
    if live.get("ok") and state.get("open"):
        # تأكيد البيانات من المصدرين
        ts = int(time.time())
        prev = load_state()
        
        if prev.get("ts") and (ts - int(prev["ts"])) > RESUME_LOOKBACK_SECS:
            log_w("🔄 Found old local state — overriding with exchange live snapshot")
        
        # تحديث STATE بالبيانات الأحدث
        state.update({
            "in_position": True,
            "side": live["side"],
            "entry_price": live["entry"],
            "position_qty": live["qty"],
            "leverage": live.get("leverage") or state.get("leverage") or LEVERAGE,
            "partial_taken": prev.get("partial_taken", False),
            "breakeven_armed": prev.get("breakeven_armed", False),
            "trail_active": prev.get("trail_active", False),
            "trail_tightened": prev.get("trail_tightened", False),
            "mode": prev.get("mode", "trend"),
            "gz_snapshot": prev.get("gz_snapshot", {}),
            "cv_snapshot": prev.get("cv_snapshot", {}),
            "footprint_snapshot": prev.get("footprint_snapshot", {}),
            "ema_snapshot": prev.get("ema_snapshot", {}),
            "opened_at": prev.get("opened_at", ts),
            
            # تحديث Stop Loss
            "max_loss_price": live["entry"] * (1.0 + MAX_LOSS_PCT) if live["side"] == "LONG" else live["entry"] * (1.0 - MAX_LOSS_PCT),
        })
        
        # حفظ الحالة المحدثة
        save_state(state)
        log_g(f"🔄 ENHANCED RESUME: {state['side']} qty={state['position_qty']} @ {state['entry_price']:.6f} "
              f"lev={state['leverage']}x | Stop Loss: {state.get('max_loss_price', 'N/A'):.6f}")
        
        # تسجيل حالة Post-Big-Win
        if state.get("post_big_win_mode"):
            log_i(f"🛡️ Post-Big-Win Mode ACTIVE: {state['post_big_win_bars_left']} bars remaining")
    
    return state

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
        "ema_snapshot": prev.get("ema_snapshot", {}),
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
        
        # ========== إغلاق Hedge وإجبار OneWay ==========
        try:
            ex.set_position_mode(False, SYMBOL)  # force OneWay (NOT hedge)
            POSITION_MODE = "oneway"
            log_g(f"✅ Position mode set to OneWay (Hedge disabled)")
        except Exception as e:
            log_w(f"set_position_mode error: {e} - تأكد من تغيير Position Mode في واجهة BingX")
        # ================================================
        
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
    يطبع Snapshot موحّد: Bookmap + Flow + Council + Strategy + Balance/PnL + VWAP + OTC + EMA + ChartPrime
    """
    try:
        bm = bookmap_snapshot(exchange, symbol)
        flow = compute_flow_metrics(df)
        cv = council_votes_pro_enhanced(df)
        mode = decide_strategy_mode_enhanced(df, 
                                            adx=cv["ind"].get("adx"),
                                            di_plus=cv["ind"].get("plus_di"),
                                            di_minus=cv["ind"].get("minus_di"),
                                            rsi_ctx=rsi_ma_context(df),
                                            footprint=cv.get("footprint", {}),
                                            ema_ctx=cv.get("ema", {}))
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
        strat = f"Strategy: {strat_icon} {mode['mode'].upper()} ({mode['why']})"

        bal_note = f"Balance={bal:.2f}" if bal is not None else ""
        pnl_note = f"CompoundPnL={cpnl:.6f}" if cpnl is not None else ""
        wallet = (" | ".join(x for x in [bal_note, pnl_note] if x)) or ""

        gz_note = ""
        if gz and gz.get("ok"):
            gz_note = f" | 🟡 {gz['zone']['type']} s={gz['score']:.1f}"

        # OTC info
        otc_note = ""
        if cv.get("otc", {}).get("otc_buy"):
            otc_note = f" | 💰 OTC BUY s={cv['otc'].get('strength',0):.1f}"
        elif cv.get("otc", {}).get("otc_sell"):
            otc_note = f" | 💰 OTC SELL s={cv['otc'].get('strength',0):.1f}"
            
        # EMA info
        ema_note = ""
        ema_ctx = cv.get("ema", {})
        if ema_ctx.get("label") != "none":
            ema_label = ema_ctx.get("label", "")
            ema_score = ema_ctx.get("score", 0.0)
            ema_note = f" | 📈 EMA:{ema_label}({ema_score:.1f})"
            
        # ChartPrime info
        cp_note = ""
        cp_boxes = cv.get("cp_boxes", {})
        if cp_boxes.get("side"):
            cp_note = f" | 📦 {cp_boxes['side'].upper()} box({cp_boxes.get('box_score',0):.1f})"
            
        # Liquidity Cycle info
        liq_note = ""
        liq_cycle = cv.get("liquidity_cycle", {})
        if liq_cycle.get("regime") != "chop":
            liq_note = f" | 💧 {liq_cycle['regime'].replace('_', ' ').upper()}"

        if LOG_ADDONS:
            print(f"🧱 {bm_note}", flush=True)
            print(f"📦 {fl_note}", flush=True)
            print(f"📊 {dash}{gz_note}{otc_note}{ema_note}{cp_note}{liq_note}", flush=True)
            print(f"{strat}{(' | ' + wallet) if wallet else ''}", flush=True)
            
            gz_snap_note = ""
            if gz and gz.get("ok"):
                zone_type = gz["zone"]["type"]
                zone_score = gz["score"]
                gz_snap_note = f" | 🟡{zone_type} s={zone_score:.1f}"
            
            flow_z = flow['delta_z'] if flow and flow.get('ok') else 0.0
            bm_imb = bm['imbalance'] if bm and bm.get('ok') else 1.0
            
            # إضافة معلومات VWAP للسنابشوت
            vwap_info = ""
            if VWAP_ENABLED and cv['ind'].get('vwap'):
                vwap_val = cv['ind']['vwap']
                current_price = float(df['close'].iloc[-1])
                vwap_diff_bps = abs(current_price - vwap_val) / vwap_val * 10000.0
                vwap_status = "NEAR" if vwap_diff_bps <= VWAP_SCALP_BAND_BPS else "FAR" if vwap_diff_bps >= VWAP_TREND_BAND_BPS else "MID"
                vwap_info = f" | VWAP:{vwap_status}({vwap_diff_bps:.1f}bps)"
            
            # OTC info for snapshot
            otc_snap = ""
            if cv.get("otc", {}).get("otc_buy"):
                otc_snap = f" | 💰OTC-BUY({cv['otc'].get('strength',0):.1f})"
            elif cv.get("otc", {}).get("otc_sell"):
                otc_snap = f" | 💰OTC-SELL({cv['otc'].get('strength',0):.1f})"
            
            # EMA info for snapshot
            ema_snap = ""
            if ema_ctx.get("label") != "none":
                ema_snap = f" | 📈EMA:{ema_ctx.get('label','')}({ema_ctx.get('score',0):.1f})"
                
            # ChartPrime snapshot
            cp_snap = ""
            if cp_boxes.get("side"):
                cp_snap = f" | 📦{cp_boxes['side'][:3].upper()}({cp_boxes.get('box_score',0):.1f})"
                
            # Liquidity snapshot
            liq_snap = ""
            if liq_cycle.get("regime") != "chop":
                liq_snap = f" | 💧{liq_cycle['regime'][:8]}"
            
            print(f"🧠 SNAP | {side_hint} | votes={cv['b']}/{cv['s']} score={cv['score_b']:.1f}/{cv['score_s']:.1f} "
                  f"| ADX={cv['ind'].get('adx',0):.1f} DI={cv['ind'].get('di_spread',0):.1f} | "
                  f"z={flow_z:.2f} | imb={bm_imb:.2f}{gz_snap_note}{vwap_info}{otc_snap}{ema_snap}{cp_snap}{liq_snap}", 
                  flush=True)
            
            # إضافة معلومات Footprint وSMC
            if cv.get('footprint', {}).get('ok'):
                fp = cv['footprint']
                print(f"👣 FOOTPRINT | Delta={fp['delta']:.0f} | CVD={fp['cumulative_delta']:.0f} | "
                      f"Spike={fp['volume_spike']} | AbsBull={fp['absorption_bull']} | AbsBear={fp['absorption_bear']}", flush=True)
            
            if cv.get('candles', {}).get('smc_pattern'):
                print(f"🕯️ SMC | {cv['candles']['smc_pattern']} | Trap={cv['candles']['liquidity_trap']}", flush=True)
            
            # OTC detailed info
            if cv.get('otc', {}).get('otc_buy') or cv.get('otc', {}).get('otc_sell'):
                otc = cv['otc']
                print(f"💰 OTC | {'BUY' if otc.get('otc_buy') else 'SELL'} | strength={otc.get('strength',0):.1f} | "
                      f"move={otc.get('move_bps',0):.1f}bps | flow={otc.get('visible_flow_ratio',0)*100:.1f}% | "
                      f"reason={otc.get('reason','')}", flush=True)
            
            # EMA detailed info
            if ema_ctx.get("label") != "none":
                print(f"📈 EMA CROSSOVER | {ema_ctx.get('label')} | score={ema_ctx.get('score',0):.1f} | side={ema_ctx.get('side','flat')}", flush=True)
                
            # ChartPrime detailed info
            if cp_boxes.get("side"):
                print(f"📦 CHART PRIME | {cp_boxes['side'].upper()} | score={cp_boxes.get('box_score',0):.1f} | "
                      f"sup={cp_boxes.get('support_level','N/A')} | res={cp_boxes.get('resistance_level','N/A')}", flush=True)
                
            # Liquidity Cycle detailed info
            if liq_cycle.get("regime") != "chop":
                print(f"💧 LIQUIDITY CYCLE | {liq_cycle['regime']} | bias={liq_cycle.get('continuation_bias','none')} | "
                      f"ADX={liq_cycle.get('adx',0):.1f} | ART={liq_cycle.get('art_phase','')}", flush=True)
            
            print("✅ ENHANCED ADDONS LIVE", flush=True)

        return {"bm": bm, "flow": flow, "cv": cv, "mode": mode, "gz": gz, "wallet": wallet}
    except Exception as e:
        print(f"🟨 AddonLog error: {e}", flush=True)
        return {"bm": None, "flow": None, "cv": {"b":0,"s":0,"score_b":0.0,"score_s":0.0,"ind":{}},
                "mode": {"mode":"n/a"}, "gz": None, "wallet": ""}

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
    
    # OTC note
    otc_note = ""
    if council_data.get("otc", {}).get("otc_buy"):
        otc_note = f" | 💰 OTC BUY s={council_data['otc'].get('strength',0):.1f}"
    elif council_data.get("otc", {}).get("otc_sell"):
        otc_note = f" | 💰 OTC SELL s={council_data['otc'].get('strength',0):.1f}"
    
    # EMA note
    ema_note = ""
    ema_ctx = council_data.get("ema", {})
    if ema_ctx.get("label") != "none":
        ema_note = f" | 📈 EMA:{ema_ctx.get('label','')}"
        
    # ChartPrime note
    cp_note = ""
    cp_boxes = council_data.get("cp_boxes", {})
    if cp_boxes.get("side"):
        cp_note = f" | 📦 {cp_boxes['side'][:3].upper()}({cp_boxes.get('box_score',0):.1f})"
        
    # Liquidity note
    liq_note = ""
    liq_cycle = council_data.get("liquidity_cycle", {})
    if liq_cycle.get("regime") != "chop":
        liq_note = f" | 💧 {liq_cycle['regime'][:8]}"
    
    votes = council_data
    print(f"🎯 EXECUTE: {side.upper()} {qty:.4f} @ {price:.6f} | "
          f"mode={mode} | votes={votes['b']}/{votes['s']} score={votes['score_b']:.1f}/{votes['score_s']:.1f}"
          f"{gz_note}{otc_note}{ema_note}{cp_note}{liq_note}", flush=True)

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
def open_market_enhanced(side, qty, price):
    if qty <= 0: 
        log_e("skip open (qty<=0)")
        return False
    
    df = fetch_ohlcv()
    current_price = price or float(df['close'].iloc[-1])
    
    # Enhanced analysis
    snap = emit_snapshots(ex, SYMBOL, df)
    votes = snap["cv"]
    footprint = votes.get("footprint", {})
    ema_ctx = votes.get("ema", {})
    
    mode_data = decide_strategy_mode_enhanced(df, 
                                   adx=votes["ind"].get("adx"),
                                   di_plus=votes["ind"].get("plus_di"),
                                   di_minus=votes["ind"].get("minus_di"),
                                   rsi_ctx=rsi_ma_context(df),
                                   footprint=footprint,
                                   ema_ctx=ema_ctx)
    
    mode = mode_data["mode"]
    gz = snap["gz"]
    
    # Enhanced management config
    management_config = setup_trade_management(mode)
    
    success = execute_trade_decision(side, price, qty, mode, votes, gz)
    
    if success:
        # نحسب قوة الإشارة
        side_label = "long" if side == "buy" else "short"
        signal_strength = calculate_signal_strength(df, votes["ind"], side_label)
        
        # نحدّد Profile جني الأرباح (سكالب خفيف / نص ترند / ترند قوي)
        trade_profile = classify_trade_profile(signal_strength, mode)
        
        # تحديث Risk Management
        risk_on_new_trade(STATE, side_label, price)
        
        STATE.update({
            "open": True, 
            "side": side_label, 
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
            "trade_profile": trade_profile,
        })
        
        # ===================== HUNTER PATCH: IRON SL =====================
        if USE_IRON_SL_TICKS:
            tick = get_tick_size(ex, SYMBOL)
            STATE["hard_sl"] = compute_iron_sl(
                STATE["entry"], STATE["side"], tick, HARD_SL_TICKS
            )
            log_w(f"🎯 IRON SL SET @ {STATE['hard_sl']:.6f} (8 ticks)")
        # ================================================================
        
        save_state({
            "in_position": True,
            "side": "LONG" if side.upper().startswith("B") else "SHORT",
            "entry_price": price,
            "position_qty": qty,
            "leverage": LEVERAGE,
            "mode": mode,
            "management": management_config,
            "signal_strength": signal_strength,
            "trade_profile": trade_profile,
            "gz_snapshot": gz if isinstance(gz, dict) else {},
            "cv_snapshot": votes if isinstance(votes, dict) else {},
            "footprint_snapshot": footprint if isinstance(footprint, dict) else {},
            "ema_snapshot": ema_ctx if isinstance(ema_ctx, dict) else {},
            "opened_at": int(time.time()),
            "partial_taken": False,
            "breakeven_armed": False,
            "trail_active": False,
            "trail_tightened": False,
            "last_status_log_ts": 0.0,
            "hard_sl": STATE.get("hard_sl"),
        })

        # === لوج افتتاح الصفقة + خطة جني الأرباح ===
        mgmt = management_config or {}
        tp1_pct           = mgmt.get("tp1_pct", TP1_PCT_BASE / 100.0) * 100.0
        be_activate_pct   = mgmt.get("be_activate_pct", BREAKEVEN_AFTER / 100.0) * 100.0
        trail_activate_pct= mgmt.get("trail_activate_pct", TRAIL_ACTIVATE_PCT / 100.0) * 100.0
        atr_trail_mult    = mgmt.get("atr_trail_mult", ATR_TRAIL_MULT)

        side_txt = "BUY" if side == "buy" else "SELL"

        log_banner("NEW POSITION OPENED")
        log_g(
            f"📌 OPEN {side_txt} | mode={mode} | qty={qty:.4f} | entry={price:.6f} | "
            f"signal_strength={signal_strength:.1f}"
        )
        log_i(
            f"🎯 PROFIT PLAN | TP1≈{tp1_pct:.2f}% | BE@{be_activate_pct:.2f}% | "
            f"TRAIL@{trail_activate_pct:.2f}% ATR×{atr_trail_mult:.2f}"
        )
        
        log_g(f"✅ ENHANCED POSITION OPENED: {side.upper()} | mode={mode} | signal_strength={signal_strength:.1f}")
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

# =================== STATE ===================
STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
    "tp1_done": False, "highest_profit_pct": 0.0,
    "profit_targets_achieved": 0,
    
    # نمط الصفقة (لجني الأرباح): SCALP_LIGHT / MID_TREND / STRONG_TREND
    "trade_profile": None,
    "signal_strength": 0.0,
    
    # --- Risk / Guards ---
    "max_loss_price": None,           # سعر ستوب الصفقة الحالية
    "last_closed_pnl_pct": 0.0,       # نسبة ربح/خسارة آخر صفقة مغلقة
    "post_big_win_mode": False,       # هل إحنا في وضع حماية بعد مكسب كبير؟
    "post_big_win_bars_left": 0,      # عدد الشموع المتبقية في وضع Post Big Win
    
    # --- HUNTER PATCH ---
    "hard_sl": None,                  # ستوب حديدي 8 نقاط
    "hist_adx": [],                   # تاريخ ADX للمراقبة
    "hist_atr": [],                   # تاريخ ATR للمراقبة
}
compound_pnl = 0.0
wait_for_next_signal_side = None

# =================== WAIT FOR NEXT SIGNAL ===================
def wait_gate_allow(df, info, allow_override=False, override_tag=""):
    """بوابة الانتظار: تتحول من Hard Block إلى Soft Block عند إشارات قوية."""
    if wait_for_next_signal_side is None:
        return True, ""

    if allow_override:
        return True, f"wait-gate OVERRIDDEN: {override_tag}"

    need = ((wait_for_next_signal_side == "buy"  and info.get("long")) or
            (wait_for_next_signal_side == "sell" and info.get("short")))

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
                
                # تحديث Risk Management
                realized_pnl_pct = compute_unrealized_pnl_pct(STATE, px)
                risk_on_trade_closed(STATE, realized_pnl_pct)
                
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
    
    # تحديث Risk Management لو كان هناك صفقة مفتوحة
    if STATE["open"] and STATE["entry"]:
        current_price = price_now() or STATE["entry"]
        realized_pnl_pct = compute_unrealized_pnl_pct(STATE, current_price)
        risk_on_trade_closed(STATE, realized_pnl_pct)
    
    STATE.update({
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
        "tp1_done": False, "highest_profit_pct": 0.0, "profit_targets_achieved": 0,
        "trail_tightened": False, "partial_taken": False,
        "max_loss_price": None,  # إعادة تعيين Stop Loss
        "trade_profile": None,
        "signal_strength": 0.0,
        "last_status_log_ts": 0.0,
        "hard_sl": None,  # إعادة تعيين Iron SL
        "hist_adx": [],
        "hist_atr": [],
    })
    save_state({"in_position": False, "position_qty": 0})
    
    # تفعيل انتظار الإشارة التالية
    wait_for_next_signal_side = "sell" if prev_side=="long" else ("buy" if prev_side=="short" else None)
    log_i(f"🛑 WAIT FOR NEXT SIGNAL: {wait_for_next_signal_side}")
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

    # =================== ATR REGIME → TRADE MANAGEMENT ===================
    regime, buf_mult = atr_regime_ind(ind)   # quiet | normal | expansion

    # افتراضات إدارة (قيمك الحالية)
    trail_mult = state["management"].get("atr_trail_mult", 1.6)
    tp1_pct    = state["management"].get("tp1_pct", 0.40/100.0)

    # Quiet: سوق هادي → شدّ الحماية
    if regime == "quiet":
        trail_mult = max(1.2, trail_mult * 0.85)
        tp1_pct    = min(tp1_pct, 0.35/100.0)
        state["management"]["mode_hint"] = "TIGHTEN (quiet)"

    # Normal: سيبه زي ما هو
    elif regime == "normal":
        state["management"]["mode_hint"] = "NORMAL"

    # Expansion: انفجار → امسك الصفقة وركّب
    elif regime == "expansion":
        trail_mult = min(2.2, trail_mult * 1.20)
        tp1_pct    = max(tp1_pct, 0.50/100.0)
        state["management"]["mode_hint"] = "HOLD (expansion)"

    # طبّق القيم
    state["management"]["atr_trail_mult"] = trail_mult
    state["management"]["tp1_pct"] = tp1_pct

    # =================== GOLDEN FAILURE → EXIT / REVERSAL ===================
    # لو الصفقة اتفتحت بسبب Golden
    opened_by_golden = state.get("opened_by") == "golden"

    if opened_by_golden:
        # شروط الفشل (متوازنة)
        px = float(ind.get("price", df["close"].iloc[-1]))
        adx = float(ind.get("adx", 0.0))
        pdi = float(ind.get("plus_di", 0.0))
        mdi = float(ind.get("minus_di", 0.0))

        # قفل عكسي داخل نفس المنطقة + انقلاب DI
        fail_long  = (side == "long"  and (mdi > pdi) and adx_falling)
        fail_short = (side == "short" and (pdi > mdi) and adx_falling)

        # تأكيد إضافي: ADX مش بيبني
        if (fail_long or fail_short) and adx < 20:
            log_w("🟨 GOLDEN FAILURE detected → early exit")
            return {"action": "close", "why": "golden_failure", "log": "GOLDEN FAILURE → early exit"}

        # خيار الانعكاس (اختياري — خليها False لو عايز خروج فقط)
        ALLOW_REVERSAL = True
        if ALLOW_REVERSAL and (fail_long or fail_short) and adx >= 18:
            rev_side = "sell" if side == "long" else "buy"
            log_w(f"🟣 GOLDEN FAILURE → REVERSAL {rev_side.upper()}")
            return {"action": "reverse", "why": "golden_failure_reversal", "side": rev_side}

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

    # --- OTC Reversal بعد TP1 (سيولة مخفية عكسية) ---
    # ملاحظة: هنا pnl_pct = كسر (0.01 = 1%)
    if state.get('tp1_done') and pnl_pct >= OTC_EXIT_MIN_PNL_PCT:
        try:
            otc = detect_otc_flows(df)
        except Exception as e:
            otc = {"otc_buy": False, "otc_sell": False, "strength": 0.0, "reason": f"error:{e}"}

        opp_otc = False
        if side == "long" and otc.get("otc_sell"):
            opp_otc = True
        elif side == "short" and otc.get("otc_buy"):
            opp_otc = True

        if opp_otc and otc.get("strength", 0.0) >= OTC_EXIT_MIN_STRENGTH:
            return {
                "action": "close",
                "why": "otc_reversal",
                "log": (
                    f"🔴 CLOSE STRONG | OTC reversal after TP1 | "
                    f"side={side} pnl={pnl_pct*100:.2f}% "
                    f"strength={otc.get('strength',0):.1f} "
                    f"move={otc.get('move_bps',0):.1f}bps "
                    f"flow={otc.get('visible_flow_ratio',0)*100:.1f}% "
                    f"reason={otc.get('reason','')}"
                )
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

# =================== ENHANCED TRADE MANAGEMENT ===================
def manage_after_entry_enhanced(df, ind, info):
    """إدارة محسنة للمركز مع Smart Profit AI + Smart Exit Guard + Hard Stop + HUNTER PATCH"""
    if not STATE["open"] or STATE["qty"] <= 0:
        return

    px   = info["price"]
    entry = STATE["entry"]
    side  = STATE["side"]
    qty   = STATE["qty"]
    mode  = STATE.get("mode", "trend")
    
    # ========= HUNTER PATCH: IRON SL =========
    if USE_IRON_SL_TICKS and STATE.get("hard_sl"):
        sl = STATE["hard_sl"]
        if side == "long" and px <= sl:
            pnl_pct = compute_unrealized_pnl_pct(STATE, px)
            log_w(f"🛑 IRON SL HIT (LONG) @ {px:.6f} | Loss: {pnl_pct*100:.2f}%")
            close_market_strict("IRON_SL")
            risk_on_trade_closed(STATE, pnl_pct)
            return
        elif side == "short" and px >= sl:
            pnl_pct = compute_unrealized_pnl_pct(STATE, px)
            log_w(f"🛑 IRON SL HIT (SHORT) @ {px:.6f} | Loss: {pnl_pct*100:.2f}%")
            close_market_strict("IRON_SL")
            risk_on_trade_closed(STATE, pnl_pct)
            return
    
    # ========= HARD MAX LOSS GUARD (ستوب صارم) =========
    max_loss_price = STATE.get("max_loss_price")
    if max_loss_price:
        if side == "long" and px <= max_loss_price:
            pnl_pct = compute_unrealized_pnl_pct(STATE, px)
            log_w(f"🛑 MAX-LOSS GUARD HIT (LONG) @ {px:.6f} | Loss: {pnl_pct*100:.2f}%")
            close_market_strict("max_loss_guard_long")
            risk_on_trade_closed(STATE, pnl_pct)
            return
        elif side == "short" and px >= max_loss_price:
            pnl_pct = compute_unrealized_pnl_pct(STATE, px)
            log_w(f"🛑 MAX-LOSS GUARD HIT (SHORT) @ {px:.6f} | Loss: {pnl_pct*100:.2f}%")
            close_market_strict("max_loss_guard_short")
            risk_on_trade_closed(STATE, pnl_pct)
            return
    
    # PnL % (كـ نسبة مئوية)
    pnl_pct = (px - entry) / entry * 100.0 * (1 if side == "long" else -1)
    STATE["pnl"] = pnl_pct

    if pnl_pct > STATE.get("highest_profit_pct", 0.0):
        STATE["highest_profit_pct"] = pnl_pct

    # ========= HUNTER PATCH: ADX/ATR WEAKNESS EXIT =========
    watch = adx_atr_watcher(ind, STATE)
    
    weak_votes = 0
    if watch["adx"] < ADX_EXIT_WEAK: weak_votes += 1
    if watch["atr_reg"] == "contract": weak_votes += 1
    if watch["adx_slope"] < 0: weak_votes += 1

    if pnl_pct >= MIN_PROFIT_TO_EXIT and weak_votes >= EXIT_WEAKNESS_VOTES:
        log_w(f"🛑 ADX/ATR WEAKNESS EXIT | votes={weak_votes} | pnl={pnl_pct:.2f}%")
        close_market_strict("ADX_ATR_WEAKNESS")
        return

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
                    
                    # لو دي آخر مرحلة جني أرباح في السلم: اقفل الصفقة بالكامل
                    if target_num >= (len(SCALP_TPS) if mode == "scalp" else len(TREND_TPS)):
                        close_market_strict("all_targets_achieved")
                        return
                except Exception as e:
                    log_e(f"❌ Smart TP failed: {e}")
            else:
                log_i(f"DRY_RUN: Smart TP{target_num} close {close_qty:.4f}")

    # === لوج حالة الصفقة كل 30 ثانية ===
    now_ts = time.time()
    last_ts = STATE.get("last_status_log_ts", 0.0)

    if now_ts - last_ts >= POSITION_STATUS_LOG_INTERVAL:
        STATE["last_status_log_ts"] = now_ts

        management = STATE.get("management", {}) or {}
        tp1_pct            = management.get("tp1_pct", TP1_PCT_BASE / 100.0) * 100.0
        be_activate_pct    = management.get("be_activate_pct", BREAKEVEN_AFTER / 100.0) * 100.0
        trail_activate_pct = management.get("trail_activate_pct", TRAIL_ACTIVATE_PCT / 100.0) * 100.0
        atr_trail_mult     = management.get("atr_trail_mult", ATR_TRAIL_MULT)

        mode = STATE.get("mode", "scalp")
        max_pnl = STATE.get("highest_profit_pct", 0.0)
        targets_done = STATE.get("profit_targets_achieved", 0)
        side_txt = "BUY" if side == "long" else "SELL"

        log_i(
            f"📊 POSITION STATUS | {side_txt} | mode={mode} | "
            f"price={px:.6f} | entry={entry:.6f} | qty={qty:.4f}"
        )
        log_i(
            f"   PnL={pnl_pct:.2f}% (max={max_pnl:.2f}%) | TP_done={targets_done} | "
            f"plan: TP1≈{tp1_pct:.2f}% / BE@{be_activate_pct:.2f}% / "
            f"TRAIL@{trail_activate_pct:.2f}% ATR×{atr_trail_mult:.2f}"
        )

    # ========= الإدارة الكلاسيك (TP1 + BE + Trail + Dust) =========
    current_atr      = ind.get("atr", 0.0)
    management       = STATE.get("management", {})
    
    tp1_pct          = management.get("tp1_pct", TP1_PCT_BASE/100.0)
    be_activate_pct  = management.get("be_activate_pct", BREAKEVEN_AFTER/100.0)
    trail_activate_pct = management.get("trail_activate_pct", TRAIL_ACTIVATE_PCT/100.0)
    atr_trail_mult   = management.get("atr_trail_mult", ATR_TRAIL_MULT)

    # نحول PnL من % إلى كسور عشان نستخدمه مع الحراس اللي شغّالين بالـ fraction
    pnl_frac = pnl_pct / 100.0

    # TP1 جزئي (مرة واحدة فقط)
    if not STATE.get("tp1_done") and pnl_frac >= tp1_pct:
        close_fraction = TP1_CLOSE_FRAC
        close_qty = safe_qty(STATE["qty"] * close_fraction)
        if close_qty > 0:
            close_side = "sell" if STATE["side"] == "long" else "buy"
            if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                try:
                    ex.create_order(SYMBOL, "market", close_side, close_qty, None, _params_close())
                    log_g(f"✅ TP1 HIT: closed {close_fraction*100:.0f}%")
                except Exception as e:
                    log_e(f"❌ TP1 close failed: {e}")
            STATE["qty"] = safe_qty(STATE["qty"] - close_qty)
            STATE["tp1_done"] = True
            STATE["profit_targets_achieved"] += 1

    # تفعيل Breakeven
    if not STATE.get("breakeven_armed") and pnl_frac >= be_activate_pct:
        STATE["breakeven_armed"] = True
        STATE["breakeven"] = entry
        log_i("BREAKEVEN ARMED")

    # تفعيل التريل
    if not STATE.get("trail_active") and pnl_frac >= trail_activate_pct:
        STATE["trail_active"] = True
        log_i("TRAIL ACTIVATED")

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
            log_w(f"TRAIL STOP: {px} vs trail {STATE['trail']}")
            close_market_strict("trail_stop")
            return

    # تنفيذ Breakeven الصارم
    if STATE.get("breakeven"):
        if (side == "long" and px <= STATE["breakeven"]) or (side == "short" and px >= STATE["breakeven"]):
            log_w(f"BREAKEVEN STOP: {px} vs breakeven {STATE['breakeven']}")
            close_market_strict("breakeven_stop")
            return

    # Dust guard: لو الكمية بقت فتات اقفل وخلاص
    if STATE["qty"] <= FINAL_CHUNK_QTY:
        log_w(f"DUST GUARD: qty {STATE['qty']} <= {FINAL_CHUNK_QTY}, closing...")
        close_market_strict("dust_guard")
        return

    # ========= Smart Exit Guard (Golden Reversal + Wick/Flow/Wall + OTC) =========
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

        # إغلاق صارم عند Golden Reversal أو Hard Close أو OTC Reversal
        elif act == "close":
            close_market_strict(guard.get("why", "smart_exit_guard"))
            return
        elif act == "reverse":
            # تنفيذ الانعكاس
            rev_side = guard.get("side")
            if rev_side and STATE["qty"] > 0:
                close_market_strict("golden_failure_reverse_close")
                # بعد الإغلاق، افتح صفقة في الاتجاه المعاكس
                time.sleep(1)
                qty = compute_size(balance_usdt(), px)
                open_market_enhanced(rev_side, qty, px)
        # متعمدين نتجاهل "partial" هنا عشان ما نعملش TP1 مزدوج (Smart AI + Guard)

# =================== MANUAL CLOSE SYNC FUNCTION ===================
def sync_manual_close():
    """
    تزامن الإغلاق اليدوي من المنصة:
    - يتحقق من المركز الحقيقي على البورصة
    - يضبط STATE إذا كان هناك تعارض
    """
    global STATE, wait_for_next_signal_side
    
    # قراءة المركز من البورصة
    exch_qty, exch_side, exch_entry = _read_position()
    
    # 1) البورصة Flat والبوت فاكر في صفقة ⇒ أغلق داخليًا
    if exch_qty <= 0 and STATE.get("open"):
        log_w("⚠️ Detected manual close on exchange → syncing state (flat)")
        _reset_after_close("manual_close_sync")
        return
    
    # 2) البورصة فيها صفقة والبوت فاكر إنه Flat ⇒ نعتبرها صفقة معلّقة ونكمّل إدارتها
    if exch_qty > 0 and not STATE.get("open"):
        side = exch_side or "long"
        STATE.update({
            "open": True,
            "side": side,
            "entry": exch_entry,
            "qty": exch_qty,
            "bars": 0,
            "highest_profit_pct": 0.0,
            "profit_targets_achieved": 0,
            "max_loss_price": exch_entry * (1.0 + MAX_LOSS_PCT) if side == "long" else exch_entry * (1.0 - MAX_LOSS_PCT),
        })
        log_g(f"♻️ Re-attached to existing exchange position | side={side} qty={exch_qty} entry={exch_entry}")

# =================== ENHANCED TRADE LOOP ===================
def trade_loop_enhanced():
    """حلقة تداول محسنة مع Golden Zone Pro وSmart Profit AI وVWAP وOTC Detection وEMA Cross + Risk Guards + ChartPrime + HUNTER PATCH + Market State Engine"""
    global wait_for_next_signal_side
    loop_i = 0
    
    while True:
        try:
            # 🔄 أولاً: مزامنة الإغلاق اليدوي
            sync_manual_close()
            
            # تحديث عدّاد Post Big Win كل لوب
            tick_post_big_win_decay(STATE)
            
            # جمع البيانات الأساسية
            bal = balance_usdt()
            px = price_now()
            df = fetch_ohlcv()
            info = rf_signal_live(df)
            ind = compute_indicators(df)
            spread_bps = orderbook_spread_bps()
            
            # Enhanced Snapshots
            snap = emit_snapshots(ex, SYMBOL, df,
                                balance_fn=lambda: float(bal) if bal else None,
                                pnl_fn=lambda: float(compound_pnl))
            
            # تحديث حالة الربح/الخسارة
            if STATE["open"] and px:
                STATE["pnl"] = (px-STATE["entry"])*STATE["qty"] if STATE["side"]=="long" else (STATE["entry"]-px)*STATE["qty"]
            
            # إدارة الصفقة المفتوحة مع Smart Profit AI + Hard Stop
            if STATE["open"]:
                manage_after_entry_enhanced(df, ind, {
                    "price": px or info["price"], 
                    "bm": snap["bm"],
                    "flow": snap["flow"],
                    **info
                })
            
            # ===================== HUNTER PATCH: UNIFIED ENTRY DECISION =====================
            reason = None
            
            # (A) Spread Guard
            if spread_bps is not None and spread_bps > MAX_SPREAD_BPS:
                gate_block(
                    "SPREAD_GUARD",
                    gate_value=spread_bps,
                    missing=f"spread_bps <= {MAX_SPREAD_BPS}",
                    extra={"px": px, "spread_bps": spread_bps}
                )
                reason = f"spread too high {spread_bps:.2f}bps"
                continue

            council_data = council_votes_pro_enhanced(df)
            gz = council_data.get("gz")
            footprint = council_data.get("footprint", {})
            otc = council_data.get("otc", {})
            ema_ctx = council_data.get("ema", {})
            cp_boxes = council_data.get("cp_boxes", {})
            liq_cycle = council_data.get("liquidity_cycle", {})

            # ===================== MARKET STATE ENGINE (NEW) =====================
            ms = market_state_engine(df, council_data["ind"], px)
            momentum_gate_log(ms)

            # SR/ATR buffer rule (قاعدتك)
            sr = ms["sr"] if ms.get("ok") else {"ok": False}
            reg = ms["atr_regime"] if ms.get("ok") else {"buf_atr":0.40, "atr": _tf(ind.get("atr"),0.0)}

            # منطقة "قرب الدعم/المقاومة" الديناميكية
            buf_px = reg["buf_atr"] * max(1e-12, reg["atr"])
            near_res = sr.get("ok") and (px >= (sr["resistance"] - buf_px) and px < sr["resistance"])
            near_sup = sr.get("ok") and (px <= (sr["support"] + buf_px) and px > sr["support"])

            # Breakout/Breaddown confirmed
            breakout_ok  = ms.get("breakout_up", {}).get("ok", False)
            breakdown_ok = ms.get("breakout_dn", {}).get("ok", False)

            # تطبيق قاعدتك حرفيًا كـ Gate
            if near_res and not breakout_ok:
                gate_block(
                    "BUY_NEAR_RES_BLOCKED",
                    gate_value=f"px~res buf={reg['buf_atr']:.2f}ATR",
                    missing="need Breakout Confirmed (Close فوق المقاومة + Body + ADX/DI)",
                    extra={"px": px, "res": sr.get("resistance"), "atr": reg.get("atr"), "adx": ind.get("adx"), "pdi": ind.get("plus_di"), "mdi": ind.get("minus_di")}
                )

            if near_sup and not breakdown_ok:
                gate_block(
                    "SELL_NEAR_SUP_BLOCKED",
                    gate_value=f"px~sup buf={reg['buf_atr']:.2f}ATR",
                    missing="need Breakdown Confirmed (Close تحت الدعم + Body + ADX/DI)",
                    extra={"px": px, "sup": sr.get("support"), "atr": reg.get("atr"), "adx": ind.get("adx"), "pdi": ind.get("plus_di"), "mdi": ind.get("minus_di")}
                )

            # ===================== HUNTER PATCH v2 (SMART ENTRY, NOT RANDOM) =====================

            # مراقبة ADX/ATR
            watch = adx_atr_watcher(ind, STATE)

            # Launch/Compression (كان عامل قفل كامل — هنحوّله لفلتر)
            cl = accumulation_launch_signal(ind, STATE)

            # Zones / Signals
            gb = gz and gz.get("ok") and gz.get("zone") and gz["zone"]["type"] == "golden_bottom"
            gt = gz and gz.get("ok") and gz.get("zone") and gz["zone"]["type"] == "golden_top"

            buy_liquidity  = footprint.get('ok') and footprint.get('absorption_bull')
            sell_liquidity = footprint.get('ok') and footprint.get('absorption_bear')

            # Flow (Order Flow spike)
            ob_signal = None
            flow = snap.get("flow")
            if flow and flow.get('ok'):
                if flow['delta_z'] >= FLOW_SPIKE_Z and flow.get('cvd_trend') == 'up':
                    ob_signal = ("bullish", flow['delta_z'])
                elif flow['delta_z'] <= -FLOW_SPIKE_Z and flow.get('cvd_trend') == 'down':
                    ob_signal = ("bearish", flow['delta_z'])

            # OTC Hidden flow (لو متوفر في council_data أو snap)
            otc_buy  = bool(otc.get("otc_buy"))
            otc_sell = bool(otc.get("otc_sell"))

            # ChartPrime Boxes + Liquidity Cycle (دي كانت خارج zone_ok وبالتالي بتقتل الصفقات)
            cb_side = cp_boxes.get("side")
            cb_score = float(cp_boxes.get("box_score", 0) or 0)

            near_support = bool(liq_cycle.get("near_support", False))
            near_resistance = bool(liq_cycle.get("near_resistance", False))

            cp_support_ok = (cb_side == "support" and cb_score >= 5.0 and near_support)
            cp_resist_ok  = (cb_side == "resistance" and cb_score >= 5.0 and near_resistance)

            liq_reg = liq_cycle.get("regime")  # bull_accumulation / bear_accumulation / ...
            bull_accum_ok = (liq_reg == "bull_accumulation")
            bear_accum_ok = (liq_reg == "bear_accumulation")

            # ---- SMART zone_ok (مش زون واحدة بس) ----
            # Include shock, breakout, breakdown, and liquidity grabs
            shock_ok = bool(ms.get("shock", {}).get("shock"))
            break_ok = bool(breakout_ok or breakdown_ok)
            liq_grab_ok = bool(ms.get("liq", {}).get("fakeout_up") or ms.get("liq", {}).get("fakeout_dn"))
            
            zone_ok = bool(
                gb or gt or
                buy_liquidity or sell_liquidity or
                otc_buy or otc_sell or
                (ob_signal is not None) or
                cp_support_ok or cp_resist_ok or
                bull_accum_ok or bear_accum_ok or
                shock_ok or break_ok or liq_grab_ok
            )

            # Debug واضح بدل "price خارج Zone" المبهم
            log_i(
                f"ENTRY_CTX | px={px:.6f} zone_ok={zone_ok} "
                f"GB={gb} GT={gt} LIQ(B/S)={buy_liquidity}/{sell_liquidity} "
                f"CP(S/R)={cp_support_ok}/{cp_resist_ok} LIQREG={liq_reg} "
                f"OTC(B/S)={otc_buy}/{otc_sell} FLOW={ob_signal} "
                f"SHOCK={shock_ok} BREAK={break_ok} LIQ_GRAB={liq_grab_ok}"
            )
            
            # STATE ENGINE logging
            log_i(
                f"STATE_ENGINE | phase={ms.get('phase')} bias={ms.get('bias')} "
                f"atr%={ms.get('atr_regime',{}).get('atr_pct',0)*100:.2f}% reg={ms.get('atr_regime',{}).get('regime')} "
                f"SR(sup/res)={ms.get('sr',{}).get('support')}/{ms.get('sr',{}).get('resistance')} "
                f"BO={breakout_ok} BD={breakdown_ok} SHOCK={ms.get('shock',{}).get('shock')} "
                f"LIQ(up/dn)={ms.get('liq',{}).get('fakeout_up')}/{ms.get('liq',{}).get('fakeout_dn')}"
            )

            # (B) Compression بدون Zone
            if cl.get("compression") and not zone_ok:
                gate_block(
                    "ACCUM_COMPRESSION",
                    gate_value=True,
                    missing="need zone_ok OR golden OR strong confluence",
                    extra={"px": px, "atr": ind.get("atr"), "adx": ind.get("adx"), "regime": cl.get("regime")}
                )
                log_i("🧊 ACCUMULATION: compression بدون zone confluence → WAIT")
                reason = "accumulation_wait_no_zone"
                continue

            # بدل قفل التجميع: نخليه "فلتر مخاطرة"
            # - لو تجميع (compression) بس مفيش أي Zone قوية => انتظر
            if cl.get("compression") and not zone_ok:
                log_i("🧊 ACCUMULATION: compression بدون zone confluence → WAIT")
                reason = "accumulation_wait_no_zone"
            else:
                # ==== Score بناءً على Confluence ====
                buy_score = 0
                sell_score = 0
                reasons = []

                # Strong zones
                if gb: buy_score += 3; reasons.append("GoldenBottom")
                if gt: sell_score += 3; reasons.append("GoldenTop")

                if cp_support_ok: buy_score += 2; reasons.append(f"CP_Support({cb_score:.1f})")
                if cp_resist_ok:  sell_score += 2; reasons.append(f"CP_Resist({cb_score:.1f})")

                if bull_accum_ok: buy_score += 2; reasons.append("LiqReg_BullAccum")
                if bear_accum_ok: sell_score += 2; reasons.append("LiqReg_BearAccum")

                if buy_liquidity:  buy_score += 2; reasons.append("Absorption_Bull")
                if sell_liquidity: sell_score += 2; reasons.append("Absorption_Bear")

                if otc_buy:  buy_score += 2; reasons.append("OTC_Buy")
                if otc_sell: sell_score += 2; reasons.append("OTC_Sell")

                if ob_signal and ob_signal[0] == "bullish":
                    buy_score += 2; reasons.append(f"Flow_Bull(z={ob_signal[1]:.2f})")
                if ob_signal and ob_signal[0] == "bearish":
                    sell_score += 2; reasons.append(f"Flow_Bear(z={ob_signal[1]:.2f})")
                    
                # Shock detection
                if shock_ok:
                    if ms.get("shock", {}).get("dir") == "pump":
                        buy_score += 3; reasons.append(f"SHOCK_PUMP(str={ms['shock'].get('strength')})")
                    elif ms.get("shock", {}).get("dir") == "dump":
                        sell_score += 3; reasons.append(f"SHOCK_DUMP(str={ms['shock'].get('strength')})")
                
                # Breakout/Breakdown
                if breakout_ok: buy_score += 3; reasons.append(f"BREAKOUT({ms['breakout_up'].get('why','')})")
                if breakdown_ok: sell_score += 3; reasons.append(f"BREAKDOWN({ms['breakout_dn'].get('why','')})")
                
                # Liquidity grab
                if liq_grab_ok:
                    if ms.get("liq", {}).get("fakeout_up"):
                        sell_score += 2; reasons.append("LIQ_GRAB_UP")
                    if ms.get("liq", {}).get("fakeout_dn"):
                        buy_score += 2; reasons.append("LIQ_GRAB_DN")

                # ADX/DI سياق الترند (Boost فقط — مش فيتو)
                if watch["regime"] == "TREND":
                    if watch["side"] == "up":
                        buy_score += 2; reasons.append("Trend_Up(ADX/DI)")
                    elif watch["side"] == "down":
                        sell_score += 2; reasons.append("Trend_Down(ADX/DI)")

                # Dynamic Score Threshold بدل 6 ثابتة
                dyn_thr = 6
                if watch.get("regime") == "TREND" and float(ind.get("adx",0)) >= 25:
                    dyn_thr = 5   # في الترند القوي ندخل بدري
                elif cl.get("compression"):
                    dyn_thr = 7   # في الضغط/التجميع نطلب تأكيد أعلى

                final_signal = None
                if buy_score >= dyn_thr and buy_score > sell_score + 1:
                    final_signal = "buy"
                elif sell_score >= dyn_thr and sell_score > buy_score + 1:
                    final_signal = "sell"

                # (C) Score Threshold Gate
                if not final_signal:
                    gate_block(
                        "SCORE_THRESHOLD",
                        gate_value=f"buy={buy_score} sell={sell_score}",
                        missing="need score >= threshold AND dominance (+1)",
                        extra={
                            "px": px, "buy_score": buy_score, "sell_score": sell_score,
                            "adx": ind.get("adx"), "plus_di": ind.get("plus_di"), "minus_di": ind.get("minus_di"),
                            "atr": ind.get("atr"), "rsi": ind.get("rsi"),
                            "regime": watch.get("regime") if isinstance(watch, dict) else None
                        }
                    )
                    log_i(f"NO TRADE | buy={buy_score} sell={sell_score} reasons={reasons}")
                    reason = f"insufficient_score buy={buy_score} sell={sell_score}"
                    continue

                # ---- Launch gate: مطلوب فقط لو السوق تجميع + مفيش Trend واضح ----
                need_launch = (watch["regime"] == "ACCUMULATION" and watch["side"] == "flat")
                if need_launch and not cl.get("launch") and not (gb or gt or cp_support_ok or cp_resist_ok or shock_ok or break_ok or liq_grab_ok):
                    log_i("NO TRADE: accumulation يحتاج Launch (لا يوجد Trend ولا Zone قوية)")
                    reason = "need_launch_in_pure_accum"
                else:
                    # Override قوي: Golden أو Confluence عالي
                    override = False
                    tag = ""

                    # Golden مباشرة (أقوى override)
                    if gb or gt:
                        override = True
                        tag = "GOLDEN"

                    # Confluence قوي (OTC/Flow/CP/LiqReg)
                    elif (cp_support_ok or cp_resist_ok or otc_buy or otc_sell or (ob_signal is not None) or bull_accum_ok or bear_accum_ok or shock_ok or break_ok or liq_grab_ok):
                        # بشرط إن السكور محترم
                        if (buy_score >= 7) or (sell_score >= 7):
                            override = True
                            tag = "CONFLUENCE>=7"

                    allow_wait, wait_reason = wait_gate_allow(df, info, allow_override=override, override_tag=tag)

                    # (D) Wait Gate (أكبر قاتل للدخول)
                    if not allow_wait:
                        gate_block(
                            "WAIT_FOR_NEXT_SIGNAL",
                            gate_value=wait_for_next_signal_side,
                            missing=f"need RF signal: {wait_for_next_signal_side}",
                            extra={"px": px, "adx": ind.get("adx"), "plus_di": ind.get("plus_di"), "minus_di": ind.get("minus_di")}
                        )
                        reason = wait_reason
                        continue
                    
                    # Smart Entry Gate Check
                    ind["price"] = float(px or info["price"])
                    allow_entry, gate_reason = smart_entry_gate(final_signal, df, ind, gz)
                    
                    # 🔥 SHOCK FORCE ENTRY 🔥
                    if ms.get("shock", {}).get("shock"):
                        shock_dir = ms["shock"]["dir"]
                        if shock_dir == "dump" and (spread_bps or 0) <= MAX_SPREAD_BPS:
                            final_signal, allow_entry, gate_reason = "sell", True, "SHOCK_DUMP → FORCE SHORT (guards ok)"
                        elif shock_dir == "pump" and (spread_bps or 0) <= MAX_SPREAD_BPS:
                            final_signal, allow_entry, gate_reason = "buy", True, "SHOCK_PUMP → FORCE LONG (guards ok)"
                    
                    if not allow_entry:
                        log_w(f"🚫 ENTRY BLOCKED: {gate_reason}")
                        continue
                    else:
                        log_i(f"✅ ENTRY ALLOWED: {gate_reason}")
                    
                    qty = compute_size(bal, px or info["price"])
                    if qty > 0:
                        ok = open_market_enhanced(final_signal, qty, px or info["price"])
                        if ok:
                            wait_for_next_signal_side = None
                            log_i(f"✅ ENTRY_PASS | sig={final_signal} buy={buy_score} sell={sell_score} | wait_gate=PASS")
                            log_i(f"🎯 HUNTER v2 ENTRY: {final_signal.upper()} | buy={buy_score} sell={sell_score} reasons={reasons}")
                        else:
                            reason = "open_failed"
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
        print(colored(f"📊 {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-d %H:%M:%S')} UTC","cyan"))
        print(colored("─"*100,"cyan"))
        print("📈 INDICATORS & RF")
        print(f"   💲 Price {fmt(info.get('price'))} | RF filt={fmt(info.get('filter'))}  hi={fmt(info.get('hi'))} lo={fmt(info.get('lo'))}")
        print(f"   🧮 RSI={fmt(ind.get('rsi'))}  +DI={fmt(ind.get('plus_di'))}  -DI={fmt(ind.get('minus_di'))}  ADX={fmt(ind.get('adx'))}  ATR={fmt(ind.get('atr'))}")
        print(f"   🎯 ENTRY: COUNCIL PRO + GOLDEN ENTRY + VWAP STRATEGY + OTC DETECTION + EMA CROSSOVER ENGINE + CHART PRIME + HUNTER PATCH + MARKET STATE ENGINE |  spread_bps={fmt(spread_bps,2)}")
        print(f"   ⏱️ closes_in ≈ {left_s}s")
        
        # عرض معلومات الـ Guards
        print("\n🛡️ RISK GUARDS")
        if STATE["open"]:
            stop_price = STATE.get("max_loss_price")
            hard_sl = STATE.get("hard_sl")
            current_price = info.get("price") or 0
            if stop_price:
                stop_distance_pct = abs(stop_price - current_price) / current_price * 100
                stop_side = "BELOW" if STATE['side'] == 'long' else "ABOVE"
                print(f"   🔴 HARD STOP: {stop_side} {fmt(stop_price)} ({stop_distance_pct:.2f}%)")
            if hard_sl:
                sl_distance_pct = abs(hard_sl - current_price) / current_price * 100
                print(f"   🎯 IRON SL: {fmt(hard_sl)} ({sl_distance_pct:.2f}%)")
        
        if STATE.get("post_big_win_mode"):
            print(f"   🛡️ POST-BIG-WIN: ACTIVE ({STATE.get('post_big_win_bars_left', 0)} bars left)")
        
        print("\n🧭 POSITION")
        bal_line = f"Balance={fmt(bal,2)}  Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x  CompoundPnL={fmt(compound_pnl)}  Eq~{fmt((bal or 0)+compound_pnl,2)}"
        print(colored(f"   {bal_line}", "yellow"))
        
        if STATE["open"]:
            lamp='🟩 LONG' if STATE['side']=='long' else '🟥 SHORT'
            print(f"   {lamp}  Entry={fmt(STATE['entry'])}  Qty={fmt(STATE['qty'],4)}  Bars={STATE['bars']}  Trail={fmt(STATE['trail'])}  BE={fmt(STATE['breakeven'])}")
            print(f"   🎯 TP_done={STATE['profit_targets_achieved']}  HP={fmt(STATE['highest_profit_pct'],2)}%")
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
    return f"✅ Council PRO Bot — {SYMBOL} {INTERVAL} — {mode} — Enhanced Candles + Golden Zone Pro + Smart Profit AI + VWAP Strategy + OTC Detection + EMA Crossover Engine + Hard Stop Loss + Post-Big-Win Guard + Auto-Recovery + ChartPrime + HUNTER PATCH + Market State Engine"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "state": STATE, "compound_pnl": compound_pnl,
        "entry_mode": "COUNCIL_PRO_GOLDEN_ENHANCED_VWAP_OTC_EMA_CHARTPRIME_HUNTER_MARKET_STATE", "wait_for_next_signal": wait_for_next_signal_side,
        "risk_guards": {
            "hard_stop_pct": abs(MAX_LOSS_PCT)*100,
            "big_win_pct": BIG_WIN_PCT*100,
            "post_big_win_bars": POST_BIG_WIN_BARS,
            "max_spread_bps": MAX_SPREAD_BPS,
            "final_chunk_qty": FINAL_CHUNK_QTY
        },
        "vwap_strategy": {
            "enabled": VWAP_ENABLED,
            "scalp_band_bps": VWAP_SCALP_BAND_BPS,
            "trend_band_bps": VWAP_TREND_BAND_BPS
        },
        "otc_detection": {
            "enabled": True,
            "window_bars": OTC_WINDOW_BARS,
            "min_move_bps": OTC_MIN_MOVE_BPS,
            "exit_min_strength": OTC_EXIT_MIN_STRENGTH
        },
        "ema_crossover": {
            "enabled": True,
            "fast_period": 9,
            "mid_period": 21,
            "slow_period": 50
        },
        "chart_prime": {
            "enabled": True,
            "lookback_period": 20,
            "vol_len": 2
        },
        "liquidity_cycle": {
            "enabled": True,
            "regime": STATE.get("liquidity_cycle", {}).get("regime", "unknown")
        },
        "hunter_patch": {
            "enabled": True,
            "iron_sl_ticks": HARD_SL_TICKS,
            "adx_accum_max": ADX_ACCUM_MAX,
            "adx_trend_min": ADX_TREND_MIN,
            "adx_exit_weak": ADX_EXIT_WEAK,
            "min_profit_to_exit": MIN_PROFIT_TO_EXIT
        },
        "market_state_engine": {
            "enabled": True,
            "detects": ["accumulation", "breakout", "liquidity_grab", "shock"]
        },
        "current_guards": {
            "post_big_win_active": STATE.get("post_big_win_mode", False),
            "post_big_win_bars_left": STATE.get("post_big_win_bars_left", 0),
            "hard_stop_price": STATE.get("max_loss_price"),
            "iron_sl_price": STATE.get("hard_sl"),
            "last_closed_pnl_pct": STATE.get("last_closed_pnl_pct", 0.0)
        }
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True, "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"], "side": STATE["side"], "qty": STATE["qty"],
        "compound_pnl": compound_pnl, "timestamp": datetime.utcnow().isoformat(),
        "entry_mode": "COUNCIL_PRO_GOLDEN_ENHANCED_VWAP_OTC_EMA_CHARTPRIME_HUNTER_MARKET_STATE", "wait_for_next_signal": wait_for_next_signal_side,
        "risk_guards": {
            "hard_stop_active": STATE.get("max_loss_price") is not None,
            "post_big_win_active": STATE.get("post_big_win_mode", False),
            "iron_sl_active": STATE.get("hard_sl") is not None
        }
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
            # استخدام الدالة المحسنة للاستئناف
            state = enhanced_resume_open_position(ex, SYMBOL, state)
        except Exception as e:
            log_w(f"🔄 Enhanced resume error: {e}\n{traceback.format_exc()}")

    verify_execution_environment()

    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'}  •  {SYMBOL}  •  {INTERVAL}", "yellow"))
    print(colored(f"RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x  •  COUNCIL_PRO=ENHANCED", "yellow"))
    print(colored(f"GOLDEN ENTRY PRO: score≥{GOLDEN_ENTRY_SCORE} | ADX≥{GOLDEN_ENTRY_ADX}", "yellow"))
    print(colored(f"ENHANCED CANDLES: SMC Patterns + Wick exhaustion + Golden reversal", "yellow"))
    print(colored(f"FOOTPRINT ANALYSIS: Volume spikes + Absorption detection", "yellow"))
    print(colored(f"OTC DETECTION: Hidden flow detection + Protection system", "yellow"))
    print(colored(f"SMART PROFIT AI: Dynamic profit taking + Signal strength", "yellow"))
    print(colored(f"VWAP STRATEGY: SCALP(near {VWAP_SCALP_BAND_BPS}bps) | TREND(far {VWAP_TREND_BAND_BPS}bps)", "yellow"))
    print(colored(f"EMA CROSSOVER ENGINE: Strong/Weak Trend Detection (9/21/50)", "yellow"))
    print(colored(f"📦 CHART PRIME: High-Volume Boxes + Liquidity Cycle Monitor", "green"))
    print(colored(f"🛡️ RISK GUARDS: Hard Stop Loss (-{abs(MAX_LOSS_PCT)*100}%) | Post-Big-Win Guard (+{BIG_WIN_PCT*100}%)", "red"))
    print(colored(f"🎯 HUNTER PATCH: Unified Entry | ADX/ATR Smart Monitor | Iron SL {HARD_SL_TICKS} ticks | Launch Detection", "cyan"))
    print(colored(f"🚀 MARKET STATE ENGINE: Accumulation / Liquidity / Breakout / Shock Detection", "magenta"))
    print(colored(f"🔄 AUTO RECOVERY: Enhanced position sync on restart", "green"))
    print(colored(f"🔄 MANUAL CLOSE SYNC: Enabled (detects manual close from exchange)", "green"))
    print(colored(f"EXECUTION: {'ACTIVE' if EXECUTE_ORDERS and not DRY_RUN else 'SIMULATION'}", "yellow"))
    
    logging.info("enhanced service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    
    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)

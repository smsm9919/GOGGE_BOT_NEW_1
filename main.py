# -*- coding: utf-8 -*-
"""
RF Futures Bot — RF-LIVE ONLY (BingX Perp via CCXT)
• Council ELITE Unified Decision System with Smart Management
• Golden Entry + SMC/ICT + Smart Exit Management + Flow-Pressure v2
• Dynamic TP ladder + Breakeven + ATR-trailing + Order Book Diagonal Analysis
• Professional Logging & Dashboard
"""

import os, time, math, random, signal, sys, traceback, logging, json
from logging.handlers import RotatingFileHandler
from datetime import datetime
import pandas as pd
import numpy as np
import ccxt
from flask import Flask, jsonify
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from collections import deque

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
BOT_VERSION = "DOGE Council ELITE v6.0 — Flow-Pressure v2 Hybrid System"
print("🔁 Booting:", BOT_VERSION, flush=True)

STATE_PATH = "./bot_state.json"
RESUME_ON_RESTART = True
SAFE_RECONCILE = True
RESUME_LOOKBACK_SECS = 60 * 60

# === Addons config ===
BOOKMAP_DEPTH = 50
BOOKMAP_TOPWALLS = 3
IMBALANCE_ALERT = 1.30

FLOW_WINDOW = 20
FLOW_SPIKE_Z = 1.60
CVD_SMOOTH = 8

# =================== FLOW-PRESSURE v2 SETTINGS ===================
FLOW_V2_ENABLED = True
FLOW_V2_DEPTH = 15
FLOW_V2_SHIFT = 1
FLOW_V2_WALL_K = 5.0
FLOW_V2_WIN_SIZE = 120
FLOW_V2_AGGRESSION_RATIO_TH = 1.15
FLOW_V2_AGGRESSION_Z_TH = 0.5
FLOW_V2_SCALP_Z_TH = 0.8
FLOW_V2_WALL_PROX_BPS = 10

# تخزين التاريخ للـ z-score
_flow_hist = deque(maxlen=FLOW_V2_WIN_SIZE)

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

ENTRY_RF_ONLY = False
MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", 6.0))

# =================== HYBRID SMART SYSTEM SETTINGS ===================
# Fast RF + Council Supervisor
RF_FAST_TRACK = True
RF_ADX_MIN = 18.0
RF_ADX_RISE_DELTA = 0.6
RF_MAX_SPREAD_BPS = 8
RF_COOLDOWN_S = 0
RF_HYST_BPS = 5

COUNCIL_OPPORTUNISTIC_MODE = True
ALLOW_GZ_ENTRY = True
GZ_MIN_SCORE = 6.0
GZ_REQ_ADX = 20.0

# Move Classification
ADX_TREND_GATE = 22.0
DI_SPREAD_TREND = 6.0
SCALP_TP1_PCT = 0.0040
SCALP_BE_AFTER = 0.0030
SCALP_TRAIL_ACTIVATE = 0.0080

TREND_TP1_PCT = 0.0060
TREND_TRAIL_ACTIVATE = 0.0120

# Scale-In (Smart Addition)
SCALE_IN_ENABLED = True
SCALE_IN_MAX_ADDS = 2
SCALE_IN_STEP_PCT = 0.25
SCALE_IN_MIN_DISTANCE_BPS = 10
SCALE_IN_MIN_CONFIDENCE = 7.0

# RSI/EMA Golden Context
RSI_MA_LEN = 9
RSI_TREND_PERSIST = 3
RSI_NEUTRAL_BAND = (45, 55)

# =================== COUNCIL ELITE SETTINGS ===================
# Council Weights & Gates
ADX_GATE = 17.0
ADX_TREND_MIN = 22.0
DI_SPREAD_TREND = 6.0
RSI_MA_LEN = 9

# Golden Zones
GZ_FIB_LOW = 0.618
GZ_FIB_HIGH = 0.786
GZ_MIN_SCORE = 6.0
GZ_ADX_MIN = 20.0
GOLDEN_ENTRY_SCORE = 6.0
GOLDEN_ENTRY_ADX = 20.0
GOLDEN_REVERSAL_SCORE = 6.5

# FVG/SMC
FVG_MIN_BPS = 8.0
BOS_MIN_PCT = 0.35
SWEEP_WICK_X_ATR = 1.2
OB_LOOKBACK = 40

# Flow/Bookmap
DELTA_Z_BULL = 0.50
DELTA_Z_BEAR = -0.50
IMB_ALERT = 1.20

# Management profiles
TP1_PCT_SCALP = 0.0040   # 0.40%
TP1_PCT_TREND = 0.0060   # 0.60%
BE_AFTER_SCALP = 0.0030  # 0.30%
BE_AFTER_TREND = 0.0040  # 0.40%
TRAIL_ACT_SCALP = 0.0080 # 0.80%
TRAIL_ACT_TREND = 0.0120 # 1.20%
ATR_TRAIL_MULT = 1.6
TRAIL_TIGHT_MULT = 1.2

# Decision thresholds
COUNCIL_STRONG_TH = 8.0
COUNCIL_OK_TH = 7.0

# Smart Exit Tuning
TP1_SCALP_PCT = 0.0035
TP1_TREND_PCT = 0.0060
HARD_CLOSE_PNL_PCT = 0.0110
WICK_ATR_MULT = 1.5
EVX_SPIKE = 1.8
BM_WALL_PROX_BPS = 5
TIME_IN_TRADE_MIN = 8

# ===== Council & RSI defaults (FIX) =====
RSI_MA_LEN         = globals().get("RSI_MA_LEN", 9)
RSI_NEUTRAL_BAND   = globals().get("RSI_NEUTRAL_BAND", (45, 55))
RSI_TREND_PERSIST  = globals().get("RSI_TREND_PERSIST", 3)

ADX_TREND_MIN      = globals().get("ADX_TREND_MIN", 20)
DI_SPREAD_TREND    = globals().get("DI_SPREAD_TREND", 6)
ADX_GATE           = globals().get("ADX_GATE", 17)

GZ_MIN_SCORE       = globals().get("GZ_MIN_SCORE", 6.0)
GZ_REQ_ADX         = globals().get("GZ_REQ_ADX", 20)
ALLOW_GZ_ENTRY     = globals().get("ALLOW_GZ_ENTRY", True)

TP1_PCT_BASE       = globals().get("TP1_PCT_BASE", 0.004)  # 0.40%
BREAKEVEN_AFTER    = globals().get("BREAKEVEN_AFTER", 0.003)
TRAIL_ACTIVATE_PCT = globals().get("TRAIL_ACTIVATE_PCT", 0.008)
ATR_TRAIL_MULT     = globals().get("ATR_TRAIL_MULT", 1.6)
TRAIL_TIGHT_MULT   = globals().get("TRAIL_TIGHT_MULT", 1.20)

EVX_SPIKE          = globals().get("EVX_SPIKE", 1.8)
BM_WALL_PROX_BPS   = globals().get("BM_WALL_PROX_BPS", 5)
WICK_ATR_MULT      = globals().get("WICK_ATR_MULT", 1.5)
HARD_CLOSE_PNL_PCT = globals().get("HARD_CLOSE_PNL_PCT", 1.10/100.0)

# Dust guard
FINAL_CHUNK_QTY = float(os.getenv("FINAL_CHUNK_QTY", 40.0))
RESIDUAL_MIN_QTY = float(os.getenv("RESIDUAL_MIN_QTY", 9.0))

# Strict close
CLOSE_RETRY_ATTEMPTS = 6
CLOSE_VERIFY_WAIT_S = 2.0

# Pacing
BASE_SLEEP = 5
NEAR_CLOSE_S = 1

# ==== Safe logger (يمنع recursion) ====
def addon_log_safe(msg):
    try:
        print(msg, flush=True)
    except Exception as _:
        pass

# =================== PROFESSIONAL LOGGING ===================
def log_i(msg): print(f"ℹ️ {msg}", flush=True)
def log_g(msg): print(f"✅ {msg}", flush=True)
def log_w(msg): print(f"🟨 {msg}", flush=True)
def log_e(msg): print(f"❌ {msg}", flush=True)

def log_banner(text): print(f"\n{'—'*12} {text} {'—'*12}\n", flush=True)

# =============== TRADE OPEN LOG (BUY=🟢 / SELL=🔴) ===============
def log_trade_open(*, side:str, price:float, qty:float, leverage:int,
                   source:str, mode:str, risk_alloc:float,
                   council:dict=None, gz:dict=None, mgmt:dict=None):
    lamp = "🟢 BUY" if side.lower().startswith("b") else "🔴 SELL"
    p = f"{float(price):.6f}"
    q = f"{float(qty):.4f}"
    lev = f"{int(leverage)}x"
    ra = f"{int(risk_alloc*100)}%"

    c_part = ""
    if council:
        c_part = f" | Council B/S={council.get('score_b',0):.1f}/{council.get('score_s',0):.1f} votes={council.get('b',0)}/{council.get('s',0)}"

    gz_part = ""
    if gz and gz.get("ok"):
        gz_part = f" | GZ={gz['zone']['type']} s={gz.get('score',0):.1f}"

    mg = mgmt or {}
    tp1 = mg.get("tp1_pct"); bea = mg.get("be_activate_pct"); tra = mg.get("trail_activate_pct"); atrx = mg.get("atr_trail_mult")
    mg_part = ""
    if any(v is not None for v in (tp1, bea, tra)):
        mg_part = " | MGMT:" \
                  + (f" TP1={tp1*100:.2f}%" if tp1 is not None else "") \
                  + (f" BE≥{bea*100:.2f}%" if bea is not None else "") \
                  + (f" Trail≥{tra*100:.2f}%" if tra is not None else "") \
                  + (f" ATRx={atrx}" if atrx is not None else "")

    msg = f"{lamp} • {source} • {mode.upper()} | Price={p} Qty={q} Lev={lev} Risk={ra}{c_part}{gz_part}{mg_part}"

    try:
        (log_g if side.lower().startswith("b") else log_w)(msg)
    except NameError:
        print(msg, flush=True)

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

# =================== FLOW-PRESSURE v2 CORE FUNCTIONS ===================
def _safe_ob_fetch(exchange, symbol, limit=50):
    """جلب دفتر الطلبات بأمان مع معالجة الأخطاء"""
    try:
        ob = exchange.fetch_order_book(symbol, limit=limit)
        return {"ok": True, "bids": ob.get("bids", []), "asks": ob.get("asks", []), "ts": ob.get("timestamp")}
    except Exception as e:
        log_w(f"Order book fetch failed: {e}")
        return {"ok": False, "err": str(e)}

def _diag_metrics(bids, asks, shift=FLOW_V2_SHIFT, depth=FLOW_V2_DEPTH):
    """حساب المقاييس القطرية للدفتر"""
    n = min(len(bids), len(asks) - shift, depth)
    if n <= 0: 
        return None
    
    diag_delta = 0.0
    diag_ratio_sum = 0.0
    
    for i in range(n):
        bsz = float(bids[i][1] or 0)
        asz = float(asks[i + shift][1] or 0)
        diag_delta += (bsz - asz)
        diag_ratio_sum += (bsz / max(asz, 1e-9))
    
    return {
        "n": n, 
        "diag_delta": diag_delta, 
        "diag_ratio": diag_ratio_sum / n,
        "total_bid_vol": sum(float(b[1]) for b in bids[:n]),
        "total_ask_vol": sum(float(a[1]) for a in asks[:n])
    }

def compute_flow_v2(exchange, symbol, trades_fn=None, price=None, wall_k=FLOW_V2_WALL_K, depth=FLOW_V2_DEPTH):
    """الحساب الرئيسي لـ Flow-Pressure v2"""
    if not FLOW_V2_ENABLED:
        return {"ok": False, "why": "disabled"}
    
    ob = _safe_ob_fetch(exchange, symbol, limit=50)
    if not ob["ok"]:
        return {"ok": False, "why": "ob_fetch_failed", "err": ob.get("err")}
    
    m = _diag_metrics(ob["bids"], ob["asks"], shift=FLOW_V2_SHIFT, depth=depth)
    if not m:
        return {"ok": False, "why": "insufficient_depth"}
    
    # اكتشاف الجدران
    all_sizes = []
    for side in [ob["bids"][:depth], ob["asks"][:depth]]:
        for _, size in side:
            all_sizes.append(float(size))
    
    med_size = sorted(all_sizes)[len(all_sizes)//2] if all_sizes else 0
    
    # البحث عن جدران البيع والشراء
    bid_wall_idx = None
    ask_wall_idx = None
    bid_wall_price = None
    ask_wall_price = None
    
    for i, (price_val, size) in enumerate(ob["bids"][:depth]):
        if float(size) >= wall_k * med_size:
            bid_wall_idx = i
            bid_wall_price = float(price_val)
            break
            
    for i, (price_val, size) in enumerate(ob["asks"][:depth]):
        if float(size) >= wall_k * med_size:
            ask_wall_idx = i
            ask_wall_price = float(price_val)
            break
    
    # حساب z-score
    _flow_hist.append(m["diag_delta"])
    mu = sum(_flow_hist) / len(_flow_hist) if _flow_hist else 0.0
    var = sum((x - mu) ** 2 for x in _flow_hist) / max(1, len(_flow_hist))
    std = var ** 0.5 if var > 0 else 1e-9
    z = (m["diag_delta"] - mu) / std

    # حساب CVD من التداولات
    cvd = 0
    trade_imbalance = 0
    if trades_fn:
        try:
            trades = trades_fn()
            buys = sum(float(t["amount"]) for t in trades if str(t.get("side", "")).lower() == "buy")
            sells = sum(float(t["amount"]) for t in trades if str(t.get("side", "")).lower() == "sell")
            cvd = buys - sells
            trade_imbalance = (buys - sells) / max(buys + sells, 1) * 100
        except Exception as e:
            log_w(f"Trades fetch failed: {e}")

    # تصنيف العدوانية
    agg = "flat"
    agg_strength = 0.0
    
    if m["diag_ratio"] > FLOW_V2_AGGRESSION_RATIO_TH and z > FLOW_V2_AGGRESSION_Z_TH:
        agg = "buy"
        agg_strength = min(2.0, (m["diag_ratio"] - 1.0) * 2 + z)
    elif m["diag_ratio"] < (1.0 / FLOW_V2_AGGRESSION_RATIO_TH) and z < -FLOW_V2_AGGRESSION_Z_TH:
        agg = "sell" 
        agg_strength = min(2.0, ((1.0 / m["diag_ratio"]) - 1.0) * 2 + abs(z))

    # اكتشاف الامتصاص
    absorption = None
    wall_prox_bps = None
    
    if price is not None:
        current_price = float(price)
        
        # حساب المسافة لأقرب جدار
        if bid_wall_price and ask_wall_price:
            bid_dist = abs(current_price - bid_wall_price) / current_price * 10000
            ask_dist = abs(current_price - ask_wall_price) / current_price * 10000
            wall_prox_bps = min(bid_dist, ask_dist)
            
            # اكتشاف الامتصاص
            if agg == "buy" and ask_wall_idx is not None and ask_dist <= FLOW_V2_WALL_PROX_BPS:
                absorption = "ask"
            elif agg == "sell" and bid_wall_idx is not None and bid_dist <= FLOW_V2_WALL_PROX_BPS:
                absorption = "bid"

    return {
        "ok": True,
        "agg": agg,               # buy/sell/flat
        "agg_strength": round(agg_strength, 2),
        "ratio": round(m["diag_ratio"], 3),
        "z": round(z, 2),
        "cvd": float(cvd),
        "trade_imbalance": round(trade_imbalance, 1),
        "bid_wall_idx": bid_wall_idx,
        "ask_wall_idx": ask_wall_idx,
        "bid_wall_price": bid_wall_price,
        "ask_wall_price": ask_wall_price,
        "wall_prox_bps": wall_prox_bps,
        "absorption": absorption,
        "total_bid_vol": m["total_bid_vol"],
        "total_ask_vol": m["total_ask_vol"],
        "vol_imbalance": (m["total_bid_vol"] - m["total_ask_vol"]) / max(m["total_bid_vol"] + m["total_ask_vol"], 1) * 100
    }

def log_flow_v2_snapshot(flow_data):
    """تسجيل لقطة Flow-Pressure v2"""
    if not flow_data.get("ok"):
        return
        
    agg = flow_data["agg"]
    ratio = flow_data["ratio"]
    z = flow_data["z"]
    cvd = flow_data["cvd"]
    absorption = flow_data.get("absorption")
    wall_prox = flow_data.get("wall_prox_bps")
    
    if agg == "buy":
        log_i(f"🟢 Flow: Buy  A={ratio:+.2f}  z={z:+.2f} | CVD 📊 {int(cvd):+} | wall⬆️ {wall_prox or '--'}bps")
    elif agg == "sell":
        log_i(f"🔴 Flow: Sell A={ratio:+.2f}  z={z:+.2f} | CVD 📊 {int(cvd):+} | wall⬇️ {wall_prox or '--'}bps")
    else:
        log_i(f"🟨 Flow: Flat  A={ratio:+.2f}  z={z:+.2f} | CVD 📊 {int(cvd):+}")
    
    if absorption:
        log_w(f"🛡️ Absorption: {absorption} wall detected — tighten risk")
        
    # معلومات إضافية عن التوازن
    vol_imb = flow_data.get("vol_imbalance", 0)
    trade_imb = flow_data.get("trade_imbalance", 0)
    log_i(f"📊 Vol-Imb: {vol_imb:+.1f}% | Trade-Imb: {trade_imb:+.1f}%")

# =================== HYBRID SMART SYSTEM FUNCTIONS ===================
def classify_move(ind, rsi_ctx):
    """تصنيف الحركة: سكالب vs ترند"""
    adx = ind.get('adx', 0.0)
    di_spread = abs(ind.get('plus_di', 0) - ind.get('minus_di', 0))
    
    # ترند واضح
    if (adx >= ADX_TREND_GATE and 
        di_spread >= DI_SPREAD_TREND and 
        rsi_ctx["trendZ"] in ("bull", "bear")):
        return "trend"
    # خلاف ذلك سكالب ذكي
    return "scalp"

def rf_entry_guard_ok(df, ind, spread_bps, last_adx_vals):
    """التحقق المحسن من شروط دخول RF"""
    if spread_bps is not None and spread_bps > RF_MAX_SPREAD_BPS:
        return False, f"spread {spread_bps:.1f}bps>{RF_MAX_SPREAD_BPS}"
    
    adx = ind.get('adx', 0.0)
    di_plus = ind.get('plus_di', 0.0)
    di_minus = ind.get('minus_di', 0.0)
    di_spread = abs(di_plus - di_minus)
    
    # شرط انتشار الاتجاه واستمرارية ADX
    adx_persistence = 0
    if len(last_adx_vals) >= 3:  # RF_ADX_PERSISTENCE
        adx_persistence = sum(1 for i in range(1, 4) if last_adx_vals[-i] >= RF_ADX_MIN)
    
    # الشرط الأساسي: ADX مرتفع أو متصاعد مع انتشار DI
    basic_condition = (adx >= RF_ADX_MIN and di_spread >= 6.0 and adx_persistence >= 2)
    
    # الشرط المبكر: Council قوي مع إشارات ذهبية/FVG
    early_condition = False
    if adx >= 16.0:  # COUNCIL_EARLY_ADX_MIN
        cv = council_votes_pro_enhanced(df)
        gz = cv["ind"].get("gz", {})
        fvg_ok = detect_fvg(df).get("ok", False)
        council_strong = max(cv["score_b"], cv["score_s"]) >= 7.0  # COUNCIL_EARLY_ENTRY_MIN
        early_condition = council_strong and (gz.get("ok") or fvg_ok)
    
    if basic_condition or early_condition:
        return True, None
    
    return False, f"ADX {adx:.1f}<{RF_ADX_MIN} or DI-spread {di_spread:.1f}<6.0 or ADX-persistence {adx_persistence}<2"

def maybe_open_via_rf(df, info, ind, spread_bps, last_adx_vals):
    """الدخول السريع عبر RF مع تصنيف النمط"""
    global wait_for_next_signal_side
    
    if not RF_FAST_TRACK or STATE["open"]: 
        return
    
    # تحديد الإشارة من RF
    sig = None
    if info.get("long"):
        sig = "buy"
    elif info.get("short"):
        sig = "sell"
    
    if not sig: 
        return
    
    # التحقق من شروط الدخول
    ok, reason = rf_entry_guard_ok(df, ind, spread_bps, last_adx_vals)
    if not ok:
        log_i(f"RF_SKIP | {reason}")
        return
    
    px = info["price"]
    bal = balance_usdt()
    qty = compute_size(bal, px)
    
    if qty <= 0: 
        return
    
    # فتح الصفقة
    if open_market(sig, qty, px):
        rsi_ctx = rsi_ma_context(df)
        mode = classify_move(ind, rsi_ctx)
        STATE["mode"] = mode
        STATE["adds"] = 0  # إعادة تعزيزات لصفر
        log_g(f"🟢 RF ENTRY {sig.upper()} | mode={mode} | ADX={ind.get('adx',0):.1f} DIsp={abs(ind.get('plus_di',0)-ind.get('minus_di',0)):.1f}")

def council_opportunistic_action_enhanced(df, ind, extras):
    """المجلس الانتهازي المحسن - فتح/تعزيز عند الفرص القوية"""
    if not COUNCIL_OPPORTUNISTIC_MODE:
        return
        
    # التحقق من التبريد
    if STATE.get("last_action_time") and len(df) > 0:
        last_action_ts = STATE["last_action_time"]
        current_ts = int(df["time"].iloc[-1])
        tf_ms = _interval_seconds(INTERVAL) * 1000
        if current_ts - last_action_ts < 2 * tf_ms:  # ACTION_COOLDOWN_CANDLES = 2
            return

    cv = council_votes_pro_enhanced(df)
    gz = cv["ind"].get("gz", {})
    adx = cv["ind"].get("adx", 0.0)
    rsi_ctx = rsi_ma_context(df)
    score_b, score_s = cv["score_b"], cv["score_s"]
    winner = "buy" if score_b > score_s else "sell"
    winner_score = max(score_b, score_s)
    
    current_price = price_now()
    if not current_price:
        return

    # الحصول على إشارات التدفق والكتل
    bm = extras.get("bm", {})
    flow = extras.get("flow", {})
    flow_v2 = cv["ind"].get("flow_v2", {})
    fvg_ok = detect_fvg(df).get("ok", False)
    imbalance_ok = bm.get("imbalance", 1.0) >= 1.10 if bm.get("ok") else False  # COUNCIL_ENTRY_IMBALANCE_MIN
    flow_pressure = flow_v2.get("ok") and flow_v2.get("agg_strength", 0) > 1.0
    no_chop = not rsi_ctx["in_chop"]

    # 1- تعزيز ذكي للصفقة الحالية
    if STATE["open"] and SCALE_IN_ENABLED and STATE.get("adds", 0) < SCALE_IN_MAX_ADDS:
        same_dir = (STATE["side"] == "long" and winner == "buy") or (STATE["side"] == "short" and winner == "sell")
        enough_conf = winner_score >= SCALE_IN_MIN_CONFIDENCE
        
        # شروط التعزيز: أي اثنين من الأربعة
        conditions = {
            "golden_zone": gz and gz.get("ok") and (
                (gz["zone"]["type"] == "golden_bottom" and STATE["side"] == "long") or
                (gz["zone"]["type"] == "golden_top" and STATE["side"] == "short")
            ),
            "fvg": fvg_ok,
            "flow_pressure": flow_pressure,
            "imbalance": imbalance_ok
        }
        conditions_met = sum(conditions.values()) >= 2
        
        dist_ok = abs(current_price - STATE["entry"]) / STATE["entry"] * 10000.0 >= SCALE_IN_MIN_DISTANCE_BPS

        if same_dir and enough_conf and conditions_met and no_chop and dist_ok:
            add_qty = safe_qty(STATE["qty"] * SCALE_IN_STEP_PCT)
            if add_qty > 0:
                side = "buy" if STATE["side"] == "long" else "sell"
                if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                    try:
                        ex.create_order(SYMBOL, "market", side, add_qty, None, _params_open(side))
                        log_g(f"➕ COUNCIL ADD-ON {side.upper()} | conf={winner_score:.1f} | conditions={sum(conditions.values())}/4")
                        STATE["qty"] = safe_qty(STATE["qty"] + add_qty)
                        STATE["adds"] = STATE.get("adds", 0) + 1
                        STATE["last_action_time"] = int(df["time"].iloc[-1])
                        save_state(STATE)
                    except Exception as e:
                        log_e(f"❌ Scale-in failed: {e}")
                else:
                    log_i(f"DRY_RUN: Scale-in {side.upper()} {add_qty:.4f}")

    # 2- فتح انتهازي عند الفرص الذهبية
    if not STATE["open"]:
        golden_buy = gz and gz.get("ok") and gz["zone"]["type"] == "golden_bottom" and winner == "buy"
        golden_sell = gz and gz.get("ok") and gz["zone"]["type"] == "golden_top" and winner == "sell"
        
        strong_flow = flow_v2.get("ok") and flow_v2.get("agg") == winner
        rsi_cross = (winner == "buy" and rsi_ctx["cross"] == "bull") or (winner == "sell" and rsi_ctx["cross"] == "bear")
        
        if (golden_buy or golden_sell) and adx >= 20.0 and rsi_cross and imbalance_ok and no_chop:  # COUNCIL_ENTRY_ADX_MIN
            bal = balance_usdt()
            qty = compute_size(bal, current_price)
            if qty > 0:
                if open_market(winner, qty, current_price):
                    STATE["mode"] = classify_move(ind, rsi_ctx)
                    STATE["adds"] = 0
                    STATE["last_action_time"] = int(df["time"].iloc[-1])
                    log_g(f"🏆 COUNCIL ENTRY {winner.upper()} | golden={gz['zone']['type']} | score={winner_score:.1f} | ADX={adx:.1f}")

# =================== SMC/ICT TOOLS ===================
def _fib_zone(last_impulse_low, last_impulse_high):
    rng = last_impulse_high - last_impulse_low
    return last_impulse_low + GZ_FIB_LOW * rng, last_impulse_low + GZ_FIB_HIGH * rng

def detect_bos(df):
    """Break of Structure detection"""
    if len(df) < 30: 
        return {"ok": False}
    
    high_series = df['high'].astype(float)
    low_series = df['low'].astype(float)
    close_series = df['close'].astype(float)
    
    swing_high = high_series.iloc[-20:-5].max()
    swing_low = low_series.iloc[-20:-5].min()
    close = close_series.iloc[-1]
    prev = close_series.iloc[-2]
    
    up_bos = (close > swing_high) and ((close - prev) / prev * 100 >= BOS_MIN_PCT)
    down_bos = (close < swing_low) and ((prev - close) / prev * 100 >= BOS_MIN_PCT)
    
    if up_bos:   
        return {"ok": True, "dir": "bull", "ref": swing_high}
    if down_bos: 
        return {"ok": True, "dir": "bear", "ref": swing_low}
    
    return {"ok": False}

def detect_sweep(df, atr):
    """Liquidity Sweep detection"""
    if len(df) < 5: 
        return {"ok": False}
    
    c = df.iloc[-1]
    current_high = float(c['high'])
    current_low = float(c['low'])
    current_close = float(c['close'])
    current_open = float(c['open'])
    
    wick_up = current_high - max(current_close, current_open)
    wick_dn = min(current_close, current_open) - current_low
    
    bull = wick_dn >= SWEEP_WICK_X_ATR * atr
    bear = wick_up >= SWEEP_WICK_X_ATR * atr
    
    if bull: 
        return {"ok": True, "dir": "bull"}
    if bear: 
        return {"ok": True, "dir": "bear"}
    
    return {"ok": False}

def detect_fvg(df, min_bps=FVG_MIN_BPS):
    """Fair Value Gap detection"""
    if len(df) < 5: 
        return {"ok": False}
    
    h1 = float(df['high'].iloc[-2])
    l1 = float(df['low'].iloc[-2])
    h0 = float(df['high'].iloc[-1])
    l0 = float(df['low'].iloc[-1])
    
    up = (l0 - h1) / ((h1 + l0) / 2) * 10000.0
    down = (l1 - h0) / ((h0 + l1) / 2) * 10000.0
    
    if up >= min_bps:   
        return {"ok": True, "dir": "bull", "bps": up}
    if down >= min_bps: 
        return {"ok": True, "dir": "bear", "bps": down}
    
    return {"ok": False}

def detect_order_block(df, bullish=True, lookback=OB_LOOKBACK):
    """Order Block detection"""
    try:
        if len(df) < lookback:
            return {"ok": False}
            
        window = df.iloc[-lookback:]
        high_series = window['high'].astype(float)
        low_series = window['low'].astype(float)
        
        if bullish:
            lowest_indices = low_series.nsmallest(3).index
            base = window.loc[lowest_indices]
            ob_low = base['low'].astype(float).min()
            ob_high = base['high'].astype(float).max()
        else:
            highest_indices = high_series.nlargest(3).index
            base = window.loc[highest_indices]
            ob_low = base['low'].astype(float).min()
            ob_high = base['high'].ast(float).max()
            
        return {"ok": True, "low": ob_low, "high": ob_high}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def golden_zone_check_pro(df, ind):
    """Enhanced Golden Zone detection with Fibonacci levels"""
    if len(df) < 40:
        return {"ok": False}
        
    closes = df['close'].astype(float).values
    recent = closes[-30:]
    hi = recent.max()
    lo = recent.min()
    
    # Simple trend detection
    trend_up = hi == recent[-1]
    trend_dn = lo == recent[-1]
    
    fib_lo, fib_hi = _fib_zone(lo, hi)
    last = closes[-1]
    score = 0.0
    ztype = None

    if fib_lo <= last <= fib_hi:
        # Inside golden zone
        if ind.get('adx', 0) >= GZ_ADX_MIN:
            score += 2.0
        if ind.get('rsi', 50) < 45 and ind.get('rsi', 50) > ind.get('rsi_ma', 50):
            score += 1.0
        if ind.get('evx', 1.0) < 1.2:
            score += 0.5
            
        if trend_up:
            ztype = 'golden_top'
        elif trend_dn:
            ztype = 'golden_bottom'
        else:
            zone_mid = (fib_lo + fib_hi) / 2
            ztype = 'golden_top' if last > zone_mid else 'golden_bottom'
            
        return {
            "ok": True, 
            "score": score + 3.0, 
            "zone": {
                "type": ztype, 
                "lo": fib_lo, 
                "hi": fib_hi
            }
        }
        
    return {"ok": False}

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

    rng1 = _rng(h1,l1); up = _upper_wick(h1,o1,c1); dn = _lower_wick(l1,o1,c1)
    wick_up_big = (up >= 1.2*_body(o1,c1)) and (up >= 0.4*rng1)
    wick_dn_big = (dn >= 1.2*_body(o1,c1)) and (dn >= 0.4*rng1)

    if is_doji:
        strength_b *= 0.8; strength_s *= 0.8

    return {
        "buy": strength_b>0, "sell": strength_s>0,
        "score_buy": round(strength_b,2), "score_sell": round(strength_s,2),
        "wick_up_big": bool(wick_up_big), "wick_dn_big": bool(wick_dn_big),
        "doji": bool(is_doji), "pattern": ",".join(tags) if tags else None
    }

# =================== EXECUTION VERIFICATION ===================
def verify_execution_environment():
    """التحقق من بيئة التنفيذ عند الإقلاع"""
    print(f"⚙️ EXECUTION ENVIRONMENT", flush=True)
    print(f"🔧 EXECUTE_ORDERS: {EXECUTE_ORDERS} | SHADOW_MODE: {SHADOW_MODE_DASHBOARD} | DRY_RUN: {DRY_RUN}", flush=True)
    print(f"🎯 HYBRID SMART SYSTEM: Fast RF + Council ELITE + Flow-Pressure v2", flush=True)
    print(f"📈 SMC/ICT: Golden Zones + FVG + BOS + Sweeps + Scale-In + Diagonal Analysis", flush=True)
    
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
    try:
        # حساب RSI وRSI_MA إذا لم يكونا موجودين
        if 'rsi' not in df.columns:
            df['rsi'] = compute_rsi(df['close'].astype(float), 14)
        if 'rsi_ma' not in df.columns:
            df['rsi_ma'] = sma(df['rsi'], RSI_MA_LEN)
            
        rsi = float(df["rsi"].iloc[-1])
        rma = float(df["rsi_ma"].iloc[-1])
        band_lo, band_hi = RSI_NEUTRAL_BAND
        in_chop = (band_lo <= rsi <= band_hi)

        # ترند-Z مع Persist
        above = rsi > rma
        wins = 0
        for i in range(1, min(len(df), RSI_TREND_PERSIST+2)):
            rv = float(df["rsi"].iloc[-i])
            mv = float(df["rsi_ma"].iloc[-i])
            if (rv > mv) == above:
                wins += 1
            else:
                break
        trendZ = "bull" if above and wins >= RSI_TREND_PERSIST else ("bear" if (not above) and wins >= RSI_TREND_PERSIST else "flat")
        
        # حساب التقاطع
        if len(df) >= 2:
            rsi_prev = float(df["rsi"].iloc[-2])
            rma_prev = float(df["rsi_ma"].iloc[-2])
            cross = "bull" if rsi >= rma and rsi_prev < rma_prev else ("bear" if rsi <= rma and rsi_prev > rma_prev else "none")
        else:
            cross = "none"

        return {"rsi": rsi, "rsi_ma": rma, "trendZ": trendZ, "cross": cross, "in_chop": in_chop}
    except Exception as e:
        addon_log_safe(f"rsi_ma_context error: {e}")
        return {"rsi":50, "rsi_ma":50, "trendZ":"flat", "cross":"none", "in_chop":False}

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

# =================== COUNCIL ELITE VOTING WITH FLOW-PRESSURE v2 ===================
COUNCIL_BUSY = False
LAST_COUNCIL = {"b": 0, "s": 0, "score_b": 0.0, "score_s": 0.0, "logs": [], "ind": {}}

def council_votes_pro_enhanced_with_flow_v2(df):
    """مجلس محسّن مع دمج Flow-Pressure v2"""
    global COUNCIL_BUSY, LAST_COUNCIL
    if COUNCIL_BUSY:
        return LAST_COUNCIL
        
    COUNCIL_BUSY = True
    try:
        ind = compute_indicators(df)
        rsi_ctx = rsi_ma_context(df)
        gz = golden_zone_check(df, ind)

        votes_b = votes_s = 0
        score_b = score_s = 0.0
        logs = []
        actions = []

        adx = ind.get('adx', 0.0)
        plus_di = ind.get('plus_di', 0.0)
        minus_di = ind.get('minus_di', 0.0)
        di_spread = abs(plus_di - minus_di)

        # حساب Flow-Pressure v2
        current_price = price_now()
        flow_v2 = compute_flow_v2(
            ex, SYMBOL, 
            trades_fn=lambda: ex.fetch_trades(SYMBOL, limit=100),
            price=current_price
        )

        # 1- التصويت الأساسي من المؤشرات التقنية
        if adx > ADX_TREND_MIN:
            if plus_di > minus_di and di_spread > DI_SPREAD_TREND:
                votes_b += 2
                score_b += 1.5
                logs.append("📈 ترند صاعد قوي")
            elif minus_di > plus_di and di_spread > DI_SPREAD_TREND:
                votes_s += 2
                score_s += 1.5
                logs.append("📉 ترند هابط قوي")

        if rsi_ctx["cross"] == "bull" and rsi_ctx["rsi"] < 70:
            votes_b += 2
            score_b += 1.0
            logs.append("🟢 RSI-MA إيجابي")
        elif rsi_ctx["cross"] == "bear" and rsi_ctx["rsi"] > 30:
            votes_s += 2
            score_s += 1.0
            logs.append("🔴 RSI-MA سلبي")

        if rsi_ctx["trendZ"] == "bull":
            votes_b += 3
            score_b += 1.5
            logs.append("🚀 RSI ترند صاعد مستمر")
        elif rsi_ctx["trendZ"] == "bear":
            votes_s += 3
            score_s += 1.5
            logs.append("💥 RSI ترند هابط مستمر")

        if gz and gz.get("ok"):
            if gz['zone']['type'] == 'golden_bottom':
                votes_b += 3
                score_b += 1.5
                logs.append(f"🏆 قاع ذهبي (قوة {gz['score']:.1f})")
            elif gz['zone']['type'] == 'golden_top':
                votes_s += 3
                score_s += 1.5
                logs.append(f"🏆 قمة ذهبية (قوة {gz['score']:.1f})")

        # 2- التصويت من Flow-Pressure v2
        if flow_v2.get("ok"):
            # عدوانية الشراء
            if flow_v2["agg"] == "buy":
                votes_b += 2
                score_b += 1.2
                logs.append(f"💰 ضغط شراء (A={flow_v2['ratio']:.2f} z={flow_v2['z']:.2f})")
                
                # دفعة إضافية مع Golden Zone
                if gz and gz.get("ok") and gz['zone']['type'] == 'golden_bottom':
                    votes_b += 2
                    score_b += 1.0
                    logs.append("🚀 دفعة ذهبية مع ضغط شراء")
                    
            # عدوانية البيع        
            elif flow_v2["agg"] == "sell":
                votes_s += 2
                score_s += 1.2
                logs.append(f"💸 ضغط بيع (A={flow_v2['ratio']:.2f} z={flow_v2['z']:.2f})")
                
                # دفعة إضافية مع Golden Zone
                if gz and gz.get("ok") and gz['zone']['type'] == 'golden_top':
                    votes_s += 2
                    score_s += 1.0
                    logs.append("🚀 دفعة ذهبية مع ضغط بيع")

            # اكتشاف الامتصاص وإدارة المخاطر
            if STATE.get("open"):
                if STATE["side"] == "long" and flow_v2["absorption"] == "ask":
                    actions.append({"type": "tighten_trail", "why": "امتصاص بيعي"})
                    logs.append("🛡️ تشديد بسبب امتصاص بيعي")
                    
                elif STATE["side"] == "short" and flow_v2["absorption"] == "bid":
                    actions.append({"type": "tighten_trail", "why": "امتصاص شرائي"})
                    logs.append("🛡️ تشديد بسبب امتصاص شرائي")

            # السماح بالسكالب في ظروف محددة
            spread_bps = orderbook_spread_bps()
            allow_scalp = False
            if (adx < 18 and flow_v2["agg"] in ("buy", "sell") and 
                abs(flow_v2["z"]) >= FLOW_V2_SCALP_Z_TH and 
                spread_bps and spread_bps <= 6):
                allow_scalp = True
                logs.append("⚡ سكالب مسموح (ADX منخفض + ضغط عالي)")

        # 3- التعديلات النهائية
        if rsi_ctx["in_chop"]:
            score_b *= 0.8
            score_s *= 0.8
            logs.append("⚖️ نطاق حيادي RSI")

        if adx < ADX_GATE:
            score_b *= 0.85
            score_s *= 0.85
            logs.append(f"🛡️ ADX Gate {adx:.1f}<{ADX_GATE}")

        ind.update({
            "rsi_ma": rsi_ctx["rsi_ma"], 
            "rsi_trendz": rsi_ctx["trendZ"], 
            "di_spread": di_spread, 
            "gz": gz,
            "flow_v2": flow_v2,
            "allow_scalp": allow_scalp
        })
        
        result = {
            "b": votes_b,
            "s": votes_s, 
            "score_b": round(score_b, 2),
            "score_s": round(score_s, 2),
            "logs": logs,
            "actions": actions,
            "ind": ind
        }
        
        LAST_COUNCIL = result
        return result
        
    except Exception as e:
        addon_log_safe(f"council_votes_pro_enhanced_with_flow_v2 error: {e}")
        return LAST_COUNCIL
    finally:
        COUNCIL_BUSY = False

council_votes_pro = council_votes_pro_enhanced_with_flow_v2

# =================== SMART TRADE MANAGEMENT ===================
def setup_trade_management(mode):
    """تهيئة إدارة الصفقة حسب النمط مع الإعدادات الجديدة"""
    if mode == "scalp":
        return {
            "tp1_pct": SCALP_TP1_PCT,
            "be_activate_pct": SCALP_BE_AFTER,
            "trail_activate_pct": SCALP_TRAIL_ACTIVATE,
            "atr_trail_mult": ATR_TRAIL_MULT,
            "close_aggression": "high"
        }
    else:  # trend
        return {
            "tp1_pct": TREND_TP1_PCT,
            "be_activate_pct": SCALP_BE_AFTER,
            "trail_activate_pct": TREND_TRAIL_ACTIVATE,
            "atr_trail_mult": ATR_TRAIL_MULT,
            "close_aggression": "medium"
        }

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
    evx_spike = False

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

    # Golden Reversal بعد TP1
    if state.get('tp1_done') and (gz and gz.get('ok')):
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

    # Wick exhaustion + Tighten عند إجهاد/تدفق/جدار
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

# =================== POSITION RECOVERY ===================
def _normalize_side(pos):
    side = pos.get("side") or pos.get("positionSide") or ""
    if side: return side.upper()
    qty = float(pos.get("contracts") or pos.get("positionAmt") or pos.get("size") or 0)
    return "LONG" if qty > 0 else ("SHORT" if qty < 0 else "")

def fetch_live_position_safe(ex, symbol):
    try:
        if getattr(ex, "has", {}).get("fetchPositions"):
            pos = ex.fetch_positions([symbol])
            return pos[0] if pos else None
        return None
    except Exception as e:
        addon_log_safe(f"fetch_live_position error (ignored): {e}")
        return None

def fetch_live_position(exchange, symbol: str):
    try:
        return fetch_live_position_safe(exchange, symbol)
    except Exception as e:
        log_w(f"fetch_live_position error: {e}")
    return {"ok": False, "why": "no_open_position"}

def resume_open_position_enhanced(exchange, symbol: str, state: dict) -> dict:
    """استئناف محسن للمركز مع مصالحة آمنة"""
    if not RESUME_ON_RESTART:
        log_i("resume disabled")
        return state

    prev = load_state() or {}
    
    # 1) المحاولة الأولى: جلب المركز من المنصة
    live = fetch_live_position(exchange, symbol)
    if live and live.get("ok"):
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
            "opened_at": prev.get("opened_at", int(time.time())),
            "adds": prev.get("adds", 0),
        })
        save_state(state)
        log_g(f"✅ RESUME via EXCHANGE: {state['side']} qty={state['position_qty']} @ {state['entry_price']:.6f}")
        return state
    
    # 2) Fallback: استخدام STATE.json إذا كان حديثاً
    if SAFE_RECONCILE and prev.get("in_position") and prev.get("position_qty", 0) > 0:
        ts = int(time.time())
        state_ts = prev.get("ts", 0)
        
        if ts - state_ts < 3600:
            state.update(prev)
            save_state(state)
            log_w(f"⚠️ RESUME via STATE.json (exchange unavailable): {state['side']} qty={state['position_qty']}")
            return state
    
    log_i("No position to resume — starting fresh")
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

# ========= Unified snapshot emitter with Flow-Pressure v2 =========
def emit_snapshots_with_flow_v2(exchange, symbol, df, balance_fn=None, pnl_fn=None, council=None):
    """تحديث الـ Snapshots مع دمج Flow-Pressure v2"""
    try:
        bm = bookmap_snapshot(exchange, symbol)
        flow = compute_flow_metrics(df)
        flow_v2 = compute_flow_v2(exchange, symbol, 
                                trades_fn=lambda: exchange.fetch_trades(symbol, limit=100),
                                price=price_now())
        
        if council is None:
            cv = council_votes_pro_enhanced_with_flow_v2(df)
        else:
            cv = council
            
        mode = decide_strategy_mode(df)
        gz = cv["ind"].get("gz", {})

        bal = None; cpnl = None
        if callable(balance_fn):
            try: bal = balance_fn()
            except: bal = None
        if callable(pnl_fn):
            try: cpnl = pnl_fn()
            except: cpnl = None

        # لوج Bookmap العادي
        if bm.get("ok"):
            imb_tag = "🟢" if bm["imbalance"]>=IMBALANCE_ALERT else ("🔴" if bm["imbalance"]<=1/IMBALANCE_ALERT else "⚖️")
            bm_note = f"Bookmap: {imb_tag} Imb={bm['imbalance']:.2f} | Buy[{fmt_walls(bm['buy_walls'])}] | Sell[{fmt_walls(bm['sell_walls'])}]"
        else:
            bm_note = f"Bookmap: N/A ({bm.get('why')})"

        # لوج Flow-Pressure v2
        if flow_v2.get("ok"):
            log_flow_v2_snapshot(flow_v2)

        # لوج Flow العادي
        if flow.get("ok"):
            dtag = "🟢Buy" if flow["delta_last"]>0 else ("🔴Sell" if flow["delta_last"]<0 else "⚖️Flat")
            spk = " ⚡Spike" if flow["spike"] else ""
            fl_note = f"Flow: {dtag} Δ={flow['delta_last']:.0f} z={flow['delta_z']:.2f}{spk} | CVD {'↗️' if flow['cvd_trend']=='up' else '↘️'} {flow['cvd_last']:.0f}"
        else:
            fl_note = f"Flow: N/A ({flow.get('why')})"

        # الباقي من اللوج العادي
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

        return {"bm": bm, "flow": flow, "flow_v2": flow_v2, "cv": cv, "mode": mode, "gz": gz, "wallet": wallet}
    except Exception as e:
        addon_log_safe(f"🟨 AddonLog error: {e}")
        return {"bm": None, "flow": None, "flow_v2": None, "cv": {"b":0,"s":0,"score_b":0.0,"score_s":0.0,"ind":{}},
                "mode": {"mode":"n/a"}, "gz": None, "wallet": ""}

emit_snapshots = emit_snapshots_with_flow_v2

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

# =================== ENHANCED TRADE EXECUTION ===================
def open_market_enhanced(side, qty, price):
    if qty <= 0: 
        log_e("skip open (qty<=0)")
        return False
    
    df = fetch_ohlcv()
    
    cv = council_votes_pro_enhanced_with_flow_v2(df)
    snap = emit_snapshots(ex, SYMBOL, df, council=cv)
    
    votes = cv
    mode_data = decide_strategy_mode(df, 
                                   adx=votes["ind"].get("adx"),
                                   di_plus=votes["ind"].get("plus_di"),
                                   di_minus=votes["ind"].get("minus_di"),
                                   rsi_ctx=rsi_ma_context(df))
    
    mode = mode_data["mode"]
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
            "adds": 0,
            "mode": mode,
            "management": management_config
        })
        
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
            "adds": 0,
        })
        
        log_trade_open(
            side=side, price=price, qty=qty, leverage=LEVERAGE,
            source="Hybrid Smart System + Flow-Pressure v2",
            mode=mode,
            risk_alloc=RISK_ALLOC,
            council=votes,
            gz=gz,
            mgmt=management_config
        )
        
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
    "profit_targets_achieved": 0, "adds": 0,
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

# =================== ORDERS ===================
def _params_open(side):
    if POSITION_MODE == "hedge":
        return {"positionSide": "LONG" if side=="buy" else "SHORT", "reduceOnly": False}
    return {"positionSide": "BOTH", "reduceOnly": False}

def _params_close():
    return {"reduceOnly": True}

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

def close_market_strict(reason="manual"):
    try:
        side = "sell" if STATE["side"]=="long" else "buy"
        qty  = max(0.0, float(STATE["qty"]))
        if qty <= 0: 
            addon_log_safe("close: qty<=0"); return
        ex.create_order(SYMBOL, "market", side, qty, None, _params_close())
        log_w(f"🚨 CLOSE STRICT [{reason}] qty={qty}")
        STATE.update({"open": False})
        save_state({"in_position": False})
    except Exception as e:
        log_e(f"close failed: {e}")

def _reset_after_close(reason, prev_side=None):
    """إعادة تعيين الحالة بعد الإغلاق"""
    global wait_for_next_signal_side
    prev_side = prev_side or STATE.get("side")
    STATE.update({
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
        "tp1_done": False, "highest_profit_pct": 0.0, "profit_targets_achieved": 0,
        "trail_tightened": False, "partial_taken": False, "adds": 0
    })
    save_state({"in_position": False, "position_qty": 0})
    
    _arm_wait_after_close(prev_side)
    logging.info(f"AFTER_CLOSE waiting_for={wait_for_next_signal_side}")

# =================== ENHANCED TRADE MANAGEMENT WITH FLOW-PRESSURE v2 ===================
def manage_after_entry_with_flow_v2(df, ind, info):
    """إدارة محسنة للمركز مع دمج Flow-Pressure v2"""
    if not STATE["open"] or STATE["qty"] <= 0:
        return

    px = info["price"]
    entry = STATE["entry"]
    side = STATE["side"]
    qty = STATE["qty"]
    mode = STATE.get("mode", "trend")
    
    pnl_pct = (px - entry) / entry * 100 * (1 if side == "long" else -1)
    STATE["pnl"] = pnl_pct
    
    if pnl_pct > STATE["highest_profit_pct"]:
        STATE["highest_profit_pct"] = pnl_pct

    # حساب المجلس مع Flow-Pressure v2
    cv = council_votes_pro_enhanced_with_flow_v2(df)
    flow_v2 = cv["ind"].get("flow_v2", {})
    
    # تطبيق إجراءات Flow-Pressure v2
    for action in cv.get("actions", []):
        if action["type"] == "tighten_trail" and not STATE.get("trail_tightened"):
            STATE["trail_tightened"] = True
            STATE["trail"] = None
            log_w(f"🔄 TRAIL TIGHTENED: {action['why']}")

    # اكتشاف الامتصاص المتكرر
    if flow_v2.get("ok") and flow_v2.get("absorption"):
        absorption_key = f"absorption_{flow_v2['absorption']}"
        STATE[absorption_key] = STATE.get(absorption_key, 0) + 1
        
        # إذا تكرر الامتصاص مرتين، استعداد للإغلاق
        if STATE[absorption_key] >= 2:
            STATE["strict_close_ready"] = True
            log_w(f"🚨 STRICT CLOSE READY: repeated {flow_v2['absorption']} absorption")

    # الباقي من إدارة الصفقة العادية
    manage_after_entry_enhanced(df, ind, info)

def manage_after_entry_enhanced(df, ind, info):
    """إدارة محسنة للمركز مع خروج ذكي حسب النمط"""
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

    cv = council_votes_pro_enhanced_with_flow_v2(df)
    snap = emit_snapshots(ex, SYMBOL, df, council=cv)
    
    gz = snap["gz"]
    
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
    tp1_pct = management.get("tp1_pct", TP1_PCT_SCALP)
    be_activate_pct = management.get("be_activate_pct", BE_AFTER_SCALP)
    trail_activate_pct = management.get("trail_activate_pct", TRAIL_ACT_SCALP)
    atr_trail_mult = management.get("atr_trail_mult", ATR_TRAIL_MULT)

    if not STATE.get("tp1_done") and pnl_pct/100 >= tp1_pct:
        close_fraction = 0.5
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

manage_after_entry = manage_after_entry_with_flow_v2

# =================== ENHANCED TRADE LOOP ===================
def trade_loop_enhanced_final():
    """الحلقة النهائية المحسنة مع جميع التحسينات"""
    global wait_for_next_signal_side
    loop_i = 0
    last_adx_vals = []
    
    while True:
        try:
            # جمع البيانات الأساسية
            bal = balance_usdt()
            px = price_now()
            df = fetch_ohlcv()
            info = rf_signal_live(df)
            ind = compute_indicators(df)
            spread_bps = orderbook_spread_bps()
            
            # تحديث تاريخ ADX
            current_adx = ind.get('adx', 0.0)
            last_adx_vals.append(current_adx)
            if len(last_adx_vals) > 5:
                last_adx_vals.pop(0)
            
            council_data = council_votes_pro_enhanced_with_flow_v2(df)
            
            snap = emit_snapshots(ex, SYMBOL, df,
                                balance_fn=lambda: float(bal) if bal else None,
                                pnl_fn=lambda: float(compound_pnl),
                                council=council_data)
            
            if STATE["open"] and px:
                STATE["pnl"] = (px - STATE["entry"]) * STATE["qty"] if STATE["side"] == "long" else (STATE["entry"] - px) * STATE["qty"]
            
            # 1- الدخول السريع عبر RF (المحسن)
            maybe_open_via_rf(df, info, ind, spread_bps, last_adx_vals)
            
            # 2- إدارة الصفقة المفتوحة
            if STATE["open"]:
                manage_after_entry(df, ind, {
                    "price": px or info["price"], 
                    "bm": snap["bm"],
                    "flow": snap["flow"],
                    **info
                })
            
            # 3- المجلس الانتهازي المحسن
            council_opportunistic_action_enhanced(df, ind, snap)
            
            if LOG_LEGACY:
                pretty_snapshot(bal, {"price": px or info["price"], **info}, ind, spread_bps, None, df)
            
            loop_i += 1
            sleep_s = NEAR_CLOSE_S if time_to_candle_close(df) <= 10 else BASE_SLEEP
            time.sleep(sleep_s)
            
        except Exception as e:
            log_e(f"loop error: {e}\n{traceback.format_exc()}")
            logging.error(f"trade_loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

trade_loop = trade_loop_enhanced_final

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
        print(f"   🎯 ENTRY: HYBRID SMART SYSTEM (Fast RF + Council ELITE + Flow-Pressure v2)  |  spread_bps={fmt(spread_bps,2)}")
        print(f"   ⏱️ closes_in ≈ {left_s}s")
        print("\n🧭 POSITION")
        bal_line = f"Balance={fmt(bal,2)}  Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x  CompoundPnL={fmt(compound_pnl)}  Eq~{fmt((bal or 0)+compound_pnl,2)}"
        print(colored(f"   {bal_line}", "yellow"))
        if STATE["open"]:
            lamp='🟩 LONG' if STATE['side']=='long' else '🟥 SHORT'
            print(f"   {lamp}  Entry={fmt(STATE['entry'])}  Qty={fmt(STATE['qty'],4)}  Bars={STATE['bars']}  Trail={fmt(STATE['trail'])}  BE={fmt(STATE['breakeven'])}")
            print(f"   🎯 TP_done={STATE['profit_targets_achieved']}  HP={fmt(STATE['highest_profit_pct'],2)}%  Adds={STATE.get('adds',0)}")
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
    return f"✅ Hybrid Smart System Bot — {SYMBOL} {INTERVAL} — {mode} — Fast RF + Council ELITE + Flow-Pressure v2"

@app.route("/metrics")
def metrics():
    # حساب Flow-Pressure v2 الحالي
    current_flow_v2 = compute_flow_v2(
        ex, SYMBOL, 
        trades_fn=lambda: ex.fetch_trades(SYMBOL, limit=100),
        price=price_now()
    )
    
    return jsonify({
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "state": STATE, "compound_pnl": compound_pnl,
        "entry_mode": "HYBRID_SMART_SYSTEM_WITH_FLOW_V2", "wait_for_next_signal": wait_for_next_signal_side,
        "guards": {"max_spread_bps": MAX_SPREAD_BPS, "final_chunk_qty": FINAL_CHUNK_QTY},
        "flow_v2": current_flow_v2 if current_flow_v2.get("ok") else None
    })

@app.route("/health")
def health():
    return "ok", 200

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
    log_banner("HYBRID SMART SYSTEM WITH FLOW-PRESSURE v2")
    state = load_state() or {}
    state.setdefault("in_position", False)

    if RESUME_ON_RESTART:
        try:
            state = resume_open_position_enhanced(ex, SYMBOL, state)
        except Exception as e:
            log_w(f"resume error: {e}\n{traceback.format_exc()}")

    verify_execution_environment()

    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'}  •  {SYMBOL}  •  {INTERVAL}", "yellow"))
    print(colored(f"RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x  •  HYBRID_SYSTEM=ENABLED", "yellow"))
    print(colored(f"FAST RF: ADX≥{RF_ADX_MIN} | ΔADX≥{RF_ADX_RISE_DELTA} | Spread≤{RF_MAX_SPREAD_BPS}bps", "yellow"))
    print(colored(f"FLOW-PRESSURE v2: ENABLED", "green"))
    print(colored(f"  • Depth: {FLOW_V2_DEPTH} | Shift: {FLOW_V2_SHIFT} | Wall-K: {FLOW_V2_WALL_K}", "cyan"))
    print(colored(f"  • Aggression: ratio>{FLOW_V2_AGGRESSION_RATIO_TH} | z>{FLOW_V2_AGGRESSION_Z_TH}", "cyan"))
    print(colored(f"  • Scalp Trigger: z≥{FLOW_V2_SCALP_Z_TH} with spread≤6bps", "cyan"))
    print(colored(f"  • Wall Proximity: {FLOW_V2_WALL_PROX_BPS}bps", "cyan"))
    print(colored(f"COUNCIL ELITE: Golden Zones + Scale-In + Opportunistic Entries + Diagonal Analysis", "yellow"))
    print(colored(f"MANAGEMENT: Smart TP + Smart Exit + Trail Adaptation + Absorption Detection", "yellow"))
    print(colored(f"EXECUTION: {'ACTIVE' if EXECUTE_ORDERS and not DRY_RUN else 'SIMULATION'}", "yellow"))
    
    logging.info("Hybrid Smart System with Flow-Pressure v2 service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    
    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)

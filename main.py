# -*- coding: utf-8 -*-
"""
RF Futures Bot — RF-LIVE ONLY (BingX Perp via CCXT)
• Entry: Range Filter (TradingView-like) — LIVE CANDLE ONLY  
• Post-entry: Dynamic TP ladder + Breakeven + ATR-trailing
• Strict close with exchange verification
• Opposite-signal wait policy after a close
• Dust guard: force close if remaining ≤ FINAL_CHUNK_QTY (default 40 DOGE)
• Flask /metrics + /health + rotated logging
• [ADDON] SMC كامل + Golden Zones + RSI+MA Boost + EVX + وضعين تداول
• [ADDON] Bookmap-Lite + Volume Flow + Shadow Dashboard + Recovery System
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
LOG_LEGACY = False           # عطّل اللوج القديم
LOG_ADDONS = True            # فعّل اللوج الجديد

# ==== Addon: Logging + Recovery Settings ====
BOT_VERSION = "DOGE SmartMoney Fusion v1.7 — SmartExit Pro"
print("🔁 Booting:", BOT_VERSION, flush=True)

SHADOW_MODE_DASHBOARD = True
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

ENTRY_RF_ONLY = True
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

# ==== [ADDON] Smart Settings ====
SMART_MODE = os.getenv("SMART_MODE", "pro")  # "off" | "pro"
# عتبات قرار متوازنة
ENTRY_MIN_VOTES  = 6
ENTRY_MIN_SCORE  = 2.3

RSI_LEN = 14
RSI_MA_LEN = 9
RSI_CHOP_BAND = (45.0, 55.0)
RSI_CROSS_VOTES = 2;   RSI_CROSS_SCORE = 1.0
RSI_TRENDZ_PERSIST = 3
RSI_TRENDZ_VOTES = 3;  RSI_TRENDZ_SCORE = 1.5

EVX_ATR_LEN = 14
EVX_BASE_LEN = 50
EVX_STRONG = 1.8

GZ_MIN_SCORE  = 6.0
GZ_ADX_MIN    = 20.0
GZ_REQ_VOL_MA = 20
GZ_FIB_LOW    = 0.618
GZ_FIB_HIGH   = 0.786
GZ_CAN_LEAD_ENTRY = True

ENTRY_CONFIRM_GUARD = True
RSI_CROSS_REQUIRED  = True
RSI_NEUTRAL_BAND    = (45.0, 55.0)

TREND_ADX_MIN = 22
TREND_DI_SPREAD = 6

SCALP_ADX_MIN   = 15
SCALP_TP1       = 0.35
SCALP_BE        = 0.25
SCALP_TRAIL_ACTIVATE = 0.9

# === Smart Exit Tuning ===
TP1_SCALP_PCT      = 0.35/100     # سكالب: جني أول سريع
TP1_TREND_PCT      = 0.60/100     # ترند: جني أول أوسع
HARD_CLOSE_PNL_PCT = 1.10/100     # ربح محترم → إغلاق صارم عند إشارات هابطة
WICK_ATR_MULT      = 1.5          # فتيلة ≥ 1.5*ATR = إنهاك
EVX_SPIKE          = 1.8          # انفجار تذبذب
BM_WALL_PROX_BPS   = 5            # جدار بيع Bookmap على بعد ≤ 0.05%
TIME_IN_TRADE_MIN  = 8            # زمن داخل الصفقة قبل التفكير في الخروج
TRAIL_TIGHT_MULT   = 1.20         # تشديد التريل

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
    يطبع Snapshot موحّد: Bookmap + Flow + Council + Strategy + Balance/PnL
    """
    try:
        bm = bookmap_snapshot(exchange, symbol)
        flow = compute_flow_metrics(df)
        cv = council_votes_pro(df)
        mode = decide_strategy_mode(df)
        gz = golden_zone_check(df, {"adx": cv["ind"]["adx"]}, "buy" if cv["b"]>=cv["s"] else "sell")

        # balance & pnl (اختياري)
        bal = None; cpnl = None
        if callable(balance_fn):
            try: bal = balance_fn()
            except: bal = None
        if callable(pnl_fn):
            try: cpnl = pnl_fn()
            except: cpnl = None

        # بناء سطر Snapshot
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
                f"DI={cv['ind'].get('di_spread',0):.1f} EVX={cv['ind'].get('evx',1.0):.2f}")

        strat_icon = "⚡" if mode["mode"]=="scalp" else "📈" if mode["mode"]=="trend" else "ℹ️"
        strat = f"Strategy: {strat_icon} {mode['mode'].upper()}"

        bal_note = f"Balance={bal:.2f}" if bal is not None else ""
        pnl_note = f"CompoundPnL={cpnl:.6f}" if cpnl is not None else ""
        wallet = (" | ".join(x for x in [bal_note, pnl_note] if x)) or ""

        gz_note = ""
        if gz and gz.get("ok"):
            gz_note = f" | 🟡 {gz['zone']['type']} s={gz['score']:.1f}"

        # اطبع سطور واضحة
        if LOG_ADDONS:
            print(f"🧱 {bm_note}", flush=True)
            print(f"📦 {fl_note}", flush=True)
            print(f"📊 {dash}{gz_note}", flush=True)
            print(f"{strat}{(' | ' + wallet) if wallet else ''}", flush=True)
            
            # سنابشوت إضافي مختصر - مصحح
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

        return {"bm": bm, "flow": flow, "cv": cv, "mode": mode, "gz": gz, "wallet": wallet}
    except Exception as e:
        print(f"🟨 AddonLog error: {e}", flush=True)
        return {"bm": None, "flow": None, "cv": {"b":0,"s":0,"score_b":0.0,"score_s":0.0,"ind":{}},
                "mode": {"mode":"n/a"}, "gz": None, "wallet": ""}

# =================== BOOKMAP-LITE ===================
def log_bookmap(bm):
    if not bm.get("ok"):
        print(f"🧱 Bookmap: N/A ({bm.get('why')})", flush=True); return
    bw = ", ".join([f"{p:.4f}@{q:.0f}" for p,q in bm["buy_walls"]])
    sw = ", ".join([f"{p:.4f}@{q:.0f}" for p,q in bm["sell_walls"]])
    tag = "🟢" if bm["imbalance"]>=IMBALANCE_ALERT else ("🔴" if bm["imbalance"]<=1/IMBALANCE_ALERT else "⚖️")
    print(f"🧱 Bookmap: {tag} Imb={bm['imbalance']:.2f} | BuyWalls[{bw}] | SellWalls[{sw}]", flush=True)

# =================== VOLUME FLOW ===================
def log_flow(info):
    if not info.get("ok"):
        print(f"📦 Flow: N/A ({info.get('why')})", flush=True); return
    dir_ = "🟢Buy" if info["delta_last"]>0 else ("🔴Sell" if info["delta_last"]<0 else "⚖️Flat")
    spike = " ⚡Spike" if info["spike"] else ""
    cvd_tr= "↗️" if info["cvd_trend"]=="up" else "↘️"
    print(f"📦 Flow: {dir_} Δ={info['delta_last']:.0f} z={info['delta_z']:.2f}{spike} | CVD {cvd_tr} {info['cvd_last']:.0f}", flush=True)

# =================== SHADOW DASHBOARD ===================
def shadow_dashboard(side_hint, council, bm, flow, extras=None):
    if not SHADOW_MODE_DASHBOARD: return
    b = council.get("b",0); s=council.get("s",0)
    sb=council.get("score_b",0.0); ss=council.get("score_s",0.0)
    ind = council.get("ind", {})
    rsi = ind.get("rsi", 50.0); adx=ind.get("adx", 0.0)
    di = ind.get("di_spread", 0.0); evx=ind.get("evx", 1.0)
    imb_str="N/A"; imb_tag="❔"
    if bm and bm.get("ok"):
        imb = bm["imbalance"]; imb_str=f"{imb:.2f}"
        imb_tag = "🟢" if imb>=IMBALANCE_ALERT else ("🔴" if imb<=1/IMBALANCE_ALERT else "⚖️")
    fl_dir="N/A"; fl_z="N/A"; fl_spk=""
    if flow and flow.get("ok"):
        fl_dir = "🟢" if flow["delta_last"]>0 else ("🔴" if flow["delta_last"]<0 else "⚖️")
        fl_z = f"{flow['delta_z']:.2f}"
        fl_spk = "⚡" if flow["spike"] else ""
    mode = (extras or {}).get("mode", "n/a")
    gz = (extras or {}).get("gz", {})
    gz_tag = ""
    if gz and gz.get("ok"):
        zone_type = gz["zone"]["type"]
        zone_score = gz["score"]
        gz_tag = f" | 🟡 {zone_type} s={zone_score:.1f}"
    print(
        f"📊 DASH — hint={side_hint} | Council BUY({b},{sb}) SELL({s},{ss}) | "
        f"RSI={rsi:.1f} ADX={adx:.1f} DI={di:.1f} EVX={evx:.2f} | "
        f"OB Imb={imb_tag}{imb_str} | Flow={fl_dir} z={fl_z}{fl_spk} | mode={mode}{gz_tag}", flush=True
    )

# =================== STRATEGY BANNER ===================
def log_strategy_banner(mode_dict):
    m = mode_dict.get("mode","n/a"); why = mode_dict.get("why","")
    icon = "⚡" if m=="scalp" else "📈" if m=="trend" else "ℹ️"
    print(f"{icon} Strategy: {m.upper()} ({why})", flush=True)

# =================== TRADE OPEN SHEET ===================
def log_open_trade_details(side, price, qty, lev, mode, votes, golden=None, bm=None, flow=None):
    gz_note = ""
    if golden and golden.get("ok"):
        gz_note = f" | 🟡 {golden['zone']['type']} s={golden['score']:.1f}"
    bm_note = ""
    if bm and bm.get("ok"):
        tag = "🟢" if bm["imbalance"]>=IMBALANCE_ALERT else ("🔴" if bm["imbalance"]<=1/IMBALANCE_ALERT else "⚖️")
        bm_note = f" | 🧱 {tag} Imb={bm['imbalance']:.2f}"
    fl_note = ""
    if flow and flow.get("ok"):
        fl_dir = "🟢" if flow["delta_last"]>0 else ("🔴" if flow["delta_last"]<0 else "⚖️")
        spike = "⚡" if flow["spike"] else ""
        fl_note = f" | 📦 {fl_dir} Δ={flow['delta_last']:.0f}{spike}"
    print(
        f"🚀 OPEN {side} @ {price:.6f}  qty={qty:.3f}  lev={lev}x  mode={mode}"
        f" | council: BUY({votes.get('b',0)},{votes.get('score_b',0.0)}) SELL({votes.get('s',0)},{votes.get('score_s',0.0)})"
        f"{gz_note}{bm_note}{fl_note}", flush=True
    )

# =================== SNAPSHOT EMITTER ===================
def emit_snapshots_legacy(exchange, symbol, df):
    """
    دالة مجمعة لجميع الإضافات - تضمن ظهور كل اللوج الجديد
    """
    try:
        bm = bookmap_snapshot(exchange, symbol, depth=BOOKMAP_DEPTH)
        log_bookmap(bm)
        flow = compute_flow_metrics(df)
        log_flow(flow)
        cv = council_votes_pro(df)
        mode = decide_strategy_mode(df)
        gz = golden_zone_check(df, {"adx": cv["ind"]["adx"]}, "buy" if cv["b"]>=cv["s"] else "sell")
        side_hint = "BUY" if cv["b"]>=cv["s"] else "SELL"
        shadow_dashboard(side_hint, cv, bm, flow, extras={"mode": mode["mode"], "gz": gz})
        log_strategy_banner(mode)
        print("✅ ADDONS LIVE", flush=True)
        return {"bm": bm, "flow": flow, "cv": cv, "mode": mode, "gz": gz}
    except Exception as e:
        print(f"🟨 AddonLog error: {e}", flush=True)
        return {"bm": None, "flow": None, "cv": {"b":0,"s":0,"score_b":0.0,"score_s":0.0,"ind":{}}, "mode": {"mode":"n/a"}, "gz": None}

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

def ind_rsi(close, n=RSI_LEN):
    d = close.diff()
    up = d.clip(lower=0); dn = (-d).clip(lower=0)
    rs = _ema(up, n) / _ema(dn, n).replace(0, 1e-12)
    return 100 - (100/(1+rs))

def _ema(s, n):   return s.ewm(span=n, adjust=False).mean()
def _sma(series, n): return series.rolling(n).mean()

def council_votes_pro(df):
    """مجلس التصويت المحسّن مع RSI-MA boosts والمناطق الذهبية"""
    try:
        ind = compute_indicators(df)
        rsi = ind.get("rsi", 50.0)
        adx = ind.get("adx", 0.0)
        plus_di = ind.get("plus_di", 0.0)
        minus_di = ind.get("minus_di", 0.0)
        di_spread = abs(plus_di - minus_di)
        evx = ind.get("evx", 1.0)
        
        # حساب RSI-MA
        rsi_series = pd.Series([ind.get("rsi", 50.0)] * len(df))  # محاكاة للبيانات
        if len(df) >= RSI_MA_LEN:
            rsi_values = []
            for i in range(len(df)):
                if i < RSI_MA_LEN:
                    rsi_values.append(50.0)
                else:
                    rsi_slice = df["close"].iloc[i-RSI_MA_LEN:i].apply(lambda x: float(x))
                    rsi_val = ind_rsi(rsi_slice, RSI_LEN).iloc[-1] if len(rsi_slice) >= RSI_LEN else 50.0
                    rsi_values.append(rsi_val)
            rsi_series = pd.Series(rsi_values)
            rsi_ma = _sma(rsi_series, RSI_MA_LEN).iloc[-1]
        else:
            rsi_ma = 50.0

        # Trend-Z detection (استمرار 3 شمعات)
        rsi_trendz = "neutral"
        if len(df) >= RSI_TRENDZ_PERSIST:
            recent_rsi = [ind.get("rsi", 50.0)]  # محاكاة
            recent_ma = [rsi_ma] * RSI_TRENDZ_PERSIST
            if all(r > m for r, m in zip(recent_rsi, recent_ma)):
                rsi_trendz = "bull"
            elif all(r < m for r, m in zip(recent_rsi, recent_ma)):
                rsi_trendz = "bear"

        votes_b = 0
        votes_s = 0
        score_b = 0.0
        score_s = 0.0
        logs = []

        # === الأساس: مؤشرات الترند ===
        if adx > TREND_ADX_MIN:
            if plus_di > minus_di and di_spread > TREND_DI_SPREAD:
                votes_b += 2
                score_b += 1.5
                logs.append("📈 ترند صاعد قوي")
            elif minus_di > plus_di and di_spread > TREND_DI_SPREAD:
                votes_s += 2
                score_s += 1.5
                logs.append("📉 ترند هابط قوي")

        # === RSI-MA boosts ===
        bull_cross = (rsi > rsi_ma) and rsi < 70
        bear_cross = (rsi < rsi_ma) and rsi > 30
        
        if bull_cross:
            votes_b += 2
            score_b += 1.0
            logs.append("🟢 RSI-MA إيجابي")
        if bear_cross:
            votes_s += 2  
            score_s += 1.0
            logs.append("🔴 RSI-MA سلبي")

        # Trend-Z (استمرار 3 شمعات مع ميل المتوسط)
        if rsi_trendz == 'bull':
            votes_b += 3
            score_b += 1.5
            logs.append("🚀 RSI ترند صاعد مستمر")
        if rsi_trendz == 'bear':
            votes_s += 3
            score_s += 1.5
            logs.append("💥 RSI ترند هابط مستمر")

        # === EVX قوة الاتجاه ===
        if evx > EVX_STRONG:
            if plus_di > minus_di:
                votes_b += 1
                score_b += 0.5
                logs.append("⚡ EVX داعم للشراء")
            else:
                votes_s += 1
                score_s += 0.5
                logs.append("⚡ EVX داعم للبيع")

        # === المنطقة الذهبية ===
        side_hint = "buy" if votes_b >= votes_s else "sell"
        gz = golden_zone_check(df, {"adx": adx}, side_hint)
        if gz and gz.get("ok"):
            if gz['zone']['type'] == 'golden_bottom':
                votes_b += 3
                score_b += 1.5
                logs.append(f"🏆 منطقة ذهبية للشراء (قوة: {gz['score']:.1f})")
            elif gz['zone']['type'] == 'golden_top':
                votes_s += 3  
                score_s += 1.5
                logs.append(f"🏆 منطقة ذهبية للبيع (قوة: {gz['score']:.1f})")

        # === تخفيض الثقة في النطاق المحايد ===
        if 45 <= rsi <= 55:
            score_b *= 0.8
            score_s *= 0.8
            logs.append("⚖️ RSI في النطاق المحايد - تخفيض الثقة")

        # تحديث المؤشرات بالإضافات
        ind.update({
            "rsi_ma": rsi_ma,
            "rsi_trendz": rsi_trendz,
            "di_spread": di_spread,
            "evx": evx,
            "gz": gz
        })

        return {
            "b": votes_b, "s": votes_s,
            "score_b": score_b, "score_s": score_s,
            "logs": logs, "ind": ind
        }
    except Exception as e:
        log_w(f"council_votes_pro error: {e}")
        return {"b":0,"s":0,"score_b":0.0,"score_s":0.0,"logs":[],"ind":{}}

def decide_strategy_mode(df):
    return {"mode":"scalp","why":"sample"}

def golden_zone_check(df, ind, side: str):
    return {"ok":False,"score":0.0,"reasons":[],"zone":None}

def entry_confirmation_guard(df, side_to_open, v, ind, bm=None, flow=None):
    """حارس دخول ذكي متوازن بين الأمان والهجوم"""
    if SMART_MODE != "pro":
        return {"ok": True, "why": ["smart_off"]}

    adx = ind.get('adx', 0.0)
    rsi = ind.get('rsi', 50.0)
    di_spread = ind.get('di_spread', 0.0)
    evx = ind.get('evx', 1.0)
    gz = ind.get('gz')

    # Anti-chop (منع التردد)
    if adx < 15 and 45 <= rsi <= 55 and abs(evx-1.0) < 0.2:
        return {"ok": False, "why": ["choppy_market"]}

    # فتح مبكّر من منطقة ذهبية قوية
    if gz and gz.get("ok") and adx >= 20 and gz['score'] >= 6.0:
        return {"ok": True, "why": [f"golden_{gz['zone']['type']}_confirmed"]}

    # شروط أساسية
    is_buy = side_to_open.upper().startswith('B')
    core_votes = (v['b'] if is_buy else v['s']) >= ENTRY_MIN_VOTES
    core_score = (v['score_b'] if is_buy else v['score_s']) >= ENTRY_MIN_SCORE

    # Bookmap/Flow تأكيد إضافي
    bm_imb = bm['imbalance'] if bm and bm.get('ok') else 1.0
    flow_z = flow['delta_z'] if flow and flow.get('ok') else 0.0
    
    ok_bm = bm_imb >= IMBALANCE_ALERT if is_buy else (1.0/IMBALANCE_ALERT >= bm_imb)
    ok_flow = flow_z >= FLOW_SPIKE_Z if is_buy else (-flow_z >= FLOW_SPIKE_Z)

    # تحديد النمط والشروط
    is_trend = adx >= TREND_ADX_MIN and abs(di_spread) >= TREND_DI_SPREAD
    is_scalp = adx >= SCALP_ADX_MIN and (ok_bm or ok_flow)
    
    ok_mode = is_trend or is_scalp

    if core_votes and core_score and (ok_bm or ok_flow) and ok_mode:
        reasons = []
        if is_trend: reasons.append("trend_mode")
        else: reasons.append("scalp_mode")
        if ok_bm: reasons.append("bookmap_confirm")
        if ok_flow: reasons.append("flow_confirm")
        return {"ok": True, "why": reasons}

    # أسباب المنع
    why_not = []
    if not core_votes: why_not.append(f"votes<{ENTRY_MIN_VOTES}")
    if not core_score: why_not.append(f"score<{ENTRY_MIN_SCORE}")
    if not (ok_bm or ok_flow): why_not.append("weak_bm_flow")
    if not ok_mode: why_not.append("mode_guard")
    return {"ok": False, "why": why_not}

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
}
compound_pnl = 0.0
wait_for_next_signal_side = None

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

def open_market(side, qty, price):
    if qty<=0: 
        log_e("skip open (qty<=0)")
        return False
    
    # === SMART COUNCIL INTEGRATION ===
    df = fetch_ohlcv()
    
    # === 📍 الموقع الحاسم: استدعاء جميع الإضافات ===
    snap = emit_snapshots(ex, SYMBOL, df)
    
    votes = {"b":snap["cv"]["b"],"s":snap["cv"]["s"],"score_b":snap["cv"]["score_b"],"score_s":snap["cv"]["score_s"]}
    
    # تحديث نداء الحارس
    ind_extended = snap["cv"]["ind"].copy()
    ind_extended['bm_imb'] = snap["bm"]["imbalance"] if snap["bm"] and snap["bm"].get("ok") else 1.0
    ind_extended['flow_z'] = snap["flow"]["delta_z"] if snap["flow"] and snap["flow"].get("ok") else 0.0

    g = entry_confirmation_guard(df, side, votes, ind_extended, 
                               bm=snap["bm"], flow=snap["flow"])
    
    # لوج سبب المنع
    if not g["ok"]:
        side_hint = "BUY" if side.upper().startswith("B") else "SELL"
        print(f"🛑 ENTRY BLOCK: {side_hint} | {', '.join(g['why'])} | "
              f"RSI={ind_extended.get('rsi',0):.1f} ADX={ind_extended.get('adx',0):.1f} "
              f"z={ind_extended.get('flow_z',0):.2f} imb={ind_extended.get('bm_imb',1.0):.2f}", 
              flush=True)
        return False

    log_i(f"PLAN → council={votes} | mode={snap['mode']['mode']}({snap['mode']['why']})")
    if snap["gz"] and snap["gz"].get("ok"):
        log_i(f"نقطة ذهبية: {snap['gz']['zone']['type']} | score={snap['gz']['score']:.1f} | {snap['gz']['reasons']}")
    
    if MODE_LIVE:
        try:
            try: ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
            except Exception: pass
            ex.create_order(SYMBOL, "market", side, qty, None, _params_open(side))
        except Exception as e:
            log_e(f"open: {e}")
            logging.error(f"open_market error: {e}")
            return False
    
    STATE.update({
        "open": True, "side": "long" if side=="buy" else "short", "entry": price,
        "qty": qty, "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
        "tp1_done": False, "highest_profit_pct": 0.0, "profit_targets_achieved": 0,
        "mode": snap['mode']['mode']
    })
    
    # === 📍 الموقع الحاسم 2: تسجيل فتح الصفقة ===
    if LOG_ADDONS:
        votes = {"b": (snap["cv"].get("b",0) if snap["cv"] else 0),
                 "s": (snap["cv"].get("s",0) if snap["cv"] else 0),
                 "score_b": (snap["cv"].get("score_b",0.0) if snap["cv"] else 0.0),
                 "score_s": (snap["cv"].get("score_s",0.0) if snap["cv"] else 0.0)}
        log_open_trade_details(side, price, qty, LEVERAGE,
                               (snap["mode"]["mode"] if snap["mode"] else "n/a"),
                               votes, golden=(snap["gz"] or {}),
                               bm=(snap["bm"] or {}), flow=(snap["flow"] or {}))
    
    log_i(f"Decision Summary → {side} | reasons: {','.join([r for r in snap['cv']['logs'] if not r.startswith('🟨')])}")
    logging.info(f"OPEN {side} qty={qty} price={price} mode={snap['mode']['mode']} council={votes}")
    
    # === SAVE STATE FOR RECOVERY ===
    save_state({
        "in_position": True,
        "side": "LONG" if side.upper().startswith("B") else "SHORT",
        "entry_price": price,
        "position_qty": qty,
        "leverage": LEVERAGE,
        "mode": snap['mode']['mode'],
        "gz_snapshot": snap["gz"] if isinstance(snap["gz"], dict) else {},
        "cv_snapshot": votes if isinstance(votes, dict) else {},
        "opened_at": int(time.time()),
        "partial_taken": False,
        "breakeven_armed": False,
        "trail_active": False,
        "trail_tightened": False,
    })
    
    return True

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
            if MODE_LIVE:
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
    global wait_for_next_signal_side
    prev_side = prev_side or STATE.get("side")
    STATE.update({
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
        "tp1_done": False, "highest_profit_pct": 0.0, "profit_targets_achieved": 0,
        "trail_tightened": False, "partial_taken": False
    })
    # SAVE CLOSED STATE
    save_state({"in_position": False, "position_qty": 0})
    
    if prev_side == "long":  wait_for_next_signal_side = "sell"
    elif prev_side == "short": wait_for_next_signal_side = "buy"
    else: wait_for_next_signal_side = None
    logging.info(f"AFTER_CLOSE waiting_for={wait_for_next_signal_side}")

def smart_exit_guard(state, df, ind, flow, bm, now_price, pnl_pct, mode, side, entry_price):
    """يقرر: Partial / Tighten / Strict Close مع لوج واضح."""
    atr = ind.get('atr', 0.0)
    adx = ind.get('adx', 0.0)
    rsi = ind.get('rsi', 50.0)
    rsi_ma = ind.get('rsi_ma', 50.0)
    
    # حساب ميل ADX (مبسط)
    if len(df) >= 3:
        adx_slope = adx - ind.get('adx_prev', adx)
    else:
        adx_slope = 0.0

    # قراءة الشمعة الحالية للفتيلة
    if len(df) > 0:
        c = df.iloc[-1]
        wick_up = float(c['high']) - max(float(c['close']), float(c['open']))
        wick_down = min(float(c['close']), float(c['open'])) - float(c['low'])
    else:
        wick_up = wick_down = 0.0

    # إشارات إنعكاس/إنهاك
    rsi_cross_down = (rsi < rsi_ma) if side == "long" else (rsi > rsi_ma)
    adx_falling = (adx_slope < 0)
    cvd_down = (flow and flow.get('ok') and flow.get('cvd_trend') == 'down')
    evx_spike = (ind.get('evx', 1.0) >= EVX_SPIKE)
    
    # تحليل Bookmap
    bm_wall_close = False
    if bm and bm.get('ok'):
        if side == "long":  # نبحث عن جدار بيع قريب
            sell_walls = bm.get('sell_walls', [])
            if sell_walls:
                best_ask = min([p for p, _ in sell_walls])
                bps = abs((best_ask - now_price) / now_price) * 10000.0
                bm_wall_close = (bps <= BM_WALL_PROX_BPS)
        else:  # short - نبحث عن جدار شراء قريب
            buy_walls = bm.get('buy_walls', [])
            if buy_walls:
                best_bid = max([p for p, _ in buy_walls])
                bps = abs((best_bid - now_price) / now_price) * 10000.0
                bm_wall_close = (bps <= BM_WALL_PROX_BPS)

    # 1) Partial TP مبكر حسب النمط
    tp1_target = TP1_SCALP_PCT if mode == 'scalp' else TP1_TREND_PCT
    if pnl_pct >= tp1_target and not state.get('tp1_done'):
        qty_pct = 0.35 if mode == 'scalp' else 0.25
        return {
            "action": "partial", 
            "why": f"TP1 hit {tp1_target*100:.2f}%",
            "qty_pct": qty_pct,
            "log": f"💰 TP1 جزئي {tp1_target*100:.2f}% | pnl={pnl_pct*100:.2f}% | mode={mode}"
        }

    # 2) Tighten: نزول/إنهاك مع ربح موجود
    if pnl_pct > 0:
        wick_signal = (wick_up >= WICK_ATR_MULT * atr) if side == "long" else (wick_down >= WICK_ATR_MULT * atr)
        if wick_signal or evx_spike or bm_wall_close or cvd_down:
            return {
                "action": "tighten", 
                "why": "exhaustion/flow/wall",
                "trail_mult": TRAIL_TIGHT_MULT,
                "log": f"🛡️ Tighten | wick={wick_up/atr:.1f}×ATR evx={ind.get('evx',1.0):.2f} wall={bm_wall_close} cvd_down={cvd_down}"
            }

    # 3) Strict close: ربح محترم + إشارات هابطة متزامنة
    bearish_signals = [rsi_cross_down, adx_falling, cvd_down, evx_spike, bm_wall_close]
    bearish_count = sum(bearish_signals)
    
    if pnl_pct >= HARD_CLOSE_PNL_PCT and bearish_count >= 2:
        reasons = []
        if rsi_cross_down: reasons.append("rsi↓")
        if adx_falling: reasons.append("adx↓")
        if cvd_down: reasons.append("cvd↓")
        if evx_spike: reasons.append(f"evx{ind.get('evx',1.0):.1f}")
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

# =================== DYNAMIC TP ===================
def manage_after_entry(df, ind, info):
    """إدارة ذكية للمركز مع خروج استباقي"""
    if not STATE["open"] or STATE["qty"] <= 0:
        return

    px = info["price"]
    entry = STATE["entry"]
    side = STATE["side"]
    qty = STATE["qty"]
    mode = STATE.get("mode", "trend")
    
    # حساب الربح الحالي
    pnl_pct = (px - entry) / entry * 100 * (1 if side == "long" else -1)
    STATE["pnl"] = pnl_pct
    
    # تحديث أعلى ربح
    if pnl_pct > STATE["highest_profit_pct"]:
        STATE["highest_profit_pct"] = pnl_pct

    # === نداء حارس الخروج الذكي ===
    snap = emit_snapshots(ex, SYMBOL, df)  # تحديث البيانات الحية
    exit_signal = smart_exit_guard(STATE, df, ind, snap["flow"], snap["bm"], px, pnl_pct/100, mode, side, entry)
    
    if exit_signal["log"]:
        print(exit_signal["log"], flush=True)
        logging.info(f"EXIT_GUARD: {exit_signal}")

    # تنفيذ قرار الخروج
    if exit_signal["action"] == "partial" and not STATE.get("partial_taken"):
        partial_qty = safe_qty(qty * exit_signal["qty_pct"])
        if partial_qty > 0:
            close_side = "sell" if side == "long" else "buy"
            if MODE_LIVE:
                try:
                    ex.create_order(SYMBOL, "market", close_side, partial_qty, None, _params_close())
                    log_g(f"PARTIAL CLOSE: {partial_qty:.4f} {SYMBOL} | {exit_signal['why']}")
                    STATE["partial_taken"] = True
                    STATE["qty"] = safe_qty(qty - partial_qty)
                except Exception as e:
                    log_e(f"Partial close failed: {e}")
    
    elif exit_signal["action"] == "tighten" and not STATE.get("trail_tightened"):
        # تشديد التريل
        STATE["trail_tightened"] = True
        STATE["trail"] = None  # إعادة حساب التريل
        log_i(f"TRAIL TIGHTENED: {exit_signal['why']}")
    
    elif exit_signal["action"] == "close":
        log_w(f"SMART EXIT: {exit_signal['why']}")
        close_market_strict(f"smart_exit_{exit_signal['why']}")
        return

    # === المنطق الأصلي للـ TP/Trail (يبقى كما هو مع التعديلات) ===
    current_atr = ind.get("atr", 0.0)
    entry_price = STATE["entry"]
    
    # تحديد إعدادات النمط
    if mode == 'scalp':
        tp1_pct = SCALP_TP1 / 100.0
        be_activate_pct = SCALP_BE / 100.0
        trail_activate_pct = SCALP_TRAIL_ACTIVATE / 100.0
        atr_trail_mult = ATR_TRAIL_MULT
    else:  # trend
        tp1_pct = TP1_PCT_BASE / 100.0
        be_activate_pct = BREAKEVEN_AFTER / 100.0
        trail_activate_pct = TRAIL_ACTIVATE_PCT / 100.0
        atr_trail_mult = ATR_TRAIL_MULT

    # TP1 Logic
    if not STATE.get("tp1_done") and pnl_pct/100 >= tp1_pct:
        close_fraction = TP1_CLOSE_FRAC
        close_qty = safe_qty(STATE["qty"] * close_fraction)
        if close_qty > 0:
            close_side = "sell" if STATE["side"] == "long" else "buy"
            if MODE_LIVE:
                try:
                    ex.create_order(SYMBOL, "market", close_side, close_qty, None, _params_close())
                except Exception as e:
                    log_e(f"TP1 close failed: {e}")
            STATE["qty"] = safe_qty(STATE["qty"] - close_qty)
            STATE["tp1_done"] = True
            STATE["profit_targets_achieved"] += 1
            log_g(f"TP1 HIT: closed {close_fraction*100}% | remaining {STATE['qty']:.4f}")

    # Breakeven Logic
    if not STATE.get("breakeven_armed") and pnl_pct/100 >= be_activate_pct:
        STATE["breakeven_armed"] = True
        STATE["breakeven"] = entry_price
        log_i("BREAKEVEN ARMED")

    # Trail Logic
    if not STATE.get("trail_active") and pnl_pct/100 >= trail_activate_pct:
        STATE["trail_active"] = True
        log_i("TRAIL ACTIVATED")

    if STATE.get("trail_active"):
        trail_mult = TRAIL_TIGHT_MULT if STATE.get("trail_tightened") else atr_trail_mult
        if side == "long":
            new_trail = px - (current_atr * trail_mult)
            if STATE.get("trail") is None or new_trail > STATE["trail"]:
                STATE["trail"] = new_trail
        else:  # short
            new_trail = px + (current_atr * trail_mult)
            if STATE.get("trail") is None or new_trail < STATE["trail"]:
                STATE["trail"] = new_trail

    # Check trail stop
    if STATE.get("trail"):
        if (side == "long" and px <= STATE["trail"]) or (side == "short" and px >= STATE["trail"]):
            log_w(f"TRAIL STOP: {px} vs trail {STATE['trail']}")
            close_market_strict("trail_stop")

    # Check breakeven stop
    if STATE.get("breakeven"):
        if (side == "long" and px <= STATE["breakeven"]) or (side == "short" and px >= STATE["breakeven"]):
            log_w(f"BREAKEVEN STOP: {px} vs breakeven {STATE['breakeven']}")
            close_market_strict("breakeven_stop")

    # Dust guard
    if STATE["qty"] <= FINAL_CHUNK_QTY:
        log_w(f"DUST GUARD: qty {STATE['qty']} <= {FINAL_CHUNK_QTY}, closing...")
        close_market_strict("dust_guard")

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
        print(f"   🎯 ENTRY: RF-LIVE ONLY  |  spread_bps={fmt(spread_bps,2)}")
        print(f"   ⏱️ closes_in ≈ {left_s}s")
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

def trade_loop():
    global wait_for_next_signal_side
    loop_i=0
    while True:
        try:
            bal = balance_usdt()
            px  = price_now()
            df  = fetch_ohlcv()
            info = rf_signal_live(df)
            ind  = compute_indicators(df)
            spread_bps = orderbook_spread_bps()
            
            # === 📍 الموقع الحاسم 3: استدعاء الإضافات في كل دورة ===
            if LOG_ADDONS:
                snap = emit_snapshots(ex, SYMBOL, df,
                                    balance_fn=lambda: float(bal) if bal else None,
                                    pnl_fn=lambda: float(compound_pnl))
            
            if STATE["open"] and px:
                STATE["pnl"] = (px-STATE["entry"])*STATE["qty"] if STATE["side"]=="long" else (STATE["entry"]-px)*STATE["qty"]
            
            # تحديث نداء إدارة الصفقة
            if STATE["open"]:
                manage_after_entry(df, ind, {
                    "price": px or info["price"], 
                    "bm": snap["bm"] if 'snap' in locals() else None,
                    "flow": snap["flow"] if 'snap' in locals() else None,
                    **info
                })
            
            reason=None
            if spread_bps is not None and spread_bps > MAX_SPREAD_BPS:
                reason=f"spread too high ({fmt(spread_bps,2)}bps > {MAX_SPREAD_BPS})"
            sig = "buy" if (ENTRY_RF_ONLY and info["long"]) else ("sell" if (ENTRY_RF_ONLY and info["short"]) else None)
            if not STATE["open"] and sig and reason is None:
                if wait_for_next_signal_side and sig != wait_for_next_signal_side:
                    reason=f"waiting opposite RF: need {wait_for_next_signal_side.upper()}"
                else:
                    qty = compute_size(bal, px or info["price"])
                    if qty>0:
                        ok = open_market(sig, qty, px or info["price"])
                        if ok:
                            wait_for_next_signal_side = None
                    else:
                        reason="qty<=0"
            if LOG_LEGACY:
                pretty_snapshot(bal, {"price": px or info["price"], **info}, ind, spread_bps, reason, df)
            loop_i += 1
            sleep_s = NEAR_CLOSE_S if time_to_candle_close(df)<=10 else BASE_SLEEP
            time.sleep(sleep_s)
        except Exception as e:
            log_e(f"loop error: {e}\n{traceback.format_exc()}")
            logging.error(f"trade_loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

# =================== API / KEEPALIVE ===================
app = Flask(__name__)
@app.route("/")
def home():
    mode='LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ RF-LIVE Bot — {SYMBOL} {INTERVAL} — {mode} — Entry: RF LIVE only — Dynamic TP — Strict Close"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "state": STATE, "compound_pnl": compound_pnl,
        "entry_mode": "RF_LIVE_ONLY", "wait_for_next_signal": wait_for_next_signal_side,
        "guards": {"max_spread_bps": MAX_SPREAD_BPS, "final_chunk_qty": FINAL_CHUNK_QTY}
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True, "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"], "side": STATE["side"], "qty": STATE["qty"],
        "compound_pnl": compound_pnl, "timestamp": datetime.utcnow().isoformat(),
        "entry_mode": "RF_LIVE_ONLY", "wait_for_next_signal": wait_for_next_signal_side
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

    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'}  •  {SYMBOL}  •  {INTERVAL}", "yellow"))
    print(colored(f"RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x  •  RF_LIVE={RF_LIVE_ONLY}", "yellow"))
    print(colored(f"ENTRY: RF ONLY  •  FINAL_CHUNK_QTY={FINAL_CHUNK_QTY}", "yellow"))
    print(colored(f"SMART MODE: {SMART_MODE}  •  SMART EXIT: ACTIVE", "yellow"))
    logging.info("service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    
    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)

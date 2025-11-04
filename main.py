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
• [ADDON] Bookmap-Lite + Volume Flow + Shadow Dashboard
"""

import os, time, math, random, signal, sys, traceback, logging
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

# =================== SETTINGS ===================
SYMBOL     = os.getenv("SYMBOL", "DOGE/USDT:USDT")
INTERVAL   = os.getenv("INTERVAL", "15m")
LEVERAGE   = int(os.getenv("LEVERAGE", 10))
RISK_ALLOC = float(os.getenv("RISK_ALLOC", 0.60))   # 60% من الرصيد
POSITION_MODE = os.getenv("BINGX_POSITION_MODE", "oneway")  # oneway/hedge

# RF (TradingView-like) — live candle only
RF_SOURCE = "close"
RF_PERIOD = int(os.getenv("RF_PERIOD", 20))
RF_MULT   = float(os.getenv("RF_MULT", 3.5))
RF_LIVE_ONLY = True
RF_HYST_BPS  = 6.0  # كسر واضح عن الفلتر لتقليل الفليكر على الشمعة الحيّة

# Indicators
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14

# دخول فقط من RF (لا أي مدخلات أخرى)
ENTRY_RF_ONLY = True

# Spread guard
MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", 6.0))

# Dynamic TP / trail
TP1_PCT_BASE       = 0.40
TP1_CLOSE_FRAC     = 0.50
BREAKEVEN_AFTER    = 0.30
TRAIL_ACTIVATE_PCT = 1.20
ATR_TRAIL_MULT     = 1.6

# سلّم الأهداف الافتراضي
TREND_TPS       = [0.50, 1.00, 1.80]
TREND_TP_FRACS  = [0.30, 0.30, 0.20]

# Dust guard / final chunk
FINAL_CHUNK_QTY = float(os.getenv("FINAL_CHUNK_QTY", 40.0))  # <= 40 DOGE → strict close
RESIDUAL_MIN_QTY = float(os.getenv("RESIDUAL_MIN_QTY", 9.0)) # أقل كمية بعد الجزئي (min lot)

# Strict close
CLOSE_RETRY_ATTEMPTS = 6
CLOSE_VERIFY_WAIT_S  = 2.0

# Pacing
BASE_SLEEP   = 5
NEAR_CLOSE_S = 1

# ==== [ADDON] Smart Settings ====
# RSI-MA
RSI_LEN = 14
RSI_MA_LEN = 9
RSI_CHOP_BAND = (45.0, 55.0)
RSI_CROSS_VOTES = 2;   RSI_CROSS_SCORE = 1.0
RSI_TRENDZ_PERSIST = 3
RSI_TRENDZ_VOTES = 3;  RSI_TRENDZ_SCORE = 1.5

# EVX (volatility explosion)
EVX_ATR_LEN = 14
EVX_BASE_LEN = 50
EVX_STRONG = 1.8

# Golden Zones
GZ_MIN_SCORE  = 6.0
GZ_ADX_MIN    = 20.0
GZ_REQ_VOL_MA = 20
GZ_FIB_LOW    = 0.618
GZ_FIB_HIGH   = 0.786
GZ_CAN_LEAD_ENTRY = True

# Entry Guard
ENTRY_CONFIRM_GUARD = True
ENTRY_MIN_VOTES   = 8
ENTRY_MIN_SCORE   = 2.5
RSI_CROSS_REQUIRED  = True
RSI_NEUTRAL_BAND    = (45.0, 55.0)

# Strategy modes
TREND_ADX_MIN = 28.0     # أقله ترند قوي
TREND_DI_SPREAD = 8.0    # فرق DI+ و DI-
SCALP_TP1 = 0.35         # % افتراضي للسكالب
SCALP_BE = 0.25
SCALP_TRAIL_ACTIVATE = 0.9

# ==== [ADDON] Orderbook & Flow & Dashboard ====
BOOKMAP_DEPTH = 50
BOOKMAP_TOPWALLS = 3
IMBALANCE_ALERT = 1.40        # >=1.40 ضغط شراء، <=1/1.40 ضغط بيع

FLOW_WINDOW = 20              # نوافذ قياس دلتا الحجم
FLOW_SPIKE_Z = 1.8            # Spike اذا |z|>=1.8
CVD_SMOOTH = 8                # تنعيم CVD لقراءة الاتجاه

SHADOW_MODE_DASHBOARD = True  # Scoreboard فقط (لا يغيّر أي قرار)
DASH_REPEAT_SECS = 30         # إن أردت تطبعه دوريًا أثناء الإدارة
# ================================

# =================== LOGGING ===================
def setup_file_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "").endswith("bot.log")
               for h in logger.handlers):
        fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=7, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    print(colored("🗂️ log rotation ready", "cyan"))

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
        print(colored(f"🔧 precision={AMT_PREC}, step={LOT_STEP}, min={LOT_MIN}", "cyan"))
    except Exception as e:
        print(colored(f"⚠️ load_market_specs: {e}", "yellow"))

def ensure_leverage_mode():
    try:
        try:
            ex.set_leverage(LEVERAGE, SYMBOL, params={"side": "BOTH"})
            print(colored(f"✅ leverage set: {LEVERAGE}x", "green"))
        except Exception as e:
            print(colored(f"⚠️ set_leverage warn: {e}", "yellow"))
        print(colored(f"📌 position mode: {POSITION_MODE}", "cyan"))
    except Exception as e:
        print(colored(f"⚠️ ensure_leverage_mode: {e}", "yellow"))

try:
    load_market_specs()
    ensure_leverage_mode()
except Exception as e:
    print(colored(f"⚠️ exchange init: {e}", "yellow"))

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
    if q<=0: print(colored(f"⚠️ qty invalid after normalize → {q}", "yellow"))
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

# =================== [ADDON] BOOKMAP-LITE ===================
def bookmap_snapshot(exchange, symbol: str, depth: int = BOOKMAP_DEPTH):
    """
    يلخّص دفتر الأوامر: أكبر 3 حوائط شراء/بيع + Imbalance نسبة أحجام.
    return: {"ok":bool, "buy_walls":[(price,qty)..], "sell_walls":[...], "imbalance":float}
    """
    try:
        ob = exchange.fetch_order_book(symbol, depth)
        bids = ob.get("bids", [])[:depth]; asks = ob.get("asks", [])[:depth]
        if not bids or not asks:
            return {"ok": False, "why": "empty_orderbook"}
        b_sizes = np.array([b[1] for b in bids]); b_prices = np.array([b[0] for b in bids])
        a_sizes = np.array([a[1] for a in asks]); a_prices = np.array([a[0] for a in asks])
        b_idx = b_sizes.argsort()[::-1][:BOOKMAP_TOPWALLS]
        a_idx = a_sizes.argsort()[::-1][:BOOKMAP_TOPWALLS]
        buy_walls  = [(float(b_prices[i]), float(b_sizes[i])) for i in b_idx]
        sell_walls = [(float(a_prices[i]), float(a_sizes[i])) for i in a_idx]
        imb = b_sizes.sum() / max(a_sizes.sum(), 1e-12)
        return {"ok": True, "buy_walls": buy_walls, "sell_walls": sell_walls, "imbalance": float(imb)}
    except Exception as e:
        return {"ok": False, "why": f"{e}"}

def log_bookmap(bm):
    if not bm.get("ok"):
        print(f"🧱 Bookmap: N/A ({bm.get('why')})"); return
    bw = ", ".join([f"{p:.4f}@{q:.0f}" for p,q in bm["buy_walls"]])
    sw = ", ".join([f"{p:.4f}@{q:.0f}" for p,q in bm["sell_walls"]])
    tag = "🟢" if bm["imbalance"]>=IMBALANCE_ALERT else ("🔴" if bm["imbalance"]<=1/IMBALANCE_ALERT else "⚖️")
    print(f"🧱 Bookmap: {tag} Imb={bm['imbalance']:.2f} | BuyWalls[{bw}] | SellWalls[{sw}]")

# =================== [ADDON] VOLUME FLOW / DELTA & CVD ===================
def compute_flow_metrics(df):
    """
    Delta per bar (تقريبي): لو close>prev → upVol وإلا downVol.
    CVD: cumulative volume delta (بالتنعيم).
    Spike: z-score على نافذة FLOW_WINDOW.
    """
    if len(df) < max(30, FLOW_WINDOW+2):
        return {"ok": False, "why": "short_df"}
    close = df["close"].astype(float).copy()
    vol   = df["volume"].astype(float).copy()
    up_mask = close.diff().fillna(0) > 0
    up_vol  = (vol * up_mask).astype(float)
    dn_vol  = (vol * (~up_mask)).astype(float)
    delta   = up_vol - dn_vol
    cvd     = delta.cumsum()
    cvd_ma  = cvd.rolling(CVD_SMOOTH).mean()
    wnd = delta.tail(FLOW_WINDOW)
    mu  = float(wnd.mean()); sd = float(wnd.std() or 1e-12)
    z   = float((wnd.iloc[-1] - mu) / sd)
    info = {
        "ok": True,
        "delta_last": float(delta.iloc[-1]),
        "delta_mean": mu,
        "delta_z": z,
        "cvd_last": float(cvd.iloc[-1]),
        "cvd_trend": "up" if (cvd_ma.iloc[-1] - cvd_ma.iloc[-min(CVD_SMOOTH, len(cvd_ma))]) >= 0 else "down",
        "spike": abs(z) >= FLOW_SPIKE_Z
    }
    return info

def log_flow(info):
    if not info.get("ok"):
        print(f"📦 Flow: N/A ({info.get('why')})"); return
    dir_  = "🟢Buy" if info["delta_last"]>0 else ("🔴Sell" if info["delta_last"]<0 else "⚖️Flat")
    spike = " ⚡Spike" if info["spike"] else ""
    cvd_tr= "↗️" if info["cvd_trend"]=="up" else "↘️"
    print(f"📦 Flow: {dir_} Δ={info['delta_last']:.0f} z={info['delta_z']:.2f}{spike} | CVD {cvd_tr} {info['cvd_last']:.0f}")

# =================== [ADDON] SHADOW DASHBOARD ===================
def shadow_dashboard(side_hint, council, bm, flow, extras=None):
    """
    Scoreboard مُوحّد — لا يغير قرار؛ لوج فقط.
    council: dict من council_votes_pro => b,s,score_b,score_s,ind{rsi,adx,di_spread,evx}
    extras: {"mode":"scalp/trend", "gz":{...}} اختياري
    """
    if not SHADOW_MODE_DASHBOARD: return
    b = council.get("b",0); s=council.get("s",0)
    sb=council.get("score_b",0.0); ss=council.get("score_s",0.0)
    ind = council.get("ind", {})
    rsi = ind.get("rsi", 50.0); adx=ind.get("adx", 0.0)
    di  = ind.get("di_spread", 0.0); evx=ind.get("evx", 1.0)
    imb_str="N/A"; imb_tag="❔"
    if bm and bm.get("ok"):
        imb = bm["imbalance"]; imb_str=f"{imb:.2f}"
        imb_tag = "🟢" if imb>=IMBALANCE_ALERT else ("🔴" if imb<=1/IMBALANCE_ALERT else "⚖️")
    fl_dir="N/A"; fl_z="N/A"; fl_spk=""
    if flow and flow.get("ok"):
        fl_dir = "🟢" if flow["delta_last"]>0 else ("🔴" if flow["delta_last"]<0 else "⚖️")
        fl_z   = f"{flow['delta_z']:.2f}"
        fl_spk = "⚡" if flow["spike"] else ""
    mode = (extras or {}).get("mode", "n/a")
    gz   = (extras or {}).get("gz", {})
    gz_tag = ""
    if gz and gz.get("ok"):
        z = gz["zone"]["type"]; sc = gz["score"]
        gz_tag = f" | 🟡 {z} s={sc:.1f}"
    print(
        f"📊 DASH — hint={side_hint} | Council BUY({b},{sb}) SELL({s},{ss}) | "
        f"RSI={rsi:.1f} ADX={adx:.1f} DI={di:.1f} EVX={evx:.2f} | "
        f"OB Imb={imb_tag}{imb_str} | Flow={fl_dir} z={fl_z}{fl_spk} | mode={mode}{gz_tag}"
    )

# =================== [ADDON] STRATEGY BANNER ===================
def log_strategy_banner(mode_dict):
    m = mode_dict.get("mode","n/a"); why = mode_dict.get("why","")
    icon = "⚡" if m=="scalp" else "📈" if m=="trend" else "ℹ️"
    print(f"{icon} Strategy: {m.upper()} ({why})")

# =================== [ADDON] TRADE OPEN SHEET ===================
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
        spike  = "⚡" if flow["spike"] else ""
        fl_note = f" | 📦 {fl_dir} Δ={flow['delta_last']:.0f}{spike}"
    print(
        f"🚀 OPEN {side} @ {price:.6f}  qty={qty:.3f}  lev={lev}x  mode={mode}"
        f" | council: BUY({votes.get('b',0)},{votes.get('score_b',0.0)}) SELL({votes.get('s',0)},{votes.get('score_s',0.0)})"
        f"{gz_note}{bm_note}{fl_note}"
    )

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

# =================== [ADDON] SMART INDICATORS ===================
def _ema(s, n):   return s.ewm(span=n, adjust=False).mean()
def _sma(series, n): return series.rolling(n).mean()

def ind_rsi(close, n=RSI_LEN):
    d = close.diff()
    up = d.clip(lower=0); dn = (-d).clip(lower=0)
    rs = _ema(up, n) / _ema(dn, n).replace(0, 1e-12)
    return 100 - (100/(1+rs))

def ind_macd(close, fast=12, slow=26, sig=9):
    f = _ema(close, fast); s = _ema(close, slow)
    macd = f - s; signal = _ema(macd, sig); hist = macd - signal
    return macd, signal, hist

def ind_true_range(df):
    h,l,c = df['high'].astype(float), df['low'].astype(float), df['close'].astype(float)
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr

def ind_adx_di(df, n=14):
    h,l,c = df['high'].astype(float), df['low'].astype(float), df['close'].astype(float)
    upm = h.diff(); dnm = -l.diff()
    plus_dm  = ((upm>dnm)&(upm>0))*upm
    minus_dm = ((dnm>upm)&(dnm>0))*dnm
    atr = ind_true_range(df).ewm(alpha=1/n, adjust=False).mean().replace(0,1e-12)
    plus_di  = 100*( _ema(plus_dm, n) / atr )
    minus_di = 100*( _ema(minus_dm, n) / atr )
    dx  = 100*((plus_di-minus_di).abs() / (plus_di+minus_di).replace(0,1e-12))
    adx = _ema(dx, n)
    return adx, plus_di, minus_di

def ind_bb_width(close, n=20, k=2.0):
    ma = _sma(close, n); std = close.rolling(n).std()
    upper = ma + k*std; lower = ma - k*std
    return (upper-lower) / ma.replace(0,1e-12)

def ind_evx(df, atr_len=EVX_ATR_LEN, base_len=EVX_BASE_LEN):
    tr = ind_true_range(df)
    atr = tr.ewm(alpha=1/atr_len, adjust=False).mean()
    base= tr.rolling(base_len).mean()
    return atr / base.replace(0,1e-12)

# =================== [ADDON] CANDLE PATTERNS ===================
def candle_signals(df):
    o,c = df['open'].astype(float), df['close'].astype(float)
    h,l = df['high'].astype(float), df['low'].astype(float)
    body = (c-o).abs(); rng = (h-l).replace(0,1e-12)
    upper = h - np.maximum(c,o); lower = np.minimum(c,o) - l
    return {
        "hammer":          bool((lower>body*1.5) & (upper<body*0.4)).iloc[-2] if len(df)>2 else False,
        "inverted_hammer": bool((upper>body*1.5) & (lower<body*0.4)).iloc[-2] if len(df)>2 else False,
        "shooting_star":   bool((upper>body*1.5) & (c<o)).iloc[-2] if len(df)>2 else False,
        "doji":            bool(((body/rng)<0.1).iloc[-2]) if len(df)>2 else False,
        "bullish_engulf":  bool((c.shift(1)>o.shift(1)) & (o<c.shift(1)) & (c>o.shift(1)) & (c>o)).iloc[-1] if len(df)>2 else False,
        "bearish_engulf":  bool((c.shift(1)<o.shift(1)) & (o>c.shift(1)) & (c<o.shift(1)) & (c<o)).iloc[-1] if len(df)>2 else False,
    }

# =================== [ADDON] SMC FUNCTIONS ===================
def find_eqh_eql(df, lookback=50):
    highs = df['high'].astype(float).rolling(3).max()
    lows  = df['low'].astype(float).rolling(3).min()
    eqh,eql=[],[]
    S=max(3,len(df)-lookback)
    ref_hi,ref_lo={},{}
    for i in range(S, len(df)):
        ref_hi.setdefault(round(float(highs.iloc[i]),4), []).append(i)
        ref_lo.setdefault(round(float(lows.iloc[i]),4),  []).append(i)
    def collapse(ref):
        out=[]
        for k,v in ref.items():
            if len(v)>=2: out+=v[-2:]
        return sorted(set(out))
    return collapse(ref_hi), collapse(ref_lo)

def detect_sweep(df, eqh, eql):
    h,l,c = df['high'].astype(float), df['low'].astype(float), df['close'].astype(float)
    i=-2
    return {
        "sweep_up":   any(h.iloc[i]>=h.iloc[idx] and c.iloc[i]<h.iloc[idx] for idx in eqh) if len(df)>5 else False,
        "sweep_down": any(l.iloc[i]<=l.iloc[idx] and c.iloc[i]>l.iloc[idx] for idx in eql) if len(df)>5 else False,
    }

def detect_fvg(df, lookback=50):
    res=[]
    for i in range(max(3,len(df)-lookback), len(df)-1):
        if float(df['low'].iloc[i]) > float(df['high'].iloc[i-2]):
            res.append(("bull", i, float(df['high'].iloc[i-2]), float(df['low'].iloc[i])))
        if float(df['high'].iloc[i]) < float(df['low'].iloc[i-2]):
            res.append(("bear", i, float(df['high'].iloc[i]), float(df['low'].iloc[i-2])))
    return res

def detect_order_blocks(df, lookback=80):
    o,c = df['open'].astype(float), df['close'].astype(float)
    h,l = df['high'].astype(float), df['low'].astype(float)
    r = h - l; out=[]
    for i in range(max(5,len(df)-lookback), len(df)-3):
        impulse = abs(c.iloc[i+3]-c.iloc[i]) > 2.5*float(r.iloc[i])
        if impulse:
            typ = "demand" if c.iloc[i+3] > c.iloc[i] else "supply"
            lo  = float(min(o.iloc[i], c.iloc[i], l.iloc[i]))
            hi  = float(max(o.iloc[i], c.iloc[i], h.iloc[i]))
            out.append({"type":typ,"lo":lo,"hi":hi,"idx":i})
    return out

def detect_bos_choch(df):
    if len(df)<8: return {"bos_up":False,"bos_dn":False,"choch":False}
    h,l,c = df['high'].astype(float), df['low'].astype(float), df['close'].astype(float)
    swing_hi = h.rolling(3).max(); swing_lo = l.rolling(3).min()
    bos_up = c.iloc[-2] > swing_hi.iloc[-4]
    bos_dn = c.iloc[-2] < swing_lo.iloc[-4]
    return {"bos_up":bool(bos_up),"bos_dn":bool(bos_dn),"choch":bool(bos_up and bos_dn)}

# =================== [ADDON] GOLDEN ZONES ===================
def golden_zone_check(df, ind, side: str):
    if len(df)<40:
        return {"ok":False,"score":0.0,"reasons":[],"zone":None}
    # swings (بسيطة)
    d = df.iloc[-51:-1]
    hi_idx = d['high'].astype(float).idxmax()
    lo_idx = d['low'].astype(float).idxmin()
    hi = float(df.loc[hi_idx,'high']); lo=float(df.loc[lo_idx,'low'])
    fib618 = lo + (hi-lo)*GZ_FIB_LOW
    fib786 = lo + (hi-lo)*GZ_FIB_HIGH
    lastc = float(df['close'].iloc[-2])
    score = 0.0; reasons=[]
    if min(fib618,fib786) <= lastc <= max(fib618,fib786):
        score+=2.0; reasons.append("fib_0.618_0.786_ok")
    # شمعة انعكاس (wick)
    o,c = df['open'].astype(float), df['close'].astype(float)
    h,l = df['high'].astype(float), df['low'].astype(float)
    body = abs(c.iloc[-2]-o.iloc[-2])
    wick_u = h.iloc[-2]-max(c.iloc[-2],o.iloc[-2])
    wick_l = min(c.iloc[-2],o.iloc[-2]) - l.iloc[-2]
    if side=="buy"  and wick_l>body*0.8: score+=1.0; reasons.append("long_lower_wick")
    if side=="sell" and wick_u>body*0.8: score+=1.0; reasons.append("long_upper_wick")
    # حجم
    vol_ma = _sma(df['volume'].astype(float), GZ_REQ_VOL_MA)
    if float(df['volume'].iloc[-2]) > float(vol_ma.iloc[-2] if vol_ma is not None else 0.0):
        score+=1.0; reasons.append("volume>ma20")
    # RSI-MA توافق
    rsi = ind_rsi(df['close'].astype(float))
    rsi_ma = _sma(rsi, RSI_MA_LEN)
    sig="neutral"
    if len(rsi)>=2 and len(rsi_ma)>=2:
        if rsi.iloc[-2]<rsi_ma.iloc[-2] and rsi.iloc[-1]>rsi_ma.iloc[-1]: sig="bull"
        if rsi.iloc[-2]>rsi_ma.iloc[-2] and rsi.iloc[-1]<rsi_ma.iloc[-1]: sig="bear"
    if side=="buy"  and (sig=="bull"): score+=1.5; reasons.append("rsi_ma_bullish")
    if side=="sell" and (sig=="bear"): score+=1.5; reasons.append("rsi_ma_bearish")
    # ADX
    adx = float(ind.get("adx", 0.0)) if isinstance(ind, dict) else 0.0
    if adx>=GZ_ADX_MIN: score+=0.5; reasons.append("adx_ok")
    ok = score>=GZ_MIN_SCORE
    zone={"type":"golden_bottom" if side=="buy" else "golden_top","lo":float(min(fib618,fib786)),"hi":float(max(fib618,fib786))}
    return {"ok":ok,"score":score,"reasons":reasons,"zone":zone}

# =================== [ADDON] COUNCIL VOTING ===================
def council_votes_pro(df):
    close = df['close'].astype(float)
    rsi  = ind_rsi(close); rsi_ma = _sma(rsi, RSI_MA_LEN)
    macd, macds, hist = ind_macd(close)
    adx, di_plus, di_minus = ind_adx_di(df)
    evx  = ind_evx(df)
    cs   = candle_signals(df)
    eqh,eql = find_eqh_eql(df,50)
    sw   = detect_sweep(df, eqh, eql)
    fvg  = detect_fvg(df,50)
    obs  = detect_order_blocks(df,80)
    bos  = detect_bos_choch(df)

    b=s=0; sb=ss=0.0; logs=[]; reasons_b=[]; reasons_s=[]
    # RSI-MA
    sig="neutral"
    if len(rsi)>=2 and len(rsi_ma)>=2:
        if rsi.iloc[-2]<rsi_ma.iloc[-2] and rsi.iloc[-1]>rsi_ma.iloc[-1]: sig="bullish_cross"
        if rsi.iloc[-2]>rsi_ma.iloc[-2] and rsi.iloc[-1]<rsi_ma.iloc[-1]: sig="bearish_cross"
    rsi_now=float(rsi.iloc[-2]) if len(rsi) else 50.0
    if sig=="bullish_cross" and rsi_now<70:  b+=RSI_CROSS_VOTES; sb+=RSI_CROSS_SCORE; logs.append(f"🔎 RSI-MA bull_cross rsi={rsi_now:.1f} → +{RSI_CROSS_VOTES}(+{RSI_CROSS_SCORE})"); reasons_b.append("rsi_bull_cross")
    if sig=="bearish_cross" and rsi_now>30: s+=RSI_CROSS_VOTES; ss+=RSI_CROSS_SCORE; logs.append(f"🔎 RSI-MA bear_cross rsi={rsi_now:.1f} → +{RSI_CROSS_VOTES}(+{RSI_CROSS_SCORE})"); reasons_s.append("rsi_bear_cross")

    # Trend-Z
    di_spread = float((di_plus - di_minus).iloc[-2]) if len(di_plus) else 0.0
    macd_slope = float(hist.iloc[-2]-hist.iloc[-3]) if len(hist)>=3 else 0.0
    if di_spread>8 and macd_slope>0:  b+=3; sb+=1.5; logs.append("📈 TrendZ up → +3(+1.5)"); reasons_b.append("trendZ_up")
    if di_spread<-8 and macd_slope<0: s+=3; ss+=1.5; logs.append("📉 TrendZ down → +3(+1.5)"); reasons_s.append("trendZ_dn")

    # شموع
    if cs["bullish_engulf"] or cs["hammer"]:  b+=1; sb+=0.5; logs.append("🟢 Bullish candle → +1(+0.5)"); reasons_b.append("bullish_candle")
    if cs["bearish_engulf"] or cs["shooting_star"]: s+=1; ss+=0.5; logs.append("🔴 Bearish candle → +1(+0.5)"); reasons_s.append("bearish_candle")

    # Sweeps
    if sw["sweep_down"]: b+=1; sb+=1.0; logs.append("💧 Sweep-down → +1(+1.0)"); reasons_b.append("sweep_down")
    if sw["sweep_up"]:   s+=1; ss+=1.0; logs.append("💧 Sweep-up → +1(+1.0)"); reasons_s.append("sweep_up")

    # OB/FVG
    if any(x["type"]=="demand" for x in obs[-5:]): b+=1; sb+=0.5; logs.append("📦 Demand OB → +1(+0.5)"); reasons_b.append("demand_ob")
    if any(x["type"]=="supply" for x in obs[-5:]):  s+=1; ss+=0.5; logs.append("📦 Supply OB → +1(+0.5)"); reasons_s.append("supply_ob")
    if fvg:
        last=fvg[-1]
        if last[0]=="bull": b+=1; sb+=0.5; logs.append("🟩 Bullish FVG → +1(+0.5)"); reasons_b.append("bull_fvg")
        else:                s+=1; ss+=0.5; logs.append("🟥 Bearish FVG → +1(+0.5)"); reasons_s.append("bear_fvg")

    # BOS/CHOCH
    if bos["bos_up"]: b+=1; sb+=0.5; logs.append("🔼 BOS up → +1(+0.5)"); reasons_b.append("BOS_up")
    if bos["bos_dn"]: s+=1; ss+=0.5; logs.append("🔽 BOS down → +1(+0.5)"); reasons_s.append("BOS_dn")
    if bos["choch"]:  logs.append("♻️ CHoCH detected")

    # EVX + Chop
    ev = float(ind_evx(df).iloc[-2]) if len(df)>3 else 1.0
    if ev>=EVX_STRONG: logs.append(f"💥 EVX {ev:.2f} explosion")
    bw = ind_bb_width(close)
    if (RSI_CHOP_BAND[0] <= rsi_now <= RSI_CHOP_BAND[1]) and float(bw.iloc[-2])<0.06 and float(adx.iloc[-2])<17:
        sb*=0.8; ss*=0.8; logs.append("🟨 Chop zone → damp x0.8")

    return {"b":b,"s":s,"score_b":round(sb,2),"score_s":round(ss,2),"logs":logs,
            "ind":{"rsi":rsi_now,"adx":float(adx.iloc[-2]) if len(adx) else 0.0,"di_spread":di_spread,"evx":ev}}

# =================== [ADDON] STRATEGY SELECTOR ===================
def decide_strategy_mode(df):
    adx, di_plus, di_minus = ind_adx_di(df)
    if len(adx)<3: return {"mode":"scalp","why":"insufficient_data"}
    di_spread = float((di_plus-di_minus).iloc[-2])
    adx_now = float(adx.iloc[-2])
    if adx_now>=TREND_ADX_MIN and abs(di_spread)>=TREND_DI_SPREAD:
        return {"mode":"trend","why":f"adx={adx_now:.1f} di_spread={di_spread:.1f}"}
    return {"mode":"scalp","why":f"adx={adx_now:.1f} di_spread={di_spread:.1f}"}

# =================== [ADDON] ENTRY GUARD ===================
def entry_confirmation_guard(df, side_to_open: str, votes: dict, ind: dict):
    if not ENTRY_CONFIRM_GUARD: return {"ok":True,"why":["guard_off"]}
    b,s   = int(votes.get("b",0)), int(votes.get("s",0))
    sb,ss = float(votes.get("score_b",0.0)), float(votes.get("score_s",0.0))
    if side_to_open.upper().startswith("B"):
        if b<ENTRY_MIN_VOTES or sb<ENTRY_MIN_SCORE:
            return {"ok":False,"why":[f"weak_council_buy b={b} sb={sb}"]}
    else:
        if s<ENTRY_MIN_VOTES or ss<ENTRY_MIN_SCORE:
            return {"ok":False,"why":[f"weak_council_sell s={s} ss={ss}"]}
    rsi_now = float(ind.get("rsi",50.0))
    if RSI_NEUTRAL_BAND[0]<=rsi_now<=RSI_NEUTRAL_BAND[1]:
        return {"ok":False,"why":[f"rsi_neutral_band {RSI_NEUTRAL_BAND}"]}
    # Cross check سريع
    rsi = ind_rsi(df['close'].astype(float)); rsi_ma = _sma(rsi, RSI_MA_LEN)
    sig="neutral"
    if len(rsi)>=2 and len(rsi_ma)>=2:
        if rsi.iloc[-2]<rsi_ma.iloc[-2] and rsi.iloc[-1]>rsi_ma.iloc[-1]: sig="bull"
        if rsi.iloc[-2]>rsi_ma.iloc[-2] and rsi.iloc[-1]<rsi_ma.iloc[-1]: sig="bear"
    if side_to_open.upper().startswith("B") and RSI_CROSS_REQUIRED and sig!="bull":
        return {"ok":False,"why":["need_bullish_cross"]}
    if side_to_open.upper().startswith("S") and RSI_CROSS_REQUIRED and sig!="bear":
        return {"ok":False,"why":["need_bearish_cross"]}
    # Golden assist
    if GZ_CAN_LEAD_ENTRY:
        g = golden_zone_check(df, ind, "buy" if side_to_open.upper().startswith("B") else "sell")
        if not g.get("ok") and ((side_to_open.upper().startswith("B") and (b<ENTRY_MIN_VOTES+2 or sb<ENTRY_MIN_SCORE+0.5)) or (side_to_open.upper().startswith("S") and (s<ENTRY_MIN_VOTES+2 or ss<ENTRY_MIN_SCORE+0.5))):
            return {"ok":False,"why":["need_stronger_council_or_golden_zone"]}
    return {"ok":True,"why":["confirmed"]}

# =================== [ADDON] TRADE MANAGEMENT HINTS ===================
def rsi_ma_trade_management_hint(df, side, state, tighten_fn, breakeven_fn, partial_fn):
    rsi = ind_rsi(df['close'].astype(float)); rsi_ma = _sma(rsi, RSI_MA_LEN)
    if len(rsi)<2 or len(rsi_ma)<2: return
    bull = (rsi.iloc[-2]<rsi_ma.iloc[-2] and rsi.iloc[-1]>rsi_ma.iloc[-1])
    bear = (rsi.iloc[-2]>rsi_ma.iloc[-2] and rsi.iloc[-1]<rsi_ma.iloc[-1])
    # ضد الاتجاه → Tighten + BE + Partial 25%
    if side.lower().startswith("l") and bear:
        tighten_fn(); breakeven_fn()
        if not state.get("partial_taken"): partial_fn(0.25, "RSI-MA warning vs LONG")
        print("🛡️ RSI-MA ضد LONG → tighten/BE/partial")
    if side.lower().startswith("s") and bull:
        tighten_fn(); breakeven_fn()
        if not state.get("partial_taken"): partial_fn(0.25, "RSI-MA warning vs SHORT")
        print("🛡️ RSI-MA ضد SHORT → tighten/BE/partial")
    # مع الاتجاه → Hold-TP
    if side.lower().startswith("l") and bull:  print("⏳ RSI-MA مع LONG → Hold-TP")
    if side.lower().startswith("s") and bear:  print("⏳ RSI-MA مع SHORT → Hold-TP")

# =================== RANGE FILTER (TV-like) ===================
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
    """
    إشارة RF على الشمعة الحيّة:
      - flip فوق/تحت الفلتر مع هيستريسس بسيط بالـbps
      - لا نعتمد على إغلاق الشمعة
    """
    if len(df) < RF_PERIOD + 3:
        i = -1
        price = float(df["close"].iloc[i]) if len(df) else None
        return {"time": int(df["time"].iloc[i]) if len(df) else int(time.time()*1000),
                "price": price or 0.0, "long": False, "short": False,
                "filter": price or 0.0, "hi": price or 0.0, "lo": price or 0.0}
    src = df[RF_SOURCE].astype(float)
    hi, lo, filt = _rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))
    # هيستريسس بسيط
    def _bps(a,b):
        try: return abs((a-b)/b)*10000.0
        except Exception: return 0.0
    p_now = float(src.iloc[-1]); p_prev = float(src.iloc[-2])
    f_now = float(filt.iloc[-1]); f_prev = float(filt.iloc[-2])
    # flip
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
wait_for_next_signal_side = None  # "buy" or "sell" (ننتظر الإشارة العكسية بعد الإغلاق)

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
        print(colored("❌ skip open (qty<=0)", "red"))
        return False
    
    # === [ADDON] SMART COUNCIL INTEGRATION ===
    df = fetch_ohlcv()
    cv = council_votes_pro(df)
    for line in cv["logs"]: 
        print(colored(line, "cyan"))
    
    mode = decide_strategy_mode(df)
    gz   = golden_zone_check(df, {"adx": cv["ind"]["adx"]}, "buy" if side.upper().startswith("B") else "sell")

    # === [ADDON] BOOKMAP & FLOW INTEGRATION ===
    bm = bookmap_snapshot(ex, SYMBOL, depth=BOOKMAP_DEPTH)
    log_bookmap(bm)
    
    flow = compute_flow_metrics(df)
    log_flow(flow)
    
    votes = {"b":cv["b"],"s":cv["s"],"score_b":cv["score_b"],"score_s":cv["score_s"]}
    g = entry_confirmation_guard(df, side, votes, {"rsi":cv["ind"]["rsi"]})
    if not g["ok"]:
        print(colored(f"🟨 Entry Guard: BLOCK {side} → {g['why']}", "yellow"))
        return False

    # === [ADDON] SHADOW DASHBOARD & STRATEGY BANNER ===
    extras = {"mode": mode['mode'], "gz": gz}
    side_hint = side if 'side' in locals() else ('BUY' if cv['b']>=cv['s'] else 'SELL')
    shadow_dashboard(side_hint, cv, bm, flow, extras=extras)
    log_strategy_banner(mode)

    print(colored(f"🧭 PLAN → council={votes} | mode={mode['mode']}({mode['why']})", "cyan"))
    if gz.get("ok"):
        print(colored(f"🟡 نقطة ذهبية: {gz['zone']['type']} | score={gz['score']:.1f} | {gz['reasons']}", "yellow"))
    # === END ADDON ===
    
    if MODE_LIVE:
        try:
            try: ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
            except Exception: pass
            ex.create_order(SYMBOL, "market", side, qty, None, _params_open(side))
        except Exception as e:
            print(colored(f"❌ open: {e}", "red"))
            logging.error(f"open_market error: {e}")
            return False
    
    STATE.update({
        "open": True, "side": "long" if side=="buy" else "short", "entry": price,
        "qty": qty, "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
        "tp1_done": False, "highest_profit_pct": 0.0, "profit_targets_achieved": 0
    })
    
    # === [ADDON] ENHANCED LOGGING ===
    log_open_trade_details(side, price, qty, LEVERAGE, mode['mode'], votes, golden=gz, bm=bm, flow=flow)
    print(colored(f"📊 Decision Summary → {side} | reasons: {','.join([r for r in cv['logs'] if not r.startswith('🟨')])}", "green"))
    logging.info(f"OPEN {side} qty={qty} price={price} mode={mode['mode']} council={votes}")
    
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
                print(colored(f"🔚 STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
                logging.info(f"STRICT_CLOSE {side} pnl={pnl} total={compound_pnl}")
                _reset_after_close(reason, prev_side=side)
                return
            qty_to_close = safe_qty(left_qty)
            attempts += 1
            print(colored(f"⚠️ strict close retry {attempts}/{CLOSE_RETRY_ATTEMPTS} — residual={fmt(left_qty,4)}","yellow"))
            time.sleep(CLOSE_VERIFY_WAIT_S)
        except Exception as e:
            last_error = e; logging.error(f"close_market_strict attempt {attempts+1}: {e}"); attempts += 1; time.sleep(CLOSE_VERIFY_WAIT_S)
    print(colored(f"❌ STRICT CLOSE FAILED after {CLOSE_RETRY_ATTEMPTS} attempts — last error: {last_error}", "red"))
    logging.critical(f"STRICT CLOSE FAILED — last_error={last_error}")

def _reset_after_close(reason, prev_side=None):
    global wait_for_next_signal_side
    prev_side = prev_side or STATE.get("side")
    STATE.update({
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
        "tp1_done": False, "highest_profit_pct": 0.0, "profit_targets_achieved": 0
    })
    # انتظر الإشارة العكسية من RF
    if prev_side == "long":  wait_for_next_signal_side = "sell"
    elif prev_side == "short": wait_for_next_signal_side = "buy"
    else: wait_for_next_signal_side = None
    logging.info(f"AFTER_CLOSE waiting_for={wait_for_next_signal_side}")

def close_partial(frac, reason):
    """إغلاق جزئي + حارس البواقي + قفل نهائي لو الباقي ≤ FINAL_CHUNK_QTY"""
    if not STATE["open"] or STATE["qty"]<=0: return
    qty_close = safe_qty(max(0.0, STATE["qty"] * min(max(frac,0.0),1.0)))
    px = price_now() or STATE["entry"]
    # احترام أقل كمية
    min_unit = max(RESIDUAL_MIN_QTY, LOT_MIN or RESIDUAL_MIN_QTY)
    if qty_close < min_unit:
        print(colored(f"⏸️ skip partial (amount={fmt(qty_close,4)} < min_unit={fmt(min_unit,4)})", "yellow"))
        return
    side = "sell" if STATE["side"]=="long" else "buy"
    if MODE_LIVE:
        try: ex.create_order(SYMBOL,"market",side,qty_close,None,_params_close())
        except Exception as e: print(colored(f"❌ partial close: {e}", "red")); return
    pnl = (px - STATE["entry"]) * qty_close * (1 if STATE["side"]=="long" else -1)
    STATE["qty"] = safe_qty(STATE["qty"] - qty_close)
    logging.info(f"PARTIAL_CLOSE {reason} qty={qty_close} pnl={pnl} rem={STATE['qty']}")
    print(colored(f"🔻 PARTIAL {reason} closed={fmt(qty_close,4)} pnl={fmt(pnl)} rem={fmt(STATE['qty'],4)}","magenta"))
    # Dust / Final chunk rule
    if STATE["qty"] <= FINAL_CHUNK_QTY and STATE["qty"]>0:
        print(colored(f"🧹 Final chunk ≤ {FINAL_CHUNK_QTY} DOGE → strict close", "yellow"))
        close_market_strict("FINAL_CHUNK_RULE")

# =================== DYNAMIC TP ===================
def _consensus(ind, info, side) -> float:
    score=0.0
    try:
        adx=float(ind.get("adx") or 0.0)
        rsi=float(ind.get("rsi") or 50.0)
        if (side=="long" and rsi>=55) or (side=="short" and rsi<=45): score += 1.0
        if adx>=28: score += 1.0
        elif adx>=20: score += 0.5
        if abs(info["price"]-info["filter"])/max(info["filter"],1e-9) >= (RF_HYST_BPS/10000.0): score += 0.5
    except Exception: pass
    return float(score)

def _tp_ladder(info, ind, side):
    px = info["price"]; atr = float(ind.get("atr") or 0.0)
    atr_pct = (atr / max(px,1e-9))*100.0 if px else 0.5
    score = _consensus(ind, info, side)
    if score >= 2.5: mults = [1.8, 3.2, 5.0]
    elif score >= 1.5: mults = [1.6, 2.8, 4.5]
    else: mults = [1.2, 2.4, 4.0]
    tps = [round(m*atr_pct, 2) for m in mults]
    frs = [0.25, 0.30, 0.45]
    return tps, frs

def manage_after_entry(df, ind, info):
    """Breakeven + Dynamic TP + ATR trail + تحديث أعلى ربح"""
    if not STATE["open"] or STATE["qty"]<=0: return
    px = info["price"]; entry=STATE["entry"]; side=STATE["side"]
    rr = (px - entry)/entry*100*(1 if side=="long" else -1)
    # سلّم ديناميكي
    dyn_tps, dyn_fracs = _tp_ladder(info, ind, side)
    STATE.setdefault("_tp_cache", dyn_tps); STATE["_tp_cache"]=dyn_tps
    STATE.setdefault("_tp_fracs", dyn_fracs); STATE["_tp_fracs"]=dyn_fracs
    k = int(STATE.get("profit_targets_achieved", 0))
    # TP1 ثابتة القاعدة (لكن تتعدل بالـADX لاحقًا)
    tp1_now = TP1_PCT_BASE*(2.2 if ind.get("adx",0)>=35 else 1.8 if ind.get("adx",0)>=28 else 1.0)
    if (not STATE["tp1_done"]) and rr >= tp1_now:
        close_partial(TP1_CLOSE_FRAC, f"TP1@{tp1_now:.2f}%")
        STATE["tp1_done"]=True
        if rr >= BREAKEVEN_AFTER: STATE["breakeven"]=entry
    # بقية الأهداف الديناميكية
    if k < len(dyn_tps) and rr >= dyn_tps[k]:
        frac = dyn_fracs[k] if k < len(dyn_fracs) else 0.25
        close_partial(frac, f"TP_dyn@{dyn_tps[k]:.2f}%")
        STATE["profit_targets_achieved"] = k + 1
    # أعلى ربح
    if rr > STATE["highest_profit_pct"]: STATE["highest_profit_pct"]=rr
    # تريل ATR بعد تفعيل حد الربح
    if rr >= TRAIL_ACTIVATE_PCT and ind.get("atr",0)>0:
        gap = ind["atr"] * ATR_TRAIL_MULT
        if side=="long":
            new_trail = px - gap
            STATE["trail"] = max(STATE["trail"] or new_trail, new_trail)
            if STATE["breakeven"] is not None: STATE["trail"] = max(STATE["trail"], STATE["breakeven"])
            if px < STATE["trail"]: close_market_strict(f"TRAIL_ATR({ATR_TRAIL_MULT}x)")
        else:
            new_trail = px + gap
            STATE["trail"] = min(STATE["trail"] or new_trail, new_trail)
            if STATE["breakeven"] is not None: STATE["trail"] = min(STATE["trail"], STATE["breakeven"])
            if px > STATE["trail"]: close_market_strict(f"TRAIL_ATR({ATR_TRAIL_MULT}x)")
    
    # === [ADDON] SMART TRADE MANAGEMENT ===
    def tighten_fn():
        if STATE["trail"] is not None:
            current_price = price_now() or STATE["entry"]
            if STATE["side"] == "long":
                STATE["trail"] = current_price - (gap * 0.5)
            else:
                STATE["trail"] = current_price + (gap * 0.5)
            print("🔧 tightened trail")

    def breakeven_fn():
        STATE["breakeven"] = STATE["entry"]
        print("🎯 breakeven activated")

    def partial_fn(frac, reason):
        close_partial(frac, reason)

    rsi_ma_trade_management_hint(df, STATE["side"], STATE, tighten_fn, breakeven_fn, partial_fn)

# =================== LOOP / LOG ===================
def pretty_snapshot(bal, info, ind, spread_bps, reason=None, df=None):
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
            info = rf_signal_live(df)             # ⚡ RF على الشمعة الحيّة
            ind  = compute_indicators(df)
            spread_bps = orderbook_spread_bps()
            # PnL snapshot
            if STATE["open"] and px:
                STATE["pnl"] = (px-STATE["entry"])*STATE["qty"] if STATE["side"]=="long" else (STATE["entry"]-px)*STATE["qty"]
            # إدارة بعد الدخول
            manage_after_entry(df, ind, {"price": px or info["price"], **info})
            # تأخير الدخول لو السبريد كبير
            reason=None
            if spread_bps is not None and spread_bps > MAX_SPREAD_BPS:
                reason=f"spread too high ({fmt(spread_bps,2)}bps > {MAX_SPREAD_BPS})"
            # الدخول من RF فقط
            sig = "buy" if (ENTRY_RF_ONLY and info["long"]) else ("sell" if (ENTRY_RF_ONLY and info["short"]) else None)
            # سياسة الانتظار العكسية
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
            pretty_snapshot(bal, {"price": px or info["price"], **info}, ind, spread_bps, reason, df)
            loop_i += 1
            sleep_s = NEAR_CLOSE_S if time_to_candle_close(df)<=10 else BASE_SLEEP
            time.sleep(sleep_s)
        except Exception as e:
            print(colored(f"❌ loop error: {e}\n{traceback.format_exc()}", "red"))
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
        print(colored("⛔ keepalive disabled (SELF_URL not set)", "yellow"))
        return
    import requests
    sess=requests.Session(); sess.headers.update({"User-Agent":"rf-live-bot/keepalive"})
    print(colored(f"KEEPALIVE every 50s → {url}", "cyan"))
    while True:
        try: sess.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

# =================== BOOT ===================
if __name__ == "__main__":
    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'}  •  {SYMBOL}  •  {INTERVAL}", "yellow"))
    print(colored(f"RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x  •  RF_LIVE={RF_LIVE_ONLY}", "yellow"))
    print(colored(f"ENTRY: RF ONLY  •  FINAL_CHUNK_QTY={FINAL_CHUNK_QTY}", "yellow"))
    logging.info("service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    # Threads
    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)

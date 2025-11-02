# -*- coding: utf-8 -*-
"""
DOGE/USDT — Council-Only Pro Trader (Closed-RF Context) — FINAL+
• Council قوي + Zone Planner (Pivot/FVG/Box) + تصنيف قوة الصفقة (Strong/Weak)
• منظومة شموع احترافية (Engulfing, Marubozu, Hammer, Inverted, Doji, Dragonfly, Gravestone)
• ركوب ترند (ADX/DI + MACD hist) + إدارة ربح: TP1/BE/Trail/Partial/Wick Harvest
• إغلاق صارم بذكاء (Opp RF confirmed, Reversal risk, Key level break, Impulse flip)
• Pullback Plan تلقائي بعد أي إغلاق صارم (يدخل بعد التصحيح بدل مطاردة السعر)
• حُرّاس: سبريد/انزلاق/معدل صفقات/تبريد بعد الإغلاق + استئناف بعد Restart
• ترقية: X-Protect بالـ VEI_K + Rate-limit على مستوى الشمعة + min_unit ذكي للجزئيات
• HTTP: / , /metrics , /health , /bookmap
"""

import os, time, math, random, signal, sys, traceback, logging, json, tempfile
from logging.handlers import RotatingFileHandler
from datetime import datetime
from decimal import Decimal, ROUND_DOWN

import pandas as pd
import ccxt
from flask import Flask, jsonify, request

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# ======== ENV / MODE ========
API_KEY    = os.getenv("BINGX_API_KEY", "")
API_SECRET = os.getenv("BINGX_API_SECRET", "")
MODE_LIVE  = bool(API_KEY and API_SECRET)
PORT       = int(os.getenv("PORT", 5000))
SELF_URL   = (os.getenv("SELF_URL") or os.getenv("RENDER_EXTERNAL_URL") or "").strip().rstrip("/")

# ======== Strategy ========
SYMBOL        = "DOGE/USDT:USDT"
INTERVAL      = "15m"
LEVERAGE      = 10
RISK_ALLOC    = 0.60
POSITION_MODE = "oneway"

# Entry switches
USE_RF_ENTRY         = False     # Council-Only Mode (RF كداتا فقط)
RF_AS_CONTEXT_ONLY   = True
BYPASS_WAIT_FOR_COUNCIL = True   # المجلس يتجاوز انتظار نفس جهة RF بعد الإغلاق

# RF (Closed candle — للمرجعية فقط)
RF_SOURCE     = "close"
RF_PERIOD     = 20
RF_MULT       = 3.5
RF_HYST_BPS   = 6.0

# Indicators
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14

# Momentum & Impulse
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIG  = 9
IMPULSE_ADX_MIN  = 25.0
IMPULSE_BODY_ATR = 1.8
IMPULSE_MARUBOZU = 0.75
IMPULSE_VEI      = 2.6
FLIP_COOLDOWN_S  = 45

# Gates/Guards
MAX_SPREAD_BPS      = 8.0
HARD_SPREAD_BPS     = 15.0
PAUSE_ADX_THRESHOLD = 17.0
MAX_TRADES_PER_HOUR = 6
CLOSE_COOLDOWN_S    = 90

# Council thresholds
ENTRY_VOTES_MIN = 7
ENTRY_SCORE_MIN = 4.5
ENTRY_ADX_MIN   = 19.0
EXIT_VOTES_MIN  = 4

# Votes weights
VOTE_SUPPLY_REJECT=2; VOTE_DEMAND_REJECT=2
VOTE_SWEEP=2; VOTE_FVG=1; VOTE_EQ_LEVELS=1
VOTE_RF_CONFIRM=1; VOTE_DI_ADX=1; VOTE_RSI_NEUT_TURN=1
VOTE_BOOKMAP_ACC=1; VOTE_BOOKMAP_SWEEP=1
VOTE_MACD_MOMENTUM=2; VOTE_CANDLE_POWER=1
VOTE_IMPULSE_BONUS=2
VOTE_TRUE_PIVOT_STRONG=3; VOTE_TRUE_PIVOT_WEAK=1
VOTE_TREND_ALIGNMENT=2; VOTE_VOLUME_CONFIRM=1

# Candles & Liquidity votes
VOTE_CANDLE_SET       = 2   # قوة نمط شموعي مُؤكِّد
VOTE_WICK_REJECT      = 1   # رفض بذيول طويلة ضد الاتجاه
VOTE_LIQ_GRAB         = 2   # سحب سيولة (sweep) مؤكد
VOTE_ACCUMULATION     = 1   # تجميع/تصريف حجمي

# Scale-In (تعزيز مركز عند دخول سيولة مع ترند)
MAX_ADDS              = 2
ADD_COOLDOWN_BARS     = 3
ADD_FRAC_OF_REMAIN    = 0.35   # 35% من القدرة المتبقية

# X-Protect
VEI_LOOKBACK = 20
VEI_K        = 2.2  # [UPGRADE] سنستخدمه لحظر الدخول وقت ذعر التقلب

# Execution (slippage)
MAX_SLIP_OPEN_BPS  = 20.0
MAX_SLIP_CLOSE_BPS = 30.0
USE_LIMIT_IOC      = True

# Sizing
SIZE_BUFFER = 0.97
FINAL_CHUNK_QTY  = 50.0
RESIDUAL_MIN_QTY = 9.0

# Management
TP1_PCT_BASE       = 0.40
TP1_CLOSE_FRAC     = 0.40
BREAKEVEN_AFTER    = 0.30
TRAIL_ACTIVATE_PCT = 1.20
ATR_TRAIL_MULT     = 1.6
RATCHET_LOCK_FALLBACK = 0.60

# Exhaustion/Wicks/Chop
EXH_MIN_PROFIT   = 0.35
OPP_RF_HYST_BPS  = 8.0
OPP_STRONG_DEBOUNCE = 2
CHOP_LOOKBACK      = 12
CHOP_ATR_FRAC_MAX  = 0.45
CHOP_ALT_BODY_RATE = 0.55
CHOP_EXIT_PROFIT   = 0.25

# Zone Planner
STRONG_ZONE_MODE      = True
ZONE_PLAN_MAX_BARS    = 12
PULLBACK_ENTRY_ENABLE = True

# Restart
STATE_FILE                  = "state_doge.json"
AUTOSAVE_EVERY_LOOP         = True
AUTOSAVE_ON_ORDER           = True
RESTART_SAFE_BARS_HOLD      = 2
RESTART_STRICT_EXCHANGE_SRC = True

# Pace
BASE_SLEEP   = 5
NEAR_CLOSE_S = 1

# ===== Logging =====
def setup_file_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) and getattr(h,"baseFilename","").endswith("bot.log")
               for h in logger.handlers):
        fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=7, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    print(colored("🗂️ log rotation ready", "cyan"))
setup_file_logging()

# ===== Bookmap Adapter (اختياري) =====
class BookmapAdapter:
    def __init__(self): self.snapshot=[]
    def supply(self, levels): self.snapshot = levels or []
    def evaluate(self, pip: float = 0.0005):
        if not self.snapshot: return {"accumulation":[], "sweep":[], "walls":[]}
        by_bucket={}
        for p,liq,imb,ab in self.snapshot:
            key=round(p/pip); by_bucket.setdefault(key,[]).append((p,liq,imb,ab))
        liqs=[r[1] for r in self.snapshot if r[1] is not None] or [1e-9]
        imbs=[r[2] for r in self.snapshot if r[2] is not None] or [0.0]
        liq_avg=max(1e-9, sum(liqs)/len(liqs)); imb_avg=(sum(imbs)/len(imbs)) if imbs else 0.0
        zones_acc,zones_walls,zones_sweep=[],[],[]
        for rows in by_bucket.values():
            prices=[r[0] for r in rows]; lo,hi=min(prices),max(prices)
            liq_sum=sum(r[1] for r in rows); imb_mean=sum(r[2] for r in rows)/max(len(rows),1)
            ab_hits=sum(1 for r in rows if r[3])
            if liq_sum>5*liq_avg: zones_acc.append((lo,hi))
            if abs(imb_mean)>2*abs(imb_avg): zones_walls.append((lo,hi))
            if ab_hits>=3: zones_sweep.append((lo,hi))
        return {"accumulation":zones_acc, "sweep":zones_sweep, "walls":zones_walls}
bookmap = BookmapAdapter()

# ===== Exchange =====
def make_ex():
    return ccxt.bingx({
        "apiKey": API_KEY, "secret": API_SECRET,
        "enableRateLimit": True, "timeout": 20000,
        "options": {"defaultType":"swap"}
    })
ex = make_ex()
MARKET={}; AMT_PREC=0; LOT_STEP=None; LOT_MIN=None

def load_market_specs():
    global MARKET, AMT_PREC, LOT_STEP, LOT_MIN
    try:
        ex.load_markets()
        MARKET = ex.markets.get(SYMBOL, {})
        AMT_PREC = int((MARKET.get("precision",{}) or {}).get("amount", 0) or 0)
        lims = (MARKET.get("limits",{}) or {}).get("amount",{}) or {}
        LOT_STEP = lims.get("step"); LOT_MIN = lims.get("min")
        print(colored(f"🔧 precision={AMT_PREC}, step={LOT_STEP}, min={LOT_MIN}","cyan"))
    except Exception as e:
        print(colored(f"⚠️ load_market_specs: {e}","yellow"))

def ensure_leverage_mode():
    try:
        try:
            ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
            print(colored(f"✅ leverage set {LEVERAGE}x","green"))
        except Exception as e:
            print(colored(f"⚠️ set_leverage warn: {e}","yellow"))
        print(colored(f"📌 position mode: {POSITION_MODE}","cyan"))
    except Exception as e:
        print(colored(f"⚠️ ensure_leverage_mode: {e}","yellow"))

try:
    load_market_specs()
    ensure_leverage_mode()
except Exception as e:
    print(colored(f"⚠️ exchange init: {e}","yellow"))

# ===== Helpers =====
def _round_amt(q):
    if q is None: return 0.0
    try:
        d=Decimal(str(q))
        if LOT_STEP and isinstance(LOT_STEP,(int,float)) and LOT_STEP>0:
            step=Decimal(str(LOT_STEP))
            d=(d/step).to_integral_value(rounding=ROUND_DOWN)*step
        prec=int(AMT_PREC) if AMT_PREC>=0 else 0
        d=d.quantize(Decimal(1).scaleb(-prec), rounding=ROUND_DOWN)
        if LOT_MIN and isinstance(LOT_MIN,(int,float)) and LOT_MIN>0 and d<Decimal(str(LOT_MIN)): return 0.0
        return float(d)
    except Exception: return max(0.0, float(q))

def safe_qty(q):
    q=_round_amt(q)
    if q<=0: print(colored(f"⚠️ qty invalid after normalize → {q}","yellow"))
    return q

def fmt(v,d=6,na="—"):
    try:
        if v is None or (isinstance(v,float) and (math.isinf(v) or math.isnan(v))): return na
        return f"{float(v):.{d}f}"
    except Exception: return na

def with_retry(fn, tries=3, base=0.4):
    for i in range(tries):
        try: return fn()
        except Exception:
            if i==tries-1: raise
            time.sleep(base*(2**i)+random.random()*0.25)

def fetch_ohlcv(limit=600):
    rows=with_retry(lambda: ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, params={"type":"swap"}))
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

def price_now():
    try:
        t=with_retry(lambda: ex.fetch_ticker(SYMBOL))
        return t.get("last") or t.get("close")
    except Exception: return None

def balance_usdt():
    if not MODE_LIVE: return 100.0
    try:
        b=with_retry(lambda: ex.fetch_balance(params={"type":"swap"}))
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT")
    except Exception: return None

def orderbook_spread_bps():
    try:
        ob=with_retry(lambda: ex.fetch_order_book(SYMBOL, limit=5))
        bid=ob["bids"][0][0] if ob["bids"] else None
        ask=ob["asks"][0][0] if ob["asks"] else None
        if not (bid and ask): return None
        mid=(bid+ask)/2.0
        return ((ask-bid)/mid)*10000.0
    except Exception: return None

def _interval_seconds(iv:str)->int:
    iv=(iv or "").lower().strip()
    if iv.endswith("m"): return int(float(iv[:-1]))*60
    if iv.endswith("h"): return int(float(iv[:-1]))*3600
    if iv.endswith("d"): return int(float(iv[:-1]))*86400
    return 15*60

def time_to_candle_close(df: pd.DataFrame)->int:
    tf=_interval_seconds(INTERVAL)
    if len(df)==0: return tf
    cur=int(df["time"].iloc[-1]); now=int(time.time()*1000)
    nxt=cur+tf*1000
    while nxt<=now: nxt+=tf*1000
    return int(max(0,nxt-now)/1000)

def cancel_all_orders():
    if not MODE_LIVE: return
    try: ex.cancel_all_orders(SYMBOL)
    except Exception as e: logging.warning(f"cancel_all_orders: {e}")

# ===== State =====
STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
    "tp1_done": False, "highest_profit_pct": 0.0,
    "profit_targets_achieved": 0, "opp_votes": 0,
    "_last_entry_ts": 0, "_last_close_ts": 0, "_rf_debounce": 0,
    "_reversal_guard_bars": 0, "_last_flip_ts": 0
}
compound_pnl=0.0
wait_for_next_signal_side=None
RESTART_HOLD_UNTIL_BAR=0
_trades_timestamps=[]

# [UPGRADE] Rate-limit على مستوى الشمعة
_last_open_bar_ts = {"ts": 0}
def can_open_this_bar(bar_ts:int)->bool:
    if not bar_ts: return True
    if _last_open_bar_ts["ts"] == bar_ts:
        return False
    _last_open_bar_ts["ts"] = bar_ts
    return True

def _within_hour_rate_limit()->bool:
    now=time.time()
    while _trades_timestamps and now-_trades_timestamps[0]>3600: _trades_timestamps.pop(0)
    return len(_trades_timestamps) < MAX_TRADES_PER_HOUR
def _mark_trade_timestamp(): _trades_timestamps.append(time.time())

# ===== State persistence =====
def _atomic_write_json(path: str, payload: dict):
    try:
        d=os.path.dirname(path) or "."
        os.makedirs(d, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", delete=False, dir=d, encoding="utf-8") as tmp:
            json.dump(payload, tmp, ensure_ascii=False, separators=(",",":"))
            tmp.flush(); os.fsync(tmp.fileno()); tmp_path=tmp.name
        os.replace(tmp_path, path)
    except Exception as e: logging.error(f"atomic_write_json: {e}")

def save_state(tag=""):
    snap={"STATE": STATE, "compound_pnl": compound_pnl,
          "symbol":SYMBOL, "interval":INTERVAL,
          "ts": int(time.time()*1000), "tag": tag}
    _atomic_write_json(STATE_FILE, snap)

def load_state():
    try:
        if not os.path.exists(STATE_FILE): return None
        with open(STATE_FILE,"r",encoding="utf-8") as f: return json.load(f)
    except Exception as e:
        logging.error(f"load_state: {e}"); return None

# ===== Indicators =====
def wilder_ema(s: pd.Series, n:int): return s.ewm(alpha=1/n, adjust=False).mean()

def compute_indicators(df: pd.DataFrame):
    if len(df) < max(ATR_LEN, RSI_LEN, ADX_LEN) + 3:
        return {"rsi":50.0,"plus_di":0.0,"minus_di":0.0,"dx":0.0,"adx":0.0,"atr":0.0,"vei":1.0,
                "macd":0.0,"macd_signal":0.0,"macd_hist":0.0}
    c,h,l = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float)
    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = wilder_ema(tr, ATR_LEN)

    delta=c.diff(); up=delta.clip(lower=0.0); dn=(-delta).clip(lower=0.0)
    rs = wilder_ema(up, RSI_LEN) / wilder_ema(dn, RSI_LEN).replace(0,1e-12)
    rsi = 100 - (100/(1+rs))

    upm=h.diff(); dnm=l.shift(1)-l
    plus_dm = upm.where((upm>dnm)&(upm>0),0.0)
    minus_dm= dnm.where((dnm>upm)&(dnm>0),0.0)
    plus_di = 100*(wilder_ema(plus_dm,ADX_LEN)/atr.replace(0,1e-12))
    minus_di= 100*(wilder_ema(minus_dm,ADX_LEN)/atr.replace(0,1e-12))
    dx = (100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,1e-12)).fillna(0.0)
    adx = wilder_ema(dx, ADX_LEN)

    rng = (h-l).astype(float)
    try:
        lb = rng.rolling(VEI_LOOKBACK).mean()
        vei = float((rng / lb.replace(0,1e-9)).iloc[-1])
        if math.isinf(vei) or math.isnan(vei): vei = 1.0
    except Exception: vei = 1.0

    ema_fast = c.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = c.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_sig  = macd_line.ewm(span=MACD_SIG, adjust=False).mean()
    macd_hist = macd_line - macd_sig

    i=len(df)-1
    return {"rsi": float(rsi.iloc[i]), "plus_di": float(plus_di.iloc[i]),
            "minus_di": float(minus_di.iloc[i]), "dx": float(dx.iloc[i]),
            "adx": float(adx.iloc[i]), "atr": float(atr.iloc[i]),
            "vei": vei, "macd": float(macd_line.iloc[i]),
            "macd_signal": float(macd_sig.iloc[i]),
            "macd_hist": float(macd_hist.iloc[i])}

# ===== RF (closed, context only) =====
def _ema(s: pd.Series, n:int): return s.ewm(span=n, adjust=False).mean()
def _rng_size(src: pd.Series, qty: float, n:int)->pd.Series:
    avrng=_ema((src-src.shift(1)).abs(), n); wper=(n*2)-1
    return _ema(avrng, wper)*qty
def _rng_filter(src: pd.Series, rsize: pd.Series):
    rf=[float(src.iloc[0])]
    for i in range(1,len(src)):
        prev=rf[-1]; x=float(src.iloc[i]); r=float(rsize.iloc[i]); cur=prev
        if x-r>prev: cur=x-r
        if x+r<prev: cur=x+r
        rf.append(cur)
    filt=pd.Series(rf, index=src.index, dtype="float64")
    return filt+rsize, filt-rsize, filt

def rf_signal_closed(df: pd.DataFrame):
    if len(df) < RF_PERIOD + 4:
        return {"time": int(time.time()*1000), "price": None, "long": False, "short": False,
                "filter": None, "hi": None, "lo": None}
    d = df.iloc[:-1]
    src = d[RF_SOURCE].astype(float)
    hi, lo, filt = _rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))
    p_now = float(src.iloc[-1]); p_prev = float(src.iloc[-2])
    f_now = float(filt.iloc[-1]); f_prev = float(filt.iloc[-2])
    def _bps(a,b): 
        try: return abs((a-b)/b)*10000.0
        except Exception: return 0.0
    long_flip  = (p_prev <= f_prev and p_now > f_now and _bps(p_now, f_now) >= RF_HYST_BPS)
    short_flip = (p_prev >= f_prev and p_now < f_now and _bps(p_now, f_now) >= RF_HYST_BPS)
    return {"time": int(d["time"].iloc[-1]), "price": p_now, "long": bool(long_flip),
            "short": bool(short_flip), "filter": f_now, "hi": float(hi.iloc[-1]), "lo": float(lo.iloc[-1])}

# ===== SMC / Candles =====
def _find_swings(df: pd.DataFrame, left:int=2, right:int=2):
    if len(df) < left+right+3: return None, None
    h=df["high"].astype(float).values; l=df["low"].astype(float).values
    ph=[None]*len(df); pl=[None]*len(df)
    for i in range(left, len(df)-right):
        if all(h[i] >= h[j] for j in range(i-left, i+right+1)): ph[i]=h[i]
        if all(l[i] <= l[j] for j in range(i-left, i+right+1)): pl[i]=l[i]
    return ph, pl

def detect_fvg(df: pd.DataFrame, lookback=40):
    if len(df) < 5: return None
    d = df.iloc[-lookback-2:-1] if len(df)>lookback+2 else df.iloc[:-1]
    res=[]
    for i in range(2, len(d)):
        h1,l1 = float(d["high"].iloc[i-2]), float(d["low"].iloc[i-2])
        h3,l3 = float(d["high"].iloc[i]),   float(d["low"].iloc[i])
        if l3 > h1:  res.append({"type":"bull","gap_top":l3,"gap_bot":h1})
        if h3 < l1:  res.append({"type":"bear","gap_top":l1,"gap_bot":h3})
    return res[-1] if res else None

def detect_sweep(df: pd.DataFrame, lookback=30, bps=8.0):
    if len(df)<5: return None
    d=df.iloc[-lookback:]
    o,h,l,c = map(float, d[["open","high","low","close"]].iloc[-1])
    prev_h = float(d["high"].iloc[-2]); prev_l = float(d["low"].iloc[-2])
    def near(a,b):
        try: return abs((a-b)/b)*10000.0 <= bps
        except: return False
    if h>prev_h and c<prev_h and near(h, prev_h): return {"type":"sweep_high"}
    if l<prev_l and c>prev_l and near(l, prev_l): return {"type":"sweep_low"}
    return None

def detect_trap_wick(df: pd.DataFrame, ratio=0.6):
    if len(df)<3: return None
    o,h,l,c = map(float, df[["open","high","low","close"]].iloc[-1])
    rng=max(h-l,1e-12); up=h-max(o,c); dn=min(o,c)-l
    if (dn/rng)>=ratio and c>o: return {"type":"bull_trap_reject"}
    if (up/rng)>=ratio and c<o: return {"type":"bear_trap_reject"}
    return None

def detect_boxes(df: pd.DataFrame):
    d=df.iloc[:-1] if len(df)>=2 else df
    ph,pl=_find_swings(d,2,2)
    highs=[p for p in ph if p is not None][-20:]
    lows =[p for p in pl if p is not None][-20:]
    sup=dem=None
    if highs:
        top=max(highs); bot=top - (top-min(highs))*0.25
        sup={"side":"supply","top":top,"bot":bot}
    if lows:
        bot=min(lows); top=bot + (max(lows)-bot)*0.25 if len(lows)>1 else bot*1.002
        dem={"side":"demand","top":top,"bot":bot}
    return {"supply":sup,"demand":dem}

def candle_features(df: pd.DataFrame):
    """
    تُعيد:
      - patterns: list[str]  (أسماء الأنماط المكتشفة)
      - body_ratio, up_wick, dn_wick
      - power_bull, power_bear (0..1)
    """
    if len(df) < 3:
        return {"patterns":[], "body_ratio":0.0, "up_wick":0.0, "dn_wick":0.0,
                "power_bull":0.0, "power_bear":0.0}

    o1,h1,l1,c1 = map(float, df[["open","high","low","close"]].iloc[-2])
    o2,h2,l2,c2 = map(float, df[["open","high","low","close"]].iloc[-1])
    rng2 = max(h2-l2, 1e-12); body2 = abs(c2-o2)
    up   = h2 - max(o2,c2);   dn   = min(o2,c2) - l2
    up_r = up/rng2;           dn_r = dn/rng2;   body_r = body2/rng2

    pats=[]
    # Engulfing
    if (c2>o2) and (c1>o1) is False and (o2<=c1) and (c2>=o1): pats.append("BullishEngulf")
    if (c2<o2) and (c1<o1) is False and (o2>=c1) and (c2<=o1): pats.append("BearishEngulf")
    # Marubozu
    if (c2>o2) and body_r>=IMPULSE_MARUBOZU: pats.append("MarubozuUp")
    if (c2<o2) and body_r>=IMPULSE_MARUBOZU: pats.append("MarubozuDown")
    # Hammers / Stars / Doji
    if (dn >= 2.0*max(body2,1e-12)) and (up <= 0.35*max(body2,1e-12)): pats.append("Hammer")
    if (up >= 2.0*max(body2,1e-12)) and (dn <= 0.35*max(body2,1e-12)): pats.append("InvertedHammer")
    if body_r <= 0.1: 
        pats.append("Doji")
        if dn_r>=0.6: pats.append("Dragonfly")
        if up_r>=0.6: pats.append("Gravestone")
    # Piercing / Dark Cloud (مبسّط)
    mid1 = (o1+c1)/2.0
    if (c2>o2) and (c1<o1) and (c2>mid1) and (o2< c1): pats.append("Piercing")
    if (c2<o2) and (c1>o1) and (c2<mid1) and (o2> c1): pats.append("DarkCloud")
    # Three Soldiers / Black Crows (مبسّط)
    if len(df) >= 4:
        closes = df["close"].astype(float).values[-4:]
        opens  = df["open"].astype(float).values[-4:]
        if all(closes[i]>opens[i] for i in [-3,-2,-1]) and closes[-1]>closes[-2]>closes[-3]:
            pats.append("ThreeWhiteSoldiers")
        if all(closes[i]<opens[i] for i in [-3,-2,-1]) and closes[-1]<closes[-2]<closes[-3]:
            pats.append("ThreeBlackCrows")

    power_bull = min(1.0, (1.0 if "BullishEngulf" in pats else 0.0) + (0.5 if "Hammer" in pats else 0.0) + (0.5 if "MarubozuUp" in pats else 0.0))
    power_bear = min(1.0, (1.0 if "BearishEngulf" in pats else 0.0) + (0.5 if "InvertedHammer" in pats else 0.0) + (0.5 if "MarubozuDown" in pats else 0.0))

    return {"patterns":pats, "body_ratio":body_r, "up_wick":up_r, "dn_wick":dn_r,
            "power_bull":power_bull, "power_bear":power_bear}

def structure_trend(df: pd.DataFrame, look=20):
    if len(df) < look+3: return "range"
    d=df.iloc[-look:]
    ph,pl=_find_swings(d,2,2)
    highs=[(i,v) for i,v in enumerate(ph) if v is not None]
    lows =[(i,v) for i,v in enumerate(pl) if v is not None]
    if len(highs)<2 or len(lows)<2: return "range"
    hh = highs[-1][1] > highs[-2][1]
    hl = lows[-1][1]  > lows[-2][1]
    lh = highs[-1][1] < highs[-2][1]
    ll = lows[-1][1]  < lows[-2][1]
    if hh and hl: return "bull"
    if lh and ll: return "bear"
    return "range"

def pivot_candidates(df: pd.DataFrame, look=50):
    if len(df)<look: return None,None
    d=df.iloc[-look:]
    ph,pl=_find_swings(d,2,2)
    last_high = [(i,v) for i,v in enumerate(ph) if v is not None]
    last_low  = [(i,v) for i,v in enumerate(pl) if v is not None]
    sh = last_high[-1][1] if last_high else None
    sl = last_low[-1][1]  if last_low  else None
    return sh, sl

def true_pivot(df: pd.DataFrame, ind: dict, feats: dict, rf: dict, tol_bps=30.0):
    if len(df)<5: return None
    why=[]; adx=float(ind.get("adx") or 0.0); hist=float(ind.get("macd_hist") or 0.0); atr=float(ind.get("atr") or 0.0)
    o,h,l,c = map(float, df[["open","high","low","close"]].iloc[-1])
    sh, sl = pivot_candidates(df, look=50)
    if not atr: atr=1e-9
    sw = detect_sweep(df, lookback=30, bps=tol_bps/2)

    bottom_score=0.0
    if sl:
        if (l < sl and c > sl) or (sw and sw.get("type")=="sweep_low"): bottom_score+=1.2; why.append("sweep_low/close>SL")
        # استخدام الأنماط الجديدة بدل المفاتيح القديمة
        patterns = feats.get("patterns", [])
        if "BullishEngulf" in patterns or "Hammer" in patterns or "Dragonfly" in patterns or "MarubozuUp" in patterns:
            bottom_score+=1.0; why.append("bullish_candle")
        if hist>=0: bottom_score+=0.6; why.append("MACD_hist>=0")
        if adx>=17: bottom_score+=0.6; why.append("ADX>=17")
        if (c-o) >= 1.0*atr: bottom_score+=0.6; why.append("body>=1*ATR")
        if rf.get("long"): bottom_score+=0.5; why.append("RF_long")
        ret_lo = sl; ret_hi = max(min(c,h)-0.25*(h-l), sl)
        if bottom_score>=2.2: return {"type":"bottom","conf":min(1.0,bottom_score/4.0),"retest_zone":(ret_lo,ret_hi),"why":why}

    why=[]; top_score=0.0
    if sh:
        if (h > sh and c < sh) or (sw and sw.get("type")=="sweep_high"): top_score+=1.2; why.append("sweep_high/close<SH")
        # استخدام الأنماط الجديدة بدل المفاتيح القديمة
        patterns = feats.get("patterns", [])
        if "BearishEngulf" in patterns or "InvertedHammer" in patterns or "Gravestone" in patterns or "MarubozuDown" in patterns:
            top_score+=1.0; why.append("bearish_candle")
        if hist<=0: top_score+=0.6; why.append("MACD_hist<=0")
        if adx>=17: top_score+=0.6; why.append("ADX>=17")
        if (o-c) >= 1.0*atr: top_score+=0.6; why.append("body>=1*ATR")
        if rf.get("short"): top_score+=0.5; why.append("RF_short")
        ret_hi = sh; ret_lo = min(max(c,l)+0.25*(h-l), sh)
        if top_score>=2.2: return {"type":"top","conf":min(1.0,top_score/4.0),"retest_zone":(ret_lo,ret_hi),"why":why}
    return None

def detect_retest_displacement(df: pd.DataFrame, ind: dict, mult=1.2, lookback=10):
    if len(df)<lookback+3: return None
    atr=float(ind.get("atr") or 0.0)
    if atr<=0: return None
    d=df.iloc[-lookback:]
    body=(d["close"]-d["open"]).abs().astype(float)
    rng =(d["high"]-d["low"]).astype(float)
    if float(body.iloc[-1]) >= mult*atr and float(rng.iloc[-1]) >= mult*atr:
        prev_low=float(d["low"].iloc[-2]); prev_high=float(d["high"].iloc[-2])
        last_low=float(d["low"].iloc[-1]); last_high=float(d["high"].iloc[-1])
        if last_low<=prev_high or last_high>=prev_low:
            return {"type":"displacement_retest"}
    return None

# ===== وحدات السيولة =====
def liquidity_accumulation(df: pd.DataFrame, win=12):
    """ تجميع/تصريف: نطاقات صغيرة + حجم مرتفع = احتمالية تراكم أو تصريف """
    if len(df) < win+2: return None
    d = df.iloc[-win-1:-1]
    ranges = (d["high"]-d["low"]).astype(float)
    vol    = d["volume"].astype(float)
    small_range = float(ranges.mean())
    vol_boost   = float(vol.mean())
    last_r = float(ranges.iloc[-1]); last_v = float(vol.iloc[-1])
    if last_r <= small_range*0.8 and last_v >= vol_boost*1.25:
        # اتجاه التراكم التقريبي = إغلاق بالنسبة للمدى
        rel_close = (float(d["close"].iloc[-1]) - float(d["low"].iloc[-1])) / max(last_r,1e-12)
        if rel_close >= 0.6:  return {"type":"acc_bull"}  # شراء متراكم
        if rel_close <= 0.4:  return {"type":"acc_bear"}  # تصريف
    return None

def liquidity_inflow_metric(df: pd.DataFrame):
    """ تقدير تدفق السيولة لحظة-بلحظة باستخدام إشارة الحجم × اتجاه الجسم """
    if len(df)<3: return 0.0
    d = df.iloc[-20:]
    vol = d["volume"].astype(float)
    z = (vol - vol.mean())/max(vol.std(ddof=0), 1e-9)
    o,h,l,c = map(float, d[["open","high","low","close"]].iloc[-1])
    dir_ = 1.0 if c>o else -1.0 if c<o else 0.0
    return float(z.iloc[-1]) * dir_

# ===== Zone Planner =====
class ZonePlanner:
    def __init__(self):
        self.active=False; self.side=None; self.lo=None; self.hi=None; self.reason=None; self.bars_left=0
    def set(self, side, lo, hi, reason, bars=ZONE_PLAN_MAX_BARS):
        self.active=True; self.side=side; self.lo=float(min(lo,hi)); self.hi=float(max(lo,hi))
        self.reason=reason; self.bars_left=int(bars)
    def clear(self): self.__init__()
    def on_new_bar(self):
        if not self.active: return
        self.bars_left=max(0, self.bars_left-1)
        if self.bars_left==0: self.clear()
    def ready(self, price)->bool:
        return self.active and price is not None and self.lo<=price<=self.hi

ZONE_PLAN = ZonePlanner()
_last_bar_time_for_plan = 0

def build_zone_from_pivot_or_fvg(df, ind, piv, fvg, boxes):
    try:
        if piv and piv.get("retest_zone"):
            zlo,zhi=piv["retest_zone"]
            if piv.get("type")=="bottom": return ("buy", float(zlo), float(zhi), "PIVOT_RETEST_ZONE")
            if piv.get("type")=="top":    return ("sell", float(zlo), float(zhi), "PIVOT_RETEST_ZONE")
        if fvg:
            if fvg["type"]=="bull": return ("buy", float(fvg["gap_bot"]), float(fvg["gap_top"]), "FVG_FILL")
            else:                   return ("sell", float(fvg["gap_bot"]), float(fvg["gap_top"]), "FVG_FILL")
        sup=(boxes or {}).get("supply"); dem=(boxes or {}).get("demand")
        if sup:
            mid=(sup["top"]+sup["bot"])/2.0; span=(sup["top"]-sup["bot"])*0.25
            return ("sell", float(mid-span), float(mid+span), "BOX_SUPPLY_MID")
        if dem:
            mid=(dem["top"]+dem["bot"])/2.0; span=(dem["top"]-dem["bot"])*0.25 if (dem["top"]>dem["bot"]) else dem["top"]*0.001
            return ("buy", float(mid-span), float(mid+span), "BOX_DEMAND_MID")
    except Exception: pass
    return None

def plan_pullback_after_close(df: pd.DataFrame, ind: dict):
    rf = rf_signal_closed(df)
    boxes = detect_boxes(df)
    fvg = detect_fvg(df)
    trend = structure_trend(df)
    px = float(df["close"].iloc[-1])
    ind_atr = float(ind.get("atr") or 0.0) or 0.0
    pad = max(1e-6, 0.35*ind_atr)
    if trend=="bull" and rf.get("filter"):
        lo=min(rf["filter"], px)-pad; hi=min(rf["filter"], px)+pad
        return ("buy", lo, hi, "PULLBACK_RF_RETEST")
    if trend=="bear" and rf.get("filter"):
        lo=max(rf["filter"], px)-pad; hi=max(rf["filter"], px)+pad
        return ("sell", lo, hi, "PULLBACK_RF_RETEST")
    return build_zone_from_pivot_or_fvg(df, ind, None, fvg, boxes)

# ===== Council =====
class Council:
    def __init__(self):
        self.state={"open":False,"side":None,"entry":None}
        self._last_log=None; self._last_impulse=None; self._last_pivot=None
    def votes(self, df, ind, rf):
        b=s=0; score=0.0; rb=[]; rs=[]
        boxes=detect_boxes(df); sup=boxes.get("supply"); dem=boxes.get("demand")
        feats=candle_features(df)
        bm = bookmap.evaluate()
        if bm["accumulation"]: b += VOTE_BOOKMAP_ACC; score += 0.5; rb.append("BM-acc")
        if bm["sweep"]:        s += VOTE_BOOKMAP_SWEEP; score += 0.5; rs.append("BM-sweep")

        trap = detect_trap_wick(df, 0.6)
        if dem and trap and trap.get("type") == "bull_trap_reject":
            if (df["low"].iloc[-1] <= dem["top"]) and (df["close"].iloc[-1] > (dem["top"]+dem["bot"])/2.0):
                b += VOTE_DEMAND_REJECT; score += 1.6; rb.append("reject@demand")
        if sup and trap and trap.get("type") == "bear_trap_reject":
            if (df["high"].iloc[-1] >= sup["bot"]) and (df["close"].iloc[-1] < (sup["top"]+sup["bot"])/2.0):
                s += VOTE_SUPPLY_REJECT; score += 1.6; rs.append("reject@supply")
                
        sw = detect_sweep(df)
        if sw:
            if sw["type"]=="sweep_low":
                b += VOTE_LIQ_GRAB; score += 0.8; rb.append("liq_grab_low")
            else:
                s += VOTE_LIQ_GRAB; score += 0.8; rs.append("liq_grab_high")

        fvg = detect_fvg(df)
        if fvg:
            if fvg["type"]=="bull": b += VOTE_FVG; score += 0.5; rb.append("FVG(bull)")
            else:                   s += VOTE_FVG; score += 0.5; rs.append("FVG(bear)")

        acc = liquidity_accumulation(df)
        if acc and acc["type"]=="acc_bull":
            b += VOTE_ACCUMULATION; score += 0.4; rb.append("accum(bull)")
        if acc and acc["type"]=="acc_bear":
            s += VOTE_ACCUMULATION; score += 0.4; rs.append("accum(bear)")

        disp=detect_retest_displacement(df, ind, mult=1.2, lookback=10)
        if disp:
            if float(df["close"].iloc[-1])>float(df["open"].iloc[-1]): b+=1; score+=0.7; rb.append("displacement")
            else: s+=1; score+=0.7; rs.append("displacement")
        pdi,mdi,adx = ind.get("plus_di",0), ind.get("minus_di",0), ind.get("adx",0)
        if adx>=18 and pdi>mdi: b+=VOTE_DI_ADX; score+=0.5; rb.append("DI+>DI- & ADX")
        if adx>=18 and mdi>pdi: s+=VOTE_DI_ADX; score+=0.5; rs.append("DI->DI+ & ADX")
        rsi=ind.get("rsi",50.0); o=float(df["open"].iloc[-1]); c=float(df["close"].iloc[-1])
        if 45<=rsi<=55:
            if c>o: b+=VOTE_RSI_NEUT_TURN; score+=0.5; rb.append("RSI_neutral_up")
            else:   s+=VOTE_RSI_NEUT_TURN; score+=0.5; rs.append("RSI_neutral_down")
        hist=float(ind.get("macd_hist") or 0.0)
        if adx>=17 and hist>0: b += VOTE_MACD_MOMENTUM; score += 0.8; rb.append("MACD+ADX↑")
        if adx>=17 and hist<0: s += VOTE_MACD_MOMENTUM; score += 0.8; rs.append("MACD+ADX↓")

        # Candle set vote
        if feats["power_bull"] >= 0.8:
            b += VOTE_CANDLE_SET; score += 0.6; rb.append("candle_set↑")
        if feats["power_bear"] >= 0.8:
            s += VOTE_CANDLE_SET; score += 0.6; rs.append("candle_set↓")

        # Wick rejection ضد الاتجاه الحالي
        if feats["up_wick"]>=0.6 and float(ind.get("macd_hist") or 0.0) <= 0:
            s += VOTE_WICK_REJECT; score += 0.3; rs.append("wick_reject_up")
        if feats["dn_wick"]>=0.6 and float(ind.get("macd_hist") or 0.0) >= 0:
            b += VOTE_WICK_REJECT; score += 0.3; rb.append("wick_reject_down")

        # استخدام الأنماط الجديدة بدل المفاتيح القديمة
        patterns = feats.get("patterns", [])
        if "BullishEngulf" in patterns or "MarubozuUp" in patterns or "Hammer" in patterns or "Dragonfly" in patterns:
            b += VOTE_CANDLE_POWER; score += 0.4; rb.append("candle↑")
        if "BearishEngulf" in patterns or "MarubozuDown" in patterns or "InvertedHammer" in patterns or "Gravestone" in patterns:
            s += VOTE_CANDLE_POWER; score += 0.4; rs.append("candle↓")
            
        impulse = self._last_impulse = detect_impulse(df, ind)
        if impulse and impulse["type"]=="explosion_up":   b += VOTE_IMPULSE_BONUS; score += 0.8; rb.append("IMPULSE↑")
        if impulse and impulse["type"]=="explosion_down": s += VOTE_IMPULSE_BONUS; score += 0.8; rs.append("IMPULSE↓")
        piv = self._last_pivot = true_pivot(df, ind, feats, rf_signal_closed(df))
        if piv and piv["type"]=="bottom":
            add = VOTE_TRUE_PIVOT_STRONG if piv["conf"]>=0.7 else VOTE_TRUE_PIVOT_WEAK
            b += add; score += 1.0 if add==VOTE_TRUE_PIVOT_STRONG else 0.5; rb.append("TRUE_BOTTOM["+ ";".join(piv["why"]) +"]")
        if piv and piv["type"]=="top":
            add = VOTE_TRUE_PIVOT_STRONG if piv["conf"]>=0.7 else VOTE_TRUE_PIVOT_WEAK
            s += add; score += 1.0 if add==VOTE_TRUE_PIVOT_STRONG else 0.5; rs.append("TRUE_TOP["+ ";".join(piv["why"]) +"]")
        trend = structure_trend(df)
        if trend == "bull": b += VOTE_TREND_ALIGNMENT; score += 0.8; rb.append("trend_bull")
        elif trend == "bear": s += VOTE_TREND_ALIGNMENT; score += 0.8; rs.append("trend_bear")
        # Volume confirm بسيط
        try:
            vol_ok = float(df["volume"].iloc[-1]) > float(df["volume"].tail(5).mean())*0.8
        except Exception:
            vol_ok = True
        if vol_ok:
            if b>s: b+=VOTE_VOLUME_CONFIRM; score+=0.3; rb.append("volume_confirm")
            elif s>b: s+=VOTE_VOLUME_CONFIRM; score+=0.3; rs.append("volume_confirm")

        self._last_log=f"🏛 BUY={b} [{', '.join(rb) or '—'}] | SELL={s} [{', '.join(rs) or '—'}] | score={score:.2f} | ADX={ind.get('adx'):.1f} | MACD_hist={hist:.4f}"
        print(colored(self._last_log, "green" if b>s else "red" if s>b else "cyan"))
        return b,s,score

    def strength_bucket(self, b,s,score, ind):
        adx=float(ind.get("adx") or 0.0)
        gap=abs(b-s)
        if score>=ENTRY_SCORE_MIN+1 and adx>=ENTRY_ADX_MIN+5 and gap>=3: return "STRONG"
        if score>=ENTRY_SCORE_MIN and adx>=ENTRY_ADX_MIN: return "NORMAL"
        return "WEAK"

    def decide(self, df, ind, rf):
        b,s,score = self.votes(df, ind, rf)
        adx=float(ind.get("adx") or 0.0)
        entry=None
        
        # Turbo trend (اختياري)
        if not self.state["open"]:
            trend = structure_trend(df)
            hist = float(ind.get("macd_hist") or 0.0)
            if adx >= 25 and score >= (ENTRY_SCORE_MIN - 0.5):
                if trend == "bull" and b >= (ENTRY_VOTES_MIN - 1) and hist >= 0:
                    entry = {"side":"buy","reason":self._last_log + " | TREND_TURBO_BULL"}
                    return {"entry":entry,"exit":None,"log":self._last_log}
                if trend == "bear" and s >= (ENTRY_VOTES_MIN - 1) and hist <= 0:
                    entry = {"side":"sell","reason":self._last_log + " | TREND_TURBO_BEAR"}
                    return {"entry":entry,"exit":None,"log":self._last_log}
        
        if not self.state["open"]:
            # Pause عام للسوق الهادي
            if adx < PAUSE_ADX_THRESHOLD:
                return {"entry":None,"exit":None,"log":self._last_log+" | PAUSE_ADX"}
            # شروط المجلس (مع قوة)
            if b>=ENTRY_VOTES_MIN and score>=ENTRY_SCORE_MIN and adx>=ENTRY_ADX_MIN:
                bucket=self.strength_bucket(b,s,score,ind)
                entry={"side":"buy","reason":self._last_log+f" | {bucket}"}
            elif s>=ENTRY_VOTES_MIN and score>=ENTRY_SCORE_MIN and adx>=ENTRY_ADX_MIN:
                bucket=self.strength_bucket(b,s,score,ind)
                entry={"side":"sell","reason":self._last_log+f" | {bucket}"}
        return {"entry":entry,"exit":None,"log":self._last_log}

    def impulse_flip(self, df, ind, state_side):
        imp = self._last_impulse or detect_impulse(df, ind)
        if not imp: return None
        if state_side=="long" and imp["type"]=="explosion_down": return {"flip":"sell","reason":imp["reason"]}
        if state_side=="short" and imp["type"]=="explosion_up":  return {"flip":"buy","reason":imp["reason"]}
        return None

def detect_impulse(df: pd.DataFrame, ind: dict):
    if len(df) < 3: return None
    atr=float(ind.get("atr") or 0.0); vei=float(ind.get("vei") or 1.0)
    if atr<=0: return None
    o,h,l,c = map(float, df[["open","high","low","close"]].iloc[-1])
    body=abs(c-o); rng=max(h-l,1e-12)
    body_atr_ok=(body >= IMPULSE_BODY_ATR*atr); maru=(body/rng)>=IMPULSE_MARUBOZU
    hist=float(ind.get("macd_hist") or 0.0); adx=float(ind.get("adx") or 0.0)
    if vei>=IMPULSE_VEI and body_atr_ok and maru and adx>=IMPULSE_ADX_MIN:
        if c>o and hist>=0: return {"type":"explosion_up","reason":f"VEI{vei:.2f} body≥{IMPULSE_BODY_ATR}ATR marubozu↑ ADX{adx:.1f}"}
        if c<o and hist<=0: return {"type":"explosion_down","reason":f"VEI{vei:.2f} body≥{IMPULSE_BODY_ATR}ATR marubozu↓ ADX{adx:.1f}"}
    return None

council = Council()

# ===== Orders =====
def _params_open(side):
    if POSITION_MODE=="hedge":
        return {"positionSide":"LONG" if side=="buy" else "SHORT", "reduceOnly":False}
    return {"positionSide":"BOTH","reduceOnly":False}

def _params_close():
    if POSITION_MODE=="hedge":
        return {"positionSide":"LONG" if STATE.get("side")=="long" else "SHORT","reduceOnly":True}
    return {"positionSide":"BOTH","reduceOnly":True}

def _best_quotes():
    ob=with_retry(lambda: ex.fetch_order_book(SYMBOL, limit=5))
    bid=ob["bids"][0][0] if ob["bids"] else None
    ask=ob["asks"][0][0] if ob["asks"] else None
    mid=(bid+ask)/2.0 if (bid and ask) else price_now()
    return bid, ask, mid

def _ioc_price(side: str, mid: float, max_bps: float) -> float:
    if mid is None or mid<=0: return None
    slip = max_bps/10000.0
    return (mid*(1+slip)) if side=="buy" else (mid*(1-slip))

def _create_order_ioc(symbol, side, qty, limit_price, reduce_only=False):
    params={"timeInForce":"IOC","reduceOnly":reduce_only}
    if POSITION_MODE=="hedge":
        params["positionSide"] = "LONG" if (side=="buy") else "SHORT"
    return ex.create_order(symbol, "limit", side, qty, limit_price, params)

def _read_position():
    try:
        poss=with_retry(lambda: ex.fetch_positions(params={"type":"swap"}))
        for p in poss:
            sym=(p.get("symbol") or p.get("info",{}).get("symbol") or "")
            if SYMBOL.split(":")[0] not in sym: continue
            qty=abs(float(p.get("contracts") or p.get("info",{}).get("positionAmt") or 0))
            if qty<=0: return 0.0,None,None
            entry=float(p.get("entryPrice") or p.get("info",{}).get("avgEntryPrice") or 0)
            side_raw=(p.get("side") or p.get("info",{}).get("positionSide") or "").lower()
            side="long" if "long" in side_raw or float(p.get("cost",0))>0 else "short"
            return qty, side, entry
    except Exception as e:
        logging.error(f"_read_position: {e}")
    return 0.0,None,None

def compute_size(balance, price):
    cap=(balance or 0.0)*RISK_ALLOC*LEVERAGE*SIZE_BUFFER
    raw=max(0.0, cap/max(float(price or 0.0),1e-9))
    return safe_qty(raw)

_last_add_bar = {"ts": 0}
def maybe_scale_in(df, ind):
    if not STATE["open"] or STATE["qty"]<=0: return
    # تبريد زمن الشموع
    cur_bar = int(df["time"].iloc[-1])
    if _last_add_bar["ts"] and (STATE["bars"] < ADD_COOLDOWN_BARS or cur_bar == _last_add_bar["ts"]):
        return

    adx = float(ind.get("adx") or 0.0)
    if adx < 25: return  # تعزيز فقط في ترند واضح

    inflow = liquidity_inflow_metric(df)
    side = STATE["side"]
    good_flow = (inflow>0.8 and side=="long") or (inflow<-0.8 and side=="short")
    if not good_flow: return

    # لا نتجاوز MAX_ADDS (نُخزّن عدد التعزيزات داخل STATE)
    adds = int(STATE.get("_adds",0))
    if adds >= MAX_ADDS: return

    bal = balance_usdt(); px = price_now() or float(df["close"].iloc[-1])
    if not px or not bal: return
    # حجم التعزيز = نسبة من القدرة المتبقية
    base_qty = compute_size(bal, px)
    add_qty  = safe_qty(base_qty * ADD_FRAC_OF_REMAIN)
    if add_qty <= 0: return

    ok = open_market("buy" if side=="long" else "sell", add_qty, px, tag="[SCALE_IN:liquidity]")
    if ok:
        STATE["_adds"] = adds+1
        _last_add_bar["ts"] = cur_bar

def open_market(side, qty, price, tag=""):
    if qty<=0: print(colored("❌ skip open (qty<=0)","red")); return False
    
    # يسمح للمجلس/الزون فقط أثناء Reversal-Guard
    if STATE["_reversal_guard_bars"]>0:
        is_council = isinstance(tag, str) and (tag.startswith("[COUNCIL") or tag.startswith("[COUNCIL-ZONE]") or tag.startswith("[IMPULSE]"))
        if not is_council:
            print(colored("⛔ Reversal-Guard active — council-only entries","yellow"))
            return False
            
    spr=orderbook_spread_bps()
    if spr is not None and (spr>HARD_SPREAD_BPS or spr>MAX_SPREAD_BPS):
        print(colored(f"⛔ spread {fmt(spr,2)}bps — guard","yellow")); return False
    if not _within_hour_rate_limit():
        print(colored("⛔ rate-limit: too many trades/hour","yellow")); return False
    _,_,mid=_best_quotes()
    if MODE_LIVE and USE_LIMIT_IOC:
        limit_price=_ioc_price(side, mid, MAX_SLIP_OPEN_BPS)
        try:
            try: ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
            except Exception: pass
            _create_order_ioc(SYMBOL, side, qty, limit_price, reduce_only=False)
        except Exception as e:
            print(colored(f"❌ IOC open fail: {e}","red")); logging.error(e); return False
    elif MODE_LIVE:
        try: ex.create_order(SYMBOL,"market",side,qty,None,_params_open(side))
        except Exception as e: print(colored(f"❌ market open fail: {e}","red")); logging.error(e); return False
    STATE.update({
        "open":True, "side":"long" if side=="buy" else "short", "entry":price,
        "qty":qty, "pnl":0.0, "bars":0, "trail":None, "breakeven":None,
        "tp1_done":False, "highest_profit_pct":0.0, "profit_targets_achieved":0,
        "opp_votes":0, "_last_entry_ts": int(time.time())
    })
    STATE["_adds"] = 0
    _mark_trade_timestamp()
    print(colored(f"🚀 OPEN {('🟩 LONG' if side=='buy' else '🟥 SHORT')} qty={fmt(qty,4)} @ {fmt(price)} {tag}","green" if side=='buy' else 'red'))
    if AUTOSAVE_ON_ORDER: save_state(tag="open")
    return True

def _reset_after_close(reason, prev_side=None):
    global wait_for_next_signal_side
    prev_side = prev_side or STATE.get("side")
    STATE.update({
        "open":False,"side":None,"entry":None,"qty":0.0,"pnl":0.0,"bars":0,
        "trail":None,"breakeven":None,"tp1_done":False,
        "highest_profit_pct":0.0,"profit_targets_achieved":0,
        "opp_votes":0,"_last_close_ts": int(time.time())
    })
    wait_for_next_signal_side = "sell" if prev_side=="long" else "buy" if prev_side=="short" else None
    try:
        if PULLBACK_ENTRY_ENABLE:
            df_plan = fetch_ohlcv(limit=200); ind_plan = compute_indicators(df_plan)
            z = plan_pullback_after_close(df_plan, ind_plan)
            if z:
                z_side, z_lo, z_hi, z_reason = z
                ZONE_PLAN.set(z_side, z_lo, z_hi, f"[{z_reason}] after_close:{reason}", bars=ZONE_PLAN_MAX_BARS)
                print(colored(f"🧭 Pullback plan set: {z_side.upper()} [{fmt(z_lo)}, {fmt(z_hi)}] • {z_reason}", "cyan"))
    except Exception: pass

def close_market_strict(reason="STRICT"):
    global compound_pnl
    exch_qty, exch_side, exch_entry = _read_position()
    if exch_qty<=0 and not STATE.get("open"): return
    if exch_qty<=0 and STATE.get("open"):
        px = price_now() or STATE["entry"]; entry=STATE["entry"]; side=STATE["side"]
        pnl=(px-entry)*STATE["qty"]*(1 if side=="long" else -1); compound_pnl+=pnl
        print(colored(f"🔚 STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
        _reset_after_close(reason, prev_side=side)
        if AUTOSAVE_ON_ORDER: save_state(tag="strict_close"); return
    side_to_close="sell" if exch_side=="long" else "buy"
    qty_to_close=safe_qty(exch_qty)
    attempts=0; last=None
    while attempts<6:
        try:
            if MODE_LIVE and USE_LIMIT_IOC:
                _,_,mid=_best_quotes()
                limit_price=_ioc_price(side_to_close, mid, MAX_SLIP_CLOSE_BPS)
                _create_order_ioc(SYMBOL, side_to_close, qty_to_close, limit_price, reduce_only=True)
            elif MODE_LIVE:
                params=_params_close(); params["reduceOnly"]=True
                ex.create_order(SYMBOL,"market",side_to_close,qty_to_close,None,params)
            time.sleep(2.0)
            left,_,_= _read_position()
            if left<=0:
                px=price_now() or STATE.get("entry") or exch_entry
                entry_px=STATE.get("entry") or exch_entry or px
                side=STATE.get("side") or exch_side
                pnl=(px-entry_px)*exch_qty*(1 if side=="long" else -1); compound_pnl+=pnl
                print(colored(f"🔚 STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
                _reset_after_close(reason, prev_side=side)
                if AUTOSAVE_ON_ORDER: save_state(tag="strict_close")
                return
            qty_to_close=safe_qty(left); attempts+=1
            print(colored(f"⚠️ strict close retry {attempts} residual={fmt(left,4)}","yellow"))
        except Exception as e:
            last=e; attempts+=1; time.sleep(2.0)
    print(colored(f"❌ STRICT CLOSE FAILED last_error={last}","red"))

def close_partial(frac, reason):
    if not STATE["open"] or STATE["qty"]<=0: return
    qty_close=safe_qty(max(0.0, STATE["qty"]*min(max(frac,0.0),1.0)))
    px=price_now() or STATE["entry"]
    # [UPGRADE] min_unit ذكي يعتمد LOT_STEP و LOT_MIN
    step_unit = float(LOT_STEP or 0.0) if isinstance(LOT_STEP,(int,float)) else 0.0
    min_unit = max(RESIDUAL_MIN_QTY, step_unit, float(LOT_MIN or 0.0))
    if qty_close<min_unit:
        print(colored(f"⏸️ skip partial (amount={fmt(qty_close,4)} < min_unit={fmt(min_unit,4)})","yellow")); return
    side="sell" if STATE["side"]=="long" else "buy"
    if MODE_LIVE:
        try: ex.create_order(SYMBOL,"market",side,qty_close,None,_params_close())
        except Exception as e: print(colored(f"❌ partial: {e}","red")); return
    pnl=(px-STATE["entry"])*qty_close*(1 if STATE["side"]=="long" else -1)
    STATE["qty"]=safe_qty(STATE["qty"]-qty_close)
    print(colored(f"🔻 PARTIAL {reason} closed={fmt(qty_close,4)} pnl={fmt(pnl)} rem={fmt(STATE['qty'],4)}","magenta"))
    if STATE["qty"]<=FINAL_CHUNK_QTY and STATE["qty"]>0:
        close_market_strict("FINAL_CHUNK_RULE")

# ===== Reconcile on restart =====
def reconcile_state_with_exchange():
    global compound_pnl, RESTART_HOLD_UNTIL_BAR
    loaded=load_state()
    exch_qty, exch_side, exch_entry=_read_position()
    exch_open = exch_qty>0 and exch_side in ("long","short") and exch_entry and exch_entry>0
    if loaded and (loaded.get("symbol")==SYMBOL and loaded.get("interval")==INTERVAL):
        st=loaded.get("STATE") or {}
        STATE["highest_profit_pct"]=float(st.get("highest_profit_pct") or 0.0)
        STATE["trail"]=st.get("trail"); STATE["breakeven"]=st.get("breakeven")
        STATE["profit_targets_achieved"]=int(st.get("profit_targets_achieved") or 0)
        try: globals()["compound_pnl"]=float(loaded.get("compound_pnl") or 0.0)
        except Exception: pass
        print(colored("💾 loaded local state (non-pos)","cyan"))
    if RESTART_STRICT_EXCHANGE_SRC:
        if exch_open:
            STATE.update({"open":True,"side":exch_side,"entry":float(exch_entry),
                          "qty":float(exch_qty),"pnl":0.0,"bars":0})
            px=price_now() or exch_entry
            STATE["pnl"]=(px-exch_entry)*exch_qty*(1 if exch_side=="long" else -1)
            RESTART_HOLD_UNTIL_BAR=RESTART_SAFE_BARS_HOLD
            print(colored(f"♻️ resumed live position {exch_side} qty={fmt(exch_qty,4)} entry={fmt(exch_entry)}","yellow"))
        else:
            STATE.update({"open":False,"side":None,"entry":None,"qty":0.0,"pnl":0.0})
            print(colored("♻️ no live position — flat","yellow"))
    save_state(tag="reconcile_boot")

# ===== Management (pro) =====
def _tp_ladder(info, ind, side):
    px=info["price"]; atr=float(ind.get("atr") or 0.0)
    atr_pct=(atr/max(px,1e-9))*100.0 if px else 0.5
    adx=float(ind.get("adx") or 0.0)
    rsi=float(ind.get("rsi") or 50.0)
    score = (1.0 if adx>=28 else 0.5 if adx>=20 else 0.0) + (1.0 if (side=="long" and rsi>=55) or (side=="short" and rsi<=45) else 0.0)
    mults = [1.8,3.2,5.0] if score>=2.0 else [1.6,2.8,4.5] if score>=1.0 else [1.2,2.4,4.0]
    return [round(m*atr_pct,2) for m in mults],[0.25,0.30,0.45]

def is_chop(df: pd.DataFrame, atr: float) -> bool:
    if len(df) < CHOP_LOOKBACK+2 or atr<=0: return False
    d = df.iloc[-CHOP_LOOKBACK-1:-1]
    ranges = (d["high"]-d["low"]).astype(float)
    avg_range = float(ranges.mean())
    if avg_range <= CHOP_ATR_FRAC_MAX * atr:
        bodies = (d["close"]>d["open"]).astype(int).values
        altern = sum(bodies[i]!=bodies[i-1] for i in range(1,len(bodies))) / max(len(bodies)-1,1)
        return altern >= CHOP_ALT_BODY_RATE
    return False

def should_take_reversal_profit(df: pd.DataFrame, ind: dict, side: str, rr_pct: float) -> bool:
    adx = float(ind.get("adx") or 0.0)
    if rr_pct <= 0.0: return False
    o,h,l,c = map(float, df[["open","high","low","close"]].iloc[-1])
    rng=max(h-l,1e-12); up=h-max(o,c); dn=min(o,c)-l
    wick_against = (side=="long" and up/rng>=0.6) or (side=="short" and dn/rng>=0.6)
    sw = detect_sweep(df)
    return wick_against and (adx<18 or sw is not None)

def defensive_on_opposite_rf(ind: dict, info: dict):
    if not STATE["open"] or STATE["qty"]<=0: return
    STATE["opp_votes"]=int(STATE.get("opp_votes",0))+1
    adx=float(ind.get("adx") or 0.0); px=info.get("price"); rf=info.get("filter")
    hyst=0.0
    try:
        if px and rf: hyst = abs((px-rf)/rf)*10000.0
    except Exception: pass
    if STATE["opp_votes"]>=OPP_STRONG_DEBOUNCE and adx>=28 and hyst>=OPP_RF_HYST_BPS:
        close_market_strict("OPPOSITE_RF_CONFIRMED")

def smart_position_management(df, ind, info):
    if not STATE["open"]: return
    current_price = price_now() or float(df["close"].iloc[-1])
    entry_price = STATE["entry"]; side = STATE["side"]
    pnl_pct = ((current_price - entry_price)/entry_price*100) if side=="long" else ((entry_price - current_price)/entry_price*100)

    # 1) خروج ذكي
    reason = advanced_exit_strategy(df, ind, info, pnl_pct)
    if reason:
        close_market_strict(reason); return

    # 2) جني أرباح ديناميكي + Wick/Long Candle
    adx=float(ind.get("adx") or 0.0)
    trend_strength = min(adx / 50.0, 1.0)
    base_tp = TP1_PCT_BASE * (1 + trend_strength)
    if not STATE["tp1_done"] and pnl_pct >= base_tp:
        close_partial(TP1_CLOSE_FRAC, f"SMART_TP1@{base_tp:.2f}%"); STATE["tp1_done"]=True
        if pnl_pct>=BREAKEVEN_AFTER: STATE["breakeven"]=entry_price

    dyn_tps,dyn_fracs=_tp_ladder(info, ind, side)
    k=int(STATE.get("profit_targets_achieved",0))
    if k<len(dyn_tps) and pnl_pct>=dyn_tps[k]:
        close_partial(dyn_fracs[k], f"TP_dyn@{dyn_tps[k]:.2f}%")
        STATE["profit_targets_achieved"]=k+1

    # Wick harvest / Long candle against position → اغلاق صارم لو الربح محترم
    if should_take_reversal_profit(df, ind, side, pnl_pct) and pnl_pct>=EXH_MIN_PROFIT:
        print(colored(f"🔄 Reversal risk → lock profit {pnl_pct:.2f}%","yellow"))
        close_market_strict("REVERSAL_LOCK"); STATE["_reversal_guard_bars"]=4

    # Ratchet / Trail
    if pnl_pct>STATE["highest_profit_pct"]: STATE["highest_profit_pct"]=pnl_pct
    if STATE["highest_profit_pct"]>=TRAIL_ACTIVATE_PCT and pnl_pct<STATE["highest_profit_pct"]*RATCHET_LOCK_FALLBACK:
        close_partial(0.50, f"Ratchet {STATE['highest_profit_pct']:.2f}%→{pnl_pct:.2f}%")
    atr=float(ind.get("atr") or 0.0); px=current_price
    if pnl_pct>=TRAIL_ACTIVATE_PCT and atr>0:
        gap=atr*ATR_TRAIL_MULT
        if side=="long":
            new=px-gap; STATE["trail"]=max(STATE["trail"] or new, new)
            if STATE["breakeven"] is not None: STATE["trail"]=max(STATE["trail"], STATE["breakeven"])
            if px<STATE["trail"]: close_market_strict(f"TRAIL_ATR({ATR_TRAIL_MULT}x)")
        else:
            new=px+gap; STATE["trail"]=min(STATE["trail"] or new, new)
            if STATE["breakeven"] is not None: STATE["trail"]=min(STATE["trail"], STATE["breakeven"])
            if px>STATE["trail"]: close_market_strict(f"TRAIL_ATR({ATR_TRAIL_MULT}x)")

def advanced_exit_strategy(df, ind, info, pnl_pct):
    side = STATE["side"]
    # 1) Council opposite (بديل مؤقت: opposite RF)
    rf = rf_signal_closed(df)
    opp = (side=="long" and rf["short"]) or (side=="short" and rf["long"])
    if opp and pnl_pct>0.5:
        return "COUNCIL_EXIT_PROXY_OPP_RF"
    # 2) تغيّر حالة السوق (ضعف اتجاه ضد الصفقة)
    adx=float(ind.get("adx") or 0.0); pdi=float(ind.get("plus_di") or 0.0); mdi=float(ind.get("minus_di") or 0.0)
    if side=="long" and adx>=25 and mdi>pdi and pnl_pct>-1.5: return "MARKET_CONDITION_LONG_WEAK"
    if side=="short" and adx>=25 and pdi>mdi and pnl_pct>-1.5: return "MARKET_CONDITION_SHORT_WEAK"
    return None

# ===== UI =====
def pretty_snapshot(bal, info, ind, spread_bps, reason=None, df=None, council_log=None):
    left_s=time_to_candle_close(df) if df is not None else 0
    print(colored("─"*110,"cyan"))
    print(colored(f"📊 {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
    print(colored("─"*110,"cyan"))
    print("📈 RF CLOSED (context only)" if not USE_RF_ENTRY else "📈 RF (entry enabled)")
    print(f"   💲 Price {fmt(info.get('price'))} | filt={fmt(info.get('filter'))} hi={fmt(info.get('hi'))} lo={fmt(info.get('lo'))} | spread={fmt(spread_bps,2)}bps")
    print(f"   🧮 RSI={fmt(ind.get('rsi'))} +DI={fmt(ind.get('plus_di'))} -DI={fmt(ind.get('minus_di'))} ADX={fmt(ind.get('adx'))} ATR={fmt(ind.get('atr'))} VEI~{fmt(ind.get('vei'),2)} MACD_hist={fmt(ind.get('macd_hist'),4)}")
    
    # 🕯️ CANDLE SUITE
    cfeat = candle_features(df)
    pats = cfeat.get("patterns", [])
    pat_str = ", ".join(pats) if pats else "—"
    print(colored("🕯️ CANDLE SUITE", "white"))
    print(f"   body={fmt(cfeat['body_ratio'],2)} up_wick={fmt(cfeat['up_wick'],2)} dn_wick={fmt(cfeat['dn_wick'],2)}")
    if pats: 
        print(colored(f"   🎭 patterns: {pat_str}", "magenta"))
    print(f"   power_bull={cfeat['power_bull']:.2f} | power_bear={cfeat['power_bear']:.2f}")

    # 💧 LIQUIDITY
    sw = detect_sweep(df); acc = liquidity_accumulation(df); inflow = liquidity_inflow_metric(df)
    liq_tags=[]
    if sw: liq_tags.append("sweep_low" if sw["type"]=="sweep_low" else "sweep_high")
    if acc: liq_tags.append(acc["type"])
    liq_tags.append(f"inflow~{inflow:+.2f}")
    print(colored("💧 LIQUIDITY", "white"))
    print(colored(f"   signals: {', '.join(liq_tags)}", "cyan"))
    
    if council_log: print(colored(council_log,"white"))
    if council._last_pivot and council._last_pivot.get("retest_zone"):
        z=council._last_pivot["retest_zone"]
        print(colored(f"   🔁 Retest zone: [{fmt(z[0])}, {fmt(z[1])}] • {council._last_pivot['type'].upper()} conf={council._last_pivot.get('conf',0):.2f}","yellow"))
    if ZONE_PLAN.active:
        print(colored(f"   🎯 ZONE PLAN: {ZONE_PLAN.side.upper()} [{fmt(ZONE_PLAN.lo)}, {fmt(ZONE_PLAN.hi)}] • {ZONE_PLAN.reason} • bars_left={ZONE_PLAN.bars_left}","cyan"))
    print(f"   ⏱️ closes_in ≈ {left_s}s")
    print("\n🧭 POSITION")
    bal_line=f"Balance={fmt(bal,2)} Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x CompoundPnL={fmt(compound_pnl)} Eq~{fmt((bal or 0)+compound_pnl,2)}"
    print(colored(f"   {bal_line}","yellow"))
    if STATE["open"]:
        lamp='🟩 LONG' if STATE['side']=='long' else '🟥 SHORT'
        print(f"   {lamp} Entry={fmt(STATE['entry'])} Qty={fmt(STATE['qty'],4)} Bars={STATE['bars']} Trail={fmt(STATE['trail'])} BE={fmt(STATE['breakeven'])}")
        print(f"   🎯 TP_done={STATE['profit_targets_achieved']} HP={fmt(STATE['highest_profit_pct'],2)}% OppVotes={STATE.get('opp_votes',0)} GuardBars={STATE.get('_reversal_guard_bars',0)}")
        print(f"   🔋 Scale-Ins: {STATE.get('_adds', 0)}/{MAX_ADDS}")
    else:
        print("   ⚪ FLAT")
        if wait_for_next_signal_side: print(colored(f"   ⏳ Waiting same-side RF: {wait_for_next_signal_side.upper()}", "cyan"))
    if reason: print(colored(f"   ℹ️ reason: {reason}","white"))
    print(colored("─"*110,"cyan"))

# ===== Main loop =====
app=Flask(__name__)

def trade_loop():
    global wait_for_next_signal_side, RESTART_HOLD_UNTIL_BAR, _last_bar_time_for_plan
    reconcile_state_with_exchange()
    last_decision_bar_time=0

    while True:
        try:
            bal=balance_usdt()
            df = fetch_ohlcv()
            ind=compute_indicators(df)
            rf = rf_signal_closed(df)
            spread=orderbook_spread_bps()
            px=price_now() or rf["price"] or STATE.get("entry") or 0.0

            if STATE["open"] and px:
                STATE["pnl"]=(px-STATE["entry"])*STATE["qty"]*(1 if STATE["side"]=="long" else -1)

            council_decision = council.decide(df, ind, rf)
            council_log = council_decision.get("log")

            # إدارة الصفقة أثناء الفتح
            if STATE["open"]:
                opp = (STATE["side"]=="long" and rf["short"]) or (STATE["side"]=="short" and rf["long"])
                if opp: defensive_on_opposite_rf(ind, {"price":px, **rf})
                rr = (px-STATE["entry"])/STATE["entry"]*100*(1 if STATE["side"]=="long" else -1)
                atr=float(ind.get("atr") or 0.0)
                if is_chop(df, atr) and rr>=CHOP_EXIT_PROFIT:
                    print(colored(f"⚪ Chop → light profit {rr:.2f}%","yellow"))
                    close_market_strict("CHOP_EXIT"); STATE["_reversal_guard_bars"]=4
                flip = council.impulse_flip(df, ind, STATE["side"])
                if flip and (time.time()-STATE.get("_last_flip_ts",0) >= FLIP_COOLDOWN_S):
                    if spread is None or (spread <= MAX_SPREAD_BPS and spread <= HARD_SPREAD_BPS):
                        # [UPGRADE] block double-open on same bar for flip as well
                        if can_open_this_bar(rf["time"]):
                            print(colored(f"⚡ IMPULSE FLIP → {flip['flip'].upper()} ({flip['reason']})","magenta"))
                            close_market_strict("IMPULSE_FLIP")
                            STATE["_last_flip_ts"] = int(time.time())
                            qty=compute_size(bal, px)
                            if qty>0: 
                                open_market(flip["flip"], qty, px, tag="[IMPULSE]"); wait_for_next_signal_side=None
                        else:
                            print(colored("⏸️ flip blocked: same-bar rate-limit","yellow"))
                    else:
                        print(colored(f"⏸️ IMPULSE flip blocked by spread {fmt(spread,2)}bps","yellow"))
                # إدارة أرباح ذكية
                smart_position_management(df, ind, {"price":px, **rf})
                # بعد الإدارة، جرّب تعزيز ذكي إذا الترند قوي وتدفّق السيولة في صالحنا
                maybe_scale_in(df, ind)

            # منع مبكر عام
            reason=None
            if spread is not None and spread>HARD_SPREAD_BPS:
                reason=f"hard spread guard {fmt(spread,2)}bps>{HARD_SPREAD_BPS}"
            elif spread is not None and spread>MAX_SPREAD_BPS:
                reason=f"spread guard {fmt(spread,2)}bps>{MAX_SPREAD_BPS}"

            # [UPGRADE] X-Protect: ذعر VEI — إيقاف فتح صفقات هذا البار
            if reason is None:
                try:
                    vei=float(ind.get("vei") or 1.0)
                    if vei > VEI_K*2.8:  # عتبة حذرة
                        reason=f"X-Protect VEI spike ~{vei:.2f}"
                except Exception:
                    pass

            if reason is None and (float(ind.get("adx") or 0.0)<PAUSE_ADX_THRESHOLD):
                reason=f"ADX<{PAUSE_ADX_THRESHOLD:.0f} — PAUSE"

            decision_time = rf["time"]
            new_bar = decision_time != last_decision_bar_time
            if new_bar:
                last_decision_bar_time = decision_time
                if STATE["_reversal_guard_bars"]>0: STATE["_reversal_guard_bars"]-=1
                if RESTART_HOLD_UNTIL_BAR>0: RESTART_HOLD_UNTIL_BAR-=1
                if int(df["time"].iloc[-1]) != _last_bar_time_for_plan:
                    ZONE_PLAN.on_new_bar()
                _last_bar_time_for_plan = int(df["time"].iloc[-1])

                if not STATE["open"] and reason is None and RESTART_HOLD_UNTIL_BAR<=0:
                    # 1) ZONE_PLAN
                    zone_triggered=False
                    if ZONE_PLAN.active:
                        px_now = price_now() or rf["price"] or float(df["close"].iloc[-1])
                        if ZONE_PLAN.ready(px_now):
                            zone_sig = ZONE_PLAN.side
                            zone_tag = f"[COUNCIL-ZONE] {ZONE_PLAN.reason}"
                            # احترام انتظار RF إلا لو المجلس bypass
                            if wait_for_next_signal_side and zone_sig != wait_for_next_signal_side:
                                if not (BYPASS_WAIT_FOR_COUNCIL):
                                    reason = f"waiting same-side RF: need {wait_for_next_signal_side.upper()}"
                            if reason is None:
                                if (time.time()-STATE.get("_last_close_ts",0)) < CLOSE_COOLDOWN_S:
                                    reason = f"cooldown {int(CLOSE_COOLDOWN_S - (time.time()-STATE.get('_last_close_ts',0)))}s"
                                elif not _within_hour_rate_limit():
                                    reason = "rate-limit trades/hour"
                                elif not can_open_this_bar(decision_time):  # [UPGRADE]
                                    reason = "bar-opened-already"
                                else:
                                    qty = compute_size(bal, px_now)
                                    if qty>0 and px_now:
                                        if open_market(zone_sig, qty, px_now, zone_tag):
                                            wait_for_next_signal_side=None
                                            ZONE_PLAN.clear(); zone_triggered=True
                                    else:
                                        reason = "qty<=0 or price=None"

                    # 2) مجلس/RF
                    if not zone_triggered and reason is None:
                        sig=None; tag=""
                        if council_decision.get("entry"):
                            sig=council_decision["entry"]["side"]; tag=f"[COUNCIL] {council_decision['entry']['reason']}"
                        elif USE_RF_ENTRY:
                            if ((rf["long"] or rf["short"]) and float(ind.get("adx") or 0.0)>=PAUSE_ADX_THRESHOLD):
                                sig="buy" if rf["long"] else "sell"; tag=f"[RF-closed]"
                        if sig:
                            is_council_sig = tag.startswith("[COUNCIL]") or tag.startswith("[COUNCIL-ZONE]")
                            if wait_for_next_signal_side and sig != wait_for_next_signal_side:
                                if not (BYPASS_WAIT_FOR_COUNCIL and is_council_sig):
                                    reason = f"waiting same-side RF: need {wait_for_next_signal_side.upper()}"
                            if reason is None:
                                if (time.time()-STATE.get("_last_close_ts",0)) < CLOSE_COOLDOWN_S:
                                    reason = f"cooldown {int(CLOSE_COOLDOWN_S - (time.time()-STATE.get('_last_close_ts',0)))}s"
                                elif not _within_hour_rate_limit():
                                    reason = "rate-limit trades/hour"
                                elif not can_open_this_bar(decision_time):  # [UPGRADE]
                                    reason = "bar-opened-already"
                                else:
                                    qty=compute_size(bal, px or rf["price"])
                                    if qty>0 and (px or rf["price"]):
                                        if open_market(sig, qty, px or rf["price"], tag):
                                            wait_for_next_signal_side=None
                                            if ZONE_PLAN.active: ZONE_PLAN.clear()
                                    else:
                                        reason="qty<=0 or price=None"

            pretty_snapshot(bal, {"price":px, **rf}, ind, spread, reason, df, council_log)
            if len(df)>=2 and int(df["time"].iloc[-1])!=int(df["time"].iloc[-2]) and STATE["open"]:
                STATE["bars"]+=1
            if AUTOSAVE_EVERY_LOOP: save_state(tag="loop")
            time.sleep(NEAR_CLOSE_S if time_to_candle_close(df)<=10 else BASE_SLEEP)
        except Exception as e:
            print(colored(f"❌ loop error: {e}\n{traceback.format_exc()}","red"))
            logging.error(f"loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

# ===== HTTP =====
@app.route("/")
def home():
    mode='LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ Council-Only Pro Trader — {SYMBOL} {INTERVAL} — {mode} — IOC/Slippage — Restart-safe"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol":SYMBOL,"interval":INTERVAL,"mode":"live" if MODE_LIVE else "paper",
        "leverage":LEVERAGE,"risk_alloc":RISK_ALLOC,"price":price_now(),
        "state":STATE,"compound_pnl":compound_pnl,
        "council_log": council._last_log,
        "last_pivot": council._last_pivot,
        "zone_plan": {"active":ZONE_PLAN.active,"side":ZONE_PLAN.side,"lo":ZONE_PLAN.lo,"hi":ZONE_PLAN.hi,"bars_left":ZONE_PLAN.bars_left,"reason":ZONE_PLAN.reason},
        "guards":{"max_spread_bps":MAX_SPREAD_BPS,"hard_spread_bps":HARD_SPREAD_BPS,"pause_adx":PAUSE_ADX_THRESHOLD,"vei_k":VEI_K}
    })

@app.route("/health")
def health():
    return jsonify({"ok":True,"ts":datetime.utcnow().isoformat(),"open":STATE["open"],"side":STATE["side"],"qty":STATE["qty"]}),200

@app.route("/bookmap", methods=["POST"])
def bookmap_feed():
    try:
        payload = request.get_json(silent=True) or {}
        levels = payload.get("levels", [])
        parsed=[]
        for row in levels:
            p=float(row[0]); liq=float(row[1]); imb=float(row[2]); ab=int(row[3])
            parsed.append((p,liq,imb,ab))
        bookmap.supply(parsed)
        return jsonify({"ok":True,"count":len(parsed)})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)}),400

def keepalive_loop():
    if not SELF_URL:
        print(colored("⛔ keepalive disabled (no SELF_URL/RENDER_EXTERNAL_URL)","yellow")); return
    import requests
    sess=requests.Session(); sess.headers.update({"User-Agent":"council-pro/keepalive"})
    print(colored(f"KEEPALIVE every 50s → {SELF_URL}","cyan"))
    while True:
        try: sess.get(SELF_URL, timeout=8)
        except Exception: pass
        time.sleep(50)

# ===== Boot =====
if __name__=="__main__":
    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'} • {SYMBOL} • {INTERVAL}","yellow"))
    print(colored(f"RISK: {int(RISK_ALLOC*100)}%×{LEVERAGE}x • ENTRY: Council-Only (RF as context)","yellow"))
    print(colored("🎯 Smart Council + Candle System + Zone Planner + Trend Rider + Strict Close (+VEI_K/bar-limit/min_unit)","green"))
    logging.info("service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)

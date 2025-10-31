# -*- coding: utf-8 -*-
"""
COUNCIL-ONLY PRO — BingX USDT Perps (DOGE/USDT:USDT) — Single Position
• Entry/Exit: Smart Council only (VWAP/MACD/Delta/SMC/Bookmap) — NO RF waiting
• ADX Gate ≥ 17 — Spread/Slippage guards — Circuit Breaker + Retries
• Smart management: TP1 partial -> Breakeven -> ATR Trail (Ratchet) -> Strict Close
• Chop exit: VEI<0.8 & ADX<18 & PnL>0 → Close & wait
• Hard SL% emergency
• Resume after restart (sync with exchange) — /metrics /health /bookmap /logs
"""

import os, sys, time, math, json, random, signal, traceback, logging, tempfile, functools, threading
from logging.handlers import RotatingFileHandler
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from collections import deque

import pandas as pd
import numpy as np
import ccxt
from flask import Flask, jsonify, request

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# ===================== CONFIG =====================
SYMBOL         = os.getenv("SYMBOL", "DOGE/USDT:USDT")
TIMEFRAME      = os.getenv("TIMEFRAME", "15m")
LEVERAGE       = int(os.getenv("LEVERAGE", "10"))
RISK_FRAC      = float(os.getenv("RISK_FRAC", "0.60"))   # 60% من الرصيد
MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS","6.0"))
HARD_SPREAD_BPS= float(os.getenv("HARD_SPREAD_BPS","15.0"))
SLIP_OPEN_BPS  = float(os.getenv("SLIP_OPEN_BPS","20.0"))
SLIP_CLOSE_BPS = float(os.getenv("SLIP_CLOSE_BPS","30.0"))

# Council thresholds
ENTRY_VOTES_MIN = int(os.getenv("COUNCIL_VOTES","6"))
ENTRY_SCORE_MIN = float(os.getenv("COUNCIL_SCORE_MIN","2.5"))
ENTRY_ADX_MIN   = float(os.getenv("ENTRY_ADX_MIN","17.0"))   # طلبك
EXIT_VOTES_MIN  = 4

# Indicators
RSI_LEN=14; ADX_LEN=14; ATR_LEN=14
MACD_FAST=12; MACD_SLOW=26; MACD_SIG=9
VEI_FAST=5; VEI_SLOW=30

# Management
TP1_FRAC=0.30
BREAKEVEN_AFTER=0.0030        # 0.30% (قيمة عشرية)
TRAIL_ACTIVATE_PCT=0.012       # 1.2% (قيمة عشرية)
ATR_TRAIL_MULT=1.6
HARD_SL_PCT=0.018              # 1.8% طوارئ إغلاق صارم

# Pacing & limits
BASE_SLEEP=5; NEAR_CLOSE_SLEEP=1
MAX_TRADES_PER_HOUR=6
CLOSE_COOLDOWN_S=90

# Persistence & server
STATE_FILE="state_council_only.json"
PORT=int(os.getenv("PORT","5000"))
SELF_URL=(os.getenv("SELF_URL") or os.getenv("RENDER_EXTERNAL_URL") or "").strip().rstrip("/")
API_KEY=os.getenv("BINGX_API_KEY","")
API_SECRET=os.getenv("BINGX_API_SECRET","")
MODE_LIVE=bool(API_KEY and API_SECRET)

# ===================== PRO LOGGING =====================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
HUMAN_LOG_FILE = "bot.log"
JSON_LOG_FILE  = "events.ndjson"

logger = logging.getLogger("PRO")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

_console = logging.StreamHandler(sys.stdout)
_console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_console)

_file = RotatingFileHandler(HUMAN_LOG_FILE, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
_file.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_file)

def _append_ndjson(obj: dict):
    try:
        line = json.dumps(obj, ensure_ascii=False, separators=(",",":"))
        with open(JSON_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line+"\n")
    except Exception as e:
        logger.error(f"_append_ndjson error: {e}")

def trade_log(event: str, **data):
    ts=int(time.time())
    human=" | ".join([event.upper()]+[f"{k}={v}" for k,v in data.items()])
    logger.info(human)
    payload={"ts":ts, "event":event, **data}
    _append_ndjson(payload)

# ===================== Circuit Breaker + Retries =====================
CIRCUIT_MAX_ERRORS=5
CIRCUIT_RESET_TIMEOUT=300
RETRY_ATTEMPTS=3; RETRY_BASE=1.0; RETRY_MAX=8.0

class CircuitBreaker:
    def __init__(self):
        self.error_count=0; self.last_trip=0.0; self.last_error=""
    @property
    def is_open(self):
        return self.error_count>=CIRCUIT_MAX_ERRORS and (time.time()-self.last_trip)<CIRCUIT_RESET_TIMEOUT
    def record_success(self): self.error_count=0
    def record_failure(self, msg):
        self.error_count+=1; self.last_error=msg
        if self.error_count>=CIRCUIT_MAX_ERRORS:
            self.last_trip=time.time()
            logger.error(f"🚨 CIRCUIT TRIPPED for {CIRCUIT_RESET_TIMEOUT}s | {msg}")

breaker=CircuitBreaker()

def _is_transient(e:Exception)->bool:
    s=str(e).lower()
    keys=["ddos","rate","timeout","etimedout","econnreset","network","503","504","500"]
    return any(k in s for k in keys) or isinstance(e, getattr(ccxt,"NetworkError",Exception))

def with_resilience(name):
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*a, **kw):
            if breaker.is_open:
                logger.warning(f"🧯 CIRCUIT OPEN — block '{name}' | last={breaker.last_error}")
                raise RuntimeError("CircuitOpen")
            wait=RETRY_BASE
            for i in range(1, RETRY_ATTEMPTS+1):
                try:
                    r=fn(*a, **kw); breaker.record_success(); return r
                except Exception as e:
                    msg=f"{type(e).__name__}: {e}"
                    if _is_transient(e) and i<RETRY_ATTEMPTS:
                        sl=min(wait, RETRY_MAX)+random.random()*0.4
                        logger.warning(f"⏳ {name} transient ({i}/{RETRY_ATTEMPTS}) → retry in {sl:.1f}s | {msg}")
                        time.sleep(sl); wait*=2; continue
                    breaker.record_failure(msg); logger.error(f"💥 {name} failed | {msg}"); raise
        return wrap
    return deco

# ===================== Exchange =====================
def make_ex():
    ex=ccxt.bingx({
        "apiKey":API_KEY,"secret":API_SECRET,
        "enableRateLimit":True,"timeout":20000,
        "options":{"defaultType":"swap"}
    })
    try:
        ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
    except Exception as e:
        logger.warning(f"set_leverage: {e}")
    return ex

ex=make_ex()
AMT_PREC=0; LOT_STEP=None; LOT_MIN=None
try:
    ex.load_markets()
    m=ex.markets.get(SYMBOL,{})
    AMT_PREC=int((m.get("precision",{}) or {}).get("amount",0) or 0)
    lims=(m.get("limits",{}) or {}).get("amount",{}) or {}
    LOT_STEP=lims.get("step"); LOT_MIN=lims.get("min")
except Exception as e:
    logger.warning(f"load_markets: {e}")

@with_resilience("fetch_ohlcv")
def ex_fetch_ohlcv(sym, tf, limit=600): return ex.fetch_ohlcv(sym, timeframe=tf, limit=limit, params={"type":"swap"})
@with_resilience("fetch_ticker")
def ex_fetch_ticker(sym): return ex.fetch_ticker(sym)
@with_resilience("fetch_order_book")
def ex_fetch_order_book(sym,limit=5): return ex.fetch_order_book(sym, limit=limit)
@with_resilience("fetch_balance")
def ex_fetch_balance(): return ex.fetch_balance(params={"type":"swap"})
@with_resilience("fetch_positions")
def ex_fetch_positions(symbols=None): return ex.fetch_positions(symbols or [SYMBOL])
@with_resilience("create_order")
def ex_create_order(sym, typ, side, qty, price=None, params=None): return ex.create_order(sym, typ, side, qty, price, params or {})
@with_resilience("cancel_all_orders")
def ex_cancel_all_orders(sym): return ex.cancel_all_orders(sym)

# ===================== Helpers =====================
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
    if q<=0: logger.warning(f"qty invalid → {q}")
    return q

def fetch_ohlcv(limit=600):
    rows=ex_fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit)
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

def price_now():
    try:
        t=ex_fetch_ticker(SYMBOL); return t.get("last") or t.get("close")
    except Exception: return None

def orderbook_spread_bps():
    try:
        ob=ex_fetch_order_book(SYMBOL, limit=5)
        bid=ob["bids"][0][0] if ob["bids"] else None
        ask=ob["asks"][0][0] if ob["asks"] else None
        if not (bid and ask): return None
        mid=(bid+ask)/2.0
        return ((ask-bid)/mid)*10000.0
    except Exception: return None

def balance_usdt():
    if not MODE_LIVE: return 1000.0
    try:
        b=ex_fetch_balance()
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT")
    except Exception: return None

def _interval_seconds(iv:str)->int:
    iv=(iv or "").lower().strip()
    if iv.endswith("m"): return int(float(iv[:-1]))*60
    if iv.endswith("h"): return int(float(iv[:-1]))*3600
    if iv.endswith("d"): return int(float(iv[:-1]))*86400
    return 900

def time_to_candle_close(df: pd.DataFrame)->int:
    tf=_interval_seconds(TIMEFRAME)
    if len(df)==0: return tf
    cur=int(df["time"].iloc[-1]); now=int(time.time()*1000)
    nxt=cur+tf*1000
    while nxt<=now: nxt+=tf*1000
    return int(max(0,nxt-now)/1000)

# ===================== Indicators =====================
def atr(df, n=ATR_LEN):
    h=df["high"].astype(float); l=df["low"].astype(float); c=df["close"].astype(float)
    tr=pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def adx(df, n=ADX_LEN):
    h=df["high"].astype(float); l=df["low"].astype(float); c=df["close"].astype(float)
    up=h.diff(); dn=-l.diff()
    plusDM=np.where((up>dn)&(up>0), up, 0.0)
    minusDM=np.where((dn>up)&(dn>0), dn, 0.0)
    tr=pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr14=tr.rolling(n).mean()
    plusDI=100*(pd.Series(plusDM).rolling(n).sum()/(atr14+1e-12))
    minusDI=100*(pd.Series(minusDM).rolling(n).sum()/(atr14+1e-12))
    dx=((plusDI-minusDI).abs()/((plusDI+minusDI)+1e-12))*100
    return dx.rolling(n).mean(), plusDI, minusDI

def rsi(series, n=RSI_LEN):
    delta=series.diff()
    up=np.where(delta>0, delta, 0.0); dn=np.where(delta<0, -delta, 0.0)
    roll_up=pd.Series(up).rolling(n).mean(); roll_dn=pd.Series(dn).rolling(n).mean()
    rs=roll_up/(roll_dn+1e-12); return 100-(100/(1+rs))

def macd(series):
    ema_f=series.ewm(span=MACD_FAST, adjust=False).mean()
    ema_s=series.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_=ema_f-ema_s; sig=macd_.ewm(span=MACD_SIG, adjust=False).mean()
    return macd_, sig, macd_-sig

def vwap(df):
    tp=(df["high"]+df["low"]+df["close"])/3.0
    vol=df["volume"]; return (tp*vol).cumsum()/(vol.cumsum()+1e-12)

def vei(df, f=VEI_FAST, s=VEI_SLOW):
    rng=(df["high"]-df["low"]).rolling(f).mean()
    base=(df["high"]-df["low"]).rolling(s).mean()
    return (rng/(base+1e-12))

# ===================== Bookmap-lite + SMC =====================
class Bookmap:
    def __init__(self): self.snapshot=[]; self.history=deque(maxlen=100)
    def supply(self, levels): 
        self.snapshot=levels or []; self.history.append({"ts":time.time(),"levels":self.snapshot[:20]})
    def eval(self, pip=0.0005):
        if not self.snapshot: return {"accum":[], "sweep":[], "walls":[], "imbalance":0.0}
        liqs=[x[1] for x in self.snapshot if x[1] is not None]; imbs=[x[2] for x in self.snapshot if x[2] is not None]
        liq_avg=max(1e-9, sum(liqs)/max(1,len(liqs))); imb_avg=(sum(imbs)/max(1,len(imbs))) if imbs else 0.0
        buckets={}
        for p,liq,imb,ab in self.snapshot:
            k=round(p/pip); buckets.setdefault(k,[]).append((p,liq,imb,ab))
        zones_acc=[]; zones_walls=[]; zones_sweep=[]; total=0.0
        for rows in buckets.values():
            liq_sum=sum(r[1] for r in rows); imb_mean=sum(r[2] for r in rows)/max(1,len(rows))
            ab_hits=sum(1 for r in rows if r[3])
            if liq_sum>5*liq_avg: zones_acc.append(liq_sum)
            if abs(imb_mean)>2*abs(imb_avg): zones_walls.append(imb_mean)
            if ab_hits>=3: zones_sweep.append(ab_hits)
            total+=imb_mean*liq_sum
        return {"accum":zones_acc,"sweep":zones_sweep,"walls":zones_walls,"imbalance":total}

bookmap=Bookmap()

def detect_eq_levels(df, lookback=50): 
    hi=df["high"][-lookback:].max(); lo=df["low"][-lookback:].min()
    return {"eqh":float(hi),"eql":float(lo),"tol_bps":10.0}

def detect_true_extrema(df, ind, bm):
    last=df.iloc[-1]; prev=df.iloc[-2]
    rng=max(last["high"]-last["low"],1e-12)
    wick_up=last["high"]-max(last["close"], last["open"])
    wick_dn=min(last["close"], last["open"])-last["low"]
    sweep_up = (wick_dn/rng>0.5) and (last["close"]>prev["high"]) and (ind["rsi"]<40)
    sweep_dn = (wick_up/rng>0.5) and (last["close"]<prev["low"]) and (ind["rsi"]>60)
    imbb=bm["imbalance"]
    true_bottom = sweep_up and imbb>0
    true_top    = sweep_dn and imbb<0
    return bool(true_bottom), bool(true_top)

def near_ob_fvg(df):
    at=atr(df,14).iloc[-1]; last=df.iloc[-1]
    impulse=( (last["high"]-last["low"]) > 1.2*(at+1e-12) )
    return {"near_ob_side": "buy" if impulse and last["close"]>last["open"] else "sell" if impulse and last["close"]<last["open"] else None}

# ===================== Council =====================
def council_vote(df: pd.DataFrame):
    c=df["close"].astype(float)
    ind={}
    ind["atr"]=float(atr(df).iloc[-1])
    adxv,pdi,mdi=adx(df); ind["adx"]=float(adxv.iloc[-1]); ind["pdi"]=float(pdi.iloc[-1]); ind["mdi"]=float(mdi.iloc[-1])
    ind["rsi"]=float(rsi(c).iloc[-1])
    macd_,sig,hist=macd(c); ind["macd"]=float(macd_.iloc[-1]); ind["macd_sig"]=float(sig.iloc[-1]); ind["macd_hist"]=float(hist.iloc[-1])
    ind["vwap"]=float(vwap(df).iloc[-1]); ind["vei"]=float(vei(df).iloc[-1])
    last=float(c.iloc[-1]); ind["price"]=last
    bm=bookmap.eval(); levels=detect_eq_levels(df); tb,tt=detect_true_extrema(df, ind, bm); smc=near_ob_fvg(df)

    buy=sell=0; reasons=[]
    if last>ind["vwap"] and ind["macd"]>ind["macd_sig"]: buy+=2; reasons.append("Price>VWAP,MACD+")
    if last<ind["vwap"] and ind["macd"]<ind["macd_sig"]: sell+=2; reasons.append("Price<VWAP,MACD-")
    if ind["adx"]>=28 and ind["pdi"]>ind["mdi"] and last>ind["vwap"]: buy+=1; reasons.append("StrongTrend+")
    if ind["adx"]>=28 and ind["mdi"]>ind["pdi"] and last<ind["vwap"]: sell+=1; reasons.append("StrongTrend-")
    if bm["imbalance"]>0: buy+=1; reasons.append("BM+")
    if bm["imbalance"]<0: sell+=1; reasons.append("BM-")
    if smc["near_ob_side"]=="buy": buy+=1; reasons.append("OB(buy)")
    if smc["near_ob_side"]=="sell": sell+=1; reasons.append("OB(sell)")
    tb_flag, tt_flag = tb, tt
    if tb_flag: buy+=2; reasons.append("TrueBottom++")
    if tt_flag: sell+=2; reasons.append("TrueTop++")

    score=buy-sell
    side="buy" if score>0 else "sell" if score<0 else None
    chop=(ind["vei"]<0.8) and (ind["adx"]<18)
    council={"buy":buy,"sell":sell,"score":float(score),"side":side,"reasons":reasons,"ind":ind,"bm":bm,"smc":smc,"chop":chop}
    trade_log("council",
              buy=buy, sell=sell, score=round(score,2),
              side=side, adx=round(ind["adx"],1),
              rsi=round(ind["rsi"],1), vei=round(ind["vei"],2),
              vwap=round(ind["vwap"],6), price=round(ind["price"],6),
              chop=bool(chop))
    return council

# ===================== State =====================
STATE={"open":False,"side":None,"entry":None,"qty":0.0,"pnl":0.0,"bars":0,"trail":None,"breakeven":None,"hp":0.0,"_last_close_ts":0}
compound_pnl=0.0
_trades=[]

def _rate_ok():
    now=time.time()
    while _trades and now-_trades[0]>3600: _trades.pop(0)
    return len(_trades)<MAX_TRADES_PER_HOUR
def _mark_trade(): _trades.append(time.time())

def save_state(tag=""):
    snap={"STATE":STATE,"compound_pnl":compound_pnl,"symbol":SYMBOL,"interval":TIMEFRAME,"ts":int(time.time()*1000),"tag":tag}
    try:
        d=os.path.dirname(STATE_FILE) or "."
        os.makedirs(d, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", delete=False, dir=d, encoding="utf-8") as tmp:
            json.dump(snap, tmp, ensure_ascii=False, separators=(",",":")); tmp.flush(); os.fsync(tmp.fileno()); p=tmp.name
        os.replace(p, STATE_FILE)
    except Exception as e: logger.error(f"save_state: {e}")

def load_state():
    try:
        if not os.path.exists(STATE_FILE): return None
        with open(STATE_FILE,"r",encoding="utf-8") as f: return json.load(f)
    except Exception as e:
        logger.error(f"load_state: {e}"); return None

def reconcile():
    global compound_pnl
    loaded=load_state()
    if loaded and loaded.get("symbol")==SYMBOL and loaded.get("interval")==TIMEFRAME:
        st=loaded.get("STATE") or {}
        for k in ["trail","breakeven","hp"]: STATE[k]=st.get(k)
        try: compound_pnl=float(loaded.get("compound_pnl") or 0.0)
        except Exception: pass
        logger.info("💾 Loaded local state.")
    try:
        pos=ex_fetch_positions([SYMBOL])
        active=None
        for p in pos:
            sym=p.get("symbol") or (p.get("info",{}).get("symbol") if isinstance(p.get("info",{}),dict) else "")
            if SYMBOL.split(":")[0] not in str(sym): continue
            contracts=float(p.get("contracts") or p.get("info",{}).get("positionAmt") or 0)
            if abs(contracts)>0:
                side="long" if contracts>0 else "short"
                entry=float(p.get("entryPrice") or p.get("info",{}).get("avgEntryPrice") or 0)
                active={"side":side,"entry":entry,"qty":abs(contracts)}
                break
        if active:
            STATE.update({"open":True,"side":active["side"],"entry":active["entry"],"qty":active["qty"],"pnl":0.0,"bars":0})
            logger.info(f"🔁 Resume {active['side'].upper()} {active['qty']} @ {active['entry']}")
        else:
            STATE.update({"open":False,"side":None,"entry":None,"qty":0.0,"pnl":0.0})
            logger.info("♻️ No live position — FLAT.")
        save_state("reconcile")
    except Exception as e:
        logger.error(f"reconcile: {e}")

# ===================== Sizing/Execution =====================
def compute_qty(balance, price, atr_val):
    if not balance or not price: return 0.0
    vol_adj=max(0.5, min(1.2, 1.0/(max(atr_val/price, 1e-6)*10)))
    cap=balance*RISK_FRAC*LEVERAGE*vol_adj
    return safe_qty(cap/price)

def _best_quotes():
    ob=ex_fetch_order_book(SYMBOL,5)
    bid=ob["bids"][0][0] if ob["bids"] else None
    ask=ob["asks"][0][0] if ob["asks"] else None
    mid=(bid+ask)/2.0 if (bid and ask) else price_now()
    return bid, ask, mid

def _ioc_price(side, mid, bps):
    if not mid: return None
    slip=bps/10000.0
    return mid*(1+slip) if side=="buy" else mid*(1-slip)

def open_position(side, qty, px, reason="COUNCIL"):
    if STATE["open"] or qty<=0: return False
    spr=orderbook_spread_bps()
    if spr is not None and (spr>HARD_SPREAD_BPS or spr>MAX_SPREAD_BPS):
        logger.warning(f"⛔ spread {spr:.2f}bps guard"); return False
    if not _rate_ok():
        logger.warning("⛔ rate-limit"); return False
    if breaker.is_open:
        logger.warning("🧯 CIRCUIT OPEN — block entry"); return False
    bid,ask,mid=_best_quotes()
    try:
        if MODE_LIVE:
            # حاول IOC Limit – وإلا سقط لـ Market
            limit=_ioc_price("buy" if side=="long" else "sell", mid, SLIP_OPEN_BPS)
            try:
                ex_create_order(SYMBOL,"limit","buy" if side=="long" else "sell", qty, limit, {"timeInForce":"IOC","reduceOnly":False,"positionSide":"BOTH"})
            except Exception:
                ex_create_order(SYMBOL,"market","buy" if side=="long" else "sell", qty, None, {"reduceOnly":False,"positionSide":"BOTH"})
        STATE.update({"open":True,"side":side,"entry":px,"qty":qty,"pnl":0.0,"bars":0,"trail":None,"breakeven":None,"hp":0.0})
        _mark_trade()
        trade_log("open",
                  side=side.upper(), qty=round(qty,4), price=px,
                  leverage=LEVERAGE, risk_frac=RISK_FRAC,
                  spread_bps=round(spr or 0.0,2), reason=reason)
        save_state("open")
        return True
    except Exception as e:
        logger.error(f"open_position: {e}")
        return False

def close_partial(frac, tag):
    if not STATE["open"] or STATE["qty"]<=0: return
    qty=safe_qty(STATE["qty"]*min(max(frac,0.0),1.0))
    if qty<=0: return
    side="sell" if STATE["side"]=="long" else "buy"
    try:
        if MODE_LIVE:
            try:
                ex_create_order(SYMBOL,"market",side,qty,None,{"reduceOnly":True,"positionSide":"BOTH"})
            except Exception:
                ex_create_order(SYMBOL,"limit",side,qty,_ioc_price(side, _best_quotes()[2], SLIP_CLOSE_BPS),{"timeInForce":"IOC","reduceOnly":True,"positionSide":"BOTH"})
        px=price_now() or STATE["entry"]
        STATE["qty"]=safe_qty(STATE["qty"]-qty)
        trade_log("partial_close", side=STATE["side"], qty=round(qty,4), rem_qty=round(STATE["qty"],4), reason=tag)
        if STATE["qty"]<=0: strict_close("Partial->Zero")
    except Exception as e:
        logger.error(f"close_partial: {e}")

def strict_close(tag="STRICT"):
    global compound_pnl
    if not STATE["open"]: return
    side=STATE["side"]; qty=STATE["qty"]
    try:
        if MODE_LIVE:
            bid,ask,mid=_best_quotes()
            limit=_ioc_price("sell" if side=="long" else "buy", mid, SLIP_CLOSE_BPS)
            try:
                ex_create_order(SYMBOL,"limit","sell" if side=="long" else "buy", qty, limit, {"timeInForce":"IOC","reduceOnly":True,"positionSide":"BOTH"})
            except Exception:
                ex_create_order(SYMBOL,"market","sell" if side=="long" else "buy", qty, None, {"reduceOnly":True,"positionSide":"BOTH"})
        px=price_now() or STATE["entry"]
        pnl=(px-STATE["entry"])*qty*(1 if side=="long" else -1)
        compound_pnl+=pnl
        trade_log("strict_close", side=side, qty=round(qty,4), price=px, pnl=round(pnl,6), total=round(compound_pnl,6), reason=tag)
    except Exception as e:
        logger.error(f"strict_close: {e}")
    finally:
        STATE.update({"open":False,"side":None,"entry":None,"qty":0.0,"pnl":0.0,"bars":0,"trail":None,"breakeven":None,"hp":0.0,"_last_close_ts":int(time.time())})
        save_state("strict_close")

# ===================== Management =====================
def manage_position(df, council):
    if not STATE["open"] or STATE["qty"]<=0: return
    px=float(df["close"].iloc[-1]); entry=STATE["entry"]; side=STATE["side"]
    rr=(px-entry)/entry*(1 if side=="long" else -1)  # decimal
    STATE["pnl"]=rr*entry*STATE["qty"]
    if rr>STATE["hp"]: STATE["hp"]=rr

    ind=council["ind"]; atrv=float(ind["atr"]); adxv=float(ind["adx"])

    # Hard SL طوارئ
    if rr <= -HARD_SL_PCT:
        trade_log("hard_sl_trigger", rr=round(rr*100,2))
        strict_close("HARD_SL"); return

    # Chop exit (ربح + سوق هادي)
    if council["chop"] and rr>0:
        trade_log("chop_exit", pnl_pct=round(rr*100,2), adx=round(adxv,1), vei=round(ind["vei"],2))
        strict_close("ChopProfit"); return

    # TP1 + Breakeven
    if STATE["breakeven"] is None and rr >= BREAKEVEN_AFTER:
        close_partial(TP1_FRAC, f"TP1@{round(rr*100,2)}%")
        STATE["breakeven"]=entry
        trade_log("breakeven_set", price=entry)

    # Trail activation + Ratchet
    if rr >= TRAIL_ACTIVATE_PCT and atrv>0:
        dist=atrv*ATR_TRAIL_MULT
        if side=="long":
            new=px - dist
            prev=STATE["trail"] or new
            STATE["trail"]=max(prev, new, STATE["breakeven"] or -1e9)
            if px<STATE["trail"]: strict_close(f"TRAIL({ATR_TRAIL_MULT}x)")
        else:
            new=px + dist
            prev=STATE["trail"] or new
            STATE["trail"]=min(prev, new, STATE["breakeven"] or 1e9)
            if px>STATE["trail"]: strict_close(f"TRAIL({ATR_TRAIL_MULT}x)")

    # انعكاس معاكس قوي من المجلس
    b=council["buy"]; s=council["sell"]; score=council["score"]
    if side=="long" and s>=EXIT_VOTES_MIN and score<=-3:
        strict_close("CouncilOppStrong")
    elif side=="short" and b>=EXIT_VOTES_MIN and score>=3:
        strict_close("CouncilOppStrong")

# ===================== Main Loop =====================
def time_to_next_sleep(df):
    return NEAR_CLOSE_SLEEP if time_to_candle_close(df)<=10 else BASE_SLEEP

def council_entry_ok(council, spread_bps):
    if STATE["open"]: return (False,"HasOpen")
    if breaker.is_open: return (False,"CircuitOpen")
    if spread_bps is not None and spread_bps>MAX_SPREAD_BPS: return (False,f"Spread>{MAX_SPREAD_BPS}bps")
    if council["side"] is None: return (False,"NoSide")
    if council["ind"]["adx"]<ENTRY_ADX_MIN: return (False,f"ADX<{ENTRY_ADX_MIN}")
    if max(council["buy"],council["sell"])<ENTRY_VOTES_MIN: return (False,"CouncilWeak")
    if abs(council["score"])<ENTRY_SCORE_MIN: return (False,"ScoreLow")
    if council["chop"]: return (False,"Chop")
    return (True,"OK")

def main_loop():
    reconcile()
    last_bar=0
    while True:
        try:
            df=fetch_ohlcv()
            df[["open","high","low","close","volume"]]=df[["open","high","low","close","volume"]].astype(float)
            px=price_now()
            spread=orderbook_spread_bps()
            council=council_vote(df)

            # إدارة مركز
            if STATE["open"]:
                manage_position(df, council)

            # دخول عند شمعة جديدة فقط
            bar=int(df["time"].iloc[-1])
            if bar!=last_bar:
                last_bar=bar
                cooldown_left=max(0, CLOSE_COOLDOWN_S-(time.time()-STATE.get("_last_close_ts",0)))
                ok,why=council_entry_ok(council, spread or 0.0)
                trade_log("entry_check",
                          side_hint=council["side"],
                          votes_buy=council["buy"], votes_sell=council["sell"],
                          score=round(council["score"],2),
                          adx=round(council["ind"]["adx"],1),
                          spread_bps=round(spread or 0.0,2),
                          chop=bool(council["chop"]),
                          cooldown=int(cooldown_left),
                          rate_ok=_rate_ok(),
                          blockers=("None" if ok and cooldown_left==0 and _rate_ok() else
                                    ",".join([x for x in [
                                        None if ok else why,
                                        f"Cooldown{int(cooldown_left)}s" if cooldown_left>0 else None,
                                        "RateLimit" if not _rate_ok() else None
                                    ] if x])))
                if ok and cooldown_left==0 and _rate_ok() and px:
                    bal=balance_usdt()
                    qty=compute_qty(bal, px, council["ind"]["atr"])
                    if qty>0:
                        opened=open_position("long" if council["side"]=="buy" else "short", qty, px, "COUNCIL")
                        if not opened:
                            logger.info("⏸️ entry blocked at execution.")
                elif not ok:
                    logger.info(f"⏸️ skip entry: {why}")

            save_state("loop")
            time.sleep(time_to_next_sleep(df))
        except Exception as e:
            logger.error(f"Loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

# ===================== Flask =====================
app=Flask(__name__)

@app.route("/")
def home():
    return f"COUNCIL-ONLY PRO — {SYMBOL} {TIMEFRAME} — {'LIVE' if MODE_LIVE else 'PAPER'}"

@app.route("/metrics")
def metrics():
    try_spread=orderbook_spread_bps()
    return jsonify({
        "symbol":SYMBOL,"interval":TIMEFRAME,"mode":"live" if MODE_LIVE else "paper",
        "price": price_now(), "spread_bps": try_spread,
        "state": STATE, "compound_pnl": compound_pnl,
        "circuit": {"open": breaker.is_open, "errors": breaker.error_count, "last_trip": int(breaker.last_trip or 0), "last_error": breaker.last_error}
    })

@app.route("/health")
def health():
    return jsonify({"ok":True,"ts":datetime.utcnow().isoformat(),"open":STATE["open"],"side":STATE["side"],"qty":STATE["qty"]}),200

@app.route("/bookmap", methods=["POST"])
def bookmap_feed():
    try:
        payload=request.get_json(silent=True) or {}
        levels=[]
        for row in payload.get("levels",[]):
            p=float(row[0]); liq=float(row[1]); imb=float(row[2]); ab=int(row[3])
            levels.append((p,liq,imb,ab))
        bookmap.supply(levels)
        return jsonify({"ok":True,"count":len(levels)})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)}),400

@app.route("/logs")
def logs_tail():
    n = int(request.args.get("n", "200"))
    try:
        if not os.path.exists(JSON_LOG_FILE):
            return jsonify({"ok":True, "lines":[]})
        with open(JSON_LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()[-n:]
        return "[" + ",".join(l.strip() for l in lines) + "]"
    except Exception as e:
        return jsonify({"ok":False, "error":str(e)}), 500

def keepalive_loop():
    if not SELF_URL:
        logger.warning("Keepalive disabled (no SELF_URL)"); return
    import requests
    s=requests.Session(); s.headers.update({"User-Agent":"council-only-keepalive"})
    while True:
        try:
            r=s.get(SELF_URL, timeout=8); logger.info(f"Keepalive: {r.status_code}")
        except Exception as e: logger.warning(f"Keepalive err: {e}")
        time.sleep(50)

# ===================== Boot =====================
if __name__=="__main__":
    print(colored("🎯 COUNCIL-ONLY PRO — BingX USDT Perps", "green"))
    print(colored(f"Mode: {'LIVE' if MODE_LIVE else 'PAPER'} • {SYMBOL} • {TIMEFRAME}", "yellow"))
    print(colored("ENTRY MODE: Council-Only — ADX Gate ≥ 17 — Single Position", "magenta"))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    threading.Thread(target=main_loop, daemon=True).start()
    if SELF_URL: threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)

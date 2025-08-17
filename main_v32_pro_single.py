# HELIOS ONE — V5.1 ELITE+ (single file)
# Requirements (requirements.txt) :
# streamlit
# pandas
# numpy
# yfinance
# ccxt
#
# ⚠️ Si ccxt n'est pas dispo, l'app tombe en mode "fallback" yfinance (scan OK, prix OK, pas d'order routing)

import os, json, math, sqlite3, datetime, datetime as dt
from typing import Dict, List, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np

# ───────────────────────────────── Page config ─────────────────────────────────
st.set_page_config(page_title="HELIOS ONE — V5.1 ELITE+", page_icon="🌞", layout="wide")
st.title("HELIOS ONE — V5.1 ELITE+")
st.caption("Ensemble multi-strats • Macro/VIX • Kill-switch • Corrélations • Multi-TP (TP1→BE, TP2→trail) • Matrice Strats×Mode×Symbole • Exécution 1-clic")

# ─────────────────────────────── Dépendances externes ─────────────────────────
try:
    import ccxt
    HAVE_CCXT = True
except Exception:
    HAVE_CCXT = False

try:
    import yfinance as yf
    HAVE_YF = True
except Exception:
    HAVE_YF = False

# ──────────────────────────────── Constantes globales ─────────────────────────
DB_PATH = "portfolio.sqlite"
DEFAULT_CAPITAL = 1000.0
MIN_NOTIONAL_USD = 5.0
DEFAULT_LOT_SIZE = 1e-6

FALLBACK_EXCHANGES = ['okx','bybit','kraken','coinbase','kucoin','binance']
EX_COST = {  # bps = basis points
    'okx':     {'fee_bps': 8,  'slip_bps': 3},
    'bybit':   {'fee_bps': 10, 'slip_bps': 4},
    'kraken':  {'fee_bps': 16, 'slip_bps': 5},
    'coinbase':{'fee_bps': 40, 'slip_bps': 6},
    'kucoin':  {'fee_bps': 10, 'slip_bps': 5},
    'binance': {'fee_bps': 8,  'slip_bps': 3},
}

SYMBOLS_DEFAULT = ['BTC/USDT','ETH/USDT','BNB/USDT','SOL/USDT','XRP/USDT','ADA/USDT','AVAX/USDT','LINK/USDT','TON/USDT','DOGE/USDT']

# mapping yfinance pour fallback (USD spot)
YF_MAP = {
    "BTC/USDT":"BTC-USD","ETH/USDT":"ETH-USD","BNB/USDT":"BNB-USD","SOL/USDT":"SOL-USD",
    "XRP/USDT":"XRP-USD","ADA/USDT":"ADA-USD","AVAX/USDT":"AVAX-USD","LINK/USDT":"LINK-USD",
    "TON/USDT":"TON-USD","DOGE/USDT":"DOGE-USD","MATIC/USDT":"MATIC-USD"
}

# ────────────────────────────────── Utils DB (SQLite) ─────────────────────────
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with get_conn() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS positions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            open_ts TEXT,
            close_ts TEXT,
            symbol TEXT,
            side TEXT,
            entry REAL,
            sl REAL,
            tp REAL,
            qty REAL,
            status TEXT,
            exit_price REAL,
            pnl REAL,
            note TEXT
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)""")
        con.commit()

def kv_get(k, default=None):
    init_db()
    with get_conn() as con:
        r = con.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else default

def kv_set(k, v):
    init_db()
    with get_conn() as con:
        con.execute("INSERT OR REPLACE INTO kv(k,v) VALUES(?,?)", (k, json.dumps(v)))
        con.commit()

if "db_inited" not in st.session_state:
    init_db()
    st.session_state.db_inited = True

def list_positions(status=None, limit=100000):
    with get_conn() as con:
        q = "SELECT id,open_ts,close_ts,symbol,side,entry,sl,tp,qty,status,exit_price,pnl,note FROM positions"
        pr = ()
        if status in ("OPEN","CLOSED"):
            q += " WHERE status=?"; pr = (status,)
        q += " ORDER BY id DESC LIMIT ?"
        pr = pr + (int(limit),)
        rows = list(con.execute(q, pr))
    return pd.DataFrame(rows, columns=['id','open_ts','close_ts','symbol','side','entry','sl','tp','qty','status','exit_price','pnl','note'])

def _meta_from_note(note: str):
    if isinstance(note, str) and note.startswith("META:"):
        try: return json.loads(note[5:])
        except Exception: return None
    return None

def _meta_to_note(meta: dict):
    return "META:" + json.dumps(meta, separators=(',',':'))

def open_position(symbol, side, entry, sl, tp, qty, meta: Optional[dict]=None):
    if qty is None or float(qty) <= 0: return None
    # safety: SL de l'autre côté
    if side.upper()=="LONG"  and sl>=entry: sl = entry - max(abs(sl-entry), 1e-9)
    if side.upper()=="SHORT" and sl<=entry: sl = entry + max(abs(sl-entry), 1e-9)
    init_db()
    with get_conn() as con:
        con.execute("""INSERT INTO positions(open_ts,close_ts,symbol,side,entry,sl,tp,qty,status,exit_price,pnl,note)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (dt.datetime.utcnow().isoformat(),None,symbol,side.upper(),
                     float(entry),float(sl),float(tp),float(qty),"OPEN",None,None,
                     _meta_to_note(meta) if isinstance(meta,dict) else ""))
        con.commit()
        rid = con.execute("SELECT last_insert_rowid()").fetchone()[0]
    return rid

def update_sl(pid, new_sl):
    with get_conn() as con:
        con.execute('UPDATE positions SET sl=? WHERE id=? AND status="OPEN"', (float(new_sl), int(pid)))
        con.commit()

def _encode_close_note(reason, trade_mode, top=None):
    try:
        obj={"mode":str(trade_mode),"reason":str(reason)}
        if top: obj["top"]=top
        return "META2:"+json.dumps(obj, separators=(',',':'))
    except Exception:
        return str(reason)

def close_position(pid, px, note='CLOSE'):
    with get_conn() as con:
        row = con.execute('SELECT open_ts,symbol,side,entry,sl,tp,qty,note FROM positions WHERE id=? AND status="OPEN"', (pid,)).fetchone()
        if not row: return None
        open_ts,symbol,side,entry,sl,tp,qty,note_open=row
        meta_open=_meta_from_note(note_open) or {}
        trade_mode = meta_open.get('trade_mode','unknown')
        top = meta_open.get('top_strats')
        pnl=(float(px)-float(entry))*float(qty)*(1 if side.upper()=="LONG" else -1)
        con.execute('UPDATE positions SET close_ts=?, status=?, exit_price=?, pnl=?, note=? WHERE id=?',
                    (dt.datetime.utcnow().isoformat(),"CLOSED",float(px),float(pnl),
                     _encode_close_note(note, trade_mode, top=top), int(pid)))
        con.commit()
        return pnl

def partial_close(pid, px, qty_close, reason="TP"):
    with get_conn() as con:
        row = con.execute('SELECT open_ts,symbol,side,entry,sl,tp,qty,note FROM positions WHERE id=? AND status="OPEN"', (pid,)).fetchone()
        if not row: return None
        open_ts,symbol,side,entry,sl,tp,qty,note_open=row
        meta_open=_meta_from_note(note_open) or {}
        trade_mode = meta_open.get('trade_mode','unknown')
        top = meta_open.get('top_strats')
        qty_close = float(min(max(qty_close,0.0),float(qty)))
        if qty_close<=0: return None
        sign=1 if side.upper()=="LONG" else -1
        pnl=(float(px)-float(entry))*qty_close*sign
        con.execute('INSERT INTO positions (open_ts,close_ts,symbol,side,entry,sl,tp,qty,status,exit_price,pnl,note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                    (open_ts,dt.datetime.utcnow().isoformat(),symbol,side,float(entry),float(sl),float(tp),float(qty_close),"CLOSED",float(px),float(pnl),
                     _encode_close_note(reason, trade_mode, top=top)))
        remain=float(qty)-qty_close
        if remain>1e-12:
            con.execute('UPDATE positions SET qty=? WHERE id=? AND status="OPEN"', (remain, int(pid)))
        else:
            con.execute('UPDATE positions SET close_ts=?, status=?, exit_price=?, pnl=?, note=? WHERE id=?',
                        (dt.datetime.utcnow().isoformat(),"CLOSED",float(px),float(pnl),
                         _encode_close_note(reason+"(FULL)", trade_mode, top=top), int(pid)))
        con.commit()
        return pnl

# ─────────────────────────────────── Market Data ─────────────────────────────
def build_exchange(name: str):
    if not HAVE_CCXT: return None
    try:
        ex_cls = getattr(ccxt, name.lower())
        ex = ex_cls({'enableRateLimit': True, 'options': {'adjustForTimeDifference': True}})
        ex.load_markets()
        return ex
    except Exception:
        return None

def _map_symbol(exchange_id: str, symbol: str) -> str:
    if exchange_id == 'kraken' and symbol.startswith('BTC/'):  return symbol.replace('BTC/','XBT/')
    if exchange_id == 'coinbase' and symbol.endswith('/USDT'): return symbol.replace('/USDT','/USDC')
    return symbol

def fetch_ohlcv_ccxt(exchange_id: str, symbol: str, timeframe='1h', limit=1500) -> pd.DataFrame:
    ex = build_exchange(exchange_id)
    if ex is None: raise RuntimeError("ccxt indisponible")
    sym = _map_symbol(exchange_id, symbol)
    data = ex.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    return df.set_index('ts')

def fetch_ohlcv_yf(symbol: str, tf: str='1h', limit=1500) -> pd.DataFrame:
    if not HAVE_YF: raise RuntimeError("yfinance indisponible")
    yf_sym = YF_MAP.get(symbol)
    if not yf_sym: raise RuntimeError(f"Mapping yfinance manquant pour {symbol}")
    period = "730d" if tf=="1h" else "max"
    df = yf.download(yf_sym, period=period, interval=tf, progress=False)
    if df is None or df.empty: raise RuntimeError(f"yfinance vide pour {symbol}")
    df = df[['Open','High','Low','Close','Volume']].rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'}).dropna()
    df.index = pd.to_datetime(df.index, utc=True)
    return df.tail(limit)

def load_or_fetch(exchange_id: str, symbol: str, tf: str, limit=1500) -> pd.DataFrame:
    last_err = None
    # 1) tente ccxt
    if HAVE_CCXT:
        for ex in [exchange_id] + [e for e in FALLBACK_EXCHANGES if e != exchange_id]:
            try: return fetch_ohlcv_ccxt(ex, symbol, tf, limit)
            except Exception as e: last_err = e
    # 2) fallback yfinance
    try: return fetch_ohlcv_yf(symbol, tf, limit)
    except Exception as e: last_err = e
    raise RuntimeError(f"OHLCV échec {symbol} {tf}: {last_err}")

def fetch_last_price(exchange_id: str, symbol: str) -> float:
    # ccxt si possible
    if HAVE_CCXT:
        for ex in [exchange_id] + [e for e in FALLBACK_EXCHANGES if e != exchange_id]:
            try:
                inst = build_exchange(ex); 
                if inst is None: continue
                sym = _map_symbol(ex, symbol)
                t = inst.fetch_ticker(sym); px = t.get('last') or t.get('close')
                if px: return float(px)
            except Exception:
                continue
    # sinon yfinance
    if HAVE_YF:
        try:
            df = fetch_ohlcv_yf(symbol, '1h', 2)
            if not df.empty: return float(df['close'].iloc[-1])
        except Exception:
            pass
    return np.nan

def yf_series(ticker: str, period="5y"):
    if not HAVE_YF: return None
    try:
        y = yf.download(ticker, period=period, interval="1d", progress=False)
        if y is None or y.empty: return None
        s = y['Adj Close'].tz_localize("UTC")
        s.index = pd.to_datetime(s.index, utc=True)
        return s.rename(ticker)
    except Exception:
        return None

# ───────────────────────────────── Indicateurs ───────────────────────────────
def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    dn=-d.clip(upper=0).ewm(alpha=1/n,adjust=False).mean(); rs=up/(dn+1e-9)
    return 100-100/(1+rs)
def atr_df(df,n=14):
    hl=df['high']-df['low']; hc=(df['high']-df['close'].shift()).abs(); lc=(df['low']-df['close'].shift()).abs()
    tr=pd.concat([hl,hc,lc],axis=1).max(axis=1); return tr.rolling(n).mean()
def kama(series, er_len=10, fast=2, slow=30):
    change=series.diff(er_len).abs(); vol=series.diff().abs().rolling(er_len).sum(); er=change/(vol+1e-9)
    sc=(er*(2/(fast+1)-2/(slow+1))+2/(slow+1))**2; out=[series.iloc[0]]
    for i in range(1,len(series)): out.append(out[-1]+sc.iloc[i]*(series.iloc[i]-out[-1]))
    return pd.Series(out,index=series.index)

# Signals (24+)
def sig_ema_trend(df): f=ema(df['close'],12); s=ema(df['close'],48); return ((f>s).astype(int)-(f<s).astype(int)).rename('signal')
def sig_macd(df): f=ema(df['close'],12); s=ema(df['close'],26); m=f-s; sig=ema(m,9); return ((m>sig).astype(int)-(m<sig).astype(int)).rename('signal')
def sig_donchian(df,look=55): hh=df['high'].rolling(look).max(); ll=df['low'].rolling(look).min(); return ((df['close']>hh.shift()).astype(int)-(df['close']<ll.shift()).astype(int)).clip(-1,1).rename('signal')
def sig_supertrend(df,period=10,mult=3.0):
    atr=atr_df(df,period); mid=(df['high']+df['low'])/2; bu=mid+mult*atr; bl=mid-mult*atr; fu=bu.copy(); fl=bl.copy()
    for i in range(1,len(df)):
        fu.iloc[i]=min(bu.iloc[i],fu.iloc[i-1]) if df['close'].iloc[i-1]>fu.iloc[i-1] else bu.iloc[i]
        fl.iloc[i]=max(bl.iloc[i],fl.iloc[i-1]) if df['close'].iloc[i-1]<fl.iloc[i-1] else bl.iloc[i]
    up=df['close']>fl; dn=df['close']<fu
    return (up.astype(int)-dn.astype(int)).rename('signal')
def sig_atr_channel(df,n=14,m=2.0): e=ema(df['close'],n); a=atr_df(df,n); up=e+m*a; lo=e-m*a; c=df['close']; return ((c>up).astype(int)-(c<lo).astype(int)).rename('signal')
def sig_boll_mr(df,n=20,k=2.0): ma=df['close'].rolling(n).mean(); sd=df['close'].rolling(n).std(); up=ma+k*sd; lo=ma-k*sd; c=df['close']; return ((c<lo).astype(int)-(c>up).astype(int)).rename('signal')
def sig_ichimoku(df,conv=9,base=26,spanb=52):
    high9=df['high'].rolling(conv).max(); low9=df['low'].rolling(conv).min(); ten=(high9+low9)/2
    high26=df['high'].rolling(base).max(); low26=df['low'].rolling(base).min(); kij=(high26+low26)/2
    spanA=((ten+kij)/2).shift(base); high52=df['high'].rolling(spanb).max(); low52=df['low'].rolling(spanb).min(); spanB=((high52+low52)/2).shift(base)
    cross=(ten>kij).astype(int)-(ten<kij).astype(int); up=(df['close']>spanA)&(df['close']>spanB); dn=(df['close']<spanA)&(df['close']<spanB)
    sig=cross.where(up,0).where(~dn,-1); return sig.fillna(0).rename('signal')
def sig_kama_trend(df): k=kama(df['close']); return ((df['close']>k).astype(int)-(df['close']<k).astype(int)).rename('signal')
def sig_rsi_mr(df,n=14,lo=30,hi=70): r=rsi(df['close'],n); return ((r<lo).astype(int)-(r>hi).astype(int)).rename('signal')
def sig_ppo(df,fast=12,slow=26,sig=9): emaf=ema(df['close'],fast); emas=ema(df['close'],slow); ppo=(emaf-emas)/emas; ppos=ema(ppo,sig); return ((ppo>ppos).astype(int)-(ppo<ppos).astype(int)).rename('signal')
def sig_adx_trend(df,n=14,th=20):
    up=df['high'].diff(); dn=-df['low'].diff()
    plusDM=np.where((up>dn)&(up>0),up,0.0); minusDM=np.where((dn>up)&(dn>0),dn,0.0)
    tr=atr_df(df,n)*(n/(n-1))
    plusDI=100*pd.Series(plusDM,index=df.index).ewm(alpha=1/n,adjust=False).mean()/tr
    minusDI=100*pd.Series(minusDM,index=df.index).ewm(alpha=1/n,adjust=False).mean()/tr
    dx=100*((plusDI-minusDI).abs()/(plusDI+minusDI+1e-9)); adx=dx.ewm(alpha=1/n,adjust=False).mean()
    return (((plusDI>minusDI)&(adx>th)).astype(int)-((minusDI>plusDI)&(adx>th)).astype(int)).rename('signal')
def sig_stoch_rsi(df,n=14,k=3,d=3,lo=0.2,hi=0.8):
    r=rsi(df['close'],n); sr=(r-r.rolling(n).min())/(r.rolling(n).max()-r.rolling(n).min()+1e-9)
    kf=sr.rolling(k).mean(); df_=kf.rolling(d).mean()
    return ((kf>df_)&(kf<lo)).astype(int)-((kf<df_)&(kf>hi)).astype(int)
def sig_cci_mr(df,n=20):
    tp=(df['high']+df['low']+df['close'])/3; ma=tp.rolling(n).mean(); md=(tp-ma).abs().rolling(n).mean()
    cci=(tp-ma)/(0.015*md+1e-9); return ((cci<-100).astype(int)-(cci>100).astype(int)).rename('signal')
def sig_heikin_trend(df):
    ha_c=(df['open']+df['high']+df['low']+df['close'])/4; ha_o=ha_c.copy()
    for i in range(1,len(df)): ha_o.iloc[i]=(ha_o.iloc[i-1]+ha_c.iloc[i-1])/2
    return ((ha_c>ha_o).astype(int)-(ha_c<ha_o).astype(int)).rename('signal')
def sig_chandelier(df,n=22,k=3.0):
    a=atr_df(df,n); long_stop=df['high'].rolling(n).max()-k*a; short_stop=df['low'].rolling(n).min()+k*a
    return ((df['close']>long_stop).astype(int)-(df['close']<short_stop).astype(int)).rename('signal')
def sig_vwap_mr(df,n=48):
    pv=(df['close']*df['volume']).rolling(n).sum(); vol=df['volume'].rolling(n).sum().replace(0,np.nan); v=pv/vol
    return ((df['close']<v*0.985).astype(int)-(df['close']>v*1.015).astype(int)).rename('signal')
def sig_turtle_soup(df,look=20):
    ll=df['low'].rolling(look).min(); hh=df['high'].rolling(look).max()
    lg=((df['low']<ll.shift())&(df['close']>df['open'])).astype(int)
    sh=-((df['high']>hh.shift())&(df['close']<df['open'])).astype(int)
    return (lg+sh).rename('signal')
def sig_zscore(df,n=50,k=2.0):
    z=(df['close']-df['close'].rolling(n).mean())/(df['close'].rolling(n).std()+1e-9)
    return ((z<-k).astype(int)-(z>k).astype(int)).rename('signal')
def sig_tsi(df,r=25,s=13):
    m=df['close'].diff(); a=ema(ema(m,r),s); b=ema(ema(m.abs(),r),s); tsi=100*a/(b+1e-9); sg=ema(tsi,13)
    return ((tsi>sg).astype(int)-(tsi<sg).astype(int)).rename('signal')
def sig_ema_ribbon(df):
    e=[ema(df['close'],n) for n in (8,13,21,34,55)]
    up=sum([e[i]>e[i+1] for i in range(len(e)-1)])
    dn=sum([e[i]<e[i+1] for i in range(len(e)-1)])
    return pd.Series(np.where(up>dn,1,np.where(dn>up,-1,0)),index=df.index,name='signal')
def sig_keltner(df,n=20,k=2.0): e=ema(df['close'],n); a=atr_df(df,n); up=e+k*a; lo=e-k*a; c=df['close']; return ((c>up).astype(int)-(c<lo).astype(int)).rename('signal')

def sig_psar(df, af_step=0.02, af_max=0.2):
    # Parabolic SAR corrigé (aucune parenthèse en trop)
    h = df['high']; l = df['low']; c = df['close']
    ps = l.copy()
    bull = True
    af = af_step
    ep = float(h.iloc[0])
    ps.iloc[0] = float(l.iloc[0])
    for i in range(1, len(df)):
        prev = float(ps.iloc[i-1])
        if bull:
            ps_val = prev + af * (ep - prev)
            ps_val = min(ps_val, float(l.iloc[i-1]))
            if i > 1: ps_val = min(ps_val, float(l.iloc[i-2]))
            if float(h.iloc[i]) > ep:
                ep = float(h.iloc[i]); af = min(af + af_step, af_max)
            if float(l.iloc[i]) < ps_val:
                bull = False; ps.iloc[i] = ep; ep = float(l.iloc[i]); af = af_step
            else:
                ps.iloc[i] = ps_val
        else:
            ps_val = prev + af * (ep - prev)
            ps_val = max(ps_val, float(h.iloc[i-1]))
            if i > 1: ps_val = max(ps_val, float(h.iloc[i-2]))
            if float(l.iloc[i]) < ep:
                ep = float(l.iloc[i]); af = min(af + af_step, af_max)
            if float(h.iloc[i]) > ps_val:
                bull = True; ps.iloc[i] = ep; ep = float(h.iloc[i]); af = af_step
            else:
                ps.iloc[i] = ps_val
    signal = np.where(c > ps, 1, np.where(c < ps, -1, 0))
    return pd.Series(signal, index=df.index, name='signal')

STRATS = {
    'EMA Trend':sig_ema_trend,'MACD Momentum':sig_macd,'Donchian Breakout':sig_donchian,'SuperTrend':sig_supertrend,
    'ATR Channel':sig_atr_channel,'Bollinger MR':sig_boll_mr,'Ichimoku':sig_ichimoku,'KAMA Trend':sig_kama_trend,'RSI MR':sig_rsi_mr,
    'PPO':sig_ppo,'ADX Trend':sig_adx_trend,'StochRSI':sig_stoch_rsi,'CCI MR':sig_cci_mr,'Heikin Trend':sig_heikin_trend,'Chandelier':sig_chandelier,
    'VWAP MR':sig_vwap_mr,'TurtleSoup':sig_turtle_soup,'ZScore MR':sig_zscore,'TSI Momentum':sig_tsi,'EMA Ribbon':sig_ema_ribbon,
    'Keltner BO':sig_keltner,'PSAR Trend':sig_psar,'MFI MR':lambda df: (( ((df['high']+df['low']+df['close'])/3).diff()>0 ).astype(int) - ( ((df['high']+df['low']+df['close'])/3).diff()<0 ).astype(int)).rename('signal'),
    'OBV Trend':lambda df: ( (df['volume']*np.sign(df['close'].diff().fillna(0))).cumsum().diff().apply(lambda x: 1 if x>0 else (-1 if x<0 else 0)) ).rename('signal')
}

# ────────────────────────────── Ensemble / Scores ────────────────────────────
def compute(df, signal, fee_bps=8.0, slip_bps=3.0):
    ret = df['close'].pct_change().fillna(0.0)
    pos = signal.shift().fillna(0.0).clip(-1,1)
    cost = (pos.diff().abs().fillna(0.0))*((fee_bps+slip_bps)/10000.0)
    pnl = pos*ret - cost
    eq = (1+pnl).cumprod()
    return ret,pos,pnl,eq

def sharpe(pnl,pp=365*24):
    s=pnl.std()
    return 0.0 if s==0 or np.isnan(s) else float(pnl.mean()/s*np.sqrt(pp))
def maxdd(eq):
    peak=eq.cummax(); dd=eq/peak-1
    return float(dd.min())

def _score(p, eq):
    s = max(0.0, min(3.0, sharpe(p)))
    dd = abs(maxdd(eq))
    return s + (1.0 - min(dd, 0.4))

def ensemble_weights(df, signals, window=300, fee_bps=8.0, slip_bps=3.0):
    if not signals: return pd.Series(dtype=float)
    start=max(0,len(df)-int(window)); sc={}
    for n,s in signals.items():
        try: _,_,p,eq = compute(df.iloc[start:], s.iloc[start:], fee_bps=fee_bps, slip_bps=slip_bps)
        except Exception: p=pd.Series([0]); eq=pd.Series([1.0])
        sc[n]=_score(p,eq)
    keys=list(sc.keys()); arr=np.array([sc[k] for k in keys])
    arr = arr - np.nanmax(arr); w=np.exp(arr)
    w = w/np.nansum(w) if np.nansum(w)!=0 else np.ones_like(w)/len(w)
    return pd.Series(w,index=keys)

def blended_signal(signals, weights):
    if not signals: return pd.Series(dtype=float, name="signal")
    df = pd.concat(signals.values(), axis=1).fillna(0.0); df.columns = list(signals.keys())
    w = weights.reindex(df.columns).fillna(0.0).values.reshape(1,-1)
    pos = (df.values * w).sum(axis=1)
    return pd.Series(pos, index=df.index, name="signal").clip(-1,1)

def htf_gate(df_ltf, df_htf): return sig_ema_trend(df_htf).reindex(df_ltf.index).ffill().fillna(0.0)

def macro_gate(enable, vix_caution=20.0, vix_riskoff=28.0, gold_mom_thr=0.10):
    if not enable: return 1.0, "macro OFF"
    if not HAVE_YF: return 1.0, "no_yfinance"
    vix = yf_series("^VIX"); gold = yf_series("GC=F")
    if vix is None or vix.empty: return 1.0, "no_vix"
    lvl=float(vix.iloc[-1]); mult=1.0; note=[]
    if lvl>float(vix_riskoff): mult*=0.0; note.append(f"VIX>{vix_riskoff} risk-off")
    elif lvl>float(vix_caution): mult*=0.5; note.append(f"VIX>{vix_caution} caution")
    else: note.append("VIX benign")
    if gold is not None and not gold.empty:
        mom=float(gold.pct_change(63).iloc[-1])
        if mom>float(gold_mom_thr): mult*=0.8; note.append("Gold strong")
    return mult, " | ".join(note)

# ───────────────────────────── Risk / sizing / niveaux ───────────────────────
def atr_levels(df, d, sl_mult=2.5, tp_mult=4.0):
    if d==0 or len(df)<2: return None
    a=float(atr_df(df,14).iloc[-1]); price=float(df['close'].iloc[-1])
    sl = price - sl_mult*a if d>0 else price + sl_mult*a
    tp = price + tp_mult*a if d>0 else price - tp_mult*a
    return {'entry':price,'sl':sl,'tp':tp,'atr':a}

def size_fixed_pct(equity, entry, stop, risk_pct):
    per=abs(entry-stop); risk=equity*(risk_pct/100.0)
    return 0.0 if per<=0 else risk/per

def rr(entry, sl, tp):
    R=abs(entry-sl); return float(abs(tp-entry)/(R if R>0 else 1e-9))

# ───────────────────────────── Clusters / corrélations ───────────────────────
def base_from_symbol(sym): return sym.split('/')[0].upper()
CLUSTER_MAP = {
    'BTC':'Majors','ETH':'Majors','BNB':'Exchange','SOL':'L1','ADA':'L1','AVAX':'L1','TON':'L1','XRP':'Payments','LINK':'Infra','DOGE':'Meme'
}
def symbol_cluster(sym): return CLUSTER_MAP.get(base_from_symbol(sym),'Other')

# ───────────────────────────── Presets par MODE ──────────────────────────────
modes={
 'Conservateur': dict(risk_pct=1.0, max_expo=60.0,  per_trade_cap=25.0, min_rr=1.8, max_positions=2, splits=(0.5,0.35,0.15), tpR=(0.9,1.7,2.6), gate_thr=0.35, leverage=1.0),
 'Normal':       dict(risk_pct=2.0, max_expo=100.0, per_trade_cap=35.0, min_rr=1.8, max_positions=3, splits=(0.4,0.4,0.2), tpR=(1.0,2.0,3.5), gate_thr=0.30, leverage=1.0),
 'Agressif':     dict(risk_pct=5.0, max_expo=150.0, per_trade_cap=40.0, min_rr=1.6, max_positions=5, splits=(0.30,0.40,0.30), tpR=(1.0,2.5,5.0), gate_thr=0.22, leverage=1.5),
 'Super agressif (x5)': dict(risk_pct=2.5, max_expo=120.0, per_trade_cap=20.0, min_rr=2.2, max_positions=3, splits=(0.34,0.33,0.33), tpR=(1.2,3.0,6.0), gate_thr=0.60, leverage=5.0)
}

# ───────────────────────────── UI — Réglages & risque ───────────────────────
with st.sidebar:
    mode = st.selectbox("Mode (unique levier UX)", list(modes.keys()), index=1)
    exchange = st.selectbox('Exchange (ccxt, sinon fallback yfinance)', FALLBACK_EXCHANGES, index=0)
    tf = st.selectbox('Timeframe', ['15m','1h','4h'], index=1)
    htf = st.selectbox('HTF confirm', ['1h','4h','1d'], index=2 if tf!='4h' else 1)
    symbols = st.multiselect('Paires', SYMBOLS_DEFAULT, default=SYMBOLS_DEFAULT[:8])
    sl_mult = st.slider("SL (×ATR)", 1.0, 4.0, 2.5, 0.1)
    tp_mult = st.slider("TP (×ATR suggéré)", 1.0, 6.0, 4.0, 0.1)
    macro_enabled = st.toggle("Macro gate (VIX/Gold)", value=True)
    capital = st.number_input("Capital de base", min_value=0.0, value=float(kv_get('base_capital',DEFAULT_CAPITAL)), step=100.0)
    if st.button("💾 Enregistrer capital"): kv_set('base_capital', float(capital)); st.success("Capital mis à jour.")

with st.sidebar.expander("Risque avancé (kill-switch, corrélation, time-stop)"):
    daily_loss_limit = st.number_input("Limite perte journalière (USD)", min_value=0.0, value=float(kv_get('daily_loss_limit', 150.0)), step=50.0)
    cooldown_minutes = st.number_input("Cooldown après dépassement (minutes)", min_value=0, value=int(kv_get('cooldown_minutes', 120)), step=15)
    cluster_cap_pct = st.slider("Cap d'expo par cluster (%) du cap global", 10, 100, int(kv_get('cluster_cap_pct', 60)), 5)
    time_stop_bars = st.number_input("Time-stop (barres avant sortie si pas d’avancée)", min_value=0, value=int(kv_get('time_stop_bars', 0)), step=1)
    if st.button("💾 Enregistrer (risque avancé)"):
        kv_set('daily_loss_limit', float(daily_loss_limit))
        kv_set('cooldown_minutes', int(cooldown_minutes))
        kv_set('cluster_cap_pct', int(cluster_cap_pct))
        kv_set('time_stop_bars', int(time_stop_bars))
        st.success("Risque avancé enregistré.")

m = modes[mode]
fee_bps = EX_COST.get(exchange, EX_COST['okx'])['fee_bps']
slip_bps = EX_COST.get(exchange, EX_COST['okx'])['slip_bps']

# ───────────────────────────── Suggestion de MODE ────────────────────────────
def suggest_mode():
    macro_mult, note = macro_gate(macro_enabled)
    try:
        df_btc = load_or_fetch(exchange, 'BTC/USDT', '4h', 600)
        tr = sig_adx_trend(df_btc, n=14, th=20).iloc[-1]
    except Exception:
        tr = 0
    if macro_mult==0.0: return "Conservateur","Risk-off (macro)", note
    if tr>0 and macro_mult>=1.0: return "Agressif","Tendance haussière forte", note
    if tr<0 and macro_mult>=1.0: return "Conservateur","Tendance baissière", note
    if macro_mult<1.0: return "Conservateur","Caution macro", note
    return "Normal","Rien de spécial", note

s_mode, s_reason, s_macro = suggest_mode()
st.info(f"💡 Suggestion de mode: **{s_mode}** — {s_reason} · Macro: {s_macro}")

# ───────────────────────────── Equity (dynamique) ────────────────────────────
def portfolio_equity(base_capital, price_map=None):
    open_df=list_positions(status='OPEN'); closed_df=list_positions(status='CLOSED')
    realized=0.0 if closed_df.empty else float(closed_df['pnl'].sum()); latent=0.0
    if not open_df.empty:
        if price_map is None: price_map={s:fetch_last_price(exchange,s) for s in open_df['symbol'].unique()}
        for _,r in open_df.iterrows():
            px=float(price_map.get(r['symbol'],r['entry'])); sign=1 if r['side']=='LONG' else -1
            latent+=(px-float(r['entry']))*float(r['qty'])*sign
    return base_capital+realized+latent

eq_now  = portfolio_equity(capital)
st.metric("💼 Portefeuille (équity dynamique)", f"{eq_now:.2f} USD")

# ───────────────────────────── Kill-switch jour & cooldown ───────────────────
def realized_today():
    hist=list_positions(status='CLOSED')
    if hist.empty: return 0.0
    today=dt.datetime.utcnow().date()
    vals=hist.copy(); vals['date']=pd.to_datetime(vals['close_ts']).dt.date
    return float(vals.loc[vals['date']==today,'pnl'].sum())

cooldown_until = kv_get('cooldown_until', None)
now_utc = dt.datetime.utcnow().timestamp()
if cooldown_until and now_utc < float(cooldown_until):
    eta = dt.datetime.utcfromtimestamp(float(cooldown_until)).strftime("%Y-%m-%d %H:%M:%S UTC")
    st.warning(f"⏳ Kill-switch actif jusqu’à {eta}. Pas de nouveaux scans.")
    kill_active = True
else:
    kill_active = False
    pnl_today = realized_today()
    if daily_loss_limit>0 and pnl_today <= -daily_loss_limit:
        until = dt.datetime.utcnow() + dt.timedelta(minutes=cooldown_minutes)
        kv_set('cooldown_until', until.timestamp())
        st.warning(f"❌ Limite journalière atteinte ({pnl_today:.2f} USD). Cooldown {cooldown_minutes} min activé.")
        kill_active = True

# ───────────────────────────── Fonctions multi-TP & trailing ─────────────────
def r_targets(entry, sl, side, tpR=(1.0,2.0,3.5)):
    side=side.upper(); sign=1 if side=="LONG" else -1
    R=(entry-sl) if side=="LONG" else (sl-entry)
    if R<=0: return None
    return [entry+sign*r*R for r in tpR]

def build_meta_r(entry, sl, side, qty, splits=(0.4,0.4,0.2), tpR=(1.0,2.0,3.5),
                 be_after_tp1=True, trade_mode="Normal", top_strats=None, confidence=None):
    tps=r_targets(entry,sl,side,tpR)
    if not tps: return None
    q1,q2,q3=[float(qty*max(0.0,s)) for s in splits]; diff=float(qty)-(q1+q2+q3); q3=max(0.0,q3+diff)
    meta={'multi_tp':True,'mode':'R','trade_mode':str(trade_mode),'tpR':list(tpR),'splits':list(splits),
          'targets':[{'name':'TP1','px':tps[0],'qty':q1,'filled':False},
                     {'name':'TP2','px':tps[1],'qty':q2,'filled':False},
                     {'name':'TP3','px':tps[2],'qty':q3,'filled':False}],
          'be_after_tp1':bool(be_after_tp1),'trail_after_tp2':True}
    if isinstance(top_strats, list): meta['top_strats']=top_strats[:5]
    if confidence is not None: meta['confidence']=float(confidence)
    return meta

def normalize_meta(meta, qty, entry, sl, side, default_splits=(0.4,0.4,0.2), default_tpR=(1.0,2.0,3.5), trade_mode="Normal"):
    try:
        if not isinstance(meta,dict) or not meta.get('multi_tp'):
            return build_meta_r(entry,sl,side,qty,splits=default_splits,tpR=default_tpR,trade_mode=trade_mode)
        splits=tuple(meta.get('splits',default_splits)); tpR=tuple(meta.get('tpR',default_tpR)); tgt=meta.get('targets',[])
        want=r_targets(entry,sl,side,tpR)
        if (not isinstance(tgt,list)) or (len(tgt)==0) or isinstance(tgt[0],(int,float,str)):
            return build_meta_r(entry,sl,side,qty,splits=splits,tpR=tpR,be_after_tp1=bool(meta.get('be_after_tp1',True)),trade_mode=trade_mode, top_strats=meta.get('top_strats'), confidence=meta.get('confidence'))
        for i in range(min(3,len(tgt))):
            if not isinstance(tgt[i],dict): tgt[i]={}
            tgt[i].setdefault('name',f"TP{i+1}"); tgt[i].setdefault('px',want[i])
            tgt[i].setdefault('qty',max(0.0,qty*(splits[i] if i<len(splits) else 0.0))); tgt[i].setdefault('filled',False)
        meta['targets']=tgt[:3]; meta.setdefault('be_after_tp1',True); meta.setdefault('trail_after_tp2',True)
        meta['tpR']=list(tpR); meta['splits']=list(splits); meta['trade_mode']=str(meta.get('trade_mode',trade_mode))
        return meta
    except Exception:
        return build_meta_r(entry,sl,side,qty,splits=default_splits,tpR=default_tpR,trade_mode=trade_mode)

def _atr_vec(h,l,c,n=22):
    h,l,c=h.values,l.values,c.values; tr=[h[0]-l[0]]
    for i in range(1,len(c)): tr.append(max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1])))
    return pd.Series(tr).rolling(n).mean().values
def _chandelier_stop(df,n=22,k=3.0,side="LONG"):
    atr=_atr_vec(df["high"],df["low"],df["close"],n=n)
    return (df["high"].rolling(n).max().values-k*atr) if side.upper()=="LONG" else (df["low"].rolling(n).min().values+k*atr)

def sanitize_all_positions():
    df=list_positions(); changed=False
    for _,r in df[df["status"]=="OPEN"].iterrows():
        side=str(r["side"]).upper(); entry=float(r["entry"]); sl=float(r["sl"]); new=sl; fix=False
        if side=="LONG" and sl>=entry: new=entry-max(1e-9,abs(sl-entry)); fix=True
        if side=="SHORT" and sl<=entry: new=entry+max(1e-9,abs(sl-entry)); fix=True
        if fix:
            with get_conn() as con:
                con.execute('UPDATE positions SET sl=? WHERE id=?',(float(new),int(r['id'])))
                con.commit(); changed=True
    return changed

def auto_manage_positions(price_map, ohlc_map=None, mode="Normal", be_after_tp1=True, trail_after_tp2=True, fee_buffer_bps=5, time_stop_bars=0, tf_minutes=60):
    sanitize_all_positions()
    df=list_positions(status='OPEN')
    if df.empty: return []
    # presets
    if str(mode).lower().startswith("conserv"): parts=(0.50,0.35,0.15); tpsR=(0.9,1.7,2.6)
    elif str(mode).lower().startswith("aggr") and "super" not in str(mode).lower(): parts=(0.30,0.40,0.30); tpsR=(1.0,2.5,5.0)
    elif "super" in str(mode).lower(): parts=(0.34,0.33,0.33); tpsR=(1.2,3.0,6.0)
    else: parts=(0.40,0.40,0.20); tpsR=(1.0,2.0,3.5)

    evts=[]; now=dt.datetime.utcnow()
    for _,r in df.iterrows():
        sym=r['symbol']; side=r['side'].upper()
        if sym not in price_map: continue
        px=float(price_map[sym]); entry=float(r['entry']); sl=float(r['sl']); qty=float(r['qty'])
        if qty<=1e-12: continue
        R=(entry-sl) if side=="LONG" else (sl-entry)
        if R<=0: continue
        meta_raw=_meta_from_note(r['note']); meta=normalize_meta(meta_raw,qty,entry,sl,side,trade_mode=mode)
        if meta_raw!=meta:
            with get_conn() as con: con.execute('UPDATE positions SET note=? WHERE id=?',(_meta_to_note(meta),int(r['id']))); con.commit()

        # recalc R targets by mode preset
        tps=r_targets(entry,sl,side,tuple(meta.get('tpR',tpsR)))
        for j in range(3):
            if 'targets' in meta and len(meta['targets'])>j: meta['targets'][j]['px']=tps[j]

        def hit_tp(p,t): return p>=t if side=="LONG" else p<=t
        def hit_sl(p,s): return p<=s if side=="LONG" else p>=s

        changed=False
        # TP1
        if not meta['targets'][0]['filled'] and hit_tp(px,meta['targets'][0]['px']):
            q=min(qty,float(meta['targets'][0]['qty']))
            if q>0: partial_close(int(r['id']),px,q,"TP1"); evts.append((sym,"TP1",px,q)); meta['targets'][0]['filled']=True; meta['targets'][0]['qty']=0.0; changed=True
            if be_after_tp1:
                be_px=entry+(fee_buffer_bps/10000.0)*entry*(1 if side=="LONG" else -1); update_sl(int(r['id']),be_px)
        # refresh qty
        cur=list_positions(status='OPEN'); cur=cur.loc[cur['id']==r['id']]
        qty_left=0.0 if cur.empty else float(cur.iloc[0]['qty'])
        if qty_left<=1e-12:
            if changed:
                with get_conn() as con: con.execute('UPDATE positions SET note=? WHERE id=?',(_meta_to_note(meta),int(r['id']))); con.commit()
            continue
        # TP2
        if not meta['targets'][1]['filled'] and hit_tp(px,meta['targets'][1]['px']):
            q=min(qty_left,float(meta['targets'][1]['qty']))
            if q>0: partial_close(int(r['id']),px,q,"TP2"); evts.append((sym,"TP2",px,q)); meta['targets'][1]['filled']=True; meta['targets'][1]['qty']=0.0; changed=True
            if trail_after_tp2 and ohlc_map and sym in ohlc_map:
                df_sym=ohlc_map[sym]; trail=float(_chandelier_stop(df_sym,22,3.0,side)[-1])
                cur_sl=float(list_positions(status='OPEN').loc[list_positions(status='OPEN')['id']==r['id'],'sl'].iloc[0])
                if (side=="LONG" and trail>cur_sl) or (side=="SHORT" and trail<cur_sl): update_sl(int(r['id']),trail)
        cur=list_positions(status='OPEN'); cur=cur.loc[cur['id']==r['id']]
        qty_left=0.0 if cur.empty else float(cur.iloc[0]['qty'])
        if qty_left<=1e-12:
            if changed:
                with get_conn() as con: con.execute('UPDATE positions SET note=? WHERE id=?',(_meta_to_note(meta),int(r['id']))); con.commit()
            continue
        # TP3
        if not meta['targets'][2]['filled'] and hit_tp(px,meta['targets'][2]['px']):
            partial_close(int(r['id']),px,qty_left,"TP3"); evts.append((sym,"TP3",px,qty_left)); meta['targets'][2]['filled']=True; meta['targets'][2]['qty']=0.0; changed=True
            qty_left=0.0
        # SL
        cur=list_positions(status='OPEN'); cur=cur.loc[cur['id']==r['id']]
        if not cur.empty:
            stop=float(cur.iloc[0]['sl'])
            if qty_left>0 and hit_sl(px,stop): partial_close(int(r['id']),stop,qty_left,"SL"); evts.append((sym,"SL",stop,qty_left)); qty_left=0.0

        # Time-stop
        if time_stop_bars and qty_left>0:
            try:
                opened=dt.datetime.fromisoformat(r['open_ts'])
            except Exception:
                opened=now
            age_minutes=(now-opened).total_seconds()/60.0
            if age_minutes >= time_stop_bars*tf_minutes:
                sign=1 if side=="LONG" else -1
                pnl_now=(px-entry)*sign
                if (not meta['targets'][0]['filled']) and (pnl_now<=0):
                    partial_close(int(r['id']),px,qty_left,"TIME_STOP")
                    evts.append((sym,"TIME_STOP",px,qty_left))
                    qty_left=0.0

        if changed:
            with get_conn() as con: con.execute('UPDATE positions SET note=? WHERE id=?',(_meta_to_note(meta),int(r['id']))); con.commit()
    return evts

# ───────────────────────────── Scanner (ensemble) ────────────────────────────
def scan_once(symbols:List[str], exchange_id:str, tf:str, htf:str, sl_mult:float, tp_mult:float,
              max_positions:int, min_rr:float, gate_thr:float, fee_bps=8.0, slip_bps=3.0):
    rows=[]
    for sym in symbols:
        try:
            df = load_or_fetch(exchange_id, sym, tf, 1200)
            df_htf = load_or_fetch(exchange_id, sym, htf, 600)
        except Exception as e:
            st.warning(f"Skip {sym}: {e}"); continue

        # ensemble
        signals = {nm: fn(df) for nm,fn in STRATS.items()}
        w = ensemble_weights(df, signals, window=300, fee_bps=fee_bps, slip_bps=slip_bps)
        sig = blended_signal(signals, w)
        # top stratégies (5)
        w_sorted = w.sort_values(ascending=False)
        top = [(k, float(v)) for k,v in w_sorted.head(5).items()]

        # confidence
        conf = float(abs(sig.iloc[-1])) * float(w_sorted.head(3).sum())

        gate = htf_gate(df, df_htf)
        macro_mult, _ = macro_gate(macro_enabled)
        blended = (sig * gate).clip(-1,1) * macro_mult

        if abs(float(blended.iloc[-1])) < gate_thr: continue

        d = int(np.sign(blended.iloc[-1]))
        if d==0: continue
        lvl = atr_levels(df, d, sl_mult, tp_mult)
        if not lvl: continue
        this_rr = rr(lvl['entry'], lvl['sl'], lvl['tp'])
        if this_rr < min_rr: continue

        qty = size_fixed_pct(eq_now, lvl['entry'], lvl['sl'], m['risk_pct'])
        if qty<=0: continue

        tps = r_targets(lvl['entry'], lvl['sl'], 'LONG' if d>0 else 'SHORT', m['tpR'])
        rows.append({
            'symbol': sym, 'dir':'LONG' if d>0 else 'SHORT',
            'entry': lvl['entry'], 'sl': lvl['sl'], 'tp': lvl['tp'],
            'rr': this_rr, 'qty_suggested': qty,
            'tp1': tps[0], 'tp2': tps[1], 'tp3': tps[2],
            'top': top, 'confidence': conf
        })
    if not rows: return pd.DataFrame(columns=['symbol','dir','entry','sl','tp','rr','qty_suggested','tp1','tp2','tp3','top','confidence'])
    return pd.DataFrame(rows).sort_values(['confidence','rr'],ascending=False).head(max_positions).reset_index(drop=True)

# ───────────────────────────── Exécution depuis scan ─────────────────────────
def round_lot(qty, lot=DEFAULT_LOT_SIZE):
    if qty<=0: return 0.0
    steps = max(1, int(qty/lot))
    return round(steps*lot, 6)

def qty_from_risk(capital_usd, risk_pct, entry, sl,
                  lot=DEFAULT_LOT_SIZE, min_notional=MIN_NOTIONAL_USD):
    try:
        risk_usd = max(0.0, float(risk_pct)/100.0 * float(capital_usd))
        dist = abs(float(entry) - float(sl))
        if dist <= 0: return 0.0
        qty = risk_usd / dist
        if qty * float(entry) < min_notional:
            qty = min_notional / max(float(entry), 1e-9)
        return round_lot(qty, lot)
    except Exception:
        return 0.0

def qty_from_percent(capital_usd, alloc_pct, entry,
                     lot=DEFAULT_LOT_SIZE, min_notional=MIN_NOTIONAL_USD):
    try:
        alloc_usd = max(0.0, float(alloc_pct)/100.0 * float(capital_usd))
        if alloc_usd <= 0: return 0.0
        qty = alloc_usd / max(float(entry), 1e-9)
        if qty * float(entry) < min_notional:
            qty = min_notional / max(float(entry), 1e-9)
        return round_lot(qty, lot)
    except Exception:
        return 0.0

def render_execution_from_scan(picks_df: pd.DataFrame, capital_total_usd: float, mode_label: str,
                               max_gross_expo_pct: float, cluster_cap_pct: int):
    if picks_df is None or picks_df.empty:
        st.info("Aucun setup à exécuter.")
        return

    st.subheader("Exécution")
    c1,c2,c3 = st.columns(3)
    with c1:
        entry_choice = st.selectbox("Prix d'entrée", ["Suggéré (entry)", "Dernier (market)"], index=0)
    with c2:
        select_mode = st.radio("Sélection", ["Sélection manuelle", "Tout prendre"], horizontal=True)
    with c3:
        max_expo_pct = st.slider("Plafond d'exposition (%)", 50, 300, int(m['max_expo']*m.get('leverage',1.0)))

    alloc_choice = st.radio("Mode d’allocation", ["Par risque (%)", "% du capital / trade", "Répartir équitablement (Tout)"])
    risk_pct = m['risk_pct']
    alloc_pct_per_trade = 10.0
    if alloc_choice == "Par risque (%)":
        risk_pct = st.slider("Risque par trade (%)", 0.1, 10.0, float(risk_pct), 0.1)
    elif alloc_choice == "% du capital / trade":
        alloc_pct_per_trade = st.slider("% du capital / trade", 1.0, 50.0, 10.0, 0.5)

    # exposition actuelle & cap
    open_now=list_positions(status='OPEN')
    expo_now = 0.0 if open_now.empty else float((open_now['entry']*open_now['qty']).sum())
    cap_max = capital_total_usd * (max_expo_pct/100.0) * m.get('leverage',1.0)
    cap_dispo = max(0.0, cap_max - expo_now)
    st.write(f"Expo actuelle **{expo_now:,.2f}** • Plafond **{cap_max:,.2f}** • Enveloppe **{cap_dispo:,.2f} USD**")

    # expo cluster actuelle
    cluster_now={}
    if not open_now.empty:
        for _,r in open_now.iterrows():
            cl = symbol_cluster(r['symbol'])
            cluster_now[cl] = cluster_now.get(cl,0.0) + float(r['entry']*r['qty'])
    cap_cluster_abs = cap_max * (cluster_cap_pct/100.0)

    picks_df = picks_df.reset_index(drop=True).copy()
    to_open=[]

    equal_usd = None
    if alloc_choice == "Répartir équitablement (Tout)":
        n = len(picks_df) if select_mode=="Tout prendre" else max(1,len(picks_df))
        equal_usd = cap_dispo / max(n,1)

    for i, r in picks_df.iterrows():
        symbol = str(r["symbol"]); side = "LONG" if str(r["dir"]).upper().startswith("L") else "SHORT"
        entry, sl, tp3 = float(r["entry"]), float(r["sl"]), float(r["tp"])
        if side=="LONG":
            tp1 = entry + 0.5*(tp3-entry); tp2 = entry + 0.75*(tp3-entry)
        else:
            tp1 = entry - 0.5*(entry-tp3); tp2 = entry - 0.75*(entry-tp3)

        # qty preview selon allocation
        if alloc_choice == "Par risque (%)":
            qty = qty_from_risk(capital_total_usd, risk_pct, entry, sl)
        elif alloc_choice == "% du capital / trade":
            qty = qty_from_percent(capital_total_usd, alloc_pct_per_trade, entry)
        else:
            qty = qty_from_percent(equal_usd or 0.0, 100.0, entry)

        checked = (select_mode=="Tout prendre") or st.checkbox(f"{symbol} • {side}", key=f"pick_{i}", value=False)
        st.write(f"- {symbol} • {side} • entry={entry:.6g} • SL={sl:.6g} • TP1={tp1:.6g} • TP2={tp2:.6g} • TP3={tp3:.6g} • qty≈**{qty}**")

        if checked and qty>0:
            to_open.append(dict(symbol=symbol, side=side, entry=entry, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
                                qty=qty, top=picks_df.loc[i,'top'] if 'top' in picks_df.columns else None,
                                confidence=float(picks_df.loc[i,'confidence']) if 'confidence' in picks_df.columns else None))

    cta1, cta2, _ = st.columns([1,1,2])
    def _apply(orders):
        if not orders:
            st.warning("Aucune ligne sélectionnée."); return
        # contrôle cap global et cap cluster
        total_cost = sum(o["entry"]*o["qty"] for o in orders)
        if total_cost > cap_dispo + 1e-9:
            st.error("Dépasse le plafond d’exposition."); return
        ok=0
        for o in orders:
            notional=o["entry"]*o["qty"]
            # cap cluster
            cl = symbol_cluster(o["symbol"]); cl_now=cluster_now.get(cl,0.0)
            if cl_now + notional > cap_cluster_abs:
                st.warning(f"Cap atteint pour cluster {cl}, {o['symbol']} réduit/ignoré.")
                continue
            # prix d'entrée
            if entry_choice.startswith("Dernier"):
                lp = fetch_last_price(exchange, o["symbol"])
                if lp and not np.isnan(lp): o["entry"]=float(lp)
            # meta multi-TP
            meta = build_meta_r(o["entry"], o["sl"], o["side"], o["qty"], splits=m['splits'], tpR=m['tpR'],
                                be_after_tp1=True, trade_mode=mode, top_strats=o.get('top'), confidence=o.get('confidence'))
            pid = open_position(o["symbol"], o["side"], o["entry"], o["sl"], o["tp3"], o["qty"], meta=meta)
            if pid: ok+=1; cluster_now[cl]=cluster_now.get(cl,0.0)+notional
        st.success(f"{ok} position(s) ajoutée(s)."); st.rerun()

    with cta1:
        if st.button("📌 Prendre la sélection", type="primary", use_container_width=True): _apply(to_open)
    with cta2:
        if st.button("📌 Prendre TOUT", use_container_width=True):
            if select_mode!="Tout prendre":
                # force tout
                all_orders=[]
                n=len(picks_df)
                for i, r in picks_df.iterrows():
                    side = "LONG" if str(r["dir"]).upper().startswith("L") else "SHORT"
                    entry, sl, tp3 = float(r["entry"]), float(r["sl"]), float(r["tp"])
                    if side=="LONG":
                        tp1 = entry + 0.5*(tp3-entry); tp2 = entry + 0.75*(tp3-entry)
                    else:
                        tp1 = entry - 0.5*(entry-tp3); tp2 = entry - 0.75*(entry-tp3)
                    if alloc_choice == "Par risque (%)":
                        qty = qty_from_risk(capital_total_usd, risk_pct, entry, sl)
                    elif alloc_choice == "% du capital / trade":
                        qty = qty_from_percent(capital_total_usd, alloc_pct_per_trade, entry)
                    else:
                        alloc_usd = (cap_dispo / max(n,1))
                        qty = qty_from_percent(alloc_usd, 100.0, entry)
                    if qty>0:
                        all_orders.append(dict(symbol=str(r["symbol"]), side=side, entry=entry, sl=sl,
                                               tp1=tp1,tp2=tp2,tp3=tp3, qty=qty,
                                               top=r.get('top'), confidence=float(r.get('confidence',0.0))))
                _apply(all_orders)
            else:
                _apply(to_open)

# ───────────────────────────── Tabs structure ────────────────────────────────
tabs = st.tabs(['🏠 Décision','📈 Portefeuille','🧾 Historique','📊 Analyse avancée','🔬 Lab'])

# ───────────────────────────── 1) Décision ───────────────────────────────────
with tabs[0]:
    st.subheader("Top Picks (1 clic)")
    st.caption(f"Macro gate: { 'ON' if macro_enabled else 'OFF' } • Coûts: fee {fee_bps} bps, slip {slip_bps} bps · Mode **{mode}**")

    if st.button("🚀 Scanner maintenant", use_container_width=True, disabled=kill_active):
        if kill_active: st.stop()
        picks = scan_once(symbols, exchange, tf, htf, sl_mult, tp_mult, m['max_positions'], m['min_rr'], m['gate_thr'], fee_bps=fee_bps, slip_bps=slip_bps)
        st.session_state['last_picks'] = picks
        st.success("Scan terminé.")
    picks = st.session_state.get('last_picks', pd.DataFrame())
    if picks is None or picks.empty:
        st.info("Aucun setup selon le mode choisi.")
    else:
        st.dataframe(picks[['symbol','dir','entry','sl','tp','tp1','tp2','tp3','rr','qty_suggested','confidence']], use_container_width=True, hide_index=True)
        render_execution_from_scan(picks, capital, mode, m['max_expo'], cluster_cap_pct)

# ───────────────────────────── 2) Portefeuille ───────────────────────────────
with tabs[1]:
    st.subheader("Positions ouvertes")
    open_df=list_positions(status='OPEN')
    if open_df.empty:
        st.info("Aucune position.")
    else:
        last={s:fetch_last_price(exchange,s) for s in open_df['symbol'].unique()}
        open_df=open_df[open_df['qty']>1e-12]
        if open_df.empty:
            st.info("Aucune position.")
        else:
            open_df['last']=open_df['symbol'].map(last)
            open_df['ret_%']=((open_df['last']-open_df['entry']).where(open_df['side']=='LONG', open_df['entry']-open_df['last'])/open_df['entry']*100).round(3)
            open_df['PnL_latent']=((open_df['last']-open_df['entry']).where(open_df['side']=='LONG', open_df['entry']-open_df['last'])*open_df['qty']).round(6)
            st.dataframe(open_df[['id','symbol','side','entry','sl','tp','qty','last','ret_%','PnL_latent','note']], use_container_width=True)

            # Mise à jour auto-manage
            tf_minutes_map={'15m':15,'1h':60,'4h':240}
            tf_minutes = tf_minutes_map.get(tf, 60)

            if st.button("🔄 Mettre à jour (TP/SL + BE/Trailing + Time-stop)"):
                ohlc_map={s:load_or_fetch(exchange,s,tf,300) for s in open_df['symbol'].unique()}
                events=auto_manage_positions(last, ohlc_map, mode=mode, be_after_tp1=True, trail_after_tp2=True,
                                            fee_buffer_bps=fee_bps+slip_bps, time_stop_bars=time_stop_bars, tf_minutes=tf_minutes)
                for sym,why,px,q in events: st.success(f"{sym}: {why} @ {px:.6f} (qty {q:.4f})")
                st.rerun()

            st.markdown("### Actions rapides")
            for _,r in open_df.iterrows():
                meta=_meta_from_note(r['note']) or {}
                cols=st.columns([3,1.1,1.1,1.1,1.3])
                tags=[]
                if isinstance(meta,dict) and meta.get('multi_tp'):
                    meta=normalize_meta(meta,r['qty'],r['entry'],r['sl'],r['side'],trade_mode=mode)
                    with get_conn() as con: con.execute('UPDATE positions SET note=? WHERE id=?',(_meta_to_note(meta),int(r['id']))); con.commit()
                    for t in meta.get('targets',[]):
                        if isinstance(t,dict):
                            tick="✅" if t.get('filled') else "🟡"; nm=str(t.get('name','TP')); px=float(t.get('px',r['tp']))
                            tags.append(f"{tick} {nm}@{px:.6f}")
                cols[0].markdown(f"**{r['symbol']}** · {r['side']} · qty `{r['qty']:.4f}` · SL `{r['sl']:.6f}`  \n" + (" | ".join(tags) if tags else "—"))
                if cols[1].button("SL→BE", key=f"be_{r['id']}"): update_sl(int(r['id']),float(r['entry'])); st.rerun()
                next_qty=0.0; next_name="NEXT"
                if isinstance(meta,dict) and meta.get('multi_tp'):
                    for t in meta['targets']:
                        if not t.get('filled'): next_qty=float(min(r['qty'],t.get('qty',0.0))); next_name=t.get('name','NEXT'); break
                if cols[2].button(f"Force {next_name}", key=f"force_{r['id']}", disabled=(next_qty<=0)):
                    px=last.get(r['symbol'],r['entry']); partial_close(int(r['id']),float(px),float(next_qty),f"FORCE_{next_name}")
                    if next_name=="TP1": update_sl(int(r['id']),float(r['entry'])); st.rerun()
                dyn=int(round(100.0*next_qty/max(r['qty'],1e-12))) if next_qty>0 else 25
                if cols[3].button(f"Close {dyn}%", key=f"closep_{r['id']}", disabled=(next_qty<=0)):
                    px=last.get(r['symbol'],r['entry']); partial_close(int(r['id']),float(px),float(r['qty'])*dyn/100.0,f"MANUAL_{dyn}"); st.rerun()
                if cols[4].button("Trail", key=f"trail_{r['id']}"):
                    df_now=load_or_fetch(exchange,r['symbol'],tf,220); side=r['side'].upper()
                    trail=float(_chandelier_stop(df_now,22,3.0,side)[-1])
                    if (side=='LONG' and trail>r['sl']) or (side=='SHORT' and trail<r['sl']): update_sl(int(r['id']),trail)
                    st.rerun()

            st.markdown("### Fermer totalement")
            for _,r in open_df.iterrows():
                if st.button(f"Fermer {r['symbol']} (100%)", key=f"close_{r['id']}"):
                    px=last.get(r['symbol'],r['entry']); close_position(int(r['id']),float(px),"MANUAL_CLOSE"); st.rerun()

# ───────────────────────────── 3) Historique ─────────────────────────────────
with tabs[2]:
    st.subheader("Historique (clôturées)")
    hist=list_positions(status='CLOSED')
    if hist.empty:
        st.info("Pas encore d’historique.")
    else:
        # extractions META
        def _mode_from_note(n):
            if isinstance(n,str):
                if n.startswith("META2:"):
                    try: return json.loads(n[6:]).get("mode","unknown")
                    except Exception: return "unknown"
                if n.startswith("META:"):
                    try: return (json.loads(n[5:]) or {}).get("trade_mode","unknown")
                    except Exception: return "unknown"
            return "unknown"
        def _top_from_note(n):
            if isinstance(n,str) and n.startswith("META2:"):
                try: return json.loads(n[6:]).get("top")
                except Exception: return None
            if isinstance(n,str) and n.startswith("META:"):
                try: return (json.loads(n[5:]) or {}).get("top_strats")
                except Exception: return None
            return None

        hist["mode"]=hist["note"].apply(_mode_from_note)
        hist["top"]=hist["note"].apply(_top_from_note)
        hist["result"]=np.where(hist["pnl"]>0,"WIN",np.where(hist["pnl"]<0,"LOSS","FLAT"))
        st.dataframe(hist[['close_ts','symbol','qty','exit_price','pnl','result','mode','note']], use_container_width=True)

        pnl=float(hist['pnl'].sum()); wins=(hist['pnl']>0).sum(); total=len(hist)
        winrate=0 if total==0 else wins/total*100
        avgwin=float(hist.loc[hist['pnl']>0,'pnl'].mean()) if wins>0 else 0.0
        avgloss=float(hist.loc[hist['pnl']<=0,'pnl'].mean()) if (total-wins)>0 else 0.0
        pf=(avgwin/abs(avgloss)) if avgloss<0 else np.nan
        c1,c2,c3,c4=st.columns(4)
        c1.metric("P&L réalisé", f"{pnl:.2f}")
        c2.metric("Win rate", f"{winrate:.1f}%")
        c3.metric("Profit factor", f"{pf:.2f}" if not np.isnan(pf) else "—")
        c4.metric("Avg win/loss", f"{avgwin:.2f} / {avgloss:.2f}")

        st.markdown("#### Perf par mode")
        def agg(dfm):
            wins=(dfm["pnl"]>0).sum(); total=len(dfm)
            avgw=dfm.loc[dfm["pnl"]>0,"pnl"].mean() if wins>0 else 0.0
            avgl=dfm.loc[dfm["pnl"]<=0,"pnl"].mean() if (total-wins)>0 else 0.0
            pf=(avgw/abs(avgl)) if avgl<0 else np.nan
            return pd.Series({"trades":total,"wins":wins,"winrate%":100*wins/total if total>0 else 0.0,"pnl_sum":dfm["pnl"].sum(),"profit_factor":pf})
        bymode=hist.groupby("mode",dropna=False).apply(agg).reset_index()
        st.dataframe(bymode.sort_values("pnl_sum",ascending=False).round(3), use_container_width=True)

# ───────────────────────── 4) Analyse avancée ────────────────────────────────
with tabs[3]:
    st.subheader("📊 Analyse avancée")
    hist=list_positions(status='CLOSED')
    if hist.empty:
        st.info("Pas de données encore.")
    else:
        # Equity curve (réalisé) par MODE
        def _mode(n):
            try:
                if isinstance(n,str) and n.startswith("META2:"): return json.loads(n[6:]).get("mode","unknown")
                if isinstance(n,str) and n.startswith("META:"):  return (json.loads(n[5:]) or {}).get("trade_mode","unknown")
            except Exception: pass
            return "unknown"
        df=hist[['close_ts','pnl','note']].copy(); df['mode']=df['note'].apply(_mode); df=df.sort_values('close_ts')
        curves={}
        for mo,grp in df.groupby('mode'):
            eq=capital + grp['pnl'].cumsum(); curves[mo]=eq.values
        st.markdown("#### Equity curve (réalisé) par MODE")
        eq_plot=pd.DataFrame({mo: pd.Series(vals) for mo,vals in curves.items()})
        st.line_chart(eq_plot)

        # Matrice Stratégie × Mode × Symbole
        def _top(n):
            try:
                if isinstance(n,str) and n.startswith("META2:"): return json.loads(n[6:]).get("top")
                if isinstance(n,str) and n.startswith("META:"):  return (json.loads(n[5:]) or {}).get("top_strats")
            except Exception: return None
        hist['top']=hist['note'].apply(_top)
        rows=[]
        for _,r in hist.iterrows():
            mode_row=_mode(r['note']); sym=r['symbol']; pnl=float(r['pnl'])
            top=r['top'] or []
            for t in top:
                try:
                    name, w = t[0], float(t[1])
                except Exception:
                    continue
                rows.append({'mode':mode_row,'symbol':sym,'strategy':name,'weight':w,'pnl':pnl})
        if rows:
            M=pd.DataFrame(rows)
            grp=M.groupby(['strategy','mode','symbol']).agg(trades=('pnl','count'),
                                                            pnl_sum=('pnl','sum'),
                                                            avg_w=('weight','mean')).reset_index()
            grp['pnl_per_trade']=grp['pnl_sum']/grp['trades']
            st.dataframe(grp.sort_values('pnl_sum',ascending=False).round(3), use_container_width=True)
            st.caption("Strats qui reviennent avec bon P&L = candidates robustes pour renforcement.")
        else:
            st.info("Pas encore assez de trades contenant des top_strats.")

# ───────────────────────────── 5) Lab (backtest rapide) ──────────────────────
with tabs[4]:
    st.subheader("Lab — Backtest rapide")
    sym_b=st.selectbox("Symbole", SYMBOLS_DEFAULT, index=0, key="lab_sym")
    tf_b=st.selectbox("TF", ['15m','1h','4h'], index=1, key="lab_tf")
    names=st.multiselect("Stratégies à tester", list(STRATS.keys()),
                         default=['EMA Trend','MACD Momentum','SuperTrend','Bollinger MR','Ichimoku'])
    if st.button("▶︎ Lancer le backtest"):
        try:
            df=load_or_fetch(exchange, sym_b, tf_b, 2000); res=[]
            for nm in names:
                sig=STRATS[nm](df); _,_,p,eq=compute(df,sig, fee_bps=fee_bps, slip_bps=slip_bps)
                res.append(dict(name=nm, sharpe=sharpe(p), mdd=maxdd(eq), cagr=(eq.iloc[-1]**(365*24/len(eq))-1)))
            st.dataframe(pd.DataFrame(res).sort_values("sharpe",ascending=False).round(4), use_container_width=True)
        except Exception as e:
            st.error(str(e))

# HELIOS ONE — V4 PRO (single-file)
# Multi-TP auto · SL->BE après TP1 · Trailing après TP2 · Ensemble 20+ strats
# Circuit-breaker (DD & daily) · Macro-gate VIX/Gold · Portefeuille/Journal/Lab
# Mobile-first · Manual only · SQLite persistant

import os, json, sqlite3, datetime
import streamlit as st
import pandas as pd
import numpy as np

# ────────────────────────────────────────────────────────────────────────────────
# Page config + icon
# ────────────────────────────────────────────────────────────────────────────────
ICON = "☀️"
try:
    from PIL import Image
    if os.path.exists("app_icon.png"):
        ICON = Image.open("app_icon.png")
except Exception:
    pass
st.set_page_config(page_title="HELIOS ONE — V4 PRO", page_icon=ICON, layout="centered")
st.title("HELIOS ONE — V4 PRO")
st.caption("Ensemble 20+ strats • Multi-TP • BE auto • Trailing • Circuit-breaker • Manual only")

# ────────────────────────────────────────────────────────────────────────────────
# Ext deps
# ────────────────────────────────────────────────────────────────────────────────
try:
    import ccxt
except Exception:
    st.error("❗ Il manque `ccxt` (ajoute-le à requirements.txt)."); st.stop()

try:
    import yfinance as yf
    HAVE_YF = True
except Exception:
    HAVE_YF = False

# ────────────────────────────────────────────────────────────────────────────────
# Exchange helpers (fallbacks pour prix/données)
# ────────────────────────────────────────────────────────────────────────────────
FALLBACK = ['okx','bybit','kraken','coinbase','kucoin','binance']

def build_exchange(name: str):
    ex_cls = getattr(ccxt, name.lower())
    ex = ex_cls({'enableRateLimit': True, 'options': {'adjustForTimeDifference': True}})
    try: ex.load_markets()
    except Exception: pass
    return ex

def _map_symbol(exchange_id: str, symbol: str) -> str:
    if exchange_id=='kraken' and symbol.startswith('BTC/'):
        return symbol.replace('BTC/','XBT/')
    if exchange_id=='coinbase' and symbol.endswith('/USDT'):
        return symbol.replace('/USDT','/USDC')
    return symbol

def fetch_ohlcv(exchange: str, symbol: str, timeframe='1h', limit=1500):
    ex = build_exchange(exchange); sym=_map_symbol(exchange, symbol)
    data = ex.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df.set_index('ts', inplace=True)
    return df

def load_or_fetch(exchange: str, symbol: str, timeframe: str, limit=1500):
    last_err=None
    for ex in [exchange]+[e for e in FALLBACK if e!=exchange]:
        try: return fetch_ohlcv(ex, symbol, timeframe, limit)
        except Exception as e: last_err=e
    raise RuntimeError(f"Echec fetch {symbol} {timeframe} : {last_err}")

def fetch_last_price(exchange: str, symbol: str) -> float:
    for ex in [exchange]+[e for e in FALLBACK if e!=exchange]:
        try:
            inst = build_exchange(ex); sym=_map_symbol(ex, symbol)
            t = inst.fetch_ticker(sym)
            px = t.get('last') or t.get('close')
            if px: return float(px)
        except Exception:
            continue
    return np.nan

def yf_series(ticker: str, period="5y"):
    if not HAVE_YF: return None
    try:
        y = yf.download(ticker, period=period, interval="1d", progress=False)
        if y is None or y.empty: return None
        return y['Adj Close'].rename(ticker).tz_localize("UTC")
    except Exception:
        return None

# ────────────────────────────────────────────────────────────────────────────────
# Indicators
# ────────────────────────────────────────────────────────────────────────────────
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(s, n=14):
    d=s.diff(); up=d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn=-d.clip(upper=0).ewm(alpha=1/n, adjust=False).mean()
    rs=up/(dn+1e-9); return 100-100/(1+rs)
def atr_df(df, n=14):
    hl=df['high']-df['low']; hc=(df['high']-df['close'].shift()).abs(); lc=(df['low']-df['close'].shift()).abs()
    tr=pd.concat([hl,hc,lc],axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()
def kama(series, er_len=10, fast=2, slow=30):
    change = series.diff(er_len).abs()
    vol = series.diff().abs().rolling(er_len).sum()
    er = change / (vol + 1e-9)
    sc = (er*(2/(fast+1) - 2/(slow+1)) + 2/(slow+1))**2
    out=[series.iloc[0]]
    for i in range(1,len(series)):
        out.append(out[-1] + sc.iloc[i]*(series.iloc[i]-out[-1]))
    return pd.Series(out, index=series.index)

# ────────────────────────────────────────────────────────────────────────────────
# 24 stratégies
# ────────────────────────────────────────────────────────────────────────────────
def sig_ema_trend(df): f=ema(df['close'],12); s=ema(df['close'],48); return ((f>s).astype(int)-(f<s).astype(int)).rename('signal')
def sig_macd(df): f=ema(df['close'],12); s=ema(df['close'],26); m=f-s; sig=ema(m,9); return ((m>sig).astype(int)-(m<sig).astype(int)).rename('signal')
def sig_donchian(df, look=55): hh=df['high'].rolling(look).max(); ll=df['low'].rolling(look).min(); return ((df['close']>hh.shift()).astype(int)-(df['close']<ll.shift()).astype(int)).clip(-1,1).rename('signal')
def sig_supertrend(df, period=10, mult=3.0):
    atr=atr_df(df, period); mid=(df['high']+df['low'])/2
    bu=mid+mult*atr; bl=mid-mult*atr; fu=bu.copy(); fl=bl.copy()
    for i in range(1,len(df)):
        fu.iloc[i]=min(bu.iloc[i], fu.iloc[i-1]) if df['close'].iloc[i-1]>fu.iloc[i-1] else bu.iloc[i]
        fl.iloc[i]=max(bl.iloc[i], fl.iloc[i-1]) if df['close'].iloc[i-1]<fl.iloc[i-1] else bl.iloc[i]
    up=df['close']>fl; down=df['close']<fu; return (up.astype(int)-down.astype(int)).rename('signal')
def sig_atr_channel(df, n=14, mult=2.0):
    e=ema(df['close'],n); a=atr_df(df,n); up=e+mult*a; lo=e-mult*a
    return ((df['close']>up).astype(int)-(df['close']<lo).astype(int)).rename('signal')
def sig_boll_mr(df, n=20, k=2.0): ma=df['close'].rolling(n).mean(); sd=df['close'].rolling(n).std(); up=ma+k*sd; lo=ma-k*sd; return (((df['close']<lo).astype(int))-((df['close']>up).astype(int))).rename('signal')
def sig_ichimoku(df, conv=9, base=26, spanb=52):
    high9=df['high'].rolling(conv).max(); low9=df['low'].rolling(conv).min(); tenkan=(high9+low9)/2
    high26=df['high'].rolling(base).max(); low26=df['low'].rolling(base).min(); kijun=(high26+low26)/2
    spanA=((tenkan+kijun)/2).shift(base); high52=df['high'].rolling(spanb).max(); low52=df['low'].rolling(spanb).min(); spanB=((high52+low52)/2).shift(base)
    cross=(tenkan>kijun).astype(int)-(tenkan<kijun).astype(int); up=(df['close']>spanA)&(df['close']>spanB); down=(df['close']<spanA)&(df['close']<spanB)
    sig=cross.where(up,0).where(~down,-1); return sig.fillna(0).rename('signal')
def sig_kama_trend(df): k=kama(df['close']); return ((df['close']>k).astype(int)-(df['close']<k).astype(int)).rename('signal')
def sig_rsi_mr(df, n=14, lo=30, hi=70): r=rsi(df['close'],n); return ((r<lo).astype(int)-(r>hi).astype(int)).rename('signal')
def sig_ppo(df, fast=12, slow=26, sig=9): emaf=ema(df['close'],fast); emas=ema(df['close'],slow); ppo=(emaf-emas)/emas; ppo_sig=ema(ppo,sig); return ((ppo>ppo_sig).astype(int)-(ppo<ppo_sig).astype(int)).rename('signal')
def sig_adx_trend(df, n=14, th=20):
    up = df['high'].diff(); down = -df['low'].diff()
    plusDM = np.where((up>down)&(up>0), up, 0.0); minusDM = np.where((down>up)&(down>0), down, 0.0)
    tr = atr_df(df, n)*(n/(n-1))
    plusDI = 100*pd.Series(plusDM,index=df.index).ewm(alpha=1/n,adjust=False).mean()/tr
    minusDI = 100*pd.Series(minusDM,index=df.index).ewm(alpha=1/n,adjust=False).mean()/tr
    dx = 100*((plusDI - minusDI).abs() / (plusDI + minusDI + 1e-9))
    adx = dx.ewm(alpha=1/n, adjust=False).mean()
    trend = ((plusDI>minusDI)&(adx>th)).astype(int) - ((minusDI>plusDI)&(adx>th)).astype(int)
    return trend.rename('signal')
def sig_stoch_rsi(df, n=14, k=3, d=3, lo=0.2, hi=0.8):
    r = rsi(df['close'], n)
    sr = (r - r.rolling(n).min()) / (r.rolling(n).max() - r.rolling(n).min() + 1e-9)
    kf = sr.rolling(k).mean(); df_ = kf.rolling(d).mean()
    return ((kf>df_)&(kf<lo)).astype(int) - ((kf<df_)&(kf>hi)).astype(int)
def sig_cci_mr(df, n=20):
    tp=(df['high']+df['low']+df['close'])/3; ma=tp.rolling(n).mean()
    md=(tp-ma).abs().rolling(n).mean()
    cci=(tp-ma)/(0.015*md+1e-9)
    return ((cci<-100).astype(int)-(cci>100).astype(int)).rename('signal')
def sig_heikin_trend(df):
    ha_close=(df['open']+df['high']+df['low']+df['close'])/4
    ha_open=ha_close.copy()
    for i in range(1,len(df)):
        ha_open.iloc[i]=(ha_open.iloc[i-1]+ha_close.iloc[i-1])/2
    return ((ha_close>ha_open).astype(int)-(ha_close<ha_open).astype(int)).rename('signal')
def sig_chandelier(df, n=22, mult=3.0):
    a=atr_df(df,n); long_stop=df['high'].rolling(n).max()-mult*a; short_stop=df['low'].rolling(n).min()+mult*a
    long=(df['close']>long_stop).astype(int); short=-(df['close']<short_stop).astype(int)
    return (long+short).clip(-1,1).rename('signal')
def sig_vwap_mr(df, n=48):
    pv=(df['close']*df['volume']).rolling(n).sum(); vol=df['volume'].rolling(n).sum().replace(0,np.nan)
    v=pv/vol
    return ((df['close']<v*0.985).astype(int) - (df['close']>v*1.015).astype(int)).rename('signal')
def sig_turtle_soup(df, look=20):
    ll=df['low'].rolling(look).min(); hh=df['high'].rolling(look).max()
    long=((df['low']<ll.shift())&(df['close']>df['open'])).astype(int)
    short=-((df['high']>hh.shift())&(df['close']<df['open'])).astype(int)
    return (long+short).rename('signal')
def sig_zscore(df, n=50, k=2.0):
    z=(df['close']-df['close'].rolling(n).mean())/(df['close'].rolling(n).std()+1e-9)
    return ((z<-k).astype(int)-(z>k).astype(int)).rename('signal')
def sig_tsi(df, r=25, s=13):
    m=df['close'].diff(); a=ema(ema(m,r),s); b=ema(ema(m.abs(),r),s)
    tsi=100*a/(b+1e-9); sig=ema(tsi,13)
    return ((tsi>sig).astype(int)-(tsi<sig).astype(int)).rename('signal')
def sig_ema_ribbon(df):
    e=[ema(df['close'],n) for n in (8,13,21,34,55)]
    up=sum([e[i]>e[i+1] for i in range(len(e)-1)]); down=sum([e[i]<e[i+1] for i in range(len(e)-1)])
    return pd.Series(np.where(up>down,1,np.where(down>up,-1,0)), index=df.index, name='signal')
def sig_keltner(df, n=20, mult=2.0):
    e=ema(df['close'],n); a=atr_df(df,n); up=e+mult*a; lo=e-mult*a; c=df['close']
    return ((c>up).astype(int)-(c<lo).astype(int)).rename('signal')
def sig_psar(df, af=0.02, max_af=0.2):
    high, low = df['high'], df['low']
    psar = low.copy(); bull = True; af_val = af; ep = high.iloc[0]; psar.iloc[0] = low.iloc[0]
    for i in range(2, len(df)):
        prev = psar.iloc[i-1]
        if bull:
            psar.iloc[i] = min(prev + af_val*(ep - prev), low.iloc[i-1], low.iloc[i-2])
            if high.iloc[i] > ep: ep = high.iloc[i]; af_val = min(max_af, af_val + af)
            if low.iloc[i] < psar.iloc[i]: bull=False; psar.iloc[i]=ep; ep=low.iloc[i]; af_val=af
        else:
            psar.iloc[i] = max(prev + af_val*(ep - prev), high.iloc[i-1], high.iloc[i-2])
            if low.iloc[i] < ep: ep = low.iloc[i]; af_val = min(max_af, af_val + af)
            if high.iloc[i] > psar.iloc[i]: bull=True; psar.iloc[i]=ep; ep=high.iloc[i]; af_val=af
    return ((df['close'] > psar).astype(int) - (df['close'] < psar).astype(int)).rename('signal')
def sig_mfi_mr(df, n=14, lo=20, hi=80):
    tp=(df['high']+df['low']+df['close'])/3; mf=tp*df['volume']
    pos=mf.where(tp>tp.shift(),0.0); neg=mf.where(tp<tp.shift(),0.0).abs()
    mr=100-100/(1+(pos.rolling(n).sum()/(neg.rolling(n).sum()+1e-9)))
    return ((mr<lo).astype(int)-(mr>hi).astype(int)).rename('signal')
def sig_obv_trend(df, n=20):
    ch=np.sign(df['close'].diff().fillna(0.0)); obv=(df['volume']*ch).cumsum(); e=ema(obv,n)
    return ((obv>e).astype(int)-(obv<e).astype(int)).rename('signal')

STRATS = {
    'EMA Trend': sig_ema_trend, 'MACD Momentum': sig_macd, 'Donchian Breakout': sig_donchian,
    'SuperTrend': sig_supertrend, 'ATR Channel': sig_atr_channel, 'Bollinger MR': sig_boll_mr,
    'Ichimoku': sig_ichimoku, 'KAMA Trend': sig_kama_trend, 'RSI MR': sig_rsi_mr, 'PPO': sig_ppo,
    'ADX Trend': sig_adx_trend, 'StochRSI': sig_stoch_rsi, 'CCI MR': sig_cci_mr, 'Heikin Trend': sig_heikin_trend,
    'Chandelier': sig_chandelier, 'VWAP MR': sig_vwap_mr, 'TurtleSoup': sig_turtle_soup, 'ZScore MR': sig_zscore,
    'TSI Momentum': sig_tsi, 'EMA Ribbon': sig_ema_ribbon, 'Keltner BO': sig_keltner, 'PSAR Trend': sig_psar,
    'MFI MR': sig_mfi_mr, 'OBV Trend': sig_obv_trend,
}

# ────────────────────────────────────────────────────────────────────────────────
# Ensemble / gating / scoring
# ────────────────────────────────────────────────────────────────────────────────
def compute(df, signal, fee_bps=2.0, slippage_bps=1.0):
    ret=df['close'].pct_change().fillna(0.0)
    pos=signal.shift().fillna(0.0).clip(-1,1)
    cost=(pos.diff().abs().fillna(0.0))*((fee_bps+slippage_bps)/10000.0)
    pnl=pos*ret - cost
    equity=(1+pnl).cumprod()
    return ret, pos, pnl, equity

def sharpe(pnl, periods_per_year=365*24):
    s=pnl.std(); return 0.0 if s==0 or np.isnan(s) else float(pnl.mean()/s * np.sqrt(periods_per_year))
def max_drawdown(eq):
    peak=eq.cummax(); dd=eq/peak-1; return float(dd.min())

def _score(pnl, eq):
    s=max(0.0, min(3.0, sharpe(pnl))); dd=abs(max_drawdown(eq)); return s + (1.0 - min(dd,0.4))

def ensemble_weights(df: pd.DataFrame, signals: dict, window: int = 300) -> pd.Series:
    if not signals: return pd.Series(dtype=float)
    start=max(0, len(df)-int(window))
    scores={}
    for name, sig in signals.items():
        try:
            _,_,pnl,eq=compute(df.iloc[start:], sig.iloc[start:])
            scores[name]=_score(pnl,eq)
        except Exception: scores[name]=-1e9
    keys=list(scores.keys()); arr=np.array([scores[k] for k in keys], dtype=float)
    arr=arr-np.nanmax(arr); w=np.exp(arr); w=w/np.nansum(w) if np.nansum(w)!=0 else np.ones_like(w)/len(w)
    return pd.Series(w, index=keys)

def blended_signal(signals: dict, weights: pd.Series) -> pd.Series:
    if not signals: return pd.Series(dtype=float, name="signal")
    df=pd.concat(signals.values(), axis=1).fillna(0.0); df.columns=list(signals.keys())
    w=weights.reindex(df.columns).fillna(0.0).values.reshape(1,-1)
    pos=(df.values*w).sum(axis=1)
    return pd.Series(pos, index=df.index, name="signal").clip(-1,1)

def htf_gate(df_ltf, df_htf):
    trend = sig_ema_trend(df_htf).reindex(df_ltf.index).ffill().fillna(0.0)
    return trend

def macro_gate(enable_macro: bool, vix_caution=20.0, vix_riskoff=28.0, gold_mom_thresh=0.10):
    if not enable_macro: return 1.0, "macro OFF"
    if not HAVE_YF: return 1.0, "no_yfinance"
    vix = yf_series("^VIX"); gold = yf_series("GC=F")
    if vix is None or vix.empty: return 1.0, "no_vix"
    lvl = float(vix.iloc[-1])
    mult = 1.0; note = []
    if lvl>float(vix_riskoff): mult *= 0.0; note.append(f"VIX>{vix_riskoff} (risk-off)")
    elif lvl>float(vix_caution): mult *= 0.5; note.append(f"VIX>{vix_caution} (caution)")
    else: note.append("VIX benign")
    if gold is not None and not gold.empty:
        mom = float(gold.pct_change(63).iloc[-1])
        if mom>float(gold_mom_thresh): mult *= 0.8; note.append("Gold strong (defensive)")
    return mult, " | ".join(note)

# ────────────────────────────────────────────────────────────────────────────────
# Risk / Levels / Sizing
# ────────────────────────────────────────────────────────────────────────────────
def atr_levels(df: pd.DataFrame, direction: int, atr_mult_sl=2.5, atr_mult_tp=3.8):
    if direction==0 or len(df)<2: return None
    a=float(atr_df(df,14).iloc[-1]); price=float(df['close'].iloc[-1])
    if direction>0: sl=price-atr_mult_sl*a; tp=price+atr_mult_tp*a
    else: sl=price+atr_mult_sl*a; tp=price-atr_mult_tp*a
    return {'entry':price, 'sl':sl, 'tp':tp, 'atr':a}

def size_fixed_pct(account_equity: float, entry: float, stop: float, risk_pct: float):
    per_unit=abs(entry-stop); risk_amt=account_equity*(risk_pct/100.0)
    return 0.0 if per_unit<=0 else risk_amt/per_unit

def rr(entry, sl, tp):
    R = abs(entry-sl); return float(abs(tp-entry)/(R if R>0 else 1e-9))

# ────────────────────────────────────────────────────────────────────────────────
# SQLite (positions + kv)
# ────────────────────────────────────────────────────────────────────────────────
DB = os.path.join(os.path.dirname(__file__), 'portfolio.db')

def _init_db():
    conn = sqlite3.connect(DB); c=conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS positions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      open_ts TEXT, close_ts TEXT, symbol TEXT, side TEXT,
      entry REAL, sl REAL, tp REAL, qty REAL,
      status TEXT, exit_price REAL, pnl REAL, note TEXT)''')
    c.execute('CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)')
    conn.commit(); conn.close()

def kv_get(key, default=None):
    _init_db(); conn=sqlite3.connect(DB)
    row = conn.execute('SELECT v FROM kv WHERE k=?', (key,)).fetchone()
    conn.close()
    return json.loads(row[0]) if row else default

def kv_set(key, value):
    _init_db(); conn=sqlite3.connect(DB)
    conn.execute('INSERT OR REPLACE INTO kv (k,v) VALUES (?,?)', (key, json.dumps(value)))
    conn.commit(); conn.close()

def list_positions(status=None, limit=100000):
    _init_db(); conn=sqlite3.connect(DB)
    q='SELECT id, open_ts, close_ts, symbol, side, entry, sl, tp, qty, status, exit_price, pnl, note FROM positions'
    params=()
    if status in ("OPEN","CLOSED"): q+=' WHERE status=?'; params=(status,)
    q+=' ORDER BY id DESC LIMIT ?'; params=params+(int(limit),)
    rows=list(conn.execute(q, params)); conn.close()
    return pd.DataFrame(rows, columns=['id','open_ts','close_ts','symbol','side','entry','sl','tp','qty','status','exit_price','pnl','note'])

def _meta_from_note(note:str):
    if isinstance(note,str) and note.startswith("META:"):
        try: return json.loads(note[5:])
        except Exception: return None
    return None
def _meta_to_note(meta:dict) -> str: return "META:" + json.dumps(meta, separators=(',',':'))
def _set_meta(pos_id:int, meta:dict):
    conn=sqlite3.connect(DB)
    conn.execute('UPDATE positions SET note=? WHERE id=? AND status="OPEN"', (_meta_to_note(meta), int(pos_id)))
    conn.commit(); conn.close()

def open_position(symbol, side, entry, sl, tp, qty, meta=None):
    _init_db()
    if qty is None or float(qty) <= 0: return None
    # Sanity SL côté logique
    if side.upper()=="LONG" and sl>=entry: sl = entry - max(1e-9, abs(sl-entry))
    if side.upper()=="SHORT" and sl<=entry: sl = entry + max(1e-9, abs(sl-entry))
    note = _meta_to_note(meta) if isinstance(meta, dict) else ''
    conn=sqlite3.connect(DB); c=conn.cursor()
    c.execute('INSERT INTO positions (open_ts, close_ts, symbol, side, entry, sl, tp, qty, status, exit_price, pnl, note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
              (datetime.datetime.utcnow().isoformat(), None, symbol, side.upper(), float(entry), float(sl), float(tp), float(qty), "OPEN", None, None, note))
    conn.commit(); rid=c.lastrowid; conn.close()
    return rid

def update_sl(pos_id:int, new_sl: float):
    _init_db(); conn=sqlite3.connect(DB)
    conn.execute('UPDATE positions SET sl=? WHERE id=? AND status="OPEN"', (float(new_sl), int(pos_id)))
    conn.commit(); conn.close()

def close_position(pos_id:int, exit_price:float, note='CLOSE'):
    _init_db(); conn=sqlite3.connect(DB); c=conn.cursor()
    row=c.execute('SELECT open_ts, symbol, side, entry, sl, tp, qty FROM positions WHERE id=? AND status="OPEN"', (pos_id,)).fetchone()
    if not row: conn.close(); return None
    open_ts, symbol, side, entry, sl, tp, qty = row
    pnl=(float(exit_price)-float(entry))*float(qty)*(1 if side.upper()=="LONG" else -1)
    c.execute('UPDATE positions SET close_ts=?, status=?, exit_price=?, pnl=?, note=? WHERE id=?',
              (datetime.datetime.utcnow().isoformat(), "CLOSED", float(exit_price), float(pnl), note, pos_id))
    conn.commit(); conn.close()
    return pnl

def partial_close(pos_id:int, exit_price:float, qty_to_close:float, reason:str="TP"):
    _init_db(); conn=sqlite3.connect(DB); c=conn.cursor()
    row=c.execute('SELECT open_ts, symbol, side, entry, sl, tp, qty FROM positions WHERE id=? AND status="OPEN"', (pos_id,)).fetchone()
    if not row: conn.close(); return None
    open_ts, symbol, side, entry, sl, tp, qty = row
    qty_to_close = float(min(max(qty_to_close, 0.0), float(qty)))
    if qty_to_close <= 0: conn.close(); return None
    sign = 1 if side.upper()=="LONG" else -1
    pnl = (float(exit_price)-float(entry))*qty_to_close*sign
    c.execute('INSERT INTO positions (open_ts, close_ts, symbol, side, entry, sl, tp, qty, status, exit_price, pnl, note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
              (open_ts, datetime.datetime.utcnow().isoformat(), symbol, side, float(entry), float(sl), float(tp), float(qty_to_close),
               "CLOSED", float(exit_price), float(pnl), reason))
    remain = float(qty) - qty_to_close
    if remain > 1e-12:
        c.execute('UPDATE positions SET qty=? WHERE id=? AND status="OPEN"', (remain, pos_id))
    else:
        c.execute('UPDATE positions SET close_ts=?, status=?, exit_price=?, pnl=?, note=? WHERE id=?',
                  (datetime.datetime.utcnow().isoformat(), "CLOSED", float(exit_price), float(pnl), reason+"(FULL)", pos_id))
    conn.commit(); conn.close()
    return pnl

# ────────────────────────────────────────────────────────────────────────────────
# Multi-TP meta + sanity
# ────────────────────────────────────────────────────────────────────────────────
def r_targets(entry: float, sl: float, side: str, tpR=(1.0, 2.0, 3.5)):
    sign = 1 if side.upper()=="LONG" else -1
    R = (entry - sl) if side.upper()=="LONG" else (sl - entry)
    if R <= 0: return None
    return [entry + sign*r*R for r in tpR]

def build_meta_r(entry, sl, side, qty, splits=(0.4,0.4,0.2), tpR=(1.0,2.0,3.5), be_after_tp1=True):
    tps = r_targets(entry, sl, side, tpR)
    if not tps: return None
    q1, q2, q3 = [float(qty*max(0.0,s)) for s in splits]
    diff = float(qty) - (q1+q2+q3); q3 = max(0.0, q3 + diff)
    targets = [{'name':'TP1','px':tps[0],'qty':q1,'filled':False},
               {'name':'TP2','px':tps[1],'qty':q2,'filled':False},
               {'name':'TP3','px':tps[2],'qty':q3,'filled':False}]
    return {'multi_tp': True, 'mode':'R', 'tpR':list(tpR), 'splits':list(splits),
            'targets': targets, 'be_after_tp1': bool(be_after_tp1), 'trail_after_tp2': True}

def _fix_sl_side_for_row(r):
    side = str(r["side"]).upper()
    entry = float(r["entry"]); sl = float(r["sl"])
    changed = False
    if side == "LONG" and sl >= entry:
        sl = entry - max(1e-9, abs(sl - entry)); changed = True
    if side == "SHORT" and sl <= entry:
        sl = entry + max(1e-9, abs(sl - entry)); changed = True
    return sl, changed

def sanitize_all_positions():
    df = list_positions()
    changed_any = False
    for _, r in df[df["status"]=="OPEN"].iterrows():
        sl_new, changed = _fix_sl_side_for_row(r)
        if changed:
            conn=sqlite3.connect(DB); conn.execute('UPDATE positions SET sl=? WHERE id=?', (float(sl_new), int(r['id']))); conn.commit(); conn.close()
            changed_any = True
    return changed_any

# ────────────────────────────────────────────────────────────────────────────────
# Auto-engine (TP1/2/3 + BE + trailing)
# ────────────────────────────────────────────────────────────────────────────────
def _atr_vec(h, l, c, n=22):
    h, l, c = h.values, l.values, c.values
    tr = [h[0]-l[0]]
    for i in range(1, len(c)):
        tr.append(max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
    return pd.Series(tr).rolling(n).mean().values

def _chandelier_stop(df, n=22, k=3.0, side="LONG"):
    atr = _atr_vec(df["high"], df["low"], df["close"], n=n)
    if side.upper()=="LONG":  return (df["high"].rolling(n).max().values - k*atr)
    else:                      return (df["low"].rolling(n).min().values + k*atr)

def auto_manage_positions(price_map, ohlc_map=None,
                          mode="neutral",
                          be_after_tp1=True,
                          trail_after_tp2=True,
                          fee_buffer_bps=0):
    sanitize_all_positions()
    df = list_positions(status='OPEN')
    if df.empty: return []

    if mode.lower().startswith("conserv"): parts=(0.50,0.35,0.15); tpsR=(0.8,1.6,2.5)
    elif mode.lower().startswith("aggr"):  parts=(0.30,0.40,0.30); tpsR=(1.0,2.5,5.0)
    else:                                  parts=(0.40,0.40,0.20); tpsR=(1.0,2.0,3.5)

    evts=[]
    for _, r in df.iterrows():
        sym = r['symbol']; side = r['side'].upper()
        if sym not in price_map: continue
        px=float(price_map[sym]); entry=float(r['entry']); sl=float(r['sl'])
        qty=float(r['qty'])
        if qty<=1e-12: continue

        # R directionnel
        R = (entry - sl) if side=="LONG" else (sl - entry)
        if R <= 0:  # incohérence → skip
            continue

        # Meta
        meta = _meta_from_note(r['note'])
        if not (isinstance(meta, dict) and meta.get('multi_tp')):
            meta = build_meta_r(entry, sl, side, qty, splits=parts, tpR=tpsR, be_after_tp1=be_after_tp1)
            _set_meta(int(r['id']), meta)

        # Targets (update px si recalcul nécessaire)
        tps = r_targets(entry, sl, side, tuple(meta.get('tpR', tpsR)))
        for j, nm in enumerate(['TP1','TP2','TP3']):
            if 'targets' in meta and len(meta['targets'])>j:
                meta['targets'][j]['px'] = tps[j]

        def hit_tp(price, tgt): return price >= tgt if side=="LONG" else price <= tgt
        def hit_sl(price, stop): return price <= stop if side=="LONG" else price >= stop

        changed = False
        qty_left = float(qty)

        # TP1
        if not meta['targets'][0]['filled'] and hit_tp(px, meta['targets'][0]['px']):
            q = min(qty_left, float(meta['targets'][0]['qty']))
            if q>0:
                partial_close(int(r['id']), px, q, "TP1"); evts.append((sym,"TP1",px,q))
                meta['targets'][0]['filled']=True; meta['targets'][0]['qty']=0.0; changed=True
                qty_left -= q
                if be_after_tp1:
                    be = entry + (fee_buffer_bps/10000.0)*entry*(1 if side=="LONG" else -1)
                    update_sl(int(r['id']), be)

        rr = list_positions(status='OPEN'); rr = rr.loc[rr['id']==r['id']]
        qty_left = 0.0 if rr.empty else float(rr.iloc[0]['qty'])
        if qty_left <= 1e-12:
            if changed: _set_meta(int(r['id']), meta)
            continue

        # TP2
        if not meta['targets'][1]['filled'] and hit_tp(px, meta['targets'][1]['px']):
            q = min(qty_left, float(meta['targets'][1]['qty']))
            if q>0:
                partial_close(int(r['id']), px, q, "TP2"); evts.append((sym,"TP2",px,q))
                meta['targets'][1]['filled']=True; meta['targets'][1]['qty']=0.0; changed=True
                qty_left -= q
                if trail_after_tp2 and ohlc_map and sym in ohlc_map:
                    df_sym = ohlc_map[sym]
                    trail = float(_chandelier_stop(df_sym, n=22, k=3.0, side=side)[-1])
                    cur_sl = float(list_positions(status='OPEN').loc[list_positions(status='OPEN')['id']==r['id'], 'sl'].iloc[0])
                    if side=="LONG" and trail>cur_sl: update_sl(int(r['id']), trail)
                    if side=="SHORT" and trail<cur_sl: update_sl(int(r['id']), trail)

        rr = list_positions(status='OPEN'); rr = rr.loc[rr['id']==r['id']]
        qty_left = 0.0 if rr.empty else float(rr.iloc[0]['qty'])
        if qty_left <= 1e-12:
            if changed: _set_meta(int(r['id']), meta)
            continue

        # TP3 (tout)
        if not meta['targets'][2]['filled'] and hit_tp(px, meta['targets'][2]['px']):
            partial_close(int(r['id']), px, qty_left, "TP3"); evts.append((sym,"TP3",px,qty_left))
            meta['targets'][2]['filled']=True; meta['targets'][2]['qty']=0.0; changed=True
            qty_left = 0.0

        # SL
        rr = list_positions(status='OPEN'); rr = rr.loc[rr['id']==r['id']]
        if not rr.empty:
            stop = float(rr.iloc[0]['sl'])
            if qty_left > 0 and hit_sl(px, stop):
                partial_close(int(r['id']), stop, qty_left, "SL"); evts.append((sym,"SL",stop,qty_left))
                qty_left = 0.0

        if changed:
            _set_meta(int(r['id']), meta)

    return evts

# ────────────────────────────────────────────────────────────────────────────────
# Circuit-breaker & equity
# ────────────────────────────────────────────────────────────────────────────────
def portfolio_equity(base_capital, price_map=None):
    open_df = list_positions(status='OPEN'); closed_df = list_positions(status='CLOSED')
    realized = 0.0 if closed_df.empty else float(closed_df['pnl'].sum())
    latent = 0.0
    if not open_df.empty:
        if price_map is None:
            price_map = {s: fetch_last_price('okx', s) for s in open_df['symbol'].unique()}
        for _, r in open_df.iterrows():
            px = float(price_map.get(r['symbol'], r['entry']))
            sign = 1 if r['side']=='LONG' else -1
            latent += (px - float(r['entry'])) * float(r['qty']) * sign
    return base_capital + realized + latent

def circuit_breaker_update(capital, dd_limit=0.08, daily_limit=0.03, pause_hours=12):
    eq = portfolio_equity(capital)
    log = kv_get('equity_log', [])
    now = datetime.datetime.utcnow().isoformat()
    log.append({'ts': now, 'equity': eq})
    if len(log) > 2000: log = log[-2000:]
    kv_set('equity_log', log)

    peak = max(x['equity'] for x in log)
    dd = (eq/peak - 1.0) if peak>0 else 0.0
    today = now[:10]
    start_day = [x['equity'] for x in log if x['ts'][:10]==today]
    open_day = start_day[0] if start_day else eq
    day_chg = (eq/open_day - 1.0) if open_day>0 else 0.0

    paused_until = kv_get('risk_pause_until', None)
    if dd <= -abs(dd_limit) or day_chg <= -abs(daily_limit):
        until = (datetime.datetime.utcnow()+datetime.timedelta(hours=pause_hours)).isoformat()
        kv_set('risk_pause_until', until)
        return eq, dd, day_chg, until
    return eq, dd, day_chg, paused_until

def circuit_breaker_active():
    until = kv_get('risk_pause_until', None)
    if not until: return False
    return datetime.datetime.utcnow().isoformat() < until

# ────────────────────────────────────────────────────────────────────────────────
# UI — Réglages
# ────────────────────────────────────────────────────────────────────────────────
symbols_default = ['BTC/USDT','ETH/USDT','BNB/USDT','SOL/USDT','XRP/USDT','ADA/USDT','AVAX/USDT','LINK/USDT','TON/USDT','DOGE/USDT']

st.markdown("### ⚙️ Réglages")
with st.expander("Ouvrir les réglages", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        exchange = st.selectbox('Exchange', ['okx','bybit','kraken','coinbase','kucoin','binance'], index=0)
        tf = st.selectbox('Timeframe', ['15m','1h','4h'], index=1)
        htf = st.selectbox('HTF (confirmation)', ['1h','4h','1d'], index=2 if tf!='4h' else 1)
        capital = st.number_input('Capital (USD)', value=1000.0, step=50.0)
        symbols = st.multiselect('Paires', symbols_default, default=symbols_default[:8])
    with c2:
        mode = st.selectbox('Mode risque', ['Conservateur','Neutre','Agressif'], index=1)
        presets = {'Conservateur': dict(risk_pct=0.6, max_expo=50.0, min_rr=1.6, max_positions=2, splits=(0.5,0.35,0.15), tpR=(0.8,1.6,2.5)),
                   'Neutre':       dict(risk_pct=1.2, max_expo=80.0, min_rr=1.7, max_positions=3, splits=(0.4,0.4,0.2), tpR=(1.0,2.0,3.5)),
                   'Agressif':     dict(risk_pct=2.0, max_expo=120.0, min_rr=1.8, max_positions=5, splits=(0.3,0.4,0.3), tpR=(1.0,2.5,5.0))}
        p=presets[mode]
        risk_pct=st.slider('Risque %/trade', 0.1, 5.0, p['risk_pct'], 0.1)
        max_expo=st.slider('Cap exposition (%)', 10.0, 200.0, p['max_expo'], 1.0)
        min_rr=st.slider('R/R minimum', 1.0, 5.0, p['min_rr'], 0.1)
        max_pos=st.slider('Nb max positions', 1, 8, p['max_positions'], 1)
        splits=p['splits']; tpR=p['tpR']
    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        sl_mult = st.slider("SL (×ATR)", 1.0, 4.0, 2.5, 0.1)
        tp_mult = st.slider("TP (×ATR suggéré)", 1.0, 6.0, 3.8, 0.1)
        allow_shorts = st.toggle("Autoriser les shorts", value=True)
        active_names = st.multiselect("Stratégies actives", list(STRATS.keys()), default=list(STRATS.keys()))
    with c4:
        macro_enabled = st.toggle("Activer macro gate (VIX/Gold)", value=True)
        vix_caution = st.slider("Seuil VIX prudence", 10.0, 35.0, 20.0, 0.5)
        vix_riskoff = st.slider("Seuil VIX risk-off", 15.0, 50.0, 28.0, 0.5)
        gold_mom_thresh = st.slider("Seuil Gold momentum (3m)", 0.0, 0.3, 0.10, 0.01)

tabs = st.tabs(['🏠 Décision', '📈 Portefeuille', '🧾 Historique', '🧪 Lab'])

# ────────────────────────────────────────────────────────────────────────────────
# 1) Décision — Scanner + Prendre
# ────────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Top Picks (1 clic)")

    eq, dd, day_chg, paused_until = circuit_breaker_update(capital, dd_limit=0.08, daily_limit=0.03, pause_hours=12)
    st.caption(f"Equity={eq:.2f} · DD={dd*100:.1f}% · Jour={day_chg*100:.1f}%")
    if circuit_breaker_active():
        st.warning(f"⛔ Circuit-breaker actif jusqu’à {paused_until}. Pas de nouveaux trades.")
    else:
        macro_mult, macro_note = macro_gate(macro_enabled, vix_caution, vix_riskoff, gold_mom_thresh)
        st.caption(f"Macro gate: {macro_note} → multiplicateur {macro_mult}")

        if st.button("🚀 Scanner maintenant", use_container_width=True):
            sanitize_all_positions()
            rows=[]
            for sym in symbols:
                try:
                    df = load_or_fetch(exchange, sym, tf, limit=1200)
                    df_htf = load_or_fetch(exchange, sym, htf, limit=600)
                except Exception as e:
                    st.warning(f"Skip {sym}: {e}"); continue

                use = {name: STRATS[name] for name in active_names if name in STRATS}
                if not use:
                    st.warning("Aucune stratégie active."); break

                signals = {name: fn(df) for name, fn in use.items()}
                w = ensemble_weights(df, signals, window=300)
                sig = blended_signal(signals, w)
                gate = htf_gate(df, df_htf)
                sig = (sig * gate).clip(-1,1) * macro_mult
                d = int(np.sign(sig.iloc[-1]))
                if d==0 or (d<0 and not allow_shorts): continue

                lvl = atr_levels(df, d, atr_mult_sl=sl_mult, atr_mult_tp=tp_mult)
                if not lvl: continue
                this_rr = rr(lvl['entry'], lvl['sl'], lvl['tp'])
                if this_rr < min_rr: continue
                qty = size_fixed_pct(capital, lvl['entry'], lvl['sl'], risk_pct)
                if qty <= 0: continue

                rows.append({'symbol': sym, 'dir':'LONG' if d>0 else 'SHORT',
                             'entry':lvl['entry'],'sl':lvl['sl'],'tp':lvl['tp'],
                             'rr': this_rr, 'qty': qty})

            if not rows:
                st.info("Aucun setup solide pour l’instant.")
            else:
                picks = pd.DataFrame(rows).sort_values('rr', ascending=False).head(int(max_pos))
                st.dataframe(picks[['symbol','dir','entry','sl','tp','rr','qty']].round(6), use_container_width=True)
                st.markdown("#### Exécution")
                price_mode = st.selectbox("Prix d'entrée", ["Suggéré (entry)", "Prix du marché"])
                if st.button("📌 Prendre tous (sélection)", type="primary"):
                    for _, r in picks.iterrows():
                        entry = float(r['entry'])
                        if price_mode=="Prix du marché":
                            pxm = fetch_last_price(exchange, r['symbol'])
                            if not np.isnan(pxm): entry=float(pxm)
                        meta = build_meta_r(entry, float(r['sl']), r['dir'], float(r['qty']), splits=splits, tpR=tpR, be_after_tp1=True)
                        open_position(r['symbol'], r['dir'], entry, float(r['sl']), float(r['tp']), float(r['qty']), meta=meta)
                    st.success("Ouvert ✅"); st.rerun()

# ────────────────────────────────────────────────────────────────────────────────
# 2) Portefeuille — suivi + actions dynamiques
# ────────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Positions ouvertes")
    open_df = list_positions(status='OPEN')
    if open_df.empty:
        st.info("Aucune position ouverte.")
    else:
        last_prices = {s: fetch_last_price(exchange, s) for s in open_df['symbol'].unique()}
        def latent(row):
            last = last_prices.get(row['symbol'], row['entry'])
            sign = 1 if row['side']=='LONG' else -1
            return (last-row['entry'])*row['qty']*sign
        def ret_pct(row):
            last = last_prices.get(row['symbol'], row['entry'])
            sign = 1 if row['side']=='LONG' else -1
            return 100.0 * (last-row['entry'])/row['entry'] * sign

        open_df['last'] = open_df['symbol'].map(last_prices)
        open_df['PnL_latent'] = open_df.apply(latent, axis=1)
        open_df['ret_%'] = open_df.apply(ret_pct, axis=1).round(3)
        open_df['sens'] = np.where(open_df['ret_%']>=0, '✅ OK', '❌ Pas OK')
        st.dataframe(open_df[['id','symbol','side','entry','sl','tp','qty','last','ret_%','PnL_latent','sens','note']].round(6),
                     use_container_width=True)

        equity = portfolio_equity(capital, price_map=last_prices)
        st.metric("Équity dynamique", f"{equity:.2f} USD")

        # Gestion auto (TP/SL/BE/Trail)
        if st.button("🔄 Mettre à jour (TP/SL + BE/Trailing)"):
            ohlc_map = {s: load_or_fetch(exchange, s, tf, limit=300) for s in open_df['symbol'].unique()}
            events = auto_manage_positions(last_prices, ohlc_map=ohlc_map, mode=mode, be_after_tp1=True, trail_after_tp2=True, fee_buffer_bps=5)
            for sym, why, px, q in events:
                st.success(f"{sym}: {why} @ {px:.6f} (qty {q:.4f})")
            st.rerun()

        st.markdown("### Actions rapides (dynamiques)")
        for _, r in open_df.iterrows():
            meta = _meta_from_note(r['note'])
            cols = st.columns([3,1.1,1.1,1.1,1.3])
            cols[0].markdown(f"**{r['symbol']}** · {r['side']} · qty `{r['qty']:.4f}` · SL `{r['sl']:.6f}`")
            # Badges TP
            if isinstance(meta, dict) and meta.get('multi_tp'):
                tags=[]
                for t in meta.get('targets',[]):
                    tick = "✅" if t.get('filled') else "🟡"
                    tags.append(f"{tick} {t.get('name')} @ {t.get('px',0):.6f}")
                cols[0].caption(" | ".join(tags))

            # SL -> BE
            if cols[1].button("SL→BE", key=f"be_{r['id']}"):
                update_sl(int(r['id']), float(r['entry'])); st.rerun()

            # Force next TP (ferme la part prévue suivante)
            next_qty = 0.0
            next_name = "NEXT"
            if isinstance(meta, dict) and meta.get('multi_tp'):
                for t in meta['targets']:
                    if not t.get('filled'):
                        next_qty = float(min(r['qty'], t.get('qty',0.0)))
                        next_name = t.get('name','NEXT')
                        break
            label_next = f"Force {next_name}" if next_qty>0 else "Rien à forcer"
            if cols[2].button(label_next, key=f"force_{r['id']}", disabled=(next_qty<=0)):
                px = last_prices.get(r['symbol'], r['entry'])
                partial_close(int(r['id']), float(px), float(next_qty), f"FORCE_{next_name}")
                # si on vient de forcer TP1 → mettre BE
                if next_name=="TP1":
                    update_sl(int(r['id']), float(r['entry']))
                st.rerun()

            # Close next % (dynamique, basé sur part du prochain TP vs qty restante)
            dynamic_pct = int(round(100.0*next_qty/max(r['qty'],1e-12))) if next_qty>0 else 25
            if cols[3].button(f"Close {dynamic_pct}%", key=f"dyn_{r['id']}", disabled=(next_qty<=0)):
                px = last_prices.get(r['symbol'], r['entry'])
                partial_close(int(r['id']), float(px), float(r['qty'])*dynamic_pct/100.0, f"MANUAL_{dynamic_pct}")
                st.rerun()

            # Trail manuel (chandelier)
            if cols[4].button("Trail", key=f"trail_{r['id']}"):
                df_now = load_or_fetch(exchange, r['symbol'], tf, limit=220)
                side = r['side'].upper()
                trail = float(_chandelier_stop(df_now, n=22, k=3.0, side=side)[-1])
                if side=='LONG' and trail>r['sl']: update_sl(int(r['id']), trail)
                if side=='SHORT' and trail<r['sl']: update_sl(int(r['id']), trail)
                st.rerun()

        # Fermer 100% (séparé)
        st.markdown("### Fermer totalement")
        for _, r in open_df.iterrows():
            if st.button(f"Fermer {r['symbol']} (100%)", key=f"close_{r['id']}"):
                px = last_prices.get(r['symbol'], r['entry'])
                close_position(int(r['id']), float(px), "MANUAL_CLOSE")
                st.rerun()

# ────────────────────────────────────────────────────────────────────────────────
# 3) Historique — stats
# ────────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Historique (clôturées)")
    hist = list_positions(status='CLOSED')
    if hist.empty:
        st.info("Pas encore d’historique.")
    else:
        st.dataframe(hist[['close_ts','symbol','side','entry','exit_price','qty','pnl','note']].round(6), use_container_width=True)
        pnl = float(hist['pnl'].sum())
        wins = (hist['pnl']>0).sum(); total=len(hist); winrate = 0.0 if total==0 else wins/total*100
        avgwin = float(hist.loc[hist['pnl']>0,'pnl'].mean()) if wins>0 else 0.0
        avgloss = float(hist.loc[hist['pnl']<=0,'pnl'].mean()) if (total-wins)>0 else 0.0
        pf = (avgwin/abs(avgloss)) if avgloss<0 else np.nan
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("P&L réalisé", f"{pnl:.2f}")
        c2.metric("Win rate", f"{winrate:.1f}%")
        c3.metric("Profit factor", f"{pf:.2f}" if not np.isnan(pf) else "—")
        c4.metric("Avg win / loss", f"{avgwin:.2f} / {avgloss:.2f}")

# ────────────────────────────────────────────────────────────────────────────────
# 4) Lab — backtest rapide (visuel simple)
# ────────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Lab — Backtest rapide")
    sym_b = st.selectbox("Symbole", symbols_default, index=0, key="lab_sym")
    tf_b  = st.selectbox("TF", ['15m','1h','4h'], index=1, key="lab_tf")
    n_years = st.slider("Années d'historique (approx)", 1, 4, 3)
    names = st.multiselect("Stratégies à tester", list(STRATS.keys()), default=['EMA Trend','MACD Momentum','SuperTrend','Bollinger MR','Ichimoku'])
    if st.button("▶︎ Lancer le backtest"):
        try:
            df = load_or_fetch(exchange, sym_b, tf_b, limit=2000)
            res = []
            for nm in names:
                sig = STRATS[nm](df)
                _,_,pnl,eq = compute(df, sig)
                res.append(dict(name=nm, sharpe=sharpe(pnl), mdd=max_drawdown(eq), cagr=(eq.iloc[-1]**(365*24/len(eq))-1)))
            out = pd.DataFrame(res).sort_values("sharpe", ascending=False)
            st.dataframe(out.round(4), use_container_width=True)
        except Exception as e:
            st.error(str(e))

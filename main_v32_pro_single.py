# HELIOS ONE — V4.5 ULTRA (single file)
# Tout-en-un : sélection par trade + batch + simulation, Lab rapide, Equity visible,
# cap d'expo global + par trade (+ levier par mode), enregistrement du MODE,
# Multi-TP (TP1->BE, TP2->trail, TP3->full), auto-manage, stats par MODE.

import os, json, sqlite3, datetime
import streamlit as st
import pandas as pd
import numpy as np

# ─────────── Page config ───────────
ICON = "☀️"
try:
    from PIL import Image
    if os.path.exists("app_icon.png"):
        ICON = Image.open("app_icon.png")
except Exception:
    pass

st.set_page_config(page_title="HELIOS ONE — V4.5 ULTRA", page_icon=ICON, layout="centered")
st.title("HELIOS ONE — V4.5 ULTRA")
st.caption("Ensemble 20+ strats • Sélection par trade • Multi-TP • BE & Trailing • Cap expo • Stats par MODE")

# ─────────── Dépendances externes ───────────
try:
    import ccxt
except Exception:
    st.error("❗ Il manque `ccxt` (ajoute-le à requirements.txt)."); st.stop()
try:
    import yfinance as yf
    HAVE_YF = True
except Exception:
    HAVE_YF = False

# ─────────── Exchange helpers ───────────
FALLBACK = ['okx','bybit','kraken','coinbase','kucoin','binance']

def build_exchange(name: str):
    ex_cls = getattr(ccxt, name.lower())
    ex = ex_cls({'enableRateLimit': True, 'options': {'adjustForTimeDifference': True}})
    try: ex.load_markets()
    except Exception: pass
    return ex

def _map_symbol(exchange_id: str, symbol: str) -> str:
    if exchange_id == 'kraken' and symbol.startswith('BTC/'): return symbol.replace('BTC/','XBT/')
    if exchange_id == 'coinbase' and symbol.endswith('/USDT'): return symbol.replace('/USDT','/USDC')
    return symbol

def fetch_ohlcv(exchange_id: str, symbol: str, timeframe='1h', limit=1500) -> pd.DataFrame:
    ex = build_exchange(exchange_id); sym = _map_symbol(exchange_id, symbol)
    data = ex.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    return df.set_index('ts')

def load_or_fetch(exchange_id: str, symbol: str, tf: str, limit=1500) -> pd.DataFrame:
    last_err = None
    for ex in [exchange_id] + [e for e in FALLBACK if e != exchange_id]:
        try: return fetch_ohlcv(ex, symbol, tf, limit)
        except Exception as e: last_err = e
    raise RuntimeError(f"OHLCV échec {symbol} {tf}: {last_err}")

def fetch_last_price(exchange_id: str, symbol: str) -> float:
    for ex in [exchange_id] + [e for e in FALLBACK if e != exchange_id]:
        try:
            inst = build_exchange(ex); sym = _map_symbol(ex, symbol)
            t = inst.fetch_ticker(sym); px = t.get('last') or t.get('close')
            if px: return float(px)
        except Exception: continue
    return np.nan

def yf_series(ticker: str, period="5y"):
    if not HAVE_YF: return None
    try:
        y = yf.download(ticker, period=period, interval="1d", progress=False)
        if y is None or y.empty: return None
        return y['Adj Close'].rename(ticker).tz_localize("UTC")
    except Exception: return None

# ─────────── Indicateurs & Strats (24) ───────────
def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    dn=-d.clip(upper=0).ewm(alpha=1/n,adjust=False).mean(); rs=up/(dn+1e-9)
    return 100-100/(1+rs)
def atr_df(df,n=14):
    hl=df['high']-df['low']; hc=(df['high']-df['close'].shift()).abs(); lc=(df['low']-df['close'].shift()).abs()
    tr=pd.concat([hl,hc,lc],axis=1).max(axis=1); return tr.ewm(span=n,adjust=False).mean()
def kama(series, er_len=10, fast=2, slow=30):
    change=series.diff(er_len).abs(); vol=series.diff().abs().rolling(er_len).sum(); er=change/(vol+1e-9)
    sc=(er*(2/(fast+1)-2/(slow+1))+2/(slow+1))**2; out=[series.iloc[0]]
    for i in range(1,len(series)): out.append(out[-1]+sc.iloc[i]*(series.iloc[i]-out[-1]))
    return pd.Series(out,index=series.index)

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
def sig_psar(df,af=0.02,max_af=0.2):
    h,l=df['high'],df['low']; ps=l.copy(); bull=True; a=af; ep=h.iloc[0]; ps.iloc[0]=l.iloc[0]
    for i in range(2,len(df)):
        pv=ps.iloc[i-1]
        if bull:
            ps.iloc[i]=min(pv+a*(ep-pv), l.iloc[i-1], l.iloc[i-2])
            if h.iloc[i]>ep: ep=h.iloc[i]; a=min(max_af,a+af)
            if l.iloc[i]<ps.iloc[i]: bull=False; ps.iloc[i]=ep; ep=l.iloc[i]; a=af
        else:
            ps.iloc[i]=max(pv+a*(ep-pv), h.iloc[i-1], h.iloc[i-2])
            if l.iloc[i]<ep: ep=l.iloc[i]; a=min(max_af,a+af)
            if h.iloc[i]>ps.iloc[i]: bull=True; ps.iloc[i]=ep; ep=h.iloc[i]; a=af
    return ((df['close']>ps).astype(int)-(df['close']<ps).astype(int)).rename('signal')
def sig_mfi_mr(df,n=14,lo=20,hi=80):
    tp=(df['high']+df['low']+df['close'])/3; mf=tp*df['volume']
    pos=mf.where(tp>tp.shift(),0.0); neg=mf.where(tp<tp.shift(),0.0).abs()
    mr=100-100/(1+(pos.rolling(n).sum()/(neg.rolling(n).sum()+1e-9)))
    return ((mr<lo).astype(int)-(mr>hi).astype(int)).rename('signal')
def sig_obv_trend(df,n=20):
    ch=np.sign(df['close'].diff().fillna(0.0)); obv=(df['volume']*ch).cumsum(); e=ema(obv,n)
    return ((obv>e).astype(int)-(obv<e).astype(int)).rename('signal')

STRATS = {
    'EMA Trend':sig_ema_trend,'MACD Momentum':sig_macd,'Donchian Breakout':sig_donchian,'SuperTrend':sig_supertrend,
    'ATR Channel':sig_atr_channel,'Bollinger MR':sig_boll_mr,'Ichimoku':sig_ichimoku,'KAMA Trend':sig_kama_trend,'RSI MR':sig_rsi_mr,
    'PPO':sig_ppo,'ADX Trend':sig_adx_trend,'StochRSI':sig_stoch_rsi,'CCI MR':sig_cci_mr,'Heikin Trend':sig_heikin_trend,'Chandelier':sig_chandelier,
    'VWAP MR':sig_vwap_mr,'TurtleSoup':sig_turtle_soup,'ZScore MR':sig_zscore,'TSI Momentum':sig_tsi,'EMA Ribbon':sig_ema_ribbon,
    'Keltner BO':sig_keltner,'PSAR Trend':sig_psar,'MFI MR':sig_mfi_mr,'OBV Trend':sig_obv_trend
}

# ─────────── Ensemble / Gating ───────────
def compute(df, signal, fee_bps=2.0, slip_bps=1.0):
    ret=df['close'].pct_change().fillna(0.0)
    pos=signal.shift().fillna(0.0).clip(-1,1)
    cost=(pos.diff().abs().fillna(0.0))*((fee_bps+slip_bps)/10000.0)
    pnl=pos*ret - cost
    eq=(1+pnl).cumprod()
    return ret,pos,pnl,eq

def sharpe(pnl,pp=365*24):
    s=pnl.std()
    return 0.0 if s==0 or np.isnan(s) else float(pnl.mean()/s*np.sqrt(pp))

def maxdd(eq):
    peak=eq.cummax(); dd=eq/peak-1
    return float(dd.min())

def _score(p,eq):
    s=max(0.0,min(3.0,sharpe(p))); dd=abs(maxdd(eq))
    return s+(1.0-min(dd,0.4))

def ensemble_weights(df,signals,window=300):
    if not signals: return pd.Series(dtype=float)
    start=max(0,len(df)-int(window)); sc={}
    for n,s in signals.items():
        try:_,_,p,eq=compute(df.iloc[start:],s.iloc[start:]); sc[n]=_score(p,eq)
        except Exception: sc[n]=-1e9
    keys=list(sc.keys()); arr=np.array([sc[k] for k in keys]); arr=arr-np.nanmax(arr)
    w=np.exp(arr); w=w/np.nansum(w) if np.nansum(w)!=0 else np.ones_like(w)/len(w)
    return pd.Series(w,index=keys)

def blended_signal(signals,weights):
    if not signals: return pd.Series(dtype=float,name="signal")
    df=pd.concat(signals.values(),axis=1).fillna(0.0); df.columns=list(signals.keys())
    w=weights.reindex(df.columns).fillna(0.0).values.reshape(1,-1)
    pos=(df.values*w).sum(axis=1)
    return pd.Series(pos,index=df.index,name="signal").clip(-1,1)

def htf_gate(df_ltf,df_htf): return sig_ema_trend(df_htf).reindex(df_ltf.index).ffill().fillna(0.0)

def yf_macro(enable, vix_caution=20.0, vix_riskoff=28.0, gold_mom_thr=0.10):
    if not enable: return 1.0,"macro OFF"
    if not HAVE_YF: return 1.0,"no_yfinance"
    vix=yf_series("^VIX"); gold=yf_series("GC=F")
    if vix is None or vix.empty: return 1.0,"no_vix"
    lvl=float(vix.iloc[-1]); mult=1.0; note=[]
    if lvl>float(vix_riskoff): mult*=0.0; note.append(f"VIX>{vix_riskoff} risk-off")
    elif lvl>float(vix_caution): mult*=0.5; note.append(f"VIX>{vix_caution} caution")
    else: note.append("VIX benign")
    if gold is not None and not gold.empty:
        mom=float(gold.pct_change(63).iloc[-1])
        if mom>float(gold_mom_thr): mult*=0.8; note.append("Gold strong")
    return mult," | ".join(note)

# ─────────── Risk & niveaux ───────────
def atr_levels(df,d,sl_mult=2.5,tp_mult=4.0):
    if d==0 or len(df)<2: return None
    a=float(atr_df(df,14).iloc[-1]); price=float(df['close'].iloc[-1])
    sl = price - sl_mult*a if d>0 else price + sl_mult*a
    tp = price + tp_mult*a if d>0 else price - tp_mult*a
    return {'entry':price,'sl':sl,'tp':tp,'atr':a}

def size_fixed_pct(equity,entry,stop,risk_pct):
    per=abs(entry-stop); risk=equity*(risk_pct/100.0)
    return 0.0 if per<=0 else risk/per

def rr(entry,sl,tp):
    R=abs(entry-sl)
    return float(abs(tp-entry)/(R if R>0 else 1e-9))

# ─────────── SQLite (positions + kv) ───────────
DB=os.path.join(os.path.dirname(__file__),'portfolio.db')

def _init_db():
    conn=sqlite3.connect(DB); c=conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS positions (
      id INTEGER PRIMARY KEY AUTOINCREMENT, open_ts TEXT, close_ts TEXT, symbol TEXT, side TEXT,
      entry REAL, sl REAL, tp REAL, qty REAL, status TEXT, exit_price REAL, pnl REAL, note TEXT)''')
    c.execute('CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)')
    conn.commit(); conn.close()

def kv_get(k,default=None):
    _init_db(); conn=sqlite3.connect(DB)
    r=conn.execute('SELECT v FROM kv WHERE k=?',(k,)).fetchone(); conn.close()
    return json.loads(r[0]) if r else default

def kv_set(k,v):
    _init_db(); conn=sqlite3.connect(DB)
    conn.execute('INSERT OR REPLACE INTO kv(k,v) VALUES(?,?)',(k,json.dumps(v)))
    conn.commit(); conn.close()

def list_positions(status=None,limit=100000):
    _init_db(); conn=sqlite3.connect(DB)
    q='SELECT id,open_ts,close_ts,symbol,side,entry,sl,tp,qty,status,exit_price,pnl,note FROM positions'; pr=()
    if status in("OPEN","CLOSED"): q+=' WHERE status=?'; pr=(status,)
    q+=' ORDER BY id DESC LIMIT ?'; pr=pr+(int(limit),)
    rows=list(conn.execute(q,pr)); conn.close()
    return pd.DataFrame(rows,columns=['id','open_ts','close_ts','symbol','side','entry','sl','tp','qty','status','exit_price','pnl','note'])

def _meta_from_note(note):
    if isinstance(note,str) and note.startswith("META:"):
        try: return json.loads(note[5:])
        except Exception: return None
    return None

def _meta_to_note(meta): return "META:"+json.dumps(meta,separators=(',',':'))

def _encode_close_note(reason,trade_mode):
    try: return "META2:"+json.dumps({"mode":str(trade_mode),"reason":str(reason)})
    except Exception: return str(reason)

def _set_meta(pid,meta):
    conn=sqlite3.connect(DB)
    conn.execute('UPDATE positions SET note=? WHERE id=? AND status="OPEN"',(_meta_to_note(meta),int(pid)))
    conn.commit(); conn.close()

def open_position(symbol,side,entry,sl,tp,qty,meta=None):
    _init_db()
    if qty is None or float(qty)<=0: return None
    if side.upper()=="LONG" and sl>=entry: sl=entry-abs(sl-entry) or entry-1e-9
    if side.upper()=="SHORT" and sl<=entry: sl=entry+abs(sl-entry) or entry+1e-9
    note=_meta_to_note(meta) if isinstance(meta,dict) else ''
    conn=sqlite3.connect(DB); c=conn.cursor()
    c.execute('INSERT INTO positions (open_ts,close_ts,symbol,side,entry,sl,tp,qty,status,exit_price,pnl,note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
              (datetime.datetime.utcnow().isoformat(),None,symbol,side.upper(),float(entry),float(sl),float(tp),float(qty),"OPEN",None,None,note))
    conn.commit(); rid=c.lastrowid; conn.close(); return rid

def update_sl(pid,new_sl):
    _init_db(); conn=sqlite3.connect(DB)
    conn.execute('UPDATE positions SET sl=? WHERE id=? AND status="OPEN"',(float(new_sl),int(pid)))
    conn.commit(); conn.close()

def close_position(pid,px,note='CLOSE'):
    _init_db(); conn=sqlite3.connect(DB); c=conn.cursor()
    row=c.execute('SELECT open_ts,symbol,side,entry,sl,tp,qty,note FROM positions WHERE id=? AND status="OPEN"',(pid,)).fetchone()
    if not row: conn.close(); return None
    open_ts,symbol,side,entry,sl,tp,qty,note_open=row
    trade_mode=(_meta_from_note(note_open) or {}).get('trade_mode','unknown')
    pnl=(float(px)-float(entry))*float(qty)*(1 if side.upper()=="LONG" else -1)
    c.execute('UPDATE positions SET close_ts=?, status=?, exit_price=?, pnl=?, note=? WHERE id=?',
              (datetime.datetime.utcnow().isoformat(),"CLOSED",float(px),float(pnl),_encode_close_note(note,trade_mode),pid))
    conn.commit(); conn.close(); return pnl

def partial_close(pid,px,qty_close,reason="TP"):
    _init_db(); conn=sqlite3.connect(DB); c=conn.cursor()
    row=c.execute('SELECT open_ts,symbol,side,entry,sl,tp,qty,note FROM positions WHERE id=? AND status="OPEN"',(pid,)).fetchone()
    if not row: conn.close(); return None
    open_ts,symbol,side,entry,sl,tp,qty,note_open=row
    trade_mode=(_meta_from_note(note_open) or {}).get('trade_mode','unknown')
    qty_close=float(min(max(qty_close,0.0),float(qty)))
    if qty_close<=0: conn.close(); return None
    sign=1 if side.upper()=="LONG" else -1; pnl=(float(px)-float(entry))*qty_close*sign
    c.execute('INSERT INTO positions (open_ts,close_ts,symbol,side,entry,sl,tp,qty,status,exit_price,pnl,note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
              (open_ts,datetime.datetime.utcnow().isoformat(),symbol,side,float(entry),float(sl),float(tp),float(qty_close),
               "CLOSED",float(px),float(pnl),_encode_close_note(reason,trade_mode)))
    remain=float(qty)-qty_close
    if remain>1e-12:
        c.execute('UPDATE positions SET qty=? WHERE id=? AND status="OPEN"',(remain,pid))
    else:
        c.execute('UPDATE positions SET close_ts=?, status=?, exit_price=?, pnl=?, note=? WHERE id=?',
                  (datetime.datetime.utcnow().isoformat(),"CLOSED",float(px),float(pnl),_encode_close_note(reason+"(FULL)",trade_mode),pid))
    conn.commit(); conn.close(); return pnl

# ─────────── Multi-TP helpers ───────────
def r_targets(entry,sl,side,tpR=(1.0,2.0,3.5)):
    side=side.upper(); sign=1 if side=="LONG" else -1
    R=(entry-sl) if side=="LONG" else (sl-entry)
    if R<=0: return None
    return [entry+sign*r*R for r in tpR]

def build_meta_r(entry,sl,side,qty,splits=(0.4,0.4,0.2),tpR=(1.0,2.0,3.5),be_after_tp1=True,trade_mode="Normal"):
    tps=r_targets(entry,sl,side,tpR)
    if not tps: return None
    q1,q2,q3=[float(qty*max(0.0,s)) for s in splits]; diff=float(qty)-(q1+q2+q3); q3=max(0.0,q3+diff)
    return {'multi_tp':True,'mode':'R','trade_mode':str(trade_mode),'tpR':list(tpR),'splits':list(splits),
            'targets':[{'name':'TP1','px':tps[0],'qty':q1,'filled':False},
                       {'name':'TP2','px':tps[1],'qty':q2,'filled':False},
                       {'name':'TP3','px':tps[2],'qty':q3,'filled':False}],
            'be_after_tp1':bool(be_after_tp1),'trail_after_tp2':True}

def normalize_meta(meta,qty,entry,sl,side,default_splits=(0.4,0.4,0.2),default_tpR=(1.0,2.0,3.5),trade_mode="Normal"):
    try:
        if not isinstance(meta,dict) or not meta.get('multi_tp'):
            return build_meta_r(entry,sl,side,qty,splits=default_splits,tpR=default_tpR,trade_mode=trade_mode)
        splits=tuple(meta.get('splits',default_splits)); tpR=tuple(meta.get('tpR',default_tpR)); tgt=meta.get('targets',[])
        want=r_targets(entry,sl,side,tpR)
        if (not isinstance(tgt,list)) or (len(tgt)==0) or isinstance(tgt[0],(int,float,str)):
            return build_meta_r(entry,sl,side,qty,splits=splits,tpR=tpR,be_after_tp1=bool(meta.get('be_after_tp1',True)),trade_mode=trade_mode)
        for i in range(min(3,len(tgt))):
            if not isinstance(tgt[i],dict): tgt[i]={}
            tgt[i].setdefault('name',f"TP{i+1}"); tgt[i].setdefault('px',want[i])
            tgt[i].setdefault('qty',max(0.0,qty*(splits[i] if i<len(splits) else 0.0))); tgt[i].setdefault('filled',False)
        meta['targets']=tgt[:3]; meta.setdefault('be_after_tp1',True); meta.setdefault('trail_after_tp2',True)
        meta['tpR']=list(tpR); meta['splits']=list(splits); meta['trade_mode']=str(meta.get('trade_mode',trade_mode))
        return meta
    except Exception:
        return build_meta_r(entry,sl,side,qty,splits=default_splits,tpR=default_tpR,trade_mode=trade_mode)

# ─────────── Auto-manage (TP/SL/BE/Trail) ───────────
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
            conn=sqlite3.connect(DB); conn.execute('UPDATE positions SET sl=? WHERE id=?',(float(new),int(r['id']))); conn.commit(); conn.close(); changed=True
    return changed

def auto_manage_positions(price_map,ohlc_map=None,mode="Normal",be_after_tp1=True,trail_after_tp2=True,fee_buffer_bps=5):
    sanitize_all_positions(); df=list_positions(status='OPEN'); 
    if df.empty: return []
    if str(mode).lower().startswith("conserv"): parts=(0.50,0.35,0.15); tpsR=(0.9,1.7,2.6)
    elif str(mode).lower().startswith("aggr") and "super" not in str(mode).lower(): parts=(0.30,0.40,0.30); tpsR=(1.0,2.5,5.0)
    elif "super" in str(mode).lower(): parts=(0.34,0.33,0.33); tpsR=(1.2,3.0,6.0)
    else: parts=(0.40,0.40,0.20); tpsR=(1.0,2.0,3.5)
    evts=[]
    for _,r in df.iterrows():
        sym=r['symbol']; side=r['side'].upper()
        if sym not in price_map: continue
        px=float(price_map[sym]); entry=float(r['entry']); sl=float(r['sl']); qty=float(r['qty'])
        if qty<=1e-12: continue
        R=(entry-sl) if side=="LONG" else (sl-entry)
        if R<=0: continue
        meta_raw=_meta_from_note(r['note']); meta=normalize_meta(meta_raw,qty,entry,sl,side,trade_mode=mode)
        if meta_raw!=meta: _set_meta(int(r['id']),meta)
        tps=r_targets(entry,sl,side,tuple(meta.get('tpR',tpsR)))
        for j in range(3):
            if 'targets'in meta and len(meta['targets'])>j: meta['targets'][j]['px']=tps[j]
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
            if changed: _set_meta(int(r['id']),meta); 
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
            if changed: _set_meta(int(r['id']),meta); 
            continue
        # TP3
        if not meta['targets'][2]['filled'] and hit_tp(px,meta['targets'][2]['px']):
            partial_close(int(r['id']),px,qty_left,"TP3"); evts.append((sym,"TP3",px,qty_left)); meta['targets'][2]['filled']=True; meta['targets'][2]['qty']=0.0; changed=True
            qty_left=0.0
        cur=list_positions(status='OPEN'); cur=cur.loc[cur['id']==r['id']]
        if not cur.empty:
            stop=float(cur.iloc[0]['sl'])
            if qty_left>0 and hit_sl(px,stop): partial_close(int(r['id']),stop,qty_left,"SL"); evts.append((sym,"SL",stop,qty_left)); qty_left=0.0
        if changed: _set_meta(int(r['id']),meta)
    return evts

# ─────────── Equity & expo ───────────
def portfolio_equity(base_capital,price_map=None):
    open_df=list_positions(status='OPEN'); closed_df=list_positions(status='CLOSED')
    realized=0.0 if closed_df.empty else float(closed_df['pnl'].sum()); latent=0.0
    if not open_df.empty:
        if price_map is None: price_map={s:fetch_last_price('okx',s) for s in open_df['symbol'].unique()}
        for _,r in open_df.iterrows():
            px=float(price_map.get(r['symbol'],r['entry'])); sign=1 if r['side']=='LONG' else -1
            latent+=(px-float(r['entry']))*float(r['qty'])*sign
    return base_capital+realized+latent

# ─────────── UI: Modes & réglages ───────────
symbols_default=['BTC/USDT','ETH/USDT','BNB/USDT','SOL/USDT','XRP/USDT','ADA/USDT','AVAX/USDT','LINK/USDT','TON/USDT','DOGE/USDT']

st.markdown("### ⚙️ Mode & Réglages")
modes={
 'Conservateur': dict(risk_pct=1.0, max_expo=60.0,  per_trade_cap=25.0, min_rr=1.8, max_positions=2, splits=(0.5,0.35,0.15), tpR=(0.9,1.7,2.6), gate_thr=0.35, leverage=1.0),
 'Normal':       dict(risk_pct=2.0, max_expo=100.0, per_trade_cap=35.0, min_rr=1.8, max_positions=3, splits=(0.4,0.4,0.2), tpR=(1.0,2.0,3.5), gate_thr=0.30, leverage=1.0),
 'Agressif':     dict(risk_pct=5.0, max_expo=150.0, per_trade_cap=40.0, min_rr=1.6, max_positions=5, splits=(0.30,0.40,0.30), tpR=(1.0,2.5,5.0), gate_thr=0.22, leverage=1.5),
 'Super agressif (x5)': dict(risk_pct=2.5, max_expo=120.0, per_trade_cap=20.0, min_rr=2.2, max_positions=3, splits=(0.34,0.33,0.33), tpR=(1.2,3.0,6.0), gate_thr=0.60, leverage=5.0)
}
mode = st.selectbox("Mode (je touche seulement à ça)", list(modes.keys()), index=1)
m = modes[mode]

with st.expander("Réglages avancés (optionnel)"):
    exchange=st.selectbox('Exchange', FALLBACK, index=0)
    tf=st.selectbox('Timeframe', ['15m','1h','4h'], index=1)
    htf=st.selectbox('HTF confirm', ['1h','4h','1d'], index=2 if tf!='4h' else 1)
    symbols=st.multiselect('Paires', symbols_default, default=symbols_default[:8])
    sl_mult=st.slider("SL (×ATR)", 1.0, 4.0, 2.5, 0.1)
    tp_mult=st.slider("TP (×ATR suggéré)", 1.0, 6.0, 4.0, 0.1)
    macro_enabled=st.toggle("Macro gate (VIX/Gold)", value=True)
    # capital editable
    capital_edit = st.number_input("Capital de base (pour l'équity)", min_value=0.0, value=float(kv_get('base_capital',1000.0)), step=100.0)
    if st.button("💾 Enregistrer capital"):
        kv_set('base_capital', float(capital_edit)); st.success("Capital mis à jour.")

# défaut si expander fermé
exchange  = locals().get('exchange','okx')
tf        = locals().get('tf','1h')
htf       = locals().get('htf','4h')
symbols   = locals().get('symbols',symbols_default[:8])
sl_mult   = locals().get('sl_mult',2.5)
tp_mult   = locals().get('tp_mult',4.0)
macro_enabled = locals().get('macro_enabled',True)

capital=float(kv_get('base_capital',1000.0))
eq_now=portfolio_equity(capital)
st.metric("💼 Portefeuille (équity dynamique)", f"{eq_now:.2f} USD")

tabs = st.tabs(['🏠 Décision','📈 Portefeuille','🧾 Historique','🔬 Lab'])

# ─────────── 1) Décision ───────────
with tabs[0]:
    st.subheader("Top Picks (1 clic)")
    macro_mult, macro_note = yf_macro(macro_enabled)
    st.caption(f"Macro: {macro_note} → multiplicateur {macro_mult}")
    if st.button("🚀 Scanner maintenant", use_container_width=True):
        rows=[]
        for sym in symbols:
            try:
                df=load_or_fetch(exchange,sym,tf,1200); df_htf=load_or_fetch(exchange,sym,htf,600)
            except Exception as e:
                st.warning(f"Skip {sym}: {e}"); continue
            signals={nm:fn(df) for nm,fn in STRATS.items()}
            w=ensemble_weights(df,signals,window=300); sig=blended_signal(signals,w)
            gate=htf_gate(df,df_htf); blended=(sig*gate).clip(-1,1)*macro_mult
            if abs(float(blended.iloc[-1])) < m['gate_thr']: continue
            d=int(np.sign(blended.iloc[-1])); 
            if d==0: continue
            lvl=atr_levels(df,d,sl_mult,tp_mult)
            if not lvl: continue
            this_rr=rr(lvl['entry'],lvl['sl'],lvl['tp'])
            if this_rr < m['min_rr']: continue
            qty=size_fixed_pct(eq_now,lvl['entry'],lvl['sl'],m['risk_pct'])
            if qty<=0: continue
            tps=r_targets(lvl['entry'],lvl['sl'],'LONG' if d>0 else 'SHORT', m['tpR'])
            rows.append({'symbol':sym,'dir':'LONG' if d>0 else 'SHORT','entry':lvl['entry'],'sl':lvl['sl'],'tp':lvl['tp'],
                         'rr':this_rr,'qty':qty,'tp1':tps[0],'tp2':tps[1],'tp3':tps[2]})
        if not rows:
            st.info("Aucun setup selon le mode choisi.")
        else:
            picks=pd.DataFrame(rows).sort_values('rr',ascending=False).head(int(m['max_positions']))
            # Expo & caps
            open_now=list_positions(status='OPEN'); open_notional=0.0 if open_now.empty else float((open_now['entry']*open_now['qty']).sum())
            eq=portfolio_equity(capital); cap_gross=eq*(m['max_expo']/100.0)*m['leverage']; room=max(0.0, cap_gross-open_notional)
            per_trade_cap = eq*(m['per_trade_cap']/100.0)
            st.caption(f"Expo en cours: {open_notional:.2f} / Cap: {cap_gross:.2f} (lev {m['leverage']}x) → Reste: {room:.2f} · Cap/trade: {per_trade_cap:.2f}")

            picks_display=picks[['symbol','dir','entry','sl','tp','tp1','tp2','tp3','rr','qty']].copy()
            picks_display.insert(0,'take',True)
            st.markdown("#### Sélection (cocher/ajuster) + actions par ligne")
            edit=st.data_editor(
                picks_display, hide_index=True, num_rows="fixed", use_container_width=True,
                column_config={
                    "take": st.column_config.CheckboxColumn("Prendre"),
                    "qty":  st.column_config.NumberColumn("qty", step=0.0001, format="%.6f"),
                    "rr":   st.column_config.NumberColumn("R/R", format="%.2f", disabled=True),
                    "tp1":  st.column_config.NumberColumn("TP1", format="%.6f", disabled=True),
                    "tp2":  st.column_config.NumberColumn("TP2", format="%.6f", disabled=True),
                    "tp3":  st.column_config.NumberColumn("TP3", format="%.6f", disabled=True),
                }
            )

            # Actions par ligne
            st.markdown("##### Actions par trade")
            for i, r in edit.iterrows():
                c1,c2,c3 = st.columns([2.3,1,1])
                c1.write(f"**{r['symbol']}** · {r['dir']} · entry `{r['entry']:.6f}` · SL `{r['sl']:.6f}` · R/R `{r['rr']:.2f}`")
                if c2.button("📌 Prendre ce trade", key=f"take_{i}"):
                    qty=float(r['qty']); notional=qty*float(r['entry'])
                    scale=1.0
                    if notional>per_trade_cap: scale=min(scale, per_trade_cap/max(notional,1e-9))
                    if notional>room:          scale=min(scale, room/max(notional,1e-9))
                    if scale<=0: st.warning("Cap atteint."); st.rerun()
                    entry=float(r['entry'])
                    meta=build_meta_r(entry,float(r['sl']),r['dir'],qty*scale,splits=m['splits'],tpR=m['tpR'],be_after_tp1=True,trade_mode=mode)
                    open_position(r['symbol'],r['dir'],entry,float(r['sl']),float(r['tp']),qty*scale,meta=meta)
                    st.success(f"{r['symbol']} ouvert ✅"); st.rerun()
                if c3.button("🔬 Lab", key=f"lab_{i}"):
                    try:
                        df=load_or_fetch(exchange,r['symbol'],tf,2000)
                        bucket=['EMA Trend','MACD Momentum','SuperTrend','Bollinger MR','Ichimoku','ADX Trend','OBV Trend']
                        res=[]
                        for nm in bucket:
                            sig=STRATS[nm](df); _,_,p,eq_=compute(df,sig)
                            res.append(dict(name=nm, sharpe=sharpe(p), mdd=maxdd(eq_), cagr=(eq_.iloc[-1]**(365*24/len(eq_))-1)))
                        st.info(f"Lab rapide — {r['symbol']} ({tf})")
                        st.dataframe(pd.DataFrame(res).sort_values("sharpe",ascending=False).round(4), use_container_width=True)
                    except Exception as e:
                        st.warning(str(e))

            # Batch
            sel=edit[edit['take']].copy()
            price_mode=st.selectbox("Prix d'entrée", ["Suggéré (entry)", "Prix du marché"])
            cA,cB=st.columns(2)
            if cA.button("📌 Prendre la sélection"):
                if sel.empty: st.warning("Rien de sélectionné.")
                else:
                    total_alloc=float((sel['qty']*sel['entry']).sum()); scale=1.0
                    if total_alloc>room: scale=min(scale, room/max(total_alloc,1e-9))
                    for _,r in sel.iterrows():
                        entry=float(r['entry']) if price_mode=="Suggéré (entry)" else float(fetch_last_price(exchange,r['symbol']) or r['entry'])
                        qty=float(r['qty'])*scale; notional=qty*entry
                        if notional>per_trade_cap: qty*=per_trade_cap/max(notional,1e-9)
                        if qty<=0: continue
                        meta=build_meta_r(entry,float(r['sl']),r['dir'],qty,splits=m['splits'],tpR=m['tpR'],be_after_tp1=True,trade_mode=mode)
                        open_position(r['symbol'],r['dir'],entry,float(r['sl']),float(r['tp']),qty,meta=meta)
                    st.success("Ouvert ✅"); st.rerun()
            if cB.button("🧮 Simuler l’allocation (sans ouvrir)"):
                alloc=float((sel['qty']*sel['entry']).sum()) if not sel.empty else 0.0
                st.info(f"Sélection: {len(sel)} trades · Allocation brute {alloc:.2f} USD  \n"
                        f"Cap/trade {per_trade_cap:.2f} USD · Espace cap global restant {room:.2f} USD")

# ─────────── 2) Portefeuille ───────────
with tabs[1]:
    st.subheader("Positions ouvertes")
    open_df=list_positions(status='OPEN')
    if open_df.empty:
        st.info("Aucune position.")
    else:
        open_df=open_df[open_df['qty']>1e-12]  # pas d'affichage qty=0
        if open_df.empty:
            st.info("Aucune position.")
        else:
            last={s:fetch_last_price(exchange,s) for s in open_df['symbol'].unique()}
            open_df['last']=open_df['symbol'].map(last)
            open_df['ret_%']=((open_df['last']-open_df['entry']).where(open_df['side']=='LONG', open_df['entry']-open_df['last'])/open_df['entry']*100).round(3)
            open_df['PnL_latent']=((open_df['last']-open_df['entry']).where(open_df['side']=='LONG', open_df['entry']-open_df['last'])*open_df['qty']).round(6)
            st.dataframe(open_df[['id','symbol','side','entry','sl','tp','qty','last','ret_%','PnL_latent','note']], use_container_width=True)
            st.metric("Équity dynamique", f"{portfolio_equity(capital,last):.2f} USD")
            if st.button("🔄 Mettre à jour (TP/SL + BE/Trailing)"):
                ohlc_map={s:load_or_fetch(exchange,s,tf,300) for s in open_df['symbol'].unique()}
                events=auto_manage_positions(last, ohlc_map, mode=mode, be_after_tp1=True, trail_after_tp2=True, fee_buffer_bps=5)
                for sym,why,px,q in events: st.success(f"{sym}: {why} @ {px:.6f} (qty {q:.4f})")
                st.rerun()

            st.markdown("### Actions rapides")
            for _,r in open_df.iterrows():
                meta=_meta_from_note(r['note']) or {}
                cols=st.columns([3,1.1,1.1,1.1,1.3])
                # badges TP
                tags=[]
                if isinstance(meta,dict) and meta.get('multi_tp'):
                    meta=normalize_meta(meta,r['qty'],r['entry'],r['sl'],r['side'],trade_mode=mode); _set_meta(int(r['id']),meta)
                    for t in meta.get('targets',[]):
                        if isinstance(t,dict):
                            tick="✅" if t.get('filled') else "🟡"; nm=str(t.get('name','TP')); px=float(t.get('px',r['tp']))
                            tags.append(f"{tick} {nm}@{px:.6f}")
                cols[0].markdown(f"**{r['symbol']}** · {r['side']} · qty `{r['qty']:.4f}` · SL `{r['sl']:.6f}`  \n" + (" | ".join(tags) if tags else "—"))
                if cols[1].button("SL→BE", key=f"be_{r['id']}"): update_sl(int(r['id']),float(r['entry'])); st.rerun()
                # Prochain TP
                next_qty=0.0; next_name="NEXT"
                if isinstance(meta,dict) and meta.get('multi_tp'):
                    for t in meta['targets']:
                        if not t.get('filled'):
                            next_qty=float(min(r['qty'],t.get('qty',0.0))); next_name=t.get('name','NEXT'); break
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

# ─────────── 3) Historique ───────────
with tabs[2]:
    st.subheader("Historique (clôturées)")
    hist=list_positions(status='CLOSED')
    if hist.empty:
        st.info("Pas encore d’historique.")
    else:
        def _mode_from_note(n):
            if isinstance(n,str):
                if n.startswith("META2:"):
                    try: return json.loads(n[6:]).get("mode","unknown")
                    except Exception: return "unknown"
                if n.startswith("META:"):
                    try: return (json.loads(n[5:]) or {}).get("trade_mode","unknown")
                    except Exception: return "unknown"
            return "unknown"
        hist["mode"]=hist["note"].apply(_mode_from_note)
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
        st.caption("NB: les anciens trades peuvent être 'unknown'. Les nouveaux sont tous tagués.")

# ─────────── 4) Lab ───────────
with tabs[3]:
    st.subheader("Lab — Backtest rapide")
    sym_b=st.selectbox("Symbole", symbols_default, index=0, key="lab_sym")
    tf_b=st.selectbox("TF", ['15m','1h','4h'], index=1, key="lab_tf")
    names=st.multiselect("Stratégies à tester", list(STRATS.keys()),
                         default=['EMA Trend','MACD Momentum','SuperTrend','Bollinger MR','Ichimoku'])
    if st.button("▶︎ Lancer le backtest"):
        try:
            df=load_or_fetch(exchange, sym_b, tf_b, 2000); res=[]
            for nm in names:
                sig=STRATS[nm](df); _,_,p,eq=compute(df,sig)
                res.append(dict(name=nm, sharpe=sharpe(p), mdd=maxdd(eq), cagr=(eq.iloc[-1]**(365*24/len(eq))-1)))
            st.dataframe(pd.DataFrame(res).sort_values("sharpe",ascending=False).round(4), use_container_width=True)
        except Exception as e:
            st.error(str(e))

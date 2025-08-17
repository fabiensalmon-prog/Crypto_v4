# HELIOS ONE — V5.2 ELITE+ (single file, mobile-friendly)
# Features: Ensemble 20+ strats • Multi-TP (TP1→BE, TP2, TP3) • Trailing • Time-stop
#           Kill-switch • Caps global & cluster • Sélection & %cap par trade • Stats par MODE
# requirements.txt (recommandé) :
#   streamlit
#   pandas
#   numpy
#   yfinance
#   ccxt           # optionnel (réel temps-réel exchange). Fallback YF auto si absent.
#   pillow         # optionnel pour app_icon.png

import os, json, sqlite3, datetime, math
import streamlit as st
import pandas as pd
import numpy as np

# ───────── Page & icône
ICON = "☀️"
try:
    from PIL import Image
    if os.path.exists("app_icon.png"):
        ICON = Image.open("app_icon.png")
except Exception:
    pass
st.set_page_config(page_title="HELIOS ONE — V5.2 ELITE+", page_icon=ICON, layout="centered")
st.title("HELIOS ONE — V5.2 ELITE+")
st.caption("Ensemble multi-strats • Sélection & %cap • Multi-TP • BE/Trailing • Kill-switch • Caps global/cluster • Stats par MODE")

# ───────── Dépendances externes
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

# ───────── Univers & mapping YF (fallback)
SYMBOLS_DEFAULT = ['BTC/USDT','ETH/USDT','BNB/USDT','SOL/USDT','XRP/USDT','ADA/USDT','AVAX/USDT','LINK/USDT','TON/USDT','DOGE/USDT','MATIC/USDT']
YF_TICK = {
    'BTC/USDT':'BTC-USD','ETH/USDT':'ETH-USD','BNB/USDT':'BNB-USD','SOL/USDT':'SOL-USD','XRP/USDT':'XRP-USD',
    'ADA/USDT':'ADA-USD','AVAX/USDT':'AVAX-USD','LINK/USDT':'LINK-USD','TON/USDT':'TON-USD','DOGE/USDT':'DOGE-USD','MATIC/USDT':'MATIC-USD'
}

# ───────── Coûts par exchange (influence BE/trailing/time-stop via fee_buffer)
FALLBACK_EX = ['okx','bybit','kraken','coinbase','kucoin','binance']
EX_COST = {
    'okx':     {'fee_bps': 8,  'slip_bps': 3},
    'bybit':   {'fee_bps': 10, 'slip_bps': 4},
    'kraken':  {'fee_bps': 16, 'slip_bps': 5},
    'coinbase':{'fee_bps': 40, 'slip_bps': 6},
    'kucoin':  {'fee_bps': 10, 'slip_bps': 5},
    'binance': {'fee_bps': 8,  'slip_bps': 3},
}

# ───────── DB (SQLite)
DB = os.path.join(os.path.dirname(__file__), 'portfolio.db')
def _init_db():
    conn=sqlite3.connect(DB); c=conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS positions (
      id INTEGER PRIMARY KEY AUTOINCREMENT, open_ts TEXT, close_ts TEXT, symbol TEXT, side TEXT,
      entry REAL, sl REAL, tp REAL, qty REAL, status TEXT, exit_price REAL, pnl REAL, note TEXT)''')
    c.execute('CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)'); conn.commit(); conn.close()
def kv_get(k, default=None):
    _init_db(); conn=sqlite3.connect(DB)
    r=conn.execute('SELECT v FROM kv WHERE k=?',(k,)).fetchone(); conn.close()
    return json.loads(r[0]) if r else default
def kv_set(k,v):
    _init_db(); conn=sqlite3.connect(DB)
    conn.execute('INSERT OR REPLACE INTO kv(k,v) VALUES(?,?)',(k,json.dumps(v))); conn.commit(); conn.close()
def list_positions(status=None,limit=999999):
    _init_db(); conn=sqlite3.connect(DB)
    q='SELECT id,open_ts,close_ts,symbol,side,entry,sl,tp,qty,status,exit_price,pnl,note FROM positions'; pr=()
    if status in("OPEN","CLOSED"): q+=' WHERE status=?'; pr=(status,)
    q+=' ORDER BY id DESC LIMIT ?'; pr=pr+(int(limit),); rows=list(conn.execute(q,pr)); conn.close()
    return pd.DataFrame(rows,columns=['id','open_ts','close_ts','symbol','side','entry','sl','tp','qty','status','exit_price','pnl','note'])

def _meta_from_note(note):
    if isinstance(note,str) and note.startswith("META:"):
        try: return json.loads(note[5:])
        except Exception: return None
    return None
def _meta_to_note(meta): return "META:"+json.dumps(meta, separators=(',',':'))

def open_position(symbol, side, entry, sl, tp, qty, meta=None):
    _init_db()
    if qty is None or float(qty)<=0: return None
    if side.upper()=="LONG" and sl>=entry:  sl=entry-max(1e-9,abs(sl-entry))
    if side.upper()=="SHORT" and sl<=entry: sl=entry+max(1e-9,abs(sl-entry))
    note=_meta_to_note(meta) if isinstance(meta,dict) else ''
    conn=sqlite3.connect(DB); c=conn.cursor()
    c.execute('INSERT INTO positions (open_ts,close_ts,symbol,side,entry,sl,tp,qty,status,exit_price,pnl,note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
              (datetime.datetime.utcnow().isoformat(),None,symbol,side.upper(),float(entry),float(sl),float(tp),float(qty),"OPEN",None,None,note))
    conn.commit(); rid=c.lastrowid; conn.close(); return rid

def update_sl(pid,new_sl):
    _init_db(); conn=sqlite3.connect(DB)
    conn.execute('UPDATE positions SET sl=? WHERE id=? AND status="OPEN"',(float(new_sl),int(pid))); conn.commit(); conn.close()

def close_position(pid, px, note='CLOSE'):
    _init_db(); conn=sqlite3.connect(DB); c=conn.cursor()
    row=c.execute('SELECT open_ts,symbol,side,entry,sl,tp,qty,note FROM positions WHERE id=? AND status="OPEN"',(pid,)).fetchone()
    if not row: conn.close(); return None
    open_ts,symbol,side,entry,sl,tp,qty,note_open=row; meta=_meta_from_note(note_open) or {}
    trade_mode=meta.get('trade_mode','unknown'); top=meta.get('top_strats')
    pnl=(float(px)-float(entry))*float(qty)*(1 if side.upper()=="LONG" else -1)
    c.execute('UPDATE positions SET close_ts=?, status=?, exit_price=?, pnl=?, note=? WHERE id=?',
              (datetime.datetime.utcnow().isoformat(),"CLOSED",float(px),float(pnl),
               "META2:"+json.dumps({"mode":trade_mode,"reason":note,"top":top},separators=(',',':')), pid))
    conn.commit(); conn.close(); return pnl

def partial_close(pid, px, qty_close, reason="TP"):
    _init_db(); conn=sqlite3.connect(DB); c=conn.cursor()
    row=c.execute('SELECT open_ts,symbol,side,entry,sl,tp,qty,note FROM positions WHERE id=? AND status="OPEN"',(pid,)).fetchone()
    if not row: conn.close(); return None
    open_ts,symbol,side,entry,sl,tp,qty,note_open=row; meta=_meta_from_note(note_open) or {}
    trade_mode=meta.get('trade_mode','unknown'); top=meta.get('top_strats')
    qty_close=float(min(max(qty_close,0.0),float(qty))); 
    if qty_close<=0: conn.close(); return None
    sign=1 if side.upper()=="LONG" else -1; pnl=(float(px)-float(entry))*qty_close*sign
    c.execute('INSERT INTO positions (open_ts,close_ts,symbol,side,entry,sl,tp,qty,status,exit_price,pnl,note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
              (open_ts,datetime.datetime.utcnow().isoformat(),symbol,side,float(entry),float(sl),float(tp),float(qty_close),"CLOSED",float(px),float(pnl),
               "META2:"+json.dumps({"mode":trade_mode,"reason":reason,"top":top},separators=(',',':'))))
    remain=float(qty)-qty_close
    if remain>1e-12:
        c.execute('UPDATE positions SET qty=? WHERE id=? AND status="OPEN"',(remain,pid))
    else:
        c.execute('UPDATE positions SET close_ts=?, status=?, exit_price=?, pnl=?, note=? WHERE id=?',
                  (datetime.datetime.utcnow().isoformat(),"CLOSED",float(px),float(pnl),
                   "META2:"+json.dumps({"mode":trade_mode,"reason":reason+"(FULL)","top":top},separators=(',',':')), pid))
    conn.commit(); conn.close(); return pnl

# ───────── CCXT helpers + Fallback YF
def build_exchange(name: str):
    if not HAVE_CCXT: return None
    ex_cls = getattr(ccxt, name.lower()); return ex_cls({'enableRateLimit': True, 'options': {'adjustForTimeDifference': True}})
def _map_symbol(ex: str, symbol: str) -> str:
    if ex=='kraken' and symbol.startswith('BTC/'):  return symbol.replace('BTC/','XBT/')
    if ex=='coinbase' and symbol.endswith('/USDT'): return symbol.replace('/USDT','/USDC')
    return symbol

def fetch_ohlcv_ccxt(exchange_id: str, symbol: str, timeframe='1h', limit=1500) -> pd.DataFrame:
    ex = build_exchange(exchange_id); 
    if ex is None: raise RuntimeError("ccxt indisponible")
    ex.load_markets()
    data = ex.fetch_ohlcv(_map_symbol(exchange_id, symbol), timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True); return df.set_index('ts')

def fetch_ohlcv_yf(symbol: str, timeframe='1h', limit=1500) -> pd.DataFrame:
    if not HAVE_YF: raise RuntimeError("yfinance indisponible")
    t = YF_TICK.get(symbol, None); 
    if t is None: raise RuntimeError(f"Pas de mapping YF pour {symbol}")
    period = "730d" if timeframe in ("15m","1h","4h") else "max"
    interval = dict(**{'15m':'15m','1h':'60m','4h':'240m'}).get(timeframe,'60m')
    df = yf.download(t, period=period, interval=interval, progress=False)
    if df is None or df.empty: raise RuntimeError("YF vide")
    df = df[['Open','High','Low','Close','Volume']].rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
    df.index = pd.to_datetime(df.index, utc=True); return df.tail(limit)

def load_or_fetch(exchange_id: str, symbol: str, tf: str, limit=1500) -> pd.DataFrame:
    last_err = None
    if HAVE_CCXT:
        for ex in [exchange_id] + [e for e in FALLBACK_EX if e != exchange_id]:
            try: return fetch_ohlcv_ccxt(ex, symbol, tf, limit)
            except Exception as e: last_err = e
    # fallback YF
    try: return fetch_ohlcv_yf(symbol, tf, limit)
    except Exception as e: last_err = e
    raise RuntimeError(f"OHLCV échec {symbol} {tf}: {last_err}")

def fetch_last_price(exchange_id: str, symbol: str) -> float:
    if HAVE_CCXT:
        for ex in [exchange_id] + [e for e in FALLBACK_EX if e != exchange_id]:
            try:
                inst = build_exchange(ex); inst.load_markets()
                t = inst.fetch_ticker(_map_symbol(ex, symbol)); px = t.get('last') or t.get('close')
                if px: return float(px)
            except Exception: continue
    # fallback YF
    if HAVE_YF:
        try:
            y = yf.download(YF_TICK.get(symbol,symbol), period="5d", interval="1m", progress=False)
            if y is not None and not y.empty: return float(y['Close'].iloc[-1])
        except Exception: pass
    return np.nan

# ───────── Indicateurs / Strats
def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean(); dn=-d.clip(upper=0).ewm(alpha=1/n,adjust=False).mean()
    rs=up/(dn+1e-9); return 100-100/(1+rs)
def atr_df(df,n=14):
    hl=df['high']-df['low']; hc=(df['high']-df['close'].shift()).abs(); lc=(df['low']-df['close'].shift()).abs()
    tr=pd.concat([hl,hc,lc],axis=1).max(axis=1); return tr.ewm(span=n,adjust=False).mean()
def kama(series, er_len=10, fast=2, slow=30):
    change=series.diff(er_len).abs(); vol=series.diff().abs().rolling(er_len).sum(); er=change/(vol+1e-9)
    sc=(er*(2/(fast+1)-2/(slow+1))+2/(slow+1))**2; out=[series.iloc[0]]
    for i in range(1,len(series)): out.append(out[-1]+sc.iloc[i]*(series.iloc[i]-out[-1]))
    return pd.Series(out,index=series.index)

# … 20+ strats (condensé, même logique que V5.0)
def s_ema_trend(df): f=ema(df['close'],12); s=ema(df['close'],48); return ((f>s).astype(int)-(f<s).astype(int)).rename('signal')
def s_macd(df): f=ema(df['close'],12); s=ema(df['close'],26); m=f-s; sg=ema(m,9); return ((m>sg).astype(int)-(m<sg).astype(int)).rename('signal')
def s_donchian(df,n=55): hh=df['high'].rolling(n).max(); ll=df['low'].rolling(n).min(); return ((df['close']>hh.shift()).astype(int)-(df['close']<ll.shift()).astype(int)).clip(-1,1).rename('signal')
def s_supertrend(df,p=10,m=3.0):
    a=atr_df(df,p); mid=(df['high']+df['low'])/2; bu=mid+m*a; bl=mid-m*a; fu,fl=bu.copy(),bl.copy()
    for i in range(1,len(df)):
        fu.iloc[i]=min(bu.iloc[i],fu.iloc[i-1]) if df['close'].iloc[i-1]>fu.iloc[i-1] else bu.iloc[i]
        fl.iloc[i]=max(bl.iloc[i],fl.iloc[i-1]) if df['close'].iloc[i-1]<fl.iloc[i-1] else bl.iloc[i]
    return ((df['close']>fl).astype(int)-(df['close']<fu).astype(int)).rename('signal')
def s_atr_chan(df,n=14,k=2.0): e=ema(df['close'],n); a=atr_df(df,n); up=e+k*a; lo=e-k*a; c=df['close']; return ((c>up).astype(int)-(c<lo).astype(int)).rename('signal')
def s_boll_mr(df,n=20,k=2.0): ma=df['close'].rolling(n).mean(); sd=df['close'].rolling(n).std(); up=ma+k*sd; lo=ma-k*sd; c=df['close']; return ((c<lo).astype(int)-(c>up).astype(int)).rename('signal')
def s_ichimoku(df):
    h9=df['high'].rolling(9).max(); l9=df['low'].rolling(9).min(); ten=(h9+l9)/2
    h26=df['high'].rolling(26).max(); l26=df['low'].rolling(26).min(); kij=(h26+l26)/2
    spanA=((ten+kij)/2).shift(26); h52=df['high'].rolling(52).max(); l52=df['low'].rolling(52).min(); spanB=((h52+l52)/2).shift(26)
    cross=(ten>kij).astype(int)-(ten<kij).astype(int); up=(df['close']>spanA)&(df['close']>spanB); dn=(df['close']<spanA)&(df['close']<spanB)
    return cross.where(up,0).where(~dn,-1).fillna(0).rename('signal')
def s_kama(df): k=kama(df['close']); return ((df['close']>k).astype(int)-(df['close']<k).astype(int)).rename('signal')
def s_rsi_mr(df): r=rsi(df['close'],14); return ((r<30).astype(int)-(r>70).astype(int)).rename('signal')
def s_ppo(df): ef=ema(df['close'],12); es=ema(df['close'],26); p=(ef-es)/es; ps=ema(p,9); return ((p>ps).astype(int)-(p<ps).astype(int)).rename('signal')
def s_adx(df,n=14,th=20):
    up=df['high'].diff(); dn=-df['low'].diff()
    plusDM=np.where((up>dn)&(up>0),up,0.0); minusDM=np.where((dn>up)&(dn>0),dn,0.0)
    tr=atr_df(df,n)*(n/(n-1))
    plusDI=100*pd.Series(plusDM,index=df.index).ewm(alpha=1/n,adjust=False).mean()/tr
    minusDI=100*pd.Series(minusDM,index=df.index).ewm(alpha=1/n,adjust=False).mean()/tr
    dx=100*((plusDI-minusDI).abs()/(plusDI+minusDI+1e-9)); adx=dx.ewm(alpha=1/n,adjust=False).mean()
    return (((plusDI>minusDI)&(adx>th)).astype(int)-((minusDI>plusDI)&(adx>th)).astype(int)).rename('signal')
def s_stochrsi(df,n=14,k=3,d=3,lo=0.2,hi=0.8):
    r=rsi(df['close'],n); sr=(r-r.rolling(n).min())/(r.rolling(n).max()-r.rolling(n).min()+1e-9)
    kf=sr.rolling(k).mean(); df_=kf.rolling(d).mean()
    return ((kf>df_)&(kf<lo)).astype(int)-((kf<df_)&(kf>hi)).astype(int)
def s_cci(df,n=20):
    tp=(df['high']+df['low']+df['close'])/3; ma=tp.rolling(n).mean(); md=(tp-ma).abs().rolling(n).mean()
    cci=(tp-ma)/(0.015*md+1e-9); return ((cci<-100).astype(int)-(cci>100).astype(int)).rename('signal')
def s_heikin(df):
    ha_c=(df['open']+df['high']+df['low']+df['close'])/4; ha_o=ha_c.copy()
    for i in range(1,len(df)): ha_o.iloc[i]=(ha_o.iloc[i-1]+ha_c.iloc[i-1])/2
    return ((ha_c>ha_o).astype(int)-(ha_c<ha_o).astype(int)).rename('signal')
def s_chandelier_sig(df,n=22,k=3.0):
    a=atr_df(df,n); long_stop=df['high'].rolling(n).max()-k*a; short_stop=df['low'].rolling(n).min()+k*a
    return ((df['close']>long_stop).astype(int)-(df['close']<short_stop).astype(int)).rename('signal')
def s_vwap(df,n=48):
    pv=(df['close']*df['volume']).rolling(n).sum(); vol=df['volume'].rolling(n).sum().replace(0,np.nan); v=pv/vol
    return ((df['close']<v*0.985).astype(int)-(df['close']>v*1.015).astype(int)).rename('signal')
def s_turtlesoup(df,look=20):
    ll=df['low'].rolling(look).min(); hh=df['high'].rolling(look).max()
    lg=((df['low']<ll.shift())&(df['close']>df['open'])).astype(int)
    sh=-((df['high']>hh.shift())&(df['close']<df['open'])).astype(int)
    return (lg+sh).rename('signal')
def s_zscore(df,n=50,k=2.0):
    z=(df['close']-df['close'].rolling(n).mean())/(df['close'].rolling(n).std()+1e-9)
    return ((z<-k).astype(int)-(z>k).astype(int)).rename('signal')
def s_tsi(df,r=25,s=13):
    m=df['close'].diff(); a=ema(ema(m,r),s); b=ema(ema(m.abs(),r),s); tsi=100*a/(b+1e-9); sg=ema(tsi,13)
    return ((tsi>sg).astype(int)-(tsi<sg).astype(int)).rename('signal')
def s_ribbon(df):
    e=[ema(df['close'],n) for n in (8,13,21,34,55)]
    up=sum([e[i]>e[i+1] for i in range(len(e)-1)]); dn=sum([e[i]<e[i+1] for i in range(len(e)-1)])
    return pd.Series(np.where(up>dn,1,np.where(dn>up,-1,0)),index=df.index,name='signal')
def s_keltner(df,n=20,k=2.0): e=ema(df['close'],n); a=atr_df(df,n); up=e+k*a; lo=e-k*a; c=df['close']; return ((c>up).astype(int)-(c<lo).astype(int)).rename('signal')
def s_psar(df, af_step=0.02, af_max=0.2):
    h,l,c=df['high'],df['low'],df['close']; ps=l.copy(); bull=True; af=af_step; ep=float(h.iloc[0]); ps.iloc[0]=float(l.iloc[0])
    for i in range(1,len(df)):
        prev=float(ps.iloc[i-1])
        if bull:
            val=prev+af*(ep-prev); val=min(val,float(l.iloc[i-1])); 
            if i>1: val=min(val,float(l.iloc[i-2]))
            if float(h.iloc[i])>ep: ep=float(h.iloc[i]); af=min(af+af_step,af_max)
            if float(l.iloc[i])<val: bull=False; ps.iloc[i]=ep; ep=float(l.iloc[i]); af=af_step
            else: ps.iloc[i]=val
        else:
            val=prev+af*(ep-prev); val=max(val,float(h.iloc[i-1])); 
            if i>1: val=max(val,float(h.iloc[i-2]))
            if float(l.iloc[i])<ep: ep=float(l.iloc[i]); af=min(af+af_step,af_max)
            if float(h.iloc[i])>val: bull=True; ps.iloc[i]=ep; ep=float(h.iloc[i]); af=af_step
            else: ps.iloc[i]=val
    return pd.Series(np.where(c>ps,1,np.where(c<ps,-1,0)), index=df.index, name='signal')
def s_mfi(df,n=14,lo=20,hi=80):
    tp=(df['high']+df['low']+df['close'])/3; mf=tp*df['volume']
    pos=mf.where(tp>tp.shift(),0.0); neg=mf.where(tp<tp.shift(),0.0).abs()
    mr=100-100/(1+(pos.rolling(n).sum()/(neg.rolling(n).sum()+1e-9)))
    return ((mr<lo).astype(int)-(mr>hi).astype(int)).rename('signal')
def s_obv(df,n=20):
    ch=np.sign(df['close'].diff().fillna(0.0)); obv=(df['volume']*ch).cumsum(); e=ema(obv,n)
    return ((obv>e).astype(int)-(obv<e).astype(int)).rename('signal')

STRATS = {
    'EMA Trend':s_ema_trend,'MACD':s_macd,'Donchian BO':s_donchian,'SuperTrend':s_supertrend,'ATR Channel':s_atr_chan,
    'Bollinger MR':s_boll_mr,'Ichimoku':s_ichimoku,'KAMA Trend':s_kama,'RSI MR':s_rsi_mr,'PPO':s_ppo,'ADX Trend':s_adx,
    'StochRSI':s_stochrsi,'CCI MR':s_cci,'Heikin Trend':s_heikin,'Chandelier':s_chandelier_sig,'VWAP MR':s_vwap,
    'TurtleSoup':s_turtlesoup,'ZScore MR':s_zscore,'TSI':s_tsi,'EMA Ribbon':s_ribbon,'Keltner BO':s_keltner,
    'PSAR Trend':s_psar,'MFI MR':s_mfi,'OBV Trend':s_obv
}

# ───────── Ensemble / score
def compute(df, signal, fee_bps=8.0, slip_bps=3.0):
    ret=df['close'].pct_change().fillna(0.0); pos=signal.shift().fillna(0.0).clip(-1,1)
    cost=(pos.diff().abs().fillna(0.0))*((fee_bps+slip_bps)/10000.0); pnl=pos*ret - cost; eq=(1+pnl).cumprod()
    return ret,pos,pnl,eq
def sharpe(pnl,pp=365*24):
    s=pnl.std(); return 0.0 if s==0 or np.isnan(s) else float(pnl.mean()/s*np.sqrt(pp))
def maxdd(eq): peak=eq.cummax(); dd=eq/peak-1; return float(dd.min())
def _score(p,eq): s=max(0.0,min(3.0,sharpe(p))); dd=abs(maxdd(eq)); return s+(1.0-min(dd,0.4))
def ensemble_weights(df, signals, window=300):
    if not signals: return pd.Series(dtype=float)
    start=max(0,len(df)-int(window)); sc={}
    for n,s in signals.items():
        try: _,_,p,eq=compute(df.iloc[start:], s.iloc[start:])
        except Exception: p=pd.Series([0.0]); eq=pd.Series([1.0])
        sc[n]=_score(p,eq)
    keys=list(sc.keys()); arr=np.array([sc[k] for k in keys]); arr=arr-np.nanmax(arr); w=np.exp(arr)
    w=w/np.nansum(w) if np.nansum(w)!=0 else np.ones_like(w)/len(w)
    return pd.Series(w,index=keys)
def blended_signal(signals, weights):
    if not signals: return pd.Series(dtype=float, name="signal")
    M=pd.concat(signals.values(),axis=1).fillna(0.0); M.columns=list(signals.keys())
    w=weights.reindex(M.columns).fillna(0.0).values.reshape(1,-1)
    v=(M.values*w).sum(axis=1)
    return pd.Series(v, index=M.index, name="signal").clip(-1,1)

# ───────── Gates HTF + Macro
def htf_gate(df_ltf, df_htf): return s_ema_trend(df_htf).reindex(df_ltf.index).ffill().fillna(0.0)
def yf_series(ticker: str, period="5y"):
    """
    Récupère une série 1D (UTC) pour un ticker Yahoo Finance.
    Gère MultiIndex (plusieurs tickers), absence de 'Adj Close', et fallback sur 'Close'.
    Retourne None si indisponible.
    """
    if not HAVE_YF:
        return None
    try:
        y = yf.download(
            tickers=ticker,
            period=period,
            interval="1d",
            auto_adjust=False,     # on veut 'Adj Close' si dispo
            group_by="ticker",     # évite certaines surprises de colonnes
            progress=False,
        )
        if y is None or len(y) == 0:
            return None

        # Normalisation -> série 1D nommée <ticker>
        s = None
        if isinstance(y.columns, pd.MultiIndex):
            # Deux schémas possibles: (champ, ticker) ou (ticker, champ)
            lvl0 = y.columns.get_level_values(0)
            lvl1 = y.columns.get_level_values(1)

            if "Adj Close" in lvl0:
                s = y["Adj Close"]
            elif "Adj Close" in lvl1:
                s = y.xs("Adj Close", axis=1, level=1)
            elif "Close" in lvl0:
                s = y["Close"]
            elif "Close" in lvl1:
                s = y.xs("Close", axis=1, level=1)

            if isinstance(s, pd.DataFrame):
                # Un seul ticker attendu → on prend la 1ère colonne
                s = s.iloc[:, 0]
        else:
            # Colonnes simples
            s = y["Adj Close"] if "Adj Close" in y.columns else y.get("Close")

        if s is None or s.empty:
            return None

        s = s.astype(float)
        # Assure un index datetime UTC
        try:
            s.index = pd.to_datetime(s.index, utc=True)
        except Exception:
            s.index = pd.to_datetime(s.index).tz_localize("UTC")

        s.name = ticker
        return s
    except Exception:
        return None
def macro_gate(enable, vix_caution=20.0, vix_riskoff=28.0, gold_mom_thr=0.10):
    if not enable: return 1.0, "macro OFF"
    vix=yf_series("^VIX"); gold=yf_series("GC=F")
    if vix is None or vix.empty: return 1.0, "no_vix"
    lvl=float(vix.iloc[-1]); mult=1.0; note=[]
    if lvl>vix_riskoff: mult=0.0; note.append("risk-off")
    elif lvl>vix_caution: mult=0.5; note.append("caution")
    else: note.append("benign")
    if gold is not None and not gold.empty:
        mom=float(gold.pct_change(63).iloc[-1])
        if mom>gold_mom_thr: mult*=0.8; note.append("gold↑")
    return mult," | ".join(note)

# ───────── Risk & sizing
def rr(entry, sl, tp): R=abs(entry-sl); return float(abs(tp-entry)/(R if R>0 else 1e-9))
def atr_levels(df, d, sl_mult=2.5, tp_mult=4.0):
    if d==0 or len(df)<2: return None
    a=float(atr_df(df,14).iloc[-1]); price=float(df['close'].iloc[-1])
    sl = price - sl_mult*a if d>0 else price + sl_mult*a
    tp = price + tp_mult*a if d>0 else price - tp_mult*a
    return {'entry':price,'sl':sl,'tp':tp,'atr':a}
def size_fixed_pct(equity, entry, stop, risk_pct):
    per=abs(entry-stop); risk=equity*(risk_pct/100.0); return 0.0 if per<=0 else risk/per

# ───────── Clusters
def base(sym): return sym.split('/')[0].upper()
CLUSTER = {'BTC':'Majors','ETH':'Majors','BNB':'Exchange','SOL':'L1','ADA':'L1','AVAX':'L1','TON':'L1','XRP':'Payments','LINK':'Infra','DOGE':'Meme','MATIC':'Infra'}
def symbol_cluster(sym): return CLUSTER.get(base(sym),'Other')

# ───────── Modes (presets)
modes={
 'Conservateur': dict(risk_pct=1.0, max_expo=60.0,  per_trade_cap=25.0, min_rr=1.8, max_positions=2, splits=(0.5,0.35,0.15), tpR=(0.9,1.7,2.6), gate_thr=0.35, leverage=1.0),
 'Normal':       dict(risk_pct=2.0, max_expo=100.0, per_trade_cap=35.0, min_rr=1.8, max_positions=3, splits=(0.4,0.4,0.2), tpR=(1.0,2.0,3.5), gate_thr=0.30, leverage=1.0),
 'Agressif':     dict(risk_pct=5.0, max_expo=150.0, per_trade_cap=40.0, min_rr=1.6, max_positions=5, splits=(0.30,0.40,0.30), tpR=(1.0,2.5,5.0), gate_thr=0.22, leverage=1.5),
 'Super agressif (x5)': dict(risk_pct=2.5, max_expo=120.0, per_trade_cap=20.0, min_rr=2.2, max_positions=3, splits=(0.34,0.33,0.33), tpR=(1.2,3.0,6.0), gate_thr=0.60, leverage=5.0)
}

# ───────── Sidebar (simple)
st.markdown("### ⚙️ Mode & Réglages")
mode = st.selectbox("Mode (je touche seulement à ça)", list(modes.keys()), index=1)
m = modes[mode]

with st.expander("Réglages avancés (optionnel)"):
    exchange = st.selectbox('Exchange (si ccxt dispo)', FALLBACK_EX, index=0)
    tf = st.selectbox('Timeframe', ['15m','1h','4h'], index=1)
    htf = st.selectbox('HTF confirm', ['1h','4h','1d'], index=2 if tf!='4h' else 1)
    symbols = st.multiselect('Paires', SYMBOLS_DEFAULT, default=SYMBOLS_DEFAULT[:8])
    sl_mult = st.slider("SL (×ATR)", 1.0, 4.0, 2.5, 0.1)
    tp_mult = st.slider("TP (×ATR suggéré)", 1.0, 6.0, 4.0, 0.1)
    macro_enabled = st.toggle("Macro gate (VIX/Gold via yfinance)", value=True)
    capital_edit = st.number_input("Capital de base (équity)", min_value=0.0, value=float(kv_get('base_capital',1000.0)), step=100.0)
    if st.button("💾 Enregistrer capital"): kv_set('base_capital', float(capital_edit)); st.success("Capital mis à jour.")
with st.expander("Risque avancé"):
    daily_loss_limit = st.number_input("Limite perte journalière (USD)", min_value=0.0, value=float(kv_get('daily_loss_limit', 150.0)), step=50.0)
    cooldown_minutes = st.number_input("Cooldown après dépassement (min)", min_value=0, value=int(kv_get('cooldown_minutes', 120)), step=15)
    cluster_cap_pct = st.slider("Cap d'expo par cluster (%) du cap global", 10, 100, int(kv_get('cluster_cap_pct', 60)), 5)
    time_stop_bars = st.number_input("Time-stop (barres sans progrès)", min_value=0, value=int(kv_get('time_stop_bars', 0)), step=1)
    if st.button("💾 Enregistrer (risque avancé)"):
        kv_set('daily_loss_limit', float(daily_loss_limit)); kv_set('cooldown_minutes', int(cooldown_minutes))
        kv_set('cluster_cap_pct', int(cluster_cap_pct));     kv_set('time_stop_bars', int(time_stop_bars))
        st.success("Paramètres risque avancé enregistrés.")

# défauts si expander fermé
exchange  = locals().get('exchange','okx')
tf        = locals().get('tf','1h')
htf       = locals().get('htf','4h')
symbols   = locals().get('symbols', SYMBOLS_DEFAULT[:8])
sl_mult   = locals().get('sl_mult',2.5)
tp_mult   = locals().get('tp_mult',4.0)
macro_enabled = locals().get('macro_enabled', True)
daily_loss_limit = float(locals().get('daily_loss_limit', kv_get('daily_loss_limit',150.0)))
cooldown_minutes = int(locals().get('cooldown_minutes', kv_get('cooldown_minutes',120)))
cluster_cap_pct  = int(locals().get('cluster_cap_pct', kv_get('cluster_cap_pct',60)))
time_stop_bars   = int(locals().get('time_stop_bars', kv_get('time_stop_bars',0)))
fee_bps = EX_COST.get(exchange, EX_COST['okx'])['fee_bps']; slip_bps = EX_COST.get(exchange, EX_COST['okx'])['slip_bps']

# ───────── Suggestion de mode
def suggest_mode():
    mm, note = macro_gate(macro_enabled)
    try:
        df_btc = load_or_fetch(exchange, 'BTC/USDT', '4h', 600); trend = s_adx(df_btc).iloc[-1]
    except Exception: trend = 0
    if mm==0.0: return "Conservateur","Risk-off (macro)", note
    if trend>0 and mm>=1.0: return "Agressif","Tendance haussière forte", note
    if trend<0 and mm>=1.0: return "Conservateur","Tendance baissière", note
    if mm<1.0: return "Conservateur","Caution macro", note
    return "Normal","Rien de spécial", note
s_mode, s_reason, s_macro = suggest_mode()
st.info(f"💡 Suggestion de mode: **{s_mode}** — {s_reason} · Macro: {s_macro}")

# ───────── Equity / kill-switch
def portfolio_equity(base_capital, price_map=None):
    open_df=list_positions(status='OPEN'); closed_df=list_positions(status='CLOSED')
    realized=0.0 if closed_df.empty else float(closed_df['pnl'].sum()); latent=0.0
    if not open_df.empty:
        if price_map is None: price_map={s:fetch_last_price(exchange,s) for s in open_df['symbol'].unique()}
        for _,r in open_df.iterrows():
            px=float(price_map.get(r['symbol'],r['entry'])); sign=1 if r['side']=='LONG' else -1
            latent+=(px-float(r['entry']))*float(r['qty'])*sign
    return base_capital+realized+latent

capital = float(kv_get('base_capital',1000.0))
eq_now = portfolio_equity(capital)
st.metric("💼 Portefeuille (équity dynamique)", f"{eq_now:.2f} USD")

def realized_today():
    hist=list_positions(status='CLOSED')
    if hist.empty: return 0.0
    today=datetime.datetime.utcnow().date()
    vals=hist.copy(); vals['date']=pd.to_datetime(vals['close_ts']).dt.date
    return float(vals.loc[vals['date']==today,'pnl'].sum())

cooldown_until = kv_get('cooldown_until', None); now_utc = datetime.datetime.utcnow().timestamp()
if cooldown_until and now_utc < float(cooldown_until):
    eta = datetime.datetime.utcfromtimestamp(float(cooldown_until)).strftime("%Y-%m-%d %H:%M:%S UTC")
    st.warning(f"⏳ Kill-switch actif jusqu’à {eta}."); kill_active = True
else:
    kill_active = False
    pnl_today = realized_today()
    if daily_loss_limit>0 and pnl_today <= -daily_loss_limit:
        until = datetime.datetime.utcnow() + datetime.timedelta(minutes=cooldown_minutes)
        kv_set('cooldown_until', until.timestamp())
        st.warning(f"❌ Limite journalière atteinte ({pnl_today:.2f} USD). Cooldown {cooldown_minutes} min activé.")
        kill_active = True

# ───────── Utilitaires TP/SL
def r_targets(entry, sl, side, tpR=(1.0,2.0,3.5)):
    side=side.upper(); sign=1 if side=="LONG" else -1; R=(entry-sl) if side=="LONG" else (sl-entry)
    if R<=0: return None
    return [entry+sign*r*R for r in tpR]
def build_meta_r(entry, sl, side, qty, splits=(0.4,0.4,0.2), tpR=(1.0,2.0,3.5),
                 be_after_tp1=True, trade_mode="Normal", top_strats=None, confidence=None):
    tps=r_targets(entry,sl,side,tpR); q1,q2,q3=[float(qty*max(0.0,s)) for s in splits]; diff=float(qty)-(q1+q2+q3); q3=max(0.0,q3+diff)
    meta={'multi_tp':True,'mode':'R','trade_mode':str(trade_mode),'tpR':list(tpR),'splits':list(splits),
          'targets':[{'name':'TP1','px':tps[0],'qty':q1,'filled':False},
                     {'name':'TP2','px':tps[1],'qty':q2,'filled':False},
                     {'name':'TP3','px':tps[2],'qty':q3,'filled':False}],
          'be_after_tp1':bool(be_after_tp1),'trail_after_tp2':True}
    if isinstance(top_strats, list): meta['top_strats']=top_strats[:5]
    if confidence is not None: meta['confidence']=float(confidence)
    return meta

def _chandelier_stop(df,n=22,k=3.0,side="LONG"):
    a=atr_df(df,n).values
    if side.upper()=="LONG": return (df['high'].rolling(n).max().values - k*a)
    return (df['low'].rolling(n).min().values + k*a)

def sanitize_all_positions():
    df=list_positions(status='OPEN'); changed=False
    for _,r in df.iterrows():
        side=str(r['side']).upper(); entry=float(r['entry']); sl=float(r['sl']); new=sl; fix=False
        if side=="LONG" and sl>=entry:  new=entry-max(1e-9,abs(sl-entry)); fix=True
        if side=="SHORT" and sl<=entry: new=entry+max(1e-9,abs(sl-entry)); fix=True
        if fix: 
            conn=sqlite3.connect(DB); conn.execute('UPDATE positions SET sl=? WHERE id=?',(float(new),int(r['id']))); conn.commit(); conn.close(); changed=True
    return changed

def auto_manage_positions(price_map, ohlc_map=None, mode="Normal", be_after_tp1=True, trail_after_tp2=True, fee_buffer_bps=5, time_stop_bars=0, tf_minutes=60):
    sanitize_all_positions()
    df=list_positions(status='OPEN'); 
    if df.empty: return []
    # presets par mode (splits/tpR)
    if str(mode).startswith("Conserv"): parts=(0.50,0.35,0.15); tpsR=(0.9,1.7,2.6)
    elif str(mode).startswith("Agressif") and "Super" not in str(mode): parts=(0.30,0.40,0.30); tpsR=(1.0,2.5,5.0)
    elif "Super" in str(mode): parts=(0.34,0.33,0.33); tpsR=(1.2,3.0,6.0)
    else: parts=(0.40,0.40,0.20); tpsR=(1.0,2.0,3.5)
    evts=[]; now=datetime.datetime.utcnow()
    for _,r in df.iterrows():
        sym=r['symbol']; side=r['side'].upper()
        if sym not in price_map: continue
        px=float(price_map[sym]); entry=float(r['entry']); sl=float(r['sl']); qty=float(r['qty'])
        if qty<=1e-12: continue
        R=(entry-sl) if side=="LONG" else (sl-entry)
        if R<=0: continue
        # Targets R
        tps=r_targets(entry,sl,side,tpsR); tp1,tp2,tp3=tps
        # Hits
        def hit_tp(p,t): return p>=t if side=="LONG" else p<=t
        def hit_sl(p,s): return p<=s if side=="LONG" else p>=s
        # TP1
        if hit_tp(px,tp1):
            q=qty*parts[0]; partial_close(int(r['id']),px,q,"TP1"); evts.append((sym,"TP1",px,q))
            if be_after_tp1:
                be_px=entry*(1+(fee_buffer_bps/10000.0)*(1 if side=="LONG" else -1)); update_sl(int(r['id']),be_px)
            # refresh qty
            cur=list_positions(status='OPEN'); cur=cur[cur['id']==r['id']]
            if cur.empty: continue
            qty=float(cur.iloc[0]['qty'])
        # TP2
        if qty>0 and hit_tp(px,tp2):
            q=qty*parts[1]; partial_close(int(r['id']),px,q,"TP2"); evts.append((sym,"TP2",px,q))
            if trail_after_tp2 and ohlc_map and sym in ohlc_map:
                trail=float(_chandelier_stop(ohlc_map[sym],22,3.0,side)[-1])
                cur=list_positions(status='OPEN'); cur=cur[cur['id']==r['id']]
                if not cur.empty:
                    cur_sl=float(cur.iloc[0]['sl'])
                    if (side=="LONG" and trail>cur_sl) or (side=="SHORT" and trail<cur_sl): update_sl(int(r['id']),trail)
            cur=list_positions(status='OPEN'); cur=cur[cur['id']==r['id']]
            qty=0.0 if cur.empty else float(cur.iloc[0]['qty'])
        # TP3
        if qty>0 and hit_tp(px,tp3):
            partial_close(int(r['id']),px,qty,"TP3"); evts.append((sym,"TP3",px,qty)); qty=0.0
        # SL
        cur=list_positions(status='OPEN'); cur=cur[cur['id']==r['id']]
        if not cur.empty:
            stop=float(cur.iloc[0]['sl']); qty=float(cur.iloc[0]['qty'])
            if qty>0 and hit_sl(px,stop): partial_close(int(r['id']),stop,qty,"SL"); evts.append((sym,"SL",stop,qty)); qty=0.0
        # Time-stop (optionnel)
        if time_stop_bars and qty>0:
            try: opened=datetime.datetime.fromisoformat(r['open_ts'])
            except Exception: opened=now
            age_min=(now-opened).total_seconds()/60.0
            if age_min>=time_stop_bars*tf_minutes:
                sign=1 if side=="LONG" else -1
                pnl_now=(px-entry)*sign
                if pnl_now<=0: partial_close(int(r['id']),px,qty,"TIME_STOP"); evts.append((sym,"TIME_STOP",px,qty))
    return evts

# ───────── Onglets
tabs = st.tabs(['🏠 Décision','📈 Portefeuille','🧾 Historique','📊 Analyse','🔬 Lab'])

# ───────── 1) Décision
# ───────────────────── 1) Décision (persistant & actionnable) ─────────────────────
with tabs[0]:
    st.subheader("Top Picks (1 clic)")
    mm, macro_note = macro_gate(macro_enabled)
    st.caption(f"Macro: {macro_note} (×{mm}) • Coûts: fee {fee_bps} bps, slip {slip_bps} bps")

    # État persistant pour garder le dernier scan
    if 'scan_df' not in st.session_state: st.session_state.scan_df = pd.DataFrame()
    if 'scan_top' not in st.session_state: st.session_state.scan_top = {}   # idx -> top_strats
    if 'scan_conf' not in st.session_state: st.session_state.scan_conf = {} # idx -> confidence

    # Refresh des caps/expo actuelles
    def _caps_snapshot():
        open_now = list_positions(status='OPEN')
        open_notional = 0.0 if open_now.empty else float((open_now['entry']*open_now['qty']).sum())
        eq = portfolio_equity(float(kv_get('base_capital',1000.0)))
        cap_gross = eq*(m['max_expo']/100.0)*m['leverage']
        room = max(0.0, cap_gross - open_notional)
        per_trade_cap = eq*(m['per_trade_cap']/100.0)
        cluster_now={}
        if not open_now.empty:
            for _,r__ in open_now.iterrows():
                cl = symbol_cluster(r__['symbol'])
                cluster_now[cl] = cluster_now.get(cl,0.0) + float(r__['entry']*r__['qty'])
        return eq, cap_gross, open_notional, room, per_trade_cap, cluster_now
    eq, cap_gross, open_notional, room, per_trade_cap, cluster_now = _caps_snapshot()
    cap_cluster_abs = cap_gross * (cluster_cap_pct/100.0)

    # ---------- SCAN ----------
    if st.button("🚀 Scanner maintenant", use_container_width=True, disabled=kill_active):
        if kill_active: st.stop()
        rows=[]; tops={}; confs={}
        for sym in symbols:
            try:
                df  = load_or_fetch(exchange, sym, tf, 1200)
                dfH = load_or_fetch(exchange, sym, htf, 600)
            except Exception as e:
                st.warning(f"Skip {sym}: {e}"); continue

            signals={nm: fn(df) for nm,fn in STRATS.items()}
            w = ensemble_weights(df, signals, window=300)
            sig = blended_signal(signals, w)
            gate = htf_gate(df, dfH)
            blended = (sig*gate).clip(-1,1)*mm
            if abs(float(blended.iloc[-1])) < m['gate_thr']: continue

            d = int(np.sign(blended.iloc[-1])); 
            if d==0: continue
            lvl = atr_levels(df, d, sl_mult, tp_mult)
            if not lvl: continue
            r_r = rr(lvl['entry'], lvl['sl'], lvl['tp'])
            if r_r < m['min_rr']: continue

            qty0 = size_fixed_pct(eq, lvl['entry'], lvl['sl'], m['risk_pct'])
            if qty0 <= 0: continue
            tps = r_targets(lvl['entry'], lvl['sl'], 'LONG' if d>0 else 'SHORT', m['tpR'])

            rows.append({
                'symbol':sym,'dir':'LONG' if d>0 else 'SHORT',
                'entry':lvl['entry'],'sl':lvl['sl'],'tp':lvl['tp'],
                'tp1':tps[0],'tp2':tps[1],'tp3':tps[2],
                'rr':r_r,'qty':qty0,'pct_cap':0.0,
                'confidence': float(abs(sig.iloc[-1]) * w.sort_values(ascending=False).head(3).sum()),
            })
            tops[len(rows)-1]  = [(k,float(v)) for k,v in w.sort_values(ascending=False).head(5).items()]
            confs[len(rows)-1] = rows[-1]['confidence']

        st.session_state.scan_df   = pd.DataFrame(rows).sort_values(['confidence','rr'],ascending=False).head(int(m['max_positions'])).reset_index(drop=True)
        st.session_state.scan_top  = {i: tops.get(i, []) for i in range(len(st.session_state.scan_df))}
        st.session_state.scan_conf = {i: confs.get(i, 0.0) for i in range(len(st.session_state.scan_df))}
        st.rerun()

    # ---------- AFFICHAGE / ACTION ----------
    picks = st.session_state.scan_df
    if picks.empty:
        st.info("Clique d’abord sur **Scanner maintenant**.")
    else:
        st.caption(f"Expo: {open_notional:.2f} / Cap: {cap_gross:.2f} (lev {m['leverage']}x) → Reste: {room:.2f} · Cap/trade: {per_trade_cap:.2f}")
        st.caption("Expo cluster: " + (", ".join([f"{k}:{v:.0f}" for k,v in cluster_now.items()]) if cluster_now else "—"))

        table = picks[['symbol','dir','entry','sl','tp','tp1','tp2','tp3','rr','qty','pct_cap','confidence']].copy()
        table.insert(0,'take',True)
        edit = st.data_editor(
            table, hide_index=True, num_rows="fixed", use_container_width=True,
            column_config={
                "take": st.column_config.CheckboxColumn("Prendre"),
                "qty": st.column_config.NumberColumn("qty (si %cap=0)", step=0.0001, format="%.6f"),
                "pct_cap": st.column_config.NumberColumn("%cap (override)", min_value=0.0, max_value=100.0, step=0.5, format="%.1f"),
                "rr": st.column_config.NumberColumn("R/R", format="%.2f", disabled=True),
                "confidence": st.column_config.NumberColumn("Confiance", format="%.3f", disabled=True),
                "tp1": st.column_config.NumberColumn("TP1", format="%.6f", disabled=True),
                "tp2": st.column_config.NumberColumn("TP2", format="%.6f", disabled=True),
                "tp3": st.column_config.NumberColumn("TP3", format="%.6f", disabled=True),
            }
        )
        price_mode = st.selectbox("Prix d'entrée", ["Suggéré (entry)", "Prix du marché"], index=0)

    
        # Boutons par ligne
        st.markdown("##### Actions par trade")
        for i, r in edit.iterrows():
            c1, c2, c3 = st.columns([3,1,1])
            c1.write(f"**{r['symbol']}** · {r['dir']} · entry `{float(r['entry']):.6f}` · SL `{float(r['sl']):.6f}` · R/R `{float(r['rr']):.2f}` · conf `{float(r['confidence']):.3f}` · %cap `{float(r['pct_cap']):.1f}`")
            if c2.button("📌 Prendre ce trade", key=f"take_row_{i}"):
                _take_one(i, r); st.success("Ouvert ✅"); st.rerun()
            if c3.button("🔬 Lab", key=f"lab_row_{i}"):
                try:
                    df=load_or_fetch(exchange, r['symbol'], tf, 2000)
                    bucket=['EMA Trend','MACD','SuperTrend','Bollinger MR','Ichimoku','ADX Trend','OBV Trend']
                    res=[]
                    for nm in bucket:
                        sig=STRATS[nm](df); _,_,p,eq_=compute(df,sig, fee_bps=fee_bps, slip_bps=slip_bps)
                        res.append(dict(name=nm, sharpe=sharpe(p), mdd=maxdd(eq_), cagr=(eq_.iloc[-1]**(365*24/len(eq_))-1)))
                    st.dataframe(pd.DataFrame(res).sort_values("sharpe",ascending=False).round(4), use_container_width=True)
                except Exception as e:
                    st.warning(str(e))

        # Actions groupées
        sel = edit[edit['take']].copy()
        cA, cB = st.columns(2)
        if cA.button("📌 Prendre la sélection"):
            if sel.empty: st.warning("Rien de sélectionné.")
            else:
                tmp = sel.copy()
                for idx in tmp.index:
                    if float(tmp.loc[idx,'pct_cap'])>0:
                        tmp.loc[idx,'qty'] = (eq * (float(tmp.loc[idx,'pct_cap'])/100.0)) / max(float(tmp.loc[idx,'entry']),1e-9)
                total_alloc=float((tmp['qty']*tmp['entry']).sum()); scale_g=1.0
                if total_alloc>room: scale_g = room/max(total_alloc,1e-9)
                for i, r in tmp.iterrows():
                    r = r.copy(); r['qty'] = float(r['qty'])*scale_g
                    _take_one(i, r)
                st.success("Trades ouverts ✅"); st.rerun()

        if cB.button("🧮 Simuler l’allocation"):
            tmp = sel.copy()
            for idx in tmp.index:
                if float(tmp.loc[idx,'pct_cap'])>0:
                    tmp.loc[idx,'qty'] = (eq * (float(tmp.loc[idx,'pct_cap'])/100.0)) / max(float(tmp.loc[idx,'entry']),1e-9)
            alloc=float((tmp['qty']*tmp['entry']).sum()) if not tmp.empty else 0.0
            st.info(f"{len(tmp)} trades · Allocation brute {alloc:.2f} USD  \nCap/trade {per_trade_cap:.2f} · Espace cap {room:.2f} · Cap/cluster {cap_cluster_abs:.2f}")

# ───────── 2) Portefeuille
with tabs[1]:
    st.subheader("Positions ouvertes")
    open_df=list_positions(status='OPEN')
    if open_df.empty or (open_df['qty']<=1e-12).all():
        st.info("Aucune position.")
    else:
        open_df=open_df[open_df['qty']>1e-12]
        last={s:fetch_last_price(exchange,s) for s in open_df['symbol'].unique()}
        open_df['last']=open_df['symbol'].map(last)
        open_df['ret_%']=((open_df['last']-open_df['entry']).where(open_df['side']=='LONG', open_df['entry']-open_df['last'])/open_df['entry']*100).round(3)
        open_df['PnL_latent']=((open_df['last']-open_df['entry']).where(open_df['side']=='LONG', open_df['entry']-open_df['last'])*open_df['qty']).round(6)
        st.dataframe(open_df[['id','symbol','side','entry','sl','tp','qty','last','ret_%','PnL_latent','note']], use_container_width=True)
        st.metric("Équity dynamique", f"{portfolio_equity(capital,last):.2f} USD")

        if st.button("🔄 Mettre à jour (TP/SL + BE/Trailing + Time-stop)"):
            ohlc_map={s:load_or_fetch(exchange,s,tf,300) for s in open_df['symbol'].unique()}
            events=auto_manage_positions(last, ohlc_map, mode=mode, be_after_tp1=True, trail_after_tp2=True,
                                        fee_buffer_bps=fee_bps+slip_bps, time_stop_bars=time_stop_bars, tf_minutes={'15m':15,'1h':60,'4h':240}.get(tf,60))
            for sym,why,px,q in events: st.success(f"{sym}: {why} @ {px:.6f} (qty {q:.4f})")
            st.rerun()

        st.markdown("### Actions rapides")
        for _,r in open_df.iterrows():
            cols=st.columns([3,1.1,1.1,1.1,1.3])
            cols[0].markdown(f"**{r['symbol']}** · {r['side']} · qty `{r['qty']:.4f}` · SL `{r['sl']:.6f}`")
            if cols[1].button("SL→BE", key=f"be_{r['id']}"): update_sl(int(r["id"]),float(r["entry"])); st.rerun()
            if cols[2].button("−25%", key=f"m25_{r['id']}"):
                px=last.get(r['symbol'],r['entry']); partial_close(int(r['id']),float(px),float(r['qty'])*0.25,"MANUAL_25"); st.rerun()
            if cols[3].button("−50%", key=f"m50_{r['id']}"):
                px=last.get(r['symbol'],r['entry']); partial_close(int(r['id']),float(px),float(r['qty'])*0.50,"MANUAL_50"); st.rerun()
            if cols[4].button("Fermer 100%", key=f"m100_{r['id']}"):
                px=last.get(r['symbol'],r['entry']); close_position(int(r['id']),float(px),"MANUAL_CLOSE"); st.rerun()

# ───────── 3) Historique
with tabs[2]:
    st.subheader("Historique (clôturées)")
    hist=list_positions(status='CLOSED')
    if hist.empty:
        st.info("Pas encore d’historique.")
    else:
        def _mode_from_note(n):
            if isinstance(n,str) and n.startswith("META2:"):
                try: return json.loads(n[6:]).get("mode","unknown")
                except Exception: return "unknown"
            if isinstance(n,str) and n.startswith("META:"):
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
        c1.metric("P&L réalisé", f"{pnl:.2f}"); c2.metric("Win rate", f"{winrate:.1f}%")
        c3.metric("Profit factor", f"{pf:.2f}" if not np.isnan(pf) else "—"); c4.metric("Avg win/loss", f"{avgwin:.2f} / {avgloss:.2f}")

        st.markdown("#### Perf par mode")
        def agg(dfm):
            wins=(dfm["pnl"]>0).sum(); total=len(dfm)
            avgw=dfm.loc[dfm["pnl"]>0,"pnl"].mean() if wins>0 else 0.0
            avgl=dfm.loc[dfm["pnl"]<=0,"pnl"].mean() if (total-wins)>0 else 0.0
            pf=(avgw/abs(avgl)) if avgl<0 else np.nan
            return pd.Series({"trades":total,"wins":wins,"winrate%":100*wins/total if total>0 else 0.0,"pnl_sum":dfm["pnl"].sum(),"profit_factor":pf})
        bymode=hist.groupby("mode",dropna=False).apply(agg).reset_index()
        st.dataframe(bymode.sort_values("pnl_sum",ascending=False).round(3), use_container_width=True)

# ───────── 4) Analyse
with tabs[3]:
    st.subheader("📊 Analyse")
    hist=list_positions(status='CLOSED')
    if hist.empty:
        st.info("Pas de données.")
    else:
        # Equity per mode (réalisé)
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
        st.line_chart(pd.DataFrame({mo:pd.Series(vals) for mo,vals in curves.items()}))

# ───────── 5) Lab
with tabs[4]:
    st.subheader("Lab — Backtest rapide")
    sym_b=st.selectbox("Symbole", SYMBOLS_DEFAULT, index=0, key="lab_sym")
    tf_b=st.selectbox("TF", ['15m','1h','4h'], index=1, key="lab_tf")
    names=st.multiselect("Stratégies à tester", list(STRATS.keys()),
                         default=['EMA Trend','MACD','SuperTrend','Bollinger MR','Ichimoku'])
    if st.button("▶︎ Lancer le backtest"):
        try:
            df=load_or_fetch(exchange, sym_b, tf_b, 2000); res=[]
            for nm in names:
                sig=STRATS[nm](df); _,_,p,eq=compute(df,sig, fee_bps=fee_bps, slip_bps=slip_bps)
                res.append(dict(name=nm, sharpe=sharpe(p), mdd=maxdd(eq), cagr=(eq.iloc[-1]**(365*24/len(eq))-1)))
            st.dataframe(pd.DataFrame(res).sort_values("sharpe",ascending=False).round(4), use_container_width=True)
        except Exception as e:
            st.error(str(e))

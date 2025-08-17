# HELIOS ONE — V3.3 PRO (single-file)
# One-tap crypto signals · 24+ strats · Ensemble pondéré · HTF + Macro gating
# Manual only · Portfolio journal (SQLite) · Sélection de trades · TP/SL auto
# Améliorations : trailing stop ATR, break-even auto, anti-duplicats, stats avancées

import os, sqlite3, datetime, requests
import streamlit as st
import pandas as pd
import numpy as np

# ────────────────────────────────────────────────────────────────────────────────
# App config
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="HELIOS ONE — V3.3 PRO", page_icon="☀️", layout="centered")
st.title("HELIOS ONE — V3.3 PRO")
st.caption("One-tap crypto signals · 24+ strats · Ensemble pondéré · HTF + Macro gating · Manual only")

# ────────────────────────────────────────────────────────────────────────────────
# External deps
# ────────────────────────────────────────────────────────────────────────────────
try:
    import ccxt
except Exception:
    st.error("❗ ccxt manquant. Ajoute-le dans requirements.txt"); st.stop()

try:
    import yfinance as yf
    HAVE_YF = True
except Exception:
    HAVE_YF = False

# ────────────────────────────────────────────────────────────────────────────────
# Exchange utils
# ────────────────────────────────────────────────────────────────────────────────
def build_exchange(name: str):
    ex_class = getattr(ccxt, name.lower())
    params = {'enableRateLimit': True, 'options': {'adjustForTimeDifference': True}}
    api_key=os.getenv('API_KEY',''); api_secret=os.getenv('API_SECRET',''); password=os.getenv('PASSWORD','')
    if api_key and api_secret: params.update({'apiKey': api_key, 'secret': api_secret})
    if password: params['password']=password
    ex = ex_class(params)
    try: ex.load_markets()
    except Exception: pass
    return ex

FALLBACK = ['okx','bybit','kraken','coinbase','kucoin','binance']

def _map_symbol(exchange_id: str, symbol: str) -> str:
    if exchange_id=='kraken' and symbol.startswith('BTC/'):
        return symbol.replace('BTC/','XBT/')
    if exchange_id=='coinbase' and symbol.endswith('/USDT'):
        return symbol.replace('/USDT','/USDC')
    return symbol

def fetch_ohlcv(exchange: str, symbol: str, timeframe='1h', limit=2000):
    ex = build_exchange(exchange); sym=_map_symbol(exchange, symbol)
    data = ex.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df.set_index('ts', inplace=True)
    return df

def load_or_fetch(exchange: str, symbol: str, timeframe: str, limit=2000):
    last_err=None
    for ex in [exchange]+[e for e in FALLBACK if e!=exchange]:
        try: return fetch_ohlcv(ex, symbol, timeframe, limit)
        except Exception as e: last_err=e; continue
    raise RuntimeError(f"Echec fetch {symbol} {timeframe}. Dernière erreur: {last_err}")

def fetch_last_price(exchange: str, symbol: str) -> float:
    ex = build_exchange(exchange); sym=_map_symbol(exchange, symbol)
    t = ex.fetch_ticker(sym); return float(t.get('last') or t.get('close') or 0.0)

def yf_series(ticker: str, period="5y"):
    if not HAVE_YF: return None
    try:
        y = yf.download(ticker, period=period, interval="1d", progress=False)
        if y is None or y.empty: return None
        s = y['Adj Close'].rename(ticker).tz_localize("UTC")
        return s
    except Exception:
        return None

# ────────────────────────────────────────────────────────────────────────────────
# Indicators & helpers
# ────────────────────────────────────────────────────────────────────────────────
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(s, n=14):
    d=s.diff(); up=d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn=-d.clip(upper=0).ewm(alpha=1/n, adjust=False).mean()
    rs=up/(dn+1e-9); return 100-100/(1+rs)
def kama(series, er_len=10, fast=2, slow=30):
    change = series.diff(er_len).abs()
    vol = series.diff().abs().rolling(er_len).sum()
    er = change / (vol + 1e-9)
    sc = (er*(2/(fast+1) - 2/(slow+1)) + 2/(slow+1))**2
    out=[series.iloc[0]]
    for i in range(1,len(series)):
        out.append(out[-1] + sc.iloc[i]*(series.iloc[i]-out[-1]))
    return pd.Series(out, index=series.index)
def atr_df(df, n=14):
    hl=df['high']-df['low']; hc=(df['high']-df['close'].shift()).abs(); lc=(df['low']-df['close'].shift()).abs()
    tr=pd.concat([hl,hc,lc],axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()
def vwap_roll(df, n=48):
    pv=(df['close']*df['volume']).rolling(n).sum()
    vol=df['volume'].rolling(n).sum().replace(0,np.nan)
    return pv/vol

# ────────────────────────────────────────────────────────────────────────────────
# Strategies (24)
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
    md=(tp-tp.rolling(n).mean()).abs().rolling(n).mean()
    cci=(tp-ma)/(0.015*md+1e-9)
    return ((cci<-100).astype(int)-(cci>100).astype(int)).rename('signal')
def sig_heikin_trend(df):
    ha_close=(df['open']+df['high']+df['low']+df['close'])/4
    ha_open=ha_close.copy()
    for i in range(1,len(df)):
        ha_open.iloc[i]=(ha_open.iloc[i-1]+ha_close.iloc[i-1])/2
    trend=((ha_close>ha_open).astype(int)-(ha_close<ha_open).astype(int)).rename('signal'); return trend
def sig_chandelier(df, n=22, mult=3.0):
    a=atr_df(df,n); long_stop=df['high'].rolling(n).max()-mult*a; short_stop=df['low'].rolling(n).min()+mult*a
    long=(df['close']>long_stop).astype(int); short=-(df['close']<short_stop).astype(int)
    return (long+short).clip(-1,1).rename('signal')
def sig_vwap_mr(df, n=48):
    v=vwap_roll(df,n); return ((df['close']<v*0.985).astype(int) - (df['close']>v*1.015).astype(int)).rename('signal')
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
    e=ema(df['close'],n); a=atr_df(df,n); up=e+mult*a; lo=e-mult*a; close=df['close']
    return ((close>up).astype(int)-(close<lo).astype(int)).rename('signal')
def sig_psar(df, af=0.02, max_af=0.2):
    high, low = df['high'], df['low']
    psar = low.copy()
    bull = True
    af_val = af
    ep = high.iloc[0]
    psar.iloc[0] = low.iloc[0]
    for i in range(2, len(df)):
        prev = psar.iloc[i-1]
        if bull:
            psar.iloc[i] = min(prev + af_val*(ep - prev), low.iloc[i-1], low.iloc[i-2])
            if high.iloc[i] > ep:
                ep = high.iloc[i]
                af_val = min(max_af, af_val + af)
            if low.iloc[i] < psar.iloc[i]:
                bull = False
                psar.iloc[i] = ep
                ep = low.iloc[i]
                af_val = af
        else:
            psar.iloc[i] = max(prev + af_val*(ep - prev), high.iloc[i-1], high.iloc[i-2])
            if low.iloc[i] < ep:
                ep = low.iloc[i]
                af_val = min(max_af, af_val + af)
            if high.iloc[i] > psar.iloc[i]:
                bull = True
                psar.iloc[i] = ep
                ep = high.iloc[i]
                af_val = af
    sig = ((df['close'] > psar).astype(int) - (df['close'] < psar).astype(int)).rename('signal')
    return sig
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
# Ensemble & gating
# ────────────────────────────────────────────────────────────────────────────────
def compute(df, signal, fee_bps=2.0, slippage_bps=1.0):
    ret=df['close'].pct_change().fillna(0.0)
    pos=signal.shift().fillna(0.0).clip(-1,1)
    cost=(pos.diff().abs().fillna(0.0))*((fee_bps+slippage_bps)/10000.0)
    pnl=pos*ret - cost
    equity=(1+pnl).cumprod()
    return ret, pos, pnl, equity

def backtest(df, signal, initial_cash=1.0, fee_bps=2.0, slippage_bps=1.0):
    ret,pos,pnl,eq=compute(df,signal,fee_bps,slippage_bps)
    return {'ret':ret,'pos':pos,'pnl':pnl,'equity':initial_cash*eq}

def sharpe(pnl: pd.Series, periods_per_year=365*24):
    s=pnl.std(); return 0.0 if s==0 or np.isnan(s) else float(pnl.mean()/s * np.sqrt(periods_per_year))
def max_drawdown(equity: pd.Series):
    peak=equity.cummax(); dd=equity/peak-1; return float(dd.min())
def _score(pnl, equity):
    s=max(0.0, min(3.0, sharpe(pnl))); dd=abs(max_drawdown(equity)); return s + (1.0 - min(dd,0.4))

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

def htf_gate(df_ltf: pd.DataFrame, df_htf: pd.DataFrame):
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
# Risk / sizing / confidence
# ────────────────────────────────────────────────────────────────────────────────
def adaptive_levels(df: pd.DataFrame, direction: int, atr_mult_sl=2.5, atr_mult_tp=3.8):
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

def confidence(df, sig):
    bt = backtest(df, sig, initial_cash=1.0, fee_bps=2.0, slippage_bps=1.0)
    s = max(0.0, min(3.0, sharpe(bt['pnl'])))
    dd = abs(max_drawdown(bt['equity']))
    return round((s/3.0)*70.0 + (1.0 - min(dd,0.4)/0.4)*30.0, 1)

def exposure_value(df_open):
    if df_open.empty: return 0.0
    return float((df_open['entry'].abs()*df_open['qty'].abs()).sum())

# ────────────────────────────────────────────────────────────────────────────────
# Telegram helper
# ────────────────────────────────────────────────────────────────────────────────
def tg(msg:str):
    try:
        token=st.secrets.get("TELEGRAM_BOT_TOKEN"); chat=st.secrets.get("TELEGRAM_CHAT_ID")
        if not token or not chat: return
        requests.get(f"https://api.telegram.org/bot{token}/sendMessage", params={"chat_id": chat, "text": msg})
    except Exception:
        pass

# ────────────────────────────────────────────────────────────────────────────────
# Portfolio (SQLite)
# ────────────────────────────────────────────────────────────────────────────────
DB = os.path.join(os.path.dirname(__file__), 'portfolio.db')
def _init_db():
    conn = sqlite3.connect(DB); c=conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS positions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      open_ts TEXT, close_ts TEXT, symbol TEXT, side TEXT,
      entry REAL, sl REAL, tp REAL, qty REAL,
      status TEXT, exit_price REAL, pnl REAL, note TEXT)''')
    conn.commit(); conn.close()

def _open_exists(symbol: str, side: str):
    _init_db(); conn=sqlite3.connect(DB)
    row = conn.execute('SELECT id FROM positions WHERE symbol=? AND side=? AND status="OPEN" LIMIT 1',
                       (symbol, side.upper())).fetchone()
    conn.close()
    return row[0] if row else None

def open_position(symbol, side, entry, sl, tp, qty, note=''):
    _init_db()
    if qty is None or float(qty) <= 0:
        return None
    # anti-duplicat (même symbole & sens)
    dup = _open_exists(symbol, side)
    if dup:
        return None  # on ignore silencieusement pour éviter les doublons
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
    row=conn.execute('SELECT side, entry, qty FROM positions WHERE id=? AND status="OPEN"', (pos_id,)).fetchone()
    if not row: conn.close(); raise ValueError("Position introuvable ou déjà close")
    side, entry, qty = row
    pnl=(float(exit_price)-float(entry))*float(qty)*(1 if side.upper()=="LONG" else -1)
    conn.execute('UPDATE positions SET close_ts=?, status=?, exit_price=?, pnl=?, note=? WHERE id=?',
                 (datetime.datetime.utcnow().isoformat(), "CLOSED", float(exit_price), float(pnl), note, pos_id))
    conn.commit(); conn.close()
    return pnl

def list_positions(status=None, limit=500):
    _init_db(); conn=sqlite3.connect(DB)
    q='SELECT id, open_ts, close_ts, symbol, side, entry, sl, tp, qty, status, exit_price, pnl, note FROM positions'
    params=()
    if status in ("OPEN","CLOSED"): q+=' WHERE status=?'; params=(status,)
    q+=' ORDER BY id DESC LIMIT ?'; params=params+(int(limit),)
    rows=list(conn.execute(q, params)); conn.close()
    return pd.DataFrame(rows, columns=['id','open_ts','close_ts','symbol','side','entry','sl','tp','qty','status','exit_price','pnl','note'])

# ────────────────────────────────────────────────────────────────────────────────
# Session helpers (sauver les résultats de scan entre les reruns)
# ────────────────────────────────────────────────────────────────────────────────
def save_df(key: str, df: pd.DataFrame):
    try:
        st.session_state[key] = df.to_json(orient='split', index=False)
    except Exception:
        st.session_state[key] = None

def load_df(key: str) -> pd.DataFrame:
    js = st.session_state.get(key)
    try:
        return pd.read_json(js, orient='split') if js else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# ────────────────────────────────────────────────────────────────────────────────
# Settings (UI)
# ────────────────────────────────────────────────────────────────────────────────
symbols_default = ['BTC/USDT','ETH/USDT','BNB/USDT','SOL/USDT','XRP/USDT','ADA/USDT','LINK/USDT','AVAX/USDT','TON/USDT','DOGE/USDT']

st.markdown("### ⚙️ Réglages")
with st.expander("Ouvrir / modifier les réglages", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        exchange = st.selectbox('Exchange', ['okx','bybit','kraken','coinbase','kucoin','binance'], index=0)
        tf = st.selectbox('Timeframe', ['15m','1h','4h'], index=1)
        htf = st.selectbox('HTF (confirmation)', ['1h','4h','1d'], index=2 if tf!='4h' else 1)
        capital = st.number_input('Capital (USD)', value=1000.0, step=50.0)
        symbols = st.multiselect('Paires', symbols_default, default=symbols_default[:8])
    with c2:
        mode = st.selectbox('Mode de risque', ['Conservateur','Balancé','Agressif','Custom'], index=1)
        presets = {'Conservateur': dict(risk_pct=0.6, max_expo=50.0, min_rr=1.6, max_positions=2),
                   'Balancé':     dict(risk_pct=1.2, max_expo=80.0, min_rr=1.7, max_positions=3),
                   'Agressif':     dict(risk_pct=2.0, max_expo=120.0, min_rr=1.8, max_positions=5),}
        if mode!='Custom':
            p=presets[mode]
            risk_pct=st.slider('Risque %/trade', 0.1, 5.0, p['risk_pct'], 0.1)
            max_expo=st.slider('Cap exposition (%)', 10.0, 200.0, p['max_expo'], 1.0)
            min_rr=st.slider('R/R minimum', 1.0, 5.0, p['min_rr'], 0.1)
            max_pos=st.slider('Nb max positions', 1, 8, p['max_positions'], 1)
        else:
            risk_pct=st.slider('Risque %/trade', 0.1, 5.0, 1.0, 0.1)
            max_expo=st.slider('Cap exposition (%)', 10.0, 200.0, 80.0, 1.0)
            min_rr=st.slider('R/R minimum', 1.0, 5.0, 1.5, 0.1)
            max_pos=st.slider('Nb max positions', 1, 8, 3, 1)
    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        sl_mult = st.slider("SL (×ATR)", 1.0, 4.0, 2.5, 0.1)
        tp_mult = st.slider("TP (×ATR)", 1.0, 6.0, 3.8, 0.1)
        allow_shorts = st.toggle("Autoriser les shorts", value=True)
        active_names = st.multiselect("Stratégies actives", list(STRATS.keys()), default=list(STRATS.keys()))
        telegram_on = st.toggle("Alertes Telegram (si secrets présents)", value=False)
    with c4:
        macro_enabled = st.toggle("Activer macro gating (VIX / Gold)", value=True)
        vix_caution = st.slider("Seuil VIX prudence", 10.0, 35.0, 20.0, 0.5)
        vix_riskoff = st.slider("Seuil VIX risk-off", 15.0, 50.0, 28.0, 0.5)
        gold_mom_thresh = st.slider("Seuil Gold momentum (3m)", 0.0, 0.3, 0.10, 0.01)
    st.markdown("---")
    st.markdown("**Gestion avancée des positions**")
    colA, colB = st.columns(2)
    with colA:
        trail_on = st.toggle("Trailing stop ATR", value=True)
        trail_mult = st.slider("ATR trail (×)", 0.8, 3.0, 1.5, 0.1)
    with colB:
        be_on = st.toggle("Break-even auto", value=True)
        be_trigger = st.slider("Déclencheur BE (ret_% ≥)", 0.5, 10.0, 2.0, 0.5)

tabs = st.tabs(['🏠 Top Picks', '📈 Portefeuille', '🧾 Journal', '🧪 Backtest 3Y', '🔬 Lab'])

# ────────────────────────────────────────────────────────────────────────────────
# Tab 1 — Scanner / sélection
# ────────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader('Top Picks (1 clic)')
    open_df = list_positions(status='OPEN', limit=500)
    current_expo = exposure_value(open_df)
    cap_allowed = capital*(max_expo/100.0)
    budget_left = max(0.0, cap_allowed - current_expo)
    st.metric('Capital dispo à engager', f"{budget_left:.2f} USD")

    macro_mult, macro_note = macro_gate(macro_enabled, vix_caution, vix_riskoff, gold_mom_thresh)
    st.caption(f"Macro gate: {macro_note} → multiplicateur {macro_mult}")

    if st.button('🚀 Scanner maintenant'):
        rows=[]
        for sym in symbols:
            df = load_or_fetch(exchange, sym, tf, limit=2000)
            df_htf = load_or_fetch(exchange, sym, htf, limit=600)
            use = {name: STRATS[name] for name in active_names if name in STRATS}
            if not use:
                st.warning("Aucune stratégie active — active au moins une dans les réglages.")
                break
            signals = {name: fn(df) for name, fn in use.items()}
            w = ensemble_weights(df, signals, window=300)
            sig = blended_signal(signals, w)
            gate = htf_gate(df, df_htf)
            sig = (sig * gate).clip(-1,1) * macro_mult
            d = int(np.sign(sig.iloc[-1]))
            if d==0 or (d<0 and not allow_shorts): 
                continue
            lvl = adaptive_levels(df, d, atr_mult_sl=sl_mult, atr_mult_tp=tp_mult)
            if not lvl: continue
            this_rr = rr(lvl['entry'], lvl['sl'], lvl['tp'])
            if this_rr < min_rr: continue
            qty = size_fixed_pct(capital, lvl['entry'], lvl['sl'], risk_pct)
            if qty <= 0: continue
            conf = confidence(df, sig)
            score = conf * this_rr
            rows.append({'symbol': sym, 'dir':'LONG' if d>0 else 'SHORT', 'entry':lvl['entry'],'sl':lvl['sl'],'tp':lvl['tp'],
                         'rr': this_rr, 'qty': qty, 'confiance': conf, 'score': score})

        if not rows:
            st.info("Aucun setup solide pour l’instant.")
            save_df("last_picks", pd.DataFrame()); save_df("last_picks_alloc", pd.DataFrame())
        else:
            picks = (pd.DataFrame(rows).sort_values(['score','confiance','rr'], ascending=False).head(int(max_pos)))
            # Allocation budget restant
            w = picks['score'].clip(lower=0.0)
            if w.sum() > 0 and budget_left > 0:
                alloc = (w/w.sum())*budget_left
                picks['alloc_qty'] = (alloc / picks['entry'].abs().clip(lower=1e-9))
            else:
                picks['alloc_qty'] = 0.0
            save_df("last_picks", picks.dropna(axis=1, how='all'))
            save_df("last_picks_alloc", picks.dropna(axis=1, how='all'))
            st.success("Scan terminé ✅ — sélectionne ci-dessous.")

    picks_alloc = load_df("last_picks_alloc")
    if not picks_alloc.empty:
        picks_alloc = picks_alloc.loc[(picks_alloc['qty'].abs()>0) | (picks_alloc['alloc_qty'].abs()>0)]
        if picks_alloc.empty:
            st.info("Tous les setups ont une quantité nulle.")
        else:
            st.write("Résultats :")
            st.dataframe(picks_alloc[['symbol','dir','entry','sl','tp','rr','confiance','score','qty','alloc_qty']].round(6),
                         use_container_width=True)
            picks_alloc = picks_alloc.reset_index(drop=True)
            options = [f"{i+1} — {row['symbol']} ({row['dir']}) · RR {row['rr']:.2f} · conf {row['confiance']:.0f}"
                       for i, row in picks_alloc.iterrows()]
            default_sel = options
            sel = st.multiselect("Sélectionne les trades à enregistrer :", options, default=default_sel)
            colA, colB = st.columns(2)
            with colA:
                if st.button('📌 Enregistrer la sélection'):
                    n = 0
                    for i, label in enumerate(options):
                        if label not in sel: continue
                        r = picks_alloc.iloc[i]
                        q = float(r.get('alloc_qty', 0.0)) if 'alloc_qty' in r and r['alloc_qty']>0 else float(r['qty'])
                        if q <= 0: continue
                        rid = open_position(r['symbol'], r['dir'], float(r['entry']), float(r['sl']), float(r['tp']), q, note='PICK')
                        if rid:
                            n += 1
                            if telegram_on: tg(f"📌 OUVERT {r['symbol']} {r['dir']} qty {q:.4f} entry {r['entry']:.6f}")
                    st.success(f"{n} trade(s) ajouté(s) au portefeuille.")
                    st.rerun()
            with colB:
                if st.button('⚙️ Enregistrer tout (affiché)'):
                    n = 0
                    for _, r in picks_alloc.iterrows():
                        q = float(r.get('alloc_qty', 0.0)) if 'alloc_qty' in r and r['alloc_qty']>0 else float(r['qty'])
                        if q <= 0: continue
                        rid = open_position(r['symbol'], r['dir'], float(r['entry']), float(r['sl']), float(r['tp']), q, note='ALL_VISIBLE')
                        if rid:
                            n += 1
                            if telegram_on: tg(f"📌 OUVERT {r['symbol']} {r['dir']} qty {q:.4f} entry {r['entry']:.6f}")
                    st.success(f"{n} trade(s) ajouté(s) au portefeuille.")
                    st.rerun()
    else:
        st.info("Clique d’abord sur **Scanner maintenant** pour obtenir des propositions.")

# ────────────────────────────────────────────────────────────────────────────────
# Tab 2 — Portefeuille (suivi, trailing, BE, TP/SL)
# ────────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader('Positions ouvertes')
    open_df = list_positions(status='OPEN', limit=500)
    open_df = open_df.loc[open_df['qty'].abs() > 0]

    if open_df.empty:
        st.info('Aucune position ouverte.')
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

        open_df = open_df.copy()
        open_df['last'] = open_df['symbol'].map(last_prices)
        open_df['PnL_latent'] = open_df.apply(latent, axis=1)
        open_df['ret_%'] = open_df.apply(ret_pct, axis=1).round(3)
        open_df['sens'] = np.where(open_df['ret_%']>=0, '✅ OK', '❌ Pas OK')

        st.dataframe(open_df[['id','symbol','side','entry','sl','tp','qty','last','ret_%','PnL_latent','sens']].round(6),
                     use_container_width=True)

        # Équity dynamique
        closed_df = list_positions(status='CLOSED', limit=100000)
        realized = 0.0 if closed_df.empty else float(closed_df['pnl'].sum())
        latent_total = float(open_df['PnL_latent'].sum())
        equity = capital + realized + latent_total
        st.metric("Équity dynamique", f"{equity:.2f} USD")

        # ── Mise à jour : trailing / BE / TP/SL
        if st.button('🔍 Mettre à jour (TP/SL + trailing/BE)'):
            closed = 0
            updated = 0
            for _, r in open_df.iterrows():
                px = last_prices.get(r['symbol'], r['entry'])

                # Trailing ATR + Break-even (sur la TF choisie)
                try:
                    if trail_on or be_on:
                        df_now = load_or_fetch(exchange, r['symbol'], tf, limit=200)
                        atr_now = float(atr_df(df_now, 14).iloc[-1])
                        new_sl = r['sl']
                        # BE : si gain en % ≥ trigger → SL = entry
                        if be_on:
                            rp = 100.0 * (px - r['entry'])/r['entry'] * (1 if r['side']=='LONG' else -1)
                            if rp >= be_trigger:
                                new_sl = max(new_sl, r['entry']) if r['side']=='LONG' else min(new_sl, r['entry'])
                        # Trailing ATR
                        if trail_on:
                            if r['side']=='LONG':
                                new_sl = max(new_sl, px - trail_mult*atr_now)
                            else:
                                new_sl = min(new_sl, px + trail_mult*atr_now)
                        if (r['side']=='LONG' and new_sl>r['sl']) or (r['side']=='SHORT' and new_sl<r['sl']):
                            update_sl(int(r['id']), float(new_sl)); updated += 1
                except Exception:
                    pass

                # Fermeture auto TP/SL
                if r['side'] == 'LONG' and ((px >= r['tp']) or (px <= r['sl'])):
                    pnl = close_position(int(r['id']), float(px), note='AUTO_TP_SL')
                    if telegram_on: tg(f"✅ FERMÉ {r['symbol']} LONG · PnL {pnl:.2f}")
                    st.success(f"Position {int(r['id'])} clôturée. PnL ≈ {pnl:.2f}")
                    closed += 1
                if r['side'] == 'SHORT' and ((px <= r['tp']) or (px >= r['sl'])):
                    pnl = close_position(int(r['id']), float(px), note='AUTO_TP_SL')
                    if telegram_on: tg(f"✅ FERMÉ {r['symbol']} SHORT · PnL {pnl:.2f}")
                    st.success(f"Position {int(r['id'])} clôturée. PnL ≈ {pnl:.2f}")
                    closed += 1

            if updated: st.info(f"SL mis à jour sur {updated} position(s) (BE/Trailing).")
            if closed: st.rerun()

        st.markdown('---')
        st.write('Clôture manuelle :')
        for _, r in open_df.iterrows():
            cols = st.columns([3,1,1])
            with cols[0]: st.write(f"{r['symbol']} · {r['side']} — entry {r['entry']:.6f} qty {r['qty']:.4f} · SL {r['sl']:.6f} · TP {r['tp']:.6f}")
            with cols[1]: mkt = last_prices.get(r['symbol'], r['entry']); st.write(f"⚡ {mkt:.6f}")
            with cols[2]:
                if st.button(f"Clôturer #{int(r['id'])}", key=f"close_{int(r['id'])}"):
                    pnl = close_position(int(r['id']), float(mkt), note='MANUAL')
                    if telegram_on: tg(f"ℹ️ FERMÉ MANUEL {r['symbol']} {r['side']} · PnL {pnl:.2f}")
                    st.success(f"Position {int(r['id'])} clôturée. PnL ≈ {pnl:.2f}")
                    st.rerun()

# ────────────────────────────────────────────────────────────────────────────────
# Tab 3 — Journal + stats
# ────────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader('Historique (clôturées)')
    closed_df = list_positions(status='CLOSED', limit=100000)
    if closed_df.empty:
        st.info('Aucun trade clôturé.')
    else:
        st.dataframe(closed_df[['id','open_ts','close_ts','symbol','side','entry','exit_price','qty','pnl','note']],
                     use_container_width=True)
        total = float(closed_df['pnl'].sum())
        wins = (closed_df['pnl']>0).sum()
        losses = (closed_df['pnl']<=0).sum()
        wr = 100.0 * wins / max(1, wins+losses)
        avg_win = float(closed_df.loc[closed_df['pnl']>0,'pnl'].mean() or 0)
        avg_loss = float(closed_df.loc[closed_df['pnl']<=0,'pnl'].mean() or 0)
        pf = (closed_df.loc[closed_df['pnl']>0,'pnl'].sum() / abs(closed_df.loc[closed_df['pnl']<=0,'pnl'].sum() or 1)) if losses>0 else np.nan
        expectancy = wr/100.0 * avg_win + (1-wr/100.0) * avg_loss
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("P&L réalisé", f"{total:.2f} USD")
        c2.metric("Win rate", f"{wr:.1f}%")
        c3.metric("Profit factor", f"{pf:.2f}" if not np.isnan(pf) else "∞")
        c4.metric("Avg win", f"{avg_win:.2f}")
        c5.metric("Avg loss", f"{avg_loss:.2f}")
        st.caption(f"Expectancy/trade ≈ {expectancy:.2f} USD")

# ────────────────────────────────────────────────────────────────────────────────
# Tab 4 — Backtest 3Y (1D)
# ────────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader('Backtest rapide 3 ans (1D)')
    sym = st.selectbox('Symbole', symbols_default)
    if st.button('▶️ Lancer backtest'):
        df = load_or_fetch(exchange, sym, '1d', limit=1500)
        signals = {k: fn(df) for k, fn in STRATS.items()}
        w = ensemble_weights(df, signals, window=300)
        sig = blended_signal(signals, w)
        bt = backtest(df, sig, initial_cash=1.0, fee_bps=2.0, slippage_bps=1.0)
        st.line_chart(pd.Series(bt['equity'], name='Equity (norm.)'))
        st.write({'Sharpe': round(sharpe(bt['pnl']),2), 'MaxDD': round(max_drawdown(bt['equity']),3)})

# ────────────────────────────────────────────────────────────────────────────────
# Tab 5 — Lab (qualité par stratégie)
# ────────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader('Lab — Perfs par stratégie (lookback 300 barres)')
    sym = st.selectbox('Symbole (Lab)', symbols_default, index=0, key="lab_sym")
    if st.button('📊 Tester stratégies (Lab)'):
        df = load_or_fetch(exchange, sym, tf, limit=800)
        look = 300; dfw = df.iloc[-look:]
        rows=[]
        for name, fn in STRATS.items():
            try:
                sig = fn(dfw)
                _,_,pnl,eq = compute(dfw, sig)
                rows.append({'strategie':name, 'Sharpe':sharpe(pnl), 'MaxDD':max_drawdown(eq)})
            except Exception:
                rows.append({'strategie':name, 'Sharpe':np.nan, 'MaxDD':np.nan})
        res = pd.DataFrame(rows).sort_values('Sharpe', ascending=False)
        st.dataframe(res, use_container_width=True)
        st.caption("Astuce: active uniquement les stratégies top-Sharpe dans les réglages pour prioriser l'ensemble.")

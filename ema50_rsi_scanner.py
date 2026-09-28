import os, time, requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

EMA_LEN, RSI_LEN, VOL_LEN = 50, 14, 20
RSI_LO, RSI_HI = 45, 50
VOL_MULT, LOOKBACK, NEAR_PCT = 1.5, 3, 1.5
HDR = {"User-Agent": "Mozilla/5.0"}

def fetch(sym, rng, interval):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}.NS"
    for _ in range(3):
        try:
            r = requests.get(url, params={"range": rng, "interval": interval},
                             headers=HDR, timeout=20)
            res = r.json()["chart"]["result"][0]
            q = res["indicators"]["quote"][0]
            df = pd.DataFrame(q, index=pd.to_datetime(res["timestamp"], unit="s", utc=True)
                              .tz_convert("Asia/Kolkata"))
            return df[["open", "high", "low", "close", "volume"]].dropna()
        except Exception:
            time.sleep(1.5)
    return None

def to_4h(df):
    d = df.copy()
    d["day"] = d.index.date
    d["blk"] = d.groupby("day").cumcount() // 4
    g = d.groupby(["day", "blk"])
    return pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                         "low": g["low"].min(), "close": g["close"].last(),
                         "volume": g["volume"].sum()})

def check(df):
    if df is None or len(df) < EMA_LEN + 5:
        return None
    c = df["close"]
    ema = c.ewm(span=EMA_LEN, adjust=False).mean()
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / RSI_LEN, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / RSI_LEN, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn)
    vr = df["volume"] / df["volume"].rolling(VOL_LEN).mean()
    cross = ((c.shift(1) <= ema.shift(1)) & (c > ema)).tail(LOOKBACK).any()
    vol_ok = vr.tail(LOOKBACK).max() >= VOL_MULT
    r, rp = rsi.iloc[-1], rsi.iloc[-2]
    rsi_ok = RSI_LO <= r <= RSI_HI
    gap = (c.iloc[-1] - ema.iloc[-1]) / ema.iloc[-1] * 100
    near = c.iloc[-1] < ema.iloc[-1] and gap >= -NEAR_PCT and rsi_ok and r > rp
    if cross and vol_ok and rsi_ok:
        return "CROSS", r, gap
    if near:
        return "NEAR", r, gap
    return None

def scan(row):
    sym, sec = row
    out = []
    d = fetch(sym, "1y", "1d")
    res = check(d)
    if res:
        out.append((sec, sym, "D", *res))
    h = fetch(sym, "60d", "60m")
    res = check(to_4h(h)) if h is not None else None
    if res:
        out.append((sec, sym, "4H", *res))
    return out

def send(text):
    tok, chat = os.environ["TELEGRAM_BOT_TOKEN"], os.environ["TELEGRAM_CHAT_ID"]
    for i in range(0, len(text), 3800):
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      data={"chat_id": chat, "text": text[i:i + 3800]}, timeout=20)

def main():
    stocks = pd.read_csv("stocks.csv").drop_duplicates("symbol")
    rows = list(zip(stocks["symbol"], stocks["sector"]))
    with ThreadPoolExecutor(max_workers=6) as ex:
        hits = [h for r in ex.map(scan, rows) for h in r]
    if not hits:
        send("EMA50+RSI scan: koi CROSS/NEAR nahi mila.")
        return
    msg = f"EMA50 + RSI(45-50) scan: {len(hits)} signals\n"
    for sec in sorted({h[0] for h in hits}):
        msg += f"\n{sec}\n"
        for _, sym, tf, kind, r, gap in sorted([h for h in hits if h[0] == sec], key=lambda x: (x[3], x[1])):
            msg += f"  {kind} {sym} [{tf}] RSI {r:.1f} gap {gap:+.1f}%\n"
    send(msg)

main()

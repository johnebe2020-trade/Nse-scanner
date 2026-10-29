import os
import pandas as pd
import yfinance as yf
import requests

FILE = "EQUITY_L.csv"
LOOKBACK = 20            # resistance = highest high of previous 20 days
MIN_PRICE = 20
MIN_AVG_VOL = 100000
TREND_DAYS = 10          # EMA50 must be rising vs this many days ago
VOL_MULT = 1.5           # breakout day volume vs 20-day average

COLUMNS = ["status", "symbol", "close", "resistance", "date", "note"]

df = pd.read_csv(FILE)
df.columns = df.columns.str.strip()
df = df[df["SERIES"].str.strip() == "EQ"]
symbols = df["SYMBOL"].str.strip().str.upper().tolist()

hits, no_data = [], []

for i in range(0, len(symbols), 100):
    chunk = symbols[i:i + 100]
    tickers = [s + ".NS" for s in chunk]
    data = yf.download(tickers, period="1y", interval="1d",
                       group_by="ticker", auto_adjust=True,
                       threads=True, progress=False)
    for s, t in zip(chunk, tickers):
        try:
            d = data[t].dropna()
            if len(d) < 80:
                no_data.append(s)
                continue

            close, high, vol = d["Close"], d["High"], d["Volume"]
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()
            res = high.shift(1).rolling(LOOKBACK).max()

            if close.iloc[-1] < MIN_PRICE or vol.tail(20).mean() < MIN_AVG_VOL:
                continue

            vol_x = vol.iloc[-1] / vol.tail(20).mean()
            checks = {
                "BREAKOUT": close.iloc[-1] > res.iloc[-1]
                            and close.iloc[-2] <= res.iloc[-2],
                "ABOVE EMA50": close.iloc[-1] > ema50.iloc[-1],
                "UPTREND": ema20.iloc[-1] > ema50.iloc[-1]
                           and ema50.iloc[-1] > ema50.iloc[-1 - TREND_DAYS],
                "VOLUME 1.5x": vol_x >= VOL_MULT,
            }
            score = sum(checks.values())
            if score < 3:
                continue

            missing = [k for k, v in checks.items() if not v]
            hits.append({
                "status": "ACTIVE" if score == 4 else "WATCH",
                "symbol": s,
                "close": round(close.iloc[-1], 2),
                "resistance": round(res.iloc[-1], 2),
                "date": d.index[-1].strftime("%d-%b-%y"),
                "note": "4/4 OK" if score == 4 else f"3/4 NEED {missing[0]}",
            })
        except Exception:
            no_data.append(s)

out = pd.DataFrame(hits, columns=COLUMNS)
if not out.empty:
    out = out.sort_values(["status", "symbol"]).reset_index(drop=True)

out.to_csv("results_breakout.csv", index=False)
pd.Series(no_data, name="SYMBOL").to_csv("no_data.csv", index=False)

print(out.to_string(index=False) if not out.empty else "No breakouts today")
print("\nNo data / wrong ticker:", len(no_data))

# Telegram alert (skipped if secrets are missing)
token = os.getenv("TELEGRAM_BOT_TOKEN")
chat = os.getenv("TELEGRAM_CHAT_ID")
if token and chat:
    if out.empty:
        msg = "Breakout scan: no stocks today"
    else:
        lines = [f"{r.status} | {r.symbol} | C {r.close} | R {r.resistance} | {r.note}"
                 for r in out.head(30).itertuples()]
        msg = f"Breakouts ({len(out)}):\n" + "\n".join(lines)
    requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                  data={"chat_id": chat, "text": msg}, timeout=30)

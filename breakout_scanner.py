import os
import math
import numpy as np
import pandas as pd
import yfinance as yf
import requests

FILE = "EQUITY_L.csv"
LOOKBACK = 20            # resistance = highest high of previous 20 days
MIN_PRICE = 20
MIN_AVG_VOL = 100000
TREND_DAYS = 10          # EMA50 must be rising vs this many days ago
VOL_MULT = 1.5           # breakout day volume vs 20-day average
ST_PERIOD = 10           # Supertrend ATR period
ST_MULT = 3.0            # Supertrend multiplier
MAX_CHASE = 0.05         # flag if close is >5% above resistance
EXT_ATR = 2.5            # flag if close is >2.5 ATR above EMA20
MAX_SL_PCT = 10.0        # flag if stop is wider than this % of entry
WATCH_NEAR = 0.97        # WATCH rows must be within 3% of resistance

# ---- trade plan settings (change these) ----
RISK_PER_TRADE = 1000    # rupees you accept to lose per trade
ENTRY_BUFFER = 0.002     # entry = signal high + 0.2%
SL_BUFFER = 0.002        # candle-low stop sits 0.2% below the low
GAP_LIMIT = 0.03         # skip if next open is >3% above entry
T1_R, T2_R = 1.5, 3.0    # targets as multiples of risk
RISKY_SIZE = 0.5         # RISKY setups get half quantity

COLUMNS = ["status", "symbol", "close", "entry", "sl", "risk_pct", "qty",
           "cost", "t1", "t2", "skip_above", "date", "flags", "note"]


def tick(x):
    return round(round(x / 0.05) * 0.05, 2)


def supertrend(high, low, close, period=10, mult=3.0):
    hl2 = (high + low) / 2
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()],
                   axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    ub = (hl2 + mult * atr).values
    lb = (hl2 - mult * atr).values
    c = close.values
    n = len(c)
    fu, fl = ub.copy(), lb.copy()
    st = np.zeros(n)
    up = np.ones(n, dtype=bool)
    st[0] = fl[0]
    for i in range(1, n):
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if up[i - 1]:
            if c[i] < fl[i]:
                up[i], st[i] = False, fu[i]
            else:
                up[i], st[i] = True, fl[i]
        else:
            if c[i] > fu[i]:
                up[i], st[i] = True, fl[i]
            else:
                up[i], st[i] = False, fu[i]
    return st, up, atr


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

            op, high, low, close, vol = (d["Open"], d["High"], d["Low"],
                                         d["Close"], d["Volume"])
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()
            res = high.shift(1).rolling(LOOKBACK).max()
            st, st_up, atr = supertrend(high, low, close, ST_PERIOD, ST_MULT)

            c0, c1 = close.iloc[-1], close.iloc[-2]
            if c0 < MIN_PRICE or vol.tail(20).mean() < MIN_AVG_VOL:
                continue

            vol_x = vol.iloc[-1] / vol.tail(20).mean()
            checks = {
                "BREAKOUT": c0 > res.iloc[-1] and c1 <= res.iloc[-2],
                "ABOVE EMA50": c0 > ema50.iloc[-1],
                "UPTREND": ema20.iloc[-1] > ema50.iloc[-1]
                           and ema50.iloc[-1] > ema50.iloc[-1 - TREND_DAYS],
                "VOLUME 1.5x": vol_x >= VOL_MULT,
            }
            score = sum(checks.values())
            if score < 3:
                continue

            missing = [k for k, v in checks.items() if not v]
            if "BREAKOUT" in missing and c0 < res.iloc[-1] * WATCH_NEAR:
                continue   # too far from resistance

            # ---- candle and risk flags (today's candle) ----
            o0, h0, l0 = op.iloc[-1], high.iloc[-1], low.iloc[-1]
            rng = h0 - l0
            body = abs(c0 - o0)
            upper_wick = h0 - max(o0, c0)
            close_pos = (c0 - l0) / rng if rng > 0 else 0.5

            flags = []
            if rng > 0 and upper_wick / rng >= 0.4 and upper_wick > body:
                flags.append("WICK")
            if close_pos < 0.4:
                flags.append("WEAK CLOSE")
            if vol_x >= 3 and close_pos < 0.5:
                flags.append("CLIMAX")
            if (c0 - ema20.iloc[-1]) > EXT_ATR * atr.iloc[-1]:
                flags.append("EXTENDED")
            if (close.diff().tail(3) > 0).all():
                flags.append("3 GREEN")
            if c0 > res.iloc[-1] * (1 + MAX_CHASE):
                flags.append("CHASE")
            if not st_up[-1]:
                flags.append("ST DOWN")

            # ---- trade plan ----
            entry = tick(h0 * (1 + ENTRY_BUFFER))
            candle_sl = l0 * (1 - SL_BUFFER)
            sl_raw = max(float(st[-1]), candle_sl) if st_up[-1] else candle_sl
            sl = tick(sl_raw)
            risk_ps = entry - sl
            risk_pct = round(risk_ps / entry * 100, 1) if risk_ps > 0 else None

            if risk_ps <= 0:
                flags.append("BAD SL")
            elif risk_pct > MAX_SL_PCT:
                flags.append("WIDE SL")

            if score == 4 and not flags:
                status, note = "ACTIVE", "4/4 OK"
            elif score == 4:
                status, note = "RISKY", "4/4 BUT CAUTION"
            else:
                status, note = "WATCH", f"3/4 NEED {missing[0]}"

            plan = {"entry": "-", "sl": "-", "risk_pct": "-", "qty": "-",
                    "cost": "-", "t1": "-", "t2": "-", "skip_above": "-"}
            can_plan = (status in ("ACTIVE", "RISKY") and risk_ps > 0
                        and "WIDE SL" not in flags)
            if can_plan:
                qty = math.floor(RISK_PER_TRADE / risk_ps)
                if status == "RISKY":
                    qty = math.floor(qty * RISKY_SIZE)
                if qty >= 1:
                    plan = {
                        "entry": entry,
                        "sl": sl,
                        "risk_pct": risk_pct,
                        "qty": qty,
                        "cost": round(entry * qty),
                        "t1": tick(entry + T1_R * risk_ps),
                        "t2": tick(entry + T2_R * risk_ps),
                        "skip_above": tick(entry * (1 + GAP_LIMIT)),
                    }
                else:
                    flags.append("QTY<1")

            hits.append({
                "status": status,
                "symbol": s,
                "close": round(c0, 2),
                **plan,
                "date": d.index[-1].strftime("%d-%b-%y"),
                "flags": " ".join(flags) if flags else "-",
                "note": note,
            })
        except Exception:
            no_data.append(s)

out = pd.DataFrame(hits, columns=COLUMNS)
if not out.empty:
    order = {"ACTIVE": 0, "RISKY": 1, "WATCH": 2}
    out["_o"] = out["status"].map(order)
    out = out.sort_values(["_o", "symbol"]).drop(columns="_o").reset_index(drop=True)

out.to_csv("results_breakout.csv", index=False)
pd.Series(no_data, name="SYMBOL").to_csv("no_data.csv", index=False)

print(out.to_string(index=False) if not out.empty else "No breakouts today")
print("\nNo data / wrong ticker:", len(no_data))

# Telegram alert (skipped if secrets are missing)
token = os.getenv("TELEGRAM_BOT_TOKEN")
chat = os.getenv("TELEGRAM_CHAT_ID")
if token and chat:
    plans = out[out["status"].isin(["ACTIVE", "RISKY"])]
    if out.empty:
        msg = "Breakout scan: no stocks today"
    else:
        lines = []
        for r in plans.head(20).itertuples():
            if r.entry == "-":
                lines.append(f"{r.status} | {r.symbol} | no plan | {r.flags}")
            else:
                lines.append(
                    f"{r.status} | {r.symbol} | BUY above {r.entry} | SL {r.sl} "
                    f"| Qty {r.qty} | T1 {r.t1} | skip if open > {r.skip_above}"
                    f" | {r.flags}")
        watch = int((out["status"] == "WATCH").sum())
        msg = (f"Trade plans ({len(plans)}), WATCH {watch}:\n"
               + "\n".join(lines))
    requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                  data={"chat_id": chat, "text": msg}, timeout=30)

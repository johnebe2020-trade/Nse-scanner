import os
import numpy as np
import pandas as pd
import yfinance as yf
import requests

FILE = "positions.csv"
ST_PERIOD = 10
ST_MULT = 3.0
NEAR_SL_PCT = 3.0        # warn if close is within 3% of the stop

COLUMNS = ["action", "symbol", "buy", "close", "pnl_pct", "sl",
           "to_sl_pct", "date", "note"]


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
    return st, up


def send_telegram(msg):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": msg}, timeout=30)


if not os.path.exists(FILE):
    print("positions.csv not found")
    raise SystemExit(0)

pos = pd.read_csv(FILE)
pos.columns = pos.columns.str.strip().str.lower()
pos = pos.dropna(subset=["symbol"])
if pos.empty:
    print("No open positions")
    send_telegram("Position monitor: no open positions")
    raise SystemExit(0)

rows = []
for r in pos.itertuples():
    s = str(r.symbol).strip().upper().replace("NSE:", "")
    try:
        buy = float(r.buy_price)
        my_sl = getattr(r, "my_sl", np.nan)
        my_sl = float(my_sl) if pd.notna(my_sl) else None

        d = yf.Ticker(s + ".NS").history(period="1y", interval="1d",
                                         auto_adjust=True).dropna()
        if len(d) < 30:
            rows.append({"action": "ERR", "symbol": s, "buy": buy,
                         "close": "-", "pnl_pct": "-", "sl": "-",
                         "to_sl_pct": "-", "date": "-",
                         "note": "NO DATA CHECK SYMBOL"})
            continue

        st, st_up = supertrend(d["High"], d["Low"], d["Close"],
                               ST_PERIOD, ST_MULT)
        c0 = float(d["Close"].iloc[-1])
        st_line = float(st[-1])
        pnl = round((c0 - buy) / buy * 100, 1)
        date = d.index[-1].strftime("%d-%b-%y")

        # stop to use: the higher of Supertrend and your own stop
        stop = st_line if st_up[-1] else None
        if my_sl is not None:
            stop = max(stop, my_sl) if stop is not None else my_sl

        if not st_up[-1]:
            action, note = "EXIT", "CLOSED BELOW SUPERTREND"
        elif my_sl is not None and c0 < my_sl:
            action, note = "EXIT", "CLOSED BELOW MY SL"
        else:
            to_sl = (c0 - stop) / c0 * 100
            if to_sl <= NEAR_SL_PCT:
                action, note = "WARN", f"NEAR SL ({to_sl:.1f}%)"
            else:
                action, note = "HOLD", "TREND OK"

        to_sl_pct = round((c0 - stop) / c0 * 100, 1) if stop else "-"
        rows.append({
            "action": action, "symbol": s, "buy": buy, "close": round(c0, 2),
            "pnl_pct": pnl,
            "sl": round(stop, 2) if stop else "-",
            "to_sl_pct": to_sl_pct, "date": date, "note": note,
        })
    except Exception as e:
        rows.append({"action": "ERR", "symbol": s, "buy": "-", "close": "-",
                     "pnl_pct": "-", "sl": "-", "to_sl_pct": "-",
                     "date": "-", "note": f"ERROR {type(e).__name__}"})

out = pd.DataFrame(rows, columns=COLUMNS)
order = {"EXIT": 0, "WARN": 1, "ERR": 2, "HOLD": 3}
out["_o"] = out["action"].map(order)
out = out.sort_values(["_o", "symbol"]).drop(columns="_o").reset_index(drop=True)
out.to_csv("positions_status.csv", index=False)
print(out.to_string(index=False))

lines = [f"{r.action} | {r.symbol} | C {r.close} | SL {r.sl} | P&L {r.pnl_pct}% | {r.note}"
         for r in out.itertuples()]
header = "EXIT ALERT" if (out["action"] == "EXIT").any() else "Position monitor"
send_telegram(f"{header} ({len(out)} positions):\n" + "\n".join(lines))

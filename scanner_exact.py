import math
import sys
from collections import defaultdict
import numpy as np
import pandas as pd
import simple_backtest as sb

# ---- exactly your breakout_scanner.py settings ----
LOOKBACK = 20
MIN_PRICE = 20
MIN_AVG_VOL = 100000
TREND_DAYS = 10
VOL_MULT = 1.5
ST_PERIOD, ST_MULT = 10, 3.0
MAX_CHASE = 0.05
EXT_ATR = 2.5
MAX_SL_PCT = 10.0
RISK_PER_TRADE = 1000
ENTRY_BUFFER = 0.002
SL_BUFFER = 0.002
GAP_LIMIT = 0.03
T1_R, T2_R = 1.5, 3.0
RISKY_SIZE = 0.5
# ---- exit settings (from your backtest.py) ----
TP1_PART = 0.5
MAX_BARS = 20
MIN_N_HALF = 100


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


def prep_sig(d):
    op, high, low, close, vol = (d["Open"], d["High"], d["Low"],
                                 d["Close"], d["Volume"])
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    res = high.shift(1).rolling(LOOKBACK).max()
    st, st_up, atr = supertrend(high, low, close, ST_PERIOD, ST_MULT)
    avgv = vol.rolling(20).mean()
    return dict(o=op.values, h=high.values, l=low.values, c=close.values,
                v=vol.values, ema20=ema20.values, ema50=ema50.values,
                res=res.values, st=st, up=st_up, atr=atr.values,
                avgv=avgv.values, dates=d.index)


def plan_at(S, i):
    """Same logic as breakout_scanner.py, evaluated on bar i.
    Returns None or dict(status, flags, entry, sl, qty, t1, t2, skip)."""
    c, h, l, o = S["c"], S["h"], S["l"], S["o"]
    res, ema50, ema20 = S["res"], S["ema50"], S["ema20"]
    if i < 80 or not (res[i] == res[i]) or not (res[i - 1] == res[i - 1]):
        return None
    c0, c1 = c[i], c[i - 1]
    if c0 < MIN_PRICE or not (S["avgv"][i] >= MIN_AVG_VOL):
        return None
    vol_x = S["v"][i] / S["avgv"][i]
    ok = (c0 > res[i] and c1 <= res[i - 1],
          c0 > ema50[i],
          ema20[i] > ema50[i] and ema50[i] > ema50[i - TREND_DAYS],
          vol_x >= VOL_MULT)
    if sum(ok) < 4:
        return None
    o0, h0, l0 = o[i], h[i], l[i]
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
    if (c0 - ema20[i]) > EXT_ATR * S["atr"][i]:
        flags.append("EXTENDED")
    if c[i] > c[i - 1] and c[i - 1] > c[i - 2] and c[i - 2] > c[i - 3]:
        flags.append("3 GREEN")
    if c0 > res[i] * (1 + MAX_CHASE):
        flags.append("CHASE")
    if not S["up"][i]:
        flags.append("ST DOWN")
    entry = tick(h0 * (1 + ENTRY_BUFFER))
    candle_sl = l0 * (1 - SL_BUFFER)
    sl_raw = max(float(S["st"][i]), candle_sl) if S["up"][i] else candle_sl
    sl = tick(sl_raw)
    risk_ps = entry - sl
    risk_pct = round(risk_ps / entry * 100, 1) if risk_ps > 0 else None
    if risk_ps <= 0:
        flags.append("BAD SL")
    elif risk_pct > MAX_SL_PCT:
        flags.append("WIDE SL")
    status = "ACTIVE" if not flags else "RISKY"
    if risk_ps <= 0 or "WIDE SL" in flags:
        return dict(status=status, flags=flags, entry=None)
    qty = math.floor(RISK_PER_TRADE / risk_ps)
    if status == "RISKY":
        qty = math.floor(qty * RISKY_SIZE)
    if qty < 1:
        return dict(status=status, flags=flags + ["QTY<1"], entry=None)
    return dict(status=status, flags=flags, entry=entry, sl=sl, qty=qty,
                risk_pct=risk_pct, t1=tick(entry + T1_R * risk_ps),
                t2=tick(entry + T2_R * risk_ps),
                skip=tick(entry * (1 + GAP_LIMIT)))


def _step(p, i, o, h, l, c, first, move_to_be):
    done = [(TP1_PART, p["t1"])] if p["half"] else []
    rem = 1 - TP1_PART if p["half"] else 1.0
    if l[i] <= p["sl"]:                                  # stop first
        px = p["sl"] if first else min(o[i], p["sl"])
        return done + [(rem, px)], "SL" if not p["half"] else "SL2"
    if not p["half"] and h[i] >= p["t1"]:
        p["half"] = True
        p["t1"] = p["t1"] if first else max(o[i], p["t1"])
        if move_to_be:
            p["sl"] = max(p["sl"], p["entry"])
        done = [(TP1_PART, p["t1"])]
        rem = 1 - TP1_PART
    if p["half"] and h[i] >= p["t2"]:
        px = p["t2"] if first else max(o[i], p["t2"])
        return done + [(rem, px)], "T2"
    if p["days"] >= MAX_BARS:
        return done + [(rem, c[i])], "TIME"
    return None


def simulate(sym, S, first_i, move_to_be):
    o, h, l, c, dates = S["o"], S["h"], S["l"], S["c"], S["dates"]
    n = len(c)
    out = []
    busy_until = -1
    for i in range(max(first_i, 80), n - 1):
        if i <= busy_until:
            continue
        pl = plan_at(S, i)
        if pl is None:
            continue
        flags = pl["flags"]
        if pl["entry"] is None:
            continue
        j = i + 1                                   # order valid next session
        E = pl["entry"]
        if o[j] >= E:
            if o[j] > pl["skip"]:
                continue
            fill = o[j]
        elif h[j] >= E:
            fill = E
        else:
            continue
        pos = dict(entry=fill, sl0=pl["sl"], sl=pl["sl"],
                   t1=pl["t1"], t2=pl["t2"], days=1, half=False)
        legs = why = None
        k = j
        r = _step(pos, j, o, h, l, c, True, move_to_be)
        while r is None and k < n - 1:
            k += 1
            pos["days"] += 1
            r = _step(pos, k, o, h, l, c, False, move_to_be)
        if r is None:
            rem = 1 - TP1_PART if pos["half"] else 1.0
            legs = ([(TP1_PART, pos["t1"])] if pos["half"] else []) + [(rem, c[n - 1])]
            why = "OPEN"
        else:
            legs, why = r
        e = fill * (1 + sb.SLIP_PCT)
        q = pl["qty"]
        risk = fill - pl["sl"]
        gross = sum(f * (px * (1 - sb.SLIP_PCT) - e) for f, px in legs) * q
        cost = sb.COST_PCT * e * q + sb.DP_CHARGE * len(legs)
        pnl = gross - cost
        if risk <= 0:
            risk = E - pl["sl"]
        out.append(dict(sym=sym, sig_date=dates[i], entry_date=dates[j],
                        status=pl["status"], flags=flags, why=why, pnl=pnl,
                        Rn=pnl / (q * risk), risk_pct=pl["risk_pct"],
                        value=e * q))
        busy_until = k
    return out


def stat(rows):
    if not rows:
        return None
    r = np.array([x["Rn"] for x in rows])
    pos, neg = r[r > 0].sum(), -r[r <= 0].sum()
    return dict(n=len(r), win=(r > 0).mean() * 100, R=r.mean(),
                pf=(pos / neg) if neg > 0 else float("inf"),
                pnl=sum(x["pnl"] for x in rows))


def line(lab, rows, mid):
    s = stat(rows)
    if not s:
        print(f"  {lab:<14}     0")
        return
    a = stat([t for t in rows if t["entry_date"] <= mid])
    b = stat([t for t in rows if t["entry_date"] > mid])
    fa = f"{a['n']:>4} {a['R']:>5.2f}" if a else "   0    --"
    fb = f"{b['n']:>4} {b['R']:>5.2f}" if b else "   0    --"
    ok = (a and b and a["n"] >= MIN_N_HALF and b["n"] >= MIN_N_HALF
          and a["R"] > 0 and b["R"] > 0)
    print(f"  {lab:<14} {s['n']:>5} {s['win']:>4.0f}% {s['R']:>6.2f} "
          f"{sb.fmt_pf(s['pf']):>5} | {fa} | {fb} | {'YES' if ok else ''}")


ND = 1


def report(name, rows, mid):
    print(f"\n=== {name} ===")
    print("  group             N  win%   NetR    PF |  H1 N  H1 R |  H2 N  H2 R | HOLDS")
    line("ALL", rows, mid)
    line("ACTIVE", [t for t in rows if t["status"] == "ACTIVE"], mid)
    line("RISKY", [t for t in rows if t["status"] == "RISKY"], mid)
    for f in ["WICK", "WEAK CLOSE", "CLIMAX", "EXTENDED", "3 GREEN",
              "CHASE", "ST DOWN"]:
        line("flag " + f, [t for t in rows if f in t["flags"]], mid)
    for lab, lo, hi in [("risk <3%", 0, 3), ("risk 3-6%", 3, 6),
                        ("risk 6-10%", 6, 10.01)]:
        line(lab, [t for t in rows if lo <= t["risk_pct"] < hi], mid)
    s = stat(rows)
    print(f"  total P&L at Rs1000 risk/trade: Rs {s['pnl']:.0f}  "
          f"avg per trade Rs {s['pnl'] / s['n']:.0f}  signals/day "
          f"{s['n'] / ND:.1f}  avg position Rs "
          f"{np.mean([t['value'] for t in rows]):.0f}")
    ex = defaultdict(int)
    for t in rows:
        ex[t["why"]] += 1
    print("  exits:", dict(ex))
    months = defaultdict(list)
    for t in rows:
        months[t["entry_date"].strftime("%Y-%m")].append(t["Rn"])
    print("  by month (N/NetR):", "  ".join(
        f"{m}:{len(v)}/{np.mean(v):.2f}" for m, v in sorted(months.items())))


def main():
    global ND
    df = pd.read_csv(sb.FILE)
    df.columns = df.columns.str.strip()
    df = df[df["SERIES"].str.strip() == "EQ"]
    symbols = df["SYMBOL"].str.strip().str.upper().tolist()
    data = sb.download_all(symbols)
    print(f"Price data for {len(data)} symbols", flush=True)
    if not data:
        sys.exit("No data downloaded.")
    end = max(d.index[-1] for d in data.values())
    start = end - pd.Timedelta(days=sb.TEST_DAYS)
    mid = start + (end - start) / 2
    print(f"Window {start:%d-%b-%Y} to {end:%d-%b-%Y}; H1 up to "
          f"{mid:%d-%b-%Y}, H2 after", flush=True)
    ND = len(set().union(*[set(d.index[d.index >= start])
                           for d in data.values()]))
    A, B = [], []
    for s, d in data.items():
        if len(d) < 90:
            continue
        S = prep_sig(d)
        first_i = max(80, int(np.searchsorted(d.index.values,
                                              np.datetime64(start))))
        if first_i >= len(d) - 1:
            continue
        A += simulate(s, S, first_i, False)
        B += simulate(s, S, first_i, True)
    print(f"{len(A)} trades (plan A), {len(B)} (plan B)")
    report("PLAN A: half at T1, half at T2, stop stays", A, mid)
    report("PLAN B: half at T1, half at T2, stop to entry after T1", B, mid)


if __name__ == "__main__":
    main()

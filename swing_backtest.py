import csv, io, sys, subprocess
import datetime as dt
from collections import Counter
from zoneinfo import ZoneInfo
import pandas as pd

IST = ZoneInfo("Asia/Kolkata")

# ---------- CONFIG ----------
CAPITAL = 100000
CSV_FILE = "results_breakout.csv"
ENTRY_DAYS = 1            # sessions the buy-stop stays valid after signal
HOLD_LIST = [5, 10, 20]   # max trading days to hold
DEFAULT_HOLD = 10
DEFAULT_MODE = "HALF_T1_HALF_T2"
SLIP_PCT = 0.0005         # 0.05% each side
COST_PCT = 0.0025         # rough delivery costs, of entry value
DP_CHARGE = 15            # per sell leg, rupees
MIN_SCAN_TIME = dt.time(15, 45)   # drop scans made before candle closed
MODES = {
    "ALL_T1": [(1.0, "t1")],
    "ALL_T2": [(1.0, "t2")],
    "HALF_T1_HALF_T2": [(0.5, "t1"), (0.5, "t2")],
}


def git(*args):
    return subprocess.run(["git", *args], capture_output=True,
                          text=True, check=True).stdout


# ---------- SIGNALS FROM GIT HISTORY ----------
def load_signals():
    log = git("log", "--format=%H|%cI", "--", CSV_FILE).strip().splitlines()
    print(f"{len(log)} commits touch {CSV_FILE}")
    seen = {}                       # (date, symbol) -> row, newest wins
    for line in log:                # newest commit first
        h, cdate = line.split("|", 1)
        ct = dt.datetime.fromisoformat(cdate).astimezone(IST)
        try:
            txt = git("show", f"{h}:{CSV_FILE}")
        except subprocess.CalledProcessError:
            continue
        rows = list(csv.DictReader(io.StringIO(txt)))
        if not rows or "status" not in rows[0] or "skip_above" not in rows[0]:
            continue
        for r in rows:
            try:
                sd = dt.datetime.strptime(r["date"].strip(), "%d-%b-%y").date()
            except ValueError:
                continue
            key = (sd, r["symbol"].strip())
            if key in seen:
                continue
            if sd == ct.date() and ct.time() < MIN_SCAN_TIME:
                continue            # scan ran before the candle closed
            seen[key] = r
    sigs = []
    for (sd, sym), r in seen.items():
        if r["status"].strip() != "ACTIVE":
            continue
        try:
            sigs.append(dict(sym=sym, date=sd,
                             entry=float(r["entry"]), sl=float(r["sl"]),
                             t1=float(r["t1"]), t2=float(r["t2"]),
                             skip=float(r["skip_above"]),
                             qty=int(float(r["qty"]))))
        except ValueError:
            continue
    return sorted(sigs, key=lambda s: (s["date"], s["sym"]))


# ---------- PRICE DATA ----------
def download(symbols, start):
    import yfinance as yf
    out = {}
    syms = sorted(symbols)
    for i in range(0, len(syms), 100):
        chunk = syms[i:i + 100]
        tk = [s + ".NS" for s in chunk]
        data = yf.download(tk, start=start, interval="1d",
                           auto_adjust=False, group_by="ticker",
                           threads=True, progress=False)
        for s, t in zip(chunk, tk):
            try:
                try:
                    d = data[t]
                except KeyError:
                    d = data
                d = d.dropna(subset=["Open", "High", "Low", "Close"]).copy()
                if d.empty:
                    continue
                d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
                out[s] = d
            except Exception:
                continue
    return out


# ---------- ONE TRADE ----------
def simulate(sg, bars, legs, hold):
    E, S, K = sg["entry"], sg["sl"], sg["skip"]
    targets = {"t1": sg["t1"], "t2": sg["t2"]}
    start = None
    for i, (d, o, h, l, c) in enumerate(bars[:ENTRY_DAYS]):
        if i == 0 and not (0.75 * E <= o <= 1.25 * E):
            return dict(status="PRICE MISMATCH")
        if o > K:
            return dict(status="GAP SKIP")
        if h >= E:
            start = i
            break
    if start is None:
        return dict(status="NO TRIGGER")

    ed, eo = bars[start][0], bars[start][1]
    entry_px = max(E, eo) * (1 + SLIP_PCT)
    rem, sl_cur = 1.0, S
    pending = list(legs)
    exits = []
    day = 0
    for j in range(start, len(bars)):
        d, o, h, l, c = bars[j]
        first = (j == start)
        day += 1
        if l <= sl_cur:             # stop first: worst case inside a day
            px = sl_cur if first else min(o, sl_cur)
            why = "SL" if sl_cur == S else "BE"
            exits.append((rem, px * (1 - SLIP_PCT), why, d))
            rem = 0.0
            break
        for leg in list(pending):
            frac, key = leg
            tp = targets[key]
            if h >= tp:
                px = tp if first else max(tp, o)
                exits.append((frac, px * (1 - SLIP_PCT), key.upper(), d))
                rem -= frac
                pending.remove(leg)
                if key == "t1" and len(legs) > 1:
                    sl_cur = max(sl_cur, max(E, eo))   # breakeven stop
        if rem <= 1e-9:
            rem = 0.0
            break
        if day >= hold:
            exits.append((rem, c * (1 - SLIP_PCT), "TIME", d))
            rem = 0.0
            break
    if rem > 1e-9:
        d, o, h, l, c = bars[-1]
        exits.append((rem, c * (1 - SLIP_PCT), "OPEN", d))

    qty = sg["qty"]
    per_share = sum(f * (px - entry_px) for f, px, _, _ in exits)
    cost = COST_PCT * entry_px * qty + DP_CHARGE * len(exits)
    return dict(status="TRADE", sym=sg["sym"], sig_date=sg["date"],
                entry_date=ed, entry_px=entry_px,
                exit_date=exits[-1][3], exits=exits, days=day,
                R=per_share / (entry_px - S), pnl=per_share * qty - cost,
                cost_basis=entry_px * qty, is_open=exits[-1][2] == "OPEN")


# ---------- STATS ----------
def stats(trades):
    n = len(trades)
    if n == 0:
        return None
    wins = [t for t in trades if t["pnl"] > 0]
    gp = sum(t["pnl"] for t in wins)
    gl = -sum(t["pnl"] for t in trades if t["pnl"] <= 0)
    cum = peak = dd = 0.0
    streak = worst = 0
    for t in sorted(trades, key=lambda t: t["exit_date"]):
        cum += t["pnl"]
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
        streak = streak + 1 if t["pnl"] <= 0 else 0
        worst = max(worst, streak)
    return dict(n=n, win=len(wins) / n * 100,
                avgR=sum(t["R"] for t in trades) / n,
                pf=(gp / gl) if gl > 0 else float("inf"),
                net=sum(t["pnl"] for t in trades), dd=dd, streak=worst)


def peak_invested(trades):
    ev = []
    for t in trades:
        ev.append((t["entry_date"], 0, t["cost_basis"]))
        ev.append((t["exit_date"], 1, -t["cost_basis"]))
    ev.sort(key=lambda x: (x[0], x[1]))
    cur = pk = 0.0
    for _, _, v in ev:
        cur += v
        pk = max(pk, cur)
    return pk


def main():
    if git("rev-parse", "--is-shallow-repository").strip() == "true":
        print("WARNING: shallow clone. Set fetch-depth: 0 in the workflow.")
    sigs = load_signals()
    if not sigs:
        sys.exit("No ACTIVE signals found in history.")
    days = sorted({s["date"] for s in sigs})
    print(f"{len(sigs)} ACTIVE signals on {len(days)} days: "
          f"{days[0]} to {days[-1]}")

    data = download({s["sym"] for s in sigs}, days[0].isoformat())
    print(f"Prices loaded for {len(data)} symbols")

    prepared, nodata, notyet = [], 0, 0
    for s in sigs:
        d = data.get(s["sym"])
        if d is None:
            nodata += 1
            continue
        sub = d.loc[d.index > pd.Timestamp(s["date"])]
        if sub.empty:
            notyet += 1
            continue
        bars = list(zip(sub.index.date, sub["Open"], sub["High"],
                        sub["Low"], sub["Close"]))
        prepared.append((s, bars))
    print(f"Testable {len(prepared)}  no data {nodata}  "
          f"no session yet {notyet}")

    print("\nMODE             HOLD  N   WIN%   AvgR    PF     NET   MAXDD")
    for mode, legs in MODES.items():
        for hold in HOLD_LIST:
            res = [simulate(s, b, legs, hold) for s, b in prepared]
            st = stats([r for r in res if r["status"] == "TRADE"])
            if not st:
                print(f"{mode:<16} {hold:>3}d  no trades")
                continue
            pf = f"{st['pf']:.2f}" if st["pf"] != float("inf") else "inf"
            print(f"{mode:<16} {hold:>3}d {st['n']:>3} {st['win']:>5.0f}%"
                  f" {st['avgR']:>6.2f} {pf:>5} {st['net']:>7.0f}"
                  f" {st['dd']:>7.0f}")

    legs = MODES[DEFAULT_MODE]
    res = [simulate(s, b, legs, DEFAULT_HOLD) for s, b in prepared]
    trades = [r for r in res if r["status"] == "TRADE"]
    print(f"\nDetail: {DEFAULT_MODE}, hold {DEFAULT_HOLD} days")
    print("Not traded:", dict(Counter(r["status"] for r in res
                                      if r["status"] != "TRADE")))
    if not trades:
        return
    final = Counter(t["exits"][-1][2] for t in trades)
    print("Last exit reasons:", dict(final))
    print(f"Peak money in open trades: Rs {peak_invested(trades):.0f}"
          f" (capital {CAPITAL})")
    if any(t["is_open"] for t in trades):
        print("Some trades are still open, marked at the last close.")
    shown = sorted(trades, key=lambda t: t["entry_date"])[-40:]
    print(f"\nLast {len(shown)} trades:")
    for t in shown:
        why = "+".join(x[2] for x in t["exits"])
        print(f"{t['sym']:<10} {t['entry_date']:%d%b} {t['entry_px']:8.2f}"
              f" {t['days']:>2}d {why:<9} R {t['R']:>5.2f} "
              f"Rs {t['pnl']:>6.0f}")


if __name__ == "__main__":
    main()

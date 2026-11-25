import sys
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

# ---------- CONFIG ----------
FILE = "EQUITY_L.csv"
CAPITAL = 100000
TARGET_PROFIT = 2000       # rupees at target, sets the quantity
LOOKBACK = 20              # resistance = highest high of previous 20 days
EMA_SPAN = 50
NEAR_PCT = 0.01            # enter if open is within 1% above resistance
MIN_PRICE = 20
MIN_AVG_VOL = 100000
MIN_RISK_PCT = 0.5         # skip if stop is closer than this % of entry
MIN_QTY_FRAC = 0.5         # skip if cash allows less than half the quantity
ARM_DAYS = 20              # a crossed resistance stays valid this long
MAX_HOLD = 40              # trading days
TEST_DAYS = 365            # backtest window in calendar days
SLIP_PCT = 0.0005          # 0.05% each side
COST_PCT = 0.0025          # delivery costs, share of entry value
DP_CHARGE = 15             # per sell, rupees
# target price rule: R = multiple of risk, P = fixed percent above entry
VARIANTS = {"R1.5": ("R", 1.5), "R2": ("R", 2.0),
            "R3": ("R", 3.0), "P10": ("P", 0.10)}
DEFAULT_VARIANT = "R2"


# ---------- DATA ----------
def extract(data, t):
    try:
        d = data[t]
    except KeyError:
        return None
    try:
        d = d.dropna(subset=["Open", "High", "Low", "Close"]).copy()
        d = d[(d["Low"] > 0) & (d["High"] >= d["Low"])]
        if len(d) < 120:
            return None
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        d["Volume"] = d["Volume"].fillna(0)
        return d
    except Exception:
        return None


def download_all(symbols):
    import yfinance as yf
    out = {}
    for i in range(0, len(symbols), 100):
        chunk = symbols[i:i + 100]
        tk = [s + ".NS" for s in chunk]
        try:
            data = yf.download(tk, period="2y", interval="1d",
                               group_by="ticker", auto_adjust=True,
                               threads=True, progress=False)
        except Exception as e:
            print("chunk failed", i, e, flush=True)
            continue
        for s, t in zip(chunk, tk):
            d = extract(data, t)
            if d is not None:
                out[s] = d
        print(f"downloaded {min(i + 100, len(symbols))}/{len(symbols)}",
              flush=True)
    missing = [s for s in symbols if s not in out]
    print(f"missing after first pass: {len(missing)}, retrying up to 200",
          flush=True)
    for s in missing[:200]:
        try:
            data = yf.download([s + ".NS"], period="2y", interval="1d",
                               group_by="ticker", auto_adjust=True,
                               threads=False, progress=False)
            d = extract(data, s + ".NS")
            if d is not None:
                out[s] = d
        except Exception:
            pass
    return out


# ---------- INDICATORS ----------
def prep(d):
    o = d["Open"].to_numpy(float)
    h = d["High"].to_numpy(float)
    l = d["Low"].to_numpy(float)
    c = d["Close"].to_numpy(float)
    v = d["Volume"].to_numpy(float)
    n = len(c)
    ema = pd.Series(c).ewm(span=EMA_SPAN, adjust=False).mean().to_numpy()
    avgv = pd.Series(v).shift(1).rolling(20).mean().to_numpy()
    res = np.full(n, np.nan)
    resl = np.full(n, np.nan)
    if n > LOOKBACK:
        sw = np.lib.stride_tricks.sliding_window_view(h, LOOKBACK)[:-1]
        pos = LOOKBACK - 1 - np.argmax(sw[:, ::-1], axis=1)
        j = np.arange(len(sw)) + pos        # candle that made the high
        res[LOOKBACK:] = h[j]
        resl[LOOKBACK:] = l[j]
    return dict(o=o, h=h, l=l, c=c, ema=ema, avgv=avgv, res=res,
                resl=resl, dates=d.index)


# ---------- ONE STOCK ----------
def simulate(sym, P, kind, param, first_i, stats):
    o, h, l, c = P["o"], P["h"], P["l"], P["c"]
    ema, avgv, res, resl, dates = (P["ema"], P["avgv"], P["res"],
                                   P["resl"], P["dates"])
    n = len(c)
    trades = []

    def check_exit(p, i, first):
        if l[i] <= p["sl"]:                 # stop first inside a day
            return (p["sl"] if first else min(o[i], p["sl"])), "SL"
        if h[i] >= p["tp"]:
            return (p["tp"] if first else max(o[i], p["tp"])), "TARGET"
        if p["days"] >= MAX_HOLD:
            return c[i], "TIME"
        return None

    def close(p, i, px, why):
        e_eff = p["entry"] * (1 + SLIP_PCT)
        x_eff = px * (1 - SLIP_PCT)
        risk = p["entry"] - p["sl"]
        q_t = max(1, int(round(TARGET_PROFIT / (p["tp"] - p["entry"]))))
        trades.append(dict(sym=sym, entry_date=dates[p["ei"]],
                           exit_date=dates[i], entry=p["entry"], exit=px,
                           sl=p["sl"], tp=p["tp"], reason=why,
                           days=p["days"], risk_pct=risk / p["entry"] * 100,
                           entry_eff=e_eff, exit_eff=x_eff,
                           R=(x_eff - e_eff) / risk, qty_t=q_t))

    armed = None
    pos = None
    for i in range(first_i, n):
        if pos is not None:
            pos["days"] += 1
            r = check_exit(pos, i, False)
            if r:
                close(pos, i, r[0], r[1])
                pos = None
                armed = None
            continue
        if armed is not None and (c[i - 1] < armed[1]
                                  or i - armed[2] > ARM_DAYS):
            armed = None
        r_i = res[i]
        if (r_i == r_i and h[i] >= r_i and res[i - 1] == res[i - 1]
                and c[i - 1] <= res[i - 1]):
            armed = (r_i, resl[i], i)       # fresh break of resistance
        if armed is None:
            continue
        if not (c[i - 1] > ema[i - 1] and c[i - 1] >= MIN_PRICE
                and avgv[i] >= MIN_AVG_VOL):
            continue
        R, L, _ = armed
        if o[i] >= R:
            fill = o[i] if o[i] <= R * (1 + NEAR_PCT) else None
        elif h[i] >= R:
            fill = R
        else:
            fill = None
        if fill is None:
            continue
        risk = fill - L
        if risk <= 0 or risk / fill * 100 < MIN_RISK_PCT:
            stats["tiny_risk"] += 1
            armed = None
            continue
        tp = fill + param * risk if kind == "R" else fill * (1 + param)
        pos = dict(entry=fill, sl=L, tp=tp, days=1, ei=i)
        armed = None
        r = check_exit(pos, i, True)
        if r:
            close(pos, i, r[0], r[1])
            pos = None
    if pos is not None:
        close(pos, n - 1, c[n - 1], "OPEN")
    return trades


# ---------- PORTFOLIO WITH 1 LAKH ----------
def cost_of(t, q):
    return COST_PCT * t["entry_eff"] * q + DP_CHARGE


def portfolio(trades, closes, cal):
    by_entry = defaultdict(list)
    for t in trades:
        by_entry[t["entry_date"]].append(t)
    for k in by_entry:
        by_entry[k].sort(key=lambda t: t["sym"])
    cash = CAPITAL
    open_, taken, curve = [], [], []
    skipped = peak_pos = 0
    for d in cal:
        for t in by_entry.get(d, []):
            q = min(t["qty_t"], int(cash // t["entry_eff"]))
            if q < 1 or q < MIN_QTY_FRAC * t["qty_t"]:
                skipped += 1
                continue
            cash -= q * t["entry_eff"]
            open_.append(dict(t=t, q=q))
        keep = []
        for p in open_:
            t, q = p["t"], p["q"]
            if t["exit_date"] == d:
                cost = cost_of(t, q)
                cash += q * t["exit_eff"] - cost
                taken.append(dict(t, qty=q, pnl=(t["exit_eff"]
                                  - t["entry_eff"]) * q - cost))
            else:
                keep.append(p)
        open_ = keep
        peak_pos = max(peak_pos, len(open_))
        mtm = sum(p["q"] * closes[p["t"]["sym"]].asof(d) for p in open_)
        curve.append((d, cash + mtm))
    return dict(taken=taken, skipped=skipped, peak_pos=peak_pos,
                curve=curve)


# ---------- STATS ----------
def summarize(rows):
    n = len(rows)
    if n == 0:
        return None
    wins = [r for r in rows if r["pnl"] > 0]
    gp = sum(r["pnl"] for r in wins)
    gl = -sum(r["pnl"] for r in rows if r["pnl"] <= 0)
    streak = worst = 0
    for r in sorted(rows, key=lambda r: r["exit_date"]):
        streak = streak + 1 if r["pnl"] <= 0 else 0
        worst = max(worst, streak)
    return dict(n=n, win=len(wins) / n * 100,
                avgR=sum(r["R"] for r in rows) / n,
                pf=(gp / gl) if gl > 0 else float("inf"),
                net=sum(r["pnl"] for r in rows), streak=worst)


def with_unlimited_pnl(trades):
    out = []
    for t in trades:
        q = t["qty_t"]
        out.append(dict(t, qty=q, pnl=(t["exit_eff"] - t["entry_eff"]) * q
                        - cost_of(t, q)))
    return out


def peak_money(rows):
    ev = []
    for r in rows:
        v = r["qty"] * r["entry_eff"]
        ev.append((r["entry_date"], 0, v))
        ev.append((r["exit_date"], 1, -v))
    ev.sort(key=lambda x: (x[0], x[1]))
    cur = pk = 0.0
    for _, _, v in ev:
        cur += v
        pk = max(pk, cur)
    return pk


def max_dd(curve):
    peak = dd = 0.0
    for _, e in curve:
        peak = max(peak, e)
        dd = max(dd, peak - e)
    return dd


def fmt_pf(x):
    return "inf" if x == float("inf") else f"{x:.2f}"


def main():
    df = pd.read_csv(FILE)
    df.columns = df.columns.str.strip()
    df = df[df["SERIES"].str.strip() == "EQ"]
    symbols = df["SYMBOL"].str.strip().str.upper().tolist()
    print(f"{len(symbols)} EQ symbols in {FILE}", flush=True)

    data = download_all(symbols)
    print(f"Price data for {len(data)} symbols", flush=True)
    if not data:
        sys.exit("No data downloaded.")

    end = max(d.index[-1] for d in data.values())
    start = end - pd.Timedelta(days=TEST_DAYS)
    print(f"Test window {start:%d-%b-%Y} to {end:%d-%b-%Y}", flush=True)

    cal = sorted(set().union(*[set(d.index[d.index >= start])
                               for d in data.values()]))
    closes = {s: d["Close"] for s, d in data.items()}
    all_trades = {k: [] for k in VARIANTS}
    stats = Counter()
    for s, d in data.items():
        P = prep(d)
        first_i = max(60, int(np.searchsorted(d.index.values,
                                              np.datetime64(start))))
        if first_i >= len(d) - 1:
            continue
        for name, (kind, param) in VARIANTS.items():
            if name == DEFAULT_VARIANT:
                all_trades[name] += simulate(s, P, kind, param, first_i,
                                             stats)
            else:
                all_trades[name] += simulate(s, P, kind, param, first_i,
                                             Counter())
    print(f"Skipped for tiny stop (<{MIN_RISK_PCT}%): "
          f"{stats['tiny_risk']}")

    print("\nTARGET  TRADES WIN%  AvgR   PF   NETnocap | "
          "TAKEN  NET1L   RET%  MAXDD  SKIP")
    detail = None
    for name in VARIANTS:
        tr = all_trades[name]
        unl = with_unlimited_pnl(tr)
        su = summarize(unl)
        if not su:
            print(f"{name:<6} no trades")
            continue
        pf = portfolio(tr, closes, cal)
        sc = summarize(pf["taken"])
        net_cap = pf["curve"][-1][1] - CAPITAL
        print(f"{name:<6} {su['n']:>6} {su['win']:>5.0f} {su['avgR']:>6.2f}"
              f" {fmt_pf(su['pf']):>5} {su['net']:>8.0f} | "
              f"{sc['n'] if sc else 0:>5} {net_cap:>7.0f}"
              f" {net_cap / CAPITAL * 100:>6.1f} {max_dd(pf['curve']):>6.0f}"
              f" {pf['skipped']:>5}")
        if name == DEFAULT_VARIANT:
            detail = (tr, unl, su, pf, sc)

    if not detail:
        return
    tr, unl, su, pf, sc = detail
    print(f"\n=== Detail: {DEFAULT_VARIANT} ===")
    print("Needed if every trade were taken: peak money "
          f"Rs {peak_money(unl):.0f} (you have {CAPITAL})")
    print(f"With Rs {CAPITAL}: peak open positions {pf['peak_pos']}, "
          f"signals skipped for cash {pf['skipped']}")
    if sc:
        print(f"Taken: {sc['n']}  win {sc['win']:.0f}%  PF "
              f"{fmt_pf(sc['pf'])}  worst losing streak {sc['streak']}")
    print("Exit reasons (all signals):",
          dict(Counter(t["reason"] for t in tr)))
    rp = sorted(t["risk_pct"] for t in tr)
    print(f"Stop distance: median {rp[len(rp) // 2]:.1f}%  "
          f"over 10%: {sum(x > 10 for x in rp)} of {len(rp)}")
    print(f"Average days held: "
          f"{sum(t['days'] for t in tr) / len(tr):.1f}")
    print("Open at end of data (marked at last close):",
          sum(t["reason"] == "OPEN" for t in tr))

    months = defaultdict(float)
    for r in pf["taken"]:
        months[r["exit_date"].strftime("%Y-%m")] += r["pnl"]
    print("\nMonthly P&L with Rs 1 lakh (by exit month):")
    for m in sorted(months):
        print(f"  {m}  {months[m]:>8.0f}")

    ranked = sorted(pf["taken"], key=lambda r: r["pnl"])
    print("\nWorst 5:")
    for r in ranked[:5]:
        print(f"  {r['sym']:<10} {r['entry_date']:%d%b%y} x{r['qty']:<4}"
              f" {r['reason']:<6} Rs {r['pnl']:>7.0f}")
    print("Best 5:")
    for r in ranked[-5:][::-1]:
        print(f"  {r['sym']:<10} {r['entry_date']:%d%b%y} x{r['qty']:<4}"
              f" {r['reason']:<6} Rs {r['pnl']:>7.0f}")


if __name__ == "__main__":
    main()

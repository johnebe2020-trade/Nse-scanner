import sys
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import simple_backtest as sb

# ---------- CONFIG ----------
TARGET = ("R", 0.5)         # target rule used in this study
MIN_N_HALF = 100            # trades needed in EACH half for HOLDS = YES
# Filters for the FILTERED run at the bottom. Leave empty for none.
# Example: {"nifty": "UP", "slope": "RISING", "fresh": "FRESH"}
USE_FILTERS = {}

FEATURES = [("fresh", "entry type (fresh break vs already crossed)"),
            ("fill", "fill (buy-stop at resistance vs at the open)"),
            ("risk", "stop distance"),
            ("age", "age of the resistance candle"),
            ("vol", "yesterday volume vs 20-day average"),
            ("xage", "days since close was last below EMA50"),
            ("slope", "EMA50 trend (vs 10 days ago)"),
            ("ext", "close above EMA50"),
            ("nifty", "Nifty above its own EMA50")]


def bucket(v, edges, labels):
    for e, lab in zip(edges, labels):
        if v < e:
            return lab
    return labels[-1]


def prep2(d):
    P = sb.prep(d)
    h = P["h"]
    n = len(h)
    resj = np.full(n, -1)
    if n > sb.LOOKBACK:
        sw = np.lib.stride_tricks.sliding_window_view(h, sb.LOOKBACK)[:-1]
        pos = sb.LOOKBACK - 1 - np.argmax(sw[:, ::-1], axis=1)
        resj[sb.LOOKBACK:] = np.arange(len(sw)) + pos
    P["resj"] = resj
    P["v"] = d["Volume"].to_numpy(float)
    idx = np.where(P["c"] <= P["ema"], np.arange(n), -1)
    P["lb"] = np.maximum.accumulate(idx)       # last bar closed at/below EMA50
    return P


def nifty_map():
    import yfinance as yf
    try:
        d = yf.download("^NSEI", period="2y", interval="1d",
                        auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        c = d["Close"].dropna()
        c.index = pd.to_datetime(c.index).tz_localize(None).normalize()
        ema = c.ewm(span=50, adjust=False).mean()
        return {k: bool(v) for k, v in (c > ema).items()}
    except Exception as e:
        print("Nifty download failed:", e)
        return {}


def simulate_ft(sym, P, kind, param, first_i, nmap):
    """Same rules as simple_backtest.simulate, plus features at entry."""
    o, h, l, c, v = P["o"], P["h"], P["l"], P["c"], P["v"]
    ema, avgv, res, resl, resj, dates = (P["ema"], P["avgv"], P["res"],
                                         P["resl"], P["resj"], P["dates"])
    lb = P["lb"]
    n = len(c)
    trades = []

    def check_exit(p, i, first):
        if l[i] <= p["sl"]:
            return (p["sl"] if first else min(o[i], p["sl"])), "SL"
        if h[i] >= p["tp"]:
            return (p["tp"] if first else max(o[i], p["tp"])), "TARGET"
        if p["days"] >= sb.MAX_HOLD:
            return c[i], "TIME"
        return None

    def close(p, i, px, why):
        e_eff = p["entry"] * (1 + sb.SLIP_PCT)
        x_eff = px * (1 - sb.SLIP_PCT)
        risk = p["entry"] - p["sl"]
        q_t = max(1, int(round(sb.TARGET_PROFIT / (p["tp"] - p["entry"]))))
        t = dict(sym=sym, entry_date=dates[p["ei"]], exit_date=dates[i],
                 entry_eff=e_eff, exit_eff=x_eff, qty=q_t, reason=why,
                 days=p["days"], risk_pct=risk / p["entry"] * 100,
                 feat=p["feat"])
        t["pnl"] = (x_eff - e_eff) * q_t - sb.cost_of(t, q_t)
        t["Rn"] = t["pnl"] / (q_t * risk)       # R after ALL costs
        trades.append(t)

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
                                  or i - armed[2] > sb.ARM_DAYS):
            armed = None
        r_i = res[i]
        if (r_i == r_i and h[i] >= r_i and res[i - 1] == res[i - 1]
                and c[i - 1] <= res[i - 1]):
            armed = (r_i, resl[i], i, resj[i])
        if armed is None:
            continue
        if not (c[i - 1] > ema[i - 1] and c[i - 1] >= sb.MIN_PRICE
                and avgv[i] >= sb.MIN_AVG_VOL):
            continue
        R, L, d0, j = armed
        if o[i] >= R:
            fill = o[i] if o[i] <= R * (1 + sb.NEAR_PCT) else None
        elif h[i] >= R:
            fill = R
        else:
            fill = None
        if fill is None:
            continue
        risk = fill - L
        if risk <= 0 or risk / fill * 100 < sb.MIN_RISK_PCT:
            armed = None
            continue
        tp = fill + param * risk if kind == "R" else fill * (1 + param)
        vp = v[i - 1] / avgv[i - 1] if avgv[i - 1] and avgv[i - 1] > 0 \
            else 1.0
        nf = nmap.get(dates[i - 1])
        feat = dict(
            fresh="FRESH" if i == d0 else "LATE",
            fill="STOP" if o[i] < R else "OPEN",
            risk=bucket(risk / fill * 100, [3, 6, 10],
                        ["<3%", "3-6%", "6-10%", ">10%"]),
            age=bucket(i - j, [6, 13], ["1-5d", "6-12d", "13-20d"]),
            vol=bucket(vp, [1.0, 1.5, 3.0],
                       ["<1x", "1-1.5x", "1.5-3x", ">3x"]),
            xage=bucket((i - 1) - lb[i - 1], [6, 16, 41],
                        ["a:<=5d", "b:6-15d", "c:16-40d", "d:>40d"]),
            slope="RISING" if ema[i - 1] > ema[i - 11] else "NOT RISING",
            ext=bucket((c[i - 1] / ema[i - 1] - 1) * 100, [3, 8, 15],
                       ["<3%", "3-8%", "8-15%", ">15%"]),
            nifty="NA" if nf is None else ("UP" if nf else "DOWN"))
        pos = dict(entry=fill, sl=L, tp=tp, days=1, ei=i, feat=feat)
        armed = None
        r = check_exit(pos, i, True)
        if r:
            close(pos, i, r[0], r[1])
            pos = None
    if pos is not None:
        close(pos, n - 1, c[n - 1], "OPEN")
    return trades


# ---------- STATS ----------
def stat(rows):
    if not rows:
        return None
    r = np.array([x["Rn"] for x in rows])
    pos, neg = r[r > 0].sum(), -r[r <= 0].sum()
    return dict(n=len(r), win=(r > 0).mean() * 100, R=r.mean(),
                pf=(pos / neg) if neg > 0 else float("inf"))


def cell(s):
    return f"{s['n']:>5} {s['R']:>6.2f}" if s else f"{0:>5}    -- "


def table(rows, mid, key, title):
    print(f"\n{title}")
    print("  bucket        N   WIN%  NetR    PF |  H1 N   H1 R |  H2 N"
          "   H2 R | HOLDS")
    groups = defaultdict(list)
    for t in rows:
        groups[t["feat"][key]].append(t)
    for lab in sorted(groups):
        g = groups[lab]
        s = stat(g)
        a = stat([t for t in g if t["entry_date"] <= mid])
        b = stat([t for t in g if t["entry_date"] > mid])
        ok = (a and b and a["n"] >= MIN_N_HALF and b["n"] >= MIN_N_HALF
              and a["R"] > 0 and b["R"] > 0)
        print(f"  {lab:<12} {s['n']:>5} {s['win']:>5.0f} {s['R']:>6.2f}"
              f" {sb.fmt_pf(s['pf']):>5} | {cell(a)} | {cell(b)} |"
              f" {'YES' if ok else ''}")


def passes(t):
    for k, want in USE_FILTERS.items():
        ok = want if isinstance(want, (list, tuple, set)) else [want]
        if t["feat"][k] not in ok:
            return False
    return True


def main():
    df = pd.read_csv(sb.FILE)
    df.columns = df.columns.str.strip()
    df = df[df["SERIES"].str.strip() == "EQ"]
    symbols = df["SYMBOL"].str.strip().str.upper().tolist()
    data = sb.download_all(symbols)
    print(f"Price data for {len(data)} symbols", flush=True)
    if not data:
        sys.exit("No data downloaded.")
    nmap = nifty_map()
    print(f"Nifty days loaded: {len(nmap)}", flush=True)

    end = max(d.index[-1] for d in data.values())
    start = end - pd.Timedelta(days=sb.TEST_DAYS)
    mid = start + (end - start) / 2
    print(f"Window {start:%d-%b-%Y} to {end:%d-%b-%Y}; "
          f"H1 up to {mid:%d-%b-%Y}, H2 after", flush=True)
    ndays = len(set().union(*[set(d.index[d.index >= start])
                              for d in data.values()]))

    rows = []
    for s, d in data.items():
        P = prep2(d)
        first_i = max(60, int(np.searchsorted(d.index.values,
                                              np.datetime64(start))))
        if first_i >= len(d) - 1:
            continue
        rows += simulate_ft(s, P, TARGET[0], TARGET[1], first_i, nmap)
    print(f"{len(rows)} trades, target rule {TARGET}")

    base = stat(rows)
    a = stat([t for t in rows if t["entry_date"] <= mid])
    b = stat([t for t in rows if t["entry_date"] > mid])
    print(f"\nALL: N {base['n']}  win {base['win']:.0f}%  NetR "
          f"{base['R']:.2f}  PF {sb.fmt_pf(base['pf'])}  | H1 {a['R']:.2f}"
          f"  H2 {b['R']:.2f}   (NetR = after all costs)")
    for key, title in FEATURES:
        table(rows, mid, key, f"By {title}")

    sel = [t for t in rows if passes(t)]
    print(f"\n=== FILTERED: {USE_FILTERS or 'no filters'} ===")
    s = stat(sel)
    if not s:
        print("No trades pass the filters.")
        return
    a = stat([t for t in sel if t["entry_date"] <= mid])
    b = stat([t for t in sel if t["entry_date"] > mid])
    print(f"N {s['n']}  win {s['win']:.0f}%  NetR {s['R']:.2f}  PF "
          f"{sb.fmt_pf(s['pf'])}")
    print(f"H1: N {a['n'] if a else 0} NetR "
          f"{a['R'] if a else float('nan'):.2f}   H2: N "
          f"{b['n'] if b else 0} NetR {b['R'] if b else float('nan'):.2f}")
    print(f"Signals per trading day: {s['n'] / ndays:.1f}   "
          f"peak money if all taken: Rs {sb.peak_money(sel):.0f}")
    print("Exit reasons:", dict(Counter(t["reason"] for t in sel)))
    months = defaultdict(list)
    for t in sel:
        months[t["entry_date"].strftime("%Y-%m")].append(t["Rn"])
    print("By entry month (N, NetR):")
    for m in sorted(months):
        print(f"  {m}  {len(months[m]):>4}  {np.mean(months[m]):>6.2f}")


if __name__ == "__main__":
    main()

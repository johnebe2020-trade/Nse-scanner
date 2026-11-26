"""Paper trader for the breakout scanner plan. NO real orders, ever.
Runs once per day after the market closes (GitHub Actions, weekdays).
Order of work each day:  fill yesterday's orders -> manage open positions
-> scan for new signals (orders for tomorrow) -> write CSV files."""
import math
import os
import sys
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

# ================= SETTINGS =================
FILE = "EQUITY_L.csv"
START_DATE = "2026-11-26"      # first trading day of the paper run
CAPITAL = 500000               # paper capital in rupees
TARGET_PROFIT = 2000           # rupees if the full position hits T1
MAX_POSITION_PCT = 0.20        # one position never exceeds 20% of CAPITAL
MIN_QTY_FRAC = 0.5             # if cash allows < half the qty, skip the trade
RISKY_SIZE = 0.5               # RISKY setups get half quantity
# --- scanner rules (same as breakout_scanner.py) ---
LOOKBACK = 20
MIN_PRICE = 20
MIN_AVG_VOL = 100000
TREND_DAYS = 10
VOL_MULT = 1.5
ST_PERIOD, ST_MULT = 10, 3.0
MAX_CHASE = 0.05
EXT_ATR = 2.5
MAX_SL_PCT = 10.0
ENTRY_BUFFER = 0.002
SL_BUFFER = 0.002
GAP_LIMIT = 0.03
T1_R, T2_R = 1.5, 3.0
# --- exits ---
TP1_PART = 0.5                 # half of the shares sold at T1, rest at T2
MAX_BARS = 20                  # close everything after 20 trading days
MOVE_STOP_TO_ENTRY = False     # backtests showed no difference
# --- costs (same as backtests) ---
SLIP_PCT = 0.0005
COST_PCT = 0.0025
DP_CHARGE = 15
# --- files ---
DIR = os.environ.get("PAPER_DIR", ".")
F_PENDING = "paper_pending.csv"
F_POS = "paper_positions.csv"
F_TRADES = "paper_trades.csv"
F_EVENTS = "paper_events.csv"
F_ACCOUNT = "paper_account.csv"

COLS = {
    F_PENDING: ["symbol", "signal_date", "setup", "flags", "entry", "sl",
                "t1", "t2", "skip_above", "risk_pct", "vol_x", "qty"],
    F_POS: ["id", "symbol", "setup", "flags", "signal_date", "entry_date",
            "entry_time", "fill", "qty", "q1", "sl0", "sl", "t1", "t1p", "t2",
            "days", "half", "leg1_date", "leg1_time", "leg1_px"],
    F_TRADES: ["trade_id", "symbol", "setup", "flags", "signal_date",
               "entry_date", "entry_time", "entry_price", "qty", "invested",
               "stop_loss", "target1", "target2", "leg1_date", "leg1_time",
               "leg1_qty", "leg1_price", "exit_date", "exit_time",
               "exit_qty", "exit_price", "exit_reason", "days_held",
               "gross_pnl", "costs", "net_pnl", "net_pct", "r_multiple",
               "cash_after"],
    F_EVENTS: ["logged_at", "candle_date", "time", "event", "symbol",
               "price", "qty", "detail"],
    F_ACCOUNT: ["date", "cash", "invested_at_cost", "open_positions",
                "market_value", "equity", "realized_pnl_total",
                "closed_trades", "pending_orders"],
}


# ================= DATA =================
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


_intra = {}


def event_time(sym, day, kind, level, at_open):
    """Clock time (IST, HH:MM) at which a price level was first touched on
    that day, from 5-minute candles. Best effort: '' when unavailable."""
    if at_open:
        return "09:15"
    try:
        if (pd.Timestamp.now() - day).days > 55:
            return ""
        key = (sym, day)
        if key not in _intra:
            import yfinance as yf
            df = yf.download(sym + ".NS", start=day.strftime("%Y-%m-%d"),
                             end=(day + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                             interval="5m", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            _intra[key] = df
        df = _intra[key]
        if df is None or df.empty:
            return ""
        m = (df["High"] >= level) if kind == "up" else (df["Low"] <= level)
        if not m.any():
            return ""
        return m.idxmax().strftime("%H:%M")
    except Exception:
        return ""


# ================= SIGNALS (same logic as breakout_scanner.py) ==========
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
    return dict(status=status, flags=flags, entry=entry, sl=sl,
                risk_pct=risk_pct, t1=tick(entry + T1_R * risk_ps),
                t2=tick(entry + T2_R * risk_ps),
                skip=tick(entry * (1 + GAP_LIMIT)), vol_x=vol_x)


def size_qty(entry, t1, status):
    """Quantity so that the full position makes about TARGET_PROFIT at T1."""
    q = int(round(TARGET_PROFIT / (t1 - entry)))
    if status == "RISKY":
        q = int(math.floor(q * RISKY_SIZE))
    q = min(q, int(math.floor(CAPITAL * MAX_POSITION_PCT / entry)))
    return q


# ================= BOOK (state kept in CSV files) =================
def _path(name):
    return os.path.join(DIR, name)


def _load(name):
    p = _path(name)
    if os.path.exists(p):
        return pd.read_csv(p, keep_default_na=False)
    return pd.DataFrame(columns=COLS[name])


def _num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class Book:
    def __init__(self):
        self.pending = [dict(r) for r in _load(F_PENDING).to_dict("records")]
        self.positions = []
        for r in _load(F_POS).to_dict("records"):
            for k in ("fill", "sl0", "sl", "t1", "t1p", "t2", "leg1_px"):
                r[k] = _num(r[k], float("nan"))
            for k in ("id", "qty", "q1", "days"):
                r[k] = int(_num(r[k]))
            r["half"] = bool(int(_num(r["half"])))
            self.positions.append(r)
        self.trades = _load(F_TRADES).to_dict("records")
        self.events = _load(F_EVENTS).to_dict("records")
        self.account = _load(F_ACCOUNT).to_dict("records")
        if self.account:
            self.cash = float(self.account[-1]["cash"])
            self.last_date = pd.Timestamp(self.account[-1]["date"])
        else:
            self.cash = float(CAPITAL)
            self.last_date = None
        ids = [int(_num(t["trade_id"])) for t in self.trades] + \
              [p["id"] for p in self.positions]
        self.next_id = (max(ids) + 1) if ids else 1
        self.now = datetime.now()

    def log(self, day, tm, event, sym, price, qty, detail=""):
        self.events.append(dict(
            logged_at=self.now.strftime("%Y-%m-%d %H:%M:%S"),
            candle_date=day.strftime("%Y-%m-%d"), time=tm, event=event,
            symbol=sym, price=round(float(price), 2) if price != "" else "",
            qty=qty, detail=detail))

    def save(self):
        os.makedirs(DIR, exist_ok=True)
        pos = pd.DataFrame(self.positions, columns=COLS[F_POS])
        if len(pos):
            pos["half"] = pos["half"].astype(int)
        pd.DataFrame(self.pending, columns=COLS[F_PENDING]).to_csv(
            _path(F_PENDING), index=False)
        pos.to_csv(_path(F_POS), index=False)
        pd.DataFrame(self.trades, columns=COLS[F_TRADES]).to_csv(
            _path(F_TRADES), index=False)
        pd.DataFrame(self.events, columns=COLS[F_EVENTS]).to_csv(
            _path(F_EVENTS), index=False)
        pd.DataFrame(self.account, columns=COLS[F_ACCOUNT]).to_csv(
            _path(F_ACCOUNT), index=False)


def fmt(d):
    return d.strftime("%Y-%m-%d")


def idx_of(S, day):
    k = S["dates"].searchsorted(day)
    if k < len(S["dates"]) and S["dates"][k] == day:
        return int(k)
    return None


# ================= ONE TRADING DAY =================
def step(p, i, S, first):
    """Same exit rules as the backtest. Returns (t1_px or None,
    (exit_px, reason) or None)."""
    o, h, l, c = S["o"], S["h"], S["l"], S["c"]
    t1_px = None
    if l[i] <= p["sl"]:                                  # stop first
        px = p["sl"] if first else min(o[i], p["sl"])
        return None, (px, "SL2" if p["half"] else "SL")
    if not p["half"] and h[i] >= p["t1"]:
        t1_px = p["t1"] if first else max(o[i], p["t1"])
        p["half"] = True
        p["t1"] = t1_px
        if MOVE_STOP_TO_ENTRY:
            p["sl"] = max(p["sl"], p["fill"])
    if p["half"] and h[i] >= p["t2"]:
        px = p["t2"] if first else max(o[i], p["t2"])
        return t1_px, (px, "T2")
    if p["days"] >= MAX_BARS:
        return t1_px, (c[i], "TIME")
    return t1_px, None


def handle_step(bk, p, i, S, day, first, closed_today):
    sym = p["symbol"]
    o, l = S["o"], S["l"]
    t1_px, ex = step(p, i, S, first)
    if t1_px is not None and p["q1"] > 0:
        tm = event_time(sym, day, "up", t1_px, bool(o[i] >= t1_px and not first))
        e1 = t1_px * (1 - SLIP_PCT)
        bk.cash += p["q1"] * e1 - DP_CHARGE
        p["leg1_date"], p["leg1_time"], p["leg1_px"] = fmt(day), tm, t1_px
        bk.log(day, tm, "T1_HIT", sym, t1_px, p["q1"], "half sold at T1")
    elif t1_px is not None:                      # q1 == 0: nothing to sell
        p["leg1_date"], p["leg1_time"], p["leg1_px"] = "", "", float("nan")
    if ex is None:
        return False
    px, why = ex
    rem = p["qty"] - (p["q1"] if (p["half"] and p["q1"] > 0) else 0)
    if why == "TIME":
        tm = "15:30"
    elif why in ("SL", "SL2"):
        tm = event_time(sym, day, "down", px, bool(o[i] <= px and not first))
    else:
        tm = event_time(sym, day, "up", px, bool(o[i] >= px and not first))
    ex_eff = px * (1 - SLIP_PCT)
    bk.cash += rem * ex_eff - DP_CHARGE
    e = p["fill"] * (1 + SLIP_PCT)
    legs = 2 if (p["half"] and p["q1"] > 0) else 1
    l1 = p["q1"] * (p["leg1_px"] * (1 - SLIP_PCT) - e) if legs == 2 else 0.0
    gross = l1 + rem * (ex_eff - e)
    costs = COST_PCT * e * p["qty"] + DP_CHARGE * legs
    net = gross - costs
    invested = e * p["qty"]
    risk = p["fill"] - p["sl0"]
    bk.trades.append(dict(
        trade_id=p["id"], symbol=sym, setup=p["setup"], flags=p["flags"],
        signal_date=p["signal_date"], entry_date=p["entry_date"],
        entry_time=p["entry_time"], entry_price=round(p["fill"], 2),
        qty=p["qty"], invested=round(invested, 2),
        stop_loss=round(p["sl0"], 2), target1=round(p["t1p"], 2),
        target2=round(p["t2"], 2),
        leg1_date=p["leg1_date"] if legs == 2 else "",
        leg1_time=p["leg1_time"] if legs == 2 else "",
        leg1_qty=p["q1"] if legs == 2 else "",
        leg1_price=round(p["leg1_px"], 2) if legs == 2 else "",
        exit_date=fmt(day), exit_time=tm, exit_qty=rem,
        exit_price=round(px, 2), exit_reason=why, days_held=p["days"],
        gross_pnl=round(gross, 2), costs=round(costs, 2),
        net_pnl=round(net, 2),
        net_pct=round(net / invested * 100, 2),
        r_multiple=round(net / (p["qty"] * risk), 2) if risk > 0 else "",
        cash_after=round(bk.cash, 2)))
    bk.log(day, tm, {"SL": "STOP_HIT", "SL2": "STOP_HIT_AFTER_T1",
                     "T2": "T2_HIT", "TIME": "TIME_EXIT"}[why],
           sym, px, rem, f"net P&L {net:.0f}")
    closed_today.add(sym)
    return True


def fill_orders(bk, day, data_S):
    closed_today = set()
    order = sorted(bk.pending, key=lambda r: (r["setup"] != "ACTIVE",
                                              -_num(r["vol_x"])))
    bk.pending = []
    new_positions = []
    for r in order:
        sym = r["symbol"]
        S = data_S.get(sym)
        i = idx_of(S, day) if S else None
        if i is None:
            bk.log(day, "", "ORDER_EXPIRED", sym, "", "", "no candle today")
            continue
        o, h = S["o"][i], S["h"][i]
        E, skip = float(r["entry"]), float(r["skip_above"])
        if o >= E:
            if o > skip:
                bk.log(day, "09:15", "ORDER_SKIPPED_GAP", sym, o, "",
                       f"opened {o:.2f} above skip level {skip:.2f}")
                continue
            fill, at_open = o, True
        elif h >= E:
            fill, at_open = E, False
        else:
            bk.log(day, "", "ORDER_NOT_TRIGGERED", sym, E, "",
                   f"high {h:.2f} stayed below entry")
            continue
        q = int(_num(r["qty"]))
        q = min(q, int(math.floor(CAPITAL * MAX_POSITION_PCT / fill)))
        e = fill * (1 + SLIP_PCT)
        need = lambda n: n * e * (1 + COST_PCT)
        if need(q) > bk.cash:
            q_aff = int(math.floor(bk.cash / (e * (1 + COST_PCT))))
            if q_aff < 1 or q_aff < q * MIN_QTY_FRAC:
                bk.log(day, "", "ORDER_SKIPPED_NO_CASH", sym, fill, q,
                       f"cash {bk.cash:.0f}")
                continue
            q = q_aff
        if q < 1:
            bk.log(day, "", "ORDER_SKIPPED_QTY0", sym, fill, 0, "")
            continue
        q1 = int(math.floor(q * TP1_PART))
        tm = event_time(sym, day, "up", fill, at_open)
        bk.cash -= e * q + COST_PCT * e * q
        p = dict(id=bk.next_id, symbol=sym, setup=r["setup"],
                 flags=r["flags"], signal_date=r["signal_date"],
                 entry_date=fmt(day), entry_time=tm, fill=fill, qty=q,
                 q1=q1, sl0=float(r["sl"]), sl=float(r["sl"]),
                 t1=float(r["t1"]) if q1 > 0 else float("inf"),
                 t1p=float(r["t1"]), t2=float(r["t2"]), days=1,
                 half=False, leg1_date="", leg1_time="",
                 leg1_px=float("nan"))
        bk.next_id += 1
        bk.log(day, tm, "BUY_FILLED", sym, fill, q,
               f"{r['setup']} {r['flags']} SL {p['sl']:.2f} "
               f"T1 {p['t1p']:.2f} T2 {p['t2']:.2f}")
        # manage the entry day too (stop first, same as the backtest)
        if not handle_step(bk, p, i, S, day, True, closed_today):
            new_positions.append(p)
    return closed_today, new_positions


def process_day(bk, day, data_S, seed_only=False):
    closed_today = set()
    if not seed_only:
        old_positions = bk.positions
        c1, new_positions = fill_orders(bk, day, data_S)
        closed_today |= c1
        keep = []
        for p in old_positions:
            sym = p["symbol"]
            S = data_S.get(sym)
            i = idx_of(S, day) if S else None
            if i is None:
                keep.append(p)
                continue
            p["days"] += 1
            if not handle_step(bk, p, i, S, day, False, closed_today):
                keep.append(p)
        bk.positions = keep + new_positions
    busy = {p["symbol"] for p in bk.positions} | closed_today
    new_orders = []
    for sym, S in data_S.items():
        if sym in busy:
            continue
        i = idx_of(S, day)
        if i is None:
            continue
        pl = plan_at(S, i)
        if pl is None:
            continue
        if pl["entry"] is None:
            bk.log(day, "EOD", "SIGNAL_NO_PLAN", sym, S["c"][i], "",
                   f"{pl['status']} {' '.join(pl['flags'])}")
            continue
        q = size_qty(pl["entry"], pl["t1"], pl["status"])
        if q < 1:
            bk.log(day, "EOD", "SIGNAL_QTY0", sym, pl["entry"], 0, "")
            continue
        new_orders.append(dict(
            symbol=sym, signal_date=fmt(day), setup=pl["status"],
            flags=" ".join(pl["flags"]) if pl["flags"] else "-",
            entry=pl["entry"], sl=pl["sl"], t1=pl["t1"], t2=pl["t2"],
            skip_above=pl["skip"], risk_pct=pl["risk_pct"],
            vol_x=round(pl["vol_x"], 2), qty=q))
        bk.log(day, "EOD", "ORDER_PLACED", sym, pl["entry"], q,
               f"{pl['status']} buy above {pl['entry']} SL {pl['sl']} "
               f"T1 {pl['t1']} T2 {pl['t2']} skip>{pl['skip']}")
    bk.pending = new_orders
    # ---- account line ----
    mv = 0.0
    cost_basis = 0.0
    for p in bk.positions:
        S = data_S.get(p["symbol"])
        i = idx_of(S, day) if S else None
        px = S["c"][i] if i is not None else p["fill"]
        rem = p["qty"] - (p["q1"] if p["half"] else 0)
        mv += rem * px
        cost_basis += rem * p["fill"]
    realized = sum(_num(t["net_pnl"]) for t in bk.trades)
    bk.account.append(dict(
        date=fmt(day), cash=round(bk.cash, 2),
        invested_at_cost=round(cost_basis, 2),
        open_positions=len(bk.positions), market_value=round(mv, 2),
        equity=round(bk.cash + mv, 2),
        realized_pnl_total=round(realized, 2),
        closed_trades=len(bk.trades), pending_orders=len(bk.pending)))
    bk.last_date = day
    bk.cash = float(bk.account[-1]["cash"])


# ================= RUN =================
def run_engine(data, now):
    """data: {symbol: daily OHLCV frame with COMPLETE candles only}."""
    bk = Book()
    bk.now = now
    data_S = {s: prep_sig(d) for s, d in data.items() if len(d) >= 90}
    all_days = sorted(set().union(*[set(S["dates"]) for S in data_S.values()]))
    start = pd.Timestamp(START_DATE)
    if bk.last_date is None:
        prev = [d for d in all_days if d < start]
        if prev:
            process_day(bk, prev[-1], data_S, seed_only=True)
            print(f"Seeded orders from the {prev[-1]:%d-%b-%Y} close")
        else:
            print("No candle before START_DATE; nothing to seed.")
            return bk
    todo = [d for d in all_days if d > bk.last_date and d >= start]
    if not todo:
        print("No new completed trading day to process.")
    for day in todo:
        process_day(bk, day, data_S)
        a = bk.account[-1]
        print(f"{day:%d-%b-%Y}: equity {a['equity']:,.0f} cash "
              f"{a['cash']:,.0f} open {a['open_positions']} closed "
              f"{a['closed_trades']} orders for next session "
              f"{a['pending_orders']}")
    bk.save()
    return bk


def last_complete_date(now):
    today = pd.Timestamp(now.date())
    return today if now.hour * 60 + now.minute >= 16 * 60 + 15 \
        else today - pd.Timedelta(days=1)


def main():
    ist = datetime.now(timezone(timedelta(hours=5, minutes=30))).replace(tzinfo=None)
    lcd = last_complete_date(ist)
    df = pd.read_csv(FILE)
    df.columns = df.columns.str.strip()
    df = df[df["SERIES"].str.strip() == "EQ"]
    symbols = df["SYMBOL"].str.strip().str.upper().tolist()
    data = download_all(symbols)
    data = {s: d[d.index <= lcd] for s, d in data.items()}
    data = {s: d for s, d in data.items() if len(d) >= 120}
    print(f"Price data for {len(data)} symbols, last complete day "
          f"{lcd:%d-%b-%Y}", flush=True)
    if not data:
        sys.exit("No data downloaded.")
    bk = run_engine(data, ist)
    t = pd.DataFrame(bk.trades)
    if len(t):
        net = t["net_pnl"].astype(float)
        print(f"\nClosed trades {len(t)}, win {100 * (net > 0).mean():.0f}%, "
              f"net P&L Rs {net.sum():,.0f}")
    print(f"Open positions: {len(bk.positions)}")


if __name__ == "__main__":
    main()

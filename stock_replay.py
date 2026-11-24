import os, csv, math, sys, time
import datetime as dt
from replay import get_bars, load_master, DAY

# ---------- CONFIG ----------
CAPITAL = 100000
TARGET_PROFIT = 2000     # rupees at T1, sets the quantity
MAX_RISK = 1500          # max loss per trade at the stop-loss
USE_SCANNER_QTY = False  # True = use the CSV qty (Rs 1000 risk)
ENTRY_START = dt.time(9, 15)
ENTRY_CUTOFF = dt.time(14, 30)
EXIT_TIME = dt.time(15, 15)
INTRADAY = False         # False = hold, report open P&L at close
SLIP_PCT = 0.0005        # 0.05% each side
CHARGES = 50             # rough flat cost per trade
CSV_PATH = os.getenv("SIGNAL_CSV", "results_breakout.csv")


def load_signals():
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            if r["status"].strip() != "ACTIVE":
                continue
            try:
                rows.append(dict(sym=r["symbol"].strip(),
                                 entry=float(r["entry"]),
                                 sl=float(r["sl"]),
                                 t1=float(r["t1"]),
                                 skip=float(r["skip_above"]),
                                 qty=int(float(r["qty"])),
                                 date=r["date"].strip()))
            except ValueError:
                print("Bad row skipped:", r.get("symbol"))
    return rows


def sec_id(sym):
    for r in load_master():
        if (r.get("SEM_EXM_EXCH_ID") == "NSE"
                and r.get("SEM_INSTRUMENT_NAME") == "EQUITY"
                and r.get("SEM_TRADING_SYMBOL") == sym
                and r.get("SEM_SERIES", "EQ") == "EQ"):
            return r["SEM_SMST_SECURITY_ID"]
    return None


def simulate(s, bars):
    if bars[0]["o"] > s["skip"]:
        return dict(sym=s["sym"], status="SKIPPED",
                    why=f"opened {bars[0]['o']} above skip level {s['skip']}")
    pos = None
    for b in bars:
        tm = b["t"].time()
        if pos is None:
            if tm >= ENTRY_CUTOFF:
                break
            if tm >= ENTRY_START and b["h"] >= s["entry"]:
                fill = max(s["entry"], b["o"])
                if fill > s["skip"]:
                    return dict(sym=s["sym"], status="SKIPPED",
                                why=f"fill {fill:.2f} above skip "
                                    f"level {s['skip']}")
                pos = dict(t=b["t"], px=fill * (1 + SLIP_PCT))
            continue            # exits are checked from the next bar
        # position open
        if b["l"] <= s["sl"]:
            px, why = min(b["o"], s["sl"]), "STOP LOSS"
        elif b["h"] >= s["t1"]:
            px, why = max(b["o"], s["t1"]), "TARGET T1"
        elif INTRADAY and tm >= EXIT_TIME:
            px, why = b["c"], "EOD SQUARE-OFF"
        else:
            continue
        return dict(sym=s["sym"], status="TRADE", sig=s,
                    entry_t=pos["t"], entry_px=pos["px"],
                    exit_t=b["t"], exit_px=px * (1 - SLIP_PCT), why=why)
    if pos:
        last = bars[-1]
        return dict(sym=s["sym"], status="TRADE", sig=s,
                    entry_t=pos["t"], entry_px=pos["px"],
                    exit_t=last["t"], exit_px=last["c"] * (1 - SLIP_PCT),
                    why="OPEN AT CLOSE")
    return dict(sym=s["sym"], status="NO TRADE",
                why="price never reached entry")


def main():
    print(f"Stock breakout replay: {DAY}  capital {CAPITAL}")
    sigs = load_signals()
    if not sigs:
        sys.exit("No ACTIVE rows in the CSV.")
    sdate = dt.datetime.strptime(sigs[0]["date"], "%d-%b-%y").date()
    if sdate >= dt.date.fromisoformat(DAY):
        print(f"WARNING: signals dated {sdate}, not before {DAY}. "
              f"This replay is not valid.")
    else:
        print(f"Signals dated {sdate}")
    print(f"{len(sigs)} ACTIVE: " + ", ".join(s["sym"] for s in sigs))

    results = []
    for s in sigs:
        sid = sec_id(s["sym"])
        if not sid:
            print(f"{s['sym']}: not found in scrip master")
            continue
        try:
            bars = get_bars(sid, "NSE_EQ", "EQUITY")
        except SystemExit as e:
            print(f"{s['sym']}: data error {e}")
            continue
        time.sleep(0.5)
        if not bars:
            print(f"{s['sym']}: no bars")
            continue
        results.append(simulate(s, bars))

    for r in results:
        if r["status"] != "TRADE":
            print(f"{r['sym']:<12} {r['status']}: {r['why']}")

    trades = sorted([r for r in results if r["status"] == "TRADE"],
                    key=lambda r: r["entry_t"])
    accepted = []
    net = 0.0
    wins = open_n = 0
    print("-" * 60)
    for r in trades:
        s, e = r["sig"], r["entry_px"]
        blocked = sum(a["qty"] * a["px"] for a in accepted
                      if a["exit_t"] > r["entry_t"])
        avail = CAPITAL - blocked
        if e >= s["t1"] or e <= s["sl"]:
            print(f"{r['sym']}: entry {e:.2f} outside SL/T1 range, skipped")
            continue
        q_c = int(avail // e)
        if USE_SCANNER_QTY:
            q, lim = min(s["qty"], q_c), "scanner"
        else:
            q_t = math.ceil(TARGET_PROFIT / (s["t1"] - e))
            q_r = int(MAX_RISK // (e - s["sl"]))
            q = min(q_t, q_r, q_c)
            lim = "target" if q == q_t else ("risk" if q == q_r else "cash")
        if q < 1:
            print(f"{r['sym']}: skipped, quantity 0 (cash {avail:.0f})")
            continue
        pnl = (r["exit_px"] - e) * q - CHARGES
        net += pnl
        wins += pnl > 0
        open_n += r["why"] == "OPEN AT CLOSE"
        accepted.append(dict(qty=q, px=e, exit_t=r["exit_t"]))
        print(f"{r['sym']:<10} {r['entry_t']:%H:%M} buy {e:8.2f} x{q:<4}"
              f" -> {r['exit_t']:%H:%M} {r['exit_px']:8.2f}"
              f" [{r['why']}] sized by {lim}  P&L {pnl:7.0f}")

    print("-" * 60)
    print(f"Trades {len(accepted)}  Wins {wins}  Still open {open_n}  "
          f"NET Rs {net:.0f} ({net / CAPITAL * 100:.2f}% of capital)")
    if open_n:
        print("Open trades are marked at the day's close, not final.")


if __name__ == "__main__":
    main()

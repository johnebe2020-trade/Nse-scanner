import os, csv, io, sys, datetime as dt
from zoneinfo import ZoneInfo
import requests

try:
    from dhanhq import DhanContext, dhanhq   # newer library
    _ctx = DhanContext(os.environ["DHAN_CLIENT_ID"],
                       os.environ["DHAN_TOKEN"])
    dhan = dhanhq(_ctx)
except ImportError:                           # older library
    from dhanhq import dhanhq
    dhan = dhanhq(os.environ["DHAN_CLIENT_ID"],
                  os.environ["DHAN_TOKEN"])

IST = ZoneInfo("Asia/Kolkata")

# ---------- CONFIG ----------
CAPITAL = 100000
MAX_LOTS = 1
SL_PCT = 0.20          # initial SL on premium
TRAIL_START = 0.30     # trail starts after +30%
TRAIL_PCT = 0.25       # trail 25% below peak
SLIPPAGE = 0.5         # rupees per share, each side
CHARGES = 60           # rough flat cost per round trip
ENTRY_CUTOFF = dt.time(14, 30)
EXIT_TIME = dt.time(15, 15)
TS_SHIFT_HOURS = 0     # fix if bar times look wrong

DAY = os.getenv("REPLAY_DATE") or dt.datetime.now(IST).strftime("%Y-%m-%d")

MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"


# ---------- DATA ----------
def get_bars(sec_id, seg, inst):
    d0 = dt.date.fromisoformat(DAY)
    d1 = d0 + dt.timedelta(days=1)
    r = dhan.intraday_minute_data(str(sec_id), seg, inst,
                                  DAY, d1.isoformat(), 1)
    if r.get("status") != "success":
        sys.exit(f"Dhan error: {r}")
    x = r["data"]
    bars = []
    for i, ts in enumerate(x["timestamp"]):
        t = dt.datetime.fromtimestamp(ts + TS_SHIFT_HOURS * 3600, IST)
        if t.strftime("%Y-%m-%d") != DAY:
            continue
        bars.append(dict(t=t, o=x["open"][i], h=x["high"][i],
                         l=x["low"][i], c=x["close"][i]))
    bars.sort(key=lambda b: b["t"])
    return bars


_master = None
def load_master():
    global _master
    if _master is None:
        txt = requests.get(MASTER_URL, timeout=60).text
        _master = list(csv.DictReader(io.StringIO(txt)))
    return _master


def find_option(strike, side):
    rows = [r for r in load_master()
            if r.get("SEM_INSTRUMENT_NAME") == "OPTIDX"
            and r.get("SEM_TRADING_SYMBOL", "").startswith("NIFTY-")
            and r["SEM_EXPIRY_DATE"][:10] >= DAY]
    if not rows:
        sys.exit("No NIFTY options found. Print load_master()[0].keys() "
                 "and check column names.")
    exp = min(r["SEM_EXPIRY_DATE"][:10] for r in rows)
    for r in rows:
        if (r["SEM_EXPIRY_DATE"][:10] == exp
                and abs(float(r["SEM_STRIKE_PRICE"]) - strike) < 0.01
                and r["SEM_OPTION_TYPE"] == side):
            return (r["SEM_SMST_SECURITY_ID"],
                    int(float(r["SEM_LOT_UNITS"])),
                    r["SEM_TRADING_SYMBOL"], exp)
    sys.exit(f"Strike {strike} {side} not found for expiry {exp}")


# ---------- STRATEGY REPLAY ----------
def main():
    print(f"Replay date: {DAY}")
    idx = get_bars(13, "IDX_I", "INDEX")
    if not idx:
        sys.exit("No index bars. Market closed or data API not enabled.")
    print("Index bars:", idx[0]["t"].strftime("%H:%M"), "to",
          idx[-1]["t"].strftime("%H:%M"), f"({len(idx)} bars)")

    first15 = [b for b in idx
               if dt.time(9, 15) <= b["t"].time() < dt.time(9, 30)]
    if len(first15) < 10:
        sys.exit("9:15 candle bars missing. Check timestamps (TS_SHIFT_HOURS).")

    H = max(b["h"] for b in first15)
    L = min(b["l"] for b in first15)
    C = first15[-1]["c"]
    base = round(C / 100) * 100
    ce_k, pe_k = base - 100, base + 100
    print(f"15m candle  H {H}  L {L}  C {C}")
    print(f"Base {base}  CE strike {ce_k}  PE strike {pe_k}")

    # --- find breakout ---
    side = strike = entry_t = None
    for b in idx:
        tm = b["t"].time()
        if tm < dt.time(9, 30):
            continue
        if tm >= ENTRY_CUTOFF:
            break
        if b["c"] > H:
            side, strike, entry_t = "CE", ce_k, b["t"]
            break
        if b["c"] < L:
            side, strike, entry_t = "PE", pe_k, b["t"]
            break
    if not side:
        print("No breakout before cutoff. No trade today.")
        return
    print(f"Signal: {side} {strike} at {entry_t.strftime('%H:%M')}")

    # --- option data ---
    sec_id, lot, sym, exp = find_option(strike, side)
    print(f"Contract {sym}  expiry {exp}  lot {lot}")
    ob = get_bars(sec_id, "NSE_FNO", "OPTIDX")
    if not ob:
        sys.exit("No option bars returned.")

    ebar = next((b for b in ob if b["t"] >= entry_t), None)
    if not ebar:
        sys.exit("No option bar at or after entry time.")
    entry_t = ebar["t"]
    entry = ebar["c"] + SLIPPAGE

    lots = min(MAX_LOTS, int(CAPITAL // (entry * lot)))
    if lots < 1:
        print(f"Premium {entry:.1f} x lot {lot} exceeds capital. Skipped.")
        return
    qty = lots * lot
    sl = entry * (1 - SL_PCT)
    peak = entry
    print(f"ENTRY {entry_t.strftime('%H:%M')}  {side} {strike} "
          f"@ {entry:.2f}  qty {qty}  SL {sl:.2f}")

    idx_map = {b["t"]: b for b in idx}
    exit_px = exit_t = reason = None

    for b in ob:
        if b["t"] <= entry_t:
            continue
        tm = b["t"].time()

        # 1) stop-loss / trailing hit inside the bar
        if b["l"] <= sl:
            exit_px, exit_t = min(b["o"], sl), b["t"]
            reason = "SL / TRAIL"
            break

        # 2) spot closed back through base against us
        sb = idx_map.get(b["t"])
        if sb and ((side == "CE" and sb["c"] < base)
                   or (side == "PE" and sb["c"] > base)):
            exit_px, exit_t = b["c"], b["t"]
            reason = f"SPOT BACK THROUGH BASE ({sb['c']})"
            break

        # 3) end of day exit
        if tm >= EXIT_TIME:
            exit_px, exit_t = b["c"], b["t"]
            reason = "EOD EXIT"
            break

        # 4) update peak and trailing SL (never below entry once trailing)
        peak = max(peak, b["h"])
        if peak >= entry * (1 + TRAIL_START):
            sl = max(sl, entry, peak * (1 - TRAIL_PCT))

    if exit_px is None:
        exit_px, exit_t, reason = ob[-1]["c"], ob[-1]["t"], "DATA END"

    exit_px -= SLIPPAGE
    pnl = (exit_px - entry) * qty - CHARGES
    print(f"EXIT  {exit_t.strftime('%H:%M')}  @ {exit_px:.2f}  [{reason}]")
    print(f"Peak premium {peak:.2f}")
    print(f"NET PnL: Rs {pnl:.0f}  ({pnl / CAPITAL * 100:.2f}% of capital)")

    last = ob[-1]["c"] - SLIPPAGE
    print(f"What-if hold to close: Rs {(last - entry) * qty - CHARGES:.0f}")


if __name__ == "__main__":
    main()

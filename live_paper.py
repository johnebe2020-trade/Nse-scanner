import os, time, datetime as dt
from zoneinfo import ZoneInfo
import requests
from replay import (dhan, get_bars, find_option, CAPITAL, MAX_LOTS,
                    SL_PCT, TRAIL_START, TRAIL_PCT, SLIPPAGE, CHARGES,
                    ENTRY_CUTOFF, EXIT_TIME)

IST = ZoneInfo("Asia/Kolkata")
POLL = 4  # option_chain limit is 1 call per 3 sec


def now():
    return dt.datetime.now(IST)


def minute():
    return now().replace(second=0, microsecond=0)


def alert(msg):
    line = f"[{now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    tok, chat = os.getenv("TG_TOKEN"), os.getenv("TG_CHAT")
    if tok and chat:
        try:
            requests.post(
                f"https://api.telegram.org/bot{tok}/sendMessage",
                data={"chat_id": chat, "text": line}, timeout=10)
        except Exception:
            pass


def chain(expiry):
    for _ in range(3):
        try:
            r = dhan.option_chain(under_security_id=13,
                                  under_exchange_segment="IDX_I",
                                  expiry=expiry)
            if r.get("status") == "success":
                d = r["data"]["data"]
                return d["last_price"], d["oc"]
            print("chain error:", r, flush=True)
        except Exception as e:
            print("chain exception:", e, flush=True)
        time.sleep(4)
    return None, None


def ltp(oc, strike, side):
    for k, v in oc.items():
        if abs(float(k) - strike) < 0.01:
            return v[side.lower()]["last_price"]
    return None


def main():
    while now().time() < dt.time(9, 30, 5):
        time.sleep(5)

    idx = get_bars(13, "IDX_I", "INDEX")
    first15 = [b for b in idx
               if dt.time(9, 15) <= b["t"].time() < dt.time(9, 30)]
    if len(first15) < 10:
        alert("9:15 candle missing. Holiday or data issue. Stopping.")
        return

    H = max(b["h"] for b in first15)
    L = min(b["l"] for b in first15)
    C = first15[-1]["c"]
    base = round(C / 100) * 100
    ce_k, pe_k = base - 100, base + 100
    ce = find_option(ce_k, "CE")
    pe = find_option(pe_k, "PE")
    expiry, lot = ce[3], ce[1]
    alert(f"PAPER | H {H} L {L} C {C} | base {base} "
          f"CE {ce_k} PE {pe_k} | expiry {expiry} lot {lot}")

    spot, oc = chain(expiry)
    if spot is None:
        alert("Option chain unavailable. Check Data APIs and token.")
        return
    print(f"Spot {spot}, strikes in chain: {len(oc)}", flush=True)

    # ---- breakout that already happened (late start) ----
    side = None
    for b in idx:
        if b["t"].time() < dt.time(9, 30):
            continue
        if b["c"] > H:
            side, bt = "CE", b["t"]
            break
        if b["c"] < L:
            side, bt = "PE", b["t"]
            break
    if side:
        still = (side == "CE" and spot > H) or (side == "PE" and spot < L)
        if still:
            alert(f"LATE ENTRY: breakout was at {bt:%H:%M}, still valid")
        else:
            alert(f"Breakout at {bt:%H:%M} faded. Watching for a new one.")
            side = None

    # ---- wait for breakout ----
    last_min, last_spot = minute(), spot
    while not side and now().time() < ENTRY_CUTOFF:
        time.sleep(POLL)
        spot, oc = chain(expiry)
        if spot is None:
            continue
        m = minute()
        if m > last_min:
            if last_spot > H:
                side = "CE"
            elif last_spot < L:
                side = "PE"
            print(f"{last_min:%H:%M} close {last_spot}", flush=True)
            last_min = m
            if side:
                break
        last_spot = spot
    if not side:
        alert("No breakout before cutoff. No trade today.")
        return

    # ---- paper entry ----
    strike = ce_k if side == "CE" else pe_k
    p = ltp(oc, strike, side)
    if p is None:
        alert("Strike price not in chain. Stopping.")
        return
    entry = p + SLIPPAGE
    lots = min(MAX_LOTS, int(CAPITAL // (entry * lot)))
    if lots < 1:
        alert(f"Premium {entry:.1f} x lot {lot} exceeds capital. Skipped.")
        return
    qty = lots * lot
    sl = entry * (1 - SL_PCT)
    peak = entry
    alert(f"PAPER BUY {side} {strike} @ {entry:.2f} qty {qty} "
          f"SL {sl:.2f} | spot {spot}")

    # ---- manage trade ----
    last_min, last_spot = minute(), spot
    reason = None
    while True:
        time.sleep(POLL)
        spot, oc = chain(expiry)
        if spot is None:
            continue
        p = ltp(oc, strike, side)
        if p is None:
            continue
        peak = max(peak, p)
        if peak >= entry * (1 + TRAIL_START):
            sl = max(sl, peak * (1 - TRAIL_PCT))
        if p <= sl:
            reason = "SL / TRAIL"
            break
        m = minute()
        if m > last_min:
            print(f"{m:%H:%M} spot {last_spot} prem {p} "
                  f"peak {peak:.1f} sl {sl:.1f}", flush=True)
            if ((side == "CE" and last_spot < base)
                    or (side == "PE" and last_spot > base)):
                reason = f"SPOT BACK THROUGH BASE ({last_spot})"
                break
            last_min = m
        last_spot = spot
        if now().time() >= EXIT_TIME:
            reason = "EOD EXIT"
            break

    exit_px = p - SLIPPAGE
    pnl = (exit_px - entry) * qty - CHARGES
    alert(f"PAPER SELL {side} {strike} @ {exit_px:.2f} [{reason}] | "
          f"NET PnL Rs {pnl:.0f} ({pnl / CAPITAL * 100:.2f}%)")


if __name__ == "__main__":
    main()

""" TrendTwin Dhan Paper Trader Paper trading only: this program never submits orders. Reads Dhan quotes; stores independent daily-resistance and 15-minute-resistance paper books in separate CSV files. Required environment: DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN Input: universe.csv columns: symbol,security_id security_id must be the Dhan NSE_EQ security ID for each stock. """
import os, csv, time, math, json, logging
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
import requests
import pandas as pd

TZ = ZoneInfo("Asia/Kolkata")
BASE_URL = "https://api.dhan.co/v2"
CLIENT_ID = os.environ["DHAN_CLIENT_ID"]
ACCESS_TOKEN = os.environ["DHAN_ACCESS_TOKEN"]
UNIVERSE_FILE = os.getenv("UNIVERSE_FILE", "universe.csv")
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "15"))
STARTING_CAPITAL = 500_000.0
TARGET_PROFIT = 2_000.0
TARGET_MOVE = 20.0                  # user's example: ₹20 move -> 100 shares
MAX_POSITION_VALUE = 100_000.0      # safety cap: 20% of ₹5 lakh per trade
PIVOT_LEFT = 3
PIVOT_RIGHT = 3
DAILY_LOOKBACK = 120
INTRADAY_LOOKBACK_DAYS = 15
OUT = Path("paper_logs")
OUT.mkdir(exist_ok=True)
LOG = logging.getLogger("TrendTwin")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

HEADERS = {"access-token": ACCESS_TOKEN, "client-id": CLIENT_ID,
           "Content-Type": "application/json", "Accept": "application/json"}

def now():
    return datetime.now(TZ)

def api_post(path, body):
    r = requests.post(BASE_URL + path, headers=HEADERS, json=body, timeout=20)
    r.raise_for_status()
    payload = r.json()
    if isinstance(payload, dict) and str(payload.get("status", "")).lower() == "failure":
        raise RuntimeError(f"Dhan API failure: {payload}")
    return payload

def read_universe():
    df = pd.read_csv(UNIVERSE_FILE, dtype={"security_id": str})
    required = {"symbol", "security_id"}
    if not required.issubset(df.columns):
        raise ValueError("universe.csv needs columns: symbol,security_id")
    df = df.dropna(subset=["symbol", "security_id"])
    return [{"symbol": str(r.symbol).strip().upper(),
             "security_id": str(r.security_id).strip()} for r in df.itertuples()]

def quote_batch(items):
    # Dhan quote endpoint supports up to 100 instruments per request.
    out = {}
    for i in range(0, len(items), 100):
        ids = [int(x["security_id"]) for x in items[i:i+100]]
        payload = api_post("/marketfeed/quote", {"NSE_EQ": ids})
        data = payload.get("data", {})
        rows = data.get("NSE_EQ", data) if isinstance(data, dict) else data
        if isinstance(rows, dict):
            rows = list(rows.values())
        for row in rows or []:
            if not isinstance(row, dict): continue
            sid = str(row.get("securityId", row.get("security_id", "")))
            ltp = row.get("last_price", row.get("lastPrice", row.get("ltp")))
            if sid and ltp is not None:
                out[sid] = float(ltp)
    return out

def chart(security_id, interval, start, end):
    body = {"securityId": str(security_id), "exchangeSegment": "NSE_EQ",
            "instrument": "EQUITY", "fromDate": start.strftime("%Y-%m-%d %H:%M:%S"),
            "toDate": end.strftime("%Y-%m-%d %H:%M:%S"), "interval": str(interval)}
    payload = api_post("/charts/historical" if interval == "D" else "/charts/intraday", body)
    data = payload.get("data", payload)
    # Dhan chart response commonly returns parallel arrays.
    if not isinstance(data, dict) or not all(k in data for k in ("open","high","low","close","timestamp")):
        raise RuntimeError(f"Unexpected Dhan chart response for {security_id}: {str(payload)[:300]}")
    df = pd.DataFrame({k: data[k] for k in ("timestamp","open","high","low","close")})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(TZ)
    df = df.set_index("timestamp").sort_index()
    df.columns = ["Open","High","Low","Close"]
    return df.astype(float)

def confirmed_pivot_high(df, left, right, lookback):
    # Return the latest confirmed pivot high; only candles with right bars after it qualify.
    if len(df) < left + right + 2: return None
    x = df.tail(lookback).reset_index(drop=True)
    highs = x["High"].tolist()
    found = []
    for i in range(left, len(highs)-right):
        h = highs[i]
        if all(h > highs[j] for j in range(i-left, i)) and all(h >= highs[j] for j in range(i+1, i+right+1)):
            found.append((i, h))
    return found[-1][1] if found else None

def daily_uptrend(daily):
    # Use completed daily candles only; never use today's unfinished daily candle.
    if len(daily) < 60: return False, None, None
    d = daily.copy()
    today = now().date()
    d = d[d.index.date < today]
    if len(d) < 55: return False, None, None
    ema = d["Close"].ewm(span=50, adjust=False).mean()
    close = float(d["Close"].iloc[-1])
    e_now, e_5 = float(ema.iloc[-1]), float(ema.iloc[-6])
    return close > e_now and e_now > e_5, close, e_now

def ensure_csv(path, fields):
    if not path.exists():
        with path.open("w", newline="") as f: csv.DictWriter(f, fieldnames=fields).writeheader()

def append_csv(path, fields, row):
    ensure_csv(path, fields)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writerow({k: row.get(k, "") for k in fields})

TRADE_FIELDS = ["trade_id","strategy","symbol","signal_date","entry_date","entry_time",
 "daily_close","daily_ema50","resistance","entry_price","target_price","target_move",
 "quantity","capital_used","target_profit","exit_date","exit_time","exit_price",
 "gross_pnl","net_pnl","status","exit_reason","holding_minutes"]
EVENT_FIELDS = ["timestamp","strategy","symbol","event","price","resistance","detail"]
STATE_FIELDS = ["strategy","symbol","entry_date","entry_time","entry_price","target_price",
 "quantity","capital_used","daily_close","daily_ema50","resistance","trade_id"]

def load_open(path):
    if not path.exists(): return {}
    df = pd.read_csv(path)
    return {str(r["symbol"]): r.to_dict() for _,r in df.iterrows()} if len(df) else {}

def save_open(path, positions):
    pd.DataFrame(list(positions.values()), columns=STATE_FIELDS).to_csv(path, index=False)

def log_event(strategy, symbol, event, price="", resistance="", detail=""):
    append_csv(OUT / f"{strategy}_events.csv", EVENT_FIELDS,
       {"timestamp":now().isoformat(timespec="seconds"),"strategy":strategy,"symbol":symbol,
        "event":event,"price":price,"resistance":resistance,"detail":detail})

def run():
    universe = read_universe()
    if not universe: raise RuntimeError("universe.csv has no stocks.")
    strategies = {
      "daily_resistance": {"interval":"D", "lookback":DAILY_LOOKBACK},
      "15m_resistance": {"interval":"15", "lookback":400},
    }
    books = {s: load_open(OUT/f"{s}_open.csv") for s in strategies}
    trades = {s: OUT/f"{s}_trades.csv" for s in strategies}
    for s in strategies: ensure_csv(trades[s], TRADE_FIELDS)

    # Cache historical candles, refresh 15m history every 15 minutes.
    cache = {}
    last_refresh = {}
    last_ltp = {}
    trade_seq = int(now().strftime("%y%m%d")) * 10000
    LOG.info("Paper-only run started. Universe=%d; polling=%ss", len(universe), POLL_SECONDS)

    while True:
        t = now()
        if t.weekday() >= 5 or not (dtime(9,15) <= t.time() <= dtime(15,30)):
            if t.time() > dtime(15,30):
                LOG.info("Market session ended; stopping.")
                break
            time.sleep(30); continue
        try:
            prices = quote_batch(universe)
        except Exception as e:
            LOG.exception("Quote request failed: %s", e); time.sleep(POLL_SECONDS); continue

        for item in universe:
            sym, sid = item["symbol"], item["security_id"]
            ltp = prices.get(sid)
            if ltp is None or ltp <= 0: continue
            for strategy, cfg in strategies.items():
                key = (strategy, sid)
                refresh = key not in cache or strategy == "15m_resistance" and (
                    t - last_refresh.get(key, t-timedelta(minutes=16)) >= timedelta(minutes=15))
                if refresh:
                    try:
                        if strategy == "daily_resistance":
                            hist = chart(sid, "D", t-timedelta(days=400), t)
                        else:
                            hist = chart(sid, "15", t-timedelta(days=INTRADAY_LOOKBACK_DAYS), t)
                        cache[key] = hist
                        last_refresh[key] = t
                    except Exception as e:
                        LOG.warning("%s %s chart unavailable: %s", strategy, sym, e); continue
                hist = cache[key]
                if strategy == "daily_resistance":
                    # daily swing high is based on completed daily bars, not today's partial bar
                    d = hist[hist.index.date < t.date()]
                    resistance = confirmed_pivot_high(d, PIVOT_LEFT, PIVOT_RIGHT, DAILY_LOOKBACK)
                else:
                    # Ignore the current incomplete 15-minute candle.
                    bars = hist[hist.index < pd.Timestamp(t).floor("15min")]
                    resistance = confirmed_pivot_high(bars, PIVOT_LEFT, PIVOT_RIGHT, 400)
                daily = cache.get(("daily_resistance", sid))
                if daily is None:
                    try:
                        daily = chart(sid, "D", t-timedelta(days=400), t)
                        cache[("daily_resistance",sid)] = daily
                    except Exception: continue
                up, dclose, ema50 = daily_uptrend(daily)
                if not up or resistance is None: continue
                pos = books[strategy].get(sym)
                if pos:
                    if ltp >= float(pos["target_price"]):
                        qty = int(pos["quantity"]); entry = float(pos["entry_price"])
                        target = float(pos["target_price"])
                        pnl = (target-entry)*qty
                        entered = datetime.fromisoformat(str(pos["entry_date"])+"T"+str(pos["entry_time"])).replace(tzinfo=TZ)
                        minutes = int((t-entered).total_seconds()/60)
                        append_csv(trades[strategy], TRADE_FIELDS, {
                          **pos,"exit_date":t.date().isoformat(),"exit_time":t.strftime("%H:%M:%S"),
                          "exit_price":target,"gross_pnl":round(pnl,2),"net_pnl":round(pnl,2),
                          "status":"CLOSED","exit_reason":"TARGET","holding_minutes":minutes})
                        log_event(strategy,sym,"PAPER_SELL_TARGET",target,resistance,f"qty={qty}; pnl={pnl:.2f}")
                        del books[strategy][sym]
                    continue
                prev = last_ltp.get((strategy,sym))
                if prev is not None and prev <= resistance < ltp:
                    qty = min(int(TARGET_PROFIT // TARGET_MOVE), int(MAX_POSITION_VALUE // ltp),
                               int(STARTING_CAPITAL // ltp))
                    if qty < 1: continue
                    trade_seq += 1
                    entry = float(ltp); target = entry + TARGET_MOVE
                    pos = {"strategy":strategy,"symbol":sym,"signal_date":t.date().isoformat(),
                      "entry_date":t.date().isoformat(),"entry_time":t.strftime("%H:%M:%S"),
                      "daily_close":round(dclose,2),"daily_ema50":round(ema50,2),
                      "resistance":round(resistance,2),"entry_price":round(entry,2),
                      "target_price":round(target,2),"target_move":TARGET_MOVE,"quantity":qty,
                      "capital_used":round(entry*qty,2),"target_profit":round((target-entry)*qty,2),
                      "trade_id":trade_seq}
                    books[strategy][sym] = pos
                    append_csv(trades[strategy],TRADE_FIELDS,pos)
                    log_event(strategy,sym,"PAPER_BUY",entry,resistance,
                              f"qty={qty}; target={target:.2f}; capital={entry*qty:.2f}")
                last_ltp[(strategy,sym)] = ltp

        for s in strategies: save_open(OUT/f"{s}_open.csv",books[s])
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    run()

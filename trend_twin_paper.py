
"""
TrendTwin Dhan Paper Trader
PAPER TRADING ONLY — NO REAL ORDERS ARE EVER PLACED.

Input:
  EQUITY_L.csv (NSE equity list)
  Dhan instrument master (for security ID matching)

Output:
  paper_logs/daily_resistance_trades.csv
  paper_logs/15m_resistance_trades.csv
  paper_logs/daily_resistance_open.csv
  paper_logs/15m_resistance_open.csv
  paper_logs/*_events.csv

Required environment:
  DHAN_CLIENT_ID
  DHAN_ACCESS_TOKEN
"""

import os
import csv
import time
import logging
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

import requests
import pandas as pd

# =========================================================
# SETTINGS
# =========================================================

TZ = ZoneInfo("Asia/Kolkata")
BASE_URL = "https://api.dhan.co/v2"

CLIENT_ID = os.environ["DHAN_CLIENT_ID"]
ACCESS_TOKEN = os.environ["DHAN_ACCESS_TOKEN"]

NSE_FILE = os.getenv("NSE_FILE", "EQUITY_L.csv")
UNIVERSE_FILE = os.getenv("UNIVERSE_FILE", "universe.csv")

DHAN_MASTER_URL = (
    "https://images.dhan.co/api-data/api-scrip-master.csv"
)

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "15"))

STARTING_CAPITAL = 500_000.0
TARGET_PROFIT = 2_000.0
TARGET_MOVE = 20.0
MAX_POSITION_VALUE = 100_000.0

PIVOT_LEFT = 3
PIVOT_RIGHT = 3
DAILY_LOOKBACK = 120
INTRADAY_LOOKBACK_DAYS = 15

OUT = Path("paper_logs")
OUT.mkdir(exist_ok=True)

LOG = logging.getLogger("TrendTwin")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

HEADERS = {
    "access-token": ACCESS_TOKEN,
    "client-id": CLIENT_ID,
    "Content-Type": "application/json",
    "Accept": "application/json"
}

# =========================================================
# TIME
# =========================================================

def now():
    return datetime.now(TZ)


# =========================================================
# DHAN API
# =========================================================

def api_post(path, body):
    response = requests.post(
        BASE_URL + path,
        headers=HEADERS,
        json=body,
        timeout=30
    )

    response.raise_for_status()
    payload = response.json()

    if (
        isinstance(payload, dict)
        and str(payload.get("status", "")).lower() == "failure"
    ):
        raise RuntimeError(f"Dhan API failure: {payload}")

    return payload


# =========================================================
# AUTOMATIC NSE UNIVERSE GENERATION
# =========================================================

def normalize_symbol(value):
    return str(value).strip().upper()


def create_universe():
    nse_path = Path(NSE_FILE)

    if not nse_path.exists():
        raise FileNotFoundError(
            f"{NSE_FILE} not found. Put it beside this script."
        )

    nse = pd.read_csv(nse_path, dtype=str)
    nse.columns = [str(c).strip().upper() for c in nse.columns]

    required = {"SYMBOL", "SERIES"}

    if not required.issubset(set(nse.columns)):
        raise ValueError(
            f"{NSE_FILE} must contain SYMBOL and SERIES columns. "
            f"Found: {list(nse.columns)}"
        )

    nse["SYMBOL"] = nse["SYMBOL"].map(normalize_symbol)
    nse["SERIES"] = nse["SERIES"].astype(str).str.strip().str.upper()

    nse = nse[nse["SERIES"] == "EQ"]
    nse = nse.drop_duplicates(subset=["SYMBOL"])

    LOG.info("NSE EQ symbols found: %d", len(nse))

    LOG.info("Downloading Dhan instrument master...")
    master = pd.read_csv(DHAN_MASTER_URL, dtype=str)
    master.columns = [str(c).strip().upper() for c in master.columns]

    required_master = {
        "SEM_EXM_EXCH_ID",
        "SEM_SEGMENT",
        "SEM_SMST_SECURITY_ID",
        "SEM_TRADING_SYMBOL"
    }

    missing = required_master - set(master.columns)

    if missing:
        raise ValueError(
            "Dhan instrument master format has changed. "
            f"Missing columns: {sorted(missing)}. "
            f"Available columns: {list(master.columns)}"
        )

    master["SEM_TRADING_SYMBOL"] = (
        master["SEM_TRADING_SYMBOL"].map(normalize_symbol)
    )

    master["SEM_EXM_EXCH_ID"] = (
        master["SEM_EXM_EXCH_ID"].astype(str).str.strip().str.upper()
    )

    master["SEM_SEGMENT"] = (
        master["SEM_SEGMENT"].astype(str).str.strip().str.upper()
    )

    # Dhan's NSE equity segment is normally E.
    # Validate this filter against the downloaded master if
    # the resulting match count is unexpectedly low.
    master = master[
        (master["SEM_EXM_EXCH_ID"] == "NSE") &
        (master["SEM_SEGMENT"] == "E")
    ].copy()

    if "SEM_INSTRUMENT_NAME" in master.columns:
        master["SEM_INSTRUMENT_NAME"] = (
            master["SEM_INSTRUMENT_NAME"]
            .astype(str).str.strip().str.upper()
        )
        equity_rows = master[
            master["SEM_INSTRUMENT_NAME"] == "EQUITY"
        ]
        if not equity_rows.empty:
            master = equity_rows

    master = master.drop_duplicates(
        subset=["SEM_TRADING_SYMBOL"]
    )

    merged = nse.merge(
        master[
            ["SEM_TRADING_SYMBOL", "SEM_SMST_SECURITY_ID"]
        ],
        left_on="SYMBOL",
        right_on="SEM_TRADING_SYMBOL",
        how="left"
    )

    merged = merged.dropna(
        subset=["SEM_SMST_SECURITY_ID"]
    )

    merged["security_id"] = (
        merged["SEM_SMST_SECURITY_ID"].astype(str).str.strip()
    )

    result = merged[["SYMBOL", "security_id"]].copy()
    result.columns = ["symbol", "security_id"]

    result.to_csv(UNIVERSE_FILE, index=False)

    LOG.info(
        "Universe created: %d matched stocks; %d unmatched.",
        len(result),
        len(nse) - len(result)
    )

    if result.empty:
        raise RuntimeError(
            "No NSE symbols matched Dhan's instrument master. "
            "Check the master columns and segment filter."
        )

    return [
        {
            "symbol": normalize_symbol(row.symbol),
            "security_id": str(row.security_id).strip()
        }
        for row in result.itertuples(index=False)
    ]


def read_universe():
    # Rebuild the universe from the NSE EQ file each run.
    return create_universe()


# =========================================================
# LIVE QUOTES
# =========================================================

def quote_batch(items):
    result = {}

    for i in range(0, len(items), 100):
        batch = items[i:i + 100]
        ids = [int(x["security_id"]) for x in batch]

        payload = api_post(
            "/marketfeed/quote",
            {"NSE_EQ": ids}
        )

        data = payload.get("data", {})

        rows = data.get("NSE_EQ", data) if isinstance(data, dict) else data

        if isinstance(rows, dict):
            # Dhan may return a dictionary keyed by security ID.
            for key, value in rows.items():
                if not isinstance(value, dict):
                    continue

                sid = str(
                    value.get(
                        "securityId",
                        value.get("security_id", key)
                    )
                )

                ltp = value.get(
                    "last_price",
                    value.get("lastPrice", value.get("ltp"))
                )

                if ltp is not None:
                    result[sid] = float(ltp)

        elif isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue

                sid = str(
                    row.get("securityId", row.get("security_id", ""))
                )

                ltp = row.get(
                    "last_price",
                    row.get("lastPrice", row.get("ltp"))
                )

                if sid and ltp is not None:
                    result[sid] = float(ltp)

    return result


# =========================================================
# HISTORICAL CHART DATA
# =========================================================

def chart(security_id, interval, start, end):
    body = {
        "securityId": str(security_id),
        "exchangeSegment": "NSE_EQ",
        "instrument": "EQUITY",
        "fromDate": start.strftime("%Y-%m-%d %H:%M:%S"),
        "toDate": end.strftime("%Y-%m-%d %H:%M:%S"),
        "interval": str(interval)
    }

    endpoint = (
        "/charts/historical"
        if interval == "D"
        else "/charts/intraday"
    )

    payload = api_post(endpoint, body)
    data = payload.get("data", payload)

    required = ("open", "high", "low", "close", "timestamp")

    if not isinstance(data, dict) or not all(
        key in data for key in required
    ):
        raise RuntimeError(
            f"Unexpected chart response for {security_id}: "
            f"{str(payload)[:300]}"
        )

    df = pd.DataFrame({
        key: data[key] for key in required
    })

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        utc=True
    ).dt.tz_convert(TZ)

    df = df.set_index("timestamp").sort_index()
    df.columns = ["Open", "High", "Low", "Close"]

    return df.astype(float)


# =========================================================
# RESISTANCE CALCULATION
# =========================================================

def confirmed_pivot_high(df, left, right, lookback):
    if len(df) < left + right + 2:
        return None

    x = df.tail(lookback).reset_index(drop=True)
    highs = x["High"].tolist()

    found = []

    for i in range(left, len(highs) - right):
        h = highs[i]

        left_ok = all(
            h > highs[j]
            for j in range(i - left, i)
        )

        right_ok = all(
            h >= highs[j]
            for j in range(i + 1, i + right + 1)
        )

        if left_ok and right_ok:
            found.append((i, h))

    return found[-1][1] if found else None


# =========================================================
# DAILY TREND FILTER
# =========================================================

def daily_uptrend(daily):
    if len(daily) < 60:
        return False, None, None

    d = daily.copy()
    today = now().date()

    # Completed daily candles only.
    d = d[d.index.date < today]

    if len(d) < 55:
        return False, None, None

    ema = d["Close"].ewm(
        span=50,
        adjust=False
    ).mean()

    close = float(d["Close"].iloc[-1])
    ema_now = float(ema.iloc[-1])
    ema_5 = float(ema.iloc[-6])

    is_uptrend = (
        close > ema_now and
        ema_now > ema_5
    )

    return is_uptrend, close, ema_now


# =========================================================
# CSV HELPERS
# =========================================================

def ensure_csv(path, fields):
    if not path.exists():
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fields
            )
            writer.writeheader()


def append_csv(path, fields, row):
    ensure_csv(path, fields)

    with path.open("a", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            extrasaction="ignore"
        )

        writer.writerow({
            key: row.get(key, "")
            for key in fields
        })


TRADE_FIELDS = [
    "trade_id", "strategy", "symbol", "signal_date",
    "entry_date", "entry_time", "daily_close", "daily_ema50",
    "resistance", "entry_price", "target_price", "target_move",
    "quantity", "capital_used", "target_profit",
    "exit_date", "exit_time", "exit_price",
    "gross_pnl", "net_pnl", "status", "exit_reason",
    "holding_minutes"
]

EVENT_FIELDS = [
    "timestamp", "strategy", "symbol", "event",
    "price", "resistance", "detail"
]

STATE_FIELDS = [
    "strategy", "symbol", "entry_date", "entry_time",
    "entry_price", "target_price", "quantity",
    "capital_used", "daily_close", "daily_ema50",
    "resistance", "trade_id", "signal_date",
    "target_move", "target_profit"
]


def load_open(path):
    if not path.exists():
        return {}

    df = pd.read_csv(path)

    if df.empty:
        return {}

    return {
        str(row["symbol"]): row.to_dict()
        for _, row in df.iterrows()
    }


def save_open(path, positions):
    pd.DataFrame(
        list(positions.values()),
        columns=STATE_FIELDS
    ).to_csv(path, index=False)


def log_event(strategy, symbol, event, price="", resistance="", detail=""):
    append_csv(
        OUT / f"{strategy}_events.csv",
        EVENT_FIELDS,
        {
            "timestamp": now().isoformat(timespec="seconds"),
            "strategy": strategy,
            "symbol": symbol,
            "event": event,
            "price": price,
            "resistance": resistance,
            "detail": detail
        }
    )


# =========================================================
# PAPER TRADING ENGINE
# =========================================================

def run():
    universe = read_universe()

    if not universe:
        raise RuntimeError("No stocks in universe.")

    strategies = {
        "daily_resistance": {
            "interval": "D",
            "lookback": DAILY_LOOKBACK
        },
        "15m_resistance": {
            "interval": "15",
            "lookback": 400
        }
    }

    books = {
        strategy: load_open(OUT / f"{strategy}_open.csv")
        for strategy in strategies
    }

    trades = {
        strategy: OUT / f"{strategy}_trades.csv"
        for strategy in strategies
    }

    for strategy in strategies:
        ensure_csv(trades[strategy], TRADE_FIELDS)

    cache = {}
    last_refresh = {}
    last_ltp = {}

    trade_seq = int(now().strftime("%y%m%d")) * 10000

    LOG.info(
        "Paper-only run started. Universe=%d; polling=%ss",
        len(universe),
        POLL_SECONDS
    )

    while True:
        t = now()

        if t.weekday() >= 5:
            LOG.info("Weekend. Stopping.")
            break

        if t.time() < dtime(9, 15):
            time.sleep(30)
            continue

        if t.time() > dtime(15, 30):
            LOG.info("Market session ended; stopping.")
            break

        try:
            prices = quote_batch(universe)

        except Exception as error:
            LOG.exception("Quote request failed: %s", error)
            time.sleep(POLL_SECONDS)
            continue

        for item in universe:
            sym = item["symbol"]
            sid = item["security_id"]

            ltp = prices.get(sid)

            if ltp is None or ltp <= 0:
                continue

            # Fetch daily history once per stock.
            daily_key = ("daily_history", sid)

            if daily_key not in cache:
                try:
                    cache[daily_key] = chart(
                        sid,
                        "D",
                        t - timedelta(days=400),
                        t
                    )
                except Exception as error:
                    LOG.warning(
                        "%s daily chart unavailable: %s",
                        sym, error
                    )
                    continue

            daily = cache[daily_key]
            up, dclose, ema50 = daily_uptrend(daily)

            if not up:
                continue

            for strategy, cfg in strategies.items():
                key = (strategy, sid)

                refresh = (
                    key not in cache or
                    (
                        strategy == "15m_resistance"
                        and t - last_refresh.get(
                            key,
                            t - timedelta(minutes=16)
                        ) >= timedelta(minutes=15)
                    )
                )

                if refresh:
                    try:
                        hist = chart(
                            sid,
                            cfg["interval"],
                            t - timedelta(
                                days=400 if strategy == "daily_resistance"
                                else INTRADAY_LOOKBACK_DAYS
                            ),
                            t
                        )

                        cache[key] = hist
                        last_refresh[key] = t

                    except Exception as error:
                        LOG.warning(
                            "%s %s chart unavailable: %s",
                            strategy, sym, error
                        )
                        continue

                hist = cache[key]

                if strategy == "daily_resistance":
                    bars = hist[hist.index.date < t.date()]

                else:
                    # Ignore the current unfinished 15-minute candle.
                    cutoff = pd.Timestamp(t).floor("15min")
                    bars = hist[hist.index < cutoff]

                resistance = confirmed_pivot_high(
                    bars,
                    PIVOT_LEFT,
                    PIVOT_RIGHT,
                    cfg["lookback"]
                )

                if resistance is None:
                    continue

                pos = books[strategy].get(sym)

                # -----------------------------------------
                # EXIT: TARGET HIT
                # -----------------------------------------

                if pos:
                    target = float(pos["target_price"])

                    if ltp >= target:
                        qty = int(pos["quantity"])
                        entry = float(pos["entry_price"])

                        pnl = (target - entry) * qty

                        entered = datetime.fromisoformat(
                            str(pos["entry_date"]) +
                            "T" +
                            str(pos["entry_time"])
                        ).replace(tzinfo=TZ)

                        minutes = int(
                            (t - entered).total_seconds() / 60
                        )

                        append_csv(
                            trades[strategy],
                            TRADE_FIELDS,
                            {
                                **pos,
                                "exit_date": t.date().isoformat(),
                                "exit_time": t.strftime("%H:%M:%S"),
                                "exit_price": round(target, 2),
                                "gross_pnl": round(pnl, 2),
                                "net_pnl": round(pnl, 2),
                                "status": "CLOSED",
                                "exit_reason": "TARGET",
                                "holding_minutes": minutes
                            }
                        )

                        log_event(
                            strategy,
                            sym,
                            "PAPER_SELL_TARGET",
                            target,
                            resistance,
                            f"qty={qty}; pnl={pnl:.2f}"
                        )

                        del books[strategy][sym]

                    last_ltp[(strategy, sym)] = ltp
                    continue

                # -----------------------------------------
                # ENTRY: LTP CROSSES RESISTANCE
                # -----------------------------------------

                previous = last_ltp.get((strategy, sym))

                if previous is not None and previous <= resistance < ltp:

                    # Quantity based on ₹2,000 target at ₹20 move.
                    # Capital and per-position limits can reduce it.
                    qty_for_profit = int(
                        TARGET_PROFIT // TARGET_MOVE
                    )

                    qty_for_position = int(
                        MAX_POSITION_VALUE // ltp
                    )

                    qty_for_capital = int(
                        STARTING_CAPITAL // ltp
             

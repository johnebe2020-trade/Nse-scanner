"""
micro_cap_scanner.py
Runs after market close (4PM IST) via GitHub Actions.

Flags micro/small-cap stocks showing:
 1. HH/HL structure forming
 2. Volume surge (confirmed or expected)
 3. Bounce from strong support
 4. Approaching a demand zone

Data source: Yahoo Finance v8/finance/chart (direct, no yfinance lib —
avoids curl_cffi compile issues, consistent with other scanners in this repo)
"""

import requests
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime

# ---------------- CONFIG ----------------
LOOKBACK_DAYS = 120          # history pulled per stock
PIVOT_WINDOW = 3             # bars each side for swing high/low detection
VOL_SURGE_MULT = 2.0         # today's vol >= this x 20d avg -> confirmed surge
VOL_DRYUP_LOOKBACK = 15      # bars to check for 5d-avg vol at multi-week low
SUPPORT_TOUCH_TOLERANCE = 0.015   # 1.5% band to count as "touching" a level
SUPPORT_MIN_TOUCHES = 2
DEMAND_ZONE_PROXIMITY = 0.04      # within 4% above zone top = "approaching"
STOCK_LIST_CSV = "microcap_universe.csv"   # optional: column named "symbol"
OUTPUT_CSV = f"scan_results_{datetime.now().strftime('%Y%m%d')}.csv"
MIN_SCORE_ALERT = 2          # only send stocks scoring >= this to Telegram

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
HEADERS = {"User-Agent": "Mozilla/5.0"}

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# Fallback list — micro + small cap. Used only if microcap_universe.csv
# is not found in the repo. Edit freely, or replace with a CSV for full control.
DEFAULT_STOCK_LIST = [
    # Micro caps
    "RVNL.NS", "IRFC.NS", "SUZLON.NS", "YESBANK.NS", "IDEA.NS",
    "SOUTHBANK.NS", "PNB.NS", "IOB.NS", "UCOBANK.NS", "CENTRALBK.NS",
    "TTML.NS", "RPOWER.NS", "JPPOWER.NS", "IFCI.NS", "NHPC.NS",
    "SJVN.NS", "GMRINFRA.NS", "HUDCO.NS", "IRCON.NS", "RITES.NS",
    "MAZDOCK.NS", "COCHINSHIP.NS", "GRSE.NS", "BEML.NS",
    # Small caps
    "BHEL.NS", "SAIL.NS", "NATIONALUM.NS", "HINDCOPPER.NS", "MOIL.NS",
    "NMDC.NS", "RAILTEL.NS", "RCF.NS", "FACT.NS", "GNFC.NS",
    "GSFC.NS", "NFL.NS", "BALRAMCHIN.NS", "TRIVENI.NS",
    "DCMSHRIRAM.NS", "SHYAMMETL.NS", "JINDALSAW.NS", "RATNAMANI.NS",
    "WELCORP.NS", "APLAPOLLO.NS", "SURYAROSNI.NS", "KIRLOSENG.NS",
    "GRAPHITE.NS", "HEG.NS", "CENTURYPLY.NS", "GREENPANEL.NS",
    "ORIENTCEM.NS", "JKCEMENT.NS", "HEIDELBERG.NS", "PRSMJOHNSN.NS",
]


# ---------------- STOCK LIST ----------------
def load_stock_list():
    if os.path.exists(STOCK_LIST_CSV):
        try:
            symbols = pd.read_csv(STOCK_LIST_CSV)["symbol"].tolist()
            print(f"Loaded {len(symbols)} symbols from {STOCK_LIST_CSV}")
            return symbols
        except Exception as e:
            print(f"Found {STOCK_LIST_CSV} but couldn't read it ({e}); using default list.")
    else:
        print(f"{STOCK_LIST_CSV} not found — using hardcoded DEFAULT_STOCK_LIST ({len(DEFAULT_STOCK_LIST)} symbols).")
    return DEFAULT_STOCK_LIST


# ---------------- DATA FETCH ----------------
def fetch_daily_data(symbol, days=LOOKBACK_DAYS):
    params = {
        "range": f"{days}d",
        "interval": "1d",
        "includePrePost": "false"
    }
    try:
        r = requests.get(YAHOO_URL.format(symbol=symbol), params=params,
                          headers=HEADERS, timeout=10)
        data = r.json()
        result = data["chart"]["result"][0]
        ts = result["timestamp"]
        q = result["indicators"]["quote"][0]

        df = pd.DataFrame({
            "date": pd.to_datetime(ts, unit="s"),
            "open": q["open"],
            "high": q["high"],
            "low": q["low"],
            "close": q["close"],
            "volume": q["volume"],
        }).dropna()

        return df.reset_index(drop=True)
    except Exception as e:
        print(f"  [!] fetch failed for {symbol}: {e}")
        return None


# ---------------- 1. HH/HL STRUCTURE ----------------
def find_swing_points(df, window=PIVOT_WINDOW):
    highs, lows = [], []
    for i in range(window, len(df) - window):
        seg_h = df["high"].iloc[i - window: i + window + 1]
        seg_l = df["low"].iloc[i - window: i + window + 1]
        if df["high"].iloc[i] == seg_h.max():
            highs.append((i, df["high"].iloc[i]))
        if df["low"].iloc[i] == seg_l.min():
            lows.append((i, df["low"].iloc[i]))
    return highs, lows


def check_hh_hl(df):
    highs, lows = find_swing_points(df)
    if len(highs) < 2 or len(lows) < 2:
        return False, None

    last2_highs = [h[1] for h in highs[-2:]]
    last2_lows = [l[1] for l in lows[-2:]]

    is_hh = last2_highs[1] > last2_highs[0]
    is_hl = last2_lows[1] > last2_lows[0]

    detail = f"HH: {last2_highs[0]:.2f}->{last2_highs[1]:.2f} | HL: {last2_lows[0]:.2f}->{last2_lows[1]:.2f}"
    return (is_hh and is_hl), detail


# ---------------- 2. VOLUME SURGE ----------------
def check_volume_surge(df):
    if len(df) < 25:
        return False, None

    avg20 = df["volume"].iloc[-21:-1].mean()
    today_vol = df["volume"].iloc[-1]
    vol_ratio = today_vol / avg20 if avg20 > 0 else 0

    confirmed = vol_ratio >= VOL_SURGE_MULT

    df = df.copy()
    df["vol_5d_avg"] = df["volume"].rolling(5).mean()
    recent_5d_avgs = df["vol_5d_avg"].iloc[-VOL_DRYUP_LOOKBACK:]
    current_5d_avg = df["vol_5d_avg"].iloc[-1]
    expected = (current_5d_avg <= recent_5d_avgs.min() * 1.05) and not confirmed

    if confirmed:
        return True, f"CONFIRMED surge, {vol_ratio:.1f}x 20d avg"
    elif expected:
        return True, f"EXPECTED (vol dried up to {VOL_DRYUP_LOOKBACK}d low, watch for breakout)"
    else:
        return False, None


# ---------------- 3. SUPPORT BOUNCE ----------------
def find_support_levels(df, tolerance=SUPPORT_TOUCH_TOLERANCE, min_touches=SUPPORT_MIN_TOUCHES):
    _, lows = find_swing_points(df)
    if not lows:
        return []

    levels = []
    used = set()
    low_vals = [l[1] for l in lows]

    for i, lv in enumerate(low_vals):
        if i in used:
            continue
        cluster = [lv]
        cluster_idx = {i}
        for j, lv2 in enumerate(low_vals):
            if j != i and j not in used and abs(lv2 - lv) / lv <= tolerance:
                cluster.append(lv2)
                cluster_idx.add(j)
        if len(cluster) >= min_touches:
            levels.append(np.mean(cluster))
            used |= cluster_idx

    return levels


def check_support_bounce(df):
    levels = find_support_levels(df)
    if not levels:
        return False, None

    today = df.iloc[-1]
    candle_range = today["high"] - today["low"]
    if candle_range == 0:
        return False, None

    is_bullish = today["close"] > today["open"]
    closed_upper_half = (today["close"] - today["low"]) / candle_range >= 0.5

    for lvl in levels:
        touched = abs(today["low"] - lvl) / lvl <= SUPPORT_TOUCH_TOLERANCE
        if touched and is_bullish and closed_upper_half:
            return True, f"Bounced from support @ {lvl:.2f} (touches: strong level)"

    return False, None


# ---------------- 4. DEMAND ZONE PROXIMITY ----------------
def find_demand_zones(df, base_max_bars=5, min_rally_pct=0.08):
    zones = []
    for i in range(base_max_bars, len(df) - 3):
        for base_len in range(2, base_max_bars + 1):
            base = df.iloc[i - base_len:i]
            base_range = base["high"].max() - base["low"].min()
            base_mid = base["close"].mean()
            if base_mid == 0:
                continue
            tightness = base_range / base_mid
            if tightness > 0.035:
                continue

            post = df.iloc[i:i + 3]
            if post.empty:
                continue
            move_pct = (post["close"].iloc[-1] - base["close"].iloc[-1]) / base["close"].iloc[-1]
            if move_pct >= min_rally_pct:
                zones.append({
                    "top": base["high"].max(),
                    "bottom": base["low"].min(),
                    "idx": i
                })
    return zones


def check_demand_zone_approach(df):
    zones = find_demand_zones(df)
    if not zones:
        return False, None

    current_price = df["close"].iloc[-1]
    zones_below = [z for z in zones if z["top"] < current_price]
    if not zones_below:
        return False, None

    nearest = max(zones_below, key=lambda z: z["idx"])
    dist_pct = (current_price - nearest["top"]) / nearest["top"]

    if 0 <= dist_pct <= DEMAND_ZONE_PROXIMITY:
        return True, f"Approaching demand zone {nearest['bottom']:.2f}-{nearest['top']:.2f} ({dist_pct*100:.1f}% away)"

    return False, None


# ---------------- SCAN ----------------
def scan_stock(symbol):
    df = fetch_daily_data(symbol)
    if df is None or len(df) < 30:
        return None

    hh_hl, hh_hl_detail = check_hh_hl(df)
    vol_surge, vol_detail = check_volume_surge(df)
    support_bounce, support_detail = check_support_bounce(df)
    demand_zone, demand_detail = check_demand_zone_approach(df)

    score = sum([hh_hl, vol_surge, support_bounce, demand_zone])
    if score == 0:
        return None

    return {
        "symbol": symbol,
        "close": df["close"].iloc[-1],
        "score": score,
        "hh_hl": hh_hl_detail if hh_hl else "",
        "vol_surge": vol_detail if vol_surge else "",
        "support_bounce": support_detail if support_bounce else "",
        "demand_zone": demand_detail if demand_zone else "",
    }


# ---------------- TABLE FORMAT ----------------
def build_table_text(df):
    """Fixed-width plain-text table for Telegram (monospace <pre> block)."""
    headers = ["Symbol", "Close", "Score", "Tags"]
    rows = []
    for _, r in df.iterrows():
        tags = []
        if r["hh_hl"]:
            tags.append("HH/HL")
        if r["vol_surge"]:
            tags.append("VOLc" if "CONFIRMED" in r["vol_surge"] else "VOLe")
        if r["support_bounce"]:
            tags.append("SUP")
        if r["demand_zone"]:
            tags.append("DEM")
        rows.append([
            r["symbol"].replace(".NS", ""),
            f"{r['close']:.2f}",
            str(r["score"]),
            "+".join(tags)
        ])

    col_widths = [max(len(str(x)) for x in [h] + [row[i] for row in rows])
                  for i, h in enumerate(headers)]

    def fmt_row(row):
        return "  ".join(str(v).ljust(w) for v, w in zip(row, col_widths))

    lines = [fmt_row(headers), "-" * (sum(col_widths) + 2 * (len(headers) - 1))]
    lines += [fmt_row(row) for row in rows]
    return "\n".join(lines)


# ---------------- TELEGRAM ----------------
def send_telegram_message(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[!] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — skipping send.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": f"<pre>{text}</pre>",
        "parse_mode": "HTML"
    }
    try:
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            print(f"[!] Telegram send failed: {r.text}")
    except Exception as e:
        print(f"[!] Telegram send error: {e}")


# ---------------- MAIN ----------------
def run_scan():
    symbols = load_stock_list()

    results = []
    for i, sym in enumerate(symbols):
        print(f"[{i+1}/{len(symbols)}] scanning {sym}...")
        res = scan_stock(sym)
        if res:
            results.append(res)
        time.sleep(0.3)   # be polite to Yahoo's endpoint

    if not results:
        print("No matches today.")
        send_telegram_message("Micro/Small-cap scan: no setups found today.")
        return

    out = pd.DataFrame(results).sort_values("score", ascending=False)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(out)} results to {OUTPUT_CSV}")

    full_table = build_table_text(out)
    print("\n" + full_table)

    alert_df = out[out["score"] >= MIN_SCORE_ALERT]
    if alert_df.empty:
        print(f"\nNo stocks met MIN_SCORE_ALERT={MIN_SCORE_ALERT}; not sending Telegram.")
        return

    table_text = build_table_text(alert_df)
    header = f"Micro/Small-cap Scan — {datetime.now().strftime('%d-%b-%Y')}\n\n"
    send_telegram_message(header + table_text)


if __name__ == "__main__":
    run_scan()

import requests
import pandas as pd
import os
import time
from datetime import datetime

os.makedirs("data", exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
BREAKOUT_TARGET_RATIO = 1.5
TOP_N = 5
ENTRY_FILE = "data/monthlylevels_paper_entries.csv"
LOG_FILE = "data/monthlylevels_paper_log.csv"

STOCK_LIST = [
"HDFCBANK.NS","SBIN.NS","PERSISTENT.NS","COFORGE.NS","ASHOKLEY.NS","M&M.NS",
"BEL.NS","COALINDIA.NS","TATAMOTORS.NS","RELIANCE.NS","TCS.NS","INFY.NS",
"ICICIBANK.NS","AXISBANK.NS","KOTAKBANK.NS","LT.NS","BAJFINANCE.NS","BAJAJFINSV.NS",
"MARUTI.NS","TITAN.NS","SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS",
"NTPC.NS","POWERGRID.NS","ONGC.NS","BPCL.NS","IOC.NS","HINDALCO.NS","JSWSTEEL.NS",
"TATASTEEL.NS","ADANIPORTS.NS","ADANIENT.NS","ULTRACEMCO.NS","GRASIM.NS",
"NESTLEIND.NS","BRITANNIA.NS","ITC.NS","HINDUNILVR.NS","ASIANPAINT.NS",
"WIPRO.NS","HCLTECH.NS","TECHM.NS","LTIM.NS","BHARTIARTL.NS","INDUSINDBK.NS",
"BAJAJ-AUTO.NS","EICHERMOT.NS","HEROMOTOCO.NS","DLF.NS","GODREJPROP.NS",
"SIEMENS.NS","ABB.NS","CUMMINSIND.NS","HAL.NS","BEML.NS","BHEL.NS",
"IRCTC.NS","IRFC.NS","ZOMATO.NS","TRENT.NS","VEDL.NS","SAIL.NS"
]

def fetch_monthly_data(symbol, retries=2):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"range": "1y", "interval": "1mo"}
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=10)
            data = r.json()
            result = data["chart"]["result"][0]
            ts = result["timestamp"]
            q = result["indicators"]["quote"][0]
            df = pd.DataFrame({"High": q["high"], "Low": q["low"], "Close": q["close"]}, index=pd.to_datetime(ts, unit="s"))
            df = df.dropna()
            if len(df) < 2:
                return None
            return df
        except Exception:
            time.sleep(1)
    return None

def fetch_daily_data(symbol, retries=2):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"range": "3mo", "interval": "1d"}
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=10)
            data = r.json()
            result = data["chart"]["result"][0]
            ts = result["timestamp"]
            q = result["indicators"]["quote"][0]
            df = pd.DataFrame({"High": q["high"], "Low": q["low"], "Close": q["close"]}, index=pd.to_datetime(ts, unit="s"))
            df = df.dropna()
            if len(df) < 3:
                return None
            return df
        except Exception:
            time.sleep(1)
    return None

def get_current_price(symbol):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"range": "1d", "interval": "1m"}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
        data = r.json()
        return round(float(data["chart"]["result"][0]["meta"]["regularMarketPrice"]), 2)
    except Exception:
        return None

def analyze(symbol, monthly_df, daily_df):
    prev_month = monthly_df.iloc[-2]
    level_0 = float(prev_month["Low"])
    level_1 = float(prev_month["High"])
    distance = level_1 - level_0
    level_032 = round(level_0 + 0.32 * distance, 2)
    level_068 = round(level_0 + 0.68 * distance, 2)
    today = daily_df.index[-1]
    month_start = pd.Timestamp(year=today.year, month=today.month, day=1)
    current_month_df = daily_df[daily_df.index >= month_start]
    if current_month_df.empty:
        return None
    today_close = float(current_month_df["Close"].iloc[-1])
    month_min_low = float(current_month_df["Low"].min())
    month_max_high = float(current_month_df["High"].max())
    swept_below_low = month_min_low < level_0
    swept_above_high = month_max_high > level_1
    reversal_buy = swept_below_low and today_close > level_032
    reversal_sell = swept_above_high and today_close < level_068
    breakout_buy = today_close > level_1
    breakout_sell = today_close < level_0
    if reversal_buy:
        signal = "REVERSAL BUY"; entry = today_close; target = level_068; stop_loss = level_0
    elif breakout_buy:
        signal = "BREAKOUT BUY"; entry = today_close; stop_loss = level_068
        risk = entry - stop_loss; target = round(entry + risk * BREAKOUT_TARGET_RATIO, 2)
    elif reversal_sell:
        signal = "AVOID (Reversal Sell)"; entry = None; target = level_032; stop_loss = level_1
    elif breakout_sell:
        signal = "AVOID (Breakout Sell)"; entry = None; target = None; stop_loss = level_032
    else:
        signal = "NEUTRAL"; entry = None; target = None; stop_loss = None
    return {
        "Symbol": symbol.replace(".NS", ""), "Date": today.strftime("%Y-%m-%d"), "CMP": round(today_close, 2),
        "MonthlyLow": round(level_0, 2), "BuyZone032": level_032, "SellZone068": level_068,
        "MonthlyHigh": round(level_1, 2), "Signal": signal, "Entry": entry, "Target": target, "StopLoss": stop_loss,
    }

def run_scan():
    results = []
    for i, sym in enumerate(STOCK_LIST, 1):
        print(f"[{i}/{len(STOCK_LIST)}] Scanning {sym} ...")
        monthly_df = fetch_monthly_data(sym)
        daily_df = fetch_daily_data(sym)
        if monthly_df is None or daily_df is None:
            continue
        try:
            res = analyze(sym, monthly_df, daily_df)
            if res:
                results.append(res)
        except Exception as e:
            print(f"  skip {sym}: {e}")
        time.sleep(0.4)
    dfres = pd.DataFrame(results)
    if dfres.empty:
        print("No data fetched.")
        return None
    buy_signals = dfres[dfres["Signal"].isin(["REVERSAL BUY", "BREAKOUT BUY"])]
    dfres.to_csv("data/monthlylevels_full.csv", index=False)
    buy_signals.to_csv("data/monthlylevels_buy.csv", index=False)
    print(f"BUY signals: {len(buy_signals)}")
    return buy_signals

def setup_entries(buy_signals):
    if os.path.exists(ENTRY_FILE):
        return pd.read_csv(ENTRY_FILE)
    if buy_signals is None or buy_signals.empty:
        return None
    top = buy_signals.head(TOP_N)
    rows = []
    for _, r in top.iterrows():
        if pd.isna(r["Entry"]) or pd.isna(r["Target"]) or pd.isna(r["StopLoss"]):
            continue
        rows.append({"Symbol": r["Symbol"], "EntryDate": datetime.now().strftime("%Y-%m-%d %H:%M"),
                      "Signal": r["Signal"], "BuyPrice": r["Entry"], "Target": r["Target"], "StopLoss": r["StopLoss"]})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.to_csv(ENTRY_FILE, index=False)
    return df

def run_paper_trade_check(entries):
    if entries is None or entries.empty:
        return
    today_rows = []
    for _, row in entries.iterrows():
        cur_price = get_current_price(row["Symbol"] + ".NS")
        if cur_price is None:
            continue
        pnl_pct = round((cur_price - row["BuyPrice"]) / row["BuyPrice"] * 100, 2)
        status = "HOLDING"
        if cur_price >= row["Target"]: status = "TARGET HIT"
        elif cur_price <= row["StopLoss"]: status = "SL HIT"
        today_rows.append({"Date": datetime.now().strftime("%Y-%m-%d %H:%M"), "Symbol": row["Symbol"],
                            "Signal": row["Signal"], "BuyPrice": row["BuyPrice"], "CurrentPrice": cur_price,
                            "Target": row["Target"], "StopLoss": row["StopLoss"], "PnL%": pnl_pct, "Status": status})
        print(f"{row['Symbol']:12s} Buy:{row['BuyPrice']:>9.2f} Now:{cur_price:>9.2f} P&L:{pnl_pct:>6.2f}% [{status}]")
    log_df = pd.DataFrame(today_rows)
    if log_df.empty:
        return
    if os.path.exists(LOG_FILE):
        log_df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        log_df.to_csv(LOG_FILE, index=False)

if __name__ == "__main__":
    print("=== MONTHLY LEVELS SCANNER ===")
    buy_signals = run_scan()
    entries = setup_entries(buy_signals)
    run_paper_trade_check(entries)

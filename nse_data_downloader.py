import io
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf


# ============================================================
# JOHN'S NSE 500 DATA DOWNLOADER
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

DATA_DIR.mkdir(exist_ok=True)

UNIVERSE_FILE = ROOT / "universe.csv"
OHLCV_FILE = DATA_DIR / "sample_ohlcv.csv"

# One year of daily history
PERIOD = "1y"

NIFTY500_URL = (
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/plain,*/*",
}


# ------------------------------------------------------------
# 1. DOWNLOAD NIFTY 500 UNIVERSE
# ------------------------------------------------------------

def download_nifty500_universe():

    print("Downloading Nifty 500 constituent list...")

    try:
        response = requests.get(
            NIFTY500_URL,
            headers=HEADERS,
            timeout=30
        )

        response.raise_for_status()

        df = pd.read_csv(io.BytesIO(response.content))

        print("Columns received:")
        print(list(df.columns))

        # Find symbol column safely
        symbol_column = None

        for col in df.columns:
            if str(col).strip().lower() == "symbol":
                symbol_column = col
                break

        if symbol_column is None:
            raise ValueError("Symbol column not found.")

        # Find industry/sector column
        industry_column = None

        for col in df.columns:
            name = str(col).strip().lower()

            if name in ["industry", "sector"]:
                industry_column = col
                break

        result = pd.DataFrame()

        result["symbol"] = (
            df[symbol_column]
            .astype(str)
            .str.strip()
        )

        if industry_column:
            result["sector"] = (
                df[industry_column]
                .astype(str)
                .str.strip()
            )
        else:
            result["sector"] = "Nifty 500"

        # Remove bad rows
        result = result[
            (result["symbol"] != "") &
            (result["symbol"].str.lower() != "nan")
        ]

        result = result.drop_duplicates(
            subset=["symbol"]
        )

        result.to_csv(
            UNIVERSE_FILE,
            index=False
        )

        print(
            f"Nifty 500 universe saved: "
            f"{len(result)} stocks"
        )

        return result

    except Exception as e:

        print("ERROR downloading Nifty 500 universe:")
        print(e)

        # If an old universe exists, use it
        if UNIVERSE_FILE.exists():

            print(
                "Using existing universe.csv instead."
            )

            return pd.read_csv(UNIVERSE_FILE)

        raise


# ------------------------------------------------------------
# 2. DOWNLOAD OHLCV
# ------------------------------------------------------------

def download_ohlcv(universe):

    symbols = (
        universe["symbol"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    print()
    print("=" * 60)
    print("Downloading historical OHLCV")
    print("=" * 60)

    print(f"Stocks: {len(symbols)}")
    print(f"Period: {PERIOD}")
    print()

    yahoo_symbols = [
        symbol + ".NS"
        for symbol in symbols
    ]

    all_data = []

    # Download in batches to reduce failures
    batch_size = 50

    for start in range(0, len(yahoo_symbols), batch_size):

        batch = yahoo_symbols[
            start:start + batch_size
        ]

        print(
            f"Downloading batch "
            f"{start + 1} - "
            f"{min(start + batch_size, len(yahoo_symbols))} "
            f"of {len(yahoo_symbols)}"
        )

        try:

            data = yf.download(
                tickers=batch,
                period=PERIOD,
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True
            )

            if data.empty:
                print("No data returned for this batch.")
                continue

            # ----------------------------------------
            # Multiple stocks
            # ----------------------------------------

            if isinstance(data.columns, pd.MultiIndex):

                for yahoo_symbol in batch:

                    if yahoo_symbol not in data.columns.levels[0]:
                        continue

                    stock = data[yahoo_symbol].copy()

                    stock = stock.reset_index()

                    required = [
                        "Date",
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume"
                    ]

                    if not all(
                        col in stock.columns
                        for col in required
                    ):
                        continue

                    stock = stock[required]

                    stock["symbol"] = (
                        yahoo_symbol.replace(".NS", "")
                    )

                    stock = stock.dropna(
                        subset=["Close"]
                    )

                    all_data.append(stock)

            # ----------------------------------------
            # Single stock fallback
            # ----------------------------------------

            else:

                stock = data.reset_index()

                required = [
                    "Date",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume"
                ]

                if all(
                    col in stock.columns
                    for col in required
                ):

                    stock = stock[required]

                    stock["symbol"] = (
                        batch[0].replace(".NS", "")
                    )

                    stock = stock.dropna(
                        subset=["Close"]
                    )

                    all_data.append(stock)

        except Exception as e:

            print(
                f"Batch failed: {e}"
            )

        time.sleep(1)

    if not all_data:

        raise RuntimeError(
            "No OHLCV data was downloaded."
        )

    # Combine everything
    final = pd.concat(
        all_data,
        ignore_index=True
    )

    # Standardize column names
    final = final.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }
    )

    final["date"] = pd.to_datetime(
        final["date"]
    ).dt.strftime("%Y-%m-%d")

    final = final[
        [
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    ]

    final = final.sort_values(
        ["symbol", "date"]
    )

    final = final.drop_duplicates(
        subset=["symbol", "date"]
    )

    final.to_csv(
        OHLCV_FILE,
        index=False
    )

    print()
    print("=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

    print(
        f"Stocks with data: "
        f"{final['symbol'].nunique()}"
    )

    print(
        f"Total candles: "
        f"{len(final)}"
    )

    print(
        f"Saved to: "
        f"{OHLCV_FILE}"
    )

    print()
    print(final.head())

    return final


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("JOHN'S NSE 500 DATA DOWNLOADER")
    print("=" * 60)
    print()

    universe = download_nifty500_universe()

    download_ohlcv(universe)

    print()
    print("Data preparation finished successfully.")

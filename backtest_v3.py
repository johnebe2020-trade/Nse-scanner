from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent

DATA_FILE = ROOT / "data" / "sample_ohlcv.csv"
OUTPUT_DIR = ROOT / "output"

OUTPUT_CSV = OUTPUT_DIR / "backtest_v3_results.csv"

EMA_LEN = 50
RSI_LEN = 14
VOL_LEN = 20

VOL_MULT = 1.5
PULLBACK_TOL = 0.01

MAX_SETUP = 5
MAX_PULLBACK = 3
MAX_HOLD = 60


def rsi_wilder(close, length):

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


print()
print("====================================")
print("JOHN BACKTEST V3")
print("====================================")
print()

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "Missing file: " + str(DATA_FILE)
    )

df = pd.read_csv(DATA_FILE)

print("Rows:", len(df))
print("Stocks:", df["symbol"].nunique())
print()


df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

for col in [
    "open",
    "high",
    "low",
    "close",
    "volume"
]:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


df = df.dropna(
    subset=[
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]
)

df = df.sort_values(
    ["symbol", "date"]
)


results = []


for number, (symbol, stock) in enumerate(
    df.groupby("symbol"),
    start=1
):

    stock = stock.copy()

    stock = stock.sort_values(
        "date"
    ).reset_index(drop=True)


    if len(stock) < 80:
        continue


    stock["ema50"] = stock["close"].ewm(
        span=EMA_LEN,
        adjust=False
    ).mean()


    stock["rsi"] = rsi_wilder(
        stock["close"],
        RSI_LEN
    )


    stock["vol_avg"] = stock["volume"].rolling(
        VOL_LEN
    ).mean()


    stock["vol_ratio"] = (
        stock["volume"]
        /
        stock["vol_avg"]
    )


    stock["cross"] = (
        (stock["close"] > stock["ema50"])
        &
        (
            stock["close"].shift(1)
            <=
            stock["ema50"].shift(1)
        )
    )


    for i in range(
        EMA_LEN + VOL_LEN,
        len(stock)
    ):

        cross_index = None

        start = max(
            0,
            i - MAX_SETUP
        )

        for j in range(
            i,
            start - 1,
            -1
        ):

            if stock.loc[j, "cross"]:
                cross_index = j
                break


        if cross_index is None:
            continue


        pullback_index = None

        start_pb = cross_index + 1

        end_pb = min(
            i,
            cross_index + MAX_PULLBACK
        )


        for j in range(
            start_pb,
            end_pb + 1
        ):

            ema = stock.loc[j, "ema50"]

            low = stock.loc[j, "low"]

            if pd.isna(ema):
                continue

            distance = abs(
                low - ema
            ) / ema

            if distance <= PULLBACK_TOL:

                pullback_index = j
                break


        if pullback_index is None:
            continue


        rsi = stock.loc[i, "rsi"]

        if pd.isna(rsi):
            continue

        if rsi <= 50:
            continue


        vol_ratio = stock.loc[i, "vol_ratio"]

        if pd.isna(vol_ratio):
            continue

        if vol_ratio < VOL_MULT:
            continue


        if stock.loc[i, "close"] <= stock.loc[i, "open"]:
            continue


        entry = float(
            stock.loc[i, "close"]
        )

        entry_date = stock.loc[i, "date"]

        pullback_low = float(
            stock.loc[pullback_index, "low"]
        )


        risk = entry - pullback_low

        if risk <= 0:
            continue


        tp1 = entry + risk

        tp2 = entry + (risk * 2)


        future = stock.iloc[
            i + 1:
            i + 1 + MAX_HOLD
        ]


        if future.empty:
            continue


        tp1_hit = False
        tp2_hit = False

        tp1_bar = None
        tp2_bar = None

        max_high = entry
        min_low = entry

        last_close = entry
        last_date = entry_date


        for bar, (_, candle) in enumerate(
            future.iterrows(),
            start=1
        ):

            high = float(
                candle["high"]
            )

            low = float(
                candle["low"]
            )

            close = float(
                candle["close"]
            )

            last_close = close
            last_date = candle["date"]

            max_high = max(
                max_high,
                high
            )

            min_low = min(
                min_low,
                low
            )


            if not tp1_hit and high >= tp1:

                tp1_hit = True
                tp1_bar = bar


            if not tp2_hit and high >= tp2:

                tp2_hit = True
                tp2_bar = bar

                break


        if tp2_hit:

            status = "TP2 HIT"

            exit_price = tp2

            return_pct = (
                ((tp1 - entry) / entry) * 50
                +
                ((tp2 - entry) / entry) * 50
            )


        elif tp1_hit:

            status = "TP1 HIT"

            exit_price = tp1

            return_pct = (
                ((tp1 - entry) / entry) * 50
                +
                ((last_close - entry) / entry) * 50
            )


        else:

            status = "OPEN"

            exit_price = last_close

            return_pct = (
                (last_close - entry)
                / entry
                * 100
            )


        max_upside = (
            (max_high - entry)
            / entry
            * 100
        )

        max_downside = (
            (min_low - entry)
            / entry
            * 100
        )


        results.append({

            "symbol": symbol,

            "ema_cross_date":
                stock.loc[cross_index, "date"],

            "pullback_date":
                stock.loc[pullback_index, "date"],

            "entry_date":
                entry_date,

            "entry":
                round(entry, 2),

            "pullback_low":
                round(pullback_low, 2),

            "risk":
                round(risk, 2),

            "tp1":
                round(tp1, 2),

            "tp2":
                round(tp2, 2),

            "status":
                status,

            "bars_to_tp1":
                tp1_bar if tp1_hit else "",

            "bars_to_tp2":
                tp2_bar if tp2_hit else "",

            "exit_date":
                last_date,

            "exit_price":
                round(exit_price, 2),

            "return_pct":
                round(return_pct, 2),

            "max_upside_pct":
                round(max_upside, 2),

            "max_downside_pct":
                round(max_downside, 2)
        })


    if number % 25 == 0:

        print(
            "Processed:",
            number,
            "Stocks | Signals:",
            len(results)
        )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


result_df = pd.DataFrame(results)


if result_df.empty:

    print()
    print("NO SIGNALS FOUND")
    print()

    result_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    raise SystemExit(0)


result_df.to_csv(
    OUTPUT_CSV,
    index=False
)


total = len(result_df)

tp1_count = result_df[
    result_df["status"].isin(
        ["TP1 HIT", "TP2 HIT"]
    )
].shape[0]

tp2_count = result_df[
    result_df["status"] == "TP2 HIT"
].shape[0]

open_count = result_df[
    result_df["status"] == "OPEN"
].shape[0]


tp1_rate = (
    tp1_count / total * 100
)

tp2_rate = (
    tp2_count / total * 100
)


avg_return = result_df[
    "return_pct"
].mean()


avg_upside = result_df[
    "max_upside_pct"
].mean()


avg_downside = result_df[
    "max_downside_pct"
].mean()


bars1 = pd.to_numeric(
    result_df["bars_to_tp1"],
    errors="coerce"
).dropna()


bars2 = pd.to_numeric(
    result_df["bars_to_tp2"],
    errors="coerce"
).dropna()


avg_bars1 = (
    bars1.mean()
    if len(bars1)
    else 0
)


avg_bars2 = (
    bars2.mean()
    if len(bars2)
    else 0
)


print()
print("====================================")
print("JOHN BACKTEST V3 COMPLETE")
print("====================================")
print()

print("Total signals       :", total)

print("TP1 reached         :", tp1_count)

print(
    "TP1 hit rate        :",
    f"{tp1_rate:.2f}%"
)

print("TP2 reached         :", tp2_count)

print(
    "TP2 hit rate        :",
    f"{tp2_rate:.2f}%"
)

print("Still open          :", open_count)

print(
    "Average return      :",
    f"{avg_return:.2f}%"
)

print(
    "Average bars TP1    :",
    f"{avg_bars1:.2f}"
)

print(
    "Average bars TP2    :",
    f"{avg_bars2:.2f}"
)

print(
    "Average max upside  :",
    f"{avg_upside:.2f}%"
)

print(
    "Average max down    :",
    f"{avg_downside:.2f}%"
)

print()

print(
    "Entry = confirmation candle CLOSE"
)

print(
    "Stop loss = NONE"
)

print(
    "Maximum holding =",
    MAX_HOLD,
    "bars"
)

print()

print(
    "CSV:",
    OUTPUT_CSV
)

print()

print("====================================")
print("BACKTEST V3 FINISHED")
print("====================================")

from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# JOHN'S BACKTEST V3
# EMA50 CROSS -> PULLBACK -> RSI -> VOLUME -> CONFIRMATION
# NO STOP LOSS
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_FILE = ROOT / "data" / "sample_ohlcv.csv"
OUTPUT_DIR = ROOT / "output"

OUTPUT_CSV = OUTPUT_DIR / "backtest_v3_results.csv"
OUTPUT_HTML = OUTPUT_DIR / "backtest_v3.html"

# ============================================================
# SETTINGS
# ============================================================

EMA_LENGTH = 50
RSI_LENGTH = 14
VOLUME_LENGTH = 20

VOLUME_MULTIPLIER = 1.5

PULLBACK_TOLERANCE = 0.01

MAX_SETUP_BARS = 5
MAX_PULLBACK_BARS = 3

MAX_HOLDING_BARS = 60

TP1_RR = 1.0
TP2_RR = 2.0


# ============================================================
# RSI - WILDER
# ============================================================

def calculate_rsi(series, length=14):

    delta = series.diff()

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


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(df, names):

    mapping = {}

    for column in df.columns:
        mapping[str(column).strip().lower()] = column

    for name in names:

        key = name.strip().lower()

        if key in mapping:
            return mapping[key]

    return None


# ============================================================
# START
# ============================================================

print()
print("==============================================")
print("JOHN'S BACKTEST V3")
print("==============================================")
print()

if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"OHLCV file not found: {DATA_FILE}"
    )

print("Loading:")
print(DATA_FILE)
print()

df = pd.read_csv(DATA_FILE)

print("Rows loaded:", len(df))
print()


# ============================================================
# DETECT COLUMNS
# ============================================================

symbol_col = find_column(
    df,
    ["symbol", "stock", "ticker"]
)

date_col = find_column(
    df,
    ["date", "datetime", "timestamp"]
)

open_col = find_column(
    df,
    ["open"]
)

high_col = find_column(
    df,
    ["high"]
)

low_col = find_column(
    df,
    ["low"]
)

close_col = find_column(
    df,
    ["close"]
)

volume_col = find_column(
    df,
    ["volume", "vol"]
)


print("Detected columns:")
print("--------------------------------")
print("Symbol :", symbol_col)
print("Date   :", date_col)
print("Open   :", open_col)
print("High   :", high_col)
print("Low    :", low_col)
print("Close  :", close_col)
print("Volume :", volume_col)
print()


required = [
    symbol_col,
    date_col,
    open_col,
    high_col,
    low_col,
    close_col,
    volume_col
]

if any(x is None for x in required):

    raise ValueError(
        "Required OHLCV column missing."
    )


# ============================================================
# CLEAN DATA
# ============================================================

df[date_col] = pd.to_datetime(
    df[date_col],
    errors="coerce"
)

for column in [
    open_col,
    high_col,
    low_col,
    close_col,
    volume_col
]:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


df = df.dropna(
    subset=[
        symbol_col,
        date_col,
        open_col,
        high_col,
        low_col,
        close_col,
        volume_col
    ]
)


df = df.sort_values(
    [symbol_col, date_col]
).reset_index(drop=True)


print("Clean rows:", len(df))
print("Stocks:", df[symbol_col].nunique())
print()


# ============================================================
# RESULTS
# ============================================================

results = []

total_stocks = df[symbol_col].nunique()


# ============================================================
# PROCESS STOCKS
# ============================================================

for stock_number, (symbol, stock) in enumerate(
    df.groupby(symbol_col),
    start=1
):

    stock = stock.copy()

    stock = stock.sort_values(
        date_col
    ).reset_index(drop=True)


    # --------------------------------------------------------
    # EMA50
    # --------------------------------------------------------

    stock["ema50"] = stock[close_col].ewm(
        span=EMA_LENGTH,
        adjust=False,
        min_periods=EMA_LENGTH
    ).mean()


    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    stock["rsi"] = calculate_rsi(
        stock[close_col],
        RSI_LENGTH
    )


    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    stock["volume_avg"] = stock[
        volume_col
    ].rolling(
        VOLUME_LENGTH,
        min_periods=VOLUME_LENGTH
    ).mean()


    stock["volume_ratio"] = (
        stock[volume_col]
        /
        stock["volume_avg"]
    )


    # --------------------------------------------------------
    # EMA CROSS
    # --------------------------------------------------------

    stock["ema_cross"] = (
        (stock[close_col] > stock["ema50"])
        &
        (
            stock[close_col].shift(1)
            <=
            stock["ema50"].shift(1)
        )
    )


    n = len(stock)


    # ========================================================
    # SCAN
    # ========================================================

    for i in range(
        EMA_LENGTH + VOLUME_LENGTH,
        n
    ):

        # ----------------------------------------------------
        # FIND RECENT EMA CROSS
        # ----------------------------------------------------

        cross_index = None

        search_start = max(
            0,
            i - MAX_SETUP_BARS
        )

        for j in range(
            i,
            search_start - 1,
            -1
        ):

            if bool(stock.iloc[j]["ema_cross"]):

                cross_index = j

                break


        if cross_index is None:
            continue


        # ----------------------------------------------------
        # FIND PULLBACK
        # ----------------------------------------------------

        pullback_index = None

        pullback_start = cross_index + 1

        pullback_end = min(
            i,
            cross_index + MAX_PULLBACK_BARS
        )


        for j in range(
            pullback_start,
            pullback_end + 1
        ):

            ema = stock.iloc[j]["ema50"]

            low = stock.iloc[j][low_col]

            if pd.isna(ema):
                continue

            distance = abs(
                low - ema
            ) / ema

            if distance <= PULLBACK_TOLERANCE:

                pullback_index = j

                break


        if pullback_index is None:
            continue


        # ----------------------------------------------------
        # RSI
        # ----------------------------------------------------

        rsi_value = stock.iloc[i]["rsi"]

        if pd.isna(rsi_value):
            continue

        if rsi_value <= 50:
            continue


        # ----------------------------------------------------
        # VOLUME
        # ----------------------------------------------------

        volume_ratio = stock.iloc[i][
            "volume_ratio"
        ]

        if pd.isna(volume_ratio):
            continue

        if volume_ratio < VOLUME_MULTIPLIER:
            continue


        # ----------------------------------------------------
        # BULLISH CONFIRMATION
        # ----------------------------------------------------

        current_open = stock.iloc[i][open_col]

        current_close = stock.iloc[i][close_col]

        if current_close <= current_open:
            continue


        # ----------------------------------------------------
        # ENTRY
        # ----------------------------------------------------

        entry_index = i

        entry_date = stock.iloc[i][date_col]

        entry = float(current_close)


        # ----------------------------------------------------
        # PULLBACK LOW
        # ----------------------------------------------------

        pullback_low = float(
            stock.iloc[pullback_index][low_col]
        )


        risk = entry - pullback_low


        if risk <= 0:
            continue


        # ----------------------------------------------------
        # TARGETS
        # ----------------------------------------------------

        tp1 = entry + (
            risk * TP1_RR
        )

        tp2 = entry + (
            risk * TP2_RR
        )


        # ----------------------------------------------------
        # FUTURE DATA
        # ----------------------------------------------------

        future = stock.iloc[
            entry_index + 1:
            entry_index + 1 + MAX_HOLDING_BARS
        ].copy()


        if future.empty:

            results.append({

                "symbol": symbol,

                "ema_cross_date":
                    stock.iloc[cross_index][date_col],

                "pullback_date":
                    stock.iloc[pullback_index][date_col],

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
                    "NO FUTURE DATA",

                "bars_to_tp1":
                    "",

                "bars_to_tp2":
                    "",

                "tp1_date":
                    "",

                "tp2_date":
                    "",

                "exit_date":
                    "",

                "exit_price":
                    "",

                "return_pct":
                    "",

                "max_upside_pct":
                    "",

                "max_downside_pct":
                    ""

            })

            continue


        # ----------------------------------------------------
        # TRACK FUTURE
        # ----------------------------------------------------

        tp1_hit = False
        tp2_hit = False

        tp1_bar = None
        tp2_bar = None

        tp1_date = None
        tp2_date = None

        max_high = entry
        min_low = entry

        last_close = entry
        last_date = entry_date


        for bar_number, (_, candle) in enumerate(
            future.iterrows(),
            start=1
        ):

            high = float(
                candle[high_col]
            )

            low = float(
                candle[low_col]
            )

            close = float(
                candle[close_col]
            )

            current_date = candle[date_col]


            # ------------------------------------------------
            # MAX UPSIDE / DOWNSIDE
            # ------------------------------------------------

            max_high = max(
                max_high,
                high
            )

            min_low = min(
                min_low,
                low
            )

            last_close = close

            last_date = current_date


            # ------------------------------------------------
            # TP1
            # ------------------------------------------------

            if not tp1_hit and high >= tp1:

                tp1_hit = True

                tp1_bar = bar_number

                tp1_date = current_date


            # ------------------------------------------------
            # TP2
            # ------------------------------------------------

            if not tp2_hit and high >= tp2:

                tp2_hit = True

                tp2_bar = bar_number

                tp2_date = current_date

                break


        # ----------------------------------------------------
        # CALCULATE RETURN
        # ----------------------------------------------------

        if tp2_hit:

            status = "TP2 HIT"

            exit_price = tp2

            exit_date = tp2_date

            tp1_return = (
                (tp1 - entry)
                / entry
                * 100
            )

            tp2_return = (
                (tp2 - entry)
                / entry
                * 100
            )

            return_pct = (
                0.50 * tp1_return
                +
                0.50 * tp2_return
            )


        elif tp1_hit:

            status = "TP1 HIT"

            exit_price = tp1

            exit_date = tp1_date

            tp1_return = (
                (tp1 - entry)
                / entry
                * 100
            )

            remaining_return = (
                (last_close - entry)
                / entry
                * 100
            )

            return_pct = (
                0.50 * tp1_return
                +
                0.50 * remaining_return
            )


        else:

            status = "OPEN"

            exit_price = last_close

            exit_date = last_date

            return_pct = (
                (last_close - entry)
                / entry
                * 100
            )


        # ----------------------------------------------------
        # MAX UPSIDE / DOWNSIDE
        # ----------------------------------------------------

        max_upside_pct = (
            (max_high - entry)
            / entry
            * 100
        )

        max_downside_pct = (
            (min_low - entry)
            / entry
            * 100
        )


        # ----------------------------------------------------
        # SAVE TRADE
        # ----------------------------------------------------

        results.append({

            "symbol": symbol,

            "ema_cross_date":
                stock.iloc[cross_index][date_col],

            "pullback_date":
                stock.iloc[pullback_index][date_col],

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

            "tp1_date":
                tp1_date if tp1_hit else "",

            "tp2_date":
                tp2_date if tp2_hit else "",

            "exit_date":
                exit_date,

            "exit_price":
                round(exit_price, 2),

            "return_pct":
                round(return_pct, 2),

            "max_upside_pct":
                round(max_upside_pct, 2),

            "max_downside_pct":
                round(max_downside_pct, 2)

        })


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (
        stock_number % 25 == 0
        or stock_number == total_stocks
    ):

        print(
            "Processed "
            + str(stock_number)
            + "/"
            + str(total_stocks)
            + " stocks | Signals: "
            + str(len(results))
        )


# ============================================================
# CREATE RESULTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


results_df = pd.DataFrame(results)


# ============================================================
# NO SIGNALS
# ============================================================

if results_df.empty:

    print()
    print("==============================================")
    print("NO SIGNALS FOUND")
    print("==============================================")

    results_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    raise SystemExit(0)


# ============================================================
# STATISTICS
# ============================================================

total = len(results_df)

tp1_count = (
    results_df["status"]
    .isin(["TP1 HIT", "TP2 HIT"])
    .sum()
)

tp2_count = (
    results_df["status"]
    == "TP2 HIT"
).sum()

open_count = (
    results_df["status"]
    == "OPEN"
).sum()

no_future_count = (
    results_df["status"]
    == "NO FUTURE DATA"
).sum()


tp1_rate = (
    tp1_count / total * 100
)

tp2_rate = (
    tp2_count / total * 100
)


returns = pd.to_numeric(
    results_df["return_pct"],
    errors="coerce"
).dropna()


bars_tp1 = pd.to_numeric(
    results_df["bars_to_tp1"],
    errors="coerce"
).dropna()


bars_tp2 = pd.to_numeric(
    results_df["bars_to_tp2"],
    errors="coerce"
).dropna()


upside = pd.to_numeric(
    results_df["max_upside_pct"],
    errors="coerce"
).dropna()


downside = pd.to_numeric(
    results_df["max_downside_pct"],
    errors="coerce"
).dropna()


average_return = (
    returns.mean()
    if not returns.empty
    else 0
)


average_bars_tp1 = (
    bars_tp1.mean()
    if not bars_tp1.empty
    else 0
)


average_bars_tp2 = (
    bars_tp2.mean()
    if not bars_tp2.empty
    else 0
)


average_upside = (
    upside.mean()
    if not upside.empty
    else 0
)


average_downside = (
    downside.mean()
    if not downside.empty
    else 0
)


# ============================================================
# SAVE CSV
# ============================================================

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# CREATE SIMPLE HTML
# ============================================================

html_parts = []

html_parts.append("<!DOCTYPE html>")
html_parts.append("<html>")
html_parts.append("<head>")
html_parts.append("<meta charset='UTF-8'>")
html_parts.append("<title>John Backtest V3</title>")

html_parts.append("""
<style>
body {
    background:#0b0f14;
    color:#e8eef5;
    font-family:Arial,sans-serif;
    margin:30px;
}

.card {
    display:inline-block;
    background:#151b23;
    border:1px solid #27313d;
    border-radius:10px;
    padding:15px;
    margin:5px;
    min-width:150px;
}

.label {
    color:#9aa7b5;
    font-size:12px;
}

.value {
    font-size:24px;
    font-weight:bold;
    margin-top:5px;
}

table {
    width:100%;
    border-collapse:collapse;
    margin-top:30px;
}

th {
    background:#1b232d;
    padding:8px;
    white-space:nowrap;
}

td {
    padding:8px;
    border-bottom:1px solid #252d36;
    white-space:nowrap;
}

.container {
    overflow-x:auto;
}
</style>
""")

html_parts.append("</head>")
html_parts.append("<body>")

html_parts.append("<h1>JOHN'S BACKTEST V3</h1>")

html_parts.append(
    "<p>EMA50 Cross → Pullback → RSI → Volume → Bullish Confirmation</p>"
)

html_parts.append(
    "<p>Entry = Confirmation Candle Close | "
    "Stop Loss = Disabled | "
    "Maximum Holding = 60 bars</p>"
)


# ============================================================
# CARDS
# ============================================================

def card(label, value):

    return (
        "<div cla

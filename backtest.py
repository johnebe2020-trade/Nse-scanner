import html
from pathlib import Path

import pandas as pd


# ============================================================
# JOHN'S BACKTEST ENGINE V2
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_FILE = ROOT / "data" / "sample_ohlcv.csv"
SIGNALS_FILE = ROOT / "output" / "scanner_results.csv"

OUTPUT_CSV = ROOT / "output" / "backtest_results.csv"
OUTPUT_HTML = ROOT / "output" / "backtest.html"

# Maximum number of daily candles to follow a signal
MAX_BARS = 20

# Position management:
# 50% booked at TP1
# Remaining 50% targeted at TP2
TP1_PART = 0.50
TP2_PART = 0.50


# ============================================================
# HELPERS
# ============================================================

def find_column(df, names):

    lower_map = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for name in names:

        if name.lower() in lower_map:
            return lower_map[name.lower()]

    return None


def money(value):

    if pd.isna(value):
        return "-"

    return f"₹{float(value):,.2f}"


def pct(value):

    if pd.isna(value):
        return "-"

    return f"{float(value):.2f}%"


def num(value):

    if pd.isna(value):
        return "-"

    return f"{float(value):.2f}"


# ============================================================
# LOAD FILES
# ============================================================

print()
print("=" * 70)
print("JOHN'S BACKTEST ENGINE V2")
print("=" * 70)
print()

if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"OHLCV file not found: {DATA_FILE}"
    )

if not SIGNALS_FILE.exists():

    raise FileNotFoundError(
        f"Scanner results not found: {SIGNALS_FILE}"
    )


ohlcv = pd.read_csv(DATA_FILE)

signals = pd.read_csv(SIGNALS_FILE)


print(
    f"OHLCV rows       : {len(ohlcv):,}"
)

print(
    f"Scanner signals  : {len(signals):,}"
)


# ============================================================
# NORMALIZE OHLCV
# ============================================================

ohlcv.columns = [
    str(c).strip().lower()
    for c in ohlcv.columns
]

required = [
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close"
]

missing = [
    c for c in required
    if c not in ohlcv.columns
]

if missing:

    raise ValueError(
        f"OHLCV missing columns: {missing}"
    )


ohlcv["symbol"] = (
    ohlcv["symbol"]
    .astype(str)
    .str.strip()
    .str.upper()
)

ohlcv["date"] = pd.to_datetime(
    ohlcv["date"],
    errors="coerce"
)

for column in [
    "open",
    "high",
    "low",
    "close"
]:

    ohlcv[column] = pd.to_numeric(
        ohlcv[column],
        errors="coerce"
    )


ohlcv = ohlcv.dropna(
    subset=[
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close"
    ]
)

ohlcv = ohlcv.sort_values(
    ["symbol", "date"]
)


# ============================================================
# SIGNAL COLUMNS
# ============================================================

symbol_col = find_column(
    signals,
    ["stock", "symbol"]
)

entry_col = find_column(
    signals,
    ["entry"]
)

sl_col = find_column(
    signals,
    ["sl", "stop_loss", "stop loss"]
)

tp1_col = find_column(
    signals,
    ["tp1", "tp 1"]
)

tp2_col = find_column(
    signals,
    ["tp2", "tp 2"]
)

confirmation_col = find_column(
    signals,
    [
        "confirmation",
        "confirmation_date",
        "confirmation date"
    ]
)


if symbol_col is None:
    raise ValueError(
        "STOCK/SYMBOL column not found."
    )

if entry_col is None:
    raise ValueError(
        "ENTRY column not found."
    )

if sl_col is None:
    raise ValueError(
        "SL column not found."
    )

if tp1_col is None:
    raise ValueError(
        "TP1 column not found."
    )

if tp2_col is None:
    raise ValueError(
        "TP2 column not found."
    )

if confirmation_col is None:
    raise ValueError(
        "CONFIRMATION column not found."
    )


# ============================================================
# PREPARE SIGNALS
# ============================================================

signals["symbol"] = (
    signals[symbol_col]
    .astype(str)
    .str.strip()
    .str.upper()
)

signals["entry_price"] = pd.to_numeric(
    signals[entry_col],
    errors="coerce"
)

signals["sl_price"] = pd.to_numeric(
    signals[sl_col],
    errors="coerce"
)

signals["tp1_price"] = pd.to_numeric(
    signals[tp1_col],
    errors="coerce"
)

signals["tp2_price"] = pd.to_numeric(
    signals[tp2_col],
    errors="coerce"
)

signals["confirmation_date"] = pd.to_datetime(
    signals[confirmation_col],
    errors="coerce"
)


signals = signals.dropna(
    subset=[
        "symbol",
        "entry_price",
        "sl_price",
        "tp1_price",
        "tp2_price",
        "confirmation_date"
    ]
)


# ============================================================
# BACKTEST
# ============================================================

results = []

print()
print("Running V2 historical test...")
print()


for _, signal in signals.iterrows():

    symbol = signal["symbol"]

    signal_date = signal[
        "confirmation_date"
    ]

    entry = float(
        signal["entry_price"]
    )

    sl = float(
        signal["sl_price"]
    )

    tp1 = float(
        signal["tp1_price"]
    )

    tp2 = float(
        signal["tp2_price"]
    )


    # --------------------------------------------------------
    # Get stock candles
    # --------------------------------------------------------

    stock = ohlcv[
        ohlcv["symbol"] == symbol
    ].copy()


    if stock.empty:

        continue


    # Only candles AFTER confirmation.
    # This prevents using future information.
    future = stock[
        stock["date"] > signal_date
    ].head(MAX_BARS)


    if future.empty:

        results.append({

            "symbol": symbol,
            "signal_date": signal_date,
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,

            "status": "NO FUTURE DATA",

            "tp1_hit": False,
            "tp2_hit": False,
            "sl_hit": False,

            "tp1_date": None,
            "tp2_date": None,
            "sl_date": None,

            "bars_to_tp1": None,
            "bars_to_tp2": None,
            "bars_to_sl": None,

            "exit_date": None,
            "exit_price": None,

            "return_pct": None,

            "max_favourable_pct": None,
            "max_adverse_pct": None

        })

        continue


    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    tp1_hit = False
    tp2_hit = False
    sl_hit = False

    tp1_date = None
    tp2_date = None
    sl_date = None

    bars_to_tp1 = None
    bars_to_tp2 = None
    bars_to_sl = None

    max_high = entry
    min_low = entry

    status = "OPEN"

    final_exit_date = None
    final_exit_price = None

    remaining_position = TP2_PART

    realised_return = 0.0


    # --------------------------------------------------------
    # Follow each future candle
    # --------------------------------------------------------

    for bar_number, (_, bar) in enumerate(
        future.iterrows(),
        start=1
    ):

        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])

        bar_date = bar["date"]


        max_high = max(
            max_high,
            high
        )

        min_low = min(
            min_low,
            low
        )


        # ====================================================
        # IMPORTANT SAME-CANDLE RULE
        #
        # If SL is touched on the same candle before/alongside
        # target information, we conservatively treat SL as
        # happening first.
        # ====================================================

        hit_sl_now = low <= sl

        hit_tp1_now = high >= tp1

        hit_tp2_now = high >= tp2


        # ====================================================
        # CASE 1:
        # SL happens before TP1
        # ====================================================

        if not tp1_hit and hit_sl_now:

            sl_hit = True

            sl_date = bar_date

            bars_to_sl = bar_number

            status = "SL BEFORE TP1"

            final_exit_date = bar_date

            final_exit_price = sl

            realised_return = (
                (sl - entry)
                / entry
                * 100
            )

            break


        # ====================================================
        # CASE 2:
        # TP1 reached
        # ====================================================

        if not tp1_hit and hit_tp1_now:

            tp1_hit = True

            tp1_date = bar_date

            bars_to_tp1 = bar_number

            # Book 50% at TP1
            realised_return += (
                TP1_PART
                *
                (
                    (tp1 - entry)
                    / entry
                    * 100
                )
            )

            remaining_position = TP2_PART


            # ------------------------------------------------
            # If TP2 is ALSO reached on this same candle,
            # we can count TP2 because price's high reached
            # the higher target.
            # ------------------------------------------------

            if hit_tp2_now:

                tp2_hit = True

                tp2_date = bar_date

                bars_to_tp2 = bar_number

                realised_return += (
                    TP2_PART
                    *
                    (
                        (tp2 - entry)
                        / entry
                        * 100
                    )
                )

                remaining_position = 0

                status = "TP2"

                final_exit_date = bar_date

                final_exit_price = tp2

                break


            continue


        # ====================================================
        # CASE 3:
        # TP1 already reached
        # ====================================================

        if tp1_hit:

            # -----------------------------------------------
            # Remaining half reaches TP2
            # -----------------------------------------------

            if hit_tp2_now:

                tp2_hit = True

                tp2_date = bar_date

                bars_to_tp2 = bar_number

                realised_return += (
                    TP2_PART
                    *
                    (
                        (tp2 - entry)
                        / entry
                        * 100
                    )
                )

                remaining_position = 0

                status = "TP2"

                final_exit_date = bar_date

                final_exit_price = tp2

                break


            # -----------------------------------------------
            # Remaining half hits SL
            # -----------------------------------------------

            if hit_sl_now:

                sl_hit = True

                sl_date = bar_date

                bars_to_sl = bar_number

                realised_return += (
                    TP2_PART
                    *
                    (
                        (sl - entry)
                        / entry
                        * 100
                    )
                )

                remaining_position = 0

                status = "TP1 → SL"

                final_exit_date = bar_date

                final_exit_price = sl

                break


    # ========================================================
    # STILL OPEN AFTER MAX BARS
    # ========================================================

    if status == "OPEN":

        last_bar = future.iloc[-1]

        final_exit_date = last_bar["date"]

        final_close = float(
            last_bar["close"]
        )


        if tp1_hit:

            # TP1 half already realised.
            # Remaining half valued at last close.

            remaining_return = (
                (final_close - entry)
                / entry
                * 100
            )

            realised_return += (
                remaining_position
                *
                remaining_return
            )

            status = "TP1 → OPEN"


        else:

            realised_return = (
                (final_close - entry)
                / entry
                * 100
            )


        final_exit_price = final_close


    # ========================================================
    # MAX FAVOURABLE / ADVERSE MOVE
    # ========================================================

    max_favourable_pct = (
        (max_high - entry)
        / entry
        * 100
    )

    max_adverse_pct = (
        (min_low - entry)
        / entry
        * 100
    )


    results.append({

        "symbol": symbol,

        "signal_date": signal_date,

        "entry": entry,

        "sl": sl,

        "tp1": tp1,

        "tp2": tp2,

        "status": status,

        "tp1_hit": tp1_hit,

        "tp2_hit": tp2_hit,

        "sl_hit": sl_hit,

        "tp1_date": tp1_date,

        "tp2_date": tp2_date,

        "sl_date": sl_date,

        "bars_to_tp1": bars_to_tp1,

        "bars_to_tp2": bars_to_tp2,

        "bars_to_sl": bars_to_sl,

        "exit_date": final_exit_date,

        "exit_price": final_exit_price,

        "return_pct": realised_return,

        "max_favourable_pct":
            max_favourable_pct,

        "max_adverse_pct":
            max_adverse_pct

    })


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


if results_df.empty:

    print()
    print("No backtest results generated.")
    raise SystemExit(0)


# ============================================================
# FORMAT DATES
# ============================================================

date_columns = [
    "signal_date",
    "tp1_date",
    "tp2_date",
    "sl_date",
    "exit_date"
]


for column in date_columns:

    results_df[column] = pd.to_datetime(
        results_df[column],
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")


# ============================================================
# SAVE CSV
# ============================================================

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# STATISTICS
# ============================================================

total = len(results_df)

tp1_count = int(
    results_df["tp1_hit"].sum()
)

tp2_count = int(
    results_df["tp2_hit"].sum()
)

sl_before_tp1 = int(
    (
        results_df["status"]
        == "SL BEFORE TP1"
    ).sum()
)

tp1_then_sl = int(
    (
        results_df["status"]
        == "TP1 → SL"
    ).sum()
)

tp1_open = int(
    (
        results_df["status"]
        == "TP1 → OPEN"
    ).sum()
)

tp2_full = int(
    (
        results_df["status"]
        == "TP2"
    ).sum()
)

no_future = int(
    (
        results_df["status"]
        == "NO FUTURE DATA"
    ).sum()
)


# ============================================================
# R-MULTIPLE
# ============================================================

# Risk based on scanner's original entry/SL.

results_df["risk"] = (
    results_df["entry"]
    - results_df["sl"]
)


valid_risk = results_df[
    results_df["risk"] > 0
].copy()


if not valid_risk.empty:

    valid_risk["r_multiple"] = (
        valid_risk["return_pct"]
        /
        (
            valid_risk["risk"]
            /
            valid_risk["entry"]
            *
            100
        )
    )

    average_r = valid_risk[
        "r_multiple"
    ].mean()

else:

    average_r = 0


# ============================================================
# RETURN
# ============================================================

closed_for_return = results_df[
    results_df["status"] != "NO FUTURE DATA"
]


if not closed_for_return.empty:

    average_return = (
        closed_for_return[
            "return_pct"
        ].mean()
    )

else:

    average_return = 0


# ============================================================
# TP1 HIT RATE
# ============================================================

usable = total - no_future

if usable > 0:

    tp1_rate = (
        tp1_count
        /
        usable
        *
        100
    )

else:

    tp1_rate = 0


# ============================================================
# TP2 RATE
# ============================================================

if usable > 0:

    tp2_rate = (
        tp2_count
        /
        usable
        *
        100
    )

else:

    tp2_rate = 0


# ============================================================
# SL BEFORE TP1 RATE
# ============================================================

if usable > 0:

    sl_rate = (
        sl_before_tp1
        /
        usable
        *
        100
    )

else:

    sl_rate = 0


# ============================================================
# AVERAGE BARS
# ============================================================

avg_bars_tp1 = results_df[
    "bars_to_tp1"
].dropna().mean()

avg_bars_tp2 = results_df[
    "bars_to_tp2"
].dropna().mean()

avg_bars_sl = results_df[
    "bars_to_sl"
].dropna().mean()

avg_max_up = results_df[
    "max_favourable_pct"
].mean()

avg_max_down = results_df[
    "max_adverse_pct"
].mean()


# ============================================================
# HTML TABLE
# ============================================================

rows_html = ""


for _, row in results_df.iterrows():

    status = str(
        row["status"]
    )


    if status == "TP2":

        badge = "green"

    elif status == "TP1 → SL":

        badge = "yellow"

    elif status == "TP1 → OPEN":

        badge = "yellow"

    elif status == "SL BEFORE TP1":

        badge = "red"

    elif status == "OPEN":

        badge = "blue"

    else:

        badge = "grey"


    rows_html += f"""
    <tr>

        <td>
            <b>{html.escape(str(row["symbol"]))}</b>
        </td>

        <td>
            {row["signal_date"]}
        </td>

        <td>
            {money(row["entry"])}
        </td>

        <td>
            {money(row["sl"])}
        </td>

        <td>
            {money(row["tp1"])}
        </td>

        <td>
            {money(row["tp2"])}
        </td>

        <td>
            <span class="badge {badge}">
                {html.escape(status)}
            </span>
        </td>

        <td>
            {row["tp1_date"]
             if pd.notna(row["tp1_date"])
             else "-"}
        </td>

        <td>
            {row["tp2_date"]
             if pd.notna(row["tp2_date"])
             else "-"}
        </td>

        <td>
            {row["bars_to_tp1"]
             if pd.notna(row["bars_to_tp1"])
             else "-"}
        </td>

        <td>
            {row["bars_to_tp2"]
             if pd.notna(row["bars_to_tp2"])
             else "-"}
        </td>

        <td>
            {pct(row["return_pct"])}
        </td>

        <td>
            {pct(row["max_favourable_pct"])}
        </td>

        <td>
            {pct(row["max_adverse_pct"])}
        </td>

    </tr>
    """


# ============================================================
# HTML
# ============================================================

html_page = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width, initial-scale=1.0">

<title>
John's Backtest V2
</title>

<style>

body {{

    margin: 0;

    background: #090e1c;

    color: #e8edf7;

    font-family:
        Arial,

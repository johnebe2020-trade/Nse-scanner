import html
from pathlib import Path

import pandas as pd


# ============================================================
# JOHN'S BACKTEST ENGINE
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_FILE = ROOT / "data" / "sample_ohlcv.csv"
SIGNALS_FILE = ROOT / "output" / "scanner_results.csv"

OUTPUT_CSV = ROOT / "output" / "backtest_results.csv"
OUTPUT_HTML = ROOT / "output" / "backtest.html"


# Maximum number of daily bars to track after confirmation
MAX_BARS = 20


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


def number(value, decimals=2):

    if pd.isna(value):
        return "-"

    return f"{float(value):.{decimals}f}"


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 65)
print("JOHN'S BACKTEST ENGINE")
print("=" * 65)
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
    f"OHLCV rows: {len(ohlcv):,}"
)

print(
    f"Scanner signals: {len(signals):,}"
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

for col in [
    "open",
    "high",
    "low",
    "close"
]:

    ohlcv[col] = pd.to_numeric(
        ohlcv[col],
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
# FIND SIGNAL COLUMNS
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
        "Could not find STOCK/SYMBOL column in scanner_results.csv"
    )

if entry_col is None:
    raise ValueError(
        "Could not find ENTRY column in scanner_results.csv"
    )

if sl_col is None:
    raise ValueError(
        "Could not find SL column in scanner_results.csv"
    )

if tp1_col is None:
    raise ValueError(
        "Could not find TP1 column in scanner_results.csv"
    )

if tp2_col is None:
    raise ValueError(
        "Could not find TP2 column in scanner_results.csv"
    )

if confirmation_col is None:
    raise ValueError(
        "Could not find CONFIRMATION column in scanner_results.csv"
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
print("Running historical test...")
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


    stock = ohlcv[
        ohlcv["symbol"] == symbol
    ].copy()


    if stock.empty:
        continue


    # Only candles AFTER confirmation
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
            "result": "NO FUTURE DATA",
            "bars": 0,
            "exit_date": None,
            "exit_price": None,
            "return_pct": None,
            "max_favourable_pct": None,
            "max_adverse_pct": None

        })

        continue


    result = "OPEN"

    exit_date = None
    exit_price = None
    bars_used = 0


    max_high = entry
    min_low = entry


    for bar_number, (_, bar) in enumerate(
        future.iterrows(),
        start=1
    ):

        high = float(bar["high"])
        low = float(bar["low"])


        max_high = max(
            max_high,
            high
        )

        min_low = min(
            min_low,
            low
        )


        # ----------------------------------------------------
        # IMPORTANT:
        # If both SL and TP are touched on the same candle,
        # we assume SL happened first.
        #
        # This is conservative because daily OHLC does not
        # tell us the intraday order of high/low.
        # ----------------------------------------------------

        hit_sl = low <= sl

        hit_tp2 = high >= tp2

        hit_tp1 = high >= tp1


        if hit_sl:

            result = "SL"

            exit_date = bar["date"]

            exit_price = sl

            bars_used = bar_number

            break


        if hit_tp2:

            result = "TP2"

            exit_date = bar["date"]

            exit_price = tp2

            bars_used = bar_number

            break


        if hit_tp1:

            result = "TP1"

            exit_date = bar["date"]

            exit_price = tp1

            bars_used = bar_number

            break


        bars_used = bar_number


    # --------------------------------------------------------
    # Still open after tracking period
    # --------------------------------------------------------

    if result == "OPEN":

        last_bar = future.iloc[-1]

        exit_date = last_bar["date"]

        exit_price = float(
            last_bar["close"]
        )


    return_pct = (
        (exit_price - entry)
        / entry
        * 100
    )


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

        "result": result,

        "bars": bars_used,

        "exit_date": exit_date,

        "exit_price": exit_price,

        "return_pct": return_pct,

        "max_favourable_pct":
            max_favourable_pct,

        "max_adverse_pct":
            max_adverse_pct

    })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)


if results_df.empty:

    print()
    print("No backtest results were generated.")
    print()

    raise SystemExit(0)


results_df["signal_date"] = pd.to_datetime(
    results_df["signal_date"]
).dt.strftime("%Y-%m-%d")


results_df["exit_date"] = pd.to_datetime(
    results_df["exit_date"],
    errors="coerce"
).dt.strftime("%Y-%m-%d")


results_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# STATISTICS
# ============================================================

total = len(results_df)

tp1_count = (
    results_df["result"] == "TP1"
).sum()

tp2_count = (
    results_df["result"] == "TP2"
).sum()

sl_count = (
    results_df["result"] == "SL"
).sum()

open_count = (
    results_df["result"] == "OPEN"
).sum()

no_data_count = (
    results_df["result"] == "NO FUTURE DATA"
).sum()


closed = results_df[
    results_df["result"].isin(
        ["TP1", "TP2", "SL"]
    )
]


if len(closed) > 0:

    avg_return = closed[
        "return_pct"
    ].mean()

else:

    avg_return = 0


win_count = tp1_count + tp2_count

if len(closed) > 0:

    win_rate = (
        win_count
        / len(closed)
        * 100
    )

else:

    win_rate = 0


# ============================================================
# HTML DASHBOARD
# ============================================================

rows_html = ""


for _, row in results_df.iterrows():

    result = str(
        row["result"]
    )

    if result == "TP2":
        badge = "tp2"

    elif result == "TP1":
        badge = "tp1"

    elif result == "SL":
        badge = "sl"

    elif result == "OPEN":
        badge = "open"

    else:
        badge = "nodata"


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
                {html.escape(result)}
            </span>
        </td>

        <td>
            {row["bars"]}
        </td>

        <td>
            {money(row["exit_price"])}
        </td>

        <td>
            {number(row["return_pct"])}%
        </td>

        <td>
            {number(row["max_favourable_pct"])}%
        </td>

        <td>
            {number(row["max_adverse_pct"])}%
        </td>

    </tr>
    """


html_page = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width,
               initial-scale=1.0">

<title>
John's Scanner - Backtest
</title>

<style>

body {{

    margin: 0;

    background: #0b1020;

    color: #e8ecf5;

    font-family:
        Arial,
        sans-serif;

}}

.container {{

    padding: 20px;

}}

h1 {{

    margin-bottom: 5px;

}}

.subtitle {{

    color: #9ba6bd;

    margin-bottom: 20px;

}}

.cards {{

    display: grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(
                150px,
                1fr
            )
        );

    gap: 12px;

    margin-bottom: 25px;

}}

.card {{

    background: #171d30;

    border: 1px solid #29324a;

    border-radius: 10px;

    padding: 16px;

}}

.card-title {{

    color: #9ba6bd;

    font-size: 13px;

}}

.card-value {{

    font-size: 25px;

    font-weight: bold;

    margin-top: 7px;

}}

.table-wrap {{

    overflow-x: auto;

}}

table {{

    width: 100%;

    border-collapse:
        collapse;

    background: #171d30;

}}

th {{

    background: #11172a;

    color: #9ba6bd;

    padding: 12px;

    text-align: left;

    white-space: nowrap;

}}

td {{

    padding: 11px;

    border-top:
        1px solid #29324a;

    white-space: nowrap;

}}

.badge {{

    padding:
        5px 9px;

    border-radius: 6px;

    font-weight: bold;

    font-size: 12px;

}}

.tp1 {{

    background: #164e32;

    color: #5ee49a;

}}

.tp2 {{

    background: #14532d;

    color: #86efac;

}}

.sl {{

    background: #542020;

    color: #ff8585;

}}

.open {{

    background: #423714;

    color: #ffd75e;

}}

.nodata {{

    background: #303644;

    color: #aeb7ca;

}}

.note {{

    margin-top: 20px;

    color: #8e99b0;

    font-size: 13px;

    line-height: 1.6;

}}

</style>

</head>


<body>

<div class="container">

<h1>
📊 JOHN'S BACKTEST
</h1>

<div class="subtitle">

Historical test of confirmed
EMA Pullback signals

</div>


<div class="cards">

<div class="card">

<div class="card-title">
TOTAL SIGNALS
</div>

<div class="card-value">
{total}
</div>

</div>


<div class="card">

<div class="card-title">
TP1
</div>

<div class="card-value">
{tp1_count}
</div>

</div>


<div class="card">

<div class="card-title">
TP2
</div>

<div class="card-value">
{tp2_count}
</div>

</div>


<div class="card">

<div class="card-title">
SL
</div>

<div class="card-value">
{sl_count}
</div>

</div>


<div class="card">

<div class="card-title">
OPEN
</div>

<div class="card-value">
{open_count}
</div>

</div>


<div class="card">

<div class="card-title">
WIN RATE
</div>

<div class="card-value">
{number(win_rate)}%
</div>

</div>


<div class="card">

<div class="card-title">
AVG RETURN
</div>

<div class="card-value">
{number(avg_return)}%
</div>

</div>

</div>


<div class="table-wrap">

<table>

<thead>

<tr>

<th>STOCK</th>
<th>SIGNAL</th>
<th>ENTRY</th>
<th>SL</th>
<th>TP1</th>
<th>TP2</th>
<th>RESULT</th>
<th>BARS</th>
<th>EXIT</th>
<th>RETURN</th>
<th>MAX ↑</th>
<th>MAX ↓</th>

</tr>

</thead>

<tbody>

{rows_html}

</tbody>

</table>

</div>


<div class="note">

<b>Important:</b>

This is a historical research test,
not a trading recommendation.

For daily candles, if both the stop-loss
and target appear to be touched during
the same candle, this backtest assumes
the stop-loss happened first because
daily OHLC data does not reveal the
intraday order.

Maximum tracking period:
{MAX_BARS} trading bars.

</div>

</div>

</body>

</html>
"""


OUTPUT_HTML.write_text(
    html_page,
    encoding="utf-8"
)


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print()
print("=" * 65)
print("BACKTEST COMPLETE")
print("=" * 65)

print()
print(f"Total signals : {total}")
print(f"TP1           : {tp1_count}")
print(f"TP2           : {tp2_count}")
print(f"SL            : {sl_count}")
print(f"Open          : {open_count}")
print(f"No future data: {no_data_count}")
print(f"Win rate      : {win_rate:.2f}%")
print(f"Average return: {avg_return:.2f}%")

print()
print(f"CSV : {OUTPUT_CSV}")
print(f"HTML: {OUTPUT_HTML}")

print()

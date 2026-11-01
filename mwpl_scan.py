import glob
import re
import pandas as pd

FOLDER = "mwpl_data"
MIN_CHG = 5.0        # rise in MWPL usage (percentage points) needed each day
DAYS = 2             # consecutive days


def find_col(cols, *keys):
    for c in cols:
        low = c.lower()
        if all(k in low for k in keys):
            return c
    return None


def load(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    hdr = next((i for i, l in enumerate(lines) if "mwpl" in l.lower()), 0)
    df = pd.read_csv(path, skiprows=hdr)
    df.columns = [c.strip() for c in df.columns]

    sym = find_col(df.columns, "symbol")
    mw = find_col(df.columns, "mwpl")
    oi = (find_col(df.columns, "future", "equiv")
          or find_col(df.columns, "open", "interest"))
    lim = find_col(df.columns, "limit")
    dt = find_col(df.columns, "date")

    out = pd.DataFrame()
    out["symbol"] = df[sym].astype(str).str.strip().str.upper()
    out["mwpl"] = pd.to_numeric(df[mw], errors="coerce")
    out["oi"] = pd.to_numeric(df[oi], errors="coerce")
    out["pct"] = out["oi"] / out["mwpl"] * 100
    out["ban"] = (df[lim].astype(str).str.contains("no fresh", case=False, na=False)
                  if lim else False)
    out = out.dropna(subset=["pct"]).drop_duplicates("symbol").set_index("symbol")

    when = None
    if dt:
        when = pd.to_datetime(df[dt].iloc[0], format="%d-%b-%y", errors="coerce")
    if when is None or pd.isna(when):
        m = re.search(r"(\d{2})(\d{2})(\d{4})", path)
        when = (pd.Timestamp(f"{m.group(3)}-{m.group(2)}-{m.group(1)}")
                if m else pd.Timestamp.min)
    return when, out


loaded = sorted((load(p) for p in glob.glob(f"{FOLDER}/*.csv")),
                key=lambda x: x[0])
need = DAYS + 1
if len(loaded) < need:
    raise SystemExit(f"Need at least {need} files in {FOLDER}, found {len(loaded)}")
loaded = loaded[-need:]
dates = [d.strftime("%d-%b-%y") for d, _ in loaded]
days = [x for _, x in loaded]

res = pd.DataFrame({"usage_pct": days[-1]["pct"], "ban": days[-1]["ban"]})
ok = pd.Series(True, index=res.index)
for k in range(1, need):
    chg = (days[k]["pct"] - days[k - 1]["pct"]).reindex(res.index)
    res[f"chg_{dates[k]}"] = chg.round(2)
    ok &= chg > MIN_CHG

res = res[ok & ~res["ban"]]
res["usage_pct"] = res["usage_pct"].round(1)
res = res.sort_values(res.columns[-1], ascending=False).drop(columns="ban")

print("Dates used:", dates)
print(res.to_string() if not res.empty else "No stocks passed the filter")
res.to_csv("mwpl_results.csv")

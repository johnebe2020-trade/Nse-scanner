import glob
import re
import pandas as pd

FOLDER = "mwpl_data"
MIN_CHG = 1.0        # rise in MWPL usage (percentage points) needed each day
                     # the video uses 5, but that is rare, so test with 2 first
DAYS = 2             # consecutive days


def find_col(cols, *keys):
    for c in cols:
        low = c.lower()
        if all(k in low for k in keys):
            return c
    return None


def num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(),
                         errors="coerce")


def file_date(p):
    m = re.search(r"(\d{2})(\d{2})(\d{4})", p)
    return pd.Timestamp(f"{m.group(3)}-{m.group(2)}-{m.group(1)}") if m else None


def load(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    hdr = next((i for i, l in enumerate(lines)
                if "isin" in l.lower() or "symbol" in l.lower()), 0)
    df = pd.read_csv(path, skiprows=hdr)
    df.columns = [c.strip() for c in df.columns]

    sym = find_col(df.columns, "symbol")
    eq = find_col(df.columns, "equiv")
    mw = find_col(df.columns, "mwpl")
    lim = find_col(df.columns, "limit")
    if sym is None or eq is None:
        raise SystemExit(f"Columns not found in {path}: {list(df.columns)}")

    out = pd.DataFrame()
    out["symbol"] = df[sym].astype(str).str.strip().str.upper()
    out["eq"] = num(df[eq])
    out["mwpl"] = num(df[mw]) if mw else float("nan")
    out["ban"] = (df[lim].astype(str).str.contains("no fresh", case=False, na=False)
                  if lim else False)
    out = out.dropna(subset=["eq"]).drop_duplicates("symbol").set_index("symbol")
    return out


by_date = {}
for p in glob.glob(f"{FOLDER}/*.csv"):
    d = file_date(p)
    if d is not None:
        by_date[d] = load(p)       # same date twice: last one wins

dates = sorted(by_date)
need = DAYS + 1
if len(dates) < need:
    raise SystemExit(f"Need {need} different dates in {FOLDER}, found {len(dates)}")
dates = dates[-need:]
days = [by_date[d] for d in dates]

# MWPL and ban status come from the latest file that has them
mwpl = None
ban = pd.Series(False, index=days[-1].index)
for d in reversed(dates):
    x = by_date[d]
    if x["mwpl"].notna().any():
        mwpl = x["mwpl"]
        ban = x["ban"]
        break
if mwpl is None:
    raise SystemExit("No file has an MWPL column")

usage = [(x["eq"] / mwpl * 100) for x in days]
labels = [d.strftime("%d-%b") for d in dates]

res = pd.DataFrame({"usage_pct": usage[-1], "ban": ban.reindex(usage[-1].index)})
res["ban"] = res["ban"].fillna(False)
ok = pd.Series(True, index=res.index)
for k in range(1, need):
    chg = (usage[k] - usage[k - 1]).reindex(res.index)
    res[f"chg_{labels[k]}"] = chg.round(2)
    ok &= chg > MIN_CHG

res = res.dropna(subset=["usage_pct"])
res["usage_pct"] = res["usage_pct"].round(1)
last = res.columns[-1]

res.sort_values(last, ascending=False).to_csv("mwpl_all.csv")
hits = res[ok.reindex(res.index).fillna(False) & ~res["ban"]]
hits = hits.sort_values(last, ascending=False).drop(columns="ban")

print("Dates used:", labels)
print(hits.to_string() if not hits.empty else "No stocks passed the filter")
hits.to_csv("mwpl_results.csv")

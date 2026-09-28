import os, time, json, glob, datetime, requests
import pandas as pd
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor

EMA_LEN, RSI_LEN, VOL_LEN = 50, 14, 20
RSI_LO, RSI_HI = 45, 50
VOL_MULT, LOOKBACK, NEAR_PCT = 1.5, 3, 1.5
MIN_AVG_VOL = 100000
PAGE = "https://johnebe2020-trade.github.io/Nse-scanner/"
HDR = {"User-Agent": "Mozilla/5.0"}


def fetch(sym, rng, interval):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(sym, safe='')}.NS"
    for _ in range(3):
        try:
            r = requests.get(url, params={"range": rng, "interval": interval},
                             headers=HDR, timeout=20)
            res = r.json()["chart"]["result"][0]
            q = res["indicators"]["quote"][0]
            idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert("Asia/Kolkata")
            df = pd.DataFrame(q, index=idx)
            return df[["open", "high", "low", "close", "volume"]].dropna()
        except Exception:
            time.sleep(1.5)
    return None


def to_4h(df):
    d = df.copy()
    d["day"] = d.index.date
    d["blk"] = d.groupby("day").cumcount() // 4
    g = d.groupby(["day", "blk"])
    return pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
    })


def check(df):
    if df is None or len(df) < EMA_LEN + 5:
        return None
    c = df["close"]
    ema = c.ewm(span=EMA_LEN, adjust=False).mean()
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / RSI_LEN, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / RSI_LEN, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn)
    vr = df["volume"] / df["volume"].rolling(VOL_LEN).mean()
    cross = ((c.shift(1) <= ema.shift(1)) & (c > ema)).tail(LOOKBACK).any()
    vol_ok = vr.tail(LOOKBACK).max() >= VOL_MULT
    r, rp = rsi.iloc[-1], rsi.iloc[-2]
    rsi_ok = RSI_LO <= r <= RSI_HI
    gap = (c.iloc[-1] - ema.iloc[-1]) / ema.iloc[-1] * 100
    near = c.iloc[-1] < ema.iloc[-1] and gap >= -NEAR_PCT and rsi_ok and r > rp
    if cross and vol_ok and rsi_ok:
        return "CROSS", r, gap
    if near:
        return "NEAR", r, gap
    return None


def scan(row):
    sym, sec = row
    out = []
    d = fetch(sym, "1y", "1d")
    if d is None or d["volume"].tail(20).mean() < MIN_AVG_VOL:
        return []
    res = check(d)
    if res:
        out.append((sec, sym, "D", *res))
    h = fetch(sym, "60d", "60m")
    res = check(to_4h(h)) if h is not None else None
    if res:
        out.append((sec, sym, "4H", *res))
    return out


def send(text):
    tok = os.environ["TELEGRAM_BOT_TOKEN"]
    chat = os.environ["TELEGRAM_CHAT_ID"]
    for i in range(0, len(text), 3800):
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      data={"chat_id": chat, "text": text[i:i + 3800]}, timeout=20)


def write_html(hits, total):
    os.makedirs("docs", exist_ok=True)
    data = [{"sector": s, "symbol": y, "tf": tf, "kind": k,
             "rsi": round(r, 1), "gap": round(g, 2)} for s, y, tf, k, r, g in hits]
    ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    meta = datetime.datetime.now(ist).strftime("%d %b %Y %H:%M IST") + f" | scanned {total} stocks"
    html = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>EMA50 RSI Scan</title>
<style>body{font-family:Arial;background:#111;color:#eee;margin:8px;font-size:13px}
select,input{background:#222;color:#eee;border:1px solid #444;padding:6px;margin:2px}
table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #333;padding:6px;text-align:left}
th{cursor:pointer;background:#1c1c1c;position:sticky;top:0}
.CROSS{color:#4caf50;font-weight:bold}.NEAR{color:#ff9800;font-weight:bold}a{color:#6cf;text-decoration:none}</style></head><body>
<h3>EMA50 + RSI(45-50) Scan</h3><div>__META__</div>
<select id="k"><option value="">All</option><option>CROSS</option><option>NEAR</option></select>
<select id="t"><option value="">D + 4H</option><option>D</option><option>4H</option></select>
<select id="s"><option value="">All sectors</option></select>
<input id="q" placeholder="search stock">
<table><thead><tr><th data-k="kind">Signal</th><th data-k="symbol">Stock</th><th data-k="tf">TF</th>
<th data-k="sector">Sector</th><th data-k="rsi">RSI</th><th data-k="gap">Gap%</th></tr></thead><tbody id="b"></tbody></table>
<script>
const D=__DATA__;let sk="kind",asc=true;const $=id=>document.getElementById(id);
[...new Set(D.map(x=>x.sector))].sort().forEach(v=>$("s").add(new Option(v,v)));
function draw(){
 let r=D.filter(x=>(!$("k").value||x.kind==$("k").value)&&(!$("t").value||x.tf==$("t").value)&&(!$("s").value||x.sector==$("s").value)&&x.symbol.includes($("q").value.toUpperCase()));
 r.sort((a,b)=>(a[sk]>b[sk]?1:-1)*(asc?1:-1));
 $("b").innerHTML=r.map(x=>`<tr><td class="${x.kind}">${x.kind}</td><td><a href="https://in.tradingview.com/chart/?symbol=NSE:${x.symbol}" target="_blank">${x.symbol}</a></td><td>${x.tf}</td><td>${x.sector}</td><td>${x.rsi}</td><td>${x.gap}</td></tr>`).join("");}
["k","t","s","q"].forEach(i=>$(i).oninput=draw);
document.querySelectorAll("th").forEach(h=>h.onclick=()=>{const k=h.dataset.k;asc=(sk==k)?!asc:true;sk=k;draw()});
draw();
</script></body></html>"""
    html = html.replace("__DATA__", json.dumps(data)).replace("__META__", meta)
    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(html)


def load_universe():
    files = glob.glob("universe/*.csv")
    if not files:
        df = pd.read_csv("stocks.csv")
        return list(zip(df["symbol"], df["sector"]))
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d.columns = [c.strip() for c in d.columns]
        frames.append(d[["Symbol", "Industry"]].rename(
            columns={"Symbol": "symbol", "Industry": "sector"}))
    df = pd.concat(frames).dropna().drop_duplicates("symbol")
    return list(zip(df["symbol"].str.strip(), df["sector"].str.strip()))


def main():
    rows = load_universe()
    with ThreadPoolExecutor(max_workers=4) as ex:
        hits = [h for r in ex.map(scan, rows) for h in r]
    write_html(hits, len(rows))
    n = sum(1 for h in hits if h[3] == "CROSS")
    send(f"EMA50+RSI scan: {n} CROSS, {len(hits) - n} NEAR ({len(rows)} stocks)\n{PAGE}")


main()

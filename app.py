import io
import re
import sqlite3
import unicodedata
import datetime as dt
import calendar
from pathlib import Path

import pandas as pd
import streamlit as st
from pdfminer.high_level import extract_text  # pentru Tankpool PDF

# ================== CONFIG ==================
APP_TITLE = "SANS – Motorină & Predict (Tankpool)"
DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "fleet.db"
for p in (DATA_DIR, UPLOAD_DIR):
    p.mkdir(parents=True, exist_ok=True)

VAT_RATE = 0.19      # 19% TVA (DE)
FUEL_PRICE_GROSS = 1.6  # €/L cu TVA – pentru EXCEL (când nu avem preț din factură PDF)

# ================== FORMAT (stil DE) ==================
def de_thousands(n: float, dec: int = 0) -> str:
    """Format european: mii cu '.' și zecimale cu ','"""
    try:
        n = float(n)
    except Exception:
        n = 0.0
    s = f"{n:,.{dec}f}"           # US: 12,345.67
    s = s.replace(",", "X")       # X = thousands
    s = s.replace(".", ",")       # 12,345.67 -> 12,345,67 temporar
    s = s.replace("X", ".")       # -> 12.345,67
    return s

def de_eur(n: float, dec: int = 2) -> str:
    return f"{de_thousands(n, dec)} €"

# ================== DB ==================
@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys=ON;")
    migrate(conn)
    init_counters(conn)
    return conn

def migrate(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,          -- data livrării (alimentare + 1 zi)
        driver TEXT,
        route TEXT,
        vehicle TEXT,
        fuel_l REAL DEFAULT 0,
        fuel_cost_net REAL DEFAULT 0,
        fuel_cost_gross REAL DEFAULT 0,
        stops INTEGER DEFAULT 0,
        packages INTEGER DEFAULT 0,
        notes TEXT
    );
    """)
    cur.execute("PRAGMA table_info(entries);")
    have = {r[1] for r in cur.fetchall()}
    for col, decl in {
        "fuel_cost_net": "REAL DEFAULT 0",
        "fuel_cost_gross": "REAL DEFAULT 0",
    }.items():
        if col not in have:
            cur.execute(f"ALTER TABLE entries ADD COLUMN {col} {decl};")
    conn.commit()

def init_counters(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS counters (
        name TEXT PRIMARY KEY,       -- 'fuel' / 'packages'
        since TEXT NOT NULL
    );
    """)
    for name in ("fuel", "packages"):
        cur.execute("INSERT OR IGNORE INTO counters(name, since) VALUES(?, ?)", (name, "2000-01-01"))
    conn.commit()

def hard_reset_db():
    if DB_PATH.exists():
        DB_PATH.unlink()
    st.cache_resource.clear()
    st.toast("Baza de date a fost ștearsă. Dă refresh la pagină.", icon="⚠️")

def get_counter_since(conn, name: str) -> dt.date:
    cur = conn.cursor()
    cur.execute("SELECT since FROM counters WHERE name=?", (name,))
    row = cur.fetchone()
    return dt.date.fromisoformat(row[0]) if row and row[0] else dt.date(2000,1,1)

def set_counter_since(conn, name: str, new_date: dt.date):
    cur = conn.cursor()
    cur.execute("UPDATE counters SET since=? WHERE name=?", (new_date.isoformat(), name))
    conn.commit()

def insert_entry(conn, row: dict):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO entries(date, driver, route, vehicle, fuel_l, fuel_cost_net, fuel_cost_gross, stops, packages, notes)
        VALUES (:date,:driver,:route,:vehicle,:fuel_l,:fuel_cost_net,:fuel_cost_gross,:stops,:packages,:notes)
    """, row)
    conn.commit()

def load_entries_df(conn) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM entries", conn)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df

# ================== HELPERS ==================
def normalize_text(s: str) -> str:
    if not s: return ""
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s

def smart_number(x) -> float:
    """
    Parsează robust: '12,762' -> 12.762 ; '12.762' -> 12.762 ; '1.234,56' -> 1234.56 ; '1234' -> 1234
    Heuristică: dacă există și ',' și '.', zecimala este ultimul separator apărut în șir.
    Dacă există doar unul, dacă are 1-3 cifre după el => decimal, altfel => separator de mii.
    """
    if x is None: return 0.0
    s = str(x).strip().replace("\u00a0", "")
    if s == "": return 0.0
    if "," in s and "." in s:
        # ultimul separator din șir e cel zecimal
        last_sep = max(s.rfind(","), s.rfind("."))
        dec = s[last_sep]
        thou = "." if dec == "," else ","
        s = s.replace(thou, "")
        s = s.replace(dec, ".")
    elif "," in s:
        idx = s.rfind(",")
        digits_after = len(re.sub(r"\D", "", s[idx+1:]))
        if 1 <= digits_after <= 3:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "." in s:
        idx = s.rfind(".")
        digits_after = len(re.sub(r"\D", "", s[idx+1:]))
        if 1 <= digits_after <= 3:
            # '.' e zecimal
            pass
        else:
            s = s.replace(".", "")
    try:
        return float(s)
    except Exception:
        try:
            return float(str(x))
        except Exception:
            return 0.0

def month_last_day(d: dt.date) -> int:
    return calendar.monthrange(d.year, d.month)[1]

# ================== Tankpool PDF Parser ==================
# Linie tipică: "02.08.25 17:01 4457 Hengersberg Diesel 34,94 L 1,2819 44,79 ..."
PDF_LINE = re.compile(
    r"(?P<date>\d{2}\.\d{2}\.\d{2})\s+\d{2}:\d{2}\s+\d+\s+[A-Za-zÄÖÜäöüß.\- ]+\s+"
    r"(?P<prod>Diesel|AdBlue)\s+(?P<liters>[\d.,]+)\s+L\s+(?P<unit_net>[\d.,]+)\s+(?P<sum_net>[\d.,]+)",
    re.IGNORECASE
)

def parse_tankpool_pdf(content: bytes) -> pd.DataFrame:
    text = extract_text(io.BytesIO(content)) or ""
    rows = []
    for m in PDF_LINE.finditer(text):
        if m.group("prod").lower() != "diesel":
            continue
        liters = smart_number(m.group("liters"))
        unit_net = smart_number(m.group("unit_net"))
        sum_net = smart_number(m.group("sum_net"))
        try:
            d = pd.to_datetime(m.group("date"), format="%d.%m.%y").date()
        except Exception:
            continue
        net = sum_net if sum_net > 0 else liters * unit_net
        rows.append({
            "date": d,
            "vehicle": "AUTO",
            "fuel_l": liters,
            "fuel_cost_net": net,
            "fuel_cost_gross": net * (1 + VAT_RATE),
            "notes": "Tankpool PDF"
        })
    return pd.DataFrame(rows)

# ================== SIDEBAR ==================
def sidebar_summary(conn, key_suffix: str):
    df = load_entries_df(conn)
    fuel_since = get_counter_since(conn, "fuel")
    pkg_since  = get_counter_since(conn, "packages")

    if df.empty:
        l_sum = net_sum = gross_sum = pk_sum = 0
    else:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        l_sum     = df.loc[df["date"]>=fuel_since, "fuel_l"].sum()
        net_sum   = df.loc[df["date"]>=fuel_since, "fuel_cost_net"].sum()
        gross_sum = df.loc[df["date"]>=fuel_since, "fuel_cost_gross"].sum()
        pk_sum    = int(df.loc[df["date"]>=pkg_since,  "packages"].sum())

    st.markdown("### Sumar live")
    st.markdown(f"**⛽ Motorină acumulată:** {de_thousands(l_sum)} L • {de_eur(gross_sum)}  \n_net: {de_eur(net_sum)}_")
    st.markdown(f"**📦 Pachete acumulate:** {de_thousands(pk_sum)}")

    today = dt.date.today()
    last_day = month_last_day(today)
    if today.day in (16, last_day):
        st.warning("⚠️ Resetează **motorina** (Tankpool).")
    if today.day in (15, last_day):
        st.warning("⚠️ Resetează **pachetele**.")

    with st.expander("Administrare"):
        c1, c2, c3 = st.columns(3)
        if c1.button("Reset motorină", key=f"rf_{key_suffix}"):
            set_counter_since(conn, "fuel", dt.date.today()); st.success("Reset motorină.")
        if c2.button("Reset pachete", key=f"rp_{key_suffix}"):
            set_counter_since(conn, "packages", dt.date.today()); st.success("Reset pachete.")
        if c3.button("Hard reset DB", key=f"hard_{key_suffix}"):
            hard_reset_db()
            st.stop()

# ================== APP ==================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    conn = get_conn()

    with st.sidebar:
        sidebar_summary(conn, "top")

    st.subheader("1) Încarcă fișiere (Tankpool: Excel/PDF, Predict: manual)")
    uploads = st.file_uploader(
        "Selectează fișierele (poți mai multe odată). Pentru costuri corecte, PDF Tankpool este ideal.",
        type=["xls","xlsx","csv","pdf"],
        accept_multiple_files=True
    )

    results = []
    if uploads:
        today = dt.date.today()
        for up in uploads:
            name = up.name
            ext = Path(name).suffix.lower()
            raw = up.getvalue()

            # ---- Tankpool PDF -> litri + cost NET/BRUT din factură
            if ext == ".pdf":
                dfp = parse_tankpool_pdf(raw)
                ins = 0
                for _, r in dfp.iterrows():
                    delivery_date = r["date"] + dt.timedelta(days=1)  # livrarea = alimentare + 1 zi
                    insert_entry(conn, {
                        "date": delivery_date.isoformat(),
                        "driver": "AUTO",
                        "route": "AUTO",
                        "vehicle": r.get("vehicle","AUTO"),
                        "fuel_l": float(r["fuel_l"]),
                        "fuel_cost_net": float(r["fuel_cost_net"]),
                        "fuel_cost_gross": float(r["fuel_cost_gross"]),
                        "stops": 0, "packages": 0, "notes": r.get("notes","Tankpool PDF")
                    })
                    ins += 1
                results.append({"fișier": name, "tip": "TANKPOOL_PDF", "rows": ins, "mesaj": "OK"})
                continue

            # ---- Tankpool Excel/CSV -> litri + cost calculat cu 1.6 €/L brut
            if ext in (".xls", ".xlsx", ".csv"):
                # folosesc engine="python" la CSV pentru delimitator autodetect
                df_x = (pd.read_excel(io.BytesIO(raw))
                        if ext != ".csv"
                        else pd.read_csv(io.BytesIO(raw), sep=None, engine="python"))
                # mapare directă pe antetele exacte din fișierele tale Tankpool
                # (Datum, Tankmenge, Kennzeichen)
                ins = 0
                for _, r in df_x.iterrows():
                    liters = smart_number(r.get("Tankmenge", 0))
                    if liters <= 0:
                        continue
                    try:
                        refuel_date = pd.to_datetime(r.get("Datum"), dayfirst=True, errors="coerce").date()
                    except Exception:
                        refuel_date = today
                    delivery_date = refuel_date + dt.timedelta(days=1)
                    veh = str(r.get("Kennzeichen", "AUTO")).upper()

                    gross = liters * FUEL_PRICE_GROSS
                    net   = gross / (1 + VAT_RATE)

                    insert_entry(conn, {
                        "date": delivery_date.isoformat(),
                        "driver": "AUTO",
                        "route": veh,
                        "vehicle": veh,
                        "fuel_l": liters,
                        "fuel_cost_net": net,
                        "fuel_cost_gross": gross,
                        "stops": 0, "packages": 0, "notes": "Tankpool Excel"
                    })
                    ins += 1
                results.append({"fișier": name, "tip": "TANKPOOL_EXCEL", "rows": ins, "mesaj": "OK"})
                continue

        if results:
            st.success("Procesare terminată ✅")
            st.markdown("### Rezumat procesare")
            st.dataframe(pd.DataFrame(results), use_container_width=True)

        with st.sidebar:
            st.divider()
            sidebar_summary(conn, "after")

    # ---- STATISTICI ----
    st.subheader("2) Statistici")
    df = load_entries_df(conn)
    if df.empty:
        st.info("Nu există date încă."); return

    # Pe zi
    daily = df.groupby("date", as_index=False).agg({
        "fuel_l":"sum","fuel_cost_net":"sum","fuel_cost_gross":"sum","stops":"sum","packages":"sum"
    }).sort_values("date")
    daily_fmt = daily.copy()
    daily_fmt["fuel_l"] = daily_fmt["fuel_l"].apply(lambda x: de_thousands(x, 3))  # păstrez 3 zecimale la L
    daily_fmt["fuel_cost_net"] = daily_fmt["fuel_cost_net"].apply(lambda x: de_eur(x, 2))
    daily_fmt["fuel_cost_gross"] = daily_fmt["fuel_cost_gross"].apply(lambda x: de_eur(x, 2))
    st.markdown("### ▶ Pe zi (litri, cost net/brut, stopuri, pachete)")
    st.dataframe(daily_fmt.rename(columns={
        "fuel_l":"litri", "fuel_cost_net":"cost (net)", "fuel_cost_gross":"cost (brut)"
    }), use_container_width=True)

    # Pe tură (Kennzeichen)
    by_route = df.groupby("route", as_index=False).agg({
        "fuel_l":"sum","fuel_cost_net":"sum","fuel_cost_gross":"sum","stops":"sum","packages":"sum"
    }).sort_values(["fuel_l","fuel_cost_gross"], ascending=False)
    br_fmt = by_route.copy()
    br_fmt["fuel_l"] = br_fmt["fuel_l"].apply(lambda x: de_thousands(x, 3))
    br_fmt["fuel_cost_net"] = br_fmt["fuel_cost_net"].apply(lambda x: de_eur(x, 2))
    br_fmt["fuel_cost_gross"] = br_fmt["fuel_cost_gross"].apply(lambda x: de_eur(x, 2))
    st.markdown("### ▶ Pe tură (litri, cost net/brut, stopuri, pachete)")
    st.dataframe(br_fmt.rename(columns={
        "route":"tură", "fuel_l":"litri", "fuel_cost_net":"cost (net)", "fuel_cost_gross":"cost (brut)"
    }), use_container_width=True)

# ---------- RUN ----------
if __name__ == "__main__":
    get_conn()
    main()

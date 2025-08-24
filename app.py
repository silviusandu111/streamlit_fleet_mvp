import io
import re
import sqlite3
import unicodedata
import datetime as dt
import calendar
from pathlib import Path

import pandas as pd
import streamlit as st

# PDF reader
from pdfminer.high_level import extract_text

# ================== CONFIG ==================
APP_TITLE = "SANS – Motorină & Predict (Tankpool)"
DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "fleet.db"
for p in (DATA_DIR, UPLOAD_DIR):
    p.mkdir(parents=True, exist_ok=True)

VAT_RATE = 0.19  # 19% TVA

# ================== FORMAT (stil DE) ==================
def de_thousands(n: float, dec: int = 0) -> str:
    try:
        n = float(n)
    except Exception:
        n = 0.0
    s = f"{n:,.{dec}f}"
    s = s.replace(",", "X").replace(".", ".").replace("X", ",")  # US->DE
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
        date TEXT NOT NULL,
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
    if "fuel_cost_net" not in have:
        cur.execute("ALTER TABLE entries ADD COLUMN fuel_cost_net REAL DEFAULT 0;")
    if "fuel_cost_gross" not in have:
        cur.execute("ALTER TABLE entries ADD COLUMN fuel_cost_gross REAL DEFAULT 0;")
    conn.commit()

def init_counters(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS counters (
        name TEXT PRIMARY KEY,
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
def parse_number_de(x) -> float:
    if pd.isna(x): return 0.0
    s = str(x).strip().replace("\u00a0","")
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except:
        return 0.0

def month_last_day(d: dt.date) -> int:
    return calendar.monthrange(d.year, d.month)[1]

# ================== Tankpool PDF Parser ==================
LINE_RX = re.compile(
    r"(?P<date>\d{2}\.\d{2}\.\d{2})\s+\d{2}:\d{2}\s+\d+\s+[A-Za-zÄÖÜäöüß.\- ]+\s+"
    r"(?P<prod>Diesel|AdBlue)\s+(?P<liters>[\d.,]+)\s+L\s+(?P<unit_net>[\d.,]+)\s+(?P<sum_net>[\d.,]+)",
    re.IGNORECASE
)

def parse_tankpool_pdf(content: bytes) -> pd.DataFrame:
    text = extract_text(io.BytesIO(content)) or ""
    rows = []
    for m in LINE_RX.finditer(text):
        prod = m.group("prod").lower()
        if prod != "diesel":
            continue
        d = m.group("date")
        liters = parse_number_de(m.group("liters"))
        unit_net = parse_number_de(m.group("unit_net"))
        sum_net = parse_number_de(m.group("sum_net"))
        try:
            date = pd.to_datetime(d, format="%d.%m.%y").date()
        except:
            continue
        rows.append({
            "date": date,
            "vehicle": "AUTO",
            "fuel_l": liters,
            "fuel_cost_net": sum_net if sum_net > 0 else liters * unit_net,
            "fuel_cost_gross": (sum_net if sum_net > 0 else liters * unit_net) * (1 + VAT_RATE),
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

    st.markdown(f"### ⛽ Motorină acumulată\n**{de_thousands(l_sum)} L • {de_eur(gross_sum)}**\n\n_net: {de_eur(net_sum)}_")
    st.markdown(f"### 📦 Pachete acumulate\n**{de_thousands(pk_sum)}**")

    today = dt.date.today()
    last_day = month_last_day(today)
    if today.day in (16, last_day):
        st.warning("⚠️ Resetează **motorina** (facturare Tankpool).")
    if today.day in (15, last_day):
        st.warning("⚠️ Resetează **pachetele** (facturare Predict).")

    with st.expander("Administrare contoare"):
        c1, c2, c3 = st.columns(3)
        if c1.button("Reset motorină", key=f"btn_rf_{key_suffix}"):
            set_counter_since(conn, "fuel", dt.date.today()); st.success("Motorină resetată.")
        if c2.button("Reset pachete", key=f"btn_rp_{key_suffix}"):
            set_counter_since(conn, "packages", dt.date.today()); st.success("Pachete resetate.")
        if c3.button("Hard reset DB", key=f"btn_hard_{key_suffix}"):
            hard_reset_db()
            st.stop()

# ================== APP ==================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    conn = get_conn()

    with st.sidebar:
        sidebar_summary(conn, key_suffix="top")

    st.subheader("1) Încarcă fișiere (Tankpool: Excel/PDF, Predict: manual)")
    uploads = st.file_uploader(
        "Fișiere",
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

            # Tankpool PDF
            if ext == ".pdf":
                dfp = parse_tankpool_pdf(raw)
                ins = 0
                for _, r in dfp.iterrows():
                    delivery_date = r["date"] + dt.timedelta(days=1)
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

            # Tankpool Excel simplu
            if ext in (".xls", ".xlsx", ".csv"):
                df_x = (pd.read_excel(io.BytesIO(raw))
                        if ext != ".csv"
                        else pd.read_csv(io.BytesIO(raw), sep=None, engine="python"))
                ins = 0
                for _, r in df_x.iterrows():
                    liters = parse_number_de(r.get("Tankmenge", 0))
                    if liters <= 0: continue
                    try:
                        refuel_date = pd.to_datetime(r.get("Datum"), dayfirst=True, errors="coerce").date()
                    except Exception:
                        refuel_date = today
                    delivery_date = refuel_date + dt.timedelta(days=1)
                    insert_entry(conn, {
                        "date": delivery_date.isoformat(),
                        "driver": "AUTO","route":"AUTO","vehicle":"AUTO",
                        "fuel_l": liters,
                        "fuel_cost_net": 0.0,"fuel_cost_gross": 0.0,
                        "stops":0,"packages":0,"notes":"Tankpool Excel"
                    })
                    ins += 1
                results.append({"fișier": name, "tip": "TANKPOOL_EXCEL", "rows": ins, "mesaj": "OK"})
                continue

        if results:
            st.success("Procesare terminată ✅")
            st.dataframe(pd.DataFrame(results), use_container_width=True)

    # Statistici
    st.subheader("2) Statistici")
    df = load_entries_df(conn)
    if df.empty:
        st.info("Nu există date încă."); return

    daily = df.groupby("date", as_index=False).agg({
        "fuel_l":"sum","fuel_cost_net":"sum","fuel_cost_gross":"sum","stops":"sum","packages":"sum"
    }).sort_values("date")

    daily_fmt = daily.copy()
    daily_fmt["fuel_l"] = daily_fmt["fuel_l"].apply(lambda x: de_thousands(x,0))
    daily_fmt["fuel_cost_net"] = daily_fmt["fuel_cost_net"].apply(lambda x: de_eur(x,2))
    daily_fmt["fuel_cost_gross"] = daily_fmt["fuel_cost_gross"].apply(lambda x: de_eur(x,2))
    st.markdown("### ▶ Pe zi")
    st.dataframe(daily_fmt, use_container_width=True)

    by_route = df.groupby("route", as_index=False).agg({
        "fuel_l":"sum","fuel_cost_net":"sum","fuel_cost_gross":"sum","stops":"sum","packages":"sum"
    })
    br_fmt = by_route.copy()
    br_fmt["fuel_l"] = br_fmt["fuel_l"].apply(lambda x: de_thousands(x,0))
    br_fmt["fuel_cost_net"] = br_fmt["fuel_cost_net"].apply(lambda x: de_eur(x,2))
    br_fmt["fuel_cost_gross"] = br_fmt["fuel_cost_gross"].apply(lambda x: de_eur(x,2))
    st.markdown("### ▶ Pe tură")
    st.dataframe(br_fmt, use_container_width=True)

# ---------- RUN ----------
if __name__ == "__main__":
    get_conn()
    main()

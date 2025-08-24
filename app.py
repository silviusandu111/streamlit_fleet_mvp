import io
import re
import sqlite3
import unicodedata
import datetime as dt
from pathlib import Path
from typing import Optional
import pandas as pd
import streamlit as st

# ================== CONFIG ==================
APP_TITLE = "SANS – Motorină & Predict"
FUEL_PRICE = 1.6  # €/L

DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "fleet.db"
for p in [DATA_DIR, UPLOAD_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ================== DB ==================
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        driver TEXT,
        route TEXT,
        vehicle TEXT,
        fuel_l REAL DEFAULT 0,
        fuel_cost REAL DEFAULT 0,
        stops INTEGER DEFAULT 0,
        packages INTEGER DEFAULT 0,
        notes TEXT
    );
    """)
    conn.commit()
    conn.close()

def insert_entry(row: dict):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO entries(date, driver, route, vehicle, fuel_l, fuel_cost, stops, packages, notes)
    VALUES (:date,:driver,:route,:vehicle,:fuel_l,:fuel_cost,:stops,:packages,:notes)
    """, row)
    conn.commit()
    conn.close()

def load_entries_df():
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM entries", conn)
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df

# ================== PARSERS ==================
def normalize_text(s: str) -> str:
    if not s: return ""
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s

def parse_packages(text: str) -> Optional[int]:
    m = re.search(r"geplante\s+zustellpaket(?:e|te)?\s*[:\-]?\s*([0-9]+)", normalize_text(text))
    return int(m.group(1)) if m else None

def parse_stops(text: str) -> Optional[int]:
    m = re.search(r"stops?\s*[:\-]?\s*([0-9]+)", normalize_text(text))
    return int(m.group(1)) if m else None

def parse_driver(text: str) -> Optional[str]:
    m = re.search(r"(fahrer|driver|sofer)\s*[:\-]?\s*([a-z0-9 ._-]{2,})", normalize_text(text))
    return m.group(2).title() if m else None

def parse_route(text: str) -> Optional[str]:
    m = re.search(r"(tour|tura|route)\s*[:\-]?\s*([a-z0-9 /._-]{2,})", normalize_text(text))
    return m.group(2).upper() if m else None

# ================== TANKPOOL HEADERS ==================
HEADER_ALIASES = {
    "date":   ["datum","date","datum_tankung"],
    "vehicle":["kennzeichen","fahrzeug","vehicle"],
    "fuel_l": ["tankmenge","menge","liter","l"],
}

def norm(s: str) -> str:
    s = normalize_text(str(s))
    s = re.sub(r"[^a-z0-9_ ]+","",s)
    return s.replace(" ","_")

def map_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = [norm(c) for c in df.columns]
    df.columns = cols
    rename = {}
    for canonical, alts in HEADER_ALIASES.items():
        for c in cols:
            if c in alts: rename[c] = canonical
    return df.rename(columns=rename)

# ================== MAIN ==================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    today = dt.date.today()

    # ========== SIDEBAR LIVE SUMMARY ==========
    with st.sidebar:
        df_all = load_entries_df()
        total_l = df_all["fuel_l"].sum() if not df_all.empty else 0
        total_c = df_all["fuel_cost"].sum() if not df_all.empty else 0
        total_p = df_all["packages"].sum() if not df_all.empty else 0
        st.metric("⛽ Motorină acumulată", f"{total_l:.0f} L / {total_c:.0f} €")
        st.metric("📦 Pachete acumulate", f"{int(total_p)}")

        # Resetări automate
        last_day = (dt.date(today.year, today.month+1,1)-dt.timedelta(days=1)).day if today.month<12 else 31
        if today.day in [16,last_day]:
            st.warning("⚠️ Resetează contorul de MOTORINĂ (facturare Tankpool).")
        if today.day in [15,last_day]:
            st.warning("⚠️ Resetează contorul de PACHETE (facturare Predict).")

    # ========== UPLOAD ==========
    st.subheader("1) Încarcă fișiere Tankpool (Excel) sau Predict (PDF/imagini)")
    uploads = st.file_uploader("Fișiere", type=["xls","xlsx","csv","pdf","png","jpg","jpeg"], accept_multiple_files=True)

    if uploads:
        for up in uploads:
            ext = Path(up.name).suffix.lower()
            raw = up.getvalue()
            # Tankpool Excel
            if ext in [".xls",".xlsx",".csv"]:
                df_x = pd.read_excel(io.BytesIO(raw)) if ext != ".csv" else pd.read_csv(io.BytesIO(raw))
                df_x = map_headers(df_x)
                for _, r in df_x.iterrows():
                    try:
                        liters = float(str(r.get("fuel_l",0)).replace(",", "."))
                    except: liters = 0
                    if liters <= 0: continue
                    try:
                        refuel_date = pd.to_datetime(r.get("date")).date()
                        delivery_date = refuel_date + dt.timedelta(days=1)
                    except: delivery_date = today
                    veh = str(r.get("vehicle","AUTO")).upper()
                    insert_entry({
                        "date": delivery_date.isoformat(),
                        "driver":"AUTO",
                        "route":veh,
                        "vehicle":veh,
                        "fuel_l":liters,
                        "fuel_cost":liters*FUEL_PRICE,
                        "stops":0,"packages":0,
                        "notes":"Tankpool"
                    })
            else:
                # Predict simplu: formular manual
                with st.expander(f"Adaugă manual date Predict pentru {up.name}"):
                    drv = st.text_input("Șofer", key=f"drv_{up.name}")
                    rte = st.text_input("Tură", key=f"rte_{up.name}")
                    stops = st.number_input("Stopuri", min_value=0, step=1, key=f"stops_{up.name}")
                    pkgs = st.number_input("Pachete", min_value=0, step=1, key=f"pkgs_{up.name}")
                    if st.button("Salvează", key=f"btn_{up.name}"):
                        insert_entry({
                            "date": today.isoformat(),
                            "driver":drv or "AUTO",
                            "route":rte or "AUTO",
                            "vehicle":"AUTO",
                            "fuel_l":0,"fuel_cost":0,
                            "stops":int(stops),"packages":int(pkgs),
                            "notes":"Predict manual"
                        })
                        st.success("Predict salvat.")

        st.success("Fișiere procesate ✅")

    # ========== STATISTICI ==========
    st.subheader("2) Statistici")
    df = load_entries_df()
    if df.empty:
        st.info("Nu există date încă.")
        return

    daily = df.groupby("date", as_index=False).agg({
        "fuel_l":"sum","fuel_cost":"sum","stops":"sum","packages":"sum"
    })
    st.markdown("### ▶ Pe zi")
    st.dataframe(daily, use_container_width=True)

    by_route = df.groupby("route", as_index=False).agg({
        "fuel_l":"sum","fuel_cost":"sum","stops":"sum","packages":"sum"
    })
    st.markdown("### ▶ Pe tură")
    st.dataframe(by_route, use_container_width=True)

# ---------- RUN ----------
if __name__=="__main__":
    init_db()
    main()

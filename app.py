import io
import re
import uuid
import sqlite3
import unicodedata
import datetime as dt
from pathlib import Path
from typing import Optional
import pandas as pd
import streamlit as st

# ================== Config ==================
APP_TITLE = "SANS – Motorină Tankpool + Predict"
FUEL_PRICE = 1.6  # €/L cu TVA

DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "fleet.db"
for p in [DATA_DIR, UPLOAD_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ================== DB ==================
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
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

def load_entries_df() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM entries", conn)
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df

# ================== Parsers ==================
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

# ================== Tankpool Excel ==================
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
            if c in alts:
                rename[c] = canonical
    return df.rename(columns=rename)

# ================== MAIN ==================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    with st.sidebar:
        today = dt.date.today()
        st.markdown("### Rezumat live")
        df_all = load_entries_df()
        if not df_all.empty:
            total_l = df_all["fuel_l"].sum()
            total_c = df_all["fuel_cost"].sum()
            total_p = df_all["packages"].sum()
            st.metric("⛽ Litri acumulați", f"{total_l:.0f} L")
            st.metric("💶 Cost motorină", f"{total_c:.0f} €")
            st.metric("📦 Pachete livrate", f"{int(total_p)}")

    st.subheader("1) Încarcă fișiere (Tankpool Excel sau Predict PDF/imagini)")
    uploads = st.file_uploader("Fișiere", type=["xls","xlsx","csv","pdf","png","jpg","jpeg"], accept_multiple_files=True)

    if uploads:
        for up in uploads:
            ext = Path(up.name).suffix.lower()
            raw = up.getvalue()

            if ext in [".xls",".xlsx",".csv"]:
                df_x = pd.read_excel(io.BytesIO(raw)) if ext != ".csv" else pd.read_csv(io.BytesIO(raw))
                df_x = map_headers(df_x)
                for _, r in df_x.iterrows():
                    try:
                        liters = float(str(r.get("fuel_l",0)).replace(",", "."))
                    except:
                        liters = 0
                    if liters <= 0: continue
                    try:
                        refuel_date = pd.to_datetime(r.get("date")).date()
                        delivery_date = refuel_date + dt.timedelta(days=1)
                    except:
                        delivery_date = dt.date.today()
                    veh = str(r.get("vehicle","AUTO")).upper()
                    insert_entry({
                        "date": delivery_date.isoformat(),
                        "driver": "AUTO",
                        "route": veh,
                        "vehicle": veh,
                        "fuel_l": liters,
                        "fuel_cost": liters * FUEL_PRICE,
                        "stops": 0,
                        "packages": 0,
                        "notes": "Tankpool Excel"
                    })
            else:
                # Predict: aici poți adăuga OCR sau completare manuală
                st.warning(f"OCR pentru {up.name} nu e implementat complet – introdu manual pentru test.")

        st.success("Procesare terminată ✅")

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

if __name__=="__main__":
    init_db()
    main()

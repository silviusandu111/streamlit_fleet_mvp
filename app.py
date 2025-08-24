import io
import re
import sqlite3
import unicodedata
import datetime as dt
import calendar
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

# ================== DB HELPERS ==================
def get_conn():
    return sqlite3.connect(DB_PATH)

def table_columns(conn, table: str):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table});")
    return [row[1] for row in cur.fetchall()]

def init_db_with_migrations():
    """
    Creează tabelele dacă lipsesc.
    Dacă 'entries' există dar îi lipsesc coloane noi (stops, packages, fuel_cost etc),
    le adaugă prin ALTER TABLE ca să nu mai dea 'no such column'.
    """
    conn = get_conn()
    cur = conn.cursor()

    # creează dacă nu există
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

    # migrare: asigură coloanele
    cols = table_columns(conn, "entries")
    needed = {
        "date": "TEXT NOT NULL DEFAULT '2000-01-01'",
        "driver": "TEXT",
        "route": "TEXT",
        "vehicle": "TEXT",
        "fuel_l": "REAL DEFAULT 0",
        "fuel_cost": "REAL DEFAULT 0",
        "stops": "INTEGER DEFAULT 0",
        "packages": "INTEGER DEFAULT 0",
        "notes": "TEXT"
    }
    for col, decl in needed.items():
        if col not in cols:
            cur.execute(f"ALTER TABLE entries ADD COLUMN {col} {decl};")

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

# ================== PARSERS ==================
def normalize_text(s: str) -> str:
    if not s: return ""
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s

def norm_col(s: str) -> str:
    s = normalize_text(str(s))
    s = re.sub(r"[^a-z0-9_ ]+","",s)
    return s.replace(" ","_")

# antete uzuale Tankpool
HEADER_ALIASES = {
    "date":   ["datum","date","datum_tankung","tankdatum","belegdatum"],
    "vehicle":["kennzeichen","fahrzeug","vehicle","kennz"],
    "fuel_l": ["tankmenge","menge","liter","l","betankte_menge"]
}

def map_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = [norm_col(c) for c in df.columns]
    df.columns = cols
    rename = {}
    for canonical, alts in HEADER_ALIASES.items():
        for c in cols:
            if c in alts:
                rename[c] = canonical
    # fallback: dacă există 'menge' și nu s-a mapat deja
    if "fuel_l" not in rename.values() and "menge" in cols:
        rename["menge"] = "fuel_l"
    return df.rename(columns=rename)

# ================== UI HELPERS ==================
def month_last_day(d: dt.date) -> int:
    return calendar.monthrange(d.year, d.month)[1]

def sidebar_thumbnails():
    today = dt.date.today()
    df_all = load_entries_df()
    total_l = df_all["fuel_l"].sum() if not df_all.empty else 0.0
    total_c = df_all["fuel_cost"].sum() if not df_all.empty else 0.0
    total_p = df_all["packages"].sum() if not df_all.empty else 0

    st.metric("⛽ Motorină acumulată", f"{total_l:.0f} L / {total_c:.0f} €")
    st.metric("📦 Pachete acumulate", f"{int(total_p)}")

    last_day = month_last_day(today)
    if today.day in (16, last_day):
        st.warning("⚠️ Resetează contorul de **motorină** (facturare Tankpool).")
    if today.day in (15, last_day):
        st.warning("⚠️ Resetează contorul de **pachete** (facturare Predict).")

# ================== APP ==================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    with st.sidebar:
        st.header("Sumar live")
        sidebar_thumbnails()

    st.subheader("1) Încarcă fișiere Tankpool (Excel) sau Predict (PDF/imagini)")
    uploads = st.file_uploader(
        "Fișiere",
        type=["xls","xlsx","csv","pdf","png","jpg","jpeg"],
        accept_multiple_files=True
    )

    if uploads:
        today = dt.date.today()
        for up in uploads:
            ext = Path(up.name).suffix.lower()
            raw = up.getvalue()

            # --- TANKPOOL EXCEL ---
            if ext in (".xls", ".xlsx", ".csv"):
                try:
                    df_x = pd.read_excel(io.BytesIO(raw)) if ext != ".csv" else pd.read_csv(io.BytesIO(raw))
                except Exception as e:
                    st.error(f"Eroare la citirea {up.name}: {e}")
                    continue

                df_x = map_headers(df_x)
                cols = set(df_x.columns)
                if not {"date","fuel_l"}.issubset(cols):
                    st.error(f"{up.name}: nu am găsit coloanele necesare (date/fuel_l). Antete detectate: {list(df_x.columns)}")
                    continue

                inserted = 0
                for _, r in df_x.iterrows():
                    # litri
                    try:
                        liters = float(str(r.get("fuel_l", 0)).replace(",", "."))
                    except Exception:
                        liters = 0.0
                    if liters <= 0:
                        continue

                    # data livrării = datum + 1 zi
                    try:
                        refuel_date = pd.to_datetime(r.get("date")).date()
                        delivery_date = refuel_date + dt.timedelta(days=1)
                    except Exception:
                        delivery_date = today

                    veh = str(r.get("vehicle", "AUTO")).upper() if "vehicle" in df_x.columns else "AUTO"

                    insert_entry({
                        "date": delivery_date.isoformat(),
                        "driver": "AUTO",
                        "route": veh,      # folosim Kennzeichen drept tură
                        "vehicle": veh,
                        "fuel_l": liters,
                        "fuel_cost": liters * FUEL_PRICE,
                        "stops": 0,
                        "packages": 0,
                        "notes": "Tankpool"
                    })
                    inserted += 1

                st.success(f"{up.name}: {inserted} rânduri importate (Tankpool).")

            # --- PREDICT: pentru simplitate aici introducere manuală ---
            else:
                with st.expander(f"Adaugă manual Predict pentru {up.name}"):
                    date_val = st.date_input("Data", value=dt.date.today(), key=f"d_{up.name}")
                    drv = st.text_input("Șofer", key=f"drv_{up.name}")
                    rte = st.text_input("Tură", key=f"rte_{up.name}")
                    stops = st.number_input("Stopuri", min_value=0, step=1, key=f"s_{up.name}")
                    pkgs = st.number_input("Pachete", min_value=0, step=1, key=f"p_{up.name}")
                    if st.button("Salvează", key=f"btn_{up.name}"):
                        insert_entry({
                            "date": date_val.isoformat(),
                            "driver": drv or "AUTO",
                            "route": (rte or "AUTO").upper(),
                            "vehicle": "AUTO",
                            "fuel_l": 0,
                            "fuel_cost": 0,
                            "stops": int(stops),
                            "packages": int(pkgs),
                            "notes": "Predict manual"
                        })
                        st.success("Predict salvat.")

        # Re-afișează thumbnail-urile cu totaluri actualizate
        with st.sidebar:
            st.divider()
            sidebar_thumbnails()

    # --- STATISTICI ---
    st.subheader("2) Statistici")
    df = load_entries_df()
    if df.empty:
        st.info("Nu există date încă.")
        return

    # Zilnic
    daily = df.groupby("date", as_index=False).agg({
        "fuel_l": "sum",
        "fuel_cost": "sum",
        "stops": "sum",
        "packages": "sum"
    })
    st.markdown("### ▶ Pe zi")
    st.dataframe(daily.sort_values("date"), use_container_width=True)

    # Pe tură (Kennzeichen)
    by_route = df.groupby("route", as_index=False).agg({
        "fuel_l": "sum",
        "fuel_cost": "sum",
        "stops": "sum",
        "packages": "sum"
    })
    st.markdown("### ▶ Pe tură")
    st.dataframe(by_route.sort_values(["fuel_l","fuel_cost"], ascending=False), use_container_width=True)

# ---------- RUN ----------
if __name__ == "__main__":
    init_db_with_migrations()
    main()

import os
import io
import re
import uuid
import sqlite3
import datetime as dt
from pathlib import Path
from typing import Optional, List

import pandas as pd
import streamlit as st

# ---- Config & Paths ----
APP_TITLE = "SANS – Fleet MVP (Streamlit)"
DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "fleet.db"
FUEL_PRICE = 1.6   # €/litru cu TVA inclus

for p in [DATA_DIR, UPLOAD_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ---- DB Helpers ----
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    # Înregistrări zilnice
    cur.execute("""
    CREATE TABLE IF NOT EXISTS entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        driver TEXT NOT NULL,
        route TEXT NOT NULL,
        vehicle TEXT NOT NULL,
        km REAL DEFAULT 0,
        fuel_l REAL DEFAULT 0,
        fuel_cost REAL DEFAULT 0,
        hours REAL DEFAULT 0,
        revenue REAL DEFAULT 0,
        stops INTEGER DEFAULT 0,
        notes TEXT
    );
    """)
    conn.commit()
    conn.close()

init_db()

# ---- OCR Helpers ----
def try_ocr_image(image_bytes: bytes) -> Optional[str]:
    try:
        from PIL import Image
        import pytesseract
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        return text
    except Exception:
        return None

def try_ocr_pdf(pdf_bytes: bytes) -> Optional[str]:
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                text_parts.append(t)
        return "\n".join(text_parts).strip()
    except Exception:
        return None

# ---- Parsere text OCR ----
NUM = r"([0-9]+(?:[.,][0-9]+)?)"

def parse_km(text: str) -> Optional[float]:
    if not text: return None
    for pat in [rf"\b{NUM}\s*km\b", rf"km\s*{NUM}\b"]:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try: return float(m.group(1).replace(",", "."))
            except: return None
    return None

def parse_liters(text: str) -> Optional[float]:
    if not text: return None
    # caută "menge" sau "litri/liter/L"
    for pat in [rf"\bmenge\s*{NUM}\b", rf"\b{NUM}\s*(l|litri|liter)\b"]:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try: return float(m.group(1).replace(",", "."))
            except: return None
    return None

def parse_euros(text: str) -> Optional[float]:
    if not text: return None
    for pat in [rf"\b{NUM}\s*(€|eur|euro)\b"]:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try: return float(m.group(1).replace(",", "."))
            except: return None
    return None

# ---- Helpers DB/UI ----
def load_dataframe(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM entries ORDER BY date DESC, id DESC", conn, parse_dates=["date"])

def upsert_entry(data: dict):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO entries (date, driver, route, vehicle, km, fuel_l, fuel_cost, hours, revenue, stops, notes)
        VALUES (:date, :driver, :route, :vehicle, :km, :fuel_l, :fuel_cost, :hours, :revenue, :stops, :notes)
    """, data)
    entry_id = cur.lastrowid
    conn.commit()
    conn.close()
    return entry_id

# ---- Streamlit App ----
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Upload → OCR (opțional) → Confirm → Import Excel/CSV (opțional) → Statistici → Export")

with st.sidebar:
    st.header("Filtre & Perioadă")
    today = dt.date.today()
    month = st.selectbox("Luna", options=list(range(1,13)), index=today.month-1,
                         format_func=lambda m: dt.date(2000, m, 1).strftime("%B"))
    year = st.number_input("An", value=today.year, step=1)
    driver_filter = st.text_input("Filtru șofer")
    route_filter = st.text_input("Filtru tură")
    vehicle_filter = st.text_input("Filtru mașină")
    st.markdown("---")
    st.info("💡 Poți încărca imagini, PDF **și** Excel/CSV. OCR este best-effort; confirmi manual valorile.")

# ----------------- 1) Upload documente -----------------
st.subheader("1) Upload documente (jpg/png/pdf/xls/xlsx/csv)")

uploads = st.file_uploader(
    "Încarcă fișiere",
    type=["png","jpg","jpeg","pdf","xls","xlsx","csv"],
    accept_multiple_files=True
)

ocr_text = ""
excel_frames: List[pd.DataFrame] = []

if uploads:
    month_dir = UPLOAD_DIR / f"{year}-{str(month).zfill(2)}"
    month_dir.mkdir(parents=True, exist_ok=True)

    for up in uploads:
        content = up.getbuffer()
        ext = Path(up.name).suffix.lower()

        # OCR
        text = None
        if ext in [".png",".jpg",".jpeg"]:
            text = try_ocr_image(bytes(content))
        elif ext == ".pdf":
            text = try_ocr_pdf(bytes(content))
        elif ext in [".xlsx", ".xls"]:
            try:
                df_x = pd.read_excel(io.BytesIO(bytes(content)))
                excel_frames.append(df_x)
            except Exception as e:
                st.error(f"Nu pot citi {up.name} ca Excel: {e}")
        elif ext == ".csv":
            try:
                df_x = pd.read_csv(io.BytesIO(bytes(content)))
                excel_frames.append(df_x)
            except Exception as e:
                st.error(f"Nu pot citi {up.name} ca CSV: {e}")

        if text:
            ocr_text += "\n" + text

prefill_km = parse_km(ocr_text) if ocr_text else None
prefill_l = parse_liters(ocr_text) if ocr_text else None

# ----------------- 2) Formular manual -----------------
st.subheader("2) Confirmă / completează datele manual")

with st.form("entry_form", clear_on_submit=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        date_val = st.date_input("Data", value=dt.date.today())
        hours = st.number_input("Ore lucrate", value=0.0, step=0.5, min_value=0.0)
    with c2:
        driver = st.text_input("Șofer", "")
        vehicle = st.text_input("Mașină", "")
    with c3:
        route = st.text_input("Tură", "")
        km = st.number_input("KM", value=float(prefill_km or 0), step=1.0, min_value=0.0)
    with c4:
        fuel_l = st.number_input("Motorină (L)", value=float(prefill_l or 0), step=0.1, min_value=0.0)

    revenue = st.number_input("Încasări (€/zi tură)", value=0.0, step=10.0, min_value=0.0)
    stops = st.number_input("Stopuri (colete)", value=0, step=1, min_value=0)
    notes = st.text_area("Note", value=(ocr_text[:500] if ocr_text else ""))

    submitted = st.form_submit_button("💾 Salvează")
    if submitted:
        if not driver or not route or not vehicle:
            st.error("Completează **Șofer**, **Tură**, **Mașină**.")
        else:
            payload = {
                "date": str(date_val),
                "driver": driver.strip(),
                "route": route.strip(),
                "vehicle": vehicle.strip(),
                "km": float(km),
                "fuel_l": float(fuel_l),
                "fuel_cost": float(fuel_l) * FUEL_PRICE,   # calcul automat
                "hours": float(hours),
                "revenue": float(revenue),
                "stops": int(stops),
                "notes": notes.strip()
            }
            eid = upsert_entry(payload)
            st.success(f"Înregistrare salvată (ID {eid}).")

# ----------------- 2b) Import Excel/CSV -----------------
st.subheader("2b) Import din Excel/CSV")

if excel_frames:
    excel_df = pd.concat(excel_frames, ignore_index=True)
    st.write("Previzualizare:")
    st.dataframe(excel_df.head(20), use_container_width=True)

    required = ['date','driver','route','vehicle','km','fuel_l','hours','revenue','stops','notes']
    missing = [c for c in required if c not in excel_df.columns]
    if missing:
        st.warning(f"Lipsesc coloane: {missing}")
    else:
        if st.button("📥 Importă rândurile"):
            def to_float(x): 
                try: return float(str(x).replace(",", ".")) if pd.notnull(x) else 0.0
                except: return 0.0
            def to_iso(x):
                try: return pd.to_datetime(x).date().isoformat()
                except: return str(x)

            conn = get_conn()
            cur = conn.cursor()
            imported = 0
            for _, row in excel_df.iterrows():
                cur.execute("""
                    INSERT INTO entries (date, driver, route, vehicle, km, fuel_l, fuel_cost, hours, revenue, stops, notes)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    to_iso(row['date']),
                    str(row['driver']).strip(),
                    str(row['route']).strip(),
                    str(row['vehicle']).strip(),
                    to_float(row['km']),
                    to_float(row['fuel_l']),
                    to_float(row['fuel_l']) * FUEL_PRICE,
                    to_float(row.get('hours', 0)),
                    to_float(row.get('revenue', 0)),
                    int(row.get('stops', 0)),
                    str(row.get('notes', '')).strip()
                ))
                imported += 1
            conn.commit()
            conn.close()
            st.success(f"Import finalizat: {imported} rânduri.")
else:
    st.caption("Încarcă fișier Excel/CSV pentru import.")

# ----------------- 3) Statistici -----------------
st.subheader("3) Statistici zilnice & lunare")

conn = get_conn()
df = load_dataframe(conn)
conn.close()

if df.empty:
    st.warning("Nu există date.")
else:
    df['date'] = pd.to_datetime(df['date']).dt.date
    mask = (pd.to_datetime(df['date']).dt.month == month) & (pd.to_datetime(df['date']).dt.year == year)
    if driver_filter: mask &= df['driver'].str.contains(driver_filter, case=False, na=False)
    if route_filter: mask &= df['route'].str.contains(route_filter, case=False, na=False)
    if vehicle_filter: mask &= df['vehicle'].str.contains(vehicle_filter, case=False, na=False)
    fdf = df[mask].copy()

    st.markdown("### Zilnic")
    daily = fdf.groupby('date', as_index=False).agg({
        'km':'sum','fuel_l':'sum','fuel_cost':'sum','hours':'sum','revenue':'sum','stops':'sum'
    })
    st.dataframe(daily, use_container_width=True)

    st.markdown("### Per șofer (lunar)")
    by_driver = fdf.groupby('driver', as_index=False).agg({
        'km':'sum','fuel_l':'sum','fuel_cost':'sum','hours':'sum','revenue':'sum','stops':'sum'
    })
    by_driver['profit'] = by_driver['revenue'] - by_driver['fuel_cost']
    st.dataframe(by_driver.sort_values('profit', ascending=False), use_container_width=True)

    st.markdown("### Per tură (lunar)")
    by_route = fdf.groupby('route', as_index=False).agg({
        'km':'sum','fuel_l':'sum','fuel_cost':'sum','hours':'sum','revenue':'sum','stops':'sum'
    })
    by_route['profit'] = by_route['revenue'] - by_route['fuel_cost']
    st.dataframe(by_route.sort_values('profit', ascending=False), use_container_width=True)

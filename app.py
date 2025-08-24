import io
import re
import uuid
import sqlite3
import unicodedata
import datetime as dt
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import pandas as pd
import streamlit as st

# ================== Config & Paths ==================
APP_TITLE = "SANS – Fleet MVP (Streamlit)"
FUEL_PRICE = 1.6  # €/litru (cu TVA)

DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
EXPORT_DIR = DATA_DIR / "exports"
DB_PATH = DATA_DIR / "fleet.db"
MASTER_XLSX = EXPORT_DIR / "master.xlsx"

for p in [DATA_DIR, UPLOAD_DIR, EXPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ================== DB Helpers ==================
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
    cur.execute("""
    CREATE TABLE IF NOT EXISTS predict_context (
        date TEXT PRIMARY KEY,
        driver TEXT,
        route TEXT,
        vehicle TEXT,
        stops INTEGER DEFAULT 0,
        raw_text TEXT
    );
    """)
    conn.commit()
    conn.close()

def insert_entry(row: dict) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO entries
        (date, driver, route, vehicle, km, fuel_l, fuel_cost, hours, revenue, stops, notes)
        VALUES (:date, :driver, :route, :vehicle, :km, :fuel_l, :fuel_cost, :hours, :revenue, :stops, :notes)
    """, row)
    eid = cur.lastrowid
    conn.commit()
    conn.close()
    return eid

def upsert_predict_context(date_iso: str, driver: Optional[str], route: Optional[str],
                           vehicle: Optional[str], stops: Optional[int], raw_text: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO predict_context(date, driver, route, vehicle, stops, raw_text)
        VALUES(?,?,?,?,?,?)
        ON CONFLICT(date) DO UPDATE SET
          driver=coalesce(excluded.driver, predict_context.driver),
          route=coalesce(excluded.route, predict_context.route),
          vehicle=coalesce(excluded.vehicle, predict_context.vehicle),
          stops=coalesce(excluded.stops, predict_context.stops),
          raw_text=excluded.raw_text
    """, (date_iso, driver, route, vehicle, stops, raw_text))
    conn.commit()
    conn.close()

def get_predict_context_for(date_iso: str) -> Optional[dict]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT date, driver, route, vehicle, stops FROM predict_context WHERE date=?", (date_iso,))
    row = cur.fetchone()
    if not row:
        d = dt.date.fromisoformat(date_iso) - dt.timedelta(days=1)
        cur.execute("SELECT date, driver, route, vehicle, stops FROM predict_context WHERE date=?", (d.isoformat(),))
        row = cur.fetchone()
    conn.close()
    if row:
        return {"date": row[0], "driver": row[1] or "", "route": row[2] or "", "vehicle": row[3] or "", "stops": row[4] or 0}
    return None

def load_entries_df() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM entries ORDER BY date DESC, id DESC", conn, parse_dates=["date"])
    conn.close()
    return df

# ================== OCR & Parsers ==================
NUM = r"([0-9]+(?:[.,][0-9]+)?)"

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s

def try_ocr_image(image_bytes: bytes) -> Optional[str]:
    try:
        from PIL import Image
        import pytesseract
        image = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(image)
    except Exception:
        return None

def try_ocr_pdf(pdf_bytes: bytes) -> Optional[str]:
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts).strip()
    except Exception:
        return None

def parse_stops_geplante(text: str) -> Optional[int]:
    if not text:
        return None
    t = normalize_text(text)
    m = re.search(r"geplante\s+zustellpaket(?:e|te)?\s*[:\-]?\s*([0-9]+)", t)
    if m:
        try:
            return int(m.group(1))
        except:
            return None
    return None

def parse_driver(text: str) -> Optional[str]:
    if not text:
        return None
    t = normalize_text(text)
    pattern = r"(fahrer|driver|sofer)\s*[:\-]?\s*([a-z0-9 ._-]{2,})"
    m = re.search(pattern, t)
    if m:
        return m.group(2).strip().title()[:50]
    return None

def parse_route(text: str) -> Optional[str]:
    if not text:
        return None
    t = normalize_text(text)
    pattern = r"(tour|tura|route)\s*[:\-]?\s*([a-z0-9 /._-]{2,})"
    m = re.search(pattern, t)
    if m:
        return m.group(2).strip().upper()[:50]
    return None

def parse_vehicle(text: str) -> Optional[str]:
    if not text:
        return None
    t = normalize_text(text)
    m = re.search(r"\b([a-z]{1,3}\s?[a-z]{1,3}\s?[0-9]{1,4})\b", t)
    if m:
        return m.group(1).upper()[:20]
    return None

def parse_liters(text: str) -> Optional[float]:
    if not text:
        return None
    t = normalize_text(text)
    m = re.search(rf"\bmenge\s+{NUM}\b", t)
    if m:
        try:
            return float(m.group(1).replace(",", "."))
        except:
            pass
    m = re.search(rf"\b{NUM}\s*(l|liter|litri)\b", t)
    if m:
        try:
            return float(m.group(1).replace(",", "."))
        except:
            pass
    return None

# ================== Header mapping ==================
HEADER_ALIASES = {
    "date":   ["date","datum","data","tag","day","datum_tankung","belegdatum"],
    "driver": ["driver","fahrer","sofer","chauffeur","conducator"],
    "route":  ["route","tour","tura","tur","linie"],
    "vehicle":["vehicle","vehicul","fahrzeug","auto","masina","kennzeichen","nr_inmatriculare","license_plate"],
    "km":     ["km","kilometer","kilometri","strecke","distanz"],
    "hours":  ["hours","ore","stunden","zeit","arbeitszeit"],
    "revenue":["revenue","venit","einnahmen","umsatz"],
    "stops":  ["stops","stopuri","pakete","zustellpakete","geplante_zustellpakette"],
    "fuel_l": ["fuel_l","litri","liter","l","menge","menge_ltr","menge_ltr.","tankmenge","betankte_menge"],
}

def norm(s: str) -> str:
    if s is None: return ""
    s = normalize_text(str(s))
    s = s.replace("ä","a").replace("ö","o").replace("ü","u").replace("ß","ss")
    s = re.sub(r"[^a-z0-9_ ]+","",s)
    s = s.replace(" ", "_")
    return s

def map_headers(df: pd.DataFrame) -> pd.DataFrame:
    cols = [norm(c) for c in df.columns]
    df = df.copy()
    df.columns = cols
    rename = {}
    for canonical, alts in HEADER_ALIASES.items():
        for c in cols:
            if c in alts:
                rename[c] = canonical
        if canonical == "fuel_l" and "menge" in cols and "fuel_l" not in rename.values():
            rename["menge"] = "fuel_l"
    df = df.rename(columns=rename)
    return df

# ================== Export live ==================
def write_master_excel():
    df = load_entries_df()
    xl = io.BytesIO()
    if df.empty:
        empty = pd.DataFrame(columns=["date","driver","route","vehicle","km","fuel_l","fuel_cost","hours","revenue","stops","notes"])
        with pd.ExcelWriter(xl, engine="xlsxwriter") as writer:
            empty.to_excel(writer, sheet_name="Entries", index=False)
        with open(MASTER_XLSX, "wb") as f:
            f.write(xl.getvalue())
        return
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"]).dt.date
    daily = df2.groupby("date", as_index=False).agg({
        "km":"sum","fuel_l":"sum","fuel_cost":"sum","hours":"sum","revenue":"sum","stops":"sum"
    })
    by_driver = df2.groupby("driver", as_index=False).agg({
        "km":"sum","fuel_l":"sum","fuel_cost":"sum","hours":"sum","revenue":"sum","stops":"sum"
    })
    by_driver["profit"] = by_driver["revenue"] - by_driver["fuel_cost"]
    by_route = df2.groupby("route", as_index=False).agg({
        "km":"sum","fuel_l":"sum","fuel_cost":"sum","hours":"sum","revenue":"sum","stops":"sum"
    })
    by_route["profit"] = by_route["revenue"] - by_route["fuel_cost"]
    with pd.ExcelWriter(xl, engine="xlsxwriter") as writer:
        df2.to_excel(writer, sheet_name="Entries", index=False)
        daily.to_excel(writer, sheet_name="Daily", index=False)
        by_driver.to_excel(writer, sheet_name="ByDriver", index=False)
        by_route.to_excel(writer, sheet_name="ByRoute", index=False)
    with open(MASTER_XLSX, "wb") as f:
        f.write(xl.getvalue())

# ================== MAIN ==================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Upload → Auto-detect (Predict & Motorină) → Auto-save → Statistici → Excel live")

    with st.sidebar:
        st.header("Perioadă & filtre")
        today = dt.date.today()
        sel_month = st.selectbox("Luna", list(range(1,13)), index=today.month-1,
                                 format_func=lambda m: dt.date(2000,m,1).strftime("%B"))
        sel_year = st.number_input("An", value=today.year, step=1)
        default_date = st.date_input("Data implicită (fișiere fără dată)", value=today)
        driver_filter = st.text_input("Filtru șofer")
        route_filter  = st.text_input("Filtru tură")
        vehicle_filter= st.text_input("Filtru mașină")
        st.info("Bonurile de motorină sunt deseori pentru ziua precedentă – le mapez automat la Predict.")

    st.subheader("1) Încarcă fișiere (jpg/png/pdf/xls/xlsx/csv)")
    uploads = st.file_uploader(
        "Alege fișiere",
        type=["png","jpg","jpeg","pdf","xls","xlsx","csv"],
        accept_multiple_files=True
    )

    auto_inserts = 0
    if uploads:
        for up in uploads:
            ext = Path(up.name).suffix.lower()
            raw_bytes = up.getvalue()
            # Excel/CSV
            if ext in [".xlsx",".xls",".csv"]:
                try:
                    if ext==".csv":
                        df_x = pd.read_csv(io.BytesIO(raw_bytes))
                    else:
                        df_x = pd.read_excel(io.BytesIO(raw_bytes))
                    df_x = map_headers(df_x)
                    for _,r in df_x.iterrows():
                        liters = float(str(r.get("fuel_l",0)).replace(",",".") or 0)
                        if liters==0: continue
                        date_iso = default_date.isoformat()
                        ctx = get_predict_context_for(date_iso) or {"driver":"AUTO","route":"AUTO","vehicle":"AUTO","stops":0}
                        row = {
                            "date": date_iso,
                            "driver": str(r.get("driver",ctx["driver"])),
                            "route": str(r.get("route",ctx["route"])),
                            "vehicle": str(r.get("vehicle",ctx["vehicle"])),
                            "km": float(r.get("km",0) or 0),
                            "fuel_l": liters,
                            "fuel_cost": liters*FUEL_PRICE,
                            "hours": float(r.get("hours",0) or 0),
                            "revenue": float(r.get("revenue",0) or 0),
                            "stops": int(r.get("stops",ctx["stops"]) or 0),
                            "notes": "Excel import"
                        }
                        insert_entry(row)
                        auto_inserts+=1
                except Exception as e:
                    st.error(f"Eroare Excel: {e}")

        write_master_excel()
        st.success(f"Procesare finalizată, inserări: {auto_inserts}")

    st.subheader("2) Statistici (zilnic & lunar)")
    df = load_entries_df()
    if df.empty:
        st.warning("Nu există date încă.")
    else:
        df["date"]=pd.to_datetime(df["date"]).dt.date
        mask = (pd.to_datetime(df["date"]).dt.month==sel_month)&(pd.to_datetime(df["date"]).dt.year==sel_year)
        fdf=df[mask].copy()
        daily=fdf.groupby("date",as_index=False).agg({"fuel_l":"sum","fuel_cost":"sum","stops":"sum"})
        by_route=fdf.groupby("route",as_index=False).agg({"fuel_l":"sum","fuel_cost":"sum","stops":"sum"})
        st.dataframe(daily,use_container_width=True)
        st.dataframe(by_route,use_container_width=True)

if __name__=="__main__":
    init_db()
    main()

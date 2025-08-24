import io
import re
import uuid
import sqlite3
import unicodedata
import datetime as dt
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
import streamlit as st

# GPT
from openai import OpenAI

# ================== Config & Paths ==================
APP_TITLE = "SANS – Fleet MVP (Motorină + Predict + ChatGPT)"
FUEL_PRICE = 1.6  # €/litru

DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
EXPORT_DIR = DATA_DIR / "exports"
DB_PATH = DATA_DIR / "fleet.db"

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

# ================== OCR & Parsers ==================
NUM = r"([0-9]+(?:[.,][0-9]+)?)"

def normalize_text(s: str) -> str:
    if not s: return ""
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
    except: return None

def try_ocr_pdf(pdf_bytes: bytes) -> Optional[str]:
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts).strip()
    except: return None

def parse_stops(text: str) -> Optional[int]:
    t = normalize_text(text)
    m = re.search(r"stop[s]?\s*[:\-]?\s*([0-9]+)", t)
    return int(m.group(1)) if m else None

def parse_packages(text: str) -> Optional[int]:
    t = normalize_text(text)
    m = re.search(r"geplante\s+zustellpakette\s*[:\-]?\s*([0-9]+)", t)
    return int(m.group(1)) if m else None

def parse_driver(text: str) -> Optional[str]:
    t = normalize_text(text)
    m = re.search(r"(fahrer|driver|sofer)\s*[:\-]?\s*([a-z0-9 ._-]{2,})", t)
    return m.group(2).title() if m else None

def parse_route(text: str) -> Optional[str]:
    t = normalize_text(text)
    m = re.search(r"(tour|tura|route)\s*[:\-]?\s*([a-z0-9 /._-]{2,})", t)
    return m.group(2).upper() if m else None

def parse_vehicle(text: str) -> Optional[str]:
    t = normalize_text(text)
    m = re.search(r"\b([a-z]{1,3}\s?[a-z]{1,3}\s?[0-9]{1,4})\b", t)
    return m.group(1).upper() if m else None

def parse_liters(text: str) -> Optional[float]:
    t = normalize_text(text)
    m = re.search(rf"\bmenge\s+{NUM}\b", t)
    if m: return float(m.group(1).replace(",", "."))
    return None

# ================== Header mapping (Tankpool Excel) ==================
HEADER_ALIASES = {
    "date":   ["date","datum","datum_tankung","belegdatum"],
    "vehicle":["vehicle","fahrzeug","kennzeichen"],
    "fuel_l": ["fuel_l","menge","menge_ltr","menge_ltr.","tankmenge","betankte_menge"],
}

def norm(s: str) -> str:
    s = normalize_text(str(s))
    s = re.sub(r"[^a-z0-9_ ]+","",s)
    return s.replace(" ","_")

def map_headers(df: pd.DataFrame) -> pd.DataFrame:
    cols = [norm(c) for c in df.columns]
    df = df.copy()
    df.columns = cols
    rename = {}
    for canonical, alts in HEADER_ALIASES.items():
        for c in cols:
            if c in alts:
                rename[c] = canonical
    return df.rename(columns=rename)

# ================== GPT Helper ==================
def ask_gpt(question: str, df: pd.DataFrame) -> str:
    if df.empty:
        return "Nu există date încă."
    # rezum datele în CSV text pt context
    context = df.to_csv(index=False)
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"Ești un analist de flotă. Răspunde clar, în română."},
            {"role":"user","content":f"Avem următoarele date:\n{context}\nÎntrebare: {question}"}
        ]
    )
    return completion.choices[0].message.content

# ================== MAIN ==================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    tab1, tab2, tab3 = st.tabs(["📂 Upload & Procesare", "📊 Statistici", "💬 ChatGPT"])

    with tab1:
        st.header("Upload fișiere Tankpool sau Predict")
        uploads = st.file_uploader("Alege fișiere", type=["png","jpg","jpeg","pdf","xls","xlsx","csv"], accept_multiple_files=True)
        default_date = dt.date.today()

        if uploads:
            for up in uploads:
                ext = Path(up.name).suffix.lower()
                raw_bytes = up.getvalue()

                if ext in [".xlsx",".xls",".csv"]:
                    df_x = pd.read_excel(io.BytesIO(raw_bytes)) if ext!=".csv" else pd.read_csv(io.BytesIO(raw_bytes))
                    df_x = map_headers(df_x)
                    for _,r in df_x.iterrows():
                        liters = float(str(r.get("fuel_l",0)).replace(",",".")) if pd.notnull(r.get("fuel_l")) else 0
                        if liters==0: continue
                        row = {
                            "date": pd.to_datetime(r.get("date",default_date)).date().isoformat(),
                            "driver":"AUTO","route":"AUTO",
                            "vehicle": str(r.get("vehicle","AUTO")),
                            "fuel_l": liters,
                            "fuel_cost": liters*FUEL_PRICE,
                            "stops":0,"packages":0,
                            "notes":"Tankpool Excel"
                        }
                        insert_entry(row)

                elif ext in [".png",".jpg",".jpeg",".pdf"]:
                    text = try_ocr_image(raw_bytes) if ext in [".png",".jpg",".jpeg"] else try_ocr_pdf(raw_bytes)
                    if text:
                        stops = parse_stops(text) or 0
                        packages = parse_packages(text) or 0
                        drv = parse_driver(text) or "AUTO"
                        rte = parse_route(text) or "AUTO"
                        veh = parse_vehicle(text) or "AUTO"
                        row = {
                            "date": default_date.isoformat(),
                            "driver": drv, "route": rte, "vehicle": veh,
                            "fuel_l":0,"fuel_cost":0,
                            "stops":stops,"packages":packages,
                            "notes":"Predict OCR"
                        }
                        insert_entry(row)
            st.success("Procesare finalizată ✅")

    with tab2:
        st.header("Statistici zilnice și pe tură")
        df = load_entries_df()
        if df.empty:
            st.warning("Nu există date încă.")
        else:
            daily = df.groupby("date",as_index=False).agg({"fuel_l":"sum","fuel_cost":"sum","stops":"sum","packages":"sum"})
            by_route = df.groupby("route",as_index=False).agg({"fuel_l":"sum","fuel_cost":"sum","stops":"sum","packages":"sum"})
            st.subheader("▶ Zilnic")
            st.dataframe(daily,use_container_width=True)
            st.subheader("▶ Pe tură")
            st.dataframe(by_route,use_container_width=True)

    with tab3:
        st.header("Întreabă ChatGPT despre date")
        df = load_entries_df()
        q = st.text_input("Întrebarea ta")
        if q and not df.empty:
            answer = ask_gpt(q, df)
            st.markdown(f"**Răspuns:** {answer}")

if __name__=="__main__":
    init_db()
    main()

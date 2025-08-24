import os
import io
import re
import uuid
import sqlite3
import datetime as dt
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

# ---- Config & Paths ----
APP_TITLE = "SANS – Fleet MVP (Streamlit)"
DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "fleet.db"

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
        notes TEXT
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_id INTEGER,
        filename TEXT NOT NULL,
        path TEXT NOT NULL,
        ocr_text TEXT,
        uploaded_at TEXT NOT NULL,
        FOREIGN KEY(entry_id) REFERENCES entries(id) ON DELETE SET NULL
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

# ---- Lightweight parsers ----
NUM = r"([0-9]+(?:[.,][0-9]+)?)"

def parse_km(text: str) -> Optional[float]:
    if not text: return None
    for pat in [rf"\b{NUM}\s*km\b", rf"km\s*{NUM}\b"]:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1).replace(",", "."))
    return None

def parse_liters(text: str) -> Optional[float]:
    if not text: return None
    for pat in [rf"\b{NUM}\s*(l|litri|liter)\b"]:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1).replace(",", "."))
    return None

def parse_euros(text: str) -> Optional[float]:
    if not text: return None
    for pat in [rf"\b{NUM}\s*(€|eur|euro)\b"]:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1).replace(",", "."))
    return None

# ---- UI ----
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Upload → OCR (opțional) → Confirm → Statistici zilnice/lunare → Export")

with st.sidebar:
    today = dt.date.today()
    month = st.selectbox("Luna", options=list(range(1,13)), index=today.month-1, format_func=lambda m: dt.date(2000, m, 1).strftime("%B"))
    year = st.number_input("An", value=today.year, step=1)
    driver_filter = st.text_input("Filtru șofer")
    route_filter = st.text_input("Filtru tură")
    vehicle_filter = st.text_input("Filtru mașină")

st.subheader("1) Upload documente")
uploads = st.file_uploader("Încarcă fișiere (jpg/png/pdf)", type=["png","jpg","jpeg","pdf"], accept_multiple_files=True)

ocr_text = ""
doc_meta = []
if uploads:
    month_dir = UPLOAD_DIR / f"{year}-{str(month).zfill(2)}"
    for up in uploads:
        content = up.getbuffer()
        ext = Path(up.name).suffix
        unique = f"{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
        out_path = month_dir / unique
        month_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(content)
        text = None
        if up.type.startswith("image/"):
            text = try_ocr_image(bytes(content))
        elif up.type == "application/pdf":
            text = try_ocr_pdf(bytes(content))
        doc_meta.append

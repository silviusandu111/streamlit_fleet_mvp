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
FUEL_PRICE = 1.6  # €/litru cu TVA

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

def parse_driver(text: str) -> Optional[str]:
    if not text:
        return None
    t = normalize_text(text)
    # caută cuvinte precum "Fahrer: Ion" sau "Driver - Gheorghe"
    pattern = r"(fahrer|driver|sofer)\s*[:\-]?\s*([a-z0-9 ._-]{2,})"
    m = re.search(pattern, t)
    if m:
        return m.group(2).strip().title()[:50]
    return None
    t = normalize_text(text)
    m = re.search(r"(fahrer|driver|sofer)\s*[:\-]?\s*([a-z0-

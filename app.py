import os
import io
import re
import uuid
import sqlite3
import unicodedata
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

FUEL_PRICE = 1.6  # €/litru cu TVA inclus

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

# ---- Utilitare parsare text ----
NUM = r"([0-9]+(?:[.,][0-9]+)?)"

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s

def parse_km(text: str) -> Optional[float]:
    if not text: return None
    for pat in [rf"\b{NUM}\s*km\b", rf"km\s*{NUM}\b"]:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try: return float(m.group(1).replace(",", "."))
            except: return None
    return None

def parse_liters(text: str) -> Optional[float]:
    """Caută cantitatea de combustibil în litri: 'menge <numar>' sau '<numar> l/liter/litri'."""
    if not text: return None
    t = normalize_text(text)
    m = re.search(rf"\bmenge\s+{NUM}\b", t, flags=re.IGNORECASE)
    if m:
        try: return float(m.group(1).replace(",", "."))
        except: pass
    m = re.search(rf"\b{NUM}\s*(l|liter|litri)\b", t, flags=re.IGNORECASE)
    if m:
        try: return float(m.group(1).replace(",", "."))
        except: pass
    return None

def parse_geplante_zustellpakette(text: str) -> Optional[int]:
    """Extrage numărul de pachete din 'Geplante Zustellpakette: N' (permite variații minore)."""
    if not text: return None
    t = normalize_text(text)
    pat = r"geplante\s+zustellpaket(?:e|te)?\s*[:\-]?\s*([0-9]+)"
    m = re.search(pat, t, flags=re.IGNORECASE)
    if m:
        try: return int(m.group(1))
        except: return None
    return None

# ---- DB ops ----
def load_dataframe(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM entries ORDER BY date DESC, id DESC", conn, parse_dates=["date"])

def insert_entry_row(row: dict) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO entries (date, driver, route, vehicle, km, fuel_l, fuel_cost, hours, revenue, stops, notes)
        VALUES (:date, :driver, :route, :vehicle, :km, :fuel_l, :fuel_cost, :hours, :revenue, :stops, :notes)
    """, row)
    eid = cur.lastrowid
    conn.commit()
    conn.close()
    return eid

# ---- Streamlit App ----
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Upload → Autodetect (Motorină & Predict) → Confirm → Statistici → Export")

with st.sidebar:
    st.header("Perioadă & filtre")
    today = dt.date.today()
    month = st.selectbox("Luna", options=list(range(1,13)), index=today.month-1,
                         format_func=lambda m: dt.date(2000, m, 1).strftime("%B"))
    year = st.number_input("An", value=today.year, step=1)
    default_date = st.date_input("Data implicită pentru fișiere", value=today,
                                 help="Se folosește când fișierul nu are o dată clară.")
    driver_filter = st.text_input("Filtru șofer (opțional)")
    route_filter = st.text_input("Filtru tură (opțional)")
    vehicle_filter = st.text_input("Filtru mașină (opțional)")
    st.markdown("---")
    st.info("💡 Urcă poze/PDF/Excel/CSV. Detectez automat **Menge** (motorină) și **Geplante Zustellpakette** (Predict).")

# ----------------- 1) Upload & Autodetect -----------------
st.subheader("1) Încarcă fișiere (jpg/png/pdf/xls/xlsx/csv)")

uploads = st.file_uploader(
    "Urcă documente",
    type=["png","jpg","jpeg","pdf","xls","xlsx","csv"],
    accept_multiple_files=True
)

detected_rows: List[dict] = []
excel_frames: List[pd.DataFrame] = []
uploaded_summary = []
ocr_debug_dump = []

def mk_base_row(date_val: dt.date) -> dict:
    return {
        "date": str(date_val),
        "driver": "",
        "route": "",
        "vehicle": "",
        "km": 0.0,
        "fuel_l": 0.0,
        "fuel_cost": 0.0,
        "hours": 0.0,
        "revenue": 0.0,
        "stops": 0,
        "notes": ""
    }

if uploads:
    month_dir = UPLOAD_DIR / f"{year}-{str(month).zfill(2)}"
    month_dir.mkdir(parents=True, exist_ok=True)

    for up in uploads:
        content = up.getbuffer()
        ext = Path(up.name).suffix.lower()

        # 1) Salvăm fizic
        unique = f"{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
        out_path = month_dir / unique
        with open(out_path, "wb") as f:
            f.write(content)

        text = None
        liters = None
        stops_val = None
        excel_rows = 0

        # 2) Autodetect & OCR
        if ext in [".png", ".jpg", ".jpeg"]:
            text = try_ocr_image(bytes(content))
        elif ext == ".pdf":
            text = try_ocr_pdf(bytes(content))
        elif ext in [".xlsx", ".xls"]:
            try:
                df_x = pd.read_excel(io.BytesIO(bytes(content)))
                excel_frames.append(df_x)
                excel_rows = len(df_x)
            except Exception as e:
                st.error(f"Nu pot citi {up.name} ca Excel: {e}")
        elif ext == ".csv":
            try:
                df_x = pd.read_csv(io.BytesIO(bytes(content)))
                excel_frames.append(df_x)
                excel_rows = len(df_x)
            except Exception as e:
                st.error(f"Nu pot citi {up.name} ca CSV: {e}")

        if text:
            ocr_debug_dump.append((up.name, text))
            liters = parse_liters(text)                      # Motorină
            stops_val = parse_geplante_zustellpakette(text) # Predict

            note_src = f"auto from {up.name}"
            if liters is not None and liters > 0:
                r = mk_base_row(default_date)
                r["fuel_l"] = float(liters)
                r["fuel_cost"] = float(liters) * FUEL_PRICE
                r["notes"] = f"Motorină (Menge) – {note_src}"
                detected_rows.append(r)

            if stops_val is not None and stops_val >= 0:
                r = mk_base_row(default_date)
                r["stops"] = int(stops_val)
                r["notes"] = f"Predict (Geplante Zustellpakette) – {note_src}"
                detected_rows.append(r)

        uploaded_summary.append({
            "fisier": up.name,
            "salvat_la": str(out_path.relative_to(DATA_DIR)),
            "tip": ext.replace(".", "").upper(),
            "litri_detectati": liters if liters is not None else "",
            "zustellpakette": stops_val if stops_val is not None else "",
            "randuri_excel": excel_rows if excel_rows else ""
        })

# 3) Afișare fișiere încărcate + OCR debug
if uploaded_summary:
    st.markdown("#### Fișiere încărcate")
    st.dataframe(pd.DataFrame(uploaded_summary), use_container_width=True)

    with st.expander("🔍 Text OCR brut (debug)"):
        if ocr_debug_dump:
            for name, txt in ocr_debug_dump:
                st.markdown(f"**{name}**")
                st.code(txt[:4000] if txt else "(fără text)", language="text")
        else:
            st.caption("Nu s-a extras text (doar Excel/CSV sau OCR indisponibil).")
else:
    st.caption("Urcă un fișier pentru a-l vedea aici (se salvează în `data/uploads/<an-lună>/`).")

# ----------------- 2) Import din Excel/CSV (ture complete) -----------------
st.subheader("2) Import din Excel/CSV (ture complete)")

if excel_frames:
    excel_df = pd.concat(excel_frames, ignore_index=True)
    st.caption("Previzualizare din Excel/CSV")
    st.dataframe(excel_df.head(20), use_container_width=True)

    required_cols = ['date','driver','route','vehicle','km','fuel_l','hours','revenue','stops','notes']
    missing = [c for c in required_cols if c not in excel_df.columns]
    if missing:
        st.warning(f"Lipsesc coloane: {missing}. Antete așteptate: {required_cols}")
    else:
        if st.button("📥 Importă rândurile din Excel/CSV în baza de date"):
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
                fuel_l = to_float(row['fuel_l'])
                fuel_cost = fuel_l * FUEL_PRICE
                cur.execute("""
                    INSERT INTO entries (date, driver, route, vehicle, km, fuel_l, fuel_cost, hours, revenue, stops, notes)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    to_iso(row['date']),
                    str(row['driver']).strip() or "AUTO",
                    str(row['route']).strip() or "AUTO",
                    str(row['vehicle']).strip() or "AUTO",
                    to_float(row['km']),
                    fuel_l,
                    fuel_cost,
                    to_float(row.get('hours', 0)),
                    to_float(row.get('revenue', 0)),
                    int(row.get('stops', 0) or 0),
                    str(row.get('notes', '')).strip()
                ))
                imported += 1
            conn.commit()
            conn.close()
            st.success(f"Import finalizat: {imported} rânduri.")
else:
    st.caption("Încarcă un Excel/CSV pentru import în bloc al turelor.")

# ----------------- 3) Detecții din poze/PDF – confirmă & salvează -----------------
st.subheader("3) Detecții din poze/PDF (auto) – completează câmpurile și salvează")

if detected_rows:
    det_df = pd.DataFrame(detected_rows)
    st.caption("✅ Completează/ajustează **driver / tură / mașină** (obligatoriu) înainte de salvare.")
    edited = st.data_editor(
        det_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={"date": st.column_config.DateColumn(format="YYYY-MM-DD")}
    )

    colA, colB = st.columns(2)
    with colA:
        if st.button("🔁 Recalculează cost motorină = L × 1.6 €"):
            if "fuel_l" in edited.columns:
                edited["fuel_cost"] = edited["fuel_l"].astype(float).fillna(0.0) * FUEL_PRICE
                st.experimental_rerun()
    with colB:
        if st.button("💾 Save detections în baza de date"):
            saved = 0
            for _, row in edited.iterrows():
                driver = (str(row.get("driver") or "")).strip() or "AUTO"
                route = (str(row.get("route") or "")).strip() or "AUTO"
                vehicle = (str(row.get("vehicle") or "")).strip() or "AUTO"
                try:
                    payload = {
                        "date": pd.to_datetime(row["date"]).date().isoformat() if pd.notnull(row["date"]) else dt.date.today().isoformat(),
                        "driver": driver,
                        "route": route,
                        "vehicle": vehicle,
                        "km": float(row.get("km", 0) or 0),
                        "fuel_l": float(row.get("fuel_l", 0) or 0),
                        "fuel_cost": float(row.get("fuel_cost", 0) or 0),
                        "hours": float(row.get("hours", 0) or 0),
                        "revenue": float(row.get("revenue", 0) or 0),
                        "stops": int(row.get("stops", 0) or 0),
                        "notes": str(row.get("notes", "") or "").strip()
                    }
                    insert_entry_row(payload)
                    saved += 1
                except Exception as e:
                    st.error(f"Eroare la salvare: {e}")
            if saved:
                st.success(f"Am salvat {saved} rânduri.")
                st.experimental_rerun()
else:
    st.caption("Urcă poze/PDF pentru a detecta **Menge** (motorină) și **Geplante Zustellpakette** (Predict).")

# ----------------- 4) Statistici -----------------
st.subheader("4) Statistici (zilnic & lunar)")

conn = get_conn()
df = load_dataframe(conn)
conn.close()

if df.empty:
    st.warning("Nu există date încă.")
else:
    df["date"] = pd.to_datetime(df["date"]).dt.date
    mask = (pd.to_datetime(df["date"]).dt.month == month) & (pd.to_datetime(df["date"]).dt.year == year)
    if driver_filter:
        mask &= df["driver"].str.contains(driver_filter, case=False, na=False)
    if route_filter:
        mask &= df["route"].str.contains(route_filter, case=False, na=False)
    if vehicle_filter:
        mask &= df["vehicle"].str.contains(vehicle_filter, case=False, na=False)
    fdf = df[mask].copy()

    st.markdown("### Zilnic (în luna selectată)")
    daily = fdf.groupby("date", as_index=False).agg({
        "km":"sum","fuel_l":"sum","fuel_cost":"sum","hours":"sum","revenue":"sum","stops":"sum"
    })
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("KM/zi (medie)", f"{daily['km'].mean():.1f}" if not daily.empty else "0.0")
    with c2: st.metric("Cost motorină/zi (medie)", f"{daily['fuel_cost'].mean():.2f}€" if not daily.empty else "0.00€")
    with c3:
        profit_series = (daily["revenue"] - daily["fuel_cost"]) if not daily.empty else pd.Series([0])
        st.metric("Profit/zi (medie)", f"{profit_series.mean():.2f}€")
    with c4: st.metric("Stopuri/zi (medie)", f"{daily['stops'].mean():.0f}" if not daily.empty else "0")

    st.dataframe(daily, use_container_width=True)

    st.markdown("### Per șofer (lunar)")
    by_driver = fdf.groupby("driver", as_index=False).agg({
        "km":"sum","fuel_l":"sum","fuel_cost":"sum","hours":"sum","revenue":"sum","stops":"sum"
    })
    by_driver["profit"] = by_driver["revenue"] - by_driver["fuel_cost"]
    st.dataframe(by_driver.sort_values(["stops","profit"], ascending=[False, False]), use_container_width=True)

    st.markdown("### Per tură (lunar)")
    by_route = fdf.groupby("route", as_index=False).agg({
        "km":"sum","fuel_l":"sum","fuel_cost":"sum","hours":"sum","revenue":"sum","stops":"sum"
    })
    by_route["profit"] = by_route["revenue"] - by_route["fuel_cost"]
    st.dataframe(by_route.sort_values(["stops","profit"], ascending=[False, False]), use_container_width=True)

# ----------------- 5) Export -----------------
st.subheader("5) Export")
if 'fdf' in locals():
    filt = fdf.copy()
    csv_bytes = filt.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descarcă CSV (luna curentă)", data=csv_bytes,
                       file_name=f"export_{year}-{str(month).zfill(2)}.csv", mime="text/csv")

    xl = io.BytesIO()
    daily_x = filt.groupby("date", as_index=False).agg({"km":"sum","fuel_l":"sum","fuel_cost":"sum","hours":"sum","revenue":"sum","stops":"sum"})
    by_driver_x = filt.groupby("driver", as_index=False).agg({"km":"sum","fuel_l":"sum","fuel_cost":"sum","hours":"sum","revenue":"sum","stops":"sum"})
    by_driver_x["profit"] = by_driver_x["revenue"] - by_driver_x["fuel_cost"]
    by_route_x = filt.groupby("route", as_index=False).agg({"km":"sum","fuel_l":"sum","fuel_cost":"sum","hours":"sum","revenue":"sum","stops":"sum"})
    by_route_x["profit"] = by_route_x["revenue"] - by_route_x["fuel_cost"]

    with pd.ExcelWriter(xl, engine="xlsxwriter") as writer:
        filt.to_excel(writer, sheet_name="Filtru", index=False)
        daily_x.to_excel(writer, sheet_name="Zilnic", index=False)
        by_driver_x.to_excel(writer, sheet_name="Sofer", index=False)
        by_route_x.to_excel(writer, sheet_name="Tura", index=False)

    st.download_button("⬇️ Descarcă Excel (luna curentă)", data=xl.getvalue(),
                       file_name=f"raport_{year}-{str(month).zfill(2)}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
with st.expander("🔧 Ajutor & format fișiere"):
    st.write("""
    - **Motorină (poze/PDF)**: detectez *Menge* sau cantități în litri și calculez costul la 1,6 €/L.
    - **Predict (poze/PDF)**: detectez *Geplante Zustellpakette* și extrag numărul exact de pachete/zi.
    - **Excel/CSV** (ture complete): antete recomandate:
      `date, driver, route, vehicle, km, fuel_l, hours, revenue, stops, notes`
      (costul motorinei se calculează automat din `fuel_l`).
    - După detecții, completează **driver/tură/mașină** și apasă **Save detections**.
    """)

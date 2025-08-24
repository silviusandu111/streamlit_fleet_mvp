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

# ================== Export (live) ==================
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

    # CSS: păstrează butonul „Alege fișier”, ascunde chenarul & textul de drag&drop
    st.markdown("""
    <style>
    [data-testid="stFileUploaderDropzone"] { border:none !important; background:transparent !important; padding:0 !important; }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderInstructions"], 
    [data-testid="stFileUploader"] .uploadFile { display:none !important; }
    [data-testid="stFileUploader"] div[role="button"] { margin: 0.25rem 0 0 0; }
    </style>
    """, unsafe_allow_html=True)

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
        st.info("Bonurile de motorină sunt deseori pentru **ziua precedentă** – le mapez automat la Predict.")

    st.subheader("1) Încarcă un fișier (jpg/png/pdf/xls/xlsx/csv)")
    up = st.file_uploader(
        label="Alege fișier",
        type=["png","jpg","jpeg","pdf","xls","xlsx","csv"],
        accept_multiple_files=False,
        label_visibility="visible",
        key="single_upload"
    )

    processed_summary: List[Dict] = []
    ocr_debug: List[Tuple[str,str]] = []
    auto_inserts = 0

    # dacă nu s-a ales nimic, nu facem nimic
    if up is None:
        st.caption("Alege un fișier și va fi procesat automat.")
    else:
        # citim conținutul o singură dată și îl refolosim peste tot
        file_name = up.name
        ext = Path(file_name).suffix.lower()
        raw_bytes = up.getvalue()
        file_size = len(raw_bytes)

        st.markdown("**Fișier primit:**")
        st.table(pd.DataFrame([{"fisier": file_name, "extensie": ext, "dimensiune_B": file_size}]))

        # salvăm fizic copia fișierului
        month_dir = UPLOAD_DIR / f"{sel_year}-{str(sel_month).zfill(2)}"
        month_dir.mkdir(parents=True, exist_ok=True)
        unique = f"{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
        saved_path = month_dir / unique
        with open(saved_path, "wb") as f:
            f.write(raw_bytes)

        status_row = {"fisier": file_name, "tip": "", "rows_saved": 0, "mesaj": ""}
        detected_type = ""
        added_rows = 0
        text = None

        # OCR pentru imagine/PDF
        try:
            if ext in [".png",".jpg",".jpeg"]:
                text = try_ocr_image(raw_bytes)
            elif ext == ".pdf":
                text = try_ocr_pdf(raw_bytes)
        except Exception as e:
            status_row.update({"tip":"ERROR","mesaj":f"OCR error: {e}"})

        if text:
            ocr_debug.append((file_name, text[:4000]))
            stops = parse_stops_geplante(text)
            drv = parse_driver(text)
            rte = parse_route(text)
            veh = parse_vehicle(text)
            date_iso = default_date.isoformat()

            if any([stops is not None, drv, rte, veh]):
                upsert_predict_context(date_iso, drv, rte, veh, stops, text)
                detected_type = "PREDICT"
                status_row.update({"tip":"PREDICT","mesaj":f"ctx({drv or 'AUTO'}/{rte or 'AUTO'}/{veh or 'AUTO'}), stops={stops or ''}"})

        # Excel/CSV
        if ext in [".xlsx",".xls",".csv"]:
            try:
                if ext == ".csv":
                    df_x = pd.read_csv(io.BytesIO(raw_bytes))
                else:
                    df_x = pd.read_excel(io.BytesIO(raw_bytes))  # openpyxl
            except Exception as e:
                status_row.update({"tip":"ERROR","mesaj":f"Eroare citire Excel/CSV: {e}"})
                processed_summary.append(status_row)
                df_x = None

            if df_x is not None:
                cols = [c.strip().lower() for c in df_x.columns]
                df_x.columns = cols

                is_fuel = ("menge" in cols) or ("fuel_l" in cols)
                is_entries = any(c in cols for c in ["driver","route","vehicle","km","stops"])

                if is_fuel:
                    detected_type = "FUEL_EXCEL"
                    dcol = "date" if "date" in cols else None
                    for _, r in df_x.iterrows():
                        liters = None
                        if "menge" in r and pd.notnull(r["menge"]):
                            try: liters = float(str(r["menge"]).replace(",",".")); 
                            except: liters = None
                        elif "fuel_l" in r and pd.notnull(r["fuel_l"]):
                            try: liters = float(str(r["fuel_l"]).replace(",",".")); 
                            except: liters = None
                        if liters is None:
                            continue

                        if dcol:
                            try:
                                date_iso = pd.to_datetime(r[dcol]).date().isoformat()
                            except Exception:
                                date_iso = (default_date - dt.timedelta(days=1)).isoformat()
                        else:
                            date_iso = (default_date - dt.timedelta(days=1)).isoformat()

                        ctx = get_predict_context_for(date_iso) or {"driver":"AUTO","route":"AUTO","vehicle":"AUTO","stops":0}
                        row = {
                            "date": date_iso,
                            "driver": ctx["driver"], "route": ctx["route"], "vehicle": ctx["vehicle"],
                            "km": float(r.get("km", 0) or 0),
                            "fuel_l": float(liters),
                            "fuel_cost": float(liters) * FUEL_PRICE,
                            "hours": float(r.get("hours", 0) or 0),
                            "revenue": float(r.get("revenue", 0) or 0),
                            "stops": int(r.get("stops", 0) or ctx.get("stops",0) or 0),
                            "notes": f"Fuel Excel {file_name}"
                        }
                        insert_entry(row)
                        auto_inserts += 1
                        added_rows += 1
                    status_row.update({"tip":"FUEL_EXCEL","rows_saved":added_rows})

                elif is_entries:
                    detected_type = "RUNS_EXCEL"
                    def to_float(x):
                        try: return float(str(x).replace(",",".")) if pd.notnull(x) else 0.0
                        except: return 0.0
                    for _, r in df_x.iterrows():
                        try:
                            date_iso = pd.to_datetime(r.get("date","")).date().isoformat()
                        except Exception:
                            date_iso = default_date.isoformat()
                        fuel_l = to_float(r.get("fuel_l", 0))
                        row = {
                            "date": date_iso,
                            "driver": str(r.get("driver","AUTO")).strip() or "AUTO",
                            "route": str(r.get("route","AUTO")).strip() or "AUTO",
                            "vehicle": str(r.get("vehicle","AUTO")).strip() or "AUTO",
                            "km": to_float(r.get("km", 0)),
                            "fuel_l": fuel_l,
                            "fuel_cost": fuel_l * FUEL_PRICE,
                            "hours": to_float(r.get("hours", 0)),
                            "revenue": to_float(r.get("revenue", 0)),
                            "stops": int(r.get("stops", 0) or 0),
                            "notes": str(r.get("notes","")).strip()
                        }
                        insert_entry(row)
                        auto_inserts += 1
                        added_rows += 1
                    status_row.update({"tip":"RUNS_EXCEL","rows_saved":added_rows})
                else:
                    if not detected_type:
                        detected_type = "UNKNOWN_EXCEL"
                        status_row.update({"tip":"UNKNOWN_EXCEL","mesaj":"Nu am recunoscut formatul (lipsesc coloane așteptate)."})

        # Imagine/PDF – bon motorină (Menge) dacă nu a fost alt tip
        if text and detected_type == "":
            liters = parse_liters(text)
            if liters is not None and liters > 0:
                date_iso = (default_date - dt.timedelta(days=1)).isoformat()
                ctx = get_predict_context_for(date_iso) or {"driver":"AUTO","route":"AUTO","vehicle":"AUTO","stops":0}
                row = {
                    "date": date_iso,
                    "driver": ctx["driver"], "route": ctx["route"], "vehicle": ctx["vehicle"],
                    "km": 0.0, "fuel_l": float(liters), "fuel_cost": float(liters) * FUEL_PRICE,
                    "hours": 0.0, "revenue": 0.0, "stops": int(ctx["stops"] or 0),
                    "notes": f"Fuel OCR {file_name}"
                }
                insert_entry(row)
                auto_inserts += 1
                detected_type = "FUEL_OCR"
                status_row.update({"tip":"FUEL_OCR","rows_saved":1,
                                   "mesaj":f"{liters} L -> {row['fuel_cost']:.2f} €"})

        if not detected_type:
            status_row.update({"tip":"NO_DETECTION","mesaj":"Nu am găsit nici Predict, nici Menge."})

        processed_summary.append(status_row)

        # Excel live
        try:
            write_master_excel()
        except Exception as e:
            st.warning(f"Nu am putut actualiza master.xlsx: {e}")

        st.success(f"Procesare finalizată. Inserări automate: {auto_inserts}.")
        st.markdown("### Rezumat procesare")
        st.dataframe(pd.DataFrame(processed_summary), use_container_width=True)

        if ocr_debug:
            with st.expander("🔍 OCR debug (primele 4000 caractere/fișier)"):
                for name, txt in ocr_debug:
                    st.markdown(f"**{name}**")
                    st.code(txt or "(fără text)", language="text")

    # Statistici + Export
    st.subheader("2) Statistici (zilnic & lunar) + Export live")
    df = load_entries_df()
    if df.empty:
        st.warning("Nu există date încă.")
    else:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        mask = (pd.to_datetime(df["date"]).dt.month == sel_month) & (pd.to_datetime(df["date"]).dt.year == sel_year)
        if driver_filter:  mask &= df["driver"].str.contains(driver_filter, case=False, na=False)
        if route_filter:   mask &= df["route"].str.contains(route_filter, case=False, na=False)
        if vehicle_filter: mask &= df["vehicle"].str.contains(vehicle_filter, case=False, na=False)
        fdf = df[mask].copy()

        daily = fdf.groupby("date", as_index=False).agg({
            "km":"sum","fuel_l":"sum","fuel_cost":"sum","hours":"sum","revenue":"sum","stops":"sum"
        })
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("KM/zi (medie)", f"{daily['km'].mean():.1f}" if not daily.empty else "0.0")
        with c2: st.metric("Cost motorină/zi (medie)", f"{daily['fuel_cost'].mean():.2f}€" if not daily.empty else "0.00€")
        with c3:
            prof = (daily["revenue"] - daily["fuel_cost"]).mean() if not daily.empty else 0.0
            st.metric("Profit/zi (medie)", f"{prof:.2f}€")
        with c4: st.metric("Stopuri/zi (medie)", f"{daily['stops'].mean():.0f}" if not daily.empty else "0")

        st.markdown("### Zilnic")
        st.dataframe(daily, use_container_width=True)

        st.markdown("### Per șofer")
        by_driver = fdf.groupby("driver", as_index=False).agg({
            "km":"sum","fuel_l":"sum","fuel_cost":"sum","hours":"sum","revenue":"sum","stops":"sum"
        })
        by_driver["profit"] = by_driver["revenue"] - by_driver["fuel_cost"]
        st.dataframe(by_driver.sort_values(["stops","profit"], ascending=[False, False]), use_container_width=True)

        st.markdown("### Per tură")
        by_route = fdf.groupby("route", as_index=False).agg({
            "km":"sum","fuel_l":"sum","fuel_cost":"sum","hours":"sum","revenue":"sum","stops":"sum"
        })
        by_route["profit"] = by_route["revenue"] - by_route["fuel_cost"]
        st.dataframe(by_route.sort_values(["stops","profit"], ascending=[False, False]), use_container_width=True)

        if MASTER_XLSX.exists():
            with open(MASTER_XLSX, "rb") as f:
                st.download_button("⬇️ Descarcă Excel live (master.xlsx)", f, file_name="master.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.caption("Excel live nu a fost generat încă.")

# ---------- Run ----------
if __name__ == "__main__":
    init_db()
    main()

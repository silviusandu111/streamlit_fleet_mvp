import io
import re
import uuid
import sqlite3
import unicodedata
import datetime as dt
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import pandas as pd
import streamlit as st

# ================== Config ==================
APP_TITLE = "SANS – Motorină (Tankpool) + Predict"
FUEL_PRICE = 1.6  # €/L cu TVA

DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
EXPORT_DIR = DATA_DIR / "exports"
DB_PATH = DATA_DIR / "fleet.db"
MASTER_XLSX = EXPORT_DIR / "master.xlsx"

for p in [DATA_DIR, UPLOAD_DIR, EXPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ================== DB ==================
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    # fiecare rând reprezintă o înregistrare (motorină sau predict) pe o zi
    cur.execute("""
    CREATE TABLE IF NOT EXISTS entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,              -- data de livrare (Predict) sau data mapată (Tankpool +1)
        driver TEXT DEFAULT 'AUTO',
        route TEXT DEFAULT 'AUTO',
        vehicle TEXT DEFAULT 'AUTO',
        fuel_l REAL DEFAULT 0,           -- litri
        fuel_cost REAL DEFAULT 0,        -- litri * 1.6
        stops INTEGER DEFAULT 0,         -- din Predict
        packages INTEGER DEFAULT 0,      -- din Predict (Geplante Zustellpakette)
        notes TEXT
    );
    """)
    # context Predict, pe zi (pentru maparea motorinei pe ture/șofer)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS predict_context (
        date TEXT PRIMARY KEY,
        driver TEXT,
        route TEXT,
        vehicle TEXT,
        stops INTEGER DEFAULT 0,
        packages INTEGER DEFAULT 0,
        raw_text TEXT
    );
    """)
    conn.commit()
    conn.close()

def insert_entry(row: dict):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO entries (date, driver, route, vehicle, fuel_l, fuel_cost, stops, packages, notes)
        VALUES (:date, :driver, :route, :vehicle, :fuel_l, :fuel_cost, :stops, :packages, :notes)
    """, row)
    conn.commit()
    conn.close()

def upsert_predict_context(date_iso: str, driver: Optional[str], route: Optional[str],
                           vehicle: Optional[str], stops: Optional[int], packages: Optional[int], raw_text: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO predict_context(date, driver, route, vehicle, stops, packages, raw_text)
        VALUES(?,?,?,?,?,?,?)
        ON CONFLICT(date) DO UPDATE SET
          driver=COALESCE(excluded.driver, predict_context.driver),
          route=COALESCE(excluded.route, predict_context.route),
          vehicle=COALESCE(excluded.vehicle, predict_context.vehicle),
          stops=COALESCE(excluded.stops, predict_context.stops),
          packages=COALESCE(excluded.packages, predict_context.packages),
          raw_text=excluded.raw_text
    """, (date_iso, driver, route, vehicle, stops, packages, raw_text))
    conn.commit()
    conn.close()

def get_predict_context_for(date_iso: str) -> Optional[dict]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT date, driver, route, vehicle, stops, packages FROM predict_context WHERE date=?", (date_iso,))
    row = cur.fetchone()
    conn.close()
    if row:
        return {
            "date": row[0],
            "driver": row[1] or "AUTO",
            "route": row[2] or "AUTO",
            "vehicle": row[3] or "AUTO",
            "stops": row[4] or 0,
            "packages": row[5] or 0
        }
    return None

def load_entries_df() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM entries ORDER BY date DESC, id DESC", conn)
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

# OCR PDF (merge în cloud)
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

# OCR imagine (poate să NU meargă în cloud din lipsă de tesseract)
def try_ocr_image(image_bytes: bytes) -> Tuple[Optional[str], bool]:
    try:
        import pytesseract
        from PIL import Image
        # verifică existența binarului tesseract
        try:
            _ = pytesseract.get_tesseract_version()
        except Exception:
            return (None, False)  # OCR indisponibil
        img = Image.open(io.BytesIO(image_bytes))
        return (pytesseract.image_to_string(img), True)
    except Exception:
        return (None, False)

def parse_int(pattern: str, text: str) -> Optional[int]:
    m = re.search(pattern, normalize_text(text))
    return int(m.group(1)) if m else None

def parse_str(pattern: str, text: str) -> Optional[str]:
    m = re.search(pattern, normalize_text(text))
    if m:
        val = m.group(2).strip()
        return val.title() if len(val) <= 60 else val[:60].title()
    return None

def parse_packages(text: str) -> Optional[int]:
    # Geplante Zustellpakette / Zustellpakete / Pakete
    for pat in [
        r"geplante\s+zustellpaket(?:e|te)?\s*[:\-]?\s*([0-9]+)",
        r"zustellpaket(?:e|te)?\s*[:\-]?\s*([0-9]+)",
        r"paket(?:e)?\s*[:\-]?\s*([0-9]+)",
    ]:
        v = parse_int(pat, text)
        if v is not None: return v
    return None

def parse_stops(text: str) -> Optional[int]:
    # Stops / Stopuri / Stopps
    for pat in [
        r"stops?\s*[:\-]?\s*([0-9]+)",
        r"stopuri?\s*[:\-]?\s*([0-9]+)",
        r"stopps?\s*[:\-]?\s*([0-9]+)",
    ]:
        v = parse_int(pat, text)
        if v is not None: return v
    return None

def parse_driver(text: str) -> Optional[str]:
    return parse_str(r"(fahrer|driver|sofer)\s*[:\-]?\s*([a-z0-9 ._-]{2,})", text)

def parse_route(text: str) -> Optional[str]:
    val = parse_str(r"(tour|tura|route)\s*[:\-]?\s*([a-z0-9 /._-]{2,})", text)
    return val.upper() if val else None

def parse_vehicle(text: str) -> Optional[str]:
    m = re.search(r"\b([a-z]{1,3}\s?[a-z]{1,3}\s?[0-9]{1,4})\b", normalize_text(text))
    return m.group(1).upper() if m else None

def parse_liters_from_text(text: str) -> Optional[float]:
    m = re.search(rf"\bmenge\s+{NUM}\b", normalize_text(text))
    if m:
        try: return float(m.group(1).replace(",", "."))
        except: return None
    return None

# ================== Header mapping (Tankpool) ==================
HEADER_ALIASES = {
    "date":   ["date","datum","datum_tankung","belegdatum","tankdatum","datum_der_tankung"],
    "vehicle":["vehicle","fahrzeug","kennzeichen","kennz","kennz_zeichen","kennzeichen_nummer"],
    "fuel_l": ["fuel_l","menge","menge_ltr","menge_ltr.","menge_l","liter","l","tankmenge","betankte_menge"],
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
    # fallback explicit pt 'menge' -> fuel_l
    if "fuel_l" not in rename.values() and "menge" in cols:
        rename["menge"] = "fuel_l"
    return df.rename(columns=rename)

# ================== Export Excel ==================
def write_master_excel():
    df = load_entries_df()
    xl = io.BytesIO()
    if df.empty:
        with pd.ExcelWriter(xl, engine="xlsxwriter") as writer:
            pd.DataFrame(columns=[
                "date","driver","route","vehicle","fuel_l","fuel_cost","stops","packages","notes"
            ]).to_excel(writer, sheet_name="Entries", index=False)
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        with open(MASTER_XLSX, "wb") as f:
            f.write(xl.getvalue())
        return

    df2 = df.copy()
    daily = df2.groupby("date", as_index=False).agg({
        "fuel_l":"sum","fuel_cost":"sum","stops":"sum","packages":"sum"
    })
    by_route = df2.groupby("route", as_index=False).agg({
        "fuel_l":"sum","fuel_cost":"sum","stops":"sum","packages":"sum"
    })

    with pd.ExcelWriter(xl, engine="xlsxwriter") as writer:
        df2.to_excel(writer, sheet_name="Entries", index=False)
        daily.to_excel(writer, sheet_name="Daily", index=False)
        by_route.to_excel(writer, sheet_name="ByRoute", index=False)

    with open(MASTER_XLSX, "wb") as f:
        f.write(xl.getvalue())

# ================== MAIN UI ==================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    with st.sidebar:
        today = dt.date.today()
        sel_month = st.selectbox("Luna", list(range(1,13)), index=today.month-1,
                                 format_func=lambda m: dt.date(2000,m,1).strftime("%B"))
        sel_year = st.number_input("An", value=today.year, step=1)
        default_date = st.date_input("Data pentru Predict (poze/PDF)", value=today)
        st.info("• Tankpool (Excel): data livrării = (Datum Tankung) + 1 zi\n• Predict (poze/PDF) folosește data selectată mai sus.")

    st.subheader("1) Încarcă fișiere Tankpool (Excel) și Predict (PDF/poze)")
    uploads = st.file_uploader("Alege unul sau mai multe fișiere",
                               type=["xls","xlsx","csv","pdf","png","jpg","jpeg"],
                               accept_multiple_files=True)

    processed = []
    ocr_warnings = []

    if uploads:
        month_dir = UPLOAD_DIR / f"{sel_year}-{str(sel_month).zfill(2)}"
        month_dir.mkdir(parents=True, exist_ok=True)

        for up in uploads:
            name = up.name
            ext = Path(name).suffix.lower()
            raw = up.getvalue()
            out_path = month_dir / f"{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}{ext}"
            with open(out_path, "wb") as f:
                f.write(raw)

            status = {"fisier": name, "tip": "", "rows": 0, "mesaj": ""}

            try:
                # --------------- TANKPOOL EXCEL ---------------
                if ext in [".xls",".xlsx",".csv"]:
                    if ext == ".csv":
                        df_x = pd.read_csv(io.BytesIO(raw))
                    else:
                        df_x = pd.read_excel(io.BytesIO(raw))
                    df_x = map_headers(df_x)
                    cols = set(df_x.columns)
                    if "fuel_l" not in cols:
                        status.update({"tip":"UNKNOWN_EXCEL","mesaj":"Nu am găsit coloana litri (fuel_l/menge/liter)."})
                    else:
                        added = 0
                        for _, r in df_x.iterrows():
                            # litri
                            lit = r.get("fuel_l", 0)
                            try:
                                liters = float(str(lit).replace(",", ".")) if pd.notnull(lit) else 0.0
                            except:
                                liters = 0.0
                            if liters <= 0:
                                continue

                            # data alimentare -> livrare (+1 zi)
                            d_val = r.get("date", None)
                            try:
                                refuel_date = pd.to_datetime(d_val).date() if pd.notnull(d_val) else (default_date - dt.timedelta(days=1))
                            except:
                                refuel_date = default_date - dt.timedelta(days=1)
                            delivery_date = (refuel_date + dt.timedelta(days=1)).isoformat()

                            # vehicul din fișier, restul din predict_context dacă există
                            veh = str(r.get("vehicle","AUTO")).upper() if pd.notnull(r.get("vehicle", None)) else "AUTO"
                            ctx = get_predict_context_for(delivery_date) or {"driver":"AUTO","route":"AUTO","vehicle":veh,"stops":0,"packages":0}
                            if ctx["vehicle"] == "AUTO":
                                ctx["vehicle"] = veh

                            row = {
                                "date": delivery_date,
                                "driver": ctx["driver"],
                                "route": ctx["route"],
                                "vehicle": ctx["vehicle"],
                                "fuel_l": liters,
                                "fuel_cost": liters * FUEL_PRICE,
                                "stops": int(ctx["stops"] or 0),
                                "packages": int(ctx["packages"] or 0),
                                "notes": f"Tankpool {name}"
                            }
                            insert_entry(row)
                            added += 1

                        status.update({"tip":"TANKPOOL_EXCEL","rows":added})
                # --------------- PREDICT PDF ---------------
                elif ext == ".pdf":
                    text = try_ocr_pdf(raw)
                    if not text:
                        status.update({"tip":"PDF_EMPTY","mesaj":"Nu am putut extrage text din PDF."})
                    else:
                        drv = parse_driver(text) or "AUTO"
                        rte = parse_route(text) or "AUTO"
                        veh = parse_vehicle(text) or "AUTO"
                        stops = parse_stops(text) or 0
                        packages = parse_packages(text) or 0
                        date_iso = default_date.isoformat()

                        upsert_predict_context(date_iso, drv, rte, veh, stops, packages, text)
                        # adăugăm și o înregistrare de tip “Predict OCR” în entries (ca să apară în totaluri zilnice)
                        insert_entry({
                            "date": date_iso, "driver": drv, "route": rte, "vehicle": veh,
                            "fuel_l": 0, "fuel_cost": 0,
                            "stops": stops, "packages": packages,
                            "notes": f"Predict PDF {name}"
                        })
                        status.update({"tip":"PREDICT_PDF","rows":1,
                                       "mesaj":f"{drv}/{rte}/{veh} | stops={stops}, packages={packages}"})
                # --------------- PREDICT IMAGINI ---------------
                elif ext in [".png",".jpg",".jpeg"]:
                    text, ocr_ok = try_ocr_image(raw)
                    date_iso = default_date.isoformat()
                    if not ocr_ok:
                        ocr_warnings.append(f"OCR pentru imagini nu este disponibil pe acest hosting. Poți introduce manual contextul pentru {name}.")
                        with st.expander(f"Introduce manual context Predict pentru {name} (data {date_iso})"):
                            drv_m = st.text_input("Șofer", key=f"drv_{name}")
                            rte_m = st.text_input("Tură", key=f"rte_{name}")
                            veh_m = st.text_input("Mașină (Kennzeichen)", key=f"veh_{name}")
                            stops_m = st.number_input("Stopuri", min_value=0, step=1, key=f"st_{name}")
                            pkgs_m = st.number_input("Pachete", min_value=0, step=1, key=f"pk_{name}")
                            if st.button(f"Salvează contextul pentru {name}"):
                                upsert_predict_context(date_iso,
                                                       drv_m or None,
                                                       (rte_m or None).upper() if rte_m else None,
                                                       (veh_m or None).upper() if veh_m else None,
                                                       int(stops_m), int(pkgs_m),
                                                       "(manual)")
                                insert_entry({
                                    "date": date_iso, "driver": drv_m or "AUTO",
                                    "route": (rte_m or "AUTO").upper(),
                                    "vehicle": (veh_m or "AUTO").upper(),
                                    "fuel_l": 0, "fuel_cost": 0,
                                    "stops": int(stops_m), "packages": int(pkgs_m),
                                    "notes": f"Predict MANUAL {name}"
                                })
                                st.success("Context salvat.")
                                status.update({"tip":"PREDICT_MANUAL","rows":1})
                    else:
                        drv = parse_driver(text) or "AUTO"
                        rte = parse_route(text) or "AUTO"
                        veh = parse_vehicle(text) or "AUTO"
                        stops = parse_stops(text) or 0
                        packages = parse_packages(text) or 0
                        upsert_predict_context(date_iso, drv, rte, veh, stops, packages, text or "")
                        insert_entry({
                            "date": date_iso, "driver": drv, "route": rte, "vehicle": veh,
                            "fuel_l": 0, "fuel_cost": 0,
                            "stops": stops, "packages": packages,
                            "notes": f"Predict IMG {name}"
                        })
                        status.update({"tip":"PREDICT_IMG","rows":1,
                                       "mesaj":f"{drv}/{rte}/{veh} | stops={stops}, packages={packages}"})
                else:
                    status.update({"tip":"IGNORED","mesaj":"Tip fișier neacoperit."})

            except Exception as e:
                status.update({"tip":"ERROR","mesaj":str(e)})

            processed.append(status)

        # export excel live
        try:
            write_master_excel()
        except Exception as e:
            st.warning(f"Nu am putut actualiza master.xlsx: {e}")

        st.success("Procesare terminată.")
        st.markdown("### Rezumat procesare")
        st.dataframe(pd.DataFrame(processed), use_container_width=True)

        if ocr_warnings:
            st.warning("• " + "\n• ".join(ocr_warnings))

    # ================== Statistici ==================
    st.subheader("2) Statistici")
    df = load_entries_df()
    if df.empty:
        st.info("Încă nu există date.")
        return

    # Filtre lună/an
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["month"] = pd.to_datetime(df["date"]).dt.month
    filt_month = st.selectbox("Luna", list(range(1,13)), index=dt.date.today().month-1)
    filt_year = st.number_input("An", value=dt.date.today().year, step=1)

    fdf = df[(df["year"]==filt_year) & (df["month"]==filt_month)].copy()
    if fdf.empty:
        st.warning("Nu sunt înregistrări pentru perioada selectată.")
        return

    # Zilnic
    daily = fdf.groupby("date", as_index=False).agg({
        "fuel_l":"sum","fuel_cost":"sum","stops":"sum","packages":"sum"
    })
    st.markdown("### ▶ Zilnic (Total litri, cost, stopuri, pachete)")
    st.dataframe(daily.sort_values("date"), use_container_width=True)

    # Pe tură
    by_route = fdf.groupby("route", as_index=False).agg({
        "fuel_l":"sum","fuel_cost":"sum","stops":"sum","packages":"sum"
    })
    st.markdown("### ▶ Pe tură (Litri & Cost + Stopuri & Pachete)")
    st.dataframe(by_route.sort_values(["fuel_l","fuel_cost"], ascending=False), use_container_width=True)

    # Export
    if MASTER_XLSX.exists():
        with open(MASTER_XLSX, "rb") as f:
            st.download_button("⬇️ Descarcă Excel (master.xlsx)", f, file_name="master.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------- Run ----------
if __name__ == "__main__":
    init_db()
    main()

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
    m = re.search(rf"\bmenge\s+{NUM}\b", t)  # ex: Menge 34,5
    if m:
        try:
            return float(m.group(1).replace(",", "."))
        except:
            pass
    m = re.search(rf"\b{NUM}\s*(l|liter|litri)\b", t)  # ex: 34 L
    if m:
        try:
            return float(m.group(1).replace(",", "."))
        except:
            pass
    return None

# ================== Header mapping pentru Excel/CSV ==================
HEADER_ALIASES = {
    # canonical -> posibile antete (normalizate)
    "date":   ["date", "datum", "data", "tag", "day"],
    "driver": ["driver", "fahrer", "sofer", "chauffeur", "conducator"],
    "route":  ["route", "tour", "tura", "tur", "linie"],
    "vehicle":["vehicle", "vehicul", "fahrzeug", "auto", "masina", "kennzeichen", "nr_inmatriculare"],
    "km":     ["km", "kilometer", "kilometri", "strecke"],
    "hours":  ["hours", "ore", "stunden", "zeit"],
    "revenue":["revenue", "venit", "einnahmen", "umsatz"],
    "stops":  ["stops", "stopuri", "pakete", "zustellpakete", "geplante_zustellpakette"],
    "fuel_l": ["fuel_l", "litri", "liter", "l", "menge"],
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
        # caz special: 'menge' -> fuel_l dacă nu s-a setat deja
        if canonical == "fuel_l" and "menge" in cols and "fuel_l" not in rename.values():
            rename["menge"] = "fuel_l"
    df = df.rename(columns=rename)
    return df

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

    # UI: Filtre
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

    # Uploader MULTI-fișiere
    st.subheader("1) Încarcă fișiere (jpg/png/pdf/xls/xlsx/csv)")
    uploads = st.file_uploader(
        "Alege unul sau mai multe fișiere",
        type=["png","jpg","jpeg","pdf","xls","xlsx","csv"],
        accept_multiple_files=True
    )

    processed_summary: List[Dict] = []
    ocr_debug: List[Tuple[str,str]] = []
    auto_inserts = 0

    # Afișează fișierele primite
    if uploads:
        rows = [{"fisier": u.name, "extensie": Path(u.name).suffix.lower(), "dimensiune_B": len(u.getvalue())} for u in uploads]
        st.markdown("**Fișiere primite:**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        month_dir = UPLOAD_DIR / f"{sel_year}-{str(sel_month).zfill(2)}"
        month_dir.mkdir(parents=True, exist_ok=True)

        for up in uploads:
            status_row = {"fisier": up.name, "tip": "", "rows_saved": 0, "mesaj": ""}
            detected_type = ""
            added_rows = 0

            try:
                raw_bytes = up.getvalue()
                ext = Path(up.name).suffix.lower()

                # Salvează copie
                unique = f"{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
                with open(month_dir / unique, "wb") as f:
                    f.write(raw_bytes)

                # OCR
                text = None
                if ext in [".png",".jpg",".jpeg"]:
                    text = try_ocr_image(raw_bytes)
                elif ext == ".pdf":
                    text = try_ocr_pdf(raw_bytes)

                if text:
                    ocr_debug.append((up.name, text[:4000]))
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
                        continue

                    df_x = map_headers(df_x)
                    cols = set(df_x.columns)

                    is_fuel = ("fuel_l" in cols) or ("menge" in cols) or ("liter" in cols) or ("l" in cols)
                    has_run_core = any(c in cols for c in ["driver","route","vehicle","km","stops","hours","revenue"])

                    if is_fuel and not has_run_core:
                        detected_type = "FUEL_EXCEL"
                        dcol = "date" if "date" in cols else None
                        for _, r in df_x.iterrows():
                            liters = None
                            for c in ["fuel_l","menge","liter","l"]:
                                if c in r and pd.notnull(r[c]):
                                    try:
                                        liters = float(str(r[c]).replace(",", "."))
                                        break
                                    except:
                                        pass
                            if liters is None:
                                continue

                            if dcol and pd.notnull(r.get(dcol, None)):
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
                                "notes": f"Fuel Excel {up.name}"
                            }
                            insert_entry(row)
                            auto_inserts += 1
                            added_rows += 1
                        status_row.update({"tip":"FUEL_EXCEL","rows_saved":added_rows})

                    elif has_run_core:
                        detected_type = "RUNS_EXCEL"
                        def to_float(x):
                            try: return float(str(x).replace(",",".")) if pd.notnull(x) else 0.0
                            except: return 0.0
                        for _, r in df_x.iterrows():
                            try:
                                date_iso = pd.to_datetime(r.get("date","")).date().isoformat()
                            except Exception:
                                date_iso = default_date.isoformat()
                            fuel_l = 0.0
                            for c in ["fuel_l","menge","liter","l"]:
                                if c in r and pd.notnull(r[c]):
                                    try:
                                        fuel_l = float(str(r[c]).replace(",", "."))
                                        break
                                    except:
                                        pass
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
                        detected_type = "UNKNOWN_EXCEL"
                        status_row.update({"tip":"UNKNOWN_EXCEL","mesaj":"Nu am recunoscut formatul (lipsesc coloane uzuale)."})

                # Imagine/PDF – bon motorină (Menge) dacă nu a fost alt tip
                if (ext in [".png",".jpg",".jpeg",".pdf"]) and text and detected_type == "":
                    liters = parse_liters(text)
                    if liters is not None and liters > 0:
                        date_iso = (default_date - dt.timedelta(days=1)).isoformat()
                        ctx = get_predict_context_for(date_iso) or {"driver":"AUTO","route":"AUTO","vehicle":"AUTO","stops":0}
                        row = {
                            "date": date_iso,
                            "driver": ctx["driver"], "route": ctx["route"], "vehicle": ctx["vehicle"],
                            "km": 0.0, "fuel_l": float(liters), "fuel_cost": float(liters) * FUEL_PRICE,
                            "hours": 0.0, "revenue": 0.0, "stops": int(ctx["stops"] or 0),
                            "notes": f"Fuel OCR {up.name}"
                        }
                        insert_entry(row)
                        auto_inserts += 1
                        detected_type = "FUEL_OCR"
                        status_row.update({"tip":"FUEL_OCR","rows_saved":1,
                                           "mesaj":f"{liters} L -> {row['fuel_cost']:.2f} €"})

                if not detected_type:
                    status_row.update({"tip":"NO_DETECTION","mesaj":"Nu am găsit nici Predict, nici Menge."})

            except Exception as e:
                status_row.update({"tip":"ERROR","mesaj":str(e)})

            processed_summary.append(status_row)

        # Excel live (safe)
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

    # ===== Statistici + Export =====
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

        # Zilnic
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

        st.markdown("### ▶ Per tură (litri & cost)")
        by_route = fdf.groupby("route", as_index=False).agg({
            "fuel_l":"sum", "fuel_cost":"sum", "km":"sum", "stops":"sum", "revenue":"sum"
        })
        by_route["profit"] = by_route["revenue"] - by_route["fuel_cost"]
        st.dataframe(by_route.sort_values(["fuel_l","fuel_cost"], ascending=False), use_container_width=True)

        st.markdown("### ▶ Per șofer")
        by_driver = fdf.groupby("driver", as_index=False).agg({
            "fuel_l":"sum", "fuel_cost":"sum", "km":"sum", "stops":"sum", "revenue":"sum"
        })
        by_driver["profit"] = by_driver["revenue"] - by_driver["fuel_cost"]
        st.dataframe(by_driver.sort_values(["fuel_l","fuel_cost"], ascending=False), use_container_width=True)

        st.markdown("### ▶ Zilnic")
        st.dataframe(daily.sort_values("date"), use_container_width=True)

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

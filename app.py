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
FUEL_PRICE = 1.6  # €/L cu TVA

DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "fleet.db"
for p in [DATA_DIR, UPLOAD_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ================== NUMERIC FORMATTING ==================
THINSPACE = "\u202f"  # thin space for thousands separator

def fmt_int(n: float | int) -> str:
    try:
        n = int(round(float(n)))
    except Exception:
        n = 0
    s = f"{n:,}"
    return s.replace(",", THINSPACE)

def fmt_eur(n: float | int, decimals: int = 0) -> str:
    try:
        n = float(n)
    except Exception:
        n = 0.0
    if decimals == 0:
        s = f"{n:,.0f}"
    else:
        s = f"{n:,.{decimals}f}"
    s = s.replace(",", THINSPACE)
    return f"{s} €"

# ================== DB (cache + migrare) ==================
@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA journal_mode=WAL;")
    migrate(conn)
    init_counters(conn)
    return conn

def migrate(conn: sqlite3.Connection):
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
    cur.execute("PRAGMA table_info(entries);")
    existing = {row[1] for row in cur.fetchall()}
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
        if col not in existing:
            cur.execute(f"ALTER TABLE entries ADD COLUMN {col} {decl};")
    conn.commit()

def init_counters(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS counters (
        name TEXT PRIMARY KEY,  -- 'fuel' / 'packages'
        since TEXT NOT NULL
    );
    """)
    for name in ("fuel", "packages"):
        cur.execute("INSERT OR IGNORE INTO counters(name, since) VALUES(?, ?)", (name, "2000-01-01"))
    conn.commit()

def get_counter_since(conn, name: str) -> dt.date:
    cur = conn.cursor()
    cur.execute("SELECT since FROM counters WHERE name=?", (name,))
    row = cur.fetchone()
    return dt.date.fromisoformat(row[0]) if row and row[0] else dt.date(2000,1,1)

def set_counter_since(conn, name: str, new_date: dt.date):
    cur = conn.cursor()
    cur.execute("UPDATE counters SET since=? WHERE name=?", (new_date.isoformat(), name))
    conn.commit()

def insert_entry(conn, row: dict):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO entries(date, driver, route, vehicle, fuel_l, fuel_cost, stops, packages, notes)
        VALUES (:date,:driver,:route,:vehicle,:fuel_l,:fuel_cost,:stops,:packages,:notes)
    """, row)
    conn.commit()

def load_entries_df(conn) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM entries", conn)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df

# ================== HELPERS ==================
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

HEADER_ALIASES = {
    "date":   ["datum","date","datum_tankung","tankdatum","belegdatum","datum_der_tankung"],
    "vehicle":["kennzeichen","fahrzeug","vehicle","kennz","kennz_zeichen"],
    "fuel_l": ["tankmenge","menge","liter","l","betankte_menge"]
}

def map_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = [norm_col(c) for c in df.columns]
    df.columns = cols
    rename = {}
    for canonical, alts in HEADER_ALIASES.items():
        for c in cols:
            if c in alts: rename[c] = canonical
    if "fuel_l" not in rename.values() and "menge" in cols:
        rename["menge"] = "fuel_l"
    return df.rename(columns=rename)

def parse_number(x) -> float:
    if pd.isna(x): return 0.0
    s = str(x).strip().replace("\u00a0","")
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except:
        try:
            return float(str(x))
        except:
            return 0.0

def month_last_day(d: dt.date) -> int:
    return calendar.monthrange(d.year, d.month)[1]

# ================== SIDEBAR CARDS ==================
CARD_CSS = """
<style>
.card {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.08);
  padding: 8px 10px;
  border-radius: 10px;
  margin-bottom: 8px;
}
.card .title {
  font-size: 0.85rem;
  opacity: 0.85;
  margin-bottom: 4px;
}
.card .value {
  font-size: 0.95rem;
  font-weight: 700;
}
.card .small {
  font-size: 0.75rem;
  opacity: 0.8;
}
</style>
"""

def sidebar_summary(conn, key_suffix: str = ""):
    st.markdown(CARD_CSS, unsafe_allow_html=True)

    df_all = load_entries_df(conn)
    fuel_since = get_counter_since(conn, "fuel")
    pkg_since  = get_counter_since(conn, "packages")

    if df_all.empty:
        total_l = total_c = total_p = 0
    else:
        df_all["date"] = pd.to_datetime(df_all["date"]).dt.date
        total_l = df_all.loc[df_all["date"] >= fuel_since, "fuel_l"].sum()
        total_c = df_all.loc[df_all["date"] >= fuel_since, "fuel_cost"].sum()
        total_p = int(df_all.loc[df_all["date"] >= pkg_since, "packages"].sum())

    st.markdown(f"""
    <div class="card">
      <div class="title">⛽ Motorină acumulată</div>
      <div class="value">{fmt_int(total_l)} L / {fmt_eur(total_c)}</div>
    </div>
    <div class="card">
      <div class="title">📦 Pachete acumulate</div>
      <div class="value">{fmt_int(total_p)}</div>
    </div>
    """, unsafe_allow_html=True)

    today = dt.date.today()
    last_day = month_last_day(today)
    if today.day in (16, last_day):
        st.warning("⚠️ Resetează contorul de **motorină** (facturare Tankpool).")
    if today.day in (15, last_day):
        st.warning("⚠️ Resetează contorul de **pachete** (facturare Predict).")

    with st.expander("Reset contoare"):
        c1, c2 = st.columns(2)
        if c1.button("Reset motorină de azi", key=f"btn_reset_fuel_{key_suffix}"):
            set_counter_since(conn, "fuel", dt.date.today())
            st.success("Contor motorină resetat (de azi).")
        if c2.button("Reset pachete de azi", key=f"btn_reset_pkg_{key_suffix}"):
            set_counter_since(conn, "packages", dt.date.today())
            st.success("Contor pachete resetat (de azi).")

# ================== APP ==================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    conn = get_conn()

    with st.sidebar:
        sidebar_summary(conn, key_suffix="top")  # chei unice -> nu mai apare DuplicateElementId

    st.subheader("1) Încarcă fișiere Tankpool (Excel) sau Predict (PDF/imagini)")
    uploads = st.file_uploader(
        "Fișiere (poți selecta mai multe odată)",
        type=["xls","xlsx","csv","pdf","png","jpg","jpeg"],
        accept_multiple_files=True
    )

    results = []
    if uploads:
        today = dt.date.today()
        for up in uploads:
            ext = Path(up.name).suffix.lower()
            raw = up.getvalue()

            # ------- TANKPOOL EXCEL -------
            if ext in (".xls", ".xlsx", ".csv"):
                try:
                    df_x = (pd.read_excel(io.BytesIO(raw))
                            if ext != ".csv"
                            else pd.read_csv(io.BytesIO(raw), sep=None, engine="python"))
                except Exception as e:
                    results.append({"fișier": up.name, "tip": "ERROR", "rows": 0, "mesaj": f"Eroare citire: {e}"})
                    continue

                df_x = map_headers(df_x)
                cols = set(df_x.columns)
                if "date" not in cols or "fuel_l" not in cols:
                    results.append({"fișier": up.name, "tip": "ERROR", "rows": 0,
                                    "mesaj": f"Coloane lipsă (cer: date & fuel_l). Găsite: {list(df_x.columns)}"})
                    continue

                inserted = 0
                for _, r in df_x.iterrows():
                    liters = parse_number(r.get("fuel_l", 0))
                    if liters <= 0:
                        continue
                    try:
                        refuel_date = pd.to_datetime(r.get("date"), dayfirst=True, errors="coerce").date()
                        if not refuel_date:
                            refuel_date = today
                    except Exception:
                        refuel_date = today
                    delivery_date = refuel_date + dt.timedelta(days=1)
                    veh = str(r.get("vehicle", "AUTO")).upper()

                    insert_entry(conn, {
                        "date": delivery_date.isoformat(),
                        "driver": "AUTO",
                        "route": veh,          # folosim Kennzeichen ca "tură"
                        "vehicle": veh,
                        "fuel_l": liters,
                        "fuel_cost": liters * FUEL_PRICE,
                        "stops": 0,
                        "packages": 0,
                        "notes": "Tankpool"
                    })
                    inserted += 1

                results.append({"fișier": up.name, "tip": "TANKPOOL_EXCEL", "rows": inserted, "mesaj": "OK"})

            # ------- PREDICT (introducere manuală simplă) -------
            else:
                with st.expander(f"Adaugă Predict manual pentru {up.name}"):
                    date_val = st.date_input("Data", value=today, key=f"d_{up.name}")
                    drv = st.text_input("Șofer", key=f"drv_{up.name}")
                    rte = st.text_input("Tură (ex. SALZWEG)", key=f"rte_{up.name}")
                    stops = st.number_input("Stopuri", min_value=0, step=1, key=f"s_{up.name}")
                    pkgs = st.number_input("Pachete (Geplante Zustellpakette)", min_value=0, step=1, key=f"p_{up.name}")
                    if st.button("Salvează Predict", key=f"btn_{up.name}"):
                        insert_entry(conn, {
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

        if results:
            st.success("Procesare terminată ✅")
            st.markdown("### Rezumat procesare")
            st.dataframe(pd.DataFrame(results), use_container_width=True)

        # afișez din nou sumarul cu chei diferite ca să nu ciocnesc ID-urile
        with st.sidebar:
            st.divider()
            sidebar_summary(conn, key_suffix="after_upload")

    # ------- STATISTICI -------
    st.subheader("2) Statistici")
    df = load_entries_df(conn)
    if df.empty:
        st.info("Nu există date încă.")
        return

    daily = df.groupby("date", as_index=False).agg({
        "fuel_l":"sum", "fuel_cost":"sum", "stops":"sum", "packages":"sum"
    })
    # Formatăm coloanele pentru vizualizare
    daily_fmt = daily.copy()
    daily_fmt["fuel_l"]   = daily["fuel_l"].apply(fmt_int)
    daily_fmt["fuel_cost"]= daily["fuel_cost"].apply(lambda x: fmt_eur(x, 0))
    st.markdown("### ▶ Pe zi (litri, cost, stopuri, pachete)")
    st.dataframe(daily_fmt.sort_values("date"), use_container_width=True)

    by_route = df.groupby("route", as_index=False).agg({
        "fuel_l":"sum", "fuel_cost":"sum", "stops":"sum", "packages":"sum"
    })
    by_route_fmt = by_route.copy()
    by_route_fmt["fuel_l"]    = by_route["fuel_l"].apply(fmt_int)
    by_route_fmt["fuel_cost"] = by_route["fuel_cost"].apply(lambda x: fmt_eur(x, 0))
    st.markdown("### ▶ Pe tură (litri, cost, stopuri, pachete)")
    st.dataframe(by_route_fmt.sort_values(["fuel_l","fuel_cost"], ascending=False), use_container_width=True)

# ---------- RUN ----------
if __name__ == "__main__":
    get_conn()  # inițializează/migrează și pune în cache
    main()

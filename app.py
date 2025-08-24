import io, re, sqlite3, unicodedata, datetime as dt, calendar
from pathlib import Path
import pandas as pd
import streamlit as st
from pdfminer.high_level import extract_text  # pentru Tankpool PDF

# ================== CONFIG ==================
APP_TITLE = "SANS – Motorină & Predict (Tankpool)"
DATA_DIR = Path("data"); UPLOAD_DIR = DATA_DIR / "uploads"; DB_PATH = DATA_DIR / "fleet.db"
for p in (DATA_DIR, UPLOAD_DIR): p.mkdir(parents=True, exist_ok=True)

VAT_RATE = 0.19
FUEL_PRICE_GROSS = 1.6  # €/L (cu TVA) pt. Excel

# ================== FORMAT (EU) ==================
def de_thousands(n: float, dec: int = 0) -> str:
    try: n = float(n)
    except: n = 0.0
    s = f"{n:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return s

def de_eur(n: float, dec: int = 2) -> str:
    return f"{de_thousands(n, dec)} €"

# ================== DB ==================
@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys=ON;"); migrate(conn); init_counters(conn); return conn

def migrate(conn):
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS entries(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL, driver TEXT, route TEXT, vehicle TEXT,
        fuel_l REAL DEFAULT 0, fuel_cost_net REAL DEFAULT 0, fuel_cost_gross REAL DEFAULT 0,
        stops INTEGER DEFAULT 0, packages INTEGER DEFAULT 0, notes TEXT );""")
    c.execute("PRAGMA table_info(entries);")
    have = {r[1] for r in c.fetchall()}
    if "fuel_cost_net" not in have:   c.execute("ALTER TABLE entries ADD COLUMN fuel_cost_net REAL DEFAULT 0;")
    if "fuel_cost_gross" not in have: c.execute("ALTER TABLE entries ADD COLUMN fuel_cost_gross REAL DEFAULT 0;")
    conn.commit()

def init_counters(conn):
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS counters(name TEXT PRIMARY KEY, since TEXT NOT NULL);""")
    for name in ("fuel","packages"):
        c.execute("INSERT OR IGNORE INTO counters(name,since) VALUES(?,?)",(name,"2000-01-01"))
    conn.commit()

def hard_reset_db():
    if DB_PATH.exists(): DB_PATH.unlink()
    st.cache_resource.clear(); st.toast("DB ștearsă. Dă refresh.", icon="⚠️")

def get_counter_since(conn, name): 
    c = conn.cursor(); c.execute("SELECT since FROM counters WHERE name=?", (name,))
    row = c.fetchone(); return dt.date.fromisoformat(row[0]) if row else dt.date(2000,1,1)

def set_counter_since(conn, name, d: dt.date):
    conn.cursor().execute("UPDATE counters SET since=? WHERE name=?", (d.isoformat(), name)); conn.commit()

def insert_entry(conn, row: dict):
    conn.cursor().execute("""INSERT INTO entries
        (date,driver,route,vehicle,fuel_l,fuel_cost_net,fuel_cost_gross,stops,packages,notes)
        VALUES (:date,:driver,:route,:vehicle,:fuel_l,:fuel_cost_net,:fuel_cost_gross,:stops,:packages,:notes)""", row)
    conn.commit()

def load_entries_df(conn)->pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM entries", conn)
    if not df.empty: df["date"] = pd.to_datetime(df["date"]).dt.date
    return df

# ================== HELPERS ==================
def month_last_day(d: dt.date) -> int: return calendar.monthrange(d.year, d.month)[1]

# --- Detect & parse Tankmenge corect pentru toată coloana ---
def parse_liters_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.replace("\u00a0","", regex=False)

    # statistici simple
    has_comma = s.str.contains(",").mean() > 0.5
    has_dot   = s.str.contains(r"\.").mean() > 0.5

    def _to_float(val: str) -> float:
        if val == "" or val.lower() == "nan": return 0.0
        v = val
        # Reguli:
        # 1) Doar virgule => virgula este zecimal
        # 2) Doar puncte: dacă toate valorile < 300 => punct zecimal, altfel punct mii
        # 3) Ambele semne => format german: . mii, , zecimal
        if "," in v and "." not in v:
            v = v.replace(",", ".")
        elif "." in v and "," not in v:
            # decidem pe baza formei locale a valorii
            # dacă are exact 3 cifre după ., îl considerăm zecimal (gen 9.626)
            m = re.search(r"\.(\d+)$", v)
            if m and 1 <= len(m.group(1)) <= 3:
                pass  # '.' zecimal
            else:
                v = v.replace(".", "")  # '.' mii
        else:
            # ambele semne -> german
            v = v.replace(".", "").replace(",", ".")
        try:
            return float(v)
        except:
            return 0.0

    return s.apply(_to_float)

# ================== Tankpool PDF ==================
PDF_LINE = re.compile(
    r"(?P<date>\d{2}\.\d{2}\.\d{2})\s+\d{2}:\d{2}\s+\d+\s+[A-Za-zÄÖÜäöüß.\- ]+\s+"
    r"(?P<prod>Diesel|AdBlue)\s+(?P<liters>[\d.,]+)\s+L\s+(?P<unit_net>[\d.,]+)\s+(?P<sum_net>[\d.,]+)",
    re.IGNORECASE
)

def parse_tankpool_pdf(content: bytes) -> pd.DataFrame:
    text = extract_text(io.BytesIO(content)) or ""
    rows=[]
    for m in PDF_LINE.finditer(text):
        if m.group("prod").lower()!="diesel": continue
        liters = parse_liters_series(pd.Series([m.group("liters")]))[0]
        unit   = parse_liters_series(pd.Series([m.group("unit_net")]))[0]
        net    = parse_liters_series(pd.Series([m.group("sum_net")]))[0]
        try: d = pd.to_datetime(m.group("date"), format="%d.%m.%y").date()
        except: continue
        if net<=0: net = liters*unit
        rows.append({
            "date": d, "vehicle":"AUTO", "fuel_l": liters,
            "fuel_cost_net": net, "fuel_cost_gross": net*(1+VAT_RATE),
            "notes":"Tankpool PDF"
        })
    return pd.DataFrame(rows)

# ================== SIDEBAR ==================
def sidebar_summary(conn, key_suffix: str):
    df = load_entries_df(conn)
    fuel_since = get_counter_since(conn, "fuel"); pkg_since = get_counter_since(conn, "packages")
    if df.empty: l_sum=net_sum=gross_sum=pk_sum=0
    else:
        df["date"]=pd.to_datetime(df["date"]).dt.date
        l_sum   = df[df["date"]>=fuel_since]["fuel_l"].sum()
        net_sum = df[df["date"]>=fuel_since]["fuel_cost_net"].sum()
        gross_sum = df[df["date"]>=fuel_since]["fuel_cost_gross"].sum()
        pk_sum  = int(df[df["date"]>=pkg_since]["packages"].sum())
    st.markdown("### Sumar live")
    st.markdown(f"**⛽ Motorină acumulată:** {de_thousands(l_sum,3)} L • {de_eur(gross_sum,2)}  \n_net: {de_eur(net_sum,2)}_")
    st.markdown(f"**📦 Pachete acumulate:** {de_thousands(pk_sum,0)}")
    today = dt.date.today(); last_day = month_last_day(today)
    if today.day in (16, last_day): st.warning("⚠️ Resetează **motorina** (Tankpool).")
    if today.day in (15, last_day): st.warning("⚠️ Resetează **pachetele**.")
    with st.expander("Administrare"):
        c1,c2,c3 = st.columns(3)
        if c1.button("Reset motorină", key=f"rf_{key_suffix}"):
            set_counter_since(conn,"fuel",dt.date.today()); st.success("Reset motorină.")
        if c2.button("Reset pachete", key=f"rp_{key_suffix}"):
            set_counter_since(conn,"packages",dt.date.today()); st.success("Reset pachete.")
        if c3.button("Hard reset DB", key=f"hard_{key_suffix}"):
            hard_reset_db(); st.stop()

# ================== APP ==================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    conn = get_conn()

    with st.sidebar: sidebar_summary(conn, "top")

    st.subheader("1) Încarcă fișiere (Tankpool: Excel/PDF)")
    uploads = st.file_uploader("Selectează fișiere (poți mai multe). Pentru costuri corecte, PDF Tankpool ideal.",
                               type=["xls","xlsx","csv","pdf"], accept_multiple_files=True)

    results=[]
    if uploads:
        today = dt.date.today()
        for up in uploads:
            name, ext = up.name, Path(up.name).suffix.lower()
            raw = up.getvalue()

            if ext==".pdf":
                dfp = parse_tankpool_pdf(raw); ins=0
                for _,r in dfp.iterrows():
                    delivery = r["date"] + dt.timedelta(days=1)
                    insert_entry(conn, {
                        "date": delivery.isoformat(), "driver":"AUTO", "route":"AUTO", "vehicle": r.get("vehicle","AUTO"),
                        "fuel_l": float(r["fuel_l"]), "fuel_cost_net": float(r["fuel_cost_net"]),
                        "fuel_cost_gross": float(r["fuel_cost_gross"]), "stops":0, "packages":0, "notes": r.get("notes","Tankpool PDF")
                    }); ins+=1
                results.append({"fișier":name,"tip":"TANKPOOL_PDF","rows":ins,"mesaj":"OK"}); continue

            if ext in (".xls",".xlsx",".csv"):
                # citește ca STRING ca să nu lase Excel să-și impună formatul
                df_x = (pd.read_excel(io.BytesIO(raw), dtype=str)
                        if ext != ".csv" else pd.read_csv(io.BytesIO(raw), dtype=str, sep=None, engine="python"))
                # Aștept: 'Datum', 'Tankmenge', 'Kennzeichen' (variază, dar astea sunt uzuale)
                if "Tankmenge" not in df_x.columns:
                    # încearcă câteva alias-uri
                    for alt in ["Menge","Liter","Betankte Menge","Tankmenge [l]"]:
                        if alt in df_x.columns: df_x.rename(columns={alt:"Tankmenge"}, inplace=True)
                if "Datum" not in df_x.columns:
                    for alt in ["Date","Belegdatum","Tankdatum","Datum Tankung"]:
                        if alt in df_x.columns: df_x.rename(columns={alt:"Datum"}, inplace=True)
                if "Kennzeichen" not in df_x.columns:
                    for alt in ["Fahrzeug","Kennz","Kennz."]:
                        if alt in df_x.columns: df_x.rename(columns={alt:"Kennzeichen"}, inplace=True)

                if "Tankmenge" not in df_x.columns or "Datum" not in df_x.columns:
                    results.append({"fișier":name,"tip":"ERROR","rows":0,"mesaj":f"Coloane lipsă. Găsite: {list(df_x.columns)}"})
                    continue

                liters_series = parse_liters_series(df_x["Tankmenge"])
                ins=0
                for idx,row in df_x.iterrows():
                    liters = float(liters_series.iloc[idx])
                    if liters<=0: continue
                    try: refuel = pd.to_datetime(row.get("Datum"), dayfirst=True, errors="coerce").date()
                    except: refuel = today
                    delivery = refuel + dt.timedelta(days=1)
                    veh = str(row.get("Kennzeichen","AUTO")).upper()

                    gross = liters * FUEL_PRICE_GROSS
                    net   = gross / (1+VAT_RATE)

                    insert_entry(conn, {
                        "date": delivery.isoformat(), "driver":"AUTO", "route": veh, "vehicle": veh,
                        "fuel_l": liters, "fuel_cost_net": net, "fuel_cost_gross": gross,
                        "stops":0, "packages":0, "notes":"Tankpool Excel"
                    }); ins+=1
                results.append({"fișier":name,"tip":"TANKPOOL_EXCEL","rows":ins,"mesaj":"OK"}); continue

        if results:
            st.success("Procesare terminată ✅")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        with st.sidebar: st.divider(); sidebar_summary(conn, "after")

    # 2) Statistici
    st.subheader("2) Statistici")
    df = load_entries_df(conn)
    if df.empty: st.info("Nu există date încă."); return

    daily = df.groupby("date", as_index=False).agg({
        "fuel_l":"sum","fuel_cost_net":"sum","fuel_cost_gross":"sum","stops":"sum","packages":"sum"
    }).sort_values("date")
    daily_fmt = daily.copy()
    daily_fmt["fuel_l"] = daily_fmt["fuel_l"].apply(lambda x: de_thousands(x,3))
    daily_fmt["fuel_cost_net"] = daily_fmt["fuel_cost_net"].apply(lambda x: de_eur(x,2))
    daily_fmt["fuel_cost_gross"] = daily_fmt["fuel_cost_gross"].apply(lambda x: de_eur(x,2))
    st.markdown("### ▶ Pe zi (litri, cost net/brut, stopuri, pachete)")
    st.dataframe(daily_fmt.rename(columns={"fuel_l":"litri","fuel_cost_net":"cost (net)","fuel_cost_gross":"cost (brut)"}),
                 use_container_width=True)

    by_route = df.groupby("route", as_index=False).agg({
        "fuel_l":"sum","fuel_cost_net":"sum","fuel_cost_gross":"sum","stops":"sum","packages":"sum"
    }).sort_values(["fuel_l","fuel_cost_gross"], ascending=False)
    br_fmt = by_route.copy()
    br_fmt["fuel_l"] = br_fmt["fuel_l"].apply(lambda x: de_thousands(x,3))
    br_fmt["fuel_cost_net"] = br_fmt["fuel_cost_net"].apply(lambda x: de_eur(x,2))
    br_fmt["fuel_cost_gross"] = br_fmt["fuel_cost_gross"].apply(lambda x: de_eur(x,2))
    st.markdown("### ▶ Pe tură (litri, cost net/brut, stopuri, pachete)")
    st.dataframe(br_fmt.rename(columns={"route":"tură","fuel_l":"litri","fuel_cost_net":"cost (net)","fuel_cost_gross":"cost (brut)"}),
                 use_container_width=True)

# ---------- RUN ----------
if __name__ == "__main__":
    get_conn(); main()

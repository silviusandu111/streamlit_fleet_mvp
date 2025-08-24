import io, re, sqlite3, unicodedata, datetime as dt, calendar
from pathlib import Path

import pandas as pd
import streamlit as st
from pdfminer.high_level import extract_text  # PDF text
import plotly.express as px

# OCR (opțional). Dacă nu se poate instala, facem fallback la formular manual.
try:
    import easyocr
    OCR_READER = easyocr.Reader(["de", "en"], gpu=False)
except Exception:
    OCR_READER = None

# ---------------- CONFIG ----------------
APP_TITLE = "SANS — Motorină & Predict"
DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "fleet.db"
for p in (DATA_DIR, UPLOAD_DIR):
    p.mkdir(parents=True, exist_ok=True)

VAT_RATE = 0.19
FUEL_PRICE_GROSS = 1.6  # €/L cu TVA (Excel Tankpool)

# ---------------- THEME/UI ----------------
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="⛽")
st.markdown(
    """
<style>
.small-muted { opacity:.7;font-size:.9rem }
.metric-card { background:#111827;border:1px solid #1f2937;padding:12px 16px;border-radius:14px }
.card{ background:#0b1220;border:1px solid #162033;padding:16px;border-radius:14px }
.section-h{ font-weight:700;font-size:1.05rem;margin-bottom:8px }
.stDataFrame { border-radius:10px;overflow:hidden }
.file-chip{display:inline-block;margin:4px 6px;padding:6px 10px;border:1px solid #1f2937;border-radius:999px;background:#0b1220}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- FORMAT (EU, fără zecimale) ----------------
def de_thousands_int(x):
    try:
        x = float(x)
    except:
        x = 0.0
    s = f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return s

def de_eur0(x):
    return f"{de_thousands_int(x)} €"

# ---------------- DB ----------------
@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys=ON;")
    migrate(conn)
    init_counters(conn)
    return conn

def migrate(conn):
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS entries(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        driver TEXT, route TEXT, vehicle TEXT,
        fuel_l REAL DEFAULT 0,
        fuel_cost_net REAL DEFAULT 0,
        fuel_cost_gross REAL DEFAULT 0,
        stops INTEGER DEFAULT 0,
        packages INTEGER DEFAULT 0,
        notes TEXT
    );"""
    )
    # coloane garantate
    c.execute("PRAGMA table_info(entries);")
    have = {r[1] for r in c.fetchall()}
    if "fuel_cost_net" not in have:
        c.execute("ALTER TABLE entries ADD COLUMN fuel_cost_net REAL DEFAULT 0;")
    if "fuel_cost_gross" not in have:
        c.execute("ALTER TABLE entries ADD COLUMN fuel_cost_gross REAL DEFAULT 0;")
    conn.commit()

def init_counters(conn):
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS counters(name TEXT PRIMARY KEY, since TEXT NOT NULL);""")
    for name in ("fuel", "packages"):
        c.execute("INSERT OR IGNORE INTO counters(name,since) VALUES(?,?)", (name, "2000-01-01"))
    conn.commit()

def hard_reset_db():
    if DB_PATH.exists():
        DB_PATH.unlink()
    st.cache_resource.clear()
    st.toast("Baza de date resetată. Reîncarcă pagina (Ctrl/Cmd+R).", icon="⚠️")

def get_counter_since(conn, name):
    c = conn.cursor()
    c.execute("SELECT since FROM counters WHERE name=?", (name,))
    row = c.fetchone()
    return dt.date.fromisoformat(row[0]) if row else dt.date(2000, 1, 1)

def set_counter_since(conn, name, d: dt.date):
    conn.cursor().execute("UPDATE counters SET since=? WHERE name=?", (d.isoformat(), name))
    conn.commit()

def insert_entry(conn, row: dict):
    conn.cursor().execute(
        """INSERT INTO entries
        (date,driver,route,vehicle,fuel_l,fuel_cost_net,fuel_cost_gross,stops,packages,notes)
        VALUES (:date,:driver,:route,:vehicle,:fuel_l,:fuel_cost_net,:fuel_cost_gross,:stops,:packages,:notes)""",
        row,
    )
    conn.commit()

def load_entries_df(conn) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM entries", conn)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df

# ---------------- HELPERS ----------------
def month_last_day(d: dt.date) -> int:
    import calendar
    return calendar.monthrange(d.year, d.month)[1]

def parse_liters_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.replace("\u00a0", "", regex=False)
    def _to_float(v: str) -> float:
        if v == "" or v.lower() == "nan":
            return 0.0
        if "," in v and "." not in v:
            v = v.replace(",", ".")
        elif "." in v and "," not in v:
            m = re.search(r"\.(\d+)$", v)
            if not (m and 1 <= len(m.group(1)) <= 3):
                v = v.replace(".", "")
        else:
            v = v.replace(".", "").replace(",", ".")
        try:
            return float(v)
        except:
            return 0.0
    return s.apply(_to_float)

# ---------------- Tankpool PDF ----------------
PDF_TANKPOOL_RX = re.compile(
    r"(?P<date>\d{2}\.\d{2}\.\d{2})\s+\d{2}:\d{2}\s+\d+\s+[A-Za-zÄÖÜäöüß.\- ]+\s+"
    r"(?P<prod>Diesel|AdBlue)\s+(?P<liters>[\d.,]+)\s+L\s+(?P<unit_net>[\d.,]+)\s+(?P<sum_net>[\d.,]+)",
    re.IGNORECASE,
)

def parse_tankpool_pdf(content: bytes) -> pd.DataFrame:
    text = extract_text(io.BytesIO(content)) or ""
    rows = []
    for m in PDF_TANKPOOL_RX.finditer(text):
        if m.group("prod").lower() != "diesel":
            continue
        liters = parse_liters_series(pd.Series([m.group("liters")]))[0]
        unit = parse_liters_series(pd.Series([m.group("unit_net")]))[0]
        net = parse_liters_series(pd.Series([m.group("sum_net")]))[0]
        try:
            d = pd.to_datetime(m.group("date"), format="%d.%m.%y").date()
        except:
            continue
        if net <= 0:
            net = liters * unit
        rows.append(
            {
                "date": d,
                "vehicle": "AUTO",
                "fuel_l": liters,
                "fuel_cost_net": net,
                "fuel_cost_gross": net * (1 + VAT_RATE),
                "notes": "Tankpool PDF",
            }
        )
    return pd.DataFrame(rows)

# ---------------- Predict (JPG/PDF) — doar PACHETE TOTALE ----------------
DATE_RX = re.compile(r"(\d{2}\.\d{2}\.\d{4})")

def ocr_image_to_text(b: bytes) -> str:
    if not OCR_READER:
        return ""
    try:
        import numpy as np
        from PIL import Image
        img = Image.open(io.BytesIO(b)).convert("RGB")
        arr = np.array(img)
        lines = OCR_READER.readtext(arr, detail=0, paragraph=True)
        return "\n".join(lines)
    except Exception:
        return ""

def extract_packages_total(text: str) -> int:
    """Heuristică: caută linii cu 'geplante' & 'paket', altfel însumează toate numerele de 2-4 cifre (>=50)."""
    if not text:
        return 0
    t = unicodedata.normalize("NFKD", text)
    total = 0
    for line in t.splitlines():
        low = line.lower()
        if "geplante" in low and "paket" in low:
            nums = [int(n) for n in re.findall(r"\b\d{2,4}\b", line)]
            total += sum(nums)
    if total > 0:
        return total
    nums = [int(n) for n in re.findall(r"\b\d{2,4}\b", t)]
    nums = [x for x in nums if 50 <= x <= 5000]
    return sum(nums)

# ---------------- SIDEBAR ----------------
def sidebar_summary(conn, key_suffix: str):
    df = load_entries_df(conn)
    fuel_since = get_counter_since(conn, "fuel")
    pkg_since = get_counter_since(conn, "packages")
    if df.empty:
        l_sum = net_sum = gross_sum = pk_sum = 0
    else:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        l_sum = df[df["date"] >= fuel_since]["fuel_l"].sum()
        net_sum = df[df["date"] >= fuel_since]["fuel_cost_net"].sum()
        gross_sum = df[df["date"] >= fuel_since]["fuel_cost_gross"].sum()
        pk_sum = int(df[df["date"] >= pkg_since]["packages"].sum())

    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(
        f"**⛽ Motorină acumulată:** {de_thousands_int(l_sum)} L • {de_eur0(gross_sum)}  \n"
        f"<span class='small-muted'>net: {de_eur0(net_sum)}</span>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="metric-card" style="margin-top:10px;">', unsafe_allow_html=True)
    st.markdown(f"**📦 Pachete acumulate:** {de_thousands_int(pk_sum)}", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    today = dt.date.today()
    last_day = month_last_day(today)
    if today.day in (16, last_day):
        st.warning("⚠️ Resetează **motorina** (Tankpool).")
    if today.day in (15, last_day):
        st.warning("⚠️ Resetează **pachetele**.")

    with st.expander("Administrare"):
        c1, c2, c3 = st.columns(3)
        if c1.button("Reset motorină", key=f"rf_{key_suffix}"):
            set_counter_since(conn, "fuel", dt.date.today())
            st.success("Reset motorină.")
        if c2.button("Reset pachete", key=f"rp_{key_suffix}"):
            set_counter_since(conn, "packages", dt.date.today())
            st.success("Reset pachete.")
        if c3.button("Hard reset DB", key=f"hard_{key_suffix}"):
            hard_reset_db()
            st.stop()

# ---------------- APP ----------------
def main():
    st.title("SANS — Motorină (Tankpool) & Predict (pachete)")
    conn = get_conn()

    # state pt. poze care cer completare manuală (nu dispar la rerun)
    if "predict_manual" not in st.session_state:
        st.session_state.predict_manual = []

    with st.sidebar:
        sidebar_summary(conn, "top")

    # ---------- Upload ----------
    with st.container():
        st.markdown('<div class="card"><div class="section-h">1) Încarcă fișiere</div>', unsafe_allow_html=True)
        default_predict_date = st.date_input(
            "Dată implicită pentru Predict (dacă nu se găsește în document)", dt.date.today()
        )
        uploads = st.file_uploader(
            "Alege Tankpool (Excel/PDF) sau Predict (PDF/JPG/PNG). Poți încărca mai multe.",
            type=["xls", "xlsx", "csv", "pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="uploader",
        )

        # Afișează instant ce ai selectat
        if uploads:
            st.write("**Fișiere selectate:** ", unsafe_allow_html=True)
            for f in uploads:
                st.markdown(f"<span class='file-chip'>{f.name} — {len(f.getvalue())//1024} KB</span>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    results = []

    # procesează fișierele
    if uploads:
        for up in uploads:
            name = up.name
            suffix = Path(name).suffix.lower()
            raw = up.getvalue()

            # Salvează fizic (ca să știi că “a intrat”)
            save_path = UPLOAD_DIR / name
            try:
                with open(save_path, "wb") as fh:
                    fh.write(raw)
            except Exception:
                pass  # dacă e readonly pe cloud, ignorăm

            # ---- PDF: Tankpool sau Predict după conținut
            if suffix == ".pdf":
                text = extract_text(io.BytesIO(raw)) or ""
                if re.search(r"Diesel\s+[\d.,]+\s+L", text, re.IGNORECASE):
                    dfp = parse_tankpool_pdf(raw)
                    ins = 0
                    for _, r in dfp.iterrows():
                        delivery = r["date"] + dt.timedelta(days=1)
                        insert_entry(
                            conn,
                            {
                                "date": delivery.isoformat(),
                                "driver": "AUTO",
                                "route": "AUTO",
                                "vehicle": "AUTO",
                                "fuel_l": float(r["fuel_l"]),
                                "fuel_cost_net": float(r["fuel_cost_net"]),
                                "fuel_cost_gross": float(r["fuel_cost_gross"]),
                                "stops": 0,
                                "packages": 0,
                                "notes": "Tankpool PDF",
                            },
                        )
                        ins += 1
                    results.append({"fișier": name, "tip": "TANKPOOL_PDF", "rows": ins, "mesaj": "OK"})
                else:
                    total_pk = extract_packages_total(text)
                    m = DATE_RX.search(text)
                    d = (
                        pd.to_datetime(m.group(1), dayfirst=True, errors="coerce").date()
                        if m
                        else default_predict_date
                    )
                    insert_entry(
                        conn,
                        {
                            "date": d.isoformat(),
                            "driver": "AUTO",
                            "route": "PREDICT",
                            "vehicle": "AUTO",
                            "fuel_l": 0,
                            "fuel_cost_net": 0,
                            "fuel_cost_gross": 0,
                            "stops": 0,
                            "packages": int(total_pk),
                            "notes": "Predict PDF (total pachete)",
                        },
                    )
                    results.append({"fișier": name, "tip": "PREDICT_PDF", "rows": 1, "mesaj": f"pachete={int(total_pk)}"})
                continue

            # ---- IMAGINI (Predict)
            if suffix in (".png", ".jpg", ".jpeg"):
                txt = ocr_image_to_text(raw)
                total_pk = extract_packages_total(txt) if txt else 0
                if total_pk > 0:
                    insert_entry(
                        conn,
                        {
                            "date": default_predict_date.isoformat(),
                            "driver": "AUTO",
                            "route": "PREDICT",
                            "vehicle": "AUTO",
                            "fuel_l": 0,
                            "fuel_cost_net": 0,
                            "fuel_cost_gross": 0,
                            "stops": 0,
                            "packages": int(total_pk),
                            "notes": "Predict IMG OCR (total pachete)",
                        },
                    )
                    results.append({"fișier": name, "tip": "PREDICT_IMG", "rows": 1, "mesaj": f"pachete={int(total_pk)}"})
                else:
                    # Păstrăm în state pentru formular manual afișat mai jos
                    st.session_state.predict_manual.append(
                        {"name": name, "date": default_predict_date, "reason": "OCR indisponibil/nesigur"}
                    )
                    results.append({"fișier": name, "tip": "PREDICT_IMG", "rows": 0, "mesaj": "OCR n/a — formular manual"})
                continue

            # ---- EXCEL/CSV (Tankpool)
            if suffix in (".xls", ".xlsx", ".csv"):
                if suffix != ".csv":
                    df_x = pd.read_excel(io.BytesIO(raw), dtype=str)
                else:
                    df_x = pd.read_csv(io.BytesIO(raw), dtype=str, sep=None, engine="python")

                # Normalizez antete
                if "Tankmenge" not in df_x.columns:
                    for alt in ["Menge", "Liter", "Betankte Menge", "Tankmenge [l]"]:
                        if alt in df_x.columns:
                            df_x.rename(columns={alt: "Tankmenge"}, inplace=True)
                if "Datum" not in df_x.columns:
                    for alt in ["Date", "Belegdatum", "Tankdatum", "Datum Tankung"]:
                        if alt in df_x.columns:
                            df_x.rename(columns={alt: "Datum"}, inplace=True)
                if "Kennzeichen" not in df_x.columns:
                    for alt in ["Fahrzeug", "Kennz", "Kennz.", "Ort", "Route"]:
                        if alt in df_x.columns:
                            df_x.rename(columns={alt: "Kennzeichen"}, inplace=True)

                if "Tankmenge" not in df_x.columns or "Datum" not in df_x.columns:
                    results.append(
                        {"fișier": name, "tip": "ERROR", "rows": 0, "mesaj": f"Coloane lipsă. Găsite: {list(df_x.columns)}"}
                    )
                    continue

                liters_series = parse_liters_series(df_x["Tankmenge"])
                ins = 0
                for idx, row in df_x.iterrows():
                    liters = float(liters_series.iloc[idx])
                    if liters <= 0:
                        continue
                    try:
                        refuel = pd.to_datetime(row.get("Datum"), dayfirst=True, errors="coerce").date()
                    except:
                        refuel = dt.date.today()
                    delivery = refuel + dt.timedelta(days=1)
                    route = (str(row.get("Kennzeichen", "AUTO")).strip() or "AUTO").upper()

                    gross = liters * FUEL_PRICE_GROSS
                    net = gross / (1 + VAT_RATE)

                    insert_entry(
                        conn,
                        {
                            "date": delivery.isoformat(),
                            "driver": "AUTO",
                            "route": route,
                            "vehicle": route,
                            "fuel_l": liters,
                            "fuel_cost_net": net,
                            "fuel_cost_gross": gross,
                            "stops": 0,
                            "packages": 0,
                            "notes": "Tankpool Excel",
                        },
                    )
                    ins += 1
                results.append({"fișier": name, "tip": "TANKPOOL_EXCEL", "rows": ins, "mesaj": "OK"})
                continue

        # Rezumat procesare
        if results:
            st.success("Procesare terminată ✅")
            st.dataframe(pd.DataFrame(results), use_container_width=True)

    # Formular manual pentru imaginile Predict care nu s-au putut citi
    if st.session_state.predict_manual:
        st.markdown('<div class="card"><div class="section-h">Completare manuală pentru imagini Predict</div>', unsafe_allow_html=True)
        to_keep = []
        for i, item in enumerate(st.session_state.predict_manual):
            c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
            with c1:
                st.write(f"**{item['name']}** — {item['reason']}")
            with c2:
                d = st.date_input("Data", value=item["date"], key=f"pm_d_{i}")
            with c3:
                p = st.number_input("Total pachete", min_value=0, step=1, key=f"pm_p_{i}")
            with c4:
                if st.button("Salvează", key=f"pm_b_{i}"):
                    insert_entry(
                        get_conn(),
                        {
                            "date": d.isoformat(),
                            "driver": "AUTO",
                            "route": "PREDICT",
                            "vehicle": "AUTO",
                            "fuel_l": 0,
                            "fuel_cost_net": 0,
                            "fuel_cost_gross": 0,
                            "stops": 0,
                            "packages": int(p),
                            "notes": "Predict manual (IMG)",
                        },
                    )
                    st.success(f"Salvat pachete={int(p)} pentru {item['name']}")
                else:
                    to_keep.append(item)
        st.session_state.predict_manual = to_keep
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Statistici + Chart ----------
    df = load_entries_df(get_conn())
    st.markdown('<div class="card"><div class="section-h">2) Statistici (zilnic & pe tură)</div>', unsafe_allow_html=True)
    if df.empty:
        st.info("Nu există date încă.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    daily = (
        df.groupby("date", as_index=False)
        .agg({"fuel_l": "sum", "fuel_cost_net": "sum", "fuel_cost_gross": "sum", "stops": "sum", "packages": "sum"})
        .sort_values("date")
    )
    daily_fmt = daily.copy()
    daily_fmt["fuel_l"] = daily_fmt["fuel_l"].apply(de_thousands_int)
    daily_fmt["fuel_cost_net"] = daily_fmt["fuel_cost_net"].apply(de_eur0)
    daily_fmt["fuel_cost_gross"] = daily_fmt["fuel_cost_gross"].apply(de_eur0)
    st.markdown("**▶ Pe zi (litri, cost net/brut, pachete)**")
    st.dataframe(
        daily_fmt.rename(columns={"fuel_l": "litri", "fuel_cost_net": "cost (net)", "fuel_cost_gross": "cost (brut)"}),
        use_container_width=True,
    )

    by_route = (
        df.groupby("route", as_index=False)
        .agg({"fuel_l": "sum", "fuel_cost_net": "sum", "fuel_cost_gross": "sum", "stops": "sum", "packages": "sum"})
        .sort_values(["fuel_l", "fuel_cost_gross"], ascending=False)
    )
    by_route_fmt = by_route.copy()
    by_route_fmt["fuel_l"] = by_route_fmt["fuel_l"].apply(de_thousands_int)
    by_route_fmt["fuel_cost_net"] = by_route_fmt["fuel_cost_net"].apply(de_eur0)
    by_route_fmt["fuel_cost_gross"] = by_route_fmt["fuel_cost_gross"].apply(de_eur0)
    st.markdown("**▶ Pe tură / oraș (litri, cost net/brut, pachete)**")
    st.dataframe(
        by_route_fmt.rename(
            columns={"route": "tură/oraș", "fuel_l": "litri", "fuel_cost_net": "cost (net)", "fuel_cost_gross": "cost (brut)"}
        ),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Chart top orașe după consum (exclude PREDICT)
    city_df = df[(df["fuel_l"] > 0) & (df["route"].notnull())].copy()
    city_df["route"] = city_df["route"].astype(str)
    city_top = city_df.groupby("route", as_index=False)["fuel_l"].sum().sort_values("fuel_l", ascending=False).head(10)
    if not city_top.empty:
        st.markdown('<div class="card"><div class="section-h">3) Top orașe după consum (L)</div>', unsafe_allow_html=True)
        fig = px.bar(
            city_top,
            x="route",
            y="fuel_l",
            labels={"route": "oraș / rută", "fuel_l": "litri"},
            text=[de_thousands_int(v) for v in city_top["fuel_l"]],
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_title=None, xaxis_title=None, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------- RUN ----------
if __name__ == "__main__":
    main()

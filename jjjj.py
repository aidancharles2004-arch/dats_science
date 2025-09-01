# app.py
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import joblib
import requests

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# -----------------------------
# Config / file paths
# -----------------------------
MODEL_PATH = "aidan.joblib"
# Hapa weka link sahihi ya direct download kutoka GitHub Release
GITHUB_RELEASE_MODEL_URL = "https://github.com/aidancharles2004-arch/dats_science/releases/download/v1.0/aidan.joblib"

# Pakua model kama haipo tayari
if not os.path.exists(MODEL_PATH):
    r = requests.get(GITHUB_RELEASE_MODEL_URL, allow_redirects=True)
    if r.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
    else:
        st.error(f"Failed to download model. Status code: {r.status_code}")

# Load ML model
try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model_loaded = False

STORAGE_FILE = "expenses_storage.csv"
HISTORICAL_SAMPLE = "group_7.csv"

CATEGORIES = [
    "Rent", "Loan_Repayment", "Insurance", "Groceries", "Transport",
    "Eating_Out", "Entertainment", "Utilities", "Healthcare", "Education", "Miscellaneous"
]

PRED_TARGETS = CATEGORIES

# -----------------------------
# Helpers
# -----------------------------
def ensure_storage():
    if not os.path.exists(STORAGE_FILE):
        cols = ["Fake_date", "Income", "Age", "Dependents", "Occupation", "City_Tier"] + CATEGORIES + ["Total_Expenses", "Disposable_Income"]
        pd.DataFrame(columns=cols).to_csv(STORAGE_FILE, index=False)

def load_storage():
    ensure_storage()
    df = pd.read_csv(STORAGE_FILE, parse_dates=["Fake_date"])
    return df

def save_entry(row: dict):
    df = load_storage()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(STORAGE_FILE, index=False)

def compute_total(row):
    return sum([row.get(cat, 0) for cat in CATEGORIES])

def get_week_df(df, end_date=None, days=7):
    if df.empty:
        return df
    if end_date is None:
        end_date = pd.Timestamp.today().normalize()
    start = end_date - pd.Timedelta(days=days-1)
    mask = (df["Fake_date"] >= start) & (df["Fake_date"] <= end_date)
    return df.loc[mask].copy()

def anomaly_iqr(series, k=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or np.isnan(iqr):
        return pd.Series([False]*len(series), index=series.index)
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (series < lower) | (series > upper)

# -----------------------------
# Language dictionary
# -----------------------------
TEXTS = {
    "EN": {
        "title": " Personal Expense Analyser & Savings Predictor - Weekly Analysis, Anomalies & Prediction",
        "subtitle": "Enter your daily expenses, view weekly summary, detect anomalies, and get next-week predictions.",
        "sidebar_header": " Add / Save Today's Expenses",
        "date": "Date",
        "income": "Income (total for period)",
        "age": "Age",
        "dependents": "Dependents",
        "occupation": "Occupation",
        "city": "City Tier",
        "save": "Save Entry",
        "saved": "✅ Entry saved to local CSV.",
        "summary": "📊 Weekly Summary & Trends",
        "no_data": "No data yet. Please add expenses in sidebar or upload 'group_7.csv'.",
        "days_in_storage": "Days in storage",
        "entries_this_week": "Entries this week",
        "week_range": "Week range",
        "week_totals": "Week totals",
        "total_exp": "Total expenses (week)",
        "total_income": "Total income (week)",
        "disposable": "Disposable after expenses (week)",
        "no_week_entries": "No entries found in selected week.",
        "anomalies": " Anomaly detection (this week)",
        "no_anomalies": "No anomalies detected this week ✅",
        "predictions": " Next week prediction (per-category totals)",
        "model_source": "Model source",
        "est_disposable": "Estimated disposable for next week (using recent income):",
        "notes": "**Notes / Tips:**",
        "footer": """
        ---
        <div style="text-align: center; font-size:14px; color: gray;">
             Personal Expense Tracker — Built for learning and smart financial planning.<br>
            Made with ❤️ using <b>Python & Streamlit</b>.<br><br>
             <i>Today’s saving is better tomorrow.</i>
        </div>
        """
    },
    "Kiswahili": {
        "title": " Kichambua  Matumizi Binafsi &Kitabiri akiba - Uchambuzi wa Wiki, Anomalies & Utabiri",
        "subtitle": "Weka matumizi yako ya kila siku, ona muhtasari wa wiki, tambua matumizi yasiyo ya kawaida, na upate utabiri wa wiki ijayo.",
        "sidebar_header": " Ongeza / Hifadhi Matumizi ya Leo",
        "date": "Tarehe",
        "income": "Kipato (jumla kwa kipindi)",
        "age": "Umri",
        "dependents": "Wategemezi",
        "occupation": "Kazi",
        "city": "Aina ya Mji",
        "save": "Hifadhi",
        "saved": " Taarifa imehifadhiwa.",
        "summary": "📊 Muhtasari wa Wiki na Mwelekeo",
        "no_data": "Bado hakuna data. Tafadhali ongeza matumizi upande wa kushoto au tumia faili 'group_7.csv'.",
        "days_in_storage": "Siku zilizo kwenye hifadhidata",
        "entries_this_week": "Taarifa za wiki hii",
        "week_range": "Kipindi cha wiki",
        "week_totals": "Jumla za wiki",
        "total_exp": "Matumizi jumla (wiki)",
        "total_income": "Kipato jumla (wiki)",
        "disposable": "Salio baada ya matumizi (wiki)",
        "no_week_entries": "Hakuna taarifa kwa wiki hiyo.",
        "anomalies": " Utambuzi wa matumizi yasiyo ya kawaida (wiki hii)",
        "no_anomalies": "Hakuna matumizi ya ajabu wiki hii ✅",
        "predictions": " Utabiri wa wiki ijayo (jumla kwa kila kipengele)",
        "model_source": "Chanzo cha mfano",
        "est_disposable": "Salio linalokadiriwa wiki ijayo (kwa kipato cha karibuni):",
        "notes": "**Vidokezo / Ushauri:**",
        "footer": """
        ---
        <div style="text-align: center; font-size:14px; color: gray;">
             Dashibodi ya Matumizi Binafsi — Imetengenezwa kwa ajili ya kujifunza na kupanga fedha kwa busara.<br>
            Imetengenezwa kwa ❤️ kwa kutumia <b>Python ,datascience concepts ,& Streamlit</b>.<br><br>
             <i>Kuhifadhi leo ni bora kwa kesho.</i>
        </div>
        """
    }
}

# -----------------------------
# Language toggle
# -----------------------------
LANG = st.sidebar.radio("🌐 Language / Lugha", ["EN", "Kiswahili"])
T = TEXTS[LANG]

# -----------------------------
# UI: Page Config & CSS for header and background
# -----------------------------
st.set_page_config(page_title="Personal Expense Dashboard", layout="wide")
st.markdown(
    f"""
    <style>
        body {{
            background-color: ghostwhite;
        }}
        .stApp .css-1v3fvcr h1 {{
            margin-top: 0rem !important;
            text-align: center !important;
        }}
        .stApp .css-1v3fvcr p {{
            text-align: center !important;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Load storage
# -----------------------------
df_storage = load_storage()

# --- Hapa code yako yote nyingine inaendelea bila kuondoa kitu ---
# Unahitaji tu kuhakikisha MODEL inaload kutoka GitHub Release direct download

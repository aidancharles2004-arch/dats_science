# app.py
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# -----------------------------
# Config / file paths
# -----------------------------
STORAGE_FILE = "expenses_storage.csv"
HISTORICAL_SAMPLE = "group_7.csv"
MODEL_PATH = "aidan.joblib"

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
# Load ML model if exists
# -----------------------------
try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except:
    model_loaded = False

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
st.title(T["title"])
st.markdown(T["subtitle"])

# -----------------------------
# Load storage
# -----------------------------
df_storage = load_storage()

# -----------------------------
# Left: Data entry form
# -----------------------------
st.sidebar.header(T["sidebar_header"])
with st.sidebar.form("entry_form", clear_on_submit=True):
    date = st.date_input(T["date"], value=datetime.date.today())
    income = st.number_input(T["income"], min_value=0, value=0, step=1000)
    age = st.number_input(T["age"], min_value=0, value=30, step=1)
    dependents = st.number_input(T["dependents"], min_value=0, value=0, step=1)
    occupation = st.selectbox(T["occupation"], options=["Salaried", "Self_Employed", "Student", "Retired", "Other"])
    city_tier = st.selectbox(T["city"], options=["Tier_1", "Tier_2", "Tier_3"])

    cat_inputs = {}
    for cat in CATEGORIES:
        cat_inputs[cat] = st.number_input(cat, min_value=0, value=0, step=100)

    submitted = st.form_submit_button(T["save"])
    if submitted:
        total_exp = compute_total(cat_inputs)
        disposable = income - total_exp
        row = {
            "Fake_date": pd.Timestamp(date),
            "Income": income,
            "Age": age,
            "Dependents": dependents,
            "Occupation": occupation,
            "City_Tier": city_tier,
            **cat_inputs,
            "Total_Expenses": total_exp,
            "Disposable_Income": disposable
        }
        save_entry(row)
        st.success(T["saved"])
        df_storage = load_storage()

# -----------------------------
# Right / Main: show summaries
# -----------------------------
st.header(T["summary"])

if df_storage.empty:
    st.info(T["no_data"])
else:
    df_storage["Fake_date"] = pd.to_datetime(df_storage["Fake_date"])
    end_date = st.date_input("Week end date (choose)", value=datetime.date.today())
    end_date = pd.Timestamp(end_date)
    week_df = get_week_df(df_storage, end_date=end_date, days=7)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(T["days_in_storage"], len(df_storage))
    with col2:
        st.metric(T["entries_this_week"], len(week_df))
    with col3:
        st.metric(T["week_range"], f"{(end_date - pd.Timedelta(days=6)).date()} → {end_date.date()}")

    if week_df.empty:
        st.warning(T["no_week_entries"])
    else:
        weekly_totals = week_df[CATEGORIES].sum().astype(float)
        total_week_exp = weekly_totals.sum()
        total_income_week = week_df["Income"].sum()
        disposable_week = total_income_week - total_week_exp

        st.subheader(T["week_totals"])
        st.write(f"{T['total_exp']}: **{total_week_exp:,.0f}**")
        st.write(f"{T['total_income']}: **{total_income_week:,.0f}**")
        st.write(f"{T['disposable']}: **{disposable_week:,.0f}**")
        st.bar_chart(weekly_totals)
        daily = week_df.set_index("Fake_date")[CATEGORIES + ["Total_Expenses"]].resample("D").sum().fillna(0)
        st.line_chart(daily["Total_Expenses"], use_container_width=True)

        # -----------------------------
        # Anomaly detection
        # -----------------------------
        st.subheader(T["anomalies"])
        anomalies_report = []

        hist_span_days = 60
        hist_start = end_date - pd.Timedelta(days=hist_span_days - 1)
        hist_df = df_storage[df_storage["Fake_date"] >= hist_start]

        for cat in CATEGORIES + ["Total_Expenses"]:
            series_hist = hist_df[cat] if not hist_df.empty else df_storage[cat]
            week_series = week_df[cat]
            hist_mean = series_hist.mean()
            hist_std = series_hist.std(ddof=0)
            if hist_std and not np.isnan(hist_std):
                z_flags = ((week_series - hist_mean).abs() / hist_std) > 2.7
            else:
                z_flags = pd.Series([False]*len(week_series), index=week_series.index)
            iqr_flags = anomaly_iqr(pd.concat([series_hist, week_series]).tail(len(week_series)))
            flags = z_flags | iqr_flags
            flagged_days = week_series[flags]
            if not flagged_days.empty:
                for idx, val in flagged_days.items():
                    anomalies_report.append({
                        "category": cat,
                        "date": pd.to_datetime(idx).date(),
                        "amount": val
                    })

        if not anomalies_report:
            st.success(T["no_anomalies"])
        else:
            st.warning(f"{len(anomalies_report)} anomaly(s) found:")
            st.table(pd.DataFrame(anomalies_report))

        # -----------------------------
        # Prediction
        # -----------------------------
        st.subheader(T["predictions"])

        def heuristic_next_week(df_all, days=14):
            recent = df_all.set_index("Fake_date").last(f"{days}D").resample("D").sum().fillna(0)
            means = recent[CATEGORIES].mean()
            return (means * 7).to_dict()

        predictions = {}
        model_source = "heuristic"

        if model_loaded and len(df_storage) >= 30:
            X_recent = df_storage[CATEGORIES + ["Income", "Age", "Dependents"]].tail(1)
            y_pred = model.predict(X_recent)
            predictions = dict(zip(CATEGORIES, y_pred.flatten()))
            model_source = "aidan.joblib"
        else:
            predictions = heuristic_next_week(df_storage)

        st.write(f"{T['model_source']}: **{model_source}**")
        pred_series = pd.Series(predictions)
        st.table(pred_series.rename("Next-week predicted total"))
        st.bar_chart(pred_series)

        predicted_week_total = pred_series.sum()
        week_income_guess = week_df["Income"].sum() if not week_df.empty else income
        predicted_disposable = week_income_guess - predicted_week_total
        st.markdown(f"**{T['est_disposable']} {predicted_disposable:,.0f}**")


st.markdown("---")
st.markdown(T["notes"])
st.markdown(T["footer"], unsafe_allow_html=True)



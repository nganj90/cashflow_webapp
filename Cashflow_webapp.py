import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# -------------------------
# DATABASE SETUP
# -------------------------
# The database file will be stored in the same folder as your app.
db_path = "RenovationCashflow.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS payments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        payment_amount REAL
    )
""")
conn.commit()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS milestones (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        milestone_name TEXT,
        expected_date TEXT,
        expected_percentage REAL
    )
""")
conn.commit()

# -------------------------
# DATABASE FUNCTIONS
# -------------------------
def add_payment(date, amount):
    cursor.execute("INSERT INTO payments (date, payment_amount) VALUES (?, ?)", (date, amount))
    conn.commit()

def get_payments():
    cursor.execute("SELECT id, date, payment_amount FROM payments ORDER BY date")
    return cursor.fetchall()

def delete_payment(payment_id):
    cursor.execute("DELETE FROM payments WHERE id = ?", (payment_id,))
    conn.commit()

def add_milestone(name, expected_date, expected_percentage):
    cursor.execute("INSERT INTO milestones (milestone_name, expected_date, expected_percentage) VALUES (?, ?, ?)",
                   (name, expected_date, expected_percentage))
    conn.commit()

def get_milestones():
    cursor.execute("SELECT id, milestone_name, expected_date, expected_percentage FROM milestones ORDER BY expected_date")
    return cursor.fetchall()

def delete_milestone(milestone_id):
    cursor.execute("DELETE FROM milestones WHERE id = ?", (milestone_id,))
    conn.commit()

# -------------------------
# CALCULATION FUNCTIONS
# -------------------------
def forecast_net(t, contract_sum, k=10):
    # Compute net forecast (S-curve) with retention cap.
    f = 1 / (1 + np.exp(-k * (t - 0.5)))
    gross = contract_sum * f
    retention = gross * 0.1 if f < 0.5 else min(gross * 0.1, contract_sum * 0.05)
    return gross - retention

def create_cash_flow_chart(contract_sum, start_date, end_date):
    # Fetch payments from the database.
    payments = get_payments()
    if payments:
        df = pd.DataFrame(payments, columns=["ID", "Date", "PaymentAmt"])
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df = df.dropna(subset=["Date"])
        df.sort_values("Date", inplace=True)
        df["CumulativeGross"] = df["PaymentAmt"].cumsum()
        cap = contract_sum * 0.05
        df["CumulativeRetention"] = df["CumulativeGross"] * 0.1
        df["EffectiveRetention"] = df["CumulativeRetention"].apply(lambda x: min(x, cap))
        df["CumulativeNetEffective"] = df["CumulativeGross"] - df["EffectiveRetention"]
    else:
        df = pd.DataFrame(columns=["Date", "CumulativeGross", "CumulativeNetEffective"])
    
    # Create forecast S-curve data.
    forecast_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    normalized = np.linspace(0, 1, len(forecast_dates))
    s_forecast = [forecast_net(t, contract_sum, k=10) for t in normalized]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    if not df.empty:
        ax.plot(df["Date"], df["CumulativeGross"], marker="x", linestyle="--", label="Cumulative Gross")
        ax.plot(df["Date"], df["CumulativeNetEffective"], marker="o", linestyle="-", label="Cumulative Net")
    ax.plot(forecast_dates, s_forecast, linestyle="--", label="S-Curve Forecast (Net)")
    
    # Plot forecast claims from milestones.
    milestones = get_milestones()
    if milestones:
        milestone_dates = []
        cumulative = 0.0
        cumulative_values = []
        for m in milestones:
            _, name, date_str, pct = m
            cumulative += pct
            milestone_dates.append(pd.to_datetime(date_str, errors='coerce'))
            gross_forecast = (cumulative / 100) * contract_sum
            retention_forecast = min(gross_forecast * 0.1, contract_sum * 0.05)
            cumulative_forecast = gross_forecast - retention_forecast
            cumulative_values.append(cumulative_forecast)
        ax.step(milestone_dates, cumulative_values, where="post", label="Forecast Claims (Cumulative)")
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Payment ($)")
    ax.set_title("Cash Flow Over Time")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig

def create_delta_analysis(contract_sum, start_date, end_date):
    cursor.execute("SELECT DISTINCT date FROM payments ORDER BY date")
    date_records = cursor.fetchall()
    if not date_records:
        return pd.DataFrame()
    rows = []
    for (d_str,) in date_records:
        current_date = pd.to_datetime(d_str, errors='coerce')
        if current_date is None:
            continue
        cursor.execute("SELECT payment_amount FROM payments WHERE date <= ? ORDER BY date", (d_str,))
        payments = cursor.fetchall()
        gross = sum([p[0] for p in payments]) if payments else 0.0
        retention = min(gross * 0.1, contract_sum * 0.05)
        actual_net = gross - retention

        cursor.execute("SELECT SUM(expected_percentage) FROM milestones WHERE expected_date <= ?", (d_str,))
        result = cursor.fetchone()[0]
        cumulative_pct = result if result is not None else 0.0
        gross_claim = (cumulative_pct / 100) * contract_sum
        retention_claim = min(gross_claim * 0.1, contract_sum * 0.05)
        forecast_claims_net = gross_claim - retention_claim

        difference = actual_net - forecast_claims_net

        # Simplified inversion of the S-curve for schedule difference.
        def forecast_gross(t, contract_sum, k=10):
            f = 1 / (1 + np.exp(-k * (t - 0.5)))
            return contract_sum * f

        def invert_forecast_gross(target, contract_sum, k=10, tol=1e-3):
            lo, hi = 0.0, 1.0
            while hi - lo > tol:
                mid = (lo + hi) / 2
                if forecast_gross(mid, contract_sum, k) < target:
                    lo = mid
                else:
                    hi = mid
            return (lo + hi) / 2

        total_days = (end_date - start_date).days
        t_est = invert_forecast_gross(gross, contract_sum, k=10)
        estimated_date = start_date + pd.Timedelta(days=t_est * total_days)
        schedule_diff = (current_date - estimated_date).days
        rows.append({
            "Payment Date": current_date.strftime("%Y-%m-%d"),
            "Actual Net": f"${actual_net:,.2f}",
            "Forecast Claims Net": f"${forecast_claims_net:,.2f}",
            "Difference": f"${difference:,.2f}",
            "Schedule Diff": f"{schedule_diff} days"
        })
    return pd.DataFrame(rows)

def create_dashboard(contract_sum):
    cursor.execute("SELECT payment_amount FROM payments")
    payments = cursor.fetchall()
    total_gross = sum([p[0] for p in payments]) if payments else 0.0
    retention = min(total_gross * 0.1, contract_sum * 0.05)
    actual_net = total_gross - retention
    s_forecast_final = forecast_net(1, contract_sum, k=10)
    percent_completed = (actual_net / s_forecast_final) * 100 if s_forecast_final else 0
    variance = actual_net - s_forecast_final
    accuracy = (actual_net / s_forecast_final) * 100 if s_forecast_final else 0

    dashboard_data = {
        "Actual Net": f"${actual_net:,.2f}",
        "S-Curve Forecast (Net)": f"${s_forecast_final:,.2f}",
        "Project Completed": f"{percent_completed:.1f}%",
        "Variance": f"${variance:,.2f}",
        "Forecast Accuracy": f"{accuracy:.1f}%"
    }
    return dashboard_data

# -------------------------
# STREAMLIT APP
# -------------------------
st.title("Renovation Cash Flow Tracker")

# Sidebar for navigation and common project settings.
page = st.sidebar.radio("Navigate to", ["Payments & Forecast", "Milestones", "Delta Analysis", "Dashboard"])

st.sidebar.subheader("Project Settings")
contract_sum = st.sidebar.number_input("Total Contract Sum ($):", value=100000)
start_date_input = st.sidebar.text_input("Project Start Date (YYYY-MM-DD):", value="2023-01-01")
end_date_input = st.sidebar.text_input("Project End Date (YYYY-MM-DD):", value="2023-12-31")
try:
    start_date = pd.to_datetime(start_date_input)
    end_date = pd.to_datetime(end_date_input)
except Exception:
    st.sidebar.error("Invalid project dates.")

if page == "Payments & Forecast":
    st.header("Payments & Forecast")
    st.subheader("Add a Payment")
    with st.form("payment_form"):
        payment_date = st.text_input("Payment Date (YYYY-MM-DD)")
        payment_amount = st.number_input("Payment Amount ($):", value=0.0)
        submitted = st.form_submit_button("Add Payment")
        if submitted:
            if payment_date == "":
                st.error("Please enter a payment date.")
            else:
                add_payment(payment_date, payment_amount)
                st.success("Payment added!")
    st.subheader("Existing Payments")
    payments = get_payments()
    if payments:
        df_payments = pd.DataFrame(payments, columns=["ID", "Date", "Payment Amount"])
        st.dataframe(df_payments)
        payment_ids = df_payments["ID"].astype(str).tolist()
        selected_payment = st.selectbox("Select Payment ID to Delete", [""] + payment_ids)
        if selected_payment != "" and st.button("Delete Selected Payment"):
            delete_payment(selected_payment)
            st.success("Payment deleted!")
    else:
        st.write("No payments recorded yet.")
    st.subheader("Cash Flow Chart")
    if st.button("Show Cash Flow Chart"):
        fig = create_cash_flow_chart(contract_sum, start_date, end_date)
        st.pyplot(fig)

elif page == "Milestones":
    st.header("Milestones")
    st.subheader("Add a Milestone")
    with st.form("milestone_form"):
        milestone_name = st.text_input("Milestone Name")
        milestone_date = st.text_input("Expected Date (YYYY-MM-DD)")
        milestone_pct = st.number_input("Expected Percentage (%):", value=0.0)
        submitted = st.form_submit_button("Add Milestone")
        if submitted:
            if milestone_name == "" or milestone_date == "":
                st.error("Please fill out all fields.")
            else:
                add_milestone(milestone_name, milestone_date, milestone_pct)
                st.success("Milestone added!")
    st.subheader("Existing Milestones")
    milestones = get_milestones()
    if milestones:
        df_milestones = pd.DataFrame(milestones, columns=["ID", "Milestone Name", "Expected Date", "Expected Percentage"])
        st.dataframe(df_milestones)
        milestone_ids = df_milestones["ID"].astype(str).tolist()
        selected_milestone = st.selectbox("Select Milestone ID to Delete", [""] + milestone_ids)
        if selected_milestone != "" and st.button("Delete Selected Milestone"):
            delete_milestone(selected_milestone)
            st.success("Milestone deleted!")
    else:
        st.write("No milestones recorded yet.")

elif page == "Delta Analysis":
    st.header("Delta Analysis")
    df_delta = create_delta_analysis(contract_sum, start_date, end_date)
    if not df_delta.empty:
        st.dataframe(df_delta)
    else:
        st.write("No payment data available for analysis.")

elif page == "Dashboard":
    st.header("Dashboard")
    dashboard_data = create_dashboard(contract_sum)
    for key, value in dashboard_data.items():
        st.write(f"**{key}:** {value}")
    st.subheader("Dashboard Chart")
    # Example dashboard chart: a bar chart and a pie chart
    cursor.execute("SELECT payment_amount FROM payments")
    payments = cursor.fetchall()
    total_gross = sum([p[0] for p in payments]) if payments else 0.0
    retention = min(total_gross * 0.1, contract_sum * 0.05)
    actual_net = total_gross - retention
    s_forecast_final = forecast_net(1, contract_sum, k=10)
    percent_completed = (actual_net / s_forecast_final) * 100 if s_forecast_final else 0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(["Actual Net", "Forecast Net"], [actual_net, s_forecast_final], color=["green", "blue"])
    ax1.set_title("Net Payment Comparison")
    ax1.set_ylabel("Amount ($)")
    remainder = 100 - percent_completed
    ax2.pie([percent_completed, remainder],
            labels=[f"Completed {percent_completed:.1f}%", f"Remaining {remainder:.1f}%"],
            autopct="%1.1f%%", colors=["green", "lightgray"])
    ax2.set_title("Project Completion")
    st.pyplot(fig)

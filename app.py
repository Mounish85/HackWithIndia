import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import json
import os
import tempfile
from fpdf import FPDF
from datetime import datetime
import base64

# ----------------------------
# Configuration / Constants
# ----------------------------
DB_FILE = "usage_and_scenarios.db"
COST_PER_SCENARIO = 0.02
COST_PER_REPORT = 0.10
BG_IMAGE = "background2.png"
MAX_REVENUE = 10_000_000
MAX_EXPENSES = 10_000_000
MAX_CASH = 100_000_000
MAX_ENGINEER_COST = 100_000
MAX_SPENDING = 1_000_000
MAX_HIRING = 500
MAX_PRICE_PCT = 200
MIN_PRICE_PCT = -50
MAX_ROAS = 10.0

# ----------------------------
# Database helpers
# ----------------------------
def init_db():
    """
    Initializes SQLite DB with usage and scenario tables.
    """
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS usage (
            id INTEGER PRIMARY KEY,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            type TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS scenarios (
            id INTEGER PRIMARY KEY,
            name TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            inputs_json TEXT,
            results_json TEXT
        )
    ''')
    conn.commit()
    return conn

def increment_usage(conn, usage_type):
    """
    Insert a usage record with type = usage_type.
    """
    c = conn.cursor()
    c.execute("INSERT INTO usage (type) VALUES (?)", (usage_type,))
    conn.commit()

def get_counts(conn):
    """
    Returns usage counts grouped by type.
    """
    c = conn.cursor()
    c.execute("SELECT type, COUNT(*) FROM usage GROUP BY type")
    rows = c.fetchall()
    return {k: v for k, v in rows}

def save_scenario(conn, name, inputs, results):
    """
    Save scenario inputs and results with a scenario name.
    """
    c = conn.cursor()
    c.execute(
        "INSERT INTO scenarios (name, inputs_json, results_json) VALUES (?, ?, ?)",
        (name, json.dumps(inputs), json.dumps(results))
    )
    conn.commit()

def list_scenarios(conn):
    """
    Returns a list of saved scenarios ordered by created_at desc.
    """
    c = conn.cursor()
    c.execute("SELECT id, name, created_at, inputs_json, results_json FROM scenarios ORDER BY created_at DESC")
    rows = c.fetchall()
    scenarios = []
    for r in rows:
        scenarios.append({
            "id": r[0],
            "name": r[1],
            "created_at": r[2],
            "inputs": json.loads(r[3]),
            "results": json.loads(r[4])
        })
    return scenarios

def delete_scenario(conn, scenario_id):
    """
    Delete scenario by id.
    """
    c = conn.cursor()
    c.execute("DELETE FROM scenarios WHERE id=?", (scenario_id,))
    conn.commit()

def reset_usage(conn):
    """
    Clear all usage records.
    """
    c = conn.cursor()
    c.execute("DELETE FROM usage")
    conn.commit()

# ----------------------------
# Pathway data loader with placeholder for live integration
# ----------------------------
def load_pathway():
    """
    Loads mock or live Pathway financial data.
    Falls back to built-in defaults if no file found.
    TODO: Integrate live API or Pathway SDK for real-time data.
    """
    # Placeholder for live data integration
    # Example:
    # try:
    #     data = fetch_live_pathway_data()
    #     return data
    # except Exception:
    #     pass

    if os.path.exists("pathway_mock.json"):
        try:
            with open("pathway_mock.json") as f:
                data = json.load(f)
                data["_last_loaded_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                return data
        except Exception as e:
            st.sidebar.error(f"Error loading pathway_mock.json: {e}")

    # Fallback defaults
    defaults = {
        "scenarios": {
            "startup_tech": {
                "base_revenue": 80000,
                "base_expenses": 60000,
                "cash": 250000,
                "engineer_cost": 7000,
                "notes": "Seed-stage tech startup with 5 engineers and moderate revenue."
            },
            "student_fest": {
                "base_revenue": 20000,
                "base_expenses": 15000,
                "cash": 50000,
                "engineer_cost": 0,
                "notes": "University fest with sponsorship revenue, prize and event expenses."
            },
            "ecommerce_store": {
                "base_revenue": 120000,
                "base_expenses": 90000,
                "cash": 100000,
                "engineer_cost": 4000,
                "notes": "Online retail business; revenue tied to ad spending and discounts."
            },
            "saas_business": {
                "base_revenue": 150000,
                "base_expenses": 80000,
                "cash": 400000,
                "engineer_cost": 6000,
                "notes": "Subscription SaaS with stable recurring revenue and product dev costs."
            },
            "bootstrapped_agency": {
                "base_revenue": 50000,
                "base_expenses": 40000,
                "cash": 30000,
                "engineer_cost": 5000,
                "notes": "Small services agency with tight margins, reliant on client projects."
            }
        }
    }
    return defaults

# ----------------------------
# Simulation logic with input validation
# ----------------------------
def validate_inputs(base_rev, base_exp, cash_on_hand, engineer_cost, spending, hiring, price_pct, roas):
    """Validate input ranges to prevent unrealistic or negative values."""
    if not (0 <= base_rev <= MAX_REVENUE):
        raise ValueError(f"Base revenue must be between 0 and {MAX_REVENUE}")
    if not (0 <= base_exp <= MAX_EXPENSES):
        raise ValueError(f"Base expenses must be between 0 and {MAX_EXPENSES}")
    if not (0 <= cash_on_hand <= MAX_CASH):
        raise ValueError(f"Cash on hand must be between 0 and {MAX_CASH}")
    if not (0 <= engineer_cost <= MAX_ENGINEER_COST):
        raise ValueError(f"Engineer cost must be between 0 and {MAX_ENGINEER_COST}")
    if not (0 <= spending <= MAX_SPENDING):
        raise ValueError(f"Spending must be between 0 and {MAX_SPENDING}")
    if not (0 <= hiring <= MAX_HIRING):
        raise ValueError(f"Hiring count must be between 0 and {MAX_HIRING}")
    if not (MIN_PRICE_PCT <= price_pct <= MAX_PRICE_PCT):
        raise ValueError(f"Price % must be between {MIN_PRICE_PCT} and {MAX_PRICE_PCT}")
    if not (0 <= roas <= MAX_ROAS):
        raise ValueError(f"ROAS must be between 0 and {MAX_ROAS}")

def simulate_scenario(base_rev, base_exp, cash_on_hand, engineer_cost,
                      spending, hiring, price_pct, roas):
    """
    Simulate financial scenario with inputs.
    Returns dict of key metrics including runway and cash projections.
    """
    validate_inputs(base_rev, base_exp, cash_on_hand, engineer_cost, spending, hiring, price_pct, roas)

    rev_from_price = base_rev * (price_pct / 100.0)
    rev_from_marketing = spending * roas
    new_revenue = base_rev + rev_from_price + rev_from_marketing
    new_expenses = base_exp + (hiring * engineer_cost) + spending
    profit = new_revenue - new_expenses
    runway_months = float('inf') if profit >= 0 else (cash_on_hand / -profit if profit != 0 else float('inf'))

    cash_projection = []
    cash = cash_on_hand
    for _ in range(12):
        cash += profit
        cash_projection.append(float(cash))

    return {
        "new_revenue": float(new_revenue),
        "new_expenses": float(new_expenses),
        "profit": float(profit),
        "runway_months": None if runway_months == float('inf') else float(runway_months),
        "cash_projection": cash_projection
    }

# ----------------------------
# PDF report creation - unchanged but added docstring
# ----------------------------
def create_pdf_report(scenario_name, scenario_inputs, results):
    """
    Create a PDF report with charts and summary text for a scenario.
    Returns report as bytes for download.
    """
    tmp_files = []
    try:
        # Chart 1: revenue vs expenses
        fig1, ax1 = plt.subplots()
        ax1.bar(["Revenue", "Expenses"], [results["new_revenue"], results["new_expenses"]])
        ax1.set_title("Monthly Revenue vs Expenses")
        ax1.set_ylabel("Amount")
        tmp1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig1.savefig(tmp1.name, bbox_inches="tight")
        plt.close(fig1)
        tmp_files.append(tmp1.name)

        # Chart 2: cash projection
        fig2, ax2 = plt.subplots()
        ax2.plot(range(1, len(results["cash_projection"])+1), results["cash_projection"], marker="o")
        ax2.set_title("12-month Cash Projection")
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Cash on Hand")
        tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig2.savefig(tmp2.name, bbox_inches="tight")
        plt.close(fig2)
        tmp_files.append(tmp2.name)

        # Build PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "CFO Helper — Scenario Report", ln=True, align="C")
        pdf.ln(6)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 6, f"Scenario name: {scenario_name}")
        pdf.multi_cell(0, 6, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        pdf.ln(4)

        # Inputs table
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Inputs", ln=True)
        pdf.set_font("Arial", size=10)
        for k, v in scenario_inputs.items():
            pdf.cell(70, 6, str(k), border=1)
            pdf.cell(0, 6, str(v), border=1, ln=True)
        pdf.ln(4)

        # Results table
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Results", ln=True)
        pdf.set_font("Arial", size=10)
        results_rows = [
            ("New revenue", f"{results['new_revenue']:.2f}"),
            ("New expenses", f"{results['new_expenses']:.2f}"),
            ("Monthly profit", f"{results['profit']:.2f}"),
            ("Runway (months)", "∞" if results.get("runway_months") is None else f"{results['runway_months']:.1f}")
        ]
        for label, val in results_rows:
            pdf.cell(70, 6, label, border=1)
            pdf.cell(0, 6, val, border=1, ln=True)
        pdf.ln(4)

        # CFO narrative
        pdf.set_font("Arial", "I", 10)
        summary_text = ("The business is profitable and cash position will grow month on month."
                        if results.get("runway_months") is None
                        else f"Monthly burn implies cash lasts approx {results['runway_months']:.1f} months.")
        pdf.multi_cell(0, 6, f"CFO Note: {summary_text}")
        pdf.ln(4)

        # Charts
        pdf.add_page()
        for fpath in tmp_files:
            try:
                pdf.image(fpath, x=10, w=190, type='PNG')
                pdf.ln(6)
            except Exception as e:
                pdf.set_font("Arial", size=10)
                pdf.multi_cell(0, 6, f"(Failed to embed chart: {e})")

        pdf.set_y(-20)
        pdf.set_font("Arial", size=8)
        pdf.cell(0, 6, "Powered by CFO Helper — Hackathon Edition", align="C")
        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        return pdf_bytes
    finally:
        for f in tmp_files:
            try:
                os.remove(f)
            except:
                pass

# ----------------------------
# Background helper - unchanged
# ----------------------------
def set_bg(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="CFO Helper", layout="wide")
set_bg(BG_IMAGE)
st.title("CFO Helper — Finance What-if Simulator")

conn = init_db()
pathway = load_pathway()

# ----------------------------
# Sidebar: Inputs & Usage
# ----------------------------
with st.sidebar:
    st.header("Scenario Inputs")
    # Bind to Pathway defaults if available (simple flatten)
    base_rev = st.number_input("Base monthly revenue",
                               value=pathway.get("base_revenue", 50000),
                               step=1000, format="%d",
                               min_value=0, max_value=MAX_REVENUE)
    base_exp = st.number_input("Base monthly expenses",
                              value=pathway.get("base_expenses", 30000),
                              step=1000, format="%d",
                              min_value=0, max_value=MAX_EXPENSES)
    cash_on_hand = st.number_input("Cash on hand",
                                  value=pathway.get("cash", 100000),
                                  step=1000, format="%d",
                                  min_value=0, max_value=MAX_CASH)
    engineer_cost = st.number_input("Monthly cost per engineer",
                                   value=pathway.get("engineer_cost", 5000),
                                   step=100,
                                   min_value=0, max_value=MAX_ENGINEER_COST)

    st.markdown("---")
    spending = st.slider("Extra monthly spending (marketing/prizes)", 0, MAX_SPENDING, 10000, step=500)
    hiring = st.slider("Hire engineers (count)", 0, MAX_HIRING, 2, step=1)
    price_pct = st.slider("Price change (%)", MIN_PRICE_PCT, MAX_PRICE_PCT, 10, step=1)
    roas = st.slider("Marketing ROAS (revenue per 1 unit spend)", 0.0, MAX_ROAS, 1.5, step=0.1)

    st.markdown("---")
    if st.button("Pull latest financials (Pathway)"):
        pathway = load_pathway()
        st.success("Reloaded pathway_mock.json (if present).")
    st.caption(f"Pathway last loaded: {pathway.get('_last_loaded_at')}")

    st.markdown("---")
    st.subheader("Billing & Usage")
    counts = get_counts(conn)
    scenarios_tested = counts.get("scenario", 0)
    reports_exported = counts.get("report", 0)
    total_bill = scenarios_tested * COST_PER_SCENARIO + reports_exported * COST_PER_REPORT
    st.write(f"Scenarios tested: **{scenarios_tested}**")
    st.write(f"Reports exported: **{reports_exported}**")
    st.write(f"Estimated bill: **{total_bill:.2f}**")

    st.markdown("---")
    st.subheader("Saved scenarios")
    scenario_name_input = st.text_input("Save scenario as (name)", value="")
    if st.button("Save current scenario"):
        scenario_inputs = {
            "base_revenue": base_rev,
            "base_expenses": base_exp,
            "cash_on_hand": cash_on_hand,
            "engineer_cost": engineer_cost,
            "spending": spending,
            "hiring": hiring,
            "price_%": price_pct,
            "marketing_ROAS": roas
        }
        try:
            results = simulate_scenario(base_rev, base_exp, cash_on_hand, engineer_cost, spending, hiring, price_pct, roas)
            save_scenario(conn, scenario_name_input or f"Scenario_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}", scenario_inputs, results)
            st.success("Scenario saved.")
        except ValueError as ve:
            st.error(f"Input error: {ve}")

    if st.button("Reset usage (admin)"):
        reset_usage(conn)
        st.warning("Usage counts reset.")

# ----------------------------
# Main: Simulation & Report
# ----------------------------
col1, col2 = st.columns([2,1])
with col1:
    if st.button("Simulate"):
        try:
            results = simulate_scenario(base_rev, base_exp, cash_on_hand, engineer_cost, spending, hiring, price_pct, roas)
            increment_usage(conn, "scenario")

            st.markdown("### Key metrics")
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Revenue", f"{results['new_revenue']:.0f}")
            mcol2.metric("Expenses", f"{results['new_expenses']:.0f}")
            mcol3.metric("Profit", f"{results['profit']:.0f}")
            mcol4.metric("Runway (months)", "∞" if results.get('runway_months') is None else f"{results['runway_months']:.1f}")

            st.markdown("### Charts")
            fig1, ax1 = plt.subplots()
            ax1.bar(["Revenue", "Expenses"], [results["new_revenue"], results["new_expenses"]])
            ax1.set_title("Revenue vs Expenses")
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            ax2.plot(range(1, len(results["cash_projection"])+1), results["cash_projection"], marker="o")
            ax2.set_title("12-month Cash Projection")
            ax2.set_xlabel("Month")
            ax2.set_ylabel("Cash on Hand")
            st.pyplot(fig2)

            st.markdown("### Report")
            report_name = st.text_input("Report name", value=f"Scenario_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}")
            if st.button("Export PDF Report"):
                scenario_inputs = {
                    "base_revenue": base_rev,
                    "base_expenses": base_exp,
                    "cash_on_hand": cash_on_hand,
                    "engineer_cost": engineer_cost,
                    "spending": spending,
                    "hiring": hiring,
                    "price_%": price_pct,
                    "marketing_ROAS": roas
                }
                pdf_bytes = create_pdf_report(report_name, scenario_inputs, results)
                increment_usage(conn, "report")
                st.download_button("Download PDF", data=pdf_bytes, file_name=f"cfo_report_{report_name}.pdf", mime="application/pdf")
        except ValueError as ve:
            st.error(f"Input error: {ve}")

with col2:
    st.markdown("### Live Billing Dashboard")
    counts = get_counts(conn)
    scenarios_tested = counts.get("scenario", 0)
    reports_exported = counts.get("report", 0)
    total_bill = scenarios_tested * COST_PER_SCENARIO + reports_exported * COST_PER_REPORT
    st.metric("Scenarios tested", scenarios_tested)
    st.metric("Reports exported", reports_exported)
    st.metric("Estimated bill", f"{total_bill:.2f}")

# ----------------------------
# Saved Scenarios Management
# ----------------------------
st.markdown("---")
st.header("Saved scenarios")
scenarios = list_scenarios(conn)
if scenarios:
    df_rows = []
    for s in scenarios:
        df_rows.append({
            "id": s["id"],
            "name": s["name"],
            "created_at": s["created_at"],
            "profit": s["results"]["profit"],
            "runway": ("∞" if s["results"].get("runway_months") is None else f"{s['results']['runway_months']:.1f}")
        })
    st.dataframe(pd.DataFrame(df_rows))
else:
    st.info("No saved scenarios yet.")

# ----------------------------
# Pathway Data Expander
# ----------------------------
st.markdown("---")
with st.expander("Pathway mock financial data"):
    st.json(pathway)
    st.caption(f"Last loaded at: {pathway.get('_last_loaded_at')}")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import Dict, Tuple, List

# Set page config
st.set_page_config(
    page_title="Business Acquisition Financial Model",
    page_icon="📊",
    layout="wide"
)

@dataclass
class Assumptions:
    # Deal & Baseline
    purchase_price: float = 4_200_000.0
    starting_revenue: float = 5_736_386.0
    starting_ebitda: float = 1_410_581.0
    years: int = 10

    # Growth & Margins
    revenue_growth: float = 0.03
    ebitda_margin_delta_per_year: float = 0.0

    # Working capital
    working_capital_pct_of_revenue: float = 0.12

    # Capex & D&A
    capex_pct_of_revenue: float = 0.02
    depreciation_pct_of_revenue: float = 0.015

    # Taxes
    tax_rate: float = 0.26

    # Financing mix
    down_payment_pct: float = 0.15
    bank_debt_pct: float = 0.60
    seller_note_pct: float = 0.25

    # Debt terms
    bank_interest_rate: float = 0.09
    bank_amort_years: int = 7
    seller_interest_rate: float = 0.06
    seller_amort_years: int = 10
    
    # Other
    ramp_months_no_owner_salary: int = 0
    owner_salary_per_year: float = 0.0

def amortization_schedule(principal: float, annual_rate: float, years: int) -> pd.DataFrame:
    """Create an annual amortization schedule with fixed payments."""
    if years <= 0 or principal <= 0:
        cols = ["Year", "BegBal", "Payment", "Interest", "Principal", "EndBal"]
        return pd.DataFrame(columns=cols)
    
    n = years
    r = annual_rate
    if r == 0:
        payment = principal / n
    else:
        payment = principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)

    rows = []
    beg = principal
    for t in range(1, n + 1):
        interest = beg * r
        principal_paid = min(payment - interest, beg)
        end_bal = max(0.0, beg - principal_paid)
        rows.append([t, beg, payment, interest, principal_paid, end_bal])
        beg = end_bal
    
    return pd.DataFrame(rows, columns=["Year", "BegBal", "Payment", "Interest", "Principal", "EndBal"])

def run_model(a: Assumptions) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    # Initial checks
    total_pct = a.down_payment_pct + a.bank_debt_pct + a.seller_note_pct
    if abs(total_pct - 1.0) > 1e-6:
        raise ValueError("Financing mix must sum to 100%.")

    # Baselines
    ebitda_margin0 = a.starting_ebitda / a.starting_revenue if a.starting_revenue > 0 else 0.0

    # Debt sizing
    down = a.purchase_price * a.down_payment_pct
    bank_principal = a.purchase_price * a.bank_debt_pct
    seller_principal = a.purchase_price * a.seller_note_pct

    bank_sched = amortization_schedule(bank_principal, a.bank_interest_rate, a.bank_amort_years)
    seller_sched = amortization_schedule(seller_principal, a.seller_interest_rate, a.seller_amort_years)

    years = np.arange(1, a.years + 1)
    revenue = np.zeros(a.years)
    ebitda = np.zeros(a.years)
    dep = np.zeros(a.years)
    ebit = np.zeros(a.years)
    bank_interest = np.zeros(a.years)
    seller_interest = np.zeros(a.years)
    bank_principal_pay = np.zeros(a.years)
    seller_principal_pay = np.zeros(a.years)
    taxes = np.zeros(a.years)
    capex = np.zeros(a.years)
    wc_level = np.zeros(a.years + 1)
    fcf_equity = np.zeros(a.years)
    dscr = np.zeros(a.years)

    # Revenue and margin trajectory
    for t in range(a.years):
        if t == 0:
            revenue[t] = a.starting_revenue * (1 + a.revenue_growth)
        else:
            revenue[t] = revenue[t - 1] * (1 + a.revenue_growth)

        margin_t = ebitda_margin0 + (t + 1) * a.ebitda_margin_delta_per_year
        ebitda[t] = revenue[t] * max(0.0, margin_t)

        # Owner salary
        if a.owner_salary_per_year > 0:
            ebitda[t] -= a.owner_salary_per_year if (t * 12) >= a.ramp_months_no_owner_salary else 0.0

        dep[t] = revenue[t] * a.depreciation_pct_of_revenue
        ebit[t] = ebitda[t] - dep[t]
        capex[t] = revenue[t] * a.capex_pct_of_revenue

        # Working capital
        next_rev = revenue[t] if t == a.years - 1 else revenue[t + 1]
        target_wc = a.working_capital_pct_of_revenue * next_rev
        if t == 0:
            wc_level[0] = target_wc
        wc_level[t + 1] = target_wc
        wc_delta = wc_level[t + 1] - wc_level[t]

        # Debt service
        if t < len(bank_sched):
            bank_interest[t] = bank_sched.loc[t, "Interest"]
            bank_principal_pay[t] = bank_sched.loc[t, "Principal"]
        else:
            bank_interest[t] = 0.0
            bank_principal_pay[t] = 0.0

        if t < len(seller_sched):
            seller_interest[t] = seller_sched.loc[t, "Interest"]
            seller_principal_pay[t] = seller_sched.loc[t, "Principal"]
        else:
            seller_interest[t] = 0.0
            seller_principal_pay[t] = 0.0

        interest_total = bank_interest[t] + seller_interest[t]
        ebt = ebit[t] - interest_total
        taxes[t] = max(0.0, ebt * a.tax_rate)

        # FCFE
        fcf_equity[t] = (
            ebitda[t]
            - taxes[t]
            - capex[t]
            - wc_delta
            - bank_interest[t] - bank_principal_pay[t]
            - seller_interest[t] - seller_principal_pay[t]
        )

        debt_service = bank_interest[t] + bank_principal_pay[t] + seller_interest[t] + seller_principal_pay[t]
        dscr[t] = (ebitda[t] / debt_service) if debt_service > 0 else np.nan

    out = pd.DataFrame({
        "Year": years,
        "Revenue": revenue,
        "EBITDA": ebitda,
        "Depreciation": dep,
        "EBIT": ebit,
        "Taxes": taxes,
        "CapEx": capex,
        "WC_Level": wc_level[1:],
        "Bank_Int": bank_interest,
        "Bank_Principal": bank_principal_pay,
        "Seller_Int": seller_interest,
        "Seller_Principal": seller_principal_pay,
        "FCF_to_Equity": fcf_equity,
        "DSCR": dscr,
        "Bank_EndBal": [bank_sched.loc[i, "EndBal"] if i < len(bank_sched) else 0.0 for i in range(len(years))],
        "Seller_EndBal": [seller_sched.loc[i, "EndBal"] if i < len(seller_sched) else 0.0 for i in range(len(years))],
    })

    details = {
        "bank_schedule": bank_sched,
        "seller_schedule": seller_sched,
        "assumptions": pd.DataFrame([a.__dict__]),
    }
    return out, details

def irr(cash_flows: List[float]) -> float:
    """Compute IRR; returns np.nan if cannot be found."""
    try:
        # Simple secant method
        def npv(rate):
            return sum(cf / ((1 + rate) ** i) for i, cf in enumerate(cash_flows))
        
        r0, r1 = 0.1, 0.2
        for _ in range(200):
            f0, f1 = npv(r0), npv(r1)
            if abs(f1 - f0) < 1e-10:
                break
            r2 = r1 - f1 * (r1 - r0) / (f1 - f0)
            if not np.isfinite(r2):
                break
            r0, r1 = r1, r2
        return r1 if np.isfinite(r1) else np.nan
    except:
        return np.nan

def equity_cash_flows(out: pd.DataFrame, a: Assumptions) -> List[float]:
    init_equity = -a.purchase_price * a.down_payment_pct
    return [init_equity] + out["FCF_to_Equity"].tolist()

def run_scenarios(base: Assumptions) -> Dict[str, pd.DataFrame]:
    """Run Conservative / Base / Aggressive scenarios."""
    scenarios = {}

    # Conservative
    cons = Assumptions(**{**base.__dict__})
    cons.revenue_growth = max(0.0, base.revenue_growth - 0.02)
    cons.ebitda_margin_delta_per_year = base.ebitda_margin_delta_per_year - 0.002
    scenarios["Conservative"], _ = run_model(cons)

    # Base
    scenarios["Base"], _ = run_model(base)

    # Aggressive
    agg = Assumptions(**{**base.__dict__})
    agg.revenue_growth = base.revenue_growth + 0.03
    agg.ebitda_margin_delta_per_year = base.ebitda_margin_delta_per_year + 0.003
    scenarios["Aggressive"], _ = run_model(agg)

    return scenarios

def main():
    st.title("📊 Business Acquisition Financial Model")
    st.markdown("---")
    
    # Sidebar for inputs
    st.sidebar.header("Model Assumptions")
    
    # Deal & Baseline
    st.sidebar.subheader("Deal Structure")
    purchase_price = st.sidebar.number_input("Purchase Price ($)", value=4_200_000, step=100_000, format="%d")
    starting_revenue = st.sidebar.number_input("Starting Revenue ($)", value=5_736_386, step=100_000, format="%d")
    starting_ebitda = st.sidebar.number_input("Starting EBITDA ($)", value=1_410_581, step=50_000, format="%d")
    years = st.sidebar.slider("Projection Years", 5, 15, 10)
    
    # Growth & Margins
    st.sidebar.subheader("Growth & Operations")
    revenue_growth = st.sidebar.slider("Annual Revenue Growth (%)", -5.0, 15.0, 3.0, 0.5) / 100
    ebitda_margin_delta = st.sidebar.slider("Annual EBITDA Margin Change (bps)", -100, 100, 0, 10) / 10000
    
    # Working Capital & Capex
    wc_pct = st.sidebar.slider("Working Capital (% of Revenue)", 0.0, 25.0, 12.0, 1.0) / 100
    capex_pct = st.sidebar.slider("CapEx (% of Revenue)", 0.0, 10.0, 2.0, 0.5) / 100
    depreciation_pct = st.sidebar.slider("Depreciation (% of Revenue)", 0.0, 5.0, 1.5, 0.1) / 100
    
    # Taxes
    tax_rate = st.sidebar.slider("Tax Rate (%)", 15.0, 40.0, 26.0, 1.0) / 100
    
    # Financing
    st.sidebar.subheader("Financing Structure")
    down_pct = st.sidebar.slider("Down Payment (%)", 10.0, 50.0, 15.0, 5.0) / 100
    bank_debt_pct = st.sidebar.slider("Bank Debt (%)", 30.0, 80.0, 60.0, 5.0) / 100
    seller_note_pct = st.sidebar.slider("Seller Note (%)", 0.0, 40.0, 25.0, 5.0) / 100
    
    # Check financing mix
    total_financing = down_pct + bank_debt_pct + seller_note_pct
    if abs(total_financing - 1.0) > 0.01:
        st.sidebar.error(f"⚠️ Financing mix totals {total_financing:.1%}, must equal 100%")
        return
    
    # Debt terms
    st.sidebar.subheader("Debt Terms")
    bank_rate = st.sidebar.slider("Bank Interest Rate (%)", 5.0, 15.0, 9.0, 0.5) / 100
    bank_years = st.sidebar.slider("Bank Amortization (Years)", 5, 10, 7)
    seller_rate = st.sidebar.slider("Seller Note Rate (%)", 3.0, 10.0, 6.0, 0.5) / 100
    seller_years = st.sidebar.slider("Seller Note Amortization (Years)", 7, 15, 10)
    
    # Owner compensation
    st.sidebar.subheader("Owner Compensation")
    owner_salary = st.sidebar.number_input("Annual Owner Salary ($)", 0, 500_000, 0, 25_000)
    ramp_months = st.sidebar.slider("Months Before Taking Salary", 0, 24, 0, 3)
    
    # Create assumptions object
    assumptions = Assumptions(
        purchase_price=purchase_price,
        starting_revenue=starting_revenue,
        starting_ebitda=starting_ebitda,
        years=years,
        revenue_growth=revenue_growth,
        ebitda_margin_delta_per_year=ebitda_margin_delta,
        working_capital_pct_of_revenue=wc_pct,
        capex_pct_of_revenue=capex_pct,
        depreciation_pct_of_revenue=depreciation_pct,
        tax_rate=tax_rate,
        down_payment_pct=down_pct,
        bank_debt_pct=bank_debt_pct,
        seller_note_pct=seller_note_pct,
        bank_interest_rate=bank_rate,
        bank_amort_years=bank_years,
        seller_interest_rate=seller_rate,
        seller_amort_years=seller_years,
        owner_salary_per_year=owner_salary,
        ramp_months_no_owner_salary=ramp_months
    )
    
    # Run model
    try:
        results, details = run_model(assumptions)
        
        # Calculate IRR
        cash_flows = equity_cash_flows(results, assumptions)
        equity_irr = irr(cash_flows)
        
        # Main results section
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Initial Equity Investment", f"${assumptions.purchase_price * down_pct:,.0f}")
            
        with col2:
            avg_fcf = results["FCF_to_Equity"].mean()
            st.metric("Avg Annual FCF to Equity", f"${avg_fcf:,.0f}")
            
        with col3:
            min_dscr = results["DSCR"].min()
            color = "normal" if min_dscr >= 1.25 else "inverse"
            st.metric("Min DSCR", f"{min_dscr:.2f}x", delta="Safe" if min_dscr >= 1.25 else "Risk", delta_color=color)
            
        with col4:
            if np.isfinite(equity_irr):
                st.metric("Equity IRR (10yr hold)", f"{equity_irr:.1%}")
            else:
                st.metric("Equity IRR", "N/A")
        
        # Charts
        st.markdown("---")
        
        # Revenue and EBITDA
        col1, col2 = st.columns(2)
        
        with col1:
            fig_rev = px.line(results, x="Year", y="Revenue", 
                             title="Revenue Projection",
                             labels={"Revenue": "Revenue ($)"})
            fig_rev.update_traces(line_color="#1f77b4", line_width=3)
            fig_rev.update_layout(yaxis_tickformat="$,.0f")
            st.plotly_chart(fig_rev, use_container_width=True)
            
        with col2:
            fig_ebitda = px.line(results, x="Year", y="EBITDA", 
                                title="EBITDA Projection",
                                labels={"EBITDA": "EBITDA ($)"})
            fig_ebitda.update_traces(line_color="#ff7f0e", line_width=3)
            fig_ebitda.update_layout(yaxis_tickformat="$,.0f")
            st.plotly_chart(fig_ebitda, use_container_width=True)
        
        # FCF and DSCR
        col1, col2 = st.columns(2)
        
        with col1:
            fig_fcf = px.line(results, x="Year", y="FCF_to_Equity", 
                             title="Free Cash Flow to Equity",
                             labels={"FCF_to_Equity": "FCF to Equity ($)"})
            fig_fcf.update_traces(line_color="#2ca02c", line_width=3)
            fig_fcf.update_layout(yaxis_tickformat="$,.0f")
            st.plotly_chart(fig_fcf, use_container_width=True)
            
        with col2:
            fig_dscr = px.line(results, x="Year", y="DSCR", 
                              title="Debt Service Coverage Ratio")
            fig_dscr.add_hline(y=1.25, line_dash="dash", line_color="red", 
                              annotation_text="Safety Line (1.25x)")
            fig_dscr.update_traces(line_color="#d62728", line_width=3)
            fig_dscr.update_layout(yaxis_title="DSCR (x)")
            st.plotly_chart(fig_dscr, use_container_width=True)
        
        # Scenario Analysis
        st.markdown("---")
        st.subheader("📈 Scenario Analysis")
        
        scenarios = run_scenarios(assumptions)
        
        # Create scenario comparison chart
        fig_scenario = go.Figure()
        colors = {"Conservative": "#ff7f0e", "Base": "#1f77b4", "Aggressive": "#2ca02c"}
        
        for name, df in scenarios.items():
            fig_scenario.add_trace(go.Scatter(
                x=df["Year"], 
                y=df["FCF_to_Equity"],
                mode='lines+markers',
                name=name,
                line=dict(color=colors[name], width=3),
                marker=dict(size=6)
            ))
        
        fig_scenario.update_layout(
            title="FCF to Equity by Scenario",
            xaxis_title="Year",
            yaxis_title="FCF to Equity ($)",
            yaxis_tickformat="$,.0f",
            legend=dict(x=0.02, y=0.98),
            height=500
        )
        
        st.plotly_chart(fig_scenario, use_container_width=True)
        
        # Scenario IRRs
        scenario_irrs = {}
        for name, df in scenarios.items():
            cf = equity_cash_flows(df, assumptions)
            scenario_irrs[name] = irr(cf)
        
        col1, col2, col3 = st.columns(3)
        scenarios_list = ["Conservative", "Base", "Aggressive"]
        colors_list = ["orange", "normal", "normal"]
        
        for i, (col, scenario) in enumerate(zip([col1, col2, col3], scenarios_list)):
            with col:
                irr_val = scenario_irrs[scenario]
                if np.isfinite(irr_val):
                    delta_color = colors_list[i] if scenario == "Conservative" else "normal"
                    st.metric(f"{scenario} IRR", f"{irr_val:.1%}", delta_color=delta_color)
                else:
                    st.metric(f"{scenario} IRR", "N/A")
        
        # Detailed Results Table
        st.markdown("---")
        st.subheader("📋 Detailed Projections")
        
        # Format the results for display
        display_df = results.copy()
        currency_cols = ["Revenue", "EBITDA", "EBIT", "Taxes", "CapEx", "WC_Level", 
                        "Bank_Int", "Bank_Principal", "Seller_Int", "Seller_Principal", 
                        "FCF_to_Equity", "Bank_EndBal", "Seller_EndBal"]
        
        for col in currency_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}")
        
        display_df["DSCR"] = results["DSCR"].apply(lambda x: f"{x:.2f}x" if pd.notna(x) else "N/A")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Financing Summary
        st.markdown("---")
        st.subheader("💰 Financing Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Initial Financing Structure:**")
            st.write(f"• Down Payment: ${assumptions.purchase_price * down_pct:,.0f} ({down_pct:.1%})")
            st.write(f"• Bank Debt: ${assumptions.purchase_price * bank_debt_pct:,.0f} ({bank_debt_pct:.1%})")
            st.write(f"• Seller Note: ${assumptions.purchase_price * seller_note_pct:,.0f} ({seller_note_pct:.1%})")
            st.write(f"• **Total: ${assumptions.purchase_price:,.0f}**")
            
        with col2:
            st.write("**Debt Terms:**")
            st.write(f"• Bank: {bank_rate:.1%} interest, {bank_years} year amortization")
            st.write(f"• Seller: {seller_rate:.1%} interest, {seller_years} year amortization")
            
            final_bank = results["Bank_EndBal"].iloc[-1]
            final_seller = results["Seller_EndBal"].iloc[-1]
            st.write(f"• Remaining Bank Debt (Year {years}): ${final_bank:,.0f}")
            st.write(f"• Remaining Seller Debt (Year {years}): ${final_seller:,.0f}")
        
    except ValueError as e:
        st.error(f"Model Error: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()
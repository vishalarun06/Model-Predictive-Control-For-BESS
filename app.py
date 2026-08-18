import streamlit as st
import pandas as pd
import io

# Import your existing class from your script
# Make sure your original file is named Minimising_Capex_EBITDA.py
from Minimising_Capex_EBITDA import BESSSettlementOptimizer

st.set_page_config(page_title="BESS Sizing Optimizer", layout="wide")

st.title("Wind-Solar-BESS Hybrid Optimizer")
st.markdown("Upload your generation/settlement data, adjust the parameters, and run the model to get the optimized schedule.")

# --- SIDEBAR FOR PARAMETERS ---
st.sidebar.header("Model Parameters")

uploaded_file = st.sidebar.file_uploader("Upload Malkapur-Nandurbar Spreadsheet (Excel)", type=["xlsx"])

max_soc = st.sidebar.number_input("Max SoC (%)", value=85)
min_soc = st.sidebar.number_input("Min SoC (%)", value=15)
max_bess_hrs = st.sidebar.number_input("Max BESS Hours", value=12.0)
tariff_caps = st.sidebar.number_input("Tariff Cap (Rs/kWh)", value=5.5)
RTEs = st.sidebar.number_input("Round-Trip Efficiency (%)", value=85)
discharge_degradations = st.sidebar.number_input("Discharge Degradation Per Year (%)", value=1)
cost_escalations = st.sidebar.number_input("Cost Escalation Per Year (%)", value=3)
min_effective_replacements = st.sidebar.number_input("Minimum Effective Replacement (%)", value=70)
solar_capexs = st.sidebar.number_input("Solar CAPEX (INR Mn./Capacity)", value=40)
wind_capexs = st.sidebar.number_input("Wind CAPEX (INR Mn./Capacity)", value=90)
BESS_capexs = st.sidebar.number_input("BESS CAPEX (INR Mn./Capacity)", value=10)
max_grid_windh = st.sidebar.number_input("Maximum Grid Allowable Wind (kWh)", value=49500)
max_grid_solarh = st.sidebar.number_input("Maximum Grid Allowable Solar (kWh)", value=150000)


run_model = st.sidebar.button("Run Optimization", type="primary")

# --- MAIN DASHBOARD AREA ---
if run_model:
    if uploaded_file is None:
        st.error("Please upload the Master Spreadsheet first.")
    else:
        with st.spinner("Solving model... this may take a minute depending on optimiser iterations."):
            try:
                # Initialize the optimizer with the uploaded file and UI parameters
                # Note: Streamlit's file uploader works perfectly in place of a file path
                opt = BESSSettlementOptimizer(
                    excel_path=uploaded_file,
                    max_soc_perc=max_soc / 100.0,
                    min_soc_perc=min_soc / 100.0,
                    max_bess_hours=max_bess_hrs,
                    max_grid_allowable=max_grid_windh+max_grid_solarh,
                    max_grid_wind=max_grid_windh,
                    max_grid_solar=max_grid_solarh,
                    RTE=RTEs / 100,
                    tariff_cap=tariff_caps,
                    solar_capex=solar_capexs,
                    wind_capex=wind_capexs,
                    BESS_capex=BESS_capexs,
                    discharge_degradation=(100 - discharge_degradations) / 100,
                    cost_escalation=(cost_escalations + 100) / 100,
                    min_effective_replacement=min_effective_replacements / 100
                )
                
                # Run the model
                opt.load_data().build_model().solve(tee=False)
                
                st.success(f"Model solved successfully! (Termination: {opt.results.solver.termination_condition})")
                
                # --- SHOW RESULTS ON SCREEN ---
                st.subheader("Summary Statistics")
                stats = opt.summary_stats()
                
                # Create a nice layout for top metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Capex/EBITDA Year 1 Ratio", f"{stats['Capex_to_EBITDA_Year1']:.2f}")
                col2.metric("Optimal BESS Storage Hours", f"{stats['BESS_Storage_Hours']:.2f}")
                col3.metric("Effective Replacement", f"{stats['Effective_Replacement_Percent']:.1f}%")
                col4.metric("Tariff (Rs/kWh)", f"{stats['Tariff_RsPerkWh']:.2f}")

                # --- EXCEL EXPORT ---
                # Instead of saving to a local disk, we save to memory so the user can download it
                output = io.BytesIO()
                opt.export_to_excel(output)
                output.seek(0)
                
                st.download_button(
                    label="📥 Download Optimised Schedule (Excel)",
                    data=output,
                    file_name="Optimised_CAPEXEBITDA_Schedule.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

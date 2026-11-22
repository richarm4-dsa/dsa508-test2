"""
DSA 508 - Big Data Platforms and Analytics
Test 2 Fall 2026
Michelle Richardson

Name: phm_dashboard_cosmos_fleet.py
Description: This script links to a beam etch column telemetry dataset stored in Azure Cosmos DB and creates a streamlit 
dashboard for tool matching and analytics.
Usage: Execute from terminal (not by running in VS Code!)
  streamlit run <path>/phm_dashboard_cosmos_fleet.py

"""
import os
import streamlit as st
from azure.cosmos import CosmosClient
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(page_title="Ion Etch Fleet Analytics", layout="wide")

COSMOS_URI = st.secrets.get("COSMOS_URI", os.getenv("COSMOS_URI"))
COSMOS_KEY = st.secrets.get("COSMOS_KEY", os.getenv("COSMOS_KEY"))

@st.cache_data(ttl=120)
def fetch_fleet_data():
    client = CosmosClient(COSMOS_URI, credential=COSMOS_KEY)
    container = client.get_database_client("SemiconColumnDB").get_container_client("IonMill_Fleet")
    docs = list(container.query_items(query="SELECT * FROM c", enable_cross_partition_query=True))
    df = pd.json_normalize(docs).sort_values(["tool_id", "timestamp_sec"])
    
    # In PHM 2018, sensor channels are z-score normalized around 0.
    # Shift voltage into a positive range (> 0) for active beam-on steps to compute Perveance proxy
    v_shifted = df["high_tension.beam_voltage_v"] - df["high_tension.beam_voltage_v"].min() + 0.1
    df["beam_perveance"] = np.where(
        df["high_tension.beam_voltage_v"] > 0,
        df["high_tension.beam_current_ma"] / (v_shifted ** 1.5),
        np.nan
    )
    return df

df = fetch_fleet_data()

st.title("Ion Beam Etch Column Analytics")
st.caption("Data queried from Azure Cosmos DB")

# Sidebar Filters
st.sidebar.header("Filters")
selected_tools = st.sidebar.multiselect("Select Tools", df["tool_id"].unique(), default=df["tool_id"].unique())
selected_states = st.sidebar.multiselect("Filter Fault / Operating State", df["fault_state"].unique(), default=df["fault_state"].unique())
active_only = st.sidebar.checkbox("Beam-On Packets Only (Normalized Voltage > 0)", value=False)
fdf = df[df["tool_id"].isin(selected_tools) & df["fault_state"].isin(selected_states)]
if active_only:
    fdf = fdf[fdf["high_tension.beam_voltage_v"] > 0]
st.sidebar.divider()
st.sidebar.caption(
    "**Data Source:** [PHM Society 2018 Data Challenge](https://github.com/ricardoevvargas/awesome-industry40-datasets/tree/master/datasets/Semicon/PHM%20Data%20Challenge%2018) "
    "(Ion Mill Etching System Telemetry)."
)

# Top KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Filtered Telemetry Packets", f"{len(fdf):,}")
c2.metric("Active Tools Monitored", fdf["tool_id"].nunique())
c3.metric("Mean Beam Current (Norm)", f"{fdf['high_tension.beam_current_ma'].mean():.2f}")

fault_count = len(fdf[fdf["fault_state"] != "Normal"])
fault_pct = (fault_count / len(fdf) * 100) if len(fdf) > 0 else 0.0
c4.metric("Fault Packets in Selection", f"{fault_count:,} ({fault_pct:.1f}%)")

# Row 1
st.header("Beam Current Matching & Faults")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Multi-Tool Beam Current Stability (3-sigma SPC Chart)")
    st.caption("Tracks rolling 15-packet within-step beam current drift from nominal baseline. Breaches beyond ±3-sigma indicate dose surges or emission droop.")

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    window_size = 15

    for idx, tool in enumerate(sorted(fdf["tool_id"].unique())):
        tdf = fdf[fdf["tool_id"] == tool].reset_index(drop=True)
        col = "high_tension.beam_current_ma"

        # Use nominal baseline median and standard deviation for each tool
        baseline = tdf[tdf["fault_state"] == "Normal"] if "Normal" in tdf["fault_state"].values else tdf
        med = baseline[col].median()
        std = float(baseline[col].std()) if float(baseline[col].std()) > 0.05 else 1.0

        # Smooth 10-packet rolling deviation in baseline standard deviation units
        z_scores = (tdf[col] - med) / std
        drift_trend = z_scores.rolling(window=10, min_periods=1).mean()

        fig.add_trace(go.Scatter(
            y=drift_trend,
            mode="lines",
            name=tool,
            line=dict(color=colors[idx % len(colors)], width=2)
        ))


    fig.add_hline(y=0.0, line_dash="dot", line_color="gray", annotation_text="Nominal (0-sigma)")
    fig.add_hline(y=3.0, line_dash="dash", line_color="red", annotation_text="+3-sigma Upper Limit")
    fig.add_hline(y=-3.0, line_dash="dash", line_color="red", annotation_text="-3-sigma Lower Limit")

    fig.update_layout(
        height=430,
        xaxis_title="Sequential Operating Packets",
        yaxis_title="Within-Step Beam Drift (sigma)",
        yaxis=dict(range=[-5, 5]), 
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Fault Frequency Distribution Across Tools")
    st.caption("Breaks down discrete fault packet counts per chamber to identify recurring subsystem faults.")
    fault_only = fdf[fdf["fault_state"] != "Normal"]
    if not fault_only.empty:
        fault_counts = fault_only.groupby(["fault_state", "tool_id"]).size().reset_index(name="packet_count")
        fig_faults = px.bar(
            fault_counts,
            x="packet_count",
            y="fault_state",
            color="tool_id",
            orientation="h",
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Plotly,
            labels={"packet_count": "Fault Packet Count", "fault_state": "", "tool_id": "Tool"}
        )
        fig_faults.update_layout(
            height=430,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(
                showline=True,
                linewidth=1.5,
                linecolor="rgba(150, 150, 150, 0.7)",
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="rgba(150, 150, 150, 0.9)"
            ),
            yaxis=dict(
                categoryorder="total ascending",
                showline=True,
                linewidth=2,
                linecolor="rgba(150, 150, 150, 0.9)"
            ),
            bargap=0.25,       # Adds clear spacing between the three fault categories
            bargroupgap=0.08,  # Keeps bars within the same fault type neatly grouped together
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1)
        )

        # Add horizontal divider lines between each fault category
        num_categories = fault_counts["fault_state"].nunique()
        for i in range(num_categories - 1):
            fig_faults.add_hline(
                y=i + 0.5,
                line_dash="dot",
                line_width=1,
                line_color="rgba(150, 150, 150, 0.45)"
            )

        st.plotly_chart(fig_faults, use_container_width=True)



# Row 2
st.header("High Tension Source Health")
col3, col4 = st.columns(2)

with col3:
    st.subheader("Beam Stability vs. Source Usage (Hrs)")
    st.caption("Monitors beam output stability as the high tension source ages.")
    st.scatter_chart(
        fdf.dropna(subset=["beam_perveance"]),
        x="high_tension.source_usage_hrs",
        y="beam_perveance",
        color="tool_id"
    )

with col4:
    st.subheader("Vacuum Pressure vs. Stray Grid Current")
    st.caption("Monitors the relationship between chamber vacuum and stray electrical leakage onto the suppressor grid, an early warning indicator of high-voltage arcing.")
    st.scatter_chart(
        fdf,
        x="vacuum.ion_gauge_pressure",
        y="high_tension.suppressor_current_ma",
        color="tool_id"
    )

# Row 3
st.header("Process Consistency")
st.subheader("Recipe-Step Summary")
st.caption("Compares average operating levels and stability across identical manufacturing steps to verify chamber-to-chamber matching.")

summary = fdf.groupby(["tool_id", "recipe_step"], dropna=False).agg(
    packet_count=("id", "count"),
    mean_pressure=("vacuum.ion_gauge_pressure", "mean"),
    mean_beam_v=("high_tension.beam_voltage_v", "mean"),
    std_beam_v=("high_tension.beam_voltage_v", "std"),
    mean_he_pressure=("cooling.helium_pressure", "mean")
).reset_index()

# Interactive Cross-Tool Step Comparison Chart
metric_map = {
    "Average Beam Voltage (Normalized)": "mean_beam_v",
    "Average Chamber Pressure (Normalized)": "mean_pressure",
    "Average Helium Cooling Pressure (Normalized)": "mean_he_pressure",
    "Beam Voltage Instability (Std Dev)": "std_beam_v"
}
selected_metric = st.selectbox("Compare Tools Across Recipe Steps By:", list(metric_map.keys()))

step_pivot = summary.pivot(
    index="recipe_step",
    columns="tool_id",
    values=metric_map[selected_metric]
)
st.bar_chart(step_pivot)

# Detailed Baseline Table
st.dataframe(summary, use_container_width=True)

st.divider()
st.caption(
    "**Dataset Citation:** Prognostics and Health Management Society (2018). "
    "*PHM Society 2018 Data Challenge: Ion Mill Etching System* [Data set]. "
    "https://github.com/ricardoevvargas/awesome-industry40-datasets/tree/master/datasets/Semicon/PHM%20Data%20Challenge%2018"
)


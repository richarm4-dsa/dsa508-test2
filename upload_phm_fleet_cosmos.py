"""
DSA 508 - Big Data Platforms and Analytics
Test 2 Fall 2026
Michelle Richardson

Name: upload_phm_fleet_cosmos.py
Upload PHM data to Azure Cosmos DB for NoSQL
This script reads PHM data from a zipped tar file for specified tools, processes it, and uploads it to Azure Cosmos DB.
Download source data tar file from Google link at, and store it in a subdirectory named /data
https://github.com/ricardoevvargas/awesome-industry40-datasets/tree/master/datasets/Semicon/PHM%20Data%20Challenge%2018

"""

import tarfile
from pathlib import Path
import pandas as pd
from azure.cosmos import CosmosClient, PartitionKey
import streamlit as st
import os

COSMOS_URI = st.secrets.get("COSMOS_URI", os.getenv("COSMOS_URI"))
COSMOS_KEY = st.secrets.get("COSMOS_KEY", os.getenv("COSMOS_KEY"))

# 2. Configuration: Add any tool IDs from the PHM dataset here
TOOLS_TO_LOAD = ["01_M01", "01_M02", "02_M01", "02_M02", "03_M01", "03_M02"]
NORMAL_SAMPLES_PER_TOOL = 600
FAULT_WINDOWS_PER_TOOL = 15

# Path to downloaded .tar.gz archive
base_dir = Path(__file__).resolve().parent
TAR_PATH = base_dir / "data" / "phm_data_challenge_2018.tar.gz"

# Connect to Azure Cosmos DB
client = CosmosClient(COSMOS_URI, credential=COSMOS_KEY)
database = client.create_database_if_not_exists(id="SemiconColumnDB")
container = database.create_container_if_not_exists(
    id="IonMill_Fleet",
    partition_key=PartitionKey(path="/tool_id")
)

def build_document(tool_id, idx, row, suffix=""):
    return {
        "id": f"{tool_id}_{suffix}{idx}",
        "tool_id": str(row["Tool"]),
        "timestamp_sec": float(row["time"]),
        "stage": str(row["stage"]),
        "recipe_step": int(row["recipe_step"]),
        "vacuum": {
            "ion_gauge_pressure": float(row["IONGAUGEPRESSURE"])
        },
        "high_tension": {
            "beam_voltage_v": float(row["ETCHBEAMVOLTAGE"]),
            "beam_current_ma": float(row["ETCHBEAMCURRENT"]),
            "suppressor_voltage_v": float(row["ETCHSUPPRESSORVOLTAGE"]),
            "suppressor_current_ma": float(row["ETCHSUPPRESSORCURRENT"]),
            "source_usage_hrs": float(row["ETCHSOURCEUSAGE"])
        },
        "cooling": {
            "helium_flow_rate": float(row["FLOWCOOLFLOWRATE"]),
            "helium_pressure": float(row["FLOWCOOLPRESSURE"])
        },
        "fault_state": str(row["fault_state"])
    }

# Stream each tool directly out of the .tar.gz archive and upload to Azure
with tarfile.open(TAR_PATH, "r:*") as tar:
    members = tar.getmembers()

    for tool in TOOLS_TO_LOAD:
        print(f"\nProcessing Tool {tool}...")
        data_member = next(m for m in members if m.name.endswith(f"train/{tool}_DC_train.csv"))
        fault_member = next(m for m in members if m.name.endswith(f"train_faults/{tool}_train_fault_data.csv"))

        raw_df = pd.read_csv(tar.extractfile(data_member))
        faults = pd.read_csv(tar.extractfile(fault_member))

        # Sample evenly spaced normal points
        step = max(1, len(raw_df) // NORMAL_SAMPLES_PER_TOOL)
        normal_df = raw_df.iloc[::step].head(NORMAL_SAMPLES_PER_TOOL).fillna(0).copy()
        normal_df["fault_state"] = "Normal"

        for idx, row in normal_df.iterrows():
            container.upsert_item(build_document(tool, idx, row))

        # Extract labeled fault windows
        fault_count = 0
        for _, f_row in faults.head(FAULT_WINDOWS_PER_TOOL).iterrows():
            window = raw_df[
                (raw_df["time"] >= f_row["time"] - 40) &
                (raw_df["time"] <= f_row["time"] + 20)
            ].iloc[::5].fillna(0).copy()
            window["fault_state"] = str(f_row["fault_name"])

            for idx, row in window.iterrows():
                container.upsert_item(build_document(tool, idx, row, suffix="fault_"))
                fault_count += 1

        print(f"Uploaded {len(normal_df)} normal + {fault_count} fault documents for {tool}.")

print("\nAll fleet tools uploaded to Azure Cosmos DB container 'IonMill_Fleet'!")

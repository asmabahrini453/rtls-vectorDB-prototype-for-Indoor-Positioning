"""
preprocessing/vector_preprocessing.py
---------------------------------------
Takes your merged parquet + Excel survey file and produces
a clean fingerprint dataset ready to load into PostgreSQL.

WHAT THIS SCRIPT DOES (step by step):
  1. Loads data/raw/all_rssi.parquet  ( 478k RSSI readings generated in 05/05/2026)
  2. Loads the Excel survey file      
  3. Builds a global anchor index     (maps anchor ID → fixed position in vector)
  4. For each reference point in Excel:
       - finds all RSSI rows within that time window on that floor
       - averages the signal per anchor
       - fills missing anchors with RSSI_FLOOR (-100 dBm = "not seen")
       - produces one clean 41-dim vector
  5. Saves result to data/processed/fingerprints.csv

HOW TO RUN:
    python preprocessing/vector_preprocessing.py

REQUIRES:
    pip install pandas pyarrow openpyxl numpy
"""


import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json

# ── paths ───
PARQUET_PATH = Path("data/raw/all_rssi.parquet")
EXCEL_PATH   = Path("C:/Users/bahri/Downloads/data_collection/data_collection/DATA 07 - Clinique de LA Sauvgarde.xlsx") 
OUT_CSV      = Path("data/processed/fingerprints.csv") # the final dataset -> fingerprints
OUT_INDEX    = Path("data/processed/anchor_index.json") #mapping for vector structure
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# ── settings ──
RSSI_FLOOR   = -100   # dBm value assigned when an anchor is NOT seen at a location
TIMEZONE     = "Europe/Paris"   # your local timezone (survey times are local)
MIN_READINGS = 3      # minimum RSSI readings required to keep a reference point

# ── step 1 · load parquet ─────────────────────────────────────────────────────
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("STEP 1 — Loading parquet data")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

if not PARQUET_PATH.exists():
    print(f" File not found: {PARQUET_PATH}")
    sys.exit(1)

df = pd.read_parquet(PARQUET_PATH)
df["real_floor"] = df["real_floor"].str.strip()   # remove :accidental spaces

#ensuring ALL timestamps are UTC
if df["time"].dt.tz is None:
    df["time"] = df["time"].dt.tz_localize("UTC")
else:
    df["time"] = df["time"].dt.tz_convert("UTC")

print(f" {len(df):,} rows loaded")
print(f"      Floors  : {sorted(df['real_floor'].unique().tolist())}")
print(f"      Tags    : {df['nodeid'].nunique()}")
print(f"      Dates   : {df['time'].min().date()} → {df['time'].max().date()}")


# ── step 2 · load excel survey ────
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("STEP 2 — Loading Excel survey file")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

if not EXCEL_PATH.exists():
    print(f" Excel file not found at: {EXCEL_PATH}")
    sys.exit(1)

xl = pd.read_excel(EXCEL_PATH)
xl.columns = xl.columns.str.strip()
xl["Etage"] = xl["Etage"].str.strip()
xl["Aile"]  = xl["Aile"].str.strip()

# Build combined datetime from Date column + time columns
xl["dt_start"] = pd.to_datetime(
    xl["Date"].dt.strftime("%Y-%m-%d") + " " + xl["timestamps_début"].astype(str)
).dt.tz_localize(TIMEZONE).dt.tz_convert("UTC")

xl["dt_end"] = pd.to_datetime(
    xl["Date"].dt.strftime("%Y-%m-%d") + " " + xl["timestamps_fin"].astype(str)
).dt.tz_localize(TIMEZONE).dt.tz_convert("UTC")

print(f"    {len(xl)} reference points loaded")
print(f"      Floors  : {sorted(xl['Etage'].unique().tolist())}")
print(f"      Wings   : {sorted(xl['Aile'].unique().tolist())}")
print(f"      Dates   : {xl['dt_start'].min().date()} → {xl['dt_end'].max().date()}")

# ── step 3 · build global anchor index ───
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("STEP 3 — Building global anchor index")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# Collect every unique anchor seen across the ENTIRE dataset
all_anchors = set()
for anchor_list in df["anchors"]:
    all_anchors.update([str(a) for a in anchor_list])

# Sort them so the index is deterministic (same order every time)
anchor_list_sorted = sorted(all_anchors)
anchor_to_idx = {anchor: i for i, anchor in enumerate(anchor_list_sorted)}
VECTOR_DIM = len(anchor_list_sorted)

print(f"   {VECTOR_DIM} unique anchors indexed")
print(f"    Vector dimension will be: {VECTOR_DIM}")

# Save the anchor index to disk, we'll need it again during ingestion & querying
anchor_index = {str(i): anchor for i, anchor in enumerate(anchor_list_sorted)}
with open(OUT_INDEX, "w") as f:
    json.dump(anchor_index, f, indent=2)
print(f"   Anchor index saved → {OUT_INDEX}")

# ── step 4 · build fingerprint vectors ──
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("STEP 4 — Building fingerprint vectors")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"  Processing {len(xl)} reference points...\n")

# Explode parquet into one row per (timestamp, anchor, rssi) for fast grouping
rows_expanded = []
for _, row in df.iterrows():
    for anchor, rssi in zip(row["anchors"], row["rssi_list"]):
        rows_expanded.append({
            "time"  : row["time"],
            "floor" : row["real_floor"],
            "anchor": str(anchor),
            "rssi"  : float(rssi)
        })

df_exp = pd.DataFrame(rows_expanded)
print(f"  Expanded to {len(df_exp):,} anchor-level rows")

# Now we build one vector per reference point ->Each RP = one known location
records = []
skipped = 0

for _, rp in xl.iterrows():
    rp_floor = rp["Etage"]
    rp_label = rp["Localisation précise"]
    rp_start = rp["dt_start"]
    rp_end   = rp["dt_end"]

    # Filter to rows within the survey time window on the correct floor
    #selects only rssi data with same floor and same time window
    mask = (
        (df_exp["floor"] == rp_floor) &
        (df_exp["time"]  >= rp_start) &
        (df_exp["time"]  <  rp_end)
    )
    window = df_exp[mask]
    #skip bad data
    if len(window) < MIN_READINGS:
        print(f"   Skipped RP {rp['N°']:>3} [{rp_floor}] '{rp_label[:45]}'"
              f"  — only {len(window)} readings (need ≥{MIN_READINGS})")
        skipped += 1
        continue

    # Average RSSI per anchor within the window
    avg_rssi = window.groupby("anchor")["rssi"].mean() #reduces noise by averaging multiple readings of the same anchor in the time window

    # Build the fixed-size vector (RSSI_FLOOR for unseen anchors : -100 dbm)
    vector = np.full(VECTOR_DIM, RSSI_FLOOR, dtype=np.float32)
    for anchor, rssi_val in avg_rssi.items():
        if anchor in anchor_to_idx:
            vector[anchor_to_idx[anchor]] = rssi_val #place the rssi value in the correct position in the vector according to the anchor index

    records.append({
        "rp_id"      : int(rp["N°"]),
        "label"      : rp_label,
        "floor"      : rp_floor,
        "wing"       : rp["Aile"],
        "orientation": rp["Orientation"],
        "survey_date": str(rp["Date"].date()),
        "n_readings" : len(window),
        "n_anchors_seen": int((vector > RSSI_FLOOR).sum()),
        "vector"     : json.dumps(vector.tolist())   # stored as JSON string in CSV
    })

# ── step 5 · save results ─────────────────
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("STEP 5 — Saving results")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

out_df = pd.DataFrame(records)
out_df.to_csv(OUT_CSV, index=False)

print(f"""
   Done!

  Reference points processed : {len(records)}
  Reference points skipped   : {skipped}  (not enough RSSI data in window)
  Vector dimension           : {VECTOR_DIM}
  Avg RSSI readings per RP   : {out_df['n_readings'].mean():.0f}
  Avg anchors seen per RP    : {out_df['n_anchors_seen'].mean():.1f} / {VECTOR_DIM}

  Output files:
    📄  {OUT_CSV}          ← fingerprint vectors (one row per location)
    📄  {OUT_INDEX}   ← anchor index (position → anchor ID)


""")
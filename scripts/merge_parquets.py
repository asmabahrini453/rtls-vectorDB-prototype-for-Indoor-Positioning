"""
scripts/merge_parquets.py
--------------------------
Merges all your parquet files into one unified file: data/raw/all_rssi.parquet

HOW TO RUN:
    python scripts/merge_parquets.py --input_dir "C:/Users/bahri/Downloads/data_collection/data_collection/new"

The script will:
  1. Load every .parquet file it finds in the input folder
  2. Add a 'source_file' column so you always know where each row came from
  3. Deduplicate any accidental duplicate rows
  4. Save the result to data/raw/all_rssi.parquet
"""

import argparse
import pandas as pd
from pathlib import Path
import sys

# ── argument: where are your parquet files? ──────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_dir",
    required=True,
    help="C:/Users/bahri/Downloads/data_collection/data_collection/new"
)
args = parser.parse_args()

input_dir = Path(args.input_dir)
output_path = Path("data/raw/all_rssi.parquet")
output_path.parent.mkdir(parents=True, exist_ok=True)

# ── find all parquet files ────────────────────────────────────────────────────
parquet_files = sorted(input_dir.glob("*.parquet"))

if not parquet_files:
    print(f"❌  No .parquet files found in: {input_dir}")
    sys.exit(1)

print(f"✅  Found {len(parquet_files)} parquet files:\n")

# ── load and merge ────────────────────────────────────────────────────────────
frames = []
for f in parquet_files:
    df = pd.read_parquet(f)
    df["source_file"] = f.name          # track which file each row came from
    frames.append(df)
    print(f"   {f.name:45s}  →  {len(df):>7,} rows  |  floors: {df['real_floor'].unique().tolist()}")

merged = pd.concat(frames, ignore_index=True)

# ── deduplicate (same tag, same timestamp, same floor) ───────────────────────
before = len(merged)
merged = merged.drop_duplicates(subset=["time", "nodeid", "real_floor"])
after  = len(merged)

if before != after:
    print(f"\n⚠️   Removed {before - after:,} duplicate rows")

# ── sort by time ─────────────────────────────────────────────────────────────
merged = merged.sort_values("time").reset_index(drop=True)

# ── save ─────────────────────────────────────────────────────────────────────
merged.to_parquet(output_path, index=False)

# ── summary ──────────────────────────────────────────────────────────────────
print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅  Merge complete → {output_path}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total rows   : {len(merged):,}
  Unique tags  : {merged['nodeid'].nunique()}
  Unique floors: {sorted(merged['real_floor'].unique().tolist())}
  Date range   : {merged['time'].min().date()}  →  {merged['time'].max().date()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
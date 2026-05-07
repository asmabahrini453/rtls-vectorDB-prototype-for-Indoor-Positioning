"""
vector_engine/fingerprint_builder.py
--------------------------------------
Reads data/processed/fingerprints.csv and inserts everything into PostgreSQL.

run:
  python vector_engine/fingerprint_builder.py
"""

import json
import sys
import numpy as np
import pandas as pd
import psycopg2
from pathlib import Path

# ── config ──── 
#i will change this to be crypted because it is not good to have it in the code but for now it is like this for simplicity
DB = dict(host="localhost", port=5432, dbname="rtls", user="rtls_user", password="rtls_password")

FINGERPRINTS_CSV  = Path("data/processed/fingerprints.csv")
ANCHOR_INDEX_JSON = Path("data/processed/anchor_index.json")

# ── connect ───────────────────────────────────────────────────────────────────
print("\n━━━ STEP 1 — Connecting to PostgreSQL ━━━")
try:
    conn = psycopg2.connect(**DB)
    cur  = conn.cursor()
    cur.execute("SELECT version();")
    print(f"   Connected: {cur.fetchone()[0][:40]}...")
except Exception as e:
    print(f"   Could not connect: {e}")
    print("      Make sure docker-compose is running: docker-compose up -d")
    sys.exit(1)

# ── enable extensions 
print("\n━━━ STEP 2 — Enabling extensions ━━━")
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
#cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
conn.commit()
print("  pgvector ready")

# ── load anchor index ──
print("\n━━━ STEP 3 — Loading anchor index ━━━")
if not ANCHOR_INDEX_JSON.exists():
    print(f"    Not found: {ANCHOR_INDEX_JSON}")
    print("      Run preprocessing first: python preprocessing/vector_preprocessing.py")
    sys.exit(1)
with open(ANCHOR_INDEX_JSON) as f:
    anchor_index = json.load(f)
VECTOR_DIM = len(anchor_index)
anchor_index_str = json.dumps(anchor_index)
print(f"   {VECTOR_DIM} anchors loaded")

# ── recreate tables cleanly ──
print("\n━━━ STEP 4 — Creating tables ━━━")
cur.execute("DROP TABLE IF EXISTS live_observations CASCADE;")
cur.execute("DROP TABLE IF EXISTS fingerprint_map    CASCADE;")
cur.execute("DROP TABLE IF EXISTS reference_points   CASCADE;")
cur.execute("DROP TABLE IF EXISTS anchors            CASCADE;")

cur.execute(f"""
CREATE TABLE reference_points (
    id          SERIAL PRIMARY KEY,
    rp_id       INT UNIQUE NOT NULL,
    label       TEXT NOT NULL,
    floor       TEXT NOT NULL,
    wing        TEXT,
    orientation TEXT,
    survey_date DATE,
    n_readings  INT,
    n_anchors_seen INT
);
""")

cur.execute(f"""
CREATE TABLE fingerprint_map (
    id           SERIAL PRIMARY KEY,
    reference_id INT REFERENCES reference_points(id) ON DELETE CASCADE,
    floor        TEXT NOT NULL,
    label        TEXT,
    embedding    vector({VECTOR_DIM}),
    anchor_index JSONB,
    created_at   TIMESTAMPTZ DEFAULT now()
);
""")

cur.execute(f"""
CREATE TABLE live_observations (
    id              SERIAL PRIMARY KEY,
    tag_id          BIGINT,
    observed_at     TIMESTAMPTZ DEFAULT now(),
    floor           TEXT,
    embedding       vector({VECTOR_DIM}),
    estimated_label TEXT,
    top_k_results   JSONB
);
""")
conn.commit()
print("   Tables created (reference_points, fingerprint_map, live_observations)")

# ── load CSV ──────────────────────────────────────────────────────────────────
print("\n━━━ STEP 5 — Loading fingerprints.csv ━━━")
if not FINGERPRINTS_CSV.exists():
    print(f"    Not found: {FINGERPRINTS_CSV}")
    sys.exit(1)
df = pd.read_csv(FINGERPRINTS_CSV)
print(f"  {len(df)} fingerprints to insert")

# ── insert ───
print("\n━━━ STEP 6 — Inserting into database ━━━")
inserted = 0
errors   = 0

for _, row in df.iterrows():
    try:
        # 1. Insert reference point
        cur.execute("""
            INSERT INTO reference_points
                (rp_id, label, floor, wing, orientation, survey_date, n_readings, n_anchors_seen)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        """, (
            int(row["rp_id"]),
            str(row["label"]),
            str(row["floor"]),
            str(row["wing"]),
            str(row["orientation"]),
            str(row["survey_date"]),
            int(row["n_readings"]),
            int(row["n_anchors_seen"])
        ))
        ref_id = cur.fetchone()[0]

        # 2. Parse the vector from JSON string → PostgreSQL vector format
        vector_list = json.loads(row["vector"])
        vector_str  = "[" + ",".join(str(round(v, 4)) for v in vector_list) + "]"

        # 3. Insert fingerprint vector
        cur.execute("""
            INSERT INTO fingerprint_map (reference_id, floor, label, embedding, anchor_index)
            VALUES (%s, %s, %s, %s::vector, %s::jsonb);
        """, (ref_id, str(row["floor"]), str(row["label"]), vector_str, anchor_index_str))

        inserted += 1

    except Exception as e:
        errors += 1
        print(f"  ⚠️  Error on RP {row['rp_id']}: {e}")
        conn.rollback()
        continue

    conn.commit()

# ── create HNSW index for fast similarity search ─
print("\n━━━ STEP 7 — Building HNSW index ━━━")
print("  (this makes nearest-neighbour queries ~100x faster)")
cur.execute("""
    CREATE INDEX fingerprint_hnsw_idx
        ON fingerprint_map
        USING hnsw (embedding vector_l2_ops)
        WITH (m = 16, ef_construction = 64);
""")
cur.execute("CREATE INDEX fp_floor_idx ON fingerprint_map (floor);")
conn.commit()
print("  HNSW index built")

# ── verify ────
print("\n━━━ STEP 8 — Verification ━━━")
cur.execute("SELECT floor, COUNT(*) FROM fingerprint_map GROUP BY floor ORDER BY floor;")
rows = cur.fetchall()
print("  Fingerprints per floor:")
for floor, count in rows:
    print(f"    {floor:>5}  →  {count} fingerprints")

cur.execute("SELECT COUNT(*) FROM fingerprint_map;")
total = cur.fetchone()[0]

cur.close()
conn.close()

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Database ready!

  Inserted  : {inserted}
  Errors    : {errors}
  Total in DB: {total}
  Vector dim : {VECTOR_DIM}


""")
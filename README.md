# RTLS Vector Database Prototype
### Indoor Positioning via RSSI Fingerprinting — PostgreSQL + pgvector

> **Clinique de la Sauvegarde** | Built by Asma Bahrini | May 2026  
> Prototype replacing MLP neural network architecture with a deterministic vector similarity search engine.

---

## Results at a Glance

| Metric | Score |
|---|---|
| **Top-1 Accuracy** (correct room, first guess) | **97.9%** |
| **Top-3 Accuracy** (correct room in top 3) | **99.9%** |
| Floors covered | E1, E2, E3, E4, RDC |
| Fingerprints in database | 141 |
| Total evaluation queries | 2,820 (L2 + Cosine x 10 trials x 141 FPs) |
| Best distance metric | **L2 (Euclidean)** — wins or ties on all 141 locations |
| System downtime when a beacon moves | **Zero** — single SQL UPDATE |

---

## What This Is

Traditional indoor positioning systems use neural networks (MLPs) to map RSSI signal fingerprints to physical locations. While accurate, MLPs are brittle: when a beacon moves, the entire model must be retrained — requiring GPU compute, data science expertise, and system downtime.

This prototype replaces the MLP with **PostgreSQL + pgvector**: each surveyed location is stored as a 111-dimensional RSSI fingerprint vector. At query time, a live RSSI scan is compared against all stored fingerprints using L2 distance via an HNSW index, returning the closest matching locations in milliseconds.

**When a beacon moves:** run one `UPDATE` SQL statement. Zero downtime. No retrain.

---

## Project Structure

```
rtls-vector-prototype/
├── docker-compose.yml              # Starts PostgreSQL + pgvector in one command
├── README.md
├── config/
│   └── settings.py                 # DB credentials and shared constants
├── data/
│   ├── raw/
│   │   └── all_rssi.parquet        # Merged RSSI dataset (478,832 rows)
│   └── processed/
│       ├── fingerprints.csv        # 141 clean fingerprint vectors
│       ├── anchor_index.json       # Maps vector position -> anchor ID
│       ├── evaluation_results.csv  # Per-location L2 accuracy
│       └── metric_comparison.csv  # L2 vs Cosine head-to-head
├── scripts/
│   └── merge_parquets.py           # Merges 12 parquet files into all_rssi.parquet
├── preprocessing/
│   └── vector_preprocessing.py    # Parquet + Excel -> clean 111-dim vectors
├── vector_engine/
│   └── fingerprint_builder.py     # Inserts vectors into PostgreSQL + builds HNSW index
├── database/
│   └── queries.py                  # Live location query function (import in production)
└── evaluation/
    ├── accuracy.py                 # Full accuracy evaluation with noise simulation
    └── compare_metrics.py          # L2 vs Cosine similarity comparison
```

---

## Quick Start

### Prerequisites
- Docker Desktop installed and running
- Python 3.8+
- The 12 parquet files exported on the 05/05/2026
- The Excel survey file `DATA_07_-_Clinique_de_LA_Sauvgarde__4_.xlsx`

### Step 1 — Start the database
```bash
docker-compose up -d
```
Verify:
```bash
docker ps   # Should show: rtls-db   Up
```

### Step 2 — Install Python dependencies
```bash
pip install pandas pyarrow openpyxl numpy psycopg2-binary
```

### Step 3 — Merge parquet files
```bash
python scripts/merge_parquets.py --input_dir "C:/path/to/parquet/folder"
# -> data/raw/all_rssi.parquet  (478,832 rows)
```

### Step 4 — Build fingerprint vectors
```bash
python preprocessing/vector_preprocessing.py
# -> data/processed/fingerprints.csv  (141 vectors x 111 dimensions)
# -> data/processed/anchor_index.json
```
> Update `EXCEL_PATH` in the script to point to your Excel file before running.

### Step 5 — Load into PostgreSQL
```bash
python vector_engine/fingerprint_builder.py
# Creates tables, inserts 141 fingerprints, builds HNSW index
```

### Step 6 — Test a live query
```bash
python database/queries.py
# Picks a random fingerprint, adds noise, queries the system, shows top-5
```

### Step 7 — Evaluate accuracy
```bash
python evaluation/accuracy.py          # L2 accuracy all 141 fingerprints
python evaluation/compare_metrics.py   # L2 vs Cosine comparison
```


## Technical Architecture

### Database Schema

```sql
-- Surveyed locations (from Excel walkthrough)
reference_points (
    id, rp_id, label, floor, wing, orientation,
    survey_date, n_readings, n_anchors_seen
)

-- Fingerprint vectors — core of the system
fingerprint_map (
    id, reference_id, floor, label,
    embedding vector(111),   -- the RSSI fingerprint
    anchor_index jsonb        -- maps vector position -> anchor ID
)

-- Optional: log every live query for drift detection over time
live_observations (
    id, tag_id, observed_at, floor,
    embedding vector(111), estimated_label, top_k_results
)
```

### HNSW Index
```sql
CREATE INDEX fingerprint_hnsw_idx
    ON fingerprint_map
    USING hnsw (embedding vector_l2_ops)
    WITH (m = 16, ef_construction = 64);
```
- `m = 16`: max graph connections per node — controls quality vs build time
- `ef_construction = 64`: candidate list size — higher = better recall, slower build
- Recall rate: >98% (true nearest neighbour found 98%+ of the time)
- Query latency: sub-millisecond on this dataset

### Distance Metrics Tested

| Operator | Name | Top-1 | Top-3 | Winner |
|---|---|---|---|---|
| `<->` | L2 / Euclidean | **97.9%** | **99.9%** | YES |
| `<=>` | Cosine | 97.7% | 99.9% | — |

**L2 wins** because RSSI vectors use -100 dBm to represent an anchor that is not visible. L2 correctly treats this as a large absolute difference — penalising anchor visibility mismatches. Cosine ignores magnitude and focuses only on signal ratios, partially discarding this information.

---

## Updating the System

### When a beacon is moved or replaced
```sql
-- Collect a new RSSI survey at the affected reference points, then:
UPDATE fingerprint_map
SET embedding = '[new_rssi_vector]'::vector
WHERE label = 'Pres de chambre 221' AND floor = 'E2';
-- System stays live. No other fingerprints affected.
```

### When a new area is added
```sql
INSERT INTO reference_points (rp_id, label, floor, ...) VALUES (...);
INSERT INTO fingerprint_map (floor, label, embedding, anchor_index) VALUES (...);
-- Existing fingerprints untouched.
```

### MLP comparison

| Update task | Vector DB | MLP Neural Network |
|---|---|---|
| Beacon moved | 1 SQL UPDATE | Full retrain + redeploy |
| New area added | INSERT rows | Collect data + retrain |
| System downtime | Zero | Yes — during redeployment |
| Expertise needed | SQL / DB admin | Python + ML + GPU |

---

## Evaluation Methodology

- **Noise simulation**: Gaussian noise (sigma = 3 dBm) added to each stored fingerprint before querying — simulates real-world signal variation between survey and live scan
- **Trials**: 10 per location = 1,410 queries per metric, 2,820 total
- **Metrics**: Top-1 (exact match), Top-3, Top-5
- **Floor filter**: always applied in evaluation (as it would be in production)

**Noise rationale:** 3 dBm is conservative for a hospital. BLE RSSI variance in indoor clinical settings is typically sigma = 2–5 dBm.

---

## Known Limitations

- **Self-evaluation**: Evaluation uses the same fingerprints as the database (with noise). A held-out test set on different days would give a more conservative accuracy estimate.
- **All-day survey windows**: 24 of 141 reference points used 24h RSSI windows. These may include interference from other moving tags, adding noise to the stored fingerprint.
- **No physical geometry**: Walls and floor constraints are not modelled. Floor filter mitigates cross-floor errors, but cross-wall errors within a floor are not prevented yet.
- **Fixed vector dimension**: The 111-dim vector is fixed at build time. Adding a new anchor requires rebuilding the database.

---

## Configuration

`config/settings.py`:
```python
DB_HOST     = "localhost"
DB_PORT     = 5432
DB_NAME     = "rtls"
DB_USER     = "rtls_user"
DB_PASSWORD = "rtls_password"  

VECTOR_DIM  = 111    # unique anchors in dataset
TOP_K       = 5      # neighbours returned per query
RSSI_FLOOR  = -100   # dBm for unseen anchors
```

---

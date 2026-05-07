"""
evaluation/accuracy.py
------------------------
Systematically tests ALL fingerprints in the database and produces
accuracy metrics you can show to Bertrand and the team.

HOW IT WORKS:
  For each of the 141 fingerprints in the DB:
    1. Pull its vector, add realistic noise (±3 dBm)
    2. Query the system — pretend it's a live tag scan
    3. Check if the correct location appears in top-1 / top-3 / top-5
  Then report accuracy per floor + overall.

Run:
  python evaluation/accuracy.py
"""

import json
import sys
import numpy as np
import psycopg2
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── config ────────────────────────────────────────────────────────────────────
DB                = dict(host="localhost", port=5432, dbname="rtls",
                         user="rtls_user", password="rtls_password")
ANCHOR_INDEX_JSON = Path("data/processed/anchor_index.json")
RSSI_FLOOR        = -100
NOISE_STD         = 3.0    # dBm — realistic signal variation between scans
TOP_K             = 5
N_TRIALS          = 10     # repeat each fingerprint N times with different noise

# ── load anchor index ─────────────────────────────────────────────────────────
with open(ANCHOR_INDEX_JSON) as f:
    anchor_index = json.load(f)
anchor_to_idx = {v: int(k) for k, v in anchor_index.items()}
VECTOR_DIM    = len(anchor_index)

# ── connect ───────────────────────────────────────────────────────────────────
conn = psycopg2.connect(**DB)
cur  = conn.cursor()

# ── fetch all fingerprints ────────────────────────────────────────────────────
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  RTLS Vector DB — Accuracy Evaluation")
print(f"  Noise level: ±{NOISE_STD} dBm | Trials per RP: {N_TRIALS}")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

cur.execute("""
    SELECT fm.id, fm.label, fm.floor, fm.embedding::text
    FROM fingerprint_map fm
    ORDER BY fm.floor, fm.label;
""")
fingerprints = cur.fetchall()
print(f"  Testing {len(fingerprints)} fingerprints × {N_TRIALS} trials = {len(fingerprints)*N_TRIALS} queries...\n")

# ── run evaluation ────────────────────────────────────────────────────────────
results_by_floor = defaultdict(lambda: {"top1": 0, "top3": 0, "top5": 0, "total": 0})
all_results      = []

for fp_id, true_label, true_floor, embedding_str in fingerprints:
    stored_vector = np.array(json.loads(embedding_str), dtype=np.float32)

    top1_hits = 0
    top3_hits = 0
    top5_hits = 0

    for trial in range(N_TRIALS):
        # Add Gaussian noise only to anchors that were actually seen
        noisy_vector = stored_vector.copy()
        seen_mask    = stored_vector > RSSI_FLOOR
        noisy_vector[seen_mask] += np.random.normal(0, NOISE_STD, seen_mask.sum())
        noisy_vector = np.clip(noisy_vector, RSSI_FLOOR, 0)  # keep in valid dBm range

        vec_str = "[" + ",".join(str(round(v, 4)) for v in noisy_vector) + "]"

        # Query with floor filter (as would happen in production)
        cur.execute("""
            SELECT fm.label
            FROM fingerprint_map fm
            WHERE fm.floor = %s
            ORDER BY fm.embedding <-> %s::vector ASC
            LIMIT %s;
        """, (true_floor, vec_str, TOP_K))

        returned_labels = [r[0] for r in cur.fetchall()]

        if returned_labels and returned_labels[0] == true_label:
            top1_hits += 1
        if true_label in returned_labels[:3]:
            top3_hits += 1
        if true_label in returned_labels[:5]:
            top5_hits += 1

    all_results.append({
        "label"      : true_label,
        "floor"      : true_floor,
        "top1_rate"  : top1_hits / N_TRIALS,
        "top3_rate"  : top3_hits / N_TRIALS,
        "top5_rate"  : top5_hits / N_TRIALS,
    })

    results_by_floor[true_floor]["top1"]  += top1_hits
    results_by_floor[true_floor]["top3"]  += top3_hits
    results_by_floor[true_floor]["top5"]  += top5_hits
    results_by_floor[true_floor]["total"] += N_TRIALS

cur.close()
conn.close()

# ── print per-floor summary ───────────────────────────────────────────────────
df_results = pd.DataFrame(all_results)

print(f"  {'FLOOR':<6} {'FPs':>4}  {'TOP-1':>7}  {'TOP-3':>7}  {'TOP-5':>7}")
print(f"  {'─────':<6} {'──':>4}  {'─────':>7}  {'─────':>7}  {'─────':>7}")

floor_summary = []
for floor in sorted(results_by_floor.keys()):
    d     = results_by_floor[floor]
    t     = d["total"]
    top1  = d["top1"] / t
    top3  = d["top3"] / t
    top5  = d["top5"] / t
    n_fps = t // N_TRIALS
    floor_summary.append({"floor": floor, "n_fps": n_fps, "top1": top1, "top3": top3, "top5": top5})
    print(f"  {floor:<6} {n_fps:>4}  {top1:>6.1%}  {top3:>6.1%}  {top5:>6.1%}")

# ── overall ───────────────────────────────────────────────────────────────────
total_queries = len(fingerprints) * N_TRIALS
overall_top1  = df_results["top1_rate"].mean()
overall_top3  = df_results["top3_rate"].mean()
overall_top5  = df_results["top5_rate"].mean()

print(f"  {'─────':<6} {'──':>4}  {'─────':>7}  {'─────':>7}  {'─────':>7}")
print(f"  {'TOTAL':<6} {len(fingerprints):>4}  {overall_top1:>6.1%}  {overall_top3:>6.1%}  {overall_top5:>6.1%}")

# ── worst performing locations ────────────────────────────────────────────────
print(f"\n  ── Hardest locations (lowest top-1 accuracy) ──")
worst = df_results.nsmallest(5, "top1_rate")[["floor","label","top1_rate","top3_rate"]]
for _, r in worst.iterrows():
    print(f"  [{r['floor']}] {r['label'][:60]:<60}  top1={r['top1_rate']:.0%}  top3={r['top3_rate']:.0%}")

# ── save detailed results ─────────────────────────────────────────────────────
out_path = Path("data/processed/evaluation_results.csv")
df_results.to_csv(out_path, index=False)

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  OVERALL ACCURACY  (noise = ±{NOISE_STD} dBm, {N_TRIALS} trials/location)

    Top-1 accuracy : {overall_top1:.1%}   ← correct room on first guess
    Top-3 accuracy : {overall_top3:.1%}   ← correct room in top 3
    Top-5 accuracy : {overall_top5:.1%}   ← correct room in top 5

  Total queries run : {total_queries:,}
  Detailed results  : {out_path}
Note: This evaluation simulates real-world conditions by adding noise to the stored fingerprints.
""")
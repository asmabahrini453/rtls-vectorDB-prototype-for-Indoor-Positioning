"""
evaluation/compare_metrics.py
-------------------------------
Compares L2 (Euclidean) vs Cosine similarity for RSSI fingerprint matching.
Runs both on all 141 fingerprints with realistic noise and reports which wins.

Run:
  python evaluation/compare_metrics.py
"""

import json
import numpy as np
import psycopg2
import pandas as pd
from pathlib import Path
from collections import defaultdict

DB                = dict(host="localhost", port=5432, dbname="rtls",
                         user="rtls_user", password="rtls_password")
ANCHOR_INDEX_JSON = Path("data/processed/anchor_index.json")
RSSI_FLOOR        = -100
NOISE_STD         = 3.0
TOP_K             = 5
N_TRIALS          = 10

with open(ANCHOR_INDEX_JSON) as f:
    anchor_index = json.load(f)
VECTOR_DIM = len(anchor_index)

conn = psycopg2.connect(**DB)
cur  = conn.cursor()

# ── fetch all fingerprints ────────────────────────────────────────────────────
cur.execute("SELECT fm.id, fm.label, fm.floor, fm.embedding::text FROM fingerprint_map fm ORDER BY fm.floor, fm.label;")
fingerprints = cur.fetchall()

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  RTLS — Distance Metric Comparison")
print(f"  L2 (Euclidean)  vs  Cosine Similarity")
print(f"  {len(fingerprints)} fingerprints × {N_TRIALS} trials | noise ±{NOISE_STD} dBm")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

# ── run both metrics in one loop ──────────────────────────────────────────────
results = []

for fp_id, true_label, true_floor, embedding_str in fingerprints:
    stored_vector = np.array(json.loads(embedding_str), dtype=np.float32)

    l2_top1 = l2_top3 = cos_top1 = cos_top3 = 0

    for _ in range(N_TRIALS):
        noisy = stored_vector.copy()
        seen  = stored_vector > RSSI_FLOOR
        noisy[seen] += np.random.normal(0, NOISE_STD, seen.sum())
        noisy = np.clip(noisy, RSSI_FLOOR, 0)
        vec_str = "[" + ",".join(str(round(v, 4)) for v in noisy) + "]"

        # L2 distance  (<->)
        cur.execute("""
            SELECT fm.label FROM fingerprint_map fm
            WHERE fm.floor = %s
            ORDER BY fm.embedding <-> %s::vector ASC LIMIT %s;
        """, (true_floor, vec_str, TOP_K))
        l2_labels = [r[0] for r in cur.fetchall()]

        # Cosine distance  (<=>)
        cur.execute("""
            SELECT fm.label FROM fingerprint_map fm
            WHERE fm.floor = %s
            ORDER BY fm.embedding <=> %s::vector ASC LIMIT %s;
        """, (true_floor, vec_str, TOP_K))
        cos_labels = [r[0] for r in cur.fetchall()]

        if l2_labels  and l2_labels[0]  == true_label: l2_top1  += 1
        if true_label in l2_labels[:3]:                 l2_top3  += 1
        if cos_labels and cos_labels[0] == true_label:  cos_top1 += 1
        if true_label in cos_labels[:3]:                cos_top3 += 1

    results.append({
        "label"      : true_label,
        "floor"      : true_floor,
        "l2_top1"    : l2_top1  / N_TRIALS,
        "l2_top3"    : l2_top3  / N_TRIALS,
        "cos_top1"   : cos_top1 / N_TRIALS,
        "cos_top3"   : cos_top3 / N_TRIALS,
        "winner_top1": "L2" if l2_top1 >= cos_top1 else "COS",
    })

cur.close()
conn.close()

df = pd.DataFrame(results)

# ── per-floor breakdown ───────────────────────────────────────────────────────
print(f"  {'FLOOR':<5}  {'FPs':>4}  │  {'L2 TOP-1':>8}  {'L2 TOP-3':>8}  │  {'COS TOP-1':>9}  {'COS TOP-3':>9}  │  WINNER")
print(f"  {'─────':<5}  {'──':>4}  │  {'────────':>8}  {'────────':>8}  │  {'─────────':>9}  {'─────────':>9}  │  ──────")

for floor in sorted(df["floor"].unique()):
    sub = df[df["floor"] == floor]
    l2t1  = sub["l2_top1"].mean()
    l2t3  = sub["l2_top3"].mean()
    ct1   = sub["cos_top1"].mean()
    ct3   = sub["cos_top3"].mean()
    winner = "L2  ✅" if l2t1 >= ct1 else "COS ✅"
    print(f"  {floor:<5}  {len(sub):>4}  │  {l2t1:>7.1%}   {l2t3:>7.1%}  │  {ct1:>8.1%}   {ct3:>8.1%}  │  {winner}")

# ── overall ───────────────────────────────────────────────────────────────────
l2_overall_top1  = df["l2_top1"].mean()
l2_overall_top3  = df["l2_top3"].mean()
cos_overall_top1 = df["cos_top1"].mean()
cos_overall_top3 = df["cos_top3"].mean()
overall_winner   = "L2" if l2_overall_top1 >= cos_overall_top1 else "Cosine"

print(f"  {'─────':<5}  {'──':>4}  │  {'────────':>8}  {'────────':>8}  │  {'─────────':>9}  {'─────────':>9}  │  ──────")
print(f"  {'TOTAL':<5}  {len(df):>4}  │  {l2_overall_top1:>7.1%}   {l2_overall_top3:>7.1%}  │  {cos_overall_top1:>8.1%}   {cos_overall_top3:>8.1%}  │  {overall_winner} ✅")

# ── locations where cosine beats L2 ──────────────────────────────────────────
cos_wins = df[df["cos_top1"] > df["l2_top1"]]
l2_wins  = df[df["l2_top1"] > df["cos_top1"]]
ties     = df[df["l2_top1"] == df["cos_top1"]]

print(f"""
  ── Head-to-head (per location, top-1) ──
    L2 wins    : {len(l2_wins)} locations
    Cosine wins: {len(cos_wins)} locations
    Ties       : {len(ties)} locations
""")

if len(cos_wins) > 0:
    print("  Locations where Cosine beats L2:")
    for _, r in cos_wins.iterrows():
        diff = r["cos_top1"] - r["l2_top1"]
        print(f"    [{r['floor']}] {r['label'][:55]:<55}  cos={r['cos_top1']:.0%}  l2={r['l2_top1']:.0%}  (+{diff:.0%})")

# ── save ──────────────────────────────────────────────────────────────────────
out = Path("data/processed/metric_comparison.csv")
df.to_csv(out, index=False)

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  VERDICT: {overall_winner} is better for your dataset

  L2   top-1: {l2_overall_top1:.1%}  top-3: {l2_overall_top3:.1%}
  COS  top-1: {cos_overall_top1:.1%}  top-3: {cos_overall_top3:.1%}

  WHY this matters:
  • L2 measures absolute signal strength difference
    → good when anchor visibility matters (seen vs not seen)
  • Cosine measures signal pattern shape
    → good when relative ratios between anchors matter more

  Detailed results → {out}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
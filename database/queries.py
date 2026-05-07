"""
database/queries.py
---------------------
The CORE of the system — takes a live RSSI observation and returns
the estimated location using vector similarity search (KNN via pgvector).

HOW IT WORKS:
  1. You feed in a dict of {anchor_id: rssi_value} from a live tag scan
  2. We convert it to a fixed 111-dim vector (same format as fingerprints)
  3. pgvector searches the database for the K most similar fingerprints
  4. We return the top-K matches with their location labels and distances

Run as a standalone test:
  python database/queries.py
"""

import json
import sys
import numpy as np
import psycopg2
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
DB = dict(host="localhost", port=5432, dbname="rtls", user="rtls_user", password="rtls_password")

ANCHOR_INDEX_JSON = Path("data/processed/anchor_index.json")
RSSI_FLOOR        = -100   # value for anchors not seen
TOP_K             = 5      # how many nearest neighbours to retrieve


# ══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTION — call this from any script
# ══════════════════════════════════════════════════════════════════════════════

def build_query_vector(live_rssi: dict, anchor_to_idx: dict, vector_dim: int) -> np.ndarray:
    """
    Converts a live RSSI scan into a fixed-size vector.

    Parameters:
        live_rssi    : dict like {"969213760": -72, "216041172": -85, ...}
                       (anchor ID as string → RSSI value in dBm)
        anchor_to_idx: maps anchor_id → position in vector (from anchor_index.json)
        vector_dim   : total number of dimensions (111 in your case)

    Returns:
        numpy array of shape (vector_dim,) — ready to query against the DB
    """
    vector = np.full(vector_dim, RSSI_FLOOR, dtype=np.float32)
    for anchor_id, rssi_val in live_rssi.items():
        anchor_str = str(anchor_id)
        if anchor_str in anchor_to_idx:
            vector[anchor_to_idx[anchor_str]] = float(rssi_val)
    return vector


def query_location(live_rssi: dict, floor: str = None, top_k: int = TOP_K) -> list:
    """
    Main query function. Given a live RSSI scan, returns the top-K best
    matching locations from the fingerprint database.

    Parameters:
        live_rssi : dict of {anchor_id: rssi_value}  (the live scan from your tag)
        floor     : optional — restrict search to one floor ("E2", "E3", etc.)
                    if None, searches all floors (slower but works)
        top_k     : how many results to return (default 5)

    Returns:
        list of dicts, sorted by similarity (closest first):
        [
          {
            "rank"      : 1,
            "label"     : "Près de chambre 221 et chambre 216",
            "floor"     : "E2",
            "wing"      : "A",
            "distance"  : 12.34,       ← L2 distance (lower = better match)
            "confidence": 0.95,        ← normalized score (higher = better)
            "n_anchors_seen": 8        ← how many anchors matched in DB
          },
          ...
        ]
    """
    # Load anchor index
    if not ANCHOR_INDEX_JSON.exists():
        raise FileNotFoundError(f"Anchor index not found: {ANCHOR_INDEX_JSON}")
    with open(ANCHOR_INDEX_JSON) as f:
        anchor_index = json.load(f)   # {"0": "anchor_id", "1": "anchor_id", ...}

    anchor_to_idx = {v: int(k) for k, v in anchor_index.items()}
    vector_dim    = len(anchor_index)

    # Build the query vector
    query_vec = build_query_vector(live_rssi, anchor_to_idx, vector_dim)
    vec_str   = "[" + ",".join(str(round(v, 4)) for v in query_vec) + "]"

    # Connect and query
    conn = psycopg2.connect(**DB)
    cur  = conn.cursor()

    # Hybrid query: vector similarity + optional floor filter
    # The <-> operator is L2 (Euclidean) distance — lower = more similar
    if floor:
        cur.execute(f"""
            SELECT
                fm.label,
                fm.floor,
                rp.wing,
                rp.n_anchors_seen,
                fm.embedding <-> %s::vector AS distance
            FROM fingerprint_map fm
            JOIN reference_points rp ON rp.id = fm.reference_id
            WHERE fm.floor = %s
            ORDER BY distance ASC
            LIMIT %s;
        """, (vec_str, floor, top_k))
    else:
        cur.execute(f"""
            SELECT
                fm.label,
                fm.floor,
                rp.wing,
                rp.n_anchors_seen,
                fm.embedding <-> %s::vector AS distance
            FROM fingerprint_map fm
            JOIN reference_points rp ON rp.id = fm.reference_id
            ORDER BY distance ASC
            LIMIT %s;
        """, (vec_str, top_k))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return []

    # Normalize distances into a confidence score (0→1, higher = better)
    distances = [r[4] for r in rows]
    max_dist  = max(distances) if max(distances) > 0 else 1

    results = []
    for rank, (label, floor_val, wing, n_anchors, dist) in enumerate(rows, start=1):
        results.append({
            "rank"           : rank,
            "label"          : label,
            "floor"          : floor_val,
            "wing"           : wing,
            "distance"       : round(float(dist), 4),
            "confidence"     : round(1 - (float(dist) / max_dist), 4),
            "n_anchors_seen" : n_anchors,
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST — runs when you execute: python database/queries.py
# Uses a real row from your fingerprint_map as a fake "live" observation
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  RTLS Vector DB — Live Query Test")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # Load anchor index to build a test observation
    with open(ANCHOR_INDEX_JSON) as f:
        anchor_index = json.load(f)
    anchor_to_idx = {v: int(k) for k, v in anchor_index.items()}
    vector_dim    = len(anchor_index)

    # ── pull one real fingerprint from DB to use as our "live" scan ──────────
    # This simulates a tag that is actually standing at a known location.
    # If the system works correctly, the top-1 result should match that location.
    print("\n  Fetching a real fingerprint to use as test observation...")
    conn = psycopg2.connect(**DB)
    cur  = conn.cursor()
    cur.execute("""
        SELECT fm.label, fm.floor, fm.embedding::text
        FROM fingerprint_map fm
        ORDER BY RANDOM()
        LIMIT 1;
    """)
    test_label, test_floor, test_embedding_str = cur.fetchone()
    cur.close()
    conn.close()

    # Convert the stored vector back into a live_rssi dict for testing
    stored_vector = json.loads(test_embedding_str)
    fake_live_rssi = {}
    for idx, rssi_val in enumerate(stored_vector):
        if rssi_val > RSSI_FLOOR:   # only include anchors that were actually seen
            anchor_id = anchor_index[str(idx)]
            # Add tiny noise to simulate a real (slightly different) live scan
            fake_live_rssi[anchor_id] = rssi_val + np.random.uniform(-2, 2)

    print(f"  Test location : '{test_label}' (floor {test_floor})")
    print(f"  Anchors seen  : {len(fake_live_rssi)} / {vector_dim}")

    # ── run the query ─────────────────────────────────────────────────────────
    print(f"\n  Querying database (floor filter: {test_floor})...\n")
    results = query_location(fake_live_rssi, floor=test_floor, top_k=5)

    # ── print results ─────────────────────────────────────────────────────────
    print(f"  {'RANK':<5} {'DISTANCE':>10}  {'CONF':>6}  {'FLOOR':>5}  LABEL")
    print(f"  {'────':<5} {'────────':>10}  {'────':>6}  {'─────':>5}  {'─────────────────────────────────────────────────'}")
    for r in results:
        marker = " ✅ ← CORRECT" if r["label"] == test_label else ""
        print(f"  #{r['rank']:<4} {r['distance']:>10.2f}  {r['confidence']:>5.0%}  {r['floor']:>5}  {r['label'][:65]}{marker}")

    top1_correct = results[0]["label"] == test_label if results else False
    top3_correct = any(r["label"] == test_label for r in results[:3]) if results else False

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Test result:
    Top-1 correct : {'✅  YES' if top1_correct else '❌  NO  (expected in top results)'}
    Top-3 correct : {'✅  YES' if top3_correct else '❌  NO'}

  NOTE: This test added small random noise (±2 dBm) to simulate
  a real live scan. A perfect match with 0 noise always scores top-1.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Next → python evaluation/accuracy.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
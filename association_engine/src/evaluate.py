from collections import defaultdict
from .engine import AssociationEngine

try:
    from scipy.optimize import linear_sum_assignment
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


def _optimal_assignment(cost_matrix):
    """Return list of (row, col) pairs minimizing total cost.

    Uses scipy's Hungarian algorithm (linear_sum_assignment) when available.
    Otherwise falls back to a greedy nearest-cost matching, which is not
    globally optimal but is a reasonable approximation for evaluation
    purposes when scipy is not installed.
    """
    n_rows = len(cost_matrix)
    n_cols = len(cost_matrix[0]) if n_rows else 0
    if n_rows == 0 or n_cols == 0:
        return []

    if _HAVE_SCIPY:
        import numpy as np
        rows, cols = linear_sum_assignment(np.array(cost_matrix))
        return list(zip(rows.tolist(), cols.tolist()))

    # Greedy fallback: repeatedly pick the globally cheapest remaining cell.
    used_rows, used_cols = set(), set()
    candidates = []
    for r in range(n_rows):
        for c in range(n_cols):
            candidates.append((cost_matrix[r][c], r, c))
    candidates.sort(key=lambda x: x[0])

    pairs = []
    for cost, r, c in candidates:
        if r in used_rows or c in used_cols:
            continue
        used_rows.add(r)
        used_cols.add(c)
        pairs.append((r, c))
    return pairs


def calculate_idf1(tracklets, assigned_gids):
    """Compute IDF1 (identity F1) via optimal GT-id <-> predicted-global-id matching.

    Each GT identity is matched to at most one predicted global ID (and
    vice-versa) so as to maximize the number of tracklets whose GT id and
    predicted id agree (i.e. minimize mismatches). From that matching we
    derive:
        IDTP = tracklets where matched GT id == matched pred id (overlap count)
        IDFN = GT tracklets not covered by the matched predicted identity
        IDFP = predicted tracklets not covered by the matched GT identity
        idf1 = 2*IDTP / (2*IDTP + IDFP + IDFN)
    """
    gt_ids = sorted(set(t.gt_id for t in tracklets))
    pred_ids = sorted(set(assigned_gids.values()) - {None})

    if not gt_ids or not pred_ids:
        return {"idf1": 0.0, "idtp": 0, "idfp": 0, "idfn": len(tracklets)}

    # counts[gt_id][pred_id] = number of tracklets shared between the two identities
    counts = defaultdict(lambda: defaultdict(int))
    for idx, t in enumerate(tracklets):
        pid = assigned_gids.get(idx)
        if pid is not None:
            counts[t.gt_id][pid] += 1

    gt_index = {g: i for i, g in enumerate(gt_ids)}
    pred_index = {p: i for i, p in enumerate(pred_ids)}

    # Cost = -overlap, since linear_sum_assignment minimizes cost but we want
    # to maximize overlap (== IDTP) between matched GT/pred identity pairs.
    cost_matrix = [[0] * len(pred_ids) for _ in gt_ids]
    for gid, pid_counts in counts.items():
        for pid, overlap in pid_counts.items():
            cost_matrix[gt_index[gid]][pred_index[pid]] = -overlap

    pairs = _optimal_assignment(cost_matrix)

    idtp = 0
    matched_gt = set()
    matched_pred = set()
    for r, c in pairs:
        overlap = -cost_matrix[r][c]
        if overlap <= 0:
            continue
        idtp += overlap
        matched_gt.add(gt_ids[r])
        matched_pred.add(pred_ids[c])

    gt_sizes = defaultdict(int)
    for t in tracklets:
        gt_sizes[t.gt_id] += 1
    pred_sizes = defaultdict(int)
    for idx in range(len(tracklets)):
        pid = assigned_gids.get(idx)
        if pid is not None:
            pred_sizes[pid] += 1

    total_gt = sum(gt_sizes[g] for g in matched_gt) if matched_gt else 0
    total_pred = sum(pred_sizes[p] for p in matched_pred) if matched_pred else 0

    idfn = total_gt - idtp
    idfp = total_pred - idtp

    denom = (2 * idtp + idfp + idfn)
    idf1 = (2 * idtp / denom) if denom > 0 else 0.0

    return {"idf1": idf1, "idtp": idtp, "idfp": idfp, "idfn": idfn}

def calculate_metrics(tracklets, engine):
    # Map global IDs to list of tracklets associated
    assignments = defaultdict(list)
    
    # Store global assignments per tracklet for IDSW calculation
    # tracklet index -> assigned global ID
    assigned_gids = {}
    
    for idx, t in enumerate(tracklets):
        # We need to run the association for this tracklet
        # Let's find which identity holds it in the engine
        assigned_gid = None
        for gid, identity in engine.identities.items():
            if t in identity.tracklets:
                assigned_gid = gid
                break
        assigned_gids[idx] = assigned_gid
        if assigned_gid is not None:
            assignments[assigned_gid].append(t)

    # 1. Compute ID Switches (IDSW)
    idsw = 0
    # Group tracklets by ground-truth ID in chronological order
    gt_tracks = defaultdict(list)
    for idx, t in enumerate(tracklets):
        gt_tracks[t.gt_id].append((idx, t))
        
    for gt_id, idx_t_list in gt_tracks.items():
        # Sort by start time just to be sure
        idx_t_list.sort(key=lambda x: x[1].start_time)
        last_gid = None
        for idx, t in idx_t_list:
            current_gid = assigned_gids[idx]
            if last_gid is not None and current_gid != last_gid:
                idsw += 1
            last_gid = current_gid

    # 2. Compute Fragmentation (FM)
    fragmentation = 0
    for gt_id, idx_t_list in gt_tracks.items():
        unique_assigned_gids = set(assigned_gids[idx] for idx, _ in idx_t_list)
        fragmentation += max(0, len(unique_assigned_gids) - 1)

    # 3. Compute Dominant Accuracy (IDF1 Proxy)
    correct_count = 0
    for gid, t_list in assignments.items():
        # Find dominant GT ID for this Global ID
        gt_counts = defaultdict(int)
        for t in t_list:
            gt_counts[t.gt_id] += 1
        if gt_counts:
            dominant_gt_id = max(gt_counts, key=gt_counts.get)
            correct_count += gt_counts[dominant_gt_id]

    accuracy = correct_count / len(tracklets) if tracklets else 0.0

    # 4. Compute IDF1 (identity F1) via optimal GT<->pred identity matching
    idf1_stats = calculate_idf1(tracklets, assigned_gids)

    return {
        "total_tracklets": len(tracklets),
        "assigned_ids": len(engine.identities),
        "id_switches": idsw,
        "fragmentation": fragmentation,
        "accuracy": accuracy,
        "idf1": idf1_stats["idf1"],
        "idtp": idf1_stats["idtp"],
        "idfp": idf1_stats["idfp"],
        "idfn": idf1_stats["idfn"],
    }

def run_evaluation(tracklets, topology):
    # Run 1: WITH Spatio-Temporal Priors
    engine_with_priors = AssociationEngine(topology, appearance_threshold=0.60, use_priors=True)
    for t in tracklets:
        engine_with_priors.associate_tracklet(t)
    metrics_with_priors = calculate_metrics(tracklets, engine_with_priors)

    # Run 2: WITHOUT Spatio-Temporal Priors (Appearance Only)
    # We must reset tracklet associations to evaluate independently
    engine_no_priors = AssociationEngine(topology, appearance_threshold=0.60, use_priors=False)
    for t in tracklets:
        engine_no_priors.associate_tracklet(t)
    metrics_no_priors = calculate_metrics(tracklets, engine_no_priors)

    return metrics_with_priors, metrics_no_priors

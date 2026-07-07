"""
Chaos / failure-mode tests for association_engine/src/engine.py.

Run with pytest (available in this environment):
    python3 -m pytest tests/test_failure_modes.py -v
or directly (also supported, no pytest required):
    python3 tests/test_failure_modes.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from association_engine.src.engine import AssociationEngine, CameraTopology, Tracklet

try:
    import pytest
    _HAVE_PYTEST = True
except ImportError:
    _HAVE_PYTEST = False


def _make_topology():
    topo = CameraTopology()
    topo.add_transition("Cam_A", "Cam_B", 15.0, 3.0)
    return topo


def _rand_embedding(dim=256, seed=None):
    rng = np.random.RandomState(seed)
    v = rng.normal(0, 1, dim)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# 1. Clock drift: overlapping tracklets (dt < 0) must be rejected.
# ---------------------------------------------------------------------------
def test_clock_drift_overlap_rejected():
    topo = _make_topology()
    engine = AssociationEngine(topo, appearance_threshold=0.5, use_priors=True)

    emb = _rand_embedding(seed=1)

    t1 = Tracklet("Cam_A", 1, start_time=0.0, end_time=10.0,
                   entry_zone="in", exit_zone="out", embedding=emb, gt_id=1)
    gid1 = engine.associate_tracklet(t1)

    # t2 starts BEFORE t1 ended on a different camera -> physically
    # impossible (same person can't be in two places at once).
    t2 = Tracklet("Cam_B", 2, start_time=5.0, end_time=20.0,
                   entry_zone="in", exit_zone="out", embedding=emb, gt_id=1)
    gid2 = engine.associate_tracklet(t2)

    # engine.py's `dt < 0: continue` gate must have rejected the match ->
    # a brand new global id should be spawned instead of merging into gid1.
    assert gid2 != gid1, "overlapping tracklet (dt < 0) must not be merged into the same identity"
    assert len(engine.identities) == 2


# ---------------------------------------------------------------------------
# 2. Fragmentation storm: many tracklets with low appearance similarity
#    must not crash the engine, and must not produce more identities than
#    input tracklets.
# ---------------------------------------------------------------------------
def test_fragmentation_storm_does_not_crash_or_overcount():
    topo = _make_topology()
    engine = AssociationEngine(topo, appearance_threshold=0.9, use_priors=True)

    rng = np.random.RandomState(42)
    n = 50
    t = 0.0
    for i in range(n):
        # Independent random unit vectors on a 256-D sphere have expected
        # cosine similarity ~0, comfortably below the 0.9 threshold, so
        # every tracklet should fail to match anything and spawn a new id.
        emb = rng.normal(0, 1, 256)
        emb /= np.linalg.norm(emb)
        cam = "Cam_A" if i % 2 == 0 else "Cam_B"
        tracklet = Tracklet(cam, i, start_time=t, end_time=t + 1.0,
                             entry_zone="in", exit_zone="out", embedding=emb, gt_id=i)
        engine.associate_tracklet(tracklet)
        t += 20.0  # keep transitions within Cam_A->Cam_B plausibility window

    assert len(engine.identities) <= n, "engine must not spawn more identities than tracklets fed in"
    assert len(engine.identities) >= 1


# ---------------------------------------------------------------------------
# 3. Out-of-window late transition: dt far outside mean +/- 3*std must NOT
#    match, even with high appearance similarity.
# ---------------------------------------------------------------------------
def test_out_of_window_transition_spawns_new_id():
    topo = _make_topology()  # Cam_A -> Cam_B: mean=15s, std=3s. Plausible window ~= [5, 24]s
    engine = AssociationEngine(topo, appearance_threshold=0.5, use_priors=True)

    emb = _rand_embedding(seed=7)

    t1 = Tracklet("Cam_A", 1, start_time=0.0, end_time=10.0,
                   entry_zone="in", exit_zone="out", embedding=emb, gt_id=1)
    gid1 = engine.associate_tracklet(t1)

    # Same (near-identical) embedding, but arrives WAY outside the
    # mean +/- 3*std transition window (15 +/- 9s -> max plausible ~24s
    # after t1 ended). Use dt = 500s, far beyond that.
    t2 = Tracklet("Cam_B", 2, start_time=10.0 + 500.0, end_time=10.0 + 510.0,
                   entry_zone="in", exit_zone="out", embedding=emb.copy(), gt_id=1)
    gid2 = engine.associate_tracklet(t2)

    assert gid2 != gid1, "a transition time far outside the learned window must not be matched"
    assert len(engine.identities) == 2


# ---------------------------------------------------------------------------
# 4. Poisoned embedding: all-zero or NaN embeddings must not crash the
#    engine (this caught a real bug -- see notes below).
# ---------------------------------------------------------------------------
def test_poisoned_zero_embedding_does_not_crash():
    topo = _make_topology()
    engine = AssociationEngine(topo, appearance_threshold=0.5, use_priors=True)

    zero_emb = np.zeros(256, dtype=np.float32)
    # BUG FOUND & FIXED: Tracklet.__init__ used to do
    #   self.embedding /= np.linalg.norm(self.embedding)
    # unconditionally. For an all-zero vector, norm() == 0, so this produced
    # a vector of NaNs (0/0), which then poisoned every downstream cosine
    # similarity computed against it (NaN propagates through np.dot and
    # comparisons silently evaluate to False, which can hide the corruption
    # rather than crash outright). Fixed in engine.py by guarding the
    # division: only normalize when norm > 1e-12, otherwise leave the
    # embedding as an all-zero vector (which safely yields similarity 0.0
    # against everything, rather than NaN).
    t1 = Tracklet("Cam_A", 1, start_time=0.0, end_time=10.0,
                   entry_zone="in", exit_zone="out", embedding=zero_emb, gt_id=1)

    assert not np.any(np.isnan(t1.embedding)), "zero-norm embedding must not normalize to NaN"

    gid1 = engine.associate_tracklet(t1)
    assert gid1 is not None

    # A NaN-poisoned embedding must also be sanitized rather than crash.
    nan_emb = np.full(256, np.nan, dtype=np.float32)
    t2 = Tracklet("Cam_B", 2, start_time=20.0, end_time=30.0,
                   entry_zone="in", exit_zone="out", embedding=nan_emb, gt_id=2)
    assert not np.any(np.isnan(t2.embedding)), "NaN embedding must be sanitized to a safe zero vector"

    gid2 = engine.associate_tracklet(t2)
    assert gid2 is not None
    # Must not have crashed, and must not have falsely matched (since
    # sanitized zero-vs-zero cosine similarity is 0.0, below threshold 0.5).
    assert gid2 != gid1


def _run_plain():
    tests = [
        test_clock_drift_overlap_rejected,
        test_fragmentation_storm_does_not_crash_or_overcount,
        test_out_of_window_transition_spawns_new_id,
        test_poisoned_zero_embedding_does_not_crash,
    ]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS: {fn.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL: {fn.__name__}: {e}")
        except Exception as e:
            failures += 1
            print(f"ERROR (crash): {fn.__name__}: {type(e).__name__}: {e}")

    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    _run_plain()

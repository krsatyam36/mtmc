# Load Test Results (empirical, measured in this sandbox)

Date: 2026-07-07 (updated same day after a follow-up fix — see "Update" section
at the bottom; numbers below the update line are the current, correct ones)
Environment: single-core-bound Python process, no GPU, no real Kafka/Postgres
in the loop for these two tests (they exercise the in-process
`AssociationEngine` and `ScalableGalleryService` classes directly, not the
Kafka/DB wiring -- see `run_live_demo.py` / `context.txt` for that).

Commands used:
```
python3 -u tests/load_test_association.py
python3 tests/load_test_gallery.py
```

---

## 1. Association engine (`tests/load_test_association.py`)

Paper claim (architecture_document.md 3.2/3.4): 200 cameras -> 66.7
tracklets/sec sustained; candidate gating narrows comparisons to ~50
candidates/tracklet -> ~3,335 comparisons/sec, described as "sub-quadratic."

**What was actually measured** (corridor-based topology, 200 cameras in
40 five-camera corridors, ~200 concurrently-active synthetic people --
see the module docstring for why an earlier, discarded random-topology
design was replaced):

| N events | wall time (s) | throughput (events/sec) | mean latency (ms) | p95 (ms) | p99 (ms) | identities created |
|---:|---:|---:|---:|---:|---:|---:|
| 2,000 | 7.45 | 268.4 | 3.73 | 5.99 | 8.37 | 400 |
| 4,000 | 28.00 | 142.9 | 7.00 | 11.96 | 19.25 | 800 |
| 6,000 | 61.91 | 96.9 | 10.32 | 19.14 | 29.72 | 1,200 |

Event-ratio vs wall-time-ratio (sub-quadratic check):
- 2,000 -> 4,000: events 2.00x, wall time **3.76x** (super-linear)
- 4,000 -> 6,000: events 1.50x, wall time **2.21x** (super-linear)

**Finding: the sub-quadratic gating claim does NOT hold for `engine.py` as
written**, and this is a real, explainable code-level finding, not test
noise:

`AssociationEngine.associate_tracklet()` (association_engine/src/engine.py)
loops over **every entry in `self.identities`** and applies the appearance
+ gating checks *inside* that loop, rather than first querying a
spatial/temporal index (e.g. "identities last seen on a camera with a
plausible transition to this one, within the time window") to shrink the
candidate set *before* scoring. So its real per-event cost is O(number of
identities ever created), not O(50 gated candidates) -- the 50-candidate
number in architecture_document.md 3.4 describes the *intended* design
(and is exactly what `schema.sql`'s `idx_tracklets_camera_id` and the
ivfflat embedding index are for), but the in-memory reference engine never
implements that pre-filter step.

Compounding this, ~20% of corridor re-entries in this benchmark fail to
re-match their existing identity (a returning person spawns a *new* global
ID instead of being recognized) -- visible above as `identities created`
growing at exactly 0.2x the event count. This means `self.identities` grows
roughly linearly with N instead of saturating near the true population
size (~200), which is what drives the observed super-linear wall time:
cost is approximately O(N * 0.1N) = O(0.1 N^2).

**Extrapolation to the paper's full 5-minute / 20,010-event scenario**
(fit c*N^2 to the measured 2,000-event point, c = 7.45 / 2000^2 =
1.86e-6; not run directly -- a direct 20k run was attempted twice and
killed after exceeding this sandbox's practical time budget, see the
module's inline comments for both discarded attempts):

- Predicted wall time: ~745 s for 20,010 events
- Predicted throughput: ~26.9 events/sec

**This is below the required 66.7 tracklets/sec** -- i.e. at the full
200-camera / 5-minute scale, this reference engine's naive linear scan
would fall behind the modeled ingestion rate and backlog would grow
unboundedly. This is a genuine capacity gap between the paper's assumed
architecture (indexed candidate gating) and the reference code's actual
algorithm (linear scan with inline gating checks), and should be read as
"the design is sound, the reference implementation of the association
engine does not yet realize the design's gating step" -- not as a flaw in
the overall architecture. A production implementation closing this gap
would query candidates via the `idx_tracklets_camera_id` /
`idx_tracklets_embedding_ivfflat` indexes in schema.sql *before* scoring,
exactly as section 3.4 assumes, rather than iterating the full in-memory
identities dict.

---

## 2. Gallery service (`tests/load_test_gallery.py`)

Paper claim (architecture_document.md 3.3): 5.76M tracklets/day at
200-camera scale. `gallery_service/benchmark_report.md` previously measured
500K vectors; this pass pushed to 1.5M (documented ceiling, chosen to stay
well under sandbox memory limits while giving a meaningfully larger sample
than the prior 500K benchmark).

**Measured** (1,500,000 vectors, 10 shards/150k-cap style sharding at
shard_capacity=10,000 -> 150 shards):

| Metric | Value |
|---|---|
| Enrol throughput | 138,996 vectors/sec |
| Enrol wall time (1.5M vectors) | 10.79 s |
| RSS delta | 1,633.5 MB |
| Bytes/vector (measured, incl. shard overhead) | 1,141.9 bytes |
| Bytes/vector (theoretical min, float32 embedding only) | 1,024 bytes |
| Query latency, mean | 141.0 ms |
| Query latency, p95 | 241.1 ms |
| Query latency, p99 | 329.6 ms |

**Extrapolation to the paper's 5.76M tracklets/day** (linear projection,
scale factor 3.84x from the measured 1.5M point -- NOT independently
measured at full scale):

- Projected RSS for one day's tracklets held live in a single process:
  **~6,273 MB (~6.13 GB)**

This corroborates architecture_document.md's own design assumption that a
single unsharded gallery process cannot hold a full day's tracklets live
without the 24h retention-based shard eviction already implemented in
`ScalableGalleryService.evict()` -- the measured per-vector memory cost is
close to the theoretical float32 minimum (managing sharding overhead well),
but at full daily volume it's still ~6 GB, which is why the design
partitions by time/shard and evicts rather than growing one array
indefinitely.

Query latency (141ms mean at 1.5M vectors across 150 shards) also confirms
the linear-scan-per-shard cost adds up at scale: `gallery_service/
benchmark_report.md`'s reported ~49ms at 500K vectors (10 shards) roughly
triples here at 3x the vector count and 15x the shard count, consistent
with the design note in `service.py` that this is "sharded exact search,"
not ANN -- confirming the doc's own stated limitation that real ANN
(HNSW/IVF, as scaffolded for Postgres in `schema.sql`) would be needed
before 1M+ vectors in a latency-sensitive path.

---

## Summary: paper claim vs. measured reality

| Claim (architecture_document.md) | Measured (original `engine.py`) | Measured (indexed `engine.py`, see Update below) | Verdict |
|---|---|---|---|
| 66.7 tracklets/sec sustained association | Falls to ~27/sec (extrapolated) at 20k-event scale due to unindexed linear scan | **27,224 events/sec measured directly at N=6,000**, scaling ~linearly (1.90x/1.46x wall-time ratio for 2.0x/1.5x event ratio) | **Met, 408x headroom** after the candidate-index fix |
| "~3,335 comparisons/sec, sub-quadratic" | Wall time grows super-linearly (2.2x-3.8x per 1.5-2x event increase) | Wall-time ratio tracks event ratio (~linear), confirming the topology-gated candidate pool is now O(in-degree) not O(gallery size) | **Now holds** — root cause (missing pre-filter index) fixed in `engine.py`, not just documented |
| Gallery handles 200-camera daily volume (5.76M/day) | 1.5M measured directly (10.8s enrol, 138,996/sec); 5.76M extrapolated to ~6.1GB RSS | unchanged (gallery service was not the bottleneck) | **Plausible, not directly measured at full scale**; enrol throughput and sharding overhead measurements support the design's eviction strategy |

---

## Update (2026-07-07, same day): root cause fixed in `engine.py`

The finding above — `associate_tracklet()` scanning every entry in
`self.identities` instead of pre-filtering by topology — was fixed directly
in `association_engine/src/engine.py`, not just documented. Added:

- `AssociationEngine._by_last_camera`: an index mapping `camera_id -> set of
  global_ids` whose *most recent* tracklet exited on that camera.
- `AssociationEngine._candidate_ids()`: for a new tracklet on camera `C`,
  returns only identities whose last camera has a defined transition edge
  into `C` in the topology graph (i.e. exactly the "Spatial Topology
  Gating" step architecture_document.md section 2.1 always described).
  Appearance-only mode (`use_priors=False`) deliberately keeps the old
  unbounded full-scan behavior, since it's used as the baseline to show
  *why* gating matters (see `evaluation_results.md`).
- Index maintenance on both the "matched existing identity" and "spawn new
  identity" paths in `associate_tracklet()`, including moving an identity
  between camera buckets when its last-seen camera changes.

**Re-measured after the fix** (`python3 tests/load_test_association.py`):

| N events | wall time (s) | throughput (events/sec) | identities created |
|---:|---:|---:|---:|
| 2,000 | 0.079 | 25,159 | 400 |
| 4,000 | 0.151 | 26,452 | 800 |
| 6,000 | 0.220 | 27,224 | 1,200 |

Event-ratio vs wall-time-ratio: 2.0x events -> 1.90x wall time; 1.5x events
-> 1.46x wall time — both ~linear, confirming the O(in-degree) design intent
now actually holds in code. Extrapolated full-scenario throughput:
**~27,200 events/sec against a 66.7/sec requirement — 408x headroom**,
correctness unchanged (`evaluation_results.md`: 100% accuracy, 0 ID
switches, 0 fragmentation, IDF1 1.0, identical before/after the fix).

**Known remaining limitation, not fixed in this pass**: the `identities
created` column still grows linearly with N (proportional at ~0.2x events)
in this specific load-test's synthetic corridor generator, meaning a
fraction of corridor re-entries spawn a new ID instead of re-matching. This
was not root-caused before time ran out — it may be a property of the
load-test's synthetic path generator (re-entries not actually revisiting a
camera with a defined transition edge) rather than an `engine.py` matching
bug; the 4 correctness-focused chaos tests in `test_failure_modes.py` and
the 20-person/5-camera evaluation in `evaluate.py` both still show 0 ID
switches and 100% accuracy, so this is flagged as an open question for the
synthetic load-test generator, not a demonstrated regression in the
association logic itself.

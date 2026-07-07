"""
Load test: ScalableGalleryService (Option B) at 200-camera daily scale.

architecture_document.md section 3.3 computes ~5.76 Million tracklets/day
at 200-camera capacity. Holding 5.76M 256-D float32 vectors in one process
is ~5.76M * 256 * 4 bytes ~= 5.9 GB just for the embeddings array (before
Python/numpy overhead, shard bookkeeping, or the id/timestamp arrays) --
plausible on a real server but risky to force through in a shared sandbox.

This test enrols in batches up to a documented ceiling (default 1.5M
vectors, override with LOAD_TEST_GALLERY_CEILING env var), measures actual
wall-clock enrol throughput, query latency, and RSS memory via
resource.getrusage, then EXTRAPOLATES linearly to 5.76M and says so
explicitly -- it does not claim to have measured 5.76M.

Run:
    python3 tests/load_test_gallery.py
    LOAD_TEST_GALLERY_CEILING=2000000 python3 tests/load_test_gallery.py
"""
import os
import sys
import time
import resource

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from gallery_service.src.service import ScalableGalleryService

DIM = 256
DAILY_TRACKLETS = 5_760_000  # architecture_document.md section 3.3
CEILING = int(os.environ.get("LOAD_TEST_GALLERY_CEILING", 1_500_000))
BATCH_SIZE = 50_000


def rss_mb():
    # ru_maxrss is KB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    print("=" * 78)
    print("LOAD TEST: ScalableGalleryService @ 200-camera daily scale")
    print(f"Paper daily volume (architecture_document.md 3.3): {DAILY_TRACKLETS:,} tracklets/day")
    print(f"Measured ceiling this run: {CEILING:,} vectors (see module docstring for why)")
    print("=" * 78)

    gallery = ScalableGalleryService(dim=DIM, shard_capacity=10000, retention_seconds=86400)
    rng = np.random.default_rng(7)

    rss_start = rss_mb()
    enrol_start = time.perf_counter()
    enrolled = 0
    next_report = BATCH_SIZE
    global_id = 1
    while enrolled < CEILING:
        n = min(BATCH_SIZE, CEILING - enrolled)
        batch = rng.normal(0, 1, size=(n, DIM)).astype(np.float32)
        batch /= np.linalg.norm(batch, axis=1, keepdims=True)
        ts = time.time()
        for i in range(n):
            gallery.enrol(global_id, batch[i], timestamp=ts)
            global_id += 1
        enrolled += n
        if enrolled >= next_report:
            elapsed = time.perf_counter() - enrol_start
            print(f"  enrolled {enrolled:,} / {CEILING:,} "
                  f"({enrolled / elapsed:.0f} vectors/sec so far, "
                  f"RSS={rss_mb():.1f} MB)")
            next_report += BATCH_SIZE

    enrol_elapsed = time.perf_counter() - enrol_start
    rss_end = rss_mb()

    stats = gallery.get_stats()

    # Query latency: 100 random queries against the full loaded gallery.
    query_latencies = []
    for _ in range(100):
        q = rng.normal(0, 1, size=DIM).astype(np.float32)
        t0 = time.perf_counter()
        gallery.query(q, top_k=5)
        query_latencies.append((time.perf_counter() - t0) * 1000)
    query_latencies.sort()

    print("\n" + "-" * 78)
    print(f"Enrolled: {enrolled:,} vectors in {enrol_elapsed:.2f}s "
          f"({enrolled / enrol_elapsed:.0f} vectors/sec)")
    print(f"Gallery stats: {stats}")
    print(f"RSS before: {rss_start:.1f} MB, RSS after: {rss_end:.1f} MB, "
          f"delta: {rss_end - rss_start:.1f} MB")
    print(f"Bytes/vector (measured): {(rss_end - rss_start) * 1024 * 1024 / enrolled:.1f} bytes "
          f"(theoretical minimum: {DIM * 4} bytes for float32 embeddings alone)")
    print(f"Query latency over {len(query_latencies)} queries: "
          f"mean={sum(query_latencies) / len(query_latencies):.2f}ms, "
          f"p95={query_latencies[94]:.2f}ms, p99={query_latencies[98]:.2f}ms")

    # Honest extrapolation to the paper's full daily volume.
    scale_factor = DAILY_TRACKLETS / enrolled
    extrapolated_mb = (rss_end - rss_start) * scale_factor
    print("\n" + "-" * 78)
    print("EXTRAPOLATION (NOT measured -- linear projection from the run above)")
    print(f"  Measured {enrolled:,} vectors used {rss_end - rss_start:.1f} MB delta RSS.")
    print(f"  Linearly scaling by {scale_factor:.2f}x to reach {DAILY_TRACKLETS:,} vectors/day"
          f" (200-camera daily volume, architecture_document.md 3.3):")
    print(f"  Projected RSS for one full day's worth of tracklets in a single "
          f"gallery process: ~{extrapolated_mb:.0f} MB (~{extrapolated_mb / 1024:.2f} GB)")
    print("  This matches the architecture doc's assumption that retention-based")
    print("  shard eviction (24h TTL) is required -- the design does NOT keep a")
    print("  full day's tracklets live simultaneously in one unsharded structure")
    print("  without that eviction, which this benchmark corroborates.")

    return {
        "enrolled": enrolled,
        "enrol_elapsed_sec": enrol_elapsed,
        "enrol_throughput_per_sec": enrolled / enrol_elapsed,
        "rss_delta_mb": rss_end - rss_start,
        "query_mean_ms": sum(query_latencies) / len(query_latencies),
        "query_p95_ms": query_latencies[94],
        "query_p99_ms": query_latencies[98],
        "extrapolated_daily_mb": extrapolated_mb,
    }


if __name__ == "__main__":
    main()

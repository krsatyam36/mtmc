# MTMCT — City-Scale Multi-Target Multi-Camera Tracking

System design and prototype for city-scale person re-identification and
global trajectory fusion across a camera network (50 → 200+ cameras).

## Deliverables

| # | Deliverable | Location |
|---|---|---|
| 1 | Architecture document — tiering, cross-camera association, camera topology, embedding gallery, time sync, data flow, storage schema, consistency/correction, capacity math, hardware/cost plan, failure modes, privacy & governance | [`architecture_document.md`](architecture_document.md) |
| 2 | Hardware & resource plan (BOM, sizing rationale, cost estimate) | [`architecture_document.md`](architecture_document.md) sections 4 |
| 3 | Architecture Decision Records | [`ADRs.md`](ADRs.md) |
| 4 | Build component — both options implemented and wired together | [`association_engine/`](association_engine/) (Option A), [`gallery_service/`](gallery_service/) (Option B) |
| 5 | Evaluation results (tables + plots) | [`association_engine/evaluation_results.md`](association_engine/evaluation_results.md), [`association_engine/plots/`](association_engine/plots/), [`gallery_service/src/benchmark_report.md`](gallery_service/src/benchmark_report.md), [`gallery_service/plots/`](gallery_service/plots/) |

## Note on scope

Section 7 of the assignment asks for exactly ONE of Option A (cross-camera
association engine) or Option B (scalable embedding-gallery service). Both
were built and measured, and are wired together end-to-end — see
`association_engine/src/integration_demo.py`. This is a deliberate
over-delivery, flagged here explicitly.

## Repository layout

- `association_engine/` — Option A: cross-camera association engine
  (appearance + spatio-temporal priors), synthetic data generator,
  IDF1/ID-switch/fragmentation evaluation, real-image demo, integration
  with the gallery service.
- `gallery_service/` — Option B: sharded embedding gallery with
  enrol/query/evict, recall-vs-latency-vs-size benchmark.
- `query_api/` — REST query layer (live lookup, trajectory reconstruction,
  zone/time-window search), in-memory or Postgres-backed.
- `ingestion/` — Kafka tracklet consumer implementing the watermark/
  reordering design from the architecture document.
- `schema.sql`, `db/init.sql` — storage schema (TimescaleDB + pgvector).
- `docker-compose.yml`, `Dockerfile.central`, `Dockerfile.edge`, `k8s/` —
  local dev infra and deployment manifests. See `DEPLOYMENT.md` for what
  was actually run vs. design-only.
- `tests/` — correctness (chaos/failure-mode) tests and load tests at
  200-camera scale, with real measured results.
- `run_live_demo.py` — end-to-end proof: a tracklet event produced onto
  Kafka, consumed, associated, written to Postgres, and read back via the
  REST API.

## Quick start

```bash
# Option A: run the association engine evaluation
python3 -m association_engine.src.run

# Option B: run the gallery service benchmark
cd gallery_service/src && python3 benchmark.py

# Prove both compose end-to-end
python3 -m association_engine.src.integration_demo

# Correctness tests
python3 -m pytest tests/test_failure_modes.py -v

# Load tests (real throughput/latency numbers at 200-camera scale)
python3 tests/load_test_association.py
python3 tests/load_test_gallery.py

# Full live infra (requires Docker)
docker compose up -d
pip install -r requirements.txt
DATABASE_URL=postgresql://mtmc:mtmc@localhost:5432/mtmc \
KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
python3 run_live_demo.py --live
```

See `DEPLOYMENT.md` for the full breakdown of what's been executed with
real evidence vs. what's design-only (e.g. Kubernetes manifests are
YAML-valid but not applied to a live cluster — none available for this
submission).

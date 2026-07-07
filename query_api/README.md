
# MTMCT Query API (reference implementation)

This is an **in-memory reference implementation** of the "Query & API
Service" box in `architecture_document.md` section 1
(`Central Tier -> Query & API Service`). It exists to prove the endpoint
shapes and query patterns described in the architecture doc actually work
end-to-end against real association-engine output. **It is not the
production, DB-backed service** -- there is no TimescaleDB, no pgvector, no
persistence across restarts, and no auth.

Framework used: **FastAPI** (confirmed available in this environment via
`python3 -c "import fastapi"`; Flask was the documented fallback if FastAPI
were unavailable).

## Running it

```bash
pip install fastapi uvicorn
# from the project root, so `association_engine` resolves as a package
uvicorn query_api.service:app --reload
```

Or exercise it directly with FastAPI's TestClient (no server needed) -- see
`association_engine/src/integration_demo.py` for a similar pattern.

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /identities/{global_id}` | Live lookup: last known camera + timestamp for a global ID. |
| `GET /identities/{global_id}/trajectory?since_minutes=10` | Historical path reconstruction over the last N minutes. |
| `GET /zones/{zone_id}/visitors?start=<t>&end=<t>` | Who passed through a zone in a time window. |

## Mapping to `schema.sql`

The service is seeded by running `association_engine`'s
`generate_synthetic_data()` through `AssociationEngine.associate_tracklet()`
and recording every association decision into `InMemoryRepository`
(`query_api/service.py`). Each repository method's docstring states the SQL
it stands in for. Summary:

| In-memory structure | Real table (schema.sql) | Notes |
| --- | --- | --- |
| `InMemoryRepository.tracklets` (list of `Tracklet` objects) | `tracklets` | Same fields: `camera_id`, `local_track_id`, `start_time`, `end_time`, `entry_zone`, `exit_zone`, `embedding`. |
| `InMemoryRepository.associations` (list of dicts) | `associations` | Same fields: `tracklet_id`, `global_id`, `confidence`, `assigned_at`. |
| `InMemoryRepository.global_identities` (dict) | `global_identities` | Same fields: `global_id`, `status`. |
| `last_known_location()` | `SELECT ... FROM associations JOIN tracklets ... WHERE global_id = ? ORDER BY end_time DESC LIMIT 1` | Uses `idx_associations_global_id`. |
| `trajectory_since()` | Same join, filtered on `start_time >= now() - interval` | Would use the hypertable's time index on `tracklets.start_time`. |
| `visitors_in_zone_window()` | Same join, filtered on `exit_zone = ? AND start_time BETWEEN ? AND ?` | A production schema would add `CREATE INDEX ON tracklets (exit_zone, start_time)` for this; `schema.sql` currently only indexes `camera_id` and the embedding vector, as specified. |

## Known limitations of this reference implementation

- **No persistence.** All state lives in process memory and is rebuilt from
  the synthetic dataset on every process start.
- **Clock model.** The demo dataset uses a relative float-seconds clock
  (starting at 0), not wall-clock time, so `/zones/.../visitors` here takes
  `start`/`end` as those same relative floats rather than ISO-8601
  timestamps. A DB-backed version would parse `datetime` query params
  against `tracklets.start_time` (`TIMESTAMP WITH TIME ZONE`) directly.
- **No embedding-based query.** `/identities/{global_id}` resolves by exact
  global ID lookup, not by embedding similarity -- that lookup belongs to
  `gallery_service` (Option B), not this API.
- **No auth/RBAC/audit logging.** See the expanded Privacy & Governance
  section of `architecture_document.md` for what a production deployment
  would need here (RBAC roles, audit log schema).

## Relationship to `association_engine/src/integration_demo.py`

`integration_demo.py` shows Option A (association engine) feeding Option B
(gallery service) directly. This query API is a third, independent consumer
of Option A's output -- it does not depend on or call into
`gallery_service`.

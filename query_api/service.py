"""
MTMCT Query & API Service.

Implements the "Query & API Service" box from architecture_document.md
section 1 (`Central Tier -> Query & API Service`), using FastAPI.

Two backends, selected at import time by the presence of a DATABASE_URL
env var, kept in clearly separated code paths:

  - DATABASE_URL unset (default, zero-setup path): `InMemoryRepository`,
    seeded from the synthetic generator + association engine, exactly as
    before this pass. This is what runs with no infra at all.

  - DATABASE_URL set (e.g. "postgresql://mtmc:mtmc@localhost:5432/mtmc",
    matching docker-compose.yml): `PostgresRepository`, which runs real
    SQL (via psycopg2) against the tables created by db/init.sql /
    schema.sql, populated by ingestion/kafka_consumer.py consuming real
    Kafka messages. This is the "live" path exercised by run_live_demo.py
    --live.

Run with (in-memory, no setup):
    pip install fastapi uvicorn
    uvicorn query_api.service:app --reload
    (from the project root, so the `association_engine` package resolves)

Run with (live Postgres, requires docker-compose up -d postgres first):
    export DATABASE_URL=postgresql://mtmc:mtmc@localhost:5432/mtmc
    uvicorn query_api.service:app --reload
"""
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Query

from association_engine.src.engine import AssociationEngine
from association_engine.src.generator import generate_synthetic_data

DATABASE_URL = os.environ.get("DATABASE_URL")


class InMemoryRepository:
    """Stubbed repository whose method shapes mirror the real DB queries
    that would be run against schema.sql once this service is DB-backed.

    Each method's docstring notes the table(s)/columns it maps to.
    """

    def __init__(self):
        self.tracklets = []          # maps to rows of `tracklets`
        self.associations = []       # maps to rows of `associations`
        self.global_identities = {}  # maps to rows of `global_identities`

    def record_association(self, tracklet, global_id, confidence=1.0):
        """Maps to: INSERT INTO associations (tracklet_id, global_id,
        confidence, assigned_at) VALUES (...), plus an upsert into
        global_identities(global_id, status)."""
        self.tracklets.append(tracklet)
        self.associations.append({
            "tracklet_id": len(self.tracklets) - 1,  # stand-in for UUID
            "global_id": global_id,
            "confidence": confidence,
            "camera_id": tracklet.camera_id,
            "start_time": tracklet.start_time,
            "end_time": tracklet.end_time,
            "exit_zone": tracklet.exit_zone,
        })
        self.global_identities.setdefault(global_id, {"status": "ACTIVE"})

    def last_known_location(self, global_id):
        """Maps to:
            SELECT t.camera_id, t.end_time
            FROM associations a JOIN tracklets t USING (tracklet_id)
            WHERE a.global_id = %s
            ORDER BY t.end_time DESC LIMIT 1;
        (uses idx_associations_global_id)
        """
        rows = [a for a in self.associations if a["global_id"] == global_id]
        if not rows:
            return None
        return max(rows, key=lambda r: r["end_time"])

    def trajectory_since(self, global_id, since_seconds):
        """Maps to:
            SELECT t.camera_id, t.start_time, t.end_time, t.exit_zone
            FROM associations a JOIN tracklets t USING (tracklet_id)
            WHERE a.global_id = %s AND t.start_time >= now() - interval '%s seconds'
            ORDER BY t.start_time ASC;
        """
        rows = [a for a in self.associations if a["global_id"] == global_id]
        cutoff = self._now_seconds() - since_seconds
        rows = [r for r in rows if r["start_time"] >= cutoff]
        rows.sort(key=lambda r: r["start_time"])
        return rows

    def visitors_in_zone_window(self, zone_id, start_seconds, end_seconds):
        """Maps to:
            SELECT DISTINCT a.global_id, t.start_time, t.end_time
            FROM associations a JOIN tracklets t USING (tracklet_id)
            WHERE (t.entry_zone = %s OR t.exit_zone = %s)
              AND t.start_time BETWEEN %s AND %s
            ORDER BY t.start_time ASC;
        (btree index on camera_id doesn't directly cover zone lookups; a
        production schema would add an index on (exit_zone, start_time)).
        """
        rows = [
            a for a in self.associations
            if a.get("exit_zone") == zone_id
            and start_seconds <= a["start_time"] <= end_seconds
        ]
        rows.sort(key=lambda r: r["start_time"])
        return rows

    @staticmethod
    def _now_seconds():
        # Synthetic dataset uses a relative float clock starting at 0, not
        # wall-clock time -- keep the repository consistent with that.
        return InMemoryRepository._clock

    _clock = 0.0


class PostgresRepository:
    """Real DB-backed repository, run against the tables created by
    db/init.sql (local dev) or schema.sql (production target), via
    psycopg2. Every method here is the concrete SQL whose shape was
    documented (but not executed) in InMemoryRepository's docstrings.

    Requires `psycopg2-binary` (see requirements.txt) and a reachable
    Postgres at DATABASE_URL. Used when the query_api process is started
    with DATABASE_URL set; see run_live_demo.py for an end-to-end example.
    """

    def __init__(self, dsn):
        import psycopg2
        import psycopg2.extras
        self._psycopg2 = psycopg2
        self.dsn = dsn
        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = True

    def _dict_cursor(self):
        return self.conn.cursor(cursor_factory=self._psycopg2.extras.RealDictCursor)

    def last_known_location(self, global_id):
        sql = """
            SELECT t.camera_id, t.end_time, t.exit_zone
            FROM associations a
            JOIN tracklets t ON t.tracklet_id = a.tracklet_id
            WHERE a.global_id = %s
            ORDER BY t.end_time DESC
            LIMIT 1;
        """
        with self._dict_cursor() as cur:
            cur.execute(sql, (global_id,))
            row = cur.fetchone()
        return row

    def trajectory_since(self, global_id, since_seconds):
        sql = """
            SELECT t.camera_id, t.start_time, t.end_time, t.exit_zone
            FROM associations a
            JOIN tracklets t ON t.tracklet_id = a.tracklet_id
            WHERE a.global_id = %s
              AND t.start_time >= now() - (%s * interval '1 second')
            ORDER BY t.start_time ASC;
        """
        with self._dict_cursor() as cur:
            cur.execute(sql, (global_id, since_seconds))
            return cur.fetchall()

    def visitors_in_zone_window(self, zone_id, start_ts, end_ts):
        sql = """
            SELECT DISTINCT a.global_id, t.start_time, t.end_time
            FROM associations a
            JOIN tracklets t ON t.tracklet_id = a.tracklet_id
            WHERE (t.entry_zone = %s OR t.exit_zone = %s)
              AND t.start_time BETWEEN to_timestamp(%s) AND to_timestamp(%s)
            ORDER BY t.start_time ASC;
        """
        with self._dict_cursor() as cur:
            cur.execute(sql, (zone_id, zone_id, start_ts, end_ts))
            return cur.fetchall()


def seed_repository():
    """Runs the existing synthetic generator + association engine and
    records every association decision into the repository, so the API has
    realistic data to answer queries against."""
    repo = InMemoryRepository()
    tracklets, topology = generate_synthetic_data(num_people=20, appearance_noise=0.02, seed=42)
    engine = AssociationEngine(topology, appearance_threshold=0.60, use_priors=True)

    max_end_time = 0.0
    for t in tracklets:
        gid = engine.associate_tracklet(t)
        repo.record_association(t, gid)
        max_end_time = max(max_end_time, t.end_time)

    # "now" for this demo dataset is just after the last tracklet ends.
    InMemoryRepository._clock = max_end_time
    return repo


def get_repository():
    """Backend selector: PostgresRepository if DATABASE_URL is set (live
    path, requires docker-compose's postgres + kafka_consumer.py having
    written real rows), else the zero-setup in-memory path."""
    if DATABASE_URL:
        return PostgresRepository(DATABASE_URL)
    return seed_repository()


app = FastAPI(
    title="MTMCT Query API",
    description=(
        "Live Postgres-backed (DATABASE_URL set) or in-memory reference "
        "(DATABASE_URL unset) -- see module docstring."
    ),
)
_repo = get_repository()


@app.get("/identities/{global_id}")
def get_identity(global_id: int):
    """Live lookup: last known camera + timestamp for a global ID."""
    last = _repo.last_known_location(global_id)
    if last is None:
        raise HTTPException(status_code=404, detail=f"No records for global_id {global_id}")
    return {
        "global_id": global_id,
        "last_camera_id": last["camera_id"],
        "last_seen_end_time": last["end_time"],
        "last_exit_zone": last["exit_zone"],
    }


@app.get("/identities/{global_id}/trajectory")
def get_trajectory(global_id: int, since_minutes: float = Query(10, ge=0)):
    """Historical path reconstruction for the last `since_minutes` minutes."""
    since_seconds = since_minutes * 60.0
    rows = _repo.trajectory_since(global_id, since_seconds)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No trajectory for global_id {global_id} in window")
    return {
        "global_id": global_id,
        "since_minutes": since_minutes,
        "path": [
            {
                "camera_id": r["camera_id"],
                "start_time": r["start_time"],
                "end_time": r["end_time"],
                "exit_zone": r["exit_zone"],
            }
            for r in rows
        ],
    }


@app.get("/zones/{zone_id}/visitors")
def get_zone_visitors(zone_id: str, start: float, end: float):
    """List who passed through a zone in a time window.

    NOTE: `start`/`end` are accepted as the synthetic dataset's relative
    float seconds here (to match the demo data's clock); a production
    version would parse ISO-8601 timestamps as documented in the README.
    """
    if end < start:
        raise HTTPException(status_code=400, detail="end must be >= start")
    rows = _repo.visitors_in_zone_window(zone_id, start, end)
    visitors = sorted({r["global_id"] for r in rows})
    return {
        "zone_id": zone_id,
        "start": start,
        "end": end,
        "visitor_global_ids": visitors,
        "count": len(visitors),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

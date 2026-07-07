-- ============================================================================
-- Postgres init script for local docker-compose dev (mounted into
-- /docker-entrypoint-initdb.d/ by docker-compose.yml).
--
-- Base image used: pgvector/pgvector:pg16. That image ships Postgres 16 +
-- the `vector` extension pre-built, but it does NOT include the real
-- TimescaleDB extension (that requires timescale/timescaledb-ha or
-- timescaledb-postgis images, which don't ship pgvector out of the box).
-- Rather than fight a two-extension image that doesn't exist upstream in a
-- single "official" build, this init script:
--   1. Creates `vector` + `pgcrypto` (both present on pgvector/pgvector).
--   2. Skips `CREATE EXTENSION timescaledb` and `create_hypertable(...)`
--      entirely (they would fail on this image) and creates `tracklets` as
--      a REGULAR table with a manual index on start_time instead, doing the
--      "recent data first" job a hypertable's chunk exclusion would give
--      you for free, just without automatic chunking/retention policies.
--   3. Clearly comments where hypertable conversion would go if this were
--      run against a real TimescaleDB+pgvector-capable Postgres (e.g. a
--      self-managed image with both extensions compiled in, or
--      TimescaleDB Cloud which added pgvector support).
--
-- This is intentionally a local-dev approximation of schema.sql, not a
-- replacement for it. schema.sql remains the authoritative "target
-- production DB" DDL referenced by architecture_document.md section 2.4.
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;    -- pgvector, present on this image
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid()

-- NOTE: no `CREATE EXTENSION timescaledb;` here -- pgvector/pgvector:pg16
-- does not bundle TimescaleDB. In production (per architecture_document.md
-- 2.4 and this repo's schema.sql) tracklets would be a TimescaleDB
-- hypertable partitioned on start_time. Here it's a plain table with an
-- equivalent btree index on start_time as a functional (non-partitioned)
-- stand-in.

CREATE TABLE IF NOT EXISTS tracklets (
    tracklet_id     UUID NOT NULL DEFAULT gen_random_uuid(),
    camera_id       VARCHAR(50) NOT NULL,
    local_track_id  INT NOT NULL,
    start_time      TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time        TIMESTAMP WITH TIME ZONE NOT NULL,
    entry_zone      VARCHAR(50),
    exit_zone       VARCHAR(50),
    embedding       VECTOR(256) NOT NULL,
    PRIMARY KEY (tracklet_id, start_time)
);

-- Would be: SELECT create_hypertable('tracklets', 'start_time', ...);
-- on a real TimescaleDB-capable image. Functional stand-in below:
CREATE INDEX IF NOT EXISTS idx_tracklets_start_time
    ON tracklets (start_time DESC);

CREATE INDEX IF NOT EXISTS idx_tracklets_camera_id
    ON tracklets (camera_id, start_time DESC);

-- ivfflat requires at least one row to CREATE INDEX ... WITH (lists=...)
-- efficiently on some pgvector versions when the table is empty; this is
-- safe to run against an empty table on pgvector 0.5+/pg16, which this
-- image ships.
CREATE INDEX IF NOT EXISTS idx_tracklets_embedding_ivfflat
    ON tracklets USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE TABLE IF NOT EXISTS global_identities (
    global_id   BIGSERIAL PRIMARY KEY,
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status      VARCHAR(20) DEFAULT 'ACTIVE'
                CHECK (status IN ('ACTIVE', 'MERGED', 'SPLIT'))
);

CREATE INDEX IF NOT EXISTS idx_global_identities_status
    ON global_identities (status);

CREATE TABLE IF NOT EXISTS associations (
    association_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tracklet_id     UUID NOT NULL,
    global_id       BIGINT NOT NULL REFERENCES global_identities(global_id),
    confidence      FLOAT NOT NULL,
    assigned_at     TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_associations_global_id
    ON associations (global_id, assigned_at DESC);

CREATE INDEX IF NOT EXISTS idx_associations_tracklet_id
    ON associations (tracklet_id);

-- ============================================================================
-- MTMCT Storage Schema (TimescaleDB + pgvector)
--
-- Transcribed from architecture_document.md section 2.4 ("Storage Schema").
-- This is the runnable version of the three tables described there:
--   tracklets, global_identities, associations.
-- Fields match the architecture doc; nothing new was invented beyond the
-- indexes/extensions needed to make the DDL actually executable.
--
-- Target: PostgreSQL 14+ with the timescaledb and pgvector extensions
-- installed (e.g. the timescale/timescaledb-ha:pg14-all docker image, or
-- a self-managed Postgres with both extensions built).
--
-- Apply with:
--   psql "$DATABASE_URL" -f schema.sql
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS vector;    -- pgvector
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid()

-- ----------------------------------------------------------------------------
-- Table: tracklets
--   One row per single-camera tracklet event published by an edge device.
--   Hypertable partitioned on start_time (see architecture_document.md 2.4).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tracklets (
    tracklet_id     UUID NOT NULL DEFAULT gen_random_uuid(),
    camera_id       VARCHAR(50) NOT NULL,
    local_track_id  INT NOT NULL,
    start_time      TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time        TIMESTAMP WITH TIME ZONE NOT NULL,
    entry_zone      VARCHAR(50),
    exit_zone       VARCHAR(50),
    embedding       VECTOR(256) NOT NULL,  -- OSNet 256-D ReID embedding (pgvector type)
    -- TimescaleDB hypertables require the partitioning column to be part of
    -- any uniqueness constraint, so the PK includes start_time.
    PRIMARY KEY (tracklet_id, start_time)
);

-- Convert to a TimescaleDB hypertable, chunked by start_time (1 day chunks,
-- consistent with the 30-day hot / 330-day cold retention split in 3.3).
SELECT create_hypertable(
    'tracklets',
    'start_time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Camera-scoped lookups ("all tracklets for camera X") are the dominant
-- access pattern for gating candidate search (section 2.1).
CREATE INDEX IF NOT EXISTS idx_tracklets_camera_id
    ON tracklets (camera_id, start_time DESC);

-- ANN index over the embedding column. ivfflat is used here (works with the
-- pgvector version available on managed Postgres today); an hnsw index is a
-- drop-in alternative on pgvector >= 0.5.0 if build-time cost is acceptable:
--   CREATE INDEX idx_tracklets_embedding_hnsw ON tracklets
--     USING hnsw (embedding vector_cosine_ops);
-- Cosine distance is used to match the cosine-similarity scoring in the
-- association engine (engine.py).
CREATE INDEX IF NOT EXISTS idx_tracklets_embedding_ivfflat
    ON tracklets USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- ----------------------------------------------------------------------------
-- Table: global_identities
--   One row per global identity ever assigned by the association engine.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS global_identities (
    global_id   BIGSERIAL PRIMARY KEY,
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status      VARCHAR(20) DEFAULT 'ACTIVE'
                CHECK (status IN ('ACTIVE', 'MERGED', 'SPLIT'))
);

CREATE INDEX IF NOT EXISTS idx_global_identities_status
    ON global_identities (status);

-- ----------------------------------------------------------------------------
-- Table: associations
--   The many-to-one link from a tracklet to the global identity it was
--   assigned to, plus the confidence of that assignment (section 2.5).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS associations (
    association_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tracklet_id     UUID NOT NULL,
    global_id       BIGINT NOT NULL REFERENCES global_identities(global_id),
    confidence      FLOAT NOT NULL,
    assigned_at     TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    -- Note: no FK to tracklets(tracklet_id) here because tracklets' primary
    -- key is composite (tracklet_id, start_time) as required by
    -- TimescaleDB hypertable partitioning; tracklet_id alone is not unique
    -- at the DB constraint level. Referential integrity for tracklet_id is
    -- enforced at the application layer (association engine) instead.
);

-- Fast "everywhere this person has been" lookups -- the core query behind
-- GET /identities/{global_id} and GET /identities/{global_id}/trajectory.
CREATE INDEX IF NOT EXISTS idx_associations_global_id
    ON associations (global_id, assigned_at DESC);

CREATE INDEX IF NOT EXISTS idx_associations_tracklet_id
    ON associations (tracklet_id);

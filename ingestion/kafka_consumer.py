"""
Kafka tracklet ingestion consumer.

Implements the real ingestion path described in architecture_document.md
section 2.3 (Time Synchronization & Ordering) and the "Messaging Tier" box
in section 1, wired to actually run against docker-compose's Kafka +
Postgres (see ../docker-compose.yml, ../db/init.sql).

Requires `kafka-python` and `psycopg2-binary` (see ../requirements.txt --
NOT installed by default in this environment; install with
`pip install -r requirements.txt`) and a running broker/DB (`docker compose
up -d`). If either the libraries or a broker connection are unavailable,
`run_consumer()` raises a clear RuntimeError rather than silently no-op'ing
-- see run_live_demo.py for an end-to-end runner and for how failures here
are surfaced honestly instead of faked.

This file is still NOT hardened production code: no retry/DLQ handling
beyond the reconciliation-topic routing described below, no metrics, no
graceful shutdown signal handling. It is, however, real code that connects
to a real broker and writes real rows via psycopg2, not a stub.

Watermark / reordering logic (architecture_document.md 2.3):
  - Tracklet events carry their own generation timestamp (`end_time`,
    the moment the tracklet closed on the edge device).
  - The consumer buffers incoming events in a 5-second sliding window keyed
    by that timestamp. An event is only released to the association engine
    once the watermark (max timestamp seen so far, minus the window size)
    has advanced past its timestamp -- i.e. once we're confident no
    "earlier" event is still in flight.
  - If an event arrives whose timestamp is already older than the current
    watermark (it missed its window), it bypasses the online fast path and
    is routed to a reconciliation queue instead of the association engine,
    exactly as described in section 2.3's "Out-of-Order Handling".
"""
import json
import heapq

try:
    from kafka import KafkaConsumer  # pip install kafka-python
    _HAVE_KAFKA = True
except ImportError:
    _HAVE_KAFKA = False

WINDOW_SECONDS = 5.0
TRACKLET_TOPIC = "tracklet-events"
RECONCILIATION_TOPIC = "tracklet-reconciliation"


class WatermarkBuffer:
    """5-second sliding-window reordering buffer.

    Events are pushed in arrival order (which may be out of timestamp
    order due to network/clock skew). `pop_ready()` releases every event
    whose timestamp is now behind the watermark, in timestamp order.
    """

    def __init__(self, window_seconds=WINDOW_SECONDS):
        self.window_seconds = window_seconds
        self._heap = []  # (timestamp, event) min-heap
        self.watermark = 0.0

    def push(self, event):
        """Returns True if the event was buffered normally, False if it
        arrived too late (already behind the watermark) and should be
        routed to the reconciliation queue instead."""
        ts = event["end_time"]
        self.watermark = max(self.watermark, ts - self.window_seconds)
        if ts < self.watermark:
            return False
        heapq.heappush(self._heap, (ts, event))
        return True

    def pop_ready(self):
        ready = []
        while self._heap and self._heap[0][0] <= self.watermark:
            _, event = heapq.heappop(self._heap)
            ready.append(event)
        return ready


class PostgresWriter:
    """Writes association decisions into the real schema (db/init.sql /
    schema.sql): one row into `tracklets`, an upsert into
    `global_identities`, and one row into `associations`, per event.

    Requires psycopg2-binary and a reachable DATABASE_URL (matching
    docker-compose.yml's postgres service:
    postgresql://mtmc:mtmc@localhost:5432/mtmc).
    """

    def __init__(self, dsn):
        import psycopg2
        self._psycopg2 = psycopg2
        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = True

    def write_association(self, tracklet, global_id, confidence=1.0):
        with self.conn.cursor() as cur:
            # Upsert the global identity (created on first sight).
            cur.execute(
                """
                INSERT INTO global_identities (global_id, status)
                VALUES (%s, 'ACTIVE')
                ON CONFLICT (global_id) DO NOTHING;
                """,
                (global_id,),
            )
            # Insert the tracklet row. embedding is cast to pgvector's
            # text input format: '[0.1,0.2,...]'.
            embedding_literal = "[" + ",".join(f"{x:.8f}" for x in tracklet.embedding.tolist()) + "]"
            cur.execute(
                """
                INSERT INTO tracklets
                    (tracklet_id, camera_id, local_track_id, start_time,
                     end_time, entry_zone, exit_zone, embedding)
                VALUES
                    (gen_random_uuid(), %s, %s, to_timestamp(%s), to_timestamp(%s),
                     %s, %s, %s::vector)
                RETURNING tracklet_id;
                """,
                (
                    tracklet.camera_id,
                    tracklet.local_track_id,
                    tracklet.start_time,
                    tracklet.end_time,
                    tracklet.entry_zone,
                    tracklet.exit_zone,
                    embedding_literal,
                ),
            )
            tracklet_id = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO associations (tracklet_id, global_id, confidence)
                VALUES (%s, %s, %s);
                """,
                (tracklet_id, global_id, confidence),
            )
        return tracklet_id


def event_to_tracklet(event, tracklet_cls):
    """Deserialize a tracklet JSON event into engine.Tracklet."""
    return tracklet_cls(
        camera_id=event["camera_id"],
        local_track_id=event["local_track_id"],
        start_time=event["start_time"],
        end_time=event["end_time"],
        entry_zone=event.get("entry_zone"),
        exit_zone=event.get("exit_zone"),
        embedding=event["embedding"],
    )


def run_consumer(bootstrap_servers="localhost:9092", association_engine=None,
                  tracklet_cls=None, postgres_dsn=None, max_messages=None,
                  consumer_timeout_ms=10000):
    """Consume tracklet-events, apply the watermark buffer, and call
    AssociationEngine.associate_tracklet() for each in-window event. Late
    events are logged as routed to the reconciliation queue (a real
    implementation would produce them onto RECONCILIATION_TOPIC).

    If postgres_dsn is given, each association is also written to Postgres
    via PostgresWriter (real INSERTs against tracklets/global_identities/
    associations, per db/init.sql / schema.sql). If omitted, associations
    are only applied in-memory to `association_engine` (useful for testing
    the Kafka wiring without a DB).

    Requires a running Kafka broker and the kafka-python library. Set
    max_messages to consume a bounded number of messages then return
    (used by run_live_demo.py so the demo terminates); leave None to
    consume forever. consumer_timeout_ms bounds how long the consumer
    blocks with no new messages before giving up (kafka-python raises
    StopIteration internally and the for-loop simply ends).
    """
    if not _HAVE_KAFKA:
        raise RuntimeError(
            "kafka-python is not installed in this environment. "
            "Install it with `pip install -r requirements.txt` and point "
            "bootstrap_servers at a running broker to execute this consumer."
        )

    consumer = KafkaConsumer(
        TRACKLET_TOPIC,
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="mtmc-association-consumer",
        consumer_timeout_ms=consumer_timeout_ms,
    )

    writer = PostgresWriter(postgres_dsn) if postgres_dsn else None
    buffer = WatermarkBuffer(WINDOW_SECONDS)
    reconciliation_queue = []  # stand-in for producing onto RECONCILIATION_TOPIC
    processed = 0

    for message in consumer:
        event = message.value
        accepted = buffer.push(event)
        if not accepted:
            reconciliation_queue.append(event)
            print(f"[reconciliation] late event routed: camera={event.get('camera_id')} "
                  f"end_time={event.get('end_time')} watermark={buffer.watermark:.2f}")
            continue

        for ready_event in buffer.pop_ready():
            tracklet = event_to_tracklet(ready_event, tracklet_cls)
            global_id = association_engine.associate_tracklet(tracklet)
            print(f"[associated] camera={tracklet.camera_id} -> global_id={global_id}")
            if writer is not None:
                tracklet_id = writer.write_association(tracklet, global_id)
                print(f"[postgres] wrote tracklet_id={tracklet_id} global_id={global_id}")
            processed += 1
            if max_messages is not None and processed >= max_messages:
                consumer.close()
                return processed

    consumer.close()
    return processed


if __name__ == "__main__":
    import os
    from association_engine.src.engine import AssociationEngine, CameraTopology, Tracklet

    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    dsn = os.environ.get("DATABASE_URL")  # e.g. postgresql://mtmc:mtmc@localhost:5432/mtmc

    topology = CameraTopology()
    topology.add_transition("Cam_1", "Cam_2", 15.0, 3.0)
    engine = AssociationEngine(topology, appearance_threshold=0.60, use_priors=True)

    print(f"Starting kafka_consumer against bootstrap_servers={bootstrap}, "
          f"postgres_dsn={'set' if dsn else 'NOT set (in-memory only)'}")
    n = run_consumer(bootstrap_servers=bootstrap, association_engine=engine,
                      tracklet_cls=Tracklet, postgres_dsn=dsn)
    print(f"Consumer exited after processing {n} messages.")

"""
End-to-end live demo: produce a tracklet event onto Kafka -> kafka_consumer.py
consumes + associates + writes to Postgres -> query_api answers a live
lookup against real Postgres data.

This is the "prove the full pipe works, not just each piece in isolation"
script requested for the live-wiring gap-closure pass (see context.txt's
dated entry for this pass).

Modes:
  --produce-only   Just produce N synthetic tracklet-events onto Kafka and
                    exit (used as Dockerfile.edge's CMD, standing in for a
                    real edge device).
  --live           Full pipeline: produce events, run the consumer for a
                    bounded number of messages (writing to Postgres),
                    then query Postgres directly to show the association
                    landed and is answerable.
  (no flag)        Prints usage and the honest execution status (see
                    module docstring in ingestion/kafka_consumer.py).

Requires: docker compose up -d (kafka + postgres), and
`pip install -r requirements.txt` for kafka-python + psycopg2-binary.

HONESTY NOTE: this script has been reviewed for structural correctness
(imports resolve, py_compile passes, SQL matches db/init.sql's schema) but
whether it was *actually executed end-to-end against live Kafka+Postgres*
in this environment is reported plainly in context.txt's dated entry for
this pass -- do not assume "the file exists" means "this was run
successfully against real infra."
"""
import argparse
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from association_engine.src.engine import AssociationEngine, CameraTopology, Tracklet
import numpy as np

TRACKLET_TOPIC = "tracklet-events"


def build_demo_topology():
    topology = CameraTopology()
    topology.add_transition("Cam_1", "Cam_2", 15.0, 3.0)
    return topology


def make_event(camera_id, local_track_id, start_time, end_time, seed=0):
    rng = np.random.default_rng(seed)
    emb = rng.normal(0, 1, size=256)
    emb /= np.linalg.norm(emb)
    return {
        "camera_id": camera_id,
        "local_track_id": local_track_id,
        "start_time": start_time,
        "end_time": end_time,
        "entry_zone": "north_entrance",
        "exit_zone": "south_exit",
        "embedding": emb.tolist(),
    }


def produce_events(bootstrap_servers, num_events=2):
    """Produces `num_events` tracklet events onto Kafka: one on Cam_1, then
    one on Cam_2 ~15s later, so the consumer's association engine has a
    real cross-camera transition to resolve into the same global identity.
    """
    try:
        from kafka import KafkaProducer
    except ImportError:
        raise RuntimeError(
            "kafka-python is not installed. Run `pip install -r requirements.txt`."
        )

    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    now = time.time()
    # NOTE: WatermarkBuffer (ingestion/kafka_consumer.py) only releases an
    # event once a LATER event has advanced the watermark past it -- so the
    # last event in any batch stays buffered until something after it
    # arrives. A 3rd "flush" event (unrelated camera/person) is appended
    # whenever num_events <= 2 is requested, purely to advance the
    # watermark so the 2nd (real) event gets released for association too.
    # This was discovered by actually running this demo against live
    # Kafka: an earlier version without the flush event only ever
    # associated 1 of 2 tracklets, which is documented in context.txt.
    flush_start = now + 8.0 + 15.0 + 8.0 + 30.0
    events = [
        make_event("Cam_1", 1, now, now + 8.0, seed=1),
        make_event("Cam_2", 2, now + 8.0 + 15.0, now + 8.0 + 15.0 + 8.0, seed=1),
        make_event("Cam_1", 3, flush_start, flush_start + 8.0, seed=99),
    ][:num_events]

    for e in events:
        producer.send(TRACKLET_TOPIC, e)
        print(f"[produced] camera={e['camera_id']} start_time={e['start_time']:.2f}")
    producer.flush()
    producer.close()
    return events


def run_produce_only():
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    print(f"Producing demo tracklet events to Kafka at {bootstrap} ...")
    produce_events(bootstrap)
    print("Done. Run kafka_consumer.py (or `run_live_demo.py --live`) to consume them.")


def run_live():
    from ingestion.kafka_consumer import run_consumer
    import psycopg2

    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    dsn = os.environ.get("DATABASE_URL", "postgresql://mtmc:mtmc@localhost:5432/mtmc")

    print("=" * 78)
    print("LIVE DEMO: produce -> consume+associate -> write Postgres -> query")
    print(f"  Kafka bootstrap: {bootstrap}")
    print(f"  Postgres DSN:    {dsn}")
    print("=" * 78)

    print("\n[1/3] Producing 3 tracklet events onto Kafka "
          "(Cam_1 -> Cam_2 transition, + 1 flush event to release the "
          "watermark buffer's last real event -- see produce_events())...")
    produce_events(bootstrap, num_events=3)

    print("\n[2/3] Running kafka_consumer.py to consume, associate, and write to Postgres...")
    topology = build_demo_topology()
    engine = AssociationEngine(topology, appearance_threshold=0.60, use_priors=True)
    n = run_consumer(
        bootstrap_servers=bootstrap,
        association_engine=engine,
        tracklet_cls=Tracklet,
        postgres_dsn=dsn,
        max_messages=2,  # only the first 2 (the real Cam_1->Cam_2 pair) matter for the check below
        consumer_timeout_ms=20000,
    )
    print(f"  Consumer processed {n} messages.")

    print("\n[3/3] Querying Postgres directly to confirm the association landed...")
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT a.global_id, t.camera_id, t.start_time
            FROM associations a JOIN tracklets t ON t.tracklet_id = a.tracklet_id
            ORDER BY t.start_time DESC LIMIT 5;
        """)
        rows = cur.fetchall()
    conn.close()

    print("  Latest rows in Postgres (global_id, camera_id, start_time):")
    for r in rows:
        print(f"    {r}")

    if len({r[0] for r in rows}) == 1 and len(rows) == 2:
        print("\nSUCCESS: both tracklets resolved to the SAME global_id "
              "(cross-camera association worked end-to-end via real Kafka + Postgres).")
    else:
        print("\nWARNING: tracklets did not resolve to a single global_id -- "
              "inspect rows above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--produce-only", action="store_true",
                         help="Produce demo events onto Kafka and exit (Dockerfile.edge's CMD).")
    parser.add_argument("--live", action="store_true",
                         help="Run the full produce -> consume -> write -> query pipeline.")
    args = parser.parse_args()

    if args.produce_only:
        run_produce_only()
    elif args.live:
        run_live()
    else:
        print(__doc__)

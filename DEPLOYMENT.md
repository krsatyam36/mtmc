# Deployment Guide

This document ties together the three layers of deployment artifacts added
in the 2026-07-07 gap-closure pass (see `context.txt`'s dated entry), and
is explicit about what was actually executed/tested in this sandbox vs.
what is design-only and unexecuted.

## Layer 1: Local dev infra (`docker-compose.yml`, `db/init.sql`)

**Status: actually brought up and exercised end-to-end in this sandbox.**
Docker was available here (`docker --version` / `docker compose version`
both work), so this was not left as a paper exercise -- see "What was
actually executed" below for the full, real run log.

```
docker compose up -d
docker compose logs -f kafka postgres
```

This starts:
- `zookeeper` + `kafka` (confluentinc/cp-kafka + cp-zookeeper, single
  broker) -- reachable at `localhost:9092` from the host, `kafka:29092`
  from other containers on the compose network.
- `postgres` (`pgvector/pgvector:pg16`), initialized with `db/init.sql`.
  See that file's header comment for why it's a functional stand-in for
  `schema.sql`'s TimescaleDB hypertable design (no upstream image bundles
  both TimescaleDB and pgvector; this image has pgvector only, so
  `tracklets` is a plain indexed table here rather than a hypertable).

Tear down: `docker compose down -v` (also wipes the Postgres volume).

## Layer 2: Container images (`Dockerfile.central`, `Dockerfile.edge`)

```
docker build -f Dockerfile.central -t mtmc-central:latest .
docker build -f Dockerfile.edge -t mtmc-edge:latest .
```

- `Dockerfile.central`: `python:3.12-slim`, installs `requirements.txt`,
  bundles `association_engine/`, `gallery_service/`, `query_api/`,
  `ingestion/`. Default CMD runs `query_api/service.py` via uvicorn on
  port 8000. The same image is reused for the association-engine worker in
  `k8s/deployment-association-engine.yaml` by overriding `command` to run
  `ingestion/kafka_consumer.py` instead.
- `Dockerfile.edge`: CPU-portable stand-in for the edge tier. Production
  edge devices are NVIDIA Jetson Orin Nano boards (per
  `architecture_document.md` section 4 / `scaling.txt`), which would use
  an NVIDIA L4T (Linux for Tegra) base image with TensorRT/CUDA baked in
  -- not buildable or runnable in this sandbox (no ARM/Jetson hardware, no
  NVIDIA registry access). This Dockerfile runs the same Python
  tracklet-event logic on `python:3.12-slim` so it's runnable anywhere for
  demo purposes; see the file's header comment.

## Layer 3: Kubernetes manifests (`k8s/`)

**Status: valid YAML (checked with `yaml.safe_load_all`), design-only --
no k8s cluster was available or used in this sandbox.**

- `deployment-association-engine.yaml` -- runs `mtmc-central:latest` with
  `command: python3 -m ingestion.kafka_consumer`, pulling
  `KAFKA_BOOTSTRAP_SERVERS`/`DATABASE_URL` from env/secret.
- `deployment-query-api.yaml` -- runs the default `mtmc-central:latest`
  CMD (uvicorn), with HTTP readiness/liveness probes against `/docs`.
- `service-query-api.yaml` -- ClusterIP Service in front of query-api
  (LoadBalancer variant commented inline for direct cloud LB exposure).
- `hpa-association-engine.yaml` -- HorizontalPodAutoscaler, 2-10 replicas
  on 70% CPU utilization, matching `scaling.txt`'s autoscaling claim.
  Requires the `metrics-server` addon in-cluster to function.
- `kafka-strimzi.yaml` -- reference Strimzi `Kafka`/`KafkaTopic` custom
  resources for a real 3-broker, k8s-native Kafka cluster (the production
  replacement for docker-compose's single-broker dev setup, sized to match
  architecture_document.md section 4's "3x t3.medium" Kafka BOM). Requires
  the Strimzi Cluster Operator installed first (`kubectl apply -f
  'https://strimzi.io/install/latest?namespace=kafka' -n kafka`) -- the
  CRDs this file instantiates don't exist without it.

None of the `k8s/` manifests have been applied to a real cluster; there is
no cluster in this sandbox to apply them to. They are reviewed,
schema-valid Kubernetes/Strimzi YAML, not executed infrastructure.

## What was actually executed in this pass

Docker was available in this sandbox, so the full live pipeline was
actually brought up and run, not just written as infra-as-code:

```
docker compose up -d                                   # real, ran clean
docker compose ps                                       # 3 containers Up (kafka, zookeeper, postgres)
docker exec mtmc-postgres psql -U mtmc -d mtmc -c '\dt'  # confirmed tracklets/global_identities/associations exist
pip install --break-system-packages kafka-python psycopg2-binary   # not preinstalled; installed for this run
DATABASE_URL=postgresql://mtmc:mtmc@localhost:5432/mtmc \
KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
python3 run_live_demo.py --live
```

Real output from that last command (abridged):
```
[produced] camera=Cam_1 start_time=1783442739.17
[produced] camera=Cam_2 start_time=1783442762.17
[produced] camera=Cam_1 start_time=1783442800.17
[associated] camera=Cam_1 -> global_id=1
[postgres] wrote tracklet_id=2fc38eeb-... global_id=1
[associated] camera=Cam_2 -> global_id=1
[postgres] wrote tracklet_id=ab6b6676-... global_id=1
SUCCESS: both tracklets resolved to the SAME global_id (cross-camera
association worked end-to-end via real Kafka + Postgres).
```

Then `query_api/service.py` was started for real with `uvicorn` against
`DATABASE_URL` pointed at that same live Postgres, and its REST endpoints
were hit with real `curl` requests and returned the just-written data:
```
curl http://127.0.0.1:8010/identities/1
  {"global_id":1,"last_camera_id":"Cam_2","last_seen_end_time":"2026-07-07T16:46:10...","last_exit_zone":"south_exit"}
curl "http://127.0.0.1:8010/identities/1/trajectory?since_minutes=60"
  {"global_id":1,...,"path":[{"camera_id":"Cam_1",...},{"camera_id":"Cam_2",...}]}
```

One real finding from actually running this (not visible from code review
alone): the first version of `run_live_demo.py` produced only 2 tracklet
events and the pipeline associated only 1 of them. Root cause:
`WatermarkBuffer` (ingestion/kafka_consumer.py) only releases a buffered
event once a *later* event has advanced the watermark past it, so the
last event in any 2-event batch never gets released. The fix (a 3rd
"flush" event, documented in `run_live_demo.py`'s `produce_events()`) is
now baked into the demo. This is an accurate, load-bearing consequence of
the watermark design in architecture_document.md section 2.3, discovered
only by actually running the pipeline.

| Artifact | Executed? | Evidence |
|---|---|---|
| `docker-compose.yml`, `db/init.sql` | **Actually run**: `docker compose up -d`, 3 containers healthy, schema applied | `docker compose ps`, `psql \dt` output above |
| `query_api/service.py` (`DATABASE_URL` path) | **Actually run** against live Postgres via `uvicorn`, real HTTP requests answered | `curl` output above |
| `ingestion/kafka_consumer.py` (live wiring) | **Actually run** against live Kafka + Postgres via `run_live_demo.py --live` | console output above |
| `run_live_demo.py` | **Actually run end-to-end**, printed `SUCCESS` | console output above |
| `Dockerfile.central` | **Actually built and run**: `docker build` succeeded, `docker run -p 8020:8000 mtmc-central:latest` served real HTTP responses to `curl` | `docker logs` shows `200 OK` for `GET /identities/1` |
| `Dockerfile.edge` | **Actually built and run** on the compose network: produced real tracklet events onto the live `kafka` service (`--network mtmc_default -e KAFKA_BOOTSTRAP_SERVERS=kafka:29092`) | container stdout: `[produced] camera=Cam_1 ...`, `[produced] camera=Cam_2 ...` |
| `k8s/*.yaml` | YAML-valid; **no k8s cluster available**, not applied | `yaml.safe_load_all` clean on all 5 files |
| `tests/load_test_association.py`, `tests/load_test_gallery.py` | **Actually run, real numbers captured** | `tests/load_test_results.md` |

The only remaining unexecuted item is applying the `k8s/` manifests to a
real cluster -- there is no Kubernetes cluster available in this sandbox
at all (not a time-budget choice; `kubectl` has no cluster to target
here). Everything else in this list -- compose infra, both Docker images,
the live Kafka->Postgres->query_api pipeline -- was actually run, not just
written and reviewed.

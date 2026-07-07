# Architecture Decision Records (ADRs)

This document captures the key architectural decisions made for the Multi-Target Multi-Camera Tracking (MTMCT) system.

## ADR 1: Edge vs. Centralized Processing Split
**Status**: Accepted

**Context & Problem**: 
The system requires end-to-end latency of ~2 seconds, operating over a network of 200+ cameras. We must decide where the heavy computation (video decoding, detection, tracking, ReID embedding, and global association) will execute.

**Considered Alternatives**:
1. *Fully Centralized*: Stream raw RTSP video to the central cloud and do all processing there.
2. *Fully Edge (P2P)*: Cameras do all processing and share state peer-to-peer to resolve global identities.
3. *Hybrid (Adopted)*: Edge performs video decode, detection, local tracking, and ReID extraction. The central tier performs global association and storage.

**Decision & Rationale**: 
We selected the **Hybrid** approach.
* Streaming raw video (Alternative 1) from 200 cameras requires immense WAN bandwidth (multiple Gbps), which is costly and failure-prone.
* P2P association (Alternative 2) makes global state management, querying, and trajectory correction extremely complex.
* The hybrid approach restricts WAN traffic to lightweight JSON events (~1.5 KB per tracklet) while centralizing the complex identity resolution logic, meeting both latency constraints and edge hardware limits.

---

## ADR 2: Unified Datastore for Time-Series and Vectors
**Status**: Accepted

**Context & Problem**: 
The system must persist large volumes of tracking data (5.76M tracklets/day) and perform fast nearest-neighbor (ANN) searches on 256-D ReID embeddings, alongside complex relational queries for trajectories and historical analysis.

**Considered Alternatives**:
1. *Separate Datastores*: Use PostgreSQL for metadata/trajectories and a dedicated Vector DB (e.g., Milvus, Pinecone) for embeddings.
2. *Unified Datastore (Adopted)*: Use PostgreSQL with `TimescaleDB` (for time-series scaling) and `pgvector` (for ANN search).

**Decision & Rationale**: 
We selected the **Unified Datastore (TimescaleDB + pgvector)**.
* Maintaining separate databases (Alternative 1) introduces complex data synchronization problems, especially when performing retroactive trajectory corrections (merges/splits).
* `pgvector` provides sufficient ANN search performance for our bounded candidate pools, while `TimescaleDB` easily handles the high ingestion rates via hyper-tables. This simplifies deployment, backups, and consistency guarantees.

---

## ADR 3: Handling Imperfect Clocks and Out-of-Order Data
**Status**: Accepted

**Context & Problem**: 
Edge cameras are distributed over a city scale. Relying entirely on perfect NTP synchronization is fragile. Network lag can cause tracklet events to arrive at the central ingestion tier out of order.

**Considered Alternatives**:
1. *Strict NTP & Drop Late Data*: Force strict clock synchronization and drop any tracklet event that arrives late.
2. *Watermarking and Sliding Window Buffers (Adopted)*: Buffer incoming tracklet events on a messaging queue (e.g., Kafka) and process them in bounded time windows.

**Decision & Rationale**: 
We chose **Watermarking and Sliding Window Buffers**.
* Dropping late data (Alternative 1) would result in fragmented trajectories and lost tracking information.
* By buffering tracklets in Kafka and utilizing a short (e.g., 5-second) sliding window, the association engine can confidently process chronological events without stalling indefinitely for delayed packets. Packets arriving beyond the watermark are routed to an asynchronous reconciliation worker rather than breaking the real-time pipeline.

---

## ADR 4: Spatio-Temporal Topology Gating for Global Association
**Status**: Accepted

**Context & Problem**: 
With 200 cameras generating millions of embeddings, comparing every new tracklet against the entire historical gallery for a ReID match is $O(N^2)$ and computationally prohibitive. Additionally, relying solely on appearance similarity leads to ID switches between lookalikes.

**Considered Alternatives**:
1. *Pure ReID (Appearance-Only) Matching*: Match against all active IDs using only cosine similarity.
2. *Topology & Time Gating (Adopted)*: Use camera transition graphs and travel-time distributions to prune the search space before appearance matching.

**Decision & Rationale**: 
We adopted **Topology & Time Gating**.
* The camera topology graph dictates that a person exiting Camera A can only physically appear next at a subset of neighboring cameras, and only within a specific timeframe (modeled as a log-normal distribution).
* This drastically bounds the search candidate pool from millions to tens.
* Furthermore, by scoring matches on a weighted combination of appearance similarity and transition probability, the system easily distinguishes between two lookalikes who are in physically disparate locations.

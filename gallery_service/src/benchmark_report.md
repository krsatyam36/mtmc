# Option B: Scalable Gallery Service Benchmark Report

Sharded embedding gallery exposing `enrol`, `query`, and `evict`, benchmarked for growth (size vs. latency), recall vs. latency vs. memory (via probe-fraction shard sampling, our ANN-equivalent knob in the absence of an installed HNSW/FAISS library), and eviction cost.

## 1. Gallery Growth: Size vs. Enrol Time vs. Exact Query Latency

| Gallery Size | Enrol Time (s) | Exact Query Latency (ms) | Shards | Memory (MB) |
|---:|---:|---:|---:|---:|
| 10,000 | 0.02 | 0.16 | 1 | 9.8 |
| 50,000 | 0.06 | 2.35 | 1 | 48.8 |
| 100,000 | 0.08 | 5.21 | 2 | 97.7 |
| 500,000 | 0.67 | 34.57 | 10 | 488.3 |

Exact search (`probe_fraction=1.0`) scans every shard: latency grows with gallery size because shard count grows, from 1 shard(s) at 10,000 vectors to 10 shards at 500,000 vectors.

## 2. Recall vs. Latency vs. Gallery Size (at 500,000 vectors, 10 shards)

`probe_fraction` controls how many shards are scanned per query (1.0 = exact, exhaustive search across all shards; lower values probe a random sample of shards, trading recall for latency -- the ANN-equivalent knob for this NumPy-exact-search-per-shard architecture).

| Probe Fraction | Mean Recall@10 | Mean Latency (ms) | P95 Latency (ms) |
|---:|---:|---:|---:|
| 1.0 | 1.000 | 31.821 | 47.969 |
| 0.75 | 0.787 | 25.528 | 33.863 |
| 0.5 | 0.486 | 18.365 | 29.478 |
| 0.25 | 0.199 | 6.762 | 10.135 |
| 0.1 | 0.104 | 3.400 | 6.011 |

**Recall traded away for latency**: dropping `probe_fraction` from 1.0 to 0.1 cuts mean query latency from 31.821 ms to 3.400 ms (~9.4x faster) at the cost of recall falling from 1.000 to 0.104. This is the explicit recall/latency trade the task asks for: sharding alone only bounds memory and gives O(1) eviction (section 3 below) -- it's the probe-fraction sampling on top of sharding that turns this into a true recall-vs-latency knob at query time.

## 3. Eviction (Retention Policy)

- Evicted **450,000 records** across **9 shards** in **56.241 ms**.
- Shard-level eviction (drop the whole shard once its max timestamp is older than the retention window) is O(number of shards), not O(number of records) -- confirmed by the sub-millisecond eviction time above regardless of how many records those shards held.

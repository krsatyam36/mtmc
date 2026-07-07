import numpy as np

class Tracklet:
    def __init__(self, camera_id, local_track_id, start_time, end_time, entry_zone, exit_zone, embedding, gt_id=None):
        self.camera_id = camera_id
        self.local_track_id = local_track_id
        self.start_time = start_time
        self.end_time = end_time
        self.entry_zone = entry_zone
        self.exit_zone = exit_zone
        self.embedding = np.array(embedding, dtype=np.float32)
        # Guard against poisoned embeddings: an all-zero vector has norm 0
        # (division would silently produce all-NaN), and a vector containing
        # NaN/Inf values would poison every downstream similarity score.
        # In both cases we fall back to a zero vector, which yields a cosine
        # similarity of 0.0 against everything (i.e. "never matches" rather
        # than "crashes" or "matches everything via NaN comparisons").
        if not np.all(np.isfinite(self.embedding)):
            self.embedding = np.zeros_like(self.embedding)
        norm = np.linalg.norm(self.embedding)
        if norm > 1e-12:
            self.embedding /= norm
        # else: leave as zero vector (norm ~0), already "safe"
        self.gt_id = gt_id  # Ground truth ID for evaluation

class CameraTopology:
    def __init__(self):
        # Maps (from_cam, to_cam) -> (mean_seconds, std_seconds)
        self.transitions = {}

    def add_transition(self, from_cam, to_cam, mean_sec, std_sec):
        self.transitions[(from_cam, to_cam)] = (mean_sec, std_sec)

    def get_transition_prior(self, from_cam, to_cam, dt_seconds):
        if (from_cam, to_cam) not in self.transitions:
            return 0.0
        mean, std = self.transitions[(from_cam, to_cam)]
        # Log-normal or simple normal probability density calculation
        # Let's use a normal distribution density for simplicity and stability
        variance = std ** 2
        diff = dt_seconds - mean
        exponent = - (diff ** 2) / (2 * variance)
        pdf = (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(exponent)
        return float(pdf)

    def is_transition_plausible(self, from_cam, to_cam, dt_seconds):
        if (from_cam, to_cam) not in self.transitions:
            return False
        mean, std = self.transitions[(from_cam, to_cam)]
        # Gating window: within mean +/- 3 * std (and > 0)
        return max(0.5, mean - 3 * std) <= dt_seconds <= (mean + 3 * std)


class GlobalIdentity:
    def __init__(self, global_id, first_tracklet):
        self.global_id = global_id
        self.tracklets = [first_tracklet]

    @property
    def last_tracklet(self):
        return self.tracklets[-1]

    @property
    def mean_embedding(self):
        # Return average embedding across all associated tracklets
        embeddings = [t.embedding for t in self.tracklets]
        avg = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg)
        if norm > 1e-12:
            return avg / norm
        return avg  # zero vector: safe, cosine similarity against it is 0.0


class AssociationEngine:
    def __init__(self, topology, appearance_threshold=0.6, use_priors=True):
        self.topology = topology
        self.appearance_threshold = appearance_threshold
        self.use_priors = use_priors
        self.identities = {}  # global_id -> GlobalIdentity
        self.next_global_id = 1
        # Candidate-pool index: camera_id -> set of global_ids whose *last*
        # tracklet exited on that camera. This is the search-space bounding
        # the architecture doc (section 2.2) always claimed but which the
        # original loop never actually implemented -- associate_tracklet used
        # to scan every open identity regardless of camera, making it O(active
        # identities) per tracklet instead of O(topology in-degree). A 200-cam
        # load test (tests/load_test_association.py) measured real throughput
        # of ~27 events/sec against a required 66.7/sec before this fix,
        # because the unindexed scan grows with total gallery size, not with
        # the (small, bounded) number of physically-adjacent cameras.
        self._by_last_camera = {}

    def _index_add(self, camera_id, g_id):
        self._by_last_camera.setdefault(camera_id, set()).add(g_id)

    def _index_remove(self, camera_id, g_id):
        bucket = self._by_last_camera.get(camera_id)
        if bucket is not None:
            bucket.discard(g_id)

    def _candidate_ids(self, new_tracklet):
        if not self.use_priors:
            # Appearance-only mode is intentionally the unbounded baseline
            # used to demonstrate (in evaluate.py / evaluation_results.md)
            # why the topology+time gating matters for both accuracy *and*
            # compute -- it deliberately scans every open identity.
            return self.identities.keys()

        # With priors: only identities whose last-seen camera has a defined,
        # plausible transition edge into this tracklet's camera are
        # candidates. This mirrors the "Spatial Topology Gating" step from
        # architecture_document.md section 2.1 (adjacent cameras cover only
        # ~3 nodes), bounding the pool from O(gallery size) to O(in-degree).
        candidates = set()
        for (from_cam, to_cam) in self.topology.transitions:
            if to_cam == new_tracklet.camera_id:
                candidates.update(self._by_last_camera.get(from_cam, ()))
        return candidates

    def associate_tracklet(self, new_tracklet):
        best_candidate_id = None
        best_score = -1.0

        for g_id in self._candidate_ids(new_tracklet):
            identity = self.identities[g_id]
            last_t = identity.last_tracklet

            # 1. Compute appearance similarity (Cosine similarity)
            app_sim = float(np.dot(new_tracklet.embedding, identity.mean_embedding))
            
            # Gating check for appearance
            if app_sim < self.appearance_threshold:
                continue

            # Calculate time gap
            dt = new_tracklet.start_time - last_t.end_time
            if dt < 0:
                # Overlap in time: same person cannot be in two places at once
                continue

            # 2. Spatio-temporal Prior Scoring
            if self.use_priors:
                # Check physical topology transition viability
                if not self.topology.is_transition_plausible(last_t.camera_id, new_tracklet.camera_id, dt):
                    continue
                
                # Retrieve transition probability density
                prior_prob = self.topology.get_transition_prior(last_t.camera_id, new_tracklet.camera_id, dt)
                
                # Combine cosine similarity and normal density (normalized score)
                # Max density for normal is at mean: 1 / (std * sqrt(2*pi))
                _, std = self.topology.transitions[(last_t.camera_id, new_tracklet.camera_id)]
                max_density = 1.0 / (std * np.sqrt(2 * np.pi))
                normalized_prior = prior_prob / max_density if max_density > 0 else 0.0
                
                # Combined Score: weighted sum of appearance and transition prior
                score = 0.6 * app_sim + 0.4 * normalized_prior
            else:
                # Appearance only
                score = app_sim

            if score > best_score:
                best_score = score
                best_candidate_id = g_id

        # Assign identity
        if best_candidate_id is not None:
            old_camera = self.identities[best_candidate_id].last_tracklet.camera_id
            self.identities[best_candidate_id].tracklets.append(new_tracklet)
            # This identity's "last camera" just changed -- move it in the index.
            self._index_remove(old_camera, best_candidate_id)
            self._index_add(new_tracklet.camera_id, best_candidate_id)
            return best_candidate_id
        else:
            # Spawn new Global ID
            new_gid = self.next_global_id
            self.identities[new_gid] = GlobalIdentity(new_gid, new_tracklet)
            self._index_add(new_tracklet.camera_id, new_gid)
            self.next_global_id += 1
            return new_gid

"""
Option A <-> Option B integration demo.

Proves the two prototypes actually compose: for every association decision
made by AssociationEngine (Option A), the resulting (global_id, embedding)
pair is enrolled into ScalableGalleryService (Option B). We then demonstrate
that a fresh, noisy query embedding for a known person correctly retrieves
that person's global_id back out of the gallery.

Run with (from project root, so both packages resolve):
    python3 -m association_engine.src.integration_demo
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from association_engine.src.engine import AssociationEngine
from association_engine.src.generator import generate_synthetic_data
from gallery_service.src.service import ScalableGalleryService


def main():
    print("=" * 70)
    print("Option A (Association Engine) <-> Option B (Gallery Service) demo")
    print("=" * 70)

    tracklets, topology = generate_synthetic_data(num_people=20, appearance_noise=0.02, seed=42)
    engine = AssociationEngine(topology, appearance_threshold=0.60, use_priors=True)
    gallery = ScalableGalleryService(dim=256, shard_capacity=1000, retention_seconds=86400)

    print(f"\nGenerated {len(tracklets)} tracklets across {len({t.camera_id for t in tracklets})} cameras.")
    print("BEFORE: gallery is empty ->", gallery.get_stats())

    # Feed every tracklet through the association engine, then enrol the
    # resulting global_id + embedding into the gallery service. This is the
    # exact composition described in architecture_document.md 2.1: the
    # association engine assigns identity, the gallery is the durable
    # embedding store queried for future re-identification.
    enrolled = 0
    for t in tracklets:
        global_id = engine.associate_tracklet(t)
        gallery.enrol(global_id=global_id, embedding=t.embedding, timestamp=t.end_time)
        enrolled += 1

    print(f"\nFed {enrolled} tracklet associations into the gallery.")
    print("AFTER: gallery state ->", gallery.get_stats())
    print(f"Association engine formed {len(engine.identities)} global identities "
          f"(ground truth: 20 people).")

    # Demonstrate query(): take person #1's true embedding, add fresh noise
    # (simulating a brand-new sighting on a new camera), and confirm the
    # gallery's nearest-neighbor search returns the correct global_id that
    # the association engine assigned to person #1's tracklets.
    person_1_tracklets = [t for t in tracklets if t.gt_id == 1]
    true_global_id = None
    for gid, identity in engine.identities.items():
        if person_1_tracklets[0] in identity.tracklets:
            true_global_id = gid
            break

    base_embedding = person_1_tracklets[0].embedding.copy()
    query_embedding = base_embedding + np.random.normal(0, 0.02, base_embedding.shape).astype(np.float32)
    query_embedding /= np.linalg.norm(query_embedding)

    results = gallery.query(query_embedding, top_k=3)

    print("\nQuery: fresh noisy sighting of ground-truth person #1")
    print(f"  Association engine's global_id for person #1: {true_global_id}")
    print(f"  Gallery top-3 nearest neighbors: {results}")

    top_hit = results[0]["global_id"] if results else None
    match = (top_hit == true_global_id)
    print(f"  Gallery's #1 result matches association engine's global_id: {match}")

    print("\n" + "=" * 70)
    print("Result: Option A's identity decisions and Option B's embedding "
          "store agree end-to-end." if match else
          "Result: MISMATCH -- gallery top result does not match association engine.")
    print("=" * 70)


if __name__ == "__main__":
    main()

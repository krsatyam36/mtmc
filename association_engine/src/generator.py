import numpy as np
from .engine import Tracklet, CameraTopology

def generate_synthetic_data(num_people=20, appearance_noise=0.02, seed=42):
    np.random.seed(seed)
    
    # 1. Define Camera Topology
    topology = CameraTopology()
    topology.add_transition("Cam_1", "Cam_2", 15.0, 3.0)
    topology.add_transition("Cam_2", "Cam_3", 20.0, 4.0)
    topology.add_transition("Cam_3", "Cam_4", 10.0, 2.0)
    topology.add_transition("Cam_4", "Cam_5", 25.0, 5.0)
    
    # Define plausible sequence of cameras for people to traverse
    paths = [
        ["Cam_1", "Cam_2", "Cam_3", "Cam_4", "Cam_5"],
        ["Cam_2", "Cam_3", "Cam_4", "Cam_5"],
        ["Cam_1", "Cam_2", "Cam_3"],
    ]

    # 2. Generate Base Embeddings for each person (256-D)
    embeddings = {}
    for i in range(1, num_people + 1):
        # Generate random vector on unit sphere
        v = np.random.normal(0, 1, 256)
        v /= np.linalg.norm(v)
        embeddings[i] = v
        
    # Introduce an "Appearance Confusion Pair"
    # Make Person 2's embedding very close to Person 1's embedding (cosine sim ~ 0.96)
    # This helps demonstrate how priors resolve appearance overlap
    if num_people >= 2:
        embeddings[2] = embeddings[1] + np.random.normal(0, 0.04, 256)
        embeddings[2] /= np.linalg.norm(embeddings[2])

    tracklets = []
    
    # 3. Simulate Trajectories
    current_time = 0.0
    for gt_id in range(1, num_people + 1):
        # Assign path
        path = paths[gt_id % len(paths)]
        
        # Start time for this person
        t = current_time + np.random.uniform(0, 30)
        
        for idx, camera in enumerate(path):
            duration = np.random.uniform(5, 12)
            start_t = t
            end_t = start_t + duration
            
            # Generate noisy embedding for this specific tracklet
            noise = np.random.normal(0, appearance_noise, 256)
            noisy_emb = embeddings[gt_id] + noise
            noisy_emb /= np.linalg.norm(noisy_emb)
            
            # Instantiate tracklet
            tracklet = Tracklet(
                camera_id=camera,
                local_track_id=100 + idx,
                start_time=start_t,
                end_time=end_t,
                entry_zone="entry",
                exit_zone="exit",
                embedding=noisy_emb.tolist(),
                gt_id=gt_id
            )
            tracklets.append(tracklet)
            
            # Compute time of transition to next camera in path
            if idx < len(path) - 1:
                next_cam = path[idx + 1]
                mean_sec, std_sec = topology.transitions[(camera, next_cam)]
                transition_time = np.random.normal(mean_sec, std_sec)
                t = end_t + max(0.5, transition_time) # Ensure dt > 0.5s
        
        # Increment time offset slightly for next person
        current_time += 15.0

    # Sort tracklets by start_time to simulate streaming order
    tracklets.sort(key=lambda x: x.start_time)
    
    return tracklets, topology

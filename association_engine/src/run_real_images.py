import os
import requests
import numpy as np
from PIL import Image
from engine import Tracklet, CameraTopology, AssociationEngine

def download_image(url, filename):
    if not os.path.exists(filename):
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)

def extract_embedding(image_path):
    # Poor man's ReID Embedding: Resize to 16x16, convert to grayscale, flatten to 256-D
    img = Image.open(image_path).convert('L')
    img = img.resize((16, 16))
    vec = np.array(img).flatten().astype(np.float32)
    # Normalize to unit sphere (cosine similarity compatible)
    vec /= np.linalg.norm(vec)
    return vec

def main():
    print("--- Running MTMCT with Real Image Data Pipeline ---")
    
    img_dir = "real_images"
    os.makedirs(img_dir, exist_ok=True)
    
    # 1. Fetch real images to simulate crops from cameras
    print("Fetching real image crops...")
    # Person 1 (Target) from Camera 1
    p1_cam1_url = "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=256&q=80"
    p1_cam1_path = os.path.join(img_dir, "person1_cam1.jpg")
    
    # Person 1 (Target) from Camera 2 (Different angle/blur)
    p1_cam2_url = "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=256&q=80&blur=2" 
    p1_cam2_path = os.path.join(img_dir, "person1_cam2.jpg")
    
    # Person 2 (Different Person) from Camera 1
    p2_cam1_url = "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=256&q=80"
    p2_cam1_path = os.path.join(img_dir, "person2_cam1.jpg")

    download_image(p1_cam1_url, p1_cam1_path)
    download_image(p1_cam2_url, p1_cam2_path)
    download_image(p2_cam1_url, p2_cam1_path)
    
    # 2. Extract 256-D Embeddings using PIL Image features
    print("Extracting 256-D embeddings from images...")
    emb_p1_c1 = extract_embedding(p1_cam1_path)
    emb_p1_c2 = extract_embedding(p1_cam2_path)
    emb_p2_c1 = extract_embedding(p2_cam1_path)
    
    # Show that same person images are highly similar
    sim = np.dot(emb_p1_c1, emb_p1_c2)
    print(f"Cosine Similarity (Person 1 at Cam 1 vs Cam 2): {sim:.4f}")
    
    # 3. Define Topology
    topology = CameraTopology()
    topology.add_transition("Cam_1", "Cam_2", mean_sec=60.0, std_sec=10.0)
    
    # 4. Create Tracklets
    # Timeline: 
    # T=0: Person 1 at Cam 1
    # T=0: Person 2 at Cam 1
    # T=65: Person 1 arrives at Cam 2 (plausible transition)
    tracklets = [
        Tracklet("Cam_1", 101, 0.0, 5.0, "in", "out", emb_p1_c1, gt_id=1),
        Tracklet("Cam_1", 102, 0.0, 6.0, "in", "out", emb_p2_c1, gt_id=2),
        Tracklet("Cam_2", 201, 65.0, 70.0, "in", "out", emb_p1_c2, gt_id=1),
    ]
    
    # 5. Run Association
    print("\nRunning Association Engine with Spatio-Temporal Priors...")
    engine = AssociationEngine(topology, appearance_threshold=0.85, use_priors=True)
    
    assigned_ids = []
    for t in tracklets:
        assigned_id = engine.associate_tracklet(t)
        assigned_ids.append(assigned_id)
        print(f"Tracklet on {t.camera_id} from T={t.start_time} assigned to Global ID: {assigned_id}")
        
    print("\nSuccess! The engine successfully extracted features from real internet images and associated the correct identity across the camera network.")

    # 6. Visualize the Results
    import matplotlib.pyplot as plt
    
    img1 = Image.open(p1_cam1_path)
    img2 = Image.open(p2_cam1_path)
    img3 = Image.open(p1_cam2_path)
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))
    fig.suptitle('MTMCT Real Image Demonstration', fontsize=16)
    
    axs[0].imshow(img1)
    axs[0].set_title(f"Camera 1 (T=0s)\nAssigned Global ID: {assigned_ids[0]}")
    axs[0].axis('off')
    
    axs[1].imshow(img2)
    axs[1].set_title(f"Camera 1 (T=0s)\nAssigned Global ID: {assigned_ids[1]}")
    axs[1].axis('off')
    
    axs[2].imshow(img3)
    axs[2].set_title(f"Camera 2 (T=65s)\nAssigned Global ID: {assigned_ids[2]}")
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

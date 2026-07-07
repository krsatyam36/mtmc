import os
import random
import requests
import concurrent.futures
from PIL import Image, ImageEnhance, ImageFilter

NUM_PEOPLE = 150 # Total unique people
MAX_SIGHTINGS = 20 # Max times a person is seen across the network
NUM_CAMERAS = 20

# We are inside src/, so dataset should be created in the parent dir
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

def fetch_base_face(person_id):
    """Fetches a real face image from randomuser.me"""
    gender = "men" if person_id % 2 == 0 else "women"
    img_idx = (person_id // 2) % 100
    url = f"https://randomuser.me/api/portraits/{gender}/{img_idx}.jpg"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return resp.content
    except Exception:
        pass
    return None

def apply_camera_augmentation(image_path, cam_id):
    """Simulates different camera qualities and lighting"""
    try:
        img = Image.open(image_path)
        
        # Random brightness shift based on camera
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.7, 1.3))
        
        # Random blur based on camera distance
        if random.random() > 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
            
        img.save(image_path)
    except Exception as e:
        pass

def process_person(person_id):
    content = fetch_base_face(person_id)
    if not content:
        return 0
        
    base_path = os.path.join(DATASET_DIR, f".base_{person_id}.jpg")
    with open(base_path, "wb") as f:
        f.write(content)
        
    num_sightings = random.randint(5, MAX_SIGHTINGS)
    current_time = random.uniform(0, 1000)
    current_cam = random.randint(1, NUM_CAMERAS)
    
    generated = 0
    for s in range(num_sightings):
        # Move to a connected camera (simplified random walk)
        next_cam = current_cam + random.choice([-1, 0, 1])
        if next_cam < 1: next_cam = 2
        if next_cam > NUM_CAMERAS: next_cam = NUM_CAMERAS - 1
        
        # Advance time
        dt = random.uniform(10, 120) 
        current_time += dt
        
        cam_dir = os.path.join(DATASET_DIR, f"Cam_{next_cam:02d}")
        os.makedirs(cam_dir, exist_ok=True)
        
        filename = f"{current_time:.2f}_gt{person_id}.jpg"
        out_path = os.path.join(cam_dir, filename)
        
        # Copy and augment
        with open(base_path, "rb") as f_in, open(out_path, "wb") as f_out:
            f_out.write(f_in.read())
            
        apply_camera_augmentation(out_path, next_cam)
        generated += 1
        current_cam = next_cam
        
    # Clean up base image
    if os.path.exists(base_path):
        os.remove(base_path)
        
    return generated

def main():
    print(f"Building massive real-face dataset for {NUM_PEOPLE} identities...")
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    total_generated = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(process_person, p_id) for p_id in range(1, NUM_PEOPLE + 1)]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            total_generated += future.result()
            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{NUM_PEOPLE} identities... ({total_generated} tracklets generated so far)")
                
    print(f"\nDataset generation complete! Created {total_generated} tracking events across {NUM_CAMERAS} cameras.")

if __name__ == "__main__":
    main()

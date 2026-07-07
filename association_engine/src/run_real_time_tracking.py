import os
import csv
import sys
import glob
import numpy as np
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

# Add current dir to path to import engine
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from engine import Tracklet, CameraTopology, AssociationEngine

# We are inside src/, so dataset is at ../dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CSV_LOG_FILE = os.path.join(BASE_DIR, "tracking_logs.csv")
NUM_CAMERAS = 20

def extract_embedding(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((16, 16))
        vec = np.array(img).flatten().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec
    except:
        return np.zeros(768, dtype=np.float32)

def build_topology():
    topology = CameraTopology()
    for c in range(1, NUM_CAMERAS + 1):
        for n in [c-1, c, c+1]:
            if 1 <= n <= NUM_CAMERAS:
                mean_t = 60.0 if n != c else 10.0
                std_t = 20.0 if n != c else 5.0
                topology.add_transition(f"Cam_{c:02d}", f"Cam_{n:02d}", mean_t, std_t)
    return topology

def show_post_run_analytics():
    """Reads the generated CSV and displays a static comprehensive Data Analysis Dashboard."""
    print("\nGenerating Post-Run Data Analysis Dashboard...")
    
    camera_counts = defaultdict(int)
    person_counts = defaultdict(int)
    person_cam_dist = defaultdict(lambda: defaultdict(int))
    
    with open(CSV_LOG_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cam = row['Camera_ID']
            gid = row['Assigned_Global_ID']
            
            camera_counts[cam] += 1
            person_counts[gid] += 1
            person_cam_dist[gid][cam] += 1
            
    if not camera_counts:
        print("No data to analyze.")
        return

    # Sort data
    top_cams = sorted(camera_counts.items(), key=lambda x: x[1], reverse=True)
    top_persons = sorted(person_counts.items(), key=lambda x: x[1], reverse=True)[:10] # top 10
    
    plt.ioff() # Ensure interactive mode is off for final plot
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Post-Run Data Analysis Dashboard', fontsize=20, fontweight='bold')
    gs = gridspec.GridSpec(2, 2)
    
    # 1. Busiest Cameras (Bar Chart)
    ax1 = fig.add_subplot(gs[0, 0])
    cams = [x[0].replace('Cam_', '') for x in top_cams]
    counts = [x[1] for x in top_cams]
    ax1.bar(cams, counts, color='skyblue', edgecolor='black')
    ax1.set_title('Busiest Cameras (Total Detections)', fontweight='bold')
    ax1.set_xlabel('Camera ID')
    ax1.set_ylabel('Detections')
    
    # 2. Most Tracked Identities (Bar Chart)
    ax2 = fig.add_subplot(gs[0, 1])
    p_ids = [f"ID {x[0]}" for x in top_persons]
    p_counts = [x[1] for x in top_persons]
    ax2.bar(p_ids, p_counts, color='salmon', edgecolor='black')
    ax2.set_title('Top 10 Most Tracked Identities', fontweight='bold')
    ax2.set_ylabel('Total Sightings')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Camera Distribution for the #1 Most Tracked Person (Pie Chart)
    ax3 = fig.add_subplot(gs[1, 0])
    if top_persons:
        top_1_id = top_persons[0][0]
        dist = person_cam_dist[top_1_id]
        labels = list(dist.keys())
        sizes = list(dist.values())
        ax3.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Set3.colors)
        ax3.set_title(f'Where was the #1 Person (ID {top_1_id}) Seen?', fontweight='bold')
        
    # 4. Network Traffic Over Time (Simulated from sorted counts)
    ax4 = fig.add_subplot(gs[1, 1])
    # Just plotting cumulative sum to show traffic growth
    cumulative = np.cumsum([1 for _ in range(sum(counts))])
    ax4.plot(cumulative, color='purple', linewidth=2)
    ax4.set_title('Cumulative Traffic Volume over Time', fontweight='bold')
    ax4.set_xlabel('Time (Events)')
    ax4.set_ylabel('Total Processed')
    ax4.grid(True)
    
    plt.tight_layout()
    print("Dashboard ready! Please close the window to exit the program.")
    plt.show()

def main():
    print("="*60)
    print("MTMCT Cross-Camera Tracking System")
    print("="*60)
    print("Please select an execution mode:")
    print("1. Logs Only (Fast Headless Mode) -> Post-Run Data Analytics Dashboard")
    print("2. Logs + Live Video GUI Feed -> Post-Run Data Analytics Dashboard")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    use_gui = (choice == "2")
    
    print("\nInitializing MTMCT Real-Time Tracking Core...\n")
    all_files = glob.glob(f"{DATASET_DIR}/Cam_*/*.jpg")
    if not all_files:
        print("Dataset not found! Please run build_large_dataset.py first.")
        return
        
    events = []
    for f in all_files:
        cam_dir = os.path.basename(os.path.dirname(f))
        filename = os.path.basename(f)
        timestamp_str, gt_str = filename.replace(".jpg", "").split("_gt")
        events.append({
            "cam": cam_dir,
            "time": float(timestamp_str),
            "path": f,
            "gt_id": int(gt_str)
        })
        
    events.sort(key=lambda x: x["time"])
    
    topology = build_topology()
    engine = AssociationEngine(topology, appearance_threshold=0.85, use_priors=True)
    analytics = defaultdict(lambda: {"total": 0, "cameras": defaultdict(int)})
    id_to_face = {}
    local_id_counter = 1000

    if use_gui:
        # === ADVANCED GUI SETUP ===
        plt.ion() 
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle('LIVE: MTMCT Cross-Camera Tracking Analytics Dashboard', fontsize=18, fontweight='bold')
        gs = gridspec.GridSpec(2, 5, height_ratios=[1.2, 1])
        
        axes_live = []
        img_plots = []
        for i in range(5):
            ax = fig.add_subplot(gs[0, i])
            ax.axis('off')
            img_plots.append(ax.imshow(np.zeros((128, 128, 3), dtype=np.uint8)))
            axes_live.append(ax)
            ax.set_title("Waiting...", fontsize=10)

        ax_pie = fig.add_subplot(gs[1, 0:2])
        ax_pie.set_title("Live Traffic Distribution", fontweight='bold')
        
        axes_faces = []
        img_faces = []
        for i in range(2, 5):
            ax = fig.add_subplot(gs[1, i])
            ax.axis('off')
            img_faces.append(ax.imshow(np.zeros((128, 128, 3), dtype=np.uint8)))
            axes_faces.append(ax)
            ax.set_title(f"Rank #{i-1} ID")

        plt.tight_layout()
        plt.show(block=False)

    print(f"Starting simulated stream. Logs will be saved to {CSV_LOG_FILE}...\n")
    
    with open(CSV_LOG_FILE, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Timestamp', 'Camera_ID', 'Local_Track_ID', 'Assigned_Global_ID', 'Ground_Truth_ID'])

        for i, event in enumerate(events):
            embedding = extract_embedding(event["path"])
            
            t = Tracklet(
                camera_id=event["cam"],
                local_track_id=local_id_counter,
                start_time=event["time"],
                end_time=event["time"] + 2.0,
                entry_zone="auto", exit_zone="auto",
                embedding=embedding, gt_id=event["gt_id"]
            )
            
            global_id = engine.associate_tracklet(t)
            
            csv_writer.writerow([f"{event['time']:.2f}", event["cam"], local_id_counter, global_id, event["gt_id"]])
            local_id_counter += 1
            
            stats = analytics[global_id]
            stats["total"] += 1
            stats["cameras"][event["cam"]] += 1
            
            # Print minimal logs if headless, otherwise full logs
            if not use_gui:
                sys.stdout.write(f"\r[LOG] Processed {i+1}/{len(events)} events... Latest: {event['cam']} -> ID {global_id}")
                sys.stdout.flush()
            else:
                print(f"[LIVE {event['time']:7.1f}s] {event['cam']} detected person -> Assigned Global ID: {global_id}")
                
                # === GUI UPDATE ===
            if use_gui:
                try:
                    display_img = Image.open(event["path"])
                    display_np = np.array(display_img)
                    
                    if global_id not in id_to_face:
                        id_to_face[global_id] = display_np

                    ax_idx = i % 5 
                    img_plots[ax_idx].set_data(display_np)
                    
                    # Properly remove old bounding boxes
                    for p in list(axes_live[ax_idx].patches):
                        p.remove()
                    
                    color = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta'][global_id % 7]
                    axes_live[ax_idx].set_title(f"LIVE: {event['cam']}\nGlobal ID: {global_id}", color=color, fontweight='bold', fontsize=11)
                    
                    rect = patches.Rectangle((25, 15), 78, 100, linewidth=3, edgecolor=color, facecolor='none')
                    axes_live[ax_idx].add_patch(rect)
                    
                    if i % 10 == 0:
                        top_ids = sorted(analytics.keys(), key=lambda k: analytics[k]["total"], reverse=True)
                        
                        ax_pie.clear()
                        top_5 = top_ids[:5]
                        counts = [analytics[k]["total"] for k in top_5]
                        labels = [f"ID {k}" for k in top_5]
                        if len(top_ids) > 5:
                            counts.append(sum(analytics[k]["total"] for k in top_ids[5:]))
                            labels.append("Others")
                        
                        ax_pie.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel1.colors)
                        ax_pie.set_title("Top 5 Most Tracked Identities", fontweight='bold')
                        
                        for rank_idx in range(3):
                            if rank_idx < len(top_ids):
                                gid = top_ids[rank_idx]
                                img_faces[rank_idx].set_data(id_to_face[gid])
                                axes_faces[rank_idx].set_title(f"Rank #{rank_idx+1}: Global ID {gid}\n(Seen {analytics[gid]['total']} times)", fontweight='bold')
                    
                    # More robust GUI update for Linux/Wayland
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    time.sleep(0.01) # Small delay to make it viewable
                    
                    if i % 50 == 0:
                        csv_file.flush()
                except Exception as e:
                    print(f"GUI Update Error: {e}")

    if use_gui:
        plt.ioff()
        plt.close(fig) # Close the live feed window

    print(f"\n\nSYSTEM RUN COMPLETE. All logs successfully saved to {CSV_LOG_FILE}.")
    
    # Finally, trigger Data Analysis Dashboard
    show_post_run_analytics()
    
if __name__ == "__main__":
    main()

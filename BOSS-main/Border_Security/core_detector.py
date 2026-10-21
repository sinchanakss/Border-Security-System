import cv2
import numpy as np
import time
import os
import winsound
from ultralytics import YOLO
import sqlite3 

# --- CONFIGURATION ---
PERSON_CLASS_ID = 0          # Class ID for 'person' in COCO dataset
ANIMAL_CLASS_ID = 15         # Class ID for 'cat' (or similar common animal in COCO dataset)
CONFIDENCE_THRESHOLD = 0.5   
ALERT_COOLDOWN_SECONDS = 5   

# --- Shared Files & DB ---
ALERT_STATUS_FILE = 'alert_status.txt'
FRAME_OUTPUT_FILE_LIVE = 'current_frame.jpg'
DB_NAME = 'surveillance_log.db' 
# Primary folders for storing threat images
ALERT_IMAGE_DIR = 'alert_images'             # <-- FOR HUMAN THREATS
ANIMAL_IMAGE_DIR = 'animal_images'           # <-- NEW: FOR ANIMAL THREATS
# --- END CONFIGURATION ---

last_alert_time = 0
ANIMAL_THREAT_TYPE = "Animal"
HUMAN_THREAT_TYPE = "Human"

# --- Database and Helper Functions ---
def init_db():
    # Ensure both alert directories exist
    os.makedirs(ALERT_IMAGE_DIR, exist_ok=True)
    os.makedirs(ANIMAL_IMAGE_DIR, exist_ok=True)
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS threat_log (
            id INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL,
            type TEXT NOT NULL,
            image_path TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def log_threat_to_db(frame, threat_type):
    """Saves the frame to the correct folder and logs the image path to the database."""
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    image_filename = f"{threat_type}_{timestamp}.png"
    
    # ⬇️ FIX: Determine the correct directory based on threat type ⬇️
    if threat_type == HUMAN_THREAT_TYPE:
        target_dir = ALERT_IMAGE_DIR
    elif threat_type == ANIMAL_THREAT_TYPE:
        target_dir = ANIMAL_IMAGE_DIR
    else:
        # Fallback for safety
        target_dir = ALERT_IMAGE_DIR
    
    image_path = os.path.join(target_dir, image_filename)
    cv2.imwrite(image_path, frame)
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO threat_log (timestamp, type, image_path) VALUES (?, ?, ?)", 
                   (timestamp, threat_type, image_path))
    conn.commit()
    conn.close()
    print(f"Logged {threat_type} threat at {timestamp}")

def update_live_feed(frame, status):
    """Updates the shared status file and the single live frame image."""
    try:
        with open(ALERT_STATUS_FILE, 'w') as f:
            f.write(status)
        cv2.imwrite(FRAME_OUTPUT_FILE_LIVE, frame)
    except Exception as e:
        print(f"Error updating live files: {e}")

def trigger_local_alert():
    """Triggers the local audio alert based on cooldown."""
    global last_alert_time
    current_time = time.time()
    if current_time - last_alert_time < ALERT_COOLDOWN_SECONDS:
        return
    print("--- ALERT: Intrusion detected. Triggering local sound alert. ---")
    try:
        winsound.Beep(500, 500) 
        last_alert_time = current_time
    except Exception as e:
        print(f"ERROR: Could not play sound. Error: {e}")
# --- End Database and Helper Functions ---


if __name__ == "__main__":
    init_db() 
    try:
        model = YOLO('yolov8n.pt') 
    except Exception as e:
        print(f"ERROR: Could not load YOLO model. {e}"); exit()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open laptop camera (Index 0). Exiting.")
        update_live_feed(np.zeros((480, 640, 3), dtype=np.uint8), "ERROR"); exit()

    print("--- Surveillance System Active (Human/Animal Detection) ---")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        current_detection_status = "NORMAL"
        detected_threat_type = None
        
        # Look for PERSON (0) and ANIMAL (15)
        # Note: YOLO's class 15 is typically 'cat', 16 is 'dog', 17 is 'horse' 
        # Using 15 is fine for a general 'animal' simulation.
        results = model(frame, verbose=False, classes=[PERSON_CLASS_ID, ANIMAL_CLASS_ID]) 
        
        for result in results:
            for box in result.boxes:
                if box.conf[0] > CONFIDENCE_THRESHOLD:
                    
                    class_id = int(box.cls[0])
                    
                    if class_id == PERSON_CLASS_ID:
                        detected_threat_type = HUMAN_THREAT_TYPE
                    elif class_id == ANIMAL_CLASS_ID:
                        detected_threat_type = ANIMAL_THREAT_TYPE

                    # Draw Bounding Box and Label
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    label = f"{detected_threat_type.upper()}: {box.conf[0]*100:.1f}%"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Log the threat
                    if detected_threat_type:
                        trigger_local_alert() 
                        log_threat_to_db(frame, detected_threat_type)
                        current_detection_status = "ALERT" 
                        break # Stop checking boxes once a high-confidence threat is logged

        update_live_feed(frame, current_detection_status)
        cv2.imshow('Detector: Press Q to Quit', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    update_live_feed(frame, "SYSTEM_OFF") 
    print("--- Surveillance System Shut Down ---")
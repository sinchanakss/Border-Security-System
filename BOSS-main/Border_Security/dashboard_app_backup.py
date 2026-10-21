import streamlit as st
from PIL import Image
import numpy as np
import time
import os
import sqlite3
import pandas as pd
import io 

# --- CONFIGURATION ---
ALERT_STATUS_FILE = 'alert_status.txt'      # Human Detection Status
FRAME_OUTPUT_FILE_LIVE = 'current_frame.jpg' # The single image for the live feed
RF_STATUS_FILE = 'rf_status.txt'            # RF Threat Status 
DB_NAME = 'surveillance_log.db'
ALERT_IMAGE_DIR = 'alert_images'
ANIMAL_IMAGE_DIR = 'animal_images'          # New folder path for animal captures
BACKGROUND_IMAGE_PATH = 'border.jpg'   # <-- SET YOUR IMAGE NAME HERE
# --- END CONFIGURATION ---

# --- Custom CSS Function (Handles background and dark theme) ---
def add_bg_from_local(image_file):
    import base64
    if not os.path.exists(image_file):
        # Allow the app to run even if the background image is missing
        print(f"Background image file not found at: {image_file}")
        return

    with open(image_file, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    
    # Inject CSS to set the background image and define text area styling
    st.markdown(
        f"""
        <style>
        /* 1. Set the MAIN background image and fixed properties */
        .stApp {{
            background-image: url("data:image/jpeg;base64,{data}");
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        /* 2. Target the main content blocks to give them a semi-transparent black background and white text. */
        .stApp, .css-1d3f9g6, .main, .block-container {{
            color: white; /* Ensures text is white */
            background-color: rgba(0, 0, 0, 0.7) !important; /* Semi-transparent black overlay */
        }}
        
        /* 3. Ensure the markdown headers and text are white */
        h1, h2, h3, h4, .stText, .stMarkdown {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Status Check Helper Functions ---
def check_alert_status():
    try:
        with open(ALERT_STATUS_FILE, 'r') as f:
            return f.read().strip()
    except FileNotFoundError: return "INIT"
    except Exception: return "ERROR"

def check_rf_status():
    try:
        with open(RF_STATUS_FILE, 'r') as f:
            return f.read().strip()
    except FileNotFoundError: return "RF_INIT"
    except Exception: return "RF_ERROR"
        
def get_most_recent_alert_image(threat_type="Human"):
    """Fetches the image path for the latest captured threat log of a specific type."""
    conn = sqlite3.connect(DB_NAME)
    query = f"""
        SELECT image_path, timestamp, type 
        FROM threat_log 
        WHERE type = '{threat_type}' 
        ORDER BY id DESC 
        LIMIT 1
    """
    try:
        df = pd.read_sql_query(query, conn)
        conn.close()
        if not df.empty:
            return df.iloc[0]['image_path'], df.iloc[0]['timestamp'], df.iloc[0]['type']
        return None, None, None
    except Exception:
        return None, None, None
# --- End Status Check Helper Functions ---


def display_dashboard():
    # --- 1. Apply Background Image ---
    add_bg_from_local(BACKGROUND_IMAGE_PATH) 

    # --- UI Setup (Run only once) ---
    st.set_page_config(
        page_title="Border Security System", 
        layout="wide", 
        initial_sidebar_state="collapsed", 
        menu_items={'About': "Border Surveillance System - Minor Project"}
    )
    
    st.title(" Border Security Monitoring System")
    st.markdown("---")

    status_header = st.empty()
    
    # Define tabs
    tab1, tab2 = st.tabs([" THREAT MONITORING", " VISUAL MONITORING"])
    
    # Define persistent placeholders for Tab 1
    with tab1:
        st.subheader("Current Surveillance Status Metrics")
        col_h, col_r = st.columns(2)
        with col_h:
            st.markdown("###  Human Intrusion Status")
            human_status_placeholder = st.empty()
        with col_r:
            st.markdown("###  Device Inference Status")
            rf_status_placeholder = st.empty()
            
    # Define persistent placeholders for Tab 2 (Three Columns)
    with tab2:
        live_col, human_log_col, animal_log_col = st.columns(3)
        
        with live_col:
            st.markdown("###  Live Surveillance Stream")
            live_placeholder = st.empty()
        with human_log_col:
            st.markdown("### Human Alert Image")
            human_log_placeholder = st.empty()
        with animal_log_col:
            st.markdown("###  Animal Alert Image")
            animal_log_placeholder = st.empty()

    # Start the main update loop
    while True:
        human_status = check_alert_status()
        rf_status = check_rf_status() 
        
        is_human_alert = (human_status == "ALERT")
        is_rf_alert = (rf_status == "RF_THREAT")
        is_major_alert = is_human_alert or is_rf_alert
        
        # 1. Update Overall Alert Banner (Global Header)
        if is_major_alert:
            status_header.error(" CRITICAL THREAT DETECTED! Review Status Panels. ")
        elif human_status == "NORMAL" and rf_status == "RF_CLEAR":
            status_header.success(" System Operational. ALL SYSTEMS NORMAL")
        else:
            status_header.info(" Monitoring...")

        
        # --- TAB 1: THREAT MONITORING (Update the persistent placeholders with direct status text) ---
        with tab1:
            # Column H: Human Detection Status
            with col_h:
                if human_status == "ALERT":
                    html = ('<div style="background-color: #ff4b4b; padding: 15px; border-radius: 8px; color: white; text-align: center;">'
                            '<h1>THREAT DETECTED</h1>'
                            '</div>')
                elif human_status == "NORMAL":
                    html = ('<div style="background-color: #008000; padding: 15px; border-radius: 8px; color: white; text-align: center;">'
                            '<h1>SAFE CONDITION</h1>'
                            '</div>')
                else:
                    html = f'<div style="padding: 15px; border-radius: 8px; color: #a9a9a9; text-align: center;">Human Detector: {human_status}</div>'
                
                human_status_placeholder.markdown(html, unsafe_allow_html=True)


            # Column R: RF Inference Status
            with col_r:
                if rf_status == "RF_THREAT":
                    html = ('<div style="background-color: #ff4b4b; padding: 15px; border-radius: 8px; color: white; text-align: center;">'
                            '<h1>THREAT DETECTED</h1>'
                            '</div>')
                elif rf_status == "RF_CLEAR":
                    html = ('<div style="background-color: #008000; padding: 15px; border-radius: 8px; color: white; text-align: center;">'
                            '<h1>SAFE CONDITION</h1>'
                            '</div>')
                else:
                    html = f'<div style="padding: 15px; border-radius: 8px; color: #a9a9a9; text-align: center;">RF Detector: {rf_status}</div>'
                
                rf_status_placeholder.markdown(html, unsafe_allow_html=True)
            
        
        # --- TAB 2: VISUAL MONITORING (Update the persistent placeholders) ---
        with tab2:
            
            # Left Column Update: Live Feed
            with live_col:
                try:
                    if os.path.exists(FRAME_OUTPUT_FILE_LIVE):
                        img = Image.open(FRAME_OUTPUT_FILE_LIVE)
                        live_placeholder.image(img, caption=f"Real-time Stream | Last Updated: {time.strftime('%H:%M:%S')}", use_container_width=True)
                    else:
                        live_placeholder.image(np.zeros((480, 640, 3), dtype=np.uint8), caption="Waiting for Camera Feed...", use_container_width=True)
                except Exception:
                    live_placeholder.error("Could not load live camera image.")

            # Center Column: Most Recent Human Alert Image
            with human_log_col:
                path, timestamp, threat_type = get_most_recent_alert_image(threat_type="Human")
                
                if path and os.path.exists(path):
                    try:
                        alert_img = Image.open(path) 
                        human_log_placeholder.image(alert_img, use_container_width=True, caption=f"Human Capture: {timestamp}")
                    except Exception:
                        human_log_placeholder.warning("Image file missing or failed to load.")
                else:
                    human_log_placeholder.info("No Human threats logged yet.")

            # Right Column: Most Recent Animal Alert Image
            with animal_log_col:
                path, timestamp, threat_type = get_most_recent_alert_image(threat_type="Animal")
                
                if path and os.path.exists(path):
                    try:
                        alert_img = Image.open(path) 
                        animal_log_placeholder.image(alert_img, use_container_width=True, caption=f"Animal Capture: {timestamp}")
                    except Exception:
                        animal_log_placeholder.warning("Image file missing or failed to load.")
                else:
                    animal_log_placeholder.info("No Animal threats logged yet.")


        # Control the refresh rate (0.5 second pause)
        time.sleep(0.5) 

if __name__ == "__main__":
    display_dashboard()
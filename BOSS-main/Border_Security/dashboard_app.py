import streamlit as st
from PIL import Image
import numpy as np
import time
import os
import sqlite3
import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

ALERT_STATUS_FILE = "alert_status.txt"
FRAME_OUTPUT_FILE_LIVE = "current_frame.jpg"
RF_STATUS_FILE = "rf_status.txt"
RF_DEVICES_FILE = "rf_devices.txt"

DB_NAME = "surveillance_log.db"

ALERT_IMAGE_DIR = "alert_images"
ANIMAL_IMAGE_DIR = "animal_images"

BACKGROUND_IMAGE_PATH = "border.jpg"


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Border Security System",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": "Border Surveillance System - Minor Project"
    }
)


# =========================================================
# BACKGROUND
# =========================================================

def add_bg_from_local(image_file):

    import base64

    if not os.path.exists(image_file):
        return

    with open(image_file, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    st.markdown(
        f"""
        <style>

        .stApp {{
            background-image: url("data:image/jpeg;base64,{data}");
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        .stApp,
        .css-1d3f9g6,
        .main,
        .block-container {{
            color: white;
            background-color: rgba(0, 0, 0, 0.7) !important;
        }}

        h1, h2, h3, h4, .stText, .stMarkdown {{
            color: white !important;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# STATUS FUNCTIONS
# =========================================================

def check_alert_status():

    try:
        with open(ALERT_STATUS_FILE, "r") as f:
            return f.read().strip()

    except FileNotFoundError:
        return "INIT"

    except Exception:
        return "ERROR"


def check_rf_status():

    try:
        with open(RF_STATUS_FILE, "r") as f:
            return f.read().strip()

    except FileNotFoundError:
        return "RF_INIT"

    except Exception:
        return "RF_ERROR"


# =========================================================
# READ NEARBY WI-FI NETWORKS
# =========================================================

def read_rf_devices():

    devices = []

    if not os.path.exists(RF_DEVICES_FILE):
        return devices

    try:

        with open(
            RF_DEVICES_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            lines = f.readlines()

        for line in lines:

            line = line.strip()

            # Skip headings and empty lines
            if not line:
                continue

            if line.startswith("Nearby Wi-Fi Networks"):
                continue

            if line.startswith("="):
                continue

            if line.startswith("No nearby"):
                continue

            try:

                # Expected format:
                #
                # 1. NetworkName | Signal: 100% | RSSI: -50.0 dBm

                parts = line.split("|")

                if len(parts) >= 3:

                    # Network name
                    network_name = parts[0]

                    if "." in network_name:
                        network_name = network_name.split(
                            ".", 1
                        )[1].strip()

                    # Signal
                    signal = parts[1].replace(
                        "Signal:",
                        ""
                    ).strip()

                    # RSSI
                    rssi = parts[2].replace(
                        "RSSI:",
                        ""
                    ).strip()

                    devices.append(
                        {
                            "Network": network_name,
                            "Signal": signal,
                            "RSSI": rssi
                        }
                    )

            except Exception:
                continue

    except Exception:
        return []

    return devices


# =========================================================
# DATABASE - LATEST ALERT IMAGE
# =========================================================

def get_most_recent_alert_image(
    threat_type="Human"
):

    conn = sqlite3.connect(DB_NAME)

    query = f"""
        SELECT image_path, timestamp, type
        FROM threat_log
        WHERE type = '{threat_type}'
        ORDER BY id DESC
        LIMIT 1
    """

    try:

        df = pd.read_sql_query(
            query,
            conn
        )

        conn.close()

        if not df.empty:

            return (
                df.iloc[0]["image_path"],
                df.iloc[0]["timestamp"],
                df.iloc[0]["type"]
            )

        return None, None, None

    except Exception:

        conn.close()

        return None, None, None


# =========================================================
# RF FRONTEND
# =========================================================

def display_rf_panel(rf_status):

    # -----------------------------------------------------
    # RF STATUS BOX
    # -----------------------------------------------------

    if rf_status == "RF_THREAT":

        st.markdown(
            """
            <div style="
                background-color:#ff4b4b;
                padding:20px;
                border-radius:12px;
                color:white;
                text-align:center;
                margin-bottom:10px;
            ">

                <h2>⚠ RF THREAT DETECTED</h2>

                <p>
                    Strong nearby RF activity detected.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    elif rf_status == "RF_CLEAR":

        st.markdown(
            """
            <div style="
                background-color:#008000;
                padding:20px;
                border-radius:12px;
                color:white;
                text-align:center;
                margin-bottom:10px;
            ">

                <h2>✓ RF SCAN ACTIVE</h2>

                <p>
                    No RF threat detected.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    elif rf_status == "RF_OFF":

        st.markdown(
            """
            <div style="
                background-color:#555555;
                padding:20px;
                border-radius:12px;
                color:white;
                text-align:center;
                margin-bottom:10px;
            ">

                <h2>RF DETECTOR OFF</h2>

                <p>
                    Start the RF detection backend
                    to scan nearby signals.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div style="
                background-color:#555555;
                padding:20px;
                border-radius:12px;
                color:white;
                text-align:center;
                margin-bottom:10px;
            ">

                <h2>RF DETECTOR</h2>

                <p>
                    Status: {rf_status}
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    # -----------------------------------------------------
    # RF INFORMATION
    # -----------------------------------------------------

    st.markdown(
        "#### Nearby Device / RF Scan"
    )

    info_col1, info_col2, info_col3 = st.columns(3)


    with info_col1:

        st.metric(
            label="RF Status",
            value=rf_status
        )


    with info_col2:

        st.metric(
            label="Detection",
            value=(
                "ACTIVE"
                if rf_status == "RF_CLEAR"
                else "CHECK"
            )
        )


    with info_col3:

        st.metric(
            label="Source",
            value="RF Scanner"
        )


    # -----------------------------------------------------
    # NEARBY WI-FI NETWORKS
    # -----------------------------------------------------

    st.markdown(
        "#### Nearby Wi-Fi Networks"
    )

    devices = read_rf_devices()


    if devices:

        device_df = pd.DataFrame(
            devices
        )

        st.dataframe(
            device_df,
            hide_index=True,
            width="stretch"
        )

        st.caption(
            "Nearby Wi-Fi networks detected "
            "using the laptop Wi-Fi adapter."
        )

    else:

        st.info(
            "No nearby Wi-Fi network information "
            "is available yet."
        )


# =========================================================
# MAIN DASHBOARD
# =========================================================

def display_dashboard():

    # Apply background
    add_bg_from_local(
        BACKGROUND_IMAGE_PATH
    )


    st.title(
        "🛡 Border Security Monitoring System"
    )

    st.markdown("---")


    status_header = st.empty()


    # =====================================================
    # TABS
    # =====================================================

    tab1, tab2 = st.tabs(
        [
            "⚠ THREAT MONITORING",
            "📹 VISUAL MONITORING"
        ]
    )


    # =====================================================
    # TAB 1
    # =====================================================

    with tab1:

        st.subheader(
            "Current Surveillance Status Metrics"
        )


        col_h, col_r = st.columns(2)


        # -------------------------------------------------
        # HUMAN STATUS
        # -------------------------------------------------

        with col_h:

            st.markdown(
                "### 👤 Human Intrusion Status"
            )

            human_status_placeholder = st.empty()


        # -------------------------------------------------
        # RF STATUS
        # -------------------------------------------------

        with col_r:

            st.markdown(
                "### 📡 Nearby Device / RF Status"
            )

            rf_status_placeholder = st.empty()


    # =====================================================
    # TAB 2
    # =====================================================

    with tab2:

        live_col, human_log_col, animal_log_col = st.columns(3)


        # -------------------------------------------------
        # LIVE CAMERA
        # -------------------------------------------------

        with live_col:

            st.markdown(
                "### 📹 Live Surveillance Stream"
            )

            live_placeholder = st.empty()


        # -------------------------------------------------
        # HUMAN ALERT
        # -------------------------------------------------

        with human_log_col:

            st.markdown(
                "### 👤 Human Alert Image"
            )

            human_log_placeholder = st.empty()


        # -------------------------------------------------
        # ANIMAL ALERT
        # -------------------------------------------------

        with animal_log_col:

            st.markdown(
                "### 🐾 Animal Alert Image"
            )

            animal_log_placeholder = st.empty()


    # =====================================================
    # MAIN UPDATE LOOP
    # =====================================================

    while True:

        human_status = check_alert_status()

        rf_status = check_rf_status()


        is_human_alert = (
            human_status == "ALERT"
        )

        is_rf_alert = (
            rf_status == "RF_THREAT"
        )

        is_major_alert = (
            is_human_alert or
            is_rf_alert
        )


        # =================================================
        # GLOBAL STATUS
        # =================================================

        if is_major_alert:

            status_header.error(
                "🚨 CRITICAL THREAT DETECTED! "
                "Review Status Panels."
            )

        elif (
            human_status == "NORMAL"
            and rf_status == "RF_CLEAR"
        ):

            status_header.success(
                "✓ System Operational. "
                "ALL SYSTEMS NORMAL"
            )

        else:

            status_header.info(
                "Monitoring..."
            )


        # =================================================
        # TAB 1
        # =================================================

        with tab1:

            # ---------------------------------------------
            # HUMAN DETECTION
            # ---------------------------------------------

            with col_h:

                if human_status == "ALERT":

                    html = (
                        '<div style="'
                        'background-color:#ff4b4b;'
                        'padding:15px;'
                        'border-radius:8px;'
                        'color:white;'
                        'text-align:center;">'

                        '<h2>⚠ THREAT DETECTED</h2>'

                        '</div>'
                    )


                elif human_status == "NORMAL":

                    html = (
                        '<div style="'
                        'background-color:#008000;'
                        'padding:15px;'
                        'border-radius:8px;'
                        'color:white;'
                        'text-align:center;">'

                        '<h2>✓ SAFE CONDITION</h2>'

                        '</div>'
                    )


                else:

                    html = (
                        '<div style="'
                        'padding:15px;'
                        'border-radius:8px;'
                        'color:#a9a9a9;'
                        'text-align:center;">'

                        f'Human Detector: '
                        f'{human_status}'

                        '</div>'
                    )


                human_status_placeholder.markdown(
                    html,
                    unsafe_allow_html=True
                )


            # ---------------------------------------------
            # RF / NEARBY DEVICE
            # ---------------------------------------------

            with col_r:

                display_rf_panel(
                    rf_status
                )


        # =================================================
        # TAB 2
        # =================================================

        with tab2:

            # ---------------------------------------------
            # LIVE CAMERA
            # ---------------------------------------------

            with live_col:

                try:

                    if os.path.exists(
                        FRAME_OUTPUT_FILE_LIVE
                    ):

                        img = Image.open(
                            FRAME_OUTPUT_FILE_LIVE
                        )


                        live_placeholder.image(
                            img,
                            caption=(
                                "Real-time Stream | "
                                "Last Updated: "
                                f"{time.strftime('%H:%M:%S')}"
                            ),
                            width="stretch"
                        )


                    else:

                        live_placeholder.image(
                            np.zeros(
                                (480, 640, 3),
                                dtype=np.uint8
                            ),
                            caption=(
                                "Waiting for Camera Feed..."
                            ),
                            width="stretch"
                        )


                except Exception:

                    live_placeholder.error(
                        "Could not load live camera image."
                    )


            # ---------------------------------------------
            # HUMAN ALERT
            # ---------------------------------------------

            with human_log_col:

                path, timestamp, threat_type = (
                    get_most_recent_alert_image(
                        threat_type="Human"
                    )
                )


                if (
                    path
                    and os.path.exists(path)
                ):

                    try:

                        alert_img = Image.open(
                            path
                        )


                        human_log_placeholder.image(
                            alert_img,
                            width="stretch",
                            caption=(
                                f"Human Capture: "
                                f"{timestamp}"
                            )
                        )


                    except Exception:

                        human_log_placeholder.warning(
                            "Image file missing "
                            "or failed to load."
                        )


                else:

                    human_log_placeholder.info(
                        "No Human threats logged yet."
                    )


            # ---------------------------------------------
            # ANIMAL ALERT
            # ---------------------------------------------

            with animal_log_col:

                path, timestamp, threat_type = (
                    get_most_recent_alert_image(
                        threat_type="Animal"
                    )
                )


                if (
                    path
                    and os.path.exists(path)
                ):

                    try:

                        animal_img = Image.open(
                            path
                        )


                        animal_log_placeholder.image(
                            animal_img,
                            width="stretch",
                            caption=(
                                f"Animal Capture: "
                                f"{timestamp}"
                            )
                        )


                    except Exception:

                        animal_log_placeholder.warning(
                            "Image file missing "
                            "or failed to load."
                        )


                else:

                    animal_log_placeholder.info(
                        "No Animal threats logged yet."
                    )


        # =================================================
        # REFRESH
        # =================================================

        time.sleep(0.5)


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    display_dashboard()
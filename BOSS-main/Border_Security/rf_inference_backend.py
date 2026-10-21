import time
import os
import subprocess

# --- CONFIGURATION ---
RF_STATUS_FILE = 'rf_status.txt'
RF_DEVICES_FILE = 'rf_devices.txt'

MAX_NETWORKS_COUNT = 5
MAX_RSSI_THRESHOLD = -50
# ---------------------


def run_wifi_scan():
    """Scan nearby Wi-Fi networks using the laptop Wi-Fi adapter."""
    cmd = ['netsh', 'wlan', 'show', 'networks', 'mode=bssid']

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        return result.stdout

    except Exception as e:
        print(f"ERROR: Failed to run Wi-Fi scan. {e}")
        return ""


def process_wifi_data(scan_output):
    """Analyze Wi-Fi networks and save their details."""

    lines = scan_output.splitlines()

    total_networks = 0
    strong_signals = 0
    devices = []

    current_ssid = None

    for line in lines:

        line = line.strip()

        # Find network name
        if line.startswith("SSID ") and ":" in line:
            current_ssid = line.split(":", 1)[1].strip()

            if current_ssid:
                total_networks += 1

        # Find signal strength
        elif line.startswith("Signal") and ":" in line and "%" in line:

            try:
                signal_percent = int(
                    line.split(":", 1)[1]
                    .strip()
                    .replace("%", "")
                )

                # Approximate RSSI from signal percentage
                rssi = (signal_percent / 2) - 100

                if rssi > MAX_RSSI_THRESHOLD:
                    strong_signals += 1

                if current_ssid:
                    devices.append({
                        "ssid": current_ssid,
                        "signal": signal_percent,
                        "rssi": round(rssi, 1)
                    })

            except ValueError:
                pass

    # Save nearby network information
    try:
        with open(RF_DEVICES_FILE, "w", encoding="utf-8") as f:

            f.write("Nearby Wi-Fi Networks\n")
            f.write("=====================\n\n")

            if devices:

                for i, device in enumerate(devices, 1):

                    f.write(
                        f"{i}. "
                        f"{device['ssid']} | "
                        f"Signal: {device['signal']}% | "
                        f"RSSI: {device['rssi']} dBm\n"
                    )

            else:
                f.write("No nearby Wi-Fi networks detected.\n")

    except Exception as e:
        print(f"Error writing device information: {e}")

    # Threat decision
    if total_networks <= MAX_NETWORKS_COUNT and strong_signals > 0:
        return "RF_THREAT"

    return "RF_CLEAR"


def update_rf_status(status):
    """Write the current RF status."""

    try:
        with open(RF_STATUS_FILE, "w", encoding="utf-8") as f:
            f.write(status)

    except Exception as e:
        print(f"Error writing RF status: {e}")


# --- MAIN PROGRAM ---

if __name__ == "__main__":

    print("--- RF Inference Backend Active ---")
    print("--- Scanning using laptop Wi-Fi adapter ---")

    update_rf_status("RF_INIT")

    try:

        while True:

            scan_data = run_wifi_scan()

            if scan_data:

                rf_status = process_wifi_data(scan_data)

                update_rf_status(rf_status)

                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"RF Status: {rf_status} | "
                    f"Networks detected: {scan_data.count('SSID ')}"
                )

            time.sleep(5)

    except KeyboardInterrupt:

        print("\n--- RF Inference Backend Shut Down ---")

        update_rf_status("RF_OFF")
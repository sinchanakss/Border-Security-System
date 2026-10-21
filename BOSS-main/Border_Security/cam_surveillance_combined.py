import os
# Force TCP for RTSP (harmless if not used) – must be set BEFORE importing cv2
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import time
import threading
import queue
import smtplib
import urllib.request
import numpy as np
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from ultralytics import YOLO
from flask import Flask, Response, render_template_string, jsonify

# ============================================================

# ============================================================
# CONFIG
# ============================================================

import os

# Email configuration
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "")

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Alert channel switches
ENABLE_EMAIL = True
ENABLE_SMS = False
ENABLE_VOICE = True
ENABLE_VOICE_CALL = False

# Twilio configuration
# Store your actual credentials in environment variables.
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "")
SMS_TO_NUMBER = os.getenv("SMS_TO_NUMBER", "")
CALL_TO_NUMBER = os.getenv("CALL_TO_NUMBER", "")

# ============================================================
# Keep the rest of your original code below this section.
# ============================================================

# ------------------------------------------------------------
#  CAMERA SOURCES
#   ESP32-CAM  → HTTP snapshot ("/capture"), polled every SNAPSHOT_INTERVAL s
#   PC Webcam  → continuous live stream (source = 0 / int)
#
#   If /capture returns 404 on your sketch, try :81/capture, /jpg, or :81/jpg
# ------------------------------------------------------------
CAMERA_SOURCES = [
    {"key": "cam0", "name": "ESP32-CAM 1",
     "source": "http://10.191.109.117/capture"},
    {"key": "cam1", "name": "ESP32-CAM 2",
     "source": "http://10.191.109.1/capture"},
    {"key": "cam2", "name": "PC Webcam",
     "source": 0},
]

# Friendly names used by the spoken alert ("Camera one", "Camera two", ...)
SPOKEN_CAM_NAMES = {
    "cam0": "Camera one",
    "cam1": "Camera two",
    "cam2": "PC webcam",
}

# COCO class IDs to detect
# person=0, cat=15, dog=16, horse=17, sheep=18, cow=19,
# elephant=20, zebra=22, giraffe=23
TARGET_CLASS_IDS = [0, 15, 16, 17, 18, 19, 20, 22, 23]

# ------------------------------------------------------------
#  CONFIDENCE THRESHOLD
#  Detections below this value are ignored entirely
#  (not drawn, not alerted, not logged).
#  Range: 0.0 – 1.0   e.g. 0.50 = 50 %
# ------------------------------------------------------------
MIN_CONFIDENCE = 0.75

# ------------------------------------------------------------
#  DETECTION / ALERT TUNING
# ------------------------------------------------------------
EMAIL_COOLDOWN    = 20        # seconds per (camera, class)
FOCAL_LENGTH_PX   = 700       # calibrate per camera for better distance
FRAME_WIDTH       = 640
FRAME_HEIGHT      = 480
JPEG_QUALITY      = 75

# ESP32-CAM: how often to grab a still + run detection
SNAPSHOT_INTERVAL = 5.0       # seconds

# Approximate real-world heights (cm) for distance estimation
KNOWN_HEIGHTS_CM = {
    "person":   170,
    "cat":       30,
    "dog":       60,
    "horse":    160,
    "sheep":     90,
    "cow":      150,
    "elephant": 300,
    "zebra":    140,
    "giraffe":  500,
}

# ------------------------------------------------------------
#  LOCAL IMAGE SAVING + EXCEL LOGGING
# ------------------------------------------------------------
ENABLE_LOCAL_IMAGE_SAVE = True            # write annotated JPEGs to disk
IMAGE_SAVE_DIR          = "detections"    # folder (created automatically)

ENABLE_EXCEL_LOG        = True            # log detections + e-mail time
EXCEL_LOG_PATH          = "detection_log.xlsx"

# How many of the most recent log rows to show on the dashboard
DASHBOARD_LOG_ROWS      = 30

# Optional dependency: openpyxl (pip install openpyxl)
try:
    import openpyxl
    from openpyxl.styles import Font as XLFont
    from openpyxl.utils import get_column_letter
    OPENPYXL_AVAILABLE = True
except ImportError:
    openpyxl = None
    XLFont = None
    get_column_letter = None
    OPENPYXL_AVAILABLE = False
    print("[WARN] openpyxl not installed -> Excel logging disabled "
          "(pip install openpyxl)")

# ============================================================
#  MODEL (shared across all threads)
# ============================================================
model = YOLO("yolov8n.pt")      # auto-download on first run
model_lock = threading.Lock()   # serialize inference across cameras

# ============================================================
#  OPTIONAL DEPENDENCIES: Twilio (SMS/call) + pyttsx3 (voice)
# ============================================================
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TwilioClient = None
    TWILIO_AVAILABLE = False
    print("[WARN] twilio not installed -> SMS / voice call disabled "
          "(pip install twilio)")

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    pyttsx3 = None
    TTS_AVAILABLE = False
    print("[WARN] pyttsx3 not installed -> local voice disabled "
          "(pip install pyttsx3)")

# ============================================================
#  EMAIL
# ============================================================
last_email_time = {}
email_lock = threading.Lock()


def send_email_with_image(subject, body, image_bytes, recipient):
    msg = MIMEMultipart()
    msg["From"]    = EMAIL_SENDER
    msg["To"]      = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    image = MIMEImage(image_bytes, _subtype="jpeg")
    image.add_header("Content-Disposition",
                     "attachment", filename="detection.jpg")
    msg.attach(image)

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        sent_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[EMAIL] Sent → {recipient}  ({subject})")
        return sent_at
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")
        return None


# ============================================================
#  LOCAL IMAGE SAVING
# ============================================================
_image_save_lock = threading.Lock()


def ensure_image_dir():
    """Create the image folder if it does not exist."""
    if ENABLE_LOCAL_IMAGE_SAVE:
        os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)


def save_detection_image(image_bytes, cam_key, class_name):
    """
    Write the annotated JPEG to IMAGE_SAVE_DIR.
    Returns the absolute path (str) or None on failure / if disabled.
    """
    if not ENABLE_LOCAL_IMAGE_SAVE:
        return None
    try:
        with _image_save_lock:
            os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
            # millisecond precision keeps filenames unique
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            safe_cls = "".join(c if c.isalnum() else "_" for c in class_name)
            filename = f"{ts}_{cam_key}_{safe_cls}.jpg"
            path = os.path.join(IMAGE_SAVE_DIR, filename)
            with open(path, "wb") as f:
                f.write(image_bytes)
        print(f"[IMG] saved → {path}")
        return os.path.abspath(path)
    except Exception as e:
        print(f"[IMG SAVE ERROR] {e}")
        return None


# ============================================================
#  EXCEL LOGGING
# ============================================================
_excel_lock = threading.Lock()

EXCEL_HEADERS = [
    "Detection Time", "Camera Key", "Camera Name", "Object", "Confidence",
    "Distance (m)", "Angle (deg)", "Image Path",
    "Email Sent", "Email Time", "SMS Sent", "Voice Alert", "Email Recipient",
]


def _create_excel_if_missing():
    """Create the workbook + header row the first time we log."""
    if os.path.exists(EXCEL_LOG_PATH):
        return
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Detections"
    ws.append(EXCEL_HEADERS)
    for cell in ws[1]:
        cell.font = XLFont(bold=True)
    ws.freeze_panes = "A2"
    widths = [20, 11, 16, 12, 11, 13, 12, 55, 11, 20, 10, 12, 28]
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    wb.save(EXCEL_LOG_PATH)


def log_detection_to_excel(row):
    """
    Append one detection row to EXCEL_LOG_PATH.
    `row` is a dict; missing keys become empty strings.
    """
    if not (ENABLE_EXCEL_LOG and OPENPYXL_AVAILABLE):
        return
    with _excel_lock:
        try:
            _create_excel_if_missing()
            wb = openpyxl.load_workbook(EXCEL_LOG_PATH)
            ws = wb.active
            ws.append([
                row.get("detection_time", ""),
                row.get("cam_key", ""),
                row.get("cam_name", ""),
                row.get("class_name", ""),
                row.get("confidence", ""),
                row.get("distance_m", ""),
                row.get("angle_deg", ""),
                row.get("image_path", ""),
                row.get("email_sent", ""),
                row.get("email_time", ""),
                row.get("sms_sent", ""),
                row.get("voice_sent", ""),
                row.get("recipient", ""),
            ])
            wb.save(EXCEL_LOG_PATH)
            print(f"[EXCEL] logged {row.get('class_name')} "
                  f"@ {row.get('detection_time')} "
                  f"(mail: {row.get('email_time') or 'not sent'})")
        except Exception as e:
            print(f"[EXCEL ERROR] {e}")


def read_excel_log(limit=DASHBOARD_LOG_ROWS):
    """
    Return the most recent `limit` rows from the Excel log,
    newest first, as a list of dicts (keys = EXCEL_HEADERS).
    """
    if not (ENABLE_EXCEL_LOG and OPENPYXL_AVAILABLE):
        return []
    if not os.path.exists(EXCEL_LOG_PATH):
        return []
    try:
        with _excel_lock:
            wb = openpyxl.load_workbook(EXCEL_LOG_PATH, read_only=True)
            ws = wb.active
            rows = list(ws.iter_rows(values_only=True))
            wb.close()
        if not rows:
            return []
        header = list(rows[0])
        data_rows = rows[1:]
        # keep only the last `limit` rows, newest first
        data_rows = data_rows[-limit:] if limit else data_rows
        data_rows = list(reversed(data_rows))
        result = []
        for r in data_rows:
            item = {}
            for i, key in enumerate(header):
                val = r[i] if i < len(r) else ""
                # make values JSON-safe
                if isinstance(val, datetime):
                    val = val.strftime("%Y-%m-%d %H:%M:%S")
                item[key] = "" if val is None else val
            result.append(item)
        return result
    except Exception as e:
        print(f"[EXCEL READ ERROR] {e}")
        return []


# ============================================================
#  SMS  (Twilio)
# ============================================================
def send_sms(body):
    """Send an SMS via Twilio. Returns True/False."""
    if not (ENABLE_SMS and TWILIO_AVAILABLE):
        return False
    try:
        client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=body,
            from_=TWILIO_FROM_NUMBER,
            to=SMS_TO_NUMBER,
        )
        print(f"[SMS] Sent → {SMS_TO_NUMBER}  ({body})")
        return True
    except Exception as e:
        print(f"[SMS ERROR] {e}")
        return False


# ============================================================
#  VOICE  (local speaker via pyttsx3)
#  pyttsx3's engine is NOT reliably reusable across runAndWait()
#  calls on Windows SAPI5 / some Linux backends, so we spin up a
#  fresh engine per utterance. Slightly slower but 100% reliable.
# ============================================================
_tts_queue = queue.Queue()
_tts_stop  = threading.Event()


def _speak_once(text):
    """Create a fresh engine, say one phrase, then tear it down."""
    if not TTS_AVAILABLE:
        return
    engine = None
    try:
        print(f"[TTS] speaking: {text}")
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.setProperty("volume", 1.0)
        # Optional: pick an English voice
        # for v in engine.getProperty("voices"):
        #     if "english" in v.name.lower():
        #         engine.setProperty("voice", v.id)
        #         break
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[TTS ERROR] {e}")
    finally:
        # Explicitly stop + release so the next call starts clean
        try:
            if engine is not None:
                engine.stop()
        except Exception:
            pass
        engine = None


def _tts_worker():
    if not TTS_AVAILABLE:
        return
    print("[TTS] voice worker ready (fresh engine per phrase)")
    while not _tts_stop.is_set():
        try:
            text = _tts_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if text is None:            # shutdown sentinel
            break
        _speak_once(text)


def start_tts_worker():
    """Call once at startup if ENABLE_VOICE."""
    if ENABLE_VOICE and TTS_AVAILABLE:
        threading.Thread(target=_tts_worker, daemon=True).start()


def speak(text):
    """Queue a phrase for spoken playback (non-blocking). Returns True/False."""
    if ENABLE_VOICE and TTS_AVAILABLE:
        _tts_queue.put(text)
        return True
    return False


# ============================================================
#  VOICE CALL  (optional, Twilio Programmable Voice)
# ============================================================
def send_voice_call(say_text):
    if not (ENABLE_VOICE_CALL and TWILIO_AVAILABLE):
        return False
    try:
        client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        twiml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f'<Response><Say voice="alice">{say_text}</Say>'
            '<Say voice="alice">Goodbye.</Say></Response>'
        )
        client.calls.create(
            twiml=twiml,
            from_=TWILIO_FROM_NUMBER,
            to=CALL_TO_NUMBER,
        )
        print(f"[CALL] Placed → {CALL_TO_NUMBER}")
        return True
    except Exception as e:
        print(f"[CALL ERROR] {e}")
        return False


# ============================================================
#  ALERT TEXT BUILDERS
# ============================================================
def angle_speech(angle_deg):
    """Turn +12.3 into '12 degrees to the right'."""
    if angle_deg is None:
        return "unknown angle"
    if abs(angle_deg) < 1.0:
        return "straight ahead"
    side = "right" if angle_deg > 0 else "left"
    return f"{abs(angle_deg):.0f} degrees to the {side}"


def build_alert_texts(cam_key, cam_name, speech_name, class_name,
                      conf, distance_m, angle_deg):
    dist_txt = f"{distance_m:.1f} m" if distance_m is not None else "n/a"
    ang_txt  = f"{angle_deg:+.1f} deg" if angle_deg is not None else "n/a"

    # ---- EMAIL ----
    subject = f"[Detection] {class_name} on {cam_name}"
    dist_line = (f"Distance  : {distance_m:.1f} m\n"
                 if distance_m is not None else "Distance  : n/a\n")
    body = (
        f"Camera    : {cam_name} ({cam_key})\n"
        f"Object    : {class_name}\n"
        f"Confidence: {conf:.2f}\n"
        f"{dist_line}"
        f"Angle     : {ang_txt}\n"
        f"Time      : {time.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # ---- SMS ----
    sms_body = (
        f"{cam_name} ({cam_key}): {class_name} detected | "
        f"dist {dist_txt} | angle {ang_txt}"
    )

    # ---- SPOKEN ----
    speech = (
        f"Alert. {speech_name}. {class_name} detected. "
        f"Distance {dist_txt}. Angle {angle_speech(angle_deg)}."
    )

    return subject, body, sms_body, speech


def dispatch_alerts(cam_key, cam_name, speech_name, class_name, conf,
                    distance_m, angle_deg, image_bytes,
                    image_path=None, detection_time=None):
    """
    Runs in its own thread → fires email + SMS + voice together,
    then writes the detection (and the e-mail time) to the Excel log.
    """
    subject, body, sms_body, speech = build_alert_texts(
        cam_key, cam_name, speech_name, class_name,
        conf, distance_m, angle_deg
    )

    if detection_time is None:
        detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---- EMAIL (records the exact send time) ----
    email_time = None
    if ENABLE_EMAIL:
        email_time = send_email_with_image(
            subject, body, image_bytes, EMAIL_RECEIVER
        )

    # ---- SMS / CALL / LOCAL VOICE ----
    sms_sent   = send_sms(sms_body)          # False if disabled
    call_sent  = send_voice_call(speech)     # False if disabled
    voice_sent = speak(speech)               # False if disabled

    # ---- EXCEL LOG ----
    log_detection_to_excel({
        "detection_time": detection_time,
        "cam_key":        cam_key,
        "cam_name":       cam_name,
        "class_name":     class_name,
        "confidence":     round(conf, 3),
        "distance_m":     round(distance_m, 2) if distance_m is not None else "",
        "angle_deg":      round(angle_deg, 1) if angle_deg is not None else "",
        "image_path":     image_path or "",
        "email_sent":     ("Yes" if email_time
                           else ("Failed" if ENABLE_EMAIL else "Disabled")),
        "email_time":     email_time or "",
        "sms_sent":       ("Yes" if sms_sent
                           else ("Disabled" if not ENABLE_SMS else "Failed")),
        "voice_sent":     ("Yes" if (voice_sent or call_sent)
                           else "Disabled"),
        "recipient":      EMAIL_RECEIVER if ENABLE_EMAIL else "",
    })


# ============================================================
#  DISTANCE & ANGLE
# ============================================================
def estimate_distance(class_name, bbox_height_px):
    real_h = KNOWN_HEIGHTS_CM.get(class_name)
    if real_h is None or bbox_height_px < 5:
        return None
    dist_cm = (real_h * FOCAL_LENGTH_PX) / bbox_height_px
    return dist_cm / 100.0        # metres


def estimate_angle(bbox_center_x, frame_width):
    offset = bbox_center_x - frame_width / 2.0
    return float(np.degrees(np.arctan(offset / FOCAL_LENGTH_PX)))


# ============================================================
#  PER-FRAME DETECTION
# ============================================================
def process_frame(frame, cam_key, cam_name):
    with model_lock:
        results = model(frame, verbose=False)[0]

    annotated = frame.copy()
    detections = []

    speech_name = SPOKEN_CAM_NAMES.get(cam_key, cam_name)

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in TARGET_CLASS_IDS:
            continue

        conf = float(box.conf[0])

        # ---- CONFIDENCE FILTER ----
        if conf < MIN_CONFIDENCE:
            continue

        class_name = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        bbox_h   = y2 - y1
        bbox_w   = x2 - x1
        center_x = x1 + bbox_w / 2.0

        distance_m = estimate_distance(class_name, bbox_h) / 10
        angle_deg  = estimate_angle(center_x, frame.shape[1])

        # --- draw bbox ---
        color = (0, 255, 0) if class_name == "person" else (0, 165, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name} {conf:.2f}"
        if distance_m is not None:
            label += f" | {distance_m:.1f}m"
        label += f" | {angle_deg:+.0f}deg"
        cv2.putText(annotated, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        detections.append({
            "class": class_name,
            "confidence": round(conf, 3),
            "distance_m": round(distance_m, 2) if distance_m is not None else None,
            "angle_deg":  round(angle_deg, 1),
            "bbox": [x1, y1, x2, y2],
        })

        # --- alert (EMAIL_COOLDOWN s cooldown per cam + class) ---
        key = (cam_key, class_name)
        now = time.time()
        with email_lock:
            last = last_email_time.get(key, 0)
            should_send = (now - last) >= EMAIL_COOLDOWN
            if should_send:
                last_email_time[key] = now

        if should_send:
            _, buf = cv2.imencode(".jpg", annotated,
                                  [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            image_bytes = buf.tobytes()

            detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            image_path     = save_detection_image(image_bytes, cam_key,
                                                  class_name)

            threading.Thread(
                target=dispatch_alerts,
                args=(cam_key, cam_name, speech_name, class_name, conf,
                      distance_m, angle_deg, image_bytes,
                      image_path, detection_time),
                daemon=True,
            ).start()

    # --- overlay summary text on the frame ---
    y = 30
    cv2.putText(annotated, cam_name, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for d in detections:
        y += 22
        txt = f"{d['class']}: {d['distance_m']}m  {d['angle_deg']:+.0f}deg"
        cv2.putText(annotated, txt, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)

    return annotated, detections


# ============================================================
#  SNAPSHOT GRABBER  (for ESP32-CAM HTTP "/capture")
# ============================================================
def snapshot_once(url, timeout=5):
    """Fetch a single JPEG over HTTP. Returns BGR frame or None."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = r.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[SNAP] {url} error: {e}")
        return None


# ============================================================
#  CAMERA WORKER  (reader thread + processor thread)
# ============================================================
class CameraWorker:
    def __init__(self, source, cam_key, cam_name):
        self.source   = source
        self.cam_key  = cam_key
        self.cam_name = cam_name

        # HTTP/ESP32 → snapshot mode; int/RTSP → live mode
        self.is_snapshot = (isinstance(source, str)
                            and source.startswith("http"))

        self.cap = None
        self.running = True

        # newest raw frame
        self.raw_frame = None
        self.raw_lock  = threading.Lock()

        # newest annotated JPEG + detections
        self.jpeg        = None
        self.jpeg_lock   = threading.Lock()
        self.detections  = []
        self.det_lock    = threading.Lock()

        self._open()

        self.reader_thread = threading.Thread(target=self._reader,    daemon=True)
        self.proc_thread   = threading.Thread(target=self._processor, daemon=True)

    # ---------- open ----------
    def _open(self):
        if isinstance(self.source, int):
            # Local webcam
            self.cap = cv2.VideoCapture(self.source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            if self.cap.isOpened():
                print(f"[OK] {self.cam_name} ({self.cam_key}) webcam opened")
            else:
                print(f"[WARN] {self.cam_name} ({self.cam_key}) webcam failed")
        elif self.is_snapshot:
            # ESP32-CAM / HTTP snapshot – handled by urllib, not OpenCV
            self.cap = None
            print(f"[OK] {self.cam_name} ({self.cam_key}) "
                  f"snapshot every {SNAPSHOT_INTERVAL:.0f}s → {self.source}")
        else:
            # RTSP fallback
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            if self.cap.isOpened():
                print(f"[OK] {self.cam_name} ({self.cam_key}) RTSP opened")
            else:
                print(f"[WARN] {self.cam_name} ({self.cam_key}) RTSP failed")

    # ---------- threads ----------
    def start(self):
        self.reader_thread.start()
        self.proc_thread.start()

    def _reader(self):
        # ---- ESP32-CAM: grab one JPEG every SNAPSHOT_INTERVAL seconds ----
        if self.is_snapshot:
            while self.running:
                frame = snapshot_once(self.source)
                if frame is not None:
                    with self.raw_lock:
                        self.raw_frame = frame
                time.sleep(SNAPSHOT_INTERVAL)
            return

        # ---- Webcam / RTSP: continuous live stream ----
        while self.running:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                with self.raw_lock:
                    self.raw_frame = frame
            else:
                time.sleep(0.05)

    def _processor(self):
        last_run = 0.0
        while self.running:
            # Throttle inference for snapshot cameras only
            if self.is_snapshot:
                now = time.time()
                if now - last_run < SNAPSHOT_INTERVAL:
                    time.sleep(0.05)
                    continue
                last_run = now

            with self.raw_lock:
                frame = self.raw_frame
            if frame is None:
                time.sleep(0.05)
                continue

            try:
                annotated, dets = process_frame(frame, self.cam_key, self.cam_name)
                ok, buf = cv2.imencode(
                    ".jpg", annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )
                if ok:
                    with self.jpeg_lock:
                        self.jpeg = buf.tobytes()
                    with self.det_lock:
                        self.detections = dets
            except Exception as e:
                print(f"[PROC ERROR {self.cam_key}] {e}")
                time.sleep(0.1)

    # ---------- accessors ----------
    def get_jpeg(self):
        with self.jpeg_lock:
            return self.jpeg

    def get_detections(self):
        with self.det_lock:
            return list(self.detections)

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()


# ============================================================
#  FLASK APP
# ============================================================

app     = Flask(__name__)
workers = {}   # cam_key -> CameraWorker

# ============================================================
#  RF STATUS + THREAT SCORE
#  Reads the RF backend files produced by rf_inference_backend.py
# ============================================================
RF_STATUS_FILE = "rf_status.txt"
RF_DEVICES_FILE = "rf_devices.txt"


def read_rf_status():
    """Read the latest RF status from the RF backend."""
    try:
        if not os.path.exists(RF_STATUS_FILE):
            return "RF_UNKNOWN"
        with open(RF_STATUS_FILE, "r", encoding="utf-8") as f:
            status = f.read().strip()
        return status or "RF_UNKNOWN"
    except Exception as e:
        print(f"[RF READ ERROR] {e}")
        return "RF_UNKNOWN"


def read_rf_devices():
    """Read nearby Wi-Fi devices from the RF backend text file."""
    devices = []
    try:
        if not os.path.exists(RF_DEVICES_FILE):
            return devices

        with open(RF_DEVICES_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("Nearby Wi-Fi") or line.startswith("==="):
                continue

            # Expected:
            # 1. SSID | Signal: 75% | RSSI: -62.5 dBm
            if ". " not in line or "|" not in line:
                continue

            try:
                number, rest = line.split(". ", 1)
                parts = [p.strip() for p in rest.split("|")]

                network = parts[0] if parts else "Unknown"
                signal = parts[1].replace("Signal:", "").strip() if len(parts) > 1 else "—"
                rssi = parts[2].replace("RSSI:", "").strip() if len(parts) > 2 else "—"

                devices.append({
                    "network": network,
                    "signal": signal,
                    "rssi": rssi
                })
            except Exception:
                continue

    except Exception as e:
        print(f"[RF DEVICES READ ERROR] {e}")

    return devices


def calculate_threat_score():
    """
    Threat Score:
      Human detected             +50
      RF threat                  +30
      Human confidence >= 90%    +20
      Human confidence 75-89%    +10
      Maximum score               100
    """
    score = 0
    human_detected = False
    highest_human_confidence = 0.0

    # Check current YOLO detections from all cameras.
    for worker in workers.values():
        for detection in worker.get_detections():
            if str(detection.get("class", "")).lower() == "person":
                human_detected = True
                confidence = float(detection.get("confidence", 0.0))
                highest_human_confidence = max(
                    highest_human_confidence, confidence
                )

    if human_detected:
        score += 50

        if highest_human_confidence >= 0.90:
            score += 20
        elif highest_human_confidence >= 0.75:
            score += 10

    rf_status = read_rf_status()
    if rf_status == "RF_THREAT":
        score += 30

    score = min(score, 100)

    if score <= 30:
        level = "LOW"
    elif score <= 60:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return {
        "score": score,
        "level": level,
        "human_detected": human_detected,
        "human_confidence": round(highest_human_confidence, 3),
        "rf_status": rf_status,
    }


# ---------- inline HTML dashboard ----------
HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Multi-Camera YOLO Detection</title>
<style>
  * { box-sizing: border-box; }
  body { margin:0; background:#0d1117; color:#e6edf3;
         font-family: system-ui, Segoe UI, Arial, sans-serif; padding:20px; }
  h1   { text-align:center; color:#3fb950; margin:0 0 6px; letter-spacing:1px; }
  .sub { text-align:center; color:#8b949e; font-size:13px; margin:0 0 22px; }
  .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
          gap:20px; }
  .cam  { background:#161b22; border:1px solid #30363d; border-radius:10px;
          padding:12px; box-shadow:0 2px 8px #0006; }
  .cam h2 { margin:0 0 8px; color:#58a6ff; font-size:16px;
            display:flex; justify-content:space-between; align-items:center; }
  .dot { width:10px; height:10px; border-radius:50%; background:#3fb950;
         animation:pulse 1.5s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
  .cam img { width:100%; border-radius:6px; display:block; background:#000;
             aspect-ratio: 4 / 3; object-fit: cover; }
  .dets { margin-top:10px; font-size:13px; max-height:160px; overflow-y:auto; }
  .det-item { padding:5px 8px; background:#21262d; border-left:3px solid #d29922;
              margin:3px 0; border-radius:3px; display:flex;
              justify-content:space-between; gap:8px; }
  .det-item.person { border-left-color:#3fb950; }
  .det-item .cls  { font-weight:600; text-transform:capitalize; }
  .det-item .meta { color:#8b949e; font-size:12px; }
  .empty { color:#6e7681; font-style:italic; padding:6px; }

  /* ---- detection log table ---- */
  .log-wrap { margin-top:34px; background:#161b22; border:1px solid #30363d;
              border-radius:10px; padding:14px 16px;
              box-shadow:0 2px 8px #0006; }
  .log-wrap h2 { margin:0 0 12px; color:#58a6ff; font-size:16px;
                 display:flex; justify-content:space-between; align-items:center; }
  .log-wrap h2 .count { color:#8b949e; font-size:12px; font-weight:400; }
  .log-table-scroll { max-height:440px; overflow:auto; border-radius:6px; }
  table.log { width:100%; border-collapse:collapse; font-size:12.5px; }
  table.log th, table.log td { padding:7px 9px; text-align:left;
                               border-bottom:1px solid #21262d; white-space:nowrap; }
  table.log th { position:sticky; top:0; background:#1c2128; color:#c9d1d9;
                 font-weight:600; z-index:1; }
  table.log tr:hover td { background:#1c2128; }
  table.log td.cls { text-transform:capitalize; font-weight:600; }
  table.log td.cls.person { color:#3fb950; }
  .pill { padding:2px 7px; border-radius:10px; font-size:11px; font-weight:600; }
  .pill.yes  { background:#1f6f38; color:#d3f9d8; }
  .pill.no   { background:#4a1f1f; color:#ffd7d7; }
  .pill.dim  { background:#30363d; color:#8b949e; }
  .img-link { color:#58a6ff; text-decoration:none; }
  .img-link:hover { text-decoration:underline; }

  /* ---- threat score ---- */
  .top-panels { display:grid; grid-template-columns: 1fr 1fr;
                gap:20px; margin-bottom:20px; }
  .status-card { background:#161b22; border:1px solid #30363d;
                 border-radius:10px; padding:18px; box-shadow:0 2px 8px #0006; }
  .status-card h2 { margin:0 0 12px; color:#58a6ff; font-size:16px; }
  .score-value { font-size:42px; font-weight:700; text-align:center;
                 margin:8px 0; }
  .score-level { text-align:center; font-size:15px; font-weight:700;
                 letter-spacing:1px; }
  .score-low { color:#3fb950; }
  .score-medium { color:#d29922; }
  .score-high { color:#f85149; }
  .score-details { margin-top:12px; color:#8b949e; font-size:13px;
                   line-height:1.7; }
  .rf-status { font-size:20px; font-weight:700; margin-bottom:12px; }
  .rf-clear { color:#3fb950; }
  .rf-threat { color:#f85149; }
  .rf-unknown { color:#d29922; }
  .rf-table-scroll { max-height:220px; overflow:auto; }
  table.rf-table { width:100%; border-collapse:collapse; font-size:13px; }
  table.rf-table th, table.rf-table td { padding:7px 9px; text-align:left;
                                         border-bottom:1px solid #21262d; }
  table.rf-table th { background:#1c2128; color:#c9d1d9; }
  @media (max-width: 800px) {
    .top-panels { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
  <h1>Multi-Camera YOLO Detection Dashboard</h1>
  <p class="sub">Minimum confidence: <b>{{ min_conf_pct }}%</b> &nbsp;·&nbsp;
     Cameras: <b>{{ cameras|length }}</b></p>

  <!-- ---------- THREAT SCORE + RF STATUS ---------- -->
  <div class="top-panels">
    <div class="status-card">
      <h2>Threat Score</h2>
      <div class="score-value" id="threat-score">0 / 100</div>
      <div class="score-level score-low" id="threat-level">LOW</div>
      <div class="score-details" id="threat-details">
        Human: No &nbsp;·&nbsp; RF: Unknown
      </div>
    </div>

    <div class="status-card">
      <h2>Nearby Wi-Fi Networks</h2>
      <div class="rf-status rf-unknown" id="rf-status">RF_UNKNOWN</div>
      <div class="rf-table-scroll">
        <table class="rf-table">
          <thead>
            <tr>
              <th>Network</th>
              <th>Signal</th>
              <th>RSSI</th>
            </tr>
          </thead>
          <tbody id="rf-body">
            <tr><td colspan="3" class="empty">Waiting for RF data…</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="grid">
    {% for cam in cameras %}
    <div class="cam">
      <h2>{{ cam.name }} <span class="dot"></span></h2>
      <img src="/video/{{ cam.key }}" alt="{{ cam.name }}">
      <div class="dets" id="dets-{{ cam.key }}">
        <div class="empty">Waiting for detections…</div>
      </div>
    </div>
    {% endfor %}
  </div>

  <!-- ---------- DETECTION LOG ---------- -->
  <div class="log-wrap">
    <h2>Detection Log
      <span class="count" id="log-count">—</span>
    </h2>
    <div class="log-table-scroll">
      <table class="log">
        <thead>
          <tr>
            <th>Detection Time</th>
            <th>Camera</th>
            <th>Object</th>
            <th>Conf.</th>
            <th>Distance</th>
            <th>Angle</th>
            <th>Image</th>
            <th>Email</th>
            <th>Email Time</th>
            <th>SMS</th>
            <th>Voice</th>
          </tr>
        </thead>
        <tbody id="log-body">
          <tr><td colspan="11" class="empty">Loading log…</td></tr>
        </tbody>
      </table>
    </div>
  </div>

<script>
/* ---------- threat score ---------- */
async function refreshThreatScore() {
  try {
    const r = await fetch('/threat_score');
    const data = await r.json();

    document.getElementById('threat-score').textContent =
      data.score + ' / 100';

    const levelEl = document.getElementById('threat-level');
    levelEl.textContent = data.level;
    levelEl.className = 'score-level score-' + data.level.toLowerCase();

    const humanText = data.human_detected
      ? 'Yes (' + (data.human_confidence * 100).toFixed(0) + '%)'
      : 'No';

    document.getElementById('threat-details').innerHTML =
      'Human: ' + humanText +
      ' &nbsp;·&nbsp; RF: ' + String(data.rf_status || 'RF_UNKNOWN');
  } catch (e) {
    /* ignore temporary connection errors */
  }
}

/* ---------- RF devices ---------- */
async function refreshRF() {
  try {
    const r = await fetch('/rf_info');
    const data = await r.json();

    const statusEl = document.getElementById('rf-status');
    const status = String(data.status || 'RF_UNKNOWN');

    statusEl.textContent = status;

    if (status === 'RF_THREAT') {
      statusEl.className = 'rf-status rf-threat';
    } else if (status === 'RF_CLEAR') {
      statusEl.className = 'rf-status rf-clear';
    } else {
      statusEl.className = 'rf-status rf-unknown';
    }

    const body = document.getElementById('rf-body');

    if (!Array.isArray(data.devices) || data.devices.length === 0) {
      body.innerHTML =
        '<tr><td colspan="3" class="empty">No nearby Wi-Fi networks found.</td></tr>';
      return;
    }

    body.innerHTML = data.devices.map(d => `
      <tr>
        <td>${esc(d.network)}</td>
        <td>${esc(d.signal)}</td>
        <td>${esc(d.rssi)}</td>
      </tr>
    `).join('');
  } catch (e) {
    /* ignore temporary connection errors */
  }
}

/* ---------- live per-camera detections ---------- */
async function refreshDetections() {
  try {
    const r = await fetch('/detections');
    const data = await r.json();

    for (const camKey in data) {
      const el = document.getElementById('dets-' + camKey);
      if (!el) continue;
      const list = data[camKey];
      if (!list || list.length === 0) {
        el.innerHTML = '<div class="empty">No detections</div>';
        continue;
      }
      el.innerHTML = list.map(d => {
        const cls  = d.class === 'person' ? 'person' : '';
        const dist = (d.distance_m != null) ? d.distance_m.toFixed(1) + ' m' : 'n/a';
        const ang  = (d.angle_deg  != null) ? d.angle_deg.toFixed(0) + '\u00B0' : 'n/a';
        const conf = (d.confidence * 100).toFixed(0) + '%';
        return `<div class="det-item ${cls}">
                  <span class="cls">${d.class}</span>
                  <span class="meta">${conf} &middot; ${dist} &middot; ${ang}</span>
                </div>`;
      }).join('');
    }
  } catch (e) { /* ignore */ }
}

/* ---------- detection log table ---------- */
function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function pill(val) {
  const v = String(val || '').toLowerCase();
  if (v === 'yes') return '<span class="pill yes">Yes</span>';
  if (v === 'failed') return '<span class="pill no">Failed</span>';
  if (v === 'disabled' || v === '') return '<span class="pill dim">—</span>';
  return '<span class="pill dim">' + esc(val) + '</span>';
}
async function refreshLog() {
  try {
    const r = await fetch('/log');
    const rows = await r.json();
    const body = document.getElementById('log-body');
    const count = document.getElementById('log-count');

    if (!Array.isArray(rows) || rows.length === 0) {
      body.innerHTML = '<tr><td colspan="11" class="empty">No log entries yet.</td></tr>';
      count.textContent = '0 entries';
      return;
    }
    count.textContent = rows.length + ' entries (newest first)';

    body.innerHTML = rows.map(r => {
      const cls = String(r['Object'] || '').toLowerCase() === 'person' ? 'person' : '';
      const dist = (r['Distance (m)'] !== '' && r['Distance (m)'] != null)
                    ? Number(r['Distance (m)']).toFixed(1) + ' m' : 'n/a';
      const ang  = (r['Angle (deg)'] !== '' && r['Angle (deg)'] != null)
                    ? Number(r['Angle (deg)']).toFixed(0) + '\u00B0' : 'n/a';
      const conf = (r['Confidence'] !== '' && r['Confidence'] != null)
                    ? (Number(r['Confidence']) * 100).toFixed(0) + '%' : 'n/a';
      const imgCell = r['Image Path']
        ? `<a class="img-link" href="/image/${encodeURIComponent(r['Image Path'].split(/[\\/]/).pop())}"
              target="_blank">view</a>`
        : '—';
      return `<tr>
        <td>${esc(r['Detection Time'])}</td>
        <td>${esc(r['Camera Name'] || r['Camera Key'])}</td>
        <td class="cls ${cls}">${esc(r['Object'])}</td>
        <td>${conf}</td>
        <td>${dist}</td>
        <td>${ang}</td>
        <td>${imgCell}</td>
        <td>${pill(r['Email Sent'])}</td>
        <td>${esc(r['Email Time']) || '—'}</td>
        <td>${pill(r['SMS Sent'])}</td>
        <td>${pill(r['Voice Alert'])}</td>
      </tr>`;
    }).join('');
  } catch (e) {
    document.getElementById('log-body').innerHTML =
      '<tr><td colspan="11" class="empty">Could not load log.</td></tr>';
  }
}

setInterval(refreshThreatScore, 1000);
setInterval(refreshRF, 2000);
refreshThreatScore();
refreshRF();
setInterval(refreshDetections, 1000);
setInterval(refreshLog, 3000);
refreshDetections();
refreshLog();
</script>
</body>
</html>
"""

# ---------- routes ----------
@app.route("/")
def index():
    return render_template_string(
        HTML_PAGE,
        cameras=CAMERA_SOURCES,
        min_conf_pct=int(round(MIN_CONFIDENCE * 100)),
    )


def _mjpeg_generator(worker):
    """Yield the worker's newest JPEG as an MJPEG stream."""
    last_jpeg = None
    while True:
        jpeg = worker.get_jpeg()
        if jpeg is not None and jpeg is not last_jpeg:
            last_jpeg = jpeg
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                   + jpeg + b"\r\n")
        time.sleep(0.03)


@app.route("/video/<cam_key>")
def video(cam_key):
    worker = workers.get(cam_key)
    if worker is None:
        return "Camera not found", 404
    return Response(_mjpeg_generator(worker),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/detections")
def detections():
    return jsonify({
        key: worker.get_detections()
        for key, worker in workers.items()
    })


@app.route("/threat_score")
def threat_score():
    return jsonify(calculate_threat_score())


@app.route("/rf_info")
def rf_info():
    return jsonify({
        "status": read_rf_status(),
        "devices": read_rf_devices()
    })


@app.route("/log")
def log_endpoint():
    """Return the most recent Excel log rows as JSON."""
    return jsonify(read_excel_log(DASHBOARD_LOG_ROWS))


@app.route("/image/<filename>")
def serve_detection_image(filename):
    """Serve a saved detection JPEG (safe: no path traversal)."""
    # strip any directory components to prevent traversal
    safe_name = os.path.basename(filename)
    path = os.path.join(IMAGE_SAVE_DIR, safe_name)
    if not os.path.exists(path):
        return "Not found", 404
    from flask import send_file
    return send_file(path, mimetype="image/jpeg")


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    # Prepare local storage for images + Excel log
    if ENABLE_LOCAL_IMAGE_SAVE:
        ensure_image_dir()
        print(f"[OK] Detection images → {os.path.abspath(IMAGE_SAVE_DIR)}")

    if ENABLE_EXCEL_LOG and OPENPYXL_AVAILABLE:
        _create_excel_if_missing()
        print(f"[OK] Excel log        → {os.path.abspath(EXCEL_LOG_PATH)}")

    print(f"[OK] Min confidence   → {MIN_CONFIDENCE:.2f} "
          f"({int(round(MIN_CONFIDENCE * 100))}%)")

    # Start the local TTS worker once (if voice is enabled)
    start_tts_worker()

    # Start camera workers
    for cam in CAMERA_SOURCES:
        w = CameraWorker(cam["source"], cam["key"], cam["name"])
        w.start()
        workers[cam["key"]] = w

    time.sleep(1.5)   # let cameras warm up

    # Start Flask (threaded so multiple camera streams work at once)
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)

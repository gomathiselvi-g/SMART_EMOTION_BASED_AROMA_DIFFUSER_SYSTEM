"""Smart Emotion-Based Aroma System - Flask web application.

Face verification is a local academic-demo feature using OpenCV LBPH.
It is not a high-assurance biometric authentication system and does not
include robust anti-spoof/liveness detection.
"""

from flask import (
    Flask,
    render_template,
    Response,
    jsonify,
    request,
    session,
    redirect,
    url_for,
    flash,
    send_file,
    send_from_directory
)
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from tensorflow.keras.models import load_model
from collections import Counter, deque
import random
from datetime import datetime, timedelta
from pathlib import Path
import base64
import os
import re
import secrets
import sqlite3
import subprocess
import time
import threading
import firebase_admin
from firebase_admin import credentials, messaging

try:
    import serial  # pyserial; Arduino is optional during software testing
except ImportError:
    serial = None

import cv2
import numpy as np


# =========================================================
# PATHS AND FLASK CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "emotion_history.db"
FACE_DIR = BASE_DIR / "face_data"
FACE_DIR.mkdir(parents=True, exist_ok=True)
FACE_MODEL_PATH = FACE_DIR / "face_recognizer.yml"  # legacy LBPH model; no longer used
YUNET_MODEL_PATH = FACE_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_PATH = FACE_DIR / "face_recognition_sface_2021dec.onnx"
RECORDINGS_DIR = BASE_DIR / "recordings"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
MAX_RECORDING_SECONDS = max(15, int(os.environ.get("AROMA_MAX_CLIP_SECONDS", "60")))  # legacy continuous-clip cap
DEFAULT_RECORDING_INTERVAL_SECONDS = 30 * 60  # legacy SQLite/API field; no longer used as a wait timer
DEFAULT_RECORDING_DURATION_SECONDS = 30 * 60
MAX_RECORDING_INTERVAL_SECONDS = 7 * 24 * 60 * 60
MAX_SCHEDULED_RECORDING_DURATION_SECONDS = 60 * 60  # up to one hour per clip to limit disk use
RECORDING_RETENTION_DAYS = max(1, int(os.environ.get("AROMA_RECORDING_RETENTION_DAYS", "7")))
CAMERA_TARGET_FPS = max(4, min(15, int(os.environ.get("AROMA_CAMERA_TARGET_FPS", "8"))))
# Browser-friendly recording frame rate. Frames may be repeated when CNN processing
# is slower than real-time so the saved MP4 duration reflects elapsed time.
RECORDING_FPS = max(1, min(6, int(os.environ.get("AROMA_RECORDING_FPS", "4"))))
DAILY_REPORT_HOUR = min(23, max(0, int(os.environ.get("AROMA_DAILY_REPORT_HOUR", "20"))))
CAMERA_DEVICE_INDEX = int(os.environ.get("AROMA_CAMERA_INDEX", "0"))

app = Flask(__name__)
# Firebase Admin SDK initialization
FIREBASE_CREDENTIALS_PATH = (
    BASE_DIR / "secrets" / "firebase-service-account.json"
)

try:
    if not firebase_admin._apps:
        if FIREBASE_CREDENTIALS_PATH.is_file():
            firebase_cred = credentials.Certificate(
                str(FIREBASE_CREDENTIALS_PATH)
            )
            firebase_admin.initialize_app(firebase_cred)
            print("Firebase Admin initialized successfully.")
        else:
            print("WARNING: Firebase service account file not found.")
except Exception:
    app.logger.exception("Firebase Admin initialization failed")

@app.route("/firebase-messaging-sw.js")
def firebase_messaging_service_worker():
    return send_from_directory(
        app.root_path,
        "firebase-messaging-sw.js",
        mimetype="application/javascript"
    )
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

if not os.environ.get("FLASK_SECRET_KEY"):
    print("WARNING: Using a temporary development session secret.")
    print("Set FLASK_SECRET_KEY in the environment before deployment.")


# Find the existing CNN model without requiring you to edit a hard-coded path.
_model_candidates = [
    os.environ.get("AROMA_MODEL_PATH", ""),
    str(BASE_DIR / "templates" / "model.h5"),
    str(BASE_DIR / "model.h5"),
    r"C:\Users\Gomathi Selvi\Desktop\templates\model.h5",
]
MODEL_PATH = next((p for p in _model_candidates if p and os.path.isfile(p)), None)
if MODEL_PATH is None:
    raise FileNotFoundError(
        "Could not find model.h5. Put it in the project's templates folder, "
        "put it beside app.py, or set the AROMA_MODEL_PATH environment variable."
    )

print(f"Loading CNN model from: {MODEL_PATH}")
classifier = load_model(MODEL_PATH)
print("CNN model loaded successfully.")

# Face detection and identity recognition models (OpenCV Zoo: YuNet + SFace).
# SFace embeddings are more suitable for identity matching across normal head turns
# than LBPH grayscale crops. Enrollment must collect multiple poses.
if not YUNET_MODEL_PATH.is_file() or not SFACE_MODEL_PATH.is_file():
    raise FileNotFoundError(
        "Face recognition model files are missing. Run `python download_face_models.py` "
        "from the project root, then restart the app."
    )

if not hasattr(cv2, "FaceDetectorYN") or not hasattr(cv2, "FaceRecognizerSF"):
    raise RuntimeError(
        "This OpenCV build lacks YuNet/SFace APIs. Install opencv-contrib-python==4.11.0.86."
    )

face_model_lock = threading.RLock()
face_detector_sface = cv2.FaceDetectorYN.create(
    str(YUNET_MODEL_PATH), "", (320, 320), 0.70, 0.30, 5000
)
face_embedding_model = cv2.FaceRecognizerSF.create(str(SFACE_MODEL_PATH), "")
print("YuNet face detector and SFace identity model loaded.")

# =========================================================
# EMOTIONS AND RECOMMENDATIONS
# =========================================================

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]

recommendations = {
    "Happy": {
        "aroma": "Citrus",
        "suggestion": "Enjoy the positive mood with relaxing music.",
    },
    "Sad": {
        "aroma": "Lavender",
        "suggestion": "Try a short breathing and relaxation break.",
    },
    "Angry": {
        "aroma": "Lavender",
        "suggestion": "Try slow and controlled breathing.",
    },
    "Fear": {
        "aroma": "Lavender",
        "suggestion": "Take calm breaths and relax for a few moments.",
    },
    "Neutral": {
        "aroma": "Rosemary",
        "suggestion": "Take a short refreshing break.",
    },
    "Surprise": {
        "aroma": "Eucalyptus",
        "suggestion": "Take a few calm breaths.",
    },
    "Disgust": {
        "aroma": "Eucalyptus",
        "suggestion": "Take a short break and relax.",
    },
}


# Emotion-linked wellness ideas are optional lifestyle suggestions, not diagnoses.
WELLNESS_DETAILS = {
    "Happy": {
        "breathing": "Take 5 comfortable, natural breaths and notice how you feel. No need to breathe deeply or force a rhythm.",
        "meditation": "Spend one minute noticing something positive from your day, without needing to change how you feel.",
        "music": "Choose upbeat music you personally enjoy at a comfortable volume.",
        "quote": "Notice what is going well and give yourself permission to enjoy the moment.",
        "reflection": "What is one thing you would like to remember about this moment?",
    },
    "Sad": {
        "breathing": "Breathe gently in and out at a comfortable pace for one minute. Stop if you feel dizzy or uncomfortable.",
        "meditation": "Sit comfortably and name one small, kind thing you can do for yourself next.",
        "music": "Consider a familiar, comforting song that you choose for yourself.",
        "quote": "You do not have to solve everything at once. One small step is enough for now.",
        "reflection": "What would feel supportive or manageable for the next few minutes?",
    },
    "Angry": {
        "breathing": "Relax your shoulders. Breathe in gently, then breathe out slowly for a few cycles; do not hold your breath.",
        "meditation": "Pause before your next task and notice your feet on the floor and the sounds around you.",
        "music": "Try soft instrumental music or quiet sounds at a volume that feels comfortable.",
        "quote": "A pause gives you room to choose your next step.",
        "reflection": "Would a short break or some space help right now?",
    },
    "Fear": {
        "breathing": "If it feels comfortable, let your breathing settle naturally and make each exhale gentle. Stop if uncomfortable.",
        "meditation": "Look around and name three things you can see to reconnect with your surroundings.",
        "music": "Try a familiar, calming track if music feels helpful; silence is also okay.",
        "quote": "Take the next manageable step; you do not need to do everything at once.",
        "reflection": "What is one small thing that would help you feel more settled?",
    },
    "Neutral": {
        "breathing": "Take a brief breathing pause and let your breath stay natural and comfortable.",
        "meditation": "Spend one minute paying attention to your surroundings without judging them.",
        "music": "Choose a calm or focus-friendly track based on what you prefer today.",
        "quote": "Small pauses can make space for clearer next steps.",
        "reflection": "What would you like to focus on next?",
    },
    "Surprise": {
        "breathing": "Take a few comfortable breaths and give yourself a moment before moving on.",
        "meditation": "Notice what changed, then gently bring your attention back to the present.",
        "music": "Choose a familiar track that matches the mood you would like to encourage.",
        "quote": "You can take a moment to process something unexpected.",
        "reflection": "What stands out to you about this moment?",
    },
    "Disgust": {
        "breathing": "Take a short, comfortable pause. Breathe normally and stop any exercise that feels unpleasant.",
        "meditation": "Shift attention to a neutral detail around you, such as a colour, shape, or sound.",
        "music": "Try a familiar neutral track, or choose a quiet break instead.",
        "quote": "You can step away for a moment and decide what feels right next.",
        "reflection": "Would changing your environment or taking a brief break help?",
    },
}

DEFAULT_WELLNESS_DETAILS = {
    "breathing": "Take a few natural, comfortable breaths. Stop if you feel dizzy or uncomfortable.",
    "meditation": "Sit comfortably and notice your surroundings for one quiet minute.",
    "music": "Choose music you enjoy, or take a quiet break.",
    "quote": "Take things one step at a time. A short pause is okay.",
    "reflection": "Notice how you feel and choose a next step that feels right for you.",
}

DEFAULT_RECOMMENDATION_INTERVAL_SECONDS = 60 * 60  # default: one hour
DEFAULT_SPRAY_DURATION_SECONDS = 5
MAX_RECOMMENDATION_INTERVAL_SECONDS = 7 * 24 * 60 * 60
MAX_SPRAY_DURATION_SECONDS = 30
SAMPLE_COUNT = 10
# Lower LBPH distance threshold to reduce false acceptance of unknown faces.
# Recalibrate only using tests with enrolled and non-enrolled people.
SFACE_COSINE_THRESHOLD = float(os.environ.get("SFACE_COSINE_THRESHOLD", "0.363"))
# Use a stricter threshold for duplicate ENROLLMENT checks than live recognition.
# This reduces false duplicate rejections for different people.
SFACE_DUPLICATE_ENROLLMENT_THRESHOLD = float(
    os.environ.get("SFACE_DUPLICATE_ENROLLMENT_THRESHOLD", "0.50")
)
# A single LBPH guess must not block a new user; duplicate detection uses a stricter
# threshold and requires the same enrolled label to match most captured samples.
SFACE_AMBIGUITY_MARGIN = float(os.environ.get("SFACE_AMBIGUITY_MARGIN", "0.035"))
LIVE_MATCH_FRAMES_REQUIRED = 5
# Allow CPU-based YuNet/SFace + CNN inference a little more time between
# successful identity checks. A missing/unknown face still blocks immediately
# in continuous_camera_worker; this is only a processing-time grace window.
IDENTITY_FRESHNESS_SECONDS = max(
    1.5, float(os.environ.get("AROMA_IDENTITY_FRESHNESS_SECONDS", "3.0"))
)
ARDUINO_PORT = os.environ.get("AROMA_ARDUINO_PORT", "COM4")
ARDUINO_BAUDRATE = int(os.environ.get("AROMA_ARDUINO_BAUDRATE", "9600"))
# Hardware remains disabled by default until face gating is tested. Set this to 1
# in PowerShell only after confirming the Arduino firmware and relay wiring.
AROMA_HARDWARE_ENABLED = os.environ.get("AROMA_HARDWARE_ENABLED", "0") == "1"

# Commands must match the existing Arduino sketch:
# H = Citrus relay, S = Lavender relay, N = Rosemary relay, X = Eucalyptus relay.
AROMA_COMMANDS = {
    "Citrus": b"H",
    "Lavender": b"S",
    "Rosemary": b"N",
    "Eucalyptus": b"X",
}
arduino = None
if AROMA_HARDWARE_ENABLED:
    if serial is None:
        print("ARDUINO DISABLED: install pyserial in this .venv first.")
    else:
        try:
            arduino = serial.Serial(
                ARDUINO_PORT,
                ARDUINO_BAUDRATE,
                timeout=1,
                write_timeout=1,
            )
            time.sleep(2)  # Arduino UNO typically resets after serial connection.
            arduino.reset_input_buffer()
            print(f"Arduino connected on {ARDUINO_PORT} at {ARDUINO_BAUDRATE} baud.")
        except Exception as exc:
            arduino = None
            print(f"ARDUINO UNAVAILABLE on {ARDUINO_PORT}: {exc}")
else:
    print("Arduino hardware is disabled (safe test mode).")

# Runtime state is kept per authenticated user. This is suitable for the
# local demo; restart clears the in-memory live state, but not SQLite records.
user_states = {}


def get_user_state(user_id):
    user_id = int(user_id)
    if user_id not in user_states:
        user_states[user_id] = {
            "current_emotion": "Waiting...",
            "current_aroma": "Waiting...",
            "current_suggestion": "Waiting for the first recommendation.",
            "prediction_history": deque(maxlen=SAMPLE_COUNT),
            "next_recommendation_time": time.time() + DEFAULT_RECOMMENDATION_INTERVAL_SECONDS,
            "timer_interval_seconds": DEFAULT_RECOMMENDATION_INTERVAL_SECONDS,
            "timer_enabled": True,
            "spray_duration_seconds": DEFAULT_SPRAY_DURATION_SECONDS,
            "fallback_enabled": False,
            "fallback_unknown_enabled": False,
            "fallback_multiple_enabled": False,
            "fallback_camera_error_enabled": False,
            "fallback_aroma": "Citrus",
            "recording_timer_enabled": False,
            "recording_interval_seconds": DEFAULT_RECORDING_INTERVAL_SECONDS,
            "recording_duration_seconds": DEFAULT_RECORDING_DURATION_SECONDS,
            "recording_status": "Recording is off.",
            "last_saved_clip": None,
            "last_saved_clip_at": None,
            "next_recording_time": time.time() + DEFAULT_RECORDING_INTERVAL_SECONDS,
            "timer_settings_loaded": False,
            "session_active": False,
            "ever_live_verified": False,
            "identity_verified": False,
            "identity_checked": False,
            "identity_match_streak": 0,
            "last_identity_check": 0.0,
            "identity_message": "Waiting for live face verification.",
            "spray_status": "Hardware disabled (safe test mode)." if not AROMA_HARDWARE_ENABLED else "Waiting for verified emotion.",
            "monitoring_enabled": False,
            "last_notification_emotion": None,
        }
    return user_states[user_id]




camera_lock = threading.Lock()  # protects the single always-on camera worker
camera_stop_event = threading.Event()
camera_condition = threading.Condition()
# The camera worker owns the active video writer. Settings changes request rotation
# through this set rather than closing an FFmpeg pipe from the Flask request thread.
recording_rotation_requests = set()
recording_rotation_lock = threading.Lock()
camera_worker_thread = None
camera_sequence = 0
latest_raw_frame = None  # in-memory preview; local video recording is independently consent-gated
latest_camera_jpeg = None
latest_camera_status = "Camera service starting"
active_camera_user_id = None
last_monitoring_user_id = None  # most recent consenting registered account; used only for opt-in no-face fallback


# =========================================================
# DATABASE AND MIGRATIONS
# =========================================================


def get_db_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_database():
    connection = sqlite3.connect(DB_PATH)
    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                face_enrolled INTEGER NOT NULL DEFAULT 0,
                phone TEXT NOT NULL DEFAULT '',
                department TEXT NOT NULL DEFAULT '',
                study_year TEXT NOT NULL DEFAULT '',
                location TEXT NOT NULL DEFAULT ''
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS emotion_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                emotion TEXT NOT NULL,
                aroma TEXT NOT NULL,
                suggestion TEXT NOT NULL,
                detected_at TEXT NOT NULL,
                user_id INTEGER
            )
            """
        )

        # Migrate databases created by the earlier version without dropping data.
        user_columns = {
            row[1] for row in cursor.execute("PRAGMA table_info(users)").fetchall()
        }
        if "face_enrolled" not in user_columns:
            cursor.execute(
                "ALTER TABLE users ADD COLUMN face_enrolled INTEGER NOT NULL DEFAULT 0"
            )
        if "monitoring_enabled" not in user_columns:
            cursor.execute(
                "ALTER TABLE users ADD COLUMN monitoring_enabled INTEGER NOT NULL DEFAULT 0"
            )
        for column in ("phone", "department", "study_year", "location"):
            if column not in user_columns:
                cursor.execute(f"ALTER TABLE users ADD COLUMN {column} TEXT NOT NULL DEFAULT ''")

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS profile_reflections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                focus TEXT NOT NULL DEFAULT '',
                thought TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_profile_reflections_user_time "
            "ON profile_reflections(user_id, created_at)"
        )

        history_columns = {
            row[1]
            for row in cursor.execute("PRAGMA table_info(emotion_history)").fetchall()
        }
        if "user_id" not in history_columns:
            cursor.execute("ALTER TABLE emotion_history ADD COLUMN user_id INTEGER")

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS face_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                face_image BLOB NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                pose_tag TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS video_recordings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                file_name TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT NOT NULL,
                duration_seconds REAL NOT NULL DEFAULT 0,
                frame_count INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                notification_type TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL,
                is_read INTEGER NOT NULL DEFAULT 0,
                dedupe_key TEXT UNIQUE,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS timer_settings (
                user_id INTEGER PRIMARY KEY,
                interval_seconds INTEGER NOT NULL DEFAULT 3600,
                timer_enabled INTEGER NOT NULL DEFAULT 1,
                spray_duration_seconds INTEGER NOT NULL DEFAULT 5,
                fallback_enabled INTEGER NOT NULL DEFAULT 0,
                fallback_unknown_enabled INTEGER NOT NULL DEFAULT 0,
                fallback_multiple_enabled INTEGER NOT NULL DEFAULT 0,
                fallback_camera_error_enabled INTEGER NOT NULL DEFAULT 0,
                fallback_aroma TEXT NOT NULL DEFAULT 'Citrus',
                recording_timer_enabled INTEGER NOT NULL DEFAULT 0,
                recording_interval_seconds INTEGER NOT NULL DEFAULT 1800,
                recording_duration_seconds INTEGER NOT NULL DEFAULT 1800,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )

        timer_columns = {row[1] for row in cursor.execute("PRAGMA table_info(timer_settings)").fetchall()}
        # Safe, additive migrations: preserve existing user settings and recordings.
        for column, declaration in (
            ("fallback_unknown_enabled", "INTEGER NOT NULL DEFAULT 0"),
            ("fallback_multiple_enabled", "INTEGER NOT NULL DEFAULT 0"),
            ("fallback_camera_error_enabled", "INTEGER NOT NULL DEFAULT 0"),
        ):
            if column not in timer_columns:
                cursor.execute(f"ALTER TABLE timer_settings ADD COLUMN {column} {declaration}")
        if "recording_timer_enabled" not in timer_columns:
            cursor.execute("ALTER TABLE timer_settings ADD COLUMN recording_timer_enabled INTEGER NOT NULL DEFAULT 0")
        if "recording_interval_seconds" not in timer_columns:
            cursor.execute("ALTER TABLE timer_settings ADD COLUMN recording_interval_seconds INTEGER NOT NULL DEFAULT 1800")
        if "recording_duration_seconds" not in timer_columns:
            cursor.execute("ALTER TABLE timer_settings ADD COLUMN recording_duration_seconds INTEGER NOT NULL DEFAULT 1800")

        # Upgrade the previous built-in 60-second default to 30-minute clip rotation
        # only for disabled, untouched defaults. Preserve active/custom settings.
        cursor.execute(
            "UPDATE timer_settings SET recording_duration_seconds = 1800 "
            "WHERE recording_timer_enabled = 0 AND recording_interval_seconds = 1800 "
            "AND recording_duration_seconds = 60"
        )

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_user_time ON emotion_history(user_id, detected_at)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_recordings_user_time ON video_recordings(user_id, started_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_notifications_user_time ON notifications(user_id, created_at)")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS browser_push_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_browser_push_tokens_user ON browser_push_tokens(user_id)"
        )
        connection.commit()
    finally:
        connection.close()


init_database()


def get_timer_settings(user_id):
    """Load per-user emotion timer, spray duration, fallback and scheduled recording settings."""
    defaults = {
        "interval_seconds": DEFAULT_RECOMMENDATION_INTERVAL_SECONDS,
        "timer_enabled": True,
        "spray_duration_seconds": DEFAULT_SPRAY_DURATION_SECONDS,
        "fallback_enabled": False,
        "fallback_unknown_enabled": False,
        "fallback_multiple_enabled": False,
        "fallback_camera_error_enabled": False,
        "fallback_aroma": "Citrus",
        "recording_timer_enabled": False,
        "recording_interval_seconds": DEFAULT_RECORDING_INTERVAL_SECONDS,
        "recording_duration_seconds": DEFAULT_RECORDING_DURATION_SECONDS,
    }
    connection = get_db_connection()
    try:
        row = connection.execute(
            "SELECT interval_seconds, timer_enabled, spray_duration_seconds, fallback_enabled, "
            "fallback_unknown_enabled, fallback_multiple_enabled, fallback_camera_error_enabled, fallback_aroma, "
            "recording_timer_enabled, recording_interval_seconds, recording_duration_seconds "
            "FROM timer_settings WHERE user_id = ?",
            (int(user_id),),
        ).fetchone()
        if row is None:
            connection.execute(
                "INSERT OR IGNORE INTO timer_settings "
                "(user_id, interval_seconds, timer_enabled, spray_duration_seconds, fallback_enabled, "
                "fallback_unknown_enabled, fallback_multiple_enabled, fallback_camera_error_enabled, fallback_aroma, "
                "recording_timer_enabled, recording_interval_seconds, recording_duration_seconds, updated_at) "
                "VALUES (?, ?, 1, ?, 0, 0, 0, 0, 'Citrus', 0, ?, ?, ?)",
                (int(user_id), DEFAULT_RECOMMENDATION_INTERVAL_SECONDS, DEFAULT_SPRAY_DURATION_SECONDS,
                 DEFAULT_RECORDING_INTERVAL_SECONDS, DEFAULT_RECORDING_DURATION_SECONDS,
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
            connection.commit()
            return defaults
        return {
            "interval_seconds": int(row["interval_seconds"]),
            "timer_enabled": bool(row["timer_enabled"]),
            "spray_duration_seconds": int(row["spray_duration_seconds"]),
            "fallback_enabled": bool(row["fallback_enabled"]),
            "fallback_unknown_enabled": bool(row["fallback_unknown_enabled"]),
            "fallback_multiple_enabled": bool(row["fallback_multiple_enabled"]),
            "fallback_camera_error_enabled": bool(row["fallback_camera_error_enabled"]),
            "fallback_aroma": row["fallback_aroma"] if row["fallback_aroma"] in ("Citrus", "Rosemary") else "Citrus",
            "recording_timer_enabled": bool(row["recording_timer_enabled"]),
            "recording_interval_seconds": int(row["recording_interval_seconds"]),
            "recording_duration_seconds": int(row["recording_duration_seconds"]),
        }
    finally:
        connection.close()


def apply_timer_settings_to_state(user_id, settings=None, reset_timer=False):
    state = get_user_state(user_id)
    was_loaded = bool(state.get("timer_settings_loaded"))
    settings = settings or get_timer_settings(user_id)
    state["timer_interval_seconds"] = int(settings["interval_seconds"])
    state["timer_enabled"] = bool(settings["timer_enabled"])
    state["spray_duration_seconds"] = int(settings["spray_duration_seconds"])
    state["fallback_enabled"] = bool(settings["fallback_enabled"])
    state["fallback_unknown_enabled"] = bool(settings.get("fallback_unknown_enabled", False))
    state["fallback_multiple_enabled"] = bool(settings.get("fallback_multiple_enabled", False))
    state["fallback_camera_error_enabled"] = bool(settings.get("fallback_camera_error_enabled", False))
    state["fallback_aroma"] = settings["fallback_aroma"]
    state["recording_timer_enabled"] = bool(settings["recording_timer_enabled"])
    # Legacy interval value is retained for migration compatibility only. Recording is continuous;
    # recording_duration_seconds controls the length of each file before it rotates.
    state["recording_interval_seconds"] = int(settings["recording_duration_seconds"])
    state["recording_duration_seconds"] = int(settings["recording_duration_seconds"])
    state["timer_settings_loaded"] = True
    if reset_timer or not was_loaded:
        now = time.time()
        state["prediction_history"].clear()
        state["next_recommendation_time"] = now + state["timer_interval_seconds"] if state["timer_enabled"] else float("inf")
        state["next_recording_time"] = now if state["recording_timer_enabled"] else float("inf")
    elif not state["recording_timer_enabled"]:
        state["next_recording_time"] = float("inf")
    return state


# =========================================================
# PERSONAL NOTIFICATIONS AND DAILY REPORTS
# =========================================================



FUN_NOTIFICATIONS = {
    "Happy": [
        ("Thalaiva, treat enga? 😂",
         "{name}, inniku semma happy-ah irukka! Enna vishayam boss, lottery adichadha? 🍊"),
        ("Happy mode ON-u! 😎",
         "{name}, sirippu paatha periya good news pola! Indha vibe appadiye maintain pannu! ✨"),
        ("Enna oru sirippu! 😂",
         "{name}, face-la 1000-watt bulb eriyudhu boss! Happy-ah iru! 💡"),
        ("Inniku un day dhaan! 🥳",
         "{name}, inniku luck un pakkam irukku pola! Poi kalakku thalaiva! 🔥"),
        ("Sirippu factory open-ah? 😁",
         "{name}, ivlo sirikka enna secret? Engalukkum sollu boss! 🤭"),
        ("Vera level vibe! 🎉",
         "{name}, un happiness-ku oru celebration pannidalaama? Semma energy! 💃"),
        ("Yaaru andha lucky person? 👀",
         "{name}, indha smile-ku pinnadi edho story irukku pola! 😂"),
        ("Positive vibes loading! ⚡",
         "{name}, happiness full charge-la irukku! Indha battery drain aaga vidaadha! 🔋"),
        ("Smile-ku attendance podu! 😁",
         "{name}, indha smile daily attendance podanum boss! ✨"),
        ("Happy-ah iru, boss! 🍊",
         "{name}, inniku un mood super-ah irukku pola! Oru favourite song potu enjoy pannu! 🎶"),
    ],

    "Sad": [
        ("Enna pa, mood off-ah? 🥹",
         "{name}, life konjam mokka-va pogudha? Parava illa boss, oru break eduthu relax pannuvom! 💜"),
        ("Sad-ku leave kuduppoma? 😂",
         "{name}, ellathayum ore naal-la solve pannanum-nu avasiyam illa boss! 🫶"),
        ("Yaaru da un mood-a damage pannadhu? 😭",
         "{name}, konjam self-care panniko boss! Unakku nee dhaan first priority! 💜"),
        ("Heart-ku konjam recharge! 🔋",
         "{name}, phone-ku charge podra maadhiri unakkum konjam rest kudu! 🥹"),
        ("Life-la interval card! 🎬",
         "{name}, padam innum mudiyala boss! Ippo oru break eduthuttu next scene-ku povom! 🌱"),
        ("Sad face-ku tea break! ☕",
         "{name}, oru tea illa favourite snack saaptu konjam relax pannalaama? 😌"),
        ("Overthinking department close! 😂",
         "{name}, brain-ku overtime kudukkaadha boss! Konjam rest eduthuko! 🧠"),
        ("Ellam seri aagum, paathukalaam! 🫂",
         "{name}, inniku kashtama irundhaalum parava illa. Oru step at a time boss! 💜"),
        ("Mood-ku maintenance work! 🛠️",
         "{name}, konjam quiet time eduthuko. Un feelings-ku space kudu boss! 🌿"),
        ("Smile-ku force panna vendam! 🫶",
         "{name}, happy-ah nadikka vendam boss. Unakku thevaiyana neram eduthuko! 💜"),
    ],

    "Angry": [
        ("Yaaru da kadupaethunadhu? 😤",
         "{name}, face-la full current-ah irukku boss! Konjam cool aagu! ⚡"),
        ("Kovam 100%, patience 0% ah? 😂",
         "{name}, reply panradhukku munnadi oru deep breath vidu boss! 😮‍💨"),
        ("Angry bird detected! 🐦",
         "{name}, parakka ready-ah irukka pola! First konjam relax pannalaam boss! 😂"),
        ("Blood pressure-ku break kudu! 😅",
         "{name}, indha situation-ku nee mattum tension aagadha boss! Konjam pause pannu! 🌿"),
        ("Yaaru un peace-a disturb pannadhu? 🤨",
         "{name}, peace-ku disturbance vandhirukku pola! Konjam time eduthu yosichu respond pannu! 🧘"),
        ("Cool down, mass hero! 🕶️",
         "{name}, interval fight scene vendam boss! First calm mode-ku switch aagu! 😂"),
        ("Keyboard-a paavam vittudu! ⌨️",
         "{name}, kovathula type panna message-a once reread pannitu send pannu boss! 😅"),
        ("Steam engine mode ON! 🚂",
         "{name}, whistle vidradhukku munnadi konjam breathe pannalaam! 😮‍💨"),
        ("Patience enga boss? 🔍",
         "{name}, patience missing complaint register pannalama? 😂 Konjam pause eduthu paaru!"),
        ("Kovathukkum timeout irukku! ⏳",
         "{name}, konjam water kudichittu situation-a calm-ah handle pannalaam boss! 💧"),
    ],

    "Fear": [
        ("Bayam oru pakkam, namma oru pakkam! 😅",
         "{name}, tension aagadha boss. Ovoru step-ah eduthu veppom! 🌱"),
        ("Brain-la 100 tabs open-ah? 😂",
         "{name}, onnu onna paathukalaam. Ellathayum ore nerathula yosikka venam! 🐢"),
        ("Anxiety express vandhucha? 🚂",
         "{name}, konjam slow down pannu boss. Ippo oru step mattum focus pannalaam! 🌿"),
        ("Courage loading... ⏳",
         "{name}, superhero aaganum-nu avasiyam illa! Chinna step-um periya vishayam dhaan! 💪"),
        ("Bayam vandha tea podu! ☕",
         "{name}, konjam relax pannu boss. Unakku comfortable-ah irukkura pace-la po! 😌"),
        ("Mind-ku emergency meeting-ah? 🧠",
         "{name}, ellathayum ippo decide panna vendam. Konjam pause eduthuko! 🌱"),
        ("Tension-ku attendance podaadha! 😂",
         "{name}, worry panradha vida next small step enna-nu paaru boss! 🐾"),
        ("Namma slow-ah pogalaam! 🐢",
         "{name}, speed mukkiyam illa boss. Safe-ah, comfortable-ah move pannradhu mukkiyam! 🌿"),
        ("Fear-ku konjam space kudu! 🌱",
         "{name}, bayama irundhaalum parava illa. Ready aana appuram next step edukkalaam! 💜"),
        ("Deep breath challenge! 😮‍💨",
         "{name}, comfortable-ah irundha oru slow breath eduthu, un attention-a ippo irukkura vishayathukku kondu vaa! 🌿"),
    ],

    "Neutral": [
        ("Face loading-la irukka? 😂",
         "{name}, happy-um illa sad-um illa! Battery saver mode-la irukkiya boss? 🔋"),
        ("Expression enga boss? 😐",
         "{name}, face-la full silence! Oru tea break potta nalla irukkum pola! ☕"),
        ("Silent mode activated! 🤫",
         "{name}, inniku face kooda silent mode-la irukku pola! 😂"),
        ("Background processing... 🖥️",
         "{name}, mind-la enna processing nadakkudhu boss? Face mattum poker face! 😎"),
        ("NPC mode detected! 🎮",
         "{name}, game-la idle character maadhiri calm-ah irukka boss! 😂"),
        ("Emotion loading... 1% 😴",
         "{name}, expression update innum varala pola! No rush boss! 😄"),
        ("Chill mode maximum! 🧊",
         "{name}, un calmness-ku oru award kudukkalaam pola! 🏆"),
        ("Enna boss, Monday face-ah? 😶",
         "{name}, face romba serious-ah irukku! Coffee break time pola! ☕"),
        ("No drama, only peace! 🕊️",
         "{name}, inniku silent-ah irukkuradhum oru vibe dhaan boss! 🌿"),
        ("Idle illa, energy save! 🔋",
         "{name}, konjam quiet-ah irundhaalum parava illa boss. Un pace-la po! 😌"),
    ],

    "Surprise": [
        ("Adei, enna plot twist-u? 😲",
         "{name}, face-a paatha edho unexpected news vandha maadhiri irukku! Enna boss nadandhuchu? 😂"),
        ("Shock aayitiya thalaiva? 🤯",
         "{name}, konjam wait pannu… brain loading complete aagattum! 😜"),
        ("Breaking news detected! 📣",
         "{name}, un expression paatha breaking news pola irukku boss! 😂"),
        ("Enna da idhu pudhusa? 👀",
         "{name}, unexpected update vandhirukku pola! Oru second process panniko! 😲"),
        ("Plot twist incoming! 🎬",
         "{name}, life screenplay-la pudhu twist pola! Next scene enna boss? 😂"),
        ("Eyes wide open mode! 👀",
         "{name}, surprise-ku reaction semma level-la irukku pola! 😆"),
        ("Brain.exe stopped working! 💻",
         "{name}, unexpected moment-la brain konjam buffering aaguradhu normal dhaan! 😂"),
        ("Idha expect pannalaya? 😳",
         "{name}, surprise package delivery aayiduchu pola boss! 🎁"),
        ("Shock-ku oru timeout! ⏳",
         "{name}, surprise-a digest pannittu next move decide pannalaam boss! 😄"),
        ("Unexpected guest: Surprise! 🎉",
         "{name}, expression semma plot twist kudukkudhu! Camera guess dhaan boss, actual reason unakkuthaan theriyum! 😜"),
    ],

    "Disgust": [
        ("Idhu enna kodumai boss? 😂",
         "{name}, edho pidikkala pola! Konjam break eduthu fresh-ah start pannuvom! 🌿"),
        ("Skip pannidalaama? 🙃",
         "{name}, vibe set aagala na parava illa boss. Konjam relax pannitu continue pannalaam! 🫧"),
        ("Yov, idhu vendaam! 😂",
         "{name}, face paatha 'enakku idhu set aagala' nu sollura maadhiri irukku! 😆"),
        ("Review: 0 stars! ⭐",
         "{name}, indha experience-ku review podra expression pola boss! 😂"),
        ("Unsubscribe pannidalaama? 🚫",
         "{name}, indha vibe pudikkala na konjam distance eduthukkalaam boss! 🌿"),
        ("Nope mode activated! 🙅",
         "{name}, 'not interested' button press pannita maadhiri irukku! 😂"),
        ("Face says: Next please! ➡️",
         "{name}, indha scene-a skip pannalaam nu face-e solludhu pola! 😅"),
        ("Taste test fail-ah? 🤢",
         "{name}, edho romba pidikkala pola boss! Konjam fresh air eduthuko! 🌱"),
        ("Dislike button enga? 👎",
         "{name}, indha moment-ku dislike podanum pola! 😂 Konjam break eduthu paaru!"),
        ("Mood-ku refresh button! 🔄",
         "{name}, vibe set aagala na parava illa boss. Unakku comfortable-ah irukkura vishayathukku switch pannalaam! 🌿"),
    ],
}


def get_user_first_name(user_id):
    connection = get_db_connection()
    try:
        row = connection.execute("SELECT full_name FROM users WHERE id = ?", (int(user_id),)).fetchone()
        return (row["full_name"].strip().split() or ["friend"])[0] if row else "friend"
    finally:
        connection.close()

def create_notification(user_id, notification_type, title, message, dedupe_key=None):
    connection = get_db_connection()
    try:
        connection.execute(
            "INSERT OR IGNORE INTO notifications (user_id, notification_type, title, message, created_at, is_read, dedupe_key) VALUES (?, ?, ?, ?, ?, 0, ?)",
            (int(user_id), notification_type, title, message, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), dedupe_key),
        )
        connection.commit()
    finally:
        connection.close()


camera_event_lock = threading.Lock()
camera_event_status = None
camera_event_number = 0


def send_browser_push(user_id, title, body):
    """Send an FCM web push to all browser tokens registered to this user."""
    if not firebase_admin._apps:
        app.logger.warning("Push skipped: Firebase Admin is not initialized.")
        return 0
    connection = get_db_connection()
    try:
        tokens = connection.execute(
            "SELECT token FROM browser_push_tokens WHERE user_id = ?", (int(user_id),)
        ).fetchall()
    finally:
        connection.close()

    sent = 0
    for row in tokens:
        try:
            msg = messaging.Message(
                notification=messaging.Notification(title=title, body=body),
                token=row["token"],
                webpush=messaging.WebpushConfig(
                    notification=messaging.WebpushNotification(title=title, body=body)
                ),
            )
            messaging.send(msg)
            sent += 1
        except Exception:
            app.logger.exception("FCM push failed for user %s", user_id)
    return sent


def handle_camera_status_notification(status):
    """Immediately notify once when a no-face/unknown/multiple-face event starts."""
    global camera_event_status, camera_event_number
    event_statuses = {
        "No Face Detected",
        "Other Face Detected",
        "Multiple Faces Detected",
    }
    normalized = status if status in event_statuses else None
    with camera_event_lock:
        if normalized == camera_event_status:
            return
        camera_event_status = normalized
        if normalized is None:
            return
        camera_event_number += 1
        event_number = camera_event_number

    recipient = last_monitoring_user_id
    if recipient is None or not get_monitoring_enabled(recipient):
        return

    titles = {
        "No Face Detected": ("No Face Detected", "No face is visible to the camera."),
        "Other Face Detected": ("Unknown Person Detected", "An unregistered face is visible. Emotion processing is blocked."),
        "Multiple Faces Detected": ("Multiple Faces Detected", "More than one face is visible. Emotion processing is blocked unless your registered face is identified."),
    }
    title, body = titles[normalized]
    key = f"camera-event:{recipient}:{event_number}:{normalized.lower().replace(' ', '-')}"
    create_notification(recipient, "camera_status", title, body, key)
    send_browser_push(recipient, title, body)


def maybe_trigger_nonregistered_spray(user_id, status):
    """Optional fallback spray for unknown/multiple faces; never runs emotion inference."""
    if user_id is None or status not in ("Other Face Detected", "Multiple Faces Detected"):
        return False
    state = get_user_state(user_id)
    if not state.get("timer_settings_loaded"):
        apply_timer_settings_to_state(user_id)
    if not get_monitoring_enabled(user_id):
        return False
    setting_key = "fallback_unknown_enabled" if status == "Other Face Detected" else "fallback_multiple_enabled"
    if not state.get(setting_key):
        state["spray_status"] = f"Spray blocked: enable the {status} fallback in timer settings."
        return False
    aroma = state.get("fallback_aroma", "Citrus")
    command = AROMA_COMMANDS.get(aroma)
    duration = max(1, min(MAX_SPRAY_DURATION_SECONDS, int(state.get("spray_duration_seconds", DEFAULT_SPRAY_DURATION_SECONDS))))
    state["current_emotion"] = status
    state["current_aroma"] = aroma
    state["current_suggestion"] = "Emotion processing is blocked for this camera condition."
    if not AROMA_HARDWARE_ENABLED:
        state["spray_status"] = f"Dry-run only: {status} fallback would spray {aroma} for {duration}s."
        return False
    if arduino is None or command is None:
        state["spray_status"] = "Spray unavailable: Arduino or aroma mapping is unavailable."
        return False
    try:
        payload = f"{command.decode('ascii')},{duration}\\n".encode("ascii")
        arduino.write(payload)
        arduino.flush()
        state["spray_status"] = f"{status} fallback spray command sent: {aroma}, {duration}s."
        return True
    except Exception:
        app.logger.exception("Non-registered fallback spray failed")
        state["spray_status"] = "Fallback spray failed."
        return False

def maybe_create_emotion_notification(user_id, emotion):
    """Create and push one wellness notification for each confirmed timer cycle."""
    options = FUN_NOTIFICATIONS.get(
        emotion,
        [("A little wellness nudge 🌿", "{name}, take a moment for yourself if that feels useful.")],
    )
    title, message = random.choice(options)
    name = get_user_first_name(user_id)
    message = message.format(name=name)
    event_key = f"emotion:{int(user_id)}:{int(time.time() * 1000)}"
    create_notification(user_id, "emotion", title, message, event_key)
    send_browser_push(user_id, title, message)


def maybe_create_daily_reports():
    """Create one local-time daily report after DAILY_REPORT_HOUR for each opted-in account."""
    now = datetime.now()
    if now.hour < DAILY_REPORT_HOUR:
        return
    report_date = now.strftime("%Y-%m-%d")
    connection = get_db_connection()
    try:
        users = connection.execute(
            "SELECT id, full_name FROM users WHERE face_enrolled = 1 AND monitoring_enabled = 1"
        ).fetchall()
    finally:
        connection.close()
    for user in users:
        user_id = int(user["id"])
        dedupe_key = f"daily-report:{user_id}:{report_date}"
        connection = get_db_connection()
        try:
            already_sent = connection.execute(
                "SELECT 1 FROM notifications WHERE dedupe_key = ?", (dedupe_key,)
            ).fetchone()
            if already_sent:
                continue
            report_day = datetime.strptime(report_date, "%Y-%m-%d").date()
            next_day = (report_day + timedelta(days=1)).strftime("%Y-%m-%d")
            rows = connection.execute(
                "SELECT emotion, aroma, COUNT(*) AS amount FROM emotion_history WHERE user_id = ? AND detected_at >= ? AND detected_at < ? GROUP BY emotion, aroma",
                (user_id, report_date + " 00:00:00", next_day + " 00:00:00"),
            ).fetchall()
        finally:
            connection.close()
        total = sum(int(row["amount"]) for row in rows)
        emotion_totals = Counter()
        aroma_totals = Counter()
        for row in rows:
            emotion_totals[row["emotion"]] += int(row["amount"])
            aroma_totals[row["aroma"]] += int(row["amount"])
        first_name = (user["full_name"].strip().split() or ["friend"])[0]
        if total:
            top_emotion, top_amount = emotion_totals.most_common(1)[0]
            top_aroma = aroma_totals.most_common(1)[0][0]
            title = f"Your daily vibe report, {first_name} 📊"
            message = (
                f"Today's camera logged {total} verified emotion check-in(s). "
                f"Most frequent model label: {top_emotion} ({top_amount}); most recommended aroma: {top_aroma}. "
                "These are camera-model estimates, not a diagnosis. Tomorrow is a fresh page 🌱"
            )
        else:
            title = f"Your daily check-in, {first_name} 🌙"
            message = (
                "No verified emotion records were saved today. The camera only records while an enrolled face is recognized and monitoring is enabled. "
                "No need to chase a score—rest counts too 💜"
            )
        create_notification(user_id, "daily_report", title, message, dedupe_key)


# =========================================================
# FACE TEMPLATE HELPERS
# =========================================================


def decode_face_image(data_url):
    """Decode a webcam data URL to an OpenCV colour image."""
    try:
        if not isinstance(data_url, str) or "," not in data_url:
            return None, "Invalid image data."
        encoded = data_url.split(",", 1)[1]
        raw = base64.b64decode(encoded, validate=True)
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return None, "Could not read the captured image."
        return frame, None
    except (ValueError, TypeError):
        return None, "Invalid image encoding."


def _sface_extract(frame):
    """Return (unit-normalized SFace embedding, bbox, error). Requires one visible face."""
    if frame is None or frame.size == 0:
        return None, None, "Camera frame is empty."
    height, width = frame.shape[:2]
    try:
        with face_model_lock:
            face_detector_sface.setInputSize((int(width), int(height)))
            _status, faces = face_detector_sface.detect(frame)
            if faces is None or len(faces) == 0:
                return None, None, "No face detected. Keep your face visible to the camera."
            if len(faces) != 1:
                return None, None, "More than one face detected. Keep only the enrolling user in frame."
            detection = faces[0]
            aligned = face_embedding_model.alignCrop(frame, detection)
            raw_feature = face_embedding_model.feature(aligned)
    except Exception:
        app.logger.exception("YuNet/SFace extraction failed")
        return None, None, "Face extraction failed. Improve lighting and try again."

    feature = np.asarray(raw_feature, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(feature))
    if norm <= 1e-12:
        return None, None, "Face template was empty; please capture again."
    feature /= norm
    x, y, box_w, box_h = [int(round(v)) for v in detection[:4]]
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    box_w = max(1, min(box_w, width - x))
    box_h = max(1, min(box_h, height - y))
    return feature, (x, y, box_w, box_h), None


def _stored_embeddings(exclude_user_id=None, enrolled_only=True):
    """Read compact embeddings; no source camera images are needed for recognition."""
    connection = get_db_connection()
    try:
        query = (
            "SELECT e.user_id, e.embedding FROM face_embeddings e "
            "JOIN users u ON u.id = e.user_id WHERE 1=1"
        )
        params = []
        if enrolled_only:
            query += " AND u.face_enrolled = 1"
        if exclude_user_id is not None:
            query += " AND e.user_id != ?"
            params.append(int(exclude_user_id))
        rows = connection.execute(query, params).fetchall()
    finally:
        connection.close()

    result = {}
    for row in rows:
        emb = np.frombuffer(row["embedding"], dtype=np.float32)
        if emb.size == 0:
            continue
        norm = float(np.linalg.norm(emb))
        if norm <= 1e-12:
            continue
        emb = emb / norm
        result.setdefault(int(row["user_id"]), []).append(emb)
    return result


def _user_similarity(feature, user_embeddings):
    """Best cosine similarity against a user's enrolled, multi-angle samples."""
    if not user_embeddings:
        return -1.0
    return max(float(np.dot(feature, sample)) for sample in user_embeddings)


def _best_enrolled_identity(feature, exclude_user_id=None):
    embeddings = _stored_embeddings(exclude_user_id=exclude_user_id, enrolled_only=True)
    scored = [
        (user_id, _user_similarity(feature, samples))
        for user_id, samples in embeddings.items()
    ]
    if not scored:
        return None, -1.0, "No face profiles are currently enrolled."
    scored.sort(key=lambda item: item[1], reverse=True)
    best_user, best_score = scored[0]
    if best_score < SFACE_COSINE_THRESHOLD:
        return None, best_score, "Face not matched to an enrolled user."
    if len(scored) > 1:
        second_user, second_score = scored[1]
        if second_score >= SFACE_COSINE_THRESHOLD and best_score - second_score < SFACE_AMBIGUITY_MARGIN:
            return None, best_score, (
                "Face match is ambiguous between enrolled accounts. Use one account per person "
                "and re-enroll with several angles."
            )
    return int(best_user), float(best_score), "SFace identity match accepted."


def user_face_enrolled(user_id):
    connection = get_db_connection()
    try:
        row = connection.execute(
            "SELECT face_enrolled FROM users WHERE id = ?", (int(user_id),)
        ).fetchone()
        has_embedding = connection.execute(
            "SELECT 1 FROM face_embeddings WHERE user_id = ? LIMIT 1", (int(user_id),)
        ).fetchone()
        return bool(row and row["face_enrolled"] and has_embedding)
    finally:
        connection.close()


def migrate_legacy_lbph_profiles():
    """Require a fresh SFace multi-angle enrollment for users who only have legacy LBPH samples."""
    connection = get_db_connection()
    try:
        legacy_users = connection.execute(
            "SELECT id FROM users WHERE face_enrolled = 1"
        ).fetchall()
        reset_ids = []
        for row in legacy_users:
            has_sface = connection.execute(
                "SELECT 1 FROM face_embeddings WHERE user_id = ? LIMIT 1",
                (int(row["id"]),),
            ).fetchone()
            if not has_sface:
                reset_ids.append(int(row["id"]))
        if reset_ids:
            connection.executemany(
                "UPDATE users SET face_enrolled = 0, monitoring_enabled = 0 WHERE id = ?",
                [(uid,) for uid in reset_ids],
            )
            connection.commit()
            print(
                "SFace upgrade: previous LBPH profiles for user IDs "
                f"{reset_ids} must be re-enrolled with the new multi-angle face page."
            )
    finally:
        connection.close()


migrate_legacy_lbph_profiles()

# =========================================================
# AUTHENTICATION AND ACCESS PROTECTION
# =========================================================


def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            if request.path.startswith("/api/") or request.path == "/video_feed":
                return jsonify({"error": "Login required"}), 401
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapper


def face_verified_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            if request.path.startswith("/api/") or request.path == "/video_feed":
                return jsonify({"error": "Login required"}), 401
            return redirect(url_for("login"))

        if session.get("face_verified") is True:
            # Settings and a user's own saved video library remain accessible after the live
            # face moves out of frame. These endpoints never grant emotion or spray access.
            # Emotion/history/analytics APIs still require a current live identity match.
            path = request.path
            identity_exempt = (
                # These endpoints return the logged-in user's safe dashboard status and
                # own notifications even when their face temporarily leaves the frame.
                # /api/current is populated by block_user_state_for_identity_loss() with
                # the face status and aroma OFF, so it does not grant emotion processing.
                path == "/api/current"
                or path == "/api/notifications"
                or path == "/api/timer-settings"
                or path == "/api/recordings"
                or path.startswith("/api/recordings/")
                or path.startswith("/recordings/")
            )
            if path.startswith("/api/") and not identity_exempt:
                state = user_states.get(int(session["user_id"]))
                if state and not state.get("session_active"):
                    return jsonify({"error": "Session is not active. Log in and verify again."}), 403
                if state and state.get("identity_checked") and not state.get("identity_verified"):
                    return jsonify({
                        "error": "Registered face not verified in live camera.",
                        "identity_status": state.get("identity_message", "Blocked"),
                    }), 403
            return view_func(*args, **kwargs)

        # Never serve camera frames or protected API data before login + face verification.
        if request.path.startswith("/api/") or request.path == "/video_feed":
            return jsonify({"error": "Face verification required"}), 403

        if user_face_enrolled(session["user_id"]):
            return redirect(url_for("face_verify"))
        return redirect(url_for("face_enroll"))

    return wrapper


# =========================================================
# REGISTRATION, LOGIN, LOGOUT
# =========================================================


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not full_name or not email or not password:
            flash("Please fill in all required fields.", "error")
        elif password != confirm_password:
            flash("Passwords do not match.", "error")
        elif len(password) < 8:
            flash("Password must contain at least 8 characters.", "error")
        else:
            connection = get_db_connection()
            try:
                connection.execute(
                    """
                    INSERT INTO users (full_name, email, password_hash, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        full_name,
                        email,
                        generate_password_hash(password),
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ),
                )
                connection.commit()
                flash("Registration successful. Please log in.", "success")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("This email is already registered.", "error")
            finally:
                connection.close()

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        connection = get_db_connection()
        try:
            user = connection.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            ).fetchone()
        finally:
            connection.close()

        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = int(user["id"])
            session["user_name"] = user["full_name"]
            session["face_verified"] = False
            state = get_user_state(user["id"])
            state["session_active"] = False
            state["identity_verified"] = False
            state["identity_checked"] = False
            state["identity_match_streak"] = 0
            state["prediction_history"].clear()

            if user["face_enrolled"]:
                return redirect(url_for("face_verify"))
            return redirect(url_for("face_enroll"))

        flash("Invalid email or password.", "error")

    return render_template("login.html")


@app.route("/logout", methods=["POST"])
def logout():
    user_id = session.get("user_id")
    if user_id is not None:
        state = get_user_state(user_id)
        state["session_active"] = False
        state["identity_verified"] = False
        state["identity_checked"] = True
        state["identity_match_streak"] = 0
        state["last_identity_check"] = 0.0
        state["prediction_history"].clear()
        state["current_emotion"] = "Logged out"
        state["current_aroma"] = "Blocked"
        state["current_suggestion"] = "Log in and verify the registered face to resume."
        state["spray_status"] = "Blocked: session logged out."
    session.clear()
    return redirect(url_for("login"))


# =========================================================
# FACE ENROLLMENT AND VERIFICATION
# =========================================================


@app.route("/face-enroll")
@login_required
def face_enroll():
    if user_face_enrolled(session["user_id"]):
        return redirect(url_for("face_verify"))
    return render_template("face_enroll.html")


@app.route("/api/face/enroll", methods=["POST"])
@login_required
def api_face_enroll():
    user_id = int(session["user_id"])
    if session.get("face_verified"):
        return jsonify({"ok": False, "message": "Log out before changing face enrollment."}), 403
    if user_face_enrolled(user_id):
        return jsonify({"ok": False, "message": "Face is already enrolled. Please verify instead."}), 409

    payload = request.get_json(silent=True) or {}
    images = payload.get("images", [])
    monitoring_consent = bool(payload.get("monitoringConsent", False))
    pose_tags = payload.get("poses", [])
    if not isinstance(images, list) or not 5 <= len(images) <= 8:
        return jsonify({"ok": False, "message": "Capture 5–8 face samples across front and visible side angles."}), 400

    features = []
    valid_pose_tags = []
    errors = []
    for index, image_data in enumerate(images):
        frame, decode_error = decode_face_image(image_data)
        if decode_error:
            errors.append(decode_error)
            continue
        feature, bbox, face_error = _sface_extract(frame)
        if face_error:
            errors.append(f"Sample {index + 1}: {face_error}")
            continue
        features.append(feature)
        tag = pose_tags[index] if isinstance(pose_tags, list) and index < len(pose_tags) else f"angle_{index + 1}"
        valid_pose_tags.append(str(tag)[:40])

    if len(features) < 5:
        detail = errors[0] if errors else "Not enough valid face samples."
        return jsonify({
            "ok": False,
            "message": f"Only {len(features)} usable samples captured; at least 5 are required. {detail} Keep one face in view, turn slowly, and retry."
        }), 400

    # Duplicate enrollment check: compare each captured sample directly against
    # existing enrolled profiles, using a stricter threshold than live verification.
    # Do not let one borderline sample reject a different person's account.
    duplicate_hits = Counter()
    existing_embeddings = _stored_embeddings(
        exclude_user_id=user_id, enrolled_only=True
    )
    for feature in features:
        if not existing_embeddings:
            break
        scores = [
            (existing_user_id, _user_similarity(feature, samples))
            for existing_user_id, samples in existing_embeddings.items()
        ]
        matched_existing_id, similarity = max(scores, key=lambda item: item[1])
        if similarity >= SFACE_DUPLICATE_ENROLLMENT_THRESHOLD:
            duplicate_hits[matched_existing_id] += 1

    # Require a strong match across most usable pose samples before rejecting.
    duplicate_minimum = max(4, int(np.ceil(len(features) * 0.67)))
    duplicate_match = duplicate_hits.most_common(1)
    if duplicate_match and duplicate_match[0][1] >= duplicate_minimum:
        matched_id, matched_count = duplicate_match[0]
        return jsonify({
            "ok": False,
            "message": (
                f"These samples strongly match an existing face profile (account ID {matched_id}). "
                f"{matched_count} samples matched. If this is a different person, retry with "
                "clear, even lighting and capture all face angles. The same person's face "
                "must not be enrolled under multiple accounts."
            )
        }), 409

    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    connection = get_db_connection()
    try:
        connection.execute("DELETE FROM face_embeddings WHERE user_id = ?", (user_id,))
        connection.executemany(
            "INSERT INTO face_embeddings (user_id, embedding, pose_tag, created_at) VALUES (?, ?, ?, ?)",
            [
                (user_id, feature.astype(np.float32).tobytes(), valid_pose_tags[i], now_text)
                for i, feature in enumerate(features)
            ],
        )
        # Old LBPH crops are no longer used by the SFace system.
        connection.execute("DELETE FROM face_samples WHERE user_id = ?", (user_id,))
        connection.execute(
            "UPDATE users SET face_enrolled = 1, monitoring_enabled = ? WHERE id = ?",
            (1 if monitoring_consent else 0, user_id),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        app.logger.exception("Saving SFace enrollment failed")
        return jsonify({"ok": False, "message": "Could not save the face profile. Please retry."}), 500
    finally:
        connection.close()

    state = get_user_state(user_id)
    state["session_active"] = False
    state["identity_verified"] = False
    state["identity_checked"] = False
    state["identity_match_streak"] = 0
    state["last_identity_check"] = 0.0
    state["prediction_history"].clear()
    session["face_verified"] = False
    return jsonify({
        "ok": True,
        "message": f"Face enrolled from {len(features)} pose samples. Please verify your face now.",
        "next": url_for("face_verify"),
    })


@app.route("/face-verify")
@login_required
def face_verify():
    if session.get("face_verified"):
        return redirect(url_for("index"))
    connection = get_db_connection()
    try:
        user = connection.execute(
            "SELECT face_enrolled, monitoring_enabled FROM users WHERE id = ?",
            (int(session["user_id"]),),
        ).fetchone()
    finally:
        connection.close()
    if not user or not user["face_enrolled"]:
        return redirect(url_for("face_enroll"))
    return render_template("face_verify.html", monitoring_enabled=bool(user["monitoring_enabled"]))


@app.route("/api/face/verify", methods=["POST"])
@login_required
def api_face_verify():
    # A client-uploaded still image must never grant live camera authorization.
    return jsonify({
        "ok": False,
        "message": "Use the always-on local camera verification screen instead.",
        "next": url_for("face_verify"),
    }), 410


# =========================================================
# EMOTION STORAGE AND PROCESSING
# =========================================================


def save_emotion(emotion, user_id):
    state = get_user_state(user_id)
    recommendation = recommendations.get(
        emotion,
        {"aroma": "None", "suggestion": "No recommendation available."},
    )
    state["current_aroma"] = recommendation["aroma"]
    state["current_suggestion"] = recommendation["suggestion"]
    detected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    connection = get_db_connection()
    try:
        connection.execute(
            """
            INSERT INTO emotion_history
                (emotion, aroma, suggestion, detected_at, user_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                emotion,
                state["current_aroma"],
                state["current_suggestion"],
                detected_at,
                int(user_id),
            ),
        )
        connection.commit()
    finally:
        connection.close()

    print(
        f"Saved for user {user_id}: {emotion} | Aroma: {state['current_aroma']}"
    )


def send_to_arduino(aroma, user_id):
    """Send one aroma command only while the current live face is authorized."""
    state = get_user_state(user_id)
    now = time.time()
    if not get_monitoring_enabled(user_id):
        state["spray_status"] = "Blocked: continuous monitoring is paused."
        print("Spray blocked: monitoring consent is off.")
        return False
    if not state.get("identity_verified") or now - state.get("last_identity_check", 0.0) > IDENTITY_FRESHNESS_SECONDS:
        state["spray_status"] = "Blocked: registered face is not currently verified."
        print("Spray blocked: no fresh registered-face match.")
        return False

    command = AROMA_COMMANDS.get(aroma)
    if command is None:
        state["spray_status"] = f"No Arduino relay mapping for aroma: {aroma}"
        print(state["spray_status"])
        return False

    if not AROMA_HARDWARE_ENABLED:
        state["spray_status"] = f"Dry-run only: {aroma} command not sent."
        print(f"SAFE TEST MODE: would send {command!r} for {aroma}.")
        return False

    if arduino is None:
        state["spray_status"] = "Hardware unavailable; no spray command sent."
        print(state["spray_status"])
        return False

    # Check identity again immediately before serial write.
    if not state.get("identity_verified") or time.time() - state.get("last_identity_check", 0.0) > IDENTITY_FRESHNESS_SECONDS:
        state["spray_status"] = "Blocked: identity expired before relay command."
        return False

    try:
        duration_seconds = max(1, min(MAX_SPRAY_DURATION_SECONDS, int(state.get("spray_duration_seconds", DEFAULT_SPRAY_DURATION_SECONDS))))
        # Timer-aware firmware protocol: one line such as H,5\n means Citrus for 5 seconds.
        payload = f"{command.decode('ascii')},{duration_seconds}\n".encode("ascii")
        arduino.write(payload)
        arduino.flush()
        state["spray_status"] = f"Command sent for {aroma}; requested spray duration: {duration_seconds}s."
        print(f"Arduino command sent for user {user_id}: {aroma} ({payload.decode('ascii').strip()})")
        return True
    except Exception as exc:
        state["spray_status"] = "Arduino write failed; no new command sent."
        app.logger.exception("Arduino command failed: %s", exc)
        return False


def process_confirmed_emotion(emotion, user_id):
    state = get_user_state(user_id)
    # Monitoring consent is checked again at the moment of record/spray.
    if not get_monitoring_enabled(user_id):
        state["prediction_history"].clear()
        state["current_emotion"] = "Monitoring paused"
        state["current_aroma"] = "Not activated"
        state["current_suggestion"] = "Monitoring is paused; no emotion record or spray command was created."
        state["spray_status"] = "Blocked: monitoring consent is off."
        return
    # Do not record or actuate on a stale / unverified identity.
    if not state.get("identity_verified") or time.time() - state.get("last_identity_check", 0.0) > IDENTITY_FRESHNESS_SECONDS:
        state["prediction_history"].clear()
        state["current_emotion"] = "Identity not verified"
        state["current_aroma"] = "Blocked"
        state["current_suggestion"] = "Emotion processing and spray are blocked until the registered face is confirmed."
        state["spray_status"] = "Blocked: registered face is not currently verified."
        print("Confirmed emotion discarded because identity is not fresh.")
        return

    state["current_emotion"] = emotion
    save_emotion(emotion, user_id)
    aroma = state["current_aroma"]
    send_to_arduino(aroma, user_id)
    maybe_create_emotion_notification(user_id, emotion)
    state["prediction_history"].clear()
    settings = get_timer_settings(user_id)
    apply_timer_settings_to_state(user_id, settings)
    state["next_recommendation_time"] = time.time() + state["timer_interval_seconds"] if state["timer_enabled"] else float("inf")
    print(
        f"User {user_id} recommendation: {emotion} / {aroma}; "
        f"next cycle in {state['timer_interval_seconds']} seconds"
    )


def maybe_trigger_no_face_fallback(user_id, identity_message):
    """Optional fallback only when the camera reports no face at all; unknown/multiple faces never trigger it."""
    if user_id is None or "no face detected" not in str(identity_message).lower():
        return False
    state = get_user_state(user_id)
    if not state.get("timer_settings_loaded"):
        apply_timer_settings_to_state(user_id)
    if not get_monitoring_enabled(user_id) or not state.get("timer_enabled") or not state.get("fallback_enabled"):
        return False
    now = time.time()
    if now < state.get("next_recommendation_time", float("inf")):
        return False
    # Never run fallback when camera is unavailable, multiple faces are visible, or an unknown face is detected.
    aroma = state.get("fallback_aroma", "Citrus")
    command = AROMA_COMMANDS.get(aroma)
    duration = max(1, min(MAX_SPRAY_DURATION_SECONDS, int(state.get("spray_duration_seconds", DEFAULT_SPRAY_DURATION_SECONDS))))
    state["prediction_history"].clear()
    state["current_emotion"] = "No face detected"
    state["current_aroma"] = aroma
    state["current_suggestion"] = f"Optional no-face fallback selected {aroma}. This is not an emotion prediction."
    if not AROMA_HARDWARE_ENABLED:
        state["spray_status"] = f"Fallback dry-run only: {aroma}, {duration}s; hardware disabled."
        print(state["spray_status"])
    elif arduino is None or command is None:
        state["spray_status"] = "Fallback blocked: Arduino is unavailable or the aroma mapping is invalid."
    else:
        try:
            payload = f"{command.decode('ascii')},{duration}\n".encode("ascii")
            arduino.write(payload)
            arduino.flush()
            state["spray_status"] = f"Optional no-face fallback command sent: {aroma}, {duration}s."
            print(f"Optional no-face fallback for user {user_id}: {payload.decode('ascii').strip()}")
        except Exception:
            app.logger.exception("No-face fallback Arduino command failed")
            state["spray_status"] = "Fallback failed; no successful hardware command was confirmed."
    state["next_recommendation_time"] = now + state["timer_interval_seconds"]
    create_notification(user_id, "fallback", "Optional no-face fallback", state["current_suggestion"])
    return True


def verify_live_registered_face(frame, user_id):
    feature, bbox, error = _sface_extract(frame)
    if error:
        return False, error, None
    matched_user, similarity, message = _best_enrolled_identity(feature)
    if matched_user != int(user_id):
        return False, "Registered face not matched to this account; processing and spray blocked.", None
    return True, f"Registered face matched (cosine similarity {similarity:.3f}).", bbox


def block_user_state_for_identity_loss(user_id, message):
    state = get_user_state(user_id)
    was_verified = state.get("identity_verified", False)
    state["identity_checked"] = True
    state["identity_verified"] = False
    state["identity_match_streak"] = 0
    state["last_identity_check"] = 0.0
    state["identity_message"] = message
    state["prediction_history"].clear()
    normalized_message = str(message or "Identity not verified")
    if normalized_message in ("No Face Detected", "Other Face Detected", "Multiple Faces Detected"):
        state["current_emotion"] = normalized_message
        state["current_suggestion"] = "Emotion processing is blocked for this camera condition."
    else:
        state["current_emotion"] = "Identity not verified"
        state["current_suggestion"] = "Only the registered person may use emotion detection."
    state["current_aroma"] = "OFF"
    state["spray_status"] = "Automatic spray is OFF for no-face; unknown/multiple fallback requires its timer setting."
    # Require a fresh recommendation interval after the registered person returns.
    # Preserve the scheduled due time while the enrolled face is absent. A pending cycle runs
    # when the registered user returns, unless an enabled no-face fallback already consumed it.



def detect_emotion(frame, user_id, face_box):
    """Run the emotion CNN only on the already identity-matched face crop."""
    state = get_user_state(user_id)
    if not state.get("timer_settings_loaded"):
        apply_timer_settings_to_state(user_id)
    if not state.get("timer_enabled"):
        state["current_emotion"] = "Timer off"
        state["current_aroma"] = "Not activated"
        state["current_suggestion"] = "The emotion cycle timer is off. Turn it on in Settings to resume scheduled predictions."
        return frame
    if time.time() < state.get("next_recommendation_time", float("inf")):
        return frame
    if not get_monitoring_enabled(user_id):
        state["current_emotion"] = "Monitoring paused"
        state["current_aroma"] = "Not activated"
        state["prediction_history"].clear()
        return frame
    if (
        not state.get("identity_verified")
        or time.time() - state.get("last_identity_check", 0.0) > IDENTITY_FRESHNESS_SECONDS
    ):
        state["current_emotion"] = "Identity not verified"
        state["current_aroma"] = "Blocked"
        state["current_suggestion"] = "Registered face check expired; emotion detection is paused."
        state["prediction_history"].clear()
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x, y, width, height = [int(v) for v in face_box]
    roi_gray = gray[y:y + height, x:x + width]
    if roi_gray.size == 0:
        return frame

    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
    roi = roi_gray.astype("float32") / 255.0
    roi = np.expand_dims(np.expand_dims(roi, axis=0), axis=-1)

    try:
        prediction = classifier.predict(roi, verbose=0)[0]
        detected_label = emotion_labels[int(np.argmax(prediction))]
    except Exception:
        app.logger.exception("CNN emotion prediction failed")
        state["current_emotion"] = "Prediction unavailable"
        return frame

    # Recheck identity and consent after the model call: do not record or spray if either expired.
    if not get_monitoring_enabled(user_id):
        state["prediction_history"].clear()
        state["current_emotion"] = "Monitoring paused"
        state["current_aroma"] = "Not activated"
        state["current_suggestion"] = "Monitoring was paused during prediction; no record or spray was created."
        state["spray_status"] = "Blocked: monitoring consent is off."
        return frame
    if not state.get("identity_verified") or time.time() - state.get("last_identity_check", 0.0) > IDENTITY_FRESHNESS_SECONDS:
        state["prediction_history"].clear()
        state["current_emotion"] = "Identity not verified"
        state["current_aroma"] = "Blocked"
        state["current_suggestion"] = "Registered face check expired; no recommendation or spray command sent."
        state["spray_status"] = "Blocked: identity check expired."
        return frame

    state["current_emotion"] = detected_label
    if time.time() >= state["next_recommendation_time"]:
        state["prediction_history"].append(detected_label)
        print(
            f"User {user_id} sampling: {len(state['prediction_history'])}/"
            f"{SAMPLE_COUNT} | Detected: {detected_label}"
        )
        if len(state["prediction_history"]) >= SAMPLE_COUNT:
            majority_emotion = Counter(state["prediction_history"]).most_common(1)[0][0]
            print(f"User {user_id} {SAMPLE_COUNT}-sample majority: {majority_emotion}")
            process_confirmed_emotion(majority_emotion, user_id)

    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 200, 100), 2)
    cv2.putText(
        frame,
        "Emotion: " + state["current_emotion"],
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        "Aroma: " + state["current_aroma"],
        (x, min(y + height + 25, frame.shape[0] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2,
    )
    return frame


def get_monitoring_enabled(user_id):
    connection = get_db_connection()
    try:
        row = connection.execute(
            "SELECT monitoring_enabled FROM users WHERE id = ?", (int(user_id),)
        ).fetchone()
        return bool(row and row["monitoring_enabled"])
    finally:
        connection.close()



def identify_registered_face(frame):
    """Detect every visible face; allow emotion only if an enrolled identity is present."""
    if frame is None or frame.size == 0:
        return None, "Face Detection Unavailable", None

    height, width = frame.shape[:2]
    try:
        with face_model_lock:
            face_detector_sface.setInputSize((int(width), int(height)))
            _status, faces = face_detector_sface.detect(frame)
            detections = [] if faces is None else list(faces)
            features_and_boxes = []
            for detection in detections:
                aligned = face_embedding_model.alignCrop(frame, detection)
                raw_feature = face_embedding_model.feature(aligned)
                feature = np.asarray(raw_feature, dtype=np.float32).reshape(-1)
                norm = float(np.linalg.norm(feature))
                if norm <= 1e-12:
                    continue
                feature /= norm
                x, y, box_w, box_h = [int(round(v)) for v in detection[:4]]
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                box_w = max(1, min(box_w, width - x))
                box_h = max(1, min(box_h, height - y))
                features_and_boxes.append((feature, (x, y, box_w, box_h)))
    except Exception:
        app.logger.exception("Multi-face YuNet/SFace extraction failed")
        return None, "Face Detection Unavailable", None

    if not detections:
        return None, "No Face Detected", None

    matches = []
    for feature, bbox in features_and_boxes:
        matched_user, similarity, match_message = _best_enrolled_identity(feature)
        if matched_user is not None and user_face_enrolled(matched_user):
            matches.append((int(matched_user), float(similarity), bbox))

    if matches:
        # Prefer the strongest enrolled match. This allows a registered user to be
        # tracked even when other people are also visible; CNN receives only this crop.
        matches.sort(key=lambda item: item[1], reverse=True)
        user_id, similarity, bbox = matches[0]
        status = (
            f"Multiple Faces Detected — Registered Face Detected (similarity {similarity:.3f})"
            if len(detections) > 1
            else f"Registered Face Detected (similarity {similarity:.3f})"
        )
        return user_id, status, bbox

    if len(detections) > 1:
        return None, "Multiple Faces Detected", None
    return None, "Other Face Detected", None


class BrowserMP4Writer:
    """Encode browser-compatible H.264 MP4 using the FFmpeg bundled by imageio-ffmpeg."""

    def __init__(self, path, width, height, fps):
        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise RuntimeError(
                "Video recording requires imageio-ffmpeg. Run: "
                "python -m pip install imageio-ffmpeg"
            ) from exc

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg_exe,
            "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo",
            "-pixel_format", "bgr24",
            "-video_size", f"{width}x{height}",
            "-framerate", str(float(fps)),
            "-i", "pipe:0",
            "-an",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "24",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(path),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        time.sleep(0.05)
        if self.process.poll() is not None:
            error = self.process.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"FFmpeg could not start MP4 recording: {error[:500]}")

    def isOpened(self):
        return self.process is not None and self.process.poll() is None

    def write(self, frame):
        if not self.isOpened() or self.process.stdin is None:
            raise RuntimeError("The MP4 encoder is not running.")
        image = np.ascontiguousarray(frame, dtype=np.uint8)
        self.process.stdin.write(image.tobytes())

    def release(self):
        if self.process is None:
            return
        try:
            if self.process.stdin and not self.process.stdin.closed:
                self.process.stdin.close()
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=3)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass
        finally:
            if self.process.stderr:
                error = self.process.stderr.read().decode("utf-8", errors="replace")
                if error.strip():
                    app.logger.warning("FFmpeg recording encoder: %s", error.strip()[:500])
            self.process = None


def _recording_start(user_id, frame, duration_seconds):
    """Start a scheduled scene clip; the frame need not contain a recognized face."""
    user_dir = RECORDINGS_DIR / f"user_{int(user_id)}"
    user_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = user_dir / f"scheduled_scene_{stamp}_{secrets.token_hex(3)}.mp4"
    height, width = frame.shape[:2]
    fps = float(RECORDING_FPS)
    duration_seconds = max(1, min(MAX_SCHEDULED_RECORDING_DURATION_SECONDS, int(duration_seconds)))
    try:
        writer = BrowserMP4Writer(path, width, height, fps)
    except Exception:
        app.logger.exception("Could not start browser-compatible video recording")
        return None
    now = time.time()
    return {
        "user_id": int(user_id),
        "path": path,
        "writer": writer,
        "started_epoch": now,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_duration_seconds": duration_seconds,
        "frame_count": 0,
        "fps": fps,
        "last_write_epoch": now,
        "next_write_epoch": now,
        "last_frame": np.ascontiguousarray(frame.copy()),
    }


def _recording_finish(recording):
    """Finalize the MP4, then commit its metadata so the clip appears in Camera Recordings."""
    if not recording:
        return
    try:
        recording["writer"].release()
    except Exception:
        app.logger.exception("Could not finalize video writer for %s", recording.get("path"))
    ended_epoch = time.time()
    ended_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = Path(recording["path"])
    try:
        file_size = path.stat().st_size if path.exists() else 0
    except OSError:
        file_size = 0
    if recording["frame_count"] < 2 or file_size == 0:
        print(
            f"[REC] Clip not saved: too few frames or empty MP4; "
            f"frames={recording.get('frame_count')}, bytes={file_size}, path={path}"
        )
        try:
            path.unlink(missing_ok=True)
        except OSError:
            app.logger.exception("Could not remove incomplete recording %s", path)
        return
    try:
        relative_path = str(path.relative_to(BASE_DIR))
        connection = get_db_connection()
        try:
            cursor = connection.execute(
                "INSERT INTO video_recordings (user_id, file_path, file_name, started_at, ended_at, duration_seconds, frame_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    recording["user_id"], relative_path, path.name,
                    recording["started_at"], ended_at,
                    round(recording["frame_count"] / recording["fps"], 2),
                    int(recording["frame_count"]),
                ),
            )
            connection.commit()
            recording_id = cursor.lastrowid
        finally:
            connection.close()
        state = get_user_state(recording["user_id"])
        state["last_saved_clip"] = path.name
        state["last_saved_clip_at"] = ended_at
        state["recording_status"] = f"Saved {path.name}; continuous recording is ready for the next clip."
        print(
            f"[REC] SAVED id={recording_id} user={recording['user_id']} "
            f"duration={recording['frame_count'] / recording['fps']:.2f}s "
            f"frames={recording['frame_count']} bytes={file_size} path={relative_path}"
        )
    except Exception:
        app.logger.exception("Could not save recording metadata for %s", path)


def cleanup_old_recordings():
    """Delete local video clips and their SQLite metadata older than the configured retention period."""
    cutoff = (datetime.now() - timedelta(days=RECORDING_RETENTION_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
    connection = get_db_connection()
    try:
        rows = connection.execute(
            "SELECT id, file_path FROM video_recordings WHERE started_at < ?", (cutoff,)
        ).fetchall()
        deleted_ids = []
        for row in rows:
            path = (BASE_DIR / row["file_path"]).resolve()
            try:
                path.relative_to(RECORDINGS_DIR.resolve())
            except ValueError:
                app.logger.warning("Skipped invalid recording path during cleanup: %s", row["file_path"])
                continue
            try:
                path.unlink(missing_ok=True)
                deleted_ids.append(int(row["id"]))
            except OSError:
                app.logger.exception("Could not remove expired recording %s", path)
        if deleted_ids:
            connection.executemany("DELETE FROM video_recordings WHERE id = ?", [(item_id,) for item_id in deleted_ids])
            connection.commit()
            print(f"Recording retention cleanup removed {len(deleted_ids)} expired clip(s).")
    finally:
        connection.close()


def _write_scheduled_recording(recording, user_id, frame, allow_start=True):
    """Continuously record the camera scene and split it into fixed-length MP4 clips.

    Video recording is intentionally independent of face/emotion detection: the scene may
    contain the room, no person, or other people. It still requires the signed-in account's
    explicit monitoring consent and the continuous-recording switch. This function never
    calls the CNN or sends an Arduino command.
    """
    now = time.time()
    was_recording = recording is not None
    # Once recording has started, preserve the consent owner's identity even when another
    # face appears or no face is detected. The video file is owned by the account that opted in.
    owner_id = int(recording["user_id"]) if recording is not None else (int(user_id) if user_id is not None else None)
    if owner_id is None:
        return recording

    state = get_user_state(owner_id)
    if not state.get("timer_settings_loaded"):
        apply_timer_settings_to_state(owner_id)

    consent_on = get_monitoring_enabled(owner_id)
    recording_on = bool(state.get("recording_timer_enabled"))
    if not consent_on or not recording_on:
        if recording:
            _recording_finish(recording)
        state["recording_status"] = "Continuous video recording is off or monitoring consent is paused."
        return None

    # Only the camera worker touches the FFmpeg writer. If the user changes clip length,
    # the request thread asks this worker to finalize the current segment safely on its next frame.
    with recording_rotation_lock:
        rotation_requested = owner_id in recording_rotation_requests
        if rotation_requested:
            recording_rotation_requests.discard(owner_id)

    rotated = False
    duration_expired = bool(
        recording and now - recording["started_epoch"] >= recording["target_duration_seconds"]
    )
    if recording and (duration_expired or rotation_requested):
        reason = "clip duration reached" if duration_expired else "clip length changed in settings"
        # Write the newly captured frame before closing. This is important for 1–3 second
        # clips on slower laptops: recognition may take longer than the clip duration, and
        # closing first would leave only one encoded frame, causing the clip to be discarded.
        try:
            recording["writer"].write(frame)
            recording["frame_count"] += 1
            recording["last_frame"] = np.ascontiguousarray(frame.copy())
        except Exception:
            app.logger.exception("Could not flush final frame before rotating recording")
        print(
            f"[REC] Finalizing clip: reason={reason}, "
            f"elapsed={now - recording['started_epoch']:.2f}s, "
            f"target={recording['target_duration_seconds']}s"
        )
        _recording_finish(recording)
        recording = None
        rotated = True
        state["recording_status"] = "Previous clip finalized; starting the next clip."

    if recording is None:
        # First recording starts only after the user has enabled it; when an existing clip just
        # rotated, always start the successor immediately even if allow_start=False was supplied.
        if not allow_start and not (was_recording and rotated):
            return None
        duration = int(state.get("recording_duration_seconds", DEFAULT_RECORDING_DURATION_SECONDS))
        recording = _recording_start(owner_id, frame, duration)
        if recording is None:
            state["recording_status"] = "Could not start MP4 recording; will retry on the next camera frame."
            return None
        state["next_recording_time"] = now + duration  # informational compatibility field only
        state["recording_status"] = f"RECORDING now; current clip will rotate after {duration} second(s)."
        print(f"[REC] START user={owner_id} clip={path.name if (path := recording.get('path')) else 'unknown'} target={duration}s")
        try:
            recording["writer"].write(frame)
            recording["frame_count"] = 1
            recording["last_write_epoch"] = now
            recording["next_write_epoch"] = now + (1.0 / recording["fps"])
            recording["last_frame"] = np.ascontiguousarray(frame.copy())
        except Exception:
            app.logger.exception("Could not write the initial scheduled video frame")
            _recording_finish(recording)
            return None
        return recording

    try:
        interval = 1.0 / recording["fps"]
        due_frames = 0
        # Repeat the last frame as needed to keep encoded clip length aligned with elapsed time
        # when CNN/face processing runs slower than the selected recording frame rate.
        while now >= recording["next_write_epoch"] and due_frames < int(recording["fps"] * 3):
            recording["writer"].write(recording["last_frame"])
            recording["frame_count"] += 1
            recording["next_write_epoch"] += interval
            due_frames += 1
        recording["last_frame"] = np.ascontiguousarray(frame.copy())
        recording["last_write_epoch"] = now
    except Exception:
        app.logger.exception("Continuous video frame recording failed")
        _recording_finish(recording)
        return None
    return recording


def _make_camera_placeholder(text):
    canvas = np.full((360, 640, 3), 248, dtype=np.uint8)
    cv2.putText(canvas, "SMART AROMA CAMERA", (24, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (90, 60, 170), 2)
    lines = [text[i:i + 52] for i in range(0, len(text), 52)] or [text]
    for index, line in enumerate(lines[:3]):
        cv2.putText(canvas, line, (24, 160 + index * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (65, 65, 80), 1)
    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 78])
    return encoded.tobytes() if ok else b""


def _publish_camera_frame(raw_frame, display_frame, owner_id, status):
    global camera_sequence, latest_raw_frame, latest_camera_jpeg, latest_camera_status, active_camera_user_id
    ok, encoded = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if not ok:
        return
    with camera_condition:
        camera_sequence += 1
        latest_raw_frame = raw_frame.copy()
        latest_camera_jpeg = encoded.tobytes()
        latest_camera_status = status
        active_camera_user_id = owner_id
        camera_condition.notify_all()


def _publish_placeholder(status):
    global camera_sequence, latest_raw_frame, latest_camera_jpeg, latest_camera_status, active_camera_user_id
    data = _make_camera_placeholder(status)
    with camera_condition:
        camera_sequence += 1
        latest_raw_frame = None
        latest_camera_jpeg = data
        latest_camera_status = status
        active_camera_user_id = None
        camera_condition.notify_all()


def generate_frames(user_id):
    """Stream a clean live camera preview; authorization status is shown outside the video, not drawn over it."""
    last_sequence = -1
    while True:
        with camera_condition:
            camera_condition.wait_for(
                lambda: camera_sequence != last_sequence or camera_stop_event.is_set(),
                timeout=1.0,
            )
            last_sequence = camera_sequence
            raw_frame = latest_raw_frame.copy() if latest_raw_frame is not None else None
        if raw_frame is not None:
            ok, encoded = cv2.imencode(".jpg", raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
            frame_bytes = encoded.tobytes() if ok else _make_camera_placeholder("Camera preview unavailable")
        else:
            frame_bytes = _make_camera_placeholder("Camera starting…")
        if frame_bytes:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        if camera_stop_event.is_set():
            break


def enrollment_preview_frames():
    """Show a clean raw camera preview during enrollment; do not expose the monitoring overlay."""
    last_sequence = -1
    while True:
        with camera_condition:
            camera_condition.wait_for(
                lambda: camera_sequence != last_sequence or camera_stop_event.is_set(),
                timeout=1.0,
            )
            last_sequence = camera_sequence
            raw_frame = latest_raw_frame.copy() if latest_raw_frame is not None else None

        if raw_frame is not None:
            # Clean in-memory preview: no previous-user identity or status text is drawn.
            ok, encoded = cv2.imencode(".jpg", raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
            frame_bytes = encoded.tobytes() if ok else _make_camera_placeholder("Camera preview unavailable")
        else:
            frame_bytes = _make_camera_placeholder("Camera starting…")

        if frame_bytes:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        if camera_stop_event.is_set():
            break


@app.route("/verification-feed")
@login_required
def verification_feed():
    """Serve the existing continuous camera preview for face verification."""
    return Response(
        enrollment_preview_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


def get_fresh_camera_snapshot(after_sequence=-1, timeout=3.0):
    """Wait for a new raw in-memory frame; it is only returned, never persisted by this route."""
    with camera_condition:
        camera_condition.wait_for(
            lambda: camera_sequence > int(after_sequence) and latest_raw_frame is not None,
            timeout=timeout,
        )
        if latest_raw_frame is None or camera_sequence <= int(after_sequence):
            return None, camera_sequence
        return latest_raw_frame.copy(), camera_sequence


def continuous_camera_worker():
    """Run while python app.py is alive; identify enrolled users for emotion/spray gating and record consented scenes continuously."""
    global active_camera_user_id, last_monitoring_user_id
    cap = None
    candidate_user_id = None
    match_streak = 0
    verified_user_id = None
    recording = None
    frame_count = 0
    last_daily_report_check = 0.0
    last_frame_tick = 0.0
    last_recording_cleanup_date = None
    print("Continuous camera worker starting. Scene recording is consent-gated; emotion/spray remain face-verified.")
    try:
        cleanup_old_recordings()
        last_recording_cleanup_date = datetime.now().date()
    except Exception:
        app.logger.exception("Initial recording retention cleanup failed")

    try:
        while not camera_stop_event.is_set():
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(CAMERA_DEVICE_INDEX, cv2.CAP_DSHOW)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 15)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    print(f"Always-on camera connected (index {CAMERA_DEVICE_INDEX}).")
                else:
                    _publish_placeholder("Camera unavailable. No emotion processing or recording.")
                    if time.time() - last_daily_report_check > 30:
                        try:
                            maybe_create_daily_reports()
                        except Exception:
                            app.logger.exception("Daily notification scheduler failed")
                        today = datetime.now().date()
                        if last_recording_cleanup_date != today:
                            try:
                                cleanup_old_recordings()
                                last_recording_cleanup_date = today
                            except Exception:
                                app.logger.exception("Recording retention cleanup failed")
                        last_daily_report_check = time.time()
                    time.sleep(3)
                    continue

            frame_started = time.perf_counter()
            ok, frame = cap.read()
            if not ok or frame is None:
                if verified_user_id is not None:
                    block_user_state_for_identity_loss(verified_user_id, "Camera frame unavailable; all processing blocked.")
                _recording_finish(recording)
                recording = None
                candidate_user_id = None
                match_streak = 0
                verified_user_id = None
                _publish_placeholder("Camera frame unavailable. Recording and spray blocked.")
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
                time.sleep(0.1)
                continue

            raw_frame = frame.copy()  # Used in memory; disk writing is handled by the separate consented scene recorder.
            frame_count += 1
            user_id, message, face_box = identify_registered_face(raw_frame)
            display_status = message

            if user_id is None:
                # Face-state notifications are event-based and ignore the wellness timer.
                handle_camera_status_notification(message)
                fallback_owner = verified_user_id if verified_user_id is not None else last_monitoring_user_id
                if verified_user_id is not None:
                    block_user_state_for_identity_loss(verified_user_id, message)
                if fallback_owner is not None:
                    if message == "No Face Detected":
                        maybe_trigger_no_face_fallback(fallback_owner, message)
                    elif message in ("Other Face Detected", "Multiple Faces Detected"):
                        maybe_trigger_nonregistered_spray(fallback_owner, message)
                    state = get_user_state(fallback_owner)
                    state["current_emotion"] = message
                    if message == "No Face Detected":
                        state["current_aroma"] = "OFF" if not state.get("fallback_enabled") else state.get("fallback_aroma", "Citrus")
                    elif state.get("spray_status", "").startswith("Dry-run only"):
                        state["current_aroma"] = state.get("fallback_aroma", "Citrus")
                    elif state.get("spray_status", "").startswith(("Other Face", "Multiple Faces")):
                        state["current_aroma"] = state.get("fallback_aroma", "Citrus")
                    else:
                        state["current_aroma"] = "OFF"
                # Scheduled clips can keep recording the camera scene even when no enrolled
                # face or no emotion is detected, provided this account opted in and enabled it.
                # Continuous video recording is handled once per camera frame below, independently
                # of whether a face is found. Emotion processing and spray remain blocked here.
                candidate_user_id = None
                match_streak = 0
                verified_user_id = None
                display_status = message
                # Keep the preview clean when an identity is absent/unknown. The separate per-frame
                # scene recorder can continue with consent; the CNN/spray paths remain blocked.
                _publish_camera_frame(raw_frame, raw_frame, None, display_status)
            else:
                handle_camera_status_notification(None)
                if user_id == candidate_user_id:
                    match_streak += 1
                else:
                    if verified_user_id is not None and verified_user_id != user_id:
                        block_user_state_for_identity_loss(verified_user_id, "A different registered user appeared; emotion processing and spray for the previous identity are blocked. Scene recording continues only under the original account's consent.")
                    # Do not interrupt scene recording when the visible person changes.
                    # Recording ownership stays with the account that explicitly enabled it.
                    candidate_user_id = user_id
                    match_streak = 1
                    verified_user_id = None

                if match_streak < LIVE_MATCH_FRAMES_REQUIRED:
                    state = get_user_state(user_id)
                    state["identity_checked"] = True
                    state["identity_verified"] = False
                    state["identity_match_streak"] = match_streak
                    state["identity_message"] = "Confirming registered face."
                    state["last_identity_check"] = 0.0
                    state["prediction_history"].clear()
                    # Show clean live video while accumulating consecutive face matches.
                    # Do not enable emotion processing or recording until the streak is complete.
                    _publish_camera_frame(raw_frame, raw_frame, None, "Confirming registered face")
                    # Keep the clip rolling during the consecutive-frame face confirmation.
                else:
                    state = get_user_state(user_id)
                    now = time.time()
                    if verified_user_id != user_id:
                        state["prediction_history"].clear()
                        if not state.get("timer_settings_loaded"):
                            apply_timer_settings_to_state(user_id)
                        if not state.get("ever_live_verified"):
                            state["next_recommendation_time"] = now + state["timer_interval_seconds"] if state["timer_enabled"] else float("inf")
                        state["ever_live_verified"] = True
                    verified_user_id = user_id
                    state["session_active"] = True
                    state["identity_checked"] = True
                    state["identity_verified"] = True
                    state["identity_match_streak"] = match_streak
                    state["last_identity_check"] = now
                    state["identity_message"] = message
                    if not state.get("timer_settings_loaded"):
                        apply_timer_settings_to_state(user_id)
                        state["next_recommendation_time"] = time.time() + state["timer_interval_seconds"] if state["timer_enabled"] else float("inf")
                    is_monitoring_on = get_monitoring_enabled(user_id)
                    state["monitoring_enabled"] = is_monitoring_on
                    if is_monitoring_on:
                        last_monitoring_user_id = user_id

                    x, y, width, height = face_box
                    # Do not draw account IDs or blocked-status text into the video preview.

                    if is_monitoring_on:
                        if state.get("timer_enabled") and time.time() >= state.get("next_recommendation_time", float("inf")) and frame_count % 2 == 0:
                            frame = detect_emotion(frame, user_id, face_box)
                        else:
                            if not state.get("timer_enabled"):
                                timer_label = "Timer: OFF"
                            else:
                                remaining = max(0, int(state.get("next_recommendation_time", time.time()) - time.time()))
                                timer_label = f"Next sample in {remaining // 3600:02d}:{(remaining % 3600) // 60:02d}:{remaining % 60:02d}"
                            cv2.putText(frame, f"Emotion: {state['current_emotion']}", (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 170, 80), 2)
                            cv2.putText(frame, f"Aroma: {state['current_aroma']}", (16, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 120, 10), 2)
                            cv2.putText(frame, timer_label, (16, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (170, 120, 20), 2)
                        # Emotion processing and spray are identity-gated above. Video recording
                        # is handled once per frame below and does not depend on face recognition.
                        state["spray_status"] = state.get("spray_status", "Waiting for the next confirmed recommendation.")
                    else:
                        state["current_emotion"] = "Monitoring paused"
                        state["current_aroma"] = "Not activated"
                        state["current_suggestion"] = "Face recognized. Enable continuous monitoring and recording in the dashboard if you consent."
                        state["spray_status"] = "Monitoring consent is off; no recording or spray command sent."
                        # Raw video remains visible; monitoring remains safely disabled.
                        display_status = f"Face recognized; monitoring paused for user {user_id}"
                    _publish_camera_frame(raw_frame, frame, user_id, display_status)

            recording_owner = recording["user_id"] if recording is not None else last_monitoring_user_id
            if recording_owner is not None:
                recording = _write_scheduled_recording(recording, recording_owner, raw_frame, allow_start=True)

            if time.time() - last_daily_report_check > 30:
                try:
                    maybe_create_daily_reports()
                except Exception:
                    app.logger.exception("Daily notification scheduler failed")
                today = datetime.now().date()
                if last_recording_cleanup_date != today:
                    try:
                        cleanup_old_recordings()
                        last_recording_cleanup_date = today
                    except Exception:
                        app.logger.exception("Recording retention cleanup failed")
                last_daily_report_check = time.time()

            # Avoid saturating the CPU while checking the camera continuously.
            minimum_frame_time = 1.0 / CAMERA_TARGET_FPS
            elapsed = time.perf_counter() - frame_started
            if elapsed < minimum_frame_time:
                time.sleep(minimum_frame_time - elapsed)
    except Exception:
        app.logger.exception("Continuous camera worker stopped unexpectedly")
    finally:
        _recording_finish(recording)
        if verified_user_id is not None:
            block_user_state_for_identity_loss(verified_user_id, "Camera service stopped; recording and spray blocked.")
        if cap is not None:
            cap.release()
        _publish_placeholder("Camera service stopped. Start python app.py to resume.")
        print("Continuous camera worker stopped.")


def camera_worker_supervisor():
    """Restart the background camera worker after unexpected failures while the app remains alive."""
    while not camera_stop_event.is_set():
        continuous_camera_worker()
        if not camera_stop_event.is_set():
            app.logger.error("Camera worker exited unexpectedly; retrying in 3 seconds.")
            camera_stop_event.wait(3.0)


def start_camera_worker():
    global camera_worker_thread
    if camera_worker_thread and camera_worker_thread.is_alive():
        return
    camera_stop_event.clear()
    camera_worker_thread = threading.Thread(
        target=camera_worker_supervisor,
        name="smart-aroma-camera-supervisor",
        daemon=True,
    )
    camera_worker_thread.start()



# =========================================================
# BACKGROUND CAMERA PREVIEW, LIVE LOGIN VERIFY, CONSENT, RECORDINGS
# =========================================================


@app.route("/enrollment-feed")
@login_required
def enrollment_feed():
    if user_face_enrolled(session["user_id"]):
        return redirect(url_for("face_verify"))
    return Response(
        enrollment_preview_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.route("/api/camera/snapshot")
@login_required
def camera_snapshot():
    if user_face_enrolled(session["user_id"]):
        return jsonify({"error": "Face is already enrolled; use live verification."}), 409
    after_seq = request.args.get("after_seq", "-1")
    try:
        after_seq = int(after_seq)
    except ValueError:
        after_seq = -1
    frame, seq = get_fresh_camera_snapshot(after_seq, timeout=3.0)
    if frame is None:
        return jsonify({"error": "Camera frame is not available. Check that the webcam is connected."}), 503
    feature, bbox, face_error = _sface_extract(frame)
    if face_error:
        return jsonify({"error": face_error}), 400
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return jsonify({"error": "Could not encode the camera frame."}), 500
    image_data = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
    return jsonify({"sequence": seq, "image": image_data})


@app.route("/api/face/live-status")
@login_required
def face_live_status():
    user_id = int(session["user_id"])
    state = get_user_state(user_id)
    is_live_match = (
        active_camera_user_id == user_id
        and state.get("identity_verified")
        and time.time() - state.get("last_identity_check", 0.0) <= IDENTITY_FRESHNESS_SECONDS
    )
    return jsonify({
        "recognized": bool(is_live_match),
        "message": state.get("identity_message") if is_live_match else "Waiting for the enrolled face to be seen by the always-on camera.",
        "monitoring_enabled": get_monitoring_enabled(user_id),
    })


@app.route("/api/face/confirm", methods=["POST"])
@login_required
def confirm_live_face_login():
    global last_monitoring_user_id
    user_id = int(session["user_id"])
    state = get_user_state(user_id)
    is_live_match = (
        active_camera_user_id == user_id
        and state.get("identity_verified")
        and time.time() - state.get("last_identity_check", 0.0) <= IDENTITY_FRESHNESS_SECONDS
    )
    if not is_live_match:
        session["face_verified"] = False
        return jsonify({"ok": False, "message": "Registered face is not currently recognized. Stand in front of the camera and try again."}), 403

    payload = request.get_json(silent=True) or {}
    enabled = bool(payload.get("monitoringEnabled", False))
    connection = get_db_connection()
    try:
        connection.execute(
            "UPDATE users SET monitoring_enabled = ? WHERE id = ?",
            (1 if enabled else 0, user_id),
        )
        connection.commit()
    finally:
        connection.close()

    session["face_verified"] = True
    state["monitoring_enabled"] = enabled
    state["session_active"] = True
    if enabled:
        last_monitoring_user_id = user_id
        settings = get_timer_settings(user_id)
        apply_timer_settings_to_state(user_id, settings, reset_timer=True)
        return jsonify({"ok": True, "message": "Face matched. Monitoring enabled; the first emotion sample will run when the selected timer expires.", "next": url_for("index")})

    _close_user_recording_if_any(user_id)
    state["spray_status"] = "Monitoring is paused by user choice. No new recordings or spray commands will be issued."
    state["current_emotion"] = "Monitoring paused"
    state["current_aroma"] = "Not activated"
    state["current_suggestion"] = "Face verified. Monitoring is paused until you enable it again."
    return jsonify({"ok": True, "message": "Face matched. Monitoring remains paused.", "next": url_for("index")})


def _close_user_recording_if_any(user_id):
    # The always-on worker owns the writer; this flag asks it to close the current clip on next frame.
    get_user_state(user_id)["monitoring_enabled"] = False


@app.route("/api/monitoring/toggle", methods=["POST"])
@login_required
def monitoring_toggle():
    global last_monitoring_user_id
    user_id = int(session["user_id"])
    payload = request.get_json(silent=True) or {}
    enabled = bool(payload.get("enabled", False))
    if enabled:
        state = get_user_state(user_id)
        if not session.get("face_verified") or active_camera_user_id != user_id or not state.get("identity_verified") or time.time() - state.get("last_identity_check", 0.0) > IDENTITY_FRESHNESS_SECONDS:
            return jsonify({"ok": False, "message": "To enable monitoring, login and verify your live registered face first."}), 403
    connection = get_db_connection()
    try:
        connection.execute("UPDATE users SET monitoring_enabled = ? WHERE id = ?", (1 if enabled else 0, user_id))
        connection.commit()
    finally:
        connection.close()
    state = get_user_state(user_id)
    state["monitoring_enabled"] = enabled
    if not enabled:
        state["prediction_history"].clear()
        state["spray_status"] = "Monitoring paused by user. No recording or new spray commands allowed."
        state["current_emotion"] = "Monitoring paused"
        state["current_aroma"] = "Not activated"
        state["current_suggestion"] = "Camera recognition may continue in memory, but emotion processing, recording and spray are paused."
    else:
        last_monitoring_user_id = user_id
        settings = get_timer_settings(user_id)
        apply_timer_settings_to_state(user_id, settings, reset_timer=True)
        state["current_suggestion"] = "Monitoring enabled. Emotion sampling waits for the configured timer and requires the registered face."
    return jsonify({"ok": True, "enabled": enabled, "message": "Monitoring enabled." if enabled else "Monitoring paused. The recording will close on the next camera frame."})


@app.route("/api/notifications/fcm-token", methods=["POST"])
@login_required
def register_browser_push_token():
    """Save the browser's Firebase Cloud Messaging token for the signed-in user."""
    payload = request.get_json(silent=True) or {}
    token = str(payload.get("token", "")).strip()
    if not token or len(token) > 4096:
        return jsonify({"ok": False, "error": "A valid browser push token is required."}), 400

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    connection = get_db_connection()
    try:
        connection.execute(
            """INSERT INTO browser_push_tokens (user_id, token, created_at, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(token) DO UPDATE SET
                 user_id=excluded.user_id, updated_at=excluded.updated_at""",
            (int(session["user_id"]), token, now, now),
        )
        connection.commit()
        return jsonify({
            "ok": True,
            "message": "Browser token registered. Firebase Admin credentials are required for server-initiated push delivery."
        }), 200
    except sqlite3.Error:
        app.logger.exception("Could not save browser push token")
        return jsonify({"ok": False, "error": "Could not save browser push token."}), 500
    finally:
        connection.close()


@app.route("/api/notifications")
@face_verified_required
def get_notifications():
    user_id = int(session["user_id"])
    connection = get_db_connection()
    try:
        rows = connection.execute(
            "SELECT id, notification_type, title, message, created_at, is_read FROM notifications WHERE user_id = ? ORDER BY id DESC LIMIT 50",
            (user_id,),
        ).fetchall()
        unread = connection.execute(
            "SELECT COUNT(*) AS total FROM notifications WHERE user_id = ? AND is_read = 0", (user_id,)
        ).fetchone()["total"]
        return jsonify({
            "unread": int(unread),
            "notifications": [dict(row) for row in rows],
        })
    finally:
        connection.close()


@app.route("/api/notifications/<int:notification_id>/read", methods=["POST"])
@face_verified_required
def mark_notification_read(notification_id):
    connection = get_db_connection()
    try:
        connection.execute(
            "UPDATE notifications SET is_read = 1 WHERE id = ? AND user_id = ?",
            (notification_id, int(session["user_id"])),
        )
        connection.commit()
        return jsonify({"ok": True})
    finally:
        connection.close()


@app.route("/api/notifications/test-push", methods=["POST"])
@face_verified_required
def test_browser_push():
    user_id = int(session["user_id"])
    connection = get_db_connection()

    try:
        rows = connection.execute(
            "SELECT token FROM browser_push_tokens WHERE user_id = ?",
            (user_id,),
        ).fetchall()

        if not rows:
            return jsonify({
                "ok": False,
                "error": "No browser push token found. Please enable browser notifications first."
            }), 404

        sent = 0
        errors = []

        for row in rows:
            token = row["token"]

            try:
                message = messaging.Message(
                    notification=messaging.Notification(
                        title="Smart AromaSense",
                        body="Test notification received successfully!"
                    ),
                    token=token,
                    webpush=messaging.WebpushConfig(
                        notification=messaging.WebpushNotification(
                            title="Smart AromaSense",
                            body="Test notification received successfully!"
                        )
                    )
                )

                messaging.send(message)
                sent += 1

            except Exception:
                app.logger.exception("Failed to send browser push notification")
                errors.append("A notification could not be delivered.")

        if sent:
            return jsonify({
                "ok": True,
                "message": f"Push notification sent successfully to {sent} browser(s)."
            }), 200

        return jsonify({
            "ok": False,
            "error": errors[0] if errors else "Push notification failed."
        }), 500

    finally:
        connection.close()


@app.route("/api/recordings")
@face_verified_required
def list_recordings():
    connection = get_db_connection()
    try:
        rows = connection.execute(
            "SELECT id, file_name, started_at, ended_at, duration_seconds, frame_count FROM video_recordings WHERE user_id = ? ORDER BY id DESC LIMIT 100",
            (int(session["user_id"]),),
        ).fetchall()
        return jsonify([{
            "id": row["id"], "file_name": row["file_name"],
            "started_at": row["started_at"], "ended_at": row["ended_at"],
            "duration_seconds": row["duration_seconds"], "frame_count": row["frame_count"],
            "url": url_for("recording_file", recording_id=row["id"]),
        } for row in rows])
    finally:
        connection.close()


@app.route("/recordings/<int:recording_id>")
@face_verified_required
def recording_file(recording_id):
    connection = get_db_connection()
    try:
        row = connection.execute(
            "SELECT file_path, file_name FROM video_recordings WHERE id = ? AND user_id = ?",
            (recording_id, int(session["user_id"])),
        ).fetchone()
    finally:
        connection.close()
    if not row:
        return jsonify({"error": "Recording not found"}), 404
    path = (BASE_DIR / row["file_path"]).resolve()
    try:
        path.relative_to(RECORDINGS_DIR.resolve())
    except ValueError:
        return jsonify({"error": "Invalid recording path"}), 400
    if not path.is_file():
        return jsonify({"error": "Video file is no longer available"}), 404
    return send_file(path, mimetype="video/mp4" if path.suffix.lower() == ".mp4" else "video/x-msvideo", conditional=True)


@app.route("/api/recordings/<int:recording_id>", methods=["DELETE"])
@face_verified_required
def delete_recording(recording_id):
    """Delete the signed-in user's recording and its database metadata."""
    user_id = int(session["user_id"])
    connection = get_db_connection()

    try:
        row = connection.execute(
            """
            SELECT file_path FROM video_recordings
            WHERE id = ? AND user_id = ?
            """,
            (recording_id, user_id),
        ).fetchone()

        if row is None:
            return jsonify({"ok": False, "message": "Recording not found."}), 404

        path = (BASE_DIR / row["file_path"]).resolve()

        try:
            path.relative_to(RECORDINGS_DIR.resolve())
        except ValueError:
            return jsonify({"ok": False, "message": "Invalid recording path."}), 400

        # Delete the file first. Keep its database row if deletion fails.
        try:
            delete_recording_file(path)
        except OSError as exc:
            app.logger.exception(
                "Recording deletion failed: id=%s, path=%s",
                recording_id,
                path,
            )
            return jsonify({
                "ok": False,
                "message": (
                    f"Could not delete video: {exc}. "
                    "Close the video player and try again."
                ),
            }), 500

        connection.execute(
            "DELETE FROM video_recordings WHERE id = ? AND user_id = ?",
            (recording_id, user_id),
        )
        connection.commit()

        try:
            path.parent.rmdir()  # Remove the user folder only if it is empty.
        except OSError:
            pass

        return jsonify({"ok": True, "message": "Recording deleted."})

    finally:
        connection.close()


def delete_recording_file(path, retries=5, delay=0.5):
    """Delete a recording, retrying briefly if Windows temporarily locks it."""
    for attempt in range(retries):
        try:
            if path.exists():
                os.chmod(path, 0o666)
                path.unlink()
            return True
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
    return False

# =========================================================
# DASHBOARD AND API ROUTES (ALL REQUIRE VERIFIED FACE)
# =========================================================


@app.route("/")
@face_verified_required
def index():
    return render_template("index.html")


@app.route("/video_feed")
@face_verified_required
def video_feed():
    return Response(
        generate_frames(int(session["user_id"])),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/timer-settings", methods=["GET", "POST"])
@face_verified_required
def timer_settings_api():
    global last_monitoring_user_id
    user_id = int(session["user_id"])
    if request.method == "GET":
        settings = get_timer_settings(user_id)
        state = get_user_state(user_id)
        apply_timer_settings_to_state(user_id, settings)
        if settings.get("recording_timer_enabled") and get_monitoring_enabled(user_id):
            last_monitoring_user_id = user_id
        next_at = state.get("next_recommendation_time", float("inf"))
        remaining = max(0, int(next_at - time.time())) if next_at != float("inf") else None
        recording_at = state.get("next_recording_time", float("inf"))
        recording_remaining = max(0, int(recording_at - time.time())) if recording_at != float("inf") else None
        return jsonify({**settings, "monitoring_enabled": get_monitoring_enabled(user_id), "seconds_until_next_cycle": remaining, "seconds_until_next_recording": recording_remaining, "sample_count": SAMPLE_COUNT, "hardware_enabled": AROMA_HARDWARE_ENABLED, "spray_status": state.get("spray_status", "Waiting"), "recording_status": state.get("recording_status", "Recording is not started yet."), "last_saved_clip": state.get("last_saved_clip"), "last_saved_clip_at": state.get("last_saved_clip_at")})

    payload = request.get_json(silent=True) or {}
    current_settings = get_timer_settings(user_id)
    try:
        interval = int(payload.get("interval_seconds", current_settings["interval_seconds"]))
        duration = int(payload.get("spray_duration_seconds", current_settings["spray_duration_seconds"]))
        recording_interval = int(payload.get("recording_interval_seconds", current_settings["recording_interval_seconds"]))
        recording_duration = int(payload.get("recording_duration_seconds", current_settings["recording_duration_seconds"]))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "message": "Timer values must be numbers."}), 400
    timer_enabled = bool(payload.get("timer_enabled", current_settings["timer_enabled"]))
    fallback_enabled = bool(payload.get("fallback_enabled", current_settings["fallback_enabled"]))
    fallback_unknown_enabled = bool(payload.get("fallback_unknown_enabled", current_settings.get("fallback_unknown_enabled", False)))
    fallback_multiple_enabled = bool(payload.get("fallback_multiple_enabled", current_settings.get("fallback_multiple_enabled", False)))
    fallback_camera_error_enabled = bool(payload.get("fallback_camera_error_enabled", current_settings.get("fallback_camera_error_enabled", False)))
    fallback_aroma = payload.get("fallback_aroma", current_settings["fallback_aroma"])
    recording_timer_enabled = bool(payload.get("recording_timer_enabled", current_settings["recording_timer_enabled"]))
    # Legacy database/API field only; clips are split based on duration and do not wait for an interval.
    recording_interval = recording_duration
    if interval < 5 or interval > MAX_RECOMMENDATION_INTERVAL_SECONDS:
        return jsonify({"ok": False, "message": "Choose an emotion sampling interval from 5 seconds up to 7 days."}), 400
    if duration < 1 or duration > MAX_SPRAY_DURATION_SECONDS:
        return jsonify({"ok": False, "message": f"Spray duration must be between 1 and {MAX_SPRAY_DURATION_SECONDS} seconds."}), 400
    if recording_duration < 1 or recording_duration > MAX_SCHEDULED_RECORDING_DURATION_SECONDS:
        return jsonify({"ok": False, "message": "Clip length must be between 1 second and 60 minutes."}), 400
    if fallback_aroma not in ("Citrus", "Rosemary"):
        return jsonify({"ok": False, "message": "No-face fallback can only be Citrus (Happy) or Rosemary (Neutral)."}), 400

    settings = {
        "interval_seconds": interval,
        "timer_enabled": timer_enabled,
        "spray_duration_seconds": duration,
        "fallback_enabled": fallback_enabled,
        "fallback_unknown_enabled": fallback_unknown_enabled,
        "fallback_multiple_enabled": fallback_multiple_enabled,
        "fallback_camera_error_enabled": fallback_camera_error_enabled,
        "fallback_aroma": fallback_aroma,
        "recording_timer_enabled": recording_timer_enabled,
        "recording_interval_seconds": recording_duration,  # legacy compatibility only
        "recording_duration_seconds": recording_duration,
    }
    connection = get_db_connection()
    try:
        connection.execute(
            "INSERT INTO timer_settings (user_id, interval_seconds, timer_enabled, spray_duration_seconds, "
            "fallback_enabled, fallback_unknown_enabled, fallback_multiple_enabled, fallback_camera_error_enabled, "
            "fallback_aroma, recording_timer_enabled, recording_interval_seconds, recording_duration_seconds, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET interval_seconds=excluded.interval_seconds, timer_enabled=excluded.timer_enabled, "
            "spray_duration_seconds=excluded.spray_duration_seconds, fallback_enabled=excluded.fallback_enabled, "
            "fallback_unknown_enabled=excluded.fallback_unknown_enabled, fallback_multiple_enabled=excluded.fallback_multiple_enabled, "
            "fallback_camera_error_enabled=excluded.fallback_camera_error_enabled, fallback_aroma=excluded.fallback_aroma, "
            "recording_timer_enabled=excluded.recording_timer_enabled, recording_interval_seconds=excluded.recording_interval_seconds, "
            "recording_duration_seconds=excluded.recording_duration_seconds, updated_at=excluded.updated_at",
            (user_id, interval, 1 if timer_enabled else 0, duration, 1 if fallback_enabled else 0,
             1 if fallback_unknown_enabled else 0, 1 if fallback_multiple_enabled else 0,
             1 if fallback_camera_error_enabled else 0, fallback_aroma,
             1 if recording_timer_enabled else 0, recording_interval, recording_duration,
             datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        connection.commit()
    finally:
        connection.close()

    state = apply_timer_settings_to_state(user_id, settings, reset_timer=True)
    if recording_timer_enabled and get_monitoring_enabled(user_id):
        last_monitoring_user_id = user_id
        with recording_rotation_lock:
            # If a clip is active, finalize it on the worker's next frame so the new setting
            # applies immediately and the previous clip is committed to SQLite now.
            recording_rotation_requests.add(user_id)
        state["recording_status"] = f"Continuous recording enabled; clip rotation is set to every {recording_duration}s."
    elif recording_timer_enabled:
        state["recording_status"] = "Recording is enabled in settings, but monitoring consent is off; no video is being recorded."
    else:
        state["recording_status"] = "Continuous video recording is off."
    if not timer_enabled:
        state["current_suggestion"] = "Emotion timer is off. No scheduled emotion sampling or spray will occur."
        state["spray_status"] = "Timer off; automatic spray is paused."
    else:
        state["current_suggestion"] = "Timer saved. The next 10-prediction sampling cycle starts after the selected interval."
        state["spray_status"] = "Waiting for next scheduled emotion sampling cycle."
    if recording_timer_enabled and get_monitoring_enabled(user_id):
        state["recording_status"] = f"Continuous recording enabled; clip rotation is set to every {recording_duration}s."
    elif recording_timer_enabled:
        state["recording_status"] = "Recording is configured, but monitoring consent is off; no video is being recorded."
    else:
        state["recording_status"] = "Continuous video recording is off."
    return jsonify({"ok": True, "message": "Timer settings saved. Emotion sampling uses its own timer; continuous video clips rotate by clip duration.", **settings})




@app.route("/api/profile", methods=["GET", "PATCH"])
@login_required
def api_profile():
    """Read/update the signed-in user's own profile and account activity."""
    user_id = int(session["user_id"])
    connection = get_db_connection()
    try:
        if request.method == "PATCH":
            payload = request.get_json(silent=True) or {}
            full_name = str(payload.get("full_name", "")).strip()
            email = str(payload.get("email", "")).strip().lower()
            phone = str(payload.get("phone", "")).strip()
            department = str(payload.get("department", "")).strip()
            study_year = str(payload.get("study_year", "")).strip()
            location = str(payload.get("location", "")).strip()
            if not full_name:
                return jsonify({"ok": False, "message": "Full name is required."}), 400
            if not email or not re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", email):
                return jsonify({"ok": False, "message": "Enter a valid email address."}), 400
            duplicate = connection.execute(
                "SELECT id FROM users WHERE lower(email)=? AND id!=?", (email, user_id)
            ).fetchone()
            if duplicate:
                return jsonify({"ok": False, "message": "That email address is already in use."}), 409
            connection.execute(
                """UPDATE users SET full_name=?, email=?, phone=?, department=?, study_year=?, location=?
                   WHERE id=?""",
                (full_name, email, phone, department, study_year, location, user_id),
            )
            connection.commit()

        user = connection.execute(
            """SELECT id, full_name, email, phone, department, study_year, location
               FROM users WHERE id=?""", (user_id,)
        ).fetchone()
        if not user:
            return jsonify({"ok": False, "message": "Account not found."}), 404
        emotion_count = connection.execute(
            "SELECT COUNT(*) FROM emotion_history WHERE user_id=?", (user_id,)
        ).fetchone()[0]
        reflection_count = connection.execute(
            "SELECT COUNT(*) FROM profile_reflections WHERE user_id=?", (user_id,)
        ).fetchone()[0]
        reflections = connection.execute(
            """SELECT id, focus, thought, created_at FROM profile_reflections
               WHERE user_id=? ORDER BY datetime(created_at) DESC, id DESC LIMIT 3""", (user_id,)
        ).fetchall()
        return jsonify({
            "ok": True,
            "profile": {
                "id": int(user["id"]), "full_name": user["full_name"] or "",
                "email": user["email"] or "", "phone": user["phone"] or "",
                "department": user["department"] or "", "study_year": user["study_year"] or "",
                "location": user["location"] or "", "emotion_count": int(emotion_count or 0),
                "reflection_count": int(reflection_count or 0),
                "reflections": [
                    {"id": int(r["id"]), "focus": r["focus"], "thought": r["thought"], "created_at": r["created_at"]}
                    for r in reflections
                ],
            }
        })
    finally:
        connection.close()


@app.route("/api/profile/reflections", methods=["POST"])
@login_required
def api_profile_reflections():
    payload = request.get_json(silent=True) or {}
    focus = str(payload.get("focus", "")).strip()[:120]
    thought = str(payload.get("thought", "")).strip()[:3000]
    if not thought:
        return jsonify({"ok": False, "message": "Reflection text is required."}), 400
    connection = get_db_connection()
    try:
        connection.execute(
            "INSERT INTO profile_reflections (user_id, focus, thought, created_at) VALUES (?, ?, ?, ?)",
            (int(session["user_id"]), focus, thought, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        connection.commit()
        return jsonify({"ok": True, "message": "Reflection saved to your account."}), 201
    finally:
        connection.close()


@app.route("/api/current")
@face_verified_required
def current():
    state = get_user_state(session["user_id"])
    emotion = state["current_emotion"]
    wellness_details = WELLNESS_DETAILS.get(emotion, DEFAULT_WELLNESS_DETAILS).copy()
    suggested_aroma = recommendations.get(emotion, {}).get("aroma", state["current_aroma"])
    return jsonify(
        {
            "emotion": emotion,
            "aroma": state["current_aroma"],
            "suggestion": state["current_suggestion"],
            "identity_verified": bool(state.get("identity_verified")),
            "identity_status": state.get("identity_message", "Unknown"),
            "spray_status": state.get("spray_status", "Unknown"),
            "monitoring_enabled": get_monitoring_enabled(session["user_id"]),
            "camera_status": latest_camera_status,
            "timer_enabled": bool(state.get("timer_enabled", True)),
            "timer_interval_seconds": int(state.get("timer_interval_seconds", DEFAULT_RECOMMENDATION_INTERVAL_SECONDS)),
            "recording_timer_enabled": bool(state.get("recording_timer_enabled", False)),
            "recording_interval_seconds": int(state.get("recording_interval_seconds", DEFAULT_RECORDING_INTERVAL_SECONDS)),
            "recording_duration_seconds": int(state.get("recording_duration_seconds", DEFAULT_RECORDING_DURATION_SECONDS)),
            "seconds_until_next_cycle": (max(0, int(state.get("next_recommendation_time", time.time()) - time.time())) if state.get("next_recommendation_time", float("inf")) != float("inf") else None),
            "spray_duration_seconds": int(state.get("spray_duration_seconds", DEFAULT_SPRAY_DURATION_SECONDS)),
            "fallback_enabled": bool(state.get("fallback_enabled", False)),
            "fallback_aroma": state.get("fallback_aroma", "Citrus"),
            "wellness": {
                **wellness_details,
                "aroma": suggested_aroma,
                "emotion": emotion,
            },
        }
    )


def analytics_start_time(period):
    now = datetime.now()
    if period == "today":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "week":
        start = now - timedelta(days=now.weekday())
        return start.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "month":
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return None


def query_user_emotion_counts(user_id, period="all"):
    start_time = analytics_start_time(period)
    connection = get_db_connection()
    try:
        query = "SELECT emotion, COUNT(*) AS amount FROM emotion_history WHERE user_id = ?"
        params = [int(user_id)]
        if start_time is not None:
            query += " AND detected_at >= ?"
            params.append(start_time.strftime("%Y-%m-%d %H:%M:%S"))
        query += " GROUP BY emotion"
        rows = connection.execute(query, params).fetchall()

        total_query = "SELECT COUNT(*) AS amount FROM emotion_history WHERE user_id = ?"
        total_params = [int(user_id)]
        if start_time is not None:
            total_query += " AND detected_at >= ?"
            total_params.append(start_time.strftime("%Y-%m-%d %H:%M:%S"))
        total = connection.execute(total_query, total_params).fetchone()["amount"]

        return total, {row["emotion"]: row["amount"] for row in rows}
    finally:
        connection.close()


@app.route("/api/analytics")
@face_verified_required
def analytics():
    period = request.args.get("period", "all")
    if period not in {"today", "week", "month", "all"}:
        return jsonify({"error": "Invalid period"}), 400
    total, emotion_counts = query_user_emotion_counts(session["user_id"], period)
    return jsonify(
        {"period": period, "total": total, "emotion_counts": emotion_counts}
    )


@app.route("/api/analytics/overall")
@face_verified_required
def overall_analytics():
    total, emotion_counts = query_user_emotion_counts(session["user_id"], "all")
    return jsonify({"total": total, "emotion_counts": emotion_counts})


@app.route("/api/daily-summary")
@face_verified_required
def daily_summary():
    """Return one authenticated user's detailed summary for a selected local date."""
    date_text = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    try:
        report_day = datetime.strptime(date_text, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Date must use YYYY-MM-DD format."}), 400

    start_text = report_day.strftime("%Y-%m-%d") + " 00:00:00"
    end_text = (report_day + timedelta(days=1)).strftime("%Y-%m-%d") + " 00:00:00"
    connection = get_db_connection()
    try:
        emotion_rows = connection.execute(
            """SELECT emotion, COUNT(*) AS amount
               FROM emotion_history
               WHERE user_id = ? AND detected_at >= ? AND detected_at < ?
               GROUP BY emotion ORDER BY amount DESC""",
            (int(session["user_id"]), start_text, end_text),
        ).fetchall()
        aroma_rows = connection.execute(
            """SELECT aroma, COUNT(*) AS amount
               FROM emotion_history
               WHERE user_id = ? AND detected_at >= ? AND detected_at < ?
               GROUP BY aroma ORDER BY amount DESC""",
            (int(session["user_id"]), start_text, end_text),
        ).fetchall()
        latest = connection.execute(
            """SELECT emotion, aroma, suggestion, detected_at
               FROM emotion_history
               WHERE user_id = ? AND detected_at >= ? AND detected_at < ?
               ORDER BY detected_at DESC, id DESC LIMIT 1""",
            (int(session["user_id"]), start_text, end_text),
        ).fetchone()
    finally:
        connection.close()

    emotion_counts = {row["emotion"]: int(row["amount"]) for row in emotion_rows}
    aroma_counts = {row["aroma"]: int(row["amount"]) for row in aroma_rows}
    total = sum(emotion_counts.values())
    top_emotion = emotion_rows[0]["emotion"] if emotion_rows else "—"
    top_aroma = aroma_rows[0]["aroma"] if aroma_rows else "—"
    return jsonify({
        "date": report_day.strftime("%Y-%m-%d"),
        "total": total,
        "emotion_counts": emotion_counts,
        "aroma_counts": aroma_counts,
        "top_emotion": top_emotion,
        "top_aroma": top_aroma,
        "latest": dict(latest) if latest else None,
        "note": ("These are counts of confirmed model recommendations, not a diagnosis." if total else "No confirmed emotion records were saved for this date. That is okay—rest counts too 💜"),
    })


@app.route("/api/history")
@face_verified_required
def history():
    connection = get_db_connection()
    try:
        rows = connection.execute(
            """
            SELECT emotion, aroma, suggestion, detected_at
            FROM emotion_history
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 20
            """,
            (int(session["user_id"]),),
        ).fetchall()
        return jsonify(
            [
                {
                    "emotion": row["emotion"],
                    "aroma": row["aroma"],
                    "suggestion": row["suggestion"],
                    "detected_at": row["detected_at"],
                }
                for row in rows
            ]
        )
    finally:
        connection.close()


if __name__ == "__main__":
    print("\n======================================")
    print(" Smart Emotion-Based Aroma System")
    print(" Flask + CNN + Live Face Gate + optional Arduino relay control")
    print("======================================")
    print(f"Database: {DB_PATH}")
    print(f"Face samples/model folder: {FACE_DIR}")
    print("Default recommendation interval: 3600 seconds (1 hour); adjust per account in Settings.")
    print(f"Fresh predictions per cycle: {SAMPLE_COUNT}")
    print(f"Live face matches required: {LIVE_MATCH_FRAMES_REQUIRED}")
    print(f"Arduino hardware enabled: {AROMA_HARDWARE_ENABLED}")
    print("Continuous recording clips: configurable 1 second to 60 minutes per file")
    print(f"Recording retention: {RECORDING_RETENTION_DAYS} day(s)")
    print(f"Camera processing target: {CAMERA_TARGET_FPS} FPS")
    print(f"Daily report notification time: {DAILY_REPORT_HOUR:02d}:00 local time")
    print("Camera service runs while this Python process is running; browser may be closed.")
    print("Dashboard: http://127.0.0.1:8080\n")
    start_camera_worker()
    try:
        app.run(host="127.0.0.1", port=8080, debug=False, threaded=True)
    finally:
        camera_stop_event.set()
        with camera_condition:
            camera_condition.notify_all()
        if camera_worker_thread and camera_worker_thread.is_alive():
            camera_worker_thread.join(timeout=5)

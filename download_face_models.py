"""Download the official OpenCV Zoo YuNet + SFace ONNX models.
Run once from the project root: python download_face_models.py
"""
from pathlib import Path
from urllib.request import Request, urlopen
import hashlib
import os
import sys

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "face_data"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODELS = [
    {
        "name": "face_detection_yunet_2023mar.onnx",
        "url": "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "min_bytes": 200_000,
    },
    {
        "name": "face_recognition_sface_2021dec.onnx",
        "url": "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        "min_bytes": 30_000_000,
    },
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(model):
    destination = MODEL_DIR / model["name"]
    if destination.exists() and destination.stat().st_size >= model["min_bytes"]:
        if sha256(destination).lower() == model["sha256"]:
            print(f"Already verified: {destination}")
            return
    partial = destination.with_suffix(destination.suffix + ".part")
    request = Request(model["url"], headers={"User-Agent": "SmartAromaDemo/1.0"})
    print(f"Downloading {model['name']} — may take a while...")
    try:
        with urlopen(request, timeout=60) as response, partial.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        if partial.stat().st_size < model["min_bytes"]:
            raise RuntimeError(f"Download is too small: {partial.stat().st_size} bytes")
        actual = sha256(partial).lower()
        if actual != model["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {model['name']}: {actual}")
        os.replace(partial, destination)
        print(f"Saved and verified: {destination} ({destination.stat().st_size:,} bytes)")
    except Exception:
        partial.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    try:
        for model in MODELS:
            download(model)
        print("\nBoth face models are ready. Start the application with: python app.py")
    except Exception as exc:
        print(f"\nModel download failed: {exc}", file=sys.stderr)
        print("Check your internet connection and rerun: python download_face_models.py", file=sys.stderr)
        sys.exit(1)

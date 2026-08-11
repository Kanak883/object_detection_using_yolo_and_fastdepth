from pathlib import Path
import os
import platform
import subprocess
import sys
import urllib.request

# Run this file from the conda environment you want to use.
# The project root is always the folder containing this script, so it can live on F:.

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "Weights"
MODELS_DIR = ROOT / "models"
YOLO_MODEL = ROOT / "yolov8n.pt"
FASTDEPTH_WEIGHTS = WEIGHTS_DIR / "FastDepthV2_L1GN_Best.pth"
REQ_FILE = ROOT / "requirements.txt"

REQUIREMENTS = """ultralytics
opencv-python
torch
torchvision
numpy
Pillow
"""

def run(cmd):
    print("\n>", " ".join(map(str, cmd)))
    subprocess.check_call(cmd)

def check_python():
    major, minor = sys.version_info[:2]
    print(f"Python: {sys.version.split()[0]}")
    if major != 3 or minor not in (10, 11, 12, 13):
        print(
            "WARNING: This project is intended for Python 3.10-3.13. "
            "A different version may cause PyTorch/torchvision issues."
        )

def check_conda():
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        print(f"Conda environment: {conda_prefix}")
    else:
        print(
            "WARNING: No active conda environment detected. "
            "Activate your conda environment first."
        )

def install_dependencies():
    REQ_FILE.write_text(REQUIREMENTS, encoding="utf-8")
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "-r", str(REQ_FILE)])

def prepare_folders():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Project root : {ROOT}")
    print(f"Weights dir  : {WEIGHTS_DIR}")
    print(f"Models dir   : {MODELS_DIR}")

def verify_fastdepth_source():
    candidates = [
        MODELS_DIR / "models.py",
        ROOT / "FastDepth" / "models" / "models.py",
    ]
    found = next((p for p in candidates if p.exists()), None)
    if found:
        print(f"FastDepth source found: {found}")
        return True

    print("\nERROR: FastDepthV2 source code was not found.")
    print("Expected one of:")
    for p in candidates:
        print(f"  {p}")
    print(
        "\nThe uploaded main.py imports `models.models.FastDepthV2`, so "
        "the FastDepth implementation must be copied into this project."
    )
    return False

def verify_fastdepth_weights():
    if FASTDEPTH_WEIGHTS.exists():
        print(f"FastDepth weights found: {FASTDEPTH_WEIGHTS}")
        return True

    print("\nWARNING: FastDepth weights are missing.")
    print(f"Put the checkpoint here:\n  {FASTDEPTH_WEIGHTS}")
    print("The checkpoint expected by main_fdrive.py is:")
    print("  FastDepthV2_L1GN_Best.pth")
    return False

def download_yolo():
    if YOLO_MODEL.exists():
        print(f"YOLO model already exists: {YOLO_MODEL}")
        return

    print("\nDownloading yolov8n.pt through Ultralytics...")
    code = (
        "from ultralytics import YOLO; "
        f"YOLO(r'{YOLO_MODEL}')"
    )
    run([sys.executable, "-c", code])

def verify_imports():
    test = (
        "import cv2, torch, torchvision, numpy, PIL; "
        "from ultralytics import YOLO; "
        "print('All Python dependencies imported successfully.'); "
        "print('CUDA available:', torch.cuda.is_available())"
    )
    run([sys.executable, "-c", test])

def main():
    print("=" * 68)
    print("FastDepth + YOLO | F: drive project setup")
    print("=" * 68)

    check_python()
    check_conda()
    prepare_folders()
    install_dependencies()
    verify_imports()
    download_yolo()

    source_ok = verify_fastdepth_source()
    weights_ok = verify_fastdepth_weights()

    print("\n" + "=" * 68)
    print("SETUP COMPLETE")
    print("=" * 68)
    print(f"Project: {ROOT}")
    print(f"Run:     python main_fdrive.py")

    if not source_ok or not weights_ok:
        print(
            "\nThe Python packages are installed, but FastDepthV2 itself "
            "is not fully available yet. Add the missing source/weights above."
        )
    else:
        print("\nFastDepthV2 source and weights are present.")

if __name__ == "__main__":
    main()

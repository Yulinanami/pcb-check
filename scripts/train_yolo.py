"""Train a YOLO baseline for PCB defect detection."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = project_root()
DATA_YAML = ROOT / "outputs" / "pcb_yolo_dataset" / "dataset.yaml"
MODEL = "yolov8n.pt"
IMAGE_SIZE = 1024
EPOCHS = 50
BATCH_SIZE = 4
DEVICE = "0"
PROJECT_DIR = ROOT / "runs" / "train"
RUN_NAME = "pcb_yolo_baseline"
WORKERS = 0


def main() -> None:
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {DATA_YAML}\nRun: python scripts/prepare_dataset.py")

    from ultralytics import YOLO

    model = YOLO(MODEL)
    model.train(
        data=str(DATA_YAML),
        imgsz=IMAGE_SIZE,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()

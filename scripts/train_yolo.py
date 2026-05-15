"""Train a YOLO baseline for PCB defect detection."""

from __future__ import annotations

import argparse
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Train Ultralytics YOLO on prepared PCB data.")
    parser.add_argument("--data", type=Path, default=root / "outputs" / "pcb_yolo_dataset" / "dataset.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=Path, default=root / "runs" / "train")
    parser.add_argument("--name", type=str, default="pcb_yolo_baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {args.data}\nRun: python scripts/prepare_dataset.py")

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()

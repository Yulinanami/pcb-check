"""Visualize PCB test-set GT boxes and YOLO predictions on the same image."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


CLASS_NAMES = ["Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]
SHORT_LABELS = ["M", "O", "Sh", "Sp", "Sc"]
COLORS = [
    (0, 255, 255),
    (255, 128, 0),
    (0, 255, 0),
    (255, 0, 255),
    (0, 128, 255),
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = project_root()
WEIGHTS = ROOT / "runs" / "train" / "pcb_yolo_baseline" / "weights" / "best.pt"
DATA_YAML = ROOT / "outputs" / "pcb_yolo_dataset" / "dataset.yaml"
OUTPUT_DIR = ROOT / "outputs" / "visualizations"
MAX_IMAGES = 10
IMAGE_SIZE = 1024
CONFIDENCE = 0.25
NMS_IOU = 0.7
DEVICE = "0"


def load_dataset_paths(data_yaml: Path) -> tuple[Path, Path]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(data["path"])
    return dataset_root / data.get("test", "images/test"), dataset_root / "labels" / "test"


def yolo_to_xyxy(line: str, width: int, height: int) -> tuple[int, tuple[int, int, int, int]]:
    parts = line.split()
    class_id = int(float(parts[0]))
    x_center, y_center, box_width, box_height = map(float, parts[1:5])
    xmin = int(round((x_center - box_width / 2.0) * width))
    ymin = int(round((y_center - box_height / 2.0) * height))
    xmax = int(round((x_center + box_width / 2.0) * width))
    ymax = int(round((y_center + box_height / 2.0) * height))
    return class_id, (xmin, ymin, xmax, ymax)


def draw_label(image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    y_text = max(text_height + baseline + 2, y)
    cv2.rectangle(
        image,
        (x, y_text - text_height - baseline - 2),
        (x + text_width + 4, y_text + 2),
        color,
        -1,
    )
    cv2.putText(image, text, (x + 2, y_text - baseline), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def draw_box(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    class_id: int,
    label: str,
    thickness: int,
) -> None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    color = COLORS[class_id % len(COLORS)]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    draw_label(image, label, x1, max(0, y1 - 4), color)


def draw_ground_truth(image: np.ndarray, label_path: Path) -> None:
    if not label_path.exists():
        return
    height, width = image.shape[:2]
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        class_id, box = yolo_to_xyxy(line, width, height)
        draw_box(image, box, class_id, f"G-{SHORT_LABELS[class_id]}", thickness=1)


def draw_predictions(image: np.ndarray, result: Any) -> None:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    for box, conf, class_id in zip(xyxy, confs, classes):
        x1, y1, x2, y2 = [int(round(value)) for value in box.tolist()]
        draw_box(image, (x1, y1, x2, y2), int(class_id), f"P-{SHORT_LABELS[class_id]} {conf:.2f}", thickness=2)


def main() -> None:
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS}")
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {DATA_YAML}\nRun: python scripts/prepare_dataset.py")

    test_images_dir, test_labels_dir = load_dataset_paths(DATA_YAML)
    image_paths = sorted(path for path in test_images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    image_paths = image_paths[:MAX_IMAGES]
    if not image_paths:
        raise FileNotFoundError(f"No test images found in {test_images_dir}")

    from ultralytics import YOLO

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        iou=NMS_IOU,
        device=DEVICE,
        verbose=False,
    )

    for image_path, result in zip(image_paths, results):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARNING] Cannot read image: {image_path}")
            continue
        draw_ground_truth(image, test_labels_dir / f"{image_path.stem}.txt")
        draw_predictions(image, result)
        output_path = OUTPUT_DIR / f"{image_path.stem}_gt_pred.jpg"
        cv2.imwrite(str(output_path), image)
        print(f"[DONE] {output_path}")


if __name__ == "__main__":
    main()

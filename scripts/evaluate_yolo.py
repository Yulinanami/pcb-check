"""Evaluate YOLO predictions with mAP@IoU=0.5 on the PCB test split."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


CLASS_NAMES = ["Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = project_root()
WEIGHTS = ROOT / "runs" / "train" / "pcb_yolo_baseline" / "weights" / "best.pt"
DATA_YAML = ROOT / "outputs" / "pcb_yolo_dataset" / "dataset.yaml"
OUTPUT_DIR = ROOT / "outputs" / "eval"
IMAGE_SIZE = 1024
CONFIDENCE = 0.001
NMS_IOU = 0.7
DEVICE = "0"
EVAL_IOU = 0.5


def load_dataset_paths(data_yaml: Path) -> tuple[Path, Path, list[str]]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(data["path"])
    test_images = dataset_root / data.get("test", "images/test")
    test_labels = dataset_root / "labels" / "test"
    names = data.get("names", CLASS_NAMES)
    if isinstance(names, dict):
        class_names = [names[idx] for idx in sorted(names)]
    else:
        class_names = list(names)
    return test_images, test_labels, class_names


def yolo_label_to_xyxy(line: str, image_width: int, image_height: int) -> tuple[int, np.ndarray]:
    parts = line.split()
    class_id = int(float(parts[0]))
    x_center, y_center, box_width, box_height = map(float, parts[1:5])
    xmin = (x_center - box_width / 2.0) * image_width
    ymin = (y_center - box_height / 2.0) * image_height
    xmax = (x_center + box_width / 2.0) * image_width
    ymax = (y_center + box_height / 2.0) * image_height
    return class_id, np.array([xmin, ymin, xmax, ymax], dtype=np.float32)


def load_ground_truth(image_paths: list[Path], label_dir: Path) -> dict[int, dict[str, list[np.ndarray]]]:
    gt_by_class: dict[int, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            class_id, box = yolo_label_to_xyxy(line, width, height)
            gt_by_class[class_id][image_path.name].append(box)
    return gt_by_class


def box_iou(box: np.ndarray, boxes: list[np.ndarray]) -> np.ndarray:
    if not boxes:
        return np.array([], dtype=np.float32)
    other = np.stack(boxes).astype(np.float32)
    x1 = np.maximum(box[0], other[:, 0])
    y1 = np.maximum(box[1], other[:, 1])
    x2 = np.minimum(box[2], other[:, 2])
    y2 = np.minimum(box[3], other[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_box = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_other = np.maximum(0.0, other[:, 2] - other[:, 0]) * np.maximum(0.0, other[:, 3] - other[:, 1])
    union = area_box + area_other - inter
    return inter / np.maximum(union, 1e-9)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for idx in range(mpre.size - 1, 0, -1):
        mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])
    change_points = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[change_points + 1] - mrec[change_points]) * mpre[change_points + 1]))


def collect_predictions(model: Any, image_paths: list[Path]) -> dict[int, list[dict[str, Any]]]:
    pred_by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        iou=NMS_IOU,
        device=DEVICE,
        verbose=False,
    )

    for image_path, result in zip(image_paths, results):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        for box, conf, class_id in zip(xyxy, confs, classes):
            pred_by_class[int(class_id)].append(
                {
                    "image": image_path.name,
                    "confidence": float(conf),
                    "box": box.astype(np.float32),
                }
            )
    return pred_by_class


def evaluate_class(
    class_id: int,
    predictions: list[dict[str, Any]],
    gt_by_image: dict[str, list[np.ndarray]],
    iou_threshold: float,
) -> dict[str, float | int]:
    n_gt = sum(len(boxes) for boxes in gt_by_image.values())
    if n_gt == 0:
        return {"class_id": class_id, "num_gt": 0, "num_predictions": len(predictions), "ap": 0.0}

    predictions = sorted(predictions, key=lambda item: item["confidence"], reverse=True)
    matched = {image_name: np.zeros(len(boxes), dtype=bool) for image_name, boxes in gt_by_image.items()}
    tp = np.zeros(len(predictions), dtype=np.float32)
    fp = np.zeros(len(predictions), dtype=np.float32)

    for idx, pred in enumerate(predictions):
        image_name = pred["image"]
        gt_boxes = gt_by_image.get(image_name, [])
        ious = box_iou(pred["box"], gt_boxes)
        if ious.size == 0:
            fp[idx] = 1.0
            continue

        best_gt_idx = int(np.argmax(ious))
        if ious[best_gt_idx] >= iou_threshold and not matched[image_name][best_gt_idx]:
            tp[idx] = 1.0
            matched[image_name][best_gt_idx] = True
        else:
            fp[idx] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / max(n_gt, 1)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    ap = compute_ap(recall, precision) if predictions else 0.0
    return {
        "class_id": class_id,
        "num_gt": int(n_gt),
        "num_predictions": int(len(predictions)),
        "ap": float(ap),
    }


def write_outputs(metrics: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "metrics_iou50.json"
    csv_path = output_dir / "metrics_iou50.csv"

    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["class_id", "class_name", "num_gt", "num_predictions", "ap"])
        writer.writeheader()
        writer.writerows(metrics["classes"])

    print(f"[DONE] Metrics json: {json_path}")
    print(f"[DONE] Metrics csv:  {csv_path}")


def main() -> None:
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS}")
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {DATA_YAML}\nRun: python scripts/prepare_dataset.py")

    test_images_dir, test_labels_dir, class_names = load_dataset_paths(DATA_YAML)
    image_paths = sorted(path for path in test_images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise FileNotFoundError(f"No test images found in {test_images_dir}")

    from ultralytics import YOLO

    gt_by_class = load_ground_truth(image_paths, test_labels_dir)
    model = YOLO(str(WEIGHTS))
    pred_by_class = collect_predictions(model, image_paths)

    class_metrics = []
    for class_id, class_name in enumerate(class_names):
        result = evaluate_class(
            class_id=class_id,
            predictions=pred_by_class.get(class_id, []),
            gt_by_image=gt_by_class.get(class_id, {}),
            iou_threshold=EVAL_IOU,
        )
        result["class_name"] = class_name
        class_metrics.append(result)

    valid_aps = [item["ap"] for item in class_metrics if item["num_gt"] > 0]
    metrics = {
        "iou_threshold": EVAL_IOU,
        "mAP": float(np.mean(valid_aps)) if valid_aps else 0.0,
        "num_images": len(image_paths),
        "classes": class_metrics,
    }
    print(f"mAP@{EVAL_IOU:.2f}: {metrics['mAP']:.6f}")
    for item in class_metrics:
        print(f"{item['class_name']}: AP={item['ap']:.6f}, GT={item['num_gt']}, Pred={item['num_predictions']}")

    write_outputs(metrics, OUTPUT_DIR)


if __name__ == "__main__":
    main()

"""计算测试集 mAP@0.5 和每类 AP。"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "runs" / "train" / "pcb_yolo_true_train" / "weights" / "best.pt"
TEST_IMAGES = ROOT / "outputs" / "pcb_yolo_dataset" / "images" / "test"
TEST_LABELS = ROOT / "outputs" / "pcb_yolo_dataset" / "labels" / "test"
OUT_DIR = ROOT / "outputs" / "eval"
CLASSES = ["Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]


def label_to_box(line, width, height):
    class_id, x, y, w, h = map(float, line.split()[:5])
    return int(class_id), np.array(
        [(x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height],
        dtype=np.float32,
    )


def load_gt(image_paths):
    gt = defaultdict(lambda: defaultdict(list))
    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size
        label_path = TEST_LABELS / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                class_id, box = label_to_box(line, width, height)
                gt[class_id][image_path.name].append(box)
    return gt


def iou(box, boxes):
    if not boxes:
        return np.array([], dtype=np.float32)
    boxes = np.stack(boxes)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area2 = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-9)


def ap_from_pr(recall, precision):
    recall = np.concatenate(([0], recall, [1]))
    precision = np.concatenate(([0], precision, [0]))
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    points = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[points + 1] - recall[points]) * precision[points + 1]))


def safe_div(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def collect_predictions(model, image_paths):
    predictions = defaultdict(list)
    results = model.predict([str(path) for path in image_paths], imgsz=1024, conf=0.001, iou=0.7, device="0", verbose=False)
    for image_path, result in zip(image_paths, results):
        if result.boxes is None:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        for box, conf, class_id in zip(boxes, confs, class_ids):
            predictions[int(class_id)].append({"image": image_path.name, "conf": float(conf), "box": box.astype(np.float32)})
    return predictions


def evaluate_class(class_id, predictions, gt_by_image):
    total_gt = sum(len(boxes) for boxes in gt_by_image.values())
    predictions = sorted(predictions, key=lambda item: item["conf"], reverse=True)
    matched = {name: np.zeros(len(boxes), dtype=bool) for name, boxes in gt_by_image.items()}
    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))

    for i, pred in enumerate(predictions):
        gt_boxes = gt_by_image.get(pred["image"], [])
        overlaps = iou(pred["box"], gt_boxes)
        if len(overlaps) == 0:
            fp[i] = 1
            continue
        best = int(np.argmax(overlaps))
        if overlaps[best] >= 0.5 and not matched[pred["image"]][best]:
            tp[i] = 1
            matched[pred["image"]][best] = True
        else:
            fp[i] = 1

    ap = 0.0
    if total_gt > 0 and len(predictions) > 0:
        recall = np.cumsum(tp) / total_gt
        precision = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-9)
        ap = ap_from_pr(recall, precision)
    tp_count = int(tp.sum())
    fp_count = int(fp.sum())
    fn_count = int(total_gt - tp_count)
    precision = safe_div(tp_count, tp_count + fp_count)
    recall = safe_div(tp_count, total_gt)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "class_id": class_id,
        "class_name": CLASSES[class_id],
        "num_gt": int(total_gt),
        "num_predictions": len(predictions),
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap": ap,
    }


def main():
    if not WEIGHTS.exists():
        raise FileNotFoundError("没有找到训练权重，请先运行 python scripts/train_yolo.py")

    image_paths = sorted(TEST_IMAGES.glob("*.*"))
    image_paths = [path for path in image_paths if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]

    from ultralytics import YOLO

    gt = load_gt(image_paths)
    predictions = collect_predictions(YOLO(str(WEIGHTS)), image_paths)
    rows = [evaluate_class(i, predictions.get(i, []), gt.get(i, {})) for i in range(len(CLASSES))]
    total_gt = sum(row["num_gt"] for row in rows)
    total_predictions = sum(row["num_predictions"] for row in rows)
    total_tp = sum(row["tp"] for row in rows)
    total_fp = sum(row["fp"] for row in rows)
    total_fn = sum(row["fn"] for row in rows)
    micro_precision = safe_div(total_tp, total_tp + total_fp)
    micro_recall = safe_div(total_tp, total_gt)
    micro_f1 = safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)
    metrics = {
        "iou_threshold": 0.5,
        "mAP": float(np.mean([row["ap"] for row in rows])),
        "num_images": len(image_paths),
        "total_gt": int(total_gt),
        "total_predictions": int(total_predictions),
        "total_tp": int(total_tp),
        "total_fp": int(total_fp),
        "total_fn": int(total_fn),
        "macro_precision": float(np.mean([row["precision"] for row in rows])),
        "macro_recall": float(np.mean([row["recall"] for row in rows])),
        "macro_f1": float(np.mean([row["f1"] for row in rows])),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "classes": rows,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics_iou50.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT_DIR / "metrics_iou50.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "class_id",
                "class_name",
                "num_gt",
                "num_predictions",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "ap",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"mAP@0.5: {metrics['mAP']:.6f}")
    print(
        "Overall: "
        f"GT={metrics['total_gt']}, Pred={metrics['total_predictions']}, "
        f"TP={metrics['total_tp']}, FP={metrics['total_fp']}, FN={metrics['total_fn']}, "
        f"P={metrics['micro_precision']:.6f}, R={metrics['micro_recall']:.6f}, F1={metrics['micro_f1']:.6f}"
    )
    for row in rows:
        print(
            f"{row['class_name']}: "
            f"AP={row['ap']:.6f}, GT={row['num_gt']}, Pred={row['num_predictions']}, "
            f"TP={row['tp']}, FP={row['fp']}, FN={row['fn']}, "
            f"P={row['precision']:.6f}, R={row['recall']:.6f}, F1={row['f1']:.6f}"
        )


if __name__ == "__main__":
    main()

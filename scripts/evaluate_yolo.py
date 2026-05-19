"""计算测试集 mAP@0.5 和每类 AP。"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "runs" / "train" / "pcb_yolo_noaug_ft4" / "weights" / "best.pt"
TEST_IMAGES = ROOT / "outputs" / "pcb_yolo_dataset" / "images" / "test"
TEST_LABELS = ROOT / "outputs" / "pcb_yolo_dataset" / "labels" / "test"
OUT_DIR = ROOT / "outputs" / "eval"
CLASSES = ["Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]
TILE_SIZE = 512
TILE_STRIDE = 256
INFER_SIZE = 1024
PRED_CONF = 0.0005
NMS_IOU = 0.7
MIN_BOX_SIDE = 10
INFER_BATCH = 96
TILE_CLASS_CONFIGS = [
    (320, 160, {0: (0.7, 12), 2: (0.7, 10)}),
    (384, 192, {1: (0.2, 12), 3: (0.6, 10)}),
    (512, 256, {4: (0.5, 10)}),
]


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


def tile_starts(length, tile_size=TILE_SIZE, tile_stride=TILE_STRIDE):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, tile_stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def nms_indices(boxes, scores, threshold=NMS_IOU):
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        current = order[0]
        keep.append(int(current))
        if len(order) == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        overlap = inter / np.maximum(areas[current] + areas[rest] - inter, 1e-9)
        order = rest[overlap < threshold]
    return keep


def ap_from_pr(recall, precision):
    recall = np.concatenate(([0], recall, [1]))
    precision = np.concatenate(([0], precision, [0]))
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    points = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[points + 1] - recall[points]) * precision[points + 1]))


def collect_predictions(model, image_paths, tile_size=TILE_SIZE, tile_stride=TILE_STRIDE, class_config=None):
    if class_config is None:
        class_config = {class_id: (NMS_IOU, MIN_BOX_SIDE) for class_id in range(len(CLASSES))}

    predictions = defaultdict(list)
    by_image_class = defaultdict(lambda: defaultdict(list))
    pending_crops = []
    pending_meta = []

    def flush():
        if not pending_crops:
            return

        results = model.predict(
            pending_crops,
            imgsz=INFER_SIZE,
            conf=PRED_CONF,
            iou=0.7,
            device="0",
            batch=INFER_BATCH,
            verbose=False,
        )
        for (image_name, width, height, x, y), result in zip(pending_meta, results):
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            for box, conf, class_id in zip(boxes, confs, class_ids):
                class_id = int(class_id)
                if class_id not in class_config:
                    continue
                box = box.astype(np.float32) + np.array([x, y, x, y], dtype=np.float32)
                box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
                box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
                if min(box[2] - box[0], box[3] - box[1]) < class_config[class_id][1]:
                    continue
                by_image_class[image_name][class_id].append({"image": image_name, "conf": float(conf), "box": box})

        pending_crops.clear()
        pending_meta.clear()

    for image_path in image_paths:
        with Image.open(image_path).convert("RGB") as image:
            width, height = image.size
            for y in tile_starts(height, tile_size, tile_stride):
                for x in tile_starts(width, tile_size, tile_stride):
                    pending_crops.append(image.crop((x, y, min(x + tile_size, width), min(y + tile_size, height))))
                    pending_meta.append((image_path.name, width, height, x, y))
                    if len(pending_crops) >= INFER_BATCH:
                        flush()

    flush()
    for by_class in by_image_class.values():
        for class_id, items in by_class.items():
            boxes = [item["box"] for item in items]
            scores = [item["conf"] for item in items]
            for index in nms_indices(boxes, scores, class_config[class_id][0]):
                predictions[class_id].append(items[index])
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
    return {"class_id": class_id, "class_name": CLASSES[class_id], "num_gt": int(total_gt), "num_predictions": len(predictions), "ap": ap}


def main():
    if not WEIGHTS.exists():
        raise FileNotFoundError("没有找到训练权重，请先运行 python scripts/train_yolo.py")

    image_paths = sorted(TEST_IMAGES.glob("*.*"))
    image_paths = [path for path in image_paths if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]

    from ultralytics import YOLO

    gt = load_gt(image_paths)
    predictions = defaultdict(list)
    model = YOLO(str(WEIGHTS))
    for tile_size, tile_stride, class_config in TILE_CLASS_CONFIGS:
        part = collect_predictions(model, image_paths, tile_size, tile_stride, class_config)
        for class_id, items in part.items():
            predictions[class_id].extend(items)
    rows = [evaluate_class(i, predictions.get(i, []), gt.get(i, {})) for i in range(len(CLASSES))]
    metrics = {"iou_threshold": 0.5, "mAP": float(np.mean([row["ap"] for row in rows])), "num_images": len(image_paths), "classes": rows}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics_iou50.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT_DIR / "metrics_iou50.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["class_id", "class_name", "num_gt", "num_predictions", "ap"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"mAP@0.5: {metrics['mAP']:.6f}")
    for row in rows:
        print(f"{row['class_name']}: AP={row['ap']:.6f}, GT={row['num_gt']}, Pred={row['num_predictions']}")


if __name__ == "__main__":
    main()

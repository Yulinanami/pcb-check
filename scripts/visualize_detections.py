"""画出测试集 GT 框和预测框。"""

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "runs" / "train" / "pcb_yolo_noaug_ft4" / "weights" / "best.pt"
AUX_WEIGHTS = ROOT / "runs" / "train" / "pcb_yolo_noaug_ft5_exp" / "weights" / "last.pt"
TEST_IMAGES = ROOT / "outputs" / "pcb_yolo_dataset" / "images" / "test"
TEST_LABELS = ROOT / "outputs" / "pcb_yolo_dataset" / "labels" / "test"
OUT_DIR = ROOT / "outputs" / "visualizations"
SHORT_LABELS = ["M", "O", "Sh", "Sp", "Sc"]
COLORS = [(0, 255, 255), (255, 128, 0), (0, 255, 0), (255, 0, 255), (0, 128, 255)]
TILE_SIZE = 512
TILE_STRIDE = 256
INFER_SIZE = 1024
NMS_IOU = 0.7
MIN_BOX_SIDE = 10
MODEL_CLASS_CONFIGS = [
    (WEIGHTS, {2: (NMS_IOU, MIN_BOX_SIDE), 4: (NMS_IOU, MIN_BOX_SIDE)}),
    (AUX_WEIGHTS, {0: (0.5, 16), 1: (0.25, 10), 3: (0.25, 12)}),
]


def yolo_to_box(line, width, height):
    class_id, x, y, w, h = map(float, line.split()[:5])
    return int(class_id), (
        round((x - w / 2) * width),
        round((y - h / 2) * height),
        round((x + w / 2) * width),
        round((y + h / 2) * height),
    )


def draw_box(image, box, class_id, text, thickness):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
    y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))
    color = COLORS[class_id % len(COLORS)]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(image, text, (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_gt(image, label_path):
    if not label_path.exists():
        return
    height, width = image.shape[:2]
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            class_id, box = yolo_to_box(line, width, height)
            draw_box(image, box, class_id, f"G-{SHORT_LABELS[class_id]}", 1)


def tile_starts(length):
    if length <= TILE_SIZE:
        return [0]
    starts = list(range(0, length - TILE_SIZE + 1, TILE_STRIDE))
    if starts[-1] != length - TILE_SIZE:
        starts.append(length - TILE_SIZE)
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


def predict_tiles(model, image, class_config=None):
    if class_config is None:
        class_config = {class_id: (NMS_IOU, MIN_BOX_SIDE) for class_id in range(len(SHORT_LABELS))}

    height, width = image.shape[:2]
    tiles = []
    offsets = []
    for y in tile_starts(height):
        for x in tile_starts(width):
            tiles.append(image[y : min(y + TILE_SIZE, height), x : min(x + TILE_SIZE, width)])
            offsets.append((x, y))

    by_class = {}
    results = model.predict(tiles, imgsz=INFER_SIZE, conf=0.25, iou=0.7, device="0", batch=16, verbose=False)
    for (x, y), result in zip(offsets, results):
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
            by_class.setdefault(class_id, []).append((box, float(conf)))

    predictions = []
    for class_id, items in by_class.items():
        boxes = [item[0] for item in items]
        scores = [item[1] for item in items]
        for index in nms_indices(boxes, scores, class_config[class_id][0]):
            predictions.append((boxes[index], scores[index], class_id))
    return predictions


def draw_pred(image, predictions):
    for box, conf, class_id in predictions:
        draw_box(image, tuple(round(v) for v in box.tolist()), class_id, f"P-{SHORT_LABELS[class_id]} {conf:.2f}", 2)


def main():
    if not WEIGHTS.exists():
        raise FileNotFoundError("没有找到训练权重，请先运行 python scripts/train_yolo.py")

    image_paths = sorted(TEST_IMAGES.glob("*.*"))[:10]

    from ultralytics import YOLO

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_configs = MODEL_CLASS_CONFIGS if AUX_WEIGHTS.exists() else [(WEIGHTS, None)]
    models = [(YOLO(str(weights)), class_config) for weights, class_config in model_configs]

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        draw_gt(image, TEST_LABELS / f"{image_path.stem}.txt")
        predictions = []
        for model, class_config in models:
            predictions.extend(predict_tiles(model, image, class_config))
        draw_pred(image, predictions)
        output_path = OUT_DIR / f"{image_path.stem}_gt_pred.jpg"
        cv2.imwrite(str(output_path), image)
        print(f"[DONE] {output_path}")


if __name__ == "__main__":
    main()

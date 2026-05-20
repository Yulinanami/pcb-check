"""画出测试集 GT 框和预测框。"""

import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "runs" / "train" / "pcb_yolo_noaug_ft4" / "weights" / "best.pt"
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
INFER_BATCH = 48
DIFF_THRESHOLD = 15
TILE_CLASS_CONFIGS = [
    (320, 160, {0: (0.7, 12), 2: (0.7, 10)}),
    (384, 192, {1: (0.2, 12), 3: (0.6, 10)}),
    (512, 256, {4: (0.5, 10)}),
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


def build_diff_masks(images):
    groups = defaultdict(list)
    for image_name, image in images.items():
        match = re.match(r"(\d+)", Path(image_name).stem)
        if match:
            groups[match.group(1)].append((image_name, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)))

    masks = {}
    for items in groups.values():
        stack = np.stack([gray for _, gray in items])
        for index, (image_name, gray) in enumerate(items):
            others = np.delete(stack, index, axis=0)
            if len(others) == 0:
                masks[image_name] = np.full_like(gray, 255)
                continue
            ref = np.median(others, axis=0).astype(np.uint8)
            diff = cv2.absdiff(gray, ref)
            diff = cv2.GaussianBlur(diff, (3, 3), 0)
            _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
            masks[image_name] = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return masks


def filter_predictions_by_diff(predictions_by_image, diff_masks):
    filtered = defaultdict(list)
    for image_name, predictions in predictions_by_image.items():
        mask = diff_masks.get(image_name)
        if mask is None:
            filtered[image_name].extend(predictions)
            continue
        height, width = mask.shape
        for box, conf, class_id in predictions:
            int_box = box.astype(int)
            x1, y1 = max(0, int_box[0]), max(0, int_box[1])
            x2, y2 = min(width, int_box[2]), min(height, int_box[3])
            if x2 > x1 and y2 > y1 and np.any(mask[y1:y2, x1:x2] > 0):
                filtered[image_name].append((box, conf, class_id))
    return filtered


def collect_predictions(model, images):
    predictions_by_image = defaultdict(list)

    for tile_size, tile_stride, class_config in TILE_CLASS_CONFIGS:
        by_image_class = defaultdict(lambda: defaultdict(list))

        def predict_batch(crops, metas):
            results = model.predict(crops, imgsz=INFER_SIZE, conf=0.25, iou=0.7, device="0", batch=INFER_BATCH, verbose=False)
            for (image_name, width, height, x, y), result in zip(metas, results):
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
                    by_image_class[image_name][class_id].append((box, float(conf)))

        crops = []
        metas = []
        for image_name, image in images.items():
            height, width = image.shape[:2]
            for y in tile_starts(height, tile_size, tile_stride):
                for x in tile_starts(width, tile_size, tile_stride):
                    crops.append(image[y : min(y + tile_size, height), x : min(x + tile_size, width)])
                    metas.append((image_name, width, height, x, y))
                    if len(crops) >= INFER_BATCH:
                        predict_batch(crops, metas)
                        crops.clear()
                        metas.clear()

        if crops:
            predict_batch(crops, metas)

        for image_name, by_class in by_image_class.items():
            for class_id, items in by_class.items():
                boxes = [item[0] for item in items]
                scores = [item[1] for item in items]
                for index in nms_indices(boxes, scores, class_config[class_id][0]):
                    predictions_by_image[image_name].append((boxes[index], scores[index], class_id))

    return predictions_by_image


def draw_pred(image, predictions):
    for box, conf, class_id in predictions:
        draw_box(image, tuple(round(v) for v in box.tolist()), class_id, f"P-{SHORT_LABELS[class_id]} {conf:.2f}", 2)


def main():
    if not WEIGHTS.exists():
        raise FileNotFoundError("没有找到训练权重，请先运行 python scripts/train_yolo.py")

    image_paths = sorted(TEST_IMAGES.glob("*.*"))[:10]

    from ultralytics import YOLO

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))
    images = {}

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        images[image_path.name] = image

    predictions_by_image = filter_predictions_by_diff(collect_predictions(model, images), build_diff_masks(images))

    for image_path in image_paths:
        image = images.get(image_path.name)
        if image is None:
            continue
        draw_gt(image, TEST_LABELS / f"{image_path.stem}.txt")
        draw_pred(image, predictions_by_image.get(image_path.name, []))
        output_path = OUT_DIR / f"{image_path.stem}_gt_pred.jpg"
        cv2.imwrite(str(output_path), image)
        print(f"[DONE] {output_path}")


if __name__ == "__main__":
    main()

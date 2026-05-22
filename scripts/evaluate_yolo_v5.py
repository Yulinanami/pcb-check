"""边缘增强评估（v5）：推理时对测试图做相同的灰度+CLAHE预处理。"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "runs" / "train" / "pcb_v5_edge" / "weights" / "best.pt"
TEST_IMAGES = ROOT / "outputs" / "pcb_yolo_dataset_v5" / "images" / "test"
TEST_LABELS = ROOT / "outputs" / "pcb_yolo_dataset_v5" / "labels" / "test"
OUT_DIR = ROOT / "outputs" / "eval_v5"
CLASSES = ["Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]

TILE_SIZE = 640
OVERLAP = 0.5


def normalize_board(image):
    """与训练相同：灰度 + CLAHE。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def imread_safe(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def tile_positions_1d(length, tile_size, overlap):
    stride = int(tile_size * (1 - overlap))
    positions = list(range(0, max(1, length - tile_size + 1), stride))
    edge = length - tile_size
    if edge > 0 and edge not in positions:
        positions.append(edge)
    if not positions:
        positions = [0]
    return positions


def nms_per_class(boxes, confs, class_ids, iou_threshold=0.5):
    if len(boxes) == 0:
        return []
    keep_all = []
    for cls_id in np.unique(class_ids):
        mask = class_ids == cls_id
        cb = boxes[mask]
        cc = confs[mask]
        ci = np.where(mask)[0]
        x1, y1, x2, y2 = cb[:, 0], cb[:, 1], cb[:, 2], cb[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = cc.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(ci[i])
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-9)
            order = order[np.where(iou <= iou_threshold)[0] + 1]
        keep_all.extend(keep)
    return sorted(keep_all)


def tiled_predict(model, image_path, tile_size, overlap, conf=0.001, iou=0.7):
    image = imread_safe(image_path)
    if image is None:
        return [], [], []
    # ★ 与训练相同的预处理
    image = normalize_board(image)
    img_h, img_w = image.shape[:2]
    xs = tile_positions_1d(img_w, tile_size, overlap)
    ys = tile_positions_1d(img_h, tile_size, overlap)
    tiles, positions = [], []
    for ty in ys:
        for tx in xs:
            ty2 = min(ty + tile_size, img_h)
            tx2 = min(tx + tile_size, img_w)
            tile = image[ty:ty2, tx:tx2]
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded
            tiles.append(tile)
            positions.append((tx, ty))
    if not tiles:
        return [], [], []
    results = model.predict(tiles, imgsz=tile_size, conf=conf, iou=iou,
                            device="0", verbose=False)
    all_boxes, all_confs, all_classes = [], [], []
    for result, (tx, ty) in zip(results, positions):
        if result.boxes is None or len(result.boxes) == 0:
            continue
        bx = result.boxes.xyxy.cpu().numpy()
        cs = result.boxes.conf.cpu().numpy()
        cl = result.boxes.cls.cpu().numpy().astype(int)
        bx[:, 0] += tx; bx[:, 1] += ty; bx[:, 2] += tx; bx[:, 3] += ty
        bx[:, 0] = np.clip(bx[:, 0], 0, img_w)
        bx[:, 1] = np.clip(bx[:, 1], 0, img_h)
        bx[:, 2] = np.clip(bx[:, 2], 0, img_w)
        bx[:, 3] = np.clip(bx[:, 3], 0, img_h)
        all_boxes.append(bx); all_confs.append(cs); all_classes.append(cl)
    if not all_boxes:
        return [], [], []
    all_boxes = np.concatenate(all_boxes)
    all_confs = np.concatenate(all_confs)
    all_classes = np.concatenate(all_classes)
    keep = nms_per_class(all_boxes, all_confs, all_classes, iou_threshold=0.5)
    return all_boxes[keep], all_confs[keep], all_classes[keep]


# ─── mAP 计算 ───

def label_to_box(line, width, height):
    class_id, x, y, w, h = map(float, line.split()[:5])
    return int(class_id), np.array(
        [(x - w / 2) * width, (y - h / 2) * height,
         (x + w / 2) * width, (y + h / 2) * height], dtype=np.float32)


def load_gt(image_paths):
    gt = defaultdict(lambda: defaultdict(list))
    for ip in image_paths:
        with Image.open(ip) as img:
            w, h = img.size
        lp = TEST_LABELS / f"{ip.stem}.txt"
        if not lp.exists():
            continue
        for line in lp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cid, box = label_to_box(line, w, h)
                gt[cid][ip.name].append(box)
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
    a1 = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    a2 = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a1 + a2 - inter, 1e-9)


def ap_from_pr(recall, precision):
    r = np.concatenate(([0], recall, [1]))
    p = np.concatenate(([0], precision, [0]))
    for i in range(len(p) - 1, 0, -1):
        p[i - 1] = max(p[i - 1], p[i])
    pts = np.where(r[1:] != r[:-1])[0]
    return float(np.sum((r[pts + 1] - r[pts]) * p[pts + 1]))


def safe_div(a, b):
    return float(a / b) if b else 0.0


def evaluate_class(class_id, predictions, gt_by_image):
    total_gt = sum(len(b) for b in gt_by_image.values())
    predictions = sorted(predictions, key=lambda x: x["conf"], reverse=True)
    matched = {n: np.zeros(len(b), dtype=bool) for n, b in gt_by_image.items()}
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
        r = np.cumsum(tp) / total_gt
        p = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-9)
        ap = ap_from_pr(r, p)
    tp_c, fp_c = int(tp.sum()), int(fp.sum())
    fn_c = int(total_gt - tp_c)
    prec = safe_div(tp_c, tp_c + fp_c)
    rec = safe_div(tp_c, total_gt)
    f1 = safe_div(2 * prec * rec, prec + rec)
    return {"class_id": class_id, "class_name": CLASSES[class_id],
            "num_gt": total_gt, "num_predictions": len(predictions),
            "tp": tp_c, "fp": fp_c, "fn": fn_c,
            "precision": prec, "recall": rec, "f1": f1, "ap": ap}


def collect_predictions_tiled(model, image_paths):
    predictions = defaultdict(list)
    for ip in image_paths:
        boxes, confs, class_ids = tiled_predict(model, ip, TILE_SIZE, OVERLAP)
        if len(boxes) == 0:
            continue
        for box, conf, cls_id in zip(boxes, confs, class_ids):
            predictions[int(cls_id)].append({
                "image": ip.name, "conf": float(conf),
                "box": box.astype(np.float32)})
    return predictions


def main():
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"没有找到权重: {WEIGHTS}")

    image_paths = sorted(TEST_IMAGES.glob("*.*"))
    image_paths = [p for p in image_paths
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]

    from ultralytics import YOLO

    print(f"[INFO] 切片推理 + 灰度CLAHE预处理")
    print(f"[INFO] tile={TILE_SIZE}, overlap={OVERLAP}, 测试图: {len(image_paths)} 张")

    gt = load_gt(image_paths)
    predictions = collect_predictions_tiled(YOLO(str(WEIGHTS)), image_paths)
    rows = [evaluate_class(i, predictions.get(i, []), gt.get(i, {}))
            for i in range(len(CLASSES))]

    total_gt = sum(r["num_gt"] for r in rows)
    total_pred = sum(r["num_predictions"] for r in rows)
    total_tp = sum(r["tp"] for r in rows)
    total_fp = sum(r["fp"] for r in rows)
    micro_p = safe_div(total_tp, total_tp + total_fp)
    micro_r = safe_div(total_tp, total_gt)
    micro_f1 = safe_div(2 * micro_p * micro_r, micro_p + micro_r)
    metrics = {
        "iou_threshold": 0.5,
        "mAP": float(np.mean([r["ap"] for r in rows])),
        "total_gt": int(total_gt), "total_predictions": int(total_pred),
        "total_tp": int(total_tp), "total_fp": int(total_fp),
        "classes": rows,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics_iou50.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"mAP@0.5: {metrics['mAP']:.6f}")
    print(f"Overall: GT={total_gt}, Pred={total_pred}, "
          f"TP={total_tp}, FP={total_fp}, "
          f"P={micro_p:.6f}, R={micro_r:.6f}, F1={micro_f1:.6f}")
    for r in rows:
        print(f"{r['class_name']}: AP={r['ap']:.6f}, "
              f"GT={r['num_gt']}, Pred={r['num_predictions']}, "
              f"TP={r['tp']}, FP={r['fp']}")


if __name__ == "__main__":
    main()

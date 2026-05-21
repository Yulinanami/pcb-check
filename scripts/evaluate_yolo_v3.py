"""切片推理评估：将测试图切成 tile 推理，合并预测后计算 mAP。

与 evaluate_yolo.py 的区别：
- 推理时将每张测试图切成 640×640 tile（与训练一致）
- 对每个 tile 单独推理
- 将预测框映射回原图坐标
- 跨 tile NMS 去重
- mAP 计算逻辑完全不变
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "runs" / "train" / "pcb_v3_tile" / "weights" / "best.pt"
TEST_IMAGES = ROOT / "outputs" / "pcb_yolo_dataset_v3" / "images" / "test"
TEST_LABELS = ROOT / "outputs" / "pcb_yolo_dataset_v3" / "labels" / "test"
OUT_DIR = ROOT / "outputs" / "eval_v3"
CLASSES = ["Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]

TILE_SIZE = 640
OVERLAP = 0.5


# ═══════════════════ 切片与 NMS ═══════════════════

def imread_safe(path):
    import cv2
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
    """对每个类别分别做 NMS，返回保留的索引。"""
    if len(boxes) == 0:
        return []
    keep_all = []
    for cls_id in np.unique(class_ids):
        mask = class_ids == cls_id
        cls_boxes = boxes[mask]
        cls_confs = confs[mask]
        cls_indices = np.where(mask)[0]

        x1 = cls_boxes[:, 0]
        y1 = cls_boxes[:, 1]
        x2 = cls_boxes[:, 2]
        y2 = cls_boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = cls_confs.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(cls_indices[i])
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
    """切片推理：切图 → 逐 tile 预测 → 映射回原图 → NMS。"""
    image = imread_safe(image_path)
    if image is None:
        return [], [], []
    img_h, img_w = image.shape[:2]
    xs = tile_positions_1d(img_w, tile_size, overlap)
    ys = tile_positions_1d(img_h, tile_size, overlap)

    tiles = []
    positions = []
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

    # 批量推理
    results = model.predict(tiles, imgsz=tile_size, conf=conf, iou=iou,
                            device="0", verbose=False)

    all_boxes = []
    all_confs = []
    all_classes = []
    for result, (tx, ty) in zip(results, positions):
        if result.boxes is None or len(result.boxes) == 0:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        cs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        # 映射回原图坐标
        boxes[:, 0] += tx
        boxes[:, 1] += ty
        boxes[:, 2] += tx
        boxes[:, 3] += ty
        # 裁剪到图像范围
        boxes[:, 0] = np.clip(boxes[:, 0], 0, img_w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, img_h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, img_w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, img_h)
        all_boxes.append(boxes)
        all_confs.append(cs)
        all_classes.append(cls_ids)

    if not all_boxes:
        return [], [], []

    all_boxes = np.concatenate(all_boxes)
    all_confs = np.concatenate(all_confs)
    all_classes = np.concatenate(all_classes)

    # 跨 tile NMS 去重
    keep = nms_per_class(all_boxes, all_confs, all_classes, iou_threshold=0.5)
    return all_boxes[keep], all_confs[keep], all_classes[keep]


# ═══════════════════ mAP 计算（与旧版一致） ═══════════════════

def label_to_box(line, width, height):
    class_id, x, y, w, h = map(float, line.split()[:5])
    return int(class_id), np.array(
        [(x - w / 2) * width, (y - h / 2) * height,
         (x + w / 2) * width, (y + h / 2) * height], dtype=np.float32)


def load_gt(image_paths):
    gt = defaultdict(lambda: defaultdict(list))
    for image_path in image_paths:
        with Image.open(image_path) as img:
            width, height = img.size
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
    a1 = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    a2 = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a1 + a2 - inter, 1e-9)


def ap_from_pr(recall, precision):
    recall = np.concatenate(([0], recall, [1]))
    precision = np.concatenate(([0], precision, [0]))
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    pts = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[pts + 1] - recall[pts]) * precision[pts + 1]))


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
    tp_c = int(tp.sum())
    fp_c = int(fp.sum())
    fn_c = int(total_gt - tp_c)
    prec = safe_div(tp_c, tp_c + fp_c)
    rec = safe_div(tp_c, total_gt)
    f1 = safe_div(2 * prec * rec, prec + rec)
    return {"class_id": class_id, "class_name": CLASSES[class_id],
            "num_gt": total_gt, "num_predictions": len(predictions),
            "tp": tp_c, "fp": fp_c, "fn": fn_c,
            "precision": prec, "recall": rec, "f1": f1, "ap": ap}


# ═══════════════════ 主流程 ═══════════════════

def collect_predictions_tiled(model, image_paths):
    """切片推理收集所有预测。"""
    predictions = defaultdict(list)
    for image_path in image_paths:
        boxes, confs, class_ids = tiled_predict(
            model, image_path, TILE_SIZE, OVERLAP)
        if len(boxes) == 0:
            continue
        for box, conf, cls_id in zip(boxes, confs, class_ids):
            predictions[int(cls_id)].append({
                "image": image_path.name,
                "conf": float(conf),
                "box": box.astype(np.float32),
            })
    return predictions


def main():
    if not WEIGHTS.exists():
        raise FileNotFoundError(
            f"没有找到训练权重: {WEIGHTS}\n"
            "请先运行 python scripts/train_yolo_v3.py")

    image_paths = sorted(TEST_IMAGES.glob("*.*"))
    image_paths = [p for p in image_paths
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]

    from ultralytics import YOLO

    print(f"[INFO] 切片推理: tile={TILE_SIZE}, overlap={OVERLAP}")
    print(f"[INFO] 测试图像: {len(image_paths)} 张")

    gt = load_gt(image_paths)
    predictions = collect_predictions_tiled(YOLO(str(WEIGHTS)), image_paths)
    rows = [evaluate_class(i, predictions.get(i, []), gt.get(i, {}))
            for i in range(len(CLASSES))]

    total_gt = sum(r["num_gt"] for r in rows)
    total_pred = sum(r["num_predictions"] for r in rows)
    total_tp = sum(r["tp"] for r in rows)
    total_fp = sum(r["fp"] for r in rows)
    total_fn = sum(r["fn"] for r in rows)
    micro_p = safe_div(total_tp, total_tp + total_fp)
    micro_r = safe_div(total_tp, total_gt)
    micro_f1 = safe_div(2 * micro_p * micro_r, micro_p + micro_r)
    metrics = {
        "iou_threshold": 0.5,
        "mAP": float(np.mean([r["ap"] for r in rows])),
        "num_images": len(image_paths),
        "total_gt": int(total_gt), "total_predictions": int(total_pred),
        "total_tp": int(total_tp), "total_fp": int(total_fp),
        "total_fn": int(total_fn),
        "macro_precision": float(np.mean([r["precision"] for r in rows])),
        "macro_recall": float(np.mean([r["recall"] for r in rows])),
        "macro_f1": float(np.mean([r["f1"] for r in rows])),
        "micro_precision": micro_p, "micro_recall": micro_r,
        "micro_f1": micro_f1, "classes": rows,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics_iou50.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT_DIR / "metrics_iou50.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "class_id", "class_name", "num_gt", "num_predictions",
            "tp", "fp", "fn", "precision", "recall", "f1", "ap"])
        w.writeheader()
        w.writerows(rows)

    print(f"mAP@0.5: {metrics['mAP']:.6f}")
    print(f"Overall: GT={total_gt}, Pred={total_pred}, "
          f"TP={total_tp}, FP={total_fp}, FN={total_fn}, "
          f"P={micro_p:.6f}, R={micro_r:.6f}, F1={micro_f1:.6f}")
    for r in rows:
        print(f"{r['class_name']}: AP={r['ap']:.6f}, "
              f"GT={r['num_gt']}, Pred={r['num_predictions']}, "
              f"TP={r['tp']}, FP={r['fp']}, FN={r['fn']}, "
              f"P={r['precision']:.6f}, R={r['recall']:.6f}, F1={r['f1']:.6f}")


if __name__ == "__main__":
    main()

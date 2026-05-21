"""画出测试集 GT 框和预测框。"""

from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "runs" / "train" / "pcb_yolo_true_train" / "weights" / "best.pt"
TEST_IMAGES = ROOT / "outputs" / "pcb_yolo_dataset" / "images" / "test"
TEST_LABELS = ROOT / "outputs" / "pcb_yolo_dataset" / "labels" / "test"
OUT_DIR = ROOT / "outputs" / "visualizations"
SHORT_LABELS = ["M", "O", "Sh", "Sp", "Sc"]
COLORS = [(0, 255, 255), (255, 128, 0), (0, 255, 0), (255, 0, 255), (0, 128, 255)]


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


def draw_pred(image, result):
    if result.boxes is None:
        return
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    for box, conf, class_id in zip(boxes, confs, class_ids):
        draw_box(image, tuple(round(v) for v in box.tolist()), int(class_id), f"P-{SHORT_LABELS[class_id]} {conf:.2f}", 2)


def main():
    if not WEIGHTS.exists():
        raise FileNotFoundError("没有找到训练权重，请先运行 python scripts/train_yolo.py")

    image_paths = sorted(TEST_IMAGES.glob("*.bmp"))[:10]

    from ultralytics import YOLO

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = YOLO(str(WEIGHTS)).predict([str(path) for path in image_paths], imgsz=1024, conf=0.25, iou=0.7, device="0", verbose=False)

    for image_path, result in zip(image_paths, results):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        draw_gt(image, TEST_LABELS / f"{image_path.stem}.txt")
        draw_pred(image, result)
        output_path = OUT_DIR / f"{image_path.stem}_gt_pred.jpg"
        cv2.imwrite(str(output_path), image)
        print(f"[DONE] {output_path}")


if __name__ == "__main__":
    main()

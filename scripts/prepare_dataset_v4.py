"""尺度匹配的切片训练数据准备。

核心改进：
- 训练 tile = 1536×1536（YOLO resize 到 640 后，缺陷 72→30px）
- 测试 tile = 640×640（缺陷 25px 不变）
- 模型输入端的缺陷尺度：训练 30px ≈ 测试 25px（仅 1.2x 差距）
"""

import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "训练集-PCB_DATASET"
TEST_ROOT = ROOT / "PCB_瑕疵测试集"
OUT_ROOT = ROOT / "outputs" / "pcb_yolo_dataset_v4"

CLASSES = ["Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]
XML_CLASS_ID = {
    "mouse_bite": 0, "open_circuit": 1, "short": 2,
    "spur": 3, "spurious_copper": 4,
}

TRAIN_TILE = 1536   # 训练 tile 大小（YOLO 会缩到 640，缺陷 72→30px）
TEST_TILE = 640      # 测试 tile 大小（不缩放，缺陷 25px）
TRAIN_OVERLAP = 0.5
TEST_OVERLAP = 0.5
NEG_RATIO = 1.0


def imread_safe(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_safe(path, image):
    ext = Path(path).suffix
    ok, buf = cv2.imencode(ext, image)
    if ok:
        buf.tofile(str(path))


def xml_to_abs_boxes(xml_path):
    """XML → 像素坐标 [(cls_id, x1, y1, x2, y2), ...]"""
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        box = obj.find("bndbox")
        if name not in XML_CLASS_ID or box is None:
            continue
        x1 = float(box.findtext("xmin", "0"))
        y1 = float(box.findtext("ymin", "0"))
        x2 = float(box.findtext("xmax", "0"))
        y2 = float(box.findtext("ymax", "0"))
        if x2 > x1 and y2 > y1:
            boxes.append((XML_CLASS_ID[name], x1, y1, x2, y2))
    return boxes


def tile_positions_1d(length, tile_size, overlap):
    if length <= tile_size:
        return [0]
    stride = int(tile_size * (1 - overlap))
    positions = list(range(0, length - tile_size + 1, stride))
    edge = length - tile_size
    if edge > 0 and edge not in positions:
        positions.append(edge)
    return positions


def generate_tiles(image, abs_boxes, tile_size, overlap):
    """切图，返回 (tile_image, yolo_labels, has_defect) 列表。"""
    img_h, img_w = image.shape[:2]
    xs = tile_positions_1d(img_w, tile_size, overlap)
    ys = tile_positions_1d(img_h, tile_size, overlap)

    tiles = []
    for ty in ys:
        for tx in xs:
            ty2 = min(ty + tile_size, img_h)
            tx2 = min(tx + tile_size, img_w)
            tile = image[ty:ty2, tx:tx2]
            th, tw = tile.shape[:2]
            if th < tile_size or tw < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[:th, :tw] = tile
                tile = padded

            labels = []
            for cls_id, bx1, by1, bx2, by2 in abs_boxes:
                cx = (bx1 + bx2) / 2
                cy = (by1 + by2) / 2
                if tx <= cx < tx + tw and ty <= cy < ty + th:
                    lx1 = max(0, bx1 - tx)
                    ly1 = max(0, by1 - ty)
                    lx2 = min(tw, bx2 - tx)
                    ly2 = min(th, by2 - ty)
                    lcx = (lx1 + lx2) / 2 / tile_size
                    lcy = (ly1 + ly2) / 2 / tile_size
                    lw = (lx2 - lx1) / tile_size
                    lh = (ly2 - ly1) / tile_size
                    if lw > 0.002 and lh > 0.002:
                        labels.append(f"{cls_id} {lcx:.6f} {lcy:.6f} {lw:.6f} {lh:.6f}")

            tiles.append((tile, labels, len(labels) > 0))
    return tiles


def group_key(class_name, image_path):
    return class_name, image_path.stem.split("_", 1)[0]


def main():
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for split in ("train", "val", "test"):
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── 收集样本 ──
    groups = {}
    for class_name in CLASSES:
        for xml_path in sorted((TRAIN_ROOT / "Annotations" / class_name).glob("*.xml")):
            xml_root = ET.parse(xml_path).getroot()
            image_path = TRAIN_ROOT / "images" / class_name / (
                xml_root.findtext("filename") or f"{xml_path.stem}.jpg")
            if not image_path.exists():
                continue
            key = group_key(class_name, image_path)
            groups.setdefault(key, []).append(
                (image_path, xml_to_abs_boxes(xml_path)))

    # ── train/val 按板分组 ──
    rng = random.Random(42)
    val_groups = set()
    for class_name in CLASSES:
        cg = sorted(k for k in groups if k[0] == class_name)
        rng.shuffle(cg)
        n = max(1, round(len(cg) * 0.2)) if len(cg) > 1 else 0
        val_groups.update(cg[:n])

    # ── 生成训练/验证 tile（1536×1536） ──
    tile_rng = random.Random(42)
    stats = {"train_pos": 0, "train_neg": 0, "val_pos": 0, "val_neg": 0}

    for key in sorted(groups):
        split = "val" if key in val_groups else "train"
        for image_path, abs_boxes in groups[key]:
            image = imread_safe(image_path)
            if image is None:
                continue
            tiles = generate_tiles(image, abs_boxes, TRAIN_TILE, TRAIN_OVERLAP)
            pos = [(t, l) for t, l, h in tiles if h]
            neg = [(t, l) for t, l, h in tiles if not h]
            n_neg = min(len(neg), max(1, int(len(pos) * NEG_RATIO)))
            neg_sampled = tile_rng.sample(neg, n_neg) if neg else []

            for tile_img, tile_labels in pos + neg_sampled:
                is_pos = len(tile_labels) > 0
                tag = "pos" if is_pos else "neg"
                idx = stats[f"{split}_{tag}"]
                name = f"{image_path.stem}_t{tag}{idx}"
                imwrite_safe(OUT_ROOT / "images" / split / f"{name}.jpg", tile_img)
                (OUT_ROOT / "labels" / split / f"{name}.txt").write_text(
                    "\n".join(tile_labels) + "\n" if tile_labels else "",
                    encoding="utf-8")
                stats[f"{split}_{tag}"] += 1

    # ── 测试集（原样复制，推理时用 640 tile） ──
    test_count = 0
    for class_name in CLASSES:
        img_dir = TEST_ROOT / f"{class_name}_Img"
        lbl_dir = TEST_ROOT / f"{class_name}_txt"
        for lp in sorted(lbl_dir.glob("*.txt")):
            if lp.name.lower() == "classes.txt":
                continue
            ip = img_dir / f"{lp.stem}.bmp"
            if not ip.exists():
                continue
            shutil.copy2(ip, OUT_ROOT / "images" / "test" / ip.name)
            shutil.copy2(lp, OUT_ROOT / "labels" / "test" / lp.name)
            test_count += 1

    # ── dataset.yaml ──
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    (OUT_ROOT / "dataset.yaml").write_text(
        f'path: "{OUT_ROOT.resolve().as_posix()}"\n'
        "train: images/train\n"
        "val: images/val\n"
        f"names:\n{names}\n", encoding="utf-8")

    total_tr = stats["train_pos"] + stats["train_neg"]
    total_va = stats["val_pos"] + stats["val_neg"]
    print(f"\n[DONE] 尺度匹配数据集已就绪")
    print(f"  训练 tile ({TRAIN_TILE}px): {total_tr}"
          f"（正 {stats['train_pos']}, 负 {stats['train_neg']}）")
    print(f"  验证 tile ({TRAIN_TILE}px): {total_va}"
          f"（正 {stats['val_pos']}, 负 {stats['val_neg']}）")
    print(f"  测试: {test_count}（原图，推理时切 {TEST_TILE}px tile）")
    print(f"\n  训练缺陷 → 模型输入: 72 × (640/{TRAIN_TILE}) ≈ {72*640/TRAIN_TILE:.0f}px")
    print(f"  测试缺陷 → 模型输入: 25px（{TEST_TILE}px tile 不缩放）")


if __name__ == "__main__":
    main()

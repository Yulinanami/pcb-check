"""切片训练数据准备：将高分辨率 PCB 图切成 640×640 tile。

核心思路：
- 模型只看 640×640 的局部区域，无法靠全局板布局作弊
- 缺陷在 tile 中占比从 <1% 提升到 ~7%，更容易学到
- 保留部分无缺陷 tile 作为负样本，减少误报
"""

import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

# ──────────────────── 路径 ────────────────────
ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "训练集-PCB_DATASET"
TEST_ROOT = ROOT / "PCB_瑕疵测试集"
OUT_ROOT = ROOT / "outputs" / "pcb_yolo_dataset_v3"

# ──────────────────── 类别 ────────────────────
CLASSES = ["Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]
XML_CLASS_ID = {
    "mouse_bite": 0, "open_circuit": 1, "short": 2,
    "spur": 3, "spurious_copper": 4,
}

# ──────────────────── 切片参数 ────────────────────
TILE_SIZE = 640
OVERLAP = 0.5          # 50% 重叠
NEG_RATIO = 1.0        # 负样本 tile 数 / 正样本 tile 数


# ═══════════════════ 工具函数 ═══════════════════

def imread_safe(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_safe(path, image):
    ext = Path(path).suffix
    ok, buf = cv2.imencode(ext, image)
    if ok:
        buf.tofile(str(path))
    return ok


def xml_to_yolo_lines(xml_path):
    root = ET.parse(xml_path).getroot()
    width = int(float(root.findtext("size/width", "0")))
    height = int(float(root.findtext("size/height", "0")))
    lines = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        box = obj.find("bndbox")
        if name not in XML_CLASS_ID or box is None:
            continue
        xmin = max(0.0, min(float(box.findtext("xmin", "0")), width))
        ymin = max(0.0, min(float(box.findtext("ymin", "0")), height))
        xmax = max(0.0, min(float(box.findtext("xmax", "0")), width))
        ymax = max(0.0, min(float(box.findtext("ymax", "0")), height))
        if xmax <= xmin or ymax <= ymin:
            continue
        x = ((xmin + xmax) / 2) / width
        y = ((ymin + ymax) / 2) / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        lines.append(f"{XML_CLASS_ID[name]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return lines


def parse_yolo_line(line):
    parts = line.strip().split()
    return int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])


def group_key(class_name, image_path):
    return class_name, image_path.stem.split("_", 1)[0]


# ═══════════════════ 切片逻辑 ═══════════════════

def tile_positions_1d(length, tile_size, overlap):
    """沿一个维度生成 tile 起始位置列表。"""
    stride = int(tile_size * (1 - overlap))
    positions = list(range(0, max(1, length - tile_size + 1), stride))
    # 确保覆盖到右/下边缘
    edge = length - tile_size
    if edge > 0 and edge not in positions:
        positions.append(edge)
    if not positions:
        positions = [0]
    return positions


def generate_tiles(image, label_lines, tile_size, overlap):
    """将一张图切成 tile，返回 (tile_image, tile_labels, has_defect)。"""
    img_h, img_w = image.shape[:2]
    xs = tile_positions_1d(img_w, tile_size, overlap)
    ys = tile_positions_1d(img_h, tile_size, overlap)

    # 预解析所有缺陷（像素坐标）
    defects = []
    for line in label_lines:
        cls_id, cx, cy, bw, bh = parse_yolo_line(line)
        abs_cx = cx * img_w
        abs_cy = cy * img_h
        abs_w = bw * img_w
        abs_h = bh * img_h
        defects.append((cls_id, abs_cx, abs_cy, abs_w, abs_h))

    tiles = []
    for ty in ys:
        for tx in xs:
            tx2 = min(tx + tile_size, img_w)
            ty2 = min(ty + tile_size, img_h)
            tile = image[ty:ty2, tx:tx2]
            # 如果不足 tile_size 则填充
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded
                actual_w = tx2 - tx
                actual_h = ty2 - ty
            else:
                actual_w = tile_size
                actual_h = tile_size

            # 查找落在此 tile 内的缺陷
            tile_labels = []
            for cls_id, abs_cx, abs_cy, abs_w, abs_h in defects:
                # 缺陷中心必须在 tile 的有效区域内
                if tx <= abs_cx < tx + actual_w and ty <= abs_cy < ty + actual_h:
                    local_cx = (abs_cx - tx) / tile_size
                    local_cy = (abs_cy - ty) / tile_size
                    local_w = abs_w / tile_size
                    local_h = abs_h / tile_size
                    # 裁剪到 [0, 1]
                    half_w = local_w / 2
                    half_h = local_h / 2
                    local_cx = max(half_w, min(local_cx, 1 - half_w))
                    local_cy = max(half_h, min(local_cy, 1 - half_h))
                    local_w = min(local_w, 1.0)
                    local_h = min(local_h, 1.0)
                    tile_labels.append(
                        f"{cls_id} {local_cx:.6f} {local_cy:.6f} {local_w:.6f} {local_h:.6f}")

            tiles.append((tile, tile_labels, len(tile_labels) > 0))
    return tiles


# ═══════════════════ 主流程 ═══════════════════

def main():
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for split in ("train", "val", "test"):
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── 1. 收集样本 ──
    groups = {}
    for class_name in CLASSES:
        for xml_path in sorted((TRAIN_ROOT / "Annotations" / class_name).glob("*.xml")):
            xml_root = ET.parse(xml_path).getroot()
            image_path = TRAIN_ROOT / "images" / class_name / (
                xml_root.findtext("filename") or f"{xml_path.stem}.jpg")
            if not image_path.exists():
                continue
            key = group_key(class_name, image_path)
            groups.setdefault(key, []).append((image_path, xml_to_yolo_lines(xml_path)))

    # ── 2. train/val 按板分组划分 ──
    rng = random.Random(42)
    val_groups = set()
    for class_name in CLASSES:
        class_groups = sorted(k for k in groups if k[0] == class_name)
        rng.shuffle(class_groups)
        n_val = max(1, round(len(class_groups) * 0.2)) if len(class_groups) > 1 else 0
        val_groups.update(class_groups[:n_val])

    # ── 3. 对每张图生成 tile ──
    tile_rng = random.Random(42)
    stats = {"train_pos": 0, "train_neg": 0, "val_pos": 0, "val_neg": 0}

    for key in sorted(groups):
        split = "val" if key in val_groups else "train"
        for image_path, label_lines in groups[key]:
            image = imread_safe(image_path)
            if image is None:
                print(f"[WARNING] 无法读取: {image_path}")
                continue

            tiles = generate_tiles(image, label_lines, TILE_SIZE, OVERLAP)
            pos_tiles = [(t, l) for t, l, has in tiles if has]
            neg_tiles = [(t, l) for t, l, has in tiles if not has]

            # 采样负样本
            n_neg = min(len(neg_tiles), max(1, int(len(pos_tiles) * NEG_RATIO)))
            neg_sampled = tile_rng.sample(neg_tiles, n_neg) if neg_tiles else []

            for tile_img, tile_labels in pos_tiles + neg_sampled:
                is_pos = len(tile_labels) > 0
                idx = stats[f"{split}_pos"] if is_pos else stats[f"{split}_neg"]
                tag = "pos" if is_pos else "neg"
                name = f"{image_path.stem}_t{tag}{idx}"
                imwrite_safe(
                    OUT_ROOT / "images" / split / f"{name}.jpg", tile_img)
                (OUT_ROOT / "labels" / split / f"{name}.txt").write_text(
                    "\n".join(tile_labels) + "\n" if tile_labels else "",
                    encoding="utf-8")
                if is_pos:
                    stats[f"{split}_pos"] += 1
                else:
                    stats[f"{split}_neg"] += 1

    # ── 4. 测试集（原样复制，推理时再切片） ──
    test_count = 0
    for class_name in CLASSES:
        image_dir = TEST_ROOT / f"{class_name}_Img"
        label_dir = TEST_ROOT / f"{class_name}_txt"
        for label_path in sorted(label_dir.glob("*.txt")):
            if label_path.name.lower() == "classes.txt":
                continue
            image_path = image_dir / f"{label_path.stem}.bmp"
            if not image_path.exists():
                print(f"[WARNING] 缺少测试图像: {label_path.name}")
                continue
            shutil.copy2(image_path, OUT_ROOT / "images" / "test" / image_path.name)
            shutil.copy2(label_path, OUT_ROOT / "labels" / "test" / label_path.name)
            test_count += 1

    # ── 5. dataset.yaml ──
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    (OUT_ROOT / "dataset.yaml").write_text(
        f'path: "{OUT_ROOT.resolve().as_posix()}"\n'
        "train: images/train\n"
        "val: images/val\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )

    total_train = stats["train_pos"] + stats["train_neg"]
    total_val = stats["val_pos"] + stats["val_neg"]
    print(f"\n[DONE] 切片数据集已就绪")
    print(f"  Tile: {TILE_SIZE}×{TILE_SIZE}, overlap={OVERLAP}")
    print(f"  训练 tile: {total_train}（正样本 {stats['train_pos']}, 负样本 {stats['train_neg']}）")
    print(f"  验证 tile: {total_val}（正样本 {stats['val_pos']}, 负样本 {stats['val_neg']}）")
    print(f"  测试: {test_count}（原图，推理时切片）")


if __name__ == "__main__":
    main()

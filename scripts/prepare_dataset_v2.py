"""增强版 PCB 数据集准备：跨板 Copy-Paste 增强 + 缺陷尺度多样化。

相比 v1 的改进：
1. 从训练图中提取缺陷 patch，跨板粘贴到其他板的图像上，
   打破 "特定板 → 特定缺陷位置" 的绑定。
2. 粘贴时随机缩放缺陷（0.3–1.0×），让模型学会检测更小的目标，
   弥补训练集缺陷 (~48px) 与测试集缺陷 (~22px) 的尺度差距。
3. 使用边缘渐变融合使粘贴更自然。
"""

import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


def imread_safe(path):
    """cv2.imread 的中文路径安全替代。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_safe(path, image):
    """cv2.imwrite 的中文路径安全替代。"""
    ext = Path(path).suffix
    ok, buf = cv2.imencode(ext, image)
    if ok:
        buf.tofile(str(path))
    return ok

# ──────────────────────── 路径 ────────────────────────
ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "训练集-PCB_DATASET"
TEST_ROOT = ROOT / "PCB_瑕疵测试集"
OUT_ROOT = ROOT / "outputs" / "pcb_yolo_dataset_v2"

# ──────────────────────── 类别 ────────────────────────
CLASSES = ["Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]
XML_CLASS_ID = {
    "mouse_bite": 0,
    "open_circuit": 1,
    "short": 2,
    "spur": 3,
    "spurious_copper": 4,
}

# ──────────────────── 增强参数 ────────────────────
AUG_COPIES = 3          # 每张原图生成的增强副本数
PASTE_PER_COPY = 2       # 每个副本粘贴的缺陷数
SCALE_RANGE = (0.3, 1.0) # 缺陷缩放范围（重点模拟更小的目标）
PAD_RATIO = 0.5          # 缺陷裁剪时的填充比例


# ═══════════════════ XML / YOLO 转换 ═══════════════════

def xml_to_yolo_lines(xml_path):
    """Pascal VOC XML → YOLO txt 行。"""
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


# ═══════════════════ Copy-Paste 增强 ═══════════════════

def extract_defect_patches(image, label_lines):
    """从图像中裁剪缺陷 patch（含上下文填充）。"""
    patches = []
    img_h, img_w = image.shape[:2]
    for line in label_lines:
        cls_id, cx, cy, bw, bh = parse_yolo_line(line)
        abs_w = bw * img_w
        abs_h = bh * img_h
        x1 = int((cx - bw / 2) * img_w)
        y1 = int((cy - bh / 2) * img_h)
        x2 = int((cx + bw / 2) * img_w)
        y2 = int((cy + bh / 2) * img_h)
        # 加填充
        pad_x = int(abs_w * PAD_RATIO)
        pad_y = int(abs_h * PAD_RATIO)
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(img_w, x2 + pad_x)
        crop_y2 = min(img_h, y2 + pad_y)
        patch = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        if patch.size == 0:
            continue
        patches.append({
            "patch": patch,
            "cls_id": cls_id,
            "box_in_patch": (x1 - crop_x1, y1 - crop_y1,
                             x2 - crop_x1, y2 - crop_y1),
        })
    return patches


def _make_feather_mask(h, w, border):
    """生成边缘渐变 mask，用于自然融合。"""
    mask = np.ones((h, w), dtype=np.float32)
    for b in range(border):
        a = (b + 1) / border
        mask[b, :] *= a
        mask[h - 1 - b, :] *= a
        mask[:, b] *= a
        mask[:, w - 1 - b] *= a
    return mask


def paste_defects(image, patches, num_paste, scale_range, rng):
    """将缺陷 patch 粘贴到图像上，返回新增的 YOLO 标签行。"""
    img_h, img_w = image.shape[:2]
    new_lines = []
    if len(patches) == 0:
        return new_lines
    selected = [patches[rng.randint(0, len(patches) - 1)]
                for _ in range(num_paste)]
    for info in selected:
        patch = info["patch"]
        ph, pw = patch.shape[:2]
        dx1, dy1, dx2, dy2 = info["box_in_patch"]
        # 随机缩放
        scale = rng.uniform(*scale_range)
        new_pw = max(6, int(pw * scale))
        new_ph = max(6, int(ph * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        patch_s = cv2.resize(patch, (new_pw, new_ph), interpolation=interp)
        if img_w <= new_pw or img_h <= new_ph:
            continue
        x = rng.randint(0, img_w - new_pw)
        y = rng.randint(0, img_h - new_ph)
        # 边缘渐变融合
        border = max(2, min(new_ph, new_pw) // 6)
        mask = _make_feather_mask(new_ph, new_pw, border)[:, :, np.newaxis]
        roi = image[y:y + new_ph, x:x + new_pw].astype(np.float32)
        blended = patch_s.astype(np.float32) * mask + roi * (1.0 - mask)
        image[y:y + new_ph, x:x + new_pw] = np.clip(blended, 0, 255).astype(np.uint8)
        # 计算新缺陷框
        nd_x1 = max(0, min(int(dx1 * scale) + x, img_w))
        nd_y1 = max(0, min(int(dy1 * scale) + y, img_h))
        nd_x2 = max(0, min(int(dx2 * scale) + x, img_w))
        nd_y2 = max(0, min(int(dy2 * scale) + y, img_h))
        if nd_x2 <= nd_x1 + 2 or nd_y2 <= nd_y1 + 2:
            continue
        cx = (nd_x1 + nd_x2) / 2 / img_w
        cy = (nd_y1 + nd_y2) / 2 / img_h
        bw = (nd_x2 - nd_x1) / img_w
        bh = (nd_y2 - nd_y1) / img_h
        new_lines.append(f"{info['cls_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return new_lines


# ═══════════════════ 数据保存 ═══════════════════

def save_sample(image_path, label_lines, split):
    shutil.copy2(image_path, OUT_ROOT / "images" / split / image_path.name)
    (OUT_ROOT / "labels" / split / f"{image_path.stem}.txt").write_text(
        "\n".join(label_lines) + "\n", encoding="utf-8")


def save_augmented(image, label_lines, name, split):
    imwrite_safe(OUT_ROOT / "images" / split / name, image)
    (OUT_ROOT / "labels" / split / f"{Path(name).stem}.txt").write_text(
        "\n".join(label_lines) + "\n", encoding="utf-8")


def group_key(class_name, image_path):
    return class_name, image_path.stem.split("_", 1)[0]


# ═══════════════════ 主流程 ═══════════════════

def main():
    # ── 清理 & 创建目录 ──
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for split in ("train", "val", "test"):
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── 1. XML → YOLO，收集样本 ──
    groups = {}
    for class_name in CLASSES:
        for xml_path in sorted((TRAIN_ROOT / "Annotations" / class_name).glob("*.xml")):
            xml_root = ET.parse(xml_path).getroot()
            image_path = TRAIN_ROOT / "images" / class_name / (
                xml_root.findtext("filename") or f"{xml_path.stem}.jpg")
            if not image_path.exists():
                print(f"[WARNING] 缺少训练图像: {image_path}")
                continue
            key = group_key(class_name, image_path)
            groups.setdefault(key, []).append((image_path, xml_to_yolo_lines(xml_path)))

    # ── 2. 按板分组 train/val 划分 ──
    rng = random.Random(42)
    val_groups = set()
    for class_name in CLASSES:
        class_groups = sorted(k for k in groups if k[0] == class_name)
        rng.shuffle(class_groups)
        n_val = max(1, round(len(class_groups) * 0.2)) if len(class_groups) > 1 else 0
        val_groups.update(class_groups[:n_val])
    train_groups_set = set(groups) - val_groups

    # ── 3. 保存原始样本 ──
    train_samples = []
    train_count = val_count = 0
    for key in sorted(groups):
        split = "val" if key in val_groups else "train"
        for image_path, label_lines in groups[key]:
            save_sample(image_path, label_lines, split)
            if split == "train":
                train_count += 1
                train_samples.append((image_path, label_lines, key))
            else:
                val_count += 1

    # ── 4. 提取缺陷 patch（按板分组） ──
    print("[INFO] 提取缺陷 patch …")
    patches_by_board = {}
    for image_path, label_lines, key in train_samples:
        board_id = key[1]
        image = imread_safe(image_path)
        if image is None:
            continue
        for p in extract_defect_patches(image, label_lines):
            patches_by_board.setdefault(board_id, []).append(p)
    total_patches = sum(len(v) for v in patches_by_board.values())
    print(f"[INFO] 共提取 {total_patches} 个缺陷 patch（来自 {len(patches_by_board)} 块板）")

    # ── 5. Copy-Paste 增强 ──
    print(f"[INFO] 生成增强样本（每张 ×{AUG_COPIES}）…")
    aug_rng = random.Random(123)
    aug_count = 0
    all_patches_flat = [p for ps in patches_by_board.values() for p in ps]
    for image_path, label_lines, key in train_samples:
        board_id = key[1]
        image = imread_safe(image_path)
        if image is None:
            continue
        # 优先选其他板的 patch（打破板级绑定）
        other = [p for bid, ps in patches_by_board.items()
                 if bid != board_id for p in ps]
        if not other:
            other = all_patches_flat
        for ai in range(AUG_COPIES):
            aug_img = image.copy()
            new_lines = paste_defects(
                aug_img, other, PASTE_PER_COPY, SCALE_RANGE, aug_rng)
            all_lines = list(label_lines) + new_lines
            save_augmented(aug_img, all_lines,
                           f"{image_path.stem}_aug{ai}.jpg", "train")
            aug_count += 1
    print(f"[INFO] 增强样本: {aug_count}")

    # ── 6. 测试集（原样复制） ──
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

    # ── 7. dataset.yaml ──
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    (OUT_ROOT / "dataset.yaml").write_text(
        f'path: "{OUT_ROOT.resolve().as_posix()}"\n'
        "train: images/train\n"
        "val: images/val\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )

    print(f"\n[DONE] 增强数据集已就绪")
    print(f"  训练（原始）: {train_count}")
    print(f"  训练（增强）: {aug_count}")
    print(f"  训练（合计）: {train_count + aug_count}")
    print(f"  验证:         {val_count}")
    print(f"  测试:         {test_count}")


if __name__ == "__main__":
    main()

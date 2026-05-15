"""把当前 PCB 数据集整理成 YOLO 格式。"""

import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "训练集-PCB_DATASET"
TEST_ROOT = ROOT / "PCB_瑕疵测试集"
OUT_ROOT = ROOT / "outputs" / "pcb_yolo_dataset"

CLASSES = ["Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]
XML_CLASS_ID = {
    "mouse_bite": 0,
    "open_circuit": 1,
    "short": 2,
    "spur": 3,
    "spurious_copper": 4,
}


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


def save_sample(image_path, label_lines, split):
    shutil.copy2(image_path, OUT_ROOT / "images" / split / image_path.name)
    (OUT_ROOT / "labels" / split / f"{image_path.stem}.txt").write_text(
        "\n".join(label_lines) + "\n",
        encoding="utf-8",
    )


def main():
    for split in ["train", "val", "test"]:
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    samples = []
    for class_name in CLASSES:
        for xml_path in sorted((TRAIN_ROOT / "Annotations" / class_name).glob("*.xml")):
            xml_root = ET.parse(xml_path).getroot()
            image_path = TRAIN_ROOT / "images" / class_name / (xml_root.findtext("filename") or f"{xml_path.stem}.jpg")
            if not image_path.exists():
                print(f"[WARNING] Missing train image: {image_path}")
                continue
            samples.append((image_path, xml_to_yolo_lines(xml_path)))

    random.Random(42).shuffle(samples)
    val_count = round(len(samples) * 0.2)
    for i, (image_path, label_lines) in enumerate(samples):
        save_sample(image_path, label_lines, "val" if i < val_count else "train")

    test_count = 0
    for class_name in CLASSES:
        image_dir = TEST_ROOT / f"{class_name}_Img"
        label_dir = TEST_ROOT / f"{class_name}_txt"
        for label_path in sorted(label_dir.glob("*.txt")):
            if label_path.name.lower() == "classes.txt":
                continue
            image_path = image_dir / f"{label_path.stem}.bmp"
            if not image_path.exists():
                print(f"[WARNING] Missing test image for {label_path.name}, skipped.")
                continue
            shutil.copy2(image_path, OUT_ROOT / "images" / "test" / image_path.name)
            shutil.copy2(label_path, OUT_ROOT / "labels" / "test" / label_path.name)
            test_count += 1

    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    (OUT_ROOT / "dataset.yaml").write_text(
        f'path: "{OUT_ROOT.resolve().as_posix()}"\n'
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )

    print("[DONE] YOLO dataset prepared.")
    print(f"Train images: {len(samples) - val_count}")
    print(f"Val images:   {val_count}")
    print(f"Test images:  {test_count}")


if __name__ == "__main__":
    main()

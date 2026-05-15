"""Prepare PCB defect data for Ultralytics YOLO.

This script reads:
- training images + VOC XML labels from ``训练集-PCB_DATASET``
- test BMP images + YOLO txt labels from ``PCB_瑕疵测试集``

It writes a YOLO-style dataset under ``outputs/pcb_yolo_dataset`` without
modifying the original data.
"""

from __future__ import annotations

import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import yaml


CLASS_NAMES = ["Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]
XML_NAME_TO_ID = {
    "mouse_bite": 0,
    "open_circuit": 1,
    "short": 2,
    "spur": 3,
    "spurious_copper": 4,
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
VAL_RATIO = 0.2
SEED = 42


@dataclass(frozen=True)
class TrainSample:
    image_path: Path
    label_lines: list[str]


@dataclass(frozen=True)
class TestSample:
    image_path: Path
    label_path: Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def voc_xml_to_yolo_lines(xml_path: Path) -> list[str]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    width = int(float(root.findtext("size/width", default="0")))
    height = int(float(root.findtext("size/height", default="0")))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}")

    lines: list[str] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in XML_NAME_TO_ID:
            print(f"[WARNING] Unknown class '{name}' in {xml_path}, skipped.")
            continue

        box = obj.find("bndbox")
        if box is None:
            print(f"[WARNING] Missing bndbox in {xml_path}, skipped.")
            continue

        xmin = float(box.findtext("xmin", default="0"))
        ymin = float(box.findtext("ymin", default="0"))
        xmax = float(box.findtext("xmax", default="0"))
        ymax = float(box.findtext("ymax", default="0"))

        xmin = max(0.0, min(xmin, width))
        xmax = max(0.0, min(xmax, width))
        ymin = max(0.0, min(ymin, height))
        ymax = max(0.0, min(ymax, height))
        if xmax <= xmin or ymax <= ymin:
            print(f"[WARNING] Invalid box in {xml_path}, skipped.")
            continue

        x_center = ((xmin + xmax) / 2.0) / width
        y_center = ((ymin + ymax) / 2.0) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        class_id = XML_NAME_TO_ID[name]
        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )

    return lines


def find_image_for_xml(train_root: Path, class_name: str, filename: str) -> Path | None:
    direct = train_root / "images" / class_name / filename
    if direct.exists():
        return direct

    stem = Path(filename).stem
    image_dir = train_root / "images" / class_name
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def collect_train_samples(train_root: Path) -> list[TrainSample]:
    samples: list[TrainSample] = []
    for class_name in CLASS_NAMES:
        ann_dir = train_root / "Annotations" / class_name
        for xml_path in sorted(ann_dir.glob("*.xml")):
            root = ET.parse(xml_path).getroot()
            filename = root.findtext("filename") or f"{xml_path.stem}.jpg"
            image_path = find_image_for_xml(train_root, class_name, filename)
            if image_path is None:
                print(f"[WARNING] Missing train image for {xml_path}, skipped.")
                continue

            label_lines = voc_xml_to_yolo_lines(xml_path)
            samples.append(TrainSample(image_path=image_path, label_lines=label_lines))
    return samples


def collect_test_samples(test_root: Path) -> list[TestSample]:
    samples: list[TestSample] = []
    for class_name in CLASS_NAMES:
        image_dir = test_root / f"{class_name}_Img"
        label_dir = test_root / f"{class_name}_txt"
        label_files = sorted(path for path in label_dir.glob("*.txt") if path.name.lower() != "classes.txt")

        for label_path in label_files:
            image_path = image_dir / f"{label_path.stem}.bmp"
            if not image_path.exists():
                print(f"[WARNING] Missing test image for {label_path.name}, skipped.")
                continue
            samples.append(TestSample(image_path=image_path, label_path=label_path))
    return samples


def ensure_dirs(output_root: Path) -> None:
    for split in ("train", "val", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_train_sample(sample: TrainSample, output_root: Path, split: str) -> None:
    image_dst = output_root / "images" / split / sample.image_path.name
    label_dst = output_root / "labels" / split / f"{sample.image_path.stem}.txt"
    shutil.copy2(sample.image_path, image_dst)
    label_dst.write_text("\n".join(sample.label_lines) + ("\n" if sample.label_lines else ""), encoding="utf-8")


def copy_test_sample(sample: TestSample, output_root: Path) -> None:
    image_dst = output_root / "images" / "test" / sample.image_path.name
    label_dst = output_root / "labels" / "test" / f"{sample.image_path.stem}.txt"
    shutil.copy2(sample.image_path, image_dst)
    shutil.copy2(sample.label_path, label_dst)


def write_dataset_yaml(output_root: Path) -> Path:
    yaml_path = output_root / "dataset.yaml"
    data = {
        "path": output_root.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: name for idx, name in enumerate(CLASS_NAMES)},
    }
    yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return yaml_path


def main() -> None:
    root = project_root()
    train_root = root / "训练集-PCB_DATASET"
    test_root = root / "PCB_瑕疵测试集"
    output_root = root / "outputs" / "pcb_yolo_dataset"

    if not train_root.exists():
        raise FileNotFoundError(f"Train root not found: {train_root}")
    if not test_root.exists():
        raise FileNotFoundError(f"Test root not found: {test_root}")

    ensure_dirs(output_root)

    train_samples = collect_train_samples(train_root)
    rng = random.Random(SEED)
    rng.shuffle(train_samples)

    val_count = max(1, int(round(len(train_samples) * VAL_RATIO)))
    val_samples = train_samples[:val_count]
    final_train_samples = train_samples[val_count:]

    for sample in final_train_samples:
        copy_train_sample(sample, output_root, "train")
    for sample in val_samples:
        copy_train_sample(sample, output_root, "val")

    test_samples = collect_test_samples(test_root)
    for sample in test_samples:
        copy_test_sample(sample, output_root)

    yaml_path = write_dataset_yaml(output_root)
    print("[DONE] YOLO dataset prepared.")
    print(f"Dataset yaml: {yaml_path}")
    print(f"Train images: {len(final_train_samples)}")
    print(f"Val images:   {len(val_samples)}")
    print(f"Test images:  {len(test_samples)}")


if __name__ == "__main__":
    main()

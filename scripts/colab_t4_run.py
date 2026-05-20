"""Colab T4 一键训练和验收入口。"""

import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import evaluate_yolo, prepare_dataset, train_yolo, visualize_detections  # noqa: E402


def zip_results():
    output_path = ROOT / "outputs" / "colab_t4_results.zip"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    include_roots = [
        ROOT / "outputs" / "eval",
        ROOT / "outputs" / "visualizations",
        ROOT / "runs" / "train" / "pcb_yolo_noaug_ft4" / "weights",
    ]
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zip_file:
        for include_root in include_roots:
            if not include_root.exists():
                continue
            for path in include_root.rglob("*"):
                if path.is_file():
                    zip_file.write(path, path.relative_to(ROOT).as_posix())

    print(f"[DONE] results zip: {output_path}")


def main():
    print(f"[INFO] project root: {ROOT}")
    if train_yolo.DATA_YAML.exists():
        print(f"[INFO] dataset exists, skip prepare: {train_yolo.DATA_YAML}")
    else:
        prepare_dataset.main()
    train_yolo.main()
    evaluate_yolo.main()
    visualize_detections.main()
    zip_results()


if __name__ == "__main__":
    main()

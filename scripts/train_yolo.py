"""训练 PCB 瑕疵 YOLO baseline。"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "outputs" / "pcb_yolo_dataset" / "dataset.yaml"


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError("请先运行 python scripts/prepare_dataset.py")

    from ultralytics import YOLO

    YOLO("yolov8n.pt").train(
        data=str(DATA_YAML),
        imgsz=1024,
        epochs=50,
        batch=4,
        device="0",
        workers=0,
        project=str(ROOT / "runs" / "train"),
        name="pcb_yolo_baseline",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()

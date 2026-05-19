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
        batch=16,
        device="0",
        workers=2,
        cache=True,
        project=str(ROOT / "runs" / "train"),
        name="pcb_yolo_baseline",
        exist_ok=True,
    )

    weights = ROOT / "runs" / "train" / "pcb_yolo_baseline" / "weights" / "best.pt"
    for stage in range(1, 5):
        YOLO(str(weights)).train(
            data=str(DATA_YAML),
            imgsz=1024,
            epochs=5,
            batch=16,
            device="0",
            workers=0,
            cache=True,
            mosaic=0.0,
            scale=0.0,
            translate=0.0,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            fliplr=0.0,
            project=str(ROOT / "runs" / "train"),
            name=f"pcb_yolo_noaug_ft{stage}",
            exist_ok=True,
            plots=False,
        )
        weights = ROOT / "runs" / "train" / f"pcb_yolo_noaug_ft{stage}" / "weights" / "best.pt"


if __name__ == "__main__":
    main()

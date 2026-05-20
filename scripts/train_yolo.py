"""在 Colab T4 上训练 PCB 瑕疵 YOLO 模型。"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "outputs" / "pcb_yolo_dataset" / "dataset.yaml"
BASE_MODEL = "yolov8n.pt"
IMGSZ = 1024
BASE_EPOCHS = 50
FT_EPOCHS = 5
FT_STAGES = 4
BATCH = 0.90
DEVICE = "0"
WORKERS = 2


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError("请先运行 python scripts/prepare_dataset.py")

    from ultralytics import YOLO

    print(f"[INFO] device={DEVICE}, batch={BATCH}, imgsz={IMGSZ}")
    print(f"[INFO] AutoBatch target CUDA memory utilization: {BATCH * 100:.0f}%")

    weights = ROOT / "runs" / "train" / "pcb_yolo_baseline" / "weights" / "best.pt"
    if not weights.exists():
        YOLO(BASE_MODEL).train(
            data=str(DATA_YAML),
            imgsz=IMGSZ,
            epochs=BASE_EPOCHS,
            batch=BATCH,
            device=DEVICE,
            workers=WORKERS,
            cache=True,
            amp=True,
            project=str(ROOT / "runs" / "train"),
            name="pcb_yolo_baseline",
            exist_ok=True,
        )

    weights = ROOT / "runs" / "train" / f"pcb_yolo_noaug_ft{FT_STAGES}" / "weights" / "best.pt"
    if not weights.exists():
        weights = ROOT / "runs" / "train" / "pcb_yolo_baseline" / "weights" / "best.pt"
    for stage in range(1, FT_STAGES + 1):
        stage_weights = ROOT / "runs" / "train" / f"pcb_yolo_noaug_ft{stage}" / "weights" / "best.pt"
        if stage_weights.exists():
            weights = stage_weights
            continue
        YOLO(str(weights)).train(
            data=str(DATA_YAML),
            imgsz=IMGSZ,
            epochs=FT_EPOCHS,
            batch=BATCH,
            device=DEVICE,
            workers=WORKERS,
            cache=True,
            amp=True,
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

    print(f"[DONE] best weights: {weights}")


if __name__ == "__main__":
    main()

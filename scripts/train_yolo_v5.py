"""边缘增强训练（v5）：灰度图，无颜色增强。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "outputs" / "pcb_yolo_dataset_v5" / "dataset.yaml"
MODEL = "yolov8s.pt"
IMGSZ = 640
EPOCHS = 20
BATCH = 8
DEVICE = "0"
WORKERS = 0
RUN_NAME = "pcb_v5_edge"


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError("请先运行 python scripts/prepare_dataset_v5.py")

    from ultralytics import YOLO

    print(f"[INFO] model={MODEL}, imgsz={IMGSZ}, epochs={EPOCHS}, batch={BATCH}")

    YOLO(MODEL).train(
        data=str(DATA_YAML),
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        cache=True,
        # ── 增强（灰度图无需颜色增强） ──
        scale=0.3,
        translate=0.1,
        fliplr=0.5,
        flipud=0.5,
        mosaic=1.0,
        close_mosaic=5,
        mixup=0.1,
        hsv_h=0.0,           # 灰度图，关闭颜色增强
        hsv_s=0.0,
        hsv_v=0.3,           # 仅保留亮度扰动
        # ── 训练调度 ──
        patience=10,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        # ── 输出 ──
        project=str(ROOT / "runs" / "train"),
        name=RUN_NAME,
        exist_ok=True,
    )

    best = ROOT / "runs" / "train" / RUN_NAME / "weights" / "best.pt"
    print(f"[DONE] 最佳权重: {best}")


if __name__ == "__main__":
    main()

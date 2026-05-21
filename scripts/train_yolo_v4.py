"""尺度匹配切片训练（v4）。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "outputs" / "pcb_yolo_dataset_v4" / "dataset.yaml"
MODEL = "yolov8s.pt"
IMGSZ = 640
EPOCHS = 20
BATCH = 8        # 1536 tile 在 GPU 上较大，batch 调小
DEVICE = "0"
WORKERS = 0          # 避免 Windows 多进程内存崩溃
RUN_NAME = "pcb_v4_scale"


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError("请先运行 python scripts/prepare_dataset_v4.py")

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
        # ── 数据增强 ──
        scale=0.3,           # 适度缩放（0.7x~1.3x），避免过度
        translate=0.1,
        fliplr=0.5,
        flipud=0.5,
        mosaic=1.0,
        close_mosaic=5,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=1.0,           # 最大饱和度扰动，去除板间颜色差异
        hsv_v=0.3,
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

"""切片训练：在 640×640 tile 上训练，模型只看局部区域。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "outputs" / "pcb_yolo_dataset_v3" / "dataset.yaml"
MODEL = "yolov8s.pt"
IMGSZ = 640
EPOCHS = 30
BATCH = 32
DEVICE = "0"
WORKERS = 2
RUN_NAME = "pcb_v3_tile"


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError("请先运行 python scripts/prepare_dataset_v3.py")

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
        # ── 数据增强（适度，切片本身已破除板级特征） ──
        scale=0.7,           # 缩放到 0.3x~1.7x，0.3x 使训练缺陷≈22px 匹配测试集
        translate=0.1,
        fliplr=0.5,
        flipud=0.5,
        mosaic=1.0,
        close_mosaic=8,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        # ── 训练调度 ──
        patience=10,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=2,
        # ── 输出 ──
        project=str(ROOT / "runs" / "train"),
        name=RUN_NAME,
        exist_ok=True,
    )

    best = ROOT / "runs" / "train" / RUN_NAME / "weights" / "best.pt"
    print(f"[DONE] 最佳权重: {best}")


if __name__ == "__main__":
    main()

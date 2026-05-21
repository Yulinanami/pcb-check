"""Colab T4 优化版训练脚本。

相比本地 v2 的改动：
- batch 16（利用 T4 的 16GB 显存）
- epochs 80 + patience 20（数据增强充分，收敛更快）
- cache="ram"（Colab 内存充足，用 RAM 缓存加速 I/O）
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "outputs" / "pcb_yolo_dataset_v2" / "dataset.yaml"
MODEL_YAML = "yolov8s-p2.yaml"
PRETRAINED = "yolov8s.pt"
IMGSZ = 1024
EPOCHS = 10
BATCH = 16
DEVICE = "0"
WORKERS = 2
RUN_NAME = "pcb_v2"


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            "请先运行 python scripts/prepare_dataset_v2.py 生成增强数据集")

    from ultralytics import YOLO

    model = YOLO(MODEL_YAML).load(PRETRAINED)

    print(f"[INFO] model={MODEL_YAML}, pretrained={PRETRAINED}")
    print(f"[INFO] imgsz={IMGSZ}, epochs={EPOCHS}, batch={BATCH}")

    model.train(
        data=str(DATA_YAML),
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        cache="ram",
        # ── 强数据增强（与 v2 一致） ──
        scale=0.9,
        translate=0.2,
        fliplr=0.5,
        flipud=0.5,
        mosaic=1.0,
        close_mosaic=10,
        mixup=0.15,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.3,
        # ── 训练调度 ──
        patience=0,            # 短期验证不早停
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

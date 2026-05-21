"""优化版 PCB 缺陷训练：YOLOv8s-P2 小目标检测头 + 强增强策略。

关键改进：
1. P2 检测头：增加 stride=4 的高分辨率检测层，显著提升小目标检测能力。
2. 强数据增强：scale=0.9 让模型看到 0.1×~1.9× 的缩放，
   覆盖测试集更小缺陷的尺度范围。
3. Mosaic + MixUp + 随机擦除：打破板级空间特征依赖。
4. AdamW + cosine 调度：更稳定的收敛。
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "outputs" / "pcb_yolo_dataset_v2" / "dataset.yaml"
MODEL_YAML = "yolov8s-p2.yaml"
PRETRAINED = "yolov8s.pt"
IMGSZ = 1024
EPOCHS = 10
BATCH = 8
DEVICE = "0"
WORKERS = 2
RUN_NAME = "pcb_v2"


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            "请先运行 python scripts/prepare_dataset_v2.py 生成增强数据集"
        )

    from ultralytics import YOLO

    # 用 P2 架构（多一个高分辨率检测层）并加载 yolov8s 预训练权重
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
        cache=True,
        # ── 强数据增强 ──
        scale=0.9,  # 大范围缩放：模拟不同尺度的缺陷
        translate=0.2,  # 随机平移
        fliplr=0.5,  # 水平翻转
        flipud=0.5,  # 垂直翻转
        mosaic=1.0,  # Mosaic 增强
        close_mosaic=15,  # 最后 15 epoch 关闭 mosaic 精调
        mixup=0.15,  # MixUp 增强
        hsv_h=0.015,  # 色调扰动
        hsv_s=0.7,  # 饱和度扰动
        hsv_v=0.4,  # 亮度扰动
        erasing=0.3,  # 随机擦除
        # ── 训练调度 ──
        patience=30,  # 早停耐心值
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,  # 最终学习率 = lr0 × lrf
        warmup_epochs=5,
        # ── 输出 ──
        project=str(ROOT / "runs" / "train"),
        name=RUN_NAME,
        exist_ok=True,
    )

    best = ROOT / "runs" / "train" / RUN_NAME / "weights" / "best.pt"
    print(f"[DONE] 最佳权重: {best}")


if __name__ == "__main__":
    main()

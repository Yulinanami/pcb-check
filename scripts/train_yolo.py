"""训练 PCB 瑕疵 YOLO 模型。"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "outputs" / "pcb_yolo_dataset" / "dataset.yaml"
MODEL = "yolov8s.pt"
IMGSZ = 1280
EPOCHS = 50
BATCH = 8
DEVICE = "0"
WORKERS = 2
RUN_NAME = "pcb_yolo_true_train"


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError("请先运行 python scripts/prepare_dataset.py")
    data_yaml_text = DATA_YAML.read_text(encoding="utf-8")
    if any(line.strip().startswith("test:") for line in data_yaml_text.splitlines()):
        raise RuntimeError("训练配置不能包含测试集 test: 字段")

    from ultralytics import YOLO

    print(f"[INFO] model={MODEL}, imgsz={IMGSZ}, epochs={EPOCHS}, batch={BATCH}, device={DEVICE}")
    YOLO(MODEL).train(
        data=str(DATA_YAML),
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        cache=True,
        project=str(ROOT / "runs" / "train"),
        name=RUN_NAME,
        exist_ok=True,
    )
    print(f"[DONE] best weights: {ROOT / 'runs' / 'train' / RUN_NAME / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()


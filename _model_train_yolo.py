import argparse
import os
from pathlib import Path
import yaml
from ultralytics import YOLO


def update_yaml_paths(yaml_path):
    """
    Ensure that train/val/test paths in data.yaml are absolute by prepending the base directory.
    Writes out a temporary data_temp.yaml next to the original and returns its path.
    """
    yaml_path = Path(yaml_path)
    base_dir = yaml_path.parent

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    for key in ('train', 'val', 'test'):
        if key in data:
            path_val = Path(data[key])
            if not path_val.is_absolute():
                data[key] = str((base_dir / path_val).resolve())

    temp_yaml_path = base_dir / "data_temp.yaml"
    with open(temp_yaml_path, "w") as f:
        yaml.dump(data, f)

    print(f"Updated YAML saved to: {temp_yaml_path}")
    return str(temp_yaml_path)


def train_yolo(
    data_yaml: str,
    epochs: int = 50,
    batch: int = 16,
    device: str = "0",
    model_size: str = "n",
    output_dir: str = "runs/train"
):
    """
    Train a YOLOv8 segmentation model of the specified size.

    Args:
        data_yaml: Path to the data.yaml file.
        epochs: Number of epochs to train.
        batch: Batch size.
        device: CUDA device index (e.g. '0') or 'cpu'.
        model_size: One of 'n', 's', 'm', 'l', 'x', or 'xl'. 
                    Maps to pretrained segment checkpoints (yolov8<n>-seg.pt, etc.).
        output_dir: Parent directory under which the run folder will be created.
    """
    size_map = {
        "n": "yolov8n-seg.pt",
        "s": "yolov8s-seg.pt",
        "m": "yolov8m-seg.pt",
        "l": "yolov8l-seg.pt",
        "x": "yolov8x-seg.pt",
        "xl": "yolov8x-seg.pt",  # 'x' and 'xl' use the same checkpoint
    }

    if model_size not in size_map:
        raise ValueError(f"Unknown model_size '{model_size}'. Choose from n, s, m, l, x, xl.")

    base_checkpoint = size_map[model_size]
    print(f"[YOLO] Using base checkpoint: {base_checkpoint} for model_size '{model_size}'")

    updated_yaml = update_yaml_paths(data_yaml)

    # Load the YOLO model (segmentation)
    model = YOLO(base_checkpoint)

    # Train
    model.train(
        data=updated_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=640,
        device=device,
        name=f"yolov8{model_size}-seg-train",
        project=output_dir,
        exist_ok=True
    )

    print(f"Finished training yolov8{model_size}. Weights: {os.path.join(output_dir, f'yolov8{model_size}-seg-train/weights/best.pt')}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 Nano (n) and X‑Large (x) segmentation models on a custom dataset."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to data.yaml (with train/val paths, nc, names)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs per model (default: 50)."
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)."
    )
    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device index (e.g. '0') or 'cpu' (default: '0')."
    )
    parser.add_argument(
        "--output-dir",
        default="runs/train",
        help="Directory under which YOLO run folders will be created (default: runs/train)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Train YOLOv8 Nano segmentation ("n")
    train_yolo(
        data_yaml=args.data,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        model_size="n",
        output_dir=args.output_dir
    )

    # Train YOLOv8 X‑Large segmentation ("x")
    train_yolo(
        data_yaml=args.data,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        model_size="x",
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

    '''
    python _modelTrain.py \
        --data powder/data_temp.yaml \
        --epochs 50 \
        --batch 16 \
        --device 0 \
        --output-dir runs/train
    '''

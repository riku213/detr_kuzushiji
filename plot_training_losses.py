import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser("Plot training/eval losses from DETR log.txt")
    parser.add_argument("--log_file", type=str, required=True, help="Path to log.txt")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--show", action="store_true", help="Show plot window")
    return parser.parse_args()


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}")
    return records


def main():
    args = parse_args()
    log_path = Path(args.log_file)
    if not log_path.exists():
        raise ValueError(f"log file does not exist: {log_path}")

    records = read_jsonl(log_path)
    if len(records) == 0:
        raise ValueError("No records found in log file")

    epochs = [r.get("epoch") for r in records]
    train_loss = [r.get("train_loss") for r in records]
    test_loss = [r.get("test_loss") for r in records]

    fig, ax = plt.subplots(figsize=(10, 5))

    if any(v is not None for v in train_loss):
        x = [e for e, v in zip(epochs, train_loss) if e is not None and v is not None]
        y = [v for e, v in zip(epochs, train_loss) if e is not None and v is not None]
        if len(x) > 0:
            ax.plot(x, y, marker="o", linewidth=1.5, label="train_loss")

    if any(v is not None for v in test_loss):
        x = [e for e, v in zip(epochs, test_loss) if e is not None and v is not None]
        y = [v for e, v in zip(epochs, test_loss) if e is not None and v is not None]
        if len(x) > 0:
            ax.plot(x, y, marker="s", linewidth=1.5, label="test_loss")

    ax.set_title("Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = Path(args.output) if args.output else log_path.parent / "loss_curves.png"
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot: {output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

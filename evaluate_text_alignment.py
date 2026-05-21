import argparse
from collections import defaultdict
from pathlib import Path

import torch

from datasets import build_dataset
from models import build_model
from util import box_ops


def parse_args():
    parser = argparse.ArgumentParser("Evaluate text-aligned DETR predictions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a checkpoint (.pth)")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Dataset split")
    parser.add_argument("--output", type=str, required=True, help="Path to the output text file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")

    # Optional overrides if the checkpoint args are missing or outdated.
    parser.add_argument("--dataset_file", type=str, default=None)
    parser.add_argument("--kuzushiji_path", type=str, default=None)
    parser.add_argument("--kuzushiji_resize_short", type=int, default=None)
    parser.add_argument("--kuzushiji_resize_max_size", type=int, default=None)
    parser.add_argument("--kuzushiji_use_crop_grid", type=bool, default=None)
    parser.add_argument("--kuzushiji_grid_size", type=int, default=None)
    return parser.parse_args()


def ensure_checkpoint_args(train_args, cli_args):
    defaults = {
        "device": cli_args.device,
        "dataset_file": "kuzushiji_text",
        "use_text_queries": True,
        "bbox_only": True,
        "masks": False,
        "aux_loss": True,
        "num_queries": 100,
        "backbone": "resnet50",
        "dilation": False,
        "position_embedding": "sine",
        "enc_layers": 6,
        "dec_layers": 6,
        "dim_feedforward": 2048,
        "hidden_dim": 256,
        "dropout": 0.1,
        "nheads": 8,
        "pre_norm": False,
        "set_cost_class": 1.0,
        "set_cost_bbox": 5.0,
        "set_cost_giou": 2.0,
        "mask_loss_coef": 1.0,
        "dice_loss_coef": 1.0,
        "bbox_loss_coef": 5.0,
        "giou_loss_coef": 2.0,
        "eos_coef": 0.1,
        "lr_backbone": 1e-5,
        "text_vocab_size": 65536,
        "text_pad_token_id": 0,
        "text_max_len": 512,
        "text_encoder_layers": 1,
        "kuzushiji_split_ratio": 0.8,
        "kuzushiji_max_samples": None,
        "kuzushiji_sort_tokens": False,
        "kuzushiji_resize_short": 640,
        "kuzushiji_resize_max_size": 1024,
        "kuzushiji_use_crop_grid": True,
        "kuzushiji_grid_size": 4,
    }

    for key, value in defaults.items():
        if not hasattr(train_args, key):
            setattr(train_args, key, value)

    train_args.device = cli_args.device
    if cli_args.dataset_file is not None:
        train_args.dataset_file = cli_args.dataset_file
    if cli_args.kuzushiji_path is not None:
        train_args.kuzushiji_path = cli_args.kuzushiji_path
    if cli_args.kuzushiji_resize_short is not None:
        train_args.kuzushiji_resize_short = cli_args.kuzushiji_resize_short
    if cli_args.kuzushiji_resize_max_size is not None:
        train_args.kuzushiji_resize_max_size = cli_args.kuzushiji_resize_max_size
    if cli_args.kuzushiji_use_crop_grid is not None:
        train_args.kuzushiji_use_crop_grid = cli_args.kuzushiji_use_crop_grid
    if cli_args.kuzushiji_grid_size is not None:
        train_args.kuzushiji_grid_size = cli_args.kuzushiji_grid_size

    if train_args.dataset_file == "kuzushiji_text":
        train_args.use_text_queries = True
        train_args.bbox_only = True

    return train_args


def is_dummy_target(gt_boxes):
    if gt_boxes.numel() == 0:
        return True
    if gt_boxes.shape[0] != 1:
        return False
    width = float(gt_boxes[0, 2].item())
    height = float(gt_boxes[0, 3].item())
    return width <= 1e-5 and height <= 1e-5


def compute_ious(pred_boxes, gt_boxes):
    pred_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes)
    gt_xyxy = box_ops.box_cxcywh_to_xyxy(gt_boxes)
    iou_matrix, _ = box_ops.box_iou(pred_xyxy, gt_xyxy)
    pair_count = min(iou_matrix.shape[0], iou_matrix.shape[1])
    if pair_count == 0:
        return []
    return [float(iou_matrix[i, i].item()) for i in range(pair_count)]


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise ValueError(f"checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "args" not in checkpoint:
        raise ValueError("checkpoint does not contain training args")

    train_args = ensure_checkpoint_args(checkpoint["args"], args)
    if train_args.dataset_file == "kuzushiji_text" and getattr(train_args, "kuzushiji_path", None) is None:
        raise ValueError("kuzushiji_path is required. Pass --kuzushiji_path.")

    device = torch.device(train_args.device)
    model, _, _ = build_model(train_args)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()

    dataset = build_dataset(image_set=args.split, args=train_args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    thresholds = [0.5, 0.75]
    threshold_stats = {threshold: defaultdict(int) for threshold in thresholds}
    total_iou = 0.0
    total_pairs = 0
    total_gt_boxes = 0
    total_pred_boxes = 0
    total_samples = 0
    skipped_dummy_samples = 0

    with torch.no_grad():
        for idx in range(len(dataset)):
            image_tensor, target = dataset[idx]
            gt_boxes = target["boxes_aligned"].cpu()

            if is_dummy_target(gt_boxes):
                skipped_dummy_samples += 1
                continue

            sample = image_tensor.unsqueeze(0).to(device)
            text_inputs = [target["token_ids"].to(device)]
            outputs = model(sample, text_inputs=text_inputs)

            pred_boxes = outputs["pred_boxes"][0].detach().cpu()
            ious = compute_ious(pred_boxes, gt_boxes)
            matched_count = len(ious)

            total_samples += 1
            total_pairs += matched_count
            total_iou += sum(ious)
            total_gt_boxes += int(gt_boxes.shape[0])
            total_pred_boxes += int(pred_boxes.shape[0])

            for threshold in thresholds:
                tp = sum(1 for iou in ious if iou >= threshold)
                threshold_stats[threshold]["tp"] += tp
                threshold_stats[threshold]["fp"] += max(0, int(pred_boxes.shape[0]) - tp)
                threshold_stats[threshold]["fn"] += max(0, int(gt_boxes.shape[0]) - tp)
                threshold_stats[threshold]["matched"] += matched_count

    mean_iou = safe_divide(total_iou, total_pairs)

    lines = []
    lines.append(f"Checkpoint: {checkpoint_path}")
    lines.append(f"Split: {args.split}")
    lines.append(f"Dataset size: {len(dataset)}")
    lines.append(f"Evaluated samples: {total_samples}")
    lines.append(f"Skipped dummy samples: {skipped_dummy_samples}")
    lines.append(f"Total GT boxes: {total_gt_boxes}")
    lines.append(f"Total predicted boxes: {total_pred_boxes}")
    lines.append(f"Mean IoU: {mean_iou:.6f}")
    lines.append("")

    for threshold in thresholds:
        stats = threshold_stats[threshold]
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        matched = stats["matched"]
        accuracy = safe_divide(tp, matched)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = 0.0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

        lines.append(f"Threshold IoU >= {threshold:.2f}")
        lines.append(f"  Accuracy@{threshold:.2f}: {accuracy:.6f}")
        lines.append(f"  Precision@{threshold:.2f}: {precision:.6f}")
        lines.append(f"  Recall@{threshold:.2f}: {recall:.6f}")
        lines.append(f"  F1@{threshold:.2f}: {f1:.6f}")
        lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Saved metrics: {output_path}")


if __name__ == "__main__":
    main()
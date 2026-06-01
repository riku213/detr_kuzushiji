import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from datasets import build_dataset
from models import build_model
from util import box_ops


def parse_args():
    parser = argparse.ArgumentParser("Visualize DETR predictions on input images")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint (.pth)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"],
                        help="Dataset split to visualize")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to visualize")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Dataset start index")
    parser.add_argument("--output_dir", type=str, default="outputs/visualizations",
                        help="Directory to save rendered images")
    parser.add_argument("--score_threshold", type=float, default=0.5,
                        help="Score threshold for drawing predicted boxes")

    # Optional overrides if checkpoint args are missing or outdated.
    parser.add_argument("--dataset_file", type=str, default=None)
    parser.add_argument("--kuzushiji_path", type=str, default=None)
    parser.add_argument("--kuzushiji_resize_short", type=int, default=None)
    parser.add_argument("--kuzushiji_resize_max_size", type=int, default=None)
    parser.add_argument("--kuzushiji_use_crop_grid", type=parse_bool, default=None)
    parser.add_argument("--kuzushiji_grid_size", type=int, default=None)
    return parser.parse_args()


def parse_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def ensure_checkpoint_args(train_args, cli_args):
    # Fill args that may not exist in old checkpoints.
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

    for k, v in defaults.items():
        if not hasattr(train_args, k):
            setattr(train_args, k, v)

    # Apply CLI overrides.
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
    elif train_args.dataset_file == "kuzushiji_det":
        train_args.use_text_queries = False
        train_args.bbox_only = False

    return train_args


def tensor_to_pil(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image_tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image_tensor.dtype).view(3, 1, 1)
    x = image_tensor.cpu() * std + mean
    x = x.clamp(0.0, 1.0)
    x = (x * 255.0).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def boxes_cxcywh_to_xyxy_abs(boxes, h, w):
    scale = torch.tensor([w, h, w, h], dtype=boxes.dtype)
    xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * scale
    return xyxy


def draw_boxes(image, gt_boxes_xyxy, pred_boxes_xyxy, gt_labels=None, pred_labels=None, pred_scores=None):
    draw = ImageDraw.Draw(image)

    for i, b in enumerate(gt_boxes_xyxy.tolist()):
        x1, y1, x2, y2 = b
        draw.rectangle([x1, y1, x2, y2], outline=(50, 220, 50), width=2)
        label = f"gt:{i}"
        if gt_labels is not None and i < len(gt_labels):
            label = f"gt:{gt_labels[i]}"
        draw.text((x1, max(0, y1 - 12)), label, fill=(50, 220, 50))

    for i, b in enumerate(pred_boxes_xyxy.tolist()):
        x1, y1, x2, y2 = b
        draw.rectangle([x1, y1, x2, y2], outline=(220, 50, 50), width=2)
        label = f"pred:{i}"
        if pred_labels is not None and i < len(pred_labels):
            label = f"pred:{pred_labels[i]}"
        if pred_scores is not None and i < len(pred_scores):
            label = f"{label}({pred_scores[i]:.2f})"
        draw.text((x1, y1 + 2), label, fill=(220, 50, 50))

    return image


def format_label_names(label_ids, label_to_char):
    if label_to_char is None:
        return [str(int(i)) for i in label_ids]
    names = []
    for idx in label_ids:
        idx = int(idx)
        if 0 <= idx < len(label_to_char):
            names.append(label_to_char[idx])
        else:
            names.append(str(idx))
    return names


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise ValueError(f"checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "args" not in checkpoint:
        raise ValueError("checkpoint does not contain training args")

    train_args = ensure_checkpoint_args(checkpoint["args"], args)

    if train_args.dataset_file in {"kuzushiji_text", "kuzushiji_det"} and getattr(train_args, "kuzushiji_path", None) is None:
        raise ValueError("kuzushiji_path is required. Pass --kuzushiji_path.")

    device = torch.device(train_args.device)

    dataset = build_dataset(image_set=args.split, args=train_args)
    if hasattr(dataset, "num_classes"):
        train_args.num_classes = dataset.num_classes

    model, _, _ = build_model(train_args)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()

    label_to_char = getattr(dataset, "label_to_char", None)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    end_index = min(len(dataset), args.start_index + args.num_samples)
    if args.start_index >= len(dataset):
        raise ValueError(f"start_index={args.start_index} is out of range for dataset size {len(dataset)}")

    with torch.no_grad():
        for idx in range(args.start_index, end_index):
            image_tensor, target = dataset[idx]
            sample = image_tensor.unsqueeze(0).to(device)

            if getattr(train_args, "use_text_queries", False):
                if "token_ids" not in target:
                    raise ValueError("target does not include token_ids required by text-query model")
                text_inputs = [target["token_ids"].to(device)]
                outputs = model(sample, text_inputs=text_inputs)
            else:
                outputs = model(sample)

            pred_boxes = outputs["pred_boxes"][0].detach().cpu()
            pred_labels = None
            pred_scores = None
            if "pred_logits" in outputs:
                probs = outputs["pred_logits"][0].softmax(-1)
                scores, labels = probs[..., :-1].max(-1)
                keep = scores >= args.score_threshold
                pred_boxes = pred_boxes[keep].cpu()
                pred_scores = scores[keep].cpu().tolist()
                pred_labels = format_label_names(labels[keep].cpu().tolist(), label_to_char)

            valid_n = None
            if "token_ids" in target:
                valid_n = int(target["token_ids"].shape[0])
                pred_boxes = pred_boxes[:valid_n]

            print('-'*30)
            if valid_n is None:
                print(f"Sample {idx}: predicted: {pred_boxes.shape[0]}")
            else:
                print(f"Sample {idx}: predicted: {pred_boxes.shape[0]} valid_n: {valid_n}")
            if pred_boxes.numel() > 0:
                print("pred cxcywh mean:", pred_boxes.mean(dim=0).tolist())
                print("pred cxcywh std :", pred_boxes.std(dim=0).tolist())
            if "boxes_aligned" in target:
                gt_boxes = target["boxes_aligned"].cpu()
            elif "boxes" in target:
                gt_boxes = target["boxes"].cpu()
            else:
                raise ValueError("target must include boxes_aligned or boxes")

            gt_labels = None
            if "labels" in target:
                gt_labels = format_label_names(target["labels"].cpu().tolist(), label_to_char)

            h = int(target["size"][0].item())
            w = int(target["size"][1].item())

            gt_xyxy = boxes_cxcywh_to_xyxy_abs(gt_boxes, h, w)
            pred_xyxy = boxes_cxcywh_to_xyxy_abs(pred_boxes, h, w)

            vis_image = tensor_to_pil(image_tensor)
            vis_image = draw_boxes(
                vis_image,
                gt_xyxy,
                pred_xyxy,
                gt_labels=gt_labels,
                pred_labels=pred_labels,
                pred_scores=pred_scores,
            )

            save_path = out_dir / f"sample_{idx:05d}.png"
            vis_image.save(save_path)
            print(f"saved: {save_path}")


if __name__ == "__main__":
    main()

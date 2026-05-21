import csv
import os
from pathlib import Path

import torch
from PIL import Image

import datasets.transforms as T


def make_kuzushiji_transforms(image_set, resize_short=640, max_size=1024):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if image_set == "train":
        return T.Compose([
            T.RandomResize([resize_short], max_size=max_size),
            normalize,
        ])

    if image_set == "val":
        return T.Compose([
            T.RandomResize([resize_short], max_size=max_size),
            normalize,
        ])

    raise ValueError(f"unknown split {image_set}")


def compute_crop_grid(img_height, img_width, grid_size=4):
    """
    Compute grid cells for cropping.
    
    Args:
        img_height: Full image height in pixels
        img_width: Full image width in pixels
        grid_size: Number of grid divisions (default: 4 for 4x4 grid)
    
    Returns:
        List of (x1, y1, x2, y2) for each grid cell (0-indexed, pixel coords)
    """
    cell_height = img_height / grid_size
    cell_width = img_width / grid_size
    
    crops = []
    for row in range(grid_size):
        for col in range(grid_size):
            x1 = int(col * cell_width)
            y1 = int(row * cell_height)
            x2 = int((col + 1) * cell_width)
            y2 = int((row + 1) * cell_height)
            crops.append((x1, y1, x2, y2))
    
    return crops


def bbox_intersects_crop(x, y, w, h, crop_box):
    """
    Check if a bounding box (in pixel coords) intersects with a crop region.
    
    Args:
        x, y, w, h: Bounding box in pixel coordinates (x, y, width, height)
        crop_box: Tuple (x1, y1, x2, y2) of crop region in pixel coordinates
    
    Returns:
        True if bbox intersects crop region
    """
    x1_crop, y1_crop, x2_crop, y2_crop = crop_box
    
    x2_bbox = x + w
    y2_bbox = y + h
    
    # Check for intersection
    if x2_bbox < x1_crop or x > x2_crop or y2_bbox < y1_crop or y > y2_crop:
        return False
    return True


def get_chars_in_crop(items, crop_box):
    """
    Get all characters that intersect with the crop region.
    
    Args:
        items: List of dicts with keys 'char', 'x', 'y', 'w', 'h'
        crop_box: Tuple (x1, y1, x2, y2) of crop region
    
    Returns:
        List of (index, item) for characters intersecting the crop
    """
    chars_in_crop = []
    for idx, item in enumerate(items):
        if bbox_intersects_crop(item["x"], item["y"], item["w"], item["h"], crop_box):
            chars_in_crop.append((idx, item))
    return chars_in_crop


def transform_bbox_to_crop_coords(x, y, w, h, crop_box, crop_w, crop_h):
    """
    Transform a bounding box from full image coordinates to cropped image coordinates.
    Then normalize to [0, 1] range.
    
    Args:
        x, y, w, h: BBox in pixel coordinates (full image)
        crop_box: Tuple (x1_crop, y1_crop, x2_crop, y2_crop) in pixel coordinates
        crop_w: Width of crop in pixels
        crop_h: Height of crop in pixels
    
    Returns:
        Tuple (cx_norm, cy_norm, w_norm, h_norm) in normalized coordinates [0, 1]
        Values may be clipped to [0, 1] for partially visible bboxes
    """
    x1_crop, y1_crop, x2_crop, y2_crop = crop_box
    
    # Clip bbox to crop boundaries
    x1_clipped = max(x, x1_crop)
    y1_clipped = max(y, y1_crop)
    x2_clipped = min(x + w, x2_crop)
    y2_clipped = min(y + h, y2_crop)
    
    # Convert to crop-local coordinates
    x1_local = x1_clipped - x1_crop
    y1_local = y1_clipped - y1_crop
    x2_local = x2_clipped - x1_crop
    y2_local = y2_clipped - y1_crop
    
    # Compute width and height in crop space
    w_local = x2_local - x1_local
    h_local = y2_local - y1_local
    
    # Normalize to [0, 1]
    cx_norm = (x1_local + x2_local) * 0.5 / crop_w
    cy_norm = (y1_local + y2_local) * 0.5 / crop_h
    w_norm = w_local / crop_w
    h_norm = h_local / crop_h
    
    # Clamp to [0, 1] for partially visible bboxes
    cx_norm = max(0.0, min(cx_norm, 1.0))
    cy_norm = max(0.0, min(cy_norm, 1.0))
    w_norm = max(0.0, min(w_norm, 1.0))
    h_norm = max(0.0, min(h_norm, 1.0))
    
    return cx_norm, cy_norm, w_norm, h_norm


class KuzushijiTextDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train", split_ratio=0.8, seed=42, max_samples=None,
                 vocab_size=65536, sort_tokens=False, resize_short=640, resize_max_size=1024,
                 use_crop_grid=True, grid_size=4):
        self.root = Path(root)
        if not self.root.exists():
            raise ValueError(f"provided dataset path {self.root} does not exist")

        self.vocab_size = int(vocab_size)
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.sort_tokens = sort_tokens
        self.use_crop_grid = use_crop_grid
        self.grid_size = grid_size

        samples = self._build_samples()
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(samples), generator=g).tolist()
        split_index = int(len(samples) * split_ratio)
        if split == "train":
            ids = perm[:split_index]
        elif split == "val":
            ids = perm[split_index:]
        else:
            raise ValueError(f"unknown split {split}")

        self.samples = [samples[i] for i in ids]
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        self._transforms = make_kuzushiji_transforms(
            split,
            resize_short=resize_short,
            max_size=resize_max_size,
        )

    def _codepoint_to_char(self, codepoint):
        if isinstance(codepoint, str) and codepoint.startswith("U+"):
            return chr(int(codepoint[2:], 16))
        return str(codepoint)

    def _char_to_token_id(self, ch):
        code = ord(ch)
        if code >= self.vocab_size:
            return self.unk_token_id
        return code

    def _build_image_index(self):
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        image_index = {}
        for doc_folder in os.listdir(self.root):
            images_dir = self.root / doc_folder / "images"
            if not images_dir.is_dir():
                continue
            for fname in os.listdir(images_dir):
                ext = Path(fname).suffix.lower()
                if ext not in exts:
                    continue
                key = Path(fname).stem
                image_index[key] = str(images_dir / fname)
        return image_index

    def _build_samples(self):
        image_index = self._build_image_index()
        rows_by_image = {}

        for doc_folder in os.listdir(self.root):
            doc_path = self.root / doc_folder
            if not doc_path.is_dir():
                continue
            for filename in os.listdir(doc_path):
                if not filename.lower().endswith(".csv"):
                    continue
                csv_path = doc_path / filename
                with open(csv_path, "r", encoding="utf-8", newline="") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if not row or len(row) < 8:
                            continue
                        if not row[0].startswith("U+"):
                            continue

                        image_id = row[1]
                        image_path = image_index.get(image_id)
                        if image_path is None:
                            continue

                        try:
                            x = int(row[2])
                            y = int(row[3])
                            w = int(row[6])
                            h = int(row[7])
                        except ValueError:
                            continue

                        ch = self._codepoint_to_char(row[0])
                        rows_by_image.setdefault(image_id, {
                            "image_path": image_path,
                            "items": [],
                        })
                        rows_by_image[image_id]["items"].append({
                            "char": ch,
                            "x": x,
                            "y": y,
                            "w": w,
                            "h": h,
                        })

        samples = []
        for image_id, data in rows_by_image.items():
            items = data["items"]
            if self.sort_tokens:
                items.sort(key=lambda r: (-r["x"], r["y"]))

            tokens = [self._char_to_token_id(it["char"]) for it in items]
            boxes_xywh = [[it["x"], it["y"], it["w"], it["h"]] for it in items]
            if len(tokens) == 0:
                continue

            samples.append({
                "image_id": image_id,
                "image_path": data["image_path"],
                "token_ids": tokens,
                "boxes_xywh": boxes_xywh,
            })
        return samples

    def __len__(self):
        if self.use_crop_grid:
            return len(self.samples) * (self.grid_size * self.grid_size)
        else:
            return len(self.samples)

    def __getitem__(self, idx):
        if self.use_crop_grid:
            # For cropped version: each sample is multiplied by grid_size^2
            sample_idx = idx // (self.grid_size * self.grid_size)
            crop_idx = idx % (self.grid_size * self.grid_size)
            return self._getitem_with_crop(sample_idx, crop_idx)
        else:
            return self._getitem_no_crop(idx)

    def _getitem_no_crop(self, idx):
        """Get item without cropping (original implementation)"""
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        width, height = image.size

        tokens = []
        boxes = []
        for token_id, (x, y, w, h) in zip(sample["token_ids"], sample["boxes_xywh"]):
            x = max(0.0, min(float(x), width - 1.0))
            y = max(0.0, min(float(y), height - 1.0))
            x2 = max(0.0, min(x + float(w), width))
            y2 = max(0.0, min(y + float(h), height))
            ww = x2 - x
            hh = y2 - y
            if ww <= 0.0 or hh <= 0.0:
                continue

            cx = (x + x2) * 0.5 / width
            cy = (y + y2) * 0.5 / height
            nw = ww / width
            nh = hh / height
            boxes.append([cx, cy, nw, nh])
            tokens.append(token_id)

        if len(tokens) == 0:
            tokens = [self.unk_token_id]
            boxes = [[0.5, 0.5, 1e-6, 1e-6]]

        target = {
            "image_id": torch.tensor([idx]),
            "boxes_aligned": torch.tensor(boxes, dtype=torch.float32),
            "token_ids": torch.tensor(tokens, dtype=torch.long),
            "orig_size": torch.tensor([int(height), int(width)]),
            "size": torch.tensor([int(height), int(width)]),
        }

        if self._transforms is not None:
            image, target = self._transforms(image, target)

        return image, target

    def _getitem_with_crop(self, sample_idx, crop_idx):
        """Get item with cropping applied"""
        sample = self.samples[sample_idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        width, height = image.size

        # Compute crop grid
        crops = compute_crop_grid(height, width, self.grid_size)
        crop_box = crops[crop_idx]
        x1_crop, y1_crop, x2_crop, y2_crop = crop_box
        crop_w = x2_crop - x1_crop
        crop_h = y2_crop - y1_crop

        # Crop image
        image_cropped = image.crop((x1_crop, y1_crop, x2_crop, y2_crop))

        # Get characters that intersect this crop
        items = []
        for i, (x, y, w, h) in enumerate(sample["boxes_xywh"]):
            items.append({
                "idx": i,
                "char": sample["token_ids"][i],  # Store token_id directly
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            })

        chars_in_crop = get_chars_in_crop(items, crop_box)

        # Transform bboxes to crop coordinates and normalize
        tokens = []
        boxes = []
        for orig_idx, item in chars_in_crop:
            token_id = item["char"]
            x, y, w, h = item["x"], item["y"], item["w"], item["h"]

            # Transform bbox to crop coordinates
            cx_norm, cy_norm, w_norm, h_norm = transform_bbox_to_crop_coords(
                x, y, w, h, crop_box, crop_w, crop_h
            )

            # Skip if bbox has no visible area
            if w_norm <= 0.0 or h_norm <= 0.0:
                continue

            boxes.append([cx_norm, cy_norm, w_norm, h_norm])
            tokens.append(token_id)

        # Handle empty crop
        if len(tokens) == 0:
            tokens = [self.unk_token_id]
            boxes = [[0.5, 0.5, 1e-6, 1e-6]]

        # Make image_id unique per crop to avoid overwriting results during evaluation
        unique_image_id = sample_idx * (self.grid_size * self.grid_size) + crop_idx
        target = {
            "image_id": torch.tensor([unique_image_id]),
            "boxes_aligned": torch.tensor(boxes, dtype=torch.float32),
            "token_ids": torch.tensor(tokens, dtype=torch.long),
            "orig_size": torch.tensor([int(crop_h), int(crop_w)]),
            "size": torch.tensor([int(crop_h), int(crop_w)]),
        }

        if self._transforms is not None:
            image_cropped, target = self._transforms(image_cropped, target)

        return image_cropped, target


def build(image_set, args):
    root = getattr(args, "kuzushiji_path", None)
    if root is None:
        raise ValueError("--kuzushiji_path is required for dataset_file=kuzushiji_text")

    return KuzushijiTextDataset(
        root=root,
        split=image_set,
        split_ratio=getattr(args, "kuzushiji_split_ratio", 0.8),
        seed=getattr(args, "seed", 42),
        max_samples=getattr(args, "kuzushiji_max_samples", None),
        vocab_size=getattr(args, "text_vocab_size", 65536),
        sort_tokens=getattr(args, "kuzushiji_sort_tokens", False),
        resize_short=getattr(args, "kuzushiji_resize_short", 640),
        resize_max_size=getattr(args, "kuzushiji_resize_max_size", 1024),
        use_crop_grid=getattr(args, "kuzushiji_use_crop_grid", True),
        grid_size=getattr(args, "kuzushiji_grid_size", 4),
    )

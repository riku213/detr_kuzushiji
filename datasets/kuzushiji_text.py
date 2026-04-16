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


class KuzushijiTextDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train", split_ratio=0.8, seed=42, max_samples=None,
                 vocab_size=65536, sort_tokens=False, resize_short=640, resize_max_size=1024):
        self.root = Path(root)
        if not self.root.exists():
            raise ValueError(f"provided dataset path {self.root} does not exist")

        self.vocab_size = int(vocab_size)
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.sort_tokens = sort_tokens

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
        return len(self.samples)

    def __getitem__(self, idx):
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
    )

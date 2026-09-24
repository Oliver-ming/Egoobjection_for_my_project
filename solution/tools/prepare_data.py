#!/usr/bin/env python3
"""Build the derived annotation files used by all training pipelines.

Outputs (in ``--out-dir``, default ``solution/data/``):

* ``trainval.json`` -- train + val merged. Image ids must be unique across the splits (asserted);
  annotation ids are re-numbered ``1..N`` (pycocotools treats a GT id of 0 as "unmatched", so ids start
  at 1). ``file_name`` is kept verbatim (it already carries the ``train/`` / ``val/`` prefix), so a single
  image prefix ``resource/data/images/`` serves both splits.
* ``val_mini.json`` -- a deterministic (seeded) subset of val for fast smoke evaluations.
  Original annotation ids are kept, so predictions on full val can be scored against it directly
  (``tools/coco_eval.py`` drops predictions whose image is not in the GT).
* optional (``--yolo-dir DIR``): an Ultralytics-ready tree ``DIR/{images,labels}/{train,val}`` (+ ``images/test``)
  with labels regenerated from COCO and CLIPPED to the image. The provided ``resource/data/labels`` are unclipped
  and Ultralytics drops images with out-of-range coordinates (see ``docs/data_check.md``).

Example::

    python tools/prepare_data.py                          # defaults point to resource/data
    python tools/prepare_data.py --val-mini-size 200 --seed 0 --yolo-dir solution/data/yolo
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.common import GENERATED_DATA_DIR, IMAGE_ROOT, SPLIT_JSON, clip_xywh, dump_json, load_json


def merge_coco(datasets: dict[str, dict]) -> dict:
    """Merge COCO datasets (``{split_name: dataset}``) into one with unique, re-numbered annotation ids."""
    splits = list(datasets)
    categories = datasets[splits[0]]["categories"]
    for split in splits[1:]:
        if datasets[split]["categories"] != categories:
            raise ValueError(f"categories of '{split}' differ from '{splits[0]}'")

    images, annotations = [], []
    image_split: dict[int, str] = {}
    for split in splits:
        for image in datasets[split]["images"]:
            if image["id"] in image_split:
                raise ValueError(f"image id {image['id']} occurs in both '{image_split[image['id']]}' and '{split}'")
            image_split[image["id"]] = split
            images.append(image)
        for ann in datasets[split]["annotations"]:
            if image_split.get(ann["image_id"]) != split:
                raise ValueError(f"annotation {ann['id']} of '{split}' refers to unknown image {ann['image_id']}")
            annotations.append(dict(ann, id=len(annotations) + 1))

    info = copy.deepcopy(datasets[splits[0]].get("info", {}))
    info["description"] = f"{info.get('description', '')} [merged: {'+'.join(splits)}]".strip()
    return {"info": info, "images": images, "annotations": annotations, "categories": categories}


def subset_coco(dataset: dict, num_images: int, seed: int) -> dict:
    """Deterministic random subset of ``num_images`` images (and their annotations); order follows the source."""
    all_ids = sorted(image["id"] for image in dataset["images"])
    if not 0 < num_images <= len(all_ids):
        raise ValueError(f"num_images must be in [1, {len(all_ids)}], got {num_images}")
    keep = set(random.Random(seed).sample(all_ids, num_images))
    info = copy.deepcopy(dataset.get("info", {}))
    info["description"] = f"{info.get('description', '')} [subset: {num_images} images, seed={seed}]".strip()
    return {
        "info": info,
        "images": [image for image in dataset["images"] if image["id"] in keep],
        "annotations": [ann for ann in dataset["annotations"] if ann["image_id"] in keep],
        "categories": dataset["categories"],
    }


def write_yolo_split(dataset: dict, split: str, image_root: Path, yolo_dir: Path, min_size: float) -> Counter:
    """Write clipped YOLO labels (if ``dataset`` has annotations) and per-image symlinks for one split.

    Images are linked file by file: Ultralytics ``resolve()``s the split directory, so a directory symlink
    would send its label lookup back to ``resource/data/labels``.
    """
    image_dir, label_dir = yolo_dir / "images" / split, yolo_dir / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    anns_by_image: dict[int, list[dict]] = {}
    for ann in dataset.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    if anns_by_image:
        label_dir.mkdir(parents=True, exist_ok=True)

    for image in dataset["images"]:
        source = image_root / image["file_name"]
        link = image_dir / source.name
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(source, link)
        stats["images"] += 1
        if not anns_by_image:
            continue
        width, height = image["width"], image["height"]
        lines = []
        for ann in anns_by_image.get(image["id"], []):
            x0, y0, w0, h0 = ann["bbox"]
            stats["clipped"] += x0 < 0 or y0 < 0 or x0 + w0 > width or y0 + h0 > height
            x, y, w, h = clip_xywh(ann["bbox"], width, height)
            if w < min_size or h < min_size:
                stats["dropped"] += 1
                continue
            lines.append(f"{ann['category_id']} {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} "
                         f"{w / width:.6f} {h / height:.6f}")
        (label_dir / f"{source.stem}.txt").write_text("".join(line + "\n" for line in lines))
        stats["boxes"] += len(lines)
    return stats


def summarize(name: str, dataset: dict) -> str:
    id_to_name = {c["id"]: c["name"] for c in dataset["categories"]}
    per_class = Counter(ann["category_id"] for ann in dataset["annotations"])
    images_with_anns = len({ann["image_id"] for ann in dataset["annotations"]})
    ann_ids = [ann["id"] for ann in dataset["annotations"]]
    lines = [
        f"[{name}] images={len(dataset['images'])} (with annotations: {images_with_anns}) "
        f"annotations={len(dataset['annotations'])} "
        f"ann_id range=[{min(ann_ids, default=0)}, {max(ann_ids, default=0)}] "
        f"unique ann ids={len(set(ann_ids)) == len(ann_ids)}",
        "    per class: " + ", ".join(f"{id_to_name[c]}={per_class.get(c, 0)}" for c in sorted(id_to_name)),
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, default=SPLIT_JSON["train"], help="train COCO json")
    parser.add_argument("--val", type=Path, default=SPLIT_JSON["val"], help="val COCO json")
    parser.add_argument("--out-dir", type=Path, default=GENERATED_DATA_DIR, help="output directory")
    parser.add_argument("--val-mini-size", type=int, default=200, help="number of images in val_mini.json")
    parser.add_argument("--seed", type=int, default=0, help="seed for the val_mini subset")
    parser.add_argument("--yolo-dir", type=Path, default=None,
                        help="if set, also write clipped YOLO labels + image symlinks under this directory")
    parser.add_argument("--test", type=Path, default=SPLIT_JSON["test"], help="test COCO json (for --yolo-dir)")
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT, help="prefix of json file_name")
    parser.add_argument("--yolo-min-size", type=float, default=1.0,
                        help="drop boxes whose clipped width or height (px) is below this")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train, val = load_json(args.train), load_json(args.val)

    trainval = merge_coco({"train": train, "val": val})
    val_mini = subset_coco(val, args.val_mini_size, args.seed)

    for name, dataset in (("train", train), ("val", val), ("trainval", trainval), ("val_mini", val_mini)):
        print(summarize(name, dataset))
    assert len(trainval["images"]) == len(train["images"]) + len(val["images"])
    assert len(trainval["annotations"]) == len(train["annotations"]) + len(val["annotations"])

    for filename, dataset in (("trainval.json", trainval), ("val_mini.json", val_mini)):
        path = dump_json(dataset, args.out_dir / filename)
        print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")

    if args.yolo_dir is not None:
        for split, dataset in (("train", train), ("val", val), ("test", load_json(args.test))):
            stats = write_yolo_split(dataset, split, args.image_root, args.yolo_dir, args.yolo_min_size)
            print(f"[yolo/{split}] " + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))
        print(f"wrote YOLO tree under {args.yolo_dir}")


if __name__ == "__main__":
    main()

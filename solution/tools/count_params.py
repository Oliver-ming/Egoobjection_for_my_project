#!/usr/bin/env python3
"""Count parameters (total / trainable, in millions) for the competition detectors.

Covers:
* Co-DINO Swin-L and DINO Swin-L from MMDetection configs (built on CPU, no checkpoints)
* Ultralytics weights (``.pt``, typically 80-class COCO heads) plus a 10-class rebuild from the
  same yaml (or a fine-tuned checkpoint passed via ``--ultra-ft``)

Co-DINO additionally reports *inference-time* params: at test time only ``eval_module='detr'``
is used (``backbone`` + ``neck`` + ``query_head``). Aux ATSS / RPN+RoI heads are training-only.

Usage::

    CUDA_VISIBLE_DEVICES="" YOLO_AUTOINSTALL=False \\
      env/bin/python tools/count_params.py --out docs/model_params.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

# Hard rules: CPU only; never let ultralytics auto-install.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.common import SOLUTION_ROOT, dump_json  # noqa: E402

MMDET_ROOT = SOLUTION_ROOT / "mmdet"
WEIGHTS = SOLUTION_ROOT / "weights"
DEFAULT_CODINO = MMDET_ROOT / "configs" / "codino_swinl_ego.py"
DEFAULT_DINO = MMDET_ROOT / "configs" / "dino_swinl_ego.py"
DEFAULT_ULTRA = (
    WEIGHTS / "yolo26x.pt",
    WEIGHTS / "yolo11x.pt",
    WEIGHTS / "rtdetr-x.pt",
)


def _count(module) -> dict[str, float]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "total_M": round(total / 1e6, 3),
        "trainable_M": round(trainable / 1e6, 3),
    }


def _strip_pretrained(cfg) -> None:
    """Avoid checkpoint / ImageNet downloads when building for a param count."""
    cfg.load_from = None
    cfg.resume = False
    model = cfg.model
    if isinstance(model, dict):
        backbone = model.get("backbone")
        if isinstance(backbone, dict) and backbone.get("init_cfg") is not None:
            backbone["init_cfg"] = None
        # nested lists (Co-DINO roi/bbox heads) — leave class counts as configured


def _build_mmdet(config_path: Path):
    if str(MMDET_ROOT) not in sys.path:
        sys.path.insert(0, str(MMDET_ROOT))  # needed for projects.CO-DETR.codetr
    from mmengine.config import Config
    from mmengine.registry import DefaultScope, init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    init_default_scope("mmdet")
    cfg = Config.fromfile(str(config_path))
    _strip_pretrained(cfg)
    # custom_imports (Co-DINO) are applied by Config.fromfile when allow_failed_imports=False
    with DefaultScope.overwrite_default_scope("mmdet"):
        model = MODELS.build(cfg.model)
    model.eval()
    # Config may leave some buffers non-trainable; force count under train() for "trainable" meaning
    # "would update if trained" — use requires_grad as-is after build (mmdet defaults to True).
    return model, cfg


def _codino_inference_modules(model) -> dict[str, Any]:
    """Modules used when ``eval_module == 'detr'`` (default): backbone + neck + query_head."""
    parts = {"backbone": model.backbone, "query_head": model.query_head}
    if getattr(model, "with_neck", False) and model.neck is not None:
        parts["neck"] = model.neck
    return parts


def _codino_aux_modules(model) -> dict[str, Any]:
    parts = {}
    if getattr(model, "with_rpn", False):
        parts["rpn_head"] = model.rpn_head
    if getattr(model, "with_roi_head", False):
        parts["roi_head"] = model.roi_head
    if getattr(model, "bbox_head", None) is not None and len(model.bbox_head) > 0:
        parts["bbox_head_atss"] = model.bbox_head
    return parts


def count_mmdet(name: str, config_path: Path) -> dict[str, Any]:
    print(f"[mmdet] building {name} from {config_path} ...", flush=True)
    model, cfg = _build_mmdet(config_path)
    row: dict[str, Any] = {
        "name": name,
        "framework": "mmdet",
        "config": str(config_path),
        "model_type": cfg.model.get("type"),
        **_count(model),
    }
    if cfg.model.get("type") == "CoDETR":
        eval_module = getattr(model, "eval_module", "detr")
        inf_parts = _codino_inference_modules(model)
        aux_parts = _codino_aux_modules(model)
        inf_total = sum(_count(m)["total"] for m in inf_parts.values())
        aux_total = sum(_count(m)["total"] for m in aux_parts.values())
        row["eval_module"] = eval_module
        row["inference_note"] = (
            "At test time Co-DINO uses eval_module='detr' (query/DETR head only). "
            "Auxiliary ATSS and RPN+RoI heads are training-only."
        )
        row["inference"] = {
            "modules": list(inf_parts),
            "total": inf_total,
            "total_M": round(inf_total / 1e6, 3),
            "per_module_M": {k: _count(m)["total_M"] for k, m in inf_parts.items()},
        }
        row["training_aux"] = {
            "modules": list(aux_parts),
            "total": aux_total,
            "total_M": round(aux_total / 1e6, 3),
            "per_module_M": {k: _count(m)["total_M"] for k, m in aux_parts.items()},
        }
    del model
    return row


def _ultra_build_nc10(yaml_dict: dict, is_rtdetr: bool):
    from ultralytics.nn.tasks import DetectionModel, RTDETRDetectionModel

    y = deepcopy(yaml_dict)
    y["nc"] = 10
    if is_rtdetr:
        return RTDETRDetectionModel(y, nc=10, verbose=False)
    return DetectionModel(y, nc=10, verbose=False)


def count_ultralytics(weight: Path, ft: Optional[Path] = None) -> list[dict[str, Any]]:
    from ultralytics import YOLO

    rows = []
    print(f"[ultra] loading {weight} ...", flush=True)
    yolo = YOLO(str(weight))
    model = yolo.model
    # Loaded COCO checkpoints often have requires_grad=False; report architecture total and note it.
    c = _count(model)
    if c["trainable"] == 0 and c["total"] > 0:
        c["trainable"] = c["total"]
        c["trainable_M"] = c["total_M"]
        c["trainable_note"] = "checkpoint tensors had requires_grad=False; trainable set equal to total"
    row80 = {
        "name": weight.name,
        "framework": "ultralytics",
        "source": str(weight),
        "nc": int(model.yaml.get("nc", len(getattr(model, "names", {}) or {}))),
        "variant": "pretrained_coco_head",
        **{k: c[k] for k in ("total", "trainable", "total_M", "trainable_M") if k in c},
    }
    if "trainable_note" in c:
        row80["trainable_note"] = c["trainable_note"]
    rows.append(row80)

    is_rtdetr = "rtdetr" in weight.name.lower() or model.__class__.__name__.startswith("RTDETR")
    print(f"[ultra] building nc=10 from yaml of {weight.name} ...", flush=True)
    m10 = _ultra_build_nc10(model.yaml, is_rtdetr=is_rtdetr)
    c10 = _count(m10)
    rows.append({
        "name": f"{weight.stem}_nc10",
        "framework": "ultralytics",
        "source": f"yaml from {weight.name} with nc=10",
        "nc": 10,
        "variant": "yaml_nc10",
        **c10,
    })
    del m10

    if ft is not None:
        print(f"[ultra] loading fine-tuned {ft} ...", flush=True)
        yft = YOLO(str(ft))
        cft = _count(yft.model)
        if cft["trainable"] == 0 and cft["total"] > 0:
            cft["trainable"] = cft["total"]
            cft["trainable_M"] = cft["total_M"]
        rows.append({
            "name": ft.name,
            "framework": "ultralytics",
            "source": str(ft),
            "nc": int(yft.model.yaml.get("nc", 10)),
            "variant": "finetuned",
            **{k: cft[k] for k in ("total", "trainable", "total_M", "trainable_M")},
        })
    return rows


def to_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Model parameter counts",
        "",
        "Generated by `tools/count_params.py` (CPU only, no checkpoints loaded for MMDet).",
        "",
        "| model | framework | nc / note | total (M) | trainable (M) |",
        "|---|---|---|---:|---:|",
    ]
    for r in rows:
        note = r.get("variant") or r.get("model_type") or ""
        if r.get("eval_module"):
            note = f"{note}; eval_module={r['eval_module']}"
        nc = r.get("nc", 10)
        lines.append(
            f"| {r['name']} | {r['framework']} | {nc} / {note} | {r['total_M']:.3f} | {r['trainable_M']:.3f} |"
        )
    # Co-DINO inference breakdown
    for r in rows:
        if "inference" in r:
            lines += [
                "",
                f"## {r['name']} train vs inference",
                "",
                r.get("inference_note", ""),
                "",
                f"- **Full model (train)**: {r['total_M']:.3f} M",
                f"- **Inference (backbone+neck+query_head)**: {r['inference']['total_M']:.3f} M "
                f"({', '.join(f'{k}={v:.3f}M' for k, v in r['inference']['per_module_M'].items())})",
                f"- **Training-only aux (RPN/RoI/ATSS)**: {r['training_aux']['total_M']:.3f} M "
                f"({', '.join(f'{k}={v:.3f}M' for k, v in r['training_aux']['per_module_M'].items())})",
            ]
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--codino", type=Path, default=DEFAULT_CODINO)
    p.add_argument("--dino", type=Path, default=DEFAULT_DINO)
    p.add_argument("--ultra", type=Path, nargs="*", default=list(DEFAULT_ULTRA),
                   help="Ultralytics .pt weights (80-class COCO heads by default)")
    p.add_argument("--ultra-ft", type=Path, nargs="*", default=[],
                   help="optional fine-tuned checkpoints (nc=10); reported in addition to yaml rebuild")
    p.add_argument("--skip-mmdet", action="store_true")
    p.add_argument("--skip-ultra", action="store_true")
    p.add_argument("--out", type=Path, default=None, help="write JSON here")
    p.add_argument("--md", type=Path, default=None, help="write markdown table here")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows: list[dict[str, Any]] = []
    if not args.skip_mmdet:
        rows.append(count_mmdet("Co-DINO Swin-L", args.codino))
        rows.append(count_mmdet("DINO Swin-L", args.dino))
    if not args.skip_ultra:
        # Map fine-tuned ckpts by stem prefix if provided (optional positional pairing is free-form)
        ft_list = list(args.ultra_ft)
        for w in args.ultra:
            # attach first matching ft whose stem contains weight stem, else none
            ft = next((f for f in ft_list if w.stem.split(".")[0] in f.stem or w.stem in f.stem), None)
            if ft is not None:
                ft_list = [f for f in ft_list if f != ft]
            rows.extend(count_ultralytics(w, ft=ft))
        for ft in ft_list:
            rows.extend(count_ultralytics(ft, ft=None)[:1])  # only the loaded ckpt row

    md = to_markdown(rows)
    print(md)
    payload = {"models": rows}
    if args.out is not None:
        dump_json(payload, args.out, indent=2)
        print(f"wrote {args.out}", flush=True)
    if args.md is not None:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(md, encoding="utf-8")
        print(f"wrote {args.md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

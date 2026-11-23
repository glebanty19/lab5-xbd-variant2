"""Reproducible xBD semantic-segmentation experiment for laboratory work 5.

Expected input: xBD image/label folders (e.g. train/images, train/labels,
hold/images, hold/labels, test/images, test/labels). See --help.
Dependencies: torch, segmentation-models-pytorch, Pillow, numpy, matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


CLASSES = ("background", "no-damage", "minor-damage", "major-damage", "destroyed")
SUBTYPE_IDS = {name: index for index, name in enumerate(CLASSES) if index}
CLASS_COLORS = np.asarray([
    (0, 0, 0), (46, 204, 113), (241, 196, 15), (230, 126, 34), (231, 76, 60),
], dtype=np.uint8)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "hold": "val",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "tier3": "test",
}


def _scene_id(stem: str) -> str:
    return re.sub(r"_(pre|post)_disaster$", "", stem, flags=re.IGNORECASE)


def _condition(stem: str) -> str | None:
    match = re.search(r"_(pre|post)_disaster$", stem, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def _find_sensor(value: Any) -> str:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {"sensor", "satellite", "platform"} and isinstance(item, str):
                return item
        for item in value.values():
            found = _find_sensor(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_sensor(item)
            if found:
                return found
    return ""


def _find_xy(feature: dict[str, Any]) -> list[list[list[float]]]:
    coordinates = feature.get("xy")
    if not coordinates and isinstance(feature.get("wkt"), str):
        rings = re.findall(r"\(([^()]*)\)", feature["wkt"])
        coordinates = []
        for ring in rings:
            points = []
            for pair in ring.split(","):
                values = pair.split()
                if len(values) >= 2:
                    points.append([float(values[0]), float(values[1])])
            if points:
                coordinates.append(points)
    if not coordinates:
        geometry = feature.get("geometry") or {}
        if geometry.get("type") == "Polygon":
            coordinates = geometry.get("coordinates")
        elif geometry.get("type") == "MultiPolygon":
            coordinates = [
                ring for polygon in geometry.get("coordinates", []) for ring in polygon
            ]
    if not coordinates:
        return []
    if coordinates and coordinates[0] and isinstance(coordinates[0][0], (int, float)):
        coordinates = [coordinates]
    return coordinates


def rasterize_label(
    label_path: Path, size: tuple[int, int]
) -> tuple[np.ndarray, dict[str, int], dict[str, list[float]]]:
    data = json.loads(label_path.read_text(encoding="utf-8"))
    mask_image = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask_image)
    object_counts = {name: 0 for name in CLASSES[1:]}
    object_areas = {name: [] for name in CLASSES[1:]}
    features = data.get("features", [])
    if isinstance(features, dict):
        features = features.get("xy", [])
    for feature in features:
        properties = feature.get("properties") or {}
        if properties.get("feature_type", "building") == "building":
            subtype = str(properties.get("subtype", "")).lower()
            if subtype not in SUBTYPE_IDS:
                for polygon in _find_xy(feature):
                    points = [(float(point[0]), float(point[1])) for point in polygon]
                    if len(points) >= 3:
                        draw.polygon(points, fill=255)
    for class_id in range(1, len(CLASSES)):
        class_name = CLASSES[class_id]
        for feature in features:
            properties = feature.get("properties") or {}
            if properties.get("feature_type", "building") != "building":
                continue
            subtype = str(properties.get("subtype", "")).lower()
            if subtype != class_name:
                continue
            polygons = _find_xy(feature)
            for polygon in polygons:
                points = [(float(point[0]), float(point[1])) for point in polygon]
                if len(points) >= 3:
                    draw.polygon(points, fill=class_id)
                    object_counts[class_name] += 1
                    area = abs(sum(
                        points[i][0] * points[(i + 1) % len(points)][1]
                        - points[(i + 1) % len(points)][0] * points[i][1]
                        for i in range(len(points))
                    )) / 2
                    object_areas[class_name].append(area)
    return np.asarray(mask_image, dtype=np.uint8), object_counts, object_areas


def prepare(args: argparse.Namespace) -> None:
    root = args.data.resolve()
    output = args.output.resolve()
    images: dict[tuple[str, str], Path] = {}
    labels: dict[tuple[str, str], Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        split = next(
            (SPLIT_ALIASES[part.lower()] for part in reversed(path.parts)
             if part.lower() in SPLIT_ALIASES),
            None,
        )
        if split is None:
            continue
        if path.suffix.lower() in IMAGE_EXTENSIONS and _condition(path.stem):
            key = (split, path.stem)
            if key in images:
                raise ValueError(f"Duplicate image stem in split {split}: {path.stem}")
            images[key] = path
        elif path.suffix.lower() == ".json" and _condition(path.stem):
            key = (split, path.stem)
            if key in labels:
                raise ValueError(f"Duplicate label stem in split {split}: {path.stem}")
            labels[key] = path
    if not images:
        raise FileNotFoundError(f"No *_pre_disaster or *_post_disaster images found under {root}")

    source_splits = {split for split, _ in images}
    split_by_scene: dict[str, str] = {}
    if source_splits == {"train"}:
        scene_ids = sorted({_scene_id(stem) for _, stem in images if _condition(stem) == "post"})
        random.Random(args.seed).shuffle(scene_ids)
        test_count = max(1, round(len(scene_ids) * 0.1))
        val_count = max(1, round(len(scene_ids) * 0.1))
        for scene_id in scene_ids[:test_count]:
            split_by_scene[scene_id] = "test"
        for scene_id in scene_ids[test_count:test_count + val_count]:
            split_by_scene[scene_id] = "val"
        for scene_id in scene_ids[test_count + val_count:]:
            split_by_scene[scene_id] = "train"

    image_by_key = {
        (split, _scene_id(stem), _condition(stem)): path
        for (split, stem), path in images.items()
    }
    records: list[dict[str, Any]] = []
    missing_labels: list[str] = []
    for (source_split, stem), image_path in sorted(images.items()):
        if _condition(stem) != "post":
            continue
        label_path = labels.get((source_split, stem))
        if label_path is None:
            missing_labels.append(stem)
            continue
        scene_id = _scene_id(stem)
        split = split_by_scene.get(scene_id, source_split)
        with Image.open(image_path) as source:
            width, height = source.size
        mask, object_counts, object_areas = rasterize_label(label_path, (width, height))
        mask_path = output / "masks" / split / f"{stem}.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(mask_path)
        label_data = json.loads(label_path.read_text(encoding="utf-8"))
        pre_image = image_by_key.get((source_split, scene_id, "pre"))
        valid_mask = mask != 255
        counts = np.bincount(mask[valid_mask].ravel(), minlength=len(CLASSES))
        records.append({
            "split": split,
            "scene_id": scene_id,
            "post_image": str(image_path),
            "pre_image": str(pre_image) if pre_image else "",
            "mask": str(mask_path),
            "label": str(label_path),
            "sensor": _find_sensor(label_data),
            "width": width,
            "height": height,
            "objects": object_counts,
            "object_areas_px": object_areas,
            "pixels": {name: int(counts[i]) for i, name in enumerate(CLASSES)},
        })
    if not records:
        raise FileNotFoundError("No post-disaster images with matching GeoJSON labels were found.")
    manifest_path = output / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["split", "scene_id", "post_image", "pre_image", "mask", "label",
               "sensor", "width", "height", "objects", "object_areas_px", "pixels"]
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "objects": json.dumps(record["objects"]),
                             "object_areas_px": json.dumps(record["object_areas_px"]),
                             "pixels": json.dumps(record["pixels"])})
    print(f"Prepared {len(records)} post-event samples; manifest: {manifest_path}")
    if split_by_scene:
        print(f"Split method: deterministic scene-level 80/10/10 from official train, seed={args.seed}")
    print(f"Missing labels: {len(missing_labels)}; paired pre-event images: "
          f"{sum(bool(row['pre_image']) for row in records)}")
    for split in ("train", "val", "test"):
        subset = [row for row in records if row["split"] == split]
        totals = {name: sum(row["pixels"][name] for row in subset) for name in CLASSES}
        total_pixels = sum(totals.values())
        fractions = {name: round(value / max(total_pixels, 1), 6)
                     for name, value in totals.items()}
        print(f"{split}: images={len(subset)}, pixel_fractions={fractions}")
        areas = {
            name: [area for row in subset for area in row["object_areas_px"][name]]
            for name in CLASSES[1:]
        }
        area_summary = {
            name: {
                "count": len(values),
                "median_px": round(float(np.median(values)), 1) if values else None,
                "p90_px": round(float(np.percentile(values, 90)), 1) if values else None,
            }
            for name, values in areas.items()
        }
        print(f"{split}: polygon_area_summary={area_summary}")
        if subset:
            sample = subset[0]
            with Image.open(sample["post_image"]) as source:
                rgb = source.convert("RGB")
            mask_rgb = Image.fromarray(CLASS_COLORS[np.asarray(
                Image.open(sample["mask"]).convert("L"), dtype=np.uint8
            ).clip(0, len(CLASSES) - 1)])
            mask_rgb = mask_rgb.resize(rgb.size, Image.Resampling.NEAREST)
            overlay = Image.blend(rgb, mask_rgb, 0.45)
            sample_dir = output / "visualizations"
            sample_dir.mkdir(parents=True, exist_ok=True)
            overlay.save(sample_dir / f"{split}_ground_truth_overlay.png")
    if missing_labels:
        print("Unmatched labels (first 10):", ", ".join(missing_labels[:10]))


def _dependencies():
    try:
        import torch
        import torch.nn.functional as functional
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise RuntimeError(
            "Install the training dependencies with: "
            "python -m pip install torch segmentation-models-pytorch"
        ) from exc
    return torch, functional, smp


class XbdDataset:
    def __init__(self, rows: list[dict[str, str]], size: int, augment: bool = False,
                 image_column: str = "post_image"):
        self.rows = rows
        self.size = size
        self.augment = augment
        self.image_column = image_column

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        import torch
        import torch.nn.functional as functional

        row = self.rows[index]
        with Image.open(row[self.image_column]) as image:
            image = image.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
            x = np.asarray(image, dtype=np.float32) / 255.0
        with Image.open(row["mask"]) as image:
            y = np.asarray(image.convert("L").resize((self.size, self.size), Image.Resampling.NEAREST),
                           dtype=np.int64)
        if self.augment and random.random() < 0.5:
            x, y = np.fliplr(x).copy(), np.fliplr(y).copy()
        if self.augment and random.random() < 0.5:
            x, y = np.flipud(x).copy(), np.flipud(y).copy()
        x = torch.from_numpy(x.transpose(2, 0, 1))
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        x = (x - mean) / std
        y = torch.from_numpy(y.copy())
        return x, y


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _confusion_metrics(confusion: np.ndarray) -> dict[str, Any]:
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    union = tp + fp + fn
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    dice_denominator = 2 * tp + fp + fn
    dice = np.divide(2 * tp, dice_denominator, out=np.zeros_like(tp),
                     where=dice_denominator > 0)
    return {
        "iou": {name: float(iou[i]) for i, name in enumerate(CLASSES)},
        "dice": {name: float(dice[i]) for i, name in enumerate(CLASSES)},
        "miou": float(iou[union > 0].mean()) if np.any(union > 0) else 0.0,
        "pixel_accuracy": float(tp.sum() / max(confusion.sum(), 1)),
    }


def _save_prediction_preview(image_path: str, mask_path: str, prediction,
                             output_path: Path) -> None:
    import torch

    with Image.open(image_path) as source:
        image = source.convert("RGB").resize((prediction.shape[1], prediction.shape[0]))
    with Image.open(mask_path) as source:
        target = np.asarray(source.convert("L").resize(image.size, Image.Resampling.NEAREST))
    pred = prediction.cpu().numpy()
    target_colors = CLASS_COLORS[target.clip(0, len(CLASSES) - 1)]
    pred_colors = CLASS_COLORS[pred.clip(0, len(CLASSES) - 1)]
    target_overlay = Image.blend(image, Image.fromarray(target_colors), 0.45)
    prediction_overlay = Image.blend(image, Image.fromarray(pred_colors), 0.45)
    valid = target != 255
    error = np.zeros((*target.shape, 3), dtype=np.uint8)
    error[target == 255] = (120, 120, 120)
    error[(target == pred) & (target == 0) & valid] = (35, 55, 85)
    error[(target == pred) & (target != 0) & valid] = (46, 204, 113)
    error[(target == 0) & (pred != 0) & valid] = (241, 196, 15)
    error[(target != 0) & (target != pred) & valid] = (231, 76, 60)
    panels = [image, target_overlay, prediction_overlay, Image.fromarray(error)]
    width, height = image.size
    canvas = Image.new("RGB", (width * len(panels), height + 28), "white")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    for index, (panel, title) in enumerate(zip(
        panels, ("Image", "Ground truth", "Prediction", "Errors")
    )):
        canvas.paste(panel, (width * index, 28))
        draw.text((width * index + 6, 6), title, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _boundary_counts(pred, target, classes: int, tol: int, valid=None):
    import torch
    import torch.nn.functional as functional

    totals = np.zeros(4, dtype=np.float64)
    for class_id in range(classes):
        p = (pred == class_id).float()[None, None]
        g = (target == class_id).float()[None, None]
        if valid is not None:
            mask = valid.float()[None, None]
            p = p * mask
            g = g * mask
        p_edge = (p - (-functional.max_pool2d(-p, 3, 1, 1))).clamp(0, 1)
        g_edge = (g - (-functional.max_pool2d(-g, 3, 1, 1))).clamp(0, 1)
        if valid is not None:
            p_edge = p_edge * mask
            g_edge = g_edge * mask
        kernel = 2 * tol + 1
        g_near = functional.max_pool2d(g_edge, kernel, 1, tol)
        p_near = functional.max_pool2d(p_edge, kernel, 1, tol)
        totals[0] += float((p_edge * g_near).sum())
        totals[1] += float((g_edge * p_near).sum())
        totals[2] += float(p_edge.sum())
        totals[3] += float(g_edge.sum())
    return totals


def train(args: argparse.Namespace) -> None:
    torch, functional, smp = _dependencies()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    manifest = _read_manifest(args.manifest)
    train_rows = [row for row in manifest if row["split"] == "train"]
    val_rows = [row for row in manifest if row["split"] == "val"]
    if not train_rows or not val_rows:
        raise ValueError("The manifest must contain both train and val samples.")
    random.Random(args.seed).shuffle(train_rows)
    train_rows = train_rows[:max(1, round(len(train_rows) * args.fraction))]
    worker_options = ({"persistent_workers": True, "prefetch_factor": 1}
                      if args.workers > 0 else {})
    train_loader = torch.utils.data.DataLoader(
        XbdDataset(train_rows, args.size, augment=True), batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, **worker_options,
    )
    val_loader = torch.utils.data.DataLoader(
        XbdDataset(val_rows, args.size), batch_size=args.batch_size, shuffle=False,
        num_workers=0,
    )
    model = smp.Unet(encoder_name="efficientnet-b3", encoder_weights="imagenet",
                     in_channels=3, classes=len(CLASSES))
    if args.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr, weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_miou = -1.0
    history = []
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_encoder:
            model.encoder.eval()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
                loss = functional.cross_entropy(logits, y, ignore_index=255)
                if args.dice_loss:
                    probabilities = logits.float().softmax(dim=1)
                    safe_y = y.clamp_max(len(CLASSES) - 1)
                    one_hot = functional.one_hot(safe_y, len(CLASSES)).permute(0, 3, 1, 2).float()
                    valid_pixels = (y != 255).float()[:, None]
                    probabilities = probabilities * valid_pixels
                    one_hot = one_hot * valid_pixels
                    intersection = (probabilities * one_hot).sum((0, 2, 3))
                    denominator = probabilities.sum((0, 2, 3)) + one_hot.sum((0, 2, 3))
                    loss = loss + 1 - ((2 * intersection + 1) / (denominator + 1)).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    prediction = model(x).argmax(dim=1)
                valid = y != 255
                bins = torch.bincount((y[valid] * len(CLASSES) + prediction[valid]).reshape(-1),
                                      minlength=len(CLASSES) ** 2)
                confusion += bins.reshape(len(CLASSES), len(CLASSES)).cpu().numpy()
        metrics = _confusion_metrics(confusion)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                  "val_miou": metrics["miou"]}
        history.append(record)
        (args.output / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8",
        )
        print(json.dumps(record))
        if metrics["miou"] > best_miou:
            best_miou = metrics["miou"]
            torch.save({
                "model": model.state_dict(),
                "classes": list(CLASSES),
                "size": args.size,
                "epoch": epoch,
                "val_miou": best_miou,
                "seed": args.seed,
                "fraction": args.fraction,
                "freeze_encoder": args.freeze_encoder,
                "dice_loss": args.dice_loss,
            }, args.output / "best.pt")
    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib to save the required learning curves.") from exc
    figure, axis = plt.subplots()
    axis.plot([row["epoch"] for row in history], [row["train_loss"] for row in history],
              label="train loss")
    axis.plot([row["epoch"] for row in history], [row["val_miou"] for row in history],
              label="validation mIoU")
    axis.set(xlabel="Epoch", ylabel="Value", title="Training history")
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output / "learning_curve.png", dpi=160)
    plt.close(figure)
    print(f"Best checkpoint: {args.output / 'best.pt'} (validation mIoU={best_miou:.4f})")


def evaluate(args: argparse.Namespace) -> None:
    torch, _, smp = _dependencies()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if tuple(checkpoint["classes"]) != CLASSES:
        raise ValueError("Checkpoint class order does not match this script.")
    model = smp.Unet(encoder_name="efficientnet-b3", encoder_weights=None,
                     in_channels=3, classes=len(CLASSES))
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    rows = [row for row in _read_manifest(args.manifest) if row["split"] == args.split]
    if not rows:
        raise ValueError(f"No {args.split} samples in manifest.")
    loader = torch.utils.data.DataLoader(
        XbdDataset(rows, int(checkpoint["size"])), batch_size=args.batch_size,
        shuffle=False, num_workers=0,
    )
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    boundary = np.zeros(4, dtype=np.float64)
    with torch.no_grad():
        preview_index = 0
        for x, y in loader:
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                prediction = model(x.to(device)).argmax(dim=1).cpu()
            valid = y != 255
            bins = torch.bincount((y[valid] * len(CLASSES) + prediction[valid]).reshape(-1),
                                  minlength=len(CLASSES) ** 2)
            confusion += bins.reshape(len(CLASSES), len(CLASSES)).numpy()
            for p, g in zip(prediction, y):
                boundary += _boundary_counts(p, g, len(CLASSES), args.boundary_tolerance,
                                             valid=g != 255)
                if preview_index < args.visual_samples:
                    row = rows[preview_index]
                    _save_prediction_preview(
                        row["post_image"], row["mask"], p,
                        args.output.parent / "visualizations" / f"{args.split}_{preview_index + 1}.png",
                    )
                preview_index += 1
    result = _confusion_metrics(confusion)
    precision = boundary[0] / max(boundary[2], 1)
    recall = boundary[1] / max(boundary[3], 1)
    result["boundary_f1"] = float(2 * precision * recall / max(precision + recall, 1e-12))
    result["split"] = args.split
    result["images"] = len(rows)
    result["classes"] = list(CLASSES)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def compare_conditions(args: argparse.Namespace) -> None:
    torch, _, smp = _dependencies()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = smp.Unet(encoder_name="efficientnet-b3", encoder_weights=None,
                     in_channels=3, classes=len(CLASSES))
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    rows = [row for row in _read_manifest(args.manifest)
            if row["split"] == args.split and row["pre_image"]]
    if not rows:
        raise ValueError(f"No paired pre/post images in the {args.split} split.")

    def score(subset: list[dict[str, str]], image_column: str) -> dict[str, Any]:
        dataset = XbdDataset(subset, int(checkpoint["size"]), image_column=image_column)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        )
        confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
        with torch.no_grad():
            for x, y in loader:
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    prediction = model(x.to(device)).argmax(dim=1).cpu()
                valid = y != 255
                bins = torch.bincount((y[valid] * len(CLASSES) + prediction[valid]).reshape(-1),
                                      minlength=len(CLASSES) ** 2)
                confusion += bins.reshape(len(CLASSES), len(CLASSES)).numpy()
        result = _confusion_metrics(confusion)
        result["images"] = len(subset)
        return result

    result: dict[str, Any] = {
        "split": args.split,
        "ground_truth": "post-event building damage labels",
        "pre_event": score(rows, "pre_image"),
        "post_event": score(rows, "post_image"),
    }
    known_sensors = sorted({row["sensor"] for row in rows if row["sensor"]})
    result["by_sensor"] = {
        sensor: score([row for row in rows if row["sensor"] == sensor], "post_image")
        for sensor in known_sensors
    }
    result["sensor_comparison_note"] = (
        "Grouped using sensor metadata found in the xBD annotation JSON."
        if known_sensors else "Sensor metadata is absent; sensor-wise comparison is unavailable."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def compare_tiles(args: argparse.Namespace) -> None:
    """Compare full-frame inference with overlapping native-resolution tiles."""
    torch, _, smp = _dependencies()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = smp.Unet(encoder_name="efficientnet-b3", encoder_weights=None,
                     in_channels=3, classes=len(CLASSES))
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    rows = [row for row in _read_manifest(args.manifest) if row["split"] == args.split]
    if not rows:
        raise ValueError(f"No {args.split} samples in manifest.")
    if args.tile_size <= 0 or not 0 <= args.overlap < args.tile_size:
        raise ValueError("Require tile_size > overlap >= 0.")
    size = int(checkpoint["size"])
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    full_confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    tile_confusion = np.zeros_like(full_confusion)
    full_boundary = np.zeros(4, dtype=np.float64)
    tile_boundary = np.zeros(4, dtype=np.float64)

    def normalized_batch(images: list[Image.Image]):
        arrays = [np.asarray(im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR),
                             dtype=np.float32) / 255.0 for im in images]
        batch = torch.from_numpy(np.stack(arrays).transpose(0, 3, 1, 2)).to(device)
        return (batch - mean) / std

    step = args.tile_size - args.overlap
    for sample_index, row in enumerate(rows):
        with Image.open(row["post_image"]) as source:
            image = source.convert("RGB")
        with Image.open(row["mask"]) as source:
            target = np.asarray(source.convert("L"), dtype=np.uint8)
        width, height = image.size
        target_eval = np.array(Image.fromarray(target).resize(
            (size, size), Image.Resampling.NEAREST), dtype=np.uint8, copy=True)

        with torch.no_grad():
            full_logits = model(normalized_batch([image]))
            full_prediction = full_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

            y_starts = list(range(0, height, step))
            x_starts = list(range(0, width, step))
            probability_sum = np.zeros((height, width, len(CLASSES)), dtype=np.float32)
            weight_sum = np.zeros((height, width), dtype=np.float32)
            for y0 in y_starts:
                for x0 in x_starts:
                    y1, x1 = min(y0 + args.tile_size, height), min(x0 + args.tile_size, width)
                    tile = image.crop((x0, y0, x1, y1))
                    if tile.size != (args.tile_size, args.tile_size):
                        canvas = Image.new("RGB", (args.tile_size, args.tile_size))
                        canvas.paste(tile, (0, 0))
                        tile = canvas
                    probabilities = model(normalized_batch([tile])).softmax(dim=1)
                    probabilities = torch.nn.functional.interpolate(
                        probabilities, size=(args.tile_size, args.tile_size),
                        mode="bilinear", align_corners=False,
                    )[0].permute(1, 2, 0).cpu().numpy()
                    valid_h, valid_w = y1 - y0, x1 - x0
                    # Raised-cosine weights reduce seams in overlapping regions.
                    axis = np.linspace(0, np.pi, args.tile_size, dtype=np.float32)
                    ramp = 0.1 + 0.9 * np.sin(axis) ** 2
                    weight = np.outer(ramp, ramp)[:valid_h, :valid_w]
                    probability_sum[y0:y1, x0:x1] += probabilities[:valid_h, :valid_w] * weight[..., None]
                    weight_sum[y0:y1, x0:x1] += weight
            tiled_native = (probability_sum / np.maximum(weight_sum[..., None], 1e-8)).argmax(axis=2).astype(np.uint8)
            tiled_prediction = np.asarray(Image.fromarray(tiled_native).resize(
                (size, size), Image.Resampling.NEAREST), dtype=np.uint8)

        valid = target_eval != 255
        for prediction, confusion, boundary in (
            (full_prediction, full_confusion, full_boundary),
            (tiled_prediction, tile_confusion, tile_boundary),
        ):
            bins = np.bincount(
                (target_eval[valid] * len(CLASSES) + prediction[valid]).ravel(),
                minlength=len(CLASSES) ** 2,
            )
            confusion += bins.reshape(len(CLASSES), len(CLASSES))
            boundary += _boundary_counts(
                torch.from_numpy(prediction), torch.from_numpy(target_eval),
                len(CLASSES), args.boundary_tolerance,
                valid=torch.from_numpy(valid),
            )
        if sample_index < args.visual_samples:
            _save_prediction_preview(
                row["post_image"], row["mask"], torch.from_numpy(tiled_prediction),
                args.output.parent / "visualizations" / f"tile_{args.split}_{sample_index + 1}.png",
            )

    def summarize(confusion: np.ndarray, boundary: np.ndarray) -> dict[str, Any]:
        result = _confusion_metrics(confusion)
        precision = boundary[0] / max(boundary[2], 1)
        recall = boundary[1] / max(boundary[3], 1)
        result["boundary_f1"] = float(2 * precision * recall / max(precision + recall, 1e-12))
        result["images"] = len(rows)
        return result

    result = {
        "split": args.split,
        "tile_size_native_px": args.tile_size,
        "overlap_native_px": args.overlap,
        "evaluation_size": size,
        "full_frame": summarize(full_confusion, full_boundary),
        "tiled_overlap": summarize(tile_confusion, tile_boundary),
        "note": "Tiles are predicted at native scale, blended by raised-cosine weights, then resized to the model evaluation grid. Scale differs from full-frame resize.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="Rasterize xBD post-event GeoJSON labels")
    prepare_parser.add_argument("--data", type=Path, required=True, help="Root of extracted xBD data")
    prepare_parser.add_argument("--output", type=Path, default=Path("xbd_prepared"))
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.set_defaults(func=prepare)

    train_parser = commands.add_parser("train", help="Train the U-Net EfficientNet-B3 model")
    train_parser.add_argument("--manifest", type=Path, default=Path("xbd_prepared/manifest.csv"))
    train_parser.add_argument("--output", type=Path, default=Path("runs/full"))
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--size", type=int, default=512)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--workers", type=int, default=2,
                              help="DataLoader worker processes (set 0 if multiprocessing is unavailable)")
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--fraction", type=float, default=1.0)
    train_parser.add_argument("--freeze-encoder", action="store_true")
    train_parser.add_argument("--dice-loss", action="store_true")
    train_parser.set_defaults(func=train)

    eval_parser = commands.add_parser("evaluate", help="Evaluate a checkpoint on val or test")
    eval_parser.add_argument("--manifest", type=Path, default=Path("xbd_prepared/manifest.csv"))
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--split", choices=("val", "test"), default="test")
    eval_parser.add_argument("--batch-size", type=int, default=4)
    eval_parser.add_argument("--boundary-tolerance", type=int, default=2)
    eval_parser.add_argument("--visual-samples", type=int, default=3)
    eval_parser.add_argument("--output", type=Path, default=Path("metrics_test.json"))
    eval_parser.set_defaults(func=evaluate)

    compare_parser = commands.add_parser(
        "compare", help="Compare paired pre/post imagery against post-event damage labels"
    )
    compare_parser.add_argument("--manifest", type=Path, default=Path("xbd_prepared/manifest.csv"))
    compare_parser.add_argument("--checkpoint", type=Path, required=True)
    compare_parser.add_argument("--split", choices=("val", "test"), default="test")
    compare_parser.add_argument("--batch-size", type=int, default=4)
    compare_parser.add_argument("--output", type=Path, default=Path("condition_comparison.json"))
    compare_parser.set_defaults(func=compare_conditions)

    tile_parser = commands.add_parser(
        "compare-tiles", help="Compare full-frame inference with overlapping tiles"
    )
    tile_parser.add_argument("--manifest", type=Path, default=Path("xbd_prepared/manifest.csv"))
    tile_parser.add_argument("--checkpoint", type=Path, required=True)
    tile_parser.add_argument("--split", choices=("val", "test"), default="test")
    tile_parser.add_argument("--tile-size", type=int, default=768)
    tile_parser.add_argument("--overlap", type=int, default=256)
    tile_parser.add_argument("--boundary-tolerance", type=int, default=2)
    tile_parser.add_argument("--visual-samples", type=int, default=3)
    tile_parser.add_argument("--output", type=Path, default=Path("tile_comparison.json"))
    tile_parser.set_defaults(func=compare_tiles)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

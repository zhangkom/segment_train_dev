from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
VALID_MASK_SUFFIX = ".png"
VALID_LABELS = {0, 1}


def collect_stems(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    files = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in suffixes:
            files[path.stem] = path
    return files


def check_split(root: Path, split: str) -> list[str]:
    errors: list[str] = []
    images_dir = root / "images" / split
    masks_dir = root / "masks" / split

    if not images_dir.exists():
        return [f"missing directory: {images_dir}"]
    if not masks_dir.exists():
        return [f"missing directory: {masks_dir}"]

    images = collect_stems(images_dir, VALID_IMAGE_SUFFIXES)
    masks = collect_stems(masks_dir, {VALID_MASK_SUFFIX})

    image_stems = set(images)
    mask_stems = set(masks)

    missing_masks = sorted(image_stems - mask_stems)
    missing_images = sorted(mask_stems - image_stems)

    if missing_masks:
        errors.append(f"{split}: missing masks for {len(missing_masks)} images, sample={missing_masks[:5]}")
    if missing_images:
        errors.append(f"{split}: missing images for {len(missing_images)} masks, sample={missing_images[:5]}")

    for stem in sorted(image_stems & mask_stems):
        mask_path = masks[stem]
        mask = np.array(Image.open(mask_path))
        unique = set(np.unique(mask).tolist())
        if not unique.issubset(VALID_LABELS):
            errors.append(f"{split}: invalid labels in {mask_path}, got={sorted(unique)}")
            break
        if mask.ndim != 2:
            errors.append(f"{split}: mask is not single channel: {mask_path}")
            break

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Check live portrait segmentation dataset")
    parser.add_argument("--root", type=Path, required=True, help="dataset root")
    args = parser.parse_args()

    root = args.root
    all_errors: list[str] = []
    for split in ("train", "val"):
        all_errors.extend(check_split(root, split))

    if all_errors:
        print("dataset check failed:")
        for error in all_errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"dataset check passed: {root}")


if __name__ == "__main__":
    main()

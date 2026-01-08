#!/usr/bin/env python3
"""
Batch dermoscopy preprocessing + classic lesion segmentation (no AI).

What it does for EACH image in the CURRENT DIRECTORY:
1) Hair detection + removal (DullRazor-style via multi-scale black-hat + inpainting)
2) Lesion likelihood map (Lab chroma-distance heuristic)
3) Otsu threshold (keeps the largest connected component immediately)
4) Hole filling + morphology cleanup
5) Contour overlay + masked lesion

Outputs:
- Creates an output folder in the current directory (default: lesion_charts/)
- For each input image, creates a subfolder and saves:
  - Original image
  - Hair cleaned input
  - Lesion likelihood map
  - Mask right after Otsu
  - Final lesion mask
  - Contour overlay
  - Masked lesion
  - A combined "charts" figure containing all of the above panels

Dependencies:
  pip install opencv-python numpy matplotlib
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Use a non-interactive backend so this works headless (servers/CI)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images_in_cwd() -> List[Path]:
    cwd = Path(".").resolve()
    imgs = [p for p in cwd.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    imgs.sort(key=lambda p: p.name.lower())
    return imgs


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    # cv2.imwrite expects BGR; use matplotlib to preserve RGB without channel swap confusion
    plt.imsave(str(path), rgb)


def save_gray(path: Path, gray_u8: np.ndarray) -> None:
    plt.imsave(str(path), gray_u8, cmap="gray", vmin=0, vmax=255)


def remove_hair_dullrazor(
    bgr: np.ndarray,
    *,
    pct: float = 91.0,
    clahe_clip_limit: float = 3.0,
    clahe_tile_grid: Tuple[int, int] = (8, 8),
    kernel_sizes: List[int] | None = None,
    min_area: int = 30,
    inpaint_radius: int = 3,
) -> Dict[str, np.ndarray]:
    """
    Multi-scale black-hat hair detection + inpainting.
    Returns intermediate images for visualization/saving.
    """
    if kernel_sizes is None:
        kernel_sizes = [9, 13, 17, 21, 27, 31]

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Contrast boost helps when hair is only slightly darker than skin
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip_limit), tileGridSize=clahe_tile_grid)
    gray_c = clahe.apply(gray)

    # Multi-scale black-hat
    blackhat_stack = []
    for ks in kernel_sizes:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        bh = cv2.morphologyEx(gray_c, cv2.MORPH_BLACKHAT, k)
        blackhat_stack.append(bh)

    blackhat = np.max(np.stack(blackhat_stack, axis=0), axis=0).astype(np.uint8)
    blackhat_n = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)

    # Percentile threshold
    thr = float(np.percentile(blackhat_n, float(pct)))
    thr = max(thr, 10.0)
    hair_mask = (blackhat_n >= thr).astype(np.uint8) * 255

    # Cleanup
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8), iterations=1)
    hair_mask = cv2.dilate(hair_mask, np.ones((3, 3), np.uint8), iterations=1)

    # Remove tiny components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hair_mask, connectivity=8)
    clean = np.zeros_like(hair_mask)
    for i in range(1, num_labels):
        if int(stats[i, cv2.CC_STAT_AREA]) >= int(min_area):
            clean[labels == i] = 255
    hair_mask = clean

    # Inpaint
    bgr_inpaint = cv2.inpaint(bgr, hair_mask, float(inpaint_radius), cv2.INPAINT_TELEA)

    # Light smoothing
    bgr_out = cv2.bilateralFilter(bgr_inpaint, d=5, sigmaColor=35, sigmaSpace=35)

    return {
        "blackhat_response": blackhat_n,
        "hair_mask": hair_mask,
        "bgr_hair_cleaned": bgr_out,
    }


def segment_lesion_classic(bgr_clean: np.ndarray, *, close_kernel: int = 15) -> Dict[str, np.ndarray]:
    """
    Classic lesion segmentation:
    - Lab chroma distance likelihood
    - Otsu threshold
    - keep largest component immediately
    - fill holes + close/open + smooth
    """
    lab = cv2.cvtColor(bgr_clean, cv2.COLOR_BGR2LAB)
    _, A, B = cv2.split(lab)

    A_f = A.astype(np.float32)
    B_f = B.astype(np.float32)

    A0 = np.median(A_f)
    B0 = np.median(B_f)

    likelihood = np.abs(A_f - A0) + 0.8 * np.abs(B_f - B0)
    likelihood = cv2.GaussianBlur(likelihood, (0, 0), sigmaX=2.5, sigmaY=2.5)
    likelihood_u8 = cv2.normalize(likelihood, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    t, otsu_mask = cv2.threshold(likelihood_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If threshold selects mostly background, invert
    if (otsu_mask.mean() / 255.0) > 0.6:
        otsu_mask = cv2.bitwise_not(otsu_mask)

    # Keep largest component immediately
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(otsu_mask, connectivity=8)
    if num_labels <= 1:
        # empty
        lesion_mask = np.zeros_like(otsu_mask)
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = 1 + int(np.argmax(areas))
        lesion_mask = (labels == largest).astype(np.uint8) * 255

    # Fill holes
    h, w = lesion_mask.shape
    ff = lesion_mask.copy()
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mask_ff, seedPoint=(0, 0), newVal=255)
    holes = cv2.bitwise_not(ff)
    lesion_mask_filled = cv2.bitwise_or(lesion_mask, holes)

    # Close then light open
    lesion_mask_clean = cv2.morphologyEx(
        lesion_mask_filled,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(close_kernel), int(close_kernel))),
        iterations=1,
    )
    lesion_mask_clean = cv2.morphologyEx(
        lesion_mask_clean,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    # Smooth boundary
    lesion_mask_clean = cv2.GaussianBlur(lesion_mask_clean, (5, 5), 0)
    _, lesion_mask_clean = cv2.threshold(lesion_mask_clean, 127, 255, cv2.THRESH_BINARY)

    # Contour overlay + masked lesion
    contours, _ = cv2.findContours(lesion_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = bgr_clean.copy()
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(overlay, [c], -1, (0, 255, 0), 2)

    masked = cv2.bitwise_and(bgr_clean, bgr_clean, mask=lesion_mask_clean)

    return {
        "likelihood_u8": likelihood_u8,
        "otsu_mask": otsu_mask,
        "lesion_mask": lesion_mask_clean,
        "bgr_overlay": overlay,
        "bgr_masked_lesion": masked,
        "otsu_threshold": np.array([t], dtype=np.float32),
    }


def save_combined_charts(
    out_path: Path,
    *,
    rgb_original: np.ndarray,
    rgb_hair_cleaned: np.ndarray,
    likelihood_u8: np.ndarray,
    otsu_mask: np.ndarray,
    lesion_mask: np.ndarray,
    rgb_overlay: np.ndarray,
    rgb_masked: np.ndarray,
) -> None:
    """
    Saves one figure containing all requested panels.
    """
    fig = plt.figure(figsize=(18, 9))

    panels = [
        (rgb_original, "Original image", "rgb"),
        (rgb_hair_cleaned, "Hair Cleaned input", "rgb"),
        (likelihood_u8, "Lesion likelihood map", "gray"),
        (otsu_mask, "Mask right after Otsu", "gray"),
        (lesion_mask, "Final lesion mask", "gray"),
        (rgb_overlay, "Contour Overlay", "rgb"),
        (rgb_masked, "Masked lesion", "rgb"),
    ]

    # 2 rows x 4 cols (last slot empty)
    for i, (img, title, mode) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 4, i)
        if mode == "rgb":
            ax.imshow(img)
        else:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")

    # empty last subplot
    ax = fig.add_subplot(2, 4, 8)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def process_one_image(img_path: Path, out_root: Path, args: argparse.Namespace) -> None:
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print(f"[WARN] Could not read: {img_path.name}")
        return

    rgb = bgr_to_rgb(bgr)

    # Step 1: hair removal
    hair = remove_hair_dullrazor(
        bgr,
        pct=args.hair_pct,
        clahe_clip_limit=args.clahe_clip,
        min_area=args.hair_min_area,
        inpaint_radius=args.inpaint_radius,
    )
    bgr_clean = hair["bgr_hair_cleaned"]
    rgb_clean = bgr_to_rgb(bgr_clean)

    # Step 2: lesion segmentation
    seg = segment_lesion_classic(bgr_clean, close_kernel=args.close_kernel)

    # Output folder per image
    img_out_dir = out_root / img_path.stem
    ensure_dir(img_out_dir)

    # Save requested panels
    save_rgb(img_out_dir / "01_original.png", rgb)
    save_rgb(img_out_dir / "02_hair_cleaned.png", rgb_clean)
    save_gray(img_out_dir / "03_likelihood_map.png", seg["likelihood_u8"])
    save_gray(img_out_dir / "04_otsu_mask.png", seg["otsu_mask"])
    save_gray(img_out_dir / "05_final_lesion_mask.png", seg["lesion_mask"])
    save_rgb(img_out_dir / "06_contour_overlay.png", bgr_to_rgb(seg["bgr_overlay"]))
    save_rgb(img_out_dir / "07_masked_lesion.png", bgr_to_rgb(seg["bgr_masked_lesion"]))

    # Save combined chart
    save_combined_charts(
        img_out_dir / f"{img_path.stem}_charts.png",
        rgb_original=rgb,
        rgb_hair_cleaned=rgb_clean,
        likelihood_u8=seg["likelihood_u8"],
        otsu_mask=seg["otsu_mask"],
        lesion_mask=seg["lesion_mask"],
        rgb_overlay=bgr_to_rgb(seg["bgr_overlay"]),
        rgb_masked=bgr_to_rgb(seg["bgr_masked_lesion"]),
    )

    print(f"[OK] {img_path.name} -> {img_out_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch hair removal + classic lesion segmentation for all images in the current directory.")
    p.add_argument("--outdir", default="lesion_charts", help="Output folder created inside the current directory.")
    p.add_argument("--hair-pct", dest="hair_pct", type=float, default=91.0, help="Percentile threshold for hair detection (lower=more detections).")
    p.add_argument("--clahe-clip", dest="clahe_clip", type=float, default=3.0, help="CLAHE clipLimit for hair detection preprocessing.")
    p.add_argument("--hair-min-area", dest="hair_min_area", type=int, default=30, help="Minimum connected component area for hair mask cleanup.")
    p.add_argument("--inpaint-radius", dest="inpaint_radius", type=int, default=3, help="Inpainting radius.")
    p.add_argument("--close-kernel", dest="close_kernel", type=int, default=15, help="Kernel size for MORPH_CLOSE during lesion mask cleanup.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    images = list_images_in_cwd()
    if not images:
        print("[INFO] No images found in the current directory.")
        print(f"[INFO] Supported extensions: {sorted(IMAGE_EXTS)}")
        return

    out_root = Path(args.outdir).resolve()
    ensure_dir(out_root)

    for img_path in images:
        process_one_image(img_path, out_root, args)


if __name__ == "__main__":
    main()

"""Candidate matcher for carpet photographs and machine-design BMP files.

The matcher intentionally ranks candidates rather than claiming a definitive ID:
machine bitmaps encode construction data, while query images show installed carpet.
Only Pillow and NumPy are required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


VERSION = 1
IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MAX_ALIGNMENT_ANGLE = 40


def load_rgb(path: Path, max_side: int = 512) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def crop_fraction(image: Image.Image, crop: str | None) -> Image.Image:
    if not crop:
        return image
    values = [float(value) for value in crop.split(",")]
    if len(values) != 4 or not all(0 <= value <= 1 for value in values):
        raise ValueError("crop must be left,top,right,bottom fractions between 0 and 1")
    left, top, right, bottom = values
    if left >= right or top >= bottom:
        raise ValueError("crop right/bottom must be greater than left/top")
    return image.crop((int(left * image.width), int(top * image.height),
                       int(right * image.width), int(bottom * image.height)))


def _vertical_alignment_score(image: Image.Image, correction: float) -> float:
    """Score how strongly texture features run vertically after a rotation."""
    rotated = image.rotate(correction, resample=Image.Resampling.BILINEAR, expand=False)
    array = np.asarray(rotated, dtype=np.float32) / 255.0
    margin_y = max(1, int(array.shape[0] * .18))
    margin_x = max(1, int(array.shape[1] * .18))
    array = array[margin_y:-margin_y, margin_x:-margin_x]
    # Vertical features remain coherent down the image. Their horizontal
    # derivative therefore has a strong column projection when upright.
    horizontal_edge = np.abs(np.diff(array, axis=1))
    projection = horizontal_edge.mean(axis=0)
    projection -= projection.mean()
    return float(np.mean(projection * projection))


def estimate_vertical_correction(image: Image.Image) -> tuple[float, float]:
    """Return rotation needed to make the dominant near-vertical texture upright."""
    gray = ImageOps.autocontrast(ImageOps.grayscale(image))
    gray.thumbnail((420, 420), Image.Resampling.LANCZOS)
    coarse_angles = np.arange(-MAX_ALIGNMENT_ANGLE, MAX_ALIGNMENT_ANGLE + .01, 2.0)
    coarse_scores = np.array([_vertical_alignment_score(gray, float(angle)) for angle in coarse_angles])
    coarse_best = float(coarse_angles[int(np.argmax(coarse_scores))])
    fine_angles = np.arange(coarse_best - 2, coarse_best + 2.01, .25)
    fine_scores = np.array([_vertical_alignment_score(gray, float(angle)) for angle in fine_angles])
    best_index = int(np.argmax(fine_scores))
    best = float(fine_angles[best_index])
    baseline = float(np.median(coarse_scores)) + 1e-12
    confidence = max(0.0, min(1.0, (float(fine_scores[best_index]) / baseline - 1.0) / 2.0))
    return best, confidence


def _largest_valid_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Find the largest axis-aligned rectangle containing no rotation fill."""
    heights = np.zeros(mask.shape[1], dtype=np.int32)
    best_area, best = 0, (0, 0, mask.shape[1], mask.shape[0])
    for row, valid in enumerate(mask):
        heights = np.where(valid, heights + 1, 0)
        stack: list[tuple[int, int]] = []
        for column in range(mask.shape[1] + 1):
            height = int(heights[column]) if column < mask.shape[1] else 0
            start = column
            while stack and stack[-1][1] > height:
                start0, previous_height = stack.pop()
                area = previous_height * (column - start0)
                if area > best_area:
                    best_area = area
                    best = (start0, row - previous_height + 1, column, row + 1)
                start = start0
            if not stack or stack[-1][1] < height:
                stack.append((start, height))
    return best


def align_and_crop(image: Image.Image) -> tuple[Image.Image, float, float]:
    correction, confidence = estimate_vertical_correction(image)
    if confidence < .08 or abs(correction) < .25:
        return image, 0.0, confidence
    rotated = image.rotate(correction, resample=Image.Resampling.BICUBIC, expand=True)
    mask = Image.new("L", image.size, 255).rotate(
        correction, resample=Image.Resampling.NEAREST, expand=True, fillcolor=0
    )
    rectangle = _largest_valid_rectangle(np.asarray(mask) > 0)
    aligned = rotated.crop(rectangle)
    return aligned, correction, confidence


def perspective_rectify(image: Image.Image, points: str | None) -> Image.Image:
    """Map a known rectangular quadrilateral (TL,TR,BR,BL) to a true rectangle."""
    if not points:
        return image
    values = [float(value.strip()) for value in points.split(",")]
    if len(values) != 8 or not all(0 <= value <= 1 for value in values):
        raise ValueError("perspective points must be 8 fractions from 0 to 1: TL,TR,BR,BL")
    source = np.array([
        (values[index] * image.width, values[index + 1] * image.height)
        for index in range(0, 8, 2)
    ], dtype=np.float64)
    tl, tr, br, bl = source
    width = max(32, int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))))
    height = max(32, int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))))
    destination = np.array([(0, 0), (width, 0), (width, height), (0, height)], dtype=np.float64)

    # Pillow requires the inverse projective map from output to source.
    matrix, target = [], []
    for (x, y), (u, v) in zip(destination, source):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        target.append(u)
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        target.append(v)
    coefficients = np.linalg.solve(np.asarray(matrix), np.asarray(target))
    return image.transform(
        (width, height), Image.Transform.PERSPECTIVE, tuple(coefficients),
        resample=Image.Resampling.BICUBIC,
    )


def grayscale_hypotheses(image: Image.Image, is_design: bool) -> list[np.ndarray]:
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray = np.asarray(ImageOps.grayscale(image), dtype=np.float32) / 255.0
    if not is_design:
        return [gray]

    # Machine BMP colors are codes.  Compare several code interpretations and
    # retain the best score for each query/candidate pair.
    packed = (np.asarray(image, dtype=np.uint32)[:, :, 0] << 16) | \
             (np.asarray(image, dtype=np.uint32)[:, :, 1] << 8) | \
             np.asarray(image, dtype=np.uint32)[:, :, 2]
    colors, inverse, counts = np.unique(packed, return_inverse=True, return_counts=True)
    order = np.argsort(-counts)
    frequency_rank = np.empty_like(order)
    frequency_rank[order] = np.arange(len(order))
    coded = frequency_rank[inverse].reshape(packed.shape).astype(np.float32)
    if len(colors) > 1:
        coded /= len(colors) - 1
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    return [gray, coded, 1.0 - coded, saturation]


def normalize(array: np.ndarray, size: int = 256) -> np.ndarray:
    lo, hi = np.percentile(array, (2, 98))
    if hi > lo:
        array = np.clip((array - lo) / (hi - lo), 0, 1)
    image = Image.fromarray(np.uint8(array * 255), mode="L").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(ImageEnhance.Contrast(image).enhance(1.5), dtype=np.float32) / 255.0


def descriptor(array: np.ndarray) -> np.ndarray:
    a = normalize(array)
    a -= a.mean()
    window = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1])).astype(np.float32)
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(a * window))))
    yy, xx = np.indices(spectrum.shape)
    cy, cx = (np.array(spectrum.shape) - 1) / 2
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    angle = (np.arctan2(yy - cy, xx - cx) + np.pi) % np.pi

    features: list[float] = []
    for bins, values, maximum in ((32, radius, radius.max()), (24, angle, np.pi)):
        ids = np.minimum((values / maximum * bins).astype(int), bins - 1)
        features.extend([float(spectrum[ids == i].mean()) for i in range(bins)])

    # Coarse spatial structure at several scales, robust to color and modest blur.
    for grid in (4, 8, 16):
        small = np.asarray(Image.fromarray(np.uint8((a - a.min()) / (np.ptp(a) + 1e-6) * 255), mode="L")
                           .resize((grid, grid), Image.Resampling.BOX), dtype=np.float32).ravel()
        features.extend(small.tolist())
    vector = np.asarray(features, dtype=np.float32)
    vector -= vector.mean()
    vector /= np.linalg.norm(vector) + 1e-8
    return vector


def design_descriptors(path: Path) -> np.ndarray:
    image = load_rgb(path)
    vectors = []
    for hypothesis in grayscale_hypotheses(image, is_design=True):
        for turns in range(4):
            vectors.append(descriptor(np.rot90(hypothesis, turns)))
    return np.stack(vectors)


def file_signature(path: Path) -> str:
    stat = path.stat()
    return f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"


def pattern_metadata(path_value: str) -> dict[str, str]:
    """Extract machine type from .../<machine type>/Bitmap files/<pattern>.BMP."""
    path = Path(path_value)
    machine_type = "Unknown"
    for parent in path.parents:
        if parent.name.casefold() == "bitmap files":
            machine_type = parent.parent.name
            break
    return {
        "machine_type": machine_type,
        "pattern_name": path.stem,
        "file_name": path.name,
        "path": str(path),
    }


def build_index(root: Path, output: Path) -> int:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".bmp")
    if not files:
        raise RuntimeError(f"No BMP files found under {root}")
    paths, signatures, vectors = [], [], []
    for number, path in enumerate(files, 1):
        print(f"[{number}/{len(files)}] {path}", flush=True)
        try:
            vectors.append(design_descriptors(path))
            paths.append(str(path.resolve()))
            signatures.append(file_signature(path))
        except Exception as error:
            print(f"  skipped: {error}", file=sys.stderr)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, version=VERSION, paths=np.array(paths), signatures=np.array(signatures),
                        vectors=np.stack(vectors))
    print(f"Indexed {len(paths)} BMPs in {output}", flush=True)
    return len(paths)


def load_index(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    if int(data["version"]) != VERSION:
        raise RuntimeError("Index version is obsolete; rebuild it")
    return data["paths"], data["vectors"]


def query_descriptors(path: Path, crop: str | None, preview: Path | None = None,
                      perspective: str | None = None) -> np.ndarray:
    image = crop_fraction(load_rgb(path, 1024), crop)
    image, correction, confidence = align_and_crop(image)
    image = perspective_rectify(image, perspective)
    print(
        f"Alignment: rotated {correction:+.2f} degrees; confidence {confidence:.0%}; "
        f"search crop {image.width} x {image.height}",
        flush=True,
    )
    if preview:
        preview.parent.mkdir(parents=True, exist_ok=True)
        image.save(preview)
        print(f"Aligned preview: {preview.resolve()}", flush=True)
    gray = grayscale_hypotheses(image, is_design=False)[0]
    # Multiple overlapping regions reduce sensitivity to walls, furniture, and seams.
    h, w = gray.shape
    regions = [gray]
    for y0, y1, x0, x1 in ((0, .75, 0, .75), (0, .75, .25, 1), (.25, 1, 0, .75), (.25, 1, .25, 1)):
        regions.append(gray[int(y0*h):int(y1*h), int(x0*w):int(x1*w)])
    return np.stack([descriptor(region) for region in regions])


def match(index: Path, query: Path, crop: str | None, top: int,
          preview: Path | None = None, perspective: str | None = None) -> list[dict[str, object]]:
    paths, candidates = load_index(index)
    queries = query_descriptors(query, crop, preview, perspective)
    # candidates: item x hypothesis x feature; queries: region x feature
    similarity = np.einsum("ihf,rf->ihr", candidates, queries)
    best = similarity.max(axis=(1, 2))
    order = np.argsort(-best)[:top]
    results = []
    for rank, i in enumerate(order, 1):
        result: dict[str, object] = {
            "rank": rank,
            "score": round(float(best[i]) * 100, 2),
        }
        result.update(pattern_metadata(str(paths[i])))
        results.append(result)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank machine-design BMPs against a carpet photograph")
    sub = parser.add_subparsers(dest="command", required=True)
    index_parser = sub.add_parser("index", help="recursively index a BMP library")
    index_parser.add_argument("library", type=Path)
    index_parser.add_argument("--output", type=Path, default=Path("carpet-index.npz"))
    match_parser = sub.add_parser("match", help="return the most likely candidate patterns")
    match_parser.add_argument("query", type=Path)
    match_parser.add_argument("--index", type=Path, default=Path("carpet-index.npz"))
    match_parser.add_argument("--top", type=int, default=10)
    match_parser.add_argument("--crop", help="optional carpet ROI: left,top,right,bottom as 0..1 fractions")
    match_parser.add_argument("--preview", type=Path, help="save the aligned/cropped query image")
    match_parser.add_argument("--perspective", help="TLx,TLy,TRx,TRy,BRx,BRy,BLx,BLy fractions")
    match_parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.command == "index":
        build_index(args.library, args.output)
    else:
        results = match(args.index, args.query, args.crop, max(1, args.top), args.preview, args.perspective)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            for result in results:
                print(
                    f"{result['rank']:>2}. {result['score']:>6.2f}  "
                    f"Machine: {result['machine_type']}  Pattern: {result['pattern_name']}  "
                    f"File: {result['path']}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

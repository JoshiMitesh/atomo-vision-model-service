#!/usr/bin/env python3
"""Benchmark plate OCR engines on real plate-crop images.

Input directory should contain cropped plate images (one plate per image).
Optional labels.csv format: image,plate  (image is the filename in the directory).
This benchmark runs locally; it does not change the production OCR engine.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import time
import sys
from pathlib import Path

import cv2

# Allow both "python tools/benchmark_plate_ocr.py" and module-style execution
# from the repository root, where the "slave" package lives.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slave.processes.worker.models.number_plate.awiros_ocr import AwirosPlateOcr
from slave.processes.worker.models.number_plate.plate_ocr import PlateOcr


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_labels(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    labels: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"image", "plate"}.issubset(reader.fieldnames):
            raise ValueError("labels CSV must have columns: image,plate")
        for row in reader:
            labels[Path(row["image"]).name] = "".join(
                ch for ch in row["plate"].upper() if ch.isalnum()
            )
    return labels


def summarize(name: str, timings: list[float], predictions: dict[str, str],
              labels: dict[str, str]) -> None:
    if not timings:
        print(f"{name}: no results")
        return
    ordered = sorted(timings)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    print(f"\n{name}")
    print(f"  images: {len(timings)}")
    print(f"  mean latency:   {statistics.mean(timings):.1f} ms")
    print(f"  median latency: {statistics.median(timings):.1f} ms")
    print(f"  p95 latency:    {p95:.1f} ms")
    print(f"  throughput:     {1000.0 / statistics.mean(timings):.2f} plates/sec")
    if labels:
        compared = [(key, predictions.get(key, ""), expected)
                    for key, expected in labels.items() if key in predictions]
        exact = sum(pred == expected for _, pred, expected in compared)
        print(f"  exact match:    {exact}/{len(compared)} ({100.0 * exact / len(compared):.1f}%)"
              if compared else "  exact match:    no matching labeled images")
        for key, pred, expected in compared:
            if pred != expected:
                print(f"    MISMATCH {key}: predicted={pred!r} expected={expected!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path,
                        help="directory of plate-crop images (one plate per image)")
    parser.add_argument("--model-dir", default="slave/processes/worker/models/number_plate/awiros",
                        help="Awiros weights/dictionary directory")
    parser.add_argument("--fast-model", default="cct-s-v2-global-model",
                        help="FastPlateOCR model used for multi-variant pipeline (default: current S model)")
    parser.add_argument("--labels", type=Path,
                        help="optional CSV with columns image,plate for exact-match accuracy")
    parser.add_argument("--warmup", type=int, default=2,
                        help="warm-up runs per engine (default: 2)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="timed repeats per image (default: 1)")
    parser.add_argument("--cpu-threads", type=int, default=2,
                        help="Paddle CPU threads (default: 2)")
    parser.add_argument("--skip-awiros", action="store_true",
                        help="skip Awiros initialization/inference")
    args = parser.parse_args()

    image_paths = sorted(p for p in args.images.iterdir()
                         if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not image_paths:
        parser.error(f"no supported images found in {args.images}")
    labels = load_labels(args.labels)
    samples = []
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            print(f"Skipping unreadable image: {path}")
            continue
        h, w = image.shape[:2]
        samples.append((path.name, image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                        (0, 0, w, h)))
    if not samples:
        parser.error("no readable images found")

    print(f"Images: {len(samples)} | CPU threads: {args.cpu_threads}")
    print("Run this while the machine is otherwise idle for comparable timings.")
    print("The first model load/download is excluded from per-image latency.")

    fast = PlateOcr(args.fast_model, pad_ratios=(0.10, 0.20))
    fast_xs = PlateOcr("cct-xs-v2-global-model", pad_ratios=(0.10,))
    awiros = None
    if not args.skip_awiros:
        awiros = AwirosPlateOcr(str(args.model_dir), device="cpu",
                               cpu_threads=args.cpu_threads)

    results: dict[str, dict[str, str]] = {}
    times: dict[str, list[float]] = {}

    def run_case(name, fn, warmup_count):
        # Warm up using the first real crop.
        for _ in range(max(0, warmup_count)):
            fn(samples[0])
        timings: list[float] = []
        predictions: dict[str, str] = {}
        for filename, image, gray, bbox in samples:
            best_plate = ""
            per_image = []
            for _ in range(max(1, args.repeat)):
                start = time.perf_counter()
                best_plate = fn((filename, image, gray, bbox))
                per_image.append((time.perf_counter() - start) * 1000.0)
            timings.append(statistics.median(per_image))
            predictions[filename] = "".join(ch for ch in best_plate.upper() if ch.isalnum())
        times[name] = timings
        results[name] = predictions
        summarize(name, timings, predictions, labels)

    try:
        run_case("FastPlateOCR XS - single pass",
                 lambda s: fast_xs._run_ocr(s[2])[0], args.warmup)
        run_case(f"FastPlateOCR {args.fast_model} - multi-variant pipeline",
                 lambda s: fast.read_plate(s[2], s[3])[0], args.warmup)
        if awiros is not None:
            run_case("Awiros PaddleOCR - current pipeline",
                     lambda s: awiros.read_plate(s[1], s[3])[0], args.warmup)
    finally:
        fast.release()
        fast_xs.release()
        if awiros is not None:
            awiros.release()

    if labels:
        print("\nAccuracy caveat: exact match is meaningful only for correctly labeled images.")
    print("\nBenchmark complete. No production config or engine was changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def enhance_to_white_bg_black_text(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (101, 101), 0)
    flattened = cv2.divide(gray, background, scale=255)
    binary = cv2.adaptiveThreshold(
        flattened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )
    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enhance a corrected paper image to white background and black text.")
    parser.add_argument("--input", required=True, help="Path to the input image.")
    parser.add_argument("--output", required=True, help="Path to the output image.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        raise SystemExit(f"Failed to read input image: {args.input}")

    enhanced = enhance_to_white_bg_black_text(image)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), enhanced)
    if not success:
        raise SystemExit(f"Failed to write output image: {args.output}")

    print(f"saved={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

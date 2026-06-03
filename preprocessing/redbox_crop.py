from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class RedBoxRegion:
    """表示从标记图中检测到的红框区域。"""

    index: int
    x: int
    y: int
    w: int
    h: int

    @property
    def bbox(self) -> list[int]:
        return [self.x, self.y, self.x + self.w, self.y + self.h]

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


def detect_red_boxes(image: np.ndarray, min_area: int = 2000) -> list[RedBoxRegion]:
    """检测图中的红色矩形框。"""

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 80, 80], dtype=np.uint8)
    upper_red1 = np.array([12, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([168, 80, 80], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: list[RedBoxRegion] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) < 4:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 30 or h < 30:
            continue
        regions.append(RedBoxRegion(index=0, x=x, y=y, w=w, h=h))

    ordered = sorted(regions, key=lambda region: (region.cy, region.x))
    normalized: list[RedBoxRegion] = []
    for idx, region in enumerate(ordered, start=1):
        normalized.append(RedBoxRegion(index=idx, x=region.x, y=region.y, w=region.w, h=region.h))
    return normalized


def crop_regions_from_image(image: np.ndarray, regions: list[RedBoxRegion], output_dir: str | Path, padding: int = 4) -> list[dict[str, object]]:
    """将检测到的区域从图中裁出并保存。"""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_h, image_w = image.shape[:2]
    outputs: list[dict[str, object]] = []

    for region in regions:
        x1 = max(0, region.x + padding)
        y1 = max(0, region.y + padding)
        x2 = min(image_w, region.x + region.w - padding)
        y2 = min(image_h, region.y + region.h - padding)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image[y1:y2, x1:x2]
        crop_path = out_dir / f"q_{region.index:02d}.png"
        success = cv2.imwrite(str(crop_path), crop)
        if not success:
            raise SystemExit(f"Failed to write crop image: {crop_path}")
        outputs.append(
            {
                "index": region.index,
                "bbox": [x1, y1, x2, y2],
                "path": str(crop_path),
            }
        )
    return outputs


def draw_detected_boxes(image: np.ndarray, regions: list[RedBoxRegion]) -> np.ndarray:
    """在图上可视化检测到的红框区域。"""

    overlay = image.copy()
    for region in regions:
        x1, y1, x2, y2 = region.bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(
            overlay,
            str(region.index),
            (x1, max(24, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return overlay


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect red rectangles from a marked image and crop each boxed region.")
    parser.add_argument("--input", required=True, help="Path to the marked input image.")
    parser.add_argument("--output-dir", required=True, help="Directory to save cropped question images.")
    parser.add_argument("--output-json", default=None, help="Optional path to save crop metadata JSON.")
    parser.add_argument("--output-overlay", default=None, help="Optional path to save detected green-box overlay.")
    parser.add_argument("--min-area", type=int, default=2000, help="Minimum contour area to keep as a red box.")
    parser.add_argument("--padding", type=int, default=4, help="Inner padding when cropping boxed regions.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        raise SystemExit(f"Failed to read input image: {args.input}")

    regions = detect_red_boxes(image, min_area=args.min_area)
    crops = crop_regions_from_image(image, regions, args.output_dir, padding=args.padding)

    payload = {
        "question_count": len(crops),
        "questions": crops,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)

    if args.output_json:
        output_json_path = Path(args.output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(rendered, encoding="utf-8")

    if args.output_overlay:
        overlay = draw_detected_boxes(image, regions)
        output_overlay_path = Path(args.output_overlay)
        output_overlay_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(output_overlay_path), overlay)
        if not success:
            raise SystemExit(f"Failed to write overlay image: {args.output_overlay}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

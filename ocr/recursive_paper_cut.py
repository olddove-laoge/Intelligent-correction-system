from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ocr.aliyun_paper_cut import draw_detected_regions, recognize_edu_paper_cut


@dataclass
class Region:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def height(self) -> int:
        return max(0, self.y1 - self.y0)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_list(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]


@dataclass
class ChildRegion:
    region: Region
    source: str
    label: str


class RecursivePaperCut:
    def __init__(
        self,
        image: np.ndarray,
        output_dir: Path,
        *,
        subject: str,
        image_type: str,
        cut_type: str,
        output_oricoord: bool,
        max_depth: int,
        min_region_height: int,
        min_region_area: int,
        strip_padding: int,
        access_key_id: str | None,
        access_key_secret: str | None,
        endpoint: str,
    ) -> None:
        self.image = image
        self.output_dir = output_dir
        self.subject = subject
        self.image_type = image_type
        self.cut_type = cut_type
        self.output_oricoord = output_oricoord
        self.max_depth = max_depth
        self.min_region_height = min_region_height
        self.min_region_area = min_region_area
        self.strip_padding = strip_padding
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
        self.endpoint = endpoint
        self.node_counter = count(1)
        self.records: list[dict[str, Any]] = []

    def run(self) -> list[dict[str, Any]]:
        root = Region(0, 0, self.image.shape[1], self.image.shape[0])
        self._process_region(root, level=0, source="root", label="root", parent_id=None)
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(self.records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self.records

    def _process_region(
        self,
        region: Region,
        *,
        level: int,
        source: str,
        label: str,
        parent_id: str | None,
    ) -> None:
        node_id = f"node_{next(self.node_counter):03d}"
        node_dir = self.output_dir / node_id
        node_dir.mkdir(parents=True, exist_ok=True)

        crop = self.image[region.y0:region.y1, region.x0:region.x1]
        crop_path = node_dir / "crop.jpg"
        cv2.imwrite(str(crop_path), crop)

        result = recognize_edu_paper_cut(
            crop_path,
            subject=self.subject,
            cut_type=self.cut_type,
            image_type=self.image_type,
            output_oricoord=self.output_oricoord,
            endpoint=self.endpoint,
            access_key_id=self.access_key_id,
            access_key_secret=self.access_key_secret,
        )
        (node_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        overlay = draw_detected_regions(crop, result)
        cv2.imwrite(str(node_dir / "overlay.jpg"), overlay)

        question_regions = extract_question_regions(result)
        children = self._build_children(region, question_regions)
        split_overlay = draw_split_overlay(crop, region, question_regions, children, region.width, region.height)
        cv2.imwrite(str(node_dir / "split_overlay.jpg"), split_overlay)

        stop_reason = self._determine_stop_reason(region, level, question_regions, children)
        record = {
            "node_id": node_id,
            "parent_id": parent_id,
            "level": level,
            "source": source,
            "label": label,
            "region": region.to_list(),
            "question_count": len(question_regions),
            "question_labels": [item["label"] for item in question_regions],
            "question_boxes": [item["bbox"] for item in question_regions],
            "children": [
                {
                    "source": child.source,
                    "label": child.label,
                    "region": child.region.to_list(),
                }
                for child in children
            ],
            "stop_reason": stop_reason,
        }
        self.records.append(record)

        if stop_reason is not None:
            return

        for child in children:
            self._process_region(
                child.region,
                level=level + 1,
                source=child.source,
                label=child.label,
                parent_id=node_id,
            )

    def _determine_stop_reason(
        self,
        region: Region,
        level: int,
        question_regions: list[dict[str, Any]],
        children: list[ChildRegion],
    ) -> str | None:
        if level >= self.max_depth:
            return "max_depth"
        if not question_regions:
            return "no_question"
        if len(question_regions) != 1:
            return "question_count_changed"
        if not children:
            return "no_child_region"
        if all(child.region.area >= region.area * 0.995 for child in children if child.source == "cut"):
            return "stable_cut_region"
        return None

    def _build_children(self, region: Region, question_regions: list[dict[str, Any]]) -> list[ChildRegion]:
        if not question_regions:
            return []

        intervals: list[tuple[int, int, str, str]] = []
        for index, item in enumerate(sorted(question_regions, key=lambda it: it["bbox"][1]), start=1):
            _, y0, _, y1 = item["bbox"]
            start = max(0, y0 - self.strip_padding)
            end = min(region.height, y1 + self.strip_padding)
            if end - start < self.min_region_height:
                continue
            label = item["label"] or str(index)
            intervals.append((start, end, "cut", label))

        if not intervals:
            return []

        merged_cut: list[tuple[int, int, str, str]] = []
        for start, end, source, label in intervals:
            if not merged_cut or start > merged_cut[-1][1]:
                merged_cut.append((start, end, source, label))
            else:
                prev_start, prev_end, _, prev_label = merged_cut[-1]
                merged_cut[-1] = (prev_start, max(prev_end, end), source, f"{prev_label}+{label}")

        children: list[ChildRegion] = []
        cursor = 0
        gap_index = 1
        for start, end, _, label in merged_cut:
            if start - cursor >= self.min_region_height:
                gap_region = Region(region.x0, region.y0 + cursor, region.x1, region.y0 + start)
                if gap_region.area >= self.min_region_area:
                    children.append(ChildRegion(gap_region, "uncut", f"gap_{gap_index}"))
                    gap_index += 1
            cut_region = Region(region.x0, region.y0 + start, region.x1, region.y0 + end)
            if cut_region.area >= self.min_region_area:
                children.append(ChildRegion(cut_region, "cut", label))
            cursor = max(cursor, end)

        if region.height - cursor >= self.min_region_height:
            gap_region = Region(region.x0, region.y0 + cursor, region.x1, region.y1)
            if gap_region.area >= self.min_region_area:
                children.append(ChildRegion(gap_region, "uncut", f"gap_{gap_index}"))

        filtered_children: list[ChildRegion] = []
        for child in children:
            if child.region.width < 10 or child.region.height < self.min_region_height:
                continue
            if child.region.area >= region.area * 0.995:
                continue
            filtered_children.append(child)
        return filtered_children


def extract_question_regions(result: dict[str, Any]) -> list[dict[str, Any]]:
    body = result.get("body")
    if not isinstance(body, dict):
        return []

    data = body.get("Data")
    if not isinstance(data, dict):
        return []

    page_list = data.get("page_list")
    if not isinstance(page_list, list):
        return []

    regions: list[dict[str, Any]] = []
    for page in page_list:
        if not isinstance(page, dict):
            continue
        subject_list = page.get("subject_list")
        if not isinstance(subject_list, list):
            continue
        for subject in subject_list:
            if not isinstance(subject, dict):
                continue
            ids = subject.get("ids")
            label = "-".join(str(item) for item in ids) if isinstance(ids, list) and ids else "?"
            content_list = subject.get("content_list_info")
            if not isinstance(content_list, list):
                continue
            for content in content_list:
                if not isinstance(content, dict):
                    continue
                points = to_point_list(content.get("pos"))
                if len(points) < 4:
                    continue
                xs = [x for x, _ in points]
                ys = [y for _, y in points]
                regions.append(
                    {
                        "label": label,
                        "points": points,
                        "bbox": [min(xs), min(ys), max(xs), max(ys)],
                    }
                )
    return regions


def to_point_list(raw: Any) -> list[tuple[int, int]]:
    if not isinstance(raw, list):
        return []
    points: list[tuple[int, int]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        x = item.get("x")
        y = item.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            points.append((int(round(x)), int(round(y))))
    return points


def draw_split_overlay(
    crop: np.ndarray,
    region: Region,
    question_regions: list[dict[str, Any]],
    children: list[ChildRegion],
    width: int,
    height: int,
) -> np.ndarray:
    overlay = crop.copy()
    for item in question_regions:
        contour = np.array(item["points"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [contour], True, (0, 255, 0), 2, cv2.LINE_AA)
    for child in children:
        local_y0 = max(0, child.region.y0 - region.y0)
        local_y1 = max(local_y0 + 1, child.region.y1 - region.y0)
        color = (255, 0, 0) if child.source == "cut" else (0, 0, 255)
        cv2.rectangle(
            overlay,
            (0, local_y0),
            (width - 1, min(height - 1, local_y1 - 1)),
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"{child.source}:{child.label}",
            (8, max(24, local_y0 + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return overlay


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recursively re-run Aliyun paper cut on cut and uncut strips.")
    parser.add_argument("--input", required=True, help="Path to the input image.")
    parser.add_argument("--output-dir", required=True, help="Directory to save recursive results.")
    parser.add_argument("--subject", default="JHighSchool_Math", help="Aliyun subject enum.")
    parser.add_argument("--image-type", default="scan", help="Aliyun image type, such as scan or photo.")
    parser.add_argument("--cut-type", default="question", help="Aliyun cut type.")
    parser.add_argument("--endpoint", default="ocr-api.cn-hangzhou.aliyuncs.com", help="Aliyun OCR endpoint.")
    parser.add_argument("--max-depth", type=int, default=6, help="Maximum recursion depth.")
    parser.add_argument("--min-region-height", type=int, default=60, help="Minimum child strip height.")
    parser.add_argument("--min-region-area", type=int, default=20000, help="Minimum child strip area.")
    parser.add_argument("--strip-padding", type=int, default=12, help="Vertical padding added around detected strips.")
    parser.add_argument("--disable-oricoord", action="store_true", help="Disable original-coordinate output.")
    parser.add_argument("--access-key-id", default=None, help="Aliyun access key id.")
    parser.add_argument("--access-key-secret", default=None, help="Aliyun access key secret.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        raise SystemExit(f"Failed to read input image: {args.input}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = RecursivePaperCut(
        image,
        output_dir,
        subject=args.subject,
        image_type=args.image_type,
        cut_type=args.cut_type,
        output_oricoord=not args.disable_oricoord,
        max_depth=args.max_depth,
        min_region_height=args.min_region_height,
        min_region_area=args.min_region_area,
        strip_padding=args.strip_padding,
        access_key_id=args.access_key_id,
        access_key_secret=args.access_key_secret,
        endpoint=args.endpoint,
    )
    records = runner.run()
    print(json.dumps({"node_count": len(records), "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

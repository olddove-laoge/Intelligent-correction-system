from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from alibabacloud_credentials.client import Client as CredentialClient
from alibabacloud_ocr_api20210707 import models as ocr_models
from alibabacloud_ocr_api20210707.client import Client as OcrClient
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_tea_util import models as util_models


DEFAULT_ENDPOINT = "ocr-api.cn-hangzhou.aliyuncs.com"
DEFAULT_SUBJECT = "Math"
DEFAULT_CUT_TYPE = "question"
DEFAULT_IMAGE_TYPE = "scan"
DEFAULT_ACCESS_KEY_ID = None
DEFAULT_ACCESS_KEY_SECRET = None


class PaperCutError(RuntimeError):
    pass


def create_client(
    endpoint: str = DEFAULT_ENDPOINT,
    *,
    access_key_id: str | None = None,
    access_key_secret: str | None = None,
) -> OcrClient:
    resolved_access_key_id = access_key_id or DEFAULT_ACCESS_KEY_ID or os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID")
    resolved_access_key_secret = access_key_secret or DEFAULT_ACCESS_KEY_SECRET or os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET")

    if resolved_access_key_id and resolved_access_key_secret:
        config = open_api_models.Config(
            access_key_id=resolved_access_key_id,
            access_key_secret=resolved_access_key_secret,
        )
    else:
        credential = CredentialClient()
        config = open_api_models.Config(credential=credential)

    config.endpoint = endpoint
    return OcrClient(config)


def recognize_edu_paper_cut(
    image_path: str | Path,
    *,
    subject: str = DEFAULT_SUBJECT,
    cut_type: str = DEFAULT_CUT_TYPE,
    image_type: str = DEFAULT_IMAGE_TYPE,
    output_oricoord: bool = True,
    endpoint: str = DEFAULT_ENDPOINT,
    access_key_id: str | None = None,
    access_key_secret: str | None = None,
) -> dict[str, Any]:
    image_file = Path(image_path)
    if not image_file.is_file():
        raise PaperCutError(f"Input image does not exist: {image_file}")

    image_bytes = image_file.read_bytes()
    client = create_client(
        endpoint=endpoint,
        access_key_id=access_key_id,
        access_key_secret=access_key_secret,
    )
    request = ocr_models.RecognizeEduPaperCutRequest(
        body=image_bytes,
        image_type=image_type,
        subject=subject,
        cut_type=cut_type,
        output_oricoord=output_oricoord,
    )
    runtime = util_models.RuntimeOptions()

    try:
        response = client.recognize_edu_paper_cut_with_options(request, runtime)
    except Exception as error:
        message = getattr(error, "message", str(error))
        recommend = None
        data = getattr(error, "data", None)
        if isinstance(data, dict):
            recommend = data.get("Recommend")

        detail = f"Aliyun RecognizeEduPaperCut failed: {message}"
        credential_hints = [
            "No usable Aliyun credentials were found.",
            "Pass --access-key-id and --access-key-secret on the command line,",
            "or set ALIBABA_CLOUD_ACCESS_KEY_ID and ALIBABA_CLOUD_ACCESS_KEY_SECRET in the environment.",
        ]
        lowered_message = message.lower()
        if "credential" in lowered_message or "accesskey" in lowered_message or "providers in the chain" in lowered_message:
            detail = f"{detail}\n" + "\n".join(credential_hints)
        if recommend:
            detail = f"{detail}\nRecommend: {recommend}"
        raise PaperCutError(detail) from error

    if hasattr(response, "to_map"):
        return _normalize_result_payload(response.to_map())
    return _normalize_result_payload(json.loads(json.dumps(response, default=str)))


def _normalize_result_payload(result: dict[str, Any]) -> dict[str, Any]:
    body = result.get("body")
    if not isinstance(body, dict):
        return result

    data = body.get("Data")
    if isinstance(data, str):
        try:
            body["Data"] = json.loads(data)
        except json.JSONDecodeError:
            pass
    return result


def _iter_content_regions(result: dict[str, Any]) -> Iterable[tuple[str, list[tuple[int, int]]]]:
    body = result.get("body")
    if not isinstance(body, dict):
        return

    data = body.get("Data")
    if not isinstance(data, dict):
        return

    page_list = data.get("page_list")
    if not isinstance(page_list, list):
        return

    for page in page_list:
        if not isinstance(page, dict):
            continue

        for list_key in ("subject_list", "answer_list"):
            region_list = page.get(list_key)
            if not isinstance(region_list, list):
                continue

            for region in region_list:
                if not isinstance(region, dict):
                    continue
                ids = region.get("ids")
                label = "-".join(str(item) for item in ids) if isinstance(ids, list) and ids else "?"
                content_list = region.get("content_list_info")
                if not isinstance(content_list, list):
                    continue

                for content in content_list:
                    if not isinstance(content, dict):
                        continue
                    points = _to_point_list(content.get("pos"))
                    if points:
                        yield label, points


def _to_point_list(raw: Any) -> list[tuple[int, int]]:
    if not isinstance(raw, list):
        return []

    if raw and all(isinstance(item, dict) for item in raw):
        points: list[tuple[int, int]] = []
        for item in raw:
            x = item.get("x")
            y = item.get("y")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                points.append((int(round(x)), int(round(y))))
        return points

    if raw and all(isinstance(item, (list, tuple)) and len(item) >= 2 for item in raw):
        return [(int(round(item[0])), int(round(item[1]))) for item in raw]

    flat_numbers = [item for item in raw if isinstance(item, (int, float))]
    if len(flat_numbers) >= 8 and len(flat_numbers) % 2 == 0:
        return [
            (int(round(flat_numbers[index])), int(round(flat_numbers[index + 1])))
            for index in range(0, len(flat_numbers), 2)
        ]

    return []


def draw_detected_regions(image: np.ndarray, result: dict[str, Any]) -> np.ndarray:
    overlay = image.copy()
    color = (0, 255, 0)
    label_color = (0, 0, 255)

    unique_polygons: set[tuple[tuple[int, int], ...]] = set()
    for label, polygon in _iter_content_regions(result):
        if len(polygon) < 4:
            continue

        polygon_key = tuple(polygon)
        if polygon_key in unique_polygons:
            continue
        unique_polygons.add(polygon_key)

        contour = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [contour], True, color, 2, cv2.LINE_AA)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(
            overlay,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            label_color,
            2,
            cv2.LINE_AA,
        )

    return overlay


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call Aliyun RecognizeEduPaperCut on a local image.")
    parser.add_argument("--input", required=True, help="Path to the input image.")
    parser.add_argument("--subject", default=DEFAULT_SUBJECT, help="Subject, for example Math or JHighSchool_Math.")
    parser.add_argument("--cut-type", default=DEFAULT_CUT_TYPE, help="Cut type: question or answer.")
    parser.add_argument("--image-type", default=DEFAULT_IMAGE_TYPE, help="Image type: scan or photo.")
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the raw JSON response.",
    )
    parser.add_argument(
        "--output-overlay",
        default=None,
        help="Optional path to save the image with detected regions drawn on it.",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="Aliyun OCR endpoint.",
    )
    parser.add_argument(
        "--disable-oricoord",
        action="store_true",
        help="Disable original-coordinate output in the API request.",
    )
    parser.add_argument(
        "--access-key-id",
        default=None,
        help="Aliyun access key id. Defaults to the built-in key when omitted.",
    )
    parser.add_argument(
        "--access-key-secret",
        default=None,
        help="Aliyun access key secret. Defaults to the built-in secret when omitted.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    result = recognize_edu_paper_cut(
        args.input,
        subject=args.subject,
        cut_type=args.cut_type,
        image_type=args.image_type,
        output_oricoord=not args.disable_oricoord,
        endpoint=args.endpoint,
        access_key_id=args.access_key_id,
        access_key_secret=args.access_key_secret,
    )

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    stdout = getattr(sys, "stdout", None)
    if stdout is not None:
        try:
            stdout.write(rendered)
            stdout.write("\n")
        except UnicodeEncodeError:
            safe_rendered = rendered.encode(stdout.encoding or "utf-8", errors="replace").decode(stdout.encoding or "utf-8")
            stdout.write(safe_rendered)
            stdout.write("\n")
    else:
        print(rendered)

    if args.output_json:
        output_json_path = Path(args.output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(rendered, encoding="utf-8")

    if args.output_overlay:
        image = cv2.imread(args.input)
        if image is None:
            raise SystemExit(f"Failed to read input image for overlay: {args.input}")
        overlay = draw_detected_regions(image, result)
        output_overlay_path = Path(args.output_overlay)
        output_overlay_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(output_overlay_path), overlay)
        if not success:
            raise SystemExit(f"Failed to write overlay image: {args.output_overlay}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

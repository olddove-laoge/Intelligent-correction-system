from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import requests


APP_KEY = "39b8d51b6eb845c5"
APP_SECRET = "oglFM7Utq1XxYJuQFXC2sWbatokTg3fp"
YOUDAO_CUT_QUESTION_URL = "https://openapi.youdao.com/cut_question"


class YoudaoPaperCutError(RuntimeError):
    pass


def truncate(text: str) -> str:
    size = len(text)
    if size <= 20:
        return text
    return text[:10] + str(size) + text[-10:]


def encrypt(sign_str: str) -> str:
    return hashlib.sha256(sign_str.encode("utf-8")).hexdigest()


def add_auth_params(app_key: str, app_secret: str, data: dict[str, Any]) -> None:
    curtime = str(int(time.time()))
    salt = str(uuid.uuid4())
    sign_str = app_key + truncate(data["q"]) + salt + curtime + app_secret
    data["signType"] = "v3"
    data["curtime"] = curtime
    data["salt"] = salt
    data["appKey"] = app_key
    data["sign"] = encrypt(sign_str)


def encode_image_for_youdao(path: str | Path, max_bytes: int = 1024 * 1024) -> str:
    image_file = Path(path)
    raw = image_file.read_bytes()
    if len(raw) <= max_bytes:
        return base64.b64encode(raw).decode("utf-8")

    image = cv2.imread(str(image_file))
    if image is None:
        raise YoudaoPaperCutError(f"Failed to read image for compression: {image_file}")

    working = image
    for max_side in (2200, 1800, 1500, 1200, 1000):
        height, width = working.shape[:2]
        current_max_side = max(height, width)
        if current_max_side > max_side:
            scale = max_side / current_max_side
            working = cv2.resize(
                working,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )

        for quality in (95, 90, 85, 80, 75, 70, 60, 50):
            success, encoded = cv2.imencode(
                ".jpg",
                working,
                [int(cv2.IMWRITE_JPEG_QUALITY), quality],
            )
            if not success:
                continue
            payload = encoded.tobytes()
            if len(payload) <= max_bytes:
                return base64.b64encode(payload).decode("utf-8")

    raise YoudaoPaperCutError(f"Image is still larger than {max_bytes} bytes after compression: {image_file}")


def call_youdao_cut_question(
    image_path: str | Path,
    *,
    image_type: str = "1",
    doc_type: str = "json",
    app_key: str = APP_KEY,
    app_secret: str = APP_SECRET,
) -> dict[str, Any]:
    image_file = Path(image_path)
    if not image_file.is_file():
        raise YoudaoPaperCutError(f"Input image does not exist: {image_file}")

    q = encode_image_for_youdao(image_file)
    data: dict[str, Any] = {
        "q": q,
        "imageType": image_type,
        "docType": doc_type,
    }
    add_auth_params(app_key, app_secret, data)

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(YOUDAO_CUT_QUESTION_URL, data=data, headers=headers, timeout=120)
    response.raise_for_status()

    try:
        result = response.json()
    except json.JSONDecodeError as error:
        raise YoudaoPaperCutError(f"Youdao response is not valid JSON: {response.text}") from error

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call Youdao cut_question API on a local image.")
    parser.add_argument("--input", required=True, help="Path to the input image.")
    parser.add_argument("--output-json", default=None, help="Optional path to save the JSON response.")
    parser.add_argument("--image-type", default="1", help="Youdao imageType parameter.")
    parser.add_argument("--doc-type", default="json", help="Youdao docType parameter.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    result = call_youdao_cut_question(
        args.input,
        image_type=args.image_type,
        doc_type=args.doc_type,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

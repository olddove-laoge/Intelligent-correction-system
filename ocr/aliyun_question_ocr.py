from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from alibabacloud_credentials.client import Client as CredentialClient
from alibabacloud_ocr_api20210707 import models as ocr_models
from alibabacloud_ocr_api20210707.client import Client as OcrClient
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_tea_util import models as util_models


DEFAULT_ENDPOINT = "ocr-api.cn-hangzhou.aliyuncs.com"
DEFAULT_ACCESS_KEY_ID = None
DEFAULT_ACCESS_KEY_SECRET = None


class QuestionOcrError(RuntimeError):
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


def recognize_edu_question_ocr(
    image_path: str | Path,
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    access_key_id: str | None = None,
    access_key_secret: str | None = None,
) -> dict[str, Any]:
    image_file = Path(image_path)
    if not image_file.is_file():
        raise QuestionOcrError(f"Input image does not exist: {image_file}")

    image_bytes = image_file.read_bytes()
    client = create_client(
        endpoint=endpoint,
        access_key_id=access_key_id,
        access_key_secret=access_key_secret,
    )
    request = ocr_models.RecognizeEduQuestionOcrRequest(body=image_bytes)
    runtime = util_models.RuntimeOptions()

    try:
        response = client.recognize_edu_question_ocr_with_options(request, runtime)
    except Exception as error:
        message = getattr(error, "message", str(error))
        recommend = None
        data = getattr(error, "data", None)
        if isinstance(data, dict):
            recommend = data.get("Recommend")

        detail = f"Aliyun RecognizeEduQuestionOcr failed: {message}"
        lowered_message = message.lower()
        if "credential" in lowered_message or "accesskey" in lowered_message or "providers in the chain" in lowered_message:
            detail = (
                f"{detail}\n"
                "No usable Aliyun credentials were found.\n"
                "Pass --access-key-id and --access-key-secret on the command line,\n"
                "or set ALIBABA_CLOUD_ACCESS_KEY_ID and ALIBABA_CLOUD_ACCESS_KEY_SECRET in the environment."
            )
        if recommend:
            detail = f"{detail}\nRecommend: {recommend}"
        raise QuestionOcrError(detail) from error

    if hasattr(response, "to_map"):
        return _normalize_result_payload(response.to_map())
    return _normalize_result_payload(json.loads(json.dumps(response, default=str)))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call Aliyun RecognizeEduQuestionOcr on a local cropped question image.")
    parser.add_argument("--input", required=True, help="Path to the input question image.")
    parser.add_argument("--output-json", default=None, help="Optional path to save the raw JSON response.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Aliyun OCR endpoint.")
    parser.add_argument("--access-key-id", default=None, help="Aliyun access key id.")
    parser.add_argument("--access-key-secret", default=None, help="Aliyun access key secret.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    result = recognize_edu_question_ocr(
        args.input,
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

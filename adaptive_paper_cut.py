from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from aliyun_paper_cut import draw_detected_regions, recognize_edu_paper_cut
from seeddream_qieti import (
    download_image,
    generate_marked_image,
    image_file_to_data_url,
)

from mimo_question_grading import (
    MIMO_API_KEY,
    MIMO_BASE_URL,
    MIMO_MODEL,
    create_mimo_client,
)


def classify_paper_style_with_mimo(
    image_path: str | Path,
    *,
    model: str = MIMO_MODEL,
    api_key: str | None = None,
) -> dict[str, str]:
    client = create_mimo_client(api_key=api_key)
    image_data_url = image_file_to_data_url(image_path)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个试卷版式分类助手。"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                    {
                        "type": "text",
                        "text": (
                            "判断这张试卷或作业页更适合哪种切题方式。"
                            "如果整页主要是手写题目和手写作答，输出 handwriting。"
                            "如果整页主要是印刷题干、规则排版、印刷小题为主，输出 printed。"
                            "只返回合法 JSON，格式为 {\"style\":\"handwriting或printed\",\"reason\":\"一句话原因\"}。"
                        ),
                    },
                ],
            },
        ],
        response_format={"type": "json_object"},
    )
    content = completion.choices[0].message.content
    if not content:
        return {"style": "printed", "reason": "Mimo returned empty response."}
    if isinstance(content, str):
        return json.loads(content)
    return {"style": "printed", "reason": str(content)}


def run_seedream_flow(
    image_path: str | Path,
    *,
    output_image: str | Path,
    output_url: str | Path | None = None,
) -> dict[str, str]:
    result_url = generate_marked_image(image_path)
    if output_url is not None:
        output_url_path = Path(output_url)
        output_url_path.parent.mkdir(parents=True, exist_ok=True)
        output_url_path.write_text(result_url, encoding="utf-8")
    download_image(result_url, output_image)
    return {"mode": "seedream", "result_url": result_url}


def run_aliyun_flow(
    image_path: str | Path,
    *,
    output_json: str | Path,
    output_overlay: str | Path,
    access_key_id: str,
    access_key_secret: str,
) -> dict[str, str]:
    result = recognize_edu_paper_cut(
        image_path,
        cut_type="question",
        image_type="scan",
        access_key_id=access_key_id,
        access_key_secret=access_key_secret,
    )
    output_json_path = Path(output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Failed to read input image for overlay: {image_path}")
    overlay = draw_detected_regions(image, result)
    output_overlay_path = Path(output_overlay)
    output_overlay_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_overlay_path), overlay)
    if not success:
        raise SystemExit(f"Failed to write overlay image: {output_overlay}")
    return {"mode": "aliyun"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Use Mimo to route paper cutting to SeedDream or Aliyun automatically.")
    parser.add_argument("--input", required=True, help="Path to the input image.")
    parser.add_argument("--output-dir", required=True, help="Directory to save outputs.")
    parser.add_argument("--mimo-model", default=MIMO_MODEL, help="Mimo model for style classification.")
    parser.add_argument("--mimo-api-key", default=None, help="Optional Mimo API key override.")
    parser.add_argument("--access-key-id", required=True, help="Aliyun access key id.")
    parser.add_argument("--access-key-secret", required=True, help="Aliyun access key secret.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    image_path = Path(args.input)
    if not image_path.is_file():
        raise SystemExit(f"Input image does not exist: {image_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    decision = classify_paper_style_with_mimo(
        image_path,
        model=args.mimo_model,
        api_key=args.mimo_api_key,
    )
    style = str(decision.get("style", "printed")).strip().lower()
    if style not in {"handwriting", "printed"}:
        style = "printed"

    stem = image_path.stem
    decision_path = output_dir / f"{stem}_routing.json"
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    if style == "handwriting":
        result = run_seedream_flow(
            image_path,
            output_image=output_dir / f"{stem}_seedream_marked.png",
            output_url=output_dir / f"{stem}_seedream_url.txt",
        )
    else:
        result = run_aliyun_flow(
            image_path,
            output_json=output_dir / f"{stem}_aliyun.json",
            output_overlay=output_dir / f"{stem}_aliyun_overlay.jpg",
            access_key_id=args.access_key_id,
            access_key_secret=args.access_key_secret,
        )

    summary = {
        "input": str(image_path),
        "style": style,
        "reason": decision.get("reason", ""),
        "mode": result.get("mode", style),
    }
    summary_path = output_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

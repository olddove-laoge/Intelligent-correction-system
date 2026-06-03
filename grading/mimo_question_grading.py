from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from preprocessing.seeddream_qieti import image_file_to_data_url

load_dotenv()

MIMO_API_KEY = os.getenv("MIMO_API_KEY", "")
MIMO_BASE_URL = os.getenv("MIMO_BASE_URL", "")
MIMO_MODEL = os.getenv("MIMO_MODEL", "mimo-v2.5")


class MimoQuestionGradingError(RuntimeError):
    pass


def create_mimo_client(api_key: str | None = None) -> OpenAI:
    resolved_api_key = api_key or MIMO_API_KEY
    if not resolved_api_key:
        raise MimoQuestionGradingError("Mimo API key is missing.")
    if not MIMO_BASE_URL:
        raise MimoQuestionGradingError("Mimo base URL is missing.")
    return OpenAI(base_url=MIMO_BASE_URL, api_key=resolved_api_key)


def parse_json_payload(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError as error:
        raise MimoQuestionGradingError(f"Mimo returned invalid JSON: {content}") from error


def grade_question_with_mimo(
    image_path: str | Path,
    *,
    model: str = MIMO_MODEL,
    api_key: str | None = None,
) -> dict[str, Any]:
    client = create_mimo_client(api_key=api_key)
    image_data_url = image_file_to_data_url(image_path)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个严谨的作业题目识别与批改助手。"
                    "请只返回合法 JSON，不要输出 Markdown，不要补充说明。"
                ),
            },
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
                            "请识别这张单题图片，并输出以下字段："
                            "question_text（题目文本）、student_answer（学生答案或作答过程）、"
                            "correct_answer（正确答案）、is_correct（true/false）、"
                            "explanation（简明解析）、mistake_analysis（若错误则说明原因，若正确可写无）、"
                            "knowledge_points（知识点数组）、confidence（0到1的小数）。"
                            "如果图片中是填空、选择、计算、应用题，都按同一格式输出。"
                            "返回格式必须是："
                            '{"question_text":"","student_answer":"","correct_answer":"","is_correct":true,"explanation":"","mistake_analysis":"","knowledge_points":[],"confidence":0.9}'
                        ),
                    },
                ],
            },
        ],
        response_format={"type": "json_object"},
    )
    content = completion.choices[0].message.content
    if not content:
        raise MimoQuestionGradingError("Mimo returned empty response.")
    if isinstance(content, str):
        return parse_json_payload(content)
    if isinstance(content, list):
        text = "".join(str(item) for item in content)
        return parse_json_payload(text)
    return parse_json_payload(str(content))


def classify_paper_style_with_mimo(
    image_path: str | Path,
    *,
    model: str = MIMO_MODEL,
    api_key: str | None = None,
) -> dict[str, str]:
    """用 Mimo 判断试卷是手写还是印刷，返回 {"style": "handwriting"|"printed", "reason": "..."}"""
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Use Mimo to recognize and grade a cropped question image.")
    parser.add_argument("--input", required=True, help="Path to the cropped question image.")
    parser.add_argument("--output-json", default=None, help="Optional path to save the grading result JSON.")
    parser.add_argument("--model", default=MIMO_MODEL, help="Mimo model id.")
    parser.add_argument("--api-key", default=None, help="Optional Mimo API key override.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    image_path = Path(args.input)
    if not image_path.is_file():
        raise SystemExit(f"Input image does not exist: {image_path}")

    result = grade_question_with_mimo(
        image_path,
        model=args.model,
        api_key=args.api_key,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)

    if args.output_json:
        output_json_path = Path(args.output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(rendered, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

import httpx
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

ARK_API_KEY = os.getenv("ARK_API_KEY", "")
BASE_URL = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
MODEL = os.getenv("ARK_MODEL", "doubao-seedream-5-0-260128")
MIMO_API_KEY = os.getenv("MIMO_API_KEY", "")
MIMO_BASE_URL = os.getenv("MIMO_BASE_URL", "")
MIMO_MODEL = os.getenv("MIMO_MODEL", "mimo-v2.5")
PROMPT = (
    "你是一个智能教育剪切助手。请分析这张包含学生手写解答的试卷图片，"
    "识别出里面的所有题目。题目不一定是印刷体，也可能是手写上去的，"
    "精确定位每道题的完整边界（包含题干、选项、图表以及学生的答题区域），"
    "修改图片，在图上用红色方框出各题目的边界线。"
    "关键要求：每道题必须用一个独立的红色矩形框标注，"
    "绝对不能把多道题合并到同一个红框内。"
    "红框之间不可相互包含，也不可相互交叉。"
    "如果页面有3道题，就必须画3个独立的红框；有几道题就画几个框。"
    "每道题完整地包含在对应红框内，不要漏题。"
)


class SeedDreamError(RuntimeError):
    pass


def guess_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def image_file_to_data_url(path: str | Path) -> str:
    image_path = Path(path)
    if not image_path.is_file():
        raise SeedDreamError(f"Input image does not exist: {image_path}")
    mime_type = guess_mime_type(image_path)
    payload = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{payload}"


def create_client(api_key: str | None = None) -> OpenAI:
    resolved_api_key = api_key or ARK_API_KEY
    if not resolved_api_key:
        raise SeedDreamError("ARK API key is missing.")
    return OpenAI(base_url=BASE_URL, api_key=resolved_api_key)


def create_mimo_client(api_key: str | None = None) -> OpenAI:
    resolved_api_key = api_key or MIMO_API_KEY
    if not resolved_api_key:
        raise SeedDreamError("Mimo API key is missing.")
    return OpenAI(base_url=MIMO_BASE_URL, api_key=resolved_api_key)


def analyze_questions_with_mimo(
    image_path: str | Path,
    *,
    model: str = MIMO_MODEL,
    api_key: str | None = None,
) -> str:
    mimo_client = create_mimo_client(api_key=api_key)
    image_data_url = image_file_to_data_url(image_path)
    completion = mimo_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个试卷切题分析助手。"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "请分析这张试卷或作业图片，识别有哪些题目。"
                            "对每道题给出：题号或顺序、题目内容摘要、学生作答是否与题目写在一起、"
                            "以及题目在页面中的大致位置。"
                            "位置只需用自然语言描述，例如左上、右上、中部偏左、下半页整块等。"
                            "请输出简洁中文列表，不要输出 Markdown。"
                        ),
                    },
                ],
            },
        ],
    )
    content = completion.choices[0].message.content
    if not content:
        raise SeedDreamError("Mimo returned empty analysis.")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)


def build_seedream_prompt(base_prompt: str, analysis_text: str | None = None) -> str:
    if not analysis_text:
        return base_prompt
    return (
        f"{base_prompt}"
        "根据上述分析内容，优先参考每道题的大致位置、题目顺序和题目类型来画框；"
        "如果分析内容与视觉内容有轻微冲突，以原图中真实可见的题目与作答区域为准。\n"
        f"以下是预分析得到的题目摘要与大致位置：\n{analysis_text}"
    )


def generate_marked_image(
    image_path: str | Path,
    *,
    prompt: str = PROMPT,
    model: str = MODEL,
    size: str = "2K",
    response_format: str = "url",
    watermark: bool = False,
    api_key: str | None = None,
) -> str:
    client = create_client(api_key=api_key)
    image_data_url = image_file_to_data_url(image_path)
    response = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        response_format=response_format,
        extra_body={
            "image": image_data_url,
            "watermark": watermark,
        },
    )
    if not getattr(response, "data", None):
        raise SeedDreamError("Seedream returned no image data.")
    first = response.data[0]
    result_url = getattr(first, "url", None)
    if not result_url:
        raise SeedDreamError("Seedream did not return a result URL.")
    return result_url


def download_image(url: str, output_path: str | Path) -> None:
    response = httpx.get(url, timeout=300)
    response.raise_for_status()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(response.content)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call Seedream to draw question bounding boxes on a paper image.")
    parser.add_argument("--input", required=True, help="Path to the input image.")
    parser.add_argument("--output", default=None, help="Optional path to save the returned image.")
    parser.add_argument("--output-url", default=None, help="Optional path to save the returned image URL as text.")
    parser.add_argument("--prompt", default=PROMPT, help="Prompt sent to Seedream.")
    parser.add_argument("--model", default=MODEL, help="Seedream model id.")
    parser.add_argument("--size", default="2K", help="Requested output image size.")
    parser.add_argument("--response-format", default="url", help="Seedream response format.")
    parser.add_argument("--watermark", action="store_true", help="Enable watermark on generated image.")
    parser.add_argument("--api-key", default=None, help="Optional Ark API key override.")
    parser.add_argument("--disable-mimo-analysis", action="store_true", help="Disable Mimo pre-analysis.")
    parser.add_argument("--mimo-model", default=MIMO_MODEL, help="Mimo model id for pre-analysis.")
    parser.add_argument("--mimo-api-key", default=None, help="Optional Mimo API key override.")
    parser.add_argument("--analysis-output", default=None, help="Optional path to save Mimo analysis text.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    analysis_text = None
    if not args.disable_mimo_analysis:
        analysis_text = analyze_questions_with_mimo(
            args.input,
            model=args.mimo_model,
            api_key=args.mimo_api_key,
        )
        if args.analysis_output:
            analysis_output_path = Path(args.analysis_output)
            analysis_output_path.parent.mkdir(parents=True, exist_ok=True)
            analysis_output_path.write_text(analysis_text, encoding="utf-8")

    final_prompt = build_seedream_prompt(args.prompt, analysis_text)

    result_url = generate_marked_image(
        args.input,
        prompt=final_prompt,
        model=args.model,
        size=args.size,
        response_format=args.response_format,
        watermark=args.watermark,
        api_key=args.api_key,
    )

    print(result_url)

    if args.output_url:
        output_url_path = Path(args.output_url)
        output_url_path.parent.mkdir(parents=True, exist_ok=True)
        output_url_path.write_text(result_url, encoding="utf-8")

    if args.output:
        download_image(result_url, args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

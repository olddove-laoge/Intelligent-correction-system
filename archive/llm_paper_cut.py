from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_MODEL = os.getenv("LLM_VISION_MODEL") or os.getenv("ARK_VISION_MODEL") or "doubao-1-5-vision-pro-32k-250115"
DEFAULT_BASE_URL = os.getenv("LLM_BASE_URL") or os.getenv("ARK_BASE_URL") or "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT") or os.getenv("ARK_REQUEST_TIMEOUT") or "180")


@dataclass(slots=True)
class LlmQuestion:
    """表示大模型返回的单题结构。"""

    question_id: str
    bbox: tuple[int, int, int, int]
    question_text: str
    student_answer: str
    is_correct: bool
    correct_answer: str
    explanation: str
    knowledge_points: list[str]
    mistake_analysis: str
    confidence: float


class LlmPaperCutError(RuntimeError):
    pass


def load_api_key_from_seeddream() -> str | None:
    """从 seeddream_qieti.py 中兜底读取 Ark API key。"""

    source_path = Path(__file__).with_name("seeddream_qieti.py")
    if not source_path.is_file():
        return None
    content = source_path.read_text(encoding="utf-8")
    match = re.search(r'^ARK_API_KEY\s*=\s*(["\'])(.*?)\1', content, flags=re.M)
    if not match:
        return None
    api_key = match.group(2).strip()
    return api_key or None


def image_to_data_url(path: str | Path) -> str:
    """将图片文件转为 data URL。"""

    image_path = Path(path)
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def resize_for_analysis(src: str | Path, dst: str | Path, max_side: int = 1800) -> tuple[int, int]:
    """按最大边缩放，返回缩放后尺寸。"""

    image = Image.open(src).convert("RGB")
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    image.save(dst, quality=92)
    return image.size


def build_prompt(width: int, height: int) -> str:
    """构建切题与批改提示词。"""

    return f"""
你是一个拍照作业批改助手。请直接观察图片，不依赖人工标准答案。

任务：
1. 自动把整页作业切成一道道题。
2. 识别每道题的题目内容和学生作答。
3. 独立求解每道题并判断正误。
4. 给出正确答案、简明解析、知识点、错因。

框题要求：
- 原图尺寸是 width={width}, height={height}。
- bbox 必须使用原图像素坐标：[x1, y1, x2, y2]。
- bbox 必须完整包含题干、选项、学生答案、竖式和过程。
- 多个独立小题必须拆开，不要合并。

其他要求：
- explanation 和 mistake_analysis 每项不超过 80 个汉字。
- 只返回合法 JSON，不要输出 Markdown。

返回格式：
{{
  "questions": [
    {{
      "question_id": "1",
      "bbox": [0, 0, 100, 100],
      "question_text": "题目内容",
      "student_answer": "学生答案或过程",
      "is_correct": true,
      "correct_answer": "正确答案",
      "explanation": "简明解析",
      "knowledge_points": ["知识点1", "知识点2"],
      "mistake_analysis": "若正确可写无",
      "confidence": 0.9
    }}
  ]
}}
""".strip()


def sanitize_json_text(content: str) -> str:
    """修正常见的非法 JSON 反斜杠转义。"""

    return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', content)


def parse_json_payload(content: str) -> dict[str, Any]:
    """解析模型 JSON，兼容包裹文本。"""

    candidates = [content]
    match = re.search(r"\{.*\}", content, flags=re.S)
    if match:
        candidates.append(match.group())

    for candidate in candidates:
        for normalized in (candidate, sanitize_json_text(candidate)):
            try:
                return json.loads(normalized)
            except json.JSONDecodeError:
                continue

    raise LlmPaperCutError("模型返回内容不是合法 JSON。")


def normalize_bbox(value: Any, width: int, height: int) -> tuple[int, int, int, int]:
    """归一化并裁剪 bbox。"""

    if isinstance(value, list) and len(value) >= 4:
        nums = [int(float(x)) for x in value[:4]]
    else:
        nums = [0, 0, width, height]

    if width > 1500 and height > 1500 and max(nums) <= 1200:
        sx = width / 1024
        sy = height / 1024
        nums = [int(nums[0] * sx), int(nums[1] * sy), int(nums[2] * sx), int(nums[3] * sy)]

    x1, y1, x2, y2 = nums
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))

    return pad_bbox((x1, y1, x2, y2), width, height)


def pad_bbox(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    """对框做轻量外扩避免裁掉笔迹。"""

    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    pad_x = max(2, min(10, bw // 20))
    pad_y = max(2, min(8, bh // 16))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def refine_bbox_on_image(image: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """基于原图局部像素将 LLM 粗框收紧到真实书写区域。"""

    image_h, image_w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(image_w - 1, x1))
    y1 = max(0, min(image_h - 1, y1))
    x2 = max(x1 + 1, min(image_w, x2))
    y2 = max(y1 + 1, min(image_h, y2))

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return bbox

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    merged = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    coords = cv2.findNonZero(merged)
    if coords is None:
        return bbox

    rx, ry, rw, rh = cv2.boundingRect(coords)
    if rw < 12 or rh < 12:
        return bbox

    refined_x1 = x1 + rx
    refined_y1 = y1 + ry
    refined_x2 = refined_x1 + rw
    refined_y2 = refined_y1 + rh

    # 留极小安全边，避免切到笔迹边缘
    pad = 2
    refined_x1 = max(0, refined_x1 - pad)
    refined_y1 = max(0, refined_y1 - pad)
    refined_x2 = min(image_w, refined_x2 + pad)
    refined_y2 = min(image_h, refined_y2 + pad)

    # 防止异常收缩过度：精修框若小于原框面积 20%，回退原框
    origin_area = max(1, (x2 - x1) * (y2 - y1))
    refined_area = max(1, (refined_x2 - refined_x1) * (refined_y2 - refined_y1))
    if refined_area < origin_area * 0.2:
        return bbox

    return refined_x1, refined_y1, refined_x2, refined_y2


def refine_questions_on_image(image_path: str | Path, questions: list[LlmQuestion]) -> list[LlmQuestion]:
    """对题框执行本地像素级精修。"""

    image = cv2.imread(str(image_path))
    if image is None:
        return questions

    refined: list[LlmQuestion] = []
    for question in questions:
        new_bbox = refine_bbox_on_image(image, question.bbox)
        refined.append(
            LlmQuestion(
                question_id=question.question_id,
                bbox=new_bbox,
                question_text=question.question_text,
                student_answer=question.student_answer,
                is_correct=question.is_correct,
                correct_answer=question.correct_answer,
                explanation=question.explanation,
                knowledge_points=question.knowledge_points,
                mistake_analysis=question.mistake_analysis,
                confidence=question.confidence,
            )
        )
    return refined


def parse_questions(raw: dict[str, Any], width: int, height: int) -> list[LlmQuestion]:
    """将模型返回结构化成题目列表。"""

    raw_questions = raw.get("questions") or []
    questions: list[LlmQuestion] = []
    for idx, item in enumerate(raw_questions, start=1):
        if not isinstance(item, dict):
            continue
        question_id = str(item.get("question_id") or idx)
        questions.append(
            LlmQuestion(
                question_id=question_id,
                bbox=normalize_bbox(item.get("bbox"), width, height),
                question_text=str(item.get("question_text") or "未识别到题干"),
                student_answer=str(item.get("student_answer") or ""),
                is_correct=bool(item.get("is_correct", False)),
                correct_answer=str(item.get("correct_answer") or ""),
                explanation=str(item.get("explanation") or ""),
                knowledge_points=[str(x) for x in item.get("knowledge_points", []) if str(x).strip()],
                mistake_analysis=str(item.get("mistake_analysis") or ""),
                confidence=float(item.get("confidence") or 0.8),
            )
        )

    if questions:
        return questions

    return [
        LlmQuestion(
            question_id="1",
            bbox=(0, 0, width, height),
            question_text="整页作业",
            student_answer="",
            is_correct=False,
            correct_answer="",
            explanation="模型没有稳定返回切题结果，请重试或换更清晰图片。",
            knowledge_points=[],
            mistake_analysis="未能切题",
            confidence=0.2,
        )
    ]


def draw_overlay(image_path: str | Path, questions: list[LlmQuestion], output_path: str | Path) -> None:
    """在原图上叠加题框和对错标识。"""

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = pick_font(20)
    for question in questions:
        color = (31, 164, 99) if question.is_correct else (220, 65, 55)
        x1, y1, x2, y2 = question.bbox
        draw.rounded_rectangle((x1, y1, x2, y2), radius=8, outline=color, width=3)
        tag = f"Q{question.question_id} {'√' if question.is_correct else '×'}"
        draw.rounded_rectangle((x1 + 6, y1 + 6, x1 + 112, y1 + 40), radius=8, fill=color)
        draw.text((x1 + 12, y1 + 10), tag, fill="white", font=font)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out, quality=95)


def pick_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """选择可用中文字体。"""

    for path in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def call_llm_vision(
    image_path: str | Path,
    *,
    api_key: str,
    model: str,
    base_url: str,
    timeout: int,
    temperature: float,
    max_tokens: int,
    image_detail: str,
) -> dict[str, Any]:
    """调用 OpenAI 兼容视觉接口。"""

    width, height = Image.open(image_path).size
    prompt = build_prompt(width, height)
    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max(max_tokens, 6000),
        "messages": [
            {"role": "system", "content": "你是严谨的作业切题和批改助手，只返回合法 JSON。"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(image_path),
                            "detail": image_detail,
                        },
                    },
                ],
            },
        ],
        "response_format": {"type": "json_object"},
    }

    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    with httpx.Client(timeout=timeout) as client:
        response = client.post(endpoint, headers=headers, json=payload)
        if response.status_code == 400 and "response_format" in response.text:
            payload.pop("response_format", None)
            response = client.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()

    content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    if not content:
        raise LlmPaperCutError("模型返回为空。")
    return parse_json_payload(content)


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""

    parser = argparse.ArgumentParser(description="Use an LLM vision endpoint to cut paper questions and annotate result.")
    parser.add_argument("--input", required=True, help="Path to input image.")
    parser.add_argument("--output-json", required=True, help="Path to save normalized JSON result.")
    parser.add_argument("--output-overlay", required=True, help="Path to save annotated image.")
    parser.add_argument("--work-dir", default="uploads/llm_paper_cut", help="Directory to save analysis image.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Vision model id.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible API base url.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Request timeout seconds.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=6000, help="Max completion tokens.")
    parser.add_argument("--image-detail", default="high", help="Image detail level for vision API.")
    parser.add_argument("--api-key", default=None, help="Optional api key override.")
    return parser


def main() -> int:
    """命令行入口。"""

    parser = build_arg_parser()
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("LLM_API_KEY") or os.getenv("ARK_API_KEY") or load_api_key_from_seeddream()
    if not api_key:
        raise SystemExit("Missing API key. Set LLM_API_KEY or pass --api-key.")

    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"Input image does not exist: {input_path}")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = work_dir / f"{input_path.stem}_analysis.jpg"
    width, height = resize_for_analysis(input_path, analysis_path)

    raw = call_llm_vision(
        analysis_path,
        api_key=api_key,
        model=args.model,
        base_url=args.base_url,
        timeout=args.timeout,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        image_detail=args.image_detail,
    )

    questions = parse_questions(raw, width, height)
    questions = refine_questions_on_image(analysis_path, questions)
    payload = {
        "image": str(analysis_path),
        "question_count": len(questions),
        "questions": [
            {
                "question_id": q.question_id,
                "bbox": list(q.bbox),
                "question_text": q.question_text,
                "student_answer": q.student_answer,
                "is_correct": q.is_correct,
                "correct_answer": q.correct_answer,
                "explanation": q.explanation,
                "knowledge_points": q.knowledge_points,
                "mistake_analysis": q.mistake_analysis,
                "confidence": q.confidence,
            }
            for q in questions
        ],
    }

    output_json_path = Path(args.output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    draw_overlay(analysis_path, questions, args.output_overlay)

    print(json.dumps({"question_count": len(questions), "output_json": str(output_json_path), "output_overlay": args.output_overlay}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""批注绘制：在试卷图像上绘制分数和总分。

使用 PIL 绘制中文文字，再转回 OpenCV 格式。
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 中文字体搜索路径（macOS 优先）
_FONT_CANDIDATES = [
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]

# 状态对应颜色（BGR）
STATUS_COLORS: dict[str, tuple[int, int, int]] = {
    "correct": (34, 197, 94),      # 绿色
    "wrong": (239, 68, 68),        # 红色
    "partial": (245, 158, 11),     # 橙色
    "ocr_failed": (148, 163, 184), # 灰色
    "need_review": (148, 163, 184),
}

_font_cache: ImageFont.FreeTypeFont | None = None


def _get_font(size: int = 28) -> ImageFont.FreeTypeFont:
    """加载中文字体，带缓存。"""
    global _font_cache
    if _font_cache is not None:
        try:
            return _font_cache.font_variant(size=size)
        except Exception:
            pass
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                _font_cache = ImageFont.truetype(path, size)
                return _font_cache
            except Exception:
                continue
    return ImageFont.load_default()


def compute_position(bbox: dict[str, int], image_width: int) -> tuple[int, int]:
    """计算批注绘制位置：题目右侧 +20px，上边缘 +10px。"""
    x = bbox["x"] + bbox["width"] + 20
    y = bbox["y"] + 10
    if x + 100 > image_width:
        x = max(bbox["x"] - 100, 10)
    return x, y


def draw_score(
    image: np.ndarray,
    bbox: dict[str, int],
    score: int,
    max_score: int,
    status: str,
) -> np.ndarray:
    """在图像上绘制单题分数批注。

    Returns:
        绘制后的图像副本。
    """
    result = image.copy()
    color_bgr = STATUS_COLORS.get(status, (59, 130, 246))
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

    # 用 PIL 绘制中文
    pil_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(28)

    text = f"{score}/{max_score}"
    x, y = compute_position(bbox, image.shape[1])

    # 绘制文字背景
    bbox_text = draw.textbbox((x, y), text, font=font)
    pad = 4
    draw.rounded_rectangle(
        [bbox_text[0] - pad, bbox_text[1] - pad, bbox_text[2] + pad, bbox_text[3] + pad],
        radius=6,
        fill=color_rgb,
    )
    draw.text((x, y), text, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_all_annotations(
    image: np.ndarray,
    questions: list[dict],
) -> np.ndarray:
    """在图像上绘制所有题目的批注。

    Args:
        image: BGR 图像。
        questions: 每道题的信息列表，需包含 bbox, score, max_score, status, order。

    Returns:
        绘制后的图像。
    """
    result = image.copy()
    for q in questions:
        bbox = q.get("bbox")
        if not bbox:
            continue
        result = draw_score(
            result,
            bbox=bbox,
            score=q.get("score", 0),
            max_score=q.get("max_score", 0),
            status=q.get("status", "need_review"),
        )
    return result


def image_to_base64(image: np.ndarray, max_side: int = 1600) -> str:
    """图像编码为 base64 JPEG 字符串，大图自动缩放以控制体积。"""
    h, w = image.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")

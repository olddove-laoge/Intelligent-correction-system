"""Flask 路由定义。

提供以下接口：
- GET  /api/health    健康检查
- POST /api/upload    上传作业图像
- POST /api/correct   触发批改（支持一步上传+批改 或 两步仅批改）

调用链路（correct 接口，对齐 docs/脚本说明.md 推荐主链）：
  上传图像 → 透视矫正 → Kimi 判定手写/印刷 →
    手写：SeedDream 画红框 → 红框检测裁切 → Kimi 批改
    印刷：阿里云切题 → Kimi 批改
  → 批注绘制 → 返回结果
"""

from __future__ import annotations

import traceback
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Blueprint, current_app, jsonify, request, send_from_directory

from backend.api.responses import error_response, success_response
from backend.annotation.drawer import draw_all_annotations, image_to_base64

# ── 预处理 ──────────────────────────────────────────────────────────────
from paper_perspective_correction import auto_correct_paper_perspective
from document_enhance import enhance_to_white_bg_black_text

# ── 自适应路由 ──────────────────────────────────────────────────────────
from mimo_question_grading import classify_paper_style_with_mimo

# ── 手写路线：SeedDream 画框 + 红框裁切 ────────────────────────────────
from seeddream_qieti import generate_marked_image, download_image
from redbox_crop import RedBoxRegion, detect_red_boxes, crop_regions_from_image

# ── 印刷路线：阿里云切题 ───────────────────────────────────────────────
from aliyun_paper_cut import recognize_edu_paper_cut, _iter_content_regions

# ── 批改 ────────────────────────────────────────────────────────────────
from kimi_question_grading import grade_question_with_kimi, KimiQuestionGradingError
from mimo_question_grading import grade_question_with_mimo, MimoQuestionGradingError

api_bp = Blueprint("api", __name__, url_prefix="/api")


# ── 健康检查 ────────────────────────────────────────────────────────────
@api_bp.route("/health", methods=["GET"])
def health():
    return success_response({"ok": True, "version": "0.1.0"})


# ── 图像上传 ────────────────────────────────────────────────────────────
@api_bp.route("/upload", methods=["POST"])
def upload():
    """接收上传图像，保存到 uploads/，返回 image_id。"""
    file = request.files.get("file")
    if not file:
        return error_response("UNKNOWN", "未选择文件")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png"}:
        return error_response("UNKNOWN", f"不支持的文件格式: {ext}")

    image_id = f"{uuid.uuid4().hex}{ext}"
    upload_dir = Path(current_app.config.get("UPLOAD_DIR", "uploads"))
    upload_dir.mkdir(exist_ok=True)
    save_path = upload_dir / image_id
    file.save(str(save_path))

    return success_response({
        "image_id": image_id,
        "image_url": f"/api/uploads/{image_id}",
        "message": "上传成功",
    })


# ── 批改接口 ────────────────────────────────────────────────────────────
@api_bp.route("/correct", methods=["POST"])
def correct():
    """触发批改完整流程。调试模式：直接抛出原始异常。"""
    try:
        return _correct_impl()
    except Exception:
        return jsonify({
            "success": False,
            "code": "DEBUG_ERROR",
            "message": traceback.format_exc(),
        }), 500


def _correct_impl():
    """批改实际逻辑。"""
    # ── 1. 获取图像路径 ─────────────────────────────────────────────
    upload_dir = Path(current_app.config.get("UPLOAD_DIR", "uploads"))

    if request.content_type and "multipart" in request.content_type:
        file = request.files.get("file")
        if not file:
            return error_response("UNKNOWN", "未选择文件")
        ext = Path(file.filename).suffix.lower()
        image_id = f"{uuid.uuid4().hex}{ext}"
        save_path = upload_dir / image_id
        save_path.parent.mkdir(exist_ok=True)
        file.save(str(save_path))
    else:
        data = request.get_json(silent=True) or {}
        image_id = data.get("image_id", "")
        save_path = upload_dir / image_id
        if not save_path.is_file():
            return error_response("UNKNOWN", f"图像不存在: {image_id}")

    image_path = str(save_path)

    # ── 2. 读取图像 ──────────────────────────────────────────────────
    image = cv2.imread(image_path)
    if image is None:
        return error_response("IMAGE_BLURRY", "图像无法读取，请重新上传")
    h, w = image.shape[:2]

    # ── 3. 图像增强 ──────────────────────────────────────────────────
    enhanced = enhance_to_white_bg_black_text(image)
    enhanced_path = upload_dir / f"enhanced_{image_id}"
    cv2.imwrite(str(enhanced_path), enhanced)

    # ── 4. 判定手写/印刷 ────────────────────────────────────────────
    decision = classify_paper_style_with_mimo(str(enhanced_path))
    style = str(decision.get("style", "printed")).strip().lower()

    # ── 5. 根据风格路由切题 ─────────────────────────────────────────
    questions_data: list[dict] = []
    cut_mode = style

    if style == "handwriting":
        # 手写路线：SeedDream 在增强图上画框 → 返回标记图直接展示
        return _handwriting_flow(enhanced, enhanced_path, upload_dir, image_id, h, w)

    # 印刷路线：矫正 → 阿里云切题
    corrected, _matrix, _points, _mask, _mode = auto_correct_paper_perspective(image)
    corrected_path = upload_dir / f"corrected_{image_id}"
    cv2.imwrite(str(corrected_path), corrected)
    questions_data = _cut_printed(corrected_path)

    if not questions_data:
        return error_response("CUT_FAILED", "未检测到题目区域")

    # ── 6. 逐题 Mimo 批改 ──────────────────────────────────────────
    questions_out, total_score, max_total = _grade_all_questions(
        corrected, questions_data, upload_dir, image_id,
    )

    # ── 7. 批注绘制 ────────────────────────────────────────────────
    annotated = draw_all_annotations(corrected, questions_out)
    annotated_base64 = image_to_base64(annotated)

    annotated_path = upload_dir / f"annotated_{image_id}"
    cv2.imwrite(str(annotated_path), annotated)

    # ── 8. 返回结果 ────────────────────────────────────────────────
    return success_response({
        "question_count": len(questions_out),
        "total_score": total_score,
        "max_score": max_total,
        "comment": _generate_comment(questions_out),
        "image_width": w,
        "image_height": h,
        "cut_mode": cut_mode,
        "annotated_image_base64": annotated_base64,
        "annotated_image_url": f"/api/uploads/annotated_{image_id}",
        "image_url": f"/api/uploads/corrected_{image_id}",
        "questions": questions_out,
    })


# ── 手写路线：SeedDream 画框 → 红框裁切 → 批改 → 返回标记图 ────────────
def _handwriting_flow(
    enhanced: np.ndarray,
    enhanced_path: Path,
    upload_dir: Path,
    image_id: str,
    h: int,
    w: int,
):
    """手写路线：SeedDream 在增强图上画红框 → 检测红框 → 裁切 → 批改 → 返回标记图。"""

    # SeedDream 画红框，检测到的框太少则重试（最多 3 次）
    marked_path = upload_dir / f"marked_{image_id}"
    eh, ew = enhanced.shape[:2]
    min_side = max(100, int(min(eh, ew) * 0.03))
    regions = []
    marked_image = None

    for attempt in range(3):
        result_url = generate_marked_image(str(enhanced_path))
        download_image(result_url, str(marked_path))
        marked_image = cv2.imread(str(marked_path))
        if marked_image is None:
            continue
        regions = detect_red_boxes(marked_image)
        regions = [r for r in regions if r.w >= min_side and r.h >= min_side]
        if len(regions) >= 2:
            break

    if not regions or marked_image is None:
        return error_response("CUT_FAILED", "SeedDream 未能检测到足够题目区域")

    # 标记图坐标 → 增强图坐标（按比例缩放），构造 RedBoxRegion 给 crop_regions_from_image
    mh, mw = marked_image.shape[:2]
    sx = ew / mw if mw > 0 else 1.0
    sy = eh / mh if mh > 0 else 1.0
    scaled_regions = [
        RedBoxRegion(
            index=r.index,
            x=max(0, int(r.x * sx)),
            y=max(0, int(r.y * sy)),
            w=int(r.w * sx),
            h=int(r.h * sy),
        )
        for r in regions
    ]

    # 用已有方法裁切增强图
    crop_dir = upload_dir / f"crops_{image_id}"
    print(f"[DEBUG] 缩放后 regions: {[(r.index, r.x, r.y, r.w, r.h) for r in scaled_regions]}", flush=True)
    crops = crop_regions_from_image(enhanced, scaled_regions, crop_dir)
    print(f"[DEBUG] 裁切结果: {len(crops)} 个", flush=True)

    if not crops:
        return error_response("CUT_FAILED", "未检测到题目区域")

    # 构造题目数据
    questions_data = [
        {
            "id": c["index"],
            "order": c["index"],
            "bbox": {"x": c["bbox"][0], "y": c["bbox"][1],
                     "width": c["bbox"][2] - c["bbox"][0],
                     "height": c["bbox"][3] - c["bbox"][1]},
            "crop_path": c["path"],
        }
        for c in crops
    ]

    # 批改
    questions_out, total_score, max_total = _grade_all_questions(
        enhanced, questions_data, upload_dir, image_id,
    )

    # 增强图作为展示图（bbox 坐标与增强图一致，红框已由 SeedDream 绘制）
    return success_response({
        "question_count": len(questions_out),
        "total_score": total_score,
        "max_score": max_total,
        "comment": _generate_comment(questions_out),
        "image_width": ew,
        "image_height": eh,
        "cut_mode": "handwriting",
        "annotated_image_base64": image_to_base64(enhanced),
        "annotated_image_url": f"/api/uploads/enhanced_{image_id}",
        "image_url": f"/api/uploads/enhanced_{image_id}",
        "questions": questions_out,
    })


# ── 印刷路线：阿里云切题 ───────────────────────────────────────────────
def _cut_printed(corrected_path: Path) -> list[dict]:
    """印刷试卷切题：阿里云 RecognizeEduPaperCut。"""
    cut_result = recognize_edu_paper_cut(str(corrected_path))

    questions: list[dict] = []
    for i, (label, polygon) in enumerate(_iter_content_regions(cut_result), start=1):
        if len(polygon) < 4:
            continue
        pts = np.array(polygon, dtype=np.int32)
        x, y, bw, bh = cv2.boundingRect(pts.reshape(-1, 1, 2))
        questions.append({
            "id": i,
            "order": i,
            "label": label,
            "bbox": {"x": int(x), "y": int(y), "width": int(bw), "height": int(bh)},
        })

    return questions


# ── 逐题批改（并行）────────────────────────────────────────────────────
from concurrent.futures import ThreadPoolExecutor, as_completed


def _grade_single_question(
    q: dict,
    corrected: np.ndarray,
    upload_dir: Path,
    image_id: str,
) -> dict:
    """批改单道题（供线程池调用）。"""
    bbox = q["bbox"]
    x1 = max(0, bbox["x"])
    y1 = max(0, bbox["y"])
    x2 = min(corrected.shape[1], bbox["x"] + bbox["width"])
    y2 = min(corrected.shape[0], bbox["y"] + bbox["height"])
    cropped = corrected[y1:y2, x1:x2]

    if cropped.size == 0:
        return _make_placeholder(q, "裁切区域为空")

    crop_path_str = q.get("crop_path")
    need_cleanup = False
    if not crop_path_str:
        crop_path = upload_dir / f"crop_{q['id']}_{image_id}"
        cv2.imwrite(str(crop_path), cropped)
        crop_path_str = str(crop_path)
        need_cleanup = True

    try:
        result = grade_question_with_mimo(crop_path_str)
        is_correct = result.get("is_correct", False)
        confidence = result.get("confidence", 0)

        q_max = 10
        if is_correct:
            score = q_max
            status = "correct"
        elif confidence < 0.5:
            score = 0
            status = "need_review"
        else:
            score = 0
            status = "wrong"

        return {
            "id": q["id"],
            "order": q["order"],
            "bbox": q["bbox"],
            "score": score,
            "max_score": q_max,
            "status": status,
            "ocr_text": result.get("question_text", ""),
            "student_answer": result.get("student_answer", ""),
            "feedback": result.get("explanation", "") or result.get("mistake_analysis", ""),
        }
    except (MimoQuestionGradingError, Exception) as e:
        return _make_placeholder(q, str(e))
    finally:
        if need_cleanup:
            Path(crop_path_str).unlink(missing_ok=True)


def _grade_all_questions(
    corrected: np.ndarray,
    questions_data: list[dict],
    upload_dir: Path,
    image_id: str,
) -> tuple[list[dict], int, int]:
    """并行批改所有题目，返回 (结果列表, 总分, 满分)。"""
    results: dict[int, dict] = {}

    with ThreadPoolExecutor(max_workers=min(len(questions_data), 5)) as pool:
        futures = {
            pool.submit(_grade_single_question, q, corrected, upload_dir, image_id): q["id"]
            for q in questions_data
        }
        for future in as_completed(futures):
            q_id = futures[future]
            try:
                results[q_id] = future.result()
            except Exception as e:
                q = next(q for q in questions_data if q["id"] == q_id)
                results[q_id] = _make_placeholder(q, str(e))

    questions_out = [results[q["id"]] for q in questions_data]
    total_score = sum(q["score"] for q in questions_out)
    max_total = sum(q["max_score"] for q in questions_out)

    return questions_out, total_score, max_total


# ── 辅助函数 ────────────────────────────────────────────────────────────
def _make_placeholder(q: dict, reason: str) -> dict:
    """生成单题占位结果（批改失败时使用）。"""
    return {
        "id": q["id"],
        "order": q["order"],
        "bbox": q["bbox"],
        "score": 0,
        "max_score": 10,
        "status": "need_review",
        "ocr_text": "",
        "student_answer": "",
        "feedback": f"批改失败: {reason}",
    }


def _generate_comment(questions: list[dict]) -> str:
    """根据各题结果生成简要总评。"""
    total = len(questions)
    correct_count = sum(1 for q in questions if q["status"] == "correct")
    wrong_count = sum(1 for q in questions if q["status"] == "wrong")
    review_count = sum(1 for q in questions if q["status"] == "need_review")

    parts = [f"共 {total} 题"]
    if correct_count:
        parts.append(f"正确 {correct_count} 题")
    if wrong_count:
        parts.append(f"错误 {wrong_count} 题")
    if review_count:
        parts.append(f"需复核 {review_count} 题")
    return "，".join(parts)


# ── 静态文件服务 ────────────────────────────────────────────────────────
@api_bp.route("/uploads/<path:filename>")
def serve_upload(filename: str):
    upload_dir = current_app.config.get("UPLOAD_DIR", "uploads")
    return send_from_directory(upload_dir, filename)

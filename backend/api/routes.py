"""Flask 路由定义。

接口清单：
- GET  /api/health        健康检查
- POST /api/upload        上传作业图像
- POST /api/cut           切题（增强 + 风格判定 + 切题，返回题目坐标）
- POST /api/grade         批改（接收切题结果，逐题调用 Mimo 批改）

流程：
  upload → cut → grade
"""

from __future__ import annotations

import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from flask import Blueprint, current_app, jsonify, request, send_from_directory

from backend.api.responses import error_response, success_response
from backend.annotation.drawer import draw_all_annotations, image_to_base64

# ── 预处理 ──────────────────────────────────────────────────────────────
from preprocessing.paper_perspective_correction import auto_correct_paper_perspective
from preprocessing.document_enhance import enhance_to_white_bg_black_text
from preprocessing.seeddream_qieti import generate_marked_image, download_image
from preprocessing.redbox_crop import RedBoxRegion, detect_red_boxes, crop_regions_from_image

# ── 切题 ────────────────────────────────────────────────────────────────
from ocr.aliyun_paper_cut import recognize_edu_paper_cut, _iter_content_regions

# ── 批改 + 风格判定 ─────────────────────────────────────────────────────
from grading.mimo_question_grading import (
    classify_paper_style_with_mimo,
    grade_question_with_mimo,
    MimoQuestionGradingError,
)

api_bp = Blueprint("api", __name__, url_prefix="/api")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  健康检查
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@api_bp.route("/health", methods=["GET"])
def health():
    return success_response({"ok": True, "version": "0.2.0"})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  图像上传
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  切题接口 POST /api/cut
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@api_bp.route("/cut", methods=["POST"])
def cut():
    """增强 → 判定手写/印刷 → 切题，返回题目坐标列表。

    请求（二选一）：
      multipart/form-data + file
      application/json + { "image_id": "xxx.jpg" }

    成功响应：
      {
        "success": true,
        "cut_mode": "handwriting" | "printed",
        "image_url": "/api/uploads/xxx",
        "image_width": 1080,
        "image_height": 1920,
        "questions": [
          {
            "id": 1,
            "order": 1,
            "bbox": { "x": 0, "y": 0, "width": 100, "height": 200 },
            "crop_url": "/api/uploads/crop_1_xxx.jpg"
          }
        ]
      }
    """
    try:
        return _cut_impl()
    except Exception:
        return jsonify({
            "success": False,
            "code": "DEBUG_ERROR",
            "message": traceback.format_exc(),
        }), 500


def _cut_impl():
    """切题实际逻辑。"""
    upload_dir = Path(current_app.config.get("UPLOAD_DIR", "uploads"))

    # ── 1. 获取图像 ─────────────────────────────────────────────────
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

    # ── 2. 读取 + 增强 ──────────────────────────────────────────────
    image = cv2.imread(str(save_path))
    if image is None:
        return error_response("IMAGE_BLURRY", "图像无法读取，请重新上传")

    enhanced = enhance_to_white_bg_black_text(image)
    enhanced_path = upload_dir / f"enhanced_{image_id}"
    cv2.imwrite(str(enhanced_path), enhanced)

    # ── 3. 判定手写/印刷 ────────────────────────────────────────────
    decision = classify_paper_style_with_mimo(str(enhanced_path))
    style = str(decision.get("style", "printed")).strip().lower()

    # ── 4. 切题 ────────────────────────────────────────────────────
    if style == "handwriting":
        return _cut_handwriting(enhanced, enhanced_path, upload_dir, image_id)
    else:
        return _cut_printed(enhanced, enhanced_path, upload_dir, image_id, image)


# ── 手写切题 ────────────────────────────────────────────────────────────
def _cut_handwriting(enhanced, enhanced_path, upload_dir, image_id):
    """SeedDream 画红框 → 检测红框 → 裁切增强图。"""
    eh, ew = enhanced.shape[:2]
    min_side = max(100, int(min(eh, ew) * 0.03))
    marked_path = upload_dir / f"marked_{image_id}"
    regions = []
    marked_image = None

    for _ in range(3):
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

    mh, mw = marked_image.shape[:2]
    sx, sy = ew / mw, eh / mh
    scaled_regions = [
        RedBoxRegion(index=r.index,
                     x=max(0, int(r.x * sx)), y=max(0, int(r.y * sy)),
                     w=int(r.w * sx), h=int(r.h * sy))
        for r in regions
    ]

    crop_dir = upload_dir / f"crops_{image_id}"
    crops = crop_regions_from_image(enhanced, scaled_regions, crop_dir)
    if not crops:
        return error_response("CUT_FAILED", "未检测到题目区域")

    questions = [
        {
            "id": c["index"], "order": c["index"],
            "bbox": {"x": c["bbox"][0], "y": c["bbox"][1],
                     "width": c["bbox"][2] - c["bbox"][0],
                     "height": c["bbox"][3] - c["bbox"][1]},
            "crop_url": f"/api/uploads/crops_{image_id}/q_{c['index']:02d}.png",
        }
        for c in crops
    ]

    return success_response({
        "cut_mode": "handwriting",
        "image_url": f"/api/uploads/enhanced_{image_id}",
        "image_width": ew, "image_height": eh,
        "questions": questions,
    })


# ── 印刷切题 ────────────────────────────────────────────────────────────
def _cut_printed(enhanced, enhanced_path, upload_dir, image_id, original):
    """透视矫正 → 阿里云切题 → 裁切增强图。"""
    corrected, *_ = auto_correct_paper_perspective(original)
    corrected_path = upload_dir / f"corrected_{image_id}"
    cv2.imwrite(str(corrected_path), corrected)

    cut_result = recognize_edu_paper_cut(str(corrected_path))
    questions_raw = []
    for i, (label, polygon) in enumerate(_iter_content_regions(cut_result), start=1):
        if len(polygon) < 4:
            continue
        pts = np.array(polygon, dtype=np.int32)
        x, y, bw, bh = cv2.boundingRect(pts.reshape(-1, 1, 2))
        questions_raw.append({"id": i, "order": i, "label": label,
                              "bbox": {"x": int(x), "y": int(y),
                                       "width": int(bw), "height": int(bh)}})

    if not questions_raw:
        return error_response("CUT_FAILED", "未检测到题目区域")

    # 裁切增强图并保存
    eh, ew = enhanced.shape[:2]
    crop_dir = upload_dir / f"crops_{image_id}"
    crop_dir.mkdir(parents=True, exist_ok=True)
    questions = []
    for q in questions_raw:
        b = q["bbox"]
        x1, y1 = max(0, b["x"]), max(0, b["y"])
        x2, y2 = min(ew, b["x"] + b["width"]), min(eh, b["y"] + b["height"])
        crop = enhanced[y1:y2, x1:x2]
        crop_path = crop_dir / f"q_{q['id']:02d}.png"
        cv2.imwrite(str(crop_path), crop)
        q["crop_url"] = f"/api/uploads/crops_{image_id}/q_{q['id']:02d}.png"
        questions.append(q)

    return success_response({
        "cut_mode": "printed",
        "image_url": f"/api/uploads/enhanced_{image_id}",
        "image_width": ew, "image_height": eh,
        "questions": questions,
    })


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  批改接口 POST /api/grade
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@api_bp.route("/grade", methods=["POST"])
def grade():
    """接收切题结果，逐题调用 Mimo 批改。

    请求体（JSON）：
      {
        "questions": [
          {
            "id": 1,
            "order": 1,
            "bbox": { "x": 0, "y": 0, "width": 100, "height": 200 },
            "crop_url": "/api/uploads/crops_xxx/q_01.png"
          }
        ]
      }

    成功响应：
      {
        "success": true,
        "question_count": 3,
        "total_score": 20,
        "max_score": 30,
        "comment": "共 3 题，正确 2 题，错误 1 题",
        "questions": [
          {
            "id": 1, "order": 1,
            "bbox": { ... },
            "score": 10, "max_score": 10,
            "status": "correct",
            "ocr_text": "...",
            "student_answer": "...",
            "feedback": "..."
          }
        ]
      }
    """
    try:
        return _grade_impl()
    except Exception:
        return jsonify({
            "success": False,
            "code": "DEBUG_ERROR",
            "message": traceback.format_exc(),
        }), 500


def _grade_impl():
    """批改实际逻辑。"""
    data = request.get_json(silent=True) or {}
    questions_input = data.get("questions", [])

    if not questions_input:
        return error_response("UNKNOWN", "请提供 questions 列表")

    upload_dir = Path(current_app.config.get("UPLOAD_DIR", "uploads"))

    def _grade_one(q):
        crop_url = q.get("crop_url", "")
        # crop_url 格式: /api/uploads/crops_xxx/q_01.png → uploads/crops_xxx/q_01.png
        if crop_url.startswith("/api/uploads/"):
            local_path = upload_dir / crop_url[len("/api/uploads/"):]
        else:
            local_path = Path(crop_url)

        if not local_path.is_file():
            return _make_placeholder(q, f"裁切图不存在: {crop_url}")

        try:
            result = grade_question_with_mimo(str(local_path))
            is_correct = result.get("is_correct", False)
            confidence = result.get("confidence", 0)
            q_max = 10

            if is_correct:
                score, status = q_max, "correct"
            elif confidence < 0.5:
                score, status = 0, "need_review"
            else:
                score, status = 0, "wrong"

            return {
                "id": q["id"], "order": q["order"], "bbox": q["bbox"],
                "score": score, "max_score": q_max, "status": status,
                "ocr_text": result.get("question_text", ""),
                "student_answer": result.get("student_answer", ""),
                "feedback": result.get("explanation", "") or result.get("mistake_analysis", ""),
            }
        except (MimoQuestionGradingError, Exception) as e:
            return _make_placeholder(q, str(e))

    # 并行批改
    results_map: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=min(len(questions_input), 5)) as pool:
        futures = {pool.submit(_grade_one, q): q["id"] for q in questions_input}
        for future in as_completed(futures):
            q_id = futures[future]
            try:
                results_map[q_id] = future.result()
            except Exception as e:
                q = next(q for q in questions_input if q["id"] == q_id)
                results_map[q_id] = _make_placeholder(q, str(e))

    questions_out = [results_map[q["id"]] for q in questions_input]
    total_score = sum(q["score"] for q in questions_out)
    max_total = sum(q["max_score"] for q in questions_out)

    return success_response({
        "question_count": len(questions_out),
        "total_score": total_score,
        "max_score": max_total,
        "comment": _generate_comment(questions_out),
        "questions": questions_out,
    })


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  辅助函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _make_placeholder(q: dict, reason: str) -> dict:
    return {
        "id": q["id"], "order": q["order"], "bbox": q.get("bbox", {}),
        "score": 0, "max_score": 10, "status": "need_review",
        "ocr_text": "", "student_answer": "",
        "feedback": f"批改失败: {reason}",
    }


def _generate_comment(questions: list[dict]) -> str:
    total = len(questions)
    correct = sum(1 for q in questions if q["status"] == "correct")
    wrong = sum(1 for q in questions if q["status"] == "wrong")
    review = sum(1 for q in questions if q["status"] == "need_review")
    parts = [f"共 {total} 题"]
    if correct:
        parts.append(f"正确 {correct} 题")
    if wrong:
        parts.append(f"错误 {wrong} 题")
    if review:
        parts.append(f"需复核 {review} 题")
    return "，".join(parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  静态文件服务
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@api_bp.route("/uploads/<path:filename>")
def serve_upload(filename: str):
    upload_dir = current_app.config.get("UPLOAD_DIR", "uploads")
    return send_from_directory(upload_dir, filename)

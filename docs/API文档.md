# API 接口文档

> 基础地址：`http://127.0.0.1:5001`

---

## 1. 健康检查

```
GET /api/health
```

**响应：**
```json
{
  "success": true,
  "ok": true,
  "version": "0.2.0"
}
```

---

## 2. 上传图像

```
POST /api/upload
Content-Type: multipart/form-data
```

**请求参数：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File | 是 | 图像文件（jpg/jpeg/png） |

**响应：**
```json
{
  "success": true,
  "image_id": "a1b2c3d4.jpg",
  "image_url": "/api/uploads/a1b2c3d4.jpg",
  "message": "上传成功"
}
```

---

## 3. 切题

```
POST /api/cut
```

增强图像 → 判定手写/印刷 → 调用对应切题方式，返回每道题的坐标和裁切图 URL。

**请求方式（二选一）：**

**方式 A — multipart 上传：**
```
Content-Type: multipart/form-data
字段: file（图像文件）
```

**方式 B — 已上传图像 ID：**
```
Content-Type: application/json
{
  "image_id": "a1b2c3d4.jpg"
}
```

**成功响应：**
```json
{
  "success": true,
  "cut_mode": "handwriting",
  "image_url": "/api/uploads/enhanced_a1b2c3d4.jpg",
  "image_width": 1080,
  "image_height": 1920,
  "questions": [
    {
      "id": 1,
      "order": 1,
      "bbox": {
        "x": 50,
        "y": 100,
        "width": 900,
        "height": 400
      },
      "crop_url": "/api/uploads/crops_a1b2c3d4/q_01.png"
    },
    {
      "id": 2,
      "order": 2,
      "bbox": {
        "x": 50,
        "y": 550,
        "width": 900,
        "height": 350
      },
      "crop_url": "/api/uploads/crops_a1b2c3d4/q_02.png"
    }
  ]
}
```

**字段说明：**

| 字段 | 说明 |
|------|------|
| cut_mode | 切题方式：`handwriting`（手写/SeedDream）或 `printed`（印刷/阿里云） |
| image_url | 增强后的展示图（bbox 坐标基于此图） |
| questions | 检测到的题目列表 |
| bbox | 题目在增强图上的矩形区域 `{x, y, width, height}` |
| crop_url | 该题的裁切图，可直接 `<img src="...">` 展示 |

**错误响应：**
```json
{
  "success": false,
  "code": "CUT_FAILED",
  "message": "未检测到题目区域"
}
```

---

## 4. 批改

```
POST /api/grade
Content-Type: application/json
```

接收切题结果，逐题调用 Mimo 大模型进行识别和批改。

**请求体：**
```json
{
  "questions": [
    {
      "id": 1,
      "order": 1,
      "bbox": { "x": 50, "y": 100, "width": 900, "height": 400 },
      "crop_url": "/api/uploads/crops_xxx/q_01.png"
    },
    {
      "id": 2,
      "order": 2,
      "bbox": { "x": 50, "y": 550, "width": 900, "height": 350 },
      "crop_url": "/api/uploads/crops_xxx/q_02.png"
    }
  ]
}
```

**字段说明：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| questions | Array | 是 | 从 `/api/cut` 返回的题目列表（原样传入即可） |
| questions[].id | Number | 是 | 题目 ID |
| questions[].order | Number | 是 | 题号 |
| questions[].bbox | Object | 是 | 题目坐标 |
| questions[].crop_url | String | 是 | 裁切图路径（cut 接口返回的 crop_url） |

**成功响应：**
```json
{
  "success": true,
  "question_count": 2,
  "total_score": 15,
  "max_score": 20,
  "comment": "共 2 题，正确 1 题，错误 1 题",
  "questions": [
    {
      "id": 1,
      "order": 1,
      "bbox": { "x": 50, "y": 100, "width": 900, "height": 400 },
      "score": 10,
      "max_score": 10,
      "status": "correct",
      "ocr_text": "有一块实心铝块，用托盘天平称得其质量为0.54kg...",
      "student_answer": "不能。G = mg = 0.54kg × 10N/kg = 5.4N > 5N",
      "feedback": "回答正确，重力超出量程不能测量。"
    },
    {
      "id": 2,
      "order": 2,
      "bbox": { "x": 50, "y": 550, "width": 900, "height": 350 },
      "score": 5,
      "max_score": 10,
      "status": "wrong",
      "ocr_text": "某石块体积为1×10⁻³m³，密度为2.8×10³kg/m³...",
      "student_answer": "G = mg = 2.8N",
      "feedback": "质量计算正确，但重力加速度未代入，应为 G=28N。"
    }
  ]
}
```

**字段说明：**

| 字段 | 说明 |
|------|------|
| total_score | 总得分 |
| max_score | 满分 |
| comment | 自动生成的总评 |
| questions[].score | 该题得分 |
| questions[].max_score | 该题满分（固定 10） |
| questions[].status | 批改状态：`correct`（正确）、`wrong`（错误）、`need_review`（需复核） |
| questions[].ocr_text | OCR 识别的题干文本 |
| questions[].student_answer | 识别的学生作答 |
| questions[].feedback | 大模型给出的批改反馈 |

**错误响应：**
```json
{
  "success": false,
  "code": "UNKNOWN",
  "message": "请提供 questions 列表"
}
```

---

## 5. 静态文件访问

```
GET /api/uploads/<filename>
```

用于访问切题图、批注图等上传/生成的图像文件。

示例：
```
GET /api/uploads/enhanced_a1b2c3d4.jpg
GET /api/uploads/crops_a1b2c3d4/q_01.png
```

---

## 前端调用流程

```
1. 用户选择图片
     ↓
2. POST /api/upload → 获取 image_id
     ↓
3. POST /api/cut { image_id } → 获取 questions（含 bbox + crop_url）
     ↓
   前端用 canvas 绘制 bbox 框，展示裁切图
     ↓
4. POST /api/grade { questions } → 获取批改结果
     ↓
   前端展示分数、反馈、总评
```

---

## 错误码

| 错误码 | 说明 |
|--------|------|
| IMAGE_BLURRY | 图像无法读取 |
| CUT_FAILED | 切题失败（未检测到题目） |
| UNKNOWN | 未知错误 |
| DEBUG_ERROR | 调试模式，message 中包含完整 traceback |

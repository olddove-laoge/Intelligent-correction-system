/**
 * Flask API 封装
 *
 * 接口：
 * GET  /api/health        → { ok, version }
 * POST /api/upload        multipart: file → { success, image_id, image_url }
 * POST /api/cut           JSON: { image_id } 或 multipart: file → { cut_mode, questions[] }
 * POST /api/grade         JSON: { questions[] } → { total_score, max_score, comment, questions[] }
 */
const ApiClient = (() => {
  const { API_BASE, REQUEST_TIMEOUT, MOCK_MODE } = AppConfig;

  class ApiError extends Error {
    constructor(message, code, status) {
      super(message);
      this.name = 'ApiError';
      this.code = code;
      this.status = status;
    }
  }

  const ErrorCode = {
    IMAGE_BLURRY: 'IMAGE_BLURRY',
    CUT_FAILED: 'CUT_FAILED',
    NETWORK: 'NETWORK',
    UNKNOWN: 'UNKNOWN',
  };

  async function fetchWithTimeout(url, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);
    try {
      return await fetch(url, { ...options, signal: controller.signal });
    } catch (err) {
      if (err.name === 'AbortError') throw new ApiError('请求超时', ErrorCode.NETWORK, 0);
      throw new ApiError('无法连接服务器', ErrorCode.NETWORK, 0);
    } finally {
      clearTimeout(timer);
    }
  }

  async function parseJson(res) {
    let data;
    try { data = await res.json(); }
    catch { throw new ApiError('服务器返回格式错误', ErrorCode.UNKNOWN, res.status); }
    if (!res.ok) {
      throw new ApiError(data.message || '操作失败', data.code || ErrorCode.UNKNOWN, res.status);
    }
    return data;
  }

  // ── 健康检查 ─────────────────────────────────────────────────────
  async function healthCheck() {
    if (MOCK_MODE) return { ok: true, mock: true };
    return parseJson(await fetchWithTimeout(`${API_BASE}/api/health`));
  }

  // ── 上传 ─────────────────────────────────────────────────────────
  async function uploadImage(file) {
    if (MOCK_MODE) return { success: true, image_id: 'mock-' + Date.now() };
    const form = new FormData();
    form.append('file', file);
    const data = await parseJson(await fetchWithTimeout(`${API_BASE}/api/upload`, { method: 'POST', body: form }));
    if (!data.success) throw new ApiError(data.message || '上传失败', data.code);
    return data;
  }

  // ── 切题 ─────────────────────────────────────────────────────────
  async function cutImage(payload) {
    if (MOCK_MODE) return mockCut();
    let res;
    if (payload instanceof FormData) {
      res = await fetchWithTimeout(`${API_BASE}/api/cut`, { method: 'POST', body: payload });
    } else {
      res = await fetchWithTimeout(`${API_BASE}/api/cut`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
    }
    const data = await parseJson(res);
    if (!data.success) throw new ApiError(data.message || '切题失败', data.code);
    return data;
  }

  // ── 批改 ─────────────────────────────────────────────────────────
  async function gradeQuestions(questions, extra = {}) {
    if (MOCK_MODE) return mockGrade();
    const res = await fetchWithTimeout(`${API_BASE}/api/grade`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ questions, ...extra }),
    });
    const data = await parseJson(res);
    if (!data.success) throw new ApiError(data.message || '批改失败', data.code);
    return data;
  }

  // ── 历史记录 ─────────────────────────────────────────────────────
  async function getHistory() {
    if (MOCK_MODE) return { records: [] };
    return parseJson(await fetchWithTimeout(`${API_BASE}/api/history`));
  }

  async function getHistoryDetail(recordId) {
    if (MOCK_MODE) return {};
    return parseJson(await fetchWithTimeout(`${API_BASE}/api/history/${recordId}`));
  }

  async function deleteHistory(recordId) {
    if (MOCK_MODE) return { success: true };
    return parseJson(await fetchWithTimeout(`${API_BASE}/api/history/${recordId}`, { method: 'DELETE' }));
  }

  // ── 收藏 ─────────────────────────────────────────────────────────
  async function getFavoriteIds() {
    if (MOCK_MODE) return { ids: [] };
    return parseJson(await fetchWithTimeout(`${API_BASE}/api/favorite_ids`));
  }

  async function addFavorite(recordId) {
    if (MOCK_MODE) return { starred: true };
    return parseJson(await fetchWithTimeout(`${API_BASE}/api/favorite/${recordId}`, { method: 'POST' }));
  }

  async function removeFavorite(recordId) {
    if (MOCK_MODE) return { starred: false };
    return parseJson(await fetchWithTimeout(`${API_BASE}/api/favorite/${recordId}`, { method: 'DELETE' }));
  }

  // ── Mock ─────────────────────────────────────────────────────────
  function mockCut() {
    return Promise.resolve({
      success: true, cut_mode: 'handwriting',
      image_url: '', image_width: 800, image_height: 1100,
      questions: [
        { id: 1, order: 1, bbox: { x: 40, y: 80, width: 720, height: 180 }, crop_url: '' },
        { id: 2, order: 2, bbox: { x: 40, y: 300, width: 720, height: 200 }, crop_url: '' },
      ],
    });
  }

  function mockGrade() {
    return Promise.resolve({
      success: true, question_count: 2, total_score: 18, max_score: 20,
      comment: '共 2 题，正确 2 题',
      questions: [
        { id: 1, order: 1, bbox: { x: 40, y: 80, width: 720, height: 180 }, score: 10, max_score: 10, status: 'correct', ocr_text: '1. 计算：25 + 37 = ?', student_answer: '62', feedback: '回答正确' },
        { id: 2, order: 2, bbox: { x: 40, y: 300, width: 720, height: 200 }, score: 8, max_score: 10, status: 'wrong', ocr_text: '2. 解方程：2x + 5 = 15', student_answer: 'x = 6', feedback: '应为 x = 5' },
      ],
    });
  }

  function resolveImageUrl(url) {
    if (!url) return null;
    if (url.startsWith('data:') || url.startsWith('http')) return url;
    const base = API_BASE.replace(/\/$/, '');
    return url.startsWith('/') ? base + url : base + '/' + url;
  }

  return { ApiError, ErrorCode, healthCheck, uploadImage, cutImage, gradeQuestions, getHistory, getHistoryDetail, deleteHistory, getFavoriteIds, addFavorite, removeFavorite, resolveImageUrl };
})();

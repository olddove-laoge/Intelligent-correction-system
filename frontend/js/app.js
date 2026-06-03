/**
 * 智能作业批改 — 主应用逻辑
 * 流程：上传 → 切题 → 批改
 */
(function () {
  const state = {
    file: null,
    previewUrl: null,
    imageId: null,
    uploaded: false,
    cameraStream: null,
    activeTab: 'upload',
    cutResult: null,  // 切题结果
    currentRecordId: null,  // 当前查看的记录 ID
  };

  const $ = (sel) => document.querySelector(sel);

  const els = {
    apiStatus: $('#apiStatus'),
    uploadZone: $('#uploadZone'),
    fileInput: $('#fileInput'),
    previewArea: $('#previewArea'),
    previewImage: $('#previewImage'),
    btnClear: $('#btnClear'),
    btnUpload: $('#btnUpload'),
    btnCut: $('#btnCut'),
    btnGrade: $('#btnGrade'),
    btnCapture: $('#btnCapture'),
    cameraVideo: $('#cameraVideo'),
    cameraCanvas: $('#cameraCanvas'),
    cameraHint: $('#cameraHint'),
    stepsList: $('#stepsList'),
    emptyState: $('#emptyState'),
    resultContent: $('#resultContent'),
    resultCanvas: $('#resultCanvas'),
    questionCount: $('#questionCount'),
    totalScore: $('#totalScore'),
    scoreDetail: $('#scoreDetail'),
    overallComment: $('#overallComment'),
    annotatedBlock: $('#annotatedBlock'),
    annotatedImage: $('#annotatedImage'),
    questionList: $('#questionList'),
    loadingOverlay: $('#loadingOverlay'),
    loadingText: $('#loadingText'),
    loadingSub: $('#loadingSub'),
    toastContainer: $('#toastContainer'),
    btnHistory: $('#btnHistory'),
    btnFavorites: $('#btnFavorites'),
    historyPanel: $('#historyPanel'),
    historyList: $('#historyList'),
    btnCloseHistory: $('#btnCloseHistory'),
    btnStar: $('#btnStar'),
  };

  function init() {
    bindTabs();
    bindUpload();
    bindCamera();
    bindActions();
    checkApiHealth();
  }

  function bindTabs() {
    document.querySelectorAll('.capture-tabs .tab').forEach((tab) => {
      tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });
  }

  function switchTab(name) {
    state.activeTab = name;
    document.querySelectorAll('.capture-tabs .tab').forEach((t) => {
      const active = t.dataset.tab === name;
      t.classList.toggle('active', active);
      t.setAttribute('aria-selected', active);
    });
    $('#panel-upload').hidden = name !== 'upload';
    $('#panel-upload').classList.toggle('active', name === 'upload');
    $('#panel-camera').hidden = name !== 'camera';
    $('#panel-camera').classList.toggle('active', name === 'camera');
    if (name === 'camera') startCamera();
    else stopCamera();
  }

  function bindUpload() {
    els.uploadZone.addEventListener('click', () => els.fileInput.click());
    els.fileInput.addEventListener('change', (e) => {
      const file = e.target.files?.[0];
      if (file) setImageFile(file);
    });
    els.uploadZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      els.uploadZone.classList.add('dragover');
    });
    els.uploadZone.addEventListener('dragleave', () => els.uploadZone.classList.remove('dragover'));
    els.uploadZone.addEventListener('drop', (e) => {
      e.preventDefault();
      els.uploadZone.classList.remove('dragover');
      const file = e.dataTransfer.files?.[0];
      if (file?.type.startsWith('image/')) setImageFile(file);
      else showToast('请上传图片文件', 'warning');
    });
    els.btnClear.addEventListener('click', clearImage);
  }

  function bindCamera() {
    els.btnCapture.addEventListener('click', captureFromCamera);
  }

  function bindActions() {
    els.btnUpload.addEventListener('click', handleUpload);
    els.btnCut.addEventListener('click', handleCut);
    els.btnGrade.addEventListener('click', handleGrade);
    els.btnHistory.addEventListener('click', () => showHistory('history'));
    els.btnFavorites.addEventListener('click', () => showHistory('favorites'));
    els.btnCloseHistory.addEventListener('click', () => { els.historyPanel.hidden = true; });
    els.historyPanel.addEventListener('click', (e) => { if (e.target === els.historyPanel) els.historyPanel.hidden = true; });
    els.btnStar.addEventListener('click', () => { toggleFavoriteFromResult(); });
  }

  async function checkApiHealth() {
    const dot = els.apiStatus.querySelector('.status-dot');
    const text = els.apiStatus.querySelector('.status-text');
    try {
      const data = await ApiClient.healthCheck();
      dot.className = 'status-dot status-dot--ok';
      text.textContent = data.mock ? '演示模式' : '后端已连接';
    } catch {
      dot.className = 'status-dot status-dot--err';
      text.textContent = AppConfig.MOCK_MODE ? '演示模式' : '后端未连接';
      if (!AppConfig.MOCK_MODE) {
        showToast('无法连接 Flask 后端（' + AppConfig.API_BASE + '）', 'warning');
      }
    }
  }

  function setImageFile(file) {
    clearPreviewUrl();
    state.file = file;
    state.imageId = null;
    state.uploaded = false;
    state.cutResult = null;
    state.previewUrl = URL.createObjectURL(file);
    els.previewImage.src = state.previewUrl;
    els.previewArea.hidden = false;
    els.btnUpload.disabled = false;
    els.btnCut.disabled = true;
    els.btnGrade.disabled = true;
    resetSteps();
    hideResults();
  }

  function clearImage() {
    clearPreviewUrl();
    state.file = null;
    state.imageId = null;
    state.uploaded = false;
    state.cutResult = null;
    els.previewArea.hidden = true;
    els.previewImage.removeAttribute('src');
    els.fileInput.value = '';
    els.btnUpload.disabled = true;
    els.btnCut.disabled = true;
    els.btnGrade.disabled = true;
    resetSteps();
    hideResults();
  }

  function clearPreviewUrl() {
    if (state.previewUrl) { URL.revokeObjectURL(state.previewUrl); state.previewUrl = null; }
  }

  async function startCamera() {
    stopCamera();
    if (!navigator.mediaDevices?.getUserMedia) { els.cameraHint.textContent = '当前浏览器不支持摄像头'; return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
      state.cameraStream = stream;
      els.cameraVideo.srcObject = stream;
      els.cameraHint.textContent = '对准作业后点击「拍摄」';
    } catch { els.cameraHint.textContent = '无法访问摄像头，请检查权限'; }
  }

  function stopCamera() {
    if (state.cameraStream) { state.cameraStream.getTracks().forEach((t) => t.stop()); state.cameraStream = null; }
    els.cameraVideo.srcObject = null;
  }

  function captureFromCamera() {
    const video = els.cameraVideo;
    const canvas = els.cameraCanvas;
    if (!video.videoWidth) { showToast('摄像头未就绪', 'warning'); return; }
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      if (!blob) return;
      const file = new File([blob], `capture-${Date.now()}.jpg`, { type: 'image/jpeg' });
      switchTab('upload');
      setImageFile(file);
      showToast('拍摄成功', 'success');
    }, 'image/jpeg', 0.92);
  }

  // ── 上传 ─────────────────────────────────────────────────────────
  async function handleUpload() {
    if (!state.file) return;
    setLoading(true, '正在上传…');
    setStep('upload', 'active');
    try {
      const data = await ApiClient.uploadImage(state.file);
      state.imageId = data.image_id;
      state.uploaded = true;
      setStep('upload', 'done');
      els.btnCut.disabled = false;
      showToast('上传成功', 'success');
    } catch (err) {
      setStep('upload', 'error');
      showToast(err.message, 'error');
    } finally {
      setLoading(false);
    }
  }

  // ── 切题 ─────────────────────────────────────────────────────────
  async function handleCut() {
    if (!state.file) return;

    setLoading(true, '正在增强图像…');
    resetSteps();
    showResultsShell();
    setStep('upload', 'done');

    const stepSequence = ['preprocess', 'cut'];
    let stepIndex = 0;
    const stepTimer = setInterval(() => {
      if (stepIndex > 0) setStep(stepSequence[stepIndex - 1], 'done');
      if (stepIndex < stepSequence.length) {
        setStep(stepSequence[stepIndex], 'active');
        els.loadingText.textContent = AppConfig.STEP_MESSAGES[stepSequence[stepIndex]];
        stepIndex++;
      }
    }, 800);

    try {
      let data;
      if (state.imageId && state.uploaded) {
        data = await ApiClient.cutImage({ image_id: state.imageId });
      } else {
        const form = new FormData();
        form.append('file', state.file);
        data = await ApiClient.cutImage(form);
      }

      clearInterval(stepTimer);
      stepSequence.forEach((s) => setStep(s, 'done'));

      state.cutResult = data;
      els.btnGrade.disabled = false;

      // 展示切题结果
      renderCutResult(data);
      showToast(`切题完成：${data.questions.length} 道题（${data.cut_mode}）`, 'success');
    } catch (err) {
      clearInterval(stepTimer);
      setStep('cut', 'error');
      showError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  // ── 批改 ─────────────────────────────────────────────────────────
  async function handleGrade() {
    if (!state.cutResult || !state.cutResult.questions.length) {
      showToast('请先完成切题', 'warning');
      return;
    }

    setLoading(true, '正在批改…');
    setStep('ocr', 'active');

    try {
      const data = await ApiClient.gradeQuestions(state.cutResult.questions, {
        cut_mode: state.cutResult.cut_mode,
        image_url: state.cutResult.image_url,
      });

      setStep('ocr', 'done');
      setStep('grade', 'done');

      state.currentRecordId = data.record_id || null;
      renderGradeResult(data);
      updateStarButton();
      showToast('批改完成', 'success');
    } catch (err) {
      setStep('grade', 'error');
      showError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  // ── 渲染切题结果 ─────────────────────────────────────────────────
  function renderCutResult(data) {
    const questions = data.questions || [];
    els.questionCount.textContent = `${questions.length} 道题`;
    els.totalScore.textContent = '—';
    els.scoreDetail.textContent = '待批改';
    els.overallComment.textContent = `切题方式：${data.cut_mode === 'handwriting' ? '手写' : '印刷'}`;
    els.annotatedBlock.hidden = true;

    // 绘制切题框
    const imageSrc = ApiClient.resolveImageUrl(data.image_url) || state.previewUrl;
    if (imageSrc && questions.length) {
      ResultCanvas.drawBoxes(els.resultCanvas, imageSrc, questions).catch(() => {
        showToast('切题框绘制失败', 'warning');
      });
    }

    // 显示题目列表（暂无分数）
    els.questionList.innerHTML = '';
    questions.forEach((q) => {
      const li = document.createElement('li');
      li.className = 'question-item question-item--default';
      li.innerHTML = `
        <div class="question-item__head">
          <span class="question-item__no">第 ${q.order ?? q.id} 题</span>
          <span class="question-item__status">待批改</span>
        </div>`;
      els.questionList.appendChild(li);
    });
  }

  // ── 渲染批改结果 ─────────────────────────────────────────────────
  function renderGradeResult(data) {
    const count = data.question_count ?? data.questions?.length ?? 0;
    els.questionCount.textContent = `${count} 道题`;
    els.totalScore.textContent = data.total_score ?? '—';
    els.scoreDetail.textContent = `满分 ${data.max_score ?? '—'} 分`;
    els.overallComment.textContent = data.comment || '';

    // 在切题框上叠加分数
    const cutQuestions = state.cutResult?.questions || [];
    const gradeMap = {};
    (data.questions || []).forEach((q) => { gradeMap[q.id] = q; });

    const merged = cutQuestions.map((cq) => {
      const gq = gradeMap[cq.id];
      return gq ? { ...cq, ...gq } : cq;
    });

    const imageSrc = ApiClient.resolveImageUrl(state.cutResult?.image_url) || state.previewUrl;
    if (imageSrc && merged.length) {
      ResultCanvas.drawBoxes(els.resultCanvas, imageSrc, merged).catch(() => {});
    }

    renderQuestionList(data.questions || []);
  }

  function renderQuestionList(questions) {
    els.questionList.innerHTML = '';
    const sorted = [...questions].sort((a, b) => (a.order ?? 0) - (b.order ?? 0));
    sorted.forEach((q) => {
      const li = document.createElement('li');
      li.className = `question-item question-item--${q.status || 'default'}`;
      const statusLabel = { correct: '正确', wrong: '错误', partial: '部分正确', need_review: '需复核' }[q.status] || '待批';
      li.innerHTML = `
        <div class="question-item__head">
          <span class="question-item__no">第 ${q.order ?? q.id} 题</span>
          <span class="question-item__score">${q.score ?? '—'} / ${q.max_score ?? '—'} 分</span>
          <span class="question-item__status">${statusLabel}</span>
        </div>
        <p class="question-item__ocr"><strong>题干：</strong>${escapeHtml(q.ocr_text || '—')}</p>
        <p class="question-item__answer"><strong>作答：</strong>${escapeHtml(q.student_answer || '—')}</p>
        <p class="question-item__feedback">${escapeHtml(q.feedback || '')}</p>`;
      els.questionList.appendChild(li);
    });
  }

  // ── 错误展示 ─────────────────────────────────────────────────────
  function showError(message) {
    showResultsShell();
    els.questionCount.textContent = '错误';
    els.totalScore.textContent = '!';
    els.scoreDetail.textContent = '失败';
    els.overallComment.textContent = '';
    els.annotatedBlock.hidden = true;
    els.questionList.innerHTML = `
      <li class="question-item question-item--need_review">
        <div class="question-item__head"><span class="question-item__no">错误信息</span></div>
        <pre style="white-space:pre-wrap;word-break:break-all;font-size:12px;color:#ef4444;max-height:400px;overflow:auto;">${escapeHtml(message)}</pre>
      </li>`;
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  // ── UI 工具 ──────────────────────────────────────────────────────
  function showResultsShell() { els.emptyState.hidden = true; els.resultContent.hidden = false; }
  function hideResults() { els.emptyState.hidden = false; els.resultContent.hidden = true; }
  function setStep(name, status) {
    const el = els.stepsList.querySelector(`[data-step="${name}"]`);
    if (!el) return;
    el.classList.remove('active', 'done', 'error');
    if (status) el.classList.add(status);
  }
  function resetSteps() { els.stepsList.querySelectorAll('.step').forEach((el) => el.classList.remove('active', 'done', 'error')); }
  function setLoading(show, text) { els.loadingOverlay.hidden = !show; if (text) els.loadingText.textContent = text; }
  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast--${type}`;
    toast.textContent = message;
    els.toastContainer.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add('show'));
    setTimeout(() => { toast.classList.remove('show'); setTimeout(() => toast.remove(), 300); }, 3500);
  }

  // ── 历史记录 / 收藏 ──────────────────────────────────────────────
  async function showHistory(type = 'history') {
    els.historyPanel.hidden = false;
    const titleEl = els.historyPanel.querySelector('h2');
    titleEl.textContent = type === 'favorites' ? '收藏记录' : '历史批改记录';
    els.historyList.innerHTML = '<p style="padding:1rem;color:var(--text-muted)">加载中…</p>';

    try {
      let records = [];
      let favIds = new Set();

      if (type === 'favorites') {
        const data = await ApiClient.getHistory();
        const allRecords = data.records || [];
        const favData = await ApiClient.getFavoriteIds();
        favIds = new Set(favData.ids || []);
        records = allRecords.filter((r) => favIds.has(r.id));
      } else {
        const [histData, favData] = await Promise.all([
          ApiClient.getHistory(),
          ApiClient.getFavoriteIds(),
        ]);
        records = histData.records || [];
        favIds = new Set(favData.ids || []);
        records.sort((a, b) => (favIds.has(a.id) ? 0 : 1) - (favIds.has(b.id) ? 0 : 1));
      }

      if (!records.length) {
        const msg = type === 'favorites' ? '暂无收藏记录' : '暂无批改记录';
        els.historyList.innerHTML = `<p style="padding:1.5rem;text-align:center;color:var(--text-muted)">${msg}</p>`;
        return;
      }

      els.historyList.innerHTML = '';
      records.forEach((r) => {
        const item = document.createElement('div');
        item.className = 'history-item';
        const time = r.created_at ? new Date(r.created_at).toLocaleString('zh-CN') : r.id;
        const cutMode = r.cut_mode === 'handwriting' ? '手写' : '印刷';
        const isFav = favIds.has(r.id);
        item.innerHTML = `
          <div class="history-item__info">
            <div class="history-item__time">${isFav ? '★ ' : ''}${escapeHtml(time)} · ${cutMode}</div>
            <div class="history-item__summary">${escapeHtml(r.comment || '')}</div>
          </div>
          <span class="history-item__score">${r.total_score}/${r.max_score}</span>
          <button class="history-item__delete" title="删除" data-id="${r.id}">×</button>`;

        item.querySelector('.history-item__info').addEventListener('click', () => loadHistoryDetail(r.id));
        item.querySelector('.history-item__delete').addEventListener('click', (e) => {
          e.stopPropagation();
          deleteHistoryRecord(r.id);
        });
        els.historyList.appendChild(item);
      });
    } catch (err) {
      els.historyList.innerHTML = `<p style="padding:1.5rem;color:var(--error)">${escapeHtml(err.message)}</p>`;
    }
  }

  async function loadHistoryDetail(recordId) {
    els.historyPanel.hidden = true;
    setLoading(true, '加载历史记录…');
    try {
      const data = await ApiClient.getHistoryDetail(recordId);
      state.cutResult = { cut_mode: data.cut_mode, image_url: data.image_url, questions: data.questions || [] };
      state.currentRecordId = recordId;
      showResultsShell();
      renderCutResult(data);
      renderGradeResult(data);
      updateStarButton();
      showToast('已加载历史记录', 'success');
    } catch (err) {
      showError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  async function deleteHistoryRecord(recordId) {
    if (!confirm('确认删除这条记录？')) return;
    try {
      await ApiClient.deleteHistory(recordId);
      showToast('已删除', 'success');
      showHistory();
    } catch (err) {
      showToast(err.message, 'error');
    }
  }

  async function toggleFavorite(recordId, btn) {
    const isStarred = btn.classList.contains('history-item__star--active');
    try {
      if (isStarred) {
        await ApiClient.removeFavorite(recordId);
        btn.classList.remove('history-item__star--active');
        showToast('已取消收藏', 'success');
      } else {
        await ApiClient.addFavorite(recordId);
        btn.classList.add('history-item__star--active');
        showToast('已收藏', 'success');
      }
    } catch (err) {
      showToast(err.message, 'error');
    }
  }

  async function updateStarButton() {
    if (!state.currentRecordId) { els.btnStar.hidden = true; return; }
    els.btnStar.hidden = false;
    try {
      const data = await ApiClient.getFavoriteIds();
      const isFav = (data.ids || []).includes(state.currentRecordId);
      els.btnStar.classList.toggle('result-star--active', isFav);
    } catch { els.btnStar.hidden = true; }
  }

  async function toggleFavoriteFromResult() {
    if (!state.currentRecordId) return;
    const isFav = els.btnStar.classList.contains('result-star--active');
    try {
      if (isFav) {
        await ApiClient.removeFavorite(state.currentRecordId);
        els.btnStar.classList.remove('result-star--active');
        showToast('已取消收藏', 'success');
      } else {
        await ApiClient.addFavorite(state.currentRecordId);
        els.btnStar.classList.add('result-star--active');
        showToast('已收藏', 'success');
      }
    } catch (err) {
      showToast(err.message, 'error');
    }
  }

  document.addEventListener('DOMContentLoaded', init);
})();

/**
 * 智能作业批改 — 主应用逻辑
 */
(function () {
  const state = {
    file: null,
    previewUrl: null,
    imageId: null,
    uploaded: false,
    cameraStream: null,
    activeTab: 'upload',
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
    els.uploadZone.addEventListener('dragleave', () => {
      els.uploadZone.classList.remove('dragover');
    });
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
    els.btnGrade.addEventListener('click', handleGrade);
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
        showToast('无法连接 Flask 后端，请确认服务已启动（' + AppConfig.API_BASE + '）', 'warning');
      }
    }
  }

  function setImageFile(file) {
    clearPreviewUrl();
    state.file = file;
    state.imageId = null;
    state.uploaded = false;
    state.previewUrl = URL.createObjectURL(file);
    els.previewImage.src = state.previewUrl;
    els.previewArea.hidden = false;
    els.btnUpload.disabled = false;
    els.btnGrade.disabled = false;
    resetSteps();
    hideResults();
  }

  function clearImage() {
    clearPreviewUrl();
    state.file = null;
    state.imageId = null;
    state.uploaded = false;
    els.previewArea.hidden = true;
    els.previewImage.removeAttribute('src');
    els.fileInput.value = '';
    els.btnUpload.disabled = true;
    els.btnGrade.disabled = true;
    resetSteps();
    hideResults();
  }

  function clearPreviewUrl() {
    if (state.previewUrl) {
      URL.revokeObjectURL(state.previewUrl);
      state.previewUrl = null;
    }
  }

  async function startCamera() {
    stopCamera();
    if (!navigator.mediaDevices?.getUserMedia) {
      els.cameraHint.textContent = '当前浏览器不支持摄像头';
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
        audio: false,
      });
      state.cameraStream = stream;
      els.cameraVideo.srcObject = stream;
      els.cameraHint.textContent = '对准作业后点击「拍摄」';
    } catch {
      els.cameraHint.textContent = '无法访问摄像头，请检查权限';
    }
  }

  function stopCamera() {
    if (state.cameraStream) {
      state.cameraStream.getTracks().forEach((t) => t.stop());
      state.cameraStream = null;
    }
    els.cameraVideo.srcObject = null;
  }

  function captureFromCamera() {
    const video = els.cameraVideo;
    const canvas = els.cameraCanvas;
    if (!video.videoWidth) {
      showToast('摄像头未就绪', 'warning');
      return;
    }
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const file = new File([blob], `capture-${Date.now()}.jpg`, { type: 'image/jpeg' });
        switchTab('upload');
        setImageFile(file);
        showToast('拍摄成功', 'success');
      },
      'image/jpeg',
      0.92
    );
  }

  async function handleUpload() {
    if (!state.file) return;
    setLoading(true, '正在上传…');
    setStep('upload', 'active');
    try {
      const data = await ApiClient.uploadImage(state.file);
      state.imageId = data.image_id;
      state.uploaded = true;
      setStep('upload', 'done');
      showToast(data.message || '上传成功', 'success');
      els.btnGrade.disabled = false;
    } catch (err) {
      setStep('upload', 'error');
      showToast(err.message, 'error');
    } finally {
      setLoading(false);
    }
  }

  async function handleGrade() {
    if (!state.file) return;

    setLoading(true, AppConfig.STEP_MESSAGES.preprocess);
    resetSteps();
    showResultsShell();

    const stepSequence = ['preprocess', 'cut', 'ocr', 'grade'];
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
        data = await ApiClient.correctHomework({ image_id: state.imageId });
      } else {
        const form = new FormData();
        form.append('file', state.file);
        data = await ApiClient.correctHomework(form);
      }

      clearInterval(stepTimer);
      stepSequence.forEach((s) => setStep(s, 'done'));
      setStep('upload', 'done');

      await renderResults(data);
      showToast('批改完成', 'success');
    } catch (err) {
      clearInterval(stepTimer);
      setStep('grade', 'error');
      showToast('批改出错，详见下方错误信息', 'error');
      // 在结果区域显示完整错误
      els.emptyState.hidden = true;
      els.resultContent.hidden = false;
      els.questionCount.textContent = '错误';
      els.totalScore.textContent = '!';
      els.scoreDetail.textContent = '批改失败';
      els.overallComment.textContent = '';
      els.annotatedBlock.hidden = true;
      els.questionList.innerHTML = `
        <li class="question-item question-item--need_review">
          <div class="question-item__head">
            <span class="question-item__no">错误信息</span>
          </div>
          <pre style="white-space:pre-wrap;word-break:break-all;font-size:12px;color:#ef4444;max-height:400px;overflow:auto;">${escapeHtml(err.message || String(err))}</pre>
        </li>`;
    } finally {
      setLoading(false);
    }
  }

  function showResultsShell() {
    els.emptyState.hidden = true;
    els.resultContent.hidden = false;
  }

  function hideResults() {
    els.emptyState.hidden = false;
    els.resultContent.hidden = true;
  }

  async function renderResults(data) {
    const count = data.question_count ?? data.questions?.length ?? 0;
    els.questionCount.textContent = `${count} 道题`;

    const total = data.total_score ?? '—';
    const max = data.max_score ?? '—';
    els.totalScore.textContent = total;
    els.scoreDetail.textContent = `满分 ${max} 分`;
    els.overallComment.textContent = data.comment || '暂无总评';

    const imageSrc =
      ApiClient.resolveImageUrl(data.annotated_image_url) ||
      ApiClient.resolveImageUrl(data.image_url) ||
      (data.annotated_image_base64 ? `data:image/png;base64,${data.annotated_image_base64}` : null) ||
      state.previewUrl;

    if (data.annotated_image_url || data.annotated_image_base64) {
      els.annotatedBlock.hidden = false;
      els.annotatedImage.src = imageSrc;
    } else {
      els.annotatedBlock.hidden = true;
    }

    const questions = data.questions || [];
    renderQuestionList(questions);

    if (imageSrc && questions.length) {
      try {
        await ResultCanvas.drawBoxes(els.resultCanvas, imageSrc, questions);
      } catch {
        showToast('切题框绘制失败', 'warning');
      }
    }
  }

  function renderQuestionList(questions) {
    els.questionList.innerHTML = '';
    const sorted = [...questions].sort((a, b) => (a.order ?? 0) - (b.order ?? 0));

    sorted.forEach((q) => {
      const li = document.createElement('li');
      li.className = `question-item question-item--${q.status || 'default'}`;

      const statusLabel = {
        correct: '正确',
        wrong: '错误',
        partial: '部分正确',
        ocr_failed: '识别失败',
        need_review: '需复核',
      }[q.status] || '待批';

      li.innerHTML = `
        <div class="question-item__head">
          <span class="question-item__no">第 ${q.order ?? q.id} 题</span>
          <span class="question-item__score">${q.score ?? '—'} / ${q.max_score ?? '—'} 分</span>
          <span class="question-item__status">${statusLabel}</span>
        </div>
        <p class="question-item__ocr"><strong>题干：</strong>${escapeHtml(q.ocr_text || '—')}</p>
        <p class="question-item__answer"><strong>作答：</strong>${escapeHtml(q.student_answer || '—')}</p>
        <p class="question-item__feedback">${escapeHtml(q.feedback || '')}</p>
      `;
      els.questionList.appendChild(li);
    });
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  function setStep(name, status) {
    const el = els.stepsList.querySelector(`[data-step="${name}"]`);
    if (!el) return;
    el.classList.remove('active', 'done', 'error');
    if (status) el.classList.add(status);
  }

  function resetSteps() {
    els.stepsList.querySelectorAll('.step').forEach((el) => {
      el.classList.remove('active', 'done', 'error');
    });
  }

  function setLoading(show, text, sub) {
    els.loadingOverlay.hidden = !show;
    if (text) els.loadingText.textContent = text;
    els.loadingSub.textContent = sub || '';
  }

  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast--${type}`;
    toast.textContent = message;
    els.toastContainer.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add('show'));
    setTimeout(() => {
      toast.classList.remove('show');
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }

  document.addEventListener('DOMContentLoaded', init);
})();

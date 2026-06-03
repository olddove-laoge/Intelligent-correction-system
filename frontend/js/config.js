/**
 * 前端配置 — 对接 Flask 时修改 API_BASE
 */
const AppConfig = {
  /** 后端地址，留空则自动使用当前页面地址（支持手机/局域网访问） */
  API_BASE: window.location.origin,

  /** 为 true 时后端不可用时使用本地模拟数据（便于纯前端调试） */
  MOCK_MODE: false,

  /** 请求超时（毫秒），批改流程含多次大模型调用，需要较长时间 */
  REQUEST_TIMEOUT: 180000,

  /** 批改流程各阶段文案 */
  STEP_MESSAGES: {
    upload: '正在上传图像…',
    preprocess: '正在矫正与增强图像…',
    cut: '正在切分题目…',
    ocr: '正在识别题目内容…',
    grade: '正在调用大模型批改…',
  },
};

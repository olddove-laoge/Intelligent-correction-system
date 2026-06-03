"""Flask 应用入口。

启动方式: python app.py
"""

from __future__ import annotations

from flask import Flask, send_from_directory

from backend.api.routes import api_bp
from config import DEBUG, HISTORY_DIR, SECRET_KEY, UPLOAD_DIR


def create_app() -> Flask:
    """创建并配置 Flask 应用。

    Returns:
        配置完成的 Flask 实例。
    """
    app = Flask(
        __name__,
        static_folder="frontend",
        static_url_path="",
    )
    app.config["SECRET_KEY"] = SECRET_KEY
    app.config["UPLOAD_DIR"] = str(UPLOAD_DIR)
    app.config["HISTORY_DIR"] = str(HISTORY_DIR)
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB

    # 注册 API 蓝图
    app.register_blueprint(api_bp)

    # 前端入口
    @app.route("/")
    def index():
        return send_from_directory("frontend", "index.html")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5001, debug=DEBUG)

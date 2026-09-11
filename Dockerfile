FROM python:3.12-slim

WORKDIR /app

# ① 【明記】メタデータとしてライセンス情報を記述する（AGPL-3.0 に変更）
LABEL maintainer="Your Name <your.email@example.com>" \
      description="Streamlit app for Fish Classification using YOLOv5" \
      license="AGPL-3.0"

# ② 【添付】自分のPC上の LICENSE ファイルをコンテナ内の /app/LICENSE へコピー
COPY LICENSE /app/LICENSE


# 依存ライブラリのインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリのコードをコピー
COPY . .

# Streamlit のポート番号設定 (Cloud Run は 8080 番を使用)
EXPOSE 8080

# コンテナ起動コマンド
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]

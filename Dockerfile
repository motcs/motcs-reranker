# ============================================================
# motcs-reranker 镜像（BGE Reranker，模型直接内置到镜像）
# ============================================================
# 构建：  docker build -t motcs-reranker .
# 运行：  docker run -d --name motcs-reranker -p 8000:8000 motcs-reranker
# 探活：  curl http://localhost:8000/health
# ============================================================
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/python:3.11

WORKDIR /app

# 仅安装运行依赖（不含 optimum/转换依赖，缩小镜像体积）
COPY requirements-run.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 应用代码
COPY app.py .

# 模型直接 COPY 进镜像（INT8 约 110MB；fp32 未转/未生成时 COPY 会失败，务必先转换）
COPY models/ ./models/

# 推理环境变量
ENV RERANKER_MODEL_DIR=/app/models/bge-reranker-base
ENV RERANKER_THREADS=4

EXPOSE 8000

# 启动服务
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

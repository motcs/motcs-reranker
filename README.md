# motcs-reranker

BGE Reranker 精排服务（ONNX + INT8，CPU 友好，FastAPI）。

在 RAG（检索增强生成）链路中，向量检索的相似度分数往往不够可靠（主题相近但答非所问的文档也会拿到高分）。
本服务对 **"问题 + 候选文档对"** 做二次相关性精排（re-ranking），返回按相关性降序的 topN 结果，用于提升检索精度、拦截无关命中。

- 模型：**BAAI/bge-reranker-base**（中文重排模型）
- 推理：ONNX Runtime **CPU** 推理，无需 GPU
- 量化：支持 INT8 动态量化（约 270MB），单条推理几十毫秒，Top-20 秒级
- 部署：Docker 镜像内置模型，运行时不联网、不下载

> **English**: see [README.en.md](./README.en.md)

---

## 特性

- 纯 CPU 推理，无 GPU 依赖
- 自动优先加载 INT8 量化模型（`model_int8.onnx`），缺失时回退 fp32（`model.onnx`）
- 兼容 2 输入 / 3 输入两种 ONNX 导出（`input_ids` / `attention_mask` / 可选 `token_type_ids`）
- FastAPI 轻量接口，HTTP/1.1 通用，任何语言可接入
- 模型文件不入版本库，按下方说明自行下载（GitHub 仓库内不含模型）

---

## 快速开始

### 1. 下载模型（必须，模型不入库）

模型文件**不随仓库上传**，首次使用前需自行下载 ONNX 版 `bge-reranker-base` 到 `models/bge-reranker-base/`。

**方式一：huggingface_hub 下载命令**

```bash
pip install -U "huggingface_hub[cli]"
hf download onnx-community/bge-reranker-base --local-dir models/bge-reranker-base
```

> 旧版 CLI 命令：`huggingface-cli download onnx-community/bge-reranker-base --local-dir models/bge-reranker-base`

**方式二：浏览器手动下载**

打开模型页 <https://huggingface.co/onnx-community/bge-reranker-base>，将以下文件下载到 `models/bge-reranker-base/`：

| 文件 | 说明 |
|------|------|
| `model.onnx` | fp32 主模型（约 1.1GB） |
| `config.json` | 模型配置 |
| `tokenizer.json` / `tokenizer_config.json` | tokenizer |
| `sentencepiece.bpe.model` | 分词模型 |
| `special_tokens_map.json` | 特殊 token |

> 国内网络加速：先执行 `export HF_ENDPOINT=https://hf-mirror.com` 再下载。

### 2.（可选）INT8 量化

量化后体积约 270MB、CPU 推理更快，服务启动时**优先加载** INT8。

```bash
python quantize_int8.py
```

产物：`models/bge-reranker-base/model_int8.onnx`

> 模型文件请勿提交到 Git 仓库，建议在 `.gitignore` 中加入 `models/`（仓库已内置该规则）。

### 3. 安装依赖并启动

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements-run.txt

uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## 接口

### POST `/v1/rerank`

精排：把 `query` 与每个候选文档组成文本对，返回按相关性降序的 `top_n` 结果。

**请求**

```json
{
  "query": "党建工作的主要职责是什么？",
  "documents": [
    { "id": "docA", "text": "人大代表职责（知识库标准文档）……" },
    { "id": "docB", "text": "党建工作职责说明……" }
  ],
  "top_n": 5
}
```

**响应**（score 为 sigmoid 归一化 0~1，越高越相关，按降序）

```json
{ "results": [ { "id": "docB", "score": 0.91 }, { "id": "docA", "score": 0.71 } ] }
```

**curl 示例**

```bash
curl -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"query":"党建工作的主要职责是什么？","documents":[{"id":"docA","text":"人大代表职责……"},{"id":"docB","text":"党建工作职责说明……"}],"top_n":5}'
```

### GET `/health`

```bash
curl http://localhost:8000/health
# {"status":"ok","model":".../model_int8.onnx"}
```

---

## 配置（环境变量）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `RERANKER_MODEL_DIR` | `models/bge-reranker-base` | 模型目录（含 ONNX + tokenizer） |
| `RERANKER_THREADS` | CPU 核数 | onnxruntime 推理线程数，调高可提速 |
| `PORT` | `8000` | 服务端口 |

---

## Docker 部署

镜像**内置模型**，运行时不联网、不下载。

```bash
# 1. 先按上文下载（/量化）模型到 models/（COPY 依赖构建上下文中有模型）
# 2. 构建镜像
docker build -t motcs-reranker .

# 3. 运行
docker run -d --name motcs-reranker -p 8000:8000 motcs-reranker

# 4. 验证
curl http://localhost:8000/health
```

> - 未生成 `models/` 前 `docker build` 会因 `COPY models/` 失败，务必先下载模型。
> - Dockerfile 的 `FROM` 默认使用华为云镜像加速源；无该网络环境时可改为 `FROM python:3.11`。
> - `.dockerignore` 排除 fp32（`model.onnx`），镜像只打包 INT8，体积更小。
> - CPU 调优：运行加 `-e RERANKER_THREADS=8`。

---

## 接入其他系统（HTTP 通用）

任何语言只需 POST JSON 即可，无 SDK 依赖。推荐接入流程：

1. 上游向量检索先召回 Top-K（如 Top-20，低阈值多召回）；
2. 调用 `/v1/rerank` 精排，取前 N 条；
3. 再用精排分数或自定义阈值（如 0.85）过滤无关命中，进入大模型上下文。

---

## 目录结构

```
motcs-reranker/
├── app.py                    # FastAPI 服务（加载/推理/接口）
├── quantize_int8.py          # INT8 动态量化脚本（需先有 model.onnx）
├── requirements-run.txt      # 运行依赖（固定已验证版本，不含转换依赖）
├── requirements.txt          # 完整依赖（含本地转换用 optimum，可选；一般用下载的 ONNX 则不需要）
├── Dockerfile                # 模型内置镜像
├── .dockerignore
├── .gitignore                # 已排除 models/ 等本地文件
├── models/bge-reranker-base/ # 模型目录（不入库，自行下载）
└── README.md / README.en.md
```

---

## 许可证

本项目采用 **MIT License**（与模型 `BAAI/bge-reranker-base` 相同的开源协议）。详见 [LICENSE](LICENSE)。

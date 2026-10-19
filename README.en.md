# motcs-reranker

BGE Re-ranker service (ONNX + INT8, CPU-friendly, FastAPI).

In RAG (retrieval-augmented generation) pipelines, vector similarity scores are often unreliable — chunks
that are topically close but do not actually answer the question can still score high. This service performs a
**second-pass relevance re-ranking over "question + candidate document" pairs** and returns the topN results
in descending relevance, improving retrieval precision and filtering irrelevant hits.

- Model: **BAAI/bge-reranker-base** (Chinese re-ranking model)
- Inference: ONNX Runtime **CPU**, no GPU required
- Quantization: INT8 dynamic quantization supported (~270MB), tens of milliseconds per pair, Top-20 in seconds
- Deployment: Docker image with the model baked in; no network access at runtime

> **中文**: see [README.md](./README.md)

---

## Features

- Pure CPU inference, no GPU dependency
- Automatically prefers the INT8 model (`model_int8.onnx`), falls back to fp32 (`model.onnx`)
- Compatible with both 2-input and 3-input ONNX exports (`input_ids` / `attention_mask` / optional `token_type_ids`)
- Lightweight FastAPI endpoints, plain HTTP/1.1, consumable from any language
- Model files are **not** committed to the repository; download them as described below

---

## Quick Start

### 1. Download the model (required — models are not in the repo)

Model files are **not** shipped with this repository. Before first use, download the ONNX version of
`bge-reranker-base` into `models/bge-reranker-base/`.

**Option A: huggingface_hub CLI**

```bash
pip install -U "huggingface_hub[cli]"
hf download onnx-community/bge-reranker-base --local-dir models/bge-reranker-base
```

> Legacy CLI: `huggingface-cli download onnx-community/bge-reranker-base --local-dir models/bge-reranker-base`

**Option B: manual download via browser**

Open <https://huggingface.co/onnx-community/bge-reranker-base> and download these files into
`models/bge-reranker-base/`:

| File | Description |
|------|-------------|
| `model.onnx` | fp32 main model (~1.1 GB) |
| `config.json` | model configuration |
| `tokenizer.json` / `tokenizer_config.json` | tokenizer |
| `sentencepiece.bpe.model` | sentencepiece model |
| `special_tokens_map.json` | special tokens |

> For mainland-China networks, run `export HF_ENDPOINT=https://hf-mirror.com` before downloading.

### 2. (Optional) INT8 quantization

The INT8 model is ~270MB and faster on CPU; the service **prefers** it on startup.

```bash
python quantize_int8.py
```

Output: `models/bge-reranker-base/model_int8.onnx`

> Do not commit model files to Git; `.gitignore` already excludes `models/`.

### 3. Install dependencies and run

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

## API

### POST `/v1/rerank`

Re-ranks `query` against each candidate document (pair-wise scoring) and returns the top `top_n` results
in descending relevance.

**Request**

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

**Response** (score is sigmoid-normalized 0~1; higher = more relevant, sorted descending)

```json
{ "results": [ { "id": "docB", "score": 0.91 }, { "id": "docA", "score": 0.71 } ] }
```

**curl example**

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

## Configuration (environment variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `RERANKER_MODEL_DIR` | `models/bge-reranker-base` | model directory (ONNX + tokenizer) |
| `RERANKER_THREADS` | CPU count | onnxruntime threads; raise to speed up |
| `PORT` | `8000` | server port |

---

## Docker Deployment

The image **bakes the model in**; no network access at runtime.

```bash
# 1. Download (/quantize) the model into models/ first (COPY needs it in the build context)
# 2. Build
docker build -t motcs-reranker .

# 3. Run
docker run -d --name motcs-reranker -p 8000:8000 motcs-reranker

# 4. Verify
curl http://localhost:8000/health
```

> - Without `models/`, `docker build` fails at `COPY models/` — download the model first.
> - The Dockerfile `FROM` defaults to a Huawei-Cloud mirror source; switch to `FROM python:3.11` if unavailable.
> - `.dockerignore` excludes fp32 (`model.onnx`), so only the INT8 model is packaged, keeping the image small.
> - CPU tuning: run with `-e RERANKER_THREADS=8`.

---

## Integrate with other systems (plain HTTP)

Any language can call the JSON endpoints without an SDK. Recommended flow:

1. Upstream vector search recalls Top-K first (e.g., Top-20 with a low threshold to maximize recall);
2. Call `/v1/rerank` to re-rank, keep the top N;
3. Filter irrelevant hits with the re-rank score / a custom threshold (e.g., 0.85), then feed the context to an LLM.

---

## Directory Layout

```
motcs-reranker/
├── app.py                    # FastAPI service (load / inference / endpoints)
├── quantize_int8.py          # INT8 dynamic quantization script (requires model.onnx)
├── requirements-run.txt      # runtime dependencies (pinned; no conversion deps)
├── requirements.txt          # full dependencies (incl. optimum for local conversion, optional)
├── Dockerfile                # model-baked image
├── .dockerignore
├── .gitignore                # excludes models/ and local files
├── models/bge-reranker-base/ # model directory (not committed; download yourself)
└── README.md / README.en.md
```

---

## License

This project is released under the **MIT License** (the same open-source license as the model `BAAI/bge-reranker-base`). See [LICENSE](LICENSE).

# -*- coding: utf-8 -*-
"""
motcs-reranker：BGE Reranker ONNX 推理服务（FastAPI）。
把"问题 + 候选文档对"精排，返回按相关性降序的 topN 文档 id + score。

启动：
    python convert_to_onnx.py          # 首次，生成 INT8 ONNX（见 README）
    uvicorn app:app --host 0.0.0.0 --port 8000

环境变量：
    RERANKER_MODEL_DIR  模型目录，默认 models/bge-reranker-base
    RERANKER_THREADS    onnxruntime 线程数，默认 = CPU 核数
接口：
    POST /v1/rerank   {"query":"...", "documents":[{"id":"..","text":".."}], "top_n":5}
    GET  /health
"""
import logging
import os
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from onnxruntime import InferenceSession, SessionOptions
from pydantic import BaseModel
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("motcs-reranker")

MODEL_DIR = os.environ.get("RERANKER_MODEL_DIR",
                           os.path.join(os.path.dirname(__file__), "models", "bge-reranker-base"))
# 优先 INT8，缺失则退回 fp32
MODEL_FILE = next(
    (os.path.join(MODEL_DIR, f) for f in ("model_int8.onnx", "model.onnx")
     if os.path.exists(os.path.join(MODEL_DIR, f))), None)

app = FastAPI(title="motcs-reranker", version="1.0")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = b""
    try:
        body = await request.body()
    except Exception:
        pass
    log.warning("Reranker 请求校验失败 body=%s errors=%s",
                body.decode("utf-8", "ignore")[:1000], exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

_tokenizer: Optional[AutoTokenizer] = None
_session: Optional[InferenceSession] = None


def load_model() -> None:
    global _tokenizer, _session
    if _session is not None:
        return
    if MODEL_FILE is None:
        raise RuntimeError(
            f"未找到 ONNX 模型: {MODEL_DIR}/model_int8.onnx（请先运行 python convert_to_onnx.py）")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    so = SessionOptions()
    so.intra_op_num_threads = max(1, int(os.environ.get("RERANKER_THREADS", os.cpu_count() or 4)))
    _session = InferenceSession(MODEL_FILE, so, providers=["CPUExecutionProvider"])
    log.info("已加载模型: %s, 输入: %s", MODEL_FILE,
             [i.name for i in _session.get_inputs()])


class Item(BaseModel):
    id: str
    text: str


class ReRankRequest(BaseModel):
    query: str
    documents: List[Item]
    top_n: Optional[int] = None


class Scored(BaseModel):
    id: str
    score: float


class ReRankResponse(BaseModel):
    results: List[Scored]


@app.post("/v1/rerank", response_model=ReRankResponse)
def rerank(req: ReRankRequest) -> ReRankResponse:
    load_model()
    if not req.documents:
        return ReRankResponse(results=[])
    top_n = (req.top_n or len(req.documents)) if req.top_n else len(req.documents)
    top_n = max(0, min(top_n, len(req.documents)))

    pairs = [(req.query, d.text) for d in req.documents]
    enc = _tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors="np")

    # 只提交 session 实际需要的输入（兼容 2/3 输入两种 ONNX 导出）
    feeds = {}
    for inp in _session.get_inputs():
        if inp.name == "input_ids":
            feeds[inp.name] = enc["input_ids"]
        elif inp.name == "attention_mask":
            feeds[inp.name] = enc["attention_mask"]
        elif inp.name == "token_type_ids":
            feeds[inp.name] = enc.get("token_type_ids", enc["input_ids"])

    logits = _session.run(None, feeds)[0]
    # 兼容两种导出：双 logit（取正类列 index 1）或单 logit（整列）
    if logits.ndim == 2 and logits.shape[1] >= 2:
        logits = logits[:, 1]
    else:
        logits = logits.reshape(-1)
    scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))  # sigmoid → 0~1，越高越相关

    order = np.argsort(-scores)
    results = [Scored(id=req.documents[i].id, score=float(scores[i])) for i in order[:top_n]]
    return ReRankResponse(results=results)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_FILE}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

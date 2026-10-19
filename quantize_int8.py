# -*- coding: utf-8 -*-
"""把现有的 model.onnx (fp32) 量化为 INT8 动态量化版 model_int8.onnx"""
import os
import time
from onnxruntime.quantization import QuantType, quantize_dynamic

base = os.path.join(os.path.dirname(__file__), "models", "bge-reranker-base")
src = os.path.join(base, "model.onnx")
dst = os.path.join(base, "model_int8.onnx")

if not os.path.exists(src):
    raise SystemExit(f"未找到 {src}")

print(f"量化前: {os.path.getsize(src)/1e6:.0f} MB")
t0 = time.time()
quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
print(f"量化完成: {os.path.getsize(dst)/1e6:.0f} MB  耗时 {time.time()-t0:.0f}s")
print(f"输出: {dst}")

# -*- coding: utf-8 -*-
import requests

body = {
    "query": "党建工作的主要职责是什么",
    "documents": [
        {"id": "c5b032e0-b9a7-4d35-a262-019533737d46",
         "text": "人大代表职责（知识库标准文档）\n\n核心定义：人大代表是国家权力机关的组成人员，代表人民的利益和意志，依法参加行使国家权力，履职分为会议期间职责、闭会期间职责，同时需履行法定义务、享有法定权利。人大代表不直接处理具体行政、司法案件，主要负责民意传达、监督建言。"},
        {"id": "ac8d52a2-f01d-4a30-aca5-12ddaccf938d",
         "text": "地情况。\n\n3. 参与活动：积极参加代表小组活动，列席人大常委会相关会议。"},
    ],
    "top_n": 5,
}

# 1) 与 Java 相同：application/json + UTF-8 编码 body
r = requests.post("http://127.0.0.1:8000/v1/rerank",
                  json=body, headers={"Content-Type": "application/json"}, timeout=10)
print("A json body      ->", r.status_code, r.text[:300])

# 2) 用字符串 body 手动编码 UTF-8（模拟 BodyPublishers.ofString(utf8)）
import json
raw = json.dumps(body, ensure_ascii=False)
r = requests.post("http://127.0.0.1:8000/v1/rerank",
                  data=raw.encode("utf-8"), headers={"Content-Type": "application/json"}, timeout=10)
print("B utf8 bytes     ->", r.status_code, r.text[:300])

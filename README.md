# 🛒 E-commerce Knowledge Graph QA System

基于 **Neo4j + NER + 向量检索 + LLM** 构建的电商知识图谱问答系统，实现从自然语言问题到图数据库查询的自动化流程。

---

## 📌 项目简介

本项目构建了一个面向电商场景的知识图谱问答系统，支持用户通过自然语言提问，系统自动：

1. 解析用户问题
2. 生成 Cypher 查询语句
3. 进行实体对齐（向量检索 + 全文检索）
4. 查询 Neo4j 图数据库
5. 基于 LLM 生成自然语言回答

👉 实现了一个完整的 **Graph + LLM + Retrieval** 的应用闭环。

---

## 🚀 核心功能

- ✅ 电商知识图谱构建（商品、品牌、类目等）
- ✅ 基于 NER 的实体识别
- ✅ 混合检索（Embedding + Fulltext）
- ✅ 自动生成 Cypher 查询
- ✅ Neo4j 图数据库查询
- ✅ LLM（DeepSeek）回答生成
- ✅ FastAPI 提供接口服务
- ✅ 简单前端页面展示问答结果

---

## 🧠 技术栈

- Python
- Neo4j
- LangChain
- HuggingFace Embeddings (bge-base-zh-v1.5)
- DeepSeek API
- FastAPI
- NER（自训练模型）
- 向量检索 + 混合检索（Hybrid Search）

---

## 🏗️ 项目结构

ec_graph/
├── src/
│ ├── configuration/ # 配置文件
│ ├── datasync/ # 数据同步模块
│ ├── ner/ # NER 模型与训练代码
│ ├── web/ # Web服务（FastAPI）
│ │ ├── app.py
│ │ ├── service.py # 核心问答逻辑
│ │ ├── schemas.py
│ │ └── static/ # 前端页面
│
├── data/ # 数据（未上传）
├── checkpoints/ # 模型（未上传）
├── logs/ # 日志
├── main.py
├── .gitignore
└── README.md


---

## 🔄 系统流程

```text
用户问题
   ↓
LLM生成 Cypher + 实体识别
   ↓
实体对齐（向量检索 + 全文检索）
   ↓
Neo4j 查询
   ↓
LLM生成答案
   ↓
返回用户
⚙️ 环境配置
1. 安装依赖
bash
pip install -r requirements.txt
2. Neo4j 配置
在 configuration/config.py 中配置：

python

NEO4J_CONFIG = {
    "uri": "bolt://localhost:7687",
    "auth": ("neo4j", "your_password")
}
3. Embedding 模型
本项目使用：

text

bge-base-zh-v1.5
需要本地下载并配置路径，例如：

python

model_name = "D:/Work/tools/models/bge-base-zh-v1.5"
4. DeepSeek API Key
python

DEEPSEEK_API_KEY = "your_api_key"
▶️ 启动项目
bash

python src/web/app.py
或者：

bash

uvicorn web.app:app --host 0.0.0.0 --port 8000
浏览器访问：

text

http://localhost:8000
💬 示例问题
某品牌有哪些商品？

某商品属于哪个类目？

某类目下有哪些商品？

📊 项目亮点
🔹 将 知识图谱 + LLM 深度结合

🔹 实现自然语言 → Cypher 自动生成

🔹 引入 混合检索（Hybrid Search）提升实体对齐准确率

🔹 构建完整的 AI 应用链路（NER → Graph → Retrieval → LLM）

🔹 支持扩展为 Agent 架构

🧩 后续优化方向
 引入 Agent 架构（任务规划 + 多步推理）

 增强 Cypher 生成准确率

 支持多轮对话

 引入缓存机制提升性能

 数据自动更新与同步
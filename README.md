
## 概述
自用的练手项目。

# 项目一：通用的RAG系统（SimpleRAGSystem.py）
- 基于Langchain和Langgraph构造的RAG（检索增强生成）系统
- 包含文档加载、文本分块、存入向量数据库、检索、生成步骤。

## 技术架构
- **大语言模型**：deepseek-v4-flash
- **Embedding模型**：text-embedding-v1
- **向量数据库**：FAISS
- **框架**：Langchain+Langgraph

## 工作流程
- 用户查询 -> 查询改写（多轮对话时） -> 向量检索 -> 上下文构建 -> 回答生成 -> 置信度评估 -> 输出结果

## 核心模块说明
### 1. DocumentProcessor（文档处理器）
- 文档加载与分块
- 向量化与 FAISS 存储
- 支持持久化保存/加载
### 2. Retriever（检索器）
- 相似度检索
- 支持返回相似度分数
### 3. Generator（生成器）
- 查询改写（多轮对话）
- 基于上下文的回答生成
- 置信度评估
### 4. RAGChain（主控制器）
- LangGraph状态图编排
- 完整RAG流程管理

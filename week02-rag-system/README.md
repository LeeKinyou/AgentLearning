# Week02: RAG实战与工程化

> 🎯 **学习目标**：掌握RAG全链路技术，引入可观测性与评估体系。

---

## 📋 目录

- [目录结构](#目录结构)
- [Day 1-2: 基础RAG流水线](#day-1-2-基础rag流水线)
- [Day 3-4: 进阶RAG策略](#day-3-4-进阶rag策略)
- [Day 5-7: 可观测性与评估](#day-5-7-可观测性与评估)
- [运行指南](#运行指南)
- [依赖安装](#依赖安装)

---

## 目录结构

```
week02-rag-system/
├── day01_02_basic_rag/           # Day 1-2: 基础RAG流水线
│   ├── __init__.py
│   └── basic_rag_pipeline.py     # 文档加载、分块、向量化、检索
├── day03_04_advanced_rag/        # Day 3-4: 进阶RAG策略
│   ├── __init__.py
│   └── advanced_rag_strategies.py # 混合检索、重排序、查询重写
├── day05_07_observability/       # Day 5-7: 可观测性与评估
│   ├── __init__.py
│   └── observability_evaluation.py # LangFuse、Ragas评估
└── tests/                        # 测试用例
    ├── __init__.py
    └── tests.py                  # 单元测试
```

---

## Day 1-2: 基础RAG流水线

### 学习内容

1. **文档加载与分块**
   - Document/Chunk数据模型
   - DocumentLoader：支持文件和目录加载
   - TextChunker：固定长度和段落分割策略

2. **向量化与向量数据库**
   - Chroma向量数据库使用
   - 文本向量化存储
   - 相似度检索

3. **答案生成**
   - LLM调用封装
   - RAG Prompt构建
   - 上下文整合

### 核心类

| 类名 | 功能 | 文件 |
|------|------|------|
| Document | 文档数据模型 | basic_rag_pipeline.py |
| Chunk | 文本块数据模型 | basic_rag_pipeline.py |
| SearchResult | 搜索结果数据模型 | basic_rag_pipeline.py |
| DocumentLoader | 文档加载器 | basic_rag_pipeline.py |
| TextChunker | 文本分块器 | basic_rag_pipeline.py |
| VectorStore | 向量数据库封装 | basic_rag_pipeline.py |
| BasicRAGSystem | 基础RAG系统 | basic_rag_pipeline.py |

### 运行示例

```bash
cd day01_02_basic_rag
python basic_rag_pipeline.py
```

---

## Day 3-4: 进阶RAG策略

### 学习内容

1. **混合检索（BM25 + 向量检索）**
   - BM25算法原理与实现
   - RRF（Reciprocal Rank Fusion）融合策略
   - 关键词检索与语义检索的优势互补

2. **重排序（Reranking）**
   - 使用LLM评估文档相关性
   - 二次排序提升结果质量

3. **查询重写与扩展**
   - 查询扩展：生成多个语义相关查询
   - 查询分解：拆分复杂查询
   - HyDE策略：假设文档生成

### 核心类

| 类名 | 功能 | 文件 |
|------|------|------|
| BM25Retriever | BM25关键词检索器 | advanced_rag_strategies.py |
| HybridRetriever | 混合检索器 | advanced_rag_strategies.py |
| Reranker | 重排序器 | advanced_rag_strategies.py |
| QueryRewriter | 查询重写器 | advanced_rag_strategies.py |
| AdvancedRAGSystem | 进阶RAG系统 | advanced_rag_strategies.py |

### 检索模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| basic | 基础向量检索 | 简单查询 |
| hybrid | 混合检索 | 通用场景 |
| rerank | 混合检索+重排序 | 高精度要求 |
| expanded | 查询扩展+混合检索 | 复杂查询 |

### 运行示例

```bash
cd day03_04_advanced_rag
python advanced_rag_strategies.py
```

---

## Day 5-7: 可观测性与评估

### 学习内容

1. **LangFuse可观测性接入**
   - Trace/Span概念理解
   - 调用链路记录
   - 关键指标埋点（延迟、Token消耗、成本）

2. **RAG系统评估**
   - Ragas评估框架
   - Faithfulness（忠实度）
   - Answer Relevance（答案相关性）
   - Context Precision（上下文精确度）

### 核心类

| 类名 | 功能 | 文件 |
|------|------|------|
| TraceSpan | 时间跨度数据模型 | observability_evaluation.py |
| Trace | 调用链路数据模型 | observability_evaluation.py |
| MetricsCollector | 指标收集器 | observability_evaluation.py |
| LangFuseObserver | LangFuse集成 | observability_evaluation.py |
| EvaluationSample | 评估样本数据模型 | observability_evaluation.py |
| RagasEvaluator | Ragas评估器 | observability_evaluation.py |
| ObservableRAGSystem | 可观测性RAG系统 | observability_evaluation.py |

### 评估指标说明

| 指标 | 说明 | 理想值 |
|------|------|--------|
| Faithfulness | 答案是否基于上下文，防止幻觉 | > 0.8 |
| Answer Relevance | 答案是否直接回答问题 | > 0.8 |
| Context Precision | 检索到的上下文是否精确相关 | > 0.7 |

### 运行示例

```bash
cd day05_07_observability
python observability_evaluation.py
```

---

## 运行指南

### 1. 环境配置

确保已配置`.env`文件（参考Week01的ENV_SETUP.md）：

```env
OPENAI_API_KEY=lm-studio
API_BASE_URL=http://localhost:1234/v1
MODEL_NAME=local-model
```

### 2. 安装依赖

```bash
pip install chromadb openai pydantic python-dotenv
```

可选依赖（用于进阶功能）：
```bash
pip install langfuse ragas rank-bm25 jieba
```

### 3. 运行测试

```bash
cd tests
python tests.py
```

---

## 学习路径建议

1. **Day 1-2**：先理解基础RAG流程
   - 运行`basic_rag_pipeline.py`观察完整流程
   - 理解Document -> Chunk -> Vector -> Search -> Answer的链路

2. **Day 3-4**：掌握检索优化策略
   - 对比不同检索模式的效果
   - 理解BM25和向量检索的互补性

3. **Day 5-7**：建立可观测性意识
   - 学会记录和分析系统指标
   - 使用评估框架量化系统性能

---

## 常见问题

### Q: Chroma数据库存储在哪里？

A: 默认使用内存存储，重启后数据丢失。如需持久化，在初始化VectorStore时指定`persist_directory`参数。

### Q: 如何选择合适的chunk_size？

A: 通常300-500字符是较好的起点。过小会丢失上下文，过大会降低检索精度。

### Q: BM25和向量检索哪个更好？

A: 没有绝对答案。BM25擅长精确关键词匹配，向量检索擅长语义理解。混合检索结合两者优势，通常效果最佳。

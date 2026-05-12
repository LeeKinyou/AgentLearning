# Agent 开发全链路学习路线图

> 🎯 **目标定位**：从"API调用侠"到"Agent系统工程师"——掌握设计、调试和运维复杂Agent系统的核心能力。

---

## 📋 目录

- [学习路线概览](#学习路线概览)
- [第一阶段：快速上手与基础夯实](#第一阶段快速上手与基础夯实第1-2周)
- [第二阶段：框架实战与多Agent协作](#第二阶段框架实战与多agent协作第3-4周)
- [第三阶段：生产级应用与综合实战](#第三阶段生产级应用与综合实战第5-6周)
- [里程碑检查清单](#里程碑检查清单)
- [学习资源推荐](#学习资源推荐)

---

## 学习路线概览

| 阶段 | 周期 | 核心主题 | 关键产出 |
|------|------|----------|----------|
| 第一阶段 | 第1-2周 | LLM基础、RAG实战、工程化预习 | 手写ReAct Agent + 可观测RAG系统 |
| 第二阶段 | 第3-4周 | LangChain/LangGraph、多Agent协作 | 复杂Agent工作流 + 多Agent协作Demo |
| 第三阶段 | 第5-6周 | API化、智能网关、综合实战 | 端到端生产级Agent应用 |

---

## 第一阶段：快速上手与基础夯实（第1-2周）

### 第一周：LLM基础与"裸写"Agent

**🎯 学习目标**：理解Agent底层原理，手写实现ReAct Agent，奠定扎实基础。

#### Day 1-2：环境与LLM初体验

- [ ] 配置Python开发环境（推荐uv/pipenv）
- [ ] 安装并测试主流LLM SDK（OpenAI、Anthropic等）
- [ ] 完成首次API调用，理解Token、Temperature等核心参数
- [ ] 实践：编写一个简单的对话脚本

#### Day 3-4：异步编程、结构化输出与Prompt工程

- [ ] 掌握Python `asyncio` 异步编程范式
- [ ] 使用Pydantic实现LLM结构化输出
- [ ] 学习Prompt工程核心技巧：
  - System Prompt设计原则
  - Few-shot Prompting
  - Chain-of-Thought (CoT)
  - ReAct Prompting
- [ ] 实践：构建一个带结构化输出的Prompt模板库

#### Day 5-7：手写ReAct Agent

- [ ] 理解ReAct范式：Reason + Act 循环
- [ ] 从零实现Agent核心组件：
  - LLM调用层
  - Tool定义与注册机制
  - Thought-Action-Observation循环
  - 终止条件判断
- [ ] 实践：手写一个能使用搜索/计算工具的ReAct Agent

**✅ 本周交付物**：一个不依赖任何Agent框架的手写ReAct Agent实现

---

### 第二周：RAG实战与工程化预习

**🎯 学习目标**：掌握RAG全链路技术，引入可观测性与评估体系。

#### Day 1-4：基础与进阶RAG

- [ ] 实现基础RAG流水线：
  - 文档加载与分块（Chunking策略）
  - 向量化与向量数据库（Chroma/Milvus等）
  - 相似度检索与答案生成
- [ ] 进阶RAG策略：
  - 混合检索（BM25 + 向量检索）
  - 重排序（Reranking）
  - 查询重写与扩展
- [ ] 实践：构建一个基于企业文档的问答系统

#### Day 5-7：可观测性与评估

- [ ] 接入LangFuse记录调用链：
  - Trace/Span概念理解
  - 关键指标埋点（延迟、Token消耗、成本）
- [ ] RAG系统评估：
  - 使用Ragas框架进行自动化评估
  - 核心指标：Faithfulness、Answer Relevance、Context Precision
- [ ] 实践：为RAG系统编写评估脚本并生成报告

**✅ 本周交付物**：一个带可观测性和评估体系的进阶RAG系统

---

## 第二阶段：框架实战与多Agent协作（第3-4周）

### 第三周：LangChain/LangGraph 框架精通

**🎯 学习目标**：掌握主流Agent框架，用可控图构建复杂Agent。

#### Day 1-3：LangChain基础

- [ ] 核心抽象学习：
  - Chain：顺序链、并行链、条件链
  - Agent：ReAct Agent、OpenAI Functions Agent
  - Tool：内置工具与自定义工具
- [ ] 记忆系统（Memory）：
  - ConversationBufferMemory
  - ConversationSummaryMemory
  - VectorStore-backed Memory
- [ ] 实践：用LangChain重写第一周的ReAct Agent

#### Day 4-7：LangGraph实战

- [ ] 核心概念：
  - State定义与类型安全
  - Node与Edge
  - 条件分支与循环
- [ ] 实战项目：
  - 用Graph构建带记忆的对话Agent
  - 实现条件分支（如：意图分类路由）
  - 加入Human-in-the-loop机制
- [ ] 实践：用LangGraph实现一个多步骤工作流Agent

**✅ 本周交付物**：基于LangGraph的复杂Agent工作流实现

---

### 第四周：多Agent协作与框架对比

**🎯 学习目标**：理解多Agent协作范式，掌握至少一种多Agent框架。

#### Day 1-3：AutoGen/CrewAI探索

- [ ] 选择并深入学习一个框架（推荐CrewAI）：
  - Agent角色定义与分工
  - Task编排与依赖管理
  - 通信机制与结果聚合
- [ ] 完成一个完整Demo：
  - 多角色协作场景（如：研究员+写手+审校）
  - 理解与LangGraph的设计差异
- [ ] 实践：对比LangGraph与CrewAI的适用场景

#### Day 4-7：多Agent协作实战

- [ ] 设计一个跨Agent协作任务：
  - 场景示例：联合撰写行业报告
  - Agent分工：信息收集 → 数据分析 → 内容撰写 → 质量审核
- [ ] 实现并调试协作流程
- [ ] 总结多Agent系统的设计模式与常见陷阱

**✅ 本周交付物**：一个完整的多Agent协作Demo及框架对比分析

---

## 第三阶段：生产级应用与综合实战（第5-6周）

### 第五周：API化、智能网关与深度调试

**🎯 学习目标**：掌握Agent服务化部署与生产级运维能力。

#### Day 1-2：FastAPI封装Agent

- [ ] 学习FastAPI核心概念：
  - 路由定义与参数校验
  - 异步端点设计
  - 流式响应（Streaming）
- [ ] 将Agent封装为RESTful API：
  - 统一请求/响应格式
  - 错误处理与状态码
  - API文档自动生成
- [ ] 实践：部署一个可远程调用的Agent服务

#### Day 3-4：LiteLLM高级配置

- [ ] 智能网关配置：
  - 多模型路由策略
  - Fallback机制（主模型失败自动切换）
  - 速率限制与配额管理
- [ ] 成本优化：
  - 模型选择策略
  - Token预算控制
- [ ] 实践：配置一个带Fallback和限流的LiteLLM网关

#### Day 5-7：鲁棒性设计与高级调试

- [ ] 常见故障模式与应对：
  - LLM幻觉检测与缓解
  - Tool调用失败重试策略
  - 超时与降级处理
- [ ] 调试技巧：
  - 利用LangFuse Trace定位问题
  - 日志分析与可视化
  - 单元测试与集成测试
- [ ] 实践：模拟故障场景并验证系统恢复能力

**✅ 本周交付物**：一个带智能网关和监控的生产级Agent服务

---

### 第六周：综合实战项目（Capstone Project）

**🎯 学习目标**：构建完整的企业级Agent系统，整合全链路技能。

#### 项目选题建议

- **智能运维助手**：查询监控、分析日志、执行预定义修复脚本
- **智能客服系统**：意图识别、知识库检索、工单生成
- **数据分析Agent**：数据查询、可视化生成、报告撰写

#### 核心任务与里程碑

##### Week 6 Day 1-2：项目规划

- [ ] 确定选题与需求分析
- [ ] 设计Agent工作流与工具集
- [ ] 技术选型与架构设计
- [ ] 输出：项目设计文档

##### Week 6 Day 3-5：核心开发

- [ ] 实现Agent核心逻辑
- [ ] 开发/集成工具集
- [ ] 通过API提供服务
- [ ] 输出：可运行的Agent系统

##### Week 6 Day 6-7：生产加固

- [ ] 接入LiteLLM网关
- [ ] 接入LangFuse监控
- [ ] 编写Ragas评估脚本
- [ ] 输出：质量保障报告

##### 项目收尾

- [ ] 编写项目README
- [ ] 录制演示视频
- [ ] 复盘整个开发过程
- [ ] 输出：完整的项目文档与演示

**🎯 最终里程碑**：完成一个端到端、可演示的生产级Agent应用，并构建一套完整的质量保障体系。

---

## 里程碑检查清单

### 阶段一检查点

- [ ] 能独立手写ReAct Agent，不依赖框架
- [ ] 理解并实现了基础RAG与进阶检索策略
- [ ] 能为系统接入可观测性并编写评估

### 阶段二检查点

- [ ] 熟练使用LangChain/LangGraph构建复杂Agent
- [ ] 理解多Agent协作范式并完成协作Demo
- [ ] 能对比不同框架的适用场景

### 阶段三检查点

- [ ] 能将Agent封装为生产级API服务
- [ ] 掌握LiteLLM网关配置与故障调试
- [ ] 完成一个完整的综合实战项目

---

## 学习资源推荐

### 官方文档

- [OpenAI API Docs](https://platform.openai.com/docs)
- [LangChain Docs](https://python.langchain.com/)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [CrewAI Docs](https://docs.crewai.com/)
- [LiteLLM Docs](https://docs.litellm.ai/)
- [LangFuse Docs](https://langfuse.com/docs)

### 推荐书籍与课程

- 《LangChain实战》
- 《Building LLM Applications for Production》
- DeepLearning.AI 相关课程

### 社区与工具

- GitHub：关注langchain-ai、microsoft/autogen等仓库
- Discord：LangChain、CrewAI官方社区
- HuggingFace：模型与数据集资源

---

## 项目结构参考

```
AgentLearning/
├── week01-llm-basics/          # 第一周：LLM基础
│   ├── async_demo/             # 异步编程示例
│   ├── prompt_engineering/     # Prompt工程
│   └── react_agent/            # 手写ReAct Agent
├── week02-rag/                 # 第二周：RAG实战
│   ├── basic_rag/              # 基础RAG
│   ├── advanced_rag/           # 进阶RAG
│   └── evaluation/             # 评估脚本
├── week03-langchain/           # 第三周：LangChain/LangGraph
│   ├── chain_demo/             # Chain示例
│   └── langgraph_agent/        # LangGraph实战
├── week04-multi-agent/         # 第四周：多Agent协作
│   └── crewai_demo/            # CrewAI/AutoGen示例
├── week05-production/          # 第五周：生产级应用
│   ├── fastapi_service/        # FastAPI封装
│   └── litellm_config/         # LiteLLM配置
└── week06-capstone/            # 第六周：综合实战
    └── project/                # Capstone Project
```

---

## 学习建议

1. **以终为始**：每个阶段开始前，先明确最终交付物是什么
2. **动手优先**：理论理解后立刻编码实践，避免"只看不练"
3. **记录问题**：建立个人知识库，记录踩坑经验与解决方案
4. **定期复盘**：每周结束时回顾学习成果，调整下周计划
5. **关注生产**：始终思考"这个功能在生产环境会怎样"

---

> 💡 **核心理念**：不追求成为"API调用侠"，而是培养"Agent系统工程师"的系统思维与工程能力。

**祝你学习顺利！🚀**

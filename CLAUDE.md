# CLAUDE.md

## 项目概述

Agent 开发全链路学习项目，涵盖 LLM 基础、RAG、LangChain/LangGraph、多 Agent 协作、生产级部署等内容。

## 环境配置（必须遵守）

**必须使用本地虚拟环境 `.venv`，禁止使用全局 Python。**

**包管理必须使用 `uv`，禁止使用 `pip`。**

```bash
# Python 解释器（运行脚本时使用）
.venv/Scripts/python          # Windows
.venv/bin/python              # Linux/macOS

# 安装依赖（统一使用 uv）
uv pip install <package>                    # 安装单个包
uv pip install -e .                         # 从 pyproject.toml 安装项目依赖
uv pip sync uv.lock                         # 根据 lock 文件精确还原环境
uv pip compile pyproject.toml -o uv.lock    # 重新生成 lock 文件

# 运行脚本
uv run python path/to/script.py             # 推荐：自动使用 .venv 环境
.venv/Scripts/python path/to/script.py      # 也可以
```

- Python 版本：3.12
- 虚拟环境路径：`.venv/`
- **不要使用裸 `pip` 或 `python` 命令**——它们可能指向全局环境
- **不要使用 `pip install`**——必须用 `uv pip install`，保证依赖解析一致
- 添加新依赖后：编辑 `pyproject.toml` → `uv pip install -e .` → 验证无冲突

## 常用命令

```bash
# 运行某个模块
uv run python -m week01-llm-basics.day01_03_llm_basics.xxx

# 安装/同步依赖
uv pip install -e .              # 安装项目及所有依赖
uv pip sync uv.lock              # 用 lock 文件还原精确环境

# 类型检查（pyright 配置在 pyproject.toml 中）
pyright
```

## 项目结构

```
week01-llm-basics/          # LLM 基础、异步编程、Prompt 工程、手写 ReAct Agent
week02-rag-system/          # 基础 RAG、高级 RAG 策略、可观测性
week03-langchain-langgraph/ # LangChain 基础、LangGraph 工作流
week04-multi-agent/         # CrewAI 多 Agent 协作
week05-production/          # 生产级 Agent 服务（API、网关、部署）
```

## 环境变量

项目使用 `.env` 文件管理密钥（已在 `.gitignore` 中排除）。参考 `.env.example` 创建。

## 注意事项

- 代码中的 `extraPaths` 配置在 `pyproject.toml` 的 `[tool.pyright]` 中，IDE 类型检查依赖此配置
- 添加新依赖时，编辑 `pyproject.toml` 后运行 `uv pip install -e .` 和 `uv lock` 同步

# Week01 环境配置指南

## 1. 安装依赖

```bash
pip install openai pydantic python-dotenv
```

## 2. 配置环境变量

### 方式一：使用 LM Studio（推荐用于学习）

LM Studio 是一个本地LLM运行工具，免费且无需API密钥。

**步骤：**
1. 下载并安装 LM Studio：https://lmstudio.ai/
2. 打开 LM Studio，下载一个模型（如 Qwen2.5-7B）
3. 点击左侧 "Local Server" 标签
4. 点击 "Start Server" 启动本地服务
5. 确认端口为 1234

**配置 `.env` 文件：**
```env
OPENAI_API_KEY=lm-studio
API_BASE_URL=http://localhost:1234/v1
MODEL_NAME=local-model
```

### 方式二：使用 Ollama

Ollama 是另一个本地LLM运行工具。

**步骤：**
1. 下载并安装 Ollama：https://ollama.com/
2. 拉取模型：`ollama pull llama2`
3. 启动服务（通常自动启动）

**配置 `.env` 文件：**
```env
OPENAI_API_KEY=ollama
API_BASE_URL=http://localhost:11434/v1
MODEL_NAME=llama2
```

### 方式三：使用 OpenAI 官方API

**步骤：**
1. 注册 OpenAI 账号：https://platform.openai.com/
2. 获取 API Key：https://platform.openai.com/api-keys
3. 充值（需要信用卡）

**配置 `.env` 文件：**
```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-3.5-turbo
```

### 方式四：使用其他兼容服务

如 OneAPI、NewAPI、DeepSeek 等兼容 OpenAI API 格式的服务。

**配置 `.env` 文件：**
```env
OPENAI_API_KEY=your-api-key
API_BASE_URL=https://your-api-server.com/v1
MODEL_NAME=gpt-3.5-turbo
```

## 3. 验证配置

运行以下命令验证配置是否正确：

```bash
cd week01-llm-basics/day01_02_llm_basics
python llm_basics.py
```

如果看到 `[LLM配置]` 输出信息，说明配置成功。

## 4. 常见问题

### Q: LM Studio 报错 "Incorrect API key provided"

A: 确保：
1. LM Studio 的 Local Server 已启动
2. `.env` 中 `API_BASE_URL` 正确指向 `http://localhost:1234/v1`
3. `OPENAI_API_KEY` 可以填写任意值（如 `lm-studio`）

### Q: 如何切换不同的LLM服务？

A: 修改 `.env` 文件中的 `API_BASE_URL` 和 `MODEL_NAME` 即可。

### Q: 本地模型响应慢怎么办？

A: 本地模型速度取决于你的硬件配置。可以尝试：
1. 使用更小的模型（如 7B 参数）
2. 使用 GPU 加速
3. 降低 `max_tokens` 参数

## 5. 环境变量说明

| 变量名 | 说明 | 示例 |
|--------|------|------|
| OPENAI_API_KEY | API密钥 | `lm-studio` 或 `sk-xxx` |
| API_BASE_URL | API基础URL | `http://localhost:1234/v1` |
| MODEL_NAME | 模型名称 | `local-model` 或 `gpt-3.5-turbo` |

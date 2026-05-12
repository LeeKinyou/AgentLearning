# Week01: LLM基础与"裸写"Agent

> 🎯 **学习目标**：理解Agent底层原理，手写实现ReAct Agent，奠定扎实基础。

---

## 📁 目录结构

```
week01-llm-basics/
├── day01_02_llm_basics/            # Day 1-2: LLM初体验
│   ├── __init__.py
│   └── llm_basics.py               # 基础对话脚本
├── day03_04_async_prompt/          # Day 3-4: 异步编程与Prompt工程
│   ├── __init__.py
│   ├── async_pydantic_demo.py      # 异步编程 + Pydantic结构化输出
│   └── prompt_templates.py         # Prompt工程模板库
├── day05_07_react_agent/           # Day 5-7: 手写ReAct Agent
│   ├── __init__.py
│   └── react_agent.py              # 完整ReAct Agent实现
└── tests/                          # 测试用例
    ├── __init__.py
    └── tests.py                    # 单元测试（13个测试，100%通过）
```

---

## 📚 学习内容与运行指南

### Day 1-2: LLM初体验

**文件**: `day01-02-llm-basics/llm_basics.py`

**学习内容**:
- 环境配置与API密钥管理
- 首次API调用
- 理解核心参数（Token、Temperature、Max Tokens）
- 简单对话脚本实现

**运行方式**:
```bash
cd day01-02-llm-basics
python llm_basics.py
```

**核心知识点**:

1. **Temperature参数**:
   - `0.0-0.3`: 输出确定、保守，适合事实性问题
   - `0.7`: 平衡创造性和一致性（推荐默认值）
   - `1.0-2.0`: 输出随机、有创造性

2. **Token概念**:
   - Token是LLM处理文本的基本单位
   - 1个Token ≈ 0.75个英文单词或1个中文字符
   - API费用基于总Token数计算

3. **关键类**:
   - `LLMChatClient`: 封装OpenAI API调用
   - `chat()`: 普通对话
   - `chat_stream()`: 流式对话（逐Token输出）

---

### Day 3-4: 异步编程与Prompt工程

#### 异步编程 + Pydantic

**文件**: `day03-04-async-prompt/async_pydantic_demo.py`

**学习内容**:
- Python `asyncio` 异步编程基础
- `async/await` 语法
- `asyncio.gather()` 并发执行
- Pydantic数据模型定义和验证
- LLM结构化输出

**运行方式**:
```bash
cd day03-04-async-prompt
python async_pydantic_demo.py
```

**核心知识点**:

1. **异步编程**:
   ```python
   async def my_function():
       await asyncio.sleep(1)  # 异步睡眠，不阻塞其他任务
   
   # 并发执行多个任务
   results = await asyncio.gather(
       task1(), task2(), task3()
   )
   ```

2. **Pydantic模型**:
   ```python
   class MovieReview(BaseModel):
       title: str = Field(description="电影名称")
       rating: float = Field(ge=1.0, le=10.0)
   ```

#### Prompt工程模板库

**文件**: `day03-04-async-prompt/prompt_templates.py`

**学习内容**:
- System Prompt设计原则
- Few-shot Prompting（少样本提示）
- Chain-of-Thought (CoT) 思维链
- ReAct Prompting
- Prompt模板管理与组合

**运行方式**:
```bash
cd day03-04-async-prompt
python prompt_templates.py
```

**核心Prompt模式**:

| 模式 | 用途 | 示例场景 |
|------|------|----------|
| System Prompt | 定义AI角色 | "你是一个代码审查专家" |
| Few-shot | 提供示例引导输出 | 情感分析、文本分类 |
| Chain-of-Thought | 激发推理能力 | 数学题、逻辑推理 |
| ReAct | 结合推理和行动 | 工具调用、多步任务 |

---

### Day 5-7: 手写ReAct Agent

**文件**: `day05-07-react-agent/react_agent.py`

**学习内容**:
- ReAct范式核心原理
- 工具定义与注册机制
- Thought-Action-Observation循环
- 终止条件判断
- 完整Agent实现

**运行方式**:
```bash
cd day05-07-react-agent
python react_agent.py
```

**ReAct工作流程**:

```
用户问题
    ↓
[Thought] LLM思考当前状态
    ↓
[Action] 选择并执行工具
    ↓
[Observation] 获取工具结果
    ↓
[循环] 重复Thought/Action/Observation
    ↓
[Final Answer] 输出最终答案
```

**核心组件**:

1. **Tool & ToolRegistry**: 工具定义和注册
2. **LLMClient**: LLM调用封装
3. **ReActAgent**: Agent核心循环
4. **AgentStep**: 执行步骤记录

**预定义工具**:
- `search`: 模拟搜索（可替换为真实API）
- `calculator`: 数学计算
- `get_date`: 日期查询
- `text_processor`: 文本处理

---

## 🧪 运行测试

```bash
cd tests
python tests.py
```

测试覆盖：
- ✅ Prompt模板功能
- ✅ 工具系统（Tool、ToolRegistry）
- ✅ 预定义工具
- ✅ Agent核心组件
- ✅ Pydantic模型验证

---

## 💡 学习要点总结

### 第一周核心技能

1. **LLM基础**:
   - 理解API调用流程
   - 掌握核心参数配置
   - 能够编写对话脚本

2. **异步编程**:
   - 理解asyncio概念
   - 掌握并发执行模式
   - 能够优化I/O密集型任务

3. **结构化输出**:
   - 使用Pydantic定义数据模型
   - 实现LLM JSON模式输出
   - 数据验证和序列化

4. **Prompt工程**:
   - 设计有效的System Prompt
   - 使用Few-shot引导输出
   - 应用CoT激发推理
   - 构建ReAct格式

5. **ReAct Agent**:
   - 理解底层工作原理
   - 掌握工具定义和调用
   - 能够从零构建Agent
   - 理解循环和状态管理

---

## 🔧 环境配置

**必需依赖**:
```bash
pip install openai pydantic python-dotenv
```

**环境变量** (`.env`文件):
```env
OPENAI_API_KEY=your_api_key_here
```

---

## 📝 实践任务

### 基础任务
- [ ] 运行所有示例代码，理解输出
- [ ] 修改Temperature参数，观察输出差异
- [ ] 添加新的Pydantic模型
- [ ] 创建自定义Prompt模板

### 进阶任务
- [ ] 替换搜索工具为真实API（如SerpAPI）
- [ ] 为Agent添加更多工具
- [ ] 实现对话历史持久化
- [ ] 添加错误重试机制

### 挑战任务
- [ ] 实现流式ReAct Agent输出
- [ ] 添加Agent执行可视化
- [ ] 实现多Agent协作原型
- [ ] 编写Agent性能评估脚本

---

## 🎯 本周交付物

✅ 一个不依赖任何Agent框架的手写ReAct Agent实现

**验收标准**:
1. 能够正确解析用户问题
2. 能够使用工具获取信息
3. 能够输出准确的最终答案
4. 代码包含详细注释
5. 通过所有测试用例

---

> 💡 **学习建议**: 在运行代码时，尝试修改参数和工具，观察Agent行为的变化。理解每一行代码的作用，而不仅仅是运行它。

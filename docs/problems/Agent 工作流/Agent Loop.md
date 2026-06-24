# Agent 循环 (Agent Loop) 知识手册

本手册专门探讨智能体核心执行回路（Agent Loop）的设计与控制策略。深入分析了 Agent 循环与普通循环的异同、各种终止条件判定、最大资源预算拦截，以及业界常用的多种典型 Loop 架构（如 Planner-Executor、Coder-Reviewer 等）的设计原则。

---

## 📋 目录
- [一、 Agent Loop 概念与边界判定 (Q1-Q7)](#一-agent-loop-概念与边界判定-q1-q7)
- [二、 资源预算与安全硬门槛设计 (Q8-Q11)](#二-资源预算与安全硬门槛设计-q8-q11)
- [三、 经典 Loop 架构与设计模式 (Q12-Q16)](#三-经典-loop-架构与设计模式-q12-q16)
- [四、 成本控制与 Loop Engineering (Q17-Q25)](#四-成本控制与-loop-engineering-q17-q25)

---

## 一、 Agent Loop 概念与边界判定 (Q1-Q7)

### Q1: 什么是 Agent loop？
**Agent Loop (智能体循环)** 是 Agent 系统的核心执行回路。
大语言模型（LLM）本身是无状态且单次交互的（即“输入 Prompt ──► 输出回答”）。为了赋予大模型自主解决复杂、多步骤任务的能力，需要将 LLM 放置在一个持续感知并影响环境的循环结构中。
在每一次 Loop 中，Agent 执行以下四个步骤（类似于军事学中的 OODA 环）：
1. **感知（Observe）**：获取当前环境的最新状态、工具执行结果或用户新的输入。
2. **分析（Orient/Reason）**：利用 LLM 结合当前状态，反思上一步的成败并对现有计划进行修正。
3. **决策（Decide）**：选择下一步要执行的最佳动作（Action）以及需要调用的工具和输入参数。
4. **行动（Act）**：运行选中的工具，获取新的环境反馈，并将状态写入下一次循环。

---

### Q2: Agent loop 和普通 while 循环有什么区别？
虽然在代码层面它们都可以用 `while` 关键字实现，但在运行机制和工程控制上存在本质差异：

| 维度 | 普通 while 循环 | Agent Loop |
| :--- | :--- | :--- |
| **跳转判定** | **确定性**：由严格的布尔表达式控制（如 `i < 10`），分支跳转完全符合预期。 | **概率性**：路由和决策基于 LLM 生成的语义，同样的输入可能会走完全不同的分支路径。 |
| **状态机制** | **结构化**：仅读写类型确定的结构化标量或对象。 | **非结构化与语义化**：状态包含聊天历史、自然语言推理过程及工具返回的复杂文本。 |
| **死循环特征** | **死循环因变量未变**：如忘记写自增 `i++`。 | **逻辑“鬼打墙”**：因 LLM 认知死角，不断用相同的错误参数调用同一个工具或陷入死胡同。 |
| **错误容忍度** | 抛出未捕获异常时通常立刻崩溃终止。 | 具备自我容错，可通过重试、反思或调用其它工具来纠偏并继续循环。 |

---

### Q3: 一个 Agent loop 的终止条件是什么？
1. **任务圆满完成 (Success)**：Agent 自主判定所有子目标已达成，或调用了特定的退出工具（如 `finish(result)`），将最终答案交付给用户。
2. **主动放弃并报错退出 (Failure)**：Agent 判定在当前环境下任务无法完成（例如权限不足、所需资源已被删除），主动调用退出机制并输出失败原因。
3. **硬性资源预算超限 (Budgets Exceeded)**：触发安全拦截器，如最大迭代步骤限制（Max Steps）、最大 Token 开销限制（Token Budget）或执行超时限制。
4. **人工干预拦截 (Human Interruption)**：在 Human-in-the-loop 节点中，用户直接选择了“终止”或“回滚”。

---

### Q4: 如何判断 Agent 已经完成任务？
1. **确定性自动化测试（推荐）**：对于可验证的任务，直接运行校验代码。例如在代码生成任务中，执行测试脚本（如 `pytest`），当通过率达到 100% 时，判定任务完成。
2. **双模型审计机制（Self-Evaluation）**：引入一个独立的、提示词非常严苛的审计 LLM。将“用户的初始需求”与“当前最新的系统状态”同时输入审计大模型，由其回答布尔值 `Is_Completed`。
3. **图终点节点控制（Final Node Control）**：不要允许 LLM 仅在文本中说“我搞定了”就判定任务结束。必须通过图编排，引导模型调用专属工具输出最终答案，只有进入特定的 `Final_Node` 且状态变量 `is_completed == True` 时，整个循环才优雅退出。

---

### Q5: 如何判断 Agent 只是在重复无效动作？
1. **动作指纹哈希（Action Hashing）**：
   在 State 中维护一个最近 N 次动作的滑窗队列。为每次动作计算 Hash 指纹：
   $$\text{Hash} = \text{MD5}(\text{tool\_name} + \text{str}(\text{tool\_args}))$$
   如果该 Hash 连续重复出现 2 次以上，或者在滑窗队列中占比超限，说明 Agent 已陷入死循环。
2. **状态停滞校验（State Stagnation）**：
   如果连续 3 次循环后，外部物理环境的状态（如目标文件大小、编译输出报错日志的堆栈信息、抓取的网页字节数）完全没有任何 Delta（差异值），说明 Agent 的所有尝试均未产生新的有效信息。

---

### Q6: 如何判断 Agent 需要更多上下文？
1. **工具层强力提示**：例如检索工具返回了 `Truncated: Content exceeds limit` 提示，或读取代码文件只显示了前 100 行。
2. **模型认知困惑检测**：LLM 在 Reason 过程中高频使用“我猜测”、“可能是由于未知的函数定义导致”等代词，并在思考区明确输出“需要读取 XXX 文件的完整上下文”。
3. **工具报错频繁发生**：Agent 频繁调用同类工具但由于找不到对应类、包或依赖而反复报 `ModuleNotFoundError`。

---

### Q7: 如何判断 Agent 应该停止并报告失败？
1. **硬熔断警报触发**：动作指纹哈希判定已经陷入死循环“鬼打墙”。
2. **遇到不可修复的物理错误**：如遭遇 `403 Forbidden`、云服务器欠费、本地文件系统空间已满等。
3. **重试次数达上限**：单一业务节点（如调用外部编译工具）连续报错且重试次数超过设定的最大限制（通常为 3-5 次）。

---

## 二、 资源预算与安全硬门槛设计 (Q8-Q11)

> [!IMPORTANT]
> **面试大加分项**：
> 在设计生产级 Agent 系统时，**绝对不能**任由大模型无限期执行循环。必须在系统的调度层（Orchestrator）设计一套多维度的资源预算与限制门槛。

### Q8: 如何设置最大 step？
- **实现手段**：在 LangGraph 的 State 中定义一个 `step_counter: int` 状态，或由图编排调度器自动计算图的迭代数。
- **逻辑控制**：在每个节点执行前，执行 `state.step_counter += 1`。在控制流的条件边（Conditional Edge）中进行判定：
  ```python
  def decide_next_step(state: AgentState):
      if state.step_counter >= MAX_STEPS: # 通常设为 10-15
          return "error_node_max_steps_exceeded"
      return ...
  ```

---

### Q9: 如何设置最大 token budget？
- **实现手段**：拦截每次 LLM API 的调用，在回复的元数据（Metadata）中读取 `usage.total_tokens`，并累加记录到 State 的 `accumulated_tokens` 字段中。
- **逻辑控制**：当 `accumulated_tokens >= TOKEN_LIMIT` 时（例如 50 万 Tokens），强制熔断后续大模型调用，直接进入降级节点处理，防止高并发下因 Bug 导致巨额 API 账单的产生。

---

### Q10: 如何设置最大 tool call budget？
- **实现手段**：在工具分发器（Tool Executor）中，记录 Agent 触发工具执行的总次数 `tool_call_count`。
- **逻辑控制**：限制工具调用总次数（如最大 20 次）。超出后即使大模型再次生成 tool_calls 意图，调度层也直接予以拦截并抛出 `ToolBudgetExceeded` 异常。这能防止失控的 Agent 反复访问外部接口导致 DDOS 攻击或越权操作。

---

### Q11: 如何设置最大执行时间？
- **实现手段**：在 Agent 任务被接收的初始节点，记录开始时间戳：`state.start_time = time.time()`。
- **逻辑控制**：每次循环流转时计算 `elapsed = time.time() - state.start_time`。如果 `elapsed >= MAX_TIMEOUT_SEC`（如 180 秒），调度器强行回收该执行线程/协程，保存快照以备复现调试。

---

## 三、 经典 Loop 架构与设计模式 (Q12-Q16)

### Q12: 如何设计 planner-executor loop？
*   **设计模式**：将规划与执行分离。避免单个 LLM 节点一边规划一边执行导致注意力分散。
*   **流程拓扑**：
    ```mermaid
    flowchart TD
        Start[开始] --> Planner["Planner (生成/修改待办队列: Todo List)"]
        Planner --> CheckTodo{Todo 队列是否为空?}
        CheckTodo -- 空 --> Finish[结束任务]
        CheckTodo -- 非空 --> Executor["Executor (取出首个任务并调用工具执行)"]
        Executor --> Feedback["Feedback (将工具返回的成功/失败信息作为环境反馈)"]
        Feedback --> Planner
    ```

---

### Q13: 如何设计 coder-reviewer loop？
*   **设计模式**：用于高质量生成任务。Coded 编写方案，Reviewer 挑刺并给修改建议。
*   **流程拓扑**：
    ```mermaid
    flowchart TD
        Start[开始] --> Coder["Coder (编写代码 / 方案初稿)"]
        Coder --> Reviewer["Reviewer (静态检查/找Bug/输出意见)"]
        Reviewer --> Verdict{代码是否有问题?}
        Verdict -- 有 --> Coder
        Verdict -- 无 --> Deploy[部署并退出]
    ```

---

### Q14: 如何设计 retrieve-read-reason loop？
*   **设计模式**：进阶版 RAG 循环。不仅是一次匹配，而是当信息不完整时多步探索。
*   **流程拓扑**：
    ```mermaid
    flowchart TD
        Start[开始] --> Retrieve["Retrieve (基于当前 Query 检索知识库)"]
        Retrieve --> Read["Read (提取 Chunk 核心事实)"]
        Read --> Reason["Reason (推理：这些事实能否完整拼凑出正确答案?)"]
        Reason --> Verdict{所需拼图是否完整?}
        Verdict -- 完整 --> Answer[输出答案退出]
        Verdict -- 缺失 --> RefineQuery["Refine Query (提炼新的缺失点，重新生成检索 Query)"]
        RefineQuery --> Retrieve
    ```

---

### Q15: 如何设计 generate-test-fix loop？
*   **设计模式**：自愈式代码编写循环，通常在 Docker 沙箱环境中运行。
*   **流程拓扑**：
    ```mermaid
    flowchart TD
        Start[开始] --> Generate["Generate (编写功能代码与测试用例)"]
        Generate --> Test["Test (运行本地测试 pytest/pyright)"]
        Test --> Verdict{测试是否 100% 通过?}
        Verdict -- 是 --> Finish[生成成功]
        Verdict -- 否 --> Fix["Fix (将 Trace 报错信息打包，进行自愈重构)"]
        Fix --> Test
    ```

---

### Q16: 如何设计 ask-human-continue loop？
*   **设计模式**：高安全性、人机协同的工作流模式。
*   **流程拓扑**：
    ```mermaid
    flowchart TD
        Start[开始] --> Step["Step (普通图节点运行)"]
        Step --> CheckRisk{检测是否触碰敏感动作?<br>如写数据库/删除/付费等}
        CheckRisk -- 是 --> Interrupt["Interrupt (保存 Checkpoint 并挂起图运行)"]
        Interrupt --> Human["Human UI (人类审查并输入意见: 同意/修改/中止)"]
        Human --> Resume["Resume (读取人类指令恢复运行)"]
        Resume --> CheckChoice{人类选择为何?}
        CheckChoice -- 中止 --> Terminate[安全终止并退出]
        CheckChoice -- 同意/修改 --> Exec["Exec (应用人类参数修改并继续)"]
        Exec --> Step
        CheckRisk -- 否 --> Exec
    ```

---

## 四、 成本控制与 Loop Engineering (Q17-Q25)

### Q17: 如何避免 loop 成本失控？
1. **阶梯式模型分发（Model Tiering）**：对于终止状态判定、状态自增处理、路由判断等低智力门槛节点，强制路由给便宜且极快的轻量模型（如 `GPT-4o-Mini`）；只在核心规划与代码编写节点调用强模型。
2. **上下文滚动压缩（Chat Context Pruning）**：不能在循环中无限把所有历史 Message 塞给大模型。应在每次迭代时，仅保留最近 2-3 次的详细工具日志，而将更早的交互记录压缩为一段 200 字的“当前运行备忘录（Running Summary）”放入 System Prompt 中。

---

### Q18: 如何避免 loop 里的错误被放大？
- **错误自反思机制（Self-Correction with Critique）**：当下发报错日志给 LLM 修复时，强制其在思考区先回答：“上一次我尝试的方案是什么？为什么失败了？报错日志指向的根本原因是什么？这一次我做出了什么改变？”如果模型不能清晰解释，禁止其直接生成新的 payload。这能防止模型盲目猜测导致错误滚雪球。

---

### Q19: 如何从 trace 里发现坏 loop？
在 Langfuse、Phoenix 等可观测性工具中，坏 loop 具有鲜明的特征：
1. **拓扑死循环**：在 Trace 拓扑树中出现大面积交织的环状流，比如 `generate` 和 `test` 两个 span 连续交替出现十次以上。
2. **Token 输入陡峭上升**：因为没有做上下文压缩，第 1 次循环输入 10k Token，到第 8 次循环由于累加报错历史，单次输入陡增到 80k Token。
3. **Tool Span 密集报红**：包含大量的 `ToolError` 或异常捕获。

---

### Q20: 如何从失败样本里改进 loop？
1. **黄金样本复现测试**：将失败运行的完整状态（State）及初始输入导出来，存入测试集，在本地建立可一键运行的回归测试。
2. **边界路由优化**：如果失败是因为“模型在 Edge 处误判导致过早退出/无效重试”，则将此 Edge 逻辑由大模型判定改为硬编码的代码控制断言。

---

### Q21: 如何判断一个 loop 适合自动化？
1. **客观可检验性高（Objective Verifiability）**：任务的成败有机器能绝对判断的物理指标（如编译通过、返回 `200 OK`），而非依赖人类的审美和主观体验。
2. **单次重试成本低**：所调用的 LLM API 价格低廉，且工具端执行速度在秒级。
3. **环境易于重置与沙箱隔离**：即使 Agent 把系统改乱了，也能在 1 秒内一键回滚环境（如虚拟机快照、Docker 容器重建）。

---

### Q22: 如何判断一个 loop 必须加人工确认？
1. **涉及物理状态变更与资金安全**：如向客户发送生产邮件、向数据库写数据、发起付款。
2. **缺乏唯一可验证标准**：如设计网页的视觉排版、撰写一篇公关新闻稿，需要人类的主观判断和指导。
3. **多次尝试后阻碍未消除**：Agent 尝试了 3 次仍无法通过编译，说明本地缺少某种特殊的依赖或库，必须由人类介入安装。

---

### Q23: 如何把 prompt engineering 升级成 loop engineering？
- **Prompt Engineering** 是一种静态思维，企图编写一个包罗万象的完美 Prompt，让大模型“毕其功于一役”，但这在长序列、高难度的真实任务下很快会因为注意力分散和长上下文遗忘而失效。
- **Loop Engineering** 是一种**系统论思维**。它承认“单次大模型调用是不完美、不稳定的”，因此不强求单个 Prompt 的绝对完美，而是通过：
  - 将大任务拆解为小而聚焦的图节点（Node）；
  - 配合确定性的图路由和逻辑边（Edge）；
  - 引入测试节点、自反思修正回路以及硬性熔断机制。
  - **用系统的确定性流程设计，来包容和驾驭大模型的概率性不确定输出**，使整体方案的可靠性达到工程级生产标准。

---

### Q24: 如何让 Agent loop 不是“无限让模型再试一次”？
1. **动态纠错反馈（Feedback Adaptation）**：
   - 第一次失败：提供原始 Trace 信息。
   - 第二次失败：触发 LLM 搜索外部静态依赖库说明。
   - 第三次失败：强制提示模型改变策略，如“不能修改该文件，换一种改配置文件的手段”。
2. **引入多样性校验算力**：如果多次重试失败，强制改变图路由方向，引导大模型进入“诊断模式”或“求助人类模式”。

---

### Q25: 如何让每次 loop 都产生新的信息？
- **负反馈状态累加（Tabu/Blacklist Mechanism）**：
  在 State 中维护一个 `tried_solutions`（已尝试代码版本哈希）和 `failed_arguments` 列表。
  在 Prompt 中强制注入：
  > “当前你的任务是解决 Bug。以下是你之前几轮尝试过且**已被测试用例证伪的错误代码/参数**：`{failed_arguments}`。请在设计新方案时**严格避开**这些路径，否则测试将继续报错。”
  通过这种禁忌搜索（Tabu Search）的工程设计，强力引导大模型在每次循环中探索全新空间。
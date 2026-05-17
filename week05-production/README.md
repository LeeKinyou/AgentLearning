# Week05: API化、智能网关与深度调试

## 学习目标

掌握Agent服务化部署与生产级运维能力。

## 目录结构

```
week05-production/
├── day01_02_fastapi_service/       # Day 1-2: FastAPI封装Agent
│   ├── __init__.py
│   └── fastapi_service.py          # FastAPI服务实现
├── day03_04_litellm_gateway/       # Day 3-4: LiteLLM智能网关
│   ├── __init__.py
│   └── litellm_gateway.py          # 网关配置与路由
├── day05_07_robustness_debugging/  # Day 5-7: 鲁棒性设计
│   ├── __init__.py
│   └── robustness_debugging.py     # 重试、降级、调试
└── tests/                          # 测试用例
    ├── __init__.py
    └── tests.py                    # 完整测试套件
```

## 安装依赖

```bash
pip install fastapi uvicorn pydantic python-dotenv langchain langchain-openai litellm pyyaml
```

## 运行方式

### 启动FastAPI服务

```bash
cd week05-production/day01_02_fastapi_service
python fastapi_service.py
```

服务启动后访问：
- API文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health

### 运行LiteLLM网关演示

```bash
cd week05-production/day03_04_litellm_gateway
python litellm_gateway.py
```

### 运行鲁棒性调试演示

```bash
cd week05-production/day05_07_robustness_debugging
python robustness_debugging.py
```

### 运行测试

```bash
cd week05-production/tests
python tests.py
```

## 核心知识点

### Day 1-2: FastAPI封装Agent

1. **核心概念**
   - 路由定义与参数校验（Pydantic模型）
   - 异步端点设计（async/await）
   - 流式响应（StreamingResponse）

2. **API设计**
   - 统一请求/响应格式
   - 错误处理与状态码
   - 自动API文档生成

3. **会话管理**
   - 会话ID生成与维护
   - 对话历史存储
   - 会话清理

### Day 3-4: LiteLLM智能网关

1. **多模型路由**
   - 权重-based路由
   - 速率限制（RPM/TPM）
   - 模型可用性检查

2. **Fallback机制**
   - 主模型失败自动切换
   - Fallback链配置
   - 故障恢复

3. **成本优化**
   - Token使用量统计
   - 成本计算
   - 预算控制

### Day 5-7: 鲁棒性设计

1. **重试策略**
   - 指数退避（Exponential Backoff）
   - 断路器（Circuit Breaker）
   - 最大重试次数

2. **降级处理**
   - Fallback函数
   - 优雅降级
   - 默认响应

3. **幻觉检测**
   - 不确定性表达检测
   - 数据一致性检查
   - 置信度评分

4. **可观测性**
   - LangFuse集成
   - Trace记录
   - 性能监控

## 测试覆盖

| 测试类 | 测试内容 | 测试数 |
|--------|----------|--------|
| TestFastAPIService | 健康检查、对话端点、参数校验 | 5 |
| TestLiteLLMGateway | 模型配置、速率限制、预算管理 | 6 |
| TestRobustnessMechanisms | 重试、断路器、降级、幻觉检测 | 5 |
| TestObservability | 可观测性初始化、Trace记录 | 2 |
| TestModelDefinitions | Pydantic模型验证 | 3 |
| TestIntegration | 端到端集成测试 | 2 |

## API端点说明

### 健康检查
```
GET /health
```

### Agent对话
```
POST /api/v1/agent/chat
Content-Type: application/json

{
    "query": "你好",
    "session_id": "optional-session-id",
    "temperature": 0.7,
    "stream": false
}
```

### 流式对话
```
POST /api/v1/agent/chat/stream
Content-Type: application/json

{
    "query": "你好",
    "stream": true
}
```

### 会话管理
```
GET /api/v1/sessions/{session_id}
DELETE /api/v1/sessions/{session_id}
```

## 生产部署建议

1. **使用Gunicorn/Uvicorn**
   ```bash
   gunicorn fastapi_service:create_app -w 4 -k uvicorn.workers.UvicornWorker
   ```

2. **配置Nginx反向代理**
   - SSL/TLS终止
   - 负载均衡
   - 静态文件服务

3. **监控与告警**
   - 接入Prometheus/Grafana
   - 设置错误率告警
   - 监控Token消耗

4. **日志管理**
   - 结构化日志（JSON格式）
   - 集中式日志收集
   - 日志轮转

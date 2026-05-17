"""
Week05 生产级Agent服务 - 测试用例

测试覆盖：
1. FastAPI服务测试
2. LiteLLM网关测试
3. 鲁棒性机制测试
4. 可观测性测试
"""

import os
import sys
import time
import asyncio
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ============================================================
# 测试1: FastAPI服务测试
# ============================================================

class TestFastAPIService(unittest.TestCase):
    """FastAPI服务测试"""
    
    def setUp(self):
        """测试准备"""
        try:
            from fastapi.testclient import TestClient
            from day01_02_fastapi_service.fastapi_service import create_app
            
            self.app = create_app()
            self.client = TestClient(self.app)
            self.has_fastapi = True
        except ImportError:
            self.has_fastapi = False
    
    def test_health_check(self):
        """测试健康检查端点"""
        if not self.has_fastapi:
            self.skipTest("FastAPI未安装")
        
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("version", data)
        self.assertIn("timestamp", data)
        self.assertIn("uptime_seconds", data)
    
    def test_chat_endpoint(self):
        """测试对话端点"""
        if not self.has_fastapi:
            self.skipTest("FastAPI未安装")
        
        # 使用mock避免实际调用LLM
        with patch('day01_02_fastapi_service.fastapi_service.AgentService.process_query') as mock_process:
            # 创建异步mock
            async def mock_async(*args, **kwargs):
                return {
                    "response": "测试响应",
                    "session_id": "test_session",
                    "latency_ms": 100.0,
                    "history_length": 2
                }
            
            mock_process.side_effect = mock_async
            
            response = self.client.post("/api/v1/agent/chat", json={
                "query": "你好",
                "temperature": 0.7,
                "stream": False
            })
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data["success"])
            self.assertIn("data", data)
    
    def test_chat_endpoint_validation(self):
        """测试请求参数校验"""
        if not self.has_fastapi:
            self.skipTest("FastAPI未安装")
        
        # 测试空查询
        response = self.client.post("/api/v1/agent/chat", json={
            "query": "",
            "temperature": 0.7
        })
        self.assertEqual(response.status_code, 422)
    
    def test_session_endpoints(self):
        """测试会话管理端点"""
        if not self.has_fastapi:
            self.skipTest("FastAPI未安装")
        
        # 测试获取不存在的会话
        response = self.client.get("/api/v1/sessions/nonexistent")
        self.assertEqual(response.status_code, 404)
    
    def test_tools_endpoint(self):
        """测试工具列表端点"""
        if not self.has_fastapi:
            self.skipTest("FastAPI未安装")
        
        response = self.client.get("/api/v1/tools")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("tools", data)
        self.assertIsInstance(data["tools"], list)


# ============================================================
# 测试2: LiteLLM网关测试
# ============================================================

class TestLiteLLMGateway(unittest.TestCase):
    """LiteLLM网关测试"""
    
    def setUp(self):
        """测试准备"""
        from day03_04_litellm_gateway.litellm_gateway import LiteLLMGateway, ModelConfig
        
        self.gateway = LiteLLMGateway()
        self.ModelConfig = ModelConfig
    
    def test_model_configuration(self):
        """测试模型配置"""
        # 验证默认模型已加载
        self.assertIn("gpt-3.5-turbo", self.gateway.models)
        self.assertIn("gpt-4", self.gateway.models)
        self.assertIn("local-llm", self.gateway.models)
    
    def test_add_remove_model(self):
        """测试添加和移除模型"""
        config = self.ModelConfig(
            model_name="test-model",
            api_base="http://test.com",
            api_key="test-key",
            rpm=10
        )
        
        # 添加模型
        self.gateway.add_model(config)
        self.assertIn("test-model", self.gateway.models)
        
        # 移除模型
        self.gateway.remove_model("test-model")
        self.assertNotIn("test-model", self.gateway.models)
    
    def test_rate_limit_status(self):
        """测试速率限制状态"""
        status = self.gateway.get_rate_limit_status()
        
        # 验证返回结构
        for model_name, info in status.items():
            self.assertIn("rpm_used", info)
            self.assertIn("rpm_limit", info)
            self.assertIn("tpm_used", info)
            self.assertIn("tpm_limit", info)
            self.assertIn("available", info)
            self.assertIsInstance(info["available"], bool)
    
    def test_budget_management(self):
        """测试预算管理"""
        # 设置预算
        self.gateway.set_budget(100.0)
        self.assertEqual(self.gateway.budget_limit, 100.0)
        
        # 检查预算
        self.assertTrue(self.gateway.check_budget())
        
        # 模拟超出预算
        self.gateway.total_cost = 150.0
        self.assertFalse(self.gateway.check_budget())
        
        # 重置
        self.gateway.total_cost = 0.0
    
    def test_cost_summary(self):
        """测试成本摘要"""
        summary = self.gateway.get_cost_summary()
        
        self.assertIn("total_cost", summary)
        self.assertIn("budget_limit", summary)
        self.assertIn("model_breakdown", summary)
        self.assertIn("timestamp", summary)
    
    def test_fallback_chain(self):
        """测试Fallback链"""
        self.assertIsInstance(self.gateway.fallback_chain, list)
        self.assertGreater(len(self.gateway.fallback_chain), 0)


# ============================================================
# 测试3: 鲁棒性机制测试
# ============================================================

class TestRobustnessMechanisms(unittest.TestCase):
    """鲁棒性机制测试"""
    
    def test_exponential_backoff(self):
        """测试指数退避重试"""
        from day05_07_robustness_debugging.robustness_debugging import RetryStrategy
        
        call_count = 0
        
        @RetryStrategy.with_exponential_backoff(max_retries=2, base_delay=0.1)
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("失败")
            return "成功"
        
        result = failing_function()
        self.assertEqual(result, "成功")
        self.assertEqual(call_count, 3)
    
    def test_circuit_breaker(self):
        """测试断路器"""
        from day05_07_robustness_debugging.robustness_debugging import RetryStrategy
        
        @RetryStrategy.with_circuit_breaker(failure_threshold=2, recovery_timeout=1)
        def always_fail():
            raise Exception("总是失败")
        
        # 触发断路器
        for _ in range(2):
            try:
                always_fail()
            except:
                pass
        
        # 断路器应该打开
        with self.assertRaises(Exception) as context:
            always_fail()
        
        self.assertIn("断路器打开", str(context.exception))
    
    def test_fallback_strategy(self):
        """测试降级策略"""
        from day05_07_robustness_debugging.robustness_debugging import FallbackStrategy
        
        def fallback_func(query: str) -> str:
            return f"降级响应: {query}"
        
        @FallbackStrategy.with_fallback(fallback_func)
        def main_func(query: str) -> str:
            raise Exception("主函数失败")
        
        result = main_func("测试")
        self.assertEqual(result, "降级响应: 测试")
    
    def test_hallucination_detection(self):
        """测试幻觉检测"""
        from day05_07_robustness_debugging.robustness_debugging import HallucinationDetector
        
        detector = HallucinationDetector()
        
        # 测试低风险响应
        response1 = "Python是一种广泛使用的编程语言。"
        result1 = detector.detect(response1)
        self.assertGreater(result1["confidence_score"], 0.7)
        self.assertFalse(result1["has_hallucination_risk"])
        
        # 测试高风险响应
        response2 = "我不太确定，但可能大概也许是这样。"
        result2 = detector.detect(response2)
        self.assertLess(result2["confidence_score"], 0.7)
        self.assertTrue(result2["has_hallucination_risk"])
    
    def test_hallucination_mitigation(self):
        """测试幻觉缓解"""
        from day05_07_robustness_debugging.robustness_debugging import HallucinationDetector
        
        detector = HallucinationDetector()
        
        response = "这是一些不确定的信息。"
        detection_result = {
            "has_hallucination_risk": True,
            "confidence_score": 0.5,
            "issues": ["不确定性表达"]
        }
        
        mitigated = detector.mitigate(response, detection_result)
        self.assertIn("系统提示", mitigated)


# ============================================================
# 测试4: 可观测性测试
# ============================================================

class TestObservability(unittest.TestCase):
    """可观测性测试"""
    
    def test_observability_manager_init(self):
        """测试可观测性管理器初始化"""
        from day05_07_robustness_debugging.robustness_debugging import ObservabilityManager
        
        manager = ObservabilityManager()
        self.assertIsNotNone(manager)
        self.assertIsInstance(manager.enabled, bool)
    
    def test_trace_llm_call(self):
        """测试LLM调用追踪"""
        from day05_07_robustness_debugging.robustness_debugging import ObservabilityManager
        
        manager = ObservabilityManager()
        
        # 不应抛出异常（即使LangFuse未配置）
        try:
            manager.trace_llm_call(
                trace_id="test_trace",
                model="gpt-3.5-turbo",
                prompt="测试",
                response="响应",
                latency_ms=100.0,
                tokens_used=10
            )
        except Exception as e:
            self.fail(f"trace_llm_call不应抛出异常: {str(e)}")


# ============================================================
# 测试5: 模型定义测试
# ============================================================

class TestModelDefinitions(unittest.TestCase):
    """模型定义测试"""
    
    def test_agent_request_model(self):
        """测试Agent请求模型"""
        from day01_02_fastapi_service.fastapi_service import AgentRequest
        
        # 测试有效请求
        request = AgentRequest(
            query="你好",
            temperature=0.7,
            stream=False
        )
        self.assertEqual(request.query, "你好")
        self.assertEqual(request.temperature, 0.7)
        self.assertFalse(request.stream)
    
    def test_agent_response_model(self):
        """测试Agent响应模型"""
        from day01_02_fastapi_service.fastapi_service import AgentResponse
        
        response = AgentResponse(
            success=True,
            data={"response": "你好！"},
            timestamp="2024-01-01T00:00:00"
        )
        self.assertTrue(response.success)
        self.assertEqual(response.data["response"], "你好！")
    
    def test_health_check_model(self):
        """测试健康检查模型"""
        from day01_02_fastapi_service.fastapi_service import HealthCheck
        
        health = HealthCheck(
            status="healthy",
            version="1.0.0",
            timestamp="2024-01-01T00:00:00",
            uptime_seconds=3600.0
        )
        self.assertEqual(health.status, "healthy")


# ============================================================
# 测试6: 集成测试
# ============================================================

class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_end_to_end_agent_service(self):
        """测试端到端Agent服务"""
        from day05_07_robustness_debugging.robustness_debugging import RobustAgentService
        
        agent = RobustAgentService()
        
        # 处理查询
        result = agent.process_query("什么是Python？")
        
        # 验证响应结构
        self.assertIn("success", result)
        self.assertIn("trace_id", result)
        
        if result["success"]:
            self.assertIn("response", result)
            self.assertIn("latency_ms", result)
        else:
            self.assertIn("error", result)
    
    def test_service_stats(self):
        """测试服务统计"""
        from day05_07_robustness_debugging.robustness_debugging import RobustAgentService
        
        agent = RobustAgentService()
        
        # 初始统计
        stats = agent.get_stats()
        self.assertEqual(stats["total_requests"], 0)
        self.assertEqual(stats["error_rate"], 0.0)
        
        # 处理一些请求
        agent.process_query("测试1")
        agent.process_query("测试2")
        
        # 更新后的统计
        stats = agent.get_stats()
        self.assertGreaterEqual(stats["total_requests"], 2)


# ============================================================
# 主函数
# ============================================================

def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Week05 测试套件")
    print("=" * 60)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestFastAPIService))
    suite.addTests(loader.loadTestsFromTestCase(TestLiteLLMGateway))
    suite.addTests(loader.loadTestsFromTestCase(TestRobustnessMechanisms))
    suite.addTests(loader.loadTestsFromTestCase(TestObservability))
    suite.addTests(loader.loadTestsFromTestCase(TestModelDefinitions))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出摘要
    print("\n" + "=" * 60)
    print("测试完成！")
    print(f"总计: {result.testsRun} 个测试")
    print(f"通过: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    run_tests()

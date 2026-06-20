"""
Day 3-4: LiteLLM智能网关配置

本模块演示如何配置LiteLLM作为智能网关，包括：
1. 多模型路由策略
2. Fallback机制（故障自动切换）
3. 速率限制与配额管理
4. 成本优化策略
5. Token预算控制

学习目标：
- 理解LiteLLM网关的核心功能
- 掌握多模型路由配置
- 学会配置Fallback和限流
- 了解成本优化策略

依赖安装：
    pip install litellm fastapi uvicorn
"""

import os
import sys
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ============================================================
# 配置模型
# ============================================================

@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str
    api_base: str
    api_key: str
    rpm: int = 60  # 每分钟请求数限制
    tpm: int = 100000  # 每分钟Token数限制
    weight: int = 1  # 路由权重
    timeout: int = 30  # 超时时间（秒）
    cost_per_1k_tokens: float = 0.002  # 每1K Token成本


class LiteLLMGateway:
    """LiteLLM智能网关"""
    
    def __init__(self):
        self.models: Dict[str, ModelConfig] = {}
        self.fallback_chain: List[str] = []
        self.request_counts: Dict[str, List[float]] = {}
        self.token_counts: Dict[str, List[tuple]] = {}  # (timestamp, token_count)
        self.total_cost: float = 0.0
        self.budget_limit: Optional[float] = None
        self._init_default_models()
    
    def _init_default_models(self):
        """初始化默认模型配置"""
        # 主模型
        self.add_model(ModelConfig(
            model_name="gpt-3.5-turbo",
            api_base=os.getenv("API_BASE_URL", "http://localhost:1234/v1"),
            api_key=os.getenv("OPENAI_API_KEY", "lm-studio"),
            rpm=60,
            tpm=100000,
            weight=3,
            cost_per_1k_tokens=0.002
        ))
        
        # 备用模型1
        self.add_model(ModelConfig(
            model_name="gpt-4",
            api_base=os.getenv("API_BASE_URL", "http://localhost:1234/v1"),
            api_key=os.getenv("OPENAI_API_KEY", "lm-studio"),
            rpm=30,
            tpm=50000,
            weight=1,
            cost_per_1k_tokens=0.03
        ))
        
        # 备用模型2（本地模型）
        self.add_model(ModelConfig(
            model_name="local-llm",
            api_base="http://localhost:1234/v1",
            api_key="lm-studio",
            rpm=100,
            tpm=200000,
            weight=2,
            cost_per_1k_tokens=0.0
        ))
        
        # 设置Fallback链
        self.fallback_chain = ["gpt-3.5-turbo", "gpt-4", "local-llm"]
    
    def add_model(self, config: ModelConfig):
        """添加模型配置"""
        self.models[config.model_name] = config
        self.request_counts[config.model_name] = []
        self.token_counts[config.model_name] = []
    
    def remove_model(self, model_name: str):
        """移除模型配置"""
        if model_name in self.models:
            del self.models[model_name]
            del self.request_counts[model_name]
            del self.token_counts[model_name]
    
    # ============================================================
    # 路由策略
    # ============================================================
    
    def route_request(self, preferred_model: Optional[str] = None) -> str:
        """
        路由请求到合适的模型
        
        策略：
        1. 如果指定了preferred_model，优先使用
        2. 检查模型是否可用（未超限）
        3. 按权重选择可用模型
        
        参数：
            preferred_model: 首选模型
        
        返回：
            str: 选中的模型名称
        """
        # 优先使用指定模型
        if preferred_model and preferred_model in self.models:
            if self._is_model_available(preferred_model):
                return preferred_model
        
        # 按权重选择可用模型
        available_models = [
            (name, config.weight) 
            for name, config in self.models.items() 
            if self._is_model_available(name)
        ]
        
        if not available_models:
            raise Exception("所有模型均不可用，请稍后重试")
        
        # 按权重排序，选择权重最高的
        available_models.sort(key=lambda x: x[1], reverse=True)
        return available_models[0][0]
    
    def _is_model_available(self, model_name: str) -> bool:
        """检查模型是否可用（未超过速率限制）"""
        if model_name not in self.models:
            return False
        
        config = self.models[model_name]
        
        # 检查RPM
        current_time = time.time()
        minute_ago = current_time - 60
        recent_requests = [
            t for t in self.request_counts[model_name] 
            if t > minute_ago
        ]
        if len(recent_requests) >= config.rpm:
            return False
        
        # 检查TPM
        recent_tokens = sum(
            count for ts, count in self.token_counts[model_name] 
            if ts > minute_ago
        )
        if recent_tokens >= config.tpm:
            return False
        
        return True
    
    # ============================================================
    # Fallback机制
    # ============================================================
    
    async def complete_with_fallback(
        self, 
        messages: List[Dict[str, str]], 
        preferred_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        带Fallback的完成请求
        
        参数：
            messages: 消息列表
            preferred_model: 首选模型
        
        返回：
            Dict: 包含响应和使用的模型信息
        """
        import litellm
        
        # 确定尝试顺序
        if preferred_model:
            try_order = [preferred_model] + [
                m for m in self.fallback_chain if m != preferred_model
            ]
        else:
            try_order = self.fallback_chain
        
        last_error = None
        
        for model_name in try_order:
            if model_name not in self.models:
                continue
            
            config = self.models[model_name]
            
            try:
                # 记录请求
                self.request_counts[model_name].append(time.time())
                
                # 调用模型
                response = await litellm.acompletion(
                    model=model_name,
                    messages=messages,
                    api_base=config.api_base,
                    api_key=config.api_key,
                    timeout=config.timeout,
                    max_tokens=1000
                )
                
                # 记录Token使用量
                usage = response.get("usage", {})  # type: ignore[attr-defined]
                token_count = usage.get("total_tokens", 0)  # type: ignore[union-attr]
                self.token_counts[model_name].append((time.time(), token_count))
                
                # 计算成本
                cost = (token_count / 1000) * config.cost_per_1k_tokens
                self.total_cost += cost
                
                return {
                    "success": True,
                    "content": response["choices"][0]["message"]["content"],  # type: ignore[index]
                    "model": model_name,
                    "tokens": token_count,
                    "cost": cost,
                    "latency_ms": 0  # 实际应计算
                }
                
            except Exception as e:
                last_error = e
                print(f"模型 {model_name} 调用失败: {str(e)}")
                continue
        
        # 所有模型都失败
        return {
            "success": False,
            "error": f"所有模型均调用失败: {str(last_error)}",
            "model": None,
            "tokens": 0,
            "cost": 0
        }
    
    # ============================================================
    # 成本与预算管理
    # ============================================================
    
    def set_budget(self, budget: float):
        """设置预算上限"""
        self.budget_limit = budget
    
    def check_budget(self) -> bool:
        """检查是否超出预算"""
        if self.budget_limit is None:
            return True
        return self.total_cost < self.budget_limit
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """获取成本摘要"""
        model_costs = {}
        for model_name, counts in self.token_counts.items():
            total_tokens = sum(count for _, count in counts)
            cost = (total_tokens / 1000) * self.models[model_name].cost_per_1k_tokens
            model_costs[model_name] = {
                "total_tokens": total_tokens,
                "cost": cost,
                "request_count": len(self.request_counts[model_name])
            }
        
        return {
            "total_cost": self.total_cost,
            "budget_limit": self.budget_limit,
            "budget_remaining": self.budget_limit - self.total_cost if self.budget_limit else None,
            "model_breakdown": model_costs,
            "timestamp": datetime.now().isoformat()
        }
    
    # ============================================================
    # 速率限制查询
    # ============================================================
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """获取速率限制状态"""
        status = {}
        current_time = time.time()
        minute_ago = current_time - 60
        
        for model_name, config in self.models.items():
            recent_requests = [
                t for t in self.request_counts[model_name] 
                if t > minute_ago
            ]
            recent_tokens = sum(
                count for ts, count in self.token_counts[model_name] 
                if ts > minute_ago
            )
            
            status[model_name] = {
                "rpm_used": len(recent_requests),
                "rpm_limit": config.rpm,
                "rpm_remaining": config.rpm - len(recent_requests),
                "tpm_used": recent_tokens,
                "tpm_limit": config.tpm,
                "tpm_remaining": config.tpm - recent_tokens,
                "available": self._is_model_available(model_name)
            }
        
        return status


# ============================================================
# 网关配置演示
# ============================================================

def demonstrate_gateway_config():
    """演示网关配置"""
    print("=" * 60)
    print("LiteLLM智能网关配置演示")
    print("=" * 60)
    
    gateway = LiteLLMGateway()
    
    # 显示模型配置
    print("\n[模型配置]")
    print("-" * 40)
    for name, config in gateway.models.items():
        print(f"  模型: {name}")
        print(f"    API Base: {config.api_base}")
        print(f"    RPM限制: {config.rpm}")
        print(f"    TPM限制: {config.tpm}")
        print(f"    权重: {config.weight}")
        print(f"    成本: ${config.cost_per_1k_tokens}/1K tokens")
    
    # 显示Fallback链
    print("\n[Fallback链]")
    print("-" * 40)
    print(f"  {' -> '.join(gateway.fallback_chain)}")
    
    # 显示路由状态
    print("\n[速率限制状态]")
    print("-" * 40)
    status = gateway.get_rate_limit_status()
    for model, info in status.items():
        print(f"  {model}:")
        print(f"    RPM: {info['rpm_used']}/{info['rpm_limit']} (剩余: {info['rpm_remaining']})")
        print(f"    TPM: {info['tpm_used']}/{info['tpm_limit']} (剩余: {info['tpm_remaining']})")
        print(f"    可用: {'是' if info['available'] else '否'}")
    
    # 显示成本摘要
    print("\n[成本摘要]")
    print("-" * 40)
    gateway.set_budget(10.0)  # 设置$10预算
    summary = gateway.get_cost_summary()
    print(f"  总成本: ${summary['total_cost']:.4f}")
    print(f"  预算上限: ${summary['budget_limit']:.2f}")
    print(f"  预算剩余: ${summary['budget_remaining']:.2f}")


# ============================================================
# 配置文件生成
# ============================================================

def generate_config_file():
    """生成LiteLLM配置文件"""
    config = {
        "model_list": [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "openai/gpt-3.5-turbo",
                    "api_base": os.getenv("API_BASE_URL", "http://localhost:1234/v1"),
                    "api_key": os.getenv("OPENAI_API_KEY", "lm-studio"),
                    "rpm": 60,
                    "timeout": 30
                }
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "openai/gpt-4",
                    "api_base": os.getenv("API_BASE_URL", "http://localhost:1234/v1"),
                    "api_key": os.getenv("OPENAI_API_KEY", "lm-studio"),
                    "rpm": 30,
                    "timeout": 60
                }
            },
            {
                "model_name": "local-llm",
                "litellm_params": {
                    "model": "openai/local-llm",
                    "api_base": "http://localhost:1234/v1",
                    "api_key": "lm-studio",
                    "rpm": 100,
                    "timeout": 120
                }
            }
        ],
        "router_settings": {
            "routing_strategy": "simple-shuffle",
            "model_group_alias": {
                "default": ["gpt-3.5-turbo", "gpt-4", "local-llm"]
            }
        },
        "litellm_settings": {
            "fallbacks": [
                {"gpt-3.5-turbo": ["gpt-4", "local-llm"]},
                {"gpt-4": ["local-llm"]}
            ],
            "context_window_fallbacks": [
                {"gpt-3.5-turbo": ["gpt-4"]}
            ],
            "allowed_fails": 3,
            "cooldown_time": 30
        }
    }
    
    # 保存配置文件
    config_path = os.path.join(os.path.dirname(__file__), "litellm_config.yaml")
    
    import yaml
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    print(f"\n配置文件已生成: {config_path}")
    return config


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数"""
    demonstrate_gateway_config()
    
    # 生成配置文件
    try:
        import yaml
        generate_config_file()
    except ImportError:
        print("\n[提示] 安装PyYAML以生成配置文件: pip install pyyaml")


if __name__ == "__main__":
    main()

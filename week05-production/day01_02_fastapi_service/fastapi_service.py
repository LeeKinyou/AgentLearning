"""
Day 1-2: FastAPI封装Agent服务

本模块演示如何将Agent封装为生产级RESTful API服务，包括：
1. FastAPI核心概念：路由、参数校验、异步端点
2. Agent服务化：统一请求/响应格式
3. 流式响应（Streaming）实现
4. 错误处理与状态码规范
5. API文档自动生成

学习目标：
- 掌握FastAPI框架的核心用法
- 理解如何将Agent封装为API服务
- 学会设计统一的API响应格式
- 掌握流式响应的实现方法

依赖安装：
    pip install fastapi uvicorn pydantic python-dotenv langchain langchain-openai
"""

import os
import sys
import asyncio
from typing import Optional, List, Dict, Any, AsyncGenerator, TYPE_CHECKING
from datetime import datetime
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr

if TYPE_CHECKING:
    from fastapi import FastAPI

# 加载环境变量
load_dotenv()

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ============================================================
# Pydantic模型定义（请求/响应）
# ============================================================

class AgentRequest(BaseModel):
    """Agent请求模型"""
    query: str = Field(..., description="用户查询", min_length=1, max_length=4000)
    session_id: Optional[str] = Field(None, description="会话ID，用于保持对话上下文")
    temperature: float = Field(0.7, description="温度参数", ge=0.0, le=2.0)
    stream: bool = Field(False, description="是否使用流式响应")
    metadata: Optional[Dict[str, Any]] = Field(None, description="额外元数据")


class AgentResponse(BaseModel):
    """Agent响应模型"""
    success: bool = Field(..., description="是否成功")
    data: Optional[Dict[str, Any]] = Field(None, description="响应数据")
    error: Optional[str] = Field(None, description="错误信息")
    session_id: Optional[str] = Field(None, description="会话ID")
    timestamp: str = Field(..., description="响应时间戳")
    latency_ms: Optional[float] = Field(None, description="响应延迟（毫秒）")


class StreamChunk(BaseModel):
    """流式响应块"""
    chunk: str = Field(..., description="内容块")
    is_last: bool = Field(False, description="是否为最后一块")
    timestamp: str = Field(..., description="时间戳")


class HealthCheck(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    version: str = Field(..., description="API版本")
    timestamp: str = Field(..., description="检查时间")
    uptime_seconds: float = Field(..., description="运行时间（秒）")


# ============================================================
# Agent核心逻辑
# ============================================================

class AgentService:
    """Agent服务核心类"""
    
    def __init__(self):
        self.sessions: Dict[str, List[Dict]] = {}
        self.start_time = datetime.now()
        self.llm = self._create_llm()
    
    def _create_llm(self):
        """创建LLM实例"""
        from langchain_openai import ChatOpenAI
        
        api_key = os.getenv("OPENAI_API_KEY", "lm-studio")
        base_url = os.getenv("API_BASE_URL", "http://localhost:1234/v1")
        model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
        
        return ChatOpenAI(
            model=model_name,
            temperature=0.7,
            api_key=SecretStr(api_key),
            base_url=base_url,
        )
    
    async def process_query(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        处理用户查询
        
        参数：
            query: 用户查询
            session_id: 会话ID
            temperature: 温度参数
        
        返回：
            Dict: 包含响应内容和元数据
        """
        start_time = datetime.now()
        
        # 获取或创建会话
        if session_id and session_id in self.sessions:
            history = self.sessions[session_id]
        else:
            history = []
            session_id = session_id or f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # 调用LLM
            from langchain_core.messages import HumanMessage, AIMessage
            
            messages = []
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
            messages.append(HumanMessage(content=query))
            
            response = await self.llm.ainvoke(messages)
            response_text = response.content
            
            # 更新会话历史
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response_text})
            self.sessions[session_id] = history
            
            # 计算延迟
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "response": response_text,
                "session_id": session_id,
                "latency_ms": latency,
                "history_length": len(history)
            }
            
        except Exception as e:
            raise Exception(f"Agent处理失败: {str(e)}")
    
    async def stream_query(
        self, 
        query: str, 
        session_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        流式处理用户查询
        
        参数：
            query: 用户查询
            session_id: 会话ID
        
        返回：
            AsyncGenerator: 流式响应块
        """
        from langchain_core.messages import HumanMessage
        
        messages = [HumanMessage(content=query)]
        
        try:
            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    yield str(chunk.content)
        except Exception as e:
            yield f"[错误] {str(e)}"
    
    def get_health(self) -> Dict[str, Any]:
        """获取服务健康状态"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        return {
            "status": "healthy",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime
        }
    
    def get_session_history(self, session_id: str) -> List[Dict]:
        """获取会话历史"""
        return self.sessions.get(session_id, [])
    
    def clear_session(self, session_id: str) -> bool:
        """清除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False


# ============================================================
# FastAPI应用
# ============================================================

def create_app() -> "FastAPI":
    """创建FastAPI应用"""
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    
    # 创建Agent服务实例
    agent_service = AgentService()
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """应用生命周期管理"""
        print("Agent服务启动...")
        yield
        print("Agent服务关闭...")
    
    app = FastAPI(
        title="Agent API Service",
        description="生产级Agent API服务 - 支持对话、流式响应、会话管理",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # 配置CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # --------------------------------------------------------
    # 健康检查端点
    # --------------------------------------------------------
    @app.get("/health", response_model=HealthCheck)
    async def health_check():
        """健康检查端点"""
        return agent_service.get_health()
    
    # --------------------------------------------------------
    # Agent对话端点（非流式）
    # --------------------------------------------------------
    @app.post("/api/v1/agent/chat", response_model=AgentResponse)
    async def chat(request: AgentRequest):
        """
        Agent对话端点
        
        接收用户查询，返回Agent响应。
        支持会话保持，通过session_id维护对话上下文。
        """
        start_time = datetime.now()
        
        try:
            result = await agent_service.process_query(
                query=request.query,
                session_id=request.session_id,
                temperature=request.temperature
            )
            
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            return AgentResponse(
                success=True,
                data={
                    "response": result["response"],
                    "history_length": result["history_length"]
                },
                error=None,
                session_id=result["session_id"],
                timestamp=datetime.now().isoformat(),
                latency_ms=latency
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"处理请求时发生错误: {str(e)}"
            )
    
    # --------------------------------------------------------
    # Agent对话端点（流式）
    # --------------------------------------------------------
    @app.post("/api/v1/agent/chat/stream")
    async def chat_stream(request: AgentRequest):
        """
        Agent流式对话端点
        
        接收用户查询，返回流式响应。
        适用于需要实时显示生成内容的场景。
        """
        async def generate_stream() -> AsyncGenerator[str, None]:
            async for chunk in agent_service.stream_query(
                query=request.query,
                session_id=request.session_id
            ):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    
    # --------------------------------------------------------
    # 会话管理端点
    # --------------------------------------------------------
    @app.get("/api/v1/sessions/{session_id}")
    async def get_session(session_id: str):
        """获取会话历史"""
        history = agent_service.get_session_history(session_id)
        if not history:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {
            "session_id": session_id,
            "history": history,
            "message_count": len(history)
        }
    
    @app.delete("/api/v1/sessions/{session_id}")
    async def delete_session(session_id: str):
        """删除会话"""
        if agent_service.clear_session(session_id):
            return {"message": "会话已删除", "session_id": session_id}
        raise HTTPException(status_code=404, detail="会话不存在")
    
    # --------------------------------------------------------
    # 工具端点
    # --------------------------------------------------------
    @app.get("/api/v1/tools")
    async def list_tools():
        """列出可用工具"""
        return {
            "tools": [
                {
                    "name": "calculator",
                    "description": "数学计算器",
                    "parameters": {"expression": "数学表达式"}
                },
                {
                    "name": "datetime",
                    "description": "日期时间查询",
                    "parameters": {"query": "查询内容"}
                }
            ]
        }
    
    return app


# ============================================================
# 命令行启动
# ============================================================

def main():
    """主函数：启动FastAPI服务"""
    import uvicorn
    
    print("=" * 60)
    print("FastAPI Agent服务")
    print("=" * 60)
    print("\n启动服务...")
    print("API文档地址: http://localhost:8000/docs")
    print("健康检查: http://localhost:8000/health")
    print("\n按 Ctrl+C 停止服务")
    print("=" * 60)
    
    uvicorn.run(
        "week05-production.day01_02_fastapi_service.fastapi_service:create_app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        factory=True
    )


if __name__ == "__main__":
    main()

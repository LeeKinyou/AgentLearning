import os
import uuid
import time
from typing import Dict, Any, List
from dotenv import load_dotenv

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

# 尝试加载环境变量
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

app = FastAPI(title="RepoMind Agent Backend Demo")

# === 模拟数据库 (内存存储) ===
# 格式: { "task_id": { "status": "pending|running|completed|failed", "steps": [], "result": None, "issue": "..." } }
tasks_db: Dict[str, Dict[str, Any]] = {}

# === 请求模型 ===
class AnalyzeRequest(BaseModel):
    project_path: str
    issue: str

# === 后台 Agent 工作流模拟函数 ===
def run_agent_workflow(task_id: str, project_path: str, issue: str):
    """
    模拟一个需要在后台运行很久的 Agent 循环流程。
    """
    try:
        tasks_db[task_id]["status"] = "running"
        
        # Step 1: 扫描目录
        tasks_db[task_id]["steps"].append("Agent: 正在扫描项目目录结构...")
        time.sleep(1.5)  # 模拟耗时
        
        # Step 2: 搜索关键字
        tasks_db[task_id]["steps"].append(f"Agent: 尝试搜索与报错 '{issue[:10]}...' 相关的关键字...")
        time.sleep(1.5)
        
        # Step 3: 读取文件
        tasks_db[task_id]["steps"].append("Agent: 发现可疑文件 pyproject.toml，正在读取...")
        time.sleep(1.5)
        
        # 模拟生成最终结论
        result = (
            f"经过对路径 '{project_path}' 的分析，Agent 认为这是一个缺少依赖的错误。\n"
            f"建议在终端执行: uv pip install repomind"
        )
        tasks_db[task_id]["result"] = result
        tasks_db[task_id]["status"] = "completed"
        
    except Exception as e:
        tasks_db[task_id]["status"] = "failed"
        tasks_db[task_id]["result"] = str(e)


# === API 路由 ===

@app.post("/api/analyze")
async def start_analysis(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    提交分析任务。生成唯一的 task_id，并将长耗时的 Agent 执行过程放入后台。
    """
    task_id = str(uuid.uuid4())
    
    # 初始化任务状态
    tasks_db[task_id] = {
        "status": "pending",
        "issue": request.issue,
        "steps": [],
        "result": None
    }
    
    # 放入后台队列
    background_tasks.add_task(run_agent_workflow, task_id, request.project_path, request.issue)
    
    return {
        "message": "分析任务已受理并开始在后台运行",
        "task_id": task_id,
        "status": "pending"
    }


@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    """
    前端轮询该接口获取任务当前的状态、已经执行的步骤轨迹和最终结果。
    """
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
        
    return tasks_db[task_id]

# === 启动说明 ===
# 运行方式：uv run uvicorn demo05_engineering_backend:app --reload

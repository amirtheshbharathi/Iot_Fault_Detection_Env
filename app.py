import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List

from env import IoTEnvironment
from models import Action, Observation, EnvironmentState, GraderResult
from tasks import TASKS

app = FastAPI(title="IoT Fault Diagnosis OpenEnv")
environment = IoTEnvironment()

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]

@app.post("/reset")
@app.get("/reset")
def reset_env(task_name: str = "easy") -> Observation:
    try:
        return environment.reset(task_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step_env(action: Action) -> StepResponse:
    try:
        obs, reward, done, info = environment.step(action)
        return StepResponse(observation=obs, reward=reward.value, done=done, info=info)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def get_state() -> EnvironmentState:
    try:
        return environment.get_state()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/tasks")
def list_tasks() -> List[str]:
    return list(TASKS.keys())

class GraderRequest(BaseModel):
    action: Action
    state: EnvironmentState

@app.post("/grader")
def run_grader(req: GraderRequest) -> GraderResult:
    from grader import evaluate_action
    return evaluate_action(req.action, req.state)

@app.get("/baseline")
def run_baseline_endpoint():
    # We can invoke the baseline script and return its stdout.
    try:
        result = subprocess.run(["python", "baseline.py"], capture_output=True, text=True, check=True)
        return {"output": result.stdout}
    except subprocess.CalledProcessError as e:
        return {"error": e.stderr}

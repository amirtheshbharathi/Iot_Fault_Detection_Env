from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any

class Observation(BaseModel):
    timestamp: float
    # Sensor name mapping to a list of floats (time-series). None encodes missing values.
    sensor_data: Dict[str, List[Optional[float]]]
    system_metadata: Dict[str, str]
    history: List[Dict[str, Any]]

class Action(BaseModel):
    action_type: str = Field(..., description="Must be 'diagnose', 'request_data', or 'tool_call'")
    tool_call: Optional[str] = None
    tool_params: Optional[Dict[str, Any]] = None
    thought: Optional[str] = None
    diagnosis: Optional[str] = None
    root_cause: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    recommended_action: Optional[str] = None
    explanation: Optional[str] = None

    @validator("action_type")
    def validate_action_type(cls, v):
        if v not in ["diagnose", "request_data", "tool_call"]:
            raise ValueError("action_type must be 'diagnose', 'request_data', or 'tool_call'")
        return v

class Reward(BaseModel):
    value: float
    done: bool
    info: Dict[str, Any]

class GraderResult(BaseModel):
    total_score: float
    breakdown: Dict[str, float]

class EnvironmentState(BaseModel):
    task_name: str
    # Internal representation of the ENTIRE time series available in the environment
    full_sensor_data: Dict[str, List[Optional[float]]]
    system_metadata: Dict[str, str]
    # Ground truth answers for the task
    true_diagnosis: str
    true_root_cause: str
    true_recommended_action: str
    # Progress trackers
    current_time_index: int
    max_time_index: int
    history: List[Dict[str, Any]]
    is_done: bool
    total_reward: float
    # SoA Extensions
    energy_consumption: float = 0.0
    latency: float = 0.0
    is_denoised: bool = False
    current_sampling_rate: int = 1000
    noisy_sensor_data: Optional[Dict[str, List[Optional[float]]]] = None

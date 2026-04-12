import numpy as np
from typing import Tuple, Dict, List, Optional
from models import EnvironmentState

def generate_normal_task() -> EnvironmentState:
    """
    Normal Task: Healthy baseline machine. No anomalies.
    """
    np.random.seed(111)
    timesteps = 50
    temperature = np.random.normal(loc=50.0, scale=1.0, size=timesteps).tolist()
    pressure = np.random.normal(loc=110.0, scale=0.5, size=timesteps).tolist()
    vibration = np.random.normal(loc=0.8, scale=0.05, size=timesteps).tolist()

    return EnvironmentState(
        task_name="normal",
        full_sensor_data={
            "temperature": temperature,
            "pressure": pressure,
            "vibration": vibration
        },
        system_metadata={
            "machine_type": "Pump",
            "location": "Plant A, Sector 2",
            "normal_ranges": "temperature: 45-55C, pressure: 108-112, vibration: 0.6-1.0"
        },
        true_diagnosis="Normal Operation",
        true_root_cause="None",
        true_recommended_action="Continue monitoring",
        current_time_index=20,
        max_time_index=timesteps,
        history=[],
        is_done=False,
        total_reward=0.0
    )

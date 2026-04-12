import numpy as np
from typing import Tuple, Dict, List, Optional
from models import EnvironmentState

def generate_medium_task() -> EnvironmentState:
    """
    Medium Task: Multi-sensor interaction, requires correlation reasoning.
    (e.g., vibration increases slightly, followed by a delayed pressure drop).
    """
    np.random.seed(123)
    timesteps = 60
    
    temperature = np.random.normal(loc=60.0, scale=1.5, size=timesteps).tolist()
    pressure = np.random.normal(loc=120.0, scale=2.0, size=timesteps).tolist()
    vibration = np.random.normal(loc=1.2, scale=0.15, size=timesteps).tolist()

    # Sensor Drift: Slowly bias the pressure downward starting from step 10
    for i in range(10, timesteps):
        pressure[i] -= 0.5 * (i - 10) # linear downward drift (sensor bias)

    return EnvironmentState(
        task_name="medium",
        full_sensor_data={
            "temperature": temperature,
            "pressure": pressure,
            "vibration": vibration
        },
        system_metadata={
            "machine_type": "Gas Turbine",
            "location": "Plant B, Generator Room",
            "normal_ranges": "temperature: 57-63C, pressure: 115-125, vibration: 0.9-1.5"
        },
        true_diagnosis="Pressure sensor drift",
        true_root_cause="Sensor calibration decay over time",
        true_recommended_action="Recalibrate pressure sensor",
        current_time_index=15,  # Start early enough to observe the pressure drift from step 10
        max_time_index=timesteps,
        history=[],
        is_done=False,
        total_reward=0.0
    )

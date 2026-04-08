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

    # Step 25: Vibration starts increasing
    for i in range(25, timesteps):
        vibration[i] += 0.05 * (i - 25) # gradual increase
    
    # Step 35: Pressure drops due to bearing degradation causing seal leak
    for i in range(35, timesteps):
        pressure[i] -= 1.0 * (i - 35) # gradual decrease

    return EnvironmentState(
        task_name="medium",
        full_sensor_data={
            "temperature": temperature,
            "pressure": pressure,
            "vibration": vibration
        },
        system_metadata={
            "machine_type": "Gas Turbine",
            "location": "Plant B, Generator Room"
        },
        true_diagnosis="Bearing degradation causing seal failure",
        true_root_cause="Bearing wear over time",
        true_recommended_action="Shut down immediately and replace bearing/seal assembly",
        current_time_index=20,
        max_time_index=timesteps,
        history=[],
        is_done=False,
        total_reward=0.0
    )

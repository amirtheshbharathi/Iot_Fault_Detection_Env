import numpy as np
from typing import Tuple, Dict, List, Optional
from models import EnvironmentState

def generate_easy_task() -> EnvironmentState:
    """
    Easy Task: Clear anomaly (temperature spike), single-sensor reasoning.
    """
    np.random.seed(42)
    # Generate 50 time steps of normal data
    temperature = np.random.normal(loc=45.0, scale=2.0, size=50).tolist()
    pressure = np.random.normal(loc=100.0, scale=1.0, size=50).tolist()
    vibration = np.random.normal(loc=0.5, scale=0.1, size=50).tolist()

    # Introduce a massive temperature spike at step 30
    for i in range(30, 40):
        temperature[i] += 40.0  # Spike up to ~85

    return EnvironmentState(
        task_name="easy",
        full_sensor_data={
            "temperature": temperature,
            "pressure": pressure,
            "vibration": vibration
        },
        system_metadata={
            "machine_type": "Pump",
            "location": "Plant A, Sector 1"
        },
        true_diagnosis="Cooling system failure",
        true_root_cause="Blocked coolant valve",
        true_recommended_action="Replace coolant valve and restart pump",
        current_time_index=20,  # Agent starts observing before the spike
        max_time_index=50,
        history=[],
        is_done=False,
        total_reward=0.0
    )

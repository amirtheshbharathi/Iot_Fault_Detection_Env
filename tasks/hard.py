import numpy as np
from typing import Tuple, Dict, List, Optional
from models import EnvironmentState

def generate_hard_task() -> EnvironmentState:
    """
    Hard Task: Missing values, noisy/conflicting signals, ambiguous fault.
    """
    np.random.seed(999)
    timesteps = 100
    
    temperature = np.random.normal(loc=75.0, scale=5.0, size=timesteps).tolist()
    pressure = np.random.normal(loc=200.0, scale=10.0, size=timesteps).tolist()
    vibration = np.random.normal(loc=2.5, scale=0.5, size=timesteps).tolist()

    # Inject missing values (sensor dropouts) and noisy spikes
    for i in range(timesteps):
        if np.random.rand() < 0.15:
            temperature[i] = None # 15% missing
        if np.random.rand() < 0.10:
            pressure[i] = None
        if np.random.rand() < 0.05:
            vibration[i] += np.random.choice([5.0, -5.0]) # Occasional highly noisy spike
            
    # Ambiguous fault: temperature slowly rises, vibration suddenly stops dropping (becomes constant 0)
    # Could be a sensor failure or a seized rotor.
    # We will define it as a seized rotor which caused sensor line to snap (vibration -> 0.0 exactly).
    for i in range(50, timesteps):
        if temperature[i] is not None:
            temperature[i] += 0.5 * (i - 50)
    for i in range(60, timesteps):
        vibration[i] = 0.0

    return EnvironmentState(
        task_name="hard",
        full_sensor_data={
            "temperature": temperature,
            "pressure": pressure,
            "vibration": vibration
        },
        system_metadata={
            "machine_type": "Centrifugal Compressor",
            "location": "Plant C, Exterior Unit"
        },
        true_diagnosis="Rotor seizure with severed vibration sensor line",
        true_root_cause="Loss of lubrication leading to mechanical lock",
        true_recommended_action="Emergency stop, inspect lubrication lines and replace rotor",
        current_time_index=20,  # Start before anomaly window (vibration zeroes at index 60)
        max_time_index=timesteps,
        history=[],
        is_done=False,
        total_reward=0.0
    )

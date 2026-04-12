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
            
    # Intermittent electrical fault: Random noise spikes on vibration that vanish quickly
    for i in range(timesteps):
        if np.random.rand() < 0.10:
            if vibration[i] is not None:
                vibration[i] += 15.0  # Massive spike

    # Generate Gaussian noise overlay using fixed seed (reproducibility)
    rng = np.random.default_rng(seed=42)
    noisy_temperature = [t + rng.normal(0, 4.0) if t is not None else None for t in temperature]
    noisy_pressure = [p + rng.normal(0, 8.0) if p is not None else None for p in pressure]
    noisy_vibration = [v + rng.normal(0, 1.5) if v is not None else None for v in vibration]
    
    noisy_sensor_data = {
        "temperature": noisy_temperature,
        "pressure": noisy_pressure,
        "vibration": noisy_vibration
    }

    return EnvironmentState(
        task_name="hard",
        full_sensor_data={
            "temperature": temperature,
            "pressure": pressure,
            "vibration": vibration
        },
        system_metadata={
            "machine_type": "Centrifugal Compressor",
            "location": "Plant C, Exterior Unit",
            "normal_ranges": "temperature: 65-85C, pressure: 180-220, vibration: 2.0-3.5 (any single spike above 7.0 = electrical fault, do not dismiss as noise)"
        },
        true_diagnosis="Intermittent electrical fault in vibration sensor",
        true_root_cause="Loose wire connection",
        true_recommended_action="Inspect and secure vibration sensor wiring",
        current_time_index=10,  # Start early so agent can observe vibration spikes across the window
        max_time_index=timesteps,
        history=[],
        is_done=False,
        total_reward=0.0,
        noisy_sensor_data=noisy_sensor_data
    )

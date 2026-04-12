from typing import Tuple, Dict, Any
from models import Observation, Action, Reward, EnvironmentState
from tasks import get_task
from grader import evaluate_action

class IoTEnvironment:
    def __init__(self):
        self.state: EnvironmentState = None

    def reset(self, task_name: str) -> Observation:
        self.state = get_task(task_name)
        return self._get_observation()

    def _get_observation(self) -> Observation:
        current_idx = self.state.current_time_index
        obs_data = {}
        
        # Decide which data source to use: clean or noisy
        source_data = self.state.full_sensor_data
        if self.state.noisy_sensor_data is not None and not self.state.is_denoised:
            source_data = self.state.noisy_sensor_data

        for sensor, values in source_data.items():
            obs_data[sensor] = values[:current_idx]

        metadata = dict(self.state.system_metadata)
        metadata["max_time_index"] = str(self.state.max_time_index)
        metadata["current_time_index"] = str(current_idx)
        metadata["energy_consumption"] = str(round(self.state.energy_consumption, 2))
        metadata["latency"] = str(round(self.state.latency, 2))
        metadata["is_denoised"] = str(self.state.is_denoised)
        metadata["current_sampling_rate"] = str(self.state.current_sampling_rate)

        return Observation(
            timestamp=float(current_idx),
            sensor_data=obs_data,
            system_metadata=metadata,
            history=self.state.history
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self.state is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        if self.state.is_done:
            raise ValueError("Environment is already done. Please reset.")

        reward_value = 0.0
        done = False
        info = {}

        # Handle pre-computation critical power failure
        if self.state.energy_consumption > 100:
            done = True
            info["breakdown"] = {"critical_power_failure": 0.0}
            self.state.is_done = True
            return self._get_observation(), Reward(value=0.0, done=done, info=info), done, info

        if action.action_type == "request_data":
            # Scale cost based on sampling rate: default 10.0 energy at 1000Hz
            energy_cost = 10.0 * (self.state.current_sampling_rate / 1000.0)
            self.state.energy_consumption += energy_cost
            self.state.latency += 0.01
            
            # advance time index by 5 steps at a time so anomalies are reachable
            if self.state.current_time_index < self.state.max_time_index:
                self.state.current_time_index = min(
                    self.state.current_time_index + 5,
                    self.state.max_time_index
                )
                time_ratio = self.state.current_time_index / self.state.max_time_index
                if time_ratio < 0.8:
                    reward_value = 0.01
                else:
                    reward_value = -0.05  # Penalty for excessive requests
            else:
                reward_value = -0.1  # No more data available
                
            self.state.history.append({"action": "request_data", "step": self.state.current_time_index})
            
        elif action.action_type == "tool_call":
            if action.tool_call == "analyze":
                self.state.energy_consumption += 2.0
                self.state.latency += 0.05
                
                # Compute statistical summary of the current window
                source_data = self.state.noisy_sensor_data if (self.state.noisy_sensor_data is not None and not self.state.is_denoised) else self.state.full_sensor_data
                current_idx = self.state.current_time_index
                summary = {}
                for sensor, values in source_data.items():
                    window = [v for v in values[:current_idx] if v is not None]
                    if window:
                        third = max(1, len(window) // 3)
                        early_avg = sum(window[:third]) / third
                        late_avg = sum(window[-third:]) / third
                        summary[sensor] = {
                            "mean": round(sum(window) / len(window), 2),
                            "max": round(max(window), 2),
                            "trend": round(late_avg - early_avg, 2)
                        }
                    else:
                        summary[sensor] = "No data"
                info["tool_result"] = summary
                self.state.history.append({"action": "tool_call: analyze", "step": current_idx})
                
            elif action.tool_call == "denoise_signal":
                self.state.energy_consumption += 8.0
                self.state.latency += 0.20
                self.state.is_denoised = True
                info["tool_result"] = "Signal denoised successfully."
                self.state.history.append({"action": "tool_call: denoise_signal", "step": self.state.current_time_index})
            elif action.tool_call == "set_sampling_rate":
                hz = 1000
                if action.tool_params and "hz" in action.tool_params:
                    hz = max(100, min(100000, action.tool_params["hz"]))
                self.state.current_sampling_rate = hz
                self.state.energy_consumption += 1.0  # Minimal fixed cost to switch mode
                self.state.latency += 0.05
                info["tool_result"] = f"Sampling rate updated to {hz} Hz."
                self.state.history.append({"action": f"tool_call: set_sampling_rate({hz})", "step": self.state.current_time_index})
            else:
                info["tool_result"] = f"Unknown tool: {action.tool_call}"
                
        elif action.action_type == "diagnose":
            # Grade the action
            grader_result = evaluate_action(action, self.state)
            
            reward_value = grader_result.total_score
            # Penalize overconfident wrong answers but keep reward in [0,1]
            if action.confidence and action.confidence >= 0.8 and reward_value < 0.3:
                reward_value = max(0.0, reward_value - 0.2)
                
            info["breakdown"] = grader_result.breakdown
            done = True
            self.state.is_done = True
            self.state.history.append({
                "action": "diagnose",
                "diagnosis": action.diagnosis,
                "score": reward_value
            })
            
        self.state.total_reward += reward_value
        info["total_reward"] = self.state.total_reward
        
        # Post-action critical power check
        if self.state.energy_consumption > 100 and not done: # if over cap
            done = True
            reward_value = 0.0 # reset reward to 0
            info["breakdown"] = {"critical_power_failure": 0.0}
            self.state.is_done = True
            self.state.history.append({"action": "critical_failure", "reason": "Energy exceeded 100 units."})
            
        return self._get_observation(), Reward(value=reward_value, done=done, info=info), done, info
        
    def get_state(self) -> EnvironmentState:
        if self.state is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        return self.state

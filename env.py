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
        for sensor, values in self.state.full_sensor_data.items():
            obs_data[sensor] = values[:current_idx]

        metadata = dict(self.state.system_metadata)
        metadata["max_time_index"] = str(self.state.max_time_index)
        metadata["current_time_index"] = str(current_idx)

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

        if action.action_type == "request_data":
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
            
        elif action.action_type == "diagnose":
            # Grade the action
            grader_result = evaluate_action(action, self.state)
            
            reward_value = grader_result.total_score
            # Unsafe penalty heuristic: if confidence is high but score is very low
            if action.confidence and action.confidence >= 0.8 and reward_value < 0.3:
                reward_value -= 0.2
                
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
        
        return self._get_observation(), Reward(value=reward_value, done=done, info=info), done, info
        
    def get_state(self) -> EnvironmentState:
        if self.state is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        return self.state

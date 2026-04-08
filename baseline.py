import requests
import json

BASE_URL = "http://localhost:7860"

def run_baseline(task_name: str):
    print(f"--- Running baseline for task: {task_name} ---")
    response = requests.get(f"{BASE_URL}/reset", params={"task_name": task_name})
    if response.status_code != 200:
        print("Failed to reset:", response.text)
        return
        
    obs = response.json()
    done = False
    step = 0
    
    # Simple rule-based agent
    while not done and step < 5:
        step += 1
        
        temps = obs["sensor_data"]["temperature"]
        vibes = obs["sensor_data"]["vibration"]
        press = obs["sensor_data"]["pressure"]
        
        temps = [t for t in temps if t is not None]
        vibes = [v for v in vibes if v is not None]
        press = [p for p in press if p is not None]
        
        # Collect slightly more data if available and we don't have enough
        if len(temps) < 30 and step < 3:
            action = {"action_type": "request_data"}
        else:
            # Diagnose based on heuristics
            if max(temps) > 80:
                diagnosis = "Cooling system failure"
                rc = "Blocked coolant valve"
                rec = "Replace coolant valve and restart pump"
            elif max(vibes) > 1.3 and min(press) < 110:
                diagnosis = "Bearing degradation causing seal failure"
                rc = "Bearing wear over time"
                rec = "Shut down immediately and replace bearing/seal assembly"
            else:
                diagnosis = "Rotor seizure with severed vibration sensor line"
                rc = "Loss of lubrication leading to mechanical lock"
                rec = "Emergency stop, inspect lubrication lines and replace rotor"

            action = {
                "action_type": "diagnose",
                "diagnosis": diagnosis,
                "root_cause": rc,
                "confidence": 0.9,
                "recommended_action": rec,
                "explanation": f"Based on high temperature and vibration heuristics."
            }
            
        print(f"Step {step} action: {action['action_type']}")
        res = requests.post(f"{BASE_URL}/step", json=action)
        if res.status_code != 200:
            print("Step failed:", res.text)
            return
            
        res_data = res.json()
        obs = res_data["observation"]
        done = res_data["done"]
        if done:
            print(f"Finished {task_name} with reward: {res_data['reward']}")
            print(f"Info: {json.dumps(res_data['info'], indent=2)}")
            
if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_baseline(task)

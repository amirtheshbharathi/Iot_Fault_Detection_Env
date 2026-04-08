import os
import json
import requests
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads .env file into os.environ

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")  # LLM proxy endpoint
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")  # their injected key or fallback
ENV_URL = os.getenv("ENV_URL", "https://amirthesh29-iot-fault-detection-env.hf.space")  # env server
BENCHMARK = "iot_fault_diagnosis_env"
MAX_STEPS = 15

SYSTEM_PROMPT = """You are an expert IoT fault diagnosis engineer analyzing industrial machine sensor data.
You will receive time-series statistics (temperature, pressure, vibration) from a machine.

Key fault patterns to recognize:
- Temperature spike/trend up + normal pressure/vibration → cooling system failure, blocked coolant valve
- Vibration increasing trend + pressure dropping trend → bearing degradation causing seal failure, bearing wear
- Vibration suddenly drops to exactly 0.0 (flat line) + temperature rising trend + missing/noisy sensor values → rotor seizure with severed vibration sensor line, loss of lubrication leading to mechanical lock

Pay close attention to last_10 readings. If vibration last_10 are all 0.0, that is a severed sensor line from rotor seizure.

Your response must be a JSON object (no markdown, no extra text) with ONLY these fields:
- action_type: "diagnose" or "request_data"
- diagnosis: specific fault name
- root_cause: specific mechanical cause
- confidence: float 0.0-1.0
- recommended_action: specific repair action
- explanation: reasoning based on sensor trends"""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Escape newlines in action string so it stays on one line
    action_safe = action.replace("\n", " ").replace("\r", "")
    print(f"[STEP] step={step} action={action_safe} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def get_llm_action(client: OpenAI, obs: dict, task: str, step: int) -> dict:
    """Call the LLM to decide the next action."""
    sensor_summary = {}
    total_points = 0
    max_time_index = int(obs.get("system_metadata", {}).get("max_time_index", 50))

    for sensor, values in obs.get("sensor_data", {}).items():
        clean = [v for v in values if v is not None]
        total_points = max(total_points, len(clean))
        if clean:
            third = max(1, len(clean) // 3)
            early_avg = round(sum(clean[:third]) / third, 2)
            late_avg = round(sum(clean[-third:]) / third, 2)
            sensor_summary[sensor] = {
                "total_readings": len(clean),
                "min": round(min(clean), 2),
                "max": round(max(clean), 2),
                "early_avg": early_avg,
                "late_avg": late_avg,
                "trend": round(late_avg - early_avg, 2),
                "last_10": [round(v, 2) for v in clean[-10:]],
                "last_10_are_all_zero": all(v == 0.0 for v in clean[-10:]),
            }

    # Require seeing at least 70% of total timesteps before diagnosing
    enough_data = total_points >= int(max_time_index * 0.70)

    user_msg = f"""Task: {task}
Step: {step}
Machine info: {obs.get('system_metadata', {})}

Sensor analysis (IMPORTANT - check 'trend' for anomalies):
{json.dumps(sensor_summary, indent=2)}

{"You now have enough data. You MUST respond with action_type=diagnose." if enough_data else "You need more data. Respond with action_type=request_data ONLY."}

{"Diagnose JSON (only these keys):" if enough_data else "Request data JSON:"}
{"- action_type, diagnosis, root_cause, confidence, recommended_action, explanation" if enough_data else "- action_type"}

Respond with a single JSON object, no markdown."""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=350,
        )
        text = (completion.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        action = json.loads(text)
        # Sanitize: only keep valid keys
        valid_keys = {"action_type", "diagnosis", "root_cause", "confidence", "recommended_action", "explanation"}
        return {k: v for k, v in action.items() if k in valid_keys}
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return {"action_type": "request_data" if not enough_data else "diagnose",
                "diagnosis": "Unknown fault", "root_cause": "Unclear", "confidence": 0.5,
                "recommended_action": "Inspect system", "explanation": "Parse error fallback"}


def run_task(client: OpenAI, task: str) -> None:
    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    try:
        # Reset — POST as required by the validator
        res = requests.post(f"{ENV_URL}/reset", params={"task_name": task}, json={})
        res.raise_for_status()
        obs = res.json()

        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action_dict = get_llm_action(client, obs, task, step)
            action_str = json.dumps(action_dict)

            try:
                step_res = requests.post(f"{ENV_URL}/step", json=action_dict)
                step_res.raise_for_status()
                step_data = step_res.json()

                reward = float(step_data.get("reward", 0.0))
                done = bool(step_data.get("done", False))
                obs = step_data.get("observation", obs)
                error = None
            except Exception as e:
                reward = 0.0
                done = True
                error = str(e)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

        # Score: the diagnose step reward is the grader score in [0,1].
        # Find the reward from the step where done=True (the diagnose step).
        score = 0.0
        for i, r in enumerate(rewards):
            if r > 0.05:  # request_data gives 0.01; diagnose gives grader score
                score = r
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.1

    except Exception as e:
        log_step(step=steps_taken + 1, action="null", reward=0.0, done=True, error=str(e))

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    for task in ["easy", "medium", "hard"]:
        run_task(client, task)


if __name__ == "__main__":
    main()

---
title: IoT Fault Diagnosis OpenEnv
emoji: 🏭
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
license: mit
tags:
  - openenv
---

# IoT Fault Diagnosis — OpenEnv Environment

## Description & Motivation

Industrial machines (pumps, turbines, compressors) generate continuous streams of sensor data. When something goes wrong, engineers must correlate temperature, pressure, and vibration readings to identify the root cause before catastrophic failure occurs. This environment simulates that real-world diagnostic workflow.

Agents must analyze time-series sensor data, decide when they have enough information, and produce a structured diagnosis — exactly as a human engineer would. This is a genuine evaluation task for LLM agents: it requires multi-step reasoning, uncertainty handling, and domain knowledge.

## Observation Space

Each observation contains:

| Field | Type | Description |
|---|---|---|
| `timestamp` | float | Current time index in the simulation |
| `sensor_data` | dict[str, list[float\|null]] | Time-series readings for `temperature`, `pressure`, `vibration`. Null encodes missing/dropped values. |
| `system_metadata` | dict[str, str] | Machine type, location, `max_time_index`, `current_time_index` |
| `history` | list[dict] | Previous actions taken this episode |

## Action Space

Two action types:

| Action | Required Fields | Description |
|---|---|---|
| `request_data` | `action_type` | Advances the time window by 5 steps. Use to gather more sensor readings. |
| `diagnose` | `action_type`, `diagnosis`, `root_cause`, `confidence` (0–1), `recommended_action`, `explanation` | Ends the episode. Graded against ground truth. |

## Tasks

| Task | Machine | Fault | Difficulty |
|---|---|---|---|
| `easy` | Pump | Temperature spike at step 30–40, other sensors normal | Single-sensor reasoning |
| `medium` | Gas Turbine | Vibration gradually increases from step 25, pressure drops from step 35 | Multi-sensor correlation |
| `hard` | Centrifugal Compressor | 15% missing values, noisy spikes, vibration goes to exactly 0.0 at step 60 (severed sensor line from rotor seizure) | Ambiguous, noisy, missing data |

## Reward Function

- `request_data` when time_ratio < 0.8: `+0.01` (incremental progress reward)
- `request_data` when time_ratio >= 0.8: `-0.05` (penalty for over-requesting)
- `diagnose`: grader score in `[0.0, 1.0]` based on keyword overlap with ground truth
- High confidence + wrong answer: `-0.2` penalty

## Baseline Scores

Rule-based baseline (`baseline.py`) using sensor heuristics:

| Task | Score |
|---|---|
| easy | ~0.89 |
| medium | ~0.89 |
| hard | ~0.89 |

LLM baseline (`inference.py`) using `Qwen/Qwen2.5-72B-Instruct` via HF router:

| Task | Score |
|---|---|
| easy | 0.98 |
| medium | 0.71 |
| hard | 0.99 |

## Setup & Usage

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Set environment variables:**
```bash
export HF_TOKEN="your_huggingface_token"
export LLM_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export API_BASE_URL="http://localhost:7860"
```

**Start the environment server** (Terminal 1):
```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

**Run LLM inference** (Terminal 2):
```bash
python inference.py
```

**Run rule-based baseline:**
```bash
python baseline.py
```

## Docker

```bash
docker build -t iot-fault-env .
docker run -p 7860:7860 iot-fault-env
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET/POST | `/reset?task_name=easy` | Reset environment, returns initial observation |
| POST | `/step` | Submit an action, returns observation + reward + done + info |
| GET | `/state` | Returns full current environment state |
| GET | `/tasks` | Lists available task names |

## Project Structure

```
├── app.py          # FastAPI server
├── env.py          # IoTEnvironment class (step/reset/state)
├── models.py       # Pydantic models (Observation, Action, Reward, EnvironmentState)
├── grader.py       # Programmatic grader with synonym-aware scoring
├── inference.py    # LLM baseline using OpenAI client + HF_TOKEN
├── baseline.py     # Rule-based baseline
├── openenv.yaml    # OpenEnv metadata
├── Dockerfile      # Container definition
├── requirements.txt
└── tasks/
    ├── easy.py
    ├── medium.py
    └── hard.py
```

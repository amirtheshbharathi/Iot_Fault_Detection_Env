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

## Overview

Industrial machines generate continuous sensor streams. When something goes wrong, engineers must correlate temperature, pressure, and vibration readings to identify the root cause before failure occurs. This environment simulates that diagnostic workflow as a sequential decision-making task for LLM agents.

Agents observe time-series sensor data, decide when they have enough evidence, and produce a structured diagnosis — requiring multi-step reasoning, uncertainty calibration, and domain knowledge under energy constraints.

## Observation Space

| Field | Type | Description |
|---|---|---|
| `timestamp` | float | Current time index |
| `sensor_data` | dict[str, list[float\|null]] | Time-series for `temperature`, `pressure`, `vibration`. Null = missing/dropped values. |
| `system_metadata` | dict[str, str] | Machine type, location, normal operating ranges, `current_time_index`, `max_time_index`, `energy_consumption`, `latency` |
| `history` | list[dict] | Previous actions this episode |

## Action Space

| Action | Fields | Cost |
|---|---|---|
| `request_data` | `action_type` | 10 energy, advances time window +5 steps |
| `tool_call('analyze')` | `action_type`, `tool_call` | 2 energy, returns mean/max/trend stats per sensor |
| `tool_call('denoise_signal')` | `action_type`, `tool_call` | 8 energy, removes noise from readings |
| `tool_call('set_sampling_rate')` | `action_type`, `tool_call`, `tool_params` | 1 energy, adjusts Hz |
| `diagnose` | `action_type`, `diagnosis`, `root_cause`, `confidence`, `recommended_action`, `explanation` | 0 energy, terminal — ends episode |

## Tasks

| Task | Machine | Anomaly | True Diagnosis |
|---|---|---|---|
| `normal` | Pump | None — all sensors flat within baseline | Normal Operation |
| `easy` | Pump | Massive temperature spike (steps 15–35), pressure/vibration stable | Cooling system failure |
| `medium` | Gas Turbine | Sustained linear pressure drift from step 10 | Pressure sensor drift |
| `hard` | Centrifugal Compressor | Intermittent vibration spikes (15% missing values, noisy signal) | Intermittent electrical fault in vibration sensor |

## Reward Function

| Event | Reward |
|---|---|
| `request_data` at time_ratio < 0.8 | +0.01 |
| `request_data` at time_ratio 0.8–1.0 | -0.05 |
| `request_data` past max_time_index | -0.10 |
| `diagnose` | grader score [0.0, 1.0] − (energy_used × 0.005) |
| High confidence + wrong answer (confidence ≥ 0.8, score < 0.3) | −0.20 |

The energy penalty means diagnosing at 20 energy costs 0.10 points, while diagnosing at 80 energy costs 0.40 points. Efficient agents score significantly higher.

## Grader

Diagnosis quality is evaluated by an LLM judge (Groq) scoring Accuracy, Root Cause, and Reasoning on a 1–5 scale, normalized to [0, 1]. A keyword-overlap fallback activates if the LLM judge is unavailable. Synonym groups ensure "thermal fault" and "cooling system failure" score equivalently.

## LLM Agent Results

Using `meta-llama/llama-4-scout-17b-16e-instruct` via Groq:

| Task | Steps | Energy Used | Score |
|---|---|---|---|
| normal | 4 | 14 | 0.68 |
| easy | 5 | 24 | 0.63 |
| medium | 8 | 46 | 0.52 |
| hard | 4 | 22 | 0.64 |

## Setup

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Configure `.env`:**
```
GROQ_API_KEY=your_groq_api_key
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=meta-llama/llama-4-scout-17b-16e-instruct
ENV_URL=http://localhost:7860
```

**Start the environment server:**
```bash
python -m uvicorn app:app --host 0.0.0.0 --port 7860
```

**Run the agent:**
```bash
python inference.py --scenario scenario_config.json
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
| POST | `/step` | Submit action, returns observation + reward + done + info |
| GET | `/state` | Full current environment state |
| GET | `/tasks` | List available task names |
| POST | `/grader` | Run grader directly on an action + state |

## Project Structure

```
├── app.py                  # FastAPI server
├── env.py                  # IoTEnvironment (step/reset logic, energy tracking)
├── models.py               # Pydantic models (Observation, Action, Reward, EnvironmentState)
├── grader.py               # LLM judge + keyword fallback grader
├── inference.py            # LLM agent (OpenAI-compatible client, Groq backend)
├── baseline.py             # Rule-based baseline
├── scenario_config.json    # Task sequence, max steps, system prompt
├── openenv.yaml            # OpenEnv metadata
├── Dockerfile
├── requirements.txt
└── tasks/
    ├── normal.py           # Healthy baseline — no anomalies
    ├── easy.py             # Temperature spike fault
    ├── medium.py           # Pressure sensor drift
    └── hard.py             # Intermittent vibration electrical fault
```

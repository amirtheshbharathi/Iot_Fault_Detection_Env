import os
import json
from openai import OpenAI
from models import Action, EnvironmentState, GraderResult

# Synonym groups: any word in a group is treated as equivalent during scoring
SYNONYM_GROUPS = [
    {"cooling", "coolant", "overheating", "overheat", "thermal", "temperature", "heat"},
    {"valve", "blockage", "blocked", "clog", "obstruction", "restriction"},
    {"bearing", "bearings", "wear", "degradation", "worn", "degraded"},
    {"seal", "sealing", "leak", "leaking", "failure"},
    {"vibration", "vibrations", "imbalance", "misalignment", "unbalanced"},
    {"pressure", "pressurize", "drop", "dropping"},
    {"rotor", "seizure", "seized", "lock", "locked", "mechanical"},
    {"lubrication", "lubricant", "oil", "loss"},
    {"sensor", "line", "severed", "disconnected", "snapped"},
    {"pump", "turbine", "compressor", "machine"},
    {"replace", "replacement", "inspect", "inspection", "repair", "assembly"},
    {"shutdown", "shut", "stop", "emergency", "immediately"},
    {"system", "failure", "fault", "issue", "problem", "malfunction"},
    {"increase", "increasing", "rise", "rising", "spike", "spiked", "upward"},
    {"decrease", "decreasing", "drop", "dropping", "downward"},
    {"calibration", "calibrate", "recalibrate", "recalibration", "drift", "drifting", "bias", "decay"},
    {"electrical", "electric", "wiring", "wire", "connection", "loose", "intermittent"},
    {"secure", "securing", "tighten", "tightening", "reconnect", "reconnection"},
]

def _normalize(word: str) -> str:
    w = word.lower().strip(".,;:!?()")
    for group in SYNONYM_GROUPS:
        if w in group:
            return sorted(group)[0]
    return w

def _tokenize(text: str):
    return {_normalize(w) for w in text.lower().split() if len(w) > 2}

def match_score(pred: str, true: str) -> float:
    if not pred or not true:
        return 0.0
    pred_tokens = _tokenize(pred)
    true_tokens = _tokenize(true)
    if not pred_tokens or not true_tokens:
        return 0.0
    overlap = len(pred_tokens & true_tokens)
    # Recall-based: covering half the true keywords = full score (lenient)
    return min(1.0, overlap / max(1, len(true_tokens) * 0.5))


def evaluate_action_fallback(action: Action, state: EnvironmentState) -> GraderResult:
    diag_score = match_score(action.diagnosis, state.true_diagnosis) * 0.4
    root_score = match_score(action.root_cause, state.true_root_cause) * 0.3
    rec_score  = match_score(action.recommended_action, state.true_recommended_action) * 0.1

    correctness = (diag_score + root_score + rec_score) / 0.8
    conf = action.confidence if action.confidence is not None else 0.5
    conf_score = 0.1 * (1.0 - abs(correctness - conf))

    exp_score = match_score(
        action.explanation,
        state.true_diagnosis + " " + state.true_root_cause
    ) * 0.1

    total_score = min(0.99, max(0.01, diag_score + root_score + rec_score + conf_score + exp_score))

    return GraderResult(
        total_score=total_score,
        breakdown={
            "diagnosis_correctness": round(diag_score, 3),
            "root_cause_correctness": round(root_score, 3),
            "recommended_action_correctness": round(rec_score, 3),
            "confidence_calibration": round(conf_score, 3),
            "explanation_correctness": round(exp_score, 3),
        }
    )

def evaluate_action(action: Action, state: EnvironmentState) -> GraderResult:
    try:
        api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        api_key = os.getenv("HF_TOKEN")
        model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
        
        if not api_key:
            raise ValueError("No API key provided.")
            
        client = OpenAI(base_url=api_base, api_key=api_key)
        
        prompt = f"""You are an expert grader. Evaluate this IoT Fault Diagnosis agent on a scale of 1-5 for three categories: Accuracy, Root_Cause, and Reasoning.
        
Ground Truth:
- Diagnosis: {state.true_diagnosis}
- Root Cause: {state.true_root_cause}
- Recommended Action: {state.true_recommended_action}

Agent's Output:
- Diagnosis: {action.diagnosis}
- Root Cause: {action.root_cause}
- Explanation/Reasoning: {action.explanation}
- Recommended Action: {action.recommended_action}

CRITICAL XAI RUBRIC: 
When grading 'Reasoning', you MUST deduct points if the agent fails to cite specific numerical telemetry deltas (e.g., 'pressure dropped from 102 to 85'). If the reasoning is vague, hallucinated, or lacks exact numerical evidence, the maximum Reasoning score you can give is 2.

Return a single JSON object strictly matching this format (no code blocks):
{{"Accuracy": 4, "Root_Cause": 5, "Reasoning": 3}}
"""
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
            timeout=10.0
        )
        res_text = completion.choices[0].message.content.strip()
        if res_text.startswith("```"):
            res_text = res_text.split("```")[1]
            if res_text.startswith("json"):
                res_text = res_text[4:]
        
        scores = json.loads(res_text)
        acc_score = scores.get("Accuracy", 1)
        rc_score = scores.get("Root_Cause", 1)
        reas_score = scores.get("Reasoning", 1)
        
        # Max points: 3 * 5 = 15. Min points: 3 * 1 = 3
        # Normalize to [0,1]
        sum_scores = acc_score + rc_score + reas_score
        raw_llm_score = (sum_scores - 3) / 12.0
        
        breakdown = {
            "Accuracy (1-5)": acc_score,
            "Root Cause (1-5)": rc_score,
            "Reasoning (1-5)": reas_score,
            "Raw LLM Score": round(raw_llm_score, 3)
        }
        
        # Hard Filter False Alarm
        diagnosis_text = (action.diagnosis or "").lower()
        if state.task_name == "normal" and "normal" not in diagnosis_text:
            raw_llm_score = 0.01  # strictly > 0
            breakdown["false_alarm_penalty"] = 1.0

        # Clamp strictly within (0, 1)
        raw_llm_score = min(0.99, max(0.01, raw_llm_score))
        base_result = GraderResult(total_score=raw_llm_score, breakdown=breakdown)
    except Exception as e:
        print(f"[WARNING]: LLM Judge unavailable ({e}), using keyword fallback.", flush=True)
        base_result = evaluate_action_fallback(action, state)
        
    # Apply resource penalty (-0.005 per unit)
    energy_penalty = round(state.energy_consumption * 0.005, 3)
    final_score = min(0.99, max(0.01, base_result.total_score - energy_penalty))
    base_result.breakdown["energy_penalty"] = -energy_penalty
    base_result.total_score = final_score
        
    return base_result

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


def evaluate_action(action: Action, state: EnvironmentState) -> GraderResult:
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

    total_score = min(1.0, max(0.0, diag_score + root_score + rec_score + conf_score + exp_score))

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

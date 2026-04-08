from .easy import generate_easy_task
from .medium import generate_medium_task
from .hard import generate_hard_task

TASKS = {
    "easy": generate_easy_task,
    "medium": generate_medium_task,
    "hard": generate_hard_task
}

def get_task(task_name: str):
    if task_name not in TASKS:
        raise ValueError(f"Task {task_name} not found. Available tasks: {list(TASKS.keys())}")
    return TASKS[task_name]()

from load_data import load_task, TaskName
from run_queries import run_task_queries

if __name__ == "__main__":
    task = TaskName.MMLU

    # load task data
    queries, expected_responses = load_task(task)
    # run questions
    run_task_queries(task, queries, expected_responses)

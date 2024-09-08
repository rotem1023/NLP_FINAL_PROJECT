from load_data import load_task, TaskName
from run_queries import run_task_queries

if __name__ == "__main__":
    task = TaskName.summaries_topics
    # connect to chat GPT api

    # load task data
    queries, expected_responses = load_task(task)
    # run questions
    run_task_queries(None, task, queries, expected_responses)

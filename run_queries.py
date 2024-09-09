from load_data import load_prompts, TaskName
import pandas as pd
import os



def run_task_queries(gpt_client, task_name, queries, expected_responses):
    prompts = load_prompts()
    for level in prompts:
        level_prompts = prompts[level]
        for i in range(len(level_prompts)):
            prompt = level_prompts[i]
            responses = _get_gpt_response_for_prompt(gpt_client, prompt, queries, task_name)
            # save responses to a file
            dir_to_save = _get_task_prompt_dir(task_name, level)
            table = pd.DataFrame({"prompt": [prompt for i in range(len(queries))], "queries": queries, "responses": [i for i in range(len(queries))]})
            if expected_responses is not None:
                table['expected_responses'] = expected_responses
            table.to_csv(f"{dir_to_save}/{i}.csv")


def _get_gpt_response_for_prompt(gpt_client, prompt, queries, task_name):
    output = []
    for i in range(len(queries)):
        query = _get_query(prompt, queries[i], task_name)
        # run gpt_api
        if i%10 == 0:
            print(f"Finished {i} queries for prompt: {prompt} in task: {task_name.value}")
    return output


def _get_query(prompt, original_query, task_name):
    prompt_with_query = f"{prompt}: {original_query}"
    if task_name == TaskName.sentiment_analysis:
        output = f"{prompt_with_query} The return answer should be one of these three options: positive, negative , neutral. Please return one word"
    if task_name == TaskName.summaries_topics:
        output = f"{prompt_with_query} Please provide the longest summary possible"
    if task_name == TaskName.math:
        output = f"{prompt_with_query} Write minimum words possible"
    if task_name == TaskName.MMLU: # create the instruction while loading the data
        output = f"{prompt_with_query} Please return the answer one of the following responses: A, B, C, D. " \
                 f"The response should contains one character"
    return output

def _get_task_prompt_dir(task_name, prompt_level):
    output_dir = _get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    task_output_dir = f"{output_dir}/{task_name.value}"
    os.makedirs(task_output_dir, exist_ok=True)
    prompt_level_dir = f"{task_output_dir}/{prompt_level}"
    os.makedirs(prompt_level_dir, exist_ok=True)
    return prompt_level_dir

def _get_output_dir():
    return f"{os.path.dirname(os.path.realpath(__file__))}/output"
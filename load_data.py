from enum import Enum
import os
import json
import pandas as pd
import tarfile


class TaskName(Enum):
    summaries_topics = "summaries_topics" # https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset?resource=download
    math = "math"
    sentiment_analysis = "sentiment_analysis"
    MMLU = "MMLU" # Multidisciplinary Multiple Choice, address: https://www.kaggle.com/datasets/peiyuanliu2001/mmlu-dataset?select=train.csv
    next_word = "next_word" # https://www.kaggle.com/datasets/moustacheman/next-word-prediction-dataset
    next_word_text = "next_word_from_text" # https://www.kaggle.com/datasets/ronikdedhia/next-word-prediction
    generated_text = "generated_text"

def load_prompts():
    output = {}
    prompts_dir = f"{_get_data_dir()}/prompts"
    for filename in os.listdir(prompts_dir):
        file_path = os.path.join(prompts_dir, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                output[int(filename[:-5])] = data
    return output




def load_task(task_name):
    if task_name == TaskName.summaries_topics:
        return _load_summaries_topics_task()
    if task_name == TaskName.math:
        return _load_math()
    if task_name == TaskName.sentiment_analysis:
        return _load_sentiment_analysis()
    if task_name == TaskName.MMLU:
        return _load_mmlu()
    if task_name == TaskName.next_word:
        return _load_next_word()
    if task_name == TaskName.next_word_text:
        return _load_next_word_from_text()
    if task_name == TaskName.generated_text:
        return _load_generated_text()
    raise RuntimeError("Unrecognized task")


def _get_data_dir():
    return f"{os.path.dirname(os.path.realpath(__file__))}/data"


def _load_summaries_topics_task():
    output = []
    data_dir = f"{_get_data_dir()}/summaries_topics"
    files = os.listdir(data_dir)
    files.sort()
    for filename in files:
        file_path = os.path.join(data_dir, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                content = file.read()
                output.extend(content.split("\n"))
    return output, None


def _load_sentiment_analysis():
    data_dir = _get_data_dir()

    # Use the dynamically generated file path
    filepath = os.path.join(data_dir, f"{TaskName.sentiment_analysis.value}.csv")

    # Read the CSV file using the constructed file path
    table = pd.read_csv(filepath, encoding='latin1')
    positive_table = table[table['sentiment']=='positive']
    negative_table = table[table['sentiment'] == 'negative']
    neutral_table = table[table['sentiment'] == 'neutral']

    positive_sample = positive_table.head(333)
    negative_sample = negative_table.head(333)
    neutral_sample = neutral_table.head(334)

    # Concatenate the samples into one DataFrame
    combined_table = pd.concat([positive_sample, negative_sample, neutral_sample], ignore_index=True)

    shuffled_table = combined_table.sample(frac=1, random_state=42).reset_index(drop=True)
    output_queries = list(shuffled_table['text'])
    output_responses = list(shuffled_table['sentiment'])
    return output_queries, output_responses


def _load_math():
    output_queries = []
    output_responses = []
    data_dir = f"{_get_data_dir()}/math"
    dirs = os.listdir(data_dir)
    dirs = [d for d in dirs if os.path.isdir(os.path.join(data_dir, d))]
    dirs.sort()
    for dir in dirs:
        dir_path = f"{data_dir}\{dir}"
        dir_files = os.listdir(os.path.join(data_dir,dir))
        files = [f for f in dir_files if os.path.isfile(os.path.join(dir_path, f))]
        files.sort()
        for i in range(100):
            cur_file = files[i]
            query, expected_res = _read_and_split_file(os.path.join(dir_path, cur_file))
            if query is None or expected_res is None:
                raise RuntimeError(f"Wrong format file: {cur_file}")
            output_responses.append(expected_res)
            output_queries.append(query)
    return output_queries, output_responses


def _read_and_split_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Split the content by 'Answer:' to separate problem and answer
    parts = content.split('Answer:')

    if len(parts) == 2:
        problem = parts[0].replace('Problem:', '').strip()  # Get the problem part and clean it
        solution = parts[1].strip()  # Get the solution part and clean it
        return problem, solution
    else:
        return None, None


def _load_mmlu():
    data_dir = _get_data_dir()
    table = pd.read_csv(f"{data_dir}\\MMLU.csv")
    expected_responses = list(table['answer'])
    table['queries'] = table.apply(_create_mmlu_prompt, axis=1)
    queries = list(table['queries'])
    return queries, expected_responses


def _create_mmlu_prompt(row):
    prompt= row['prompt']
    answer_a = row['A']
    answer_b = row['B']
    answer_c = row['C']
    answer_d = row['D']
    output = f"what is the best possible answer for this question:" \
             f" {prompt}\n (A) {answer_a}\n (B) {answer_b}\n (C) {answer_c}\n (D) {answer_d}\n"
    return output


def _load_next_word():
    data_dir = _get_data_dir()
    table = pd.read_csv(f"{data_dir}\\predict_next_word.csv")
    queries = []
    expected_responses = []
    # Iterate over each row in the DataFrame
    for index, row in table.iterrows():
        # Assuming the columns are named 'text' for the sentence and 'response' for the response
        input = row['Quotes']
        lst = input.rsplit(" ", 1)
        queries.append(lst[0])
        expected_responses.append(lst[1][:-1])
    return queries, expected_responses



def _load_next_word_from_text():
    data_dir = _get_data_dir()
    with open(f'{data_dir}/predict_next_word_from_text.txt', 'r', encoding='utf-8') as file:
        data = file.read()
    return  _create_queries_and_exp_responses_from_text(data)

def _create_queries_and_exp_responses_from_text(data):
    data = data.replace('\n', '')
    data_lst = data.split('.')
    queries = []
    expected_responses = []
    for i in range(len(data_lst)):
        if len(queries) >= 1000:
            break
        split_sentence = data_lst[i].split()
        if (len(split_sentence) <2):
            continue
        expected_responses.append(split_sentence[-1].strip())
        queries.append(' '.join(split_sentence[0:-1]))
    return queries, expected_responses

def _load_generated_text():
    data_dir = _get_data_dir()
    with open(f'{data_dir}/generated_text.txt') as file:
        data = file.read()
    return _create_queries_and_exp_responses_from_text(data)

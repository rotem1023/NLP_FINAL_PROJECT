from enum import Enum
import os
import json
import pandas as pd
import tarfile


class TaskName(Enum):
    summaries_topics = "summaries_topics"
    math = "math"
    sentiment_analysis = "sentiment_analysis"


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
    raise RuntimeError("Unrecognized task")


def _get_data_dir():
    return f"{os.path.dirname(os.path.realpath(__file__))}/data"


def _load_summaries_topics_task():
    output = []
    data_dir = f"{_get_data_dir()}/summaries_topics"
    for filename in os.listdir(data_dir):
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
    # TODO: implement this function
    tar_file_path = "C:\\Users\\rotem\\Downloads\\amps.tar.gz"
    with tarfile.open(tar_file_path, 'r') as tar:
        # Iterate over all members in the tar file
        for member in tar.getmembers():
            # Check if the member is a file
            if member.isfile():
                # Print the file path
                print(member.name)
import os
import pandas as pd
from load_data import TaskName
import re

i = 0

class PromptLevel:
    def __init__(self, level, first_version_data, second_version_data, third_version_data):
        self.level = level
        self.first_version_data = first_version_data
        self.second_version_data = second_version_data
        self.third_version_data = third_version_data

    def get_data_version(self, version_number):
        if version_number==1:
            return self.first_version_data
        if version_number==2:
            return self.second_version_data
        if version_number==3:
            return self.third_version_data
        raise RuntimeError("Error: incorrect version number. Possible version numbers are: 1,2,3")

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        self._index +=1
        if self._index == 1:
            return self.first_version_data
        elif self._index ==2:
            return self.second_version_data
        elif self._index ==3:
            return self.third_version_data
        else:
            raise StopIteration

class TaskData:

    def __init__(self, task_name, first_level, second_level, third_level, fourth_level, fifth_level):
        self.task_name = task_name
        self.levels_data = [first_level, second_level, third_level, fourth_level, fifth_level]

    def get_level_data(self, level):
        return self.levels_data[level-1]

    def get_level_version_data(self, level, version):
        level_data = self.get_level_data(level)
        return level_data.get_data_version(version)


    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self.levels_data):
            raise StopIteration
        data = self.levels_data[self._index]
        self._index +=1
        return data

class ResponsesData:

    def __init__(self, mmlu_task_data, predict_next_word_data, sentiment_analysis_data, next_word_from_text, generated_text):
        self.mmlu_task_data = mmlu_task_data
        self.predict_next_word_data = predict_next_word_data
        self.sentiment_analysis_data = sentiment_analysis_data
        self.predict_next_word_from_text_data = next_word_from_text
        self.predict_next_word_generated_text = generated_text

    def get_task_data(self, task_name):
        if task_name == TaskName.MMLU:
            return self.mmlu_task_data
        if task_name == TaskName.sentiment_analysis:
            return self.sentiment_analysis_data
        if task_name == TaskName.next_word:
            return self.predict_next_word_data
        if task_name == TaskName.next_word_text:
            return self.predict_next_word_from_text_data
        if task_name == TaskName.generated_text:
            return self.predict_next_word_generated_text
        raise RuntimeError("Error: invalid task name")

    def __iter__(self):
        self._index = 0
        return self



    def __next__(self):
        self._index +=1
        if self._index == 1:
            return self.mmlu_task_data
        elif self._index == 2:
            return self.predict_next_word_data
        elif self._index == 3:
            return self.sentiment_analysis_data
        elif self._index == 4:
            return self.predict_next_word_from_text_data
        elif self._index == 5:
            return self.predict_next_word_generated_text
        else:
            raise StopIteration


def _get_data_dir():
    return f"{os.path.dirname(os.path.realpath(__file__))}/output"

def _get_all_dirs_in_dir(dir_path):
    return sorted([d for d in os.listdir(dir_path) if os.path.isfile(f"{dir_path}/{d}") == False])

def _get_all_files_in_dir(dir_path):
    return sorted([d for d in os.listdir(dir_path) if os.path.isfile(f"{dir_path}/{d}")])

def _normalize_word(w):
    return re.sub(r'[^a-zA-Z]', '', w).lower().strip()





def _normalize_table(task_name, table):
    table['responses'] = table['responses'].apply(_normalize_word)
    table['expected_responses'] = table['expected_responses'].apply(_normalize_word)
    return table


def _read_task_data(task_name):
    data_dir = _get_data_dir()
    task_names_dirs= {TaskName.math: "math", TaskName.MMLU: "MMLU", TaskName.sentiment_analysis: "sentiment_analysis", TaskName.next_word: "next_word", TaskName.next_word_text : "next_word_from_text", TaskName.generated_text: "generated_text"}
    task_dir = f"{data_dir}/{task_names_dirs[task_name]}"
    prompts_level = _get_all_dirs_in_dir(task_dir)
    prompts_dic = {p: [] for p in prompts_level}
    for p in prompts_level:
        level_dir= f"{task_dir}/{p}"
        versions = _get_all_files_in_dir(level_dir)
        for v in versions:
            data = pd.read_csv(f"{level_dir}/{v}")
            data = data.drop("Unnamed: 0", axis=1)
            data = _normalize_table(task_name, data)
            prompts_dic[p].append(data)
    levels_data= []
    for p in prompts_level:
        level_data = prompts_dic[p]
        levels_data.append(PromptLevel(int(p), level_data[0], level_data[1], level_data[2]))
    task_data = TaskData(task_name, levels_data[0], levels_data[1], levels_data[2], levels_data[3], levels_data[4])
    return task_data

def read_data():
    mmlu_data = _read_task_data(TaskName.MMLU)
    sentiment_analysis_data = _read_task_data(TaskName.sentiment_analysis)
    next_word_data = _read_task_data(TaskName.next_word)
    next_word_data_from_text = _read_task_data(TaskName.next_word_text)
    generated_text = _read_task_data(TaskName.generated_text)
    data = ResponsesData(mmlu_data, next_word_data, sentiment_analysis_data, next_word_data_from_text, generated_text)
    return data

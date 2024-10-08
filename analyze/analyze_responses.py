from load_responses import read_data
from analyze.plot import *
from load_data import TaskName
import copy

def _eval_acc_per_level(data):
    acc_per_level = {}
    for task in data:
        task_name = task.task_name
        i = 0
        for level in task:
            i+=1
            cur_level = i
            if cur_level not in acc_per_level:
                acc_per_level[cur_level] = []
            for version in level:
                correct = version.apply(lambda x:  1 if x['responses'] == x['expected_responses'] else 0, axis = 1)
                acc_per_level[cur_level].append(100*sum(correct)/ len(correct))
    return acc_per_level


def get_acc_per_level_per_task(data, task):
    task_data = data.get_task_data(task)
    acc_per_level = {}
    i = 0
    for level in task_data:
        i+=1
        cur_level = i
        if cur_level not in acc_per_level:
            acc_per_level[cur_level] = []
        for version in level:
            correct = version.apply(lambda x:  1 if x['responses'] == x['expected_responses'] else 0, axis = 1)
            acc_per_level[cur_level].append(100*sum(correct)/ len(correct))
    return acc_per_level

def get_acc_per_version_task(data, task,  level):
    task_data = data.get_task_data(task)
    level_data = task_data.get_level_data(level)
    acc = {}
    i= 0
    for v in level_data:
        correct = v.apply(lambda x:  1 if x['responses'] == x['expected_responses'] else 0, axis = 1)
        acc[i] = 100*sum(correct)/ len(correct)
        i+=1
    return acc

if __name__ == "__main__":
    data = read_data()
    num_levels = 5

    acc_task_version = {i:{} for i in range(num_levels)}
    acc_task_level = {}
    for task in data:
        task_name = task.task_name
        acc_per_level_per_task = get_acc_per_level_per_task(data, task_name)
        dic = {k: np.mean(acc_per_level_per_task[k]).item() for k in acc_per_level_per_task}
        # plot_lineplot(dic, f"plot_over_acc_levels_{task_name.value}", "Levels", "Accuracy")
        acc_task_level[task_name.value] = dic
        i = 0
        for level in data.get_task_data(task_name):
            acc_of_versions = get_acc_per_version_task(data, task_name, i)
            # plot_lineplot(acc_of_versions, f"boxplot_over_acc_levels_{task_name.value}_{i}", "Versions", "Accuracy")
            acc_task_version[i][task_name.value] = acc_of_versions
            i+=1
            acc_per_level = _eval_acc_per_level(copy.deepcopy(data))
    # for i in range(num_levels):
    #     plot_lineplots(acc_task_version[i],  f"plot_over_acc_level_{i}", "Versions", "Accuracy")

    plot_lineplots(acc_task_level, f"plot_over_acc_levels", "Levels", "Accuracy")
    # plot_boxplot(acc_per_level, "boxplot_over_acc_levels", "Accuracy")

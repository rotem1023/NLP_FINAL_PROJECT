import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


def _get_outputs_dir():
    return f"{os.path.dirname(os.path.realpath(__file__))}/outputs"

def plot_boxplot(data_dic, plot_name, ylabel):
    # Data and means
    keys= sorted(list(data_dic))
    data = []
    means = []
    for k in keys:
        data.append(data_dic[k])
        means.append(np.mean(data_dic[k]))

    print(f"means: {means}")

    # Seaborn style and color palette for better aesthetics
    sns.set(style="whitegrid")
    color_palette = "Paired"  # Use a nice color palette

    # Create the box plot
    fig, ax = plt.subplots(figsize=(10, 6))
    box = ax.boxplot(data, patch_artist=True, showmeans=False, boxprops=dict(linewidth=2),
                     medianprops=dict(color='black', linewidth=2), whiskerprops=dict(linewidth=1.5))

    # colors = [sns.color_palette(color_palette)[4], sns.color_palette(color_palette)[6], sns.color_palette(color_palette)[8]]
    # # Set colors for the boxes
    # for patch, color in zip(box['boxes'], colors):
    #     patch.set_facecolor(color)
    #     patch.set_edgecolor('black')  # Black border for boxes

    # Plot the mean with customized markers
    ax.plot([i+1 for i in range(len(keys))], means, marker='d', color='black', markerfacecolor='none',
            linestyle='None', markersize=10, label='Mean')

    # Set x-axis and y-axis labels
    ax.set_xticklabels(keys, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)

    # Set y-axis range and grid lines
    # ax.set_ylim(-1, 65)
    # ax.set_yticks(np.arange(0, 101, 10))  # Y-ticks from 0 to 100 in steps of 10
    plt.yticks(fontsize=14)  # Set font size for y-ticks
    ax.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)

    # Add background and frame styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Add legend for the mean
    ax.legend(fontsize=16)
    plt.yticks(fontsize=14)

    # Save the plot
    dir_to_save = _get_outputs_dir()
    os.makedirs(dir_to_save, exist_ok=True)
    plt.savefig(f"{dir_to_save}/{plot_name}.png", dpi=300, bbox_inches='tight')

    # Show the plot
    # plt.show()

def plot_lineplots(data_dic, plot_name, xlabel, ylabel):
    keys = sorted(list(data_dic))

    # Set a Seaborn style for better aesthetics
    sns.set_style("whitegrid")
    color_palette = "Paired"

    fig, ax = plt.subplots(figsize=(10, 6))

    i =0
    for key in keys:
        ax.plot(list(data_dic[key].keys()), list(data_dic[key].values()), color=sns.color_palette(color_palette)[i], label = f"{' '.join(key.split('_'))}")
        i+=1
    _add_axes_labels(ax, xlabel, ylabel)

    ax.set_ylim(0, 100)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    # save the plot
    dir_to_save = _get_outputs_dir()
    plt.savefig(f"{dir_to_save}/{plot_name}.png", dpi=300, bbox_inches='tight')

    # Show the plot
    # plt.show()

    plt.close(fig)

def plot_lineplot(data_dic, plot_name, xlabel, ylabel):
    keys= sorted(list(data_dic))
    data = []
    for k in keys:
        data.append(data_dic[k])

    # Set a Seaborn style for better aesthetics
    sns.set_style("whitegrid")
    color_palette = "Paired"

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(keys, data, color=sns.color_palette(color_palette)[0])

    _add_axes_labels(ax, xlabel, ylabel)
    ax.set_ylim(0, 100)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    # save the plot
    dir_to_save = _get_outputs_dir()
    plt.savefig(f"{dir_to_save}/{plot_name}.png", dpi=300, bbox_inches='tight')

    # Show the plot
    # plt.show()

    plt.close(fig)


def _add_axes_labels(ax, x_label, y_label):
    ax.set_xlabel(x_label, fontsize=26)
    ax.set_ylabel(y_label, fontsize=26)
    ax.legend(fontsize=20)
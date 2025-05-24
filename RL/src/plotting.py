import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Constants
FONTSIZE = 11.
COLORS = plt.cm.get_cmap('Set1').colors
colormaps = [plt.cm.spring, plt.cm.summer, plt.cm.autumn, plt.cm.winter, plt.cm.cool]
n_colors = 5
ta, ta_op = 1, 0
# ta_op, ta = ta, ta_op
# colors = ['gray'] + [colormaps[ta](v) for v in np.linspace(0., 0.6, n_colors)[::-1]]
colors = [colormaps[ta_op](1.)] + [colormaps[ta](v) for v in np.linspace(0., 0.6, n_colors)[::-1]]
# colors = ['gray'] + [mpl.colors.rgb2hex(color[:3]) for color in COLORS]
linestyles = ['-', '--', ':', '-.', '--', ':', '-.', (0, (5, 2))]
# Dictionary to map metrics to labels
dict_of = {
    "avg_reward": "Avg. R",
    "entropy": "Entropy",
    "V_E": r"$V_E$",
    "Q_E_0": r"$Q_{E}(L)$",
    "Q_I_0": r"$Q_{I}(L)$",
    "Q_E_1": r"$Q_{E}(R)$",
    "Q_I_1": r"$Q_{I}(R)$",
    "MSVE": "MSVE",
    "VE_pi": "VE_pi",
    "VE_max": "VE_max",
    "V_I": r"$V_I$",
    "reward": "Reward",
    "action": "Action",
    "pi_0": r"$\pi_0$",
    "pi_1": r"$\pi_1$",
    "log_pi_0": r"$\log \pi_0$",
    "log_pi_1": r"$\log \pi_1$",
}

def plot_results(params, histories, environment, metric):
    """
    Plot results of the experiments with one subplot.

    Parameters:
        params (dict): Experiment parameters.
        histories (dict): Experiment histories for each condition.
        environment (object): The environment object (for change step info).
        metric (str): The metric to be plotted.
    """
    linestyles = ['-', '--', '-.']
    fig, ax = plt.subplots(figsize=(6, 2), squeeze=True)
    time_steps = params["time_steps"]
    title = f"$w_I$: {params['wI']}, $\\alpha$: {params['alpha']}, $\\tau$: {params['tau']}"
    # fig.suptitle(title, fontsize=FONTSIZE)
    fig.suptitle(title, fontsize=FONTSIZE, x=0.09, ha='left')  # Align the title to the left

    for i, (label, history) in enumerate(histories.items()):
        wE = params["wE"][int(label)]
        line_label = f"$w_E$={wE}"
        mean_metric = np.mean(history[metric], axis=0)
        std_metric = np.std(history[metric], axis=0, ddof=1)

        if i == 0:
            ax.plot(np.mean(history["max_reward"], axis=0), label="optimal", color="gray", linestyle=":")
        # Plot the mean with shaded standard deviation
        ax.plot(range(time_steps), mean_metric, label=line_label, color=colors[i], linestyle=linestyles[i], alpha=0.9)
        ax.fill_between(range(time_steps), mean_metric - std_metric, mean_metric + std_metric, color=colors[i], alpha=0.2)

    # Highlight environment change steps
    for change_step in environment.change_steps:
        ax.axvline(change_step, color="gray", linestyle="--", linewidth=1)

    ax.set_ylabel(dict_of[metric], fontsize=FONTSIZE)
    ax.set_xlabel("Timesteps", fontsize=FONTSIZE)
    # ax.legend(loc='upper right', fontsize=FONTSIZE, handlelength=1, handletextpad=0.1, ncol=4)
    # ax.legend(loc='upper right', fontsize=FONTSIZE, handlelength=1, handletextpad=0.1,
    #           ncol=4, bbox_to_anchor=(0.5, -0.1), frameon=False)
    fig.legend(loc='upper right', fontsize=FONTSIZE, handlelength=1, handletextpad=0.1, columnspacing=0.5, ncol=4,
               bbox_to_anchor=(0.9, 1.05), fancybox=False, frameon=False)
    plt.show()

def plot_results_2cols(params, histories, environment, metrics, colors=None, colors_baseline=None):
    """
    Plot results of the experiments with two subplots.

    Parameters:
        params (dict): Experiment parameters.
        histories (dict): Experiment histories for each condition.
        environment (object): The environment object (for change step info).
        metric (str): The metric to be plotted.
    """
    # linestyles = ['-', '--', '-.']
    fig, axes = plt.subplots(2, 3, figsize=(7, 2.5), sharex=True, sharey=True, squeeze=False,
                             gridspec_kw={'hspace': 0., 'wspace': 0.})
    axes = axes.flatten()
    time_steps = params["time_steps"]
    title = f"$w_E$: {params['wE']}, $\\alpha$: {params['alpha']}, $\\tau$: {params['tau']}"
    fig.suptitle(title, fontsize=FONTSIZE, y=0.95)

    for i, (label, history) in enumerate(histories.items()):
        for j, metric in enumerate(metrics):
            wI = params["wI"][int(label)]
            line_label = f"$w_I$={wI}"
            mean_metric = np.mean(history[metric], axis=0)
            std_metric = np.std(history[metric], axis=0, ddof=1)

            if i ==0 and metric == "entropy":
                for ax_i, ax in enumerate(axes):
                    ax.axhline(np.log(2), label="max" if ax_i == 0 else None, color="k", linestyle=":", linewidth=1, alpha=0.9)


            if i == 0 and metric in ["avg_reward", "V_E"]:
                axes[0].plot(np.mean(history["max_reward"], axis=0), label="optimal", color="k", linestyle=":", linewidth=1, alpha=0.9)
                axes[1].plot(np.mean(history["max_reward"], axis=0), color="k", linestyle=":", linewidth=1, alpha=0.9)
                axes[2].plot(np.mean(history["max_reward"], axis=0), color="k", linestyle=":", linewidth=1, alpha=0.9)
                axes[3].plot(np.mean(history["max_reward"], axis=0), color="k", linestyle=":", linewidth=1, alpha=0.9)
                axes[4].plot(np.mean(history["max_reward"], axis=0), color="k", linestyle=":", linewidth=1, alpha=0.9)
                axes[5].plot(np.mean(history["max_reward"], axis=0), color="k", linestyle=":", linewidth=1, alpha=0.9)

            # if i == 0 and metric == "V_E":
            #     axes[0].plot(np.mean(history["avg_reward"], axis=0), label="optimal", color="gray", linestyle="--")
            #     axes[1].plot(np.mean(history["avg_reward"], axis=0), label="optimal", color="gray", linestyle="--")
            # if i == 0 and metric == "V_I":
            #     axes[0].plot(np.mean(history["VE_pi"], axis=0), label="optimal", color="gray", linestyle="--")
            #     axes[1].plot(np.mean(history["VE_pi"], axis=0), label="optimal", color="gray", linestyle="--")
            # if i == 0 and metric in ["V_I"]:
            #     axes[0].plot(np.mean(history["varR_pi"], axis=0), label="optimal", color="gray", linestyle=":")
            #     axes[1].plot(np.mean(history["varR_pi"], axis=0), label="optimal", color="gray", linestyle=":")
            if i == 1 or i == 0:
                # Plot for first subplot
                axes[0].plot(range(time_steps), mean_metric, linewidth=1 if i==0 else 2., label=line_label, color=colors_baseline[0] if i==0 else colors[i], linestyle=linestyles[i], alpha=0.9)
                axes[0].fill_between(range(time_steps), mean_metric - std_metric, mean_metric + std_metric, color=colors[i], alpha=0.2)
            if i == 2 or i == 0:
                # Plot for second subplot
                axes[1].plot(range(time_steps), mean_metric, linewidth=1 if i==0 else 2., label=line_label if i!=0 else None, color=colors_baseline[0] if i==0 else colors[i], linestyle=linestyles[i], alpha=0.9)
                axes[1].fill_between(range(time_steps), mean_metric - std_metric, mean_metric + std_metric, color=colors_baseline[0] if i==0 else colors[i], alpha=0.2)
            if i == 3 or i == 0:
                # Plot for second subplot
                axes[2].plot(range(time_steps), mean_metric, linewidth=1 if i==0 else 2., label=line_label if i!=0 else None, color=colors_baseline[0] if i==0 else colors[i], linestyle=linestyles[i], alpha=0.9)
                axes[2].fill_between(range(time_steps), mean_metric - std_metric, mean_metric + std_metric, color=colors_baseline[0] if i==0 else colors[i], alpha=0.2)
            if i == 4 or i == 0:
                # Plot for second subplot
                axes[3].plot(range(time_steps), mean_metric, linewidth=1 if i==0 else 2., label=line_label if i!=0 else None, color=colors_baseline[1] if i==0 else colors[i], linestyle=linestyles[i], alpha=0.9)
                axes[3].fill_between(range(time_steps), mean_metric - std_metric, mean_metric + std_metric, color=colors_baseline[1] if i==0 else colors[i], alpha=0.2)
            if i == 5 or i == 0:
                # Plot for second subplot
                axes[4].plot(range(time_steps), mean_metric, linewidth=1 if i==0 else 2., label=line_label if i!=0 else None, color=colors_baseline[1] if i==0 else colors[i], linestyle=linestyles[i], alpha=0.9)
                axes[4].fill_between(range(time_steps), mean_metric - std_metric, mean_metric + std_metric, color=colors_baseline[1] if i==0 else colors[i], alpha=0.2)
            if i == 6 or i == 0:
                # Plot for second subplot
                axes[5].plot(range(time_steps), mean_metric, linewidth=1 if i==0 else 2., label=line_label if i!=0 else None, color=colors_baseline[1] if i==0 else colors[i], linestyle=linestyles[i], alpha=0.9)
                axes[5].fill_between(range(time_steps), mean_metric - std_metric, mean_metric + std_metric, color=colors_baseline[1] if i==0 else colors[i], alpha=0.2)

    # Highlight environment change steps
    for change_step in environment.change_steps:
        for ax in axes:
            ax.axvline(change_step, color="gray", linestyle=":", linewidth=1, alpha=0.9)
    axes[0].set_ylabel(dict_of[metric], fontsize=FONTSIZE)
    # axes[1].set_ylabel(dict_of[metric], fontsize=FONTSIZE)
    axes[3].set_ylabel(dict_of[metric], fontsize=FONTSIZE)
    # axes[3].set_ylabel(dict_of[metric], fontsize=FONTSIZE)

    # axes[0].set_xlabel("Timesteps", fontsize=FONTSIZE)
    # axes[1].set_xlabel("Timesteps", fontsize=FONTSIZE)
    # axes[0].legend(loc='center left', bbox_to_anchor=(1., 0.5), fontsize=FONTSIZE, handlelength=1, handletextpad=0.1)
    # axes[1].legend(loc='center left', bbox_to_anchor=(1., 0.5), fontsize=FONTSIZE, handlelength=1, handletextpad=0.1)
    # axes[2].legend(loc='center left', bbox_to_anchor=(1., 0.5), fontsize=FONTSIZE, handlelength=1, handletextpad=0.1)
    # axes[3].legend(loc='center left', bbox_to_anchor=(1., 0.5), fontsize=FONTSIZE, handlelength=1, handletextpad=0.1)
    # axes[0].legend(loc='lower center', fontsize=FONTSIZE, handlelength=1, handletextpad=0.1, ncol=4,
    #           bbox_to_anchor=(0.5, -0.5))
    # axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=len(histories), fontsize=FONTSIZE,
    #                handlelength=0.8, columnspacing=0.5, handletextpad=0.1)
    # axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=len(histories), fontsize=FONTSIZE,
    #                handlelength=0.8, columnspacing=0.5, handletextpad=0.1)
    # ymin0, ymax0 = axes[0].get_ylim()
    # ymin1, ymax1 = axes[1].get_ylim()
    #
    # for i, (label, history) in enumerate(histories.items()):
    #     if i == 0 and metric in ["V_I"]:
    #         axes[0].plot(np.mean(history["varR_pi"], axis=0), label="optimal", color="gray", linestyle=":", alpha=0.2)
    #         axes[1].plot(np.mean(history["varR_pi"], axis=0), label="optimal", color="gray", linestyle=":", alpha=0.2)
    # axes[0].set_ylim(ymin0, ymax0)
    # axes[1].set_ylim(ymin1, ymax1)
# plt.tight_layout()

# Collect all handles and labels from all axes
    handles, labels = [], []
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        handles.extend(ax_handles)
        labels.extend(ax_labels)
    # Define the formatter function
    def format_ticks(x, pos):
        if x >= 1000:
            return f"{int(x/1000)}K"
        return str(int(x))

    # Apply the formatter to your axes
    formatter = FuncFormatter(format_ticks)

    # Set the x-axis labels and apply the formatter
    axes[-3].set_xlabel("Timesteps", fontsize=FONTSIZE)
    axes[-3].xaxis.set_major_formatter(formatter)

    axes[-2].set_xlabel("Timesteps", fontsize=FONTSIZE)
    axes[-2].xaxis.set_major_formatter(formatter)

    axes[-1].set_xlabel("Timesteps", fontsize=FONTSIZE)
    axes[-1].xaxis.set_major_formatter(formatter)
    # Add a single legend for the entire figure
    # columnspacing=1.
    ordering = [0, 1, 2, 5, 3, 6, 4, 7]
    new_handles = [handles[i] for i in ordering]
    new_labels = [labels[i] for i in ordering]
    fig.legend(new_handles, new_labels, loc='lower center', fontsize=FONTSIZE,
               handlelength=2, handletextpad=0.,
               ncol=4, bbox_to_anchor=(0.5, -0.35))
    plt.show()

def plot_results_2cols2(params, histories, environment, metrics, colors=None, colors_baseline=None,reverse=False):
    """
    Plot results of the experiments with two subplots.

    Parameters:
        params (dict): Experiment parameters.
        histories (dict): Experiment histories for each condition.
        environment (object): The environment object (for change step info).
        metric (str): The metric to be plotted.
    """
    # linestyles = ['-', '--', '-.']
    fig, axes = plt.subplots(1, 2, figsize=(8, 2), sharex=True, sharey=True, squeeze=False,
                             gridspec_kw={'hspace': 0., 'wspace': 0.})
    axes = axes.flatten()
    time_steps = params["time_steps"]
    title = f"$w_E$: {params['wE']}, $\\alpha$: {params['alpha']}, $\\tau$: {params['tau']}"
    fig.suptitle(title, fontsize=FONTSIZE, y=1)
    if reverse:
        ite = list(histories.items())[0:1]
        for item in list(histories.items())[1:][::-1]:
            ite.append(item)
    else:
        ite = histories.items()
    for i, (label, history) in enumerate(ite):
        for j, metric in enumerate(metrics):
            wI = params["wI"][int(label)]
            line_label = f"$w_I$={wI}"
            mean_metric = np.mean(history[metric], axis=0)
            std_metric = np.std(history[metric], axis=0, ddof=1)

            if i ==0 and metric == "entropy":
                for ax_i, ax in enumerate(axes):
                    ax.axhline(np.log(2), label="max" if ax_i == 0 else None, color="k", linestyle=":", linewidth=1, alpha=0.9)


            if i == 0 and metric in ["avg_reward", "V_E"]:
                axes[0].plot(np.mean(history["max_reward"], axis=0), label="optimal", color="k", linestyle=":", linewidth=1, alpha=0.9)
                axes[1].plot(np.mean(history["max_reward"], axis=0), color="k", linestyle=":", linewidth=1, alpha=0.9)

            # if i == 0 and metric == "V_E":
            #     axes[0].plot(np.mean(history["avg_reward"], axis=0), label="optimal", color="gray", linestyle="--")
            #     axes[1].plot(np.mean(history["avg_reward"], axis=0), label="optimal", color="gray", linestyle="--")
            # if i == 0 and metric == "V_I":
            #     axes[0].plot(np.mean(history["VE_pi"], axis=0), label="optimal", color="gray", linestyle="--")
            #     axes[1].plot(np.mean(history["VE_pi"], axis=0), label="optimal", color="gray", linestyle="--")
            # if i == 0 and metric in ["V_I"]:
            #     axes[0].plot(np.mean(history["varR_pi"], axis=0), label="optimal", color="gray", linestyle=":")
            #     axes[1].plot(np.mean(history["varR_pi"], axis=0), label="optimal", color="gray", linestyle=":")
            if i == 1 or i == 0:
                # Plot for first subplot
                axes[0].plot(range(time_steps), mean_metric, linewidth=1 if i==0 else 2., label=line_label, color=colors_baseline[0] if i==0 else colors[i], linestyle=linestyles[i], alpha=0.9)
                axes[0].fill_between(range(time_steps), mean_metric - std_metric, mean_metric + std_metric, color=colors[i], alpha=0.2)

            if i == 2 or i == 0:
                # Plot for second subplot
                axes[1].plot(range(time_steps), mean_metric, linewidth=1 if i==0 else 2., label=line_label if i!=0 else None, color=colors_baseline[0] if i==0 else colors[i], linestyle=linestyles[i], alpha=0.9)
                axes[1].fill_between(range(time_steps), mean_metric - std_metric, mean_metric + std_metric, color=colors_baseline[0] if i==0 else colors[i], alpha=0.2)

    # Highlight environment change steps
    for change_step in environment.change_steps:
        for ax in axes:
            ax.axvline(change_step, color="gray", linestyle=":", linewidth=1, alpha=0.9)
    axes[0].set_ylabel(dict_of[metric], fontsize=FONTSIZE)


    # Collect all handles and labels from all axes
    handles, labels = [], []
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        handles.extend(ax_handles)
        labels.extend(ax_labels)
    # Define the formatter function
    def format_ticks(x, pos):
        if x >= 1000:
            return f"{int(x/1000)}K"
        return str(int(x))

    # Apply the formatter to your axes
    formatter = FuncFormatter(format_ticks)
    axes[-2].set_xlabel("Timesteps", fontsize=FONTSIZE)
    axes[-2].xaxis.set_major_formatter(formatter)

    axes[-1].set_xlabel("Timesteps", fontsize=FONTSIZE)
    axes[-1].xaxis.set_major_formatter(formatter)
    # Add a single legend for the entire figure
    # columnspacing=1.
    ordering = [0, 1, 2, 5, 3, 6, 4, 7]
    # new_handles = [handles[i] for i in ordering]
    # new_labels = [labels[i] for i in ordering]
    fig.legend(handles, labels, loc='lower center', fontsize=FONTSIZE,
               handlelength=2, handletextpad=0.,
               ncol=4, bbox_to_anchor=(0.5, -0.35))
    plt.show()

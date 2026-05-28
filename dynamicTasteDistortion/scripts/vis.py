import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from dynamicTasteDistortion.ioUtils import read_experiment, read_metrics
from dynamicTasteDistortion.scripts.metrics_utils import (
    apply_rolling_avg,
    remove_burnin,
)

from dynamicTasteDistortion.simulationConstants import SEEDS

import seaborn as sns

HALF_PAGE_FIG_SIZE = (5, 3)
FULL_PAGE_FIG_SIZE = (10, 6)


def plot_metrics_smoothed_overlay(
    ax,
    metrics,
    model_names,
    metric_name,
    window_size,
    steps,
    title,
    y_label=None,
):
    ylabel = metric_name if y_label is None else y_label
    colors = sns.color_palette("flare", len(metrics))
    markers = ["x", "^", "o", "s"]

    for i, (metric, model_name, step) in enumerate(zip(metrics, model_names, steps)):
        color = colors[i]
        marker = markers[i % len(markers)]
        metric_ma = apply_rolling_avg(metric, window_size)

        sns.lineplot(ax=ax, data=metric_ma, color=color, label=f"{model_name}")
        sns.lineplot(ax=ax, data=metric, color=color, alpha=0.3, label=None)

        if step is not None:
            idx = list(range(0, len(metric), step))
            ax.scatter(
                idx,
                [metric[j] for j in idx],
                color=color,
                marker=marker,
                zorder=4,
                s=15,
                alpha=0.6,
            )

    ax.set_title(f"{title}")
    ax.set_xlabel("Simulation Round")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()


def plot_metric_boxplot_overlay(ax, metrics, model_names, metric_name, title=None):
    colors = sns.color_palette("husl", len(metrics))

    bp = ax.boxplot(
        metrics,
        labels=model_names,
        patch_artist=True,
        flierprops=dict(marker=".", markersize=2, alpha=0.3),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    set_title = f"Distribution of {metric_name}" if title is None else title
    ax.set_title(set_title)
    ax.set_ylabel(metric_name)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3)


NON_RETRAIN_MODELS = {"random", "most_popular"}


def plot_metric_comparison(
    experiment_files,  # [baseline, challenger1, challenger2, ...]
    metric_name,
    title,
    graph="line",
    window_size=10,
    plot_retrain_points=False,
    should_remove_outliers=False,
    figsize=HALF_PAGE_FIG_SIZE,
    experiment_aliases=None,
    y_label=None,
):

    if should_remove_outliers:
        title += " (Outliers Removed)"

    metric_name_to_nice_name = {
        "mace": "Mean Average Calibration Error",
        "diver": "Average Distribution divergence",
        "coverage": "Catalog Coverage",
        "gini": "Gini index",
        "div": "Diversity (ILS)",
    }

    metric_key_map = {
        "mace": "maces",
        "diver": "divergences",
        "coverage": "coverages",
        "gini": "ginis",
        "div": "diversities",
    }

    nice_name = metric_name_to_nice_name[metric_name.lower()]

    metrics, labels, steps = [], [], []
    for idx, exp_file in enumerate(experiment_files):
        config = read_experiment(exp_file)
        all_results = read_metrics(config, should_remove_outliers)
        
        n_trials = config["n_trials"]
        metric_key = metric_key_map[metric_name.lower()]
        trial_metrics = [all_results[seed][metric_key] for seed in SEEDS[:n_trials]]
        metric_raw = np.mean(trial_metrics, axis=0)
        
        metrics.append(metric_raw)
        if experiment_aliases is not None:
            label = str(experiment_aliases[idx])
        else:
            label = config["model"]
            label = label.replace("_", " ")
            label = " ".join(s.title() for s in label.split(" "))
            if config.get("calibrate") is not None:
                label += f" (Calibration: {config['calibrate']})"
        labels.append(label)
        L = config["num_rounds_per_eval"]
        should_plot_retrain = (
            plot_retrain_points and config["model"] not in NON_RETRAIN_MODELS
        )
        steps.append(L if should_plot_retrain else None)
    plt.rcParams.update({"font.size": 8})  # match paper font size
    if graph == "box":
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plot_metric_boxplot_overlay(ax, metrics, labels, nice_name, title)
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plot_metrics_smoothed_overlay(
            ax=ax,
            metrics=metrics,
            model_names=labels,
            metric_name=nice_name,
            window_size=window_size,
            steps=steps,
            title=title,
            y_label=y_label,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

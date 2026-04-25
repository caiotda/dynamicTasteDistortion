import pandas as pd
import matplotlib.pyplot as plt

from dynamicTasteDistortion.ioUtils import read_experiment, read_metrics
from dynamicTasteDistortion.scripts.metrics_utils import apply_rolling_avg


def plot_metrics_smoothed(ax, metric, model_name, metric_name, step, window_size=10):

    # if ax is None, create it
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    metric_ma = apply_rolling_avg(metric, window_size)

    ax.plot(metric_ma, label=f"{metric_name} (Moving average w = {window_size})")
    ax.plot(metric, alpha=0.3, label=metric_name)
    if step is not None:
        idx = range(0, len(metric), step)
        ax.scatter(
            idx,
            [metric[i] for i in idx],
            color="red",
            zorder=4,
            s=3,
            alpha=0.5,
            label="Model Retrain Point",
        )

    ax.set_title(f"{metric_name} over Rounds ({model_name} based Rec)")
    ax.set_xlabel("Simulation Round")
    ax.set_ylabel(metric_name)
    ax.legend()


def plot_metric_boxplot(ax, metric, model_name, metric_name):
    ax.boxplot(metric)
    ax.set_title(f"Distribution of {metric_name}\n({model_name} based Rec)")
    ax.set_ylabel(metric_name)
    ax.set_xticks([1])
    ax.set_xticklabels([model_name])


def plot_metric_comparison(
    baseline_exp_file,
    comparison_exp_file,
    metric_name,
    title,
    graph="line",  # "line" or "box"
    window_size=10,
    plot_retrain_points=False,
    should_remove_outliers=False,
):
    metric_map = {
        "mace": 0,
        "kl": 1,
        "map": 2,
        "coverage": 3,
        "mrr": 4,
        "gini": 5,
        "div": 6,
    }

    metric_name_to_nice_name = {
        "mace": "Mean Average Calibration Error",
        "kl": "Average KL Divergence",
        "map": "Mean Average Precision",
        "coverage": "Catalog Coverage",
        "mrr": "Mean Reciprocal Rank",
        "gini": "Gini index",
        "div": "Diversity (ILS)",
    }

    metric_index = metric_map[metric_name.lower()]
    nice_name = metric_name_to_nice_name[metric_name.lower()]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Baseline
    baseline_config = read_experiment(baseline_exp_file)
    L = baseline_config["num_rounds_per_eval"]
    baseline_metric = read_metrics(baseline_config, should_remove_outliers)[
        metric_index
    ]

    # Comparison
    comparison_config = read_experiment(comparison_exp_file)
    comparison_metric = read_metrics(comparison_config, should_remove_outliers)[
        metric_index
    ]

    if graph == "box":
        plot_metric_boxplot(
            ax=axes[0],
            metric=baseline_metric,
            model_name=baseline_config["model"],
            metric_name=nice_name,
        )
        plot_metric_boxplot(
            ax=axes[1],
            metric=comparison_metric,
            model_name=comparison_config["model"],
            metric_name=nice_name,
        )
    else:
        plot_metrics_smoothed(
            ax=axes[0],
            metric=baseline_metric,
            model_name=baseline_config["model"],
            metric_name=nice_name,
            window_size=window_size,
            step=L if plot_retrain_points else None,
        )
        plot_metrics_smoothed(
            ax=axes[1],
            metric=comparison_metric,
            model_name=comparison_config["model"],
            metric_name=nice_name,
            window_size=window_size,
            step=None,
        )
    if should_remove_outliers:
        title += " (Outliers Removed)"
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(title)
    plt.show()

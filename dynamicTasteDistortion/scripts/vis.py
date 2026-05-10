import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from dynamicTasteDistortion.ioUtils import read_experiment, read_metrics
from dynamicTasteDistortion.scripts.metrics_utils import apply_rolling_avg


def plot_metrics_smoothed_overlay(
    ax, metrics, model_names, metric_name, window_size, steps
):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["x", "^", "o", "s"]

    for i, (metric, model_name, step) in enumerate(zip(metrics, model_names, steps)):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        metric_ma = apply_rolling_avg(metric, window_size)

        ax.plot(metric_ma, color=color, label=f"{model_name}")
        ax.plot(metric, color=color, alpha=0.3, label=None)

        if step is not None:
            idx = range(0, len(metric), step)
            ax.scatter(
                idx,
                [metric[j] for j in idx],
                color=color,
                marker=marker,
                zorder=4,
                s=40,
                alpha=0.7,
                # label=f"{model_name} Retrain Points",
            )

    ax.set_title(f"{metric_name} over Rounds")
    ax.set_xlabel("Simulation Round")
    ax.set_ylabel(metric_name)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()


def plot_metric_boxplot(ax, metric, model_name, metric_name):
    ax.boxplot(metric)
    ax.set_title(f"Distribution of {metric_name}\n({model_name} based Rec)")
    ax.set_ylabel(metric_name)
    ax.set_xticks([1])
    ax.set_xticklabels([model_name])
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
):
    metric_map = {
        "mace": 0,
        "diver": 1,
        "map": 2,
        "coverage": 3,
        "mrr": 4,
        "gini": 5,
        "div": 6,
    }
    metric_name_to_nice_name = {
        "mace": "Mean Average Calibration Error",
        "diver": "Average Distribution divergence",
        "map": "Mean Average Precision",
        "coverage": "Catalog Coverage",
        "mrr": "Mean Reciprocal Rank",
        "gini": "Gini index",
        "div": "Diversity (ILS)",
    }

    metric_index = metric_map[metric_name.lower()]
    nice_name = metric_name_to_nice_name[metric_name.lower()]

    configs, metrics, labels, steps = [], [], [], []
    for exp_file in experiment_files:
        config = read_experiment(exp_file)
        configs.append(config)
        metrics.append(read_metrics(config, should_remove_outliers)[metric_index])
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

    if graph == "box":
        fig, axes = plt.subplots(
            1, len(experiment_files), figsize=(6 * len(experiment_files), 6)
        )
        if len(experiment_files) == 1:
            axes = [axes]  # ensure iterable
        for ax, metric, label in zip(axes, metrics, labels):
            plot_metric_boxplot(ax, metric, label, nice_name)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        plot_metrics_smoothed_overlay(
            ax=ax,
            metrics=metrics,
            model_names=labels,
            metric_name=nice_name,
            window_size=window_size,
            steps=steps,
        )

    if should_remove_outliers:
        title += " (Outliers Removed)"

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

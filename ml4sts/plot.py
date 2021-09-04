# Imports: standard library
import os
import logging

# Imports: third party
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# Imports: ml4sts
# Imports: first party
from ml4sts.utils import get_full_feature_name
from ml4sts.definitions import IMAGE_EXT, MODEL_NAMES_FULL

matplotlib.use("Agg")


FONTSIZE_SMALL = 12
FONTSIZE_MEDIUM = 18
plt.rc("font", size=FONTSIZE_MEDIUM)  # controls default text sizes
plt.rc("axes", titlesize=FONTSIZE_MEDIUM)  # fontsize of the axes title
plt.rc("axes", labelsize=FONTSIZE_MEDIUM)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=FONTSIZE_MEDIUM)  # fontsize of the tick labels
plt.rc("ytick", labelsize=FONTSIZE_MEDIUM)  # fontsize of the tick labels
plt.rc("legend", fontsize=FONTSIZE_SMALL)  # legend fontsize
plt.rc("figure", titlesize=FONTSIZE_MEDIUM)  # fontsize of the figure title


def plot_feature_coefficients(
    args,
    model_name: str,
    feature_values: pd.DataFrame,
):
    sns.set(style="white", palette="muted", color_codes=True)
    sns.set_context("talk")

    # Isolate subset of top features for plotting
    if args.top_features_to_plot < len(feature_values):
        feature_values = feature_values.iloc[: args.top_features_to_plot]

    # Calculate length of strings to enable dynamic resizing of figure
    length_longest_feature_string = max(feature_values["feature"].str.len())
    fig_width = 4 + length_longest_feature_string / 10
    fig_height = 2 + args.top_features_to_plot * 0.35

    plt.figure(
        num=None,
        figsize=(fig_width, fig_height),
        dpi=150,
        facecolor="w",
        edgecolor="k",
    )
    y_pos = np.arange(feature_values.shape[0])
    plt.barh(y_pos, feature_values.iloc[:, 1], align="center", alpha=0.75)
    plt.yticks(y_pos, feature_values["feature"])
    plt.xlabel("Coefficient")

    plt.title(f"Feature coefficient: {MODEL_NAMES_FULL[model_name]}")

    # Flip axis so highest feature_values are plotted on top of figure
    axes = plt.gca()
    axes.invert_yaxis()

    # Remove ticks on y-axis
    axes.tick_params(axis="both", which="both", length=0)

    # Remove top, right, and left border
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["left"].set_visible(False)

    # Auto-adjust layout and whitespace
    plt.tight_layout()

    fpath = os.path.join(
        args.output_directory,
        args.id,
        f"feature_coefficients_{model_name}{IMAGE_EXT}",
    )
    plt.savefig(fpath, bbox_inches="tight", pad_inches=0.01)
    plt.close()
    logging.info(f"Saved {fpath}")


def plot_calibration_curve(
    args,
    model_name: str,
    calibration_method: str,
    fraction_of_positives: np.array,
    mean_predicted_value: np.array,
):
    sns.set(style="white", palette="muted", color_codes=True)
    sns.set_context("talk")

    plt.figure(
        num=None,
        figsize=(6.5, 6),
        dpi=150,
        facecolor="w",
        edgecolor="k",
    )
    axes = plt.axes()
    axes.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    axes.plot(
        mean_predicted_value,
        fraction_of_positives,
        "s-",
        linewidth=2,
        label=calibration_method,
    )

    axes.set_xlabel("Predicted")
    axes.set_ylabel("Observed")

    axes.set_title(f"{MODEL_NAMES_FULL[model_name]}")

    if args.calibration_curve_log_scale:
        axes.set_xscale("log")
        axes.set_yscale("log")
    else:
        # Set axes to upper limit
        axes_bound = 0.4
        axes.set_xlim([0, axes_bound])
        axes.set_ylim([0, axes_bound])

        # Set ticks
        num_ticks = int(axes_bound * 10 + 1)
        axes.set_xticks(np.linspace(0, axes_bound, num_ticks))
        axes.set_yticks(np.linspace(0, axes_bound, num_ticks))

    axes.grid()
    # plt.tight_layout()

    fname = f"calibration_curve_{model_name}_{calibration_method}{IMAGE_EXT}"
    fpath = os.path.join(args.output_directory, args.id, fname)
    plt.savefig(fpath, bbox_inches="tight", pad_inches=0.01)
    plt.close()
    logging.info(f"Saved plot of calibration curve to {fpath}")


def plot_classifier_curve(
    args,
    x_axis_percentiles: dict,
    x_axis_values: np.array,
    y_axis_percentiles: dict,
    auc_percentiles: dict,
    auc_title: str,
    x_label: str,
    y_label: str,
    title: str,
    file_name: str,
    legend_location: str = "lower right",
    plot_diagonal: bool = False,
):
    sns.set(style="white", palette="muted", color_codes=True)
    sns.set_context("talk")

    matplotlib.rcParams["xtick.major.pad"] = 8
    matplotlib.rcParams["ytick.major.pad"] = 8
    plt.figure(num=None, figsize=(6, 6), dpi=150, facecolor="w", edgecolor="k")
    axes = plt.axes()

    # Plot red dashed line representing random chance
    if plot_diagonal:
        axes.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", alpha=0.8)

    if x_axis_values is not None:
        x = x_axis_values
    elif x_axis_percentiles is not None:
        x = x_axis_percentiles["median"]
    else:
        x = np.linspace(0, 1, 100)

    label_text = f"""Median {auc_title} = {auc_percentiles["median"]:0.2f}"""
    axes.plot(
        x,
        y_axis_percentiles["median"],
        color="b",
        label=label_text,
        linewidth=2,
        alpha=0.8,
    )

    axes.fill_between(
        x,
        y_axis_percentiles["lower"],
        y_axis_percentiles["upper"],
        color="grey",
        alpha=0.2,
        label=(
            f"""95% CI = [{auc_percentiles["lower"]:0.2f}"""
            f""" - {auc_percentiles["upper"]:0.2f}]"""
        ),
    )

    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])
    axes.legend(loc=legend_location, frameon=False)
    axes.set_title(title)
    axes.grid()

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.yticks()
    plt.xticks()

    fpath = os.path.join(
        args.output_directory,
        args.id,
        file_name,
    )
    plt.savefig(fpath, bbox_inches="tight", pad_inches=0.01)
    plt.close()
    logging.info(f"Saved classifier curve plot to {fpath}")


def plot_sensitivity_analysis_auc(
    axes,
    df: pd.DataFrame,
    model_name: str,
    xticks: list,
):
    axes.plot(
        df["scale_factor"],
        df["median"],
        label=f"{model_name}",
        alpha=0.85,
    )
    axes.fill_between(
        df["scale_factor"],
        df["lower"],
        df["upper"],
        alpha=0.2,
    )
    plt.xticks(xticks)
    axes.set_xscale("log")
    axes.set_xticks(xticks)
    axes.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes.set_xlabel("Scale factor")
    axes.margins(x=0)
    axes.set_ylim(0.65, 0.9)
    axes.set_ylabel("AUC")
    axes.legend(loc="lower left")


def plot_distributions_feature_y_hat(
    args,
    feature_values: np.array,
    y_hat: dict,
    model_name: str,
    feature_name: str,
):
    sns.set(style="white", palette="muted")
    sns.set_context("talk")
    _fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Plot feature distribution
    sns.distplot(
        feature_values,
        hist=True,
        kde=False,
        ax=axes[0],
        hist_kws={"alpha": 1, "linewidth": 3, "edgecolor": "dimgray"},
    )
    sns.despine()

    full_feature_name = get_full_feature_name(feature_name)
    axes[0].set_title(f"Distribution of {full_feature_name} (n={len(feature_values)})")
    axes[0].set_xlim([0, np.max(feature_values)])
    axes[0].set_xlabel("Time (minutes)")
    axes[0].set_ylabel("Count")

    # Draw vertical lines for mean and median values
    mean = np.mean(feature_values)
    axes[0].axvline(mean, linestyle="-", color="black", zorder=5, linewidth=3)
    median = np.median(feature_values)
    axes[0].axvline(
        median,
        linestyle="--",
        color="black",
        zorder=5,
        linewidth=3,
    )
    axes[0].legend({"Mean": mean, "Median": median})

    df = pd.DataFrame(y_hat).melt()

    # Plot predicted probabilities y_hat
    sns.violinplot(
        data=df,
        x="variable",
        y="value",
        ax=axes[1],
        cut=0,
        linewidth=3,
    )
    sns.despine()
    axes[1].set_ylim([0, 1])
    axes[1].set_title(f"Predicted probability of operative mortality: {model_name}")
    axes[1].set_xlabel(f"Percentile of {full_feature_name}")
    axes[1].set_ylabel("Predicted probability")

    # Get 95th percentile of predicted probability for entire plot
    upper = np.percentile(df.value, 95)

    # Round to a reasonable upper bound for ylim
    ylim_upper = np.around(upper + 0.1, 1)
    ylim_upper = 1 if ylim_upper > 1 else ylim_upper
    ylim_upper = 0.3 if ylim_upper < 0.3 else ylim_upper

    axes[1].set_ylim([0, ylim_upper])

    fpath = os.path.join(
        args.output_directory,
        args.id,
        f"sensitivity_analysis_y_hat_{feature_name}_{model_name}{IMAGE_EXT}",
    )
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()
    logging.info(f"Saved {fpath}")


def plot_bootstrap_differences_of_metric(
    axes,
    diff: np.array,
    metric: str,
    pvalue: float,
):
    # Plot distribution
    plot = sns.distplot(diff, ax=axes)

    # Set vertical line at x=0
    axes.axvline(0)

    # Set title
    plot.set_title(f"{metric} differences (p={pvalue})")

    # Remove y-axis ticks
    axes.set_yticks([])


def plot_ffs_aucs(
    df: pd.DataFrame,
    fpath: str,
):
    # Increment column names by one, to shift from indices to number of features
    df = df.rename(columns=lambda x: x + 1)

    plt.figure(
        num=None,
        figsize=(df.shape[1] * 1.25, 6),
        dpi=150,
        facecolor="w",
        edgecolor="k",
    )
    axes = plt.axes()
    sns.set(style="whitegrid", palette="muted", color_codes=True)
    sns.set_context("talk")
    sns.barplot(data=df, ci="sd", color="cornflowerblue")
    axes.set_ylim([0.5, 1.0])
    plt.xlabel("Number of top features")
    plt.ylabel("AUC")
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()
    logging.info(f"Saved {fpath}")

# Imports: standard library
import os
import logging
import multiprocessing as mp
from typing import Dict, Tuple

# Imports: third party
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.metrics import (
    auc,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

# Imports: ml4sts
# Imports: first party
from ml4sts.plot import plot_bootstrap_differences_of_metric
from ml4sts.utils import create_shared_array
from ml4sts.models import evaluate_predictions
from ml4sts.definitions import CI_PERCENTILES


def resample_precision_recall(
    precisions: np.array,
    recalls: np.array,
    num_elements: float = 100,
) -> Tuple[np.array, np.array]:

    r_precisions = []
    r_mean_recall = np.linspace(0, 1, num_elements)

    for precision, recall in zip(precisions, recalls):
        r_recall = np.flip(recall)
        r_precision = np.flip(precision)
        r_interp_precision = np.interp(r_mean_recall, r_recall, r_precision)
        r_interp_precision[0] = 1
        r_precisions.append(r_interp_precision)

    r_precisions = np.array(r_precisions)
    precisions_resampled = np.flip(r_precisions)
    mean_recall = np.flip(r_mean_recall)
    return precisions_resampled, mean_recall


def _sample_bootstrap(
    y: np.array,
    y_hat: np.array,
    y_hat_calibrated: np.array,
) -> Tuple[np.array, np.array, np.array]:
    """
    Perform one bootstrap sampling on y, y_hat, and y_hat_calibrated
    and return the sampled distributions.
    """
    if y.shape != y_hat.shape != y_hat_calibrated.shape:
        raise ValueError(
            "y, y_hat, and y_hat_bootstrap must be arrays with identical shapes",
        )
    idx = np.random.choice(len(y), len(y), replace=True)
    return y[idx], y_hat[idx], y_hat_calibrated[idx]


def generate_bootstrap_distributions(
    seed: int,
    y: np.array,
    y_hat: np.array,
    y_hat_calibrated: np.array,
    bootstrap_samplings: int,
):
    # Reset random seed to ensure the same bootstrap samplings each call
    np.random.seed(seed)

    y_bootstrap = []
    y_hat_bootstrap = []
    y_hat_bootstrap_calibrated = []
    for _sampling in range(bootstrap_samplings):
        y_i, y_hat_i, y_hat_calibrated_i = _sample_bootstrap(
            y=y,
            y_hat=y_hat,
            y_hat_calibrated=y_hat_calibrated,
        )
        y_bootstrap.append(y_i)
        y_hat_bootstrap.append(y_hat_i)
        y_hat_bootstrap_calibrated.append(y_hat_calibrated_i)
    return (
        np.array(y_bootstrap),
        np.array(y_hat_bootstrap),
        np.array(y_hat_bootstrap_calibrated),
    )


def get_bootstrap_metrics(
    y_bootstrap: np.array,
    y_hat_bootstrap: np.array,
):
    percents = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    precisions = []
    recalls = []
    average_precisions = []

    for sampling in range(y_bootstrap.shape[0]):

        # Calculate FPR and TPR for ROC curve
        fpr_i, tpr_i, _tt = roc_curve(
            y_bootstrap[sampling, :],
            y_hat_bootstrap[sampling, :],
        )
        interp_tpr = np.interp(percents, fpr_i, tpr_i)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr_i, tpr_i))

        # Calculate precision and recall arrays for PR curve
        precision_i, recall_i, _ = precision_recall_curve(
            y_bootstrap[sampling, :],
            y_hat_bootstrap[sampling, :],
        )
        average_precision_i = average_precision_score(
            y_bootstrap[sampling, :],
            y_hat_bootstrap[sampling, :],
        )
        precisions.append(precision_i)
        recalls.append(recall_i)
        average_precisions.append(average_precision_i)

    # Turn lists of arrays into arrays
    tprs = np.array(tprs)
    aucs = np.array(aucs)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    average_precisions = np.array(average_precisions)

    return tprs, aucs, precisions, recalls, average_precisions


def calculate_percentiles(arr: np.array) -> dict:
    arr_percentiles = {}
    (
        arr_percentiles["lower"],
        arr_percentiles["median"],
        arr_percentiles["upper"],
    ) = np.percentile(
        arr,
        (CI_PERCENTILES["lower"], CI_PERCENTILES["median"], CI_PERCENTILES["upper"]),
        axis=0,
    )
    return arr_percentiles


def _calculate_bootstrap_pvalue(values: np.array) -> float:
    # Check edge case where diffs are all zeroes
    if not np.any(values):
        return 1.0

    observed_mean = np.mean(values)

    if observed_mean >= 0:
        num_extreme_observations = sum(values < 0)
    else:
        num_extreme_observations = sum(values > 0)

    # If mean diff is very close to zero, the number of extreme observations
    # can exceed 50% due to random chance, which would result in p > 1.0.
    # To mitigate, set number of extreme observations to the maximum possible
    if num_extreme_observations >= len(values) / 2:
        return 0.999

    pvalue = 2 * num_extreme_observations / len(values)

    return pvalue


def _list_of_dicts_into_dict_of_arrays(list_of_dicts: list) -> dict:
    return {
        key: np.array([entry[key] for entry in list_of_dicts])
        for key in list_of_dicts[0]
    }


# Global dictionary stores variables passed from initializer
var_dict = {}


def _init_globals_bootstrap(
    y_baseline: np.array,
    y_baseline_shape: tuple,
    y_compare: np.array,
    y_compare_shape: tuple,
    y_hat_compare: np.array,
    y_hat_compare_shape: tuple,
    y_hat_compare_calibrated: np.array,
    y_hat_compare_calibrated_shape: tuple,
    y_hat_baseline: np.array,
    y_hat_baseline_shape: tuple,
    y_hat_baseline_calibrated: np.array,
    y_hat_baseline_calibrated_shape: tuple,
):
    var_dict["y_baseline"] = y_baseline
    var_dict["y_baseline_shape"] = y_baseline_shape
    var_dict["y_compare"] = y_compare
    var_dict["y_compare_shape"] = y_compare_shape
    var_dict["y_hat_compare"] = y_hat_compare
    var_dict["y_hat_compare_shape"] = y_hat_compare_shape
    var_dict["y_hat_compare_calibrated"] = y_hat_compare_calibrated
    var_dict["y_hat_compare_calibrated_shape"] = y_hat_compare_calibrated_shape
    var_dict["y_hat_baseline"] = y_hat_baseline
    var_dict["y_hat_baseline_shape"] = y_hat_baseline_shape
    var_dict["y_hat_baseline_calibrated"] = y_hat_baseline_calibrated
    var_dict["y_hat_baseline_calibrated_shape"] = y_hat_baseline_calibrated_shape


def _evaluate_predictions_parallel(i: int) -> tuple:
    """
    Get the arrays from shared memory buffer
    """
    y_baseline = np.frombuffer(var_dict["y_baseline"]).reshape(
        var_dict["y_baseline_shape"],
    )
    y_compare = np.frombuffer(var_dict["y_compare"]).reshape(
        var_dict["y_compare_shape"],
    )
    y_hat_baseline = np.frombuffer(var_dict["y_hat_baseline"]).reshape(
        var_dict["y_hat_baseline_shape"],
    )
    y_hat_baseline_calibrated = np.frombuffer(
        var_dict["y_hat_baseline_calibrated"],
    ).reshape(
        var_dict["y_hat_baseline_calibrated_shape"],
    )
    y_hat_compare = np.frombuffer(var_dict["y_hat_compare"]).reshape(
        var_dict["y_hat_compare_shape"],
    )
    y_hat_compare_calibrated = np.frombuffer(
        var_dict["y_hat_compare_calibrated"],
    ).reshape(
        var_dict["y_hat_compare_calibrated_shape"],
    )

    # Calculate metrics
    metrics_baseline = evaluate_predictions(
        y=y_baseline[i, :],
        y_hat=y_hat_baseline[i, :],
    )
    metrics_baseline_calibrated = evaluate_predictions(
        y=y_baseline[i, :],
        y_hat=y_hat_baseline_calibrated[i, :],
    )
    metrics_compare = evaluate_predictions(
        y=y_compare[i, :],
        y_hat=y_hat_compare[i, :],
    )
    metrics_compare_calibrated = evaluate_predictions(
        y=y_compare[i, :],
        y_hat=y_hat_compare_calibrated[i, :],
    )
    return (
        metrics_baseline,
        metrics_baseline_calibrated,
        metrics_compare,
        metrics_compare_calibrated,
    )


def compare_bootstrap_metrics(
    args,
    y_baseline: np.array,
    y_compare: np.array,
    y_hat_baseline: np.array,
    y_hat_baseline_calibrated: np.array,
    y_hat_compare: np.array,
    y_hat_compare_calibrated: np.array,
    prefix_str: str,
    model_name_baseline: str,
    model_name_compare: str,
) -> dict:
    bootstrap_samplings = y_hat_compare.shape[0]

    # Convert numpy arrays into shared raw arrays for multiprocess
    y_baseline, y_baseline_shape = create_shared_array(y_baseline)
    y_compare, y_compare_shape = create_shared_array(y_compare)
    y_hat_compare, y_hat_compare_shape = create_shared_array(y_hat_compare)
    y_hat_compare_calibrated, y_hat_compare_calibrated_shape = create_shared_array(
        y_hat_compare_calibrated,
    )
    y_hat_baseline, y_hat_baseline_shape = create_shared_array(y_hat_baseline)
    y_hat_baseline_calibrated, y_hat_baseline_calibrated_shape = create_shared_array(
        y_hat_baseline_calibrated,
    )

    logging.info(
        f"Assessing metrics of {model_name_compare} vs {model_name_baseline}"
        f" via {bootstrap_samplings} bootstrap samplings",
    )

    # Compare predicted labels from two models via bootstrap sampling,
    # and save metrics as list of dicts
    with mp.Pool(
        processes=None,
        initializer=_init_globals_bootstrap,
        initargs=(
            y_baseline,
            y_baseline_shape,
            y_compare,
            y_compare_shape,
            y_hat_compare,
            y_hat_compare_shape,
            y_hat_compare_calibrated,
            y_hat_compare_calibrated_shape,
            y_hat_baseline,
            y_hat_baseline_shape,
            y_hat_baseline_calibrated,
            y_hat_baseline_calibrated_shape,
        ),
    ) as pool:
        # Get list of tuples
        metrics_list_of_dicts = pool.map(
            _evaluate_predictions_parallel,
            range(bootstrap_samplings),
        )

    # Convert list of tuples into list of dicts
    (
        metrics_list_of_dicts_baseline,
        metrics_list_of_dicts_baseline_calibrated,
        metrics_list_of_dicts_compare,
        metrics_list_of_dicts_compare_calibrated,
    ) = map(
        list,
        zip(*metrics_list_of_dicts),
    )

    # Convert list of dicts into dict of arrays
    metrics_baseline = _list_of_dicts_into_dict_of_arrays(
        list_of_dicts=metrics_list_of_dicts_baseline,
    )
    metrics_baseline_calibrated = _list_of_dicts_into_dict_of_arrays(
        list_of_dicts=metrics_list_of_dicts_baseline_calibrated,
    )
    metrics_compare = _list_of_dicts_into_dict_of_arrays(
        list_of_dicts=metrics_list_of_dicts_compare,
    )
    metrics_compare_calibrated = _list_of_dicts_into_dict_of_arrays(
        list_of_dicts=metrics_list_of_dicts_compare_calibrated,
    )

    # Add the calibrated Brier score to metrics_compare
    metrics_baseline["brier_calibrated"] = metrics_baseline_calibrated["brier"]
    metrics_compare["brier_calibrated"] = metrics_compare_calibrated["brier"]

    # Free memory
    del metrics_baseline_calibrated
    del metrics_compare_calibrated

    # Initialize dict
    metrics_bootstrap: Dict[str, dict] = {}

    # Format string for distribution plots
    if prefix_str:
        prefix_str = f"{prefix_str}_"

    # If model being assessed is not baseline, prepare the plot of diffs
    if model_name_compare != model_name_baseline:
        sns.set(style="white", palette="muted", color_codes=True)
        fig, axes = plt.subplots(
            len(metrics_compare),
            figsize=(8, len(metrics_compare) * 3),
            sharex=False,
        )
        sns.despine(left=True)

    # Iterate through metrics, and for each calculate p-value of differences
    # and also calculate percentiles to save in dict
    for i, metric in enumerate(sorted(metrics_compare)):

        if metric not in metrics_bootstrap:
            metrics_bootstrap[metric] = {}

        # Calculate percentiles of metric
        (
            metrics_bootstrap[metric]["lower"],
            metrics_bootstrap[metric]["median"],
            metrics_bootstrap[metric]["upper"],
        ) = np.percentile(
            metrics_compare[metric],
            (
                CI_PERCENTILES["lower"],
                CI_PERCENTILES["median"],
                CI_PERCENTILES["upper"],
            ),
            axis=0,
        )

        # Plot distribution of diffs
        if model_name_compare != model_name_baseline:
            diff = metrics_compare[metric] - metrics_baseline[metric]

            # Compute p-value from array of differences
            metrics_bootstrap[metric]["pvalue"] = _calculate_bootstrap_pvalue(
                values=diff,
            )

            plot_bootstrap_differences_of_metric(
                axes=axes[i],
                diff=diff,
                metric=metric,
                pvalue=metrics_bootstrap[metric]["pvalue"],
            )

    # If the model being assessed is not baseline, compare against baseline
    if model_name_compare != model_name_baseline:
        fpath = os.path.join(
            args.output_directory,
            args.id,
            f"metric_diffs_{prefix_str}{model_name_compare}_vs_{model_name_baseline}.png",
        )
        fig.tight_layout()
        fig.savefig(fpath, dpi=150)
        fig.clf()
        logging.info(f"Saved {fpath}")

    return metrics_bootstrap


def format_bootstrap_metrics_to_dataframe(
    metrics: dict,
    decimals: int,
) -> pd.DataFrame:
    """
    Format a dictionary of metrics into a dataframe for easier logging and printing.
    """
    # Convert dictionary to dataframe
    df = pd.DataFrame(metrics).transpose()

    # Reorder columns and round columns (except pvalue)
    if "pvalue" in df:
        df = pd.concat(
            [df[["median", "lower", "upper"]], df["pvalue"]],
            axis=1,
        )
        df["pvalue"] = df["pvalue"].round(4)
        df.loc[:, df.columns != "pvalue"] = df.loc[:, df.columns != "pvalue"].round(
            decimals,
        )
    else:
        df = df[["median", "lower", "upper"]]
        df = df.round(decimals)
    df.sort_index(inplace=True)
    return df

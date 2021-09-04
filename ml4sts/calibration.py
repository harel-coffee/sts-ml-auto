# Imports: standard library
import logging

# Imports: third party
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

# Imports: ml4sts
# Imports: first party
from ml4sts.plot import plot_calibration_curve
from ml4sts.utils import log_dataframe, save_dataframe_to_csv


def train_calibrator(y: np.array, y_hat: np.array):
    calibration_model = LogisticRegression()
    calibration_model.fit(y_hat.reshape(-1, 1), y)
    return calibration_model


def calibrate_probabilities(calibration_model, y_hat: np.array) -> np.array:
    return calibration_model.predict_proba(y_hat.reshape(-1, 1))[:, 1]


def threshold_probabilities(y_hat: np.array, threshold: float) -> np.array:
    y_hat[y_hat > threshold] = threshold
    return y_hat


def calibration_metrics_and_curves(
    args,
    model_name: str,
    y: np.array,
    y_hat: np.array,
    y_hat_calibrated: np.array,
):
    """
    Given a model name, true labels, predictions, and calibrated predictions:
    - Calculate Brier score, calibration slope, and O/E ratio
    - Save metrics to CSV
    - Plot calibration curves
    """

    # Initialize dicts in which to save binned probabilities and metrics
    fraction_of_positives = {}
    mean_predicted_value = {}
    brier_score = {}
    calibration_slope = {}
    expected_to_observed = {}

    # Iterate through calibration methods and y_hat arrays
    for (calibration_method, y_hat) in zip(
        ["original", "platt_scaling"],
        [y_hat, y_hat_calibrated],
    ):

        # Bin predicted probabilities
        (
            fraction_of_positives[calibration_method],
            mean_predicted_value[calibration_method],
        ) = calibration_curve(y_true=y, y_prob=y_hat, n_bins=10, strategy="quantile")

        # Brier score
        brier_score[calibration_method] = brier_score_loss(y_true=y, y_prob=y_hat)

        # Calibration slope
        calibration_slope[calibration_method] = np.polyfit(
            x=mean_predicted_value[calibration_method],
            y=fraction_of_positives[calibration_method],
            deg=1,
        )[0]

        # E/O ratio
        expected_to_observed[calibration_method] = sum(y_hat) / sum(y)

        # Plot calibration curve of best method
        plot_calibration_curve(
            args=args,
            model_name=model_name,
            calibration_method=calibration_method,
            fraction_of_positives=fraction_of_positives[calibration_method],
            mean_predicted_value=mean_predicted_value[calibration_method],
        )

    # Convert calibration metrics to dataframe, round, and transpose
    df_calibration_metrics = (
        pd.DataFrame(
            [brier_score, calibration_slope, expected_to_observed],
            index=["brier_score", "calibration_slope", "expected_to_observed"],
        )
        .round(args.decimals)
        .transpose()
    )

    # Log calibration metrics
    logging.info(f"Calibration metrics for {model_name}")
    log_dataframe(
        df=df_calibration_metrics,
        format_scientific=False,
    )

    # Save calibration metrics to CSV
    save_dataframe_to_csv(
        args=args,
        df=df_calibration_metrics,
        fname=f"calibration_metrics_{model_name}",
        keep_index=True,
    )

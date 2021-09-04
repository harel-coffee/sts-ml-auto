# Imports: standard library
import os
import logging
from collections import Counter, defaultdict

# Imports: third party
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import ignore_warnings

# Imports: ml4sts
from ml4sts.plot import plot_ffs_aucs, plot_classifier_curve
from ml4sts.utils import (
    log_dataframe,
    select_cohort,
    save_results_to_disk,
    save_dataframe_to_csv,
    generate_crossfold_indices,
    get_continuous_features_mask,
    format_categorical_and_binary_features,
)
from ml4sts.models import initialize_model
from ml4sts.bootstrap import (
    calculate_percentiles,
    get_bootstrap_metrics,
    compare_bootstrap_metrics,
    resample_precision_recall,
    generate_bootstrap_distributions,
    format_bootstrap_metrics_to_dataframe,
)
from ml4sts.calibration import (
    train_calibrator,
    calibrate_probabilities,
    calibration_metrics_and_curves,
)
from ml4sts.definitions import IMAGE_EXT, OUTCOME_MAP, MODEL_NAMES_FULL
from ml4sts.hyperparameters import generate_hyperparameters


@ignore_warnings(category=ConvergenceWarning)
def ffs(args):
    num_iterations_without_improvement = 0

    # Load data from dataframe
    df = pd.read_csv(args.path_to_csv, low_memory=False)

    # Isolate desired case types
    if args.cohort:
        df = select_cohort(df=df, cohort=args.cohort)

    # Isolate features from dataframe
    df_x = df[args.features].copy()
    df_x = format_categorical_and_binary_features(df=df_x)

    # Get mask of continuous features
    continuous_features_mask = get_continuous_features_mask(df=df_x)

    # Get feature names
    feature_names = np.array(df_x.columns)

    # Get features and labels as numpy arrays
    x = df_x.values
    y = df[OUTCOME_MAP[args.outcome]].values

    n = x.shape[0]
    m = x.shape[1]
    deaths = sum(y)
    deaths_frac = sum(y) / n
    logging.info(
        f"Loaded data from {args.path_to_csv} containing "
        f"{n} patients, {deaths} deaths ({deaths_frac:0.2f}%), {m} features",
    )

    # FFS only does logistic regression
    model_name = "logreg"

    # Generate single combo of hyperparameters for each model
    hp = {}
    hp[model_name] = generate_hyperparameters(
        seed=args.seed,
        model_name=model_name,
        random_samplings=1,
        smote=args.smote,
    )

    # Reset random seed to a set value so the number of times you use random
    # number generators in generate_hyperparameters does not affect
    # reproducibility of downstream crossfold generation
    np.random.seed(args.seed)

    # Generate cross-fold indices
    skf, idx_outer_trains, idx_outer_tests = generate_crossfold_indices(
        crossfolds=args.crossfolds,
        seed=args.seed,
        y=y,
    )

    model_name = "logreg"
    model = initialize_model(model_name=model_name, hp=hp[model_name][0])

    # Perform forward feature selection using logistic regression
    top_features_idx = []
    num_top_features_asymptote = 0
    y_hat_best = []
    aucs_best = []
    results = defaultdict()
    y_hat_vs_n_top_features = defaultdict()

    # Iterate through number of top features until convergence
    while len(top_features_idx) < args.ffs_top_features:

        # If this is the first run, remaining features == all feature indices
        if len(top_features_idx) == 0:
            remaining_features_idx = np.arange(0, x.shape[1])

        logging.info(
            f"Performing FFS with {len(top_features_idx) + 1} feature(s);"
            f" {len(remaining_features_idx)} remaining to choose from",
        )
        best_auc_mean = 0
        best_auc_std = 0
        best_feature_that_was_added_idx = None
        aucs_best_for_i_top_features = None
        y_test_all = []

        for i, feature_to_add_to_list_idx in enumerate(remaining_features_idx):
            features_to_try_idx = np.array(
                top_features_idx + [feature_to_add_to_list_idx],
            )
            y_hat_all_this_feature_set = []
            y_hat_calibrated_all_this_feature_set = []
            aucs_by_crossfold = []

            for crossfold in range(args.crossfolds):
                x_train = x[
                    idx_outer_trains[crossfold].reshape(-1, 1),
                    features_to_try_idx,
                ]
                x_test = x[
                    idx_outer_tests[crossfold].reshape(-1, 1),
                    features_to_try_idx,
                ]
                y_train = y[idx_outer_trains[crossfold]]
                y_test = y[idx_outer_tests[crossfold]]

                # Because y_test_all (concatenated across folds) is identical across
                # all feature combos, only assemble this list of arrays once
                if len(y_test_all) < args.crossfolds:
                    y_test_all.append(y_test)

                # Scale continuous features
                mask = continuous_features_mask[features_to_try_idx]

                # Special case: x_train is 1D array of continuous values, must reshape
                if len(features_to_try_idx) == 1:
                    x_train = x_train.reshape(-1, 1)
                    x_test = x_test.reshape(-1, 1)

                # If any features are continuous, scale features
                if np.any(mask):
                    scaler = StandardScaler().fit(x_train[:, mask])
                    x_train[:, mask] = scaler.transform(x_train[:, mask])

                # Train model and predict on test set
                model.fit(x_train, y_train)
                y_hat = model.predict_proba(x_test)[:, 1]
                y_hat_all_this_feature_set.append(y_hat)

                # Calibrate predictions using training data
                y_hat_train = model.predict_proba(x_train)[:, 1]
                calibration_model = train_calibrator(y=y_train, y_hat=y_hat_train)
                y_hat_calibrated = calibrate_probabilities(
                    calibration_model=calibration_model,
                    y_hat=y_hat,
                )
                y_hat_calibrated_all_this_feature_set.append(y_hat_calibrated)

                auc = roc_auc_score(y_true=y_test, y_score=y_hat)
                aucs_by_crossfold.append(auc)

            auc_mean = np.mean(aucs_by_crossfold)
            auc_std = np.std(aucs_by_crossfold)
            logging.info(
                f"Feature combo {i+1}/{len(remaining_features_idx)}:"
                f" AUC = {auc_mean:0.3f} ± {auc_std:0.3f}",
            )
            if auc_mean > best_auc_mean:
                best_auc_mean = auc_mean
                best_auc_std = auc_std
                best_feature_that_was_added_idx = feature_to_add_to_list_idx
                aucs_best_for_i_top_features = aucs_by_crossfold

        # Update top_features_idx and AUCs across folds
        top_features_idx.append(best_feature_that_was_added_idx)
        aucs_best.append(aucs_best_for_i_top_features)

        # Remove best feature index from remaining feature indices
        remaining_features_idx = np.delete(
            remaining_features_idx,
            np.where(remaining_features_idx == best_feature_that_was_added_idx),
        )

        num_top_features = len(top_features_idx)
        logging.info(
            f" === FFS with {num_top_features} top feature(s): AUC = {best_auc_mean:0.3f} ± {best_auc_std:0.3f} ===",
        )

        # Store results in dict keyed by number of top features in model
        results[num_top_features] = {}
        results[num_top_features]["auc_mean"] = best_auc_mean
        results[num_top_features]["auc_std"] = best_auc_std
        results[num_top_features]["features"] = feature_names[top_features_idx].tolist()

        # Check "patience", e.g., additional features to try in FFS after performance
        # stops improving.
        if num_top_features > 2:
            auc_current = round(results[num_top_features]["auc_mean"], 2)
            auc_prior = round(results[num_top_features - 1]["auc_mean"], 2)
            auc_diff = auc_current - auc_prior
            if auc_diff > args.ffs_auc_delta:
                num_iterations_without_improvement = 0
            else:
                num_iterations_without_improvement += 1
                logging.info(
                    f"{num_iterations_without_improvement} additional features used"
                    f" without {args.ffs_auc_delta} of AUC improvement",
                )
                if num_iterations_without_improvement >= args.ffs_patience:
                    num_top_features_asymptote = (
                        num_top_features - num_iterations_without_improvement
                    )
                    auc_best_asymptote = np.mean(
                        aucs_best[num_top_features_asymptote],
                    )
                    logging.info(
                        f"{num_iterations_without_improvement} additional features added"
                        f" without improvement. AUC asymptotes at {auc_best_asymptote:0.3f}"
                        f" using the top {num_top_features_asymptote} features. Stopping FFS",
                    )
                    break

    # Summarize and save AUCs and best features in CSV
    df_results = pd.DataFrame(results).T
    df_results.auc_mean = df_results.auc_mean.astype(float).round(3)
    df_results.auc_std = df_results.auc_std.astype(float).round(3)
    df_results.index.name = "number_features"
    fpath = os.path.join(args.output_directory, args.id, "ffs-results.csv")
    df_results.to_csv(fpath, index=True, header=True)

    # Create (f, n) dataframe of AUCs
    # f: number of crossfolds (rows)
    # n: number of top features (columns)
    df_aucs = pd.DataFrame(aucs_best).T

    # Generate bar plot of best AUCs vs number of top features
    fpath = os.path.join(args.output_directory, args.id, "ffs-aucs.png")
    plot_ffs_aucs(df=df_aucs, fpath=fpath)

    # Now we know the relationship between number of top features,
    # and cross-validated test set performance
    # Next, perform bootstrap sampling to assess the best model more closely

    # Isolate subset of best features
    x_top = x[:, top_features_idx[:num_top_features_asymptote]]
    mask = continuous_features_mask[features_to_try_idx[:num_top_features_asymptote]]

    y_test_all = []
    y_hat_all = []
    y_hat_all_calibrated = []
    for crossfold in range(args.crossfolds):
        x_train = x_top[idx_outer_trains[crossfold], :]
        x_test = x_top[idx_outer_tests[crossfold], :]
        y_train = y[idx_outer_trains[crossfold]]
        y_test = y[idx_outer_tests[crossfold]]

        y_test_all.append(y_test)

        # Scale test set using training distribution
        if np.any(mask):
            scaler = StandardScaler().fit(x_train[:, mask])
            x_test[:, mask] = scaler.transform(x_test[:, mask])

        # Train model and predict outcomes on test set
        model.fit(x_train, y_train)
        y_hat_test = model.predict_proba(x_test)[:, 1]
        y_hat_all.append(y_hat_test)

        # Train and save calibration model (regressor) on training data
        y_hat_train = model.predict_proba(x_train)[:, 1]
        calibration_model = train_calibrator(y=y_train, y_hat=y_hat_train)

        # Calibrate test set predictions
        y_hat_calibrated = calibrate_probabilities(
            calibration_model=calibration_model,
            y_hat=y_hat_test,
        )
        y_hat_all_calibrated.append(y_hat_calibrated)

    # Concatenate args.crossfolds lists into single arrays
    y_test_all = np.concatenate(y_test_all)
    y_hat_all = np.concatenate(y_hat_all)
    y_hat_all_calibrated = np.concatenate(y_hat_all_calibrated)

    # Calculate calibration metrics and plot curves
    calibration_metrics_and_curves(
        args=args,
        model_name=model_name,
        y=y_test_all,
        y_hat=y_hat_all,
        y_hat_calibrated=y_hat_all_calibrated,
    )

    # Generate bootstrap distributions
    (
        y_bootstrap,
        y_hat_bootstrap,
        y_hat_bootstrap_calibrated,
    ) = generate_bootstrap_distributions(
        seed=args.seed,
        y=y_test_all,
        y_hat=y_hat_all,
        y_hat_calibrated=y_hat_all_calibrated,
        bootstrap_samplings=args.bootstrap_samplings,
    )

    # Calculate tprs and aucs from bootstrap distributions
    tprs, aucs, precisions, recalls, average_precisions = get_bootstrap_metrics(
        y_bootstrap=y_bootstrap,
        y_hat_bootstrap=y_hat_bootstrap,
    )

    # Format precisions and recalls; the arrays within these arrays are not
    # guaranteed to have the same number of elements from the i'th to i+1'th
    # bootstrap sampling so we need to resample, interpolate, etc.
    # This overwrites precisions and recalls.
    precisions, recall_resampled = resample_precision_recall(
        precisions=precisions,
        recalls=recalls,
    )

    # Calculate percentiles
    tpr_percentiles = calculate_percentiles(arr=tprs)
    auc_percentiles = calculate_percentiles(arr=aucs)
    precision_percentiles = calculate_percentiles(arr=precisions)
    average_precision_percentiles = calculate_percentiles(arr=average_precisions)

    # Plot ROC curve
    plot_classifier_curve(
        args=args,
        x_axis_percentiles=None,
        x_axis_values=None,
        y_axis_percentiles=tpr_percentiles,
        auc_percentiles=auc_percentiles,
        auc_title="AUC",
        x_label="False positive rate",
        y_label="True positive rate",
        title=f"{MODEL_NAMES_FULL[model_name]}",
        file_name=f"curve-roc-{model_name}-{num_top_features_asymptote}-top-features{IMAGE_EXT}",
        legend_location="lower right",
        plot_diagonal=True,
    )

    # Plot PR curve
    plot_classifier_curve(
        args=args,
        x_axis_percentiles=None,
        x_axis_values=recall_resampled,
        y_axis_percentiles=precision_percentiles,
        auc_percentiles=average_precision_percentiles,
        auc_title="AUPRC",
        x_label="Recall",
        y_label="Precision",
        title=f"{MODEL_NAMES_FULL[model_name]}",
        file_name=f"curve-pr-{model_name}-{num_top_features_asymptote}-top-features{IMAGE_EXT}",
        legend_location="upper right",
        plot_diagonal=False,
    )

    # Calculate metrics
    metrics_bootstrap = compare_bootstrap_metrics(
        args=args,
        y_baseline=y_bootstrap,
        y_compare=y_bootstrap,
        y_hat_baseline=y_hat_bootstrap,
        y_hat_baseline_calibrated=y_hat_bootstrap_calibrated,
        y_hat_compare=y_hat_bootstrap,
        y_hat_compare_calibrated=y_hat_bootstrap_calibrated,
        prefix_str="",
        model_name_baseline=model_name,
        model_name_compare=model_name,
    )

    # Format metrics dicts into dataframe
    metrics_df = format_bootstrap_metrics_to_dataframe(
        metrics=metrics_bootstrap,
        decimals=args.decimals,
    )

    # Log metrics dataframe
    log_dataframe(
        df=metrics_df,
        format_scientific=False,
    )

    # Save metrics dataframe to CSV
    save_dataframe_to_csv(
        args=args,
        df=metrics_df,
        fname=f"metrics-logreg-ffs.csv",
        keep_index=True,
    )

    # Use all data to train calibration model for use on other data sets
    if np.any(mask):
        scaler = StandardScaler().fit(x_top[:, mask])
        x_top[:, mask] = scaler.transform(x_top[:, mask])

    # Train classifier on all data (scaled)
    model.fit(x_top, y)

    y_hat = model.predict_proba(x_top)[:, 1]
    calibration_model = train_calibrator(y=y, y_hat=y_hat)

    # Save objects to disk; note for some objects that are plural,
    # objects are expected as a list or a dict keyed by the model (supervised learner).
    model_dict = {
        "model": model,
        "scaler": scaler,
        "mask": mask,
        "calibration_model": calibration_model,
        "top_features_idx": top_features_idx[:num_top_features_asymptote],
    }
    save_results_to_disk(args=args, dict_to_save=model_dict)

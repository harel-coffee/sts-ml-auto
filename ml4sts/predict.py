# Imports: standard library
import logging

# Imports: third party
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Imports: ml4sts
from ml4sts.plot import plot_classifier_curve
from ml4sts.utils import (
    load_results,
    log_dataframe,
    select_cohort,
    save_results_to_disk,
    save_dataframe_to_csv,
    generate_crossfold_indices,
    get_continuous_features_mask,
    format_categorical_and_binary_features,
)
from ml4sts.bootstrap import (
    calculate_percentiles,
    get_bootstrap_metrics,
    compare_bootstrap_metrics,
    resample_precision_recall,
    generate_bootstrap_distributions,
    format_bootstrap_metrics_to_dataframe,
)
from ml4sts.calibration import calibrate_probabilities, calibration_metrics_and_curves
from ml4sts.definitions import (
    IMAGE_EXT,
    OUTCOME_MAP,
    MODEL_NAMES_FULL,
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURES_TO_REMOVE_FROM_TEST,
)


def _remove_rows_of_positive_feature(
    x: np.array,
    y: np.array,
    feature_names: np.array,
    column_name: str,
) -> tuple:
    idx_column_name = np.where(feature_names == column_name)[0].item()
    rows_to_remove = np.where(x[:, idx_column_name] == 1)
    x = np.delete(x, rows_to_remove, axis=0)
    y = np.delete(y, rows_to_remove)
    return x, y


def predict_ffs(args):
    # Load trained models, scalers, and other objects as dicts keyed by results directory
    results = load_results(args)
    model = next(iter(results["model"].values()))
    mask = next(iter(results["mask"].values()))
    scaler = next(iter(results["scaler"].values()))
    calibration_model = next(iter(results["calibration_model"].values()))
    top_features_idx = next(
        iter(results["top_features_idx"].values()),
    )

    # Load data on which to predict
    df = pd.read_csv(args.path_to_csv_predict, low_memory=False)

    # Select desired cohort
    if args.cohort:
        df = select_cohort(df=df, cohort=args.cohort)

    # Isolate features from dataframe
    df_x = df[args.features].copy()
    df_x = format_categorical_and_binary_features(df=df_x)

    # Get features and labels as numpy arrays
    x = df_x.values
    y = df[OUTCOME_MAP[args.outcome]].values

    # Select subset of best features from training via FFS
    x = x[:, top_features_idx]

    # Report dimensions
    n = x.shape[0]
    m = x.shape[1]
    positive_labels = sum(y)
    positive_labels_frac = sum(y) / n
    logging.info(
        f"Remaining data from {args.path_to_csv} has"
        f" {n} patients, {positive_labels} {args.outcome}"
        f" ({positive_labels_frac:0.2f}%), {m} features",
    )

    # Scale data
    x[:, mask] = scaler.transform(x[:, mask])

    # Predict using model
    y_hat = model.predict_proba(x)[:, 1]
    auc = roc_auc_score(y, y_hat)
    logging.info(f"AUC is {auc:0.2f}")

    logging.info(f"Generating {args.bootstrap_samplings} bootstrap distributions")

    # Calibrate predictions
    y_hat_calibrated = calibrate_probabilities(
        calibration_model=calibration_model,
        y_hat=y_hat,
    )

    # Generate bootstrap distributions
    (
        y_bootstrap,
        y_hat_bootstrap,
        y_hat_bootstrap_calibrated,
    ) = generate_bootstrap_distributions(
        seed=args.seed,
        y=y,
        y_hat=y_hat,
        y_hat_calibrated=y_hat_calibrated,
        bootstrap_samplings=args.bootstrap_samplings,
    )

    logging.info(f"Calculating TPRs and AUCs")

    # Calculate tprs and aucs from bootstrap distributions
    (tprs, aucs, precisions, recalls, average_precisions) = get_bootstrap_metrics(
        y_bootstrap=y_bootstrap,
        y_hat_bootstrap=y_hat_bootstrap,
    )

    model_name = "logreg"
    model_name_baseline = "logreg"

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
        file_name=f"curve_roc_{model_name}{IMAGE_EXT}",
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
        file_name=f"curve_pr_{model_name}{IMAGE_EXT}",
        legend_location="upper right",
        plot_diagonal=False,
    )

    # Calculate metrics of results with confidence intervals via bootstraps
    metrics = compare_bootstrap_metrics(
        args=args,
        y_baseline=y_bootstrap,
        y_compare=y_bootstrap,
        y_hat_baseline=y_hat_bootstrap,
        y_hat_baseline_calibrated=y_hat_bootstrap_calibrated,
        y_hat_compare=y_hat_bootstrap,
        y_hat_compare_calibrated=y_hat_bootstrap_calibrated,
        prefix_str="",
        model_name_baseline=model_name_baseline,
        model_name_compare=model_name,
    )

    # Format metrics dicts into dataframe
    df_metrics = format_bootstrap_metrics_to_dataframe(
        metrics=metrics,
        decimals=args.decimals,
    )

    # Log metrics dataframe
    logging.info(
        f"metrics comparing {model_name_baseline} vs {model_name}"
        f" via {args.bootstrap_samplings} bootstrap samplings:",
    )
    log_dataframe(
        df=df_metrics,
        format_scientific=False,
    )

    # Save metrics dataframe to CSV
    save_dataframe_to_csv(
        args=args,
        df=df_metrics,
        fname=f"metrics_{model_name}.csv",
        keep_index=True,
    )

    # Assess calibration metrics and curves
    calibration_metrics_and_curves(
        args=args,
        model_name=model_name,
        y=y,
        y_hat=y_hat,
        y_hat_calibrated=y_hat_calibrated,
    )


def predict(args):
    # Load trained models, scalers, and other objects as dicts keyed by results directory
    results = load_results(args)
    best_models = next(iter(results["best_models"].values()))
    scalers = next(iter(results["scalers"].values()))
    calibration_models = next(iter(results["calibration_models"].values()))
    idx_best_features_folds_models = next(
        iter(results["idx_best_features_folds_models"].values()),
    )
    idx_subset_test_folds = next(iter(results["idx_subset_test_folds"].values()))
    idx_original_test_folds = next(iter(results["idx_original_test_folds"].values()))

    # Load data on which to predict
    df = pd.read_csv(args.path_to_csv_predict, low_memory=False)

    # Select desired cohort
    if args.cohort:
        df = select_cohort(df=df, cohort=args.cohort)

    # Get indices of remaining data for both original and new dataframes
    idx_original_remaining = df.index.to_numpy()

    # Isolate features from dataframe
    df_x = df[args.features].copy()
    df_x = format_categorical_and_binary_features(df=df_x)

    # Get boolean mask indicating which features are continuous
    continuous_features_mask = get_continuous_features_mask(df=df_x)

    # Get features and labels as numpy arrays
    x = df_x.values
    y = df[OUTCOME_MAP[args.outcome]].values

    # Report dimensions
    n = x.shape[0]
    m = x.shape[1]
    positive_labels = sum(y)
    positive_labels_frac = sum(y) / n
    logging.info(
        f"Remaining data from {args.path_to_csv} has"
        f" {n} patients, {positive_labels} {args.outcome}"
        f" ({positive_labels_frac:0.2f}%), {m} features",
    )

    # Initialize objects in which to store bootstrapped results
    y_bootstrap = {}
    y_hat_bootstrap = {}
    y_hat_bootstrap_calibrated = {}

    y_hat_test_outer_folds = {}
    y_hat_test_calibrated_outer_folds = {}

    # Iterate through models (dict of lists with k models, where k is number
    # of outer cross folds)
    for model_name, models_by_fold in best_models.items():

        # Only iterate over desired models
        if model_name not in args.models:
            logging.info(
                f"{model_name} is not in arguments. Skipping to next model.",
            )
            continue

        logging.info(
            f"{model_name} // Predicting outcome(s) and calibrating predictions",
        )

        # If predicting on same CSV and same cohort, perform cross-fold validation,
        # use each fold's model to generate predictions on test data, and concatenate
        # predictions
        if args.predict_same_csv_as_train:
            logging.info(
                f"Predicting on same CSV as was used to train models;"
                f" removing unwanted case types from test sets and avoiding data leak",
            )
            mrns = []
            surgery_dates = []
            y_test = []
            y_hat = []

            # Iterate over folds, obtain indices of test data in terms of original and
            # new (subset) dataframes; also get MRNs and surgery dates for test data
            for fold, idx_original_test_fold in enumerate(idx_original_test_folds):

                # Isolate best features
                idx_best_features = idx_best_features_folds_models[fold][model_name]

                # Create copy of feature array to scale for this fold
                x_temp = np.copy(x)

                # Scale data using scaler from this fold's training set
                if np.any(continuous_features_mask):
                    x_temp[:, continuous_features_mask] = scalers[fold].transform(
                        x_temp[:, continuous_features_mask],
                    )

                # Isolate best features
                x_temp = x_temp[:, idx_best_features]

                # Intersect of test set indices and original dataframe gives us test set
                # indices (of original df) for remaining patients (desired case types)
                idx_original_test_fold_remaining = np.sort(
                    np.array(
                        list(set(idx_original_test_fold) & set(idx_original_remaining)),
                    ),
                )

                # Get indices of surviving rows in new dataframe
                idx_subset_test_fold = np.array(
                    [
                        np.argwhere(idx_original_remaining == idx)[0][0]
                        for idx in idx_original_test_fold_remaining
                    ],
                )

                # Get MRNs and surgery dates for patients
                mrns_fold = df[args.key_mrn].iloc[idx_subset_test_fold].to_numpy()
                mrns = np.append(mrns, mrns_fold)

                surgery_dates_fold = (
                    df[args.key_surgery_date].iloc[idx_subset_test_fold].to_numpy()
                )
                surgery_dates = np.append(surgery_dates, surgery_dates_fold)

                # Isolate test fold of data from which to predict outcomes
                x_test_fold = x_temp[idx_subset_test_fold, :]
                y_test_fold = y[idx_subset_test_fold]

                # Isolate model trained on this fold's training data;
                # this model has never seen this fold's test data
                model = models_by_fold[fold]

                # Predict probabilities for this fold
                y_hat_fold = model.predict_proba(x_test_fold)[:, 1]

                # Assess and log AUC of predictions for this fold
                auc = roc_auc_score(y_test_fold, y_hat_fold)
                logging.info(f"fold {fold} test set AUC is {auc:0.2f}")

                # Update label and prediction arrays with values from this fold
                y_test = np.append(y_test, y_test_fold)
                y_hat = np.append(y_hat, y_hat_fold)

        # Predict on entire dataset in CSV
        else:
            # Create copy of feature array to scale for this model
            x_temp = np.copy(x)

            # Scale data using scaler from any fold's training set
            if np.any(continuous_features_mask):
                x_temp[:, continuous_features_mask] = scalers[0].transform(
                    x_temp[:, continuous_features_mask],
                )

            # Isolate top features for crossfold 0 and this model
            fold_used = 0
            idx_best_features = idx_best_features_folds_models[fold_used][model_name]
            x_temp = x_temp[:, idx_best_features]

            # Predict outcome using model from any crossfold
            y_hat = models_by_fold[0].predict_proba(x_temp)[:, 1]

            # Rename y to y_test so future code can use y_test
            y_test = y

        # Calibrate test set predictions
        # using the first fold's calibration model.
        calibration_model = calibration_models[model_name][0]
        y_hat_test_calibrated = calibrate_probabilities(
            calibration_model=calibration_model,
            y_hat=y_hat,
        )

        # Save predictions to dict of arrays, keyed by model
        y_hat_test_outer_folds[model_name] = y_hat
        y_hat_test_calibrated_outer_folds[model_name] = y_hat_test_calibrated

        if y_test.size != 0:
            auc = roc_auc_score(y_test, y_hat)
            logging.info(f"AUC using all test labels and predictions = {auc:0.2f}")

        # Calibrate predictions: use first fold's calibration model,
        # transform the value, and then average predictions over all models
        y_hat_calibrated = []
        calibration_model = calibration_models[model_name][0]
        y_hat_calibrated = calibrate_probabilities(
            calibration_model=calibration_model,
            y_hat=y_hat,
        )

        # If no labels for cohort, then we are simply applying an existing model to
        # data to perform inference. We cannot assess how well that mdoel does.
        # Save predicted and calibrated labels to CSV.
        if y_test.size == 0:

            # Convert dict of arrays into dataframe
            df_probabilities = pd.DataFrame.from_dict(
                {
                    "mrn": mrns_reordered,
                    "surgery_date": surgery_dates_reordered,
                    "y_hat": y_hat,
                    "y_hat_calibrated": y_hat_calibrated,
                },
            )

            # Save labels and predictions to CSV
            save_dataframe_to_csv(
                args=args,
                df=df_probabilities,
                fname=f"predicted_probabilities_{model_name}",
                keep_index=False,
            )

            return

        # Else, labels exist for cohort. Save labels, predicted probabilities, and
        # calibrated probabilities to CSV for future reference.
        # Perform bootstrap sampling to assess model performance.
        results = {
            "y": y_test,
            "y_hat": y_hat,
            "y_hat_calibrated": y_hat_calibrated,
        }

        if "mrn" in df:
            mrn = df.mrn.to_numpy()
            results["mrn"] = mrn
        if "mrns" in df:
            mrns = df.mrn.to_numpy()
            results["mrns"] = mrns

        # Convert dict of arrays into dataframe
        df_probabilities = pd.DataFrame.from_dict(results)

        # Save labels and predictions to CSV
        save_dataframe_to_csv(
            args=args,
            df=df_probabilities,
            fname=f"predicted_probabilities_{model_name}",
            keep_index=False,
        )

        # Assess calibration metrics and curves
        calibration_metrics_and_curves(
            args=args,
            model_name=model_name,
            y=y_test,
            y_hat=y_hat,
            y_hat_calibrated=y_hat_calibrated,
        )

        logging.info(
            f"{model_name} // Generating {args.bootstrap_samplings}"
            " bootstrap distributions",
        )

        # Generate bootstrap distributions
        (
            y_bootstrap[model_name],
            y_hat_bootstrap[model_name],
            y_hat_bootstrap_calibrated[model_name],
        ) = generate_bootstrap_distributions(
            seed=args.seed,
            y=y_test,
            y_hat=y_hat,
            y_hat_calibrated=y_hat_calibrated,
            bootstrap_samplings=args.bootstrap_samplings,
        )

        logging.info(f"{model_name} // Calculating TPRs and AUCs")

        # Calculate metrics from bootstrap distributions
        tprs, aucs, precisions, recalls, average_precisions = get_bootstrap_metrics(
            y_bootstrap=y_bootstrap[model_name],
            y_hat_bootstrap=y_hat_bootstrap[model_name],
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
        average_precision_percentiles = calculate_percentiles(
            arr=average_precisions,
        )

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
            file_name=f"roc_curve_{model_name}{IMAGE_EXT}",
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
            file_name=f"pr_curve_{model_name}{IMAGE_EXT}",
            legend_location="upper right",
            plot_diagonal=False,
        )

        # Compare models via bootstrap sampling
        if "logreg" in best_models:
            model_name_baseline = "logreg"
        else:
            model_name_baseline = model_name
            logging.info("No logistic regression to compare other models against")

        metrics_bootstrap = compare_bootstrap_metrics(
            args=args,
            y_baseline=y_bootstrap[model_name_baseline],
            y_compare=y_bootstrap[model_name],
            y_hat_baseline=y_hat_bootstrap[model_name_baseline],
            y_hat_baseline_calibrated=y_hat_bootstrap_calibrated[model_name_baseline],
            y_hat_compare=y_hat_bootstrap[model_name],
            y_hat_compare_calibrated=y_hat_bootstrap_calibrated[model_name],
            prefix_str="",
            model_name_baseline=model_name_baseline,
            model_name_compare=model_name,
        )

        # Format metrics dicts into dataframe
        df_metrics = format_bootstrap_metrics_to_dataframe(
            metrics=metrics_bootstrap,
            decimals=args.decimals,
        )

        # Log metrics dataframe
        logging.info(
            f"metrics comparing {model_name_baseline} vs {model_name}"
            f" via {args.bootstrap_samplings} bootstrap samplings:",
        )
        log_dataframe(
            df=df_metrics,
            format_scientific=False,
        )

        # Save metrics dataframe to CSV
        save_dataframe_to_csv(
            args=args,
            df=df_metrics,
            fname=f"metrics_compare_{model_name_baseline}_vs_{model_name}.csv",
            keep_index=True,
        )

    # Save labels & predicted probabilities for future use via compare
    # in arrays that have the same order as arrays generated by train mode
    results_to_save = {
        "y_test_outer_folds": y_test,
        "y_hat_test_outer_folds": y_hat_test_outer_folds,
        "y_hat_test_calibrated_outer_folds": y_hat_test_calibrated_outer_folds,
    }
    save_results_to_disk(args=args, dict_to_save=results_to_save)

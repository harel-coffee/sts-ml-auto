# Imports: standard library
import logging
import multiprocessing as mp
from collections import Counter, defaultdict

# Imports: third party
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Imports: ml4sts
from ml4sts.plot import plot_classifier_curve, plot_feature_coefficients
from ml4sts.utils import (
    log_dataframe,
    select_cohort,
    create_shared_array,
    save_results_to_disk,
    save_dataframe_to_csv,
    generate_crossfold_indices,
    get_continuous_features_mask,
    get_crossfold_indices_original_df,
    format_categorical_and_binary_features,
)
from ml4sts.models import (
    initialize_model,
    get_feature_values,
    evaluate_predictions,
    add_fold_and_model_info_to_results,
    get_descriptive_feature_coefficients,
    format_cross_validation_results_as_df,
)
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
    threshold_probabilities,
    calibration_metrics_and_curves,
)
from ml4sts.definitions import IMAGE_EXT, OUTCOME_MAP, MODEL_NAMES_FULL
from ml4sts.hyperparameters import generate_hyperparameters, get_best_hyperparameters

# Global dictionary stores variables passed from initializer
var_dict = {}


def _init_globals_train(
    x_train_inner: np.array,
    x_train_inner_shape: tuple,
    y_train_inner: np.array,
    y_train_inner_shape: tuple,
    x_test_inner: np.array,
    x_test_inner_shape: tuple,
    y_test_inner: np.array,
    y_test_inner_shape: tuple,
):
    var_dict["x_train_inner"] = x_train_inner
    var_dict["x_train_inner_shape"] = x_train_inner_shape
    var_dict["y_train_inner"] = y_train_inner
    var_dict["y_train_inner_shape"] = y_train_inner_shape
    var_dict["x_test_inner"] = x_test_inner
    var_dict["x_test_inner_shape"] = x_test_inner_shape
    var_dict["y_test_inner"] = y_test_inner
    var_dict["y_test_inner_shape"] = y_test_inner_shape


def _train_model_parallel(model_name_and_hp: tuple) -> dict:
    # Access variables from underlying memory array
    x_train = np.frombuffer(var_dict["x_train_inner"]).reshape(
        var_dict["x_train_inner_shape"],
    )
    y_train = np.frombuffer(var_dict["y_train_inner"]).reshape(
        var_dict["y_train_inner_shape"],
    )

    x_test = np.frombuffer(var_dict["x_test_inner"]).reshape(
        var_dict["x_test_inner_shape"],
    )
    y_test = np.frombuffer(var_dict["y_test_inner"]).reshape(
        var_dict["y_test_inner_shape"],
    )
    # Unpack tuple
    model_name = model_name_and_hp[0]
    hyperparameters = model_name_and_hp[1]

    # SMOTE
    if hyperparameters["smote_sampling_strategy"]:
        resampler = SMOTE(
            random_state=hyperparameters["seed"],
            n_jobs=-1,
            sampling_strategy=hyperparameters["smote_sampling_strategy"],
        )
        x_train, y_train = resampler.fit_resample(
            x_train,
            y_train,
        )
    y_train_pos = sum(y_train == 1)
    y_train_neg = sum(y_train == 0)
    y_train_pos_percent = y_train_pos / len(y_train) * 100

    y_test_pos = sum(y_test == 1)
    y_test_neg = sum(y_test == 0)
    y_test_pos_percent = y_test_pos / len(y_test) * 100

    # Initialize and train model
    model = initialize_model(model_name, hyperparameters)
    model.fit(x_train, y_train)

    # Determine number of top features
    num_top_features_float = hyperparameters["feature_select_ratio"] * x_train.shape[1]
    num_top_features = max(1, int(np.floor(num_top_features_float)))

    # Sort coefficients from highest to lowest abs value and isolate the subset
    coefs = get_feature_values(model=model)
    idx_sorted_coefs = np.argsort(np.abs(coefs))[::-1]
    idx_sorted_coefs = idx_sorted_coefs[:num_top_features]

    # Isolate subset of top features
    x_train = x_train[:, idx_sorted_coefs]
    x_test = x_test[:, idx_sorted_coefs]

    # Train model on subset
    model.fit(x_train, y_train)
    y_hat_train = model.predict_proba(x_train)[:, 1]
    auc_train = roc_auc_score(y_train, y_hat_train)
    y_hat_test = model.predict_proba(x_test)[:, 1]
    auc_test = roc_auc_score(y_test, y_hat_test)

    results = {
        "hp": hyperparameters,
        "y_train_pos": y_train_pos,
        "y_train_neg": y_train_neg,
        "y_train_pos_percent": y_train_pos_percent,
        "y_test_pos": y_test_pos,
        "y_test_neg": y_test_neg,
        "y_test_pos_percent": y_test_pos_percent,
        "auc_train": auc_train,
        "auc_test": auc_test,
        "idx_sorted_coefs": idx_sorted_coefs,
    }
    return results


def train(args):
    # Start method required to use CUDA in subprocesses
    mp.set_start_method("spawn")

    # Load data from dataframe
    df = pd.read_csv(args.path_to_csv, low_memory=False)

    # Select desired cohort
    if args.cohort:
        df = select_cohort(df=df, cohort=args.cohort)

    # Get array of indices of remaining rows, but in indices of original df
    idx_original_remaining = df.index.to_numpy()

    # Get array of binary outcomes from dataframe
    y = df[OUTCOME_MAP[args.outcome]].values

    # Generate cross-fold indices relative to indices of new dataframe with subset
    # case types, not indices of original dataframe
    skf, idx_subset_train_folds, idx_subset_test_folds = generate_crossfold_indices(
        crossfolds=args.crossfolds,
        seed=args.seed,
        y=y,
    )

    # Use indices of subset dataframe to identify indices in original dataframe;
    # these values are required for consistency in predict mode
    (
        idx_original_train_folds,
        idx_original_test_folds,
    ) = get_crossfold_indices_original_df(
        idx_original_remaining=idx_original_remaining,
        idx_subset_train_folds=idx_subset_train_folds,
        idx_subset_test_folds=idx_subset_test_folds,
    )

    # Get MRNs and surgery dates from subset df;
    # this enables exporting model predictions next to corresponding patient info.
    # If df does not contain MRN column, use row numbers as MRNs.
    if args.key_mrn in df:
        mrns = df[args.key_mrn].to_numpy(copy=True)
    else:
        mrns = df.reset_index().index

    surgery_dates = df[args.key_surgery_date].to_numpy(copy=True)

    # Isolate features from dataframe and format
    df_x = format_categorical_and_binary_features(df=df[args.features].copy())

    # Get mask of continuous features
    continuous_features_mask = get_continuous_features_mask(df=df_x)

    # Get feature names
    feature_names = np.array(df_x.columns)

    # Get features as numpy array
    x = df_x.values

    n = x.shape[0]
    m = x.shape[1]
    positive_labels = sum(y)
    positive_labels_percent = sum(y) / n * 100
    logging.info(
        f"Remaining data from {args.path_to_csv} has"
        f" {n} patients, {positive_labels} {args.outcome} labels"
        f" ({positive_labels_percent:0.1f}%), {m} features",
    )

    # Generate hyperparameter combinations for each model
    logging.info(
        f"Generating {args.hyperparameter_random_samplings} random"
        " hyperparameter combinations",
    )
    hp = {}
    for model_name in args.models:
        hp[model_name] = generate_hyperparameters(
            seed=args.seed,
            model_name=model_name,
            random_samplings=args.hyperparameter_random_samplings,
            smote=args.smote,
        )

    # Reset random seed to a set value so the number of times you use random
    # number generators in generate_hyperparameters does not affect
    # reproducibility of downstream crossfold generation
    np.random.seed(args.seed)

    mrns_reordered = []
    surgery_dates_reordered = []
    best_models = {}
    best_hps = {}
    scalers = []
    y_test_outer_folds = []
    y_hat_test_outer_folds = {}
    y_hat_test_calibrated_outer_folds = {}
    metrics = {}
    calibration_models = {}
    idx_best_features_folds_models = defaultdict()

    # Iterate through models to initialize dict elements
    for model_name in args.models:
        best_models[model_name] = []
        best_hps[model_name] = []
        y_hat_test_outer_folds[model_name] = []
        y_hat_test_calibrated_outer_folds[model_name] = []
        calibration_models[model_name] = []

    # Iterate through k outer folds
    for (outer_fold, (idx_subset_train_fold, idx_subset_test_fold)) in enumerate(
        zip(idx_subset_train_folds, idx_subset_test_folds),
    ):
        logging.info(f"Processing outer fold {outer_fold+1}")

        # Initialize dict within dict to store best indices
        idx_best_features_folds_models[outer_fold] = defaultdict()

        # Extend list of reordered MRNs using indices of test set
        mrns_reordered.extend(mrns[idx_subset_test_fold])
        surgery_dates_reordered.extend(surgery_dates[idx_subset_test_fold])

        # Split x and y into train and test sets for this fold
        x_train = x[idx_subset_train_fold, :]
        x_test = x[idx_subset_test_fold, :]
        y_train = y[idx_subset_train_fold]
        y_test = y[idx_subset_test_fold]

        # Standardize features using training data
        if np.any(continuous_features_mask):
            scaler = StandardScaler().fit(x_train[:, continuous_features_mask])
            x_train[:, continuous_features_mask] = scaler.transform(
                x_train[:, continuous_features_mask],
            )
            x_test[:, continuous_features_mask] = scaler.transform(
                x_test[:, continuous_features_mask],
            )
            scalers.append(scaler)

        # Assess distribution of scaled features
        logging.info(
            f"Calculating information about feature distributions in transformed test"
            f" set:",
        )
        for i in range(0, x_test.shape[1]):
            if continuous_features_mask[i]:
                mean = x_test[:, i].mean()
                std = x_test[:, i].std()
                logging.info(f"\t{feature_names[i]}: mu={mean:0.2f}, sigma={std:0.2f}")
            else:
                logging.info(
                    f"\t{feature_names[i]}: unique values {np.unique(x_test[:, i])}",
                )

        label_percent_train = sum(y_train) / len(y_train) * 100
        logging.info(
            f"Train labels (outer): {sorted(Counter(y_train).items())}"
            f"; {label_percent_train:0.1f}% y=1",
        )

        label_percent_test = sum(y_test) / len(y_test) * 100
        logging.info(
            f"Test labels (outer): {sorted(Counter(y_test).items())}"
            f"; {label_percent_test:0.1f}% y=1",
        )

        y_test_outer_folds.append(y_test)

        # Initialize results list
        results_outer = []

        # Iterate through K inner folds
        for (inner_fold, (idx_train_inner, idx_test_inner)) in enumerate(
            skf.split(x_train, y_train),
        ):
            logging.info(f"Processing inner fold {inner_fold+1}")

            # Generate inner training and test sets
            x_train_inner = x_train[idx_train_inner, :]
            x_test_inner = x_train[idx_test_inner, :]
            y_train_inner = y_train[idx_train_inner]
            y_test_inner = y_train[idx_test_inner]

            # Convert numpy arrays into shared raw arrays for multiprocess
            x_train_inner, x_train_inner_shape = create_shared_array(x_train_inner)
            y_train_inner, y_train_inner_shape = create_shared_array(y_train_inner)
            x_test_inner, x_test_inner_shape = create_shared_array(x_test_inner)
            y_test_inner, y_test_inner_shape = create_shared_array(y_test_inner)
            with mp.Pool(
                processes=None,
                initializer=_init_globals_train,
                initargs=(
                    x_train_inner,
                    x_train_inner_shape,
                    y_train_inner,
                    y_train_inner_shape,
                    x_test_inner,
                    x_test_inner_shape,
                    y_test_inner,
                    y_test_inner_shape,
                ),
            ) as pool:
                for model_name in args.models:
                    logging.info(
                        f"Training {model_name} on"
                        f" {args.hyperparameter_random_samplings}"
                        " hyperparameter combos",
                    )
                    results_inner = pool.map(
                        _train_model_parallel,
                        [(model_name, hp_vals) for hp_vals in hp[model_name]],
                    )

                    # Add crossfold and model info to results for this inner loop
                    results_inner = add_fold_and_model_info_to_results(
                        results=results_inner,
                        outer_fold=outer_fold,
                        inner_fold=inner_fold,
                        model_name=model_name,
                    )

                    # Update outer fold results with inner fold results
                    results_outer.extend(results_inner)

        # Format list of results of this outer fold into dataframe
        df_results_outer = format_cross_validation_results_as_df(
            results=results_outer,
            models=args.models,
            decimals=args.decimals,
        )

        # Log results
        logging.info(f"Performance in outer fold {outer_fold+1}")
        log_dataframe(
            df=df_results_outer,
            format_scientific=False,
            omit_columns=["idx_sorted_coefs"],
        )

        # At this point, all models have trained all inner folds for this outer fold

        # Iterate through models, find best hyperparameter using inner folds,
        # retrain on outer training data, predict on outer test data
        for model_name in args.models:

            # Get hp that achieved best inner test AUC
            best_hps_this_model, idx_sorted_coefs = get_best_hyperparameters(
                df=df_results_outer,
                model_name=model_name,
            )

            # Initialize model with best hyperparameters
            model = initialize_model(model_name=model_name, hp=best_hps_this_model)

            # Save best_hps for this outer fold to list of best_hps
            best_hps[model_name].append(best_hps_this_model)

            # Isolate subset of top features
            num_top_features = (
                best_hps_this_model["feature_select_ratio"] * x_train.shape[1]
            )
            num_top_features = max(1, int(np.floor(num_top_features)))
            idx_best_features = idx_sorted_coefs[:num_top_features]
            x_train_fold = x_train[:, idx_best_features]
            x_test_fold = x_test[:, idx_best_features]

            # Update object to save
            idx_best_features_folds_models[outer_fold][model_name] = idx_best_features

            # Train model on outer fold train data, subset to selected features
            model.fit(x_train_fold, y_train)

            # Predict
            y_hat_train = model.predict_proba(x_train_fold)[:, 1]
            y_hat_test = model.predict_proba(x_test_fold)[:, 1]

            # Save best trained model for this fold
            best_models[model_name].append(model)

            # Evaluate metrics on test set without calibration
            metrics[model_name] = evaluate_predictions(y=y_test, y_hat=y_hat_test)

            # Threshold predictions
            y_hat_train = threshold_probabilities(
                y_hat=y_hat_train,
                threshold=args.calibration_probability_threshold,
            )
            y_hat_test = threshold_probabilities(
                y_hat=y_hat_test,
                threshold=args.calibration_probability_threshold,
            )

            # Train and save calibration model (regressor) on training data
            calibration_model = train_calibrator(y=y_train, y_hat=y_hat_train)
            calibration_models[model_name].append(calibration_model)

            # Calibrate test set predictions
            y_hat_test_calibrated = calibrate_probabilities(
                calibration_model=calibration_model,
                y_hat=y_hat_test,
            )

            # Append predictions to dicts of arrays
            y_hat_test_outer_folds[model_name].append(y_hat_test)
            y_hat_test_calibrated_outer_folds[model_name].append(y_hat_test_calibrated)

            logging.info(
                f"{model_name} - finished evaluating outer fold {outer_fold+1}",
            )

    # Nested crossfold validation is done!

    # Concatenate lists of arrays into single array
    y_test_outer_folds = np.concatenate(y_test_outer_folds)

    y_bootstrap = {}
    y_hat_bootstrap = {}
    y_hat_bootstrap_calibrated = {}

    # Standardize all data
    if np.any(continuous_features_mask):
        scaler = StandardScaler().fit(x[:, continuous_features_mask])
        x[:, continuous_features_mask] = scaler.transform(
            x[:, continuous_features_mask],
        )

    # Iterate through models to assess:
    # 1. Feature values
    # 2. Calibration curves
    # 3. ROC curves
    for model_name in args.models:

        # Log best hyperparameters over outer folds
        logging.info(f"{model_name} - best hyperparameters (one per outer fold):")
        log_dataframe(
            df=pd.DataFrame(best_hps[model_name]),
            format_scientific=True,
        )

        # Identify fold with best test AUC
        outer_fold_with_best_test_auc = np.argmax(metrics[model_name]["auc"])

        # Get hyperparameters associated with best outer fold
        overall_best_hps = best_hps[model_name][outer_fold_with_best_test_auc]

        # Initialize and train model with hyperparameters from best outer fold
        model = initialize_model(model_name=model_name, hp=overall_best_hps)
        model = model.fit(x, y)

        # Get indices of coefficients sorted from highest to lowest absolute value
        coefs = get_feature_values(model=model)
        idx_sorted_coefs = np.argsort(np.abs(coefs))[::-1]

        # Isolate subset of top features
        num_top_features = overall_best_hps["feature_select_ratio"] * x.shape[1]
        num_top_features = max(1, int(np.floor(num_top_features)))
        idx_best_features = idx_sorted_coefs[:num_top_features]
        x_subset = x[:, idx_best_features]

        # Get names of best features subset
        feature_names_subset = feature_names[idx_best_features]

        # Train model on best features subset and get coefficients
        model = model.fit(x_subset, y)
        coefs = get_feature_values(model=model)
        feature_values_df = get_descriptive_feature_coefficients(
            coefs=coefs,
            feature_names=feature_names_subset,
            decimals=args.decimals,
        )

        # Log feature values
        logging.info(f"{model_name} - feature coefficients")
        log_dataframe(
            df=feature_values_df,
            format_scientific=False,
        )

        # Save metrics dataframe to CSV
        save_dataframe_to_csv(
            args=args,
            df=feature_values_df,
            fname=f"feature_coefs_{model_name}.csv",
            keep_index=False,
        )

        # Plot feature values
        plot_feature_coefficients(
            args=args,
            model_name=model_name,
            feature_values=feature_values_df,
        )

        # Concatenate lists of predictions across test sets into single array
        y_hat_test_outer_folds[model_name] = np.concatenate(
            y_hat_test_outer_folds[model_name],
        )
        y_hat_test_calibrated_outer_folds[model_name] = np.concatenate(
            y_hat_test_calibrated_outer_folds[model_name],
        )

        # Convert dict of arrays into dataframe
        df_probabilities = pd.DataFrame.from_dict(
            {
                "mrn": mrns_reordered,
                "surgery_date": surgery_dates_reordered,
                "y": y_test_outer_folds,
                "y_hat": y_hat_test_outer_folds[model_name],
                "y_hat_calibrated": y_hat_test_calibrated_outer_folds[model_name],
            },
        )

        # Save labels and predictions to CSV
        save_dataframe_to_csv(
            args=args,
            df=df_probabilities,
            fname=f"predicted_probabilities_{model_name}",
            keep_index=False,
        )

        # Calculate calibration metrics and plot curves
        calibration_metrics_and_curves(
            args=args,
            model_name=model_name,
            y=y_test_outer_folds,
            y_hat=y_hat_test_outer_folds[model_name],
            y_hat_calibrated=y_hat_test_calibrated_outer_folds[model_name],
        )

        # Generate bootstrap distributions
        (
            y_bootstrap[model_name],
            y_hat_bootstrap[model_name],
            y_hat_bootstrap_calibrated[model_name],
        ) = generate_bootstrap_distributions(
            seed=args.seed,
            y=y_test_outer_folds,
            y_hat=y_hat_test_outer_folds[model_name],
            y_hat_calibrated=y_hat_test_calibrated_outer_folds[model_name],
            bootstrap_samplings=args.bootstrap_samplings,
        )

        # Calculate tprs and aucs from bootstrap distributions
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

        # Compare models via bootstrap sampling
        if "logreg" in args.models:
            baseline_name = "logreg"
        else:
            baseline_name = args.models[0]

        metrics_bootstrap = compare_bootstrap_metrics(
            args=args,
            y_baseline=y_bootstrap[model_name],
            y_compare=y_bootstrap[model_name],
            y_hat_baseline=y_hat_bootstrap[baseline_name],
            y_hat_baseline_calibrated=y_hat_bootstrap_calibrated[baseline_name],
            y_hat_compare=y_hat_bootstrap[model_name],
            y_hat_compare_calibrated=y_hat_bootstrap_calibrated[model_name],
            prefix_str="",
            model_name_baseline=baseline_name,
            model_name_compare=model_name,
        )

        # Format metrics dicts into dataframe
        metrics_df = format_bootstrap_metrics_to_dataframe(
            metrics=metrics_bootstrap,
            decimals=args.decimals,
        )

        # Log metrics dataframe
        logging.info(
            f"metrics comparing {baseline_name} vs {model_name} via"
            f" {args.bootstrap_samplings} bootstrap samplings:",
        ),
        log_dataframe(
            df=metrics_df,
            format_scientific=False,
        )

        # Save metrics dataframe to CSV
        save_dataframe_to_csv(
            args=args,
            df=metrics_df,
            fname=f"metrics_compare_{baseline_name}_vs_{model_name}.csv",
            keep_index=True,
        )

    # Save objects
    results_to_save = {
        "idx_subset_train_folds": idx_subset_train_folds,
        "idx_subset_test_folds": idx_subset_test_folds,
        "idx_original_train_folds": idx_original_train_folds,
        "idx_original_test_folds": idx_original_test_folds,
        "y_test_outer_folds": y_test_outer_folds,
        "y_hat_test_outer_folds": y_hat_test_outer_folds,
        "y_hat_test_calibrated_outer_folds": y_hat_test_calibrated_outer_folds,
        "best_models": best_models,
        "scalers": scalers,
        "calibration_models": calibration_models,
        "idx_best_features_folds_models": idx_best_features_folds_models,
    }
    save_results_to_disk(args=args, dict_to_save=results_to_save)

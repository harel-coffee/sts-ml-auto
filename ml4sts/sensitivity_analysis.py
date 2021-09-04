# Imports: standard library
import os
import logging

# Imports: third party
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Imports: ml4sts
# Imports: first party
from ml4sts.plot import plot_sensitivity_analysis_auc, plot_distributions_feature_y_hat
from ml4sts.utils import (
    load_results,
    log_dataframe,
    get_col_from_csv,
    get_full_feature_name,
    save_dataframe_to_csv,
    generate_crossfold_indices,
)
from ml4sts.bootstrap import (
    calculate_percentiles,
    get_bootstrap_metrics,
    generate_bootstrap_distributions,
)
from ml4sts.definitions import (
    SCALE_FACTORS,
    OPERATIVE_FEATURES_LOOKUP,
    SENSITIVITY_ANALYSIS_FEATURES,
    OPERATIVE_FEATURES_PERCENTILES,
)


def _count_and_sort_procedure_types(procedure_types: np.array) -> tuple:
    # Convert array of procedure types into dataframe
    df = pd.DataFrame(procedure_types, columns=["procedure_types"])

    # Get value counts
    value_counts = df.procedure_types.value_counts()

    # Turn counts series into dataframe with name and counts
    df_counts = pd.DataFrame(
        value_counts.reset_index().values,
        columns=["procedure_type", "counts"],
    )

    # Return sorted from most to leaset prevalent
    return df_counts.sort_index()


def _replace_feature(
    x: np.array,
    feature_names: np.array,
    feature_to_replace: str,
    procedure_types: np.array,
    replacement_values: dict,
) -> np.array:
    # Determine index of feature to replace
    idx = np.where(feature_names == OPERATIVE_FEATURES_LOOKUP[feature_to_replace])[0][0]

    # Overwite value of feature with value from dict stratified by procedure type
    x[:, idx] = [
        replacement_values[procedure_type] for procedure_type in procedure_types
    ]
    return x


def _scale_feature_by_factor(
    x: np.array,
    feature_names: np.array,
    feature_to_scale: str,
    scale_factor: float,
) -> np.array:
    idx = np.where(feature_names == OPERATIVE_FEATURES_LOOKUP[feature_to_scale])
    x[:, idx] *= scale_factor
    return x


def _get_percentile_of_feature_stratified(
    feature_values: np.array,
    procedure_types: np.array,
    percentile: int,
) -> dict:
    if feature_values.shape != procedure_types.shape:
        raise ValueError(
            "Feature and procedure type arrays have mismatched dimensions.",
        )
    results = {}
    for unique_val in np.unique(procedure_types):
        results[unique_val] = np.percentile(
            feature_values[procedure_types == unique_val],
            percentile,
        )
    return results


def _subset_array_by_procedure(array: np.array, procedure_types: np.array) -> dict:
    phrases_to_keep_partial = []
    phrases_to_keep_exact = ["aortic_valve_aorta", "aorta"]
    idx_match = set()

    # Iterate through phrases to keep: partial
    for phrase in phrases_to_keep_partial:

        # Find indices of elements that contain the phrase
        idx_match_phrase = {
            i for i, item in enumerate(procedure_types) if phrase in item
        }

        # Update set with indices
        idx_match.update(idx_match_phrase)

    # Iterate through phrases to keep: exact
    for phrase in phrases_to_keep_exact:

        # Find indices of elements that match the phrase
        idx_match_phrase = {
            i for i, item in enumerate(procedure_types) if phrase == item
        }

        # Update set with indices
        idx_match.update(idx_match_phrase)

    return array[list(idx_match)]


def sensitivity_analysis(args):
    # Load trained models from specified results dir
    _, _, models, _, _ = load_results_from_disk(args)

    # TODO REVISE
    # Manually set features to all possible valid features
    args.features = VALID_FEATURE_ARGS

    # Load data from dataframe
    df = pd.read_csv(args.path_to_csv, low_memory=False)

    # Isolate desired case types
    if args.case_types:
        for case_type in args.case_types:
            logging.info(f"Isolating patients with case type {case_type}")
            df = df[df[case_type] == 1]
        df.reset_index(drop=True, inplace=True)

    # Get MRNs and surgery dates from df
    mrns = df[args.key_mrn].to_numpy(copy=True)
    surgery_dates = df[args.key_surgery_date].to_numpy(copy=True)

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

    # Get procedure type from CSV
    procedure_types = get_col_from_csv(
        path_to_csv=args.path_to_csv,
        key="procedure_type",
    )

    np.random.seed(args.seed)

    # Get models out of dict indexed by results dir; this only works
    # because this function can only process one --path_to_results
    models = next(iter(models.values()))

    # Check number of crossfolds from model object, overwriting default arg
    args.crossfolds = len(next(iter(models.values())))

    # Generate cross-fold indices
    (_skf_out, _skf_in, idx_train, idx_test) = generate_crossfold_indices(args, x, y)

    logging.info(
        f"Performing sensitivity analysis on {args.path_to_results[0]}"
        f" using {args.bootstrap_samplings} bootstrap samplings",
    )

    feature_percentile_by_procedure = {}

    # Iterate through features to scale
    for feature_name in SENSITIVITY_ANALYSIS_FEATURES:

        # Get feature array from CSV
        feature_array = get_col_from_csv(
            path_to_csv=args.path_to_csv,
            key=OPERATIVE_FEATURES_LOOKUP[feature_name],
        )

        # Calculate percentile of feature time stratified by procedure type
        for percentile in OPERATIVE_FEATURES_PERCENTILES:
            feature_percentile_by_procedure[
                percentile
            ] = _get_percentile_of_feature_stratified(
                feature_values=feature_array,
                procedure_types=procedure_types,
                percentile=percentile,
            )

        sns.set(style="whitegrid", palette="muted", color_codes=True)
        sns.set_context("poster")
        _fig, axes = plt.subplots(figsize=(9, 6))

        # Iterate through dict of models by model name
        for model_name, models_by_fold in models.items():
            logging.info(f"{feature_name} // {model_name} // predicting labels")

            y_test = []
            y_hat_scale_factors = {}
            y_hat_replaced = {}
            auc_percentiles = {}

            # Iterate through k outer folds
            for (fold, (idx_train_this_fold, idx_test_this_fold)) in enumerate(
                zip(idx_train, idx_test),
            ):

                # Split x and y into train and test sets for this fold
                x_train = x[idx_train_this_fold, :]
                x_test = x[idx_test_this_fold, :]

                # Append true labels for test set
                y_test.append(y[idx_test_this_fold])

                # Calculate scaler object to standardize data
                scaler = StandardScaler().fit(x_train[:, continuous_features_mask])

                # Isolate model for this outer fold
                model = models_by_fold[fold]

                # Iterate through scale factors
                for scale_factor in SCALE_FACTORS:
                    if scale_factor not in y_hat_scale_factors:
                        y_hat_scale_factors[scale_factor] = []

                    x_scaled = _scale_feature_by_factor(
                        x=np.copy(x_test),
                        feature_names=feature_names,
                        feature_to_scale=feature_name,
                        scale_factor=scale_factor,
                    )

                    # Standardize continuous features
                    x_scaled[:, continuous_features_mask] = scaler.transform(
                        x_scaled[:, continuous_features_mask],
                    )

                    # Predict y_hat
                    y_hat = np.array(model.predict_proba(x_scaled)[:, 1])

                    # Append dict of lists with y_hat array
                    y_hat_scale_factors[scale_factor].append(y_hat)

                # Iterate through percentiles of a feature and replace all
                # instances of a feature with that percentile
                for percentile in OPERATIVE_FEATURES_PERCENTILES:
                    if percentile not in y_hat_replaced:
                        y_hat_replaced[percentile] = []

                    x_replaced = _replace_feature(
                        x=np.copy(x_test),
                        feature_names=feature_names,
                        feature_to_replace=feature_name,
                        procedure_types=procedure_types[idx_test_this_fold],
                        replacement_values=feature_percentile_by_procedure[percentile],
                    )

                    # Standardize continuous features
                    x_replaced[:, continuous_features_mask] = scaler.transform(
                        x_replaced[:, continuous_features_mask],
                    )

                    # Predict y_hat
                    y_hat = np.array(model.predict_proba(x_replaced)[:, 1])

                    # Append dict of lists with y_hat array
                    y_hat_replaced[percentile].append(y_hat)

            # Concatenate list of arrays into single array
            y_test = np.concatenate(y_test)

            # Convert dict of lists of arrays into dict of arrays
            for scale_factor in SCALE_FACTORS:
                y_hat_scale_factors[scale_factor] = np.concatenate(
                    y_hat_scale_factors[scale_factor],
                )
            for percentile in OPERATIVE_FEATURES_PERCENTILES:
                y_hat_replaced[percentile] = np.concatenate(y_hat_replaced[percentile])

            # Iterate through scale factors and bootstrap each
            for scale_factor, y_hat in y_hat_scale_factors.items():

                # Generate bootstrap distributions
                (y_bootstrap, y_hat_bootstrap) = generate_bootstrap_distributions(
                    y=y_test,
                    y_hat=y_hat,
                    seed=args.seed,
                    bootstrap_samplings=args.bootstrap_samplings,
                )

                (_tprs, aucs_this_scale_factor) = get_bootstrap_metrics(
                    y_bootstrap=y_bootstrap,
                    y_hat_bootstrap=y_hat_bootstrap,
                )

                # Calculate percentiles
                auc_percentiles_this_scale_factor = calculate_percentiles(
                    arr=aucs_this_scale_factor,
                )

                auc_percentiles[scale_factor] = auc_percentiles_this_scale_factor

            # Convert AUC versus scale factor dictionary into dataframe
            df = pd.DataFrame(auc_percentiles).transpose().round(args.decimals)
            df = df.rename_axis("scale_factor").reset_index()
            df = df[["scale_factor", "median", "lower", "upper"]]
            df = df.round(args.decimals)

            # Log dataframe
            log_dataframe(
                log_text=f"{feature_name} // {model_name} // AUCs with 95% CIs",
                df=df,
            )

            # Save AUC versus scale factor dataframe to CSV
            save_dataframe_to_csv(
                args=args,
                df=df,
                fname=f"{feature_name}_{model_name}.csv",
                keep_index=False,
            )

            # Plot sensitivity analysis for this model: AUC
            plot_sensitivity_analysis_auc(
                axes=axes,
                df=df,
                model_name=model_name,
                xticks=SCALE_FACTORS,
            )

            # Iterate through y_hat and isolate values of interest in new dict
            y_hat_subset = {}
            for percentile, y_hat_percentile in y_hat_replaced.items():
                y_hat_subset[percentile] = _subset_array_by_procedure(
                    array=y_hat_percentile,
                    procedure_types=procedure_types,
                )

            # Isolate feature values of desired subset of procedure types
            feature_subset = _subset_array_by_procedure(
                array=feature_array,
                procedure_types=procedure_types,
            )

            # Plot y_hat and feature distributions
            plot_distributions_feature_y_hat(
                args,
                feature_values=feature_subset,
                y_hat=y_hat_subset,
                model_name=model_name,
                feature_name=feature_name,
            )

        # Save sensitivity analysis figure
        sns.despine()
        full_feature_name = get_full_feature_name(feature_name)
        plt.title(f"Sensitivity analysis: {full_feature_name}", fontsize=24)
        fpath = os.path.join(
            args.output_directory,
            args.id,
            f"sensitivity_analysis_auc_{feature_name}.png",
        )
        plt.tight_layout()
        plt.savefig(fpath)
        plt.close()
        logging.info(f"Saved {fpath}")

# Imports: standard library
import os
import logging
import multiprocessing as mp
from typing import Dict, List, Union
from collections import defaultdict

# Imports: third party
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Imports: ml4sts
from ml4sts.definitions import (
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURES,
    PREOPERATIVE_FEATURES,
    DESCRIPTIVE_FEATURE_NAMES,
    OPERATIVE_FEATURES_LOOKUP,
)


def select_cohort(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    if cohort == "cabg":
        return df[(df["opcab"] == 1) & (df["opvalve"] == 0) & (df["opother"] == 0)]
    elif cohort == "valve":
        return df[(df["opcab"] == 0) & (df["opvalve"] == 1) & (df["opother"] == 0)]
    elif cohort == "cabg-valve":
        return df[(df["opcab"] == 1) & (df["opvalve"] == 1) & (df["opother"] == 0)]
    elif cohort == "major":
        return df[
            ((df["opcab"] == 0) & (df["opvalve"] == 0) & (df["opother"] == 0))
            | ((df["opcab"] == 1) & (df["opvalve"] == 0) & (df["opother"] == 0))
            | ((df["opcab"] == 1) & (df["opvalve"] == 1) & (df["opother"] == 0))
        ]
    elif cohort == "other":
        return df[df["opother"] == 1]
    elif cohort == "office":
        return df[(df["status"] < 3) & (df["carshock"] == 2)]
    else:
        raise ValueError(f"{cohort} is not a valid cohort name")


def generate_crossfold_indices(crossfolds: int, seed: float, y: np.array) -> tuple:
    # Initialize stratified crossfold objects
    skf = StratifiedKFold(n_splits=crossfolds, random_state=seed, shuffle=True)

    # Initialize list of lists
    idx_train_folds = []
    idx_test_folds = []

    # Generate dummy feature array
    x = np.zeros(len(y))

    # Iterate over each fold's list of train and test indices and append to big list
    for idx_train, idx_test in skf.split(x, y):
        idx_train_folds.append(idx_train)
        idx_test_folds.append(idx_test)
    return skf, idx_train_folds, idx_test_folds


def get_crossfold_indices_original_df(
    idx_original_remaining: np.array,
    idx_subset_train_folds: List[np.array],
    idx_subset_test_folds: List[np.array],
) -> np.array:
    """Get crossfold indices in terms of original dataframe"""
    # Initialize list of lists to store indices of original dataframe
    idx_original_train_folds = []
    idx_original_test_folds = []

    # Iterate over each fold's list of train and test indices
    for idx_subset_train, idx_subset_test in zip(
        idx_subset_train_folds,
        idx_subset_test_folds,
    ):
        idx_original_train_folds.append(idx_original_remaining[idx_subset_train])
        idx_original_test_folds.append(idx_original_remaining[idx_subset_test])
    return idx_original_train_folds, idx_original_test_folds


def save_dataframe_to_csv(
    args,
    df: pd.DataFrame,
    fname: str,
    keep_index: bool = False,
):
    if not fname.endswith(".csv"):
        fname += ".csv"
    fpath = os.path.join(args.output_directory, args.id, fname)
    df.to_csv(fpath, index=keep_index)
    logging.info(f"Saved {fpath}")


def get_full_feature_name(feature_name: str) -> str:
    return DESCRIPTIVE_FEATURE_NAMES[OPERATIVE_FEATURES_LOOKUP[feature_name]].lower()


def create_shared_array(data: np.array) -> tuple:
    # Declare RawArray of appropriate size
    data_raw = mp.RawArray("d", data.size)

    # Wrap x_raw as an numpy array so we can easily manipulates its data
    data_np = np.frombuffer(data_raw).reshape(data.shape)

    # Copy data from original numpy array to shared array
    np.copyto(data_np, data)

    # Return RawArray data and its shape as a tuple
    return data_raw, data.shape


def format_categorical_and_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical feature, drop redundant feature column for
    binary features, and update df"""
    # Reset indices of new df so indices align
    df.reset_index(drop=True, inplace=True)

    # Iterate through each categorical feature in this dataframe
    for feature in df.columns:

        # If feature is categorical one-hot encode it
        if feature in CATEGORICAL_FEATURES:

            # Iterate through possible values for this categorical feature
            for categorical_value in CATEGORICAL_FEATURES[feature]:

                # Initialize new series of zeros
                new_column = pd.Series(np.zeros(df.shape[0]))

                # Set positive rows to 1
                new_column[df[feature] == categorical_value] = 1

                # Append colmn to dataframe
                df[f"{feature}_{categorical_value}"] = new_column

            # Remove original feature from dataframe
            df = df.drop(columns=feature)
        else:
            if min(df[feature]) == 1 and max(df[feature]) == 2:
                # Shift encoding of feature from {1: yes, 2: no} -> {0: no, 1: yes}
                # by replacing all values of 2 with 0
                df.loc[:, feature] = df.loc[:, feature].replace(2, 0)

    # Re-index columns in alphabetical order
    df = df.reindex(columns=sorted(df.columns))

    return df


def get_continuous_features_mask(df: pd.DataFrame) -> np.array:
    """Get Boolean mask by comparing each feature to defined continuous features"""
    return np.array([feature in CONTINUOUS_FEATURES for feature in df.columns])


def get_col_from_csv(path_to_csv: str, key: str) -> np.array:
    df = pd.read_csv(path_to_csv, encoding="latin-1", low_memory=False)
    df = df.dropna()
    return df[key].to_numpy()


def log_dataframe(
    df: pd.DataFrame,
    format_scientific: bool = False,
    omit_columns: Union[str, list] = None,
):
    if omit_columns:
        df = df.drop(labels=omit_columns, axis="columns")
    if format_scientific:
        pd.options.display.float_format = "{:.2e}".format

    df_str = df.to_string().replace("\n", "\n\t")
    logging.info(f"\n\t{df_str}")
    pd.reset_option("display.float_format")


def save_results_to_disk(args, dict_to_save: dict):
    dirpath = os.path.join(args.output_directory, args.id)
    for key, val in dict_to_save.items():
        fpath = os.path.join(dirpath, f"{key}{args.results_file_extension}")
        np.savez_compressed(fpath, key=val)
        logging.info(f"Saved {key} to {fpath}")


def _get_y_from_results(results: dict) -> dict:
    """
    Get true labels as a dict of np.arrays indexed by results directory. Note
    there is no need to index by models, since y is the same for all models
    within a directory.

    If y is not available, return an empty dictionary.
    """
    y: Dict[str, dict] = {}
    for results_dir, result in results.items():
        if "y_test_outer_folds" in result:
            y[results_dir] = result["y_test_outer_folds"]
        else:
            logging.warning(
                "y_test_outer_folds not found in"
                f" results[{results_dir}]. Returning y as empty dict.",
            )
    return y


def _get_y_hat_from_results(results: dict, compare_mode: str) -> dict:
    """
    Iterate through results (indexed by results dirs, and then models),
    and save y_hat indexed in order determined by compare_mode:
    within_dirs: [results_dir][model_name]
    across_dirs: [model_name][results_dir]

    If y_hat is not available, return an empty dictionary.
    """
    y_hat: Dict[str, dict] = {}

    for results_dir, result in results.items():
        if "y_hat_test_outer_folds" in result:
            for model_name in result["y_hat_test_outer_folds"]:
                if compare_mode == "within_dirs":
                    if results_dir not in y_hat:
                        y_hat[results_dir] = {}
                    y_hat[results_dir][model_name] = result["y_hat_test_outer_folds"][
                        model_name
                    ]

                elif compare_mode == "across_dirs":
                    if model_name not in y_hat:
                        y_hat[model_name] = {}
                    y_hat[model_name][results_dir] = result["y_hat_test_outer_folds"][
                        model_name
                    ]
        else:
            logging.warning(
                "y_hat_test_outer_folds not found in"
                f" results[{results_dir}]. Returning y_hat as empty dict.",
            )
    return y_hat


def _get_obj_from_results(results: dict, key: str) -> dict:
    obj = {}
    for results_dir, result in results.items():
        if key in result:
            obj[results_dir] = result[key]
    return obj


def _get_models_from_results(results: dict) -> dict:
    models = {}
    for results_dir, result in results.items():
        if "best_models" in result:
            models[results_dir] = result["best_models"]
    return models


def _get_scalers_from_results(results: dict) -> dict:
    scalers = {}
    for results_dir, result in results.items():
        if "scaler" in result:
            scalers[results_dir] = result["scaler"]
    return scalers


def _get_calibration_models_from_results(results: dict) -> dict:
    calibration_models = {}
    for results_dir, result in results.items():
        if "calibration_models" in results:
            calibration_models[results_dir] = result["calibration_models"]
    return calibration_models


def load_results(args) -> dict:
    results = defaultdict()

    for directory in args.path_to_results:
        logging.info(f"Loading results from {directory}")

        if not os.path.exists(directory):
            raise ValueError(f"{directory} does not exist")

        directory_name = os.path.basename(os.path.normpath(directory))
        results[directory_name] = {}

        # Load contents from this results dir into dict
        for root, _path_to_files, fnames in os.walk(directory):
            for fname in fnames:

                # Get arrays, models, and scalers
                if fname.endswith(args.results_file_extension):
                    fpath = os.path.join(root, fname)
                    data = np.load(fpath, allow_pickle=True)

                    # If data["key"] contains a dict, this try will succeed
                    try:
                        results[directory_name][
                            fname.replace(args.results_file_extension, "")
                        ] = data["key"].item()

                    # Else, it was a numpy array and can be extracted directly
                    except:
                        try:
                            results[directory_name][
                                fname.replace(args.results_file_extension, "")
                            ] = data["key"]
                        except:
                            raise ValueError(f"Problem loading data from {fname}")
                    logging.info(f"Loaded data from {fname}")

    # Reverse nesting of dictionary from:
    # dict[results_dir][object]
    #     to
    # dict[object][results_dir]
    flipped = defaultdict(dict)
    for key, val in results.items():
        for subkey, subval in val.items():
            flipped[subkey][key] = subval
    return flipped

# Imports: standard library
import logging

# Imports: third party
import numpy as np
import pandas as pd

# Imports: ml4sts
from ml4sts.utils import log_dataframe


def _sample_random_hyperparameter(
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
    scaling: str = "linear",
    primitive_type: np.dtype = float,
    categorical_options: list = None,
):
    if scaling not in ["linear", "logarithmic"]:
        raise ValueError(f"Invalid scaling parameter: {scaling}")

    if categorical_options:
        return np.random.choice(categorical_options)

    if scaling == "logarithmic":
        lower_bound = np.log(lower_bound)
        upper_bound = np.log(upper_bound)

    if primitive_type == float:
        sampled_value = np.random.uniform(low=lower_bound, high=upper_bound, size=1)
    elif primitive_type == int:
        sampled_value = np.random.randint(
            low=lower_bound,
            high=upper_bound + 1,
            size=1,
        )
    else:
        raise ValueError(f"Invalid primitive type: {primitive_type}")

    if scaling == "logarithmic":
        sampled_value = np.exp(sampled_value)

    # Return the value inside numpy array
    return sampled_value.item()


def generate_hyperparameters(
    seed: int,
    model_name: str,
    random_samplings: int,
    smote: bool,
) -> list:
    hp = []
    # Reset random seed to the same value each time you call this function;
    # this ensures the random hyperparameters with random_samplings == n
    # contain all the prior values called with random_samplings == m where m < n
    np.random.seed(seed)
    for _random_sampling in range(random_samplings):
        if model_name == "logreg":
            hp_this_sampling = {}
            hp_this_sampling["c"] = _sample_random_hyperparameter(
                lower_bound=0.01,
                upper_bound=0.1,
                scaling="logarithmic",
                primitive_type=float,
            )
            hp_this_sampling["l1_ratio"] = _sample_random_hyperparameter(
                lower_bound=0.01,
                upper_bound=0.95,
                scaling="linear",
                primitive_type=float,
            )
        elif model_name == "svm":
            hp_this_sampling = {}
            hp_this_sampling["c"] = _sample_random_hyperparameter(
                lower_bound=0.001,
                upper_bound=0.1,
                scaling="logarithmic",
                primitive_type=float,
            )
        elif model_name == "randomforest":
            hp_this_sampling = {}
            hp_this_sampling["n_estimators"] = _sample_random_hyperparameter(
                lower_bound=250,
                upper_bound=500,
                scaling="linear",
                primitive_type=int,
            )
            hp_this_sampling["max_depth"] = _sample_random_hyperparameter(
                lower_bound=5,
                upper_bound=11,
                scaling="linear",
                primitive_type=int,
            )
            hp_this_sampling["min_samples_split"] = _sample_random_hyperparameter(
                lower_bound=3,
                upper_bound=10,
                scaling="linear",
                primitive_type=int,
            )
            hp_this_sampling["min_samples_leaf"] = _sample_random_hyperparameter(
                lower_bound=6,
                upper_bound=11,
                scaling="linear",
                primitive_type=int,
            )
        elif model_name == "xgboost":
            hp_this_sampling = {}

            hp_this_sampling["n_estimators"] = _sample_random_hyperparameter(
                lower_bound=20,
                upper_bound=100,
                scaling="linear",
                primitive_type=int,
            )
            hp_this_sampling["max_depth"] = _sample_random_hyperparameter(
                lower_bound=1,
                upper_bound=6,
                scaling="linear",
                primitive_type=int,
            )
        elif model_name == "mlp":
            hp_this_sampling = {}
            num_hidden_layers = _sample_random_hyperparameter(
                lower_bound=2,
                upper_bound=4,
                scaling="linear",
                primitive_type=int,
            )
            hidden_layer_sizes = []
            for _layer in range(num_hidden_layers):
                num_neurons = _sample_random_hyperparameter(
                    lower_bound=55,
                    upper_bound=300,
                    scaling="linear",
                    primitive_type=int,
                )
                hidden_layer_sizes.append(num_neurons)
            hp_this_sampling["hidden_layer_sizes"] = tuple(hidden_layer_sizes)
            hp_this_sampling["alpha"] = _sample_random_hyperparameter(
                lower_bound=0.1,
                upper_bound=5,
                scaling="logarithmic",
                primitive_type=float,
            )

        # Hyperparameters that are relevant across models
        hp_this_sampling["seed"] = seed
        hp_this_sampling["feature_select_ratio"] = _sample_random_hyperparameter(
            lower_bound=0.02,
            upper_bound=0.5,
            scaling="linear",
            primitive_type=float,
        )

        # If we perform SMOTE, randomly sample the sampling strategy float.
        if smote:
            hp_this_sampling["smote_sampling_strategy"] = _sample_random_hyperparameter(
                categorical_options=[None, 0.10, 0.25, 0.50],
            )
        # If not performing SMOTE, set the dict value to None. The value at this key
        # is used to determine if the parallelized model training function applies
        # SMOTE to inner training fold data.
        else:
            hp_this_sampling["smote_sampling_strategy"] = None
        hp.append(hp_this_sampling)
    return hp


def get_hp_cols(df: pd.DataFrame) -> list:
    not_hp_cols = [
        "model_name",
        "auc_train",
        "auc_test",
        "top_features_idx",
        "outer_fold",
        "inner_fold",
        "idx_sorted_coefs",
    ]
    return [col for col in df.columns if col not in not_hp_cols]


def get_best_hyperparameters(df: pd.DataFrame, model_name: str) -> dict:
    coefs_col_name = "idx_sorted_coefs"

    # Isolate df view for just that model
    df_model = df[df.model_name == model_name]

    # Determine hyperparameter column names
    hp_cols = get_hp_cols(df_model)

    # Get distinct hyperparameter combos, plus top features indices
    distinct_hps = df_model.drop_duplicates(subset=hp_cols)[hp_cols + [coefs_col_name]]

    # Initialize max test AUC
    auc_test_max = 0
    best_hp = None
    best_idx_sorted_coefs = None

    # Iterate over distinct hyperparameter combos
    for hp_row in distinct_hps.iterrows():

        # Isolate top features indices
        idx_sorted_coefs = hp_row[1][coefs_col_name]

        # Convert distinct hp row from series to df
        hp_row = hp_row[1][hp_cols].to_frame().T

        # Get values in dataframe for this unique hp combo
        df_this_hp = df.merge(hp_row)

        if df_this_hp["auc_test"].mean() > auc_test_max:
            auc_test_max = df_this_hp["auc_test"].mean()
            best_hp = hp_row
            best_idx_sorted_coefs = idx_sorted_coefs

    # Convert dataframe to dict; easier to log, and usable to initialize new models
    best_hp = best_hp.dropna(axis=1).to_dict("records")[0]

    logging.info(
        f"{model_name} - for this outer fold, the best mean inner test AUC ="
        f" {auc_test_max:0.2f} is achieved with hyperparameters:",
    )

    for param_key, param_val in best_hp.items():
        logging.info(f"\t{param_key} = {param_val:0.3e}")

    return best_hp, best_idx_sorted_coefs

# Imports: standard library
import os
import sys
import argparse
import operator
from datetime import datetime

# Imports: ml4sts
# Imports: first party
from ml4sts.logger import load_config
from ml4sts.definitions import OUTCOME_MAP


def parse_args(inp_args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mode",
        type=str,
        default="",
        help=(
            "summary_statistics, train, compare, sensitivity_analysis,"
            " predict. Default is blank"
        ),
    )
    parser.add_argument(
        "--bootstrap_samplings",
        type=int,
        default=10000,
        help="Number of bootstrap samplings. Default: 10000",
    )
    parser.add_argument(
        "--calibration_curve_log_scale",
        action="store_true",
        help="Plot calibration curve on log scale. Default: false (linear scale)",
    )
    parser.add_argument(
        "--calibration_probability_threshold",
        type=float,
        default=1,
        help=(
            "Predicted probabilities above this value are removed and replaced"
            " with this value. Default: 1"
        ),
    )
    parser.add_argument(
        "--cohort",
        type=str,
        help=(
            "Cohort to use, defined by case types."
            " Options: cabg, valve, cabg-valve, major, other, office",
        ),
    )
    parser.add_argument(
        "--crossfolds",
        type=int,
        default=5,
        help="Number of folds for cross-validation. Default: 5",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Number of decimal places for logged results. Default: 2",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="",
        nargs="+",
        help="Names of features to use for training models.",
    )
    parser.add_argument(
        "--hyperparameter_random_samplings",
        type=int,
        default=100,
        help=(
            "Number of hyperparameter combinations to try for random search."
            " Default: 100"
        ),
    )
    parser.add_argument(
        "--id",
        type=str,
        default="no_id",
        help=(
            "Identifier for this run; user-defined string to keep experiments"
            " organized. Default: no_id"
        ),
    )
    parser.add_argument(
        "--key_mrn",
        type=str,
        default="medrecn",
        help="Key (column) name of MRN in input CSV file. Default: medrecn",
    )
    parser.add_argument(
        "--key_surgery_date",
        type=str,
        default="surgdt",
        help="Key (column) name of surgery date in input CSV file. Default: surgdt",
    )
    parser.add_argument(
        "--logging_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help=(
            "logging level; overrides any configuration given in the logging"
            " configuration file."
        ),
    )
    parser.add_argument(
        "--models",
        type=str,
        default="logreg",
        nargs="+",
        help="Models to train: logreg, svm, randomforest, xgboost, mlp.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of workers to use in Multiprocess.",
    )
    parser.add_argument(
        "--ffs_auc_delta",
        type=float,
        default=0.01,
        help="Improvement in AUC that must be achieved to avoid contributing to"
        " iteration of patience. In other words, a model with M features must achieve"
        " an AUC of this much more than a model with M-1 features for the model to be"
        " considered better. If the model is not better, we iterate the patience"
        " counter (see ffs_patience arg below).",
    )
    parser.add_argument(
        "--ffs_patience",
        type=int,
        default=2,
        help="Number of additional features to try in FFS after performance stops"
        " improving. If the AUC improves after adding one more feature,"
        " patience resets. If the AUC does not improve after trying this many"
        " more features, finish FFS",
    )
    parser.add_argument(
        "--ffs_top_features",
        type=int,
        default=10,
        help="Number of features to use for forward feature selection.",
    )
    parser.add_argument(
        "--outcome",
        type=str,
        default="death",
        help=(
            "Outcome label to learn. Choose one from: death, stroke,"
            " renal_failure, prolonged_ventilation, deep_sternal_wound_infection,"
            " reoperation, any_morbidity, long_stay."
        ),
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default="~/dropbox/sts-ml/results",
        help="Path to output directory for pipeline runs.",
    )
    parser.add_argument(
        "--path_to_csv",
        type=str,
        default="~/dropbox/sts-data/sts-mgh.csv",
        help="Path to CSV file with features and labels.",
    )
    parser.add_argument(
        "--path_to_csv_predict",
        type=str,
        default="",
        help="Path to CSV file with features and labels to call and assess predict on.",
    )
    parser.add_argument(
        "--path_to_results",
        type=str,
        default=[],
        nargs="+",
        help="Path to 1+ directories containing results from train mode.",
    )
    parser.add_argument(
        "--predict_same_csv_as_train",
        action="store_true",
        help=(
            " Iterates through k models trained on outer folds, and applies each model"
            " to the test set of that fold."
        ),
    )
    parser.add_argument(
        "--results_file_extension",
        type=str,
        default=".npz",
        help="Extension for results files. Default: .npz",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2020,
        help="Value of random seed. Default: 2020",
    )
    parser.add_argument(
        "--smote",
        action="store_true",
        help="If this arg is given, perform Synthetic Minority Oversampling TEchnique (SMOTE)",
    )
    parser.add_argument(
        "--tablefmt",
        type=str,
        default="fancy_grid",
        help=(
            "Format of table generated by tableone. Options: github, grid,"
            " fancy_grid, rst, html, latex."
        ),
    )
    parser.add_argument(
        "--top_features_to_plot",
        type=int,
        default=15,
        help="Number of top features to plot.",
    )
    args = parser.parse_args(inp_args)
    _process_args(args)
    return args


def _process_args(args):
    now_string = datetime.now().strftime("%Y-%m-%d_%H-%M")
    args_file = os.path.join(
        args.output_directory,
        args.id,
        "arguments_" + now_string + ".txt",
    )
    command_line = f"""\n{' '.join(sys.argv)}\n"""

    print(command_line)

    if not os.path.exists(os.path.dirname(args_file)):
        os.makedirs(os.path.dirname(args_file))

    with open(args_file, "w") as args_file_object:
        args_file_object.write(command_line)
        for key, val in sorted(args.__dict__.items(), key=operator.itemgetter(0)):
            args_file_object.write(key + " = " + str(val) + "\n")

    load_config(
        args.logging_level,
        os.path.join(args.output_directory, args.id),
        "log_" + now_string,
    )

    # If user uses ~, expand to full path to home directory
    args.output_directory = os.path.expanduser(args.output_directory)

    # If input features are a list with one string separate by spaces,
    # e.g. FEATURES="preop cpb axc cabg valve", parse into list of distinct strings
    if len(args.features) == 1:
        args.features = args.features[0].split()

    if args.mode == "train":
        if not args.models:
            raise ValueError("train mode requires models")
        if not args.features:
            raise ValueError("train mode requires features")
        if args.outcome not in OUTCOME_MAP:
            raise ValueError(f'Outcome "{args.outcome}" not among defined outcomes')

    if args.mode == "compare":
        for idx, fpath in enumerate(args.path_to_results):
            args.path_to_results[idx] = os.path.expanduser(fpath)
            if not os.path.exists(fpath):
                raise ValueError(
                    f"Invalid args.path_to_results value: {fpath} does not exist",
                )

    if args.mode == "sensitivity_analysis":
        if len(args.path_to_results) > 1:
            raise ValueError(
                f"args.path_to_results_dir has >1 entry, but {args.mode} only"
                " supports 1 entry",
            )

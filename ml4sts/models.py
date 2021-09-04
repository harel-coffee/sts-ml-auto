# Imports: third party
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    auc,
    f1_score,
    roc_curve,
    recall_score,
    precision_score,
    brier_score_loss,
    classification_report,
    average_precision_score,
    balanced_accuracy_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Imports: ml4sts
from ml4sts.definitions import DESCRIPTIVE_FEATURE_NAMES
from ml4sts.hyperparameters import get_hp_cols


def initialize_model(model_name: str, hp: tuple):
    if model_name == "logreg":
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            class_weight=None,
            max_iter=5000,
            C=hp["c"],
            l1_ratio=hp["l1_ratio"],
        )
    elif model_name == "svm":
        model = MySVM(class_weight=None, C=hp["c"])
    elif model_name == "randomforest":
        model = RandomForestClassifier(
            class_weight=None,
            n_estimators=int(hp["n_estimators"]),
            max_depth=int(hp["max_depth"]),
            min_samples_split=int(hp["min_samples_split"]),
            min_samples_leaf=int(hp["min_samples_leaf"]),
        )
    elif model_name == "xgboost":
        model = XGBClassifier(
            n_estimators=int(hp["n_estimators"]),
            max_depth=int(hp["max_depth"]),
        )
    elif model_name == "mlp":
        model = MyMLP(hidden_layer_sizes=hp["hidden_layer_sizes"], alpha=hp["alpha"])
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    model.name = model_name
    return model


def add_fold_and_model_info_to_results(
    results: list,
    outer_fold: int,
    inner_fold: int,
    model_name: str,
) -> list:
    results_new = []
    for dict_for_one_hp_row in results:
        dict_for_one_hp_row["outer_fold"] = outer_fold + 1
        dict_for_one_hp_row["inner_fold"] = inner_fold + 1
        dict_for_one_hp_row["model_name"] = model_name
        results_new.append(dict_for_one_hp_row)
    return results_new


def format_cross_validation_results_as_df(
    results: list,
    models: list,
    decimals: int,
) -> pd.DataFrame:
    hps = []
    results_without_hps = []

    for result in results:
        hps.append(result["hp"])
        results_without_hps.append({key: result[key] for key in result if key != "hp"})

    # Convert list of dicts into dataframes and concatenate
    results_without_hps = pd.DataFrame(results_without_hps)
    hps = pd.DataFrame(hps)
    df = pd.concat([results_without_hps, hps], axis=1)

    # Re-order column names
    col_names = df.columns.tolist()
    col_names.remove("model_name")
    col_names.insert(0, "model_name")
    df = df[col_names]

    df.model_name = df.model_name.astype("category")
    df.sort_values(["model_name", "outer_fold", "inner_fold"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_feature_values(model) -> pd.DataFrame:
    if model.name == "logreg":
        return model.coef_.flatten()
    elif model.name == "svm":
        return model.LSVC.coef_.flatten()
    elif model.name == "randomforest" or model.name == "xgboost":
        return model.feature_importances_.flatten()
    else:
        raise ValueError(
            f"{model} lacks an attribute with feature coefficients or importances",
        )


def get_descriptive_feature_coefficients(
    coefs: np.array,
    feature_names: np.array,
    decimals: int,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {"feature": feature_names, f"coeff": coefs, f"coeff_abs": np.abs(coefs)},
    )
    # Sort feature_values by absolute value of coefficient (high to low)
    df = df.sort_values(f"coeff_abs", ascending=False)

    # Reset indices and avoid old index being added as col
    df.reset_index(drop=True, inplace=True)

    # Replace short feature variable name with descriptive name
    for feature in df.feature:

        # Get parts of features
        feature_parts = feature.split("_")

        # Look up descriptive name from root
        full_name = DESCRIPTIVE_FEATURE_NAMES[feature_parts[0]]

        # If feature was one-hot encoded, append full name
        if len(feature_parts) == 2:
            full_name += f"_{feature_parts[1]}"

        df.loc[df.feature == feature, "feature"] = full_name

    return df.round(decimals)


def evaluate_predictions(y: np.array, y_hat: np.array) -> dict:
    """
    AUC is calculated from roc_curve -> auc instead of roc_auc_score because we
    need the other outputs from roc_curve to threshold y_hat continous
    probabilities into binary labels.
    """
    metrics = {}

    fpr, tpr, thresholds = roc_curve(y, y_hat)

    # Determine optimal threshold via Youden's J statistic
    idx_optimal_threshold = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[idx_optimal_threshold]

    # Threshold predicted probabilities into binary labels
    y_hat_label = (y_hat > optimal_threshold).astype(float)

    metrics["auc"] = auc(fpr, tpr)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y, y_hat_label)
    classification_report_results = classification_report(
        y,
        y_hat_label,
        labels=[0, 1],
        output_dict=True,
    )
    metrics["sens"] = classification_report_results["1"]["recall"]
    metrics["spec"] = classification_report_results["0"]["recall"]
    metrics["auprc"] = average_precision_score(y, y_hat)
    metrics["precision"] = precision_score(y, y_hat_label)
    metrics["recall"] = recall_score(y, y_hat_label)
    metrics["f1"] = f1_score(y, y_hat_label)
    metrics["brier"] = brier_score_loss(y, y_hat, pos_label=y.max())
    metrics["g-mean"] = np.sqrt(metrics["sens"] * metrics["spec"])
    return metrics


class MySVM(LinearSVC):
    def __init__(self, class_weight, C, select_feature_ratio=None):
        super().__init__(
            penalty="l2",
            dual=False,
            class_weight=class_weight,
            C=C,
        )

    def fit(self, x, y):
        # Fit internal LinearSVC to obtain coefs
        self.LSVC = LinearSVC(
            penalty=self.penalty,
            dual=self.dual,
            class_weight=self.class_weight,
            C=self.C,
        )

        # Fit LinearSVC using all features
        self.LSVC.fit(x, y)

        # Wrap internal SVC in calibrated model to enable predict_proba
        self.calibratedmodel = CalibratedClassifierCV(base_estimator=self.LSVC, cv=5)

        # Train calibrated model on top features
        self.calibratedmodel.fit(x, y)
        return self

    def predict_proba(self, x):
        """
        Call predict_proba on internal linear calibrated SVC
        """
        return self.calibratedmodel.predict_proba(x)


class MyMLP(MLPClassifier):
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        alpha=0.0001,
        select_feature_ratio=None,
    ):
        super().__init__(
            solver="adam",
            early_stopping=True,
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
        )
        self.select_feature_ratio = select_feature_ratio
        self.initial_fit = True

    def fit(self, x, y):
        # Fit initial MLP using all features
        super(MyMLP, self).fit(x, y)

        # Calculate feature_importances_ = dy/dx | x=xmean=0
        self.feature_importances_ = []
        half_dx = x.std() / 20.0
        for i in range(x.shape[1]):
            x1 = np.zeros((1, x.shape[1]))
            x1[0, i] -= half_dx
            x2 = np.zeros((1, x.shape[1]))
            x2[0, i] += half_dx
            y1 = super(MyMLP, self).predict_proba(x1)[0, 1]
            y2 = super(MyMLP, self).predict_proba(x2)[0, 1]
            self.feature_importances_.append((y2 - y1) / (half_dx * 2.0))
        self.feature_importances_ = np.array(self.feature_importances_)

        # Sort feature importances from high to low
        idx_coefs_all = np.argsort(self.feature_importances_)[::-1]

        # Determine number of features from select_feature_ratio fraction
        num_features = int(self.select_feature_ratio * x.shape[1])
        if num_features > x.shape[1]:
            num_features = x.shape[1]

        # Isolate indices for top features
        idx_coefs = idx_coefs_all[:num_features]

        # Convert array of indices to Boolean mask
        self.feature_support = np.in1d(np.arange(x.shape[1]), idx_coefs)

        # Call fit of parent class using subset of selected features
        super().fit(x[:, self.feature_support], y)

        # Disable initial_fit so predict and predict_proba use feature_support
        self.initial_fit = False

        return self

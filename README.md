# STS-ML

Supervised learning to predict outcomes using the Society of Thoracic Surgeons database.

- [Setup](#setup)
- [Outcomes](#outcomes)
- [Recipes](#recipes)
    - [`summary_statistics`](#summary_statistics)
    - [`train`](#train)
    - [`ffs` (forward feature selection)](#ffs)
    - [`compare_across`](#compare_across)
    - [`compare_within`](#compare_within)
    - [`sensitivity_analysis`](#sensitivity_analysis)
    - [`predict`](#predict)
- [Data](#data)
- [Docker](#docker)

## Setup

Clone the repo:
```
git clone git@github.com:aguirre-lab/sts-ml.git
cd sts-ml
```

Install the Conda environment:
```bash
make setup
```

Activate it:
```bash
conda activate ml4sts
```

You may also have to install the local package:
```bash
pip install -e .
```

## Recipes

All recipes also produce `argumentres.txt` summarizing the arguments used when the mode was called, and `log.txt` which describes pipeline output in detail. Both file names are appended with the run date and time.

### Arguments
See [`arguments.py`](https://github.com/aguirre-lab/sts-ml/blob/master/ml4sts/arguments.py) for descriptions and default values.

### `summary_statistics`
Generates summary statistics for the patient population specified by `path_to_csv`, using [`tableone`](https://github.com/tompollard/tableone).

Example call:
```bash
ml4sts summary_statistics \
--path_to_csv $HOME/dropbox/sts-data/sts-mgh.csv \
--tablefmt fancy_grid \
--outcome death \
--features \
    age \
    carshock \
    chf \
    vdstena \
    vdstenm \
    wbc \
    weightkg \
--output_directory $HOME/dropbox/sts-ecg \
--id figures-and-tables
```

Outputs:
- `summary-statistics.csv`.

### `train`
Train models to predict operative mortality on STS data.

Example call:
```bash
INSTITUTION=mgh
ml4sts train \
--top_features_to_plot 10 \
--path_to_csv $HOME/dropbox/sts-data/$INSTITUTION.csv \
--hyperparameter_random_samplings 250 \
--bootstrap_samplings 10000 \
--cohort other \
--outcome death \
--features \
    age \
    carshock \
    chf \
    chrlungd \
    classnyh \
    creatlst \
    cva \
    cvd \
    cvdpcarsurg \
    cvdtia \
    diabetes \
    dialysis \
    ethnicity \
    gender \
    hct \
    hdef \
    heightcm \
    hypertn \
    immsupp \
    incidenc \
    infendty \
    medadp5days \
    medgp \
    medinotr \
    medster \
    numdisv \
    platelets \
    pocpci \
    pocpciin \
    prcab \
    prcvint \
    prvalve \
    pvd \
    raceasian \
    raceblack \
    racecaucasian \
    racenativeam \
    raceothernativepacific \
    resusc \
    status \
    vdinsufa \
    vdinsufm \
    vdinsuft \
    vdstena \
    vdstenm \
    wbc \
    weightkg \
--models logreg svm randomforest xgboost \
--calibration_curve_log_scale \
--output_directory $HOME/dropbox/sts-ml/results \
--id train-$INSTITUTION-test-$INSTITUTION
```

Outputs:
- `calibration_curve_{model_name}.png`
- `feature_coefs_or_importances_{model_name}.csv`
- `feature_coefs_or_importances_{model_name}.png`
- `metrics_compare_{baseline_model_name}_vs_{model_name}.csv`; `baseline_model_name` is usually `logreg`
- `calibration_metrics_{model_name}.csv`
- `roc_curve_{model_name}.png`
- `scaler.npz` - contains [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) fit on all training data `x`, used later for `predict`.
- `y_hat_test_outer_folds.npz`: dict keyed by `{model_name}` containing dict of numpy arrays of `y_hat` values.
- `y_test_outer_folds.npz`: numpy array of labels; equivalent to `y`.
- `best_models.npz`: dict keyed by `{model_name}` containing lists of `k` trained best models where `k` is the number of cross-folds.

### `ffs`
Perform forward feature selection (FFS) using regularized logistic regression and `k`-fold cross-validation.

FFS iterates through the number of top features to use (e.g. 1, 2, ..., `N`). Stopping conditions are whichever comes first:
1. All `N` features have been used.
1. Test AUC does not improve after adding `--ffs_patience` number of features, where improvement is defined by the argument `--ffs_auc_delta`.

For example, with `--ffs_patience 2` (default value) and the following number of top features : AUCs,

```
1 : 0.52
2 : 0.68
3 : 0.73
4 : 0.75
5 : 0.77
6 : 0.77
7 : 0.77
```
after adding the 7th top feature, the AUC has failed to improve for a second time. The optimal number of top features is five. The data is consequently subselected to include just these five most predictive features. A new model is initialized, trained on all data, and saved to disk for use by `predict` mode.


Example call:
```bash
INSTITUTION="mgh"
ml4sts ffs \
--ffs_top_features 20 \
--ffs_auc_delta 0.005 \
--path_to_csv $HOME/dropbox/sts-data/$INSTITUTION.csv \
--crossfolds 5 \
--bootstrap_samplings 10000 \
--cohort other \
--outcome death \
--features \
    age \
    carshock \
    chf \
    chrlungd \
    classnyh \
    creatlst \
    cva \
    cvd \
    cvdpcarsurg \
    cvdtia \
    diabetes \
    dialysis \
    ethnicity \
    gender \
    hct \
    hdef \
    heightcm \
    hypertn \
    immsupp \
    incidenc \
    infendty \
    medadp5days \
    medgp \
    medinotr \
    medster \
    numdisv \
    platelets \
    pocpci \
    pocpciin \
    prcab \
    prcvint \
    prvalve \
    pvd \
    raceasian \
    raceblack \
    racecaucasian \
    racenativeam \
    raceothernativepacific \
    resusc \
    status \
    vdinsufa \
    vdinsufm \
    vdinsuft \
    vdstena \
    vdstenm \
    wbc \
    weightkg \
--calibration_curve_log_scale \
--output_directory $HOME/dropbox/sts-ml/results \
--id ffs-train-$INSTITUTION-test-$INSTITUTION-delta-0.005
```

### `compare_across`
Iterate over models, then iterate over directories in `--path_to_results` (generated by `train` or `predict`).

For each model, compare each directory against the baseline directory (which is the first directory encountered).

Example call:
```bash
ml4sts compare_across \
--bootstrap_samplings 10000 \
--path_to_results \
    $HOME/dropbox/sts-ml/results/train-mgh-test-mgh-smote \
    $HOME/dropbox/sts-ml/results/train-mgh-test-mgh \
--output_directory $HOME/dropbox/sts-ml/results \
--id compare-across-train-mgh-test-mgh-vs-train-mgh-test-mgh-smote
```

This results in metrics `.csv` files:
1. Logistic regression, SMOTE (directory 1) vs original (directory 2).
1. Random forest, SMOTE (directory 1) vs original (directory 2).
1. SVM, SMOTE (directory 1) vs original (directory 2).
1. XGBoost, SMOTE (directory 1) vs original (directory 2).

### `compare_within`:
For a given single results directory, iterate over models, calculate various discrimination and calibration metrics, and compare against `logreg` as a baseline,

> This is performed by the `train` recipe too.

Example call:
```bash
ml4sts compare_within \
--bootstrap_samplings 10000 \
--path_to_results \
    $HOME/dropbox/sts-ml/results/train-bwh-test-bwh \
--output_directory $HOME/dropbox/sts-ml/results \
--id compare-within-train-bwh-test-bwh
```

### `predict`
Predict outcomes on a dataset using a model trained by `train` mode. The dataset should differ from that used to train the model. Also, the model should be trained on the same features as the feature arguments, e.g. a model trained only on pre-op features should be applied to only pre-operative data, which is achieved by setting `--features preop`.

Example call:
```bash
ml4sts predict \
--bootstrap_samplings 10000 \
--path_to_results $HOME/dropbox/sts-ml/results/train-mgh-test-mgh \
--path_to_csv_predict $HOME/dropbox/sts-data/bwh.csv \
--cohort other \
--outcome death \
--features \
    age \
    carshock \
    chf \
    chrlungd \
    classnyh \
    creatlst \
    cva \
    cvd \
    cvdpcarsurg \
    cvdtia \
    diabetes \
    dialysis \
    ethnicity \
    gender \
    hct \
    hdef \
    heightcm \
    hypertn \
    immsupp \
    incidenc \
    infendty \
    medadp5days \
    medgp \
    medinotr \
    medster \
    numdisv \
    platelets \
    pocpci \
    pocpciin \
    prcab \
    prcvint \
    prvalve \
    pvd \
    raceasian \
    raceblack \
    racecaucasian \
    racenativeam \
    raceothernativepacific \
    resusc \
    status \
    vdinsufa \
    vdinsufm \
    vdinsuft \
    vdstena \
    vdstenm \
    wbc \
    weightkg \
--models logreg randomforest svm xgboost \
--calibration_curve_log_scale \
--output_directory $HOME/dropbox/sts-ml/results \
--id train-mgh-test-bwh
```

Outputs:
- `calibration_curve_{calibration_method}_{model_name}.png`
- `calibration_metrics_{model_name}.csv`
- `metrics_compare_{baseline_model_name}_vs_{model_name}.csv` table with metrics and p-values
- `predicted_probabilities_{model_name}.csv`
- `roc_curve_{model_name}.png`

#### Apply `predict` on sub-cohort
To apply a model on a subset of the same dataset used to train the model, a cross-validation approach must be used to avoid overfitting.

The subset of data is obtained by removing `FEATURES_TO_REMOVE_FROM_TEST` (set in `defines.py`) from the dataset at `--path_to_csv_predict`.

Currently, the subset excludes patients with cardiogenic shock, emergent status, or salvage status, in order to investigate model performance on patients who might be seen in a clinic visit.

Example call:
```bash
ml4sts predict \
--predict_mode subset_cohort \
--bootstrap_samplings 10000 \
--path_to_results $HOME/dropbox/sts-ml/results/mgh-train-preop-cpb-axc-cabg-valve \
--path_to_csv_predict $HOME/dropbox/sts-data/mgh-others-features-labels.csv \
--outcome death \
--features preop cpb axc cabg valve \
--models logreg svm randomforest xgboost \
--output_directory $HOME/dropbox/sts-ml/results \
--id mgh-predict-death-on-mgh-office-subset-preop-cpb-axc-cabg-valve
```

### Apply `predict` on output from FFS
After performing forward feature selection (`ffs`), the trained model can be applied on a new data set via `predict_ffs` mode.

Unlike for models trained using `train` and applied to new data via `predict` mode, an `ffs` model cannot be used to predict on the same CSV via cross-validation to avoid optimistic performance.

```bash
ml4sts predict_ffs \
--bootstrap_samplings 10000 \
--path_to_results $HOME/dropbox/sts-ml/results/ffs-train-mgh-test-mgh-delta-0.005 \
--path_to_csv_predict $HOME/dropbox/sts-data/bwh.csv \
--cohort other \
--outcome death \
--features \
    age \
    carshock \
    chf \
    chrlungd \
    classnyh \
    creatlst \
    cva \
    cvd \
    cvdpcarsurg \
    cvdtia \
    diabetes \
    dialysis \
    ethnicity \
    gender \
    hct \
    hdef \
    heightcm \
    hypertn \
    immsupp \
    incidenc \
    infendty \
    medadp5days \
    medgp \
    medinotr \
    medster \
    numdisv \
    platelets \
    pocpci \
    pocpciin \
    prcab \
    prcvint \
    prvalve \
    pvd \
    raceasian \
    raceblack \
    racecaucasian \
    racenativeam \
    raceothernativepacific \
    resusc \
    status \
    vdinsufa \
    vdinsufm \
    vdinsuft \
    vdstena \
    vdstenm \
    wbc \
    weightkg \
--calibration_curve_log_scale \
--output_directory $HOME/dropbox/sts-ml/results \
--id ffs-train-mgh-test-bwh-delta-0.005
```


## MRN and surgery dates

`train` and `predict` recipes generate CSV files of the predicted probabilities of the outcome for each encounter (row).

This CSV file, `predicted_probabilities_{model_name}.csv`, includes two columns, `mrn` and `surgery_date`.

The user should specify the key (column) names for MRNs via `--key_mrn` and surgery dates via `--path_to_csv_predict`.

The values of these args should correspond to column names in the CSV files specified by:
- `--path_to_csv` for `train` mode
- `--path_to_csv_predict` for `predict` mode

```
| Variable     | Argument           | Default value |
|--------------|--------------------|---------------|
| MRN          | --key_mrn          | medrecn       |
| Surgery date | --key_surgery_date | surgdt        |

```

If no key given by `--key_mrn` can be found in the source CSV file, either a) the key is wrong, or b) there are no MRNs. Regardless, the `mrn` column in `predicted_probabilities_{model_name.csv}` will instead list indices of the original CSV.

This functionality allows a user to match predicted probabilities with rows of the original input CSV.

## Data

The STS database of each institution is maintained by the respective Division of Cardiac Surgery.

Raw data exported from the STS database is converted into a `.csv` format compatible with our pipelien via a script. Unfortunately, this script no longer exists. Re-creating this script is still in progress (see [issue #397 in `ml` repo](https://github.com/aguirre-lab/ml/issues/397)).

Our data is organized as follows:
1. Data is stored in an MGH Dropbox folder, `sts-data`.
1. Each institution has a file, `sts-$INSTITUTION.csv`, i.e. STS data for MGH is in `sts-mgh.csv`.
1. Each `sts-$INSTITUTION.csv` file contains STS features (covariates) and labels (outcomes) which are described in [ml4sts/definitions.py](https://github.com/aguirre-lab/sts-ml/blob/ecd65b8dbd3c552cc8803a83c6b7f51b90d62a69/ml4sts/definitions.py#L119).
    1. MRN is encoded as `medrecn`.
    1. Case types are binary variables:
        - CABG is `opcab`
        - Valve repairs or replacements is `opvalve`
        - AVR is `opavr`
        - Other (non-major) is `opother`
    1. The procedure type (`procedure_type`) is a string describing the type of cardiac surgical procedure.
    1. `sts-mgh.csv` contains 66 columns, but `sts-bwh.csv` contains 55. The BWH cohort lacks MRNs (`medrecn`), AVR status (`opavr`), and all outcomes except mortality (`mtopd`).

# Imports: standard library
import os
import logging

# Imports: third party
import pandas as pd
from tableone import TableOne

# Imports: ml4sts
# Imports: first party
from ml4sts.definitions import (
    OUTCOME_MAP,
    OPERATIVE_FEATURES,
    CONTINUOUS_FEATURES,
    DESCRIPTIVE_FEATURE_NAMES,
)


def summary_statistics(args):
    df = pd.read_csv(args.path_to_csv, low_memory=False)

    # Determine categorical features
    categorical_features = set(args.features) - set(CONTINUOUS_FEATURES)
    categorical_features -= set(OPERATIVE_FEATURES)

    mytable = TableOne(
        data=df,
        columns=args.features,
        categorical=list(categorical_features),
        groupby=OUTCOME_MAP[args.outcome],
        rename=DESCRIPTIVE_FEATURE_NAMES,
        label_suffix=True,
        missing=False,
        decimals={"age": 0, "platelets": 0},
        pval=True,
    )

    print(mytable.tabulate(tablefmt=args.tablefmt))

    fpath = os.path.join(args.output_directory, args.id, "summary_statistics.csv")
    mytable.to_csv(fpath)
    logging.info(f"Saved {fpath}")

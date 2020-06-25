from copy import deepcopy
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from imblearn.datasets import fetch_datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from experiments.cross_validation import OptimalSamplingClassifierCV
from utils.metrics import performance_summary

OUTPUT_PATH = "results.csv"
DATASETS = [
    "ecoli",
    "sick_euthyroid",
    # "car_eval_34",
    # "isolet",
    # "us_crime",
    # "yeast_ml8",
    # "libras_move",
    # "thyroid_sick",
    # "solar_flare_m0",
    # "coil_2000",
    # "wine_quality",
    # "yeast_me2",
    # "ozone_level",
    # "mammography",
    # "abalone_19"
]
MODEL_PARAMS = [
    dict(
        estimator_type=LogisticRegression,
        grids=dict(C=np.logspace(-4, 4, 20)),
        fixed_estimator_params=dict(n_jobs=-1, max_iter=1e4),
        n_folds=5,
    ),
    dict(
        estimator_type=DecisionTreeClassifier,
        grids=dict(ccp_alpha=np.logspace(-5, -1, 10), min_samples_leaf=[10, 20, 50], max_depth=np.arange(2, 6)),
        n_folds=5,
    ),
    # dict(
    #     estimator_type=RandomForestClassifier,
    #     grids=dict(max_features=[0.1, 0.3, 0.5]),
    #     fixed_estimator_params=dict(n_estimators=200, min_samples_leaf=20, max_depth=5, n_jobs=-1),
    #     n_folds=5,
    # )
]


def run_experiment(
        dataset: Dict[str, Union[np.ndarray, str]],
        params: Dict[str, any],
        output_path: Optional[str] = None,
        cost_scaling: float = 1.0,
        test_size: float = 0.3,
        verbose: bool = False
) -> pd.DataFrame:

    # Process data
    X = dataset["data"]
    y = dataset["target"] == 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    # Set up cross validation object
    cv = OptimalSamplingClassifierCV(
        **params,
        optimal_sampling_params=dict(positive_weight=cost_scaling / y_train.mean())
    )

    # Train model with optimal sampling and cross validation
    if verbose:
        print("Training model with optimal sampling")
    cv.fit(X_train, y_train)
    clf = deepcopy(cv.estimator)

    # Train baseline models with cross validation
    cv.optimize_sampling_proba = False
    if verbose:
        print("-" * 50)
        print("Training baseline model with nominal sampling")
    cv.fit(X_train, y_train, sampling_proba=None)
    nominal_baseline_clf = deepcopy(cv.estimator)
    if verbose:
        print("-" * 50)
        print("Training baseline model with balanced sampling")
    cv.fit(X_train, y_train, sampling_proba=0.5)
    balanced_baseline_clf = deepcopy(cv.estimator)

    # Save results
    results_df = pd.DataFrame(
        [
            performance_summary(clf, X_test, y_test),
            performance_summary(nominal_baseline_clf, X_test, y_test),
            performance_summary(balanced_baseline_clf, X_test, y_test)
        ]
    )
    results_df["dataset"] = dataset["DESCR"]
    if output_path:
        try:
            results_df = pd.concat([pd.read_csv(output_path, index_col=0), results_df])
        except FileNotFoundError:
            pass
    results_df.to_csv(output_path)
    return results_df


if __name__ == "__main__":
    database = fetch_datasets()
    for dataset_name in DATASETS:
        for params in MODEL_PARAMS:
            for cost_scaling in [0.2, 1, 5]:
                run_experiment(
                    dataset=database[dataset_name],
                    params=params,
                    output_path=OUTPUT_PATH,
                    cost_scaling=cost_scaling
                )
                print(
                    f"Completed experiment on {dataset_name} dataset with {params['estimator_type']} model and " + \
                    f"cost scaling {cost_scaling}"
                )

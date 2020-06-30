from copy import deepcopy
from typing import Dict, Optional, Union
import time

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
    "car_eval_34",
    "us_crime",
    "yeast_ml8",
    "libras_move",
    "thyroid_sick",
    "solar_flare_m0",
    "coil_2000",
    "wine_quality",
    "yeast_me2",
    "ozone_level",
    "mammography",
    "abalone_19"
]
MODEL_PARAMS = [
    dict(
        estimator_type=LogisticRegression,
        grids=dict(C=np.logspace(-4, 4, 20)),
        fixed_estimator_params=dict(n_jobs=-1, max_iter=5e3),
        n_folds=5,
    ),
    dict(
        estimator_type=DecisionTreeClassifier,
        grids=dict(ccp_alpha=np.logspace(-5, -1, 10), min_samples_leaf=[10, 20, 50], max_depth=np.arange(2, 6)),
        n_folds=5,
    )
]


def time_training(
        cv: OptimalSamplingClassifierCV,
        X: np.ndarray,
        y: np.ndarray,
        sampling_proba: Optional[float] = None
) -> float:
    start_time = time.time()
    cv.fit(X, y, sampling_proba=sampling_proba)
    end_time = time.time()
    return end_time - start_time


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
        optimal_sampling_params=dict(
            positive_weight=cost_scaling / y_train.mean(),
            calibrate_class_weight=(params["estimator_type"] not in [DecisionTreeClassifier, RandomForestClassifier])
        )
    )

    # Train model with optimal sampling and cross validation
    if verbose:
        print("Training model with optimal sampling")
    optimal_sampling_time = time_training(cv=cv, X=X_train, y=y_train)
    clf = deepcopy(cv.estimator)

    # Train baseline models with cross validation
    cv.optimize_sampling_proba = False
    if verbose:
        print("-" * 50)
        print("Training baseline model with nominal sampling")
    nominal_sampling_time = time_training(cv=cv, X=X_train, y=y_train, sampling_proba=None)
    nominal_baseline_clf = deepcopy(cv.estimator)
    if verbose:
        print("-" * 50)
        print("Training baseline model with balanced sampling")
    balanced_samling_time = time_training(cv=cv, X=X_train, y=y_train, sampling_proba=0.5)
    balanced_baseline_clf = deepcopy(cv.estimator)

    # Save results
    info = dict(
        dataset=dataset["DESCR"],
        cost_scaling=cost_scaling,
        n_features=X.shape[1],
        n_training_samples=y_train.shape[0]
    )
    results_df = pd.DataFrame(
        [
            performance_summary(
                clf=clf,
                X=X_test,
                y=y_test,
                info=dict(**info, sampling_method="optimal", training_time=optimal_sampling_time)
            ),
            performance_summary(
                clf=nominal_baseline_clf,
                X=X_test,
                y=y_test,
                info=dict(**info, sampling_method="nominal", training_time=nominal_sampling_time)
            ),
            performance_summary(
                clf=balanced_baseline_clf,
                X=X_test,
                y=y_test,
                info=dict(**info, sampling_method="balanced", training_time=balanced_samling_time)
            )
        ]
    )
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
            for cost_scaling in [0.5, 1, 2]:
                run_experiment(
                    dataset=database[dataset_name],
                    params=params,
                    output_path=OUTPUT_PATH,
                    cost_scaling=cost_scaling
                )
                model_name = str(params['estimator_type']).split(".")[-1]
                print(
                    f"Completed experiment on {dataset_name} dataset with {model_name} model "
                    + f"and cost scaling {cost_scaling}"
                )

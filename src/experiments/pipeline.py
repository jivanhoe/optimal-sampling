
import ssl
from copy import deepcopy
from typing import Dict, Union, NoReturn

import numpy as np
import pandas as pd
from imblearn.datasets import fetch_datasets
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import NearMiss, TomekLinks
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from algorithm.optimal_sampling import OptimalSamplingClassifier
from experiments.baseline import fit_sampling_baseline
from utils.metrics import performance_summary

ssl._create_default_https_context = ssl._create_unverified_context

OUTPUT_PATH = "results2.csv"
DATASETS = [
    #"ecoli",
    #"abalone",
    #"sick_euthyroid",
    #"spectrometer",
    #"car_eval_34",
    "us_crime",
    "yeast_ml8",
    "libras_move",
    "thyroid_sick",
    "solar_flare_m0",
    "wine_quality",
    "yeast_me2",
    "ozone_level",
    "mammography",
    "abalone_19",
    "spectrometer",
    "arrhythmia"
]
ESTIMATORS = [
    LogisticRegressionCV(
        Cs=10,
        cv=5,
        scoring="neg_log_loss",
        max_iter=10000,
        random_state=0
    ),
    GridSearchCV(
        estimator=DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=5,
            criterion="entropy",
            random_state=0
        ),
        scoring="neg_log_loss",
        param_grid=dict(ccp_alpha=np.logspace(-5, -1, 10)),
        cv=5
    ),
    RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=5,
        max_samples=0.5,
        max_features="sqrt",
        criterion="entropy",
        random_state=0
    ),
    MLPClassifier(
        hidden_layer_sizes=[64, 32],
        activation="tanh",
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=0
    )
]
COST_SCALINGS = [0.1, 0.2, 0.5, 1, 2, 5, 10]


def run_experiment(
        dataset: Dict[str, Union[np.ndarray, str]],
        output_path: str,
        estimator: BaseEstimator,
        cost_scaling: float = 1.0,
        n_folds: int = 5,
        verbose: bool = True
) -> NoReturn:

    # Process data
    X = preprocessing.scale(dataset["data"])
    y = dataset["target"] == 1
    nominal_proba = y.mean()
    positive_weight = (1 - nominal_proba) / nominal_proba * cost_scaling

    # Initialize classifier
    clf = OptimalSamplingClassifier(
        estimator=estimator,
        positive_weight=positive_weight,
        max_change=0.1,
        n_folds=5,
        max_iter=10,
        random_state=0,
        verbose=False
    )

    results = []
    for i, (train, test) in enumerate(
        StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=0
        ).split(X, y)
    ):

        if verbose:
            print(f"Running trial for fold {i+1}/{n_folds}")

        # Save results
        info = dict(
            dataset=dataset["DESCR"],
            cost_scaling=cost_scaling,
            n_features=X.shape[1],
            n_training_samples=y[train].shape[0],
            fold=i
        )
        for sampling_method, resampler, sampling_proba in [
            ("optimal", None, None),
            ("nominal", None, None),
            ("balanced", None, 0.5),
            ("smote", SMOTE(), None),
            ("adasyn", ADASYN(), None),
            ("near_miss", NearMiss(), None),
            ("tomeks_links", TomekLinks(), None),
        ]:
            if verbose:
                print(f"Training model with {sampling_method} sampling")
            if sampling_method == "optimal":
                clf_copy = deepcopy(clf)
                clf_copy.fit(X[train], y[train])
            else:
                clf_copy = fit_sampling_baseline(
                    clf=clf,
                    X=X[train],
                    y=y[train],
                    sampling_proba=sampling_proba,
                    resampler=resampler
                )
            results.append(
                performance_summary(
                    clf=clf_copy,
                    X=X[test],
                    y=y[test],
                    info=dict(
                        **info,
                        sampling_method=sampling_method,
                        total_train_time=clf_copy._total_train_time,
                        fit_time=clf_copy._fit_time,
                        iter_to_converge=clf_copy._iter_count if sampling_method == "optimal" else None
                    )
                )
            )
            print(f"Total train time: {clf_copy._total_train_time}")
            print(f"Total fit time: {clf_copy._fit_time}")
            print(f"Total iterations: {clf_copy._iter_count}")

    # Save results
    try:
        results_df = pd.concat([pd.read_csv(output_path, index_col=0), pd.DataFrame(results)])
    except FileNotFoundError:
        results_df = pd.DataFrame(results)
    results_df.to_csv(output_path)


if __name__ == "__main__":
    database = fetch_datasets()
    for dataset_name in DATASETS:
        for estimator_ in ESTIMATORS:
            for cost_scaling_ in COST_SCALINGS:
                try:
                    run_experiment(
                        dataset=database[dataset_name],
                        estimator=estimator_,
                        cost_scaling=cost_scaling_,
                        output_path=OUTPUT_PATH,
                        verbose=True
                    )
                    print(
                        f"Completed experiment on {dataset_name} dataset with {type(estimator_)} model "
                        + f"and cost scaling {cost_scaling_}"
                    )
                except BlockingIOError:
                    print(
                        f"Error running experiment on {dataset_name} dataset with {type(estimator_)} model "
                        + f"and cost scaling {cost_scaling_}"
                    )
                print("-" * 100)

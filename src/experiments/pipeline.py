import time
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from algorithm.optimal_sampling import OptimalSamplingClassifier
from experiments.baselines import fit_sampling_baseline
from utils.metrics import performance_summary


def auc(clf: BaseEstimator, X: np.ndarray, y: np.ndarray) -> float:
    return roc_auc_score(y, clf.predict_proba(X)[:, 1])


OUTPUT_PATH = "results.csv"
DATASETS = [
    "ecoli",
    "abalone",
    "sick_euthyroid",
    "spectrometer",
    "car_eval_34",
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
        scoring=auc,
        max_iter=10000,
        random_state=100
    ),
    GridSearchCV(
        estimator=DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=5,
            random_state=42
        ),
        scoring=auc,
        param_grid=dict(ccp_alpha=np.logspace(-5, -1, 10)),
        cv=5
    ),
    RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=5,
        max_samples=0.5,
        max_features="sqrt",
        random_state=42
    ),
    MLPClassifier(
        hidden_layer_sizes=[64, 32],
        activation="tanh",
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=42
    )
]
COST_SCALINGS = [0.2, 0.5, 1, 2, 5]


def run_experiment(
        dataset: Dict[str, Union[np.ndarray, str]],
        output_path: str,
        estimator: BaseEstimator,
        cost_scaling: float = 1.0,
        n_folds: int = 5,
        verbose: bool = False
) -> NoReturn:

    # Process data
    X = preprocessing.scale(dataset["data"])
    y = dataset["target"] == 1
    nominal_proba = y.mean()
    positive_weight = (1 - nominal_proba) / nominal_proba * cost_scaling

    results = []
    for i, (train, test) in enumerate(
        StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=42
        ).split(X, y)
    ):

        if verbose:
            print(f"Running trial for fold {i+1}/{n_folds}")

        # Set up cross validation object
        clf = OptimalSamplingClassifier(
            estimator=estimator,
            positive_weight=positive_weight,
            max_change=0.1,
            n_folds=5,
            max_iter=10,
            random_state=42,
            initial_guess="midpoint",
            verbose=False
        )

        # Train model with optimal sampling
        if verbose:
            print("Training model with optimal sampling")
        start_time = time.time()
        clf.fit(X=X[train], y=y[train])
        end_time = time.time()
        optimal_sampling_time = end_time - start_time
        iterations = clf.total_iter
        fit_time = clf.time_to_train
        # Train nominal sampling baseline
        if verbose:
            print("Training baseline model with nominal sampling")
        start_time = time.time()
        nominal_baseline_clf = fit_sampling_baseline(
            clf=clf,
            X=X[train],
            y=y[train],
            sampling_proba=None
        )
        end_time = time.time()
        nominal_sampling_time = end_time - start_time
        start_time = time.time()

        # Train balanced sampling baseline
        if verbose:
            print("Training baseline model with balanced sampling")
        balanced_baseline_clf = fit_sampling_baseline(
            clf=clf,
            X=X[train],
            y=y[train],
            sampling_proba=0.5
        )
        end_time = time.time()
        balanced_sampling_time = end_time - start_time

        # #Use SMOTE Oversampling technique
        start_time = time.time()
        if verbose:
            print("Training baseline model with SMOTE")
        X_train = X[train]
        y_train = y[train]
        sm = SMOTE()
        X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

        smote_baseline_clf = fit_sampling_baseline(
            clf=clf,
            X=X_resampled,
            y=y_resampled,
            sampling_proba=None
        )
        end_time = time.time()
        smote_sampling_time = end_time - start_time

        # #Use Adaptive Synthetic Oversampling
        if verbose:
            print("Training baseline model with ADASYN")
        start_time = time.time()
        ad = ADASYN()
        X_resampled, y_resampled = ad.fit_resample(X_train, y_train)

        adasyn_baseline_clf = fit_sampling_baseline(
            clf=clf,
            X=X_resampled,
            y=y_resampled,
            sampling_proba=None
        )
        end_time = time.time()
        adasyn_sampling_time = end_time - start_time

        #Use Near Miss Undersampling
        if verbose:
            print("Training baseline model with Near Miss")
        start_time = time.time()
        nm = NearMiss()
        X_resampled, y_resampled = nm.fit_resample(X_train, y_train)

        nearmiss_baseline_clf = fit_sampling_baseline(
            clf=clf,
            X=X_resampled,
            y=y_resampled,
            sampling_proba=None
        )
        end_time = time.time()
        nearmiss_sampling_time = end_time - start_time


        #Use Tomeks Links
        if verbose:
            print("Training baseline model with Tomek's Links")
        start_time = time.time()
        tl = TomekLinks()
        X_resampled, y_resampled = tl.fit_resample(X_train, y_train)

        tomeklinks_baseline_clf = fit_sampling_baseline(
            clf=clf,
            X=X_resampled,
            y=y_resampled,
            sampling_proba=None
        )
        end_time = time.time()
        tomeklinks_sampling_time = end_time - start_time

        # Save results
        info = dict(
            dataset=dataset["DESCR"],
            cost_scaling=cost_scaling,
            n_features=X.shape[1],
            n_training_samples=y[train].shape[0],
            fold=i
        )
        results += [
            performance_summary(
                clf=clf,
                X=X[test],
                y=y[test],
                info=dict(**info, sampling_method="optimal", training_time=optimal_sampling_time, fit_time = fit_time, iter_to_converge = iterations)
             ),
            performance_summary(
                clf=nominal_baseline_clf,
                X=X[test],
                y=y[test],
                info=dict(**info, sampling_method="nominal", training_time=nominal_sampling_time)
            ),
            performance_summary(
                clf=balanced_baseline_clf,
                X=X[test],
                y=y[test],
                info=dict(**info, sampling_method="balanced", training_time=balanced_sampling_time)
            ),
            performance_summary(
                clf=smote_baseline_clf,
                X=X[test],
                y=y[test],
                info=dict(**info, sampling_method="smote", training_time=smote_sampling_time)
            ),
            performance_summary(
                clf=adasyn_baseline_clf,
                X=X[test],
                y=y[test],
                info=dict(**info, sampling_method="adasyn", training_time=adasyn_sampling_time)
            ),
            performance_summary(
                clf=nearmiss_baseline_clf,
                X=X[test],
                y=y[test],
                info=dict(**info, sampling_method="nearmiss", training_time=nearmiss_sampling_time)
            ),
            performance_summary(
                clf=tomeklinks_baseline_clf,
                X=X[test],
                y=y[test],
                info=dict(**info, sampling_method="tomeklink", training_time=tomeklinks_sampling_time)
            )
        ]

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
                        verbose=False
                    )
                    print(
                        f"Completed experiment on {dataset_name} dataset with {type(estimator_)} model "
                        + f"and cost scaling {cost_scaling_}"
                    )
                except BlockingIOError:
                    print(
                        f"Error running experiment on {dataset_name} dataset with {estimator_} model "
                        + f"and cost scaling {cost_scaling_}"
                    )
                print("-" * 100)

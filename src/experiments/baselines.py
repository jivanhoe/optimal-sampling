from algorithm.optimal_sampling import OptimalSamplingClassifier
import numpy as np
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold

from typing import Optional


def fit_sampling_baseline(
        clf: OptimalSamplingClassifier,
        X: np.ndarray,
        y: np.ndarray,
        sampling_proba: Optional[float] = None
) -> OptimalSamplingClassifier:

    # Copy optimal sampling model
    baseline_clf = deepcopy(clf)

    # Resample
    nominal_proba = (y == baseline_clf.positive_class).mean()
    if sampling_proba:
        X, y = baseline_clf.resample_(X, y, sampling_proba=sampling_proba)
    else:
        sampling_proba = nominal_proba

    # Set sampling-related attributes
    if baseline_clf.calibrate_class_weight:
        baseline_clf.estimator.class_weight = {
            baseline_clf.positive_class: baseline_clf.positive_weight * nominal_proba / sampling_proba,
            baseline_clf.negative_class: baseline_clf.negative_weight * (1 - nominal_proba) / (1 - sampling_proba),
        }
    baseline_clf.sampling_proba = sampling_proba if sampling_proba else nominal_proba

    # Tune decision threshold
    if baseline_clf.tune_threshold:
        threshold_estimates = []
        for train, val in StratifiedKFold(
                n_splits=baseline_clf.n_folds,
                random_state=baseline_clf.random_state,
                shuffle=True
        ).split(X, y):
            baseline_clf.estimator.fit(X=X[train], y=y[train])
            threshold_estimates.append(
                baseline_clf.estimate_optimal_decision_threshold_(X=X[val], y=y[val])
            )
        baseline_clf.threshold = np.mean(threshold_estimates)
    else:
        baseline_clf.threshold = 0.5

    # Refit model on all data if desired and return
    if baseline_clf.refit or not baseline_clf.tune_threshold:
        baseline_clf.estimator.fit(X, y)
    return baseline_clf

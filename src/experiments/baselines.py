from algorithm.optimal_sampling import OptimalSamplingClassifier
import numpy as np
from copy import deepcopy

from typing import Optional


def fit_sampling_baseline(
        clf: OptimalSamplingClassifier,
        X: np.ndarray,
        y: np.ndarray,
        sampling_proba: Optional[float] = None
) -> OptimalSamplingClassifier:
    baseline_clf = deepcopy(clf)
    nominal_proba = (y == baseline_clf.positive_class).mean()
    if sampling_proba:
        X, y = baseline_clf.resample_(X, y, sampling_proba=sampling_proba)
    else:
        sampling_proba = nominal_proba
    baseline_clf.estimator.class_weight = {
        baseline_clf.positive_class: baseline_clf.positive_weight * nominal_proba / sampling_proba,
        baseline_clf.negative_class: baseline_clf.negative_weight * (1 - nominal_proba) / (1 - sampling_proba),
    }
    baseline_clf.sampling_proba = sampling_proba if sampling_proba else nominal_proba
    baseline_clf.estimator.fit(X, y)
    baseline_clf.tune_threshold_(X, y)
    return baseline_clf

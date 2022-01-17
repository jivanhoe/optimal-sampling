import time
from copy import deepcopy
from typing import Optional

import numpy as np
from imblearn.base import BaseSampler

from sklearn.model_selection import train_test_split

from algorithm.optimal_sampling import OptimalSamplingClassifier


def fit_sampling_baseline(
        clf: OptimalSamplingClassifier,
        X: np.ndarray,
        y: np.ndarray,
        sampling_proba: Optional[float] = None,
        resampler: Optional[BaseSampler] = None
) -> OptimalSamplingClassifier:

    # Copy optimal sampling model
    baseline_clf = deepcopy(clf)

    # Get train start time
    train_start_time = time.time()

    # Create a new train/validation set to tune the thresholds on the holdout data
    X, X_val, y, y_val = train_test_split(X, y, test_size=0.25, shuffle=True, stratify=y)

    # Resample training set only
    nominal_proba = (y == baseline_clf.positive_class).mean()
    if sampling_proba:
        X, y = baseline_clf._resample(X, y, sampling_proba=sampling_proba)
    elif resampler is not None:
        X, y = resampler.fit_resample(X, y)
        sampling_proba = (y == baseline_clf.positive_class).mean()
    else:
        sampling_proba = nominal_proba

    # Set sampling-related attributes
    if baseline_clf.calibrate_class_weight:
        baseline_clf.estimator.class_weight = {
            baseline_clf.positive_class: baseline_clf.positive_weight * nominal_proba / sampling_proba,
            baseline_clf.negative_class: baseline_clf.negative_weight * (1 - nominal_proba) / (1 - sampling_proba),
        }
    baseline_clf._sampling_proba = sampling_proba

    # Fit model
    fit_start_time = time.time()
    baseline_clf.estimator.fit(X, y)

    #Tune threshold on validation set
    best_threshold = 0
    best_cost = 1e12
    for threshold in np.arange(0.01, 1.0, 0.01):
        baseline_clf._threshold = threshold
        new_cost = baseline_clf.cost(X_val, y_val).mean()
        if new_cost < best_cost:
            best_cost = new_cost
            best_threshold = threshold
    baseline_clf._threshold = best_threshold
    baseline_clf._fit_time = time.time() - fit_start_time
    baseline_clf._total_train_time = time.time() - train_start_time

    return baseline_clf

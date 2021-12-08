import time
from copy import deepcopy
from typing import Optional

import numpy as np
from imblearn.base import BaseSampler

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

    # Resample
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
    baseline_clf._threshold = baseline_clf.negative_weight / (baseline_clf.positive_weight + baseline_clf.negative_weight)
    baseline_clf._fit_time = time.time() - fit_start_time
    baseline_clf._total_train_time  = time.time() - train_start_time

    return baseline_clf

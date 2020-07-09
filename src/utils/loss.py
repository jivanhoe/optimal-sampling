from typing import Callable, Type

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.svm import SVC, LinearSVC, NuSVC


def log_loss(y: np.ndarray, proba: np.ndarray) -> np.ndarray:
    proba = np.clip(proba, 1e-10, 1 - 1e-10)
    target = (y > np.min(y)).astype(float)
    return - target * np.log(proba) - (1 - target) * np.log(1 - proba)


def hinge_loss(y: np.ndarray, score: np.ndarray) -> np.ndarray:
    target = 2 * (y > np.min(y)).astype(float) - 1
    return np.maximum(0, 1 - target * score)


def select_loss(estimator_type: Type[BaseEstimator]) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    if estimator_type in (SVC, LinearSVC, NuSVC):
        return hinge_loss
    return log_loss
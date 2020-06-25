import numpy as np
from typing import Callable, Type


from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def log_loss(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    proba = np.clip(proba, 1e-10, 1 - 1e-10)
    target = (y > np.min(y)).astype(float)
    return - target * np.log(proba) - (1 - target) * np.log(1 - proba)


def hinge_loss(score: np.ndarray, y: np.ndarray) -> np.ndarray:
    target = 2 * (y > np.min(y)).astype(float) - 1
    return np.maximum(0, 1 - target * score)


def gini(_: np.ndarray, proba: np.ndarray) -> np.ndarray:
    return 1 - proba ** 2 - (1 - proba) ** 2


def select_loss(estimator_type: Type[BaseEstimator]) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    if estimator_type in (LogisticRegression, LogisticRegressionCV, GradientBoostingClassifier, DecisionTreeClassifier,
                          RandomForestClassifier, KNeighborsClassifier):
        return log_loss
    elif estimator_type in (SVC, LinearSVC, NuSVC):
        return hinge_loss
    else:
        raise NotImplementedError(
            f"Automatic loss function selection not implemented for {estimator_type} estimators - please specify loss function."
        )



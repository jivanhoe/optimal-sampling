from typing import Union, Optional, Callable, Tuple, NoReturn

import numpy as np
from sklearn.base import ClassifierMixin
from functools import wraps


class OptimalSampler:

    def __init__(
            self,
            clf: ClassifierMixin,
            loss: Callable[[np.ndarray, np.ndarray], np.ndarray],
            X: np.ndarray,
            y: np.ndarray,
            positive_weight: float = 1.0,
            negative_weight: float = 1.0,
            positive_class: Union[int, bool] = True,
            negative_class: Union[int, bool] = False,
            max_iter: int = 10,
            termination_tol: float = 1e-2,
            use_oob_validation: bool = False
    ):
        self.clf = clf
        self.loss = loss
        self.X = X
        self.y = y
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.positive_class = positive_class
        self.negative_class = negative_class
        self.max_iter = max_iter
        self.termination_tol = termination_tol
        self.use_oob_validation = use_oob_validation

    def get_optimal_sampling_proba(self) -> float:
        mask = self.y == self.positive_class
        predicted = self.clf.predict_proba(self.X) if self.use_oob_validation else self.clf.oob_decision_function_
        loss = self.loss(self.y, predicted) * np.where(mask, self.positive_weight, self.negative_weight)
        nominal_proba = np.mean(mask)
        return max(
            1 - np.sqrt((1 - nominal_proba) * np.mean(loss ** 2 * ~mask) / np.mean(loss)),
            nominal_proba
        )

    def sample(self, sampling_proba: float, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:

        # Select all positive samples
        positive_mask = self.y == self.positive_class

        # Select random subset of negative samples
        if seed is not None:
            np.random.seed(seed)
        n_samples = self.y.shape[0]
        n_negative_samples = int(np.ceil(n_samples * (1 - np.mean(positive_mask)) / (1 - sampling_proba)))
        negative_mask = np.zeros(n_samples, dtype=bool)
        negative_mask[np.random.permutation(np.argwhere(~positive_mask))[:n_negative_samples]] = True

        # Return sample
        mask = positive_mask | negative_mask
        return self.X[mask], self.y[mask]

    def optimize(self) -> float:
        nominal_proba =  np.mean(self.y == self.positive_class)
        prev_sampling_proba, sampling_proba = 1, nominal_proba
        for i in range(self.max_iter):
            X, y = self.sample(sampling_proba=sampling_proba, seed=i)
            self.clf.class_weight = {
                self.positive_class: self.positive_weight * nominal_proba / sampling_proba,
                self.negative_class: self.negative_class * (1 - nominal_proba) / (1 - sampling_proba),
            }
            self.clf.fit(X, y)
            prev_sampling_proba, sampling_proba = sampling_proba, self.get_optimal_sampling_proba()
            if abs(prev_sampling_proba - sampling_proba) < self.termination_tol:
                break
        self.clf.optimal_sampling_proba_ = sampling_proba




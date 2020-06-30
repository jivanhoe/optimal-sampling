from typing import Union, Optional, Callable, Tuple, NoReturn

import numpy as np
from sklearn.base import BaseEstimator

from utils.loss import select_loss


class OptimalSamplingClassifier(BaseEstimator):

    def __init__(
            self,
            estimator: BaseEstimator,
            loss: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
            positive_weight: float = 1.0,
            negative_weight: float = 1.0,
            positive_class: Union[int, bool] = True,
            negative_class: Union[int, bool] = False,
            max_iter: int = 10,
            termination_tol: float = 1e-2,
            tune_threshold: bool = True,
            use_decision_function: bool = False,
            calibrate_class_weight: bool = True,
            verbose: bool = False,
            random_state: Optional[int] = None
    ):
        self.estimator = estimator
        self.loss = loss if loss else select_loss(type(estimator))
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.positive_class = positive_class
        self.negative_class = negative_class
        self.max_iter = max_iter
        self.termination_tol = termination_tol
        self.tune_threshold = tune_threshold
        self.use_decision_function = use_decision_function
        self.calibrate_class_weight = calibrate_class_weight
        self.verbose = verbose
        self.sampling_proba = None
        self.threshold = None
        self.random_state = random_state

    def resample_(
            self,
            X: np.ndarray,
            y: np.ndarray,
            sampling_proba: float,
    ) -> Tuple[np.ndarray, np.ndarray]:

        # Select all positive samples
        positive_mask = y == self.positive_class

        # Select random subset of negative samples
        n_samples = y.shape[0]
        n_negative_samples = int(
            np.ceil(n_samples * np.mean(positive_mask) / sampling_proba * (1 - sampling_proba)))
        negative_mask = np.zeros(n_samples, dtype=bool)
        negative_mask[np.random.permutation(np.argwhere(~positive_mask))[:n_negative_samples]] = True

        # Return sample
        mask = positive_mask | negative_mask
        return X[mask], y[mask]

    def get_optimal_sampling_proba_(self, X: np.ndarray, y: np.ndarray) -> float:
        positive_mask = y == self.positive_class
        loss = self.weighted_loss(X, y)
        nominal_proba = positive_mask.mean()
        return max(
            1 - np.sqrt((1 - nominal_proba) * (loss ** 2 * ~positive_mask).mean()) / max(loss.mean(), 1e-10) - 1e-10,
            nominal_proba
        )

    def fit_estimator_with_optimal_sampling_(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        nominal_proba = (y == self.positive_class).mean()
        prev_sampling_proba, sampling_proba = 1, 0.5
        for i in range(self.max_iter):
            if self.verbose:
                print(f"Training with sampling probability: {'{0:.3f}'.format(sampling_proba)}")
            X_resampled, y_resampled = self.resample_(X, y, sampling_proba=sampling_proba)
            if self.calibrate_class_weight:
                self.estimator.class_weight = {
                    self.positive_class: self.positive_weight * nominal_proba / sampling_proba,
                    self.negative_class: self.negative_weight * (1 - nominal_proba) / (1 - sampling_proba),
                }
            else:
                self.estimator.class_weight = "balanced"
            self.estimator.fit(X_resampled, y_resampled)
            prev_sampling_proba, sampling_proba = sampling_proba, self.get_optimal_sampling_proba_(X, y)
            if abs(prev_sampling_proba - sampling_proba) < self.termination_tol:
                break
        self.sampling_proba = sampling_proba

    def tune_threshold_(self, X: np.ndarray, y: np.ndarray):
        nominal_proba = (y == self.positive_class).mean()
        X_resampled, y_resampled = self.resample_(X, y, sampling_proba=self.sampling_proba)
        sampling_weights = np.where(
            y_resampled == self.positive_class,
            nominal_proba / self.sampling_proba,
            (1 - nominal_proba) / (1 - self.sampling_proba)
        )
        if self.tune_threshold:
            predicted_proba = self.predict_proba(X_resampled)
            min_cost, best_threshold = np.inf, np.inf
            for threshold in np.unique(predicted_proba):
                self.threshold = threshold

                cost = (self.cost(X_resampled, y_resampled) * sampling_weights).mean()
                if cost < min_cost:
                    min_cost, best_threshold = cost, threshold
            self.threshold = best_threshold
        else:
            self.threshold = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.fit_estimator_with_optimal_sampling_(X, y)
        self.tune_threshold_(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.use_decision_function:
            return self.estimator.decision_function(X)
        else:
            return self.estimator.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.predict_proba(X) > self.threshold, self.positive_class, self.negative_class)

    def weighted_loss(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        weights = np.where(y == self.positive_class, self.positive_weight, self.negative_weight)
        return self.loss(self.predict_proba(X), y=y) * weights

    def cost(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        predicted = self.predict(X)
        positive_mask = y == self.positive_class
        return (self.positive_weight * positive_mask + self.negative_weight * ~positive_mask) * (predicted != y)





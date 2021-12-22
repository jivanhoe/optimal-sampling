import time
import warnings
from copy import deepcopy
from typing import Union, Optional, Callable, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from utils.loss import select_loss

warnings.filterwarnings("error")


class OptimalSamplingClassifier(BaseEstimator):

    def __init__(
            self,
            estimator: BaseEstimator,
            loss: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
            positive_weight: float = 1.0,
            negative_weight: float = 1.0,
            positive_class: Union[int, bool] = True,
            negative_class: Union[int, bool] = False,
            max_iter: int = 50,
            max_change: float = 0.1,
            termination_tol: float = 1e-2,
            rounding_tol: float = 1e-10,
            n_folds: float = 5,
            tune_threshold: bool = False,
            use_decision_function: bool = False,
            calibrate_class_weight: bool = True,
            initial_guess: str = "balanced",
            refit: bool = True,
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
        self.max_change = max_change
        self.termination_tol = termination_tol
        self.rounding_tol = rounding_tol
        self.n_folds = n_folds
        self.tune_threshold = tune_threshold
        self.use_decision_function = use_decision_function
        self.calibrate_class_weight = calibrate_class_weight
        self.initial_guess = initial_guess
        self.refit = refit
        self.verbose = verbose
        self.random_state = random_state
        self._cross_val_sampling_probas = None
        self._prev_sampling_probas = None
        self._nominal_proba = None
        self._sampling_proba = None
        self._threshold = None
        self._iter_count = None
        self._total_train_time = None
        self._fit_time = None

    def _resample(
            self,
            X: np.ndarray,
            y: np.ndarray,
            sampling_proba: float,
    ) -> Tuple[np.ndarray, np.ndarray]:

        # Set seed
        np.random.seed(self.random_state)

        # Select all positive samples
        positive_mask = y == self.positive_class

        # Select random subset of negative samples
        n_samples = y.shape[0]
        n_negative_samples = max(int(np.ceil(n_samples * np.mean(positive_mask) / sampling_proba * (1 - sampling_proba))), 2)
        negative_mask = np.zeros(n_samples, dtype=bool)
        negative_mask[np.random.permutation(np.argwhere(~positive_mask))[:n_negative_samples]] = True

        # Return sample
        mask = positive_mask | negative_mask
        return X[mask], y[mask]

    def _make_initial_guess(self, y: np.ndarray) -> float:
        if self.initial_guess == "balanced":
            return 0.5
        elif self.initial_guess == "nominal":
            return self._nominal_proba
        elif self.initial_guess == "midpoint":
            return (self._nominal_proba + 0.5) / 2
        elif self.initial_guess == "auto":
            odds = self._nominal_proba / (1 - self._nominal_proba)
            return float(np.clip(
                1 - 1 / (odds * self.positive_weight / self.negative_weight + 1),
                a_min=self._nominal_proba,
                a_max=0.5
            ))
        else:
            raise NotImplementedError(
                f"Unable to recognise initial guess strategy {self.initial_guess}"
                " - please choose from balanced, nominal, midpoint, or auto."
            )

    def _estimate_optimal_sampling_proba(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
    ) -> float:
        # Fit model on training data using incumbent sampling probability
        X_resampled, y_resampled = self._resample(X=X_train, y=y_train, sampling_proba=self._sampling_proba)
        fit_start_time = time.time()
        self.estimator.fit(X=X_resampled, y=y_resampled)
        self._fit_time += time.time() - fit_start_time

        # Estimate optimal sampling probability on validation set
        positive_mask = y_val == self.positive_class
        loss = self.weighted_loss(X_val, y_val)
        sampling_proba = 1 - (1 - self._nominal_proba) * np.sqrt((loss[~positive_mask] ** 2).mean()) / loss.mean()
        alt_sampling_proba = self._nominal_proba * np.sqrt((loss[positive_mask] ** 2).mean()) / loss.mean()
        chosen_sampling_proba = float(
            np.clip(
                sampling_proba if sampling_proba >= self._nominal_proba else alt_sampling_proba,
                a_min=self.rounding_tol,
                a_max=1 - self.rounding_tol
            )
        )
        return chosen_sampling_proba

    def fit(self, X: np.ndarray,  y: np.ndarray) -> None:

        # Set seed
        np.random.seed(self.random_state)

        # Initialize variables
        self._nominal_proba = (y == self.positive_class).mean()
        self._sampling_proba = self._make_initial_guess(y)
        self._total_train_time = 0
        self._fit_time = 0
        self._iter_count = 0
        self._prev_sampling_probas = []
        self._cross_val_sampling_probas = []
        train_start_time = time.time()

        for i in range(self.max_iter):
            if self.verbose:
                print(f"Training with sampling probability: {'{0:.3f}'.format(self._sampling_proba)}")

            # Update iteration count
            self._iter_count += 1

            # Recalibrate class weights used by model
            class_weight = {
                self.positive_class: self.positive_weight * self._nominal_proba / self._sampling_proba,
                self.negative_class: self.negative_weight * (1 - self._nominal_proba) / (1 - self._sampling_proba),
            } if self.calibrate_class_weight else "balanced"
            if type(self.estimator) == GridSearchCV:
                self.estimator.estimator.class_weight = class_weight
            else:
                self.estimator.class_weight = class_weight

            # Perform cross validation to estimate updated sampling probability
            updated_sampling_proba_estimates = []
            for train, val in StratifiedKFold(
                n_splits=self.n_folds,
                random_state=self.random_state,
                shuffle=True
            ).split(X, y):
                updated_sampling_proba_estimates.append(
                    self._estimate_optimal_sampling_proba(
                        X_train=X[train],
                        y_train=y[train],
                        X_val=X[val],
                        y_val=y[val]
                    )
                )
            self._cross_val_sampling_probas.append(updated_sampling_proba_estimates)
            # Update sampling probability for next iteration
            self._prev_sampling_probas.append(deepcopy(self._sampling_proba))
            self._sampling_proba = np.clip(
                np.mean(updated_sampling_proba_estimates),
                a_min=self._prev_sampling_probas[-1] - self.max_change,
                a_max=self._prev_sampling_probas[-1] + self.max_change
            )

            # Check termination condition
            if abs(self._prev_sampling_probas[-1] - self._sampling_proba) < self.termination_tol:
                break
            elif len(self._prev_sampling_probas) > 1:
                # Check for oscillation
                if abs(self._prev_sampling_probas[-2] - self._sampling_proba) < self.termination_tol:
                    self._sampling_proba = (self._prev_sampling_probas[-1] + self._sampling_proba) / 2
                    self.max_change = self.max_change / 2

        # Refit model
        if self.refit:
            X_resampled, y_resampled = self._resample(X, y, sampling_proba=self._sampling_proba)
            self.estimator.fit(X=X_resampled, y=y_resampled)

        # Set threshold
        self._threshold = self.negative_weight / (self.negative_weight + self.positive_weight)

        # Compute training time
        self._total_train_time = time.time() - train_start_time

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.use_decision_function:
            raise NotImplementedError()
        else:
            return self.estimator.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.predict_proba(X) > self._threshold, self.positive_class, self.negative_class)

    def weighted_loss(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        weights = np.where(y == self.positive_class, self.positive_weight, self.negative_weight)
        return self.loss(y, self.predict_proba(X)) * weights

    def cost(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.where(
            y == self.positive_class,
            self.positive_weight,
            self.negative_weight
        ) * (self.predict(X) != y)





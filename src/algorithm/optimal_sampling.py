from typing import Union, Optional, Callable, Tuple, NoReturn, List

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from utils.loss import select_loss
from copy import deepcopy

import warnings
warnings.filterwarnings('error')


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
            n_folds: float = 10,
            tune_threshold: bool = True,
            use_decision_function: bool = False,
            calibrate_class_weight: bool = True,
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
        self.n_folds = n_folds
        self.tune_threshold = tune_threshold
        self.use_decision_function = use_decision_function
        self.calibrate_class_weight = calibrate_class_weight
        self.refit = refit
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

        # Set seed
        np.random.seed(self.random_state)

        # Select all positive samples
        positive_mask = y == self.positive_class

        # Select random subset of negative samples
        n_samples = y.shape[0]
        n_negative_samples = int(np.ceil(n_samples * np.mean(positive_mask) / sampling_proba * (1 - sampling_proba)))
        negative_mask = np.zeros(n_samples, dtype=bool)
        negative_mask[np.random.permutation(np.argwhere(~positive_mask))[:n_negative_samples]] = True

        # Return sample
        mask = positive_mask | negative_mask
        return X[mask], y[mask]

    def make_initial_guess_(self, y: np.ndarray) -> float:
        nominal_proba = (y == self.positive_class).mean()
        odds = nominal_proba / (1 - nominal_proba)
        return float(np.clip(
            1 - 2 / (odds * self.positive_weight / self.negative_weight + 1),
            a_min=nominal_proba,
            a_max=0.5
        ))

    def estimate_optimal_sampling_proba_(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
    ) -> float:

        # Fit model on training data using incumbent sampling probability
        nominal_proba = (y_train == self.positive_class).mean()
        X_resampled, y_resampled = self.resample_(X=X_train, y=y_train, sampling_proba=self.sampling_proba)
        self.estimator.fit(X=X_resampled, y=y_resampled)

        # Estimate optimal sampling probability on validation set
        positive_mask = y_val == self.positive_class
        loss = self.weighted_loss(X_val, y_val)
        nominal_proba = positive_mask.mean()
        return max(
            1 - np.sqrt((1 - nominal_proba) * (loss ** 2 * ~positive_mask).mean()) / max(loss.mean(), 1e-10) - 1e-10,
            nominal_proba
        )

    def estimate_optimal_decision_threshold_(self, X: np.ndarray, y: np.ndarray) -> float:

        # Resample data
        nominal_proba = (y == self.positive_class).mean()
        X_resampled, y_resampled = self.resample_(X, y, sampling_proba=self.sampling_proba)
        sampling_weights = np.where(
            y_resampled == self.positive_class,
            nominal_proba / self.sampling_proba,
            (1 - nominal_proba) / (1 - self.sampling_proba)
        )

        # Perform grid search for best threshold
        predicted_proba = self.predict_proba(X_resampled)
        min_cost, best_threshold = np.inf, np.inf
        for threshold in np.unique(predicted_proba):
            self.threshold = threshold
            cost = (self.cost(X_resampled, y_resampled) * sampling_weights).mean()
            if cost < min_cost:
                min_cost, best_threshold = cost, threshold
        self.threshold = best_threshold
        return best_threshold

    def fit(self, X: np.ndarray,  y: np.ndarray) -> NoReturn:

        # Set seed
        np.random.seed(self.random_state)

        # Initialize variables
        self.sampling_proba = self.make_initial_guess_(y)
        prev_sampling_probas = []
        threshold_estimates = []
        terminate_early = False
        nominal_proba = (y == self.positive_class).mean()

        for i in range(self.max_iter):
            if self.verbose:
                print(f"Training with sampling probability: {'{0:.3f}'.format(self.sampling_proba)}")

            # Recalibrate class weights used by model
            class_weight = {
                self.positive_class: self.positive_weight * nominal_proba / self.sampling_proba,
                self.negative_class: self.negative_weight * (1 - nominal_proba) / (1 - self.sampling_proba),
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
                   self.estimate_optimal_sampling_proba_(
                        X_train=X[train],
                        y_train=y[train],
                        X_val=X[val],
                        y_val=y[val]
                    )
                )

                # Estimate decision threshold if terminating
                if (terminate_early or (i == self.max_iter - 1)) and self.tune_threshold:
                    threshold_estimates.append(
                        self.estimate_optimal_decision_threshold_(
                            X=X[val],
                            y=y[val]
                        )
                    )

            # Update sampling probability for next iteration
            prev_sampling_probas.append(deepcopy(self.sampling_proba))
            self.sampling_proba = np.clip(
                np.mean(updated_sampling_proba_estimates),
                a_min=prev_sampling_probas[-1] - self.max_change,
                a_max=prev_sampling_probas[-1] + self.max_change
            )

            # Check termination condition
            if terminate_early:
                break
            elif abs(prev_sampling_probas[-1] - self.sampling_proba) < self.termination_tol:
                terminate_early = True
            elif len(prev_sampling_probas) > 1:
                # Check for oscillation
                if abs(prev_sampling_probas[-2] - self.sampling_proba) < self.termination_tol:
                    self.sampling_proba = (prev_sampling_probas[-1] + self.sampling_proba) / 2
                    self.max_change = self.max_change / 2

        # Set sampling probability and decision threshold
        self.threshold = np.mean(threshold_estimates) if self.tune_threshold else 0.5

        # Refit model
        if self.refit:
            X_resampled, y_resampled = self.resample_(X, y, sampling_proba=self.sampling_proba)
            self.estimator.fit(X=X_resampled, y=y_resampled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.use_decision_function:
            return self.estimator.decision_function(X)
        else:
            return self.estimator.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.predict_proba(X) > self.threshold, self.positive_class, self.negative_class)

    def weighted_loss(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        weights = np.where(y == self.positive_class, self.positive_weight, self.negative_weight)
        return self.loss(y, self.predict_proba(X)) * weights

    def cost(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.where(
            y == self.positive_class,
            self.positive_weight,
            self.negative_weight
        ) * (self.predict(X) != y)





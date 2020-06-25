from typing import Iterable, Dict, NoReturn, Optional, Callable, Union, Type

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.base import BaseEstimator
from copy import deepcopy

from algorithm.optimal_sampling import OptimalSamplingClassifier
from utils.metrics import performance_summary
from experiments.baselines import fit_sampling_baseline


class OptimalSamplingClassifierCV:

    def __init__(
            self,
            estimator_type: Type[BaseEstimator],
            grids: Dict[str, Iterable[any]],
            fixed_estimator_params: Optional[Dict[str, any]] = None,
            optimal_sampling_params: Optional[Dict[str, any]] = None,
            scoring_metric: str = "weighted_loss",
            greater_is_better: bool = False,
            aggregate_by: Union[str, Callable[[pd.Series], float]] = "mean",
            n_folds: int = 10,
            random_state: Optional[int] = None,
            optimize_sampling_proba: bool = True,
            refit: bool = True,
            verbose: bool = False,
    ) -> NoReturn:
        self.estimator_type = estimator_type
        self.grids = grids
        self.fixed_estimator_params = fixed_estimator_params
        self.optimal_sampling_params = optimal_sampling_params
        self.scoring_metric = scoring_metric
        self.greater_is_better = greater_is_better
        self.aggregate_by = aggregate_by
        self.n_folds = n_folds
        self.random_state = random_state
        self.optimize_sampling_proba = optimize_sampling_proba
        self.refit = refit
        self.verbose = verbose
        self.results = None
        self.estimator = None

    def select_best_params(self, return_estimator: bool = False) -> Union[Dict[str, any], OptimalSamplingClassifier]:
        assert self.results is not None, "Parameters can only be selected after model is fitted."
        results_df = pd.DataFrame(self.results)
        scores = results_df.groupby("params_id")[self.scoring_metric].agg(self.aggregate_by)
        best_params_id = scores.idxmax() if self.greater_is_better else scores.idxmin()
        if return_estimator:
            return results_df[results_df["params_id"] == best_params_id].iloc[0]["clf"]
        else:
            return list(ParameterGrid(self.grids))[best_params_id]

    def fit(
            self,
            X: np.ndarray, 
            y: np.ndarray, 
            sampling_proba: Optional[float] = None
    ) -> NoReturn:
        self.results = []
        for i, (train, test) in enumerate(
            StratifiedKFold(
                    n_splits=self.n_folds,
                    random_state=self.random_state,
                    shuffle=True
            ).split(X, y)
        ):
            if self.verbose:
                print(f"Fitting models without fold {i+1}/{self.n_folds}")
            for j, params in enumerate(ParameterGrid(self.grids)):
                if self.fixed_estimator_params:
                    params = dict(**params, **self.fixed_estimator_params)
                clf = OptimalSamplingClassifier(
                    estimator=self.estimator_type(**params),
                    **(self.optimal_sampling_params if self.optimal_sampling_params else {})
                )
                if self.optimize_sampling_proba: 
                    clf.fit(X[train], y[train])
                else:
                    clf = fit_sampling_baseline(clf=clf, X=X[train], y=y[train], sampling_proba=sampling_proba)
                result = dict(
                    clf=clf,
                    fold=i,
                    params_id=j,
                    **params,
                    **performance_summary(clf=clf, X=X[test], y=y[test])
                )
                self.results.append(result)

        if self.refit:
            if self.verbose:
                print("Best parameters selected - refitting model on all data")
            params = self.select_best_params()
            if self.fixed_estimator_params:
                params = dict(**params, **self.fixed_estimator_params)
            clf = OptimalSamplingClassifier(
                estimator=self.estimator_type(**params),
                **(self.optimal_sampling_params if self.optimal_sampling_params else {})
            )
            if self.optimize_sampling_proba:
                clf.fit(X, y)
            else:
                clf = fit_sampling_baseline(clf=clf, X=X[train], y=y[train], sampling_proba=sampling_proba)
            self.estimator = clf
        else:
            self.estimator = self.select_best_params(return_estimator=True)




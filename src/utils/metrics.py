from typing import Dict

import numpy as np
from imblearn.metrics import sensitivity_score, specificity_score, geometric_mean_score
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, average_precision_score, \
    brier_score_loss, jaccard_score

from algorithm.optimal_sampling import OptimalSamplingClassifier


def performance_summary(clf: OptimalSamplingClassifier, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    predicted_proba = clf.predict_proba(X)
    if clf.use_decision_function:
        predicted_proba = (predicted_proba - predicted_proba.min()) / (predicted_proba.max() - predicted_proba.min())
    predicted = clf.predict(X)
    nominal_proba = (y == clf.positive_class).mean()
    return dict(
        model=str(clf.estimator).replace("\n", "").replace(" ", ""),
        class_ratio=1 / nominal_proba,
        weight_ratio=clf.positive_weight / clf.negative_weight,
        sampling_ratio=clf.sampling_proba / nominal_proba,
        accuracy=accuracy_score(y, predicted),
        sensitivity=sensitivity_score(y, predicted),
        specificity=specificity_score(y, predicted),
        precision=precision_score(y, predicted) if (predicted == clf.positive_class).sum() > 0 else None,
        f1_score=f1_score(y, predicted),
        geometric_mean_score=geometric_mean_score(y, predicted),
        roc_auc_score=roc_auc_score(y, predicted_proba),
        average_precision_score=average_precision_score(y, predicted_proba),
        brier_score=brier_score_loss(y, predicted_proba),
        jaccard_score=jaccard_score(y, predicted),
        weighted_loss=clf.weighted_loss(X, y).mean(),
        cost=clf.cost(X, y).mean()
    )
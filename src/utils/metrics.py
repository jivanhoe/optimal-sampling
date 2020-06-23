from typing import Dict

import numpy as np
from imblearn.metrics import sensitivity_score, specificity_score, geometric_mean_score
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, average_precision_score, \
    brier_score_loss, jaccard_score

from algorithm.optimal_sampling import OptimalSamplingClassifier


def performance_summary(clf: OptimalSamplingClassifier, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    predicted_proba = clf.predict_proba(X)[:, 1]
    predicted = clf.predict(X)
    return dict(
        model=str(clf.estimator).replace("\n", "").replace(" ", ""),
        class_ratio=1 / (y == clf.positive_class).mean(),
        weight_ratio=clf.positive_weight / clf.negative_weight,
        accuracy=accuracy_score(y, predicted),
        sensitivity=sensitivity_score(y, predicted),
        specificity=specificity_score(y, predicted),
        precision=precision_score(y, predicted),
        f1_score=f1_score(y, predicted),
        geometric_mean_score=geometric_mean_score(y, predicted),
        roc_auc_score=roc_auc_score(y, predicted_proba),
        average_precision_score=average_precision_score(y, predicted_proba),
        brier_score=brier_score_loss(y, predicted_proba),
        jaccard_score=jaccard_score(y, predicted),
        weighted_loss=clf.weighted_loss(X, y).mean(),
        cost=clf.cost(X, y).mean()
    )
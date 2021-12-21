from typing import Dict, Optional

import numpy as np
from imblearn.metrics import sensitivity_score, specificity_score, geometric_mean_score
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, average_precision_score, \
    recall_score

from algorithm.optimal_sampling import OptimalSamplingClassifier


def performance_summary(
        clf: OptimalSamplingClassifier,
        X: np.ndarray,
        y: np.ndarray,
        info: Optional[Dict[str, any]] = None,
) -> Dict[str, float]:
    predicted_proba = clf.predict_proba(X)
    predicted = clf.predict(X)
    nominal_proba = (y == clf.positive_class).mean()
    return dict(
        model=str(clf.estimator).replace("\n", "").replace(" ", ""),
        class_ratio=1 / nominal_proba,
        weight_ratio=clf.positive_weight / clf.negative_weight,
        sampling_probability=clf._sampling_proba,
        sampling_ratio=clf._sampling_proba / nominal_proba,
        iter_to_converge=clf._iter_count,
        accuracy=accuracy_score(y, predicted),
        sensitivity=sensitivity_score(y, predicted),
        specificity=specificity_score(y, predicted),
        precision=precision_score(y, predicted) if (predicted == clf.positive_class).sum() > 0 else None,
        recall=recall_score(y, predicted) if (predicted == clf.positive_class).sum() > 0 else None,
        f1_score=f1_score(y, predicted),
        geometric_mean_score=geometric_mean_score(y, predicted) if (predicted == clf.positive_class).sum() > 0 else None,
        roc_auc_score=roc_auc_score(y, predicted_proba),
        average_precision_score=average_precision_score(y, predicted_proba),
        weighted_loss=clf.weighted_loss(X, y).mean(),
        cost=clf.cost(X, y).mean(),
        **(info if info else {})
    )

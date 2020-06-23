from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from algorithm.optimal_undersampling import OptimalSampler
from pandas import read_csv
import numpy as np


df = read_csv("../data/default_data.csv")
y = np.array(df["defaulted"])
X = np.array(df.drop(columns="defaulted"))
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)
clf_base = LogisticRegressionCV(cv=5, max_iter=1000, solver='lbfgs', class_weight={0: 1, 1: 1})

clf_base.fit(x_train, y_train)
y_train_scores = clf_base.predict_proba(x_train)
y_test_scores = clf_base.predict_proba(x_test)
in_sample_auc = roc_auc_score(y_train, y_train_scores[:,1])
out_of_sample_auc = roc_auc_score(y_test, y_test_scores[:,1])
print("Nominal Probability")
print(f"In sample: {in_sample_auc} Out of sample: {out_of_sample_auc}")

sampler = OptimalSampler(clf=clf_base, loss=log_loss, X=x_train, y=y_train, positive_weight=2.0, termination_tol=1e-2)

x_train_downsampled, y_train_downsampled = sampler.sample(sampling_proba=0.5)
clf_base.fit(x_train_downsampled, y_train_downsampled)
y_train_scores_ds = clf_base.predict_proba(x_train)
y_test_scores_ds = clf_base.predict_proba(x_test)
in_sample_auc = roc_auc_score(y_train, y_train_scores_ds[:,1])
out_of_sample_auc = roc_auc_score(y_test, y_test_scores_ds[:,1])
print("Even Class Balance")
print(f"In sample: {in_sample_auc} Out of sample: {out_of_sample_auc}")
sampler.optimize()
y_train_scores = sampler.clf.predict_proba(x_train)
y_test_scores = sampler.clf.predict_proba(x_test)

in_sample_auc = roc_auc_score(y_train, y_train_scores[:,1])
out_of_sample_auc = roc_auc_score(y_test, y_test_scores[:,1])
print("Optimal Sampling")
print(f"In sample: {in_sample_auc} Out of sample: {out_of_sample_auc}")

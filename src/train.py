import numpy as np, joblib

train_mean = X_train.mean(axis=0)
train_std  = X_train.std(axis=0)
joblib.dump({"mean": train_mean, "std": train_std}, "artifacts/feature_stats.pkl")

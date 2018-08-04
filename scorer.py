import numpy as np
import pandas as pd
import lightgbm as lgb
import math
import random
import json
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

# load dataset
train = pd.read_csv("train_engineered_one_hot.csv")
train_num = train._get_numeric_data()
target = train_num["SalePrice"]
train_num.drop("SalePrice",inplace=True,axis=1)

test = pd.read_csv("test_engineered_one_hot.csv")
test_num = test._get_numeric_data()
test_array = test_num.values

# split into input (X) and output (Y) variables
X = train_num.values
Y = target.values

with open("best.json","r") as f:
    data = json.load(f)

model_lgb = lgb.LGBMRegressor(objective='regression',
                              num_leaves=data["best_params"]["num_leaves"],
                              learning_rate=data["best_params"]["learning_rate"],
                              n_estimators=data["best_params"]["n_estimators"],
                              max_bin = data["best_params"]["max_bin"],
                              bagging_fraction = data["best_params"]["bagging_fraction"],
                              bagging_freq = data["best_params"]["bagging_freq"],
                              feature_fraction = data["best_params"]["feature_fraction"],
                              min_data_in_leaf = data["best_params"]["min_data_in_leaf"],
                              min_sum_hessian_in_leaf = data["best_params"]["min_sum_hessian_in_leaf"],
                              verbosity=-1)

model_lgb.fit(X,Y)
predictions = model_lgb.predict(test_array)
np.savetxt("lgbm_preds.csv", predictions, delimiter=",")
#predictions = model_lgb.predict(test_array)
#np.savetxt("lgbm_preds.csv", predictions, delimiter=",")

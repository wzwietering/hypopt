import json
import lightgbm as lgb
import pandas as pd
import random

from sklearn.model_selection import cross_val_score

# load dataset
train = pd.read_csv("train_engineered_one_hot.csv")
train_num = train.select_dtypes(include="number")
target = train_num["SalePrice"]
train_num.drop("SalePrice", inplace=True, axis=1)

# split into input (X) and output (Y) variables
X = train_num.values
Y = target.values

params = {
    "num_leaves":(2, 64),
    "learning_rate":(0.01, 0.1),
    "n_estimators":(25, 500),
    "max_bin": (30, 500),
    "bagging_fraction": (0.2, 0.8),
    "bagging_freq": (1, 10),
    "feature_fraction": (0.3, 1.0),
    "min_data_in_leaf":(0, 20),
    "min_sum_hessian_in_leaf": (0, 1.0)
    }

best = -100
best_i = 0
best_params = {
    "num_leaves":0,
    "learning_rate":0,
    "n_estimators":0,
    "max_bin": 0,
    "bagging_fraction": 0,
    "bagging_freq": 0,
    "feature_fraction": 0,
    "min_data_in_leaf":0,
    "min_sum_hessian_in_leaf": 0,
    }
i = 0

try:
    while True:
        num_leaves = random.randint(params["num_leaves"][0], params["num_leaves"][1])
        learning_rate = random.uniform(params["learning_rate"][0], params["learning_rate"][1])
        n_estimators = random.randint(params["n_estimators"][0], params["n_estimators"][1])
        max_bin = random.randint(params["max_bin"][0], params["max_bin"][1])
        bagging_fraction = random.uniform(params["bagging_fraction"][0], params["bagging_fraction"][1])
        bagging_freq = random.randint(params["bagging_freq"][0], params["bagging_freq"][1])
        feature_fraction = random.uniform(params["feature_fraction"][0], params["feature_fraction"][1])
        min_data_in_leaf = random.randint(params["min_data_in_leaf"][0], params["min_data_in_leaf"][1])
        min_sum_hessian_in_leaf = random.uniform(params["min_sum_hessian_in_leaf"][0], params["min_sum_hessian_in_leaf"][1])

        model_lgb = lgb.LGBMRegressor(objective='regression',
                                      num_leaves=num_leaves,
                                      learning_rate=learning_rate,
                                      n_estimators=n_estimators,
                                      max_bin=max_bin,
                                      bagging_fraction=bagging_fraction,
                                      bagging_freq=bagging_freq,
                                      feature_fraction=feature_fraction,
                                      min_data_in_leaf=min_data_in_leaf,
                                      min_sum_hessian_in_leaf=min_sum_hessian_in_leaf,
                                      verbosity=-1)

        scores = cross_val_score(model_lgb, X, Y, cv=5, scoring='neg_mean_squared_error')
        score = scores.mean()
        if score > best:
            best = score
            best_i = i
            best_params = {
                "num_leaves":num_leaves,
                "learning_rate":learning_rate,
                "n_estimators":n_estimators,
                "max_bin":max_bin,
                "bagging_fraction":bagging_fraction,
                "bagging_freq":bagging_freq,
                "feature_fraction":feature_fraction,
                "min_data_in_leaf":min_data_in_leaf,
                "min_sum_hessian_in_leaf":min_sum_hessian_in_leaf,
                }
            with open("best.json", "w", encoding="utf-8") as f:
                result = {"best":score,
                          "best_i":i,
                          "best_params":best_params,
                         }
                f.write(json.dumps(result))
        i += 1
except KeyboardInterrupt:
    pass
print("Best parameters: ")
print(best_params)
print("Best value was at attempt " + str(best_i) + " out of " + str(i))
print("The best error was " + str(best))

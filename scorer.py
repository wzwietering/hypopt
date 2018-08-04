import json
import xgboost as xgb
import numpy as np
import pandas as pd

# load dataset
train = pd.read_csv("train_engineered_one_hot.csv")
train_num = train.select_dtypes(include="number")
target = train_num["SalePrice"]
train_num.drop("SalePrice", inplace=True, axis=1)

test = pd.read_csv("test_engineered_one_hot.csv")
test_num = test.select_dtypes(include="number")
test_array = test_num.values

# split into input (X) and output (Y) variables
X = train_num.values
Y = target.values

with open("best.json", "r") as f:
    data = json.load(f)

model_xgb = xgb.XGBRegressor(colsample_bytree=data["colsample_bytree"],
                             gamma=data["gamma"],
                             learning_rate=data["learning_rate"],
                             max_depth=data["max_depth"],
                             min_child_weight=data["min_child_weight"],
                             alpha=data["alpha"],
                             reg_lambda=data["lambda"],
                             subsample=data["subsample"],
                             max_delta_step=data["max_delta_step"],
                             colsample_bylevel=data["colsample_bylevel"],
                             )

model_xgb.fit(X, Y)
predictions = model_xgb.predict(test_array)
np.savetxt("xgb_preds.csv", predictions, delimiter=",")

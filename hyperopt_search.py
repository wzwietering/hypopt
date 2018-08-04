import json
import xgboost as xgb
import pandas as pd
import random

from sklearn.model_selection import cross_val_score

def load_dataset():
    train = pd.read_csv("train_engineered_one_hot.csv")
    train_num = train.select_dtypes(include="number")
    target = train_num["SalePrice"]
    train_num.drop("SalePrice", inplace=True, axis=1)

    # split into input (X) and output (Y) variables
    X = train_num.values
    Y = target.values
    return X, Y

def get_loss(X, Y, params):
    model_xgb = xgb.XGBRegressor(colsample_bytree=params["colsample_bytree"],
                                 gamma=params["gamma"],
                                 learning_rate=params["learning_rate"],
                                 max_depth=params["max_depth"],
                                 min_child_weight=params["min_child_weight"],
                                 alpha=params["alpha"],
                                 reg_lambda=params["lambda"],
                                 subsample=params["subsample"])

    scores = cross_val_score(model_xgb, X, Y, cv=5, scoring='neg_mean_squared_error')
    return scores.mean()

def random_params(param_ranges):
    params = {}
    params["colsample_bytree"] = random.uniform(param_ranges["colsample_bytree"][0], param_ranges["colsample_bytree"][1])
    params["gamma"] = random.uniform(param_ranges["gamma"][0], param_ranges["gamma"][1])
    params["learning_rate"] = random.uniform(param_ranges["learning_rate"][0], param_ranges["learning_rate"][1])
    params["max_depth"] = random.randint(param_ranges["max_depth"][0], param_ranges["max_depth"][1])
    params["min_child_weight"] = random.uniform(param_ranges["min_child_weight"][0], param_ranges["min_child_weight"][1])
    params["alpha"] = random.uniform(param_ranges["alpha"][0], param_ranges["alpha"][1])
    params["lambda"] = random.uniform(param_ranges["lambda"][0], param_ranges["lambda"][1])
    params["subsample"] = random.uniform(param_ranges["subsample"][0], param_ranges["subsample"][1])
    return params

def optimize_param(params, param, step_size):
    global X, Y, param_ranges
    initial = params[param]
    steps = []
    old_loss = get_loss(X, Y, params)
    increased = True
    improved = True
    times_no_improvement = 0
    while True:
        if increased and improved:
            if params[param] + step_size > param_ranges[param][1]: 
                print("Reached max bound")
                break
            params[param] += step_size
            increased = True
            print("Increased value")
        elif not increased and improved:
            if params[param] - step_size < param_ranges[param][0]: 
                print("Reached min bound")
                break
            params[param] -= step_size
            increased = False 
            print("Decreased value")
        elif increased and not improved:
            if params[param] - 2 * step_size < param_ranges[param][0]: 
                print("Reached min bound")
                break
            params[param] -= 2 * step_size
            increased = False 
            print("Decreased value")
        else:
            if params[param] + 2 * step_size > param_ranges[param][1]: 
                print("Reached max bound")
                break
            params[param] += 2 * step_size
            increased = True
            print("Increased value")
        loss = get_loss(X, Y, params)
        if loss > old_loss:
            print("Improved from " + str(old_loss) + " to " + str(loss))
            old_loss = loss
            steps.append(loss)
            times_no_improvement = 0
            improved = True
        else:
            print("No improvement, old was " + str(old_loss) + " new is " + str(loss))
            times_no_improvement += 1
            if times_no_improvement == 2: break
            improved = False
    print(steps)
    return params

if __name__ == "__main__":
    X, Y = load_dataset()
    param_ranges = {
        "colsample_bytree":(0, 1.0),
        "gamma": (0, 0.5),
        "learning_rate":(0.01, 0.1),
        "max_depth":(0, 10),
        "min_child_weight": (0, 5),
        "alpha": (0, 0.5),
        "lambda": (0, 0.5),
        "subsample":(0, 1.0),
        }
    rand_params = random_params(param_ranges)
    optimize_param(rand_params, "learning_rate", 0.01)

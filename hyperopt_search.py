import json
import xgboost as xgb
import pandas as pd
import random

import datacollector

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
                                 subsample=params["subsample"],
                                 max_delta_step=params["max_delta_step"],
                                 colsample_bylevel=params["colsample_bylevel"],
                                 )

    scores = cross_val_score(model_xgb, X, Y, cv=5, scoring='neg_mean_squared_error')
    return scores.mean()

def random_params(param_ranges):
    params = {}
    for key in param_ranges.keys():
        if type(param_ranges[key][0]) == int:
            params[key] = random.randint(param_ranges[key][0], param_ranges[key][1])
        else:
            params[key] = random.uniform(param_ranges[key][0], param_ranges[key][1])
    return params

def optimize_param(params, param, step_size):
    global X, Y, param_ranges, dc
    initial = params[param]
    steps = []
    old_loss = get_loss(X, Y, params)
    increased = True
    improved = True
    times_no_improvement = 0
    best = params[param]
    while True:
        if increased and improved:
            if params[param] + step_size > param_ranges[param][1]: 
                print("Reached max bound")
                break
            params[param] += step_size
            increased = True
            print("Increased value of " + param + " to " + str(params[param]))
        elif not increased and improved:
            if params[param] - step_size < param_ranges[param][0]: 
                print("Reached min bound")
                break
            params[param] -= step_size
            increased = False 
            print("Decreased value of " + param + " to " + str(params[param]))
        elif increased and not improved:
            if params[param] - 2 * step_size < param_ranges[param][0]: 
                print("Reached min bound")
                break
            params[param] -= 2 * step_size
            increased = False 
            print("Decreased value of " + param + " to " + str(params[param]))
        else:
            if params[param] + 2 * step_size > param_ranges[param][1]: 
                print("Reached max bound")
                break
            params[param] += 2 * step_size
            increased = True
            print("Increased value of " + param + " to " + str(params[param]))
        loss = get_loss(X, Y, params)
        dc.save_params(params, loss)
        if loss > old_loss:
            print("Improved from " + str(old_loss) + " to " + str(loss))
            old_loss = loss
            best = params[param]
            steps.append(loss)
            times_no_improvement = 0
            improved = True
        else:
            print("No improvement, old was " + str(old_loss) + " new is " + str(loss))
            times_no_improvement += 1
            if times_no_improvement == 2: break
            improved = False
    print("Steps taken: " + str(steps))
    print("Best value for " + param + " is " + str(best))
    params[param] = best
    return params

def gatherMetaData(loops):
    global X, Y, dc, param_ranges
    for i in range(loops):
        rand_params = random_params(param_ranges)
        loss = get_loss(X, Y, rand_params)
        dc.save_params(params, loss)
        if i % 20 == 0:
            print("Committed data at iteration " + str(i))
            dc.commit()

if __name__ == "__main__":
    X, Y = load_dataset()
    param_ranges = {
        "colsample_bytree": (0.0, 1.0),
        "gamma": (0.0, 0.5),
        "learning_rate": (0.01, 0.2),
        "max_depth": (0, 15),
        "min_child_weight": (0, 5),
        "alpha": (0.0, 0.5),
        "lambda": (0.0, 0.5),
        "subsample": (0.0, 1.0),
        "max_delta_step": (0, 10),
        "colsample_bylevel": (0.0, 1.0),
        }
    best_loss = -10
    best_params = {}
    dc = datacollector.DataCollector(param_ranges)

    for i in range(10):
        print("\nScoring set " + str(i) + "\n")
        rand_params = random_params(param_ranges)
        for key in param_ranges.keys():
            if type(param_ranges[key][0]) == int:
                rand_params = optimize_param(rand_params, key, 1)
            else:
                rand_params = optimize_param(rand_params, key, 0.001)
        final_loss = get_loss(X, Y, rand_params)
        dc.commit()
        print("Best values: " + str(rand_params))
        print("Final loss: " + str(final_loss))
        if final_loss > best_loss:
            print("New best loss: " + str(final_loss))
            best_loss = final_loss
            best_params = rand_params 
            with open("best.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(rand_params))

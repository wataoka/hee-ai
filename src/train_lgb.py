import os

import optuna
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

from data.loader import load_data
from data.feature  import get_features
from data.trivias_list import trivias_list



def objective(trial):

    max_depth = trial.suggest_int('max_depth', 1, 20)
    learning_rate = trial.suggest_uniform('learning_rate', 0.001, 0.1)
    scale_pos_weight = trial.suggest_uniform('scale_pos_weight', 1.0, 12.0)
    params = {
        'metric': 'l2',
        'num_leaves': trial.suggest_int("num_leaves", 2, 70),
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'objective': 'regression',
        'scale_pos_weight': scale_pos_weight,
        'verbose': 0
    }

    mse_list = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    for train_idx, valid_idx in kfold.split(X, y):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_valid, y_valid)

        model = lgb.train(params,
                          lgb_train,
                          valid_sets=lgb_valid,
                          verbose_eval=10,
                          early_stopping_rounds=30)

        # f-measure
        pred_y_valid = model.predict(X_valid, num_iteration=model.best_iteration)
        true_y_valid = np.array(y_valid.data.tolist())
        mse = np.sum((pred_y_valid - true_y_valid)**2) / len(true_y_valid)
        mse_list.append(mse)

    return np.mean(mse_list)

def build_model():

    study = optuna.create_study()
    study.optimize(objective, n_trials=10)

    valid_split = 0.2
    num_train = int((1-valid_split)*len(X))
    X_train = X[:num_train]
    y_train = y[:num_train]
    X_valid = X[num_train:]
    y_valid = y[num_train:]
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid)

    lgb_data = lgb.Dataset(X, y)

    params = study.best_params
    params['metric'] = 'l2'
    print(params)
    quit()
    model = lgb.train(params,
                      lgb_data,
                      valid_sets=lgb_valid,
                      verbose_eval=10,
                      early_stopping_rounds=30)
    
    return model

X, y = load_data(trivias_list)

if __name__ == "__main__":

    model = build_model()
    
    content = 'ミツバチが一生かけて集める蜂蜜はティースプーン1杯程度。'
    content_df = get_features(trivias_list, content=content, mode='inference')
    output = model.predict(content_df)
    hee = int(output*100)

    print(f"{content}")
    print(f"{hee}へぇ")
import pickle

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import tensorflow as tf
keras = tf.keras
from lightgbm import LGBMRegressor

# Evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, max_error
from sklearn.model_selection import cross_val_score, train_test_split

# hyperparameters
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
## Tf version was 2.3.0



def plot_and_stats(predict, test, dot_color, name):   
    plt.figure(figsize=(8,8))
    plt.scatter(predict, test, color = dot_color, s = 20, alpha = 0.5, label = name)

    plt.xlabel("Prediction [MW]")
    plt.ylabel("True Values [MW]")
    plt.title("Predicted Values vs Modeled Values")
    plt.legend()
    lims = [min(predict), max(predict)]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims, color ="#A4A7AA")
    
    plt.grid(color='grey', linestyle='-', linewidth=0.5, alpha = 0.5)
    plt.show()



def auto_regression(X, y):

    # Testing With No Scaling
    X_train, X_test, y_train, y_test =  train_test_split(X, y,
                                    test_size = 0.1, random_state=69)
    X_train, X_val, y_train, y_val = train_test_split(X_train, 
                                    y_train, test_size = 0.1, random_state=69)

    X_train, X_train_2, y_train, y_train_2 = train_test_split(X_train, 
                                    y_train, test_size = 0.3, random_state=69)

    X_train, X_test_2, y_train, y_test_2 = train_test_split(X_train, 
                                    y_train, test_size = 0.15, random_state=69)


    #prep model names
    file_name_XGBoost = 'XGBoost_model.sav'
    file_name_RF = 'RF_model.sav'
    file_name_LGBM = 'LGBM_model.sav'
    file_name_GB = 'GB_model.sav'
    file_name_NN_reg = 'NN_reg_model.h5'
    file_name_ensemble = 'ensemble_model.h5'


    ## XGBoost Optimization
    space = {
            'max_depth':hp.choice('max_depth', np.arange(5, 15, 1, dtype=int)),
            'n_estimators':hp.choice('n_estimators', np.arange(1000, 10000, 10, dtype=int)),
            'colsample_bytree':hp.quniform('colsample_bytree', 0.5, 1.0, 0.1),
            'min_child_weight':hp.choice('min_child_weight', np.arange(1, 50, 5, dtype=int)),
            'subsample':hp.quniform('subsample', 0.8, 1, 0.1),
            'eta':hp.quniform('eta', 0.1, 0.4, 0.1),
            
            'objective':'reg:squarederror',
            
            'tree_method':'hist',
            'eval_metric': 'rmse',
        }

    def score(params):
        model = xgb.XGBRegressor(**params)
        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                verbose=False, early_stopping_rounds=10)
        y_pred = model.predict(X_val)
        score = mean_squared_error(y_val, y_pred) ** 0.5
        return {'loss': score, 'status': STATUS_OK}    
        
    def optimize(trials, space):
        
        best = fmin(score, space, algo=tpe.suggest, max_evals=100)
        return best

    trials = Trials()
    best_params = optimize(trials, space)

    optimal_var = space_eval(space, best_params)

    xgb_model_best = xgb.XGBRegressor(max_depth = optimal_var['max_depth'],
                                  n_estimators = optimal_var['n_estimators'],
                                  colsample_bytree= optimal_var['colsample_bytree'],
                                  min_child_weight= optimal_var['min_child_weight'],
                                 eta= optimal_var['eta'])

    xgb_model_best.fit(X_train, y_train, early_stopping_rounds=20, 
                       eval_set=[(X_val, y_val)])


    xgb_model_general = xgb.XGBRegressor(n_estimators= 2000)
    xgb_model_general.fit(X_train, y_train, early_stopping_rounds=5, 
                       eval_set=[(X_val, y_val)])

    xgb_predict_best = xgb_model_best.predict(X_test)
    xgb_predict = xgb_model_general.predict(X_test)

    mae_xgb_best = mean_absolute_error(xgb_predict_best, y_test)
    mae_xgb_general = mean_absolute_error(xgb_predict, y_test)


    ## Random Forests
    space = {
    'max_depth': hp.choice('max_depth', range(1,20)),
    'max_features': hp.quniform('max_features',.1,.5,.1),
    'n_estimators': hp.choice('n_estimators', range(100,500))}

    def score_rf(params):
        model = RandomForestRegressor(**params)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score_rf = mean_squared_error(y_val, y_pred) ** 0.5
        return {'loss': score_rf, 'status': STATUS_OK}    
        
    def optimize_rf(trials, space):
        best = fmin(score_rf, space, algo=tpe.suggest, max_evals=100)
        return best

    trials = Trials()
    best_params = optimize_rf(trials, space)

    optimal_var = space_eval(space, best_params)


    RF_model_best = RandomForestRegressor(max_depth = optimal_var['max_depth'],
                                      max_features = optimal_var['max_features'],
                                      n_estimators = optimal_var['n_estimators'], 
                                      random_state = 69)

    RF_model_best.fit(X_train, y_train)

    RF_model_general = RandomForestRegressor(random_state = 69)
    RF_model_general.fit(X_train, y_train)


    RF_predict = RF_model_general.predict(X_test)
    RF_predict_best = RF_model_best.predict(X_test)

    mae_rf_best =  mean_absolute_error(RF_predict_best, y_test)
    mae_rf_general = mean_absolute_error(RF_predict, y_test)

    ## Gradient Boosting

    space = {
    'max_depth': hp.choice('max_depth', range(1,20)),
    'max_features': hp.quniform('max_features',.1,.5,.1),
    'n_estimators': hp.choice('n_estimators', range(100,1000)),
    'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 100)),
    'min_samples_split': hp.choice('min_samples_split', range(2, 100)),
    'validation_fraction': 0.2,
    'tol':  .001,
    'n_iter_no_change': 10
    }

    def score_gb(params):
        model =  GradientBoostingRegressor(**params)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score_gb = mean_squared_error(y_val, y_pred) ** 0.5
        return {'loss': score_gb, 'status': STATUS_OK}    
        
    def optimize_gb(trials, space):
        best = fmin(score_gb, space, algo=tpe.suggest, max_evals=100)
        return best

    trials = Trials()
    best_params = optimize_gb(trials, space)
    optimal_var = space_eval(space, best_params)


    GB_model_best = GradientBoostingRegressor(n_estimators = optimal_var['n_estimators'], 
                                                random_state = 69, validation_fraction = 0.2,
                                                n_iter_no_change=10, tol=0.001,
                                               max_depth = optimal_var['max_depth'],
                                               max_features=optimal_var['max_features'],
                                              min_samples_split= optimal_var['min_samples_split'],
                                               min_samples_leaf = 1)

    GB_model_best.fit(X_train, y_train)
    GB_predict_best = GB_model_best.predict(X_test)

    GB_model_general = GradientBoostingRegressor()
    GB_model_general.fit(X_train, y_train)
    GB_predict_general = GB_model_general.predict(X_test)
    
    mae_gb_best =  mean_absolute_error(GB_predict_best, y_test)
    mae_gb_general = mean_absolute_error(GB_predict_general, y_test)


    ### light GBM
    # Choose hyperparameter domain to search over
    space = {
            'max_depth':hp.choice('max_depth', np.arange(2, 10, 1, dtype=int)),
            'num_leaves':hp.choice('num_leaves', np.arange(2, 100, 2, dtype=int)),
            'min_data_in_leaf':hp.choice('min_data_in_leaf',  np.arange(10, 100, 2, dtype=int))
        }


    def score_lgb(params):
        model = LGBMRegressor(**params)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score_lgb = mean_squared_error(y_val, y_pred) ** 0.5
        print(score_lgb)
        return {'loss': score_lgb, 'status': STATUS_OK}    
        
    def optimize_lgb(trials, space):
        
        best = fmin(score_lgb, space, algo=tpe.suggest, max_evals=20)
        return best

    trials = Trials()
    best_params = optimize_lgb(trials, space)
    optimal_var = space_eval(space, best_params)

    LGB_model_best = LGBMRegressor(max_depth = optimal_var['max_depth'],
                                  num_leaves = optimal_var['num_leaves'],
                                  min_data_in_leaf= optimal_var['min_data_in_leaf'])
    
    LGB_model_best.fit(X_train, y_train)

    GB_light_model_general = LGBMRegressor()
    GB_light_model_general.fit(X_train, y_train)

    GB_light_predict = GB_light_model_general.predict(X_test)
    LGB_predict_best = LGB_model_best.predict(X_test)

    mae_lgbm_best = mean_absolute_error(LGB_predict_best, y_test)
    mae_lgbm_general = mean_absolute_error(GB_light_predict, y_test)


#### Stats summary + model choice


    if mae_xgb_best < mae_xgb_general:
        pickle.dump(xgb_model_best, open(file_name_XGBoost, 'wb'))
        print("using best xgb model")
        xgb_train = xgb_model_best.predict(X_train_2)
        xgb_test = xgb_model_best.predict(X_test_2)
    else:
        pickle.dump(xgb_model_general, open(file_name_XGBoost, 'wb'))
        print("using general xgb model")
        xgb_train = xgb_model_general.predict(X_train_2)
        xgb_test = xgb_model_general.predict(X_test_2)

    if mae_rf_best < mae_rf_general:
        pickle.dump(RF_model_best, open(file_name_RF, 'wb'))
        print("using best RF model")
        RF_train = RF_model_best.predict(X_train_2)
        RF_test = RF_model_best.predict(X_test_2)
    else:
        pickle.dump(RF_model_general, open(file_name_RF, 'wb'))
        print("using general RF model")
        RF_train = RF_model_general.predict(X_train_2)
        RF_test = RF_model_general.predict(X_test_2)

    if mae_gb_best < mae_gb_general:
        pickle.dump(GB_model_best, open(file_name_GB, 'wb'))
        print("using best RF model")
        GB_train = GB_model_best.predict(X_train_2)
        GB_test = GB_model_best.predict(X_test_2)
    else:
        pickle.dump(GB_model_general, open(file_name_GB, 'wb'))
        print("using general RF model")
        GB_train = GB_model_general.predict(X_train_2)
        GB_test = GB_model_general.predict(X_test_2)

    if mae_lgbm_best < mae_lgbm_general:
        pickle.dump(LGB_model_best, open(file_name_LGBM, 'wb'))
        print("using best xgb model")
        GB_light_train = LGB_model_best.predict(X_train_2)
        GB_light_test = LGB_model_best.predict(X_test_2)
    else:
        pickle.dump(GB_light_model_general, open(file_name_LGBM, 'wb'))
        print("using general xgb model")
        GB_light_train = GB_light_model_general.predict(X_train_2)
        GB_light_test = GB_light_model_general.predict(X_test_2)


####### Constructing Neural Network:
    keras.backend.clear_session()
    tf.random.set_seed(69)
    np.random.seed(69)

    NN_model = tf.keras.models.Sequential()
    NN_model.add(keras.layers.Dense(256, activation='relu', input_shape= [len(X_train.keys())]))
    NN_model.add(keras.layers.Dropout(0.1))
    NN_model.add(keras.layers.Dense(64, activation='relu'))
    NN_model.add(keras.layers.Dropout(0.2))
    NN_model.add(keras.layers.Dense(1))

    NN_model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    model_checkpoint = keras.callbacks.ModelCheckpoint(file_name_NN_reg)
    early_stopping = keras.callbacks.EarlyStopping(patience = 100, restore_best_weights = True)
    NN_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2000, verbose=1, callbacks=[model_checkpoint, early_stopping])
   
    nn_predictions = NN_model.predict(X_test).flatten()
    NN_model.save(file_name_NN_reg)

##### Constructing Ensemble Model

    nn_train = NN_model.predict(X_train_2).flatten()

    Ensemble_train_df = pd.DataFrame({
        'xgb': xgb_train,
        'gb': GB_train,
        'rf': RF_train,
        'lgbm': GB_light_train,
        'nn': nn_train
    })


    keras.backend.clear_session()
    tf.random.set_seed(69)
    np.random.seed(69)

    ensemble = tf.keras.models.Sequential()
    ensemble.add(keras.layers.Dense(32, activation='relu', input_shape =[len(Ensemble_train_df.keys())]))
    ensemble.add(keras.layers.Dense(16))
    ensemble.add(keras.layers.Dense(1))

    ensemble.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])


    model_checkpoint_ensemble = keras.callbacks.ModelCheckpoint(file_name_ensemble)
    early_stopping_ensemble = keras.callbacks.EarlyStopping(patience = 100, restore_best_weights = True)
    ensemble.fit(Ensemble_train_df, y_train_2, validation_split= 0.2, epochs=2000, verbose=1, callbacks=[model_checkpoint_ensemble, early_stopping_ensemble])
    ensemble.save(file_name_ensemble)


    nn_test = NN_model.predict(X_test_2).flatten()

    Ensemble_test_df = pd.DataFrame({
        'xgb': xgb_test,
        'gb': GB_test,
        'rf': RF_test,
        'lgbm': GB_light_test,
        'nn': nn_test
    })

    ensemble_predictions = ensemble.predict(Ensemble_test_df).flatten()

    print(ensemble_predictions)
    plot_and_stats(ensemble_predictions, y_test_2, "#00AEEF", "MW model")

    print("xgboost_best:", mae_xgb_best)
    print("xgboost_general:", mae_xgb_general)
    print("rf_best:", mae_rf_best)
    print("rf_general:", mae_rf_general)
    print("gb_best:", mae_gb_best)
    print("gb_general:", mae_gb_general)
    print("lgbm_best:", mae_lgbm_best)
    print("lgbm_general:", mae_lgbm_general)




# Example Data entry:
############################
df = pd.read_csv(r'C:\Data Science\Data files\2103 Data\GPA 2103 Houton and Greeley short.csv')
df = df.drop(['MW', 'C7+ MW', 'C7+ Sp. Grav.', 'API', '22DMP', 
            'C6 Peak 1', 'C6 Peak 2', 'nC6', 'N2'], axis=1)

#drop all non float values
df = df.apply(pd.to_numeric, errors = 'coerce').dropna()

df = df.sample(n = 500, random_state=69)

X = df.drop(['Sp. Grav.'], axis=1)
y = df['Sp. Grav.']

X.info()

auto_regression(X, y)


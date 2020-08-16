import pandas as pd
import numpy as np
from scipy import stats
from xgboost.sklearn import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from pmdarima.arima import auto_arima, ADFTest
import category_encoders as ce
import time
# TODO Add hyperopt method to optimize
class ModelTrain:

    def __init__(self, masterdf, target_col, predictors, cat_cols, modelname):
        self.target_col = target_col
        self.masterdf = masterdf
        self.num_round = 300
        self.cv_folds = 5
        self.cat_cols = cat_cols
        self.predictors = predictors
        self.modelname = modelname
        return

    def get_normalized_data(self):
        """

        :return:
        """
        if self.modelname == 'catboost':
            self.enc = ""
            return
        # Convert categorical columns using OneHotEncoding
        master_catcols = self.masterdf[self.cat_cols]
        self.enc = ce.OneHotEncoder(cols=self.cat_cols, return_df=True).fit(master_catcols)
        master_catcols = self.enc.transform(master_catcols)
        num_cols = list(set(self.masterdf.columns)-set(self.cat_cols))
        master_numcols = self.masterdf[num_cols]
        self.masterdf = pd.concat([master_numcols, master_catcols], axis=1)
        self.predictors = list(set(self.predictors) - set(self.cat_cols))
        self.predictors.extend(master_catcols.columns.tolist())
        return

    def define_xgb_model_params(self):
        self.xgb1 = XGBRegressor()
        parameters = {'nthread': [4],  # when use hyperthread, xgboost may become slower
                      'objective': ['reg:squarederror'],
                      'learning_rate': stats.uniform(.03, .1),  # so called `eta` value
                      'max_depth': [3, 5, 6, 7, 9],
                      'min_child_weight': [4],
                      'subsample': stats.uniform(0.5, 0.4),
                      'colsample_bytree': stats.uniform(0.5, 0.4),
                      'n_estimators': stats.randint(100, 500),
                      'gamma': [0.1, 0.5, 1, 1.5],
                      'lambda': [0.01, 0.5, 1, 2],
                      'seed': [1]}
        self.xgb_grid = RandomizedSearchCV(self.xgb1,
                                parameters,
                                cv=4,
                                n_jobs=4,
                                verbose=True,
                                n_iter=100)
        return

    def define_random_forest_model(self):
        """

        :return:
        """
        parameters = {'criterion': ['mse'],
                      'max_depth': [5, 6, 7],
                      'min_samples_leaf': [5, 10],
                      'min_impurity_decrease': [.001, .005, 0.01],
                      'max_features': ["sqrt", "log2"],
                      'n_estimators': [100, 500],
                      'ccp_alpha': [0.01, 0.05, .1],
                      'random_state': [1]}
        self.rf = RandomForestRegressor()
        self.rf_grid = RandomizedSearchCV(self.rf,
                                     parameters,
                                     cv=4,
                                     n_jobs=4,
                                     verbose=True)
    
    def define_catboost_model_params(self):
        self.cat1 = CatBoostRegressor()

        parameters = {'loss_function': ['RMSE'],
                      'depth': [6, 8, 10],
                      'cat_features': self.cat_features_catboost,
                      'learning_rate': [0.01, 0.05, 0.1],
                      'iterations': [30, 100, 1000],
                      'l2_leaf_reg': [.1, 1, 10, 100],
                      'early_stopping_rounds': [100],
                      'task_type': ['GPU'],
                      'border_count': [32],
                      'random_seed': [1]}

        # parameters = {'loss_function': ['RMSE'],
        #               'depth': [6, 8, 10],
        #               'cat_features': self.cat_features_catboost,
        #               'learning_rate': [0.01, 0.05, 0.1],
        #               'iterations': [30, 100, 1000],
        #               'l2_leaf_reg': [.1, 1, 10, 100],
        #               'early_stopping_rounds': [100],
        #               'random_seed': [1]}

        self.cat_grid = GridSearchCV(estimator=self.cat1,
                                param_grid=parameters,
                                cv=3,
                                n_jobs=5,
                                verbose=True)

        return


    def get_test_train(self, split_col=None, split_value=None):
        """

        :return:
        """

        if (split_col is None) or (split_value is None):
            self.train_data = self.masterdf
            return
        self.train_data = self.masterdf[~self.masterdf[split_col].isin(split_value)]
        self.test_data = self.masterdf[self.masterdf[split_col].isin(split_value)]
        return

    def train_model(self, model):
        """

        :return:
        """
        start = time.time()
        X = self.train_data[self.predictors]
        y = self.train_data[self.target_col]

        if self.modelname == 'xgb':
            self.define_xgb_model_params()
            model = self.xgb1
            model_grid = self.xgb_grid
        elif self.modelname == 'rf':
            self.define_random_forest_model()
            model = self.rf
            model_grid = self.rf_grid
        elif self.modelname == 'catboost':
            self.cat_features_catboost = [X.columns.get_loc(col) for col in self.cat_cols]
            self.define_catboost_model_params()
            model = self.cat1
            model_grid = self.cat_grid
        else:
            print('Model selected is not available')
            return
        model_grid.fit(X, y)
        model.set_params(**model_grid.best_params_)
        model.fit(X, y, verbose=False)
        print(model_grid.best_score_)
        print(model_grid.best_params_)
        self.feat_imp_df = pd.DataFrame(zip(self.predictors, model_grid.best_estimator_.feature_importances_), columns=['feature_name', 'feature_importance'])

        end = time.time()
        # total time taken
        print(f"Runtime of the program is {(end - start)/60} mins")

        return self.enc, model

    @staticmethod
    def get_timeseries_forecast(masterdf, target_col, timeseries_col, pred_points):
        """

        :return:
        """
        start = time.time()
        timeseries_key_list = masterdf[timeseries_col].unique()
        prediction = pd.DataFrame(columns=[pred_points])
        for key in timeseries_key_list:
            print(key)
            ts_series = masterdf[masterdf[timeseries_col] == key][target_col]
            MIN_LEN = 5
            PRED_PERIOD = 5
            start_len = 0
            end_len = MIN_LEN
            if len(ts_series) < MIN_LEN:
                prediction = prediction.append(pd.DataFrame(pd.Series(np.nan), ts_series.index, columns=[pred_points]))
                print('Failure')
            else:
                i = 1
                prediction = prediction.append(pd.DataFrame(pd.Series(np.nan), index=ts_series[:end_len].index, columns=[pred_points]))
                while i > 0:
                    train = ts_series[start_len:end_len]
                    arima_model = auto_arima(train, random_state=1, suppress_warnings=True)
                    if len(ts_series) - len(ts_series[:end_len]) > PRED_PERIOD:
                        test = ts_series[end_len: end_len + PRED_PERIOD]
                        pred = arima_model.predict(n_periods=PRED_PERIOD)
                        prediction = prediction.append(pd.DataFrame(pred, index=test.index, columns=[pred_points]))
                        if (len(ts_series) - len(train) >= 15):
                            start_len = start_len + 5
                        end_len = end_len + PRED_PERIOD
                    else:
                        test = ts_series[end_len-1:]
                        pred = arima_model.predict(n_periods=len(test))
                        prediction = prediction.append(pd.DataFrame(pred, index=test.index, columns=[pred_points]))
                        i = 0
                print("Success")
        end = time.time()
        # total time taken
        print(f"Runtime of the program is {(end - start)/60} mins")
        return prediction


class ModelPredict:

    def __init__(self, masterdf, enc, model, modelname, predictors, cat_cols, points_pred_col):
        self.masterdf = masterdf
        self.enc = enc
        self.model = model
        self.points_pred_col = points_pred_col
        self.cat_cols = cat_cols
        self.predictors = predictors
        self.modelname = modelname

    def get_normalized_data(self):
        """

        :return:
        """
        if (self.modelname == 'xgboost') or (self.modelname == 'rf'):
            master_catcols = self.masterdf[self.cat_cols]
            master_catcols = self.enc.transform(master_catcols)
            num_cols = list(set(self.masterdf.columns)-set(self.cat_cols))
            master_numcols = self.masterdf[num_cols]
            self.masterdf = pd.concat([master_numcols, master_catcols], axis=1)
            self.predictors = list(set(self.predictors) - set(self.cat_cols))
            self.predictors.extend(master_catcols.columns.tolist())
        return

    def get_model_predictions(self):
        """

        :return:
        """

        masterdf = self.masterdf[self.predictors]
        prediction_value = self.model.predict(masterdf.values)
        return prediction_value

    @staticmethod
    def get_model_error(masterdf, pred_col, target_col, groupbycol=None):
        """
,
        :param self:
        :return:
        """
        # predictions_error = metrics.mean_absolute_error(masterdf[target_col].values, masterdf[pred_col].values)
        masterdf[target_col].fillna(0, inplace=True)
        masterdf['error'] = np.where((np.isnan(masterdf[pred_col])), np.nan, abs(masterdf[target_col] - masterdf[pred_col]))
        predictions_error = abs(masterdf['error']).mean()
        if groupbycol != None:
            yearly_summary = pd.DataFrame(masterdf.groupby([groupbycol])[['error']].mean()).reset_index()
        else:
            yearly_summary = None
        return predictions_error, yearly_summary


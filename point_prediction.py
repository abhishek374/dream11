import pandas as pd
import numpy as np
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import category_encoders as ce
# TODO Add hyperopt method to optimize
class ModelTrain:

    def __init__(self, masterdf, target_col, predictors, cat_cols):
        self.target_col = target_col
        self.masterdf = masterdf
        self.num_round = 300
        self.cv_folds = 5
        self.cat_cols = cat_cols
        self.predictors = predictors
        return
    def define_xgb_model_params(self):
        self.xgb1 = XGBRegressor()
        parameters = {'nthread': [4],  # when use hyperthread, xgboost may become slower
                      'objective': ['reg:squarederror'],
                      'learning_rate': [.03, 0.05, .07],  # so called `eta` value
                      'max_depth': [5, 6, 7],
                      'min_child_weight': [4],
                      'subsample': [0.7],
                      'colsample_bytree': [0.7],
                      'n_estimators': [100, 500],
                      'gamma': [0.1, 0.5, 1, 1.5, ],
                      'lambda': [0.01, 0.5, 1, 2],
                      'seed': [1]}
        self.xgb_grid = GridSearchCV(self.xgb1,
                                parameters,
                                cv=5,
                                n_jobs=5,
                                verbose=True)
        return

    def get_normalized_data(self):
        """

        :return:
        """
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

    def train_model(self, model ='xgb'):
        """

        :return:
        """
        X = self.train_data[self.predictors]
        y = self.train_data[self.target_col]

        if model == 'xgb':
            self.define_xgb_model_params()
            self.xgb_grid.fit(X, y)
            self.xgb1.set_params(**self.xgb_grid.best_params_)
            self.xgb1.fit(X.values, y.values, verbose=False)
            print(self.xgb_grid.best_score_)
            print(self.xgb_grid.best_params_)
            self.feat_imp_df = pd.DataFrame(zip(self.predictors, self.xgb_grid.best_estimator_.feature_importances_),
                                            columns=['feature_name', 'feature_importance'])
        return

    def get_model_objects(self):
        return self.enc, self.xgb1


class ModelPredict:

    def __init__(self, masterdf, enc, model, predictors, cat_cols, points_pred_col):
        self.masterdf = masterdf
        self.enc = enc
        self.model = model
        self.points_pred_col = points_pred_col
        self.cat_cols = cat_cols
        self.predictors = predictors

    def get_normalized_data(self):
        """

        :return:
        """
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
        dtest_predictions = self.model.predict(masterdf.values)
        return dtest_predictions

    @staticmethod
    def get_model_error(masterdf, pred_col, target_col, groupbycol=None):
        """
,
        :param self:
        :return:
        """
        predictions_error = metrics.mean_squared_error(masterdf[target_col].values, masterdf[pred_col].values)
        masterdf[target_col].fillna(0, inplace=True)
        masterdf['error'] = np.where((masterdf[target_col] == 0), np.nan, abs(masterdf[target_col] - masterdf[pred_col]) / masterdf[target_col])
        if groupbycol != None:
            yearly_summary = pd.DataFrame(masterdf.groupby([groupbycol])[['error']].mean()).reset_index()
        else:
            yearly_summary = None

        return predictions_error, yearly_summary


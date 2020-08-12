import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder
# TODO add gridsearch from sklearn.grid_search import GridSearchCV

class ModelTrain:

    def __init__(self, masterdf, target_col, predictors, cat_cols, test_year=[2019]):
        self.target_col = target_col
        self.xgb1 = XGBClassifier(
            learning_rate=0.05,
            n_estimators=1000,
            max_depth=8,
            min_child_weight=1,
            gamma=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            nthread=-1,
            seed=27)
        self.masterdf = masterdf
        self.num_round = 300
        self.cv_folds = 5
        self.test_year = test_year
        self.cat_cols = cat_cols
        self.predictors = predictors
        return

    def get_normalized_data(self):
        """

        :return:
        """
        # Convert categorical columns using OneHotEncoding
        # encoder = OneHotEncoder()
        master_catcols = self.masterdf[self.cat_cols]
        self.enc = ce.OneHotEncoder(cols=self.cat_cols, return_df=True).fit(master_catcols)
        master_catcols = self.enc.transform(master_catcols)
        num_cols = list(set(self.masterdf.columns)-set(self.cat_cols))
        master_numcols = self.masterdf[num_cols]
        self.masterdf = pd.concat([master_numcols, master_catcols], axis=1)
        self.predictors = list(set(self.predictors) - set(self.cat_cols))
        self.predictors.extend(master_catcols.columns.tolist())
        return

    def get_test_train(self):
        """

        :return:
        """
        self.train_data = self.masterdf[~self.masterdf['year'].isin(self.test_year)]
        self.test_data = self.masterdf[self.masterdf['year'].isin(self.test_year)]

        return

    def train_model(self):
        """

        :return:
        """
        X = self.train_data[self.predictors]
        y = self.train_data[self.target_col]
        xgtrain = xgb.DMatrix(X.values, label=y.values)
        xgb_params = self.xgb1.get_xgb_params()
        cv_results = xgb.cv(xgb_params, xgtrain, num_boost_round=1000,
                            verbose_eval=25, early_stopping_rounds=self.num_round, nfold=self.cv_folds, metrics='rmse')
        self.xgb1.set_params(n_estimators=cv_results.shape[0])
        self.xgb1.fit(X, y, eval_metric='rmse')
        # print(f'CV error using softprob = {error_rate}')
        return

    def get_model_accuracy(self):
        """

        :param self:
        :return:
        """
        dtrain_predictions = self.xgb1.predict(self.train_data[self.predictors])
        train_accuracy = metrics.accuracy_score(self.train_data[self.target_col].values, dtrain_predictions)
        feat_imp_df = pd.DataFrame(zip(self.predictors, self.xgb1.feature_importances_),columns = ['feature_name','feature_importance'])
        if self.test_data.empty:
            data_predictions = pd.Series(dtrain_predictions)
            test_accuracy = 'Not Applicable'
            return data_predictions, train_accuracy, test_accuracy, feat_imp_df
        dtest_predictions = self.xgb1.predict(self.test_data[self.predictors])
        test_accuracy = metrics.accuracy_score(self.test_data[self.target_col].values, dtest_predictions)
        data_predictions = pd.Series(np.concatenate((dtrain_predictions, dtest_predictions)))
        return data_predictions, train_accuracy, test_accuracy, feat_imp_df

    def get_model_objects(self):
        return self.enc, self.xgb1


class ModelPredict:

    def __init__(self, masterdf, enc, xgb1, points_pred_col):
        self.masterdf = masterdf
        self.enc = enc
        self.xgb1 = xgb1
        self.points_pred_col = points_pred_col
    def get_normalized_data(self):
        """

        :return:
        """
        self.masterdf = self.enc.transform(self.masterdf)
        return

    def predict_points(self):
        """

        :return:
        """
        self.predictors =[i for i in self.xgb1.predictors if i in self.masterdf.columns]
        dtest_predictions = self.xgb1.predict(self.test_data[self.predictors])
        return dtest_predictions
import numpy as np
import pandas as pd
import util
import ml_alg
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import time
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime,timedelta
from  sklearn.model_selection import GridSearchCV
import pickle
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression






def g_weights(moment_id1, moment_id2, moment_id3, tollgate_id, direction):
    train_path = 'dataSets/training/'
    test_path = "dataSets/testing_phase1/"
    data_path = "dataSets/data/"
    train_df = pd.read_csv(train_path + 'data_' + str(tollgate_id) + '_' + str(direction) + '_final.csv', index_col=0)
    test_df = pd.read_csv(test_path + 'data_' + str(tollgate_id) + '_' + str(direction) + '_final.csv', index_col=0)
    train_df.drop(['volume'], axis=1, inplace=True)
    test_df.drop(['volume'], axis=1, inplace=True)
    sLength1 = len(train_df)
    sLength2 = len(test_df)
    train_df = train_df.assign(e=pd.Series(np.zeros(sLength1)).values)
    test_df = test_df.assign(e=pd.Series(np.ones(sLength2)).values)

    test_df2 = test_df.ix[
        ['2016-10-18 ' + moment_id1, '2016-10-19 ' + moment_id1, '2016-10-20 ' + moment_id1, '2016-10-21 ' + moment_id1,
         '2016-10-22 ' + moment_id1, '2016-10-23 ' + moment_id1, '2016-10-24 ' + moment_id1, '2016-10-18 ' + moment_id2,
         '2016-10-19 ' + moment_id2, '2016-10-20 ' + moment_id2, '2016-10-21 ' + moment_id2, '2016-10-22 ' + moment_id2,
         '2016-10-23 ' + moment_id2, '2016-10-24 ' + moment_id2]]
    train_df2 = train_df.ix[moment_id3:]

    all_df = pd.concat((train_df2, test_df2), axis=0)
    dweek = pd.get_dummies(all_df['dayofweek'], prefix='dayofweek')
    dhour = pd.get_dummies(all_df['hour'], prefix='hour')
    numeric_cols = all_df.columns[all_df.dtypes != 'object']
    numeric_cols = numeric_cols.values[:-1]
    numeric_col_means = all_df.loc[:, numeric_cols].mean()
    numeric_col_std = all_df.loc[:, numeric_cols].std(ddof=0)
    all_df.loc[:, numeric_cols] = (all_df.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std
    all_df.drop(['hour'], axis=1, inplace=True)
    all_df.drop(['dayofweek'], axis=1, inplace=True)
    all_df = pd.concat([all_df, dweek, dhour], axis=1)
    all_df = all_df.dropna(axis=1)
    # dummy_train_df_1 = all_df.loc[train_df2.index]
    # dummy_test_df = all_df.loc[test_df2.index]
    y_train = all_df.pop('e').values
    # y_test = test_df2.pop('e').values
    # X_train = dummy_train_df_1.values
    X_train = all_df.values
    # X_test = dummy_test_df.values
    y_train = y_train.astype('int64')
    w = len(X_train) / (2 * np.bincount(y_train))
    #sample_weight = np.ones(len(train_df2)) * w[0]
    #sample_weight2 = np.ones(len(test_df2)) * w[1]
    #sample_weight = np.concatenate((sample_weight, sample_weight2))
    clf = LogisticRegression(class_weight='balanced', random_state=42, n_jobs=-1)
    #clf = XGBClassifier(n_estimators=1000, seed=42)
    #clf.fit(X_train, y_train, sample_weight=sample_weight)
    clf.fit(X_train, y_train)
    pred_prob = clf.predict_proba(X_train)
    weight = pred_prob[:, 1] / pred_prob[:, 0]
    weight = weight[:-len(test_df2)]
    return weight


def g_data(moment_id1, moment_id2, moment_id3, tollgate_id, direction):
    #first 7 is for am, second 7 is for pm.
    print(moment_id1 + str(tollgate_id) + str(direction))
    train_path = 'dataSets/training/'
    test_path = "dataSets/testing_phase1/"
    data_path = "dataSets/data/"

    train_df = pd.read_csv(train_path + 'data_' + str(tollgate_id) + '_' + str(direction) + '_final.csv', index_col=0)
    test_df = pd.read_csv(test_path + 'data_' + str(tollgate_id) + '_' + str(direction) + '_final.csv', index_col=0)

    test_df.drop(['volume'], axis=1, inplace=True)

    test_df2 = test_df.ix[['2016-10-18 ' + moment_id1, '2016-10-19 ' + moment_id1,  '2016-10-20 ' + moment_id1,  '2016-10-21 ' + moment_id1,  '2016-10-22 ' + moment_id1,  '2016-10-23 ' + moment_id1,  '2016-10-24 ' + moment_id1, '2016-10-18 ' + moment_id2, '2016-10-19 ' + moment_id2, '2016-10-20 ' + moment_id2, '2016-10-21 ' + moment_id2, '2016-10-22 ' + moment_id2, '2016-10-23 ' + moment_id2, '2016-10-24 ' + moment_id2]]
    train_df2 = train_df.ix[moment_id3:]
    y_train1 = train_df2.pop('volume').values


    all_df = pd.concat((train_df2, test_df2), axis=0)
    dweek = pd.get_dummies(all_df['dayofweek'], prefix='dayofweek')
    dhour = pd.get_dummies(all_df['hour'], prefix='hour')
    numeric_cols = all_df.columns[all_df.dtypes != 'object']
    numeric_col_means = all_df.loc[:, numeric_cols].mean()
    numeric_col_std = all_df.loc[:, numeric_cols].std(ddof=0)
    all_df.loc[:, numeric_cols] = (all_df.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std
    all_df.drop(['hour'], axis=1, inplace=True)
    all_df.drop(['dayofweek'], axis=1, inplace=True)
    all_df = pd.concat([all_df, dweek, dhour], axis=1)
    all_df = all_df.dropna(axis=1)
    dummy_train_df_1 = all_df.loc[train_df2.index]
    dummy_test_df = all_df.loc[test_df2.index]

    X_train = dummy_train_df_1.values
    weight = g_weights(moment_id1, moment_id2, moment_id3, tollgate_id, direction)

    X_test = dummy_test_df.values
    data = {'X_train':X_train, 'X_test':X_test, 'y_train':y_train1, 'weight': weight}
    with open(data_path + moment_id1[:2] + moment_id1[3:5] + str(tollgate_id) + str(direction) + 'data3.pkl', 'wb') as handle:
        pickle.dump(data, handle)


g_data('08:00:00', '17:00:00', '2016-09-19 04:00:00', 1, 0)
g_data('08:20:00', '17:20:00', '2016-09-19 04:00:00', 1, 0)
g_data('08:40:00', '17:40:00', '2016-09-19 04:00:00', 1, 0)
g_data('09:00:00', '18:00:00', '2016-09-19 04:00:00', 1, 0)
g_data('09:20:00', '18:20:00', '2016-09-19 04:00:00', 1, 0)
g_data('09:40:00', '18:40:00', '2016-09-19 04:00:00', 1, 0)

g_data('08:00:00', '17:00:00', '2016-09-19 04:00:00', 1, 1)
g_data('08:20:00', '17:20:00', '2016-09-19 04:00:00', 1, 1)
g_data('08:40:00', '17:40:00', '2016-09-19 04:00:00', 1, 1)
g_data('09:00:00', '18:00:00', '2016-09-19 04:00:00', 1, 1)
g_data('09:20:00', '18:20:00', '2016-09-19 04:00:00', 1, 1)
g_data('09:40:00', '18:40:00', '2016-09-19 04:00:00', 1, 1)

g_data('08:00:00', '17:00:00', '2016-09-19 04:00:00', 2, 0)
g_data('08:20:00', '17:20:00', '2016-09-19 04:00:00', 2, 0)
g_data('08:40:00', '17:40:00', '2016-09-19 04:00:00', 2, 0)
g_data('09:00:00', '18:00:00', '2016-09-19 04:00:00', 2, 0)
g_data('09:20:00', '18:20:00', '2016-09-19 04:00:00', 2, 0)
g_data('09:40:00', '18:40:00', '2016-09-19 04:00:00', 2, 0)

g_data('08:00:00', '17:00:00', '2016-09-19 04:00:00', 3, 0)
g_data('08:20:00', '17:20:00', '2016-09-19 04:00:00', 3, 0)
g_data('08:40:00', '17:40:00', '2016-09-19 04:00:00', 3, 0)
g_data('09:00:00', '18:00:00', '2016-09-19 04:00:00', 3, 0)
g_data('09:20:00', '18:20:00', '2016-09-19 04:00:00', 3, 0)
g_data('09:40:00', '18:40:00', '2016-09-19 04:00:00', 3, 0)

g_data('08:00:00', '17:00:00', '2016-09-19 04:00:00', 3, 1)
g_data('08:20:00', '17:20:00', '2016-09-19 04:00:00', 3, 1)
g_data('08:40:00', '17:40:00', '2016-09-19 04:00:00', 3, 1)
g_data('09:00:00', '18:00:00', '2016-09-19 04:00:00', 3, 1)
g_data('09:20:00', '18:20:00', '2016-09-19 04:00:00', 3, 1)
g_data('09:40:00', '18:40:00', '2016-09-19 04:00:00', 3, 1)
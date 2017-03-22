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

def g_data(moment_id1, moment_id2, moment_id3, tollgate_id, direction):
    #first 7 is for am, second 7 is for pm.
    train_path = 'dataSets/training/'
    test_path = "dataSets/testing_phase1/"
    data_path = "dataSets/data/"

    train_df = pd.read_csv(train_path + 'data_' + str(tollgate_id) + '_' + str(direction) + '_final2.csv', index_col=0)
    test_df = pd.read_csv(test_path + 'data_' + str(tollgate_id) + '_' + str(direction) + '_final.csv', index_col=0)

    test_df.drop(['volume'], axis=1, inplace=True)

    test_df2 = test_df.ix[['2016-10-18 ' + moment_id1, '2016-10-19 ' + moment_id1,  '2016-10-20 ' + moment_id1,  '2016-10-21 ' + moment_id1,  '2016-10-22 ' + moment_id1,  '2016-10-23 ' + moment_id1,  '2016-10-24 ' + moment_id1, '2016-10-18 ' + moment_id2, '2016-10-19 ' + moment_id2, '2016-10-20 ' + moment_id2, '2016-10-21 ' + moment_id2, '2016-10-22 ' + moment_id2, '2016-10-23 ' + moment_id2, '2016-10-24 ' + moment_id2]]
    train_df2 = train_df.ix[moment_id3:]
    y_train1 = train_df2.pop('volume')

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
    dummy_train_df_1 = all_df[:len(train_df2)]
    dummy_test_df = all_df.loc[test_df2.index]
    X_train = dummy_train_df_1.values
    X_test = dummy_test_df.values
    data = {'X_train':X_train, 'X_test':X_test, 'y_train1':y_train1}
    with open(data_path + moment_id1[:2] + moment_id1[3:5] + str(tollgate_id) + str(direction) + 'data.pkl', 'wb') as handle:
        pickle.dump(data, handle)



g_data('08:00:00', '17:00:00', '2016-09-19 02:00:00', 1, 0)
g_data('08:20:00', '17:20:00', '2016-09-19 02:20:00', 1, 0)
g_data('08:40:00', '17:40:00', '2016-09-19 02:40:00', 1, 0)
g_data('09:00:00', '18:00:00', '2016-09-19 03:00:00', 1, 0)
g_data('09:20:00', '18:20:00', '2016-09-19 03:20:00', 1, 0)
g_data('09:40:00', '18:40:00', '2016-09-19 03:40:00', 1, 0)

g_data('08:00:00', '17:00:00', '2016-09-19 02:00:00', 1, 1)
g_data('08:20:00', '17:20:00', '2016-09-19 02:20:00', 1, 1)
g_data('08:40:00', '17:40:00', '2016-09-19 02:40:00', 1, 1)
g_data('09:00:00', '18:00:00', '2016-09-19 03:00:00', 1, 1)
g_data('09:20:00', '18:20:00', '2016-09-19 03:20:00', 1, 1)
g_data('09:40:00', '18:40:00', '2016-09-19 03:40:00', 1, 1)

g_data('08:00:00', '17:00:00', '2016-09-19 02:00:00', 2, 0)
g_data('08:20:00', '17:20:00', '2016-09-19 02:20:00', 2, 0)
g_data('08:40:00', '17:40:00', '2016-09-19 02:40:00', 2, 0)
g_data('09:00:00', '18:00:00', '2016-09-19 03:00:00', 2, 0)
g_data('09:20:00', '18:20:00', '2016-09-19 03:20:00', 2, 0)
g_data('09:40:00', '18:40:00', '2016-09-19 03:40:00', 2, 0)

g_data('08:00:00', '17:00:00', '2016-09-19 02:00:00', 3, 0)
g_data('08:20:00', '17:20:00', '2016-09-19 02:20:00', 3, 0)
g_data('08:40:00', '17:40:00', '2016-09-19 02:40:00', 3, 0)
g_data('09:00:00', '18:00:00', '2016-09-19 03:00:00', 3, 0)
g_data('09:20:00', '18:20:00', '2016-09-19 03:20:00', 3, 0)
g_data('09:40:00', '18:40:00', '2016-09-19 03:40:00', 3, 0)

g_data('08:00:00', '17:00:00', '2016-09-19 02:00:00', 3, 1)
g_data('08:20:00', '17:20:00', '2016-09-19 02:20:00', 3, 1)
g_data('08:40:00', '17:40:00', '2016-09-19 02:40:00', 3, 1)
g_data('09:00:00', '18:00:00', '2016-09-19 03:00:00', 3, 1)
g_data('09:20:00', '18:20:00', '2016-09-19 03:20:00', 3, 1)
g_data('09:40:00', '18:40:00', '2016-09-19 03:40:00', 3, 1)


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


def MAPE(ground_truth, predictions):
    ground_truth[ground_truth == 0] = math.inf
    diff = np.abs((ground_truth - predictions)/ground_truth).mean()
    return diff

def train_for_moment(moment_id1, moment_id2, moment_id3, train_df, test_df):
    #first 7 is for am, second 7 is for pm.
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
    dummy_train_df_1 = all_df.loc[train_df2.index]
    dummy_test_df = all_df.loc[test_df2.index]
    X_train = dummy_train_df_1.values
    X_test = dummy_test_df.values

    loss = make_scorer(MAPE, greater_is_better=False)
    '''
    max_features = [.1, .3, .5]
    test_scores = []
    best_score = 1
    for max_feat in max_features:
        start_time = time.time()
        clf = RandomForestRegressor(n_estimators=200, criterion = 'mae', max_features=max_feat, n_jobs = -1)
        test_score = -cross_val_score(clf, X_train, y_train1, cv=5, scoring=loss).mean()
        test_scores.append(test_score)
        if test_score < best_score:
            best_score = test_score
            best_index = max_feat
        print("max_feat:" + str(max_feat) + " loss = " + str(test_score))
        print("time:" + str(time.time() - start_time))
    '''
    best_index = 0.4
    clf = RandomForestRegressor(n_estimators=500, criterion = 'mae', min_samples_leaf=10, max_features=best_index, n_jobs = -1)
    clf.fit(X_train, y_train1)
    pred = clf.predict(X_test)
    return pred


def main(tollgate_id, direction, fw):
    train_path = 'dataSets/training/'
    test_path = "dataSets/testing_phase1/"

    train_df = pd.read_csv(train_path + 'data_' + str(tollgate_id) + '_' + str(direction) + '_final.csv', index_col=0)
    test_df = pd.read_csv(test_path + 'data_' + str(tollgate_id) + '_' + str(direction) + '_final.csv', index_col=0)

    test_df.drop(['volume'], axis=1, inplace=True)

    start_time = time.time()
    pred = train_for_moment('08:00:00', '17:00:00', '2016-09-19 02:00:00', train_df, test_df)

    for i in range(18, 25):
        time_window = datetime(2016, 10, i, 8, 0, 0)
        time_window_end = time_window + timedelta(minutes=20)
        out_line = ','.join(['"' + str(tollgate_id) + '"',
                             '"[' + str(time_window) + ',' + str(time_window_end) + ')"',
                             '"' + str(direction) + '"',
                             '"' + str(pred[i - 18]) + '"',
                             ]) + '\n'
        fw.writelines(out_line)

        time_window = datetime(2016, 10, i, 17, 0, 0)
        time_window_end = time_window + timedelta(minutes=20)
        out_line = ','.join(['"' + str(tollgate_id) + '"',
                             '"[' + str(time_window) + ',' + str(time_window_end) + ')"',
                             '"' + str(direction) + '"',
                             '"' + str(pred[7 + i - 18]) + '"',
                             ]) + '\n'
        fw.writelines(out_line)
    print('1, time:' + str(time.time() - start_time))
    pred = train_for_moment('08:20:00', '17:20:00', '2016-09-19 02:20:00', train_df, test_df)

    for i in range(18, 25):
        time_window = datetime(2016, 10, i, 8, 20, 0)
        time_window_end = time_window + timedelta(minutes=20)
        out_line = ','.join(['"' + str(tollgate_id) + '"',
                             '"[' + str(time_window) + ',' + str(time_window_end) + ')"',
                             '"' + str(direction) + '"',
                             '"' + str(pred[i - 18]) + '"',
                             ]) + '\n'
        fw.writelines(out_line)

        time_window = datetime(2016, 10, i, 17, 20, 0)
        time_window_end = time_window + timedelta(minutes=20)
        out_line = ','.join(['"' + str(tollgate_id) + '"',
                             '"[' + str(time_window) + ',' + str(time_window_end) + ')"',
                             '"' + str(direction) + '"',
                             '"' + str(pred[7 + i - 18]) + '"',
                             ]) + '\n'
        fw.writelines(out_line)
    print('2, time:' + str(time.time() - start_time))

    pred = train_for_moment('08:40:00', '17:40:00', '2016-09-19 02:40:00', train_df, test_df)

    for i in range(18, 25):
        time_window = datetime(2016, 10, i, 8, 40, 0)
        time_window_end = time_window + timedelta(minutes=20)
        out_line = ','.join(['"' + str(tollgate_id) + '"',
                             '"[' + str(time_window) + ',' + str(time_window_end) + ')"',
                             '"' + str(direction) + '"',
                             '"' + str(pred[i - 18]) + '"',
                             ]) + '\n'
        fw.writelines(out_line)

        time_window = datetime(2016, 10, i, 17, 40, 0)
        time_window_end = time_window + timedelta(minutes=20)
        out_line = ','.join(['"' + str(tollgate_id) + '"',
                             '"[' + str(time_window) + ',' + str(time_window_end) + ')"',
                             '"' + str(direction) + '"',
                             '"' + str(pred[7 + i - 18]) + '"',
                             ]) + '\n'
        fw.writelines(out_line)
    print('3, time:' + str(time.time() - start_time))

    pred = train_for_moment('09:00:00', '18:00:00', '2016-09-19 03:00:00', train_df, test_df)

    for i in range(18, 25):
        time_window = datetime(2016, 10, i, 9, 0, 0)
        time_window_end = time_window + timedelta(minutes=20)
        out_line = ','.join(['"' + str(tollgate_id) + '"',
                             '"[' + str(time_window) + ',' + str(time_window_end) + ')"',
                             '"' + str(direction) + '"',
                             '"' + str(pred[i - 18]) + '"',
                             ]) + '\n'
        fw.writelines(out_line)

        time_window = datetime(2016, 10, i, 18, 0, 0)
        time_window_end = time_window + timedelta(minutes=20)
        out_line = ','.join(['"' + str(tollgate_id) + '"',
                             '"[' + str(time_window) + ',' + str(time_window_end) + ')"',
                             '"' + str(direction) + '"',
                             '"' + str(pred[7 + i - 18]) + '"',
                             ]) + '\n'
        fw.writelines(out_line)
    print('4, time:' + str(time.time() - start_time))

    pred = train_for_moment('09:20:00', '18:20:00', '2016-09-19 03:20:00', train_df, test_df)

    for i in range(18, 25):
        time_window = datetime(2016, 10, i, 9, 20, 0)
        time_window_end = time_window + timedelta(minutes=20)
        out_line = ','.join(['"' + str(tollgate_id) + '"',
                             '"[' + str(time_window) + ',' + str(time_window_end) + ')"',
                             '"' + str(direction) + '"',
                             '"' + str(pred[i - 18]) + '"',
                             ]) + '\n'
        fw.writelines(out_line)

        time_window = datetime(2016, 10, i, 18, 20, 0)
        time_window_end = time_window + timedelta(minutes=20)
        out_line = ','.join(['"' + str(tollgate_id) + '"',
                             '"[' + str(time_window) + ',' + str(time_window_end) + ')"',
                             '"' + str(direction) + '"',
                             '"' + str(pred[7 + i - 18]) + '"',
                             ]) + '\n'
        fw.writelines(out_line)
    print('5, time:' + str(time.time() - start_time))

    pred = train_for_moment('09:40:00', '18:40:00', '2016-09-19 03:40:00', train_df, test_df)

    for i in range(18, 25):
        time_window = datetime(2016, 10, i, 9, 40, 0)
        time_window_end = time_window + timedelta(minutes=20)
        out_line = ','.join(['"' + str(tollgate_id) + '"',
                             '"[' + str(time_window) + ',' + str(time_window_end) + ')"',
                             '"' + str(direction) + '"',
                             '"' + str(pred[i - 18]) + '"',
                             ]) + '\n'
        fw.writelines(out_line)

        time_window = datetime(2016, 10, i, 18, 40, 0)
        time_window_end = time_window + timedelta(minutes=20)
        out_line = ','.join(['"' + str(tollgate_id) + '"',
                             '"[' + str(time_window) + ',' + str(time_window_end) + ')"',
                             '"' + str(direction) + '"',
                             '"' + str(pred[7 + i - 18]) + '"',
                             ]) + '\n'
        fw.writelines(out_line)
    print('6, time:' + str(time.time() - start_time))


out_file_name = 'result_03_21.csv'
test_path = "dataSets/testing_phase1/"
fw = open(test_path + out_file_name, 'w')
fw.writelines(','.join(['"tollgate_id"', '"time_window"', '"direction"', '"volume"']) + '\n')
print('tollgate:' + str(1) + 'direction:' + str(0))
main(1, 0, fw)
print('tollgate:' + str(1) + 'direction:' + str(1))

main(1, 1, fw)
print('tollgate:' + str(2) + 'direction:' + str(0))

main(2, 0, fw)
print('tollgate:' + str(3) + 'direction:' + str(0))

main(3, 0, fw)
print('tollgate:' + str(3) + 'direction:' + str(1))

main(3, 1, fw)
fw.close()
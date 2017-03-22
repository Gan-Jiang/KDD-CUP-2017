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
def MAPE(ground_truth, predictions):
    ground_truth[ground_truth == 0] = math.inf
    diff = np.abs((ground_truth - predictions)/ground_truth).mean()
    return diff

def train_for_moment(moment_id1, moment_id2, moment_id3, tollgate_id, direction):
    data_path = "dataSets/data/"
    with open(data_path + moment_id1[:2] + moment_id1[3:5] + str(tollgate_id) + str(direction) + 'data.pkl', 'rb') as handle:
        data = pickle.load(handle)
    y_train1 = data['y_train1']
    X_train = data['X_train']
    X_test = data['X_test']
    '''
    loss = make_scorer(MAPE, greater_is_better=False)
    rf = RandomForestRegressor(n_estimators=100, criterion='mae', n_jobs=-1)
    parameters = {'max_features': [0.25, 0.35, 0.45], 'min_samples_leaf': [5, 10, 20]}
    clf = GridSearchCV(rf, parameters, scoring = loss, n_jobs = -1)
    params = clf.best_params_
    '''
    rf = RandomForestRegressor(n_estimators=500, criterion='mae', n_jobs=-1, max_features = 'sqrt', min_samples_leaf = 10)
    rf.fit(X_train, y_train1)
    pred = rf.predict(X_test)

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
    return pred


def main(tollgate_id, direction):
    out_file_name = 'result_03_22.csv'
    test_path = "dataSets/testing_phase1/"
    fw = open(test_path + out_file_name, 'a')

    start_time = time.time()
    pred = train_for_moment('08:00:00', '17:00:00', '2016-09-19 02:00:00', tollgate_id, direction)

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
    pred = train_for_moment('08:20:00', '17:20:00', '2016-09-19 02:20:00', tollgate_id, direction)

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

    pred = train_for_moment('08:40:00', '17:40:00', '2016-09-19 02:40:00', tollgate_id, direction)

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

    pred = train_for_moment('09:00:00', '18:00:00', '2016-09-19 03:00:00', tollgate_id, direction)

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

    pred = train_for_moment('09:20:00', '18:20:00', '2016-09-19 03:20:00', tollgate_id, direction)

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

    pred = train_for_moment('09:40:00', '18:40:00', '2016-09-19 03:40:00', tollgate_id, direction)

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
    fw.close()


print('tollgate:' + str(1) + 'direction:' + str(0))
main(1, 1)



print('tollgate:' + str(1) + 'direction:' + str(1))

main(1, 1)
print('tollgate:' + str(2) + 'direction:' + str(0))

main(2, 0)
print('tollgate:' + str(3) + 'direction:' + str(0))

main(3, 0)
print('tollgate:' + str(3) + 'direction:' + str(1))

main(3, 1)

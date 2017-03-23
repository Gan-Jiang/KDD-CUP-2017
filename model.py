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
    ground_truth[ground_truth == 0] = 100000
    predictions[ground_truth == 100000] = 100000
    diff = np.abs((ground_truth - predictions)/ground_truth).mean()
    return diff

def MAPE2(ground_truth, predictions):
    ground_truth[ground_truth == 0] = 100000
    predictions[ground_truth == 100000] = 100000
    diff = np.abs((ground_truth - predictions)/ground_truth).sum()
    return diff

def train_for_moment(moment_id1, moment_id2, moment_id3, tollgate_id, direction):
    data_path = "dataSets/data/"
    with open(data_path + moment_id1[:2] + moment_id1[3:5] + str(tollgate_id) + str(direction) + 'data.pkl', 'rb') as handle:
        data = pickle.load(handle)
        #data = {'X_train': X_train, 'X_test': X_test, 'X_valid': X_valid, 'y_train': y_train1, 'y_valid': y_valid}

    y_train = data['y_train']
    X_train = data['X_train']
    X_valid = data['X_valid']
    X_test = data['X_test']
    y_valid = data['y_valid']
    np.random.seed(42)
    valid_data = np.concatenate((X_valid, y_valid.reshape(len(y_valid), 1)), axis = 1)
    np.random.shuffle(valid_data)
    X_valid = valid_data[:, :-1]
    y_valid = valid_data[:, -1].reshape(len(y_valid),)

    '''
    loss = make_scorer(MAPE, greater_is_better=False)
    rf = RandomForestRegressor(n_estimators=100, criterion='mae', n_jobs=-1)
    parameters = {'max_features': [0.25, 0.35, 0.45], 'min_samples_leaf': [5, 10, 20]}
    clf = GridSearchCV(rf, parameters, scoring = loss, n_jobs = -1)
    params = clf.best_params_
    '''
    #CV procedure
    CV_loss = 0
    fold = len(X_valid) // 10
    for i in range(10):
        #start_time = time.time()
        X_train_real = np.concatenate((X_train, X_valid[i*fold:(i+1)*fold]), axis=0)
        y_train_real = np.concatenate((y_train, y_valid[i*fold:(i+1)*fold]), axis=0)
        X_valid_real = np.concatenate((X_valid[:i*fold], X_valid[(i+1)*fold:10*fold]), axis=0)
        y_valid_real = np.concatenate((y_valid[:i*fold], y_valid[(i+1)*fold:10*fold]), axis=0)
        if (i == 9) and (len(X_valid) % fold != 0):
            X_valid_real = np.concatenate((X_valid_real, X_valid[(i + 1) * fold:]), axis=0)
            y_valid_real = np.concatenate((y_valid_real, y_valid[(i + 1) * fold:]), axis=0)
        rf = RandomForestRegressor(n_estimators=500, criterion='mse', n_jobs=-1, max_features=0.15, min_samples_leaf=10)
        rf.fit(X_train_real, y_train_real)
        pred = rf.predict(X_valid_real)
        CV_loss += MAPE2(y_valid_real, pred)
        #print("time:" + str(time.time() - start_time))
    CV_loss = CV_loss/len(X_valid)

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
    complete_train = np.concatenate((X_train, X_valid), axis=0)
    complete_label = np.concatenate((y_train, y_valid), axis=0)
    rf = RandomForestRegressor(n_estimators=500, criterion='mse', n_jobs=-1, max_features=0.15, min_samples_leaf=10)
    rf.fit(complete_train, complete_label)
    pred = rf.predict(X_test)
    return pred, CV_loss


def main(tollgate_id, direction):
    out_file_name = 'result_03_23.csv'
    test_path = "dataSets/testing_phase1/"
    fw = open(test_path + out_file_name, 'a')
    out_file_name2 = 'cv_loss.csv'
    fw2 = open(test_path + out_file_name2, 'a')
    start_time = time.time()
    total_loss = 0
    pred, CV_loss = train_for_moment('08:00:00', '17:00:00', '2016-09-19 02:00:00', tollgate_id, direction)
    total_loss += CV_loss

    out_line2 = ','.join(['"' + str(tollgate_id) + '"',
                         '"[' + '08:00:00'+ ',' + '17:00:00' + ')"',
                         '"' + str(direction) + '"',
                         '"' + str(CV_loss) + '"',
                         ]) + '\n'
    fw2.writelines(out_line2)

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
    pred , CV_loss = train_for_moment('08:20:00', '17:20:00', '2016-09-19 02:20:00', tollgate_id, direction)
    total_loss += CV_loss


    out_line2 = ','.join(['"' + str(tollgate_id) + '"',
                         '"[' + '08:20:00' + ',' + '17:20:00' + ')"',
                         '"' + str(direction) + '"',
                         '"' + str(CV_loss) + '"',
                         ]) + '\n'
    fw2.writelines(out_line2)


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

    pred , CV_loss = train_for_moment('08:40:00', '17:40:00', '2016-09-19 02:40:00', tollgate_id, direction)
    total_loss += CV_loss

    out_line2 = ','.join(['"' + str(tollgate_id) + '"',
                         '"[' + '08:40:00' + ',' + '17:40:00' + ')"',
                         '"' + str(direction) + '"',
                         '"' + str(CV_loss) + '"',
                         ]) + '\n'
    fw2.writelines(out_line2)


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

    pred , CV_loss= train_for_moment('09:00:00', '18:00:00', '2016-09-19 03:00:00', tollgate_id, direction)
    total_loss += CV_loss

    out_line2 = ','.join(['"' + str(tollgate_id) + '"',
                         '"[' + '09:00:00' + ',' + '18:00:00' + ')"',
                         '"' + str(direction) + '"',
                         '"' + str(CV_loss) + '"',
                         ]) + '\n'
    fw2.writelines(out_line2)

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

    pred , CV_loss= train_for_moment('09:20:00', '18:20:00', '2016-09-19 03:20:00', tollgate_id, direction)
    total_loss += CV_loss
    out_line2 = ','.join(['"' + str(tollgate_id) + '"',
                         '"[' + '09:20:00' + ',' + '18:20:00' + ')"',
                         '"' + str(direction) + '"',
                         '"' + str(CV_loss) + '"',
                         ]) + '\n'
    fw2.writelines(out_line2)


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

    pred , CV_loss= train_for_moment('09:40:00', '18:40:00', '2016-09-19 03:40:00', tollgate_id, direction)
    total_loss += CV_loss
    out_line2 = ','.join(['"' + str(tollgate_id) + '"',
                         '"[' + '09:40:00' + ',' + '18:40:00' + ')"',
                         '"' + str(direction) + '"',
                         '"' + str(CV_loss) + '"',
                         ]) + '\n'
    fw2.writelines(out_line2)


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
    fw2.close()
    return total_loss/6

out_file_name = 'result_03_23.csv'
test_path = "dataSets/testing_phase1/"
fw = open(out_file_name, 'w')
fw.writelines(','.join(['"tollgate_id"', '"time_window"', '"direction"', '"volume"']) + '\n')
fw.close()
T_loss = 0
print('tollgate:' + str(1) + 'direction:' + str(0))
total_loss = main(1, 0)
T_loss += total_loss
print('tollgate:' + str(1) + 'direction:' + str(1))

total_loss = main(1, 1)
T_loss += total_loss

print('tollgate:' + str(2) + 'direction:' + str(0))

total_loss = main(2, 0)


T_loss += total_loss
print('tollgate:' + str(3) + 'direction:' + str(0))

total_loss = main(3, 0)
T_loss += total_loss

print('tollgate:' + str(3) + 'direction:' + str(1))

total_loss = main(3, 1)
T_loss += total_loss
T_loss = T_loss/5

test_path = "dataSets/testing_phase1/"
out_file_name2 = 'cv_loss.csv'
fw2 = open(test_path + out_file_name2, 'a')
fw2.writelines(str(T_loss) + '\n')
fw2.close()

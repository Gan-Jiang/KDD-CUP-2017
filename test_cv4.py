import numpy as np
import pandas as pd
import util
import ml_alg
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
import time
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime,timedelta
from  sklearn.model_selection import GridSearchCV
import pickle
from hyperopt import fmin, hp, tpe
from xgboost import XGBRegressor
from sklearn.svm import SVR
import random

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

def MAPE3(ground_truth, predictions, weight):
    ground_truth[ground_truth == 0] = 100000
    predictions[ground_truth == 100000] = 100000
    diff = (np.abs((ground_truth - predictions)/ground_truth)*weight).sum()
    return diff


def randomforest_objective(params):
    #CV procedure
    global X_train, y_train, weight
    t = params['type']
    del params['type']
    if t == 'rf':
        clf = RandomForestRegressor(n_estimators=500, criterion='mse', n_jobs=-1,random_state = 42, max_features = 0.15,min_samples_leaf = 10,  **params)
    elif t == 'xgboost':
        clf = XGBRegressor(n_estimators = 1000, seed = 42, **params)
    elif t == 'knn':
        clf = KNeighborsRegressor()
    elif t == 'SVR':
        clf = SVR()
    CV_loss = 0
    #348 of two hours in total for training data.
    CV_result = []
    candadate_set = np.array(range(29))
    m_seeds = [797,533,295,997,841,55]
    for seed in m_seeds:
        np.random.seed(seed)
        valid_index = np.random.choice(29, 6)
        X_valid_real = np.array([])
        y_valid_real = np.array([])
        weight_valid = np.array([])

        X_train_real = np.array([])
        y_train_real = np.array([])
        weight_train = np.array([])
        for i in valid_index:
            try:
                X_valid_real = np.concatenate((X_valid_real, X_train[(i*72 + 12):(i*72 + 18)]), axis=0)
                y_valid_real = np.concatenate((y_valid_real, y_train[(i*72 + 12):(i*72 + 18)]), axis=0)
                weight_valid = np.concatenate((weight_valid, weight[(i*72 + 12):(i*72 + 18)]), axis=0)
            except:
                X_valid_real = X_train[(i*72 + 12):(i*72 + 18)]
                y_valid_real = y_train[(i*72 + 12):(i*72 + 18)]
                weight_valid = weight[(i*72 + 12):(i*72 + 18)]
            X_valid_real = np.concatenate((X_valid_real, X_train[(i * 72 + 12 + 27):(i * 72 + 18 + 27)]), axis=0)
            y_valid_real = np.concatenate((y_valid_real, y_train[(i * 72 + 12 + 27):(i * 72 + 18 + 27)]), axis=0)
            weight_valid = np.concatenate((weight_valid, weight[(i * 72 + 12 + 27):(i * 72 + 18 + 27)]), axis=0)

        train_index = np.delete(candadate_set, valid_index, axis = 0)

        for i in train_index:
            if i == 0:
                try:
                    X_train_real = np.concatenate((X_train_real, X_train[:60]), axis=0)
                    y_train_real = np.concatenate((y_train_real, y_train[i:60]), axis=0)
                    weight_train = np.concatenate((weight_train, weight[:60]), axis=0)
                except:
                    X_train_real = X_train[:60]
                    y_train_real = y_train[:60]
                    weight_train = weight[:60]
            else:

                try:
                    X_train_real = np.concatenate((X_train_real, X_train[(60 + (i-1) * 72): (60 + i * 72)]), axis=0)
                    y_train_real = np.concatenate((y_train_real, y_train[(60 + (i-1) * 72): (60 + i * 72)]), axis=0)
                    weight_train = np.concatenate((weight_train, weight[(60 + (i-1) * 72): (60 + i * 72)]), axis=0)
                except:
                    X_train_real = X_train[(60 + (i-1) * 72): (60 + i * 72)]
                    y_train_real = y_train[(60 + (i-1) * 72): (60 + i * 72)]
                    weight_train = weight[(60 + (i-1) * 72): (60 + i * 72)]

        #rf = RandomForestRegressor(n_estimators=500, criterion='mse', n_jobs=-1, max_features=max_features, min_samples_leaf=min_samples_leaf)
        #clf = RandomForestRegressor(n_estimators=500, criterion='mse', n_jobs=-1, **params)
        clf.fit(X_train_real, y_train_real)
        #, sample_weight = weight_train)
        pred = clf.predict(X_valid_real)
        CV_loss += MAPE2(y_valid_real.copy(), pred)
        CV_result.append(MAPE2(y_valid_real.copy(), pred))
        #print("time:" + str(time.time() - start_time))
    CV_loss = CV_loss/(6*12*len(m_seeds))
    return CV_loss

def train_for_moment(moment_id1, moment_id2, moment_id3, tollgate_id, direction):
    global X_train, y_train, weight
    data_path = "dataSets/data/"
    with open(data_path + moment_id1[:2] + moment_id1[3:5] + str(tollgate_id) + str(direction) + 'data3.pkl', 'rb') as handle:
        data = pickle.load(handle)
        #data = {'X_train': X_train, 'X_test': X_test, 'X_valid': X_valid, 'y_train': y_train1, 'y_valid': y_valid}

    y_train = data['y_train']
    X_train = data['X_train']
    X_test = data['X_test']
    weight = data['weight']


    train_data = np.concatenate((X_train, y_train.reshape(len(y_train), 1), weight.reshape(len(y_train), 1)), axis = 1)
    X_train = train_data[:, :-2]

    y_train = train_data[:, -2].reshape(len(y_train),)
    weight = train_data[:, -1].reshape(len(y_train),)


    space = hp.choice('classifier_type', [

        {
           'type': 'rf'
        #    'max_features': hp.loguniform('max_features', np.log(0.1), np.log(0.7)),
        #    'min_samples_leaf': hp.loguniform('min_samples_leaf', np.log(0.001), np.log(0.02))
        }
        #{
            #'type': 'xgboost'
            #'learning_rate':hp.uniform('learning_rate', 0.1, 0.2),
            #'C': hp.uniform('C', 0, 10.0),
            #'kernel': hp.choice('kernel', ['linear', 'rbf']),
            #'gamma': hp.uniform('gamma', 0, 20.0)
        #}
       # {
        #    'type': 'knn'
       # }
        #{
        #    'type':'SVR'
        #}
    ])

    # minimize the objective over the space
    #best = fmin(randomforest_objective, space, algo=tpe.suggest, max_evals=1)
    best = {'classifier_type': 0}
    print(best)
    if best['classifier_type'] == 0:
        best['type'] = 'SVR'
    del best['classifier_type']
    CV_loss = randomforest_objective(best)
    best['type'] = 'SVR'
    if best['type'] == 'rf':
        del best['type']

        clf = RandomForestRegressor(n_estimators=500, criterion='mse', n_jobs=-1,random_state = 42, max_features = 0.15,min_samples_leaf = 10, **best)
    elif best['type'] == 'xgboost':
        del best['type']

        clf = XGBRegressor(n_estimators = 1000, **best)
    elif best['type'] == 'knn':
        clf = KNeighborsRegressor()
    elif best['type'] == 'SVR':
        clf = SVR()

    clf.fit(X_train, y_train)
            #, sample_weight = weight)
    pred = clf.predict(X_test)
    return pred, CV_loss


def main(tollgate_id, direction):
    out_file_name = 'result_03_27.csv'
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


out_file_name = 'result_03_27.csv'
test_path = "dataSets/testing_phase1/"
fw = open(test_path + out_file_name, 'a')
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
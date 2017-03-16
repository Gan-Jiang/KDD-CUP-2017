# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Calculate volume for each 20-minute time window.
"""
import math
from datetime import datetime,timedelta

file_suffix = '.csv'
path = 'dataSets/training/'  # set the data directory
str_vehicle_model = 'vehicle_model '
str_has_etc = 'has_etc '
str_vehicle_type = 'vehicle_type '

def update_dic(volumes, start_time_window, tollgate_id):
    volumes[start_time_window][tollgate_id] = {}

    if tollgate_id != '2':
        volumes[start_time_window][tollgate_id]['0'] = {}
        volumes[start_time_window][tollgate_id]['1'] = {}
        volumes[start_time_window][tollgate_id]['0'][0] = 0
        volumes[start_time_window][tollgate_id]['1'][0] = 0

        volumes[start_time_window][tollgate_id]['0'][str_vehicle_model + '0'] = 0
        volumes[start_time_window][tollgate_id]['0'][str_vehicle_model + '1'] = 0
        volumes[start_time_window][tollgate_id]['0'][str_vehicle_model + '2'] = 0
        volumes[start_time_window][tollgate_id]['0'][str_vehicle_model + '3'] = 0
        volumes[start_time_window][tollgate_id]['0'][str_vehicle_model + '4'] = 0
        volumes[start_time_window][tollgate_id]['0'][str_vehicle_model + '5'] = 0
        volumes[start_time_window][tollgate_id]['0'][str_vehicle_model + '6'] = 0
        volumes[start_time_window][tollgate_id]['0'][str_vehicle_model + '7'] = 0

        volumes[start_time_window][tollgate_id]['0'][str_has_etc + '0'] = 0
        volumes[start_time_window][tollgate_id]['0'][str_has_etc + '1'] = 0

        volumes[start_time_window][tollgate_id]['0'][str_vehicle_type + '0\n'] = 0
        volumes[start_time_window][tollgate_id]['0'][str_vehicle_type + '1\n'] = 0
        volumes[start_time_window][tollgate_id]['0'][str_vehicle_type + '\n'] = 0

        volumes[start_time_window][tollgate_id]['1'][str_vehicle_model + '0'] = 0
        volumes[start_time_window][tollgate_id]['1'][str_vehicle_model + '1'] = 0
        volumes[start_time_window][tollgate_id]['1'][str_vehicle_model + '2'] = 0
        volumes[start_time_window][tollgate_id]['1'][str_vehicle_model + '3'] = 0
        volumes[start_time_window][tollgate_id]['1'][str_vehicle_model + '4'] = 0
        volumes[start_time_window][tollgate_id]['1'][str_vehicle_model + '5'] = 0
        volumes[start_time_window][tollgate_id]['1'][str_vehicle_model + '6'] = 0
        volumes[start_time_window][tollgate_id]['1'][str_vehicle_model + '7'] = 0

        volumes[start_time_window][tollgate_id]['1'][str_has_etc + '0'] = 0
        volumes[start_time_window][tollgate_id]['1'][str_has_etc + '1'] = 0

        volumes[start_time_window][tollgate_id]['1'][str_vehicle_type + '0\n'] = 0
        volumes[start_time_window][tollgate_id]['1'][str_vehicle_type + '1\n'] = 0
        volumes[start_time_window][tollgate_id]['1'][str_vehicle_type + '\n'] = 0
    else:
        volumes[start_time_window][tollgate_id]['0'] = {}
        volumes[start_time_window][tollgate_id]['0'][0] = 0

        volumes[start_time_window][tollgate_id]['0'][str_vehicle_model + '0'] = 0
        volumes[start_time_window][tollgate_id]['0'][str_vehicle_model + '1'] = 0
        volumes[start_time_window][tollgate_id]['0'][str_vehicle_model + '2'] = 0
        volumes[start_time_window][tollgate_id]['0'][str_vehicle_model + '3'] = 0
        volumes[start_time_window][tollgate_id]['0'][str_vehicle_model + '4'] = 0
        volumes[start_time_window][tollgate_id]['0'][str_vehicle_model + '5'] = 0
        volumes[start_time_window][tollgate_id]['0'][str_vehicle_model + '6'] = 0
        volumes[start_time_window][tollgate_id]['0'][str_vehicle_model + '7'] = 0

        volumes[start_time_window][tollgate_id]['0'][str_has_etc + '0'] = 0
        volumes[start_time_window][tollgate_id]['0'][str_has_etc + '1'] = 0

        volumes[start_time_window][tollgate_id]['0'][str_vehicle_type + '0\n'] = 0
        volumes[start_time_window][tollgate_id]['0'][str_vehicle_type + '1\n'] = 0
        volumes[start_time_window][tollgate_id]['0'][str_vehicle_type + '\n'] = 0
    return volumes

def avgVolume(in_file):

    out_suffix = '_20min_avg_volume'
    in_file_name = in_file + file_suffix
    out_file_name = in_file.split('_')[1] + out_suffix + file_suffix

    # Step 1: Load volume data
    fr = open(path + in_file_name, 'r')
    fr.readline()  # skip the header
    vol_data = fr.readlines()
    fr.close()

    # Step 2: Create a dictionary to caculate and store volume per time window
    volumes = {}  # key: time window value: dictionary
    start = datetime(2016, 9, 19, 0, 0)
    end = datetime(2016, 10, 18, 0, 0)
    while start < end:
        volumes[start] = {}
        volumes = update_dic(volumes, start, '1')
        volumes = update_dic(volumes, start, '2')
        volumes = update_dic(volumes, start, '3')
        start = start + timedelta(minutes=20)

    for i in range(len(vol_data)):
        each_pass = vol_data[i].replace('"', '').split(',')
        tollgate_id = each_pass[1]
        direction = each_pass[2]
        vehicle_model = each_pass[3]
        has_etc = each_pass[4]
        vehicle_type = each_pass[5]


        pass_time = each_pass[0]
        pass_time = datetime.strptime(pass_time, "%Y-%m-%d %H:%M:%S")
        time_window_minute = int(math.floor(pass_time.minute / 20) * 20)
        #print pass_time
        start_time_window = datetime(pass_time.year, pass_time.month, pass_time.day,
                                     pass_time.hour, time_window_minute, 0)

        volumes[start_time_window][tollgate_id][direction][0] += 1

        volumes[start_time_window][tollgate_id][direction][str_vehicle_model + vehicle_model] += 1
        volumes[start_time_window][tollgate_id][direction][str_has_etc + has_etc] += 1
        volumes[start_time_window][tollgate_id][direction][str_vehicle_type + vehicle_type] += 1


    # Step 3: format output for tollgate and direction per time window
    fw = open(path + out_file_name, 'w')
    fw.writelines(','.join(['"tollgate_id"', '"time_window"', '"direction"', '"volume"', '"volume_model0"', '"volume_model1"', '"volume_model2"', '"volume_model3"', '"volume_model4"', '"volume_model5"', '"volume_model6"', '"volume_model7"', '"volume_etc0"', '"volume_etc1"', '"volume_type0"', '"volume_type1"', '"volume_type2"']) + '\n')
    time_windows = list(volumes.keys())
    time_windows.sort()
    for time_window_start in time_windows:
        time_window_end = time_window_start + timedelta(minutes=20)
        for tollgate_id in volumes[time_window_start]:
            for direction in volumes[time_window_start][tollgate_id]:
                out_line = ','.join(['"' + str(tollgate_id) + '"','"[' + str(time_window_start) + ',' + str(time_window_end) + ')"','"' + str(direction) + '"','"' + str(volumes[time_window_start][tollgate_id][direction][0]) + '"','"' + str(volumes[time_window_start][tollgate_id][direction][str_vehicle_model + '0']) + '"','"' + str(volumes[time_window_start][tollgate_id][direction][str_vehicle_model + '1']) + '"','"' + str(volumes[time_window_start][tollgate_id][direction][str_vehicle_model + '2']) + '"','"' + str(volumes[time_window_start][tollgate_id][direction][str_vehicle_model + '3']) + '"','"' + str(volumes[time_window_start][tollgate_id][direction][str_vehicle_model + '4']) + '"','"' + str(volumes[time_window_start][tollgate_id][direction][str_vehicle_model + '5']) + '"','"' + str(volumes[time_window_start][tollgate_id][direction][str_vehicle_model + '6']) + '"','"' + str(volumes[time_window_start][tollgate_id][direction][str_vehicle_model + '7']) + '"','"' + str(volumes[time_window_start][tollgate_id][direction][str_has_etc + '0']) + '"','"' + str(volumes[time_window_start][tollgate_id][direction][str_has_etc + '1']) + '"','"' + str(volumes[time_window_start][tollgate_id][direction][str_vehicle_type + '0\n']) + '"','"' + str(volumes[time_window_start][tollgate_id][direction][str_vehicle_type + '1\n']) + '"','"' + str(volumes[time_window_start][tollgate_id][direction][str_vehicle_type + '\n']) + '"', ]) + '\n'
                fw.writelines(out_line)
    fw.close()

def main():

    in_file = 'volume(table 6)_training'
    avgVolume(in_file)

if __name__ == '__main__':
    main()




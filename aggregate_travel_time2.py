# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Objective:
Calculate the average travel time for each 20-minute time window.

"""

# import necessary modules
import math
from datetime import datetime,timedelta

file_suffix = '.csv'
path = 'dataSets/testing_phase1/'  # set the data directory

def update_dic(travel_times, route_id):
    start = datetime(2016, 10, 18, 6, 0)
    end = datetime(2016, 10, 18, 10, 0)
    while start < end:
        travel_times[route_id][start] = [0]
        start = start + timedelta(minutes=20)
    start = datetime(2016, 10, 19, 6, 0)
    end = datetime(2016, 10, 19, 10, 0)
    while start < end:
        travel_times[route_id][start] = [0]
        start = start + timedelta(minutes=20)
    start = datetime(2016, 10, 20, 6, 0)
    end = datetime(2016, 10, 20, 10, 0)
    while start < end:
        travel_times[route_id][start] = [0]
        start = start + timedelta(minutes=20)
    start = datetime(2016, 10, 21, 6, 0)
    end = datetime(2016, 10, 21, 10, 0)
    while start < end:
        travel_times[route_id][start] = [0]
        start = start + timedelta(minutes=20)
    start = datetime(2016, 10, 22, 6, 0)
    end = datetime(2016, 10, 22, 10, 0)
    while start < end:
        travel_times[route_id][start] = [0]
        start = start + timedelta(minutes=20)
    start = datetime(2016, 10, 23, 6, 0)
    end = datetime(2016, 10, 23, 10, 0)
    while start < end:
        travel_times[route_id][start] = [0]
        start = start + timedelta(minutes=20)
    start = datetime(2016, 10, 24, 6, 0)
    end = datetime(2016, 10, 24, 10, 0)
    while start < end:
        travel_times[route_id][start] = [0]
        start = start + timedelta(minutes=20)


    start = datetime(2016, 10, 18, 15, 0)
    end = datetime(2016, 10, 18, 19, 0)
    while start < end:
        travel_times[route_id][start] = [0]
        start = start + timedelta(minutes=20)
    start = datetime(2016, 10, 19, 15, 0)
    end = datetime(2016, 10, 19, 19, 0)
    while start < end:
        travel_times[route_id][start] = [0]
        start = start + timedelta(minutes=20)
    start = datetime(2016, 10, 20, 15, 0)
    end = datetime(2016, 10, 20, 19, 0)
    while start < end:
        travel_times[route_id][start] = [0]
        start = start + timedelta(minutes=20)
    start = datetime(2016, 10, 21, 15, 0)
    end = datetime(2016, 10, 21, 19, 0)
    while start < end:
        travel_times[route_id][start] = [0]
        start = start + timedelta(minutes=20)
    start = datetime(2016, 10, 22, 15, 0)
    end = datetime(2016, 10, 22, 19, 0)
    while start < end:
        travel_times[route_id][start] = [0]
        start = start + timedelta(minutes=20)
    start = datetime(2016, 10, 23, 15, 0)
    end = datetime(2016, 10, 23, 19, 0)
    while start < end:
        travel_times[route_id][start] = [0]
        start = start + timedelta(minutes=20)
    start = datetime(2016, 10, 24, 15, 0)
    end = datetime(2016, 10, 24, 19, 0)
    while start < end:
        travel_times[route_id][start] = [0]
        start = start + timedelta(minutes=20)

    return travel_times


def avgTravelTime(in_file):

    out_suffix = '_20min_avg_travel_time'
    in_file_name = in_file + file_suffix
    out_file_name = in_file.split('_')[1] + out_suffix + file_suffix

    # Step 1: Load trajectories
    fr = open(path + in_file_name, 'r')
    fr.readline()  # skip the header
    traj_data = fr.readlines()
    fr.close()
    print(traj_data[0])

    # Step 2: Create a dictionary to store travel time for each route per time window
    travel_times = {}  # key: route_id. Value is also a dictionary of which key is the start time for the time window and value is a list of travel times
    for i in range(len(traj_data)):
        each_traj = traj_data[i].replace('"', '').split(',')
        intersection_id = each_traj[0]
        tollgate_id = each_traj[1]

        route_id = intersection_id + '-' + tollgate_id
        if route_id not in travel_times.keys():
            travel_times[route_id] = {}
            travel_times = update_dic(travel_times, route_id)

        trace_start_time = each_traj[3]
        trace_start_time = datetime.strptime(trace_start_time, "%Y-%m-%d %H:%M:%S")
        time_window_minute = math.floor(trace_start_time.minute / 20) * 20
        start_time_window = datetime(trace_start_time.year, trace_start_time.month, trace_start_time.day,
                                     trace_start_time.hour, time_window_minute, 0)
        tt = float(each_traj[-1]) # travel time
        start = datetime(2016, 10, 18, 0, 0)
        if start_time_window < start:
            continue
        travel_times[route_id][start_time_window].append(tt)

    # Step 3: Calculate average travel time for each route per time window
    fw = open(path + out_file_name, 'w')
    fw.writelines(','.join(['"intersection_id"', '"tollgate_id"', '"time_window"', '"avg_travel_time"']) + '\n')
    for route in travel_times.keys():
        route_time_windows = list(travel_times[route].keys())
        route_time_windows.sort()
        for time_window_start in route_time_windows:
            tt_set = travel_times[route][time_window_start]
            if len(tt_set) == 1:
                avg_tt = 0
            else:
                avg_tt = round(sum(tt_set) / float(len(tt_set) - 1), 2)
            out_line = ','.join(['"' + route.split('-')[0] + '"', '"' + route.split('-')[1] + '"','"' + str(time_window_start) + '"','"' + str(avg_tt) + '"']) + '\n'
            fw.writelines(out_line)
    fw.close()

def main():

    in_file = 'trajectories(table 5)_test1'
    avgTravelTime(in_file)

if __name__ == '__main__':
    main()




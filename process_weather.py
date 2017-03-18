# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Objective:
Calculate the average travel time for each 20-minute time window.
I fill the data for 10-10 by averaging data at 10-9, 10-8, 10-11, 10-12
9-29 21 filled by the previous time and the latter time.
9-30 0
"""


from datetime import datetime,timedelta

file_suffix = '.csv'
path = 'dataSets/training/'  # set the data directory

def update_dic(weathers):
    start = datetime(2016, 9, 19, 0, 0)
    end = datetime(2016, 10, 18, 0, 0)
    while start < end:
        weathers[start] = {}
        start = start + timedelta(minutes=20)
    return weathers


def process_weather(in_file):

    out_suffix = '_20min_weather'
    in_file_name = in_file + file_suffix
    out_file_name = in_file.split('_')[1] + out_suffix + file_suffix

    # Step 1: Load trajectories
    fr = open(path + in_file_name, 'r')
    fr.readline()  # skip the header
    wea_data = fr.readlines()
    fr.close()
    # Step 2: Create a dictionary to store travel time for each route per time window
    weathers = {}
    weathers = update_dic(weathers)
    for i in range(len(wea_data)):
        each_wea = wea_data[i].replace('"', '').split(',')
        date = each_wea[0]
        hour = each_wea[1]
        pressure = each_wea[2]
        sea_pressure = each_wea[3]
        wind_direction = each_wea[4]
        wind_speed = each_wea[5]
        temperature = each_wea[6]
        rel_humidity = each_wea[7]
        precipitation = each_wea[8][:-1]
        start_time_window = datetime.strptime(date + ' ' + hour + ':0:0', "%m/%d/%Y %H:%M:%S")

        start = datetime(2016, 9, 19, 0, 0)
        if start_time_window < start:
            continue
        start = start_time_window
        end_time_window = start_time_window + timedelta(hours=3)
        while (start < end_time_window):
            weathers[start]['pressure'] = pressure
            weathers[start]['sea_pressure'] = sea_pressure
            weathers[start]['wind_direction'] = wind_direction
            weathers[start]['wind_speed'] = wind_speed
            weathers[start]['temperature'] = temperature
            weathers[start]['rel_humidity'] = rel_humidity
            weathers[start]['precipitation'] = precipitation
            start = start + timedelta(minutes=20)
    # Step 3: Calculate average travel time for each route per time window
    fw = open(path + out_file_name, 'w')
    fw.writelines(','.join(['"time"', '"pressure"', '"sea_pressure"', '"wind_direction"', '"wind_speed"', '"temperature"', '"rel_humidity"', '"precipitation"']) + '\n')
    for time in weathers.keys():
        out_line = ','.join(['"' + str(time) + '"','"' + str(weathers[time]['pressure']) + '"','"' + str(weathers[time]['sea_pressure']) + '"','"' + str(weathers[time]['wind_direction']) + '"','"' + str(weathers[time]['wind_speed']) + '"','"' + str(weathers[time]['temperature']) + '"','"' + str(weathers[time]['rel_humidity']) + '"','"' + str(weathers[time]['precipitation']) + '"']) + '\n'
        fw.writelines(out_line)
    fw.close()

def main():

    in_file = 'weather (table 7)_training'
    process_weather(in_file)

if __name__ == '__main__':
    main()
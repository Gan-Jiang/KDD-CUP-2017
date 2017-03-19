import numpy as np
import pandas as pd
from dateutil.parser import parse

path = 'dataSets/testing_phase1/'  # set the data directory
from datetime import datetime,timedelta

avg_tt = pd.read_csv(path + 'test1_20min_avg_travel_time_filled.csv')
#(A, 2)，(A, 3), (B, 1), (B, 3), (C, 1), (C, 3)
avg_tt2 = avg_tt.as_matrix()

result = {}
for i in avg_tt2:
    if parse(i[2]) in result:
        if i[0] == 'A' and i[1] == 2:
            result[parse(i[2])][0] = i[3]
        elif i[0] == 'A' and i[1] == 3:
            result[parse(i[2])][1] = i[3]
        elif i[0] == 'B' and i[1] == 1:
            result[parse(i[2])][2] = i[3]
        elif i[0] == 'B' and i[1] == 3:
            result[parse(i[2])][3] = i[3]
        elif i[0] == 'C' and i[1] == 1:
            result[parse(i[2])][4] = i[3]
        elif i[0] == 'C' and i[1] == 3:
            result[parse(i[2])][5] = i[3]
    else:
        result[parse(i[2])] = [0, 0, 0, 0, 0, 0]
        if i[0] == 'A' and i[1] == 2:
            result[parse(i[2])][0] = i[3]
        elif i[0] == 'A' and i[1] == 3:
            result[parse(i[2])][1] = i[3]
        elif i[0] == 'B' and i[1] == 1:
            result[parse(i[2])][2] = i[3]
        elif i[0] == 'B' and i[1] == 3:
            result[parse(i[2])][3] = i[3]
        elif i[0] == 'C' and i[1] == 1:
            result[parse(i[2])][4] = i[3]
        elif i[0] == 'C' and i[1] == 3:
            result[parse(i[2])][5] = i[3]

##(A, 2)，(A, 3), (B, 1), (B, 3), (C, 1), (C, 3)

fw = open(path + 'processed_travel_time.csv', 'w')
fw.writelines(','.join(['"time_window"', '"A2"', '"A3"', '"B1"', '"B3"', '"C1"', '"C3"']) + '\n')
time_windows = list(result.keys())
time_windows.sort()
for time_window_start in time_windows:
    out_line = ','.join(['"' + str(time_window_start) + '"','"' + str(result[time_window_start][0]) + '"','"' + str(result[time_window_start][1]) + '"','"' + str(result[time_window_start][2]) + '"','"' + str(result[time_window_start][3]) + '"','"' + str(result[time_window_start][4]) + '"','"' + str(result[time_window_start][5]) + '"',]) + '\n'
    fw.writelines(out_line)
fw.close()
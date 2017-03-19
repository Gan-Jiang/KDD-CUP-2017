import pandas as pd
#if neighbours have value, fill the mean.
train_path = 'dataSets/training/'
test_path = "dataSets/testing_phase1/"


df = pd.read_csv(test_path + 'test1_20min_avg_travel_time.csv', header = 0, error_bad_lines=False)
temp = pd.DatetimeIndex(df['time_window'])
df['date'] = temp.date
df['time'] = temp.time
df.head()

for index, row in df.iterrows():
    if row['avg_travel_time'] == 0:
        df.set_value(index, 'avg_travel_time', df[df['time'] == row['time']][df['intersection_id'] == row['intersection_id']][df['tollgate_id'] == row['tollgate_id']][df['avg_travel_time'] != 0]['avg_travel_time'].mean())

#a = df['avg_travel_time'] == 0
#a.sum()

df.drop(['date'], axis=1, inplace=True)
df.drop(['time'], axis=1, inplace=True)

df.to_csv(test_path + 'test1_20min_avg_travel_time_filled.csv', index = False)
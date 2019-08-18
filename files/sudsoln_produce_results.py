
import pandas as pd
import sudsoln as ss

path_to_result_sudsoln_csv = 'result_sudsoln' + ss.__version__ + '.csv'


# Produce a result

def to_sec(time):
    h, m, s = time[0], time[2:4], time[5:]
    return float(h) * 3600 + float(m) * 60 + float(s)
def categorize_solved(trial):
    if trial == 0:
        return 'logically'
    else:
        if trial == 200:
            return 'not_solved'
        else:
            return 'forecfully'

result_sudsoln = pd.read_csv(path_to_result_sudsoln_csv)
result_sudsoln = result_sudsoln.iloc[:, [0, 2, 3, 4]]
result_sudsoln.time = result_sudsoln.time.apply(to_sec)
result_sudsoln['min_time'] = result_sudsoln.time
result_sudsoln['median_time'] = result_sudsoln.time
result_sudsoln['avg_time'] = result_sudsoln.time
result_sudsoln['max_time'] = result_sudsoln.time
result_sudsoln.is_solved = result_sudsoln.is_solved.apply(int)
result_sudsoln['total'] = 1
result_sudsoln['solved'] = result_sudsoln.trial.apply(categorize_solved)

result_sudsoln_report1 = result_sudsoln\
    .groupby('category')\
    .agg({
        'is_solved': 'sum', 
        'total': 'sum',
        'min_time': 'min',
        'median_time': 'median',
        'avg_time': 'mean',
        'max_time': 'max'
    })\
    .sort_values('category', ascending = False)
print(result_sudsoln_report1)

result_sudsoln.is_solved = result_sudsoln.is_solved.apply(bool)
result_sudsoln_report2 = result_sudsoln\
    .groupby(['category', 'is_solved'])\
    .agg({
        'total': 'sum',
        'min_time': 'min',
        'median_time': 'median',
        'avg_time': 'mean',
        'max_time': 'max'
    })\
    .sort_values(['category', 'is_solved'], ascending = False)
print(result_sudsoln_report2)

result_sudsoln_report3 = result_sudsoln\
    .loc[lambda x: x['is_solved']]\
    .groupby('solved')\
    .agg({
        'total': 'sum',
        'min_time': 'min',
        'median_time': 'median',
        'avg_time': 'mean',
        'max_time': 'max'
    })\
    .sort_values('solved', ascending = False)
print(result_sudsoln_report3)



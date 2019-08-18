
import numpy as np
import pandas as pd
import random
import sudsoln as ss


# Create a file

random.seed(1024)
path_to_result_sudsoln_csv = 'result_sudsoln' + ss.__version__ + '.csv'

top95 = np.loadtxt(
    'https://norvig.com/top95.txt', 
    dtype = str, 
    delimiter = '\n'
)
len_top95 = len(top95)
hardest = np.loadtxt(
    'https://norvig.com/hardest.txt',
    dtype = str,
    delimiter = '\n'
)
concat = np.concatenate((top95, hardest))
len_concat = len(concat)

result_sudsoln = open(path_to_result_sudsoln_csv, 'w')
result_sudsoln.write('category,result,time,trial,is_solved\n')
for i in range(len_concat):
    question = ss.to_sudoku(
        concat[i], 
        elements = set([j for j in range(1, 10)])
    )
    print('Question number', i + 1, '\n')
    category = 'top95' if i <= len_top95 - 1 else 'hardest'
    time, trial = question.solve()
    result = "'" + str(question) + "'"
    ans = question.is_valid_answer()
    result_sudsoln.write(
        '{0},{1},{2},{3},{4}\n'.format(category, result, time, trial, ans)
    )
result_sudsoln.close()


# Produce a result

def to_sec(time):
    h, m, s = time[0], time[2:4], time[5:]
    return float(h) * 3600 + float(m) * 60 + float(s)

result_sudsoln = pd.read_csv(path_to_result_sudsoln_csv)
result_sudsoln = result_sudsoln.iloc[:, [0, 2, 3, 4]]
result_sudsoln.time = result_sudsoln.time.apply(to_sec)
result_sudsoln['min_time'] = result_sudsoln.time
result_sudsoln['median_time'] = result_sudsoln.time
result_sudsoln['avg_time'] = result_sudsoln.time
result_sudsoln['max_time'] = result_sudsoln.time
result_sudsoln.is_solved = result_sudsoln.is_solved.apply(int)
result_sudsoln['total'] = 1

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

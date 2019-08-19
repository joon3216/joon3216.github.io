
import pandas as pd
import sudsoln as ss

path_to_result_sudsoln_csv = 'result_sudsoln' + ss.__version__ + '.csv'
max_trial = 200

# Produce a result

def to_sec(time):
    h, m, s = time[0], time[2:4], time[5:]
    return float(h) * 3600 + float(m) * 60 + float(s)

def categorize_solved(trial):
    if trial == 0:
        return 'logically'
    else:
        if trial == max_trial:
            return 'not_solved'
        else:
            return 'forcefully'

result_sudsoln = pd.read_csv(path_to_result_sudsoln_csv)
result_sudsoln = result_sudsoln.iloc[:, [0, 2, 3, 4]]
result_sudsoln.time = result_sudsoln.time.apply(to_sec)

result_sudsoln_check = result_sudsoln\
    .loc[lambda x: x['is_solved'] == False]
if set(result_sudsoln_check.trial) == {max_trial}:
    msg = '         Yes, is_solved = False iff trial == ' +\
        str(max_trial) + ':'
else:
    msg = '         No, there is at least one case where' +\
        ' is_solved == True and trial == ' + str(max_trial) + ':'
print(
    'Table 0: Check that is_solved = False iff trial == 200 ',
    'before analysis.\n' + msg + '\n\n', 
    result_sudsoln_check, '\n\n',
    sep = ''
)

result_sudsoln['min_time'] = result_sudsoln.time
result_sudsoln['median_time'] = result_sudsoln.time
result_sudsoln['avg_time'] = result_sudsoln.time
result_sudsoln['max_time'] = result_sudsoln.time
result_sudsoln.is_solved = result_sudsoln.is_solved.apply(int)
result_sudsoln['total'] = 1
result_sudsoln['solved'] = result_sudsoln.trial.apply(categorize_solved)

result_sudsoln_cp = result_sudsoln.copy()
result_sudsoln_cp.category =\
    result_sudsoln_cp.category.apply(lambda x: 'all')
result_sudsoln_dbl = result_sudsoln.append(result_sudsoln_cp)
result_sudsoln_report1 = result_sudsoln_dbl\
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
print(
    'Table 1: Within each category, how many puzzles were solved?\n' +\
    '         How long did Sudoku.solve() run on each puzzle?\n\n', 
    result_sudsoln_report1, '\n\n',
    sep = ''
)

result_sudsoln_dbl.is_solved = result_sudsoln_dbl.is_solved.apply(bool)
result_sudsoln_report2 = result_sudsoln_dbl\
    .groupby(['category', 'is_solved'])\
    .agg({
        'total': 'sum',
        'min_time': 'min',
        'median_time': 'median',
        'avg_time': 'mean',
        'max_time': 'max'
    })\
    .sort_values(['category', 'is_solved'], ascending = False)
print(
    'Table 2: Within each category, if a puzzle was solved,\n' +\
    '         how long did it take to solve one?\n\n', 
    result_sudsoln_report2, '\n\n',
    sep = ''
)

result_sudsoln_report3 = result_sudsoln\
    .groupby('solved')\
    .agg({
        'total': 'sum',
        'min_time': 'min',
        'median_time': 'median',
        'avg_time': 'mean',
        'max_time': 'max'
    })\
    .reindex(['logically', 'forcefully', 'not_solved'])
print(
    'Table 3: How many puzzles required a brute force to be solved?\n\n',
    result_sudsoln_report3, '\n\n',
    sep = ''
)

result_sudsoln['min_trial'] = result_sudsoln.trial
result_sudsoln['median_trial'] = result_sudsoln.trial
result_sudsoln['avg_trial'] = result_sudsoln.trial
result_sudsoln['max_trial'] = result_sudsoln.trial
result_sudsoln_report4 = result_sudsoln\
    .loc[lambda x: x['solved'] == 'forcefully']\
    .groupby('solved')\
    .agg({
        'total': 'sum',
        'min_trial': 'min',
        'median_trial': 'median',
        'avg_trial': 'mean',
        'max_trial': 'max'
    })
print(
    'Table 4: If a brute force is used and successfully solved a puzzle,\n' +\
    '         how many attempts did it take to solve one?\n\n',
    result_sudsoln_report4, '\n\n',
    sep = ''
)
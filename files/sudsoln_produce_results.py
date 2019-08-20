
import pandas as pd
import sudsoln as ss

path_to_result_sudsoln_csv = 'result_sudsoln' + ss.__version__ + '.csv'
max_trial = 200


# Produce results

def categorize_solved(trial):
    if trial == 0:
        return 'logically'
    else:
        if trial == max_trial:
            return 'not_solved'
        else:
            return 'forcefully'

def to_sec(time):
    h, m, s = time[0], time[2:4], time[5:]
    return float(h) * 3600 + float(m) * 60 + float(s)

def tprint(message, table):
    print(message, '\n\n', table, '\n\n', sep = '')


result_sudsoln = pd.read_csv(path_to_result_sudsoln_csv)
result_sudsoln = result_sudsoln.iloc[:, [0, 2, 3, 4]]
result_sudsoln.time = result_sudsoln.time.apply(to_sec)
result_sudsoln['total'] = 1
result_sudsoln['solved'] = result_sudsoln.trial.apply(categorize_solved)


result_sudsoln_check = result_sudsoln\
    .loc[lambda x: x['is_solved'] == False]
result_sudsoln_check2 = result_sudsoln\
    .loc[lambda x: x['trial'] == 200]
if result_sudsoln_check.equals(result_sudsoln_check2):
    msg = '         Yes, is_solved == False iff trial == ' +\
        str(max_trial) + ':'
else:
    msg = '         No, there is at least one case where' +\
        ' is_solved == True and trial == ' + str(max_trial) + ':'
table0 =\
    'Table 0: Check that is_solved == False iff trial == 200 ' +\
    'before analysis.\n'
tprint(table0 + msg, result_sudsoln_check)


result_sudsoln_cp = result_sudsoln.copy()
result_sudsoln_cp.category =\
    result_sudsoln_cp.category.apply(lambda x: 'all')
result_sudsoln_dbl = result_sudsoln.append(result_sudsoln_cp)
result_sudsoln_dbl.is_solved = result_sudsoln_dbl.is_solved.apply(int)
result_sudsoln_report1 = result_sudsoln_dbl\
    .groupby('category')\
    .agg(
        is_solved = ('is_solved', 'sum'),
        total = ('total', 'sum'),
        min_time = ('time', 'min'),
        median_time = ('time', 'median'),
        avg_time = ('time', 'mean'),
        max_time = ('time', 'max')
    )\
    .sort_values('category', ascending = False)
table1 =\
    'Table 1: Within each category, how many puzzles were solved?\n' +\
    '         How long did Sudoku.solve() run most of the time?'
tprint(table1, result_sudsoln_report1)


result_sudsoln_dbl.is_solved = result_sudsoln_dbl.is_solved.apply(bool)
result_sudsoln_report2 = result_sudsoln_dbl\
    .groupby(['category', 'is_solved'])\
    .agg(        
        total = ('total', 'sum'),
        min_time = ('time', 'min'),
        median_time = ('time', 'median'),
        avg_time = ('time', 'mean'),
        max_time = ('time', 'max')
    )\
    .sort_values(['category', 'is_solved'], ascending = False)
table2 =\
    'Table 2: Within each category, if a puzzle was solved,\n' +\
    '         how long did it take to solve one?'
tprint(table2, result_sudsoln_report2)


result_sudsoln_report3 = result_sudsoln\
    .groupby('solved')\
    .agg(
        total = ('total', 'sum'),
        min_time = ('time', 'min'),
        median_time = ('time', 'median'),
        avg_time = ('time', 'mean'),
        max_time = ('time', 'max')
    )\
    .reindex(['logically', 'forcefully', 'not_solved'])
table3 =\
    'Table 3: How many puzzles required a brute force to be solved?'
tprint(table3, result_sudsoln_report3)


logi_max = result_sudsoln_report3.iloc[0, 4]
result_sudsoln_report3_1 = result_sudsoln\
    .loc[lambda x: (x.trial > 0) & (x.time <= logi_max)]\
    .loc[:, ['category', 'time', 'trial', 'is_solved']]
table3_1 =\
    'Table 3.1: Which forcefully solved puzzles were solved faster than' +\
    '\n           the maximum time consumed by one of ' +\
    'logically solved puzzles?'
tprint(table3_1, result_sudsoln_report3_1)


result_sudsoln_report3_2 = result_sudsoln\
    .loc[lambda x: (x.trial > 0) & (x.time > logi_max) & (x.trial <= 2)]\
    .loc[:, ['category', 'time', 'trial', 'is_solved']]
table3_2 =\
    'Table 3.2: How many puzzles are forcefully solved and yet took\n' +\
    '           less than or equal to two attempts?\n'
tprint(table3_2, result_sudsoln_report3_2)


result_sudsoln_report4 = result_sudsoln\
    .loc[lambda x: x['solved'] == 'forcefully']\
    .groupby('solved')\
    .agg(
        total = ('total', 'sum'),
        min_trial = ('trial', 'min'),
        median_trial = ('trial', 'median'),
        avg_trial = ('trial', 'mean'),
        max_trial = ('trial', 'max')
    )
table4 =\
    'Table 4: If a brute force is used and successfully solved ' +\
    'a puzzle,\n' +\
    '         how many attempts did it take to solve one?'
tprint(table4, result_sudsoln_report4)


import numpy as np
import random
import sudsoln as ss

path_to_result_sudsoln_csv = 'result_sudsoln' + ss.__version__ + '.csv'


# Create a file

random.seed(1024)

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


import numpy as np
import sudsoln as ss
import random

random.seed(1024)

top95 = np.loadtxt(
    'https://norvig.com/top95.txt', 
    dtype = str, 
    delimiter = '\n'
)
hardest = np.loadtxt(
    'https://norvig.com/hardest.txt',
    dtype = str,
    delimiter = '\n'
)
concat = np.concatenate((top95, hardest))

result_sudsoln = open('result_sudsoln.csv', 'w')
result_sudsoln.write('result,time,trial,has_solved\n')
for i in range(len(concat)):
    question = ss.to_sudoku(
        concat[i], 
        elements = set([j for j in range(1, 10)])
    )
    print('Question number', i + 1, '\n')
    time, trial = question.solve()
    result = "'" + str(question) + "'"
    has_solved = question.is_valid_answer()
    result_sudsoln.write(
        '{0},{1},{2},{3}\n'.format(result, time, trial, has_solved)
    )
result_sudsoln.close()

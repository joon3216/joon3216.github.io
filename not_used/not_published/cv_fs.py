
from datetime import datetime as dt
from matplotlib import pyplot as plt
import numpy as np


# Examples
np.random.seed(1024)
n = 1000
t = np.arange(n) + 1
f = lambda t: t / 250 - .5
g = lambda t: -(.25 / 449) * t + 250 / 449
true_theta = np.concatenate((
    np.zeros(249), f(np.arange(250, 501)), 
    np.zeros(50) + .75, g(np.arange(551, 1001))
))
y = true_theta + np.random.normal(scale = .1, size = 1000)


# split algorithm
def split(x, partition):
    unique_partitions = list(set(partition))
    result = {}
    for item in unique_partitions:
        result[unique_partitions[item]] = []
    for i in range(len(x)):
        result[partition[i]].append(x[i])
    return result


def split_data_1d(y, num_folds, fold):
    ''' (np.array, int, int) -> {str: {str: np.array and str: list}}
    
    Precondition: 0 <= fold <= num_folds - 1 <= len(y) - 1
    
    Return a dictionary with two keys: 'y_fold' has a dictionary value
    with two keys 'indices' and 'data', where 'indices' contains indices
    of the validation data selected by 'num_folds' and 'fold' argument, 
    and 'data' the validation data itself selected from 'y'; 
    'y_rest' has the same structure, except it contains the remaining 
    indices and data.
    
    >>> test1 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
    >>> test1 = np.array(test1)
    >>> result1 = split_data_1d(test1, 3, 2)
    >>> np.array_equal(
    ...     result1['y_fold']['data'], 
    ...     np.array(['c', 'f', 'i'], dtype='<U1')
    ... )
    True
    >>> np.array_equal(
    ...     result1['y_fold']['indices'],
    ...     np.array([2, 5, 8])
    ... )
    True
    >>> np.array_equal(
    ...     result1['y_rest']['data'],
    ...     np.array(['a', 'b', 'd', 'e', 'g', 'h', 'j', 'k'], dtype='<U1')
    ... )
    True
    >>> np.array_equal(
    ...     result1['y_rest']['indices'],
    ...     np.array([ 0,  1,  3,  4,  6,  7,  9, 10])
    ... )
    True
    '''

    n = len(y)
    indices = np.arange(0, n)
    assign_folds = np.tile(np.arange(0, num_folds), n // num_folds + 1)[:n]
    partition = split(indices, assign_folds)
    fold_indices = partition[fold]
    rest_indices = list(set(indices).difference(fold_indices))
    y_fold = y[fold_indices]
    y_rest = y[rest_indices]
    result = {
        'y_fold': {
            'data': y_fold,
            'indices': np.array(fold_indices)
        },
        'y_rest': {
            'data': y_rest,
            'indices': np.array(rest_indices)
        }
    }
    return result


def fusion_estimates(y, lambd, theta = None, max_iter = 1000, eps = 1e-05):
    '''(np.array, number[, np.array, int, number]) -> 
        {str: np.array or number}
    
    Preconditions:
    1. len(y) == len(theta) if theta specified.
    2. lambd > 0 and eps > 0
    3. max_iter > 1

    Return the dictionary that stores: 
    - 'theta', the fusion estimates of y iterated from theta with the
      maximum iteration max_iter and the cost difference threshold eps.
    - 'phi', the differences of each 'theta'
    - 'lambd', the lambd specified
    - 'iteration', the number of iterations, and
    - 'costs', the cost function evaluated at each iteration where the
      first cost is calculated at iteration 0.
    '''
    
    n = len(y)
    if theta is None:
        theta = y.copy()
    phi = np.diff(theta)
    phisums_old = np.cumsum(phi)
    theta_1_new = (sum(y) - sum(phisums_old)) / n
    cost = sum((y - theta) ** 2) + lambd * sum(abs(phi))
    costs = []
    costs.append(cost)
    there_is_a_progress = True
    iteration = 0
    while there_is_a_progress and iteration < max_iter:
        phi_new = np.zeros(n)
        for j in range(1, n):
            phisums_new = np.cumsum(phi_new)
            req = sum(
                phisums_old[(j - 1):(n - 1)] -\
                phisums_old[j - 1] + phisums_new[j - 1]
            )
            discri = sum(y[j:n]) - (n - (j + 1) + 1) * theta_1_new - req
            if discri < -lambd / 2:
                phi_new[j] = (discri + lambd / 2) / (n - (j + 1) + 1)
            elif discri > lambd / 2:
                phi_new[j] = (discri - lambd / 2) / (n - (j + 1) + 1)
        phi_new = phi_new[1:]
        phisums_new = phisums_new[1:]
        theta = np.append(theta_1_new, theta_1_new + phisums_new)
        cost = sum((y - theta) ** 2) + lambd * sum(abs(phi_new))
        theta_1_new = (sum(y) - sum(phisums_new)) / n
        phisums_old = phisums_new
        iteration += 1
        costs.append(cost)
        there_is_a_progress = not (abs(costs[iteration - 1] - cost) <= eps)
        
    result = {
        'theta': theta,
        'phi': phi_new,
        'lambd': lambd,
        'iteration': iteration,
        'costs': np.array(costs)
    }
    
    return result



if __name__ == '__main__':
    import doctest
    doctest.testmod()

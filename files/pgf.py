
from datetime import datetime as dt
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import fmin



npmap = lambda func, *iterable: np.array(list(map(func, *iterable)))


def dX(x):
    '''(number or *iterable) -> number or np.array
    
    Return the probability assigned for x, defined by the following pmf:
    p(0) = .05, p(1) = p(3) = p(4) = .1, p(2) = .075, p(5) = .575, 
    and p(x) = 0 otherwise.
    
    >>> dX(0)
    0.05
    >>> dX([2, 5])
    array([0.075, 0.575])
    >>> dX(np.array([-1, 0, 4]))
    array([0.  , 0.05, 0.1 ])
    '''
    
    def pX(d):
        if d == 0:
            return .05
        elif d in [1, 3, 4]:
            return .1
        elif d == 2:
            return .075
        elif d == 5:
            return .575
        else:
            return 0
    if not hasattr(x, '__iter__'):
        return pX(x)
    return npmap(pX, x)


def dY(y):
    def pY(d):
        if d in [1, 4]:
            return .25
        elif d == 2:
            return .5
        else:
            return 0
    if not hasattr(y, '__iter__'):
        return pY(y)
    return npmap(pY, y)


def csum_N(pmf, support, lambd, eps = 1e-05):
    '''(function, np.array, number[, float]) -> np.array
        
    Preconditions:
    1. pmf is a pmf of X_i where the random summation S = X_1 + ... + X_N 
       with N ~ Pois(lambd) has 0, 1, ..., M - 1 as the first M element of 
       its support.
    2. pmf is a function whose output is np.array whenever the input is
       np.array.
    3. support == np.arange(0, l + 1), where l is the largest number of
       the support of pmf.
    4. lambd > 0
    5. 0 < eps < 1
        
    Return the approximate probability mass function of S, i.e. 
    P(S = x | S < M) for some appropriate integer M determined by 
    P(S >= M) < eps, where S is the sum of iid X_i's with 
    i = 1, ..., N ~ Pois(lambd), X_i ~ pmf, and X_i's support is
    a subset of np.arange(0, l + 1) (= support) with l being the largest 
    element of X_i's support.
        
    >>> def dY(y):
    ...     def pY(d):
    ...         if d in [1, 4]:
    ...             return .25
    ...         elif d == 2:
    ...             return .5
    ...         else:
    ...             return 0
    ...     if not hasattr(y, '__iter__'):
    ...         return pY(y)
    ...     return npmap(pY, y)
    ...
    >>> result_Y = csum_N(dY, np.arange(0, 5), 3)
    >>> M_Y = len(result_Y)
    >>> print(M_Y, sum(result_Y))
    39 0.9999999999999998
    >>> result_Y[0:4]
    array([0.04978729, 0.03734044, 0.08868328, 0.05951115])
    '''
        
    pmf_vec = pmf(support)
        
    # Define the pgf of X_i
    g = lambda t: npmap(lambda d: sum(d ** support * pmf_vec), t)
        
    # Find M
    Ms = lambda t: (-lambd * (1 - g(t)) - np.log(eps)) / np.log(t)
    M = np.ceil(fmin(Ms, 1.001, full_output = True, disp = False)[1])
    
    # Append 0's
    pmf_vec = np.append(pmf_vec, np.zeros(int(M - len(pmf_vec))))
        
    # Apply DFT and inverse DFT
    gtks = np.fft.fft(pmf_vec)
    gS_gtks = np.exp(-lambd * (1 - gtks))
    pS_tks = np.fft.ifft(gS_gtks).real
        
    return pS_tks


def dpmf(x, pmf_vec, support_vec = None):
    '''(object or *iterable, *iterable[, *iterable]) -> number or np.array
    
    Preconditions:
    1. Elements of x are of the same type as elements of support_vec,
       if support_vec is specified. If support_vec is not specified, then
       x must be a number or an iterable object with numeric elements.
    2. sum(pmf_vec) == 1
    3. len(pmf_vec) == len(support_vec) if support_vec is specified.
    4. If support_vec is specified, then each element of support_vec 
       must be hashable, i.e. element.__hash__ is not None
    
    Return the probability evaluated at each element of x based on
    probabilities in pmf_vec and elements of support_vec if support_vec 
    is specified (each element of support_vec is the input that corresponds
    to the probability in pmf_vec). If not specified, then support_vec will 
    be replaced with np.arange(0, len(pmf_vec)).
    
    >>> # Example 1
    >>> pmf_eg1 = [0.25, 0.5 , 0.25]
    >>> support_eg1 = np.array([1, 2, 4])
    >>> dpmf(1, pmf_eg1, support_eg1)
    0.25
    >>> dpmf([3, 4, 6], pmf_eg1, support_eg1)
    array([0.  , 0.25, 0.  ])
    >>> dpmf(np.array([3, 4, 6]), pmf_eg1, support_eg1)
    array([0.  , 0.25, 0.  ])
    >>>
    >>> # Example 2
    >>> pmf_eg2 = (.25, .4, .35)
    >>> support_eg2 = ['apple', 'orange', 'neither']
    >>> dfruit = lambda x: dpmf(x, pmf_eg2, support_eg2)
    >>> dfruit(['apple', 'neither'])
    array([0.25, 0.35])
    >>> dfruit('orange')
    0.4
    >>> dfruit(np.array(['orange', 'hello']))
    array([0.4, 0. ])
    '''
    
    M = len(pmf_vec)
    if support_vec is None:
        support_vec = np.arange(0, M)
    D = {}
    for i in range(len(support_vec)):
        D[support_vec[i]] = pmf_vec[i]
    finder = lambda d: D[d] if d in D.keys() else 0
    if hasattr(x, '__iter__'):
        if type(x) == str:
            return finder(x)
        return npmap(finder, x)
    return finder(x)


def rpmf(n, pmf, support, **kwargs):
    '''(int, function, *iterable[, **kwargs]) -> np.array
    
    Precondition: 
    1. n >= 1
    2. support is the support of pmf.
    
    Return n random samples from the specified pmf with support 'support'
    and additional arguments of pmf in **kwargs if required. Since this
    function uses **kwargs, any additional arguments of pmf you want to
    specify must be named.
    
    >>> # Example 1: dX
    >>> np.random.seed(1024)
    >>> rpmf(n = 20, pmf = dX, support = np.arange(0, 6))
    array([5, 5, 5, 5, 5, 5, 1, 0, 1, 5, 5, 5, 5, 3, 5, 5, 5, 2, 5, 1])
    >>>
    >>> # Example 2: S_Y = Y_1 + ... + Y_N
    >>> np.random.seed(1024)
    >>> result_S_Y = csum_N(dY, np.arange(0, 5), 3) # in csum_N example
    >>> result_S_Y = result_S_Y / sum(result_S_Y)
    >>> M_S_Y = len(result_S_Y)
    >>> rpmf(10, dpmf, np.arange(0, M_S_Y), pmf_vec = result_S_Y)
    array([ 8, 22,  6,  8,  7,  9,  2,  0,  2,  9])
    >>>
    >>> # Example 3: dfruit in dpmf example
    >>> np.random.seed(2048)
    >>> rpmf(7, dfruit, ['apple', 'orange', 'neither'])
    array(['orange', 'apple', 'neither', 'neither', 'neither', 'orange',
           'apple'], dtype='<U7')
    '''
    
    cmf_vec = np.append(0, np.cumsum(pmf(support, **kwargs)))
    unif_01 = np.random.random(n)
    result = []
    for k in range(n):
        for j in range(len(cmf_vec) - 1):
            if unif_01[k] >= cmf_vec[j] and unif_01[k] < cmf_vec[j + 1]:
                result.append(support[j])
    return np.array(result)



if __name__ == '__main__':
    result = csum_N(dX, np.arange(0, 6), 3)
    M = len(result)
    pmf_eg2 = (.25, .4, .35)
    support_eg2 = ['apple', 'orange', 'neither']
    dfruit = lambda x: dpmf(x, pmf_eg2, support_eg2)
    # np.random.seed(2048)
    # start_py = dt.now()
    # samples_N = rpmf(10000, dpmf, np.arange(0, M), pmf_vec = result)
    # end_py = dt.now()    
    
    
    # Scatterplot: pmf of S
    #print(M, sum(result))
    #plt.scatter(np.arange(0, M), result)
    #plt.title('pmf of S')
    #plt.xlabel('x')
    #plt.ylabel('P(S = x)')
    #plt.show()    
    
    # Histogram: samples of S
    #plt.clf()
    #bins = np.arange(-.5, M + 1.5, step = 1)
    #plt.hist(samples_N, bins = bins, density = True)
    #plt.title('Histogram of S')
    #plt.ylabel('Mass')
    #plt.xlabel('x')
    #plt.show()
    
    # Histogram and the actual pmf of S
    #plt.clf()
    #plt.scatter(np.arange(0, M), result, s = 5, c = 'black')
    #plt.hist(samples_N, bins = bins, density = True, alpha = .5)
    #plt.title('Histogram vs. the actual pmf of S')
    #plt.ylabel('Mass')
    #plt.xlabel('x')
    #plt.show()    
    
    import doctest
    doctest.testmod()

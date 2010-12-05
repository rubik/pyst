from __future__ import division
import math
import doctest
import operator

def _root(n, k):
    return n ** (1 / k)

def _sorted(func):
    def wrapper(data, *args, **kwargs):
        return func(sorted(data), *args, **kwargs)
    return wrapper

def mean(data):
    '''
    Returns the arithmetic mean of data::

        >>> mean([1, -4, 32, 5, 3, 1]) # doctest: +ELLIPSIS
        6.3333...
    '''

    return sum(data) / len(data)

def running_average(data):
	for i in xrange(1, len(data) + 1):
		m = data[:i]
		yield sum(m) / len(m)

def md(data): ## mean difference
    n = len(data)
    return sum(abs(data[j] - data[i]) for i in xrange(n) for j in xrange(n)) / (n * (n - 1))

def rmd(data):
    return md(data) / mean(data)

def afreq(data):
    return [data.count(d) for d in set(data)]

def rfreq(data):
    n = len(data)
    return map(lambda i: i / n, afreq(data))

def gini(data):
    '''
    Returns the Gini coefficients, a number between 0 and 1.
    It is 0 when all the data are equal, and it is 1 when all the data are different::

        >>> d = [1, 2, 3, 4]
        >>> gini(d)
        0.75
        >>> d = [1, 1, 2, 2, 3, 3]
        >>> gini(d) # doctest: +ELLIPSIS
        0.666666...
        >>> d = [1, 1, 2, 2]
        >>> gini(d)
        0.5
        >>> d = [1, 1, 1, 1]
        >>> gini(d)
        0.0
    '''

    return 1 - sum(map(lambda i: i ** 2, rfreq(data)))

def shannon(data):
    return - sum(map(lambda i: i * math.log(i), data))

def mode(data):
    return sorted((data.count(d), d) for d in set(data))[-1][1]

def geo_mean(data):
    '''
    Returns the geometric mean of *data*
    '''

    return _root(reduce(operator.mul, data), len(data))

def quadratic(data):
	return math.sqrt(mean([d ** 2 for d in data]))

def weighted_mean(data, weights=None):
    dt = list(set(data))
    if weights is None:
        weights = map(lambda i: data.count(i), dt)
    return sum(map(lambda i: i[0] * i[1], zip(dt, weights))) / sum(weights)

def harmonic_mean(data):
    return len(data) / sum(1 / d for d in data)

def c_moment(data, k):
    m = mean(data)
    return sum((d - m) ** k for d in data) / len(data)

def s_moment(data, k):
    return sum(d ** k for d in data)

def range(data):
    return max(data) - min(data)

def midrange(data):
	return (max(data) + min(data)) / 2

@ _sorted
def median(data):
    n = len(data)
    if n & 1:
        return data[(n + 1) // 2 - 1]
    return (data[n // 2 - 1] + data[n // 2]) / 2

@ _sorted
def quantile(data, p, m=0):
    def _helper(h):
        return data[int(math.floor(h) - 1)] + (h - math.floor(h)) * (data[int(math.floor(h))] - data[int(math.floor(h) - 1)])
    def r1():
        if p == 0:
            return data[0]
        h = n * p + .5
        i = int(math.ceil(h - .5) - 1)
        if i < 0:
            return data[0]
        return data[i]
    def r2():
        if p == 0:
            return data[0]
        if p == 1:
            return data[-1]
        h = n * p + .5
        i, j = int(math.ceil(h - .5) - 1), int(math.floor(h + .5) - 1)
        if i < 0 or j < 0:
            return data[0]
        return (data[i] + data[j]) / 2
    def r3(): ##! does not give h = (N + 1) / 2 when p = 1/2
        if p < .5 / n:
            return data[0]
        h = n * p
        return data[int(max(1, round(h)) - 1)]
    def r4(): ##! does not give h = (N + 1) / 2 when p = 1/2
        if p < 1 / n:
            return data[0]
        if p == 1:
            return data[-1]
        h = n * p
        return _helper(h)
    def r5():
        if p < .5 / n:
            return data[0]
        if p >= (n - .5) / n:
            return data[-1]
        h = n * p + .5
        return _helper(h)
    def r6():
        if p < 1 / (n + 1):
            return data[0]
        if p >= n / (n + 1):
            return data[-1]
        h = (n + 1) * p
        return _helper(h)
    def r7():
        if p == 1:
            return data[-1]
        h = (n - 1) * p + 1
        return _helper(h)
    def r8():
        if p < (2 / 3) / (n + 1/3):
            return data[0]
        if p >= (n - 1 / 3) / (n + 1 / 3):
            return data[-1]
        h = (n + 1 / 3) * p + 1 / 3
        return _helper(h)
    def r9():
        if p < (5 / 8) / (n + 1 / 4):
            return data[0]
        if p >= (n - 3 / 8) / (n + 1 / 4):
            return data[-1]
        h = (n + 1 / 4) * p + 3 / 8
        return _helper(h)
    def unknown():
        if p < (3 / 2) / (n + 2):
            return data[0]
        if p >= (n + 1 / 2) / (n + 2):
            return data[-1]
        h = (n + 2) * p - .5
        return _helper(h)

    n = len(data)
    methods = [r1, r2, r3, r4, r5, r6, r7, r8, r9, unknown]
    try:
        return methods[m]()
    except IndexError:
        return methods[m]()

@ _sorted
def quartiles(data, m=0):
    n = len(data)
    methods = [[(n * (1 / 4), n * (3 / 4)), (n * (1 / 4), n * (3 / 4))], ## Standard method
               [((n + 1) / 4, (3*n + 3) / 4), ((n + 1) / 4, (3*n + 3) / 4)], ## Minitab's method
               [((n + 2) / 4, (3*n + 2) / 4), ((n + 3) / 4, (3*n + 1) / 4)], ## Tukey's method
               [((n + 2) / 4, (3*n + 2) / 4), ((n + 1) / 4, (3*n + 3) / 4)], ## Moore and McCabe
               [((n + 1) / 4, (3*n + 3) / 4), ((n + 1) / 4, (3*n + 3) / 4)], ## Mendenhall and Sincich
               [((n + 3) / 4, (3*n + 1) / 4), ((n + 3) / 4, (3*n + 1) / 4)]] ## Freund and Perles
    q1, q3 = map(math.ceil, methods[m][n & 1])
    return (data[int(q1 - 1)], median(data), data[int(q3 - 1)])

def hinges(data):
    return quartiles(data, 2)

def midhinge(data):
	h1, _, h3 = hinges(data)
	return (h1 + h3) / 2

def trimean(data):
    q1, q2, q3 = hinges(data)
    return (q1 + 2*q2 + q3) / 4

def decile(data, d, m=0):
    if not 0 <= d <= 10:
        raise ValueError('d must be between 0 and 10')
    return quantiles(data, d / 10, m)

def percentile(data, p, m=0):
    if not 0 <= p <= 100:
        raise ValueError('p must be between 0 and 100')
    return quantiles(data, p / 100, m)

def iqr(data, m=0):
    q = quartiles(data, m)
    return q[2] - q[0]

def kurtosis(data): ## kurtosis coeff, kurtosis index
    b = c_moment(data, 4) / (c_moment(data, 2) ** 2)
    return (b - 3, b)

def adev(data, m=median): ## absolute deviation
    try:
        c = m(data)
    except TypeError:
        c = m
    return [abs(d - c) for d in data]

def adev1(data, m=median, e=1):
    try:
        c = m(data)
    except TypeError:
        c = m
    return _root(sum(abs(d - c) ** e for d in data), e)

def md(data): ## mean absolute deviation
    return mean(adev(data, mean))

def mad(data): ## median absolute deviation
    return median(adev(data))

def stdev(data):
    '''
    Returns the standard deviation of a sample of *data*::

        >>> d = [1, 2, 3, 3, 5, 5, 5, 8]
        >>> stdev(d) # doctest: +ELLIPSIS
        2.20389...
    '''

    m = mean(data)
    n = len(data)
    return math.sqrt(sum((d - m) ** 2 / (n - 1) for d in data))

def pstdev(data):
    '''
    Returns the standard deviation of a population of *data*::

        >>> d = [1, 2, 3, 3, 5, 5, 5, 8]
        >>> pstdev(d) # doctest: +ELLIPSIS
        2.06155...
    '''
    m = mean(data)
    return math.sqrt(sum(((d - m) ** 2 for d in data)) / len(data))

def _var(data): ## two-pass algorithm
    m = mean(data)
    return sum((d - m) ** 2 for d in data), len(data)

def variance(data): ## sample variance
    v, n = _var(data)
    return v / n

def pvariance(data): ## population variance
    v, n = _var(data)
    return v / (n - 1)

def covariance(data1, data2):
    l = len(data1)
    m, n = mean(data1), mean(data2)
    return sum((data1[i] - m) * (data2[i] -  n) for i in xrange(l))

def pcv(data): ## coefficient of variation of a population
    return pstdev(data) / mean(data)

def cv(data): ## coefficient of variation of a sample
    return stdev(data) / mean(data)

def id(data): ## Index of dispersion
    return variance(data) / mean(data)

def skewness(data):
    return c_moment(data, 3) / (c_moment(data, 2) ** (3 / 2))


if __name__ == '__main__':
    doctest.testmod()

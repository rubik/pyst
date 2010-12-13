#!/usr/bin/env python

from __future__ import division
import math
import operator
import collections

class StatsError(ValueError):
    pass

def _root(n, k):
    return n ** (1 / k)

def _sorted(func):
    def wrapper(data, *args, **kwargs):
        return func(sorted(data), *args, **kwargs)
    return wrapper

def _two_pass(data): ## two-pass algorithm
    m = mean(data)
    return sum((d - m) ** 2 for d in data)

def _sp(data1, data2):
    mx, my = mean(data1), mean(data2)
    return sum((x - mx) * (y - my) for x, y in zip(data1, data2))

def sum(data):
    return math.fsum(data)

## Averages

def mean(data):
    '''
    Returns the arithmetic mean of data::

        >>> mean([1, -4, 32, 5, 3, 1]) # doctest: +ELLIPSIS
        6.3333...
    '''

    return sum(data) / len(data)

def weighted_mean(data, weights=None):
    dt = list(set(data))
    if weights is None:
        weights = map(lambda i: data.count(i), dt)
    if len(data) != len(weights) or (len(data), len(weights)) == (0, 0):
        raise StatsError('data and weights must have the same length and cannot be empty')
    return sum(map(lambda i: i[0] * i[1], zip(dt, weights))) / sum(weights)

def geo_mean(data):
    '''
    Returns the geometric mean of *data*
    '''

    return _root(reduce(operator.mul, data), len(data))

def quadratic(data):
    return math.sqrt(mean([d ** 2 for d in data]))

def harmonic_mean(data):
    return len(data) / sum(1 / d for d in data)

def running_average(data, m=mean):
    for i in xrange(1, len(data) + 1):
        s = data[:i]
        yield m(s)

## Measures of central tendancy

def mode(data):
    data = sorted((data.count(d), d) for d in set(data))
    n = len(data)
    if n == 0:
        raise StatsError('no mode defined for empty data sets')
    if n > 1 and data[-1][0] == data[-2][0]:
        raise StatsError('no distinct mode')
    return data[-1][1]

@ _sorted
def median(data):
    n = len(data)
    if n == 0:
        raise StatsError('no median defined for empty data sets')
    if n & 1:
        return data[(n + 1) // 2 - 1]
    return (data[n // 2 - 1] + data[n // 2]) / 2

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
    Returns the Gini coefficient, a number between ``0`` and ``(n - 1) / n``.
    It is ``0`` when all the data are equal, and it is ``(n - 1) / n`` when all the data are different::

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

def gini1(data):
    '''
    Returns the normalized Gini coefficient, a number between 0 and 1. It is 0
    when all the data are equal and 1 when all the data are different::

        >>> d = [1, 2, 3, 4]
        >>> gini1(d)
        1.0
        >>> d = [1, 2, 3, 4, 5, 1, 3, 4, 6, 7, 2]
        >>> gini1(d)
        0.9272727272727271
        >>> d = [1]
        >>> gini1(d)
        0.0
        >>> gini([])
        0.0
    '''

    n = len(data)
    if n in (0, 1):
        return 0.
    return gini(data) * n / (n - 1)

def shannon(data):
    '''
    Returns the Shannon index of *data*, a number between ``0`` and ``ln n``.
    It is ``0`` when all the data are equal

        
    '''

    n = len(data)
    return - sum(r * math.log(r) for r in rfreq(data))

def shannon1(data):
    if len(data) in (0, 1):
        return 0.
    return shannon(data) / math.log(len(data))

def c_moment(data, k):
    m = mean(data)
    return sum((d - m) ** k for d in data) / len(data)

def s_moment(data, k):
    return sum(d ** k for d in data)

def range(data):
    if len(data) == 0:
        raise StatsError('no range defined for empty data sets')
    return abs(max(data) - min(data))

def midrange(data):
    if len(data) == 0:
        raise StatsError('no midrange defined for empty data sets')
    return (max(data) + min(data)) / 2

@ _sorted
def quantile(data, p, m=0):
    '''
    Returns the *p*-th quantile of *data*.

    :param integer m: the method to use, it can be a number from 0 to 9:

        m
        ==

    '''

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
    if n == 0:
        raise StatsError('no quartiles defined for empty data sets')
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
        raise StatsError('d must be between 0 and 10')
    return quantiles(data, d / 10, m)

def percentile(data, p, m=0):
    if not 0 <= p <= 100:
        raise StatsError('p must be between 0 and 100')
    return quantiles(data, p / 100, m)

def iqr(data, m=0):
    q1, _, q3 = quartiles(data, m)
    return abs(q3 - q1)

def kurtosis(data): ## kurtosis coeff, kurtosis index
    b = c_moment(data, 4) / (c_moment(data, 2) ** 2)
    return b - 3

def adev(data, m=median): ## absolute deviation
    '''
    Returns the absolute deviation from the point *m*.

    :param list data: the data
    :param m: can be either a callable object or a number. If it is a callable the absolute deviation will be computed from ``m(data)`` (it must be a number)
    :type m: number or callable
    :rtype: a list of numbers

    ::
        >>> data = [1, 2, 3, 4, 5]
        >>> adev(data, data[0]) ## Absolute deviation from the first point
        [0, 1, 2, 3, 4]
        >>> adev(data, data[-1]) ## Absolute deviation from the last point
        [4, 3, 2, 1, 0]
        >>> adev(data) ## Absolute deviation from the median
        [2, 1, 0, 1, 2]
        >>> adev(data, mean) ## Absolute deviation from the arithmetic mean
        [2.0, 1.0, 0.0, 1.0, 2.0]
        >>> adev(data, mode) ## Absolute deviation from the mode
        [4, 3, 2, 1, 0]
        >>> adev(data, harmonic_mean) ## Absolute deviation from the harmonic mean
        [1.1897810218978102, 0.1897810218978102, 0.8102189781021898, 1.8102189781021898, 2.81021897810219]
    '''

    try:
        c = m(data)
    except TypeError:
        c = m
    return [abs(d - c) for d in data]

def adev1(data, m=median, e=1):
    '''
    Like :func:`adev`, but raise each element to *e* and after the sum take the *e*-th root::

        >>> data = [1, 2, 3, 4, 5]
        >>> adev1(data, data[0], 2)
        5.477225575051661

    Equivalent to::

        >>> adev(data, data[0]) ## Absolute deviation from the first point
        [0, 1, 2, 3, 4]
        >>> sum(map(lambda i: i ** 2, adev(data, data[0]))) ** (1. / 2)
        5.477225575051661

    other examples::

        >>> adev1(data, data[0], 9)
        4.033122181324529
        >>> sum(map(lambda i: i ** 9, adev(data, data[0]))) ** (1. / 9)
        4.033122181324529
        >>> adev1(data, median, 9)
        2.1605878472891096
        >>> sum(map(lambda i: i ** 9, adev(data, median))) ** (1. / 9)
        2.1605878472891096
    '''

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
    if n < 2:
        raise StatsError('standard deviation requires at leas 2 elements')
    return math.sqrt(sum((d - m) ** 2for d in data) / (n - 1))

def pstdev(data):
    '''
    Returns the standard deviation of a population of *data*::

        >>> d = [1, 2, 3, 3, 5, 5, 5, 8]
        >>> pstdev(d) # doctest: +ELLIPSIS
        2.06155...
    '''
    m = mean(data)
    n = len(data)
    if n < 2:
        raise StatsError('standard deviation requires at leas 2 elements')
    return math.sqrt(sum(((d - m) ** 2 for d in data)) / n)

def pvariance(data): ## population variance
    n = len(data)
    if n < 2:
        raise StatsError('variance requires at leas 2 elements')
    return _two_pass(data) / n

def variance(data): ## sample variance
    n = len(data)
    if n < 2:
        raise StatsError('variance requires at leas 2 elements')
    return _two_pass(data) / (n - 1)

def sums(data1, data2):
    s = collections.namedtuple('Sum', 'sumx sumy sumxy')
    xy = map(lambda i: i[0] * i[1], zip(data1, data2))
    return s(sum(data1), sum(data2), sum(xy))

def pcov(data1, data2):
    sp = _sp(data1, data2)
    return sp / len(data1)

def cov(data1, data2):
    sp = _sp(data1, data2)
    return sp / (len(data1) - 1)

def pcv(data): ## coefficient of variation of a population
    return pstdev(data) / mean(data)

def cv(data): ## coefficient of variation of a sample
    return stdev(data) / mean(data)

def id(data): ## Index of dispersion
    return variance(data) / mean(data)

def sterrmean(s, n, N=None):
    if N is not None and N < n:
        raise StatsError('the population cannot be smaller than the sample')
    if n < 0:
        raise StatsError('the sample size cannot be negative')
    if s < 0.0:
        raise StatsError('standard deviation cannot be negative')
    if n == 0:
        if N == 0:
            return float('nan')
        else:
            return float('+inf')
    st = s / math.sqrt(n)
    if N is not None:
        return st * math.sqrt((N - n) / (N - 1))
    return st

def skewness(data):
    return c_moment(data, 3) / (c_moment(data, 2) ** (3 / 2))

def quartile_skewness(data, m=0): # or bowley skewness
    q1, q2, q3 = quartiles(data, m)
    return (q1 - 2*q2 + q3) / (q3 - q1)

def pearson_mode_skewness(m, mo, s):
    return (m - mo) / s

def pearson_skewness(m, mo, me, s):
    return (3 * (m - mo) / s, 3 * (m - me) / s)

def circular_mean(data, deg=False): ## FIXME does not work
    if deg:
        data = map(math.radians, data)
    n = len(data)
    return math.atan2(math.fsum(math.sin(d) for d in data) / n, math.fsum(math.cos(d) for d in data) / n)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
#!/usr/bin/env python

from __future__ import division
import math
import operator
import functools
import collections

class StatsError(ValueError):
    pass

def _root(n, k):
    return n ** (1 / k)

def _bin_coeff(n, k):
    return math.factorial(n) / (math.factorial(n - k) * math.factorial(k))

def _sorted(func):
    @functools.wraps(func)
    def wrapper(data, *args, **kwargs):
        return func(sorted(data), *args, **kwargs)
    return wrapper

def _split(func):
    @functools.wraps(func)
    def wrapper(xdata, ydata=None):
        if ydata is None:
            xdata, ydata = zip(*xdata)
        n = len(xdata)
        if n != len(ydata):
            raise StatsError('xdata and ydata must have the same length')
        if n == 0:
            raise StatsError('xdata and ydata cannot be empty')
        return func(xdata, ydata, n)
    return wrapper

def _two_pass(data): ## two-pass algorithm
    m = mean(data)
    return sum((d - m) ** 2 for d in data)

def _sp(data1, data2):
    mx, my = mean(data1), mean(data2)
    return sum((x - mx) * (y - my) for x, y in zip(data1, data2))

def sum(data):
    return math.fsum(data)

## Others

def afreq(data):
    return [data.count(d) for d in set(data)]

def rfreq(data):
    n = len(data)
    return map(lambda i: i / n, afreq(data))

## Averages

def mean(data):
    '''
    Returns the arithmetic mean of *data*.
    The mean is the sum of the data divided by the number of data.
    It is usually called *the average*::

        >>> mean([1, -4, 32, 5, 3, 1]) # doctest: +ELLIPSIS
        6.333333...
    '''

    n = len(data)
    if n == 0:
        raise StatsError('no mean defined for empty data sets')
    return sum(data) / n

def weighted_mean(data, weights=None):
    dt = list(set(data))
    if weights is None:
        weights = map(lambda i: data.count(i), dt)
    if len(data) != len(weights) or (len(data), len(weights)) == (0, 0):
        raise StatsError('data and weights must have the same length and cannot be empty')
    return sum(map(lambda i: i[0] * i[1], zip(dt, weights))) / sum(weights)

def geo_mean(data):
    '''
    Returns the geometric mean of *data*.
    The geometric mean is the n-th root of the product of the data.
    It is usually used for averaging exponential growth rates::

        >>> d = [1, .244, 12, 53]
        >>> geo_mean(d) # doctest: +ELLIPSIS
        3.5294882...
    '''

    return _root(reduce(operator.mul, data), len(data))

def quadratic(data):
    '''
    Returns the quadratic mean of *data*.
    The quadratic mean is the square root of the arithmetic mean of the squares
    of the data. It is usually used when the data vary from negative to positive::

        >>> d = [1, -23, 24, -2, 42, -2]
        >>> quadratic(d) # doctest: +ELLIPSIS
        21.9012937...
    '''

    return math.sqrt(mean([d ** 2 for d in data]))

def harmonic_mean(data):
    '''
    Returns the harmonic mean of *data*.
    The harmonic mean is the reciprocal of the arithmetic mean of the reciprocals
    of the data. It is usually used for averaging rates::

        >>> d = [1, 2, -5, 2, 0.2]
        >>> harmonic_mean(d) # doctest: +ELLIPSIS
        0.73529411...
        >>> d = [1, -24, .2, 0]
        >>> harmonic_mean(d)
        Traceback (most recent call last):
          File "<pyshell#4>", line 1, in <module>
            harmonic_mean(d)
          File "pyst.py", line 97, in harmonic_mean
            return len(data) / sum(1 / d for d in data)
          File "pyst.py", line 43, in sum
            return math.fsum(data)
          File "pyst.py", line 97, in <genexpr>
            return len(data) / sum(1 / d for d in data)
        ZeroDivisionError: float division
    '''

    return len(data) / sum(1 / d for d in data)

def running_average(data, m=mean):
    '''
    Iterates over *data* yielding the running average.

    :param m: a function that compute the average
    :rtype: generator expression

    Examples::

        >>> d = [1, -32, 42, -3, 42]
        >>> for i in running_average(d): print i # doctest: +ELLIPSIS
        1.0
        -15.5
        3.66666...
        2.0
        10.0
        >>> list(running_average(d, harmonic_mean)) # doctest: +ELLIPSIS
        [1.0, 2.06451612..., 3.022488755..., 6.06772009..., 7.320261437...]
        >>> list(running_average(d, quadratic)) # doctest: +ELLIPSIS
        [1.0, 22.6384628..., 30.49043565..., 26.4480623..., 30.20595967...]
    '''

    for i in xrange(1, len(data) + 1):
        s = data[:i]
        yield m(s)

@ _sorted
def trimmed_mean(data, p, m=mean):
    '''
    Returns the trimmed mean of *data*.
    A trimmed mean is calculated by discarding a certain percentage of the
    lowest and the highest scores and then computing the mean of the remaining scores.

    :param p: the rate to discard
    :param m: a function that accept an argument. After the data trimming, the remaining score will be passed to this function, default to ``mean``
    :rtype: float

    Some examples::

        >>> d = [1, 23, 3, 19, 45, 1003]
        >>> mean(d)
        182.33333333333334
        >>> trimmed_mean(d, 0)
        182.33333333333334
        >>> trimmed_mean(d, 20)
        22.5
        >>> trimmed_mean(d, 25)
        22.5
        >>> trimmed_mean(d, 40)
        22.5
        >>> trimmed_mean(d, 70)
        21.0
        >>> trimmed_mean(d, 100) ## the median
        21.0
        >>> trimmed_mean(d, 20, harmonic_mean) # doctest: +ELLIPSIS
        8.85611348...
        >>> trimmed_mean(d, 20, quadratic) # doctest: +ELLIPSIS
        27.0370116...
        >>> trimmed_mean(d, 20, geo_mean) # doctest: +ELLIPSIS
        15.5848921...
    '''

    n = len(data)
    if not n:
        raise StatsError('no trimmed mean defined for empty data set')
    if p == 0:
        return m(data)
    me = median(data)
    c = n / 2 - 1
    i = int(math.ceil(c * p / 100))
    d = data[i:-i]
    if not d:
        return me
    return m(d)

def circular_mean(data, deg=False):
    if deg:
        data = map(math.radians, data)
    n = len(data)
    th = math.atan2(math.fsum(math.sin(d) for d in data) / n, math.fsum(math.cos(d) for d in data) / n)
    if deg:
        th = math.degrees(th)
    return th

###############################
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
    return sum(abs(j - i) for i in data for j in data) / (n * (n - 1))

def rmd(data):
    return md(data) / mean(data)

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

    :param integer m: the method to use, it can be a number from 0 to 9
    :rtype: an element of *data*

    ======  ======================================================
    Method  Description
    ======  ======================================================
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    ======  ======================================================

    '''

    def _compute(h):
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
        return _compute(h)
    def r5():
        if p < .5 / n:
            return data[0]
        if p >= (n - .5) / n:
            return data[-1]
        h = n * p + .5
        return _compute(h)
    def r6():
        if p < 1 / (n + 1):
            return data[0]
        if p >= n / (n + 1):
            return data[-1]
        h = (n + 1) * p
        return _compute(h)
    def r7():
        if p == 1:
            return data[-1]
        h = (n - 1) * p + 1
        return _compute(h)
    def r8():
        if p < (2 / 3) / (n + 1/3):
            return data[0]
        if p >= (n - 1 / 3) / (n + 1 / 3):
            return data[-1]
        h = (n + 1 / 3) * p + 1 / 3
        return _compute(h)
    def r9():
        if p < (5 / 8) / (n + 1 / 4):
            return data[0]
        if p >= (n - 3 / 8) / (n + 1 / 4):
            return data[-1]
        h = (n + 1 / 4) * p + 3 / 8
        return _compute(h)
    def unknown():
        if p < (3 / 2) / (n + 2):
            return data[0]
        if p >= (n + 1 / 2) / (n + 2):
            return data[-1]
        h = (n + 2) * p - .5
        return _compute(h)

    n = len(data)
    methods = [r1, r2, r3, r4, r5, r6, r7, r8, r9, unknown]
    return methods[m]()

@ _sorted
def parametrized_quantile(data, p, scheme):
    n = len(data)
    a, b, c, d = scheme
    if n < 2:
        raise StatsError('need at least 2 elements to split data into quantiles')
    x = a + (n + b) * p
    i, ii = map(int, (math.ceil(x), math.floor(x)))
    return data[i] + (data[ii] - data[i]) * (c + d * math.modf(x)[0])

@ _sorted
def quartiles(data, m=1):
    '''
    Returns the quantiles Q1, Q2 and Q3, where one quarter of the data is below Q1, two quarters below Q2 and three quarters below Q3.
    The exact values Q1, Q2 and Q3 depend on the method (default to 1):

    ======  ============================================================
    Method  Description
    ======  ============================================================
    0       Standard method (1)
    1       Method used by Minitab software (2)
    2       Tukey's method, the median is included in the two halves (3)
    3       Method recommended by Moore and McCabe
    4       Method recommended by Mendenhall and Sincich (4)
    5       Method recommended by Freund and Perles (2) (5)
    ======  ============================================================

    Notes:

    (1) Compute the first quartile Q1 with ``n / 4`` and Q3 with ``3n / 4``
    (2) Uses linear interpolation between items
    (3) Equivalent to Tukey's hinges H1, M, H2
    (4) Ensure that value returned are always data points
    (5) For compatibility with Microsoft Excel and OpenOffice, use this method
    '''

    n = len(data)
    if n < 3:
        raise StatsError('need at least 3 items')
    methods = [[(n / 4, n * 3 / 4), (n / 4, n * 3 / 4)], ## Standard method
               [((n + 1) / 4, (3*n + 3) / 4), ((n + 1) / 4, (3*n + 3) / 4)], ## Minitab's method
               [((n + 2) / 4, (3*n + 2) / 4), ((n + 3) / 4, (3*n + 1) / 4)], ## Tukey's method
               [((n + 2) / 4, (3*n + 2) / 4), ((n + 1) / 4, (3*n + 3) / 4)], ## Moore and McCabe
               [map(round, ((n + 1) / 4, (3*n + 3) / 4)), map(round, ((n + 1) / 4, (3*n + 3) / 4))], ## Mendenhall and Sincich
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
    return quantile(data, d / 10, m)

def percentile(data, p, m=0):
    if not 0 <= p <= 100:
        raise StatsError('p must be between 0 and 100')
    return quantile(data, p / 100, m)

def iqr(data, m=0):
    q1, _, q3 = quartiles(data, m)
    return abs(q3 - q1)

def idr(data, m=0): ## Inter-decile range
    d1, d9 = decile(data, 0, m), decile(data, 9, m)
    return d1 - d9

def adev(data, m=median): ## absolute deviation
    '''
    Returns the absolute deviation from the point *m*.

    :param list data: the data
    :param m: can be either a callable object or a number. If it is a callable the absolute deviation will be computed from ``m(data)`` (it must be a number)
    :type m: number or callable
    :rtype: a list of numbers

    Some examples::

        >>> data = [1, 2, 3, 4, 5]
        >>> adev(data, data[0]) ## Absolute deviation from the first point
        [0, 1, 2, 3, 4]
        >>> adev(data, data[-1]) ## Absolute deviation from the last point
        [4, 3, 2, 1, 0]
        >>> adev(data) ## Absolute deviation from the median
        [2, 1, 0, 1, 2]
        >>> adev(data, mean) ## Absolute deviation from the arithmetic mean
        [2.0, 1.0, 0.0, 1.0, 2.0]
        >>> adev(data + [1], mode) ## Absolute deviation from the mode
        [0, 1, 2, 3, 4, 0]
        >>> adev(data, harmonic_mean) ## Absolute deviation from the harmonic mean # doctest: +ELLIPSIS
        [1.18978102..., 0.18978102..., 0.81021897..., 1.81021897..., 2.81021897...]
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
        >>> adev1(data, data[0], 2) # doctest: +ELLIPSIS
        5.47722557...

    Equivalent to::

        >>> adev(data, data[0]) ## Absolute deviation from the first point
        [0, 1, 2, 3, 4]
        >>> sum(map(lambda i: i ** 2, adev(data, data[0]))) ** (1. / 2) # doctest: +ELLIPSIS
        5.47722557...

    other examples::

        >>> adev1(data, data[0], 9) # doctest: +ELLIPSIS
        4.03312218...
        >>> sum(map(lambda i: i ** 9, adev(data, data[0]))) ** (1. / 9) # doctest: +ELLIPSIS
        4.03312218...
        >>> adev1(data, median, 9) # doctest: +ELLIPSIS
        2.16058784...
        >>> sum(map(lambda i: i ** 9, adev(data, median))) ** (1. / 9) # doctest: +ELLIPSIS
        2.16058784...
    '''

    try:
        c = m(data)
    except TypeError:
        c = m
    return _root(sum(abs(d - c) ** e for d in data), e)

def m_d(data): ## mean absolute deviation
    return mean(adev(data, mean))

def mad(data): ## median absolute deviation
    return median(adev(data))

#####################
## Measures of spread

def stdev(data):
    '''
    Returns the sample standard deviation of *data*::

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
    Returns the population standard deviation of *data*::

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

################
## Other moments

def moment(data, k): ## Central moment
    if k == 0:
        return 1.
    if k == 1:
        return 0.
    m = mean(data)
    return sum((d - m) ** k for d in data) / len(data)

def s_moment(data, k): ## Standardized moment 
    return moment(data, k) / (stdev(data) ** k)

def r_moment(data, k): ## Raw moment
    m = mean(data)
    if k == 1:
        return m
    return sum(_bin_coeff(k, n) * moment(data, n) * m ** (k - n) for n in xrange(k + 1))

def skewness(data):
    return moment(data, 3) / (moment(data, 2) ** (3 / 2))

def kurtosis(data): ## kurtosis coeff, excess kurtosis
    b = moment(data, 4) / (moment(data, 2) ** 2)
    return b, b - 3

def quartile_skewness(q1, q2, q3): # or bowley skewness
    return (q1 - 2*q2 + q3) / (q3 - q1)

def pearson_mode_skewness(m, mo, s):
    if s == 0:
        return float('nan') if m == mo else float('+inf')
    if s > 0:
        return (m - mo) / s
    raise StatsError('standard deviation cannot be negative')

def pearson_skewness(m, mo, me, s):
    if s < 0:
        raise StatsError('standard deviation cannot be negative')
    return (3 * (m - mo) / s, 3 * (m - me) / s)

###########################
## Indexes and coefficients

def pcv(data): ## coefficient of variation of a population
    return pstdev(data) / mean(data)

def cv(data): ## coefficient of variation of a sample
    return stdev(data) / mean(data)

def id(data): ## Index of dispersion
    return variance(data) / mean(data)

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

    if len(data) == 0:
        raise StatsError('no Gini coefficient defined for empty data set')
    return 1 - sum(map(lambda i: i ** 2, rfreq(data)))

def gini1(data):
    '''
    Returns the normalized Gini coefficient, a number between 0 and 1. It is 0
    when all the data are equal and 1 when all the data are different::

        >>> d = [1, 2, 3, 4]
        >>> gini1(d)
        1.0
        >>> d = [1, 2, 3, 4, 5, 1, 3, 4, 6, 7, 2]
        >>> gini1(d) # doctest: +ELLIPSIS
        0.92727272...
        >>> d = [1]
        >>> gini1(d)
        0.0
        >>> gini([])
        Traceback (most recent call last):
          File "<pyshell#1>", line 1, in <module>
            gini([])
          File "pyst.py", line 559, in gini
            raise StatsError('no Gini coefficient defined for empty data set')
        StatsError: no Gini coefficient defined for empty data set
    '''

    n = len(data)
    if n == 1:
        return 0.
    return gini(data) * n / (n - 1)

def shannon(data):
    '''
    Returns the Shannon index of *data*, a number between ``0`` and ``ln n``.
    It is ``0`` when all the data are equal

        
    '''

    if len(data) == 0:
        raise StatsError('no Shannon index defined for empty data set')
    return - sum(r * math.log(r) for r in rfreq(data))

def shannon1(data):
    n = len(data)
    if n == 1:
        return 0.
    return shannon(data) / math.log(n)

## Other functions

def z_score(x, m, s):
    return (x - m) / s

## Multivariate

@ _split
def pcov(xdata, ydata, n):
    sp = _sp(xdata, ydata)
    return sp / n

@ _split
def cov(xdata, ydata, n):
    sp = _sp(xdata, ydata)
    return sp / (n - 1)

@ _split
def qcorr(xdata, ydata, n):
    ac = bd = s = 0
    xmed, ymed = median(xdata), median(ydata)
    for x, y in zip(xdata, ydata):
        if (x > xmed and y > ymed) or (x < xmed and y < ymed):
            ac += 1
        elif (x > xmed and y < ymed) or (x < xmed and y > ymed):
            bd += 1
        else:
            s += 1
    return (ac - bd) / (n - s)

@ _split
def pcorr(xdata, ydata, n):
    sx, sy, sxy, sx2, sy2 = sum(xdata), sum(ydata), \
                            sum(x * y for x, y in zip(xdata, ydata)), \
                            sum(x**2 for x in xdata), sum(y**2 for y in ydata)
    return (sxy - sx * sy / n) / math.sqrt((sx2 - sx ** 2 / n) * (sy2 - sy ** 2 / n))

@ _split
def corr(xdata, ydata, n):
    mx, sx = mean(xdata), stdev(xdata)
    my, sy = mean(ydata), stdev(ydata)
    return sum(z_score(x, mx, sx) * z_score(y, my, sy) for x, y in zip(xdata, ydata)) / n


##################################################################################
############# Still in development ###############################################
##################################################################################


if __name__ == '__main__':
    import doctest
    doctest.testmod()

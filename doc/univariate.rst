.. currentmodule:: pyst

Univariate statistics
=====================

.. contents:: Table of contents

Currently :mod:`pyst` offers these functions:

Averages
--------

.. autofunction:: mean

.. autofunction:: weighted_mean

.. autofunction:: harmonic_mean

.. autofunction:: geo_mean

.. autofunction:: quadratic

.. autofunction:: running_average(data, m=mean)

.. autofunction:: trimmed_mean(data, p, m=mean)


Central tendancy
----------------

.. autofunction:: mode

.. autofunction:: median(data)

.. autofunction:: midrange

.. autofunction:: midhinge

.. autofunction:: trimean


Order statistic
---------------

.. autofunction:: quantile(data, p, m=0)

.. autofunction:: quartiles(data, m=1)

.. autofunction:: decile

.. autofunction:: percentile

.. autofunction:: hinges


Spread
------

.. autofunction:: pstdev

.. autofunction:: stdev

.. autofunction:: pvariance

.. autofunction:: variance

.. autofunction:: iqr

.. autofunction:: idr

.. autofunction:: range

.. autofunction:: adev(data, m=mean)

.. autofunction:: adev1(data, m=mean, e=1)

.. autofunction:: md

.. autofunction:: rmd

.. autofunction:: m_d

.. autofunction:: mad


Moments
-------

.. autofunction:: moment

.. autofunction:: s_moment

.. autofunction:: r_moment

.. autofunction:: skewness

.. autofunction:: kurtosis

.. autofunction:: quartile_skewness

.. autofunction:: pearson_skewness

.. autofunction:: pearson_mode_skewness
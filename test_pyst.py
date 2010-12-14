import py
from pyst import *

class TestAverages(object):
    def test_mean(self):
        assert mean([1, 2, 3, 4]) == 2.5
        assert mean([0]) == 0
        with py.test.raises(StatsError):
            mean([])

    def test_weighted_mean(self):
        assert weighted_mean([1, 2, 3, 4]) == mean([1, 2, 3, 4])
        assert weighted_mean([1, 2, 3, 4], [1, 3, 2, 1]) == mean([1, 2, 2, 2, 3, 3, 4])
        with py.test.raises(StatsError):
            weighted_mean([], [1])
            weighted_mean([], [])

    def test_harmonic_mean(self):
        assert harmonic_mean([0.25, 0.5, 1.0, 1.0]) == .5
        assert harmonic_mean([1]) == 1
        assert harmonic_mean([1] * 100) == 1
        with py.test.raises(ZeroDivisionError):
            harmonic_mean([1, 2, 3, 0])
            harmonic_mean([0])
            harmonic_mean([])

    def test_geo_mean(self):
        assert geo_mean([1.0, 2.0, 6.125, 12.25]) == 3.5
        assert geo_mean([2, 8]) == 4
        assert geo_mean([1]) == 1
        assert geo_mean([0]) == 0
        with py.test.raises(TypeError):
            geo_mean([])

    def test_quadratic(self):
        assert quadratic([2, 2, 4, 5]) == 3.5
        assert quadratic([-2, 2, 4, -5]) == 3.5
        assert quadratic([2, -2, 4, 5]) == 3.5
        assert quadratic([-2, 2, -4, 5]) == 3.5
        with py.test.raises(StatsError):
            quadratic([])


class TestCentralTendancy(object):
    def test_mode(self):
        assert mode([0, -42, 24, 24, 2, 1, 4]) == 24
        with py.test.raises(StatsError):
            mode([])
            mode([1, 2, 3, 4])

    def test_median(self):
        assert median([1, 2, -4]) == 1
        assert median([-4, -1, 4, -7]) == -2.5
        with py.test.raises(StatsError):
            median([])

    def test_midrange(self):
        assert midrange([2.0, 4.5, 7.5]) == 4.75
        with py.test.raises(StatsError):
            midrange([])

    def test_midhinge(self):
        assert midhinge([1, 1, 2, 3, 4, 5, 6, 7, 8, 8]) == 4.5
        with py.test.raises(StatsError):
            midhinge([])

    def test_trimean(self):
        assert trimean([1, 1, 3, 5, 7, 9, 10, 14, 18]) == 6.75
        assert trimean([0, 1, 2, 3, 4, 5, 6, 7, 8]) == 4
        d = xrange(100)
        assert trimean(d) == (median(d) + midhinge(d)) / 2
        with py.test.raises(StatsError):
            trimean([])


class TestOrderStatistic(object):
    def test_quantile(self):
        pass

    def test_quartiles(self):
        pass

    def test_decile(self):
        d = xrange(11)
        for i in xrange(11):
            assert decile(d, i) == d[i]
        d = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        assert decile(d, 7) == 14
        with py.test.raises(IndexError):
            decile([], 1)

    def test_percentile(self):
        d = xrange(101)
        for i in xrange(101):
            assert percentile(d, i) == d[i]
        d = xrange(1, 201)
        assert percentile(d, 7) == 15
        assert percentile(d, 7, 2) == 14
        with py.test.raises(IndexError):
            percentile([], 1)

    def test_hinges(self):
        assert hinges(xrange(9)) == (2, 4, 6)
        assert hinges([2, 4, 6, 8, 10, 12, 14, 16, 18]) == (6, 10, 14)
        with py.test.raises(StatsError):
            hinges([])


class TestSpread(object):
    def test_stdev(self):
        pass

    def test_pstdev(self):
        assert pstdev([2, 4, 4, 4, 5, 5, 7, 9]) == 2
        with py.test.raises(StatsError):
            pstdev([])
            pstdev([1])

    def test_variance(self):
        pass

    def test_pvariance(self):
        assert pvariance([2, 4, 4, 4, 5, 5, 7, 9]) == 4
        with py.test.raises(StatsError):
            pvariance([])
            pvariance([1])

    def test_iqr(self):
        d = xrange(102, 119, 2)
        for i in xrange(6):
            assert iqr(d, i) in (8, 10)

    def test_idr(self):
        assert idr(xrange(11)) == -9
        with py.test.raises(IndexError):
            idr([])

    def test_range(self):
        assert range([1, 2]) == 1
        assert range([1, -42, 4, 64, 4, -4]) == 106
        with py.test.raises(StatsError):
            range([])

    def test_adev(self):
        d = [1, 2, 3, 4, 5]
        assert adev(d, d[0]) == [0, 1, 2, 3, 4]
        assert adev(d, d[-1]) == [4, 3, 2, 1, 0]
        assert adev(d) == [2, 1, 0, 1, 2]
        assert adev(d + [6]) == [2.5, 1.5, .5, .5, 1.5, 2.5]
        assert adev(d, mean) == [2, 1, 0, 1, 2]
        with py.test.raises(StatsError):
            adev(d, mode)

    def test_adev1(self):
        pass

    def test_md(self):
        pass

    def test_mad(self):
        assert mad([1, 1, 2, 2, 4, 6, 9]) == 1
        with py.test.raises(StatsError):
            mad([])


class TestMoments(object):
    def test_moment(self):
        d = xrange(1, 6)
        assert moment(d, 0) == 1
        assert moment(d, 1) == 0
        assert moment(d, 2) == 2
        assert moment(d, 3) == 0

    def test_sample_moment(self):
        d = xrange(1, 6)
        assert s_moment(d, 0) == len(d)
        assert s_moment(d, 1) == sum(d)
        assert s_moment(d, 2) == 55

    def test_skewness(self):
        pass

    def test_skewness1(self):
        pass

    def test_kurtosis(self):
        pass

    def test_quantile_skewness(self):
        pass

    def test_pearson_skewness(self):
        pass

    def test_pearson_mode_skewness(self):
        pass
import py
from pyst import *

class TestAverages(object):
    def test_mean(self):
        assert mean([1, 2, 3, 4]) == 2.5
        assert mean([0]) == 0
        with py.test.raises(ZeroDivisionError):
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
        with py.test.raises(ZeroDivisionError):
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
        pass


class TestOrderStatistic(object):
    def test_quantile(self):
        pass

    def test_quartiles(self):
        pass

    def test_decile(self):
        pass

    def test_percentile(self):
        pass

    def test_hinges(self):
        pass


class TestSpread(object):
    def test_stdev(self):
        assert pstdev([2, 4, 4, 4, 5, 5, 7, 9]) == 2
        with py.test.raises(StatsError):
            pstdev([])
            pstdev([1])

    def test_pstdev(self):
        pass

    def test_variance(self):
        pass

    def test_pvariance(self):
        pass

    def test_iqr(self):
        pass

    def test_range(self):
        assert range([1, 2]) == 1
        assert range([1, -42, 4, 64, 4, -4]) == 106
        with py.test.raises(StatsError):
            range([])

    def test_adev(self):
        pass

    def test_adev1(self):
        pass

    def test_md(self):
        pass

    def test_mad(self):
        assert mad([1, 1, 2, 2, 4, 6, 9]) == 1
        with py.test.raises(StatsError):
            mad([])
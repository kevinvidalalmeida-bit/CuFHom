"""Accuracy control: require the true physical norm at half the public rtol.

The first Metric.norm call sets the physical RHS threshold. Scaling that
squared RHS norm by 1/4 makes the subsequent acceptance threshold stricter;
all residual evaluations and the original physical audit remain unchanged.
"""
from contextlib import contextmanager
from unittest.mock import patch
import cufhom_dual_metric as candidate


class HalfToleranceMetric(candidate.Metric):
    def __init__(self, coordinates):
        super().__init__(coordinates)
        self.first=True

    def norm(self, residual):
        values=super().norm(residual)
        if self.first:
            self.first=False
            values=values.copy()
            values[1]*=.25
        return values


@contextmanager
def trial():
    with patch.object(candidate,'Metric',HalfToleranceMetric),candidate.trial():
        yield

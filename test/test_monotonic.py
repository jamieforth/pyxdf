from pyxdf.pyxdf import _is_monotonic

from mock_data_stream import MockStreamData


# Monotonic timeseries data.

def test_strict_increasing():
    timestamps = list(range(-5, 5))
    stream = MockStreamData(timestamps, 1)
    monotonic = _is_monotonic(stream)
    assert monotonic is True


def test_non_decreasing():
    timestamps = [-5] + list(range(-5, 5))
    stream = MockStreamData(timestamps, 1)
    monotonic = _is_monotonic(stream)
    assert monotonic is True
    timestamps = list(range(-5, 5)) + [4]
    stream = MockStreamData(timestamps, 1)
    monotonic = _is_monotonic(stream)
    assert monotonic is True
    timestamps = [-5, -4, -3, -2, -1, -1, 0, 1, 2, 3, 4]
    stream = MockStreamData(timestamps, 1)
    monotonic = _is_monotonic(stream)
    assert monotonic is True
    timestamps = [-5, -4, -3, -2, -1, 0, 1, 1, 2, 3, 4]
    stream = MockStreamData(timestamps, 1)
    monotonic = _is_monotonic(stream)
    assert monotonic is True


# Non-monotonic timeseries data.

def test_decreasing():
    timestamps = [-5, -4, -2, -3, -1, 0, 1, 2, 3, 4]
    stream = MockStreamData(timestamps, 1)
    monotonic = _is_monotonic(stream)
    assert monotonic is False
    timestamps = [-5, -4, -3, -2, -1, 0, 1, 3, 2, 4]
    stream = MockStreamData(timestamps, 1)
    monotonic = _is_monotonic(stream)
    assert monotonic is False

import time
from typing import Callable, ParamSpec, Tuple, TypeVar

import pytest


def _convert_to_human(time: int) -> Tuple[float, str]:
    if time < 1e3:
        return (time, "ns")
    elif time < 1e6:
        return (time / 1e3, "us")
    elif time < 1e9:
        return (time / 1e6, "ms")
    else:
        return (time / 1e9, "s")


P = ParamSpec("P")
T = TypeVar("T")


def _benchmark(
    f: Callable[P, T], f_ref: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> Tuple[T, T]:
    start = time.perf_counter_ns()
    result = f(*args, **kwargs)
    end = time.perf_counter_ns()
    t, unit = _convert_to_human(end - start)
    print(f"\ntime: {t:.2f} {unit}")
    start_ref = time.perf_counter_ns()
    result_ref = f_ref(*args, **kwargs)
    end_ref = time.perf_counter_ns()
    t_ref, unit_ref = _convert_to_human(end_ref - start_ref)
    print(f"time_ref: {t_ref:.2f} {unit_ref}")
    speed = (end - start) / (end_ref - start_ref)
    if speed > 1:
        print(f"{speed:.2f} slower than the reference")
    else:
        print(f"{1./speed:.2f} faster than the reference")
    return (result, result_ref)


@pytest.fixture()
def benchmark():
    return _benchmark

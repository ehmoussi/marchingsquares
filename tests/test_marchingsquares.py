import marchingsquares
from marchingalgo._find_contours_cy import _get_contour_segments
from marchingalgo import find_contours

import numpy as np

import pytest


@pytest.fixture()
def array() -> np.ndarray[float]:
    return np.array(
        [
            0.60869334,
            0.66427812,
            0.52252734,
            0.56246607,
            0.24290979,
            0.60433916,
            0.50990916,
            0.68238096,
            0.76606051,
            0.20246605,
            0.04300992,
            0.00230352,
            0.12984644,
            0.49170811,
            0.17470651,
            0.39569179,
            0.54147536,
            0.10597811,
            0.68128288,
            0.55192845,
            0.53272546,
            0.35442286,
            0.0844035,
            0.53447815,
            0.71911536,
        ]
    ).reshape(5, 5)


def test_get_contour_segments(array: np.ndarray[float]) -> None:
    segments_ref = _get_contour_segments(np.array(array), 0.5, False, None)
    segments = marchingsquares.get_contour_segments(array, level=0.5)
    assert len(segments) == len(
        segments_ref
    ), f"The number of segments is different {len(segments)}!={len(segments_ref)}"
    for segment, segment_ref in zip(segments, segments_ref):
        for point, point_ref_v in zip(segment, segment_ref):
            point_ref = marchingsquares.Point.new(point_ref_v[0], point_ref_v[1])
            assert point.close(
                point_ref, 1e-16
            ), f"({point.x}, {point.y}) != ({point_ref.x}, {point_ref.y})"


def test_marching_squares(array: np.ndarray[float]) -> None:
    contours_ref = find_contours(np.array(array), 0.5)
    contours = marchingsquares.marching_squares(array, level=0.5)
    assert len(contours) == len(
        contours_ref
    ), f"The number of contours is different {len(contours)}!={len(contours_ref)}"
    for index, (contour, contour_ref) in enumerate(zip(contours, contours_ref)):
        assert len(contour) == len(contour_ref)
        for point, point_ref_v in zip(contour, contour_ref):
            point_ref = marchingsquares.Point.new(point_ref_v[0], point_ref_v[1])
            assert point.close(
                point_ref, 1e-16
            ), f"({point.x}, {point.y}) != ({point_ref.x}, {point_ref.y}) in contour {index}"


@pytest.fixture()
def random_array() -> np.ndarray[float, 2]:
    size = int(np.random.random(1)[0] * 2e3)
    print(size * size)
    array = np.random.random(size * size)
    return array.reshape(size, size)


def test_get_contour_segments_random(random_array: np.ndarray[float, 2]) -> None:
    import time

    start_ref = time.perf_counter_ns()
    segments_ref = _get_contour_segments(random_array, 0.5, False, None)
    end_ref = time.perf_counter_ns()
    start = time.perf_counter_ns()
    segments = marchingsquares.get_contour_segments(random_array, level=0.5)
    end = time.perf_counter_ns()
    print(
        f"\ntime_ref: {(end_ref - start_ref) * 1e-6} ms\ntime: {(end - start) * 1e-6} ms"
    )
    assert len(segments) == len(
        segments_ref
    ), f"The number of segments is different {len(segments)}!={len(segments_ref)}"
    for segment, segment_ref in zip(segments, segments_ref):
        for point, point_ref_v in zip(segment, segment_ref):
            point_ref = marchingsquares.Point.new(point_ref_v[0], point_ref_v[1])
            assert point.close(
                point_ref, 1e-16
            ), f"({point.x}, {point.y}) != ({point_ref.x}, {point_ref.y})"


def test_marching_squares_random_array(random_array: np.ndarray[float, 2]) -> None:
    import time

    start_ref = time.perf_counter_ns()
    contours_ref = find_contours(random_array, 0.5)
    end_ref = time.perf_counter_ns()
    start = time.perf_counter_ns()
    contours = marchingsquares.marching_squares(random_array, level=0.5, tol=1e-16)
    end = time.perf_counter_ns()
    print(
        f"\ntime_ref: {(end_ref - start_ref) * 1e-6} ms\ntime: {(end - start) * 1e-6} ms"
    )
    assert len(contours) == len(
        contours_ref
    ), f"The number of contours is different {len(contours)}!={len(contours_ref)}"
    for contour, contour_ref in zip(contours, contours_ref):
        assert len(contour) == len(contour_ref)
        for point, point_ref_v in zip(contour, contour_ref):
            point_ref = marchingsquares.Point.new(point_ref_v[0], point_ref_v[1])
            assert point.close(
                point_ref, 1e-10
            ), f"({point.x}, {point.y}) != ({point_ref.x}, {point_ref.y})"

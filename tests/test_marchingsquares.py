import contextlib
import time
from typing import Generator
import marchingsquares
from marchingalgo._find_contours_cy import _get_contour_segments
from marchingalgo import find_contours
import marchingsquares.marchingsquares
from numpy.typing import NDArray

import numpy as np

import pytest


@contextlib.contextmanager
def measure_time(is_ref: bool) -> Generator[None, None, None]:
    start = time.perf_counter_ns()
    try:
        yield
    finally:
        end = time.perf_counter_ns()
        t = float(end - start)
        if t < 0.1 * 1e3:
            unit = "ns"
        elif t < 0.1 * 1e6:
            t /= 1e3
            unit = "us"
        elif t < 0.1 * 1e9:
            t /= 1e6
            unit = "ms"
        else:
            t /= 1e9
            unit = "s"
        if is_ref:
            prefix = "\ntime_ref"
        else:
            prefix = "time"
        print(f"{prefix}: {t:.2f} {unit}")


@pytest.fixture()
def array() -> NDArray[np.float64]:
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


def test_get_contour_segments(array: NDArray[np.float64]) -> None:
    segments_ref = _get_contour_segments(array, 0.5, False, None)
    segments = np.array(
        marchingsquares.get_contour_segments(
            array.flatten().tolist(), (array.shape[0], array.shape[1]), level=0.5
        )
    ).reshape(-1, 4)
    assert len(segments) == len(
        segments_ref
    ), f"The number of segments is different {len(segments)}!={len(segments_ref)}"
    for segment, segment_ref in zip(segments, segments_ref):
        for point, point_ref in zip(segment.reshape(2, 2), segment_ref):
            assert marchingsquares.close(
                point[0], point[1], point_ref[0], point_ref[1], 1e-16
            ), f"({point[0]}, {point[1]}) != ({point_ref[0]}, {point_ref[1]})"


def test_marching_squares(array: NDArray[np.float64]) -> None:
    contours_ref = find_contours(np.array(array), 0.5)
    contours = marchingsquares.marching_squares(
        array.flatten().tolist(), (array.shape[0], array.shape[1]), level=0.5
    )
    assert len(contours) == len(
        contours_ref
    ), f"The number of contours is different {len(contours)}!={len(contours_ref)}"
    for contour, contour_ref in zip(contours, contours_ref):
        assert len(contour) / 2 == len(contour_ref)
        for point, point_ref in zip(np.array(contour).reshape(-1, 2), contour_ref):
            assert marchingsquares.close(
                point[0], point[1], point_ref[0], point_ref[1], 1e-16
            ), f"({point[0]}, {point[1]}) != ({point_ref[0]}, {point_ref[1]})"


@pytest.fixture(scope="module")
def random_array() -> NDArray[np.float64]:
    size = 2000  # int(np.random.random(1)[0] * 1e4)
    array = np.random.random(size * size)
    return array.reshape(size, size)


def test_get_contour_segments_random(random_array: NDArray[np.float64]) -> None:
    with measure_time(is_ref=True):
        segments_ref = _get_contour_segments(random_array, 0.5, False, None)
    flat_random_array = random_array.flatten().tolist()
    with measure_time(is_ref=False):
        segments = marchingsquares.get_contour_segments(
            flat_random_array,
            (random_array.shape[0], random_array.shape[1]),
            level=0.5,
        )
    segments_a = np.array(segments).reshape(-1, 4)
    assert len(segments_a) == len(
        segments_ref
    ), f"The number of segments is different {len(segments)}!={len(segments_ref)}"
    for segment, segment_ref in zip(segments_a, segments_ref):
        for point, point_ref in zip(segment.reshape(2, 2), segment_ref):
            assert marchingsquares.close(
                point[0], point[1], point_ref[0], point_ref[1], 1e-16
            ), f"({point[0]}, {point[1]}) != ({point_ref[0]}, {point_ref[1]})"


def test_marching_squares_random(random_array: NDArray[np.float64]) -> None:
    with measure_time(is_ref=True):
        contours_ref = find_contours(random_array, 0.5)
    flat_random_array = random_array.flatten().tolist()
    with measure_time(is_ref=False):
        contours = marchingsquares.marching_squares(
            flat_random_array,
            (random_array.shape[0], random_array.shape[1]),
            level=0.5,
            tol=1e-16,
        )
    assert len(contours) == len(
        contours_ref
    ), f"The number of contours is different {len(contours)}!={len(contours_ref)}"
    for contour, contour_ref in zip(contours, contours_ref):
        assert len(contour) / 2 == len(contour_ref)
        for point, point_ref in zip(np.array(contour).reshape(-1, 2), contour_ref):
            assert marchingsquares.close(
                point[0], point[1], point_ref[0], point_ref[1], 1e-16
            ), f"({point[0]}, {point[1]}) != ({point_ref[0]}, {point_ref[1]})"


@pytest.fixture(scope="module")
def random_mask(random_array: NDArray[np.float64]) -> NDArray[np.bool]:
    return (
        np.random.random(random_array.shape[0] * random_array.shape[1]).reshape(
            random_array.shape
        )
        < 0.1
    )


def test_get_contour_segments_random_with_mask(
    random_array: NDArray[np.float64], random_mask: NDArray[np.bool]
) -> None:
    with measure_time(is_ref=True):
        segments_ref = _get_contour_segments(random_array, 0.5, False, mask=random_mask)
    flat_random_array = random_array.flatten().tolist()
    flatten_random_mask = random_mask.flatten().tolist()
    with measure_time(is_ref=False):
        segments = marchingsquares.get_contour_segments(
            flat_random_array,
            (random_array.shape[0], random_array.shape[1]),
            level=0.5,
            mask=flatten_random_mask,
        )
    segments_a = np.array(segments).reshape(-1, 4)
    assert len(segments_a) == len(
        segments_ref
    ), f"The number of segments is different {len(segments_a)}!={len(segments_ref)}"
    for segment, segment_ref in zip(segments_a, segments_ref):
        for point, point_ref in zip(segment.reshape(2, 2), segment_ref):
            assert marchingsquares.close(
                point[0], point[1], point_ref[0], point_ref[1], 1e-16
            ), f"({point[0]}, {point[1]}) != ({point_ref[0]}, {point_ref[1]})"


def test_marching_squares_random_with_mask(
    random_array: NDArray[np.float64], random_mask: NDArray[np.bool]
) -> None:
    with measure_time(is_ref=True):
        contours_ref = find_contours(random_array, 0.5, mask=random_mask)
    flat_random_array = random_array.flatten().tolist()
    flatten_random_mask = random_mask.flatten().tolist()
    with measure_time(is_ref=False):
        contours = marchingsquares.marching_squares(
            flat_random_array,
            (random_array.shape[0], random_array.shape[1]),
            level=0.5,
            tol=1e-16,
            mask=flatten_random_mask,
        )
    assert len(contours) == len(
        contours_ref
    ), f"The number of contours is different {len(contours)}!={len(contours_ref)}"
    for contour, contour_ref in zip(contours, contours_ref):
        assert len(contour) / 2 == len(contour_ref)
        for point, point_ref in zip(np.array(contour).reshape(-1, 2), contour_ref):
            assert marchingsquares.close(
                point[0], point[1], point_ref[0], point_ref[1], 1e-16
            ), f"({point[0]}, {point[1]}) != ({point_ref[0]}, {point_ref[1]})"


def test_marching_squares_with_incorrect_mask_size(
    random_array: NDArray[np.float64], random_mask: NDArray[np.bool]
) -> None:
    mask = random_mask[:, :-1]
    with pytest.raises(ValueError, match="must have the same length"):
        marchingsquares.marching_squares(
            random_array.flatten().tolist(),
            (random_array.shape[0], random_array.shape[1]),
            level=0.5,
            tol=1e-16,
            mask=mask.flatten().tolist(),
        )


def test_bad_array_shape():
    array = [0, 1, 1]
    with pytest.raises(ValueError, match="given shape are incompatible"):
        marchingsquares.marching_squares(array, (2, 2), 0.5)

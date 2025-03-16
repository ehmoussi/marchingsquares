import marchingsquares
from marchingalgo._find_contours_cy import _get_contour_segments
from marchingalgo import find_contours
from numpy.typing import NDArray
import numpy as np

import pytest


def get_contour_segments(*args, **kwargs):
    segments_ref = np.asarray(
        _get_contour_segments(*args, **kwargs),
    )
    return segments_ref.reshape(len(segments_ref), 2, 2)


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
    segments = np.array(marchingsquares.get_contour_segments(array, level=0.5)).reshape(
        -1, 4
    )
    assert len(segments) == len(
        segments_ref
    ), f"The number of segments is different {len(segments)}!={len(segments_ref)}"
    for segment, segment_ref in zip(segments, segments_ref):
        for point, point_ref in zip(segment.reshape(2, 2), segment_ref):
            assert marchingsquares.close(
                point[0], point[1], point_ref[0], point_ref[1], 1e-16
            ), f"({point[0]}, {point[1]}) != ({point_ref[0]}, {point_ref[1]})"


def test_marching_squares(array: NDArray[np.float64]) -> None:
    contours_ref = find_contours(array, level=0.5)
    contours = marchingsquares.marching_squares(array, level=0.5)
    assert len(contours) == len(
        contours_ref
    ), f"The number of contours is different {len(contours)}!={len(contours_ref)}"
    for contour, contour_ref in zip(contours, contours_ref):
        assert contour.shape == contour_ref.shape
        for point, point_ref in zip(np.array(contour).reshape(-1, 2), contour_ref):
            assert marchingsquares.close(
                point[0], point[1], point_ref[0], point_ref[1], 1e-16
            ), f"({point[0]}, {point[1]}) != ({point_ref[0]}, {point_ref[1]})"


@pytest.fixture(scope="module")
def random_array() -> NDArray[np.float64]:
    size = 2000  # int(np.random.random(1)[0] * 1e4)
    array = np.random.random(size * size)
    return array.reshape(size, size)


def test_get_contour_segments_random(
    random_array: NDArray[np.float64], benchmark
) -> None:
    segments, segments_ref = benchmark(
        marchingsquares.get_contour_segments,
        get_contour_segments,
        random_array,
        0.5,
        False,
        None,
    )
    assert (
        segments.shape == segments_ref.shape
    ), f"The number of segments is different {segments.shape}!={segments_ref.shape}"
    for segment, segment_ref in zip(segments, segments_ref):
        for point, point_ref in zip(segment, segment_ref):
            assert marchingsquares.close(
                point[0], point[1], point_ref[0], point_ref[1], 1e-16
            ), f"({point[0]}, {point[1]}) != ({point_ref[0]}, {point_ref[1]})"


def test_marching_squares_random(random_array: NDArray[np.float64], benchmark) -> None:
    contours, contours_ref = benchmark(
        marchingsquares.marching_squares, find_contours, random_array, level=0.5
    )
    assert len(contours) == len(
        contours_ref
    ), f"The number of contours is different {len(contours)}!={len(contours_ref)}"
    for contour, contour_ref in zip(contours, contours_ref):
        assert contour.shape == (contour_ref.shape)
        for point, point_ref in zip(contour, contour_ref):
            assert marchingsquares.close(
                point[0], point[1], point_ref[0], point_ref[1], 1e-16
            ), f"({point[0]}, {point[1]}) != ({point_ref[0]}, {point_ref[1]})"


@pytest.fixture(scope="module")
def random_mask(random_array: NDArray[np.float64]) -> NDArray[np.bool]:
    return np.random.random(random_array.shape) < 0.1


def test_get_contour_segments_random_with_mask(
    random_array: NDArray[np.float64], random_mask: NDArray[np.bool], benchmark
) -> None:
    segments, segments_ref = benchmark(
        marchingsquares.get_contour_segments,
        get_contour_segments,
        random_array,
        0.5,
        False,
        random_mask,
    )
    assert (
        segments.shape == segments_ref.shape
    ), f"The number of segments is different {len(segments)}!={len(segments_ref)}"
    for segment, segment_ref in zip(segments, segments_ref):
        for point, point_ref in zip(segment, segment_ref):
            assert marchingsquares.close(
                point[0], point[1], point_ref[0], point_ref[1], 1e-16
            ), f"({point[0]}, {point[1]}) != ({point_ref[0]}, {point_ref[1]})"


def test_marching_squares_random_with_mask(
    random_array: NDArray[np.float64], random_mask: NDArray[np.bool], benchmark
) -> None:
    contours, contours_ref = benchmark(
        marchingsquares.marching_squares,
        find_contours,
        random_array,
        level=0.5,
        mask=random_mask,
    )
    assert len(contours) == len(
        contours_ref
    ), f"The number of contours is different {len(contours)}!={len(contours_ref)}"
    for contour, contour_ref in zip(contours, contours_ref):
        assert contour.shape == (contour_ref.shape)
        for point, point_ref in zip(contour, contour_ref):
            assert marchingsquares.close(
                point[0], point[1], point_ref[0], point_ref[1], 1e-16
            ), f"({point[0]}, {point[1]}) != ({point_ref[0]}, {point_ref[1]})"


def test_marching_squares_with_incorrect_mask_size(
    random_array: NDArray[np.float64], random_mask: NDArray[np.bool]
) -> None:
    mask = random_mask[:, :-1]
    with pytest.raises(ValueError, match="must have the same length"):
        marchingsquares.marching_squares(random_array, level=0.5, tol=1e-16, mask=mask)


def test_bad_array_shape():
    array = [0, 1, 1]
    with pytest.raises(ValueError, match="2d dimension"):
        marchingsquares.marching_squares(array, level=0.5)

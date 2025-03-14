from marchingsquares import _marchingsquares
from typing import List, Optional
import numpy as np
import typing

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray

close = _marchingsquares.close


def get_contour_segments(
    array: "NDArray[np.float64]",
    level: float,
    vertex_connect_high: bool = False,
    mask: Optional["NDArray[np.bool]"] = None,
) -> "NDArray[np.float64]":
    array = np.asarray(array, dtype=np.float64)
    if len(array.shape) != 2:
        raise ValueError("The array should have a 2d dimension")
    shape = (array.shape[0], array.shape[1])
    array = array.flatten()
    if mask is not None:
        _mask = np.asarray(mask, dtype=np.uint8)
        if _mask.shape != shape:
            raise ValueError(
                f"The array and the mask must have the same length, {mask.shape} != {shape}"
            )
        _mask = np.asarray(mask, dtype=np.uint8).flatten()
    else:
        _mask = np.ones(array.shape, dtype=np.uint8)
    return np.asarray(
        _marchingsquares.get_contour_segments(
            array,
            shape,
            level,
            _mask,
            vertex_connect_high,
        )
    ).reshape((-1, 2, 2))


def marching_squares(
    array: "NDArray[np.float64]",
    level: float,
    is_fully_connected: bool = False,
    mask: Optional["NDArray[np.bool]"] = None,
    tol=1e-10,
) -> List["NDArray[np.float64]"]:
    array = np.asarray(array, dtype=np.float64)
    if len(array.shape) != 2:
        raise ValueError("The array should have a 2d dimension")
    shape = (array.shape[0], array.shape[1])
    array = array.flatten()
    if mask is not None:
        _mask = np.asarray(mask, dtype=np.uint8)
        if _mask.shape != shape:
            raise ValueError(
                f"The array and the mask must have the same length, {mask.shape} != {shape}"
            )
        _mask = _mask.flatten()
    else:
        _mask = np.ones(array.shape, dtype=np.uint8)
    return [
        np.asarray(contour).reshape(-1, 2)
        for contour in _marchingsquares.marching_squares(
            array,
            shape,
            level,
            _mask,
            is_fully_connected,
            tol,
        )
    ]

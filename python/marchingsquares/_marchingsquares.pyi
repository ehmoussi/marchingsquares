from typing import Sequence, Tuple
from numpy.typing import ArrayLike

def close(p1_x: float, p1_y: float, p2_x: float, p2_y: float, tol: float) -> bool:
    pass

def get_contour_segments(
    array: ArrayLike,
    shape: Tuple[int, int],
    level: float,
    mask: ArrayLike,
    vertex_connect_high: bool = False,
) -> ArrayLike:
    pass

def marching_squares(
    array: ArrayLike,
    shape: Tuple[int, int],
    level: float,
    mask: ArrayLike,
    is_fully_connected: bool = False,
    tol=1e-10,
) -> Sequence[ArrayLike]:
    pass

from typing import List, Optional, Tuple

def close(p1_x: float, p1_y: float, p2_x: float, p2_y: float, tol: float) -> bool:
    pass

def get_contour_segments(
    array: List[float],
    shape: Tuple[int, int],
    level: float,
    vertex_connect_high: bool = False,
    mask: Optional[List[bool]] = None,
) -> List[float]:
    pass

def marching_squares(
    array: List[float],
    shape: Tuple[int, int],
    level: float,
    is_fully_connected: bool = False,
    mask: Optional[List[bool]] = None,
    tol=1e-10,
) -> List[List[float]]:
    pass

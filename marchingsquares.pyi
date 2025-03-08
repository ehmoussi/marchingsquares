from typing import List, Optional, Tuple

class Point:
    x: float
    y: float
    def close(self, other: Point, tol: float) -> bool:
        pass

    @staticmethod
    def new(x: float, y: float) -> "Point":
        pass

    @staticmethod
    def top(r: int, c: int, ul: float, ur: float, level: float) -> "Point":
        pass

    @staticmethod
    def bottom(r: int, c: int, ll: float, lr: float, level: float) -> "Point":
        pass

    @staticmethod
    def left(r: int, c: int, ul: float, ll: float, level: float) -> "Point":
        pass

    @staticmethod
    def right(r: int, c: int, ur: float, lr: float, level: float) -> "Point":
        pass

def get_contour_segments(
    array: List[List[float]],
    level: float,
    vertex_connect_high: bool = False,
    mask: Optional[List[List[bool]]] = None,
) -> List[Tuple[Point, Point]]:
    pass

def marching_squares(
    array: List[List[float]],
    level: float,
    is_fully_connected: bool = False,
    mask: Optional[List[List[bool]]] = None,
    tol=1e-10,
) -> List[List[Point]]:
    pass

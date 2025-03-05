import marchingsquares
from marchingalgo._find_contours_cy import _get_contour_segments

import numpy as np


def test_get_contour_segments():
    array = [
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
    segments_ref = _get_contour_segments(
        np.array(array).reshape(5, 5), 0.5, False, None
    )
    segments = marchingsquares.get_contour_segments(
        array, nb_cols=5, level=0.5, vertex_connect_high=False, mask=[]
    )
    assert len(segments) == len(
        segments_ref
    ), f"The number of segments is different {len(segments)}!={len(segments_ref)}"
    for segment, segment_ref in zip(segments, segments_ref):
        for point, point_ref_v in zip((segment.p1, segment.p2), segment_ref):
            point_ref = marchingsquares.Point.new(point_ref_v[0], point_ref_v[1])
            assert point.close(
                point_ref, 1e-16
            ), f"({point.x}, {point.y}) != ({point_ref.x}, {point_ref.y})"

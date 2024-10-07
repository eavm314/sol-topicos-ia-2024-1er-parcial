import pytest
from src.geometry import *

@pytest.mark.parametrize(
    "px, py, x1, y1, x2, y2, expected",
    [
        (2, 2, 1, 1, 3, 3, 0.0),               # Point on the segment
        (0, 2, 1, 1, 3, 3, 1.414),             # Point off the segment
        (2, 4, 3, 3, 3, 5, 1.0),               # Vertical segment
        (4, 2, 1, 1, 5, 1, 1.0),               # Horizontal segment
    ]
)
def test_point_to_segment_distance(px, py, x1, y1, x2, y2, expected):
    assert point_to_segment_distance(px, py, x1, y1, x2, y2) == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize(
    "segment, bboxes, max_distance, expected",
    [
        ([[5, 5], [10, 10]], [[0, 0, 8, 8], [15, 15, 25, 25]], 5, [0, 0, 8, 8]),  # Segment within distance
        ([[5, 5], [10, 10]], [[0, 0, 8, 8], [15, 15, 25, 25]], 2, None),          # Segment outside distance
    ]
)
def test_match_gun_bbox(segment, bboxes, max_distance, expected):
    assert match_gun_bbox(segment, bboxes, max_distance) == expected


@pytest.mark.parametrize(
    "box, expected",
    [
        ([0, 0, 10, 10], (5, 5)),     # Simple box
        ([0, 0, 5, 10], (2, 5)),      # Box with odd dimensions
    ]
)
def test_find_center_box(box, expected):
    assert find_center_box(box) == expected


@pytest.mark.parametrize(
    "polygon, expected",
    [
        ([[0, 0], [4, 0], [2, 4]], (2, 1)),  # Triangle
        ([[0, 0], [0, 4], [4, 4], [4, 0]], (2, 2)),  # Square
        ([[1, 2], [2, 4], [4, 4], [5, 2], [3, 0]], (3, 2)),  # Complex polygon
    ]
)
def test_find_center_polygon(polygon, expected):
    assert find_center_polygon(polygon) == expected


@pytest.mark.parametrize(
    "polygon, expected",
    [
        ([[0, 0], [4, 0], [2, 4]], 8),   # Triangle
        ([[0, 0], [0, 4], [4, 4], [4, 0]], 16),  # Square
        ([[1, 2], [2, 4], [4, 4], [5, 2], [3, 0]], 10),  # Complex polygon
    ]
)
def test_find_area_polygon(polygon, expected):
    assert find_area_polygon(polygon) == expected

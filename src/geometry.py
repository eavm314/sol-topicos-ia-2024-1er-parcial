import numpy as np
import cv2


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    A = np.array([px - x1, py - y1])
    B = np.array([x2 - x1, y2 - y1])

    dot_prod = np.dot(A, B)
    norm_b_sq = np.dot(B, B)
    proj = dot_prod / norm_b_sq if norm_b_sq != 0 else 0

    proj = max(0, min(1, proj))
    closest_point = np.array([x1, y1]) + proj * B
    dist = np.linalg.norm(np.array([px, py]) - closest_point)

    return dist


def match_gun_bbox(segment: list[list[int]], bboxes: list[list[int]], max_distance: int = 10) -> list[int] | None:
    for box in bboxes:
        x1, y1, x2, y2 = box
        # points = [
        #     [[x, y1] for x in np.arange(x1, x2, 5)],
        #     [[x, y2] for x in np.arange(x1, x2, 5)],
        #     [[x1, y] for y in np.arange(y1, y2, 5)],
        #     [[x2, y] for y in np.arange(y1, y2, 5)],
        # ]

        # for sx, sy in segment:
        #     for line in points:
        #         for x, y in line:
        #             d = np.sqrt((x-sx)**2+(y-sy)**2)
        #             if d <= max_distance:
        #                 return box

        edges = [
            (x1, y1, x1, y2),
            (x2, y1, x2, y2),
            (x1, y1, x2, y1),
            (x1, y2, x2, y2)
        ]

        min_distance = 1000000

        for px, py in segment:
            for x1, y1, x2, y2 in edges:
                dist = point_to_segment_distance(px, py, x1, y1, x2, y2)
                min_distance = min(min_distance, dist)

        if min_distance <= max_distance:
            return box

    return None

def find_center_box(box: list[int]) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    return (x1 + x2) // 2, (y1 + y2) // 2

def find_center_polygon(polygon: list[list[int]]) -> tuple[int, int]:
    x = [p[0] for p in polygon]
    y = [p[1] for p in polygon]
    return sum(x) // len(polygon), sum(y) // len(polygon)

def find_area_polygon(polygon: list[list[int]]) -> int:
    n = len(polygon)
    area = 0

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        area += x1 * y2 - y1 * x2

    return int(abs(area) / 2)
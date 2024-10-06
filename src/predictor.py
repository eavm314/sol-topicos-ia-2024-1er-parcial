from ultralytics import YOLO
import numpy as np
import cv2
from shapely.geometry import Polygon, box
from src.models import Detection, PredictionType, Segmentation
from src.config import get_settings

SETTINGS = get_settings()

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


def annotate_detection(image_array: np.ndarray, detection: Detection) -> np.ndarray:
    ann_color = (255, 0, 0)
    annotated_img = image_array.copy()
    for label, conf, box in zip(detection.labels, detection.confidences, detection.boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), ann_color, 3)
        cv2.putText(
            annotated_img,
            f"{label}: {conf:.1f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            ann_color,
            2,
        )
    return annotated_img


def annotate_segmentation(image_array: np.ndarray, segmentation: Segmentation, draw_boxes: bool = True) -> np.ndarray:
    annotated_img = image_array.copy()
    safe_color = (0, 255, 0)
    danger_color = (255, 0, 0)
    for label, box, polygon in zip(segmentation.labels, segmentation.boxes, segmentation.polygons):
        ann_color = safe_color if label == "safe" else danger_color
        cv2.fillPoly(annotated_img, [np.int32(polygon)], ann_color)
        if draw_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), ann_color, 2)
            cv2.putText(
                annotated_img,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                ann_color,
                2,
            )
    return cv2.addWeighted(image_array, 0.5, annotated_img, 0.5, 0)


class GunDetector:
    def __init__(self) -> None:
        print(f"loading od model: {SETTINGS.od_model_path}")
        self.od_model = YOLO(SETTINGS.od_model_path)
        print(f"loading seg model: {SETTINGS.seg_model_path}")
        self.seg_model = YOLO(SETTINGS.seg_model_path)

    def detect_guns(self, image_array: np.ndarray, threshold: float = 0.5):
        results = self.od_model(image_array, conf=threshold)[0]
        labels = results.boxes.cls.tolist()
        indexes = [
            i for i in range(len(labels)) if labels[i] in [3, 4]
        ]  # 0 = "person"
        boxes = [
            [int(v) for v in box]
            for i, box in enumerate(results.boxes.xyxy.tolist())
            if i in indexes
        ]
        confidences = [
            c for i, c in enumerate(results.boxes.conf.tolist()) if i in indexes
        ]
        labels_txt = [
            results.names[labels[i]] for i in indexes
        ]
        return Detection(
            pred_type=PredictionType.object_detection,
            n_detections=len(boxes),
            boxes=boxes,
            labels=labels_txt,
            confidences=confidences,
        )

    def segment_people(self, image_array: np.ndarray, gun_boxes: list[list[int]], threshold: float = 0.5, max_distance: int = 10):
        prediction = self.seg_model(image_array, conf=threshold)[0]
        labels = prediction.boxes.cls.tolist()
        indexes = [
            i for i in range(len(labels)) if labels[i] == 0
        ]
        boxes = [
            [int(v) for v in box]
            for i, box in enumerate(prediction.boxes.xyxy.tolist())
            if i in indexes
        ]
        segments = [
            np.int32(segment).tolist()
            for i, segment in enumerate(prediction.masks.xy)
            if i in indexes
        ]

        labels = []
        for segment in segments:
            near_gun = match_gun_bbox(segment, gun_boxes, max_distance)
            if near_gun:
                labels.append("danger")
            else:
                labels.append("safe")

        return Segmentation(
            pred_type=PredictionType.segmentation,
            n_detections=len(boxes),
            polygons=segments,
            boxes=boxes,
            labels=labels
        )

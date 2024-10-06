from ultralytics import YOLO
import numpy as np
import cv2
from src.models import Detection, PredictionType, Segmentation, PersonType
from src.config import get_settings
from src.geometry import match_gun_bbox

SETTINGS = get_settings()


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
        ann_color = safe_color if label == PersonType.safe else danger_color
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
                labels.append(PersonType.danger)
            else:
                labels.append(PersonType.safe)

        return Segmentation(
            pred_type=PredictionType.segmentation,
            n_detections=len(boxes),
            polygons=segments,
            boxes=boxes,
            labels=labels
        )

import pytest
import numpy as np
from unittest.mock import MagicMock
from src.predictor import *

@pytest.mark.parametrize(
    "image_array, detection, expected_shape",
    [
        (np.zeros((100, 100, 3), dtype=np.uint8),
         Detection(labels=["gun"], confidences=[0.9], boxes=[
                   [10, 10, 20, 20]], pred_type=PredictionType.object_detection, n_detections=1),
         (100, 100, 3)),  # One detection

        (np.zeros((200, 200, 3), dtype=np.uint8),
         Detection(labels=["knife", "gun"], confidences=[0.8, 0.7], boxes=[[30, 30, 50, 50], [
                   70, 70, 90, 90]], pred_type=PredictionType.object_detection, n_detections=2),
         (200, 200, 3)),  # Multiple detections
    ]
)
def test_annotate_detection(image_array, detection, expected_shape):
    result_img = annotate_detection(image_array, detection)
    assert result_img.shape == expected_shape


@pytest.mark.parametrize(
    "image_array, segmentation, draw_boxes, expected_shape",
    [
        (np.zeros((100, 100, 3), dtype=np.uint8),
         Segmentation(labels=[PersonType.safe], boxes=[
                      [10, 10, 20, 20]], polygons=[[[12, 12], [18, 18]]], pred_type=PredictionType.segmentation, n_detections=1),
         True,
         (100, 100, 3)),  # Safe person with box

        (np.zeros((200, 200, 3), dtype=np.uint8),
         Segmentation(labels=[PersonType.danger], boxes=[
                      [30, 30, 50, 50]], polygons=[[[32, 32], [48, 48]]], pred_type=PredictionType.segmentation, n_detections=1),
         False,
         (200, 200, 3)),  # Dangerous person without box
    ]
)
def test_annotate_segmentation(image_array, segmentation, draw_boxes, expected_shape):
    result_img = annotate_segmentation(image_array, segmentation, draw_boxes)
    assert result_img.shape == expected_shape

@pytest.mark.parametrize(
    "image_array, model_output, expected_detection",
    [
        (np.zeros((100, 100, 3), dtype=np.uint8),
         {"boxes.xyxy": np.array([[10, 10, 20, 20]]), "boxes.cls": np.array([3]),
             "boxes.conf": np.array([0.9]), "names": {3: "gun"}},
         Detection(pred_type=PredictionType.object_detection, n_detections=1, boxes=[
                   [10, 10, 20, 20]], labels=["gun"], confidences=[0.9])
         ),
        (np.zeros((200, 200, 3), dtype=np.uint8),
         {"boxes.xyxy": np.array([[30, 30, 50, 50], [70, 70, 90, 90]]), "boxes.cls": np.array([
             4, 3]), "boxes.conf": np.array([0.8, 0.7]), "names": {3: "gun", 4: "knife"}},
         Detection(pred_type=PredictionType.object_detection, n_detections=2, boxes=[[30, 30, 50, 50], [
                   70, 70, 90, 90]], labels=["knife", "gun"], confidences=[0.8, 0.7])
         ),
    ]
)
def test_detect_guns(mocker, image_array, model_output, expected_detection):
    mock_od_model = MagicMock()
    mock_od_model.return_value = [MagicMock(**model_output)]
    
    gun_detector = GunDetector()
    mocker.patch.object(gun_detector, "od_model", mock_od_model)
    
    detection = gun_detector.detect_guns(image_array)

    assert detection.pred_type == expected_detection.pred_type
    assert detection.n_detections == expected_detection.n_detections
    assert detection.boxes == expected_detection.boxes
    assert detection.labels == expected_detection.labels
    assert detection.confidences == expected_detection.confidences


@pytest.mark.parametrize(
    "image_array, gun_boxes, model_output, expected_segmentation",
    [
        (np.zeros((100, 100, 3), dtype=np.uint8),
         [[10, 10, 20, 20]],
         {"boxes.xyxy": np.array([[15, 15, 25, 25]]), "boxes.cls": np.array([0]),
             "masks.xy":[[[12, 12], [18, 18]]]},
         Segmentation(pred_type=PredictionType.segmentation, n_detections=1, polygons=[
                      [[12, 12], [18, 18]]], boxes=[[15, 15, 25, 25]], labels=[PersonType.danger])
         ),
        (np.zeros((200, 200, 3), dtype=np.uint8),
         [],
         {"boxes.xyxy": np.array([[30, 30, 50, 50]]), "boxes.cls": np.array([0]),
             "masks.xy":[[[32, 32], [48, 48]]]},
         Segmentation(pred_type=PredictionType.segmentation, n_detections=1, polygons=[
                      [[32, 32], [48, 48]]], boxes=[[30, 30, 50, 50]], labels=[PersonType.safe])
         ),
    ]
)
def test_segment_people(mocker, image_array, gun_boxes, model_output, expected_segmentation):
    mock_seg_model = MagicMock()
    mock_seg_model.return_value = [MagicMock(**model_output)]

    gun_detector = GunDetector()
    mocker.patch.object(gun_detector, "seg_model", mock_seg_model)

    segmentation = gun_detector.segment_people(image_array, gun_boxes)

    assert segmentation.pred_type == expected_segmentation.pred_type
    assert segmentation.n_detections == expected_segmentation.n_detections
    assert segmentation.polygons == expected_segmentation.polygons
    assert segmentation.boxes == expected_segmentation.boxes
    assert segmentation.labels == expected_segmentation.labels

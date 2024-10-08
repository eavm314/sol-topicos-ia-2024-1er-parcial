import io
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends
from fastapi.responses import Response
import numpy as np
from functools import cache
from PIL import Image, UnidentifiedImageError
from src.predictor import GunDetector, annotate_detection, annotate_segmentation
from src.config import get_settings
from src.models import Detection, Segmentation, Gun, Person, PixelLocation
from src.geometry import find_area_polygon, find_center_box, find_center_polygon

SETTINGS = get_settings()

app = FastAPI(title=SETTINGS.api_name, version=SETTINGS.revision)


@cache
def get_gun_detector() -> GunDetector:
    print("Creating model...")
    return GunDetector()


def process_uploaded_file(file) -> tuple[Detection, np.ndarray]:
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Not an image"
        )
    # convertir a una imagen de Pillow
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Image format not suported"
        )
    # crear array de numpy
    img_array = np.array(img_obj)
    return img_array


@app.get("/model_info")
def get_model_info(detector: GunDetector = Depends(get_gun_detector)):
    return {
        "model_name": "Gun detector",
        "gun_detector_model": detector.od_model.model.__class__.__name__,
        "semantic_segmentation_model": detector.seg_model.model.__class__.__name__,
        "input_type": "image",
    }


@app.post("/detect_guns")
def detect_guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Detection:
    img_array = process_uploaded_file(file)
    results = detector.detect_guns(img_array, threshold)

    return results


@app.post("/annotate_guns")
def annotate_guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    img_array = process_uploaded_file(file)
    detection = detector.detect_guns(img_array, threshold)
    annotated_img = annotate_detection(img_array, detection)

    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")


@app.post("/detect_people")
def detect_people(
    threshold: float = 0.5,
    max_distance: int = 10,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Segmentation:

    img_array = process_uploaded_file(file)
    guns = detector.detect_guns(img_array, threshold)
    segmentation = detector.segment_people(
        img_array, guns.boxes, threshold, max_distance)
    return segmentation


@app.post("/annotate_people")
def annotate_people(
    threshold: float = 0.5,
    max_distance: int = 10,
    draw_boxes: bool = True,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
):

    img_array = process_uploaded_file(file)
    guns = detector.detect_guns(img_array, threshold)
    segmentation = detector.segment_people(
        img_array, guns.boxes, threshold, max_distance)
    annotated_img = annotate_segmentation(img_array, segmentation, draw_boxes)

    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")


@app.post("/detect")
def detect(
    threshold: float = 0.5,
    max_distance: int = 10,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
):

    img_array = process_uploaded_file(file)
    detection = detector.detect_guns(img_array, threshold)
    segmentation = detector.segment_people(
        img_array, detection.boxes, threshold, max_distance)

    return {"detection": detection, "segmentation": segmentation}


@app.post("/annotate")
def annotate(
    threshold: float = 0.5,
    max_distance: int = 10,
    draw_boxes: bool = True,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
):

    img_array = process_uploaded_file(file)
    guns = detector.detect_guns(img_array, threshold)
    segmentation = detector.segment_people(
        img_array, guns.boxes, threshold, max_distance)
    annotated_img = annotate_segmentation(img_array, segmentation, draw_boxes)
    annotated_guns = annotate_detection(img_array, guns)
    annotated_img = annotate_segmentation(
        annotated_guns, segmentation, draw_boxes)
    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")


@app.post("/guns")
def guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> list[Gun]:

    img_array = process_uploaded_file(file)
    detection = detector.detect_guns(img_array, threshold)

    response = []
    for box, label in zip(detection.boxes, detection.labels):
        center_x, center_y = find_center_box(box)
        response.append(
            Gun(
                gun_type=label.lower(),
                location=PixelLocation(x=center_x, y=center_y)
            )
        )

    return response


@app.post("/people")
def people(
    threshold: float = 0.5,
    max_distance: int = 10,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> list[Person]:

    img_array = process_uploaded_file(file)
    guns = detector.detect_guns(img_array, threshold)
    segmentation = detector.segment_people(
        img_array, guns.boxes, threshold, max_distance)

    response = []
    for polygon, label in zip(segmentation.polygons, segmentation.labels):
        center_x, center_y = find_center_polygon(polygon)

        response.append(
            Person(
                person_type=label,
                location=PixelLocation(x=center_x, y=center_y),
                area=find_area_polygon(polygon)
            )
        )

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", port=8080, host="0.0.0.0")

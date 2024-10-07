import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
from src.main import app
from src.models import Detection, Segmentation, Gun, Person

client = TestClient(app)

def check_structure(model, data):
    try:
        model.model_validate(data)
        return True
    except:
        return False

@pytest.fixture
def mocked_image():
    image_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image_pil = Image.fromarray(image_array)
    img_byte_arr = io.BytesIO()
    image_pil.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    return img_byte_arr

def test_get_model_info():
    response = client.get("/model_info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "gun_detector_model" in data
    assert "semantic_segmentation_model" in data
    assert "input_type" in data

def test_post_detect_guns(mocked_image):
    files = {'file': ("test_image.jpg", mocked_image, "image/jpeg")}
    response = client.post("/detect_guns", files=files, data={'threshold': '0.5'})
    assert response.status_code == 200
    data = response.json()
    assert check_structure(Detection, data)

def test_post_annotate_guns(mocked_image):
    files = {'file': ("test_image.jpg", mocked_image, "image/jpeg")}
    response = client.post("/annotate_guns", files=files, data={'threshold': '0.5'})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"

def test_post_detect_people(mocked_image):
    files = {'file': ("test_image.jpg", mocked_image, "image/jpeg")}
    response = client.post("/detect_people", files=files, data={'threshold': '0.5'})
    assert response.status_code == 200
    data = response.json()
    assert check_structure(Segmentation, data)

def test_post_annotate_people(mocked_image):
    files = {'file': ("test_image.jpg", mocked_image, "image/jpeg")}
    response = client.post("/annotate_people", files=files, data={'threshold': '0.5', 'annotate': 'true'})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"

def test_post_detect(mocked_image):
    files = {'file': ("test_image.jpg", mocked_image, "image/jpeg")}
    response = client.post("/detect", files=files, data={'threshold': '0.5'})
    assert response.status_code == 200
    data = response.json()
    assert check_structure(Detection, data["detection"])
    assert check_structure(Segmentation, data["segmentation"])

def test_post_annotate(mocked_image):
    files = {'file': ("test_image.jpg", mocked_image, "image/jpeg")}
    response = client.post("/annotate", files=files, data={'threshold': '0.5', 'annotate': 'true'})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"

def test_post_guns(mocked_image):
    files = {'file': ("test_image.jpg", mocked_image, "image/jpeg")}
    response = client.post("/guns", files=files, data={'threshold': '0.5'})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if len(data)>0:
        assert check_structure(Gun, data[0])

def test_post_people(mocked_image):
    files = {'file': ("test_image.jpg", mocked_image, "image/jpeg")}
    response = client.post("/people", files=files, data={'threshold': '0.5'})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if len(data)>0:
        assert check_structure(Person, data[0])
import pytest
from app import app, extract_features
import os
import cv2
import numpy as np
from io import BytesIO

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Leaf' in response.data

def test_file_upload(client):
    # Create a dummy image
    img = np.zeros((100, 100, 3), np.uint8)
    _, buffer = cv2.imencode('.jpg', img)
    data = {
        'leaf_image': (BytesIO(buffer), 'test.jpg')
    }
    
    response = client.post(
        '/',
        data=data,
        content_type='multipart/form-data'
    )
    assert response.status_code == 200

def test_extract_features():
    # Create test image
    img = np.zeros((128, 128, 3), np.uint8)
    cv2.imwrite('test_img.jpg', img)
    
    features = extract_features('test_img.jpg')
    assert len(features) == 512  # 8x8x8 histogram
    
    # Clean up
    os.remove('test_img.jpg')

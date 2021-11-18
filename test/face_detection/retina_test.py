import sys
sys.path.append('../..')
sys.path.append('..')
import pytest
import cv2

from api.face_detection.face_detection_retina import RetinaFaceSingleton

def test_run():
    model_pack_name = 'buffalo_m'
    root = '../data/model'
    # app = FaceAnalysis(root = root, name = model_pack_name, allowed_modules=['detection', 'recognition'])
    # app.prepare(ctx_id=0, det_size=(640, 640))
    app = RetinaFaceSingleton(root, model_pack_name)
    app.prepare(ctx_id=0, det_size=(640, 640))

    img = cv2.imread('../data/input/f_actor_1.jpeg')
    assert img is not None
    faces = app.get(img)

    print(len(faces))
    # Easy case - Need have exactly 6 faces.
    assert len(faces) == 6

    # Need capture full (13 faces) or more faces.
    img = cv2.imread('../data/input/test2.jpg')
    assert img is not None
    faces = app.get(img)
    assert len(faces) >= 13

    # Need capture at least 20 faces
    img = cv2.imread('../data/input/test3.jpg')
    assert img is not None
    faces = app.get(img)
    assert len(faces) >= 20

    # Need capture exactly 2 faces
    img = cv2.imread('../data/input/test4.jpg')
    assert img is not None
    faces = app.get(img)
    assert len(faces) == 2

    # Need capture exactly 26 faces
    img = cv2.imread('../data/input/test5.jpg')
    assert img is not None
    faces = app.get(img)
    assert len(faces) == 26

    # Need capture exactly 10 faces
    img = cv2.imread('../data/input/test6.jpg')
    assert img is not None
    faces = app.get(img)
    assert len(faces) == 10

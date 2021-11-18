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
    faces = app.get(img)

    print(len(faces))
    assert len(faces) == 6
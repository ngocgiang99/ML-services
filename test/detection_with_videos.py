import sys
sys.path.append('..')
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from api.face_detection.face_detection_retina import RetinaFaceSingleton
from datetime import datetime
import os

model_pack_name = 'buffalo_m'
root = '../data/model'
# app = FaceAnalysis(root = root, name = model_pack_name, allowed_modules=['detection', 'recognition'])
# app.prepare(ctx_id=0, det_size=(640, 640))
app = RetinaFaceSingleton(root, model_pack_name)
app.prepare(ctx_id=0, det_size=(640, 640))


video_dir = '../data/videos'
saved_dir = '../data/outputs'
video_name = 'mixkit-informal-economy-in-istanbul-4463.mp4'

cap = cv2.VideoCapture(os.path.join(video_dir, video_name))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
fps = 10
out = cv2.VideoWriter(os.path.join(saved_dir, model_pack_name + "_" + video_name), fourcc, fps, (640,640))

while True:
    ret, frame = cap.read()
    if frame is None:
        break
    # cv2.imshow('camera', frame)
    # cv2.waitKey(0)
    frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_CUBIC)

    ta = datetime.now()
    faces = app.get(frame)
    tb = datetime.now()
    print('Inference time: ', (tb - ta).total_seconds() * 1000, 'ms')
    rimg = app.draw_on(frame, faces)

    # cv2.imshow('camera', frame)
    # cv2.waitKey(0)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    out.write(rimg)


    
cap.release()
out.release()
# cv2.destroyAllWindows()
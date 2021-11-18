import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from datetime import datetime

model_pack_name = 'buffalo_sc'
root = '../data/model'
app = FaceAnalysis(root = root, name = model_pack_name, allowed_modules=['detection'])
app.prepare(ctx_id=0, det_size=(640, 640))

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

while True:
    ret, frame = cap.read()

    cv2.imshow('camera', frame)
    cv2.waitKey(0)

    ta = datetime.now()
    faces = app.get(frame)
    tb = datetime.now()
    print('Inference time: ', (tb - ta).total_seconds() * 1000, 'ms')
    rimg = app.draw_on(frame, faces)

    cv2.imshow('camera', frame)
    cv2.waitKey(0)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    
cap.release()
cv2.destroyAllWindows()
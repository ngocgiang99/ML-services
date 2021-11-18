import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from datetime import datetime

model_pack_name = 'buffalo_sc'
root = 'data/model'
app = FaceAnalysis(root = root, name = model_pack_name, allowed_modules=['detection'])
app.prepare(ctx_id=0, det_size=(640, 640))
# img = ins_get_image('t1')
# img_name = 'data/input/f_actor_1.jpeg'
img_name = 'data/input/test5.jpg'
img = cv2.imread(img_name)
print(type(img))
print(img.shape)
while True:
    ta = datetime.now()
    faces = app.get(img)
    tb = datetime.now()
    print((tb - ta).total_seconds() * 1000)
print(faces[0].keys())
rimg = app.draw_on(img, faces)
cv2.imwrite("data/t2_output.jpg", rimg)
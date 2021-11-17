import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=0, det_size=(640, 640))
# img = ins_get_image('t1')
img_name = 'f_actor_1.jpeg'
img = cv2.imread(img_name)
print(type(img))
print(img.shape)
faces = app.get(img)
print(faces[0].keys())
rimg = app.draw_on(img, faces)
cv2.imwrite("./t2_output.jpg", rimg)
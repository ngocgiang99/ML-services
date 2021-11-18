import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from datetime import datetime
import os
from ..utilities.singleton_meta import SingletonMeta


class RetinaFaceSingleton(metaclass=SingletonMeta):

    def __init__(self, root: str, model_name : str) -> None:
        self.app = FaceAnalysis(root = root, name = model_name, allowed_modules=['detection'])
        
    def get(self, img, max_num=0):
        return self.app.get(img, max_num)

    def draw_on(self, img, faces):
        return self.app.draw_on(img, faces)

    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        return self.app.prepare(ctx_id, det_thresh, det_size)
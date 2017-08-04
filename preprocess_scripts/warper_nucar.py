import cv2
import numpy as np


class Warper:
    def __init__(self):
        src = np.float32([
            [958, 1384],
            [1219, 1377],
            [1817, 1750],
            [431, 1818],
        ])

        dst = np.float32([
            [431, 0],
            [1817, 0],
            [1817, 2448],
            [431, 2448],
        ])

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

    def warp(self, img):
        return cv2.warpPerspective(
            img,
            self.M,
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR
        )

    def unwarp(self, img):
        return cv2.warpPersective(
            img,
            self.Minv,
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR
        )
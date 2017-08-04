import cv2
import numpy as np


class Warper:
    def __init__(self):
        src = np.float32([
            # [580, 425],
            # [700, 425],
            # [1040, 650],
            # [260, 650],
            [380, 425],
            [1000, 425],
            [1040, 650],
            [260, 650],
        ])
        dst = np.float32([
            [260, 0],
            [1040, 0],
            [1040, 720],
            [260, 720],
        ])
        lsrc = np.float32([
            [600, 425], #top
            [720, 425],
            [740, 650], #bot
            [100, 650],
        ])
        lsrc2 = np.float32([
            [660, 425],
            [780, 425],
            [1320, 650],
            [540, 650],
        ])

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
        self.LM = cv2.getPerspectiveTransform(lsrc, dst)
        self.LMinv = cv2.getPerspectiveTransform(dst, lsrc)
        self.LM2 = cv2.getPerspectiveTransform(lsrc2, dst)
        self.LMinv2 = cv2.getPerspectiveTransform(dst, lsrc2)

    def warp(self, img):
        return cv2.warpPerspective(
            img,
            self.M,
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR
        )
    def warpleft(self, img):
        return cv2.warpPerspective(
            img,
            self.LM,
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR
        )
    def warpleft2(self, img):
        return cv2.warpPerspective(
            img,
            self.LM2,
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR
        )
    def warpright(self, img):
        return cv2.warpPerspective(
            img,
            self.RM,
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
    def unwarpleft(self, img):
        return cv2.warpPersective(
            img,
            self.LMinv,
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR
        )
    def unwarpright(self, img):
        return cv2.warpPersective(
            img,
            self.RMinv,
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR
        )
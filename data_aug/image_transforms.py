import random

import PIL
import cv2
import dlib
import numpy as np
import torch
import torchvision.transforms.functional
from PIL import ImageDraw, ImageStat

from . import functional as F
# from torchvision.transforms.functional import pad

class SquarePad(object):
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s + pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = [p_left, p_top, p_right, p_bottom]
        return torchvision.transforms.functional.pad(image, padding, 0, 'constant')

class TensorResize(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        return torch.nn.functional.interpolate(img, size=self.size)

class imgRandomLandmarkMask(object):
    """Mask the components of given faces randomly with a given probability.
    Args:
        p (float): probability of the image being masked. Default value is 0.8
    """

    def __init__(self, p=0.8):
        self.p = p
        self.detector = dlib.get_frontal_face_detector()
        cnn_detector_path = "/mnt/d/Data/Yuxuan/data/model/mmod_human_face_detector.dat"
        face_predictor_path = "/mnt/d/Data/Yuxuan/data/model/shape_predictor_68_face_landmarks.dat"
        self.cnn_detector = dlib.cnn_face_detection_model_v1(cnn_detector_path)
        self.face_predictor = dlib.shape_predictor(face_predictor_path)
        self.mode_selection = random.choice(["mouth","left_eye","right_eye"])

    def __call__(self, img):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Randomly masked clip

        MOUTH_POINTS=list(range(48,68))
        LEFT_EYE=list(range(36,42))
        RIGHT_EYE=list(range(42,48))
        """

        if random.random() < self.p:
            if self.mode_selection == "left_eye":
                start_id = 36
                end_id = 42
            elif self.mode_selection == "right_eye":
                start_id = 42
                end_id = 48
            else:
                start_id = 48
                end_id = 68

            if isinstance(img, np.ndarray):
                # image open with cv2

                image = img
                landmark = F.detect_landmark(image, self.detector, self.cnn_detector, self.face_predictor)
                if landmark is None:
                    return img
                else:
                    landmark = landmark[start_id:end_id, :]
                    minx = min(landmark[:, 0])
                    maxx = max(landmark[:, 0])
                    miny = min(landmark[:, 1])
                    maxy = max(landmark[:, 1])
                    offset = 10

                    avg_img = self.mean_img(image)
                    # blurred_img = cv2.GaussianBlur(image, (21, 21), 0)
                    # black_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float64)
                    mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float64)
                    centre = (int(np.round((minx + maxx) / 2)), int(np.round((miny + maxy) / 2)))
                    radius = int(np.round((maxx - minx) / 2)+offset)
                    mask = cv2.circle(mask, centre, radius, (255, 255, 255), -1)
                    img = np.where(mask == (0, 0, 0), image, avg_img)
                return np.array(img)
            elif isinstance(img, PIL.Image.Image):
                # image open with PIL
                # Because the transform process include normalization
                image = img
                landmark = F.detect_landmark(image, self.detector, self.cnn_detector, self.face_predictor)
                if landmark is None:
                    return img
                else:
                    landmark = landmark[start_id:end_id, :]
                    minx = min(landmark[:, 0])
                    maxx = max(landmark[:, 0])
                    miny = min(landmark[:, 1])
                    maxy = max(landmark[:, 1])

                    draw = ImageDraw.Draw(image)
                    stat = ImageStat.Stat(image)
                    means = stat.mean
                    offset = 10
                    leftUpPoint = (minx - offset, miny - offset)
                    rightDownPoint = (maxx + offset, maxy + offset)
                    twoPointList = [leftUpPoint, rightDownPoint]
                    try:
                        draw.ellipse(twoPointList, fill=(int(means[0]), int(means[1]), int(means[2]), 0))
                    except:
                        pass

            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image or torch.Tensor' +
                                ' but got list of {0}'.format(type(img)))
        return img

    @staticmethod
    def mean_img(img):
        img_shape = img.shape
        mean_rgb_image = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.float64)
        means, dev = cv2.meanStdDev(img)
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                mean_rgb_image[i, j][0] = means[0]
                mean_rgb_image[i, j][1] = means[1]
                mean_rgb_image[i, j][2] = means[2]
        return mean_rgb_image

    @staticmethod
    def mean_rgb(img):
        if isinstance(img, np.ndarray):
            means, dev = cv2.meanStdDev(img)
        elif isinstance(img, PIL.Image.Image):
            stat = ImageStat.Stat(img)
            means = stat.mean
        return means

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


# class RandomColorJitter(onject):


import numbers
import random
import warnings

import cv2
import dlib
import numpy as np
import PIL
import skimage.transform
import torchvision
import math
import torch
from PIL import ImageDraw, ImageStat
# from cv2 import GaussianBlur as cv_GB
from torchvision import transforms

from . import functional as F
from .gaussian_blur import GaussianBlur

# Thanks to : https://github.com/hassony2/torch_videovision

# cv2.setNumThreads(0)

class Compose(object):
    """Composes several transforms
    Args:
    transforms (list of ``Transform`` objects): list of transforms
    to compose
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip


class ToNumpy(object):
    def __call__(self, clip):
        clip = clip.permute(1, 2, 3, 0) # CTHW 2 THWC
        return clip.detach().numpy()

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

# TODO: add background mask to augment the data
class RandomLandmarkMask(object):
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

    def __call__(self, clip):
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

            if isinstance(clip[0], np.ndarray):
                # image open with cv2
                for i in range(len(clip)):
                    image = clip[i]
                    landmark = F.detect_landmark(image, self.detector, self.cnn_detector, self.face_predictor)
                    if landmark is None:
                        continue
                    else:
                        landmark = landmark[start_id:end_id, :]
                        minx = min(landmark[:, 0])
                        maxx = max(landmark[:, 0])
                        miny = min(landmark[:, 1])
                        maxy = max(landmark[:, 1])
                        offset = 10

                        avg_img = self.mean_rgb(image)
                        # blurred_img = cv2.GaussianBlur(image, (21, 21), 0)
                        # black_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float64)
                        mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float64)
                        centre = (int(np.round((minx + maxx) / 2)), int(np.round((miny + maxy) / 2)))
                        radius = int(np.round((maxx - minx) / 2)+offset)
                        mask = cv2.circle(mask, centre, radius, (255, 255, 255), -1)
                        clip[i] = np.where(mask == (0, 0, 0), image, avg_img)
                return np.array(clip)
            elif isinstance(clip[0], PIL.Image.Image):
                # image open with PIL
                # TODO: change black mask to avg mask in PIL
                # Because the transform process include normalization
                for image in clip:
                    landmark = F.detect_landmark(image, self.detector, self.cnn_detector, self.face_predictor)
                    if landmark is None:
                        continue
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
                                ' but got list of {0}'.format(type(clip[0])))
        return clip

    @staticmethod
    def mean_rgb(img):
        img_shape = img.shape
        mean_rgb_image = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.float64)
        means, dev = cv2.meanStdDev(img)
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                mean_rgb_image[i, j][0] = means[0]
                mean_rgb_image[i, j][1] = means[1]
                mean_rgb_image[i, j][2] = means[2]
        return mean_rgb_image

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomGaussianBlur(object):
    def __init__(self, p=0.5):
        self.p = p
        self.blur = GaussianBlur(3)

    def __call__(self, clip):
        if random.random() < self.p:
            if isinstance(clip[0], np.ndarray):
                return [cv2.GaussianBlur(img,(3,3),0) for img in clip]
            elif isinstance(clip[0], PIL.Image.Image):
                return [
                    self.blur(img) for img in clip
                ]
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                ' but got list of {0}'.format(type(clip[0])))
        return clip


class RandomHorizontalFlip(object):
    """Horizontally flip the list of given images randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Randomly flipped clip
        """
        if random.random() < self.p:
            if isinstance(clip[0], np.ndarray):
                return [np.fliplr(img) for img in clip]
            elif isinstance(clip[0], PIL.Image.Image):
                return [
                    img.transpose(PIL.Image.FLIP_LEFT_RIGHT) for img in clip
                ]
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                ' but got list of {0}'.format(type(clip[0])))
        return clip

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):
    """Vertically flip the list of given images randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip):
        """
        Args:
            img (PIL.Image or numpy.ndarray): List of images to be flipped
            in format (h, w, c) in numpy.ndarray
        Returns:
            PIL.Image or numpy.ndarray: Randomly flipped clip
        """
        if random.random() < self.p:
            if isinstance(clip[0], np.ndarray):
                return [np.flipud(img) for img in clip]
            elif isinstance(clip[0], PIL.Image.Image):
                return [
                    img.transpose(PIL.Image.FLIP_TOP_BOTTOM) for img in clip
                ]
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                ' but got list of {0}'.format(type(clip[0])))
        return clip

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomGrayscale(object):
    """Randomly convert image to grayscale with a probability of p (default 0.2).
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading
    dimensions
    Args:
        p (float): probability that image should be converted to grayscale.
    Returns:
        PIL Image or Tensor: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b
    """
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p
    def __call__(self,clip):
        """
        Args:
            list of imgs (PIL Image or Tensor): Image to be converted to grayscale.
        Returns:
            PIL Image or Tensor: Randomly grayscaled image.
        """
        if random.random() < self.p:
            if isinstance(clip[0], np.ndarray):
                return [F.np_to_grayscale(img) for img in clip]
            elif isinstance(clip[0], PIL.Image.Image):
                num_output_channels = 1 if clip[0].mode == 'L' else 3
                for i in range(len(clip)):
                    clip[i] = F.to_grayscale(clip[i], num_output_channels)
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                ' but got list of {0}'.format(type(clip[0])))

        return clip


class I3DPixelsValue(object):
    """
    Scale the pixel value between -1 and 1 instead of 0 and 1 (required for I3D)
    """

    def __call__(self, sample):
        try:
            sample * 2 - 1
        except:
            pass
        return sample * 2 - 1


class ChangeVideoShape(object):
    """
    Expect to receive a ndarray of chape (Time, Height, Width, Channel) which is the default format
    of cv2 or PIL. Change the shape of the ndarray to TCHW or CTHW.
    """

    def __init__(self, shape: str):
        """
        shape : a string with the value "CTHW" or "TCHW".
        """

        self.shape = shape

    def __call__(self, sample):
        if isinstance(sample[0], torch.Tensor):
            if self.shape == "CTHW":
                sample = sample.permute(3, 0, 1, 2)
            elif self.shape == "TCHW":
                sample = sample.permute(0, 3, 1, 2)
            else:
                raise ValueError(f"Received {self.shape}. Expecting TCHW or CTHW.")
        else:
            if self.shape == "CTHW": # THWC 2 CTHW
                sample = np.transpose(sample, (3, 0, 1, 2))
            elif self.shape == "TCHW": # THWC 2 TCHW
                sample = np.transpose(sample, (0, 3, 1, 2))
            else:
                raise ValueError(f"Received {self.shape}. Expecting TCHW or CTHW.")

        return sample


class RandomResize(object):
    """Resizes a list of (H x W x C) numpy.ndarray to the final size
    The larger the original image is, the more times it takes to
    interpolate
    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (tuple): (widht, height)
    """

    def __init__(self, ratio=(3. / 4., 4. / 3.), interpolation='nearest'):
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, clip):
        scaling_factor = random.uniform(self.ratio[0], self.ratio[1])

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_w, new_h)
        resized = F.resize_clip(
            clip, new_size, interpolation=self.interpolation)
        return resized


class ResizeVideo(object):
    """Resizes a list of (H x W x C) numpy.ndarray to the final size
    The larger the original image is, the more times it takes to
    interpolate
    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (tuple): (width, height)
    """

    def __init__(self, size, interpolation="nearest"):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, clip):
        resized = F.resize_clip(clip, self.size, interpolation=self.interpolation)
        return np.array(resized)


class RandomCropVideo(object):
    """Extract random crop at the same location for a list of images
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError(
                "Expected numpy.ndarray or PIL.Image"
                + "but got list of {0}".format(type(clip[0]))
            )
        if w > im_w or h > im_h:
            error_msg = (
                "Initial image size should be larger then "
                "cropped size but got cropped sizes : ({w}, {h}) while "
                "initial image is ({im_w}, {im_h})".format(
                    im_w=im_w, im_h=im_h, w=w, h=h
                )
            )
            raise ValueError(error_msg)

        x1 = random.randint(0, im_w - w)
        y1 = random.randint(0, im_h - h)
        cropped = F.crop_clip(clip, y1, x1, h, w)

        return np.array(cropped)


class CenterCropVideo(object):
    """Extract center crop at the same location for a list of images
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        numpy.ndarray: Cropped list of images of shape (t, h, w, c)
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError(
                "Expected numpy.ndarray or PIL.Image"
                + "but got list of {0}".format(type(clip[0]))
            )
        if w > im_w or h > im_h:
            error_msg = (
                "Initial image size should be larger then "
                "cropped size but got cropped sizes : ({w}, {h}) while "
                "initial image is ({im_w}, {im_h})".format(
                    im_w=im_w, im_h=im_h, w=w, h=h
                )
            )
            raise ValueError(error_msg)

        x1 = int(round((im_w - w) / 2.0))
        y1 = int(round((im_h - h) / 2.0))
        cropped = F.crop_clip(clip, y1, x1, h, w)

        return np.array(cropped)


class RandomRotation(object):
    """Rotate entire clip randomly by a random angle within
    given bounds
    Args:
    degrees (sequence or int): Range of degrees to select from
    If degrees is a number instead of sequence like (min, max),
    the range of degrees, will be (-degrees, +degrees).
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,'
                                 'it must be of len 2.')

        self.degrees = degrees

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(clip[0], np.ndarray):
            rotated = [skimage.transform.rotate(img, angle) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            rotated = [img.rotate(angle) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        return rotated


class TrimVideo(object):
    """Trim each video the same way. Waiting shape TCHW
    """

    def __init__(self, size, offset=None):
        self.end = size
        self.begin = 0

        if offset != None:
            self.begin = offset
            self.end += offset

    def __call__(self, clip):
        resized = clip

        if len(clip) > self.end:
            resized = clip[self.begin: self.end]
        return np.array(resized)


class RandomTrimVideo(object):
    """Trim randomly the video. Waiting shape TCHW
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, clip):
        resized = clip

        if len(clip) > self.size:
            diff = len(resized) - self.size

            start = random.randint(0, diff)
            end = start + self.size

            resized = resized[start:end]

        return np.array(resized)


class PadVideo(object):
    def __init__(self, size, loop=True):
        self.size = size
        self.loop = loop

    def __call__(self, clip):
        if self.loop:
            resized = self._loop_sequence(clip, self.size)
        else:
            resized = self._pad_sequence(clip, self.size)

        return np.array(resized)

    def _pad_sequence(self, sequence, length):
        shape = sequence.shape
        new_shape = (length, shape[1], shape[2], shape[3])

        zero_arr = np.zeros(new_shape)
        zero_arr[: shape[0]] = sequence

        return zero_arr

    def _loop_sequence(self, sequence, length):
        shape = sequence.shape
        new_shape = (length, shape[1], shape[2], shape[3])
        zero_arr = np.zeros(new_shape)

        video_len = len(sequence)

        for i in range(length):
            vid_idx = i % video_len
            zero_arr[i] = sequence[vid_idx]

        return zero_arr


class RandomColorJitterVideo(object):
    """Randomly change the brightness, contrast and saturation and hue of the clip
    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, p=0.5, brightness=0, contrast=0, saturation=0, hue=0):
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        """
        Args:
        clip (list): list of PIL.Image
        Returns:
        list PIL.Image : list of transformed PIL.Image
        """
        if random.random() < self.p:
            brightness, contrast, saturation, hue = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)

            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(lambda img: transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(lambda img: transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(lambda img: transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(lambda img: transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)

            # Apply to all images
            jittered_clip = []
            for img in clip:
                # Transforming frame
                frame = img
                pillow_frame = transforms.ToPILImage()(np.uint8(frame))

                for func in img_transforms:
                    jittered_img = func(pillow_frame)
                jittered_clip.append(np.array(jittered_img))
            clip = jittered_clip
        return clip


class ToTensor(object):
    def __call__(self, clip):
        return torch.from_numpy(np.array(clip))
    def __repr__(self):
        return self.__class__.__name__ + '()'

class Normalize(torchvision.transforms.Normalize):
    """
    Normalize the (CTHW) video clip by mean subtraction and division by standard deviation
    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        vid = x.permute(1, 0, 2, 3)  # C T H W to T C H W
        default_float_dtype = torch.get_default_dtype()
        vid = vid.to(dtype=default_float_dtype).div(255)
        vid = super().forward(vid)
        vid = vid.permute(1, 0, 2, 3)  # T C H W to C T H W
        return vid


# class Normalize(object):
#     """Normalize a clip with mean and standard deviation.
#     Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
#     will normalize each channel of the input ``torch.*Tensor`` i.e.
#     ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
#     .. note::
#         This transform acts out of place, i.e., it does not mutates the input tensor.
#     Args:
#         mean (sequence): Sequence of means for each channel.
#         std (sequence): Sequence of standard deviations for each channel.
#     """
#
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std
#
#     def __call__(self, clip):
#         """
#         Args:
#             clip (Tensor): Tensor clip of size (T, C, H, W) to be normalized.
#         Returns:
#             Tensor: Normalized Tensor clip.
#         """
#         # TODO: check the value of std and mean for vox1 & vox2
#         default_float_dtype = torch.get_default_dtype()
#         clip = clip.to(dtype=default_float_dtype).div(255)
#         return F.normalize(clip, self.mean, self.std)
#
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


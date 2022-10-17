import numbers

import dlib
import torch
import cv2
import numpy as np
import PIL
from PIL import Image

detector = dlib.get_frontal_face_detector()
cnn_detector = "/mnt/d/Data/Yuxuan/data/model/mmod_human_face_detector.dat"
face_predictor = "/mnt/d/Data/Yuxuan/data/model/shape_predictor_68_face_landmarks.dat"

def _is_tensor_clip(clip):
    return torch.is_tensor(clip) and clip.ndimension() == 4


def crop_clip(clip, min_h, min_w, h, w):
    if isinstance(clip[0], np.ndarray):
        cropped = [img[min_h:min_h + h, min_w:min_w + w, :] for img in clip]

    elif isinstance(clip[0], PIL.Image.Image):
        cropped = [
            img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip
        ]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return cropped


def to_grayscale(img, num_output_channels=1):
    """Convert image to grayscale version of image.
    Args:
        img (PIL Image): Image to be converted to grayscale.
    Returns:
        PIL Image: Grayscale version of the image.
            if num_output_channels = 1 : returned image is single channel
            if num_output_channels = 3 : returned image is 3 channel with r = g = b
    """
    if not isinstance(img,PIL.Image.Image):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if num_output_channels == 1:
        img = img.convert('L')
    elif num_output_channels == 3:
        img = img.convert('L')
        np_img = np.array(img, dtype=np.uint8)
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, 'RGB')
    else:
        raise ValueError('num_output_channels should be either 1 or 3')

    return img

def np_to_grayscale(img):
    np_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    img = np.dstack([np_img, np_img, np_img])
    return img

def resize_clip(clip, size, interpolation='bilinear'):
    if isinstance(clip[0], np.ndarray):
        if isinstance(size, numbers.Number):
            im_h, im_w, im_c = clip[0].shape
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            np_inter = cv2.INTER_LINEAR
        else:
            np_inter = cv2.INTER_NEAREST
        scaled = [
            cv2.resize(img, size, interpolation=np_inter) for img in clip
        ]
    elif isinstance(clip[0], PIL.Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            pil_inter = PIL.Image.NEAREST
        else:
            pil_inter = PIL.Image.BILINEAR
        scaled = [img.resize(size, pil_inter) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return scaled


def get_resize_sizes(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow


def normalize(tensor, mean, std, inplace=False):
    """Normalize a tensor image with mean and standard deviation.
        This transform does not support PIL Image.

        .. note::
            This transform acts out of place by default, i.e., it does not mutates the input tensor.

        See :class:`~torchvision.transforms.Normalize` for more details.

        Args:
            tensor (Tensor): Tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channel.
            inplace(bool,optional): Bool to make this operation inplace.

        Returns:
            Tensor: Normalized Tensor image.
        """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

    if tensor.ndim < 3:
        raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
                         '{}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.sub_(mean).div_(std)
    return tensor


def detect_landmark(image, detector, cnn_detector, predictor):
    if isinstance(image, PIL.Image.Image):
        image = np.array(image)
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    # gray = np.array(ImageOps.grayscale(image))
    rects = detector(gray, 1)
    if len(rects) == 0:
        rects = cnn_detector(gray)
        rects = [d.rect for d in rects]
    coords = None
    for (_, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def image_mask_image(image_file, face_predictor_path, cnn_detector_path, crop_part):
    detector = dlib.get_frontal_face_detector()
    cnn_detector = dlib.cnn_face_detection_model_v1(cnn_detector_path)
    predictor = dlib.shape_predictor(face_predictor_path)

    image = cv2.imread(image_file)
    # image = Image.open(image_file).convert("RGB")
    # image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    landmark = detect_landmark(image, detector, cnn_detector, predictor)
    if crop_part == "left_eye":
        landmark = landmark[36:42, :]
    elif crop_part == "right_eye":
        landmark = landmark[42:48, :]
    elif crop_part == "mouth":
        landmark = landmark[48:68, :]
    else:
        landmark = landmark

    minx = min(landmark[:, 0])
    maxx = max(landmark[:, 0])
    miny = min(landmark[:, 1])
    maxy = max(landmark[:, 1])
    # blurred_img = cv2.GaussianBlur(image, (FEATHER, FEATHER), 0)
    black_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float64)
    mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float64)
    centre = (int(np.round((minx+maxx)/2)), int(np.round((miny+maxy)/2)))
    radius = int(np.round((maxy-miny)/2+20))
    mask = cv2.circle(mask, centre, radius, (255, 255, 255), -1)
    out = np.where(mask==(0, 0, 0), image, black_img)
    # cv2.imwrite("/mnt/c/Data/Yuxuan/VoxCeleb/test/cp_emma.jpg", out)
    return out
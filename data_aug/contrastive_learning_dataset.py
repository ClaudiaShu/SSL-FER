
from torchvision import transforms as cv_tf
from torchaudio import transforms as au_tf
from data_aug import video_transforms as vd_tf
from data_aug import audio_transforms as ad_tf

from torchvision import datasets

from data_aug.image_transforms import imgRandomLandmarkMask, SquarePad
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from audiomentations import (
    Compose,
    AddBackgroundNoise,
    AddGaussianNoise,
    AddGaussianSNR,
    TimeStretch,
    PitchShift,
    Shift
)

from dataset.VoxCeleb_config import configs

MUSAN = "/mnt/d/Data/Yuxuan/VoxCeleb/musan"
RIR = "/mnt/d/Data/Yuxuan/VoxCeleb/rirs_noises"
ESC = '/mnt/d/Data/Yuxuan/VoxCeleb/ESC-50'


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        data_transforms = cv_tf.Compose(
            [
                imgRandomLandmarkMask(),
                cv_tf.Resize(size=(128, 128)),
                cv_tf.RandomCrop(112),
                cv_tf.RandomHorizontalFlip(p=0.5),
                # transforms.RandomApply
                cv_tf.RandomApply([cv_tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)], p=0.8),
                # cv_tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                cv_tf.RandomApply([cv_tf.GaussianBlur(3)], p=0.5),
                # cv_tf.GaussianBlur(3),
                cv_tf.RandomGrayscale(p=0.2),
                cv_tf.ToTensor(),
                cv_tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        )
        return data_transforms

    @staticmethod
    def get_crop_transform():
        transform = cv_tf.Compose([
            SquarePad(),
            cv_tf.ToTensor(),
            cv_tf.Resize(size=(64, 64)),
            cv_tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        return transform

    @staticmethod
    def get_image_transform():
        img1_transform = cv_tf.Compose(
            [
                imgRandomLandmarkMask(),
                cv_tf.Resize(size=(128, 128)),
                cv_tf.RandomCrop(112),
                cv_tf.RandomHorizontalFlip(p=0.5),
                # transforms.RandomApply
                # cv_tf.RandomApply([cv_tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)], p=0.8),
                cv_tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                cv_tf.RandomApply([cv_tf.GaussianBlur(3)], p=0.5),
                # cv_tf.GaussianBlur(3),
                cv_tf.RandomGrayscale(p=0.2),
                cv_tf.ToTensor(),
                cv_tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        )
        img2_transform = cv_tf.Compose(
            [
                imgRandomLandmarkMask(),
                cv_tf.Resize(size=(128, 128)),
                cv_tf.RandomCrop(112),
                cv_tf.RandomHorizontalFlip(p=0.5),
                # transforms.RandomApply
                cv_tf.RandomApply([cv_tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)], p=0.8),
                # cv_tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                cv_tf.RandomApply([cv_tf.GaussianBlur(3)], p=0.5),
                # cv_tf.GaussianBlur(3),
                cv_tf.RandomGrayscale(p=0.2),
                cv_tf.ToTensor(),
                cv_tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        )
        return img1_transform, img2_transform

    @staticmethod
    def get_cvimage_transform():
        img1_transform = cv_tf.Compose(
            [
                cv_tf.ToPILImage(),
                imgRandomLandmarkMask(),
                cv_tf.Resize(size=(128, 128)),
                cv_tf.RandomCrop(112),
                cv_tf.RandomHorizontalFlip(p=0.5),
                # transforms.RandomApply
                # cv_tf.RandomApply([cv_tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)], p=0.8),
                cv_tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                cv_tf.RandomApply([cv_tf.GaussianBlur(3)], p=0.5),
                # cv_tf.GaussianBlur(3),
                cv_tf.RandomGrayscale(p=0.2),
                cv_tf.ToTensor(),
                cv_tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        )
        img2_transform = cv_tf.Compose(
            [
                cv_tf.ToPILImage(),
                imgRandomLandmarkMask(),
                cv_tf.Resize(size=(128, 128)),
                cv_tf.RandomCrop(112),
                cv_tf.RandomHorizontalFlip(p=0.5),
                # transforms.RandomApply
                cv_tf.RandomApply([cv_tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)], p=0.8),
                # cv_tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                cv_tf.RandomApply([cv_tf.GaussianBlur(3)], p=0.5),
                # cv_tf.GaussianBlur(3),
                cv_tf.RandomGrayscale(p=0.2),
                cv_tf.ToTensor(),
                cv_tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        )
        return img1_transform, img2_transform


    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True),
                          }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()

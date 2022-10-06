import math

import librosa
import glob
import os
import random
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torchaudio
from PIL import Image
from torch.utils.data import Dataset
from pytorchvideo.data.encoded_video import EncodedVideo
from tqdm import tqdm

from data_aug.face_swap import *


def id_name(df_metadata, id):
    index = int(id[3:]) - 1
    id_name = str(df_metadata.loc[index]).split('\\t')[-4]
    return id_name

def id_gender(df_metadata, id):
    index = int(id[3:]) - 1
    id_gender = str(df_metadata.loc[index]).split('\\t')[-3]
    return id_gender

def id_nation(df_metadata, id):
    index = int(id[3:]) - 1
    id_nation = str(df_metadata.loc[index]).split('\\t')[-2]
    return id_nation

def Normalize(tensor):
    # Subtract the mean, and scale to the interval [-1,1]
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean/tensor_minusmean.abs().max()

def rand_crop(audio, len_s):
    if audio.shape[0] > 1:
        start_s = random.randint(0, audio.shape[0] - len_s)
        return audio[start_s: start_s + len_s]
    else:
        start_s = random.randint(0, audio.shape[1] - len_s)
        return audio[:, start_s: start_s+len_s]
    

class maskCLR(Dataset):
    def __init__(self, *args, **kwargs):
        super(maskCLR, self).__init__()
        self.args = kwargs['args']
        self.prefix_vox1 = '/mnt/d/Data/Yuxuan/VoxCeleb/vox1'
        file = "/mnt/c/Data/Yuxuan/VoxCeleb/vox1/vox1_frames_test.txt"
        self.data = open(file, 'r')
        self.paths, self.start_frames, self.end_frames, self.frames = self.get_frame()

        # Facial landmark extraction
        self.detector = dlib.get_frontal_face_detector()
        cnn_detector_path = "/mnt/d/Data/Yuxuan/data/model/mmod_human_face_detector.dat"
        face_predictor_path = "/mnt/d/Data/Yuxuan/data/model/shape_predictor_68_face_landmarks.dat"
        self.cnn_detector = dlib.cnn_face_detection_model_v1(cnn_detector_path)
        self.face_predictor = dlib.shape_predictor(face_predictor_path)

        # Training setting
        self.trainMode = self.args.training_mode # simclr videoclr
        self.time_aug = self.args.time_aug # True False
        # Main comparison: hard negative for emotion and time-interval augmentation
        # Todo: Add distributed time-augmentation create list of time interval and randomize in-between
        # in getitem function, always return anchor, positive, negativeA, negativeP
        # So it only matters in the training process on how to process the fed data
        # either (A, P, _) or (A, P, NA, NP)
        # for simclr, get an anchor and a positive
        # for videoclr, get anchor, positive, negativeA and negativeP concat (key is to have negative pair)
        # 1-hard or multi-hard

        # Input setting
        self.nb_frame  = self.args.nb_frame # The number of frames, can be 1 for img and n for video
        # Use resnet50 to evaluate the designed task for image & audio(melspec) and MC3 to evaluate training on video
        self.sec = self.args.sec # The time duration of audio input
        self.fps = 30 if self.dataset == 'Aff2' else 25/6
        self.sample_rate = 22050
        self.interval = 1.
        self.interval_frame = (np.arange(self.fps * self.interval) + math.sqrt(self.fps * self.interval)).astype(int)
        self.n_interval = 3.
        self.n_distance = int(self.fps * self.n_interval)

        # Output setting
        self.dataMode = self.args.data_mode # Whether include audio as input or not
        # 1 no audio, return (A, P, NA, NP)
        # 2 only audio, return (A, P, NA, NP) for audio
        # 3 frame and audio, return (A, P, NA, NP, aA, aP, aNA, aNP)
        self.mapping = self.get_idx_mapping()

        # Data augmentation
        # Todo: audio add noise, rir, time mask, freq mask
        self.transforms_crop = kwargs['transforms_crop']
        self.transforms_img1 = kwargs['transforms_img1']
        self.transforms_img2 = kwargs['transforms_img2']

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        data_info = self.mapping[idx]
        Anchor = Image.open(data_info["Anchor"]).convert('RGB')
        Positive = Image.open(data_info["Positive"]).convert('RGB')
        NAnchor = Image.open(data_info["NegativeAnchor"]).convert('RGB')
        NPositive = Image.open(data_info["NegativePositive"]).convert('RGB')

        A_e, A_m = self.get_patch(Anchor)
        P_e, P_m = self.get_patch(Positive)
        N_e, N_m = self.get_patch(NAnchor)
        NP_e, NP_m = self.get_patch(NPositive)

        Sample = {
            'Anchor': self.transforms_img1(Anchor),
            'Anchor_eye':self.transforms_crop(A_e),
            'Anchor_mouth': self.transforms_crop(A_m),

            'Positive': self.transforms_img2(Positive),
            'Positive_eye': self.transforms_crop(P_e),
            'Positive_mouth': self.transforms_crop(P_m),

            'NAnchor': self.transforms_img1(NAnchor),
            'NAnchor_eye': self.transforms_crop(N_e),
            'NAnchor_mouth': self.transforms_crop(N_m),

            'NPositive': self.transforms_img2(NPositive),
            'NPositive_eye': self.transforms_crop(NP_e),
            'NPositive_mouth': self.transforms_crop(NP_m),
        }
        return  Sample

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    @staticmethod
    def detect_landmark(image, detector, cnn_detector, predictor):
        if isinstance(image, Image.Image):
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

    def get_patch(self, image):
        # image = np.array(image)
        landmark = self.detect_landmark(image, self.detector, self.cnn_detector, self.face_predictor)
        try:
            landmark_e = landmark[36:48, :]
            landmark_m = landmark[48:68, :]
        except:
            return Image.new('RGB', (64, 64)), Image.new('RGB', (64, 64))

        if isinstance(image, Image.Image):
            # image xmin, ymin, xmax, ymax
            img_e_crop = (int(min(landmark_e[:, 0])), int(min(landmark_e[:, 1])) - 10,
                       int(max(landmark_e[:, 0])), int(max(landmark_e[:, 1])) + 10)
            img_m_crop = (int(min(landmark_m[:, 0])), int(min(landmark_m[:, 1])),
                         int(max(landmark_m[:, 0])), int(max(landmark_m[:, 1])))
            img_e = image.crop(img_e_crop)
            img_m = image.crop(img_m_crop)
        else:
            img_e = image[int(min(landmark_e[:, 1])) - 10:int(max(landmark_e[:, 1])) + 10,
                    int(min(landmark_e[:, 0])):int(max(landmark_e[:, 0]))]
            img_m = image[int(min(landmark_m[:, 1])):int(max(landmark_m[:, 1])),
                    int(min(landmark_m[:, 0])):int(max(landmark_m[:, 0]))]
        return img_e, img_m

    def get_idx_mapping(self):
        mapping = {}
        cpt = 0

        for i in (pbar:= tqdm(range(len(self.paths)))):
            path = self.paths[i]
            start_frame = self.start_frames[i]
            end_frame = self.end_frames[i]
            frames = self.frames[i]
            # pbar_path = path.split('/')[-1]
            if self.dataset == 'Aff2':
                pbar.set_description('Processing file %s' % path.split('/')[-1])
                img_path = glob.glob(f"{self.prefix_aff2}/origin_faces/{path}/*.jpg")
            else:
                pbar.set_description('Processing ID %s' % path.split('/')[-3])
                img_path = glob.glob(f"{self.prefix_vox1}/{path}/*.jpg")

                # Choose the Anchor1, positive, Anchor negative, positive-negative in a certain range
            # Random get Anchor
            try:
                idx = random.randint(start_frame, end_frame)
            except:
                continue
            # Get Positive
            if self.time_aug:
                # idx_p: the start frame of the positive frame,
                # it is chosen within the near few frames of the anchor frame.
                # n_dis: is the farthest distance of the selected positive frames from the anchor
                idx_p = self.get_id(anchorID=idx, start_frame=start_frame,
                                    end_frame=end_frame, mode='pos')
                if idx_p is None:
                    continue
            else:
                idx_p = idx

            if self.trainMode == 'videoclr':
                # idx_n1: the start frame of the first negative frame,
                # it is chosen within the far few frames of the anchor frame.
                # f_dis: is the farthest distance of the selected negative frames from the anchor
                # The frame interval between the negative pairs is the same as the positive pairs: interval
                idx_n1 = self.get_id(anchorID=idx, start_frame=start_frame,
                                     end_frame=end_frame, mode='neg')
                if idx_n1 is None:
                    continue
                if self.time_aug:
                    idx_n2 = self.get_id(anchorID=idx_n1, start_frame=start_frame,
                                         end_frame=end_frame, mode='pos')
                    if idx_n2 is None:
                        continue
                else:
                    idx_n2 = idx_n1

            else:
                idx_n1 = 0
                idx_n2 = 0


            mapping[cpt] = {
                "Anchor": img_path[idx],
                "Positive": img_path[idx_p],
                "NegativeAnchor": img_path[idx_n1],
                "NegativePositive": img_path[idx_n2],
            }
            cpt += 1

        return mapping

    def get_distribution(self, mode):
        '''
        a uniform
        b linear upscale
        c linear downscale
        '''

        frame_num = len(self.interval_frame)
        if mode == 'a':
            weights = np.ones(frame_num)
        elif mode == 'b':
            weights = np.arange(1, frame_num+1)
        elif mode == 'c':
            weights = np.arange(frame_num, 0, -1)
        else:
            raise ValueError
        add = np.sum(weights)
        norm = np.linalg.norm(weights)
        weights = weights/add

        return weights

    def get_id(self, anchorID, start_frame, end_frame, mode):
        if self.nb_frame == 1:
            # nb_frame = self.nb_frame
            seq_length = end_frame - start_frame + 1
            # Todo: optimize the random func, allow choosing different distribution of time interval
            if mode == 'pos':
                list_pos = self.interval_frame
                weights = self.get_distribution(mode=self.args.distr_mode)
                # Get positive within a certain range
                dis =  np.random.choice(list_pos, p=weights)
                if anchorID - dis > start_frame and anchorID + dis < end_frame:
                    p = random.random()
                    if p < 0.5:
                        id_out = anchorID + dis
                    else:
                        id_out = anchorID - dis
                elif anchorID + dis < end_frame:
                    id_out = anchorID + dis
                elif anchorID - dis >= start_frame:
                    id_out = anchorID - dis
                else:
                    return None

            else: # neg
                dis = self.n_distance
                if anchorID - dis >= start_frame and anchorID + dis < end_frame:
                    p = random.random()
                    if p < 0.5:
                        try:
                            id_out = random.randint(anchorID + dis, end_frame - 1)
                        except:
                            id_out = end_frame - 1
                    else:
                        try:
                            id_out = random.randint(start_frame, anchorID - dis - 1)
                        except:
                            id_out = start_frame
                elif anchorID + dis < end_frame:
                    try:
                        id_out = random.randint(anchorID + dis, end_frame - 1)
                    except:
                        id_out = end_frame - 1
                elif anchorID - dis >= start_frame:
                    try:
                        id_out = random.randint(start_frame, anchorID - dis - 1)
                    except:
                        id_out = start_frame
                else:
                    return None
        else:
            raise ValueError
        return id_out

    def get_frame(self):
        paths = []
        start_frames = []
        end_frames =[]
        frames = []
        if self.args.dataset_name == 'Aff2':
            for data_info in self.data.readlines():
                path = data_info.split(",")[1]

                start_frame = int(data_info.split(",")[2])
                end_frame = int(data_info.split(",")[3])
                frame = int(data_info.split(",")[4].split("\n")[0])
                frame_path = path.split('/')[-1]

                paths.append(frame_path)
                start_frames.append(start_frame)
                end_frames.append(end_frame)
                frames.append(frame)
        else:
            for data_info in self.data.readlines():
                frame_path = data_info.split(",")[1]
                start_frame = int(data_info.split(",")[2])
                end_frame = int(data_info.split(",")[3])
                frame = int(data_info.split(",")[4].split("\n")[0])
                # frame_path = path.split('/')[-1]

                paths.append(frame_path)
                start_frames.append(start_frame)
                end_frames.append(end_frame)
                frames.append(frame)
        return paths, start_frames, end_frames, frames

class swapCLR(Dataset):
    def __init__(self, *args, **kwargs):
        super(swapCLR, self).__init__()
        self.args = kwargs['args']

        # Dataset setting
        self.dataset = self.args.dataset_name # Aff2 vox1 vox2
        # Three datasets to choose
        # Aff2 contains labels, can perform downstream task validation ()
        self.prefix_aff2 = '/mnt/c/Data/Yuxuan/ABAW'
        self.prefix_vox1 = '/mnt/d/Data/Yuxuan/VoxCeleb/vox1'
        self.prefix_vox2 = '/mnt/c/Data/Yuxuan/VoxCeleb/vox2'
        if self.dataset == 'Aff2':
            file = "/mnt/c/Data/Yuxuan/ABAW/Aff2_frames.txt"
            # file = "/mnt/c/Data/Yuxuan/ABAW/Aff2_frames_test.txt"
        elif self.dataset == 'vox1':
            file = "/mnt/c/Data/Yuxuan/VoxCeleb/vox1/vox1_frames_test.txt"
            # file = "/mnt/c/Data/Yuxuan/VoxCeleb/vox1/vox1_frames_withwav.txt"
        else:
            raise ValueError
        self.data = open(file, 'r')
        self.paths, self.start_frames, self.end_frames, self.frames = self.get_frame()

        # Facial landmark extraction
        self.detector = dlib.get_frontal_face_detector()
        cnn_detector_path = "/mnt/d/Data/Yuxuan/data/model/mmod_human_face_detector.dat"
        face_predictor_path = "/mnt/d/Data/Yuxuan/data/model/shape_predictor_68_face_landmarks.dat"
        self.cnn_detector = dlib.cnn_face_detection_model_v1(cnn_detector_path)
        self.face_predictor = dlib.shape_predictor(face_predictor_path)

        # Training setting
        self.trainMode = self.args.training_mode # simclr videoclr
        self.time_aug = self.args.time_aug # True False
        # Main comparison: hard negative for emotion and time-interval augmentation
        # Todo: Add distributed time-augmentation create list of time interval and randomize in-between
        # in getitem function, always return anchor, positive, negativeA, negativeP
        # So it only matters in the training process on how to process the fed data
        # either (A, P, _) or (A, P, NA, NP)
        # for simclr, get an anchor and a positive
        # for videoclr, get anchor, positive, negativeA and negativeP concat (key is to have negative pair)
        # 1-hard or multi-hard

        # Input setting
        self.nb_frame  = self.args.nb_frame # The number of frames, can be 1 for img and n for video
        # Use resnet50 to evaluate the designed task for image & audio(melspec) and MC3 to evaluate training on video
        self.sec = self.args.sec # The time duration of audio input
        self.fps = 30 if self.dataset == 'Aff2' else 25/6
        self.sample_rate = 22050
        self.interval = 1.
        self.interval_frame = (np.arange(self.fps * self.interval) + math.sqrt(self.fps * self.interval)).astype(int)
        self.n_interval = 3.
        self.n_distance = int(self.fps * self.n_interval)

        # Output setting
        self.dataMode = self.args.data_mode # Whether include audio as input or not
        # 1 no audio, return (A, P, NA, NP)
        # 2 only audio, return (A, P, NA, NP) for audio
        # 3 frame and audio, return (A, P, NA, NP, aA, aP, aNA, aNP)
        self.mapping = self.get_idx_mapping()

        # Data augmentation
        # Todo: audio add noise, rir, time mask, freq mask
        self.transforms_img1 = kwargs['transforms_img1']
        self.transforms_img2 = kwargs['transforms_img2']

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        data_info = self.mapping[idx]

        Anchor = cv2.imread(data_info["Anchor"], cv2.IMREAD_COLOR)
        NAnchor = cv2.imread(data_info["NegativeAnchor"], cv2.IMREAD_COLOR)

        # Randomly apply face swa to create positive pairs
        p = 0.8
        q = 0.8
        id = random.randint(0, len(self.mapping) - 1)
        face_data = self.mapping[id]
        im, landmarks = read_im_and_landmarks(face_data["Face_id"])
        if random.random() < p:
            im1, landmarks1 = read_im_and_landmarks(data_info["Positive"])
            try:
                Positive = self.face_swap(im, im1, landmarks, landmarks1).astype(np.uint8)
            except:
                Positive = cv2.imread(data_info["Positive"], cv2.IMREAD_COLOR)
        else:
            Positive = cv2.imread(data_info["Positive"], cv2.IMREAD_COLOR)
        if random.random() < q:
            im2, landmarks2 = read_im_and_landmarks(data_info["NegativePositive"])
            try:
                NPositive = self.face_swap(im, im2, landmarks, landmarks2).astype(np.uint8)
            except:
                NPositive = cv2.imread(data_info["NegativePositive"], cv2.IMREAD_COLOR)
        else:
            NPositive = cv2.imread(data_info["NegativePositive"], cv2.IMREAD_COLOR)
        Sample = {
            'Anchor': self.transforms_img1(Anchor),
            'Positive': self.transforms_img2(Positive),
            'NAnchor': self.transforms_img1(NAnchor),
            'NPositive': self.transforms_img2(NPositive),
        }
        return Sample


    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    @staticmethod
    def face_swap(im1, im2, landmarks1, landmarks2):
        M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                       landmarks2[ALIGN_POINTS])

        mask = get_face_mask(im2, landmarks2)
        warped_mask = warp_im(mask, M, im1.shape)
        combined_mask = numpy.max([get_face_mask(im1, landmarks1), warped_mask],
                                  axis=0)

        warped_im2 = warp_im(im2, M, im1.shape)
        warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)

        output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
        return output_im

    def get_idx_mapping(self):
        mapping = {}
        cpt = 0

        for i in (pbar:= tqdm(range(len(self.paths)))):
            path = self.paths[i]
            start_frame = self.start_frames[i]
            end_frame = self.end_frames[i]
            frames = self.frames[i]
            # pbar_path = path.split('/')[-1]
            if self.dataset == 'Aff2':
                pbar.set_description('Processing file %s' % path.split('/')[-1])
                img_path = glob.glob(f"{self.prefix_aff2}/origin_faces/{path}/*.jpg")
            else:
                pbar.set_description('Processing ID %s' % path.split('/')[-3])
                img_path = glob.glob(f"{self.prefix_vox1}/{path}/*.jpg")

                # Choose the Anchor1, positive, Anchor negative, positive-negative in a certain range
            # Random get Anchor
            try:
                face_path = random.choice(img_path)
            except:
                continue
            try:
                idx = random.randint(start_frame, end_frame)
            except:
                continue
            # Get Positive
            if self.time_aug:
                # idx_p: the start frame of the positive frame,
                # it is chosen within the near few frames of the anchor frame.
                # n_dis: is the farthest distance of the selected positive frames from the anchor
                idx_p = self.get_id(anchorID=idx, start_frame=start_frame,
                                    end_frame=end_frame, mode='pos')
                if idx_p is None:
                    continue
            else:
                idx_p = idx

            # idx_n1: the start frame of the first negative frame,
            # it is chosen within the far few frames of the anchor frame.
            # f_dis: is the farthest distance of the selected negative frames from the anchor
            # The frame interval between the negative pairs is the same as the positive pairs: interval
            idx_n1 = self.get_id(anchorID=idx, start_frame=start_frame,
                                 end_frame=end_frame, mode='neg')
            if idx_n1 is None:
                continue
            if self.time_aug:
                idx_n2 = self.get_id(anchorID=idx_n1, start_frame=start_frame,
                                     end_frame=end_frame, mode='pos')
                if idx_n2 is None:
                    continue
            else:
                idx_n2 = idx_n1

            mapping[cpt] = {
                "Anchor": img_path[idx],
                "Positive": img_path[idx_p],
                "NegativeAnchor": img_path[idx_n1],
                "NegativePositive": img_path[idx_n2],
                "Face_id": face_path,
            }
            cpt += 1

        return mapping

    def get_distribution(self, mode):
        '''
        a uniform
        b linear upscale
        c linear downscale
        '''

        frame_num = len(self.interval_frame)
        if mode == 'a':
            weights = np.ones(frame_num)
        elif mode == 'b':
            weights = np.arange(1, frame_num+1)
        elif mode == 'c':
            weights = np.arange(frame_num, 0, -1)
        else:
            raise ValueError
        add = np.sum(weights)
        norm = np.linalg.norm(weights)
        weights = weights/add

        return weights

    def get_id(self, anchorID, start_frame, end_frame, mode):
        if self.nb_frame == 1:
            # nb_frame = self.nb_frame
            seq_length = end_frame - start_frame + 1
            # Todo: optimize the random func, allow choosing different distribution of time interval
            if mode == 'pos':
                list_pos = self.interval_frame
                weights = self.get_distribution(mode=self.args.distr_mode)
                # Get positive within a certain range
                dis =  np.random.choice(list_pos, p=weights)
                if anchorID - dis > start_frame and anchorID + dis < end_frame:
                    p = random.random()
                    if p < 0.5:
                        id_out = anchorID + dis
                    else:
                        id_out = anchorID - dis
                elif anchorID + dis < end_frame:
                    id_out = anchorID + dis
                elif anchorID - dis >= start_frame:
                    id_out = anchorID - dis
                else:
                    return None

            else: # neg
                dis = self.n_distance
                if anchorID - dis >= start_frame and anchorID + dis < end_frame:
                    p = random.random()
                    if p < 0.5:
                        try:
                            id_out = random.randint(anchorID + dis, end_frame - 1)
                        except:
                            id_out = end_frame - 1
                    else:
                        try:
                            id_out = random.randint(start_frame, anchorID - dis - 1)
                        except:
                            id_out = start_frame
                elif anchorID + dis < end_frame:
                    try:
                        id_out = random.randint(anchorID + dis, end_frame - 1)
                    except:
                        id_out = end_frame - 1
                elif anchorID - dis >= start_frame:
                    try:
                        id_out = random.randint(start_frame, anchorID - dis - 1)
                    except:
                        id_out = start_frame
                else:
                    return None
        else:
            raise ValueError
        return id_out

    def get_frame(self):
        paths = []
        start_frames = []
        end_frames =[]
        frames = []
        if self.args.dataset_name == 'Aff2':
            for data_info in self.data.readlines():
                path = data_info.split(",")[1]

                start_frame = int(data_info.split(",")[2])
                end_frame = int(data_info.split(",")[3])
                frame = int(data_info.split(",")[4].split("\n")[0])
                frame_path = path.split('/')[-1]

                paths.append(frame_path)
                start_frames.append(start_frame)
                end_frames.append(end_frame)
                frames.append(frame)
        else:
            for data_info in self.data.readlines():
                frame_path = data_info.split(",")[1]
                start_frame = int(data_info.split(",")[2])
                end_frame = int(data_info.split(",")[3])
                frame = int(data_info.split(",")[4].split("\n")[0])
                # frame_path = path.split('/')[-1]

                paths.append(frame_path)
                start_frames.append(start_frame)
                end_frames.append(end_frame)
                frames.append(frame)
        return paths, start_frames, end_frames, frames

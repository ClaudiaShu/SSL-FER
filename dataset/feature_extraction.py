import os

import torchvision.models
from moviepy.editor import VideoFileClip
from torchvision import transforms

from abaw_config import configs
import glob
import cv2

# from models import resnet50
# from utils import *
import librosa
import soundfile as sf
# import time
# import tensorflow as tf
# from tensorflow.keras.models import load_model
import lmdb

# from models.FER_model.ResNet import ResNet34

transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize(size=(112, 112)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

def video_to_audio(video_path, save_path):
    list_csv1 = glob.glob(os.path.join(video_path, 'batch1', '*'))
    list_csv2 = glob.glob(os.path.join(video_path, 'batch2', '*'))
    video_list = list_csv1 + list_csv2
    # video_list = glob.glob(os.path.join(video_path, '/*'))
    for i, video_path in enumerate(video_list):
        video_name =video_path.split('/')[-1].split('.')[0]
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(os.path.join(save_path, f"{video_name}.wav"), logger=None)
        print('\r', f"[INFO] ({i + 1}/{len(video_list)}) Converting video to wav {video_name}", end = '')

def audio_crop(video_path, audio_path, save_path, sec = 6, sample_rate = 22050):
    list_csv1 = glob.glob(os.path.join(video_path, 'batch1', '*'))
    list_csv2 = glob.glob(os.path.join(video_path, 'batch2', '*'))
    video_list = list_csv1 + list_csv2
    video_list = [x for x in video_list if 'mp4' in x.split('.') or 'avi' in x.split('.')]
    for i, video in enumerate(video_list):
        print('\r', f"[INFO] ({i + 1}/{len(video_list)}) cropping wav file {video}", end='')
        video_name = video.split("/")[-1].split(".")[0]
        save_dir = os.path.join(save_path, video_name)
        if os.path.isdir(save_dir):
            continue
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        cap = cv2.VideoCapture(video)
        frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        data, sr = librosa.load(os.path.join(audio_path, f'{video_name}.wav'), sr = 22050)
        if len(data) < sec*sample_rate:
            print(f"shorter than 10 sec: {video}")
            continue
        audio_per_frame = int(len(data) / frame_num)
        idx_list = []
        # for i in range(frame_num):
        #     if audio_per_frame * i - sample_rate * sec >= 0:
        #         idx_list = (i, audio_per_frame * i - sample_rate * sec, audio_per_frame * i)
        idx_list = [(i, audio_per_frame*i-sample_rate*sec, audio_per_frame*i) for i
                    in range(frame_num) if audio_per_frame*i-sample_rate*sec >=0]
        for j, idx in enumerate(idx_list):
            try:
                image_idx, st_idx, end_idx = idx
                wav_path = os.path.join(save_dir, f"{image_idx}.wav")
                print('\r', f"[INFO] ({i + 1}/{len(video_list)}) cropping wav file {video_name} ({j / len(idx_list) * 100:.1f}%)", end='')
                if os.path.isfile(wav_path):
                    continue
                audio_chunk = data[st_idx:end_idx]
                if len(audio_chunk) != sec*sample_rate:
                    print(f'{os.path.join(save_dir, f"{image_idx}.wav")}')
                sf.write(wav_path, audio_chunk, samplerate = sample_rate)
            except:
                print(video, idx, wav_path)


def audio_save():
    audio_path = "/mnt/c/Data/Yuxuan/ABAW/cropped_audio"
    file_name = 'video66'
    data, sr = librosa.load(os.path.join(audio_path, file_name,'1.wav'), sr=22050)
    sf.write(os.path.join(audio_path, file_name,'1.wav'), data, samplerate=22050)
    return 0


if __name__ == '__main__':
    audio_save()


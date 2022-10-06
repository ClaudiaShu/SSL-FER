import random
import os
import glob
import math
import numpy as np
from scipy import signal
import soundfile

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

def _get_sample(path, resample=None):
  effects = [
    ["remix", "1"]
  ]
  if resample:
    effects.extend([
      ["lowpass", f"{resample // 2}"],
      ["rate", f'{resample}'],
    ])
  return torchaudio.sox_effects.apply_effects_file(path, effects=effects)

class Compose(object):
    """Composes several transforms
        Args:
        transforms (list of ``Transform`` objects): list of transforms
        to compose
        """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)
        return audio

class ToNumpy(object):
    def __call__(self, audio):
        return audio.detach().numpy()

    def __repr__(self):
        return self.__class__.__name__

class ToTensor(object):
    def __call__(self, audio):
        return torch.from_numpy(audio).float()

    def __repr__(self):
        return self.__class__.__name__

class TensorSqueeze(object):
    def __call__(self, audio):
        return audio.squeeze()

    def __repr__(self):
        return self.__class__.__name__

class ToMel(object):
    def __init__(self, sr=16000):
        n_fft = 1024
        win_length = None
        hop_length = 512
        n_mels = 128

        self.melspectrogram = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm='slaney',
            onesided=True,
            n_mels=n_mels,
        )
        # self.melspectrogram = torchaudio.transforms.MelSpectrogram(sr)
    def __call__(self, audio):
        return self.melspectrogram(audio)

    def __repr__(self):
        return self.__class__.__name__

class ToImage(object):
    def __call__(self, spec, eps=1e-6):
        mean = spec.mean()
        std = spec.std()
        spec_norm = (spec - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_scaled = 1 * (spec_norm - spec_min) / (spec_max - spec_min + eps)
        return spec_scaled
    def __repr__(self):
        return self.__class__.__name__

class RemoveNAN(object):
    def __call__(self, audio):
        if torch.any(torch.isnan(audio)):
            audio = torch.zeros(audio.shape[0],audio.shape[1])
        return audio
    def __repr__(self):
        return self.__class__.__name__

def loadWAV(filename, max_frames, evalmode=True, num_eval=10):
    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    audio, sample_rate = soundfile.read(filename)

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1
        audio = np.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]

    if evalmode:
        startframe = np.linspace(0, audiosize - max_audio, num=num_eval)
    else:
        startframe = np.array([np.int64(random.random() * (audiosize - max_audio))])

    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + max_audio])

    feat = np.stack(feats, axis=0).astype(np.float)

    return feat

class AugmentWAV(object):

    def __init__(self):
        self.prefix = '/mnt/c/Data/Yuxuan/AudioAug'

        # self.max_frames = max_frames
        # self.max_audio = max_frames * 160 + 240

        self.noisetypes = ['noise', 'speech', 'music']

        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 7], 'music': [1, 1]}
        self.noiselist = {}

        musan_data = "/mnt/c/Data/Yuxuan/AudioAug/musan_all.txt"
        musan_file = open(musan_data, 'r')
        musan = []
        for line in musan_file.readlines():
            musan_path = os.path.join(self.prefix,line.split('\n')[0])
            musan.append(musan_path)
        musan_file.close()

        # augment_files = glob.glob(os.path.join(musan_path, '*/*/*/*.wav'))
        augment_files = musan
        for file in augment_files:
            if not file.split('/')[-3] in self.noiselist:
                self.noiselist[file.split('/')[-3]] = []
            self.noiselist[file.split('/')[-3]].append(file)

        rir_data = "/mnt/c/Data/Yuxuan/AudioAug/rir_all.txt"
        rir_file = open(rir_data, 'r')
        rir = []
        for line in rir_file.readlines():
            rir_path = os.path.join(self.prefix, line.split('\n')[0])
            rir.append(rir_path)
        rir_file.close()
        self.rir_files = rir
        # self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))

    def additive_noise(self, noisecat, audio):

        clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)

        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))

        noises = []

        for noise in noiselist:
            noiseaudio, sr = _get_sample(noise, resample=None)
            noiseaudio = torch.nn.functional.pad(noiseaudio, (audio.shape[0] - noiseaudio.shape[1], 0))\
                if noiseaudio.shape[1]<=audio.shape[0] else noiseaudio[:,:audio.shape[0]]
            # Todo: with tensor
            noiseaudio = np.array(noiseaudio)
            # noiseaudio = loadWAV(noise, self.max_frames, evalmode=False)
            noise_snr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noise_db = 10 * np.log10(np.mean(noiseaudio[0] ** 2) + 1e-4)
            ns = np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio
            noises.append(ns)

        return np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True) + audio

    def reverberate(self, audio):
        # audio = torch.unsqueeze(audio, dim=0).float()
        rir_file = random.choice(self.rir_files)

        rir_raw, sample_rate = _get_sample(rir_file, resample=None)
        # Clean up the rir
        # Extract the main impulse, normalise the signal and flip along the time axis
        if rir_raw.shape[1] > int(sample_rate * 1.3):
            rir = rir_raw[:, int(sample_rate * 1.01):int(sample_rate * 1.3)]
        else:
            rir = rir_raw
        rir = rir / torch.norm(rir, p=2)
        rir = torch.flip(rir, [1])

        # convolve the speech signal with the RIR filter.
        speech_ = torch.nn.functional.pad(audio, (rir.shape[1] - 1, 0))
        return torch.nn.functional.conv1d(speech_[None, ...], rir[None, ...])[0]

class myRandomMusan(object):
    def __init__(self, data, p=0.2):
        # self.MUSAN_PATH = "/mnt/d/Data/Yuxuan/VoxCeleb/musan"
        self.prefix = '/mnt/c/Data/Yuxuan/AudioAug'
        self.data = data
        self.p = p
    def __call__(self, audio):
        if random.random() < self.p:
            # ns_files = glob.glob(os.path.join(self.MUSAN_PATH, '*/*/*.wav'))
            ns_file = random.choice(self.data)
            file = os.path.join(self.prefix, ns_file)
            ns_raw, ns_sr = _get_sample(file, resample=None)
            return signal.lfilter(ns_raw.squeeze(), 1, audio)
        else:
            return audio
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class myRandomRIR(object):
    def __init__(self, data, p=0.2):
        # self.RIR_PATH = "/mnt/d/Data/Yuxuan/VoxCeleb/rirs_noises"
        self.prefix = '/mnt/c/Data/Yuxuan/AudioAug'
        self.data = data
        self.p = p
    def __call__(self, audio):
        if random.random() < self.p:
            # rir_files = glob.glob(os.path.join(self.RIR_PATH, '*/*/*.wav'))
            rir_file = random.choice(self.data)
            file = os.path.join(self.prefix, rir_file)
            rir_raw, sample_rate = _get_sample(file, resample=None)
            # Clean up the rir
            # Extract the main impulse, normalise the signal and flip along the time axis
            if rir_raw.shape[1]>int(sample_rate * 1.3):
                rir = rir_raw[:, int(sample_rate * 1.01):int(sample_rate * 1.3)]
            else:
                rir = rir_raw
            rir = rir / torch.norm(rir, p=2)
            rir = torch.flip(rir, [1])

            # convolve the speech signal with the RIR filter.
            speech_ = torch.nn.functional.pad(audio, (rir.shape[1] - 1, 0))
            audio = torch.nn.functional.conv1d(speech_[None, ...], rir[None, ...])[0]

            return audio
        else:
            return audio
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomNoise(object):
    def __init__(self, p1=0.5, p2=0.5):
        # self.MUSAN_PATH = "/mnt/d/Data/Yuxuan/VoxCeleb/musan"

        self.augment_wav = AugmentWAV()
        self.p1 = p1
        self.p2 = p2
    def __call__(self, audio):
        if random.random() < self.p1:

            augtype = random.randint(1, 3)
            if augtype == 1:
                audio = torch.from_numpy(self.augment_wav.additive_noise('music', audio))
            elif augtype == 2:
                audio = torch.from_numpy(self.augment_wav.additive_noise('speech', audio))
            elif augtype == 3:
                audio = torch.from_numpy(self.augment_wav.additive_noise('noise', audio))
            else:
                raise ValueError
        else:
            audio = torch.from_numpy(np.expand_dims(audio, axis=0))

        if random.random() < self.p2:
            audio = self.augment_wav.reverberate(audio.float())
        else:
            audio = audio.float()

        return audio
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomLowpass(object):
    def __init__(self, p=0.5, sr=22050):
        self.p = p
        self.sr = sr
        self.effects = [
            ["lowpass", "-1", "300"],  # apply single-pole lowpass filter
        ]
    def __call__(self, audio):
        if random.random() < self.p:
            waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
                audio, self.sr, self.effects)
            audio = waveform
        return audio
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomDownSample(object):
    def __init__(self, p=0.5, sr=22050):
        self.p = p
        self.sr = sr
        self.effects = [
            ["speed", "0.8"],  # reduce the speed
                             # This only changes sample rate, so it is necessary to
                             # add `rate` effect with original sample rate after this.
            ["rate", f"{sr}"],
        ]
    def __call__(self, audio):
        if random.random() < self.p:
            waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
                audio, self.sr, self.effects)
            audio = waveform
        return audio
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomUpSample(object):
    def __init__(self, p=0.5, sr=22050):
        self.p = p
        self.sr = sr
        self.effects = [
            ["speed", "1.2"],  # reduce the speed
                             # This only changes sample rate, so it is necessary to
                             # add `rate` effect with original sample rate after this.
            ["rate", f"{sr}"],
        ]
    def __call__(self, audio):
        if random.random() < self.p:
            waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
                audio, self.sr, self.effects)
            audio = waveform
        return audio
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomSpeed(object):
    def __init__(self, p=0.5, sr=22050):
        self.p = p
        self.sr = sr
        self.effects1 = [
            ["speed", "1.2"],  # increase the speed
                             # This only changes sample rate, so it is necessary to
                             # add `rate` effect with original sample rate after this.
            ["rate", f"{sr}"],
        ]
        self.effects2 = [
            ["speed", "0.8"],  # reduce the speed
            # This only changes sample rate, so it is necessary to
            # add `rate` effect with original sample rate after this.
            ["rate", f"{sr}"],
        ]
    def __call__(self, audio):
        if random.random() < self.p:
            if random.random() < 0.5:
                effects = self.effects1
            else:
                effects = self.effects2
            waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
                audio, self.sr, effects)
            audio = waveform
        return audio
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomReverb(object):
    def __init__(self, p=0.5, sr=22050):
        '''
        reverb [−w|−−wet-only] [reverberance (50%) [HF-damping (50%)
        [room-scale (100%) [stereo-depth (100%)
        [pre-delay (0ms) [wet-gain (0dB)]]]]]]
        '''
        self.p = p
        self.sr = sr
        self.effects = [
            ["reverb", "-w"],  # Reverbration gives some dramatic feeling
        ]
    def __call__(self, audio):
        if random.random() < self.p:
            waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
                audio, self.sr, self.effects)
            augtype = random.randint(0, 1)
            if augtype == 0:
                audio = torch.unsqueeze(waveform[0],dim=0)
            else:
                audio = torch.unsqueeze(waveform[1],dim=0)
        return audio
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomBackgroundNoise(object):
    def __init__(self, snr_db=10, eps=1e-6, p=0.8):
        self.prefix = '/mnt/c/Data/Yuxuan/AudioAug'
        self.p = p
        self.eps = eps
        self.snr_db = snr_db
        # self.data = data
    def __call__(self, audio):
        if random.random() < self.p:
            augtype = random.randint(1, 3)
            if augtype == 1:
                data_file = "/mnt/c/Data/Yuxuan/AudioAug/music_all.txt"
            elif augtype == 2:
                data_file = "/mnt/c/Data/Yuxuan/AudioAug/noise_all.txt"
            elif augtype == 3:
                data_file = "/mnt/c/Data/Yuxuan/AudioAug/speech_all.txt"
            else:
                raise ValueError
            ns_file = random.choice(open(data_file, 'r').readlines()).split('\n')[0]
            file = os.path.join(self.prefix, ns_file)
            ns_raw, ns_sr = _get_sample(file, resample=None)
            if ns_raw.shape[1] > audio.shape[1]:
                noise = ns_raw[:, :audio.shape[1]]
            else:
                noise = torch.nn.functional.pad(ns_raw, (audio.shape[1] - ns_raw.shape[1], 0))

            speech_power = audio.norm(p=2)
            noise_power = noise.norm(p=2)

            snr = math.exp(self.snr_db / 10)
            scale = snr * noise_power / (speech_power + self.eps)
            audio = (scale * audio + noise) / 2
        return audio

class RandomTimeMask(object):
    """Apply masking to a spectrogram in the time domain.

        Args:
            time_mask_param (int): maximum possible length of the mask.
                Indices uniformly sampled from [0, time_mask_param)
        """
    def __init__(self, p=0.5, time_mask_param=80):
        self.p = p
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
    def __call__(self, audio):
        if random.random() < self.p:
            self.time_masking(audio)
        return audio
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomFreqMask(object):
    """Apply masking to a spectrogram in the frequency domain.

        Args:
            freq_mask_param (int): maximum possible length of the mask.
                Indices uniformly sampled from [0, freq_mask_param)
        """
    def __init__(self, p=0.5, freq_mask_param=20):
        self.p = p
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
    def __call__(self, audio):
        if random.random() < self.p:
            self.freq_masking(audio)
        return audio
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


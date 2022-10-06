import glob
import os
import random

import numpy
import soundfile
from scipy import signal

MUSAN = "/mnt/d/Data/Yuxuan/VoxCeleb/musan"
RIR = "/mnt/d/Data/Yuxuan/VoxCeleb/rirs_noises"


def round_down(num, divisor):
    return num - (num % divisor)


def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)


def loadWAV(filename, max_frames, evalmode=True, num_eval=10):
    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    audio, sample_rate = soundfile.read(filename)

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1
        audio = numpy.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]

    if evalmode:
        startframe = numpy.linspace(0, audiosize - max_audio, num=num_eval)
    else:
        startframe = numpy.array([numpy.int64(random.random() * (audiosize - max_audio))])

    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + max_audio])

    feat = numpy.stack(feats, axis=0).astype(numpy.float)

    return feat

class AugmentWAV(object):

    def __init__(self, musan_path, rir_path, max_frames):

        self.max_frames = max_frames
        self.max_audio = max_frames * 160 + 240

        self.noisetypes = ['noise', 'speech', 'music']

        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 7], 'music': [1, 1]}
        self.noiselist = {}

        augment_files = glob.glob(os.path.join(musan_path, '*/*/*.wav'))

        for file in augment_files:
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)

        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))

    def additive_noise(self, noisecat, audio):

        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)

        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))

        noises = []

        for noise in noiselist:
            noiseaudio = loadWAV(noise, self.max_frames, evalmode=False)
            noise_snr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio[0] ** 2) + 1e-4)
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)

        return numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True) + audio

    def reverberate(self, audio):

        rir_file = random.choice(self.rir_files)

        rir, fs = soundfile.read(rir_file)
        rir = numpy.expand_dims(rir.astype(numpy.float), 0)
        rir = rir / numpy.sqrt(numpy.sum(rir ** 2))

        return signal.convolve(audio, rir, mode='full')[:, :self.max_audio]
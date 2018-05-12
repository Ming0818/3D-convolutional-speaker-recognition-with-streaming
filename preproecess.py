import numpy as np
import speechpy
import pyaudio
import wave
import time
import soundfile as sf
import sys


def chunk2cube(chunk, sampling_frequency, num_coefficient=40, frame_length=0.025, frame_stride=0.01):
    npdata = np.fromstring(chunk, dtype=np.int16)
    print(npdata.shape)
    frames = speechpy.processing.stack_frames(npdata, sampling_frequency=sampling_frequency, frame_length=frame_length,
                                              frame_stride=frame_stride, zero_padding=True)
    power_spectrun = speechpy.processing.power_spectrum(frames, fft_points=2 * num_coefficient)[:, 1:]
    logenergy = speechpy.feature.lmfe(npdata, sampling_frequency=16000, frame_length=frame_length,
                                      frame_stride=frame_stride,
                                      num_filters=num_coefficient, fft_length=1024, low_frequency=0,
                                      high_frequency=None)

    feature_cube = np.zeros((10, 20, num_coefficient), dtype=np.float32)

    idx = np.random.randint(logenergy.shape[0] - 30, size=10)
    for num, index in enumerate(idx):
        feature_cube[num, :, :] = logenergy[index:index + 20, :]

    return feature_cube[None, :, :, :]


def wav2cubes(wavfile, num_coefficient=40):
    signal, fs = sf.read(wavfile)
    # Staching frames
    frames = speechpy.processing.stack_frames(signal, sampling_frequency=fs, frame_length=0.025,
                                              frame_stride=0.01,
                                              zero_padding=True)

    logenergy = speechpy.feature.lmfe(signal, sampling_frequency=fs, frame_length=0.025, frame_stride=0.01,
                                      num_filters=num_coefficient, fft_length=1024, low_frequency=0,
                                      high_frequency=None)

    feature_cube = np.zeros((10, 20, num_coefficient), dtype=np.float32)

    idx = np.random.randint(logenergy.shape[0] - 30, size=10)
    for num, index in enumerate(idx):
        feature_cube[num, :, :] = logenergy[index:index + 20, :]

    return feature_cube[None, :, :, :]


def main():
    print(wav2cubes("/home/sushi/hdd/data/voxceleb1/voxceleb1_wav/A.J._Buckley/1zcIwhmdeo4_0000001.wav").shape)


if __name__ == '__main__':
    main()

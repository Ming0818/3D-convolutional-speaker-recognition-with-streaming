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


def wav2cubes(wavfile, num_frames=20, num_coefficient=40):
    signal, fs = sf.read(wavfile)
    # Staching frames

    logenergy = speechpy.feature.lmfe(signal, sampling_frequency=fs, frame_length=0.025, frame_stride=0.01,
                                      num_filters=num_coefficient, fft_length=1024, low_frequency=0,
                                      high_frequency=None)

    # random sampling(fixed count = 10, size = 20(*0.025 sec) * 40(coefficient))

    # feature_cube = np.zeros((10, 20, num_coefficient), dtype=np.float32)
    # idx = np.random.randint(logenergy.shape[0] - 30, size=10)
    # for num, index in enumerate(idx):
    #     feature_cube[num, :, :] = logenergy[index:index + 20, :]

    # sequential sampling(size = 20(*0.025 sec) * 40(coefficient)), using overlapping(10 frame interval)
    feature_cube = np.zeros((1, num_frames, num_coefficient), dtype=np.float32)

    for num, start_point in enumerate(range(0, logenergy.shape[0] - num_frames, 10)):
        feature_cube = np.concatenate(
            (feature_cube, logenergy[start_point:start_point + num_frames, :].reshape(-1, num_frames, num_coefficient)))
    return feature_cube[None, 1:, :, :].astype(np.float32)


def main():
    print(wav2cubes("recog.wav").shape)


if __name__ == '__main__':
    main()

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
    # frames = speechpy.processing.stack_frames(npdata, sampling_frequency=sampling_frequency, frame_length=frame_length,
    #                                           frame_stride=frame_stride, zero_padding=True)
    # power_spectrun = speechpy.processing.power_spectrum(frames, fft_points=2 * num_coefficient)[:, 1:]
    logenergy = speechpy.feature.lmfe(npdata, sampling_frequency=16000, frame_length=frame_length,
                                      frame_stride=frame_stride,
                                      num_filters=num_coefficient, fft_length=1024, low_frequency=0,
                                      high_frequency=None)
    # total sampling
    return logenergy.astype(np.float32), len(logenergy)

    # random sampling
    # feature_cube = np.zeros((10, 20, num_coefficient), dtype=np.float32)
    #
    # idx = np.random.randint(logenergy.shape[0] - 30, size=10)
    # for num, index in enumerate(idx):
    #     feature_cube[num, :, :] = logenergy[index:index + 20, :]
    #
    # return feature_cube[None, :, :, :]


def wav2cubes(wavfile, num_frames=20, num_coefficient=40, max_seqlen=500):
    # 500 frames
    signal, fs = sf.read(wavfile, dtype="float32", frames=((max_seqlen+1)*160))

    logenergy = speechpy.feature.lmfe(signal, sampling_frequency=fs, frame_length=0.02, frame_stride=0.01,
                                      num_filters=num_coefficient, fft_length=1024, low_frequency=0,
                                      high_frequency=None)

    seqlen = len(logenergy)

    if seqlen < max_seqlen:
        logenergy = np.concatenate(
            (logenergy,
             np.zeros((max_seqlen-seqlen, num_coefficient), dtype=np.float32)
             ),axis=0)

    # total sampling
    return logenergy.astype(np.float32).reshape(-1, max_seqlen, num_coefficient), seqlen

    # random sampling(fixed count = 10, size = 20(*0.025 sec) * 40(coefficient))

    # feature_cube = np.zeros((10, 20, num_coefficient), dtype=np.float32)
    # idx = np.random.randint(logenergy.shape[0] - 30, size=10)
    # for num, index in enumerate(idx):
    #     feature_cube[num, :, :] = logenergy[index:index + 20, :]

    # sequential sampling(size = 20(*0.025 sec) * 40(coefficient)), using overlapping(10 frame interval)
    # feature_cube = list()
    #
    # count = 0
    # for num, start_point in enumerate(range(0, logenergy.shape[0] - num_frames, 10)):
    #     if count >= max_seqlen :
    #         break
    #     feature_cube.append(logenergy[start_point:start_point + num_frames, :].tolist())
    #     count += 1
    # if count < max_seqlen:
    #     padding_arr = np.zeros((max_seqlen-count, num_frames, num_coefficient)).tolist()
    #     feature_cube = feature_cube + padding_arr
    #
    # return np.array(feature_cube).reshape((-1,max_seqlen,num_frames,num_coefficient)), np.array(count).reshape((-1,1))


def main():
    print(wav2cubes("/home/aksdmj/dataset/voxceleb1/voxceleb1_wav/A.J._Buckley/1zcIwhmdeo4_0000001.wav")[0])


if __name__ == '__main__':
    main()

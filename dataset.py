from glob import glob
from preproecess import wav2cubes
import numpy as np
import os


def make_numpy_file(dir_name):

    ret = [None, None]
    length = len(glob(dir_name+"*/"))
    for idx_person, person_name in enumerate(glob(dir_name + "*/")):
        print("%d/%d..." % (idx_person+1, length))
        for idx_wav, wav_file in enumerate(glob(person_name+"*.wav")):
            if idx_wav == 0 :
                ret[0] = wav2cubes(wav_file)
                ret[1] = [idx_person]
            else:
                ret[0] = np.concatenate((ret[0], wav2cubes(wav_file)), axis=0)
                ret[1].append(idx_person)
        np.save("data_array/%s_input" % idx_person, ret[0])
        print(ret[0].shape)
        np.save("data_array/%s_label" % idx_person, ret[1])
        print(len(ret[1]))

def main():
    make_numpy_file("/home/sushi/hdd/data/voxceleb1/voxceleb1_wav/")



if __name__ == "__main__":
    main()
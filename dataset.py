from glob import glob
from preproecess import wav2cubes
import numpy as np
import os
import h5py

def make_numpy_file(dir_name):

    ret = [None, None]
    length = len(glob(dir_name+"*/"))
    for idx_person, person_name in enumerate(glob(dir_name + "*/")):
        print("%d/%d..." % (idx_person+1, length))
        for idx_wav, wav_file in enumerate(glob(person_name+"*.wav")):
            if idx_wav == 0 and idx_person ==0 :
                ret[0] = wav2cubes(wav_file)
                ret[1] = [idx_person]
            else:
                ret[0] = np.concatenate((ret[0], wav2cubes(wav_file)), axis=0)
                ret[1].append(idx_person)
    h5f = h5py.File('data.h5', 'w')
    h5f.create_dataset('utterances', data=ret[0])
    h5f.create_dataset('labels', data=ret[1])
    h5f.close()

def main():
    make_numpy_file("/home/sushi/hdd/data/voxceleb1/voxceleb1_wav/")



if __name__ == "__main__":
    main()
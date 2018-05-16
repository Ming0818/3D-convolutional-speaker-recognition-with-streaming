from __future__ import print_function
from glob import glob
from preproecess import wav2cubes
import numpy as np
import os
import h5py
import time

def make_numpy_file(dir_name):

    length = len(glob(dir_name+"*/"))
    start_time = time.time()
    FILE_SIZE = 10

    speechs = list()
    labels = list()
    seqs = list()
    names = list()

    for idx_person, person_name in enumerate(glob(dir_name + "*/")):
        print("%d/%d..." % (idx_person+1, length), end='')
        for idx_wav, wav_file in enumerate(glob(person_name+"*.wav")):
            vec, seqlen = wav2cubes(wav_file)
            if idx_wav == 0 and (idx_person%FILE_SIZE ==0) :
                speechs = vec
                labels = [idx_person]
                seqs = [seqlen]
            else:
                speechs = np.concatenate((speechs, vec))
                labels.append(idx_person)
                seqs.append(seqlen)
        names.append(person_name.split('/')[-2])
        print(len(speechs), names[-1])

        if idx_person % FILE_SIZE == FILE_SIZE-1:
            h5f = h5py.File('data_lmfe/data_%s.h5'%(idx_person//FILE_SIZE), 'w')
            h5f.create_dataset('speechs', data=speechs)
            h5f.create_dataset('labels', data=labels)
            h5f.create_dataset('seqs', data=seqs)
            h5f.close()
            del speechs, labels
            print("%s - saved"%(idx_person//FILE_SIZE))
            print("%s sec" %(time.time() - start_time))
            start_time = time.time()


def main():

    make_numpy_file("/home/aksdmj/dataset/voxceleb1/voxceleb1_wav/")

    h5f = h5py.File('data_array/data_40.h5','r')
    print(h5f['utterances'][:].shape)
    print(len(h5f['labels'][:]))
    print(len(h5f['seqlen'][:]))



if __name__ == "__main__":
    main()
import naoqi
import queue
import numpy as np
import time
import sys
import speechpy
import queue


class SoundReceiverModule(naoqi.ALModule):
    def __init__(self, strModuleName, strNaoIp, port, rate, max_q_size):
        try:
            naoqi.ALModule.__init__(self, strModuleName)
            self.BIND_PYTHON(self.getName(), "callback")
            self.strNaoIp = strNaoIp
            self.rate = rate
            self.port = int(port)
            self.outfile = None
            self.aOutfile = [None] * (4 - 1)
            # ASSUME max nbr channels = 4
            self.q = queue.Queue(max_q_size)
            self.closed = True

        except BaseException, err:
            print(
                    "abcdk.naoqitools.SoundReceiverModule: loading error: %s"
                    % str(err))

    # __init__ - end
    def __del__(self):
        print("abcdk.SoundReceiverModule.__del__: cleaning everything")
        self.stop()

    def start(self):
        print("start... %s" % self.strNaoIp)
        audio = naoqi.ALProxy("ALAudioDevice", self.strNaoIp, self.port)
        nNbrChannelFlag = 3
        # ALL_Channels: 0,  AL::LEFTCHANNEL: 1, AL::RIGHTCHANNEL: 2; AL::FRONTCHANNEL: 3  or AL::REARCHANNEL: 4.
        nDeinterleave = 0
        nSampleRate = self.rate
        audio.setClientPreferences(self.getName(), nSampleRate,
                                   nNbrChannelFlag, nDeinterleave)
        audio.subscribe(self.getName())
        self.closed = False
        print("SoundReceiver: started!")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def stop(self):
        print("SoundReceiver: stopping... %s" % self.strNaoIp)
        audio = naoqi.ALProxy("ALAudioDevice", self.strNaoIp, self.port)
        audio.unsubscribe(self.getName())
        self.closed = True
        self.q.queue.clear()
        print("SoundReceiver: stopped!")
        if self.outfile is not None:
            self.outfile.close()

    # stop but not erasing last queue
    def pause(self):
        print("SoundReceiver: pausing... %s" % self.strNaoIp)
        audio = naoqi.ALProxy("ALAudioDevice", self.strNaoIp, self.port)
        audio.unsubscribe(self.getName())
        self.closed = True
        print("SoundReceiver: paused!")
        if self.outfile is not None:
            self.outfile.close()

    def processRemote(self, nbOfChannels, nbrOfSamplesByChannel, aTimeStamp,
                      buffer):
        """
        This is THE method that receives all the sound buffers from the "ALAudioDevice" module
        """
        print("hello")
        if self.q.full():
            self.q.get_nowait()
        self.q.put(buffer)

    def version(self):
        return "0.6"


def receiver_initialize(ip, port):
    myBroker = naoqi.ALBroker("myBroker",
                              "0.0.0.0",  # listen to anyone
                              0,  # find a free port and use it
                              ip,  # parent broker IP
                              int(port))  # parent broker port
    SM = SoundReceiverModule("SM", ip, port, 16000, 200)

    import __main__
    __main__.SM = SM
    return SM


class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk, sound_module):
        self._rate = rate
        self._chunk = chunk
        self.sound_module = sound_module
        self.closed = True

    def __enter__(self):
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        pass

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.

            chunk = self.sound_module.q.get()

            if chunk is None:
                return
            data = [chunk]

            while self.sound_module.q.qsize() < 9:
                continue

            # Now consume whatever other data's still buffered.

            while True and len(data) < 10:
                try:
                    chunk = self.sound_module.q.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            print(len(data))

            npdata = np.fromstring(b''.join(data), dtype=np.int16);

            frames = speechpy.processing.stack_frames(npdata, sampling_frequency=16000, frame_length=0.025,
                                                      frame_stride=0.01,
                                                      zero_padding=True)

            num_coefficient = 40

            power_spectrum = speechpy.processing.power_spectrum(frames, fft_points=2 * num_coefficient)[:, 1:]

            logenergy = speechpy.feature.lmfe(npdata, sampling_frequency=16000, frame_length=0.025, frame_stride=0.01,
                                              num_filters=num_coefficient, fft_length=1024, low_frequency=0,
                                              high_frequency=None)

            feature_cube = np.zeros((10, 20, num_coefficient), dtype=np.float32)

            idx = np.random.randint(logenergy.shape[0] - 30, size=10)
            for num, index in enumerate(idx):
                feature_cube[num, :, :] = logenergy[index:index + 20, :]

            yield feature_cube[None, :, :, :]


def main():
    SM = receiver_initialize(ip="192.168.1.6", port=9559)
    SM.start()
    with MicrophoneStream(16000, 1600, SM) as stream:
        audio_generator = stream.generator()
        while True:
            for content in audio_generator:
                print content.shape

    print "done?"


if __name__ == "__main__" :
    main()

"""Simple python script to determine which PyAudio device index"""

import pyaudio

def check_audio():
    """Function that checks the audio device index, name and input/output channels using the PyAudio package"""

    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        device_index = p.get_device_info_by_index(i)['index']
        device_name = p.get_device_info_by_index(i)['name']
        n_channels_in = p.get_device_info_by_index(i)['maxInputChannels']
        n_channels_out = p.get_device_info_by_index(i)['maxOutputChannels']

        print(device_index, device_name, "\tChannels in : {} \t Channels out : {}".format(n_channels_in, n_channels_out))

    p.terminate()

if __name__ == "__main__":
    check_audio()
import numpy as np
import scipy.fftpack as fftpack


# Temporal bandpass filter with Fast-Fourier Transform
def fft_filter(video, freq_min, freq_max, fps):
    fft = fftpack.fft(video, axis=0)
    frequencies = fftpack.fftfreq(video.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - freq_min)).argmin()
    bound_high = (np.abs(frequencies - freq_max)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    iff = fftpack.ifft(fft, axis=0)
    result = np.abs(iff)
    result *= 100# Amplification factor

    return result, fft, frequencies

def fft_filter1(video, freq_min, freq_max, fps):
    fft = fftpack.fft(video, axis=0)
    frequencies = fftpack.fftfreq(len(video), d=1.0 / fps)
    bound_low = (np.abs(frequencies - freq_min)).argmin()
    bound_high = (np.abs(frequencies - freq_max)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0


    return fft, frequencies


def Imeang(img):
    g = img[:,:,1]
    ave = np.average(g)
    return ave

def Imeanb(img):
    b = img[:,:,0]
    ave = np.average(b)
    return ave

def Imeanr(img):
    r = img[:,:,2]
    ave = np.average(r)
    return ave


#!/usr/bin/python

__date__ = "12/24/15"
__credits__ = ["Mohamad Kachuee"]
__maintainer__ = "Mohamad Kachuee"
__email__ = "m.kachuee@gmail.com"

import sys
import time
import pdb

import numpy as np
import scipy.signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pywt

def denoise_wavelet(input_signal, fs_in, fs_out=1000.0, interp='cubic'):
    """
    Denoise and filter using DWT.
    """
    # resample the input
    x = np.arange(np.floor(len(input_signal)*fs_out / fs_in), dtype=np.double) / float(fs_out)
    xp = np.arange(len(input_signal), dtype=np.double) / float(fs_in)
    fp = input_signal
    if interp == 'linear':
        signal_resampled = np.interp(x, xp, fp)
    elif interp == 'cubic':
        f = interp1d(xp, fp, bounds_error=False, fill_value=0, kind='cubic')
        signal_resampled = f(x)
    # dwt decomposition
    coeffs = pywt.wavedec(signal_resampled, 'db8', level=10)
    
    # remove low-freq componenets
    coeffs[0] = coeffs[0] * 0.0

    # remove high freq noise
    for ind_c in range(4,11):
        coeffs[ind_c] = coeffs[ind_c] * 0.0

    # thresholding


    # reconstruct
    signal_reconstructed = pywt.waverec(coeffs, 'db8')

    return signal_reconstructed


def filter(input_signal, fs_in, fs_out=1000.0, interp='cubic', fc1=0.8, fc2=3.0):
    """
    Resample and filter the input.
    """
    # resample the input
    x = np.arange(np.floor(len(input_signal)*fs_out / fs_in), dtype=np.double) / float(fs_out)
    xp = np.arange(len(input_signal), dtype=np.double) / float(fs_in)
    fp = input_signal
    if interp == 'linear':
        signal_resampled = np.interp(x, xp, fp)
    elif interp == 'cubic':
        f = interp1d(xp, fp, bounds_error=False, fill_value=0, kind='cubic')
        signal_resampled = f(x)
    
    # create a filter
    (bl, al) = scipy.signal.butter(N=5, Wn=2*fc2/fs_out, btype='lowpass')
    (bh, ah) = scipy.signal.butter(N=5, Wn=2*fc1/fs_out, btype='highpass')
    signal_filtered = scipy.signal.lfilter(bl, al, signal_resampled)
    signal_filtered = scipy.signal.lfilter(bh, ah, signal_filtered)

    return signal_filtered


def filter_fir(
        input_signal, fs_in, fs_out=1000.0, interp='cubic', 
        fc1=0.7, fc2=3.0, numtaps=91, plot=True):
    """
    Resample and filter the input.
    """
    
    # create a filter
    cutoff = [fc1, fc2]# [0.7,3]
    coeffs = scipy.signal.firwin(
            numtaps=numtaps, cutoff=cutoff, 
            window='blackman', pass_zero=False, 
            scale=True, nyq=fs_in/2.0)
    
    if plot:
        plt.ion()
        plt.figure()
        #Frequency response
        mfreqz(b=coeffs, a=1, fs=fs_in)
        plt.draw()
    
    # apply the filter
    signal_filtered = np.convolve(input_signal, coeffs, 'valid')

    # resample the filtered input
    x = np.arange(
        np.floor(len(signal_filtered)*fs_out / fs_in), 
        dtype=np.double) / float(fs_out)
    xp = np.arange(len(signal_filtered), dtype=np.double) / float(fs_in)
    fp = signal_filtered
    if interp == 'linear':
        signal_resampled = np.interp(x, xp, fp)
    elif interp == 'cubic':
        f = interp1d(xp, fp, bounds_error=False, fill_value=0, kind='cubic')
        signal_resampled = f(x)
    
    return signal_resampled

def filter_iir_fwbw(
        input_signal, fs_in, fs_out=1000.0, interp='linear', 
        fc1=0.7, fc2=3.0, degree=5, plot=True):
    """
    Resample and filter the input.
    """
    
    # create a filter
    (coeffs_b, coeffs_a) = scipy.signal.butter(N=degree, 
            Wn=[2.0*fc1/fs_in, 2.0*fc2/fs_in], btype='bandpass')
    if plot:
        plt.ion()
        plt.figure()
        #Frequency response
        mfreqz(b=coeffs_b, a=coeffs_a, fs=fs_in)
        plt.draw()
    
    # apply the filter
    signal_filtered = scipy.signal.lfilter(coeffs_b, coeffs_a, input_signal, axis=0)
    signal_filtered = scipy.signal.lfilter(coeffs_b, coeffs_a, signal_filtered[::-1], axis=0)[::-1]
    
    # resample the filtered input
    x = np.arange(
        np.floor(len(signal_filtered)*fs_out / fs_in), 
        dtype=np.double) / float(fs_out)
    xp = np.arange(len(signal_filtered), dtype=np.double) / float(fs_in)
    fp = signal_filtered
    f = interp1d(xp, fp, bounds_error=False, fill_value=0, kind=interp, axis=0)
    signal_resampled = f(x)
    
    return signal_resampled

#Plot frequency and phase response
def mfreqz(b, a, fs):
    w,h = scipy.signal.freqz(b,a)
    h_dB = 20 * np.log10 (abs(h))
    plt.subplot(211)
    plt.plot(w*fs/max(w)/2,h_dB)
    plt.ylim(-150, 5)
    plt.ylabel('Magnitude (db)')
    plt.xlabel(r'Frequency')
    plt.title(r'Frequency response')
    plt.subplot(212)
    h_Phase = np.unwrap(np.arctan2(np.imag(h),np.real(h)))
    plt.plot(w*fs/np.max(w),h_Phase)
    plt.ylabel('Phase (radians)')
    plt.xlabel(r'Frequency')
    plt.title(r'Phase response')
    plt.subplots_adjust(hspace=0.5)

#Plot step and impulse response
def impz(b,a=1):
    l = len(b)
    impulse = repeat(0.,l); impulse[0] =1.
    x = np.arange(0,l)
    response = scipy.signal.lfilter(b,a,impulse)
    plt.subplot(211)
    plt.stem(x, response)
    plt.ylabel('Amplitude')
    plt.xlabel(r'n (samples)')
    plt.title(r'Impulse response')
    plt.subplot(212)
    step = np.cumsum(response)
    plt.stem(x, step)
    plt.ylabel('Amplitude')
    plt.xlabel(r'n (samples)')
    plt.title(r'Step response')
    plt.subplots_adjust(hspace=0.5)

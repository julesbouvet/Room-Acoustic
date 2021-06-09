import numpy as np
import matplotlib.pyplot as plt
import librosa
import time

from numpy.core.defchararray import find

time_start = time.time()

###########################
#     Read the signal     #
###########################

path = 'HW3/wav_files/test.wav'
x, Fs = librosa.load(path, sr=None)

###########################
#     Plot the signal     #
###########################

def print_plot_play(x, Fs, text=''):
    """1. Prints information about an audio singal, 2. plots the waveform, and 3. Creates player
    
    Notebook: C1/B_PythonAudio.ipynb
    
    Args: 
        x: Input signal
        Fs: Sampling rate of x    
        text: Text to print
    """
    print('\n \n %s Fs = %d, x.shape = %s, x.dtype = %s \n' % (text, Fs, x.shape, x.dtype))
    plt.figure(figsize=(8, 2))
    plt.plot(x, color='gray')
    plt.xlim([0, x.shape[0]])
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()
    
# print_plot_play(x=x, Fs=Fs, text='WAV file: ')


###################################################
#                                                 #
#        Computing the Schroeder Curve (SC)       #
#                                                 #
###################################################


def EDC(sq_signal, fs, t):
    """
    Compute Energy Decay Curve (EDC)

    input : sq_signal = impule frequency response at the power 2
            fs = sampling frequency
            t = time (s) where the integral begins

    output : res = integral of sq_signal between t and +inf
    """
    k = int(t*fs)
    res = np.sum(sq_signal[k:-1])
    return res

def SC(signal, fs):
    """
    Compute the Schroeder frequency, ie 10*log10(A/B)
    where A is the integral of sq_signal btw t and +inf 
          B is the integral of sq_signal btw 0 and +inf 
    """
    nb_points = signal.shape[0]
    T = [k/fs for k in range(nb_points)]
    sc = []
    sq_signal = signal**2
    B = EDC(sq_signal, fs, 0)
    for t in T:  
        A = EDC(sq_signal, fs, t)
        sc_t = 10*np.log10(A/B)
        sc.append(sc_t)

    return T, sc

T, sc = SC(x, Fs)


###################################################
#                                                 #
#       Computing T_10, T_20, T_30 and T_160      #
#                                                 #
###################################################


dict_T = {'T_10':(6, -15, -5), 'T_20':(3, -25, -5), 'T_30': (2, -35, -5), 'T_160': 4}

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def compute_reverb_time(sc, parameters, fs):
    """
    parameters = (a, dB_1, db_2)
    T_X = a[t(SC = dB_1) - t(SC = dB_2)]
    """
    a = parameters[0]
    dB_1 = parameters[1]
    dB_2 = parameters[2]

    reverb_time = a*(find_nearest(sc, dB_1) - find_nearest(sc, dB_2))/fs

    return reverb_time


def reverb_time(sc, fs, T_X):
    parameters = dict_T[T_X]
    print(parameters)
    t = compute_reverb_time(sc, parameters, fs)
    print(T_X, " = ", t)
    return t

T_10 = reverb_time(sc, Fs, 'T_10')
T_20 = reverb_time(sc, Fs, 'T_20')
T_30 = reverb_time(sc, Fs, 'T_30')

def plot(sc, T, T_10, T_20, T_30):
    plt.plot(T, sc)
    plt.axvline(T_10)
    plt.axvline(T_20)
    plt.axvline(T_30)
    plt.show()

plot(sc, T, T_10, T_20, T_30)


###################################################
#                                                 #
#            Computing Early Decay Time           #
#                                                 #
###################################################

EDT = 6*find_nearest(sc, -10)/Fs
print('\n \n EDT = ', EDT)

###################################################
#                                                 #
#                  D50 and C80                    #
#                                                 #
###################################################


def D_50_calculation(signal, fs):
    sq_signal = signal**2
    k_50 = int(0.05*fs)
    D_50 = np.sum(sq_signal[:k_50])/ np.sum(sq_signal)
    return D_50

def C_80_calculation(signal, fs):
    sq_signal = signal**2
    k_80 = int(0.08*fs)
    C_80 = 10*np.log10(np.sum(sq_signal[:k_80])/ np.sum(sq_signal[k_80:-1]))
    return C_80

D_50 = D_50_calculation(x, Fs)
C_80 = C_80_calculation(x, Fs)

print('\n \n D_50 = ', D_50)
print('\n \n C_80 = ', C_80)

time_end = time.time()
print('\n Time (in s): ', time_end-time_start)
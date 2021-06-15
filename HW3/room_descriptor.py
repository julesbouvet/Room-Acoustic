import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt

###########################
#     Plot the signal     #
###########################

def print_plot_play(x, Fs, text='', plot=True):
    """1. Prints information about an audio singal, 2. plots the waveform, and 3. Creates player
    
    Notebook: C1/B_PythonAudio.ipynb
    
    Args: 
        x: Input signal
        Fs: Sampling rate of x    
        text: Text to print
    """
    if plot == True:
        print('\n \n %s Fs = %d, x.shape = %s, x.dtype = %s \n' % (text, Fs, x.shape, x.dtype))
        plt.figure(figsize=(8, 2))
        plt.plot(x, color='gray')
        plt.xlim([0, x.shape[0]])
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.show()
    
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


###################################################
#                                                 #
#            Computing T_10, T_20, T_30           #
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

def plot(sc, T, T_10, T_20, T_30, plot):
    if plot ==True:
        plt.plot(T, sc)
        plt.axvline(T_10, c='lightcoral')
        plt.axvline(T_20, c='indianred')
        plt.axvline(T_30, c='darkred')
        plt.show()

###################################################
#                                                 #
#               Computing T_160 * ts              #
#                                                 #
###################################################

def TS(signal, T):
    sq_signal = signal**2
    ts = np.sum(T*sq_signal)/np.sum(sq_signal)
    return ts

def T_160_ts(sc, T, signal):
    SC_16 = sc[find_nearest(T, 0.16)]
    T_160 = -60*0.16/SC_16
    ts = TS(signal, T)
    return T_160*ts

###################################################
#                                                 #
#            Computing Early Decay Time           #
#                                                 #
###################################################

def EDT(sc, fs):
    return 6*find_nearest(sc, -10)/fs

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

###################################################
#                                                 #
#             Octave Band Filtering               #
#                                                 #
###################################################

def octave_filter(center_freq, order, signal, fs):
    band = [center_freq/np.sqrt(2), center_freq*np.sqrt(2)]
    sos = butter(order, band, btype='bandpass', output='sos', fs=fs)
    filtered_signal = sosfilt(sos, signal)
    return filtered_signal

###################################################
#                                                 #
#               Bass Ration (BR)                  #
#                                                 #
###################################################

def BR(x, Fs):
    signal_125 = octave_filter(125, 5, x, Fs)
    signal_250 = octave_filter(250, 5, x, Fs)
    signal_500 = octave_filter(500, 5, x, Fs)
    signal_1000 = octave_filter(1000, 5, x, Fs)

    sc_125 = SC(signal_125, Fs)
    sc_250 = SC(signal_250, Fs)
    sc_500 = SC(signal_500, Fs)
    sc_1000 = SC(signal_1000, Fs)

    T_60_125 = compute_reverb_time(sc_125, (1, -60, 0), Fs)
    T_60_250 = compute_reverb_time(sc_250, (1, -60, 0), Fs)
    T_60_500 = compute_reverb_time(sc_500, (1, -60, 0), Fs)
    T_60_1000 = compute_reverb_time(sc_1000, (1, -60, 0), Fs)

    return T_60_125 + T_60_250 /(T_60_500 + T_60_1000)

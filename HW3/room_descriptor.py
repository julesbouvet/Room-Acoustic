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
    t = compute_reverb_time(sc, parameters, fs)
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

def T_160(sc, T):
    SC_16 = sc[find_nearest(T, 0.16)]
    T160 = -60*0.16/SC_16
    return T160

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

# https://www.engineeringtoolbox.com/octave-bands-frequency-limits-d_1602.html

def octave_filter(center_freq, order, signal, fs):
    band = [center_freq/np.sqrt(2), center_freq*np.sqrt(2)]
    sos = butter(order, band, btype='bandpass', output='sos', fs=fs)
    filtered_signal = sosfilt(sos, signal)
    return filtered_signal

###################################################
#                                                 #
#               Bass Ratio (BR)                   #
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

    br = T_60_125 + T_60_250 /(T_60_500 + T_60_1000)
    print('\n Bass Ratio = ',br)

    pass


#################################
#                               #
#     Synchronize beginning     #
#                               #
#################################

def synchronize(signal):
    """
    input: signal (array)
    output: synchronized_signal = signal but without the silence at the beginning 
    """
    # get the first idx when sound starts
    idx_start = np.where(signal>0.1)[0][0]
    print(idx_start)
    synchronized_signal = signal[idx_start-500:]
    return synchronized_signal


######################################
#                                    #
#     Computing Room Descriptors     #
#                                    #
######################################

def compute_room_descriptors(signal, sr, plot_curve):
    # Schroeder Curve
    T, sc = SC(signal, sr)

    # T_10, T_20, T_30 and plot 
    dict_T = {'T_10':(6, -15, -5), 'T_20':(3, -25, -5), 'T_30': (2, -35, -5)}
    T_10 = reverb_time(sc, sr, 'T_10')
    T_20 = reverb_time(sc, sr, 'T_20')
    T_30 = reverb_time(sc, sr, 'T_30')

    print('\n T_10 = ', T_10)
    print('\n T_20 = ', T_20)
    print('\n T_30 = ', T_30)

    plot(sc, T, T_10, T_20, T_30, plot=plot_curve)

    # T_160 and ts
    T160 = T_160(sc, T)
    ts = TS(signal, T)

    print('\n T_160 = ', T160)
    print('\n ts = ', ts)

    # Early Decay Time
    edt = EDT(sc, sr)
    print('\n \n EDT = ', edt)

    # Definition D50
    D_50 = D_50_calculation(signal, sr)
    print('\n \n D_50 = ', D_50)

    # Clarity C80
    C_80 = C_80_calculation(signal, sr)
    print('\n \n C_80 = ', C_80)

    pass

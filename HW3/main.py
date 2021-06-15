import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/path/to/HW3/')

import room_descriptor as rd


# import room_descriptor as rd
import librosa
import time

time_start = time.time()

###########################
#     Read the signal     #
###########################

path = 'HW3/wav_files/test.wav'
x, Fs = librosa.load(path, sr=None)

###########################
#     Plot the signal     #
###########################

rd.print_plot_play(x=x, Fs=Fs, text='WAV file: ', plot=False)

##################################
#     Filtering Octave bands     #
##################################

octave_center_frequencies = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000]
x_filtered = rd.octave_filter(center_freq=1000, order=5, signal=x, fs=Fs)

######################################
#     Computing Room Descriptors     #
######################################

def compute_room_descriptors(signal, sr):
    # Schroeder Curve
    T, sc = rd.SC(signal, sr)

    # T_10, T_20, T_30 and plot 
    dict_T = {'T_10':(6, -15, -5), 'T_20':(3, -25, -5), 'T_30': (2, -35, -5)}
    T_10 = rd.reverb_time(sc, sr, 'T_10')
    T_20 = rd.reverb_time(sc, sr, 'T_20')
    T_30 = rd.reverb_time(sc, sr, 'T_30')

    rd.plot(sc, T, T_10, T_20, T_30, plot=True)

    # T_160 * ts
    T160_ts = rd.T_160_ts(sc, T, signal)
    print('\n T_160 * ts = ', T160_ts)

    # Early Decay Time
    edt = rd.EDT(sc, sr)
    print('\n \n EDT = ', edt)

    # Definition D50
    D_50 = rd.D_50_calculation(signal, sr)
    print('\n \n D_50 = ', D_50)

    # Clarity C80
    C_80 = rd.C_80_calculation(signal, sr)
    print('\n \n C_80 = ', C_80)


# Bass Ration
br = rd.BR(x, Fs)

# displaying results
compute_room_descriptors(x_filtered, Fs)
print('\n Bass Ration = ',br)

# display time
time_end = time.time()
print('\n Time (in s): ', time_end-time_start)




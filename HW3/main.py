import sys
sys.path.insert(1, '/path/to/HW3/')
import room_descriptor as rd
import librosa
import time

time_start = time.time()

###########################
#     Read the signal     #
###########################

path = 'HW3/Measurements/IR/M7.wav'
x_original, Fs = librosa.load(path, sr=None)

print(x_original.shape)

#################################
#     Synchronize beginning     #
#################################

x = rd.synchronize(x_original)

###########################
#     Plot the signal     #
###########################

rd.print_plot_play(x=x, Fs=Fs, text='WAV file: ', plot=True) 

##################################
#     Filtering Octave bands     #
##################################

octave_center_frequencies = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000]
# x_filtered = rd.octave_filter(center_freq=1000, order=5, signal=x, fs=Fs)
# rd.print_plot_play(x=x_filtered, Fs=Fs, text='WAV file: ', plot=False)

######################################
#     Computing Room Descriptors     #
######################################

print('\n \n Computing room descriptors for BB Signal: \n \n')
rd.compute_room_descriptors(x, Fs, plot_curve=False)
rd.BR(x, Fs)

print('\n ------------------------------------– \n')

for f in octave_center_frequencies:
    print(f'\n \n Computing room descriptors for {f} Octave band Signal: \n \n')
    x_filtered = rd.octave_filter(center_freq=f, order=5, signal=x, fs=Fs)
    rd.compute_room_descriptors(x_filtered, Fs, plot_curve=False)
    print('\n ------------------------------------– \n')


# display time
time_end = time.time()
print('\n Time (in s): ', time_end-time_start)
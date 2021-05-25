import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def triplets(max):
    """
    Return every 3-uplet (a, b, c) possible with 0<a, b, c<max
    """
    triplets = []
    for x in range(max+1):
        for y in range(max+1):
            for z in range(max+1):
                triplets.append((x, y, z))

    return triplets


def get_df_eigenmodes(L1, L2, L3, nb_modes):
    """
    Return a DataFrame designed as follows:

    | mode number | 'nLi' | 'kn1' | 'kn2' | 'kn3' | 'wn' |

    with the firsts nb_modes eigenmodes

    NB: the mode number is not significant, its aim is only
    to ease the calculation of eigenmodes
    """

    # creat a dict to store nLi, kni and wi
    dict_eigenmodes ={}

    # load every combinaison possible of 3-uplets 
    modes = triplets(int(np.cbrt(nb_modes))*2)

    mode_number = 0

    # get nLi, kn1, kn2, kn3 and wn for all modes
    for mode_idx in modes:
        nL1 = mode_idx[0]
        nL2 = mode_idx[1]
        nL3 = mode_idx[2]

        kn1 = nL1*np.pi / L1
        kn2 = nL2*np.pi / L2
        kn3 = nL3*np.pi / L3

        wn = c0 * np.sqrt(kn1**2 +kn2**2 +kn3**2)

        dict_eigenmodes[mode_number]={}

        dict_eigenmodes[mode_number]['nLi']= mode_idx
        dict_eigenmodes[mode_number]['kn1']= kn1
        dict_eigenmodes[mode_number]['kn2']= kn2
        dict_eigenmodes[mode_number]['kn3']= kn3
        dict_eigenmodes[mode_number]['wn']= wn

        mode_number += 1
    
    # convert the dict into a DataFrame to ease data manipulation 
    # at this stage, each row represents a feature (nLi, kn1, wn, ...) and each column a mode 
    df_dict_eigenmodes = pd.DataFrame(dict_eigenmodes)

    # sort the columns by the value of wn by ascending order
    df_sorted = df_dict_eigenmodes.sort_values(by= 'wn', axis=1)

    # transpose so that each row corresponds to a mode (and so each column to a feature)
    df_transpose = df_sorted.transpose()

    # select only the first nb_modes eigenmodes (so the first nb_modes rows)
    df_eigenmodes = df_transpose.iloc[1: nb_modes+1]

    return df_eigenmodes


def get_pressure__(fmin, fmax, fstep, nb_eigenmodes, L, x, y, reverb):
    """
    Compute the pressure value according to formula (3.84)
    """

    # get the values of Li, xi and yi
    L1, L2, L3 = L[0], L[1], L[2]
    x1, x2, x3 = x[0], x[1], x[2]
    y1, y2, y3 = y[0], y[1], y[2]

    # compute the frequency and angular frequency ranges
    list_f = np.arange(fmin, fmax, fstep).astype(float)
    list_w = list(2*np.pi*list_f)

    # get the nb of points we are going to calculate 
    nb_pts = len(list_w)

    # create a list in which we will store pressure value 
    list_p = []

    # load eigenmodes in order to get kn1, kn2, kn3 and wn
    df_eigenmodes = get_df_eigenmodes(L1, L2, L3, nb_eigenmodes)

    kn1 = np.array(df_eigenmodes['kn1']).astype(float)
    kn2 = np.array(df_eigenmodes['kn2']).astype(float)
    kn3 = np.array(df_eigenmodes['kn3']).astype(float)
    wn = np.array(df_eigenmodes['wn'])

    print(df_eigenmodes)
    print(' \n Eigenmodes are determined ! \n')

    # compute the product cos(kni*xi) and cos(kni*yi) for each mode
    prod_x = np.multiply(np.multiply(np.cos(kn1*x1),np.cos(kn2*x2)),np.cos(kn3*x3))
    prod_y= np.multiply(np.multiply(np.cos(kn1*y1),np.cos(kn2*y2)),np.cos(kn3*y3))

    prod = np.multiply(prod_x,prod_y)

    # create idx variable to know at what stage of the computation we are
    idx = 1

    for w in list_w:

        # compute the sum on every eigenmodes for every angular frequency
        sum_n = 0
        for n in range(nb_eigenmodes):

            wn_n = wn[n]

            denum = wn_n**2 + 2j*reverb*wn_n - w**2

            sum_n += prod[n]/denum

        p = (8*c0**2/(L1*L2*L3))*sum_n 
        list_p.append(p)

        if idx%50 == 0:
            print(idx, '/', nb_pts)

        idx += 1 


    return list_p, list_f


if __name__ == '__main__':

    # size of the room (L1, L2, L3)
    L = (5, 4, 3)
    
    # location of the receiver
    x = (4.5, 0.5, 0.5)

    # location of the source
    y = (0.1, 0.1, 0.1)

    # parameters sound velocity and reverberation time
    c0 = 343
    T60 = 2


    p, f = get_pressure__(fmin = 0, fmax = 512, fstep = 1, nb_eigenmodes = 1000, L = L, x = x, y = y, reverb = T60)
   
    # convert the pressure in dB
    p_dB = 20*np.log10(np.abs(p))

    plt.plot(f, p_dB)
    plt.xlim([10, 500])
    plt.xscale("log")
    plt.title(f'Frequency response of a rectangular room of size {L} with rigid walls')
    plt.xlabel('f [Hz]')
    plt.ylabel('p [dB]')
    plt.show()

def get_eigenmodes_dict(L1, L2, L3, nb_modes):
    """
    Return a dict with {modes: wn}
    """
    dict_eigenmodes ={}
    modes = triplets(nb_modes)

    # get wn for all modes

    for mode_idx in modes:
        nL1 = mode_idx[0]
        nL2 = mode_idx[1]
        nL3 = mode_idx[2]


        wn = c0 * np.sqrt((nL1/L1)**2 +(nL2/L2)**2 +(nL3/L3)**2)/ np.pi

        dict_eigenmodes[mode_idx]= wn 
    
    # sort modes
    sorted_eigenvalues = dict(sorted(dict_eigenmodes.items(), key=lambda item: item[1]))
    

    # return only the first n_modes
    n_eigenmodes = {k: sorted_eigenvalues[k] for k in list(sorted_eigenvalues)[1:nb_modes+1]}

    return n_eigenmodes


def get_pressure(fmin, fmax, fstep, nb_eigenmodes, L, x, y, dump):

    L1, L2, L3 = L[0], L[1], L[2]
    x1, x2, x3 = x[0], x[1], x[2]
    y1, y2, y3 = y[0], y[1], y[2]

    w_min = 2*np.pi*fmin
    w_max = 2*np.pi*fmax
    w_step = 2*np.pi*fstep

    list_w = list(np.arange(w_min, w_max, w_step))
    nb_pts = len(list_w)
    list_p = []

    df_eigenmodes = get_df_eigenmodes(L1, L2, L3, nb_eigenmodes)

    kn1 = df_eigenmodes['kn1']
    kn2 = df_eigenmodes['kn2']
    kn3 = df_eigenmodes['kn3']
    wn = df_eigenmodes['wn']


    print(' Eigenmodes are determined ! \n')

    idx = 1

    for w in list_w:
        sum_n = 0
        for n in range(nb_eigenmodes):

            kn1_n = kn1.iloc[n]
            kn2_n = kn2.iloc[n]
            kn3_n = kn3.iloc[n]
            wn_n = wn.iloc[n]

            product_x = np.cos(x1 * kn1_n) * np.cos(x2 * kn2_n) * np.cos(x3 * kn3_n)
            product_y = np.cos(y1 * kn1_n) * np.cos(y2 * kn2_n) * np.cos(y3 * kn3_n)

            denum = wn_n**2 + 2j*dump*wn_n - w**2

            sum_n += (product_x*product_y)/denum

        p = (8*c0**2/(L1*L2*L3))*sum_n 
        list_p.append(p)

        p_dB = 20*np.log10(np.abs(list_p))

        if idx%50 == 0:
            print(idx, '/', nb_pts)

        idx += 1 

    return list_p, p_dB, list_w
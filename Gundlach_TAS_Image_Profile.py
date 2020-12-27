"""
Last updated: Jan 31st, 2020

Origin-like image profile
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def SVD(file, n, tstart = None, tstop = None, wstart = None, wstop = None):
    # n is the number of singular values that you want to plot
    """
    Source:
        https://cmdlinetips.com/2019/05/singular-value-decomposition-svd-in-python/
    """

    data = np.array(pd.read_csv(file, header = None, delim_whitespace = False))
    t = data[1:, 0]
    w = data[0, 1:]
    a = data[1:, 1:] # convert to mOD

    if tstart == None:
        tstart = t[0]
    else:
        pass

    if tstop == None:
        tstop = t[-1]
    else:
        pass

    if wstart == None:
        wstart = w[0]
    else:
        pass

    if wstop == None:
        wstop = w[-1]
    else:
        pass

    tstart, tstop, wstart, wstop = [tstart, tstop, wstart, wstop]

    tstart_idx = find_nearest(t, tstart)[0]
    tstop_idx = find_nearest(t, tstop)[0]
    wstart_idx = find_nearest(w, wstart)[0]
    wstop_idx = find_nearest(w, wstop)[0]

    t = t[tstart_idx:tstop_idx]
    w = w[wstart_idx:wstop_idx]
    # convert to mOD
    a = 1000 * a[tstart_idx:tstop_idx, wstart_idx:wstop_idx]    

    """
    u = left-singular vectors, (m, m) = (184, 184) --> times
    s = singular values, (m,) = (184,)
    v = right-singular vector, (n, n) = (1024, 1024) --> wavelengths
    """

    u, s, v = np.linalg.svd(a, full_matrices = True)

    var_explained = np.round(s**2 / np.sum(s**2), decimals = 3)

    plt.figure(1)
    sns.barplot(x = list(range(1, len(var_explained)+1)), y = var_explained)
    plt.xlim(0, n+1) # plot a little further than n
    plt.xlabel('SVs', fontsize = 16)
    plt.ylabel('Percent Variance Explained', fontsize = 16)

    for i in range(n):
        plt.figure(2)
        plt.subplot(2, 1, 1)
        plt.plot(t, u[:, i], label = i+1)
        print(i+1)
        plt.title('Left Singular Vectors')
        plt.xlabel('Delay Time (fs)')
    plt.legend()
    for i in range(n):
        plt.figure(2)
        plt.subplot(2, 1, 2)
        plt.plot(w, v[i, :], label = i+1)
        plt.title('Right Singular Vectors')
        plt.xlabel('Wavelength (nm)')
    plt.legend()
    plt.show()

    return u, s, v


def Time_Zero(file, tstart, tstop, wstart, wstop):
    data = np.array(pd.read_csv(file, header = None, delim_whitespace = False))
    time = data[1:, 0]
    wavelength = data[0, 1:]
    amplitude = data[1:, 1:]

    t = time
    w = wavelength
    a = amplitude

    if tstart == None:
        tstart = t[0]
    else:
        pass

    if tstop == None:
        tstop = t[-1]
    else:
        pass

    if wstart == None:
        wstart = w[0]
    else:
        pass

    if wstop == None:
        wstop = w[-1]
    else:
        pass

    tstart, tstop, wstart, wstop = [tstart, tstop, wstart, wstop]

    tstart_idx = find_nearest(t, tstart)[0]
    tstop_idx = find_nearest(t, tstop)[0]
    wstart_idx = find_nearest(w, wstart)[0]
    wstop_idx = find_nearest(w, wstop)[0]

    t = t[tstart_idx:tstop_idx]
    w = w[wstart_idx:wstop_idx]
    # convert to mOD
    a = 1000 * a[tstart_idx:tstop_idx, wstart_idx:wstop_idx]
    
    t_shift = []
    a_shift = []
    for i in range(len(w)):
        print(w[i], 'nm')
        print('a', np.size(a[:, i]))
        print('b', np.size(abs(a[:, i])))
        a_max = np.max(abs(a[:, i])) # max point of the absorption vector, abs() in case of bleach
        a_TZ = 0.6 * a_max # time-zero guess of 60% of the rise, positive number because of abs()
        t_ind, a_TZ = find_nearest(abs(a[:, i]), a_TZ) # both values are positive
        t_TZ = t[t_ind] # a[t_idx, wl] can be negative if spectrum is of a bleach signal
        t = t - t_TZ # shift time to make time-zero occur at t = 0
        a = a[:-t_ind, i] # truncate absoprtion vector to be same length as corresponding time vector
        t_shift.append(t)
        a_shift.append(a)
        # if i == 0:
        #     t_shift = t
        #     a_shift = a
        # else:
        #     t_shift = np.column_stack((t_shift, t)) # collect all of the shifted time vectors
        #     a_shift = np.column_stack((a_shift, a)) # collect all of the truncated absorption vectors
        
        print('a', np.size(a[:, i]))
        print('b', np.size(abs(a[:, i])))

    # truncate all of the time and absorption vecotrs to have the size of the smallest vector
    column_len_list = []
    for i in t_shift:
        column_len_list.append(len(i))
    min_len = min(column_len_list)
    
    # for i in range(len(t_shift[0, :])):
    #     column_len_list.append(len(t_shift[:, i]))
    # min_idx, min_len = find_nearest(column_len_list, min(column_len_list))

    stop_idx = min_len - 1

    # t_truncated = np.array([])
    # a_truncated = np.array([])
    # for i in range(len(t_shift[0, :])):
    #     column = t_shift[:, i]
    #     column = column[:stop_idx]
    #     if i == 0:
    #         t_truncated = column
    #     else:
    #         t_truncated = np.column_stack((t_truncated, column))
    # for i in range(len(a_shift[0, :])):
    #     column = a_shift[:, i]
    #     column = column[:stop_idx]
    #     if i == 0:
    #         a_truncated = column
    #     else:
    #         a_truncated = np.column_stack((a_truncated, column))

    
    t_truncated = []
    a_truncated = []
    
    for i in range(len(t_shift)):
        t_shift[i] = t_shift[i][:stop_idx]
        a_shift[i] = a_shift[i][:stop_idx]
        t_truncated.append(t_shift[i])
        a_truncated.append(a_shift[i])
        
    # return vectors to be used for Image_Profile_TZ
    return t_truncated, w, a_truncated

class Cursor(object):
    """
    Source:
        https://matplotlib.org/3.1.1/gallery/misc/cursor_demo_sgskip.html
    """
    def __init__(self, file, ax, t, w, a):
        self.file = file
        self.ax = ax
        self.t = t
        self.w = w
        self.a = a
        self.lx = ax.axhline(color='k')  # the horiz line
        self.ly = ax.axvline(color='k')  # the vert line

        # text location in axes coords
        self.txt = ax.text(0.7, 0.9, '', transform=ax.transAxes)

    def mouse_move(self, event):
        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata
        # update the line positions
        self.lx.set_ydata(y)
        self.ly.set_xdata(x)

        self.txt.set_text('x=%1.2f, y=%1.2f' % (self.t[int(x)], y))
        self.ax.figure.canvas.draw()

    def mouse_click(self, event):
        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata
        # update the line positions
        self.lx.set_ydata(y)
        self.ly.set_xdata(x)

        self.txt.set_text('x=%1.2f, y=%1.2f' % (self.t[int(x)], y))
        self.ax.figure.canvas.draw()

        t_idx = find_nearest(self.t, self.t[int(x)])[0]
        w_idx = find_nearest(self.w, y)[0]

        plt.figure(2)
        # plt.subplot(321)
        plt.subplot(211)
        # convert to mOD
        plt.plot(self.t, 1000 * self.a[:, w_idx], label = '%.f nm' % self.w[w_idx])
        plt.xlabel('Pump Delay (fs)')
        plt.ylabel('ΔA (mOD)')
        plt.title('Trace')
        plt.legend()

        # plt.subplot(322)
        plt.subplot(212)
        # convert to mOD
        plt.plot(self.w, 1000 * self.a[t_idx, :], label = '%.f fs' % self.t[t_idx])
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('ΔA (mOD)')
        plt.title('Spectrum')
        plt.legend()

        # plt.figure(5)
        # plt.plot(self.t, 1000 * self.a[:, w_idx], label = '%s, %.f nm' % tuple((self.file, self.w[w_idx])))
        # plt.legend()
        # plt.figure(6)
        # plt.plot(self.w, 1000 * self.a[t_idx, :], label = '%s, %.f fs' % tuple((self.file, self.t[t_idx])))
        # plt.legend()


        plt.show()

        return self.t, 1000 * self.a[:, w_idx], self.w, 1000 * self.a[t_idx, :]

def surf(file, tstart = None, tstop = None, wstart = None, wstop = None):
    data = np.array(pd.read_csv(file, header = None, delim_whitespace = False))
    time = data[1:, 0]
    wavelength = data[0, 1:]
    amplitude = data[1:, 1:]

    t = time
    w = wavelength
    a = amplitude

    if tstart == None:
        tstart = t[0]
    else:
        pass

    if tstop == None:
        tstop = t[-1]
    else:
        pass

    if wstart == None:
        wstart = w[0]
    else:
        pass

    if wstop == None:
        wstop = w[-1]
    else:
        pass

    tstart, tstop, wstart, wstop = [tstart, tstop, wstart, wstop]

    tstart_idx = find_nearest(t, tstart)[0]
    tstop_idx = find_nearest(t, tstop)[0]
    wstart_idx = find_nearest(w, wstart)[0]
    wstop_idx = find_nearest(w, wstop)[0]

    t = t[tstart_idx:tstop_idx]
    w = w[wstart_idx:wstop_idx]
    # convert to mOD
    a = 1000 * a[tstart_idx:tstop_idx, wstart_idx:wstop_idx]

    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    T, W = np.meshgrid(t, w)
    ax.plot_surface(T, W, a.T, cmap = cm.rainbow)
    plt.show()

    return

def surf_compare(filelist, tstart = None, tstop = None, wstart = None, wstop = None):
    a_list = []
    for file in filelist:
        data = np.array(pd.read_csv(file, header = None, delim_whitespace = False))
        time = data[1:, 0]
        wavelength = data[0, 1:]
        amplitude = data[1:, 1:]

        t = time
        w = wavelength
        a = amplitude

        if tstart == None:
            tstart = t[0]
        else:
            pass

        if tstop == None:
            tstop = t[-1]
        else:
            pass

        if wstart == None:
            wstart = w[0]
        else:
            pass

        if wstop == None:
            wstop = w[-1]
        else:
            pass

        tstart, tstop, wstart, wstop = [tstart, tstop, wstart, wstop]

        tstart_idx = find_nearest(t, tstart)[0]
        tstop_idx = find_nearest(t, tstop)[0]
        wstart_idx = find_nearest(w, wstart)[0]
        wstop_idx = find_nearest(w, wstop)[0]

        t = t[tstart_idx:tstop_idx]
        w = w[wstart_idx:wstop_idx]
        # convert to mOD
        a = 1000 * a[tstart_idx:tstop_idx, wstart_idx:wstop_idx]
        a_list.append(a)

    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    T, W = np.meshgrid(t, w)
    for a in a_list:
        ax.plot_surface(T, W, a.T, cmap = cm.rainbow)
    plt.show()

    return


def Image_Profile(file, tstart = None, tstop = None, wstart = None, wstop = None):

    data = np.array(pd.read_csv(file, header = None, delim_whitespace = False))
    time = data[1:, 0]
    wavelength = data[0, 1:]
    amplitude = data[1:, 1:]

    t = time
    w = wavelength
    a = amplitude

    if tstart == None:
        tstart = t[0]
    else:
        pass

    if tstop == None:
        tstop = t[-1]
    else:
        pass

    if wstart == None:
        wstart = w[0]
    else:
        pass

    if wstop == None:
        wstop = w[-1]
    else:
        pass

    tstart, tstop, wstart, wstop = [tstart, tstop, wstart, wstop]

    tstart_idx = find_nearest(t, tstart)[0]
    tstop_idx = find_nearest(t, tstop)[0]
    wstart_idx = find_nearest(w, wstart)[0]
    wstop_idx = find_nearest(w, wstop)[0]

    t = t[tstart_idx:tstop_idx+1]
    w = w[wstart_idx:wstop_idx+1]
    a = a[tstart_idx:tstop_idx+1, wstart_idx:wstop_idx+1]
    a_map = 1000 * a.T[::-1] # convert to mOD and flip/rotate map to match axes


    sns.set()
    sns.set_context('paper')
    fig, ax= plt.subplots()
    cursor = Cursor(file, ax, t, w, a)
    fig.canvas.mpl_connect('motion_notify_event', cursor.mouse_move)
    fig.canvas.mpl_connect('button_press_event', cursor.mouse_click)

    # fig.subplots_adjust(left = 0.25, bottom = -0.5)
    t_placement = np.linspace(0, len(t), len(t))
    plt.imshow(a_map, cmap=plt.get_cmap('rainbow'), extent = [0, len(t), w[0], w[-1]], aspect = 'auto')
    t_ticks_labels = []
    t_ticks = np.linspace(0, len(t), 10)
    print(t_ticks)
    for i in t_ticks:
        i = int(i)
        if i == len(t):
            t_ticks_labels.append(int(t[i-1]))
        else:
            t_ticks_labels.append(int(t[i]))
    plt.xticks(ticks = t_ticks, labels = t_ticks_labels)
    plt.xlabel('Pump Delay (fs)')
    plt.ylabel('Wavelength (nm)')
    plt.title('ΔA (mOD)')
    plt.colorbar()
    plt.show()

    return

def Image_Profile_TZ(time, wavelength, amplitude, tstart = None, tstop = None, wstart = None, wstop = None):
    # use with Time_Zero()

    t = time
    w = wavelength
    a = amplitude

    if tstart == None:
        tstart = t[0]
    else:
        pass

    if tstop == None:
        tstop = t[-1]
    else:
        pass

    if wstart == None:
        wstart = w[0]
    else:
        pass

    if wstop == None:
        wstop = w[-1]
    else:
        pass

    tstart, tstop, wstart, wstop = [tstart, tstop, wstart, wstop]

    tstart_idx = find_nearest(t, tstart)[0]
    tstop_idx = find_nearest(t, tstop)[0]
    wstart_idx = find_nearest(w, wstart)[0]
    wstop_idx = find_nearest(w, wstop)[0]

    t = t[tstart_idx:tstop_idx+1]
    w = w[wstart_idx:wstop_idx+1]
    a = a[tstart_idx:tstop_idx+1, wstart_idx:wstop_idx+1]
    a_map = 1000 * a.T[::-1] # convert to mOD and flip/rotate map to match axes


    sns.set()
    sns.set_context('paper')
    fig, ax= plt.subplots()
    cursor = Cursor(file, ax, t, w, a)
    fig.canvas.mpl_connect('motion_notify_event', cursor.mouse_move)
    fig.canvas.mpl_connect('button_press_event', cursor.mouse_click)

    # fig.subplots_adjust(left = 0.25, bottom = -0.5)
    t_placement = np.linspace(0, len(t), len(t))
    plt.imshow(a_map, cmap=plt.get_cmap('rainbow'), extent = [0, len(t), w[0], w[-1]], aspect = 'auto')
    t_ticks_labels = []
    t_ticks = np.linspace(0, len(t), 10)
    print(t_ticks)
    for i in t_ticks:
        i = int(i)
        if i == len(t):
            t_ticks_labels.append(int(t[i-1]))
        else:
            t_ticks_labels.append(int(t[i]))
    plt.xticks(ticks = t_ticks, labels = t_ticks_labels)
    plt.xlabel('Pump Delay (fs)')
    plt.ylabel('Wavelength (nm)')
    plt.title('ΔA (mOD)')
    plt.colorbar()
    plt.show()

    return

file = r"C:\Users\joeav\Desktop\13 scan average.csv"
Image_Profile(file)

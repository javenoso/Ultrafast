'''
Main DFWM FFT file
'''
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp1d
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.optimize import curve_fit
from astropy.timeseries import LombScargle
from collections import OrderedDict

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def copy(result):
    # transposes row arrays to columns and copies to clipboard
    result = pd.DataFrame(result.T)
    result.to_clipboard(index = False, header = False)
    return

def exp1(t, y0, A1, t1):
    return y0 + A1*np.exp(-t/t1)

def exp2(t, y0, A1, A2, t1, t2):
    return y0 + A1*np.exp(-t/t1) + A2*np.exp(-t/t2) 

def exp3(t, y0, A1, A2, A3, t1, t2, t3):
    return y0 + A1*np.exp(-t/t1) + A2*np.exp(-t/t2) + A3*np.exp(-t/t3)

def GaussAmp(x, A, x0, FWHM):
    # same as Gaussian functions below in FFT_Fit class, but given Origin GaussAmp name
    # to differentiate it from functions in FFT_Fit class
    return A * np.exp(-4*np.log(2) * ((x - x0)/FWHM)**2)

def takeFirst(elem):
    # Source: https://www.programiz.com/python-programming/methods/built-in/sorted
    return elem[0]

def list_duplicates_of(seq,item):
    # Source: https://stackoverflow.com/questions/5419204/index-of-duplicates-items-in-a-python-list
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item,start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs

def sort_lists(x, y):
    '''
    Sort x and y together based on increasing values of x
    Source:
        https://stackoverflow.com/questions/37111798/how-to-sort-a-list-of-x-y-coordinates
    '''
#    print('x, y', np.shape(x), np.shape(y))
    z = np.column_stack((x, y))
    z = sorted(z, key=lambda k: [k[0], k[1]])
    print(z)
    x2, y2 = zip(*z)
    x2 = list(x2)
    y2 = list(y2)
    print(len(x2), len(y2))
#    y2 = y2[::-1]
    print('x2, y2', np.shape(x2), np.shape(y2))
    return x2, y2
    
#def sort_lists(list1, list2):
#    # zip the two lists into one dictionary (x = keys, y = values)
#    # sort the dictionary and then convert back to lists
#    # keys are the results vector (Amplitude, Postiion, FWHM)
#    # values are the time vector
#    x = dict(zip(list1, list2))
#    sorted_dict = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
#    keys = list(sorted_dict.keys())
#    values = list(sorted_dict.values())
#    return keys, values


def data_avg(x, y, error = False):
    '''
    x vector is the list of time stamps
    y vector is either the data in the t-space or ω-space
    '''
    x = np.array(x)
    y = np.array(y)
    x_new = []
    y_new = []
#    x_new = np.zeros(len(x))
#    y_new = np.zeros(len(y))
    get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if abs(x-y) <= 10]
    
    for i in x:
        xlist = []
        ylist = []
        
        idx = get_indexes(i, x)
#        idx = duplicates(x, i)
        for ix in idx:
            xlist.append(x[ix])
            ylist.append(y[ix])
        
        if error == True:
            xmean = sum(xlist) / len(xlist)
            yylist = []
            for yy in ylist:
                yy = 1 / (yy ** 2)
                yylist.append(yy)
            ymean = np.sqrt(sum(yylist))
        else:
            xmean = np.mean(xlist)
            ymean = np.mean(ylist)
        
#        replace_idx = np.where(x == i)
        x_new.append(xmean)
        y_new.append(ymean)
    
    x_unique = []    
    y_unique = []
    for idx in np.unique(x_new, return_index = True)[1]:
        x_unique = np.append(x_unique, x_new[idx])
        y_unique = np.append(y_unique, y_new[idx])
    return x_unique, y_unique
#    x_new = list(OrderedDict.fromkeys(x_new))
#    y_new = list(OrderedDict.fromkeys(y_new))
#    return x_new, y_new

def all_data_avg(result):
	avg_result = []
	for column in range(len(result[0, :])):
		if column == 0:
			pass
		elif column == 1:
			avg = data_avg(result[:, 0], result[:, column])
			avg_result = avg
		else:
			avg = data_avg(result[:, 0], result[:, column])
			avg_result = np.row_stack((avg_result, avg[1]))
	avg_result = avg_result.T
	return avg_result
        
class Data_Lists:
    # Note that you need to sort lists externally to this object via sort_lists
    def __init__(self, dirname):
        self.dirname = dirname
        
    def importdir(self):
        directory = os.listdir(self.dirname)
        
        beforeT0 = []
        beforeT0on = []
        beforeT0off = []
        beforeT0sub = []
        
        afterT0 = []
        afterT0on = []
        afterT0off = []
        afterT0sub = []
        
        IPoff = []
        IPon = []
        IPsub = []
        
        for file in directory:
            if '_-' in file:
                beforeT0.append(file)
            if '_-' not in file:
                afterT0.append(file)
        for file in beforeT0:
            if 'IPoff' in file:
                beforeT0off.append(file)
            if 'IPon' in file:
                beforeT0on.append(file)
            if 'IPsubtracted' in file:
                beforeT0sub.append(file)
        for file in afterT0:
            if 'IPoff' in file:
                afterT0off.append(file)
            if 'IPon' in file:
                afterT0on.append(file)
            if 'IPsubtracted' in file:
                afterT0sub.append(file)
                
        '''
        functions below to sort lists by time delay taken from here:
        https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
        '''        
        
        def atof(text):
            try:
                retval = float(text)
            except ValueError:
                retval = text
            return retval
        
        def natural_keys(text):
            '''
            alist.sort(key=natural_keys) sorts in human order
            http://nedbatchelder.com/blog/200712/human_sorting.html
            (See Toothy's implementation in the comments)
            '''
            return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]
        
        beforeT0off.sort(key = natural_keys)
        afterT0off.sort(key = natural_keys)
        IPoff.extend(beforeT0off)
        IPoff.reverse()
        IPoff.extend(afterT0off)
        
        beforeT0on.sort(key = natural_keys)
        afterT0on.sort(key = natural_keys)
        IPon.extend(beforeT0on)
        IPon.reverse()
        IPon.extend(afterT0on)
        
        beforeT0sub.sort(key = natural_keys)
        afterT0sub.sort(key = natural_keys)
        IPsub.extend(beforeT0sub)
        IPsub.reverse()
        IPsub.extend(afterT0sub)
        
        off = []
        tstamp_off = []
        for file in IPoff:
            file = self.dirname + '\\' + file
            off.append(file)
            
            lhs1, rhs1 = file.split("scans_")
            lhs2, rhs2 = rhs1.split("T")
            file_tstamp = int(lhs2)
            tstamp_off.append(file_tstamp)
            
        on = []
        tstamp_on = []
        for file in IPon:
            file = self.dirname + '\\' + file
            on.append(file)
            
            lhs1, rhs1 = file.split("scans_")
            lhs2, rhs2 = rhs1.split("T")
            file_tstamp = int(lhs2)
            tstamp_on.append(file_tstamp)
            
        sub = []
        tstamp_sub = []
        for file in IPsub:
            file = self.dirname + '\\' + file
            sub.append(file)
            
            lhs1, rhs1 = file.split("scans_")
            lhs2, rhs2 = rhs1.split("T")
            file_tstamp = int(lhs2)
            tstamp_sub.append(file_tstamp)
        
#        def sort_lists(x, y):
#            '''
#            Source:
#            https://stackoverflow.com/questions/37111798/how-to-sort-a-list-of-x-y-coordinates
#            '''
#            print('HIIIIIIIIIIIIIIIIIIII')
#            z = zip(np.array(x), np.array(y))
#            z = sorted(z, key = lambda k: [k[1], k[0]])
#            x2, y2 = zip(*z)
#            x2 = list(x2)
#            y2 = list(y2)
#            return x2, y2
        
        offdict = dict(zip(off, tstamp_off))
        ondict = dict(zip(on, tstamp_on))
        subdict = dict(zip(sub, tstamp_sub))
        
        def sort_dict(x):
            # sort the dictionary and then convert back to lists
            # keys are the filenames
            # values are the tstamps for each filename
            sorted_dict = {k: v for k, v in sorted(x.items(), key = lambda item: item[1])}
            keys = list(sorted_dict.keys())
            values = list(sorted_dict.values())
            return keys, values
        
        off, tstamp_off = sort_dict(offdict)
        on, tstamp_on = sort_dict(ondict)
        sub, tstamp_sub = sort_dict(subdict)
        
#        off, tstamp_off = sort_lists(off, tstamp_off)
#        on, tstamp_on = sort_lists(on, tstamp_on)
#        sub, tstamp_sub = sort_lists(sub, tstamp_sub)
        
        return off, on, sub, tstamp_off, tstamp_on, tstamp_sub
        

    
        

class FFT:
    def __init__(self, file, cut = None, n = 1, submethod = 4, window = 0, zp = 2**12, LS = False, graphs = False):
        self.file = file
        self.cut = cut
        self.n = n
        self.submethod = submethod
        self.window = window
        self.zp = zp
        self.LS = LS
        self.graphs = graphs
        
#        if type(self.file) == str:
#            print('hi')
#            lhs1, rhs1 = self.file.split("scans_")
#            lhs2, rhs2 = rhs1.split("T")
#            tstamp = int(lhs2)
#            print(tstamp)
#        else:
#            print('bye')
#        return tstamp

    def FFT(self):
        
        # graph_output check
        if self.graphs == True:
            pass
        elif self.graphs == False:
            pass
        else:
            self.graphs == False
            print('The variable for `graphs` must be either True or False. The graphs will not be shown by default.')
        print(self.file)
        data = pd.read_csv(self.file, header = None, delim_whitespace = True)
        time = np.array(data[0])
        amplitude = np.array(data[1])
        
        if self.graphs == True:
            plt.figure(1)
            plt.plot(time, amplitude, label = 'All Data')
            plt.title('Raw Data')
            plt.xlabel('Delay (fs)')
            plt.ylabel('Amplitude')
        else:
            pass
        
        # This is for cutting out part of the data
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx, array[idx]
        
        def cut(time, amplitude, n):
            if n == None:
                return time, amplitude
            else:
                idx = find_nearest(time, n)[0]
                time = time[idx:]
                amplitude = amplitude[idx:]
                return time, amplitude
        
        time, amplitude = cut(time, amplitude, self.cut) 

        if amplitude[-1] % amplitude[-1] != 0:
            # np.nan % np.nan = np.nan
            # for any float x, x % x = 0
            time = time[:-2]
            amplitude = amplitude[:-2]
        else:
            pass
        print('length of time', len(time))
        if self.graphs == True:
            # show cut-off
            plt.figure(1)
            plt.plot(time, amplitude, label = 'After Cutoff')
            plt.title('Raw Data')
            plt.xlabel('Delay (fs)')
            plt.ylabel('Amplitude')
        else:
            pass

        print('self n =', self.n)
        N = int(self.n * len(time))
        repeat = 0
        tvals = np.linspace(time[0], time[-1], N)
        print('tvals length =', len(tvals))
        freqmax = 1/(tvals[1] - tvals[0])
        freqmin = 0
        freq_lowest_observable = 1 / (tvals[-1] - tvals[0]) / (2.99792458E-5)
        print('lowest freq = %.1f 1/cm' % freq_lowest_observable)
        
        # make points evenly spaced via interpolation
        #f = np.array(np.interp(tvals, time, amplitude)) # linear interpolation
        if self.LS is False:
            f = interp1d(time, amplitude, kind = 'cubic') # cubic interpolation
            f = f(tvals)
        else:
            f = amplitude
        print('hhh', len(tvals), len(f))
        if self.graphs == True:
            plt.figure(1)
            plt.scatter(tvals, f, c = 'red', label = 'Interpolated Data')
            plt.legend()
        else:
            pass
        
        print(amplitude[-1])
        print(f[-1])
        '''
        "Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens in 2005
        
        There are two parameters: p for asymmetry and λ for smoothness. 
        Both have to be tuned to the data at hand. 
        We found that generally 0.001 ≤ p ≤ 0.1 is a good choice (for a signal with positive peaks) 
        and 10^2 ≤ λ ≤ 10^9 , but exceptions may occur. 
        In any case one should vary λ on a grid that is approximately linear for log λ.
        '''
        
        def baseline_als(y, lam, p, niter=10):
          L = len(y)
          D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
          w = np.ones(L)
          for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w*y)
            w = p * (y > z) + (1-p) * (y < z)
          return z
        
        # fit a polynomial to flatten FWM signal curve
        
        def poly(x, f, n):
            p = np.polyfit(x, f, n)
            fp = []
            for i in np.arange(n+1):
        #            print(n-i)
        #            print(i)
                
                fp = x**(n-i) * p[i]
            fp = np.sum(fp)
        #        print(fp)
            return fp
        
        def sub(f, method):
            if method == 0:
                fb = baseline_als(f, 10**7, 0.5)
                fsub  = f - fb
                if self.graphs == True:
                    plt.figure(2)
                    plt.plot(tvals, np.zeros(np.shape(f)), 'b--')
                    plt.plot(tvals, f, 'g-')
                    plt.plot(tvals, fb, 'r-')
                    plt.plot(tvals, fsub, 'y-')
                    plt.title('Baseline Subtraction')
                    plt.xlabel('Delay (fs)')
                    plt.ylabel('Amplitude')
                else:
                    pass
            elif method == 1:
                x0 = [f[0], f[0], abs(tvals[-1]-tvals[0])/2, abs(tvals[-1]-tvals[0])/2]
#                bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])
                p = curve_fit(lambda t, A1, A2, t1, t2: A1*np.exp(-t/t1) + A2*np.exp(-t/t2), tvals, f, x0)[0] #, bounds = bounds)[0]
                fb = p[0]*np.exp(-tvals / p[1]) + p[2]*np.exp(-tvals / p[3])
                fsub = f - fb
                if self.graphs == True:
                    plt.figure(3)
                    plt.plot(tvals, np.zeros(np.shape(f)), 'b--')
                    plt.plot(tvals, f, 'g-')
                    plt.plot(tvals, fb, 'r-')
                    plt.plot(tvals, fsub, 'y-')
                    plt.title('Baseline Subtraction')
                    plt.xlabel('Delay (fs)')
                    plt.ylabel('Amplitude')
            elif method == 2:
                x0 = [0, 0, (tvals[-1]-tvals[0])/2]
                bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
                p = curve_fit(lambda t, y0, A1, t1: y0 + A1 * np.exp(-t / t1), tvals, f, x0)[0]
                print(p)
                fb = []
                for t in tvals:
                    fb.append(p[0] + p[1] * np.exp(-t / p[2]))
                fsub = f - fb
                
                if self.graphs == True:
                    plt.figure(2)
                    plt.plot(tvals, np.zeros(np.shape(f)), 'b--')
                    plt.plot(tvals, f, 'g-')
                    plt.plot(tvals, fb, 'r-')
                    plt.plot(tvals, fsub, 'y-')
                    plt.title('Baseline Subtraction')
                    plt.xlabel('Delay (fs)')
                    plt.ylabel('Amplitude')
            elif method == 3:
                x0 = [max(f), max(f), max(f), tvals[-1], tvals[-1]]
                bounds = ([-10, -10, -10, 0, 0], [10, 10, 10, tvals[-1], tvals[-1]])
                p = curve_fit(lambda t, y0, A1, A2, t1, t2: y0 + A1*np.exp(-t/t1) + A2*np.exp(-t/t2), tvals, f, x0, bounds = bounds)[0]                
                fb = []
                for t in tvals:
                    fb.append(p[0] + p[1]*np.exp(-t / p[3]) + p[2]*np.exp(-t / p[4]))
                fsub = f - fb
                
                if self.graphs == True:
                    plt.figure(2)
                    plt.plot(tvals, np.zeros(np.shape(f)), 'b--')
                    plt.plot(tvals, f, 'g-')
                    plt.plot(tvals, fb, 'r-')
                    plt.plot(tvals, fsub, 'y-')
                    plt.title('Baseline Subtraction')
                    plt.xlabel('Delay (fs)')
                    plt.ylabel('Amplitude')
            elif method == 4:
                x0 = [max(f), max(f), max(f), max(f), tvals[-1], tvals[-1], tvals[-1]]
                bounds = ([-10, -10, -10, -10, 0, 0, 0], [10, 10, 10, 10, tvals[-1], tvals[-1], tvals[-1]])
                p = curve_fit(lambda t, y0, A1, A2, A3, t1, t2, t3: y0 + A1*np.exp(-t/t1) + A2*np.exp(-t/t2) + A3*np.exp(-t/t3), tvals, f, x0, bounds = bounds)[0]                
                fb = []
                for t in tvals:
                    fb.append(p[0] + p[1]*np.exp(-t / p[4]) + p[2]*np.exp(-t / p[5]) + p[3]*np.exp(-t / p[6]))
                fsub = f - fb
                
                if self.graphs == True:
                    plt.figure(2)
                    plt.plot(tvals, np.zeros(np.shape(f)), 'b--')
                    plt.plot(tvals, f, 'g-')
                    plt.plot(tvals, fb, 'r-')
                    plt.plot(tvals, fsub, 'y-')
                    plt.title('Baseline Subtraction')
                    plt.xlabel('Delay (fs)')
                    plt.ylabel('Amplitude')
            else:
                p = np.polyfit(tvals, f, 4)
#                fp = tvals*p[0] + p[1]
                # fp = tvals**2*p[0] + tvals*p[1] + p[2]
                # fp = tvals**3*p[0] + tvals**2*p[1] + tvals*p[2] + p[3]
                fp = tvals**4*p[0] + tvals**3*p[1] + tvals**2*p[2] + tvals*p[3] + p[4]
                # fp = tvals**6*p[0] + tvals**5*p[1] + tvals**4*p[2] + tvals**3*p[3] + tvals**2*p[4] + tvals*p[5] + p[6]
                fsub = f - fp
                if self.graphs == True:
                    plt.figure(2)
                    plt.plot(tvals, np.zeros(np.shape(f)), 'b--')
                    plt.plot(tvals, f, 'g-')
                    plt.plot(tvals, fp, 'r-')
                    plt.plot(tvals, fsub, 'y-')
                    plt.title('Baseline Subtraction')
                    plt.xlabel('Delay (fs)')
                    plt.ylabel('Amplitude')
                else:
                    pass
                
            return fsub
        
        # while True:
        #     """
        #     This while loop avoids running into this numpy bug:
        #     raise LinAlgError("SVD did not converge") LinAlgError: SVD did not converge
        #     """
        #     try:
        #         fsub = sub(f, 3)
        #         f = fsub * 1
        #         # windowing and zero-padding
        #         # f = fsub * np.hamming(len(f))
        #         f = fsub * np.kaiser(len(f), 3)
        #         '''
        #         The Kaiser can approximate many other windows by varying the beta parameter.
        #         beta 	Window shape
        #         0 	Rectangular
        #         5 	Similar to a Hamming
        #         6 	Similar to a Hanning
        #         8.6 	Similar to a Blackman
        #         '''
                
        #         # Shift data up/down to zero line
        #         # if f[-1] > 0:
        #         #     f = f - f[-1]
        #         # elif f[-1] < 0:
        #         #     f = f + f[-1]
        #         # else:
        #         #     pass
        #         break
        #     except:
        #         continue
            
        fsub = sub(f, self.submethod)
        
        global GaussUseage
        GaussUseage = False
        if self.window == None:
        	f = fsub * 1
        elif type(self.window) is list:
            GaussUseage = True
            A = self.window[0]
            FWHM = self.window[1] # convert window parameter into a number
            x0 = tvals[find_nearest(fsub, max(abs(fsub)))[0]] # find index of time-zero
            global GaussWindow
            GaussWindow = GaussAmp(tvals, A*max(abs(fsub)), x0, FWHM)
            f = fsub * GaussWindow
            
            plt.figure(3)
            plt.plot(tvals, fsub/max(fsub), label = 'Before Window')
            plt.plot(tvals, GaussWindow/max(GaussWindow), label = 'A = %.1f x Max, FWHM = %.1f' % tuple(self.window))
            
        elif (self.window == 'Bartlett') or (self.window == 'bartlett'):
            print('ba')
            f = fsub * np.bartlett(len(f))
        elif (self.window == 'Blackman') or (self.window == 'blackman'):
            f = fsub * np.blackman(len(f))
        elif (self.window == 'Hamming') or (self.window == 'hamming'):
            f = fsub * np.hamming(len(f))
        elif (self.window == 'Hanning') or (self.window == 'hanning'):
            print('h')
            f = fsub * np.hanning(len(f))
        elif (self.window == 0) or (self.window == 'Rectangle') or (self.window == 'rectangle') or (self.window == 'Rectangular') or (self.window == 'rectangular'):
            f = fsub * np.kaiser(len(f), 0)
            print('Numpy does not have a rectangular window function, so Kaiser 0 is applied instead, which is similar.')
        elif (type(self.window) is int) or (type(self.window) is float):
            f = fsub * np.kaiser(len(f), self.window)
            print('If the window value is zero or less, an empty array is returned. See the documentation: https://numpy.org/doc/stable/reference/generated/numpy.kaiser.html#numpy.kaiser')
        else:
            print('The user\'s chosen window is not recognized. No windowing function will be applied.')
            f = fsub * 1


        tinc = tvals[1] - tvals[0]  
        length = len(tvals)
        for i in range(repeat):
            f = np.append(f, f)
        tvals = np.linspace(tvals[0], tinc*len(f), len(f))
        
        plt.figure(20)
        plt.plot(tvals, f)
        # zero-padding
        if self.zp == None:
            pass
        else:
            if self.zp % 2 == 0:
                pass
            else:
                self.zp += 1
            # zp = np.zeros(int(self.zp / 2))
            # f = np.concatenate((zp, f))
            zp = np.zeros(int(self.zp))
            f = np.concatenate((f, zp))
            zp_time = np.linspace(tvals[-1]+tinc, tvals[-1]+len(zp)*tinc, self.zp)
            tvals = np.concatenate((tvals, zp_time))

        if self.graphs == True:
            plt.figure(3)
            plt.plot(tvals, f)
            plt.title('Windowing and Zero-Padding')
            plt.xlabel('Delay (fs)')
            plt.ylabel('Amplitude')
        else:
            pass
        
        if self.LS is False:
            # FFT magnitude and wavenumber axis
            # ft = np.abs(np.fft.fft(f))
            # ft_real = np.real(np.fft.fft(f))
            # ft_imag = np.im(np.fft.fft(f))
            ft = np.fft.rfft(f)
            # ft = abs(scipy.fft.fft(f))
            # k = scipy.fft.fftfreq(len(ft), d = tvals[1]-tvals[0]) / (2.99792458E-5)
            # k = np.linspace(freqmin, freqmax, len(ft)) / (2.99792458E-5) #/ 33.35
            k = np.fft.rfftfreq(2*len(ft)-1, tvals[1]-tvals[0])  / (2.99792458E-5)
        else:
            k, ft = LombScargle(tvals, f).autopower()
            k /= 2.99792458E-5
        
        if self.graphs == True:
            plt.figure(4)
            plt.plot(k, ft, 'b*-')
#            for i in klist:
#                plt.plot([i, i],[min(ft), 1.2*max(ft)], label = i)
            plt.xlim((0, 2000))
            plt.title('FFT')
            plt.xlabel('Wavenumber (1/cm)')
            plt.ylabel('Amplitude')
            
#            plt.legend()
        else:
            pass
        
        # start_idx = find_nearest(k, 0)[0]
        # end_idx = find_nearest(k, 2000)[0]
        # k = k[:int(len(k)/2)]
        # ft = ft[:int(len(ft)/2)]
        plt.show()
        return np.vstack((k, ft))

class FFT_Fit:
    def __init__(self, k, ft, number_of_peaks, x0, bounds, graphs, tstamp = None):
        self.k = k
        self.ft = ft
        self.number_of_peaks = number_of_peaks
        self.x0 = x0
        self.bounds = bounds
        self.graphs = graphs
        self.tstamp = tstamp

    def fit(self):
        def Gaussian(k, A, k0, FWHM):
            return A * np.exp(-4*np.log(2) * ((k - k0)/FWHM)**2)
        def Gaussian2(k, A1, A2, k01, k02, FWHM1, FWHM2):
            return Gaussian(k, A1, k01, FWHM1) + Gaussian(k, A2, k02, FWHM2)
        def Gaussian3(k, A1, A2, A3, k01, k02, k03, FWHM1, FWHM2, FWHM3):
            return Gaussian(k, A1, k01, FWHM1) + Gaussian(k, A2, k02, FWHM2) + Gaussian(k, A3, k03, FWHM3)
        def Gaussian4(k, A1, A2, A3, A4, k01, k02, k03, k04, FWHM1, FWHM2, FWHM3, FWHM4):
            return Gaussian(k, A1, k01, FWHM1) + Gaussian(k, A2, k02, FWHM2) + Gaussian(k, A3, k03, FWHM3) + Gaussian(k, A4, k04, FWHM4)
        def Gaussian5(k, A1, A2, A3, A4, A5, k01, k02, k03, k04, k05, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5):
            return Gaussian(k, A1, k01, FWHM1) + Gaussian(k, A2, k02, FWHM2) + Gaussian(k, A3, k03, FWHM3) + Gaussian(k, A4, k04, FWHM4) + Gaussian(k, A5, k05, FWHM5)
        def Gaussian6(k, A1, A2, A3, A4, A5, A6, k01, k02, k03, k04, k05, k06, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5, FWHM6):
            return Gaussian(k, A1, k01, FWHM1) + Gaussian(k, A2, k02, FWHM2) + Gaussian(k, A3, k03, FWHM3) + Gaussian(k, A4, k04, FWHM4) + Gaussian(k, A5, k05, FWHM5) + Gaussian(k, A6, k06, FWHM6)
        def Gaussian7(k, A1, A2, A3, A4, A5, A6, A7, k01, k02, k03, k04, k05, k06, k07, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5, FWHM6, FWHM7):
            return Gaussian(k, A1, k01, FWHM1) + Gaussian(k, A2, k02, FWHM2) + Gaussian(k, A3, k03, FWHM3) + Gaussian(k, A4, k04, FWHM4) + Gaussian(k, A5, k05, FWHM5) + Gaussian(k, A6, k06, FWHM6) + Gaussian(k, A7, k07, FWHM7)
      
        
        k = self.k
        ft = self.ft
        x0 = self.x0
        bounds = self.bounds
        graphs = self.graphs
        tstamp = self.tstamp
        
        # graph_output check
        if graphs == True:
            pass
        elif graphs == False:
            pass
        else:
            graphs == False
            print('The variable for `graphs` must be either True or False. The graphs will not be shown by default.')
        
        # kdiff = int(k[1] - k[0])
        # k_neg = np.linspace(-100, 0, kdiff)
        # kfit = np.append(k_neg, k)
        if self.number_of_peaks == 1:
            # while True:
            #     try:
            #         popt, pcov = curve_fit(lambda k, y0, A, k0, FWHM: y0 + Gaussian(k, A, k0, FWHM), k, ft, x0, bounds = bounds)
            #         break
            #     except:
            #         continue
#            print(pcov)
            popt, pcov = curve_fit(lambda k, y0, A, k0, FWHM: y0 + Gaussian(k, A, k0, FWHM), k, ft, x0, bounds = bounds)
            perr = np.sqrt(np.diag(pcov))
            # print('Errors = ', perr)
            sigma_squared = sum((ft - popt[0] - Gaussian(k, *popt[1:])) ** 2)
            if tstamp != None:
                print(tstamp, 'fs ', 'sigma^2 = ', sigma_squared)
            else:
                print('sigma^2 = ', sigma_squared)
            if graphs == True:
                plt.figure()
                plt.plot(k, ft, label = tstamp)
                plt.plot(k, popt[0] + Gaussian(k, *popt[1:]), 'r-')
                plt.title('y0 = %.2f, A = %.2f, k0 = %.2f, FWHM = %.2f' % tuple(popt))
                if tstamp != None:
                    plt.legend()
                plt.show()
        elif self.number_of_peaks == 2:
            popt, pcov = curve_fit(lambda k, y0, A1, A2, k01, k02, FWHM1, FWHM2: y0 + Gaussian2(k, A1, A2, k01, k02, FWHM1, FWHM2), k, ft, x0, bounds = bounds)
            # while True:
            #     try:
            #         popt, pcov = curve_fit(lambda k, y0, A1, A2, k01, k02, FWHM1, FWHM2: y0 + Gaussian2(k, A1, A2, k01, k02, FWHM1, FWHM2), k, ft, x0, bounds = bounds)
            #         break
            #     except:
            #         continue
#            print(pcov)
            y0, A1, A2, k01, k02, FWHM1, FWHM2 = popt
            perr = np.sqrt(np.diag(pcov))
            # print('Errors = ', perr)
            sigma_squared = sum((ft - y0 - Gaussian2(k, A1, A2, k01, k02, FWHM1, FWHM2)) ** 2)
            if tstamp != None:
                print(tstamp, 'fs ', 'sigma^2 = ', sigma_squared)
            else:
                print('sigma^2 = ', sigma_squared)
            if graphs == True:
                y0, A1, A2, k01, k02, FWHM1, FWHM2 = popt
                plt.figure()
                plt.plot(k, ft, label = tstamp)
                plt.plot(k, y0 + Gaussian(k, A1, k01, FWHM1), 'r-')
                plt.plot(k, y0 + Gaussian(k, A2, k02, FWHM2), 'g-')
                plt.plot(k, y0 + Gaussian2(k, A1, A2, k01, k02, FWHM1, FWHM2), 'k--')
                plt.title('y0 = %.2f, A1 = %.2f, A2 = %.2f, k01 = %.2f, k02 = %.2f, FWHM1 = %.2f, FWHM2 = %.2f' % tuple(popt))
                if tstamp != None:
                    plt.legend()
                plt.show()
        elif self.number_of_peaks == 3:
            # while True:
            #     try:
            #         popt, pcov = curve_fit(lambda k, y0, A1, A2, A3, k01, k02, k03, FWHM1, FWHM2, FWHM3: y0 + Gaussian3(k, A1, A2, A3, k01, k02, k03, FWHM1, FWHM2, FWHM3), k, ft, x0, bounds = bounds)
            #         break
            #     except:
            #         continue
            popt, pcov = curve_fit(lambda k, y0, A1, A2, A3, k01, k02, k03, FWHM1, FWHM2, FWHM3: y0 + Gaussian3(k, A1, A2, A3, k01, k02, k03, FWHM1, FWHM2, FWHM3), k, ft, x0, bounds = bounds)
            y0, A1, A2, A3, k01, k02, k03, FWHM1, FWHM2, FWHM3 = popt
            sigma_squared = sum((ft - y0 - Gaussian3(k, A1, A2, A3, k01, k02, k03, FWHM1, FWHM2, FWHM3)) ** 2)
            if tstamp != None:
                print(tstamp, 'fs ', 'sigma^2 = ', sigma_squared)
            else:
                print('sigma^2 = ', sigma_squared)

            # print(pcov)
            perr = np.sqrt(np.diag(pcov))
            # print('Errors = ', perr)
            if graphs == True:
                y0, A1, A2, A3, k01, k02, k03, FWHM1, FWHM2, FWHM3 = popt
                plt.figure()
                plt.plot(k, ft, label = tstamp)
                plt.plot(k, y0 + Gaussian(k, A1, k01, FWHM1), 'r-')
                plt.plot(k, y0 + Gaussian(k, A2, k02, FWHM2), 'g-')
                plt.plot(k, y0 + Gaussian(k, A3, k03, FWHM3), 'y-')
                plt.plot(k, y0 + Gaussian3(k, A1, A2, A3, k01, k02, k03, FWHM1, FWHM2, FWHM3), 'k--')
                plt.title('y0 = %.2f, A1 = %.2f, A2 = %.2f, A3 = %.2f, k01 = %.2f, k02 = %.2f, k03 = %.2f, FWHM1 = %.2f, FWHM2 = %.2f, FWHM3 = %.2f' % tuple(popt))
                if tstamp != None:
                    plt.legend()
                plt.show()
        elif self.number_of_peaks == 4:
            popt, pcov = curve_fit(lambda k, y0, A1, A2, A3, A4, k01, k02, k03, k04, FWHM1, FWHM2, FWHM3, FWHM4: y0 + Gaussian4(k, A1, A2, A3, A4, k01, k02, k03, k04, FWHM1, FWHM2, FWHM3, FWHM4), k, ft, x0, bounds = bounds)
            # while True:
            #     try:
            #         popt, pcov = curve_fit(lambda k, y0, A1, A2, A3, A4, k01, k02, k03, k04, FWHM1, FWHM2, FWHM3, FWHM4: y0 + Gaussian4(k, A1, A2, A3, A4, k01, k02, k03, k04, FWHM1, FWHM2, FWHM3, FWHM4), k, ft, x0, bounds = bounds)
            #         break
            #     except:
            #         continue
#            print(pcov)
            perr = np.sqrt(np.diag(pcov))
            # print('Errors = ', perr)
            y0, A1, A2, A3, A4, k01, k02, k03, k04, FWHM1, FWHM2, FWHM3, FWHM4 = popt
            sigma_squared = sum((ft - y0 - Gaussian4(k, A1, A2, A3, A4, k01, k02, k03, k04, FWHM1, FWHM2, FWHM3, FWHM4)) ** 2)
            if tstamp != None:
                print(tstamp, 'fs ', 'sigma^2 = ', sigma_squared)
            else:
                print('sigma^2 = ', sigma_squared)
            if graphs == True:
                y0, A1, A2, A3, A4, k01, k02, k03, k04, FWHM1, FWHM2, FWHM3, FWHM4 = popt
                plt.figure()
                plt.plot(k, ft, label = tstamp)
                plt.plot(k, y0 + Gaussian(k, A1, k01, FWHM1), 'r-')
                plt.plot(k, y0 + Gaussian(k, A2, k02, FWHM2), 'g-')
                plt.plot(k, y0 + Gaussian(k, A3, k03, FWHM3), 'y-')
                plt.plot(k, y0 + Gaussian(k, A4, k04, FWHM4), 'c-')
                plt.plot(k, y0 + Gaussian4(k, A1, A2, A3, A4, k01, k02, k03, k04, FWHM1, FWHM2, FWHM3, FWHM4), 'k--')
                plt.title('y0 = %.2f, A1 = %.2f, A2 = %.2f, A3 = %.2f, A4 = %.2f, k01 = %.2f, k02 = %.2f, k03 = %.2f, k04 = %.2f, FWHM1 = %.2f, FWHM2 = %.2f, FWHM3 = %.2f, FWHM4 = %.2f' % tuple(popt))
                if tstamp != None:
                    plt.legend()
                plt.show()
        elif self.number_of_peaks == 5:
            popt, pcov = curve_fit(lambda k, y0, A1, A2, A3, A4, A5, k01, k02, k03, k04, k05, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5: y0 + Gaussian5(k, A1, A2, A3, A4, A5, k01, k02, k03, k04, k05, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5), k, ft, x0, bounds = bounds)
            # while True:
            #     try:
            #         popt, pcov = curve_fit(lambda k, y0, A1, A2, A3, A4, A5, k01, k02, k03, k04, k05, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5: y0 + Gaussian5(k, A1, A2, A3, A4, A5, k01, k02, k03, k04, k05, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5), k, ft, x0, bounds = bounds)
            #         break
            #     except:
            #         continue
#            print(pcov)
            perr = np.sqrt(np.diag(pcov))
            y0, A1, A2, A3, A4, A5, k01, k02, k03, k04, k05, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5 = popt
            # print('Errors = ', perr)
            sigma_squared = sum((ft - y0 - Gaussian5(k, A1, A2, A3, A4, A5, k01, k02, k03, k04, k05, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5)) ** 2)
            if tstamp != None:
                print(tstamp, 'fs ', 'sigma^2 = ', sigma_squared)
            else:
                print('sigma^2 = ', sigma_squared)
            if graphs == True:
                y0, A1, A2, A3, A4, A5, k01, k02, k03, k04, k05, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5 = popt
                plt.figure()
                plt.plot(k, ft, label = tstamp)
                plt.plot(k, y0 + Gaussian(k, A1, k01, FWHM1), 'r-')
                plt.plot(k, y0 + Gaussian(k, A2, k02, FWHM2), 'g-')
                plt.plot(k, y0 + Gaussian(k, A3, k03, FWHM3), 'y-')
                plt.plot(k, y0 + Gaussian(k, A4, k04, FWHM4), 'c-')
                plt.plot(k, y0 + Gaussian(k, A5, k05, FWHM5), 'm')
                plt.plot(k, y0 + Gaussian5(k, A1, A2, A3, A4, A5, k01, k02, k03, k04, k05, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5), 'k--')
                plt.title('y0 = %.2f, A1 = %.2f, A2 = %.2f, A3 = %.2f, A4 = %.2f, A5 = %.2f, k01 = %.2f, k02 = %.2f, k03 = %.2f, k04 = %.2f, k05 = %.2f, FWHM1 = %.2f, FWHM2 = %.2f, FWHM3 = %.2f, FWHM4 = %.2f, FWHM5 = %.2f' % tuple(popt))
                if tstamp != None:
                    plt.legend()
                plt.show()
        elif self.number_of_peaks == 6:
            popt, pcov = curve_fit(lambda k, y0, A1, A2, A3, A4, A5, A6, k01, k02, k03, k04, k05, k06, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5, FWHM6: y0 + Gaussian6(k, A1, A2, A3, A4, A5, A6, k01, k02, k03, k04, k05, k06, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5, FWHM6), k, ft, x0, bounds = bounds)
            # while True:
            #     try:
            #         popt, pcov = curve_fit(lambda k, y0, A1, A2, A3, A4, A5, A6, k01, k02, k03, k04, k05, k06, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5, FWHM6: y0 + Gaussian6(k, A1, A2, A3, A4, A5, A6, k01, k02, k03, k04, k05, k06, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5, FWHM6), k, ft, x0, bounds = bounds)
            #         break
            #     except:
            #         continue
#            print(pcov)
            perr = np.sqrt(np.diag(pcov))
            # print('Errors = ', perr)
            y0, A1, A2, A3, A4, A5, A6, k01, k02, k03, k04, k05, k06, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5, FWHM6 = popt
            sigma_squared = sum((ft - y0 - Gaussian6(k, A1, A2, A3, A4, A5, A6, k01, k02, k03, k04, k05, k06, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5, FWHM6)) ** 2)
            if tstamp != None:
                print(tstamp, 'fs ', 'sigma^2 = ', sigma_squared)
            else:
                print('sigma^2 = ', sigma_squared)
            if graphs == True:
                y0, A1, A2, A3, A4, A5, A6, k01, k02, k03, k04, k05, k06, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5, FWHM6 = popt
                plt.figure()
                plt.plot(k, ft, label = tstamp)
                plt.plot(k, y0 + Gaussian(k, A1, k01, FWHM1), 'r-')
                plt.plot(k, y0 + Gaussian(k, A2, k02, FWHM2), 'g-')
                plt.plot(k, y0 + Gaussian(k, A3, k03, FWHM3), 'y-')
                plt.plot(k, y0 + Gaussian(k, A4, k04, FWHM4), 'c-')
                plt.plot(k, y0 + Gaussian(k, A5, k05, FWHM5), 'm-')
                plt.plot(k, y0 + Gaussian(k, A6, k06, FWHM6), color = '#800080') # purple
                plt.plot(k, y0 + Gaussian6(k, A1, A2, A3, A4, A5, A6, k01, k02, k03, k04, k05, k06, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5, FWHM6), 'k--')
                plt.title('y0 = %.2f, A1 = %.2f, A2 = %.2f, A3 = %.2f, A4 = %.2f, A5 = %.2f, A6 = %.2f, k01 = %.2f, k02 = %.2f, k03 = %.2f, k04 = %.2f, k05 = %.2f, k06 = %.2f, FWHM1 = %.2f, FWHM2 = %.2f, FWHM3 = %.2f, FWHM4 = %.2f, FWHM5 = %.2f, FWHM6 = %.2f' % tuple(popt))
                if tstamp != None:
                    plt.legend()
                plt.show()
        elif self.number_of_peaks == 7:
            popt, pcov = curve_fit(lambda k, y0, A1, A2, A3, A4, A5, A6, A7, k01, k02, k03, k04, k05, k06, k07, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5, FWHM6, FWHM7: y0 + Gaussian7(k, A1, A2, A3, A4, A5, A6, A7, k01, k02, k03, k04, k05, k06, k07, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5, FWHM6, FWHM7), k, ft, x0, bounds = bounds)
            # while True:
            #     try:
            #         popt, pcov = curve_fit(lambda k, y0, A1, A2, A3, A4, A5, A6, A7, k01, k02, k03, k04, k05, k06, k07, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5, FWHM6, FWHM7: y0 + Gaussian7(k, A1, A2, A3, A4, A5, A6, A7, k01, k02, k03, k04, k05, k06, k07, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5, FWHM6, FWHM7), k, ft, x0, bounds = bounds)
            #         break
            #     except:
            #         continue
#            print(pcov)
            perr = np.sqrt(np.diag(pcov))
            # print('Errors = ', perr)
            y0, A1, A2, A3, A4, A5, A6, A7, k01, k02, k03, k04, k05, k06, k07, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5, FWHM6, FWHM7 = popt
            sigma_squared = sum((ft - y0 - Gaussian7(k, A1, A2, A3, A4, A5, A6, A7, k01, k02, k03, k04, k05, k06, k07, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5, FWHM6, FWHM7) ** 2))
            if tstamp != None:
                print(tstamp, 'fs ', 'sigma^2 = ', sigma_squared)
            else:
                print('sigma^2 = ', sigma_squared)
            if graphs == True:
                y0, A1, A2, A3, A4, A5, A6, A7, k01, k02, k03, k04, k05, k06, k07, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5, FWHM6, FWHM7 = popt
                plt.figure()
                plt.plot(k, ft, label = tstamp)
                plt.plot(k, y0 + Gaussian(k, A1, k01, FWHM1), 'r-')
                plt.plot(k, y0 + Gaussian(k, A2, k02, FWHM2), 'g-')
                plt.plot(k, y0 + Gaussian(k, A3, k03, FWHM3), 'y-')
                plt.plot(k, y0 + Gaussian(k, A4, k04, FWHM4), 'c-')
                plt.plot(k, y0 + Gaussian(k, A5, k05, FWHM5), 'm-')
                plt.plot(k, y0 + Gaussian(k, A6, k06, FWHM6), color = '#800080') # purple
                plt.plot(k, y0 + Gaussian(k, A7, k07, FWHM7), color = '#ffa500') # orange
                plt.plot(k, y0 + Gaussian7(k, A1, A2, A3, A4, A5, A6, A7, k01, k02, k03, k04, k05, k06, k07, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5, FWHM6, FWHM7), 'k--')
                plt.title('y0 = %.2f, A1 = %.2f, A2 = %.2f, A3 = %.2f, A4 = %.2f, A5 = %.2f, A6 = %.2f, A7 = %.2f, k01 = %.2f, k02 = %.2f, k03 = %.2f, k04 = %.2f, k05 = %.2f, k06 = %.2f, k07 = %.2f, FWHM1 = %.2f, FWHM2 = %.2f, FWHM3 = %.2f, FWHM4 = %.2f, FWHM5 = %.2f, FWHM6 = %.2f, FWHM7 = %.2f' % tuple(popt))
                if tstamp != None:
                    plt.legend()
                plt.show()
        else:
            print('More than 7 peaks not currently supported, so this will default to a 7-Gaussian fit.')
            popt, pcov = curve_fit(lambda k, y0, A1, A2, A3, A4, A5, A6, A7, k01, k02, k03, k04, k05, k06, k07, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5, FWHM6, FWHM7: y0 + Gaussian7(k, A1, A2, A3, A4, A5, A6, A7, k01, k02, k03, k04, k05, k06, k07, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5, FWHM6, FWHM7), k, ft, x0, bounds = bounds)
            # while True:
            #     try:
            #         popt, pcov = curve_fit(lambda k, y0, A1, A2, A3, A4, A5, A6, A7, k01, k02, k03, k04, k05, k06, k07, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5, FWHM6, FWHM7: y0 + Gaussian7(k, A1, A2, A3, A4, A5, A6, A7, k01, k02, k03, k04, k05, k06, k07, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5, FWHM6, FWHM7), k, ft, x0, bounds = bounds)
            #         break
            #     except:
            #         continue
#            print(pcov)
            perr = np.sqrt(np.diag(pcov))
            # print('Errors = ', perr)
            y0, A1, A2, A3, A4, A5, A6, A7, k01, k02, k03, k04, k05, k06, k07, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5, FWHM6, FWHM7 = popt
            sigma_squared = sum((ft - y0 - Gaussian7(k, A1, A2, A3, A4, A5, A6, A7, k01, k02, k03, k04, k05, k06, k07, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5, FWHM6, FWHM7) ** 2))
            if tstamp != None:
                print(tstamp, 'fs ', 'sigma^2 = ', sigma_squared)
            else:
                print('sigma^2 = ', sigma_squared)
            if graphs == True:
                y0, A1, A2, A3, A4, A5, A6, A7, k01, k02, k03, k04, k05, k06, k07, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5, FWHM6, FWHM7 = popt
                plt.figure()
                plt.plot(k, ft, label = tstamp)
                plt.plot(k, y0 + Gaussian(k, A1, k01, FWHM1), 'r-')
                plt.plot(k, y0 + Gaussian(k, A2, k02, FWHM2), 'g-')
                plt.plot(k, y0 + Gaussian(k, A3, k03, FWHM3), 'y-')
                plt.plot(k, y0 + Gaussian(k, A4, k04, FWHM4), 'c-')
                plt.plot(k, y0 + Gaussian(k, A5, k05, FWHM5), 'm-')
                plt.plot(k, y0 + Gaussian(k, A6, k06, FWHM6), color = '#800080') # purple
                plt.plot(k, y0 + Gaussian(k, A7, k07, FWHM7), color = '#ffa500') # orange
                plt.plot(k, y0 + Gaussian7(k, A1, A2, A3, A4, A5, A6, A7, k01, k02, k03, k04, k05, k06, k07, FWHM1, FWHM2, FWHM3, FWHM4, FWHM5, FWHM6, FWHM7), 'k--')
                plt.title('y0 = %.2f, A1 = %.2f, A2 = %.2f, A3 = %.2f, A4 = %.2f, A5 = %.2f, A6 = %.2f, A7 = %.2f, k01 = %.2f, k02 = %.2f, k03 = %.2f, k04 = %.2f, k05 = %.2f, k06 = %.2f, k07 = %.2f, FWHM1 = %.2f, FWHM2 = %.2f, FWHM3 = %.2f, FWHM4 = %.2f, FWHM5 = %.2f, FWHM6 = %.2f, FWHM7 = %.2f' % tuple(popt))
                if tstamp != None:
                    plt.legend()
                plt.show()
        return popt, perr

class pDFWM(Data_Lists, FFT, FFT_Fit):
    def __init__(self, directory, number_of_peaks, x0, bounds,  group = 'sub', cut = None, n = 1, submethod = 4, window = 0, zp = 2**12, y0guess = 0, y0uplim = 1, graphs = False, norm = False):
        self.directory = directory
        self.number_of_peaks = number_of_peaks

        # x0 and bounds do not need to consider y0, since it will be handled internally
        self.x0 = x0
        self.bounds = bounds    

        self.group = group
        self.cut = cut
        self.n = n
        self.submethod = submethod
        self.window = window
        self.zp = zp
        self.y0guess = y0guess
        self.y0uplim = y0uplim
        self.graphs = graphs
        self.norm = norm 
        
        '''
        0 = False for FFT() and FFT_Fit()
        1 = True for FFT(), False for FFT_Fit()
        2 = True for FFT() and FFT_Fit()
        
        False = False for FFT() and FFT_Fit()
        True = True for FFT() and FFT_Fit()
        '''
        self.graphs = graphs
        
    def transients(self):

        off, on, sub, tstamp_off, tstamp_on, tstamp_sub = Data_Lists(self.directory).importdir()

        if (self.group == 'sub') or (self.group == 'Sub') or (self.group == 'subtracted') or (self.group == 'Subtracted'):
            self.group = sub
            tstamp = tstamp_sub
        elif (self.group == 'on') or (self.group == 'On'):
            self.group = on
            tstamp = tstamp_on
        
        elif (self.group == 'off') or (self.group == 'Off'):
            self.group = off
            tstamp = tstamp_off
        else:
            print('Error in choosing lists --> subtracted list is chosen by default')
            self.group = sub
            tstamp = tstamp_sub
        
        # this round is just for getting a consistent y0 for all FFT fits
        y0guess = self.y0guess
        y0min = 0
        y0max = self.y0uplim
        y0list = []
        
        self.x0.insert(0, y0guess)
        self.bounds[0].insert(0, y0min)
        self.bounds[1].insert(0, y0max)
    
        if self.graphs == False or self.graphs == 0:
            FFT_graphs = False
            FFT_Fit_graphs = False
        elif self.graphs == 1:
            FFT_graphs = True
            FFT_Fit_graphs = False
        elif self.graphs == 2:
            FFT_graphs = False
            FFT_Fit_graphs = True
        elif self.graphs == True or self.graphs == 3:
            FFT_graphs = True
            FFT_Fit_graphs = True
        else:
            FFT_graphs = False
            FFT_Fit_graphs = False
            print('graphs must be set equal to False, True, 0, 1, 2, or 3. The default of False will now be used.')

        for i in range(len(self.group)):
            print('file =', self.group[i])
            print('number of subtracted files =', len(self.group))
            print('iteration in the list =', i)
            #self, file, cut = None, n = 1, window = 0, zp = 2**12, graphs = False
            k, ft = FFT(self.group[i], self.cut, self.n, self.submethod, self.window, self.zp, FFT_graphs).FFT()
            if self.norm == True:
                ft = ft / np.max(ft)
            else:
                pass
            params, perr = FFT_Fit(k, ft, self.number_of_peaks, self.x0, self.bounds, FFT_Fit_graphs, tstamp[i]).fit()
            y0list.append(params[0])
        y0avg = np.mean(y0list)
        # this is the round that is finalized
        paramslist = []
        perrlist = []
        for i in range(len(self.group)):
            # use i to pick a specific file and a specific tstamp that corresponds for that file 
            k, ft = FFT(self.group[i], self.cut, self.n, self.submethod, self.window, self.zp, FFT_graphs).FFT()
            
            self.x0[0] = y0avg
            self.bounds[0][0] = y0avg-1e-5
            self.bounds[1][0] = y0avg+1e-5

            params, perr = FFT_Fit(k, ft, self.number_of_peaks, self.x0, self.bounds, FFT_Fit_graphs, tstamp[i]).fit()
                
            if i == 0:
                paramslist.append(params)
                perrlist.append(perr)
            else:
                paramslist = np.row_stack((paramslist, params))
                perrlist = np.row_stack((perrlist, perr))
                
        '''
        zip the parameters and errors together, below is an example for fitting two peaks:
            y0, y0err, A1, A1err, A2, A2err, k01, k01err, k02, k02err, FWHM1, FWHM1err, FWHM2, FWHM2err
        '''
        
        result = []
        for i in range(len(paramslist[0, :])):
            result.append(paramslist[:, i])
            result.append(perrlist[:, i])
        
        resultarray = np.array(tstamp).T
        for i in range(len(result)):
            resultarray = np.column_stack((resultarray, np.array(result[i]).T))

        return resultarray
    
    
# file = r"E:\UD\Research\TMDCs\DFWM\pump-DFWM\WS2\2019\March\18\5-50ps\pDFWM_WS2phA3_540nm_33nm_5mW_632nm_14nm_2mWwithChopper_1000scans_40002T_540nm_IPsubtracted.txt"
# k, ft = FFT(file, cut = 400, graphs = False).FFT()
# k = abs(k)
# re = np.real(ft)
# im = np.imag(ft)
# plt.plot(k, abs(np.real(ft)))
# plt.plot(k, abs(np.imag(ft)))
    
        
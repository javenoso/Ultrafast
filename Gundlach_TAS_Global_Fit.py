"""
Last updated: Nov. 22nd, 2020

OOP of Gundlach_TAS_global.py, which is:
    Transient Absorption fitting with weights adding to 1 via normalization
    Fit a single spectrum or fit up to four spectra globally.

Jan 31st, 2020
New global fitting called Fit(self).TA_globalX() is NOT ready yet.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize
import seaborn as sns

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

# def SVD(file, n, tstart = None, tstop = None, wstart = None, wstop = None):
#     # n is the number of singular values that you want to plot
#     """
#     Source:
#         https://cmdlinetips.com/2019/05/singular-value-decomposition-svd-in-python/
#     """
    
#     data = np.array(pd.read_csv(file, header = None, delim_whitespace = False))
#     time = data[1:, 0]
#     wavelength = data[0, 1:]
#     amplitude = data[1:, 1:]

#     t = time
#     w = wavelength
#     a = amplitude

#     if tstart == None:
#         tstart = t[0]
#     else:
#         pass

#     if tstop == None:
#         tstop = t[-1]
#     else:
#         pass

#     if wstart == None:
#         wstart = w[0]
#     else:
#         pass

#     if wstop == None:
#         wstop = w[-1]
#     else:
#         pass

#     tstart, tstop, wstart, wstop = [tstart, tstop, wstart, wstop]

#     tstart_idx = find_nearest(t, tstart)[0]
#     tstop_idx = find_nearest(t, tstop)[0]
#     wstart_idx = find_nearest(w, wstart)[0]
#     wstop_idx = find_nearest(w, wstop)[0]

#     t = t[tstart_idx:tstop_idx]
#     w = w[wstart_idx:wstop_idx]
#     a = a[tstart_idx:tstop_idx, wstart_idx:wstop_idx]
    
#     """
#     u = left-singular vectors, (m, m) = (184, 184) --> times
#     s = singular values, (m,) = (184,)
#     v = right-singular vector, (n, n) = (1024, 1024) --> wavelengths
#     """
    
#     u, s, v = np.linalg.svd(a, full_matrices = True)
    
#     var_explained = np.round(s**2 / np.sum(s**2), decimals = 3)
    
#     plt.figure(1)
#     sns.barplot(x = list(range(1, len(var_explained)+1)), y = var_explained)
#     plt.xlim(0, n+1) # plot a little further than n
#     plt.xlabel('SVs', fontsize = 16)
#     plt.ylabel('Percent Variance Explained', fontsize = 16)
    
#     for i in range(n):
#         plt.figure(2)
#         plt.subplot(2, 1, 1)
#         plt.plot(t, u[:, i], label = i+1)
#         print(i+1)
#         plt.title('Left Singular Vectors')
#         plt.xlabel('Delay Time (fs)')
#     plt.legend()
#     for i in range(n):
#         plt.figure(2)
#         plt.subplot(2, 1, 2)
#         plt.plot(w, v[i, :], label = i+1)
#         plt.title('Right Singular Vectors')
#         plt.xlabel('Wavelength (nm)')
#     plt.legend()
#     plt.show()
    
#     return u, s, v

class Data:
    def __init__(self, file, wavelengths, tstart, tstop, tshift = None):
        self.file = file # string: file = r'filename'
        self.wavelengths = wavelengths # single value or a vector: wavelengths = 700, or wavelengths = [700, 570]
        self.tstart = tstart
        self.tstop = tstop
        self.tshift = tshift
        
    def initialize(self):
        # import data from external file
        if '.dat' in self.file:
            data = np.array(pd.read_csv(self.file, header = None, delim_whitespace = True))
            t = data[:, 0]
            a = 1000 * data[:, 1] # convert to mOD
        if '.csv' in self.file:
            data = np.array(pd.read_csv(self.file, header = None, delim_whitespace = False))
            t = data[1:, 0]
            w = data[0, 1:]
            a = 1000 * data[1:, 1:] # convert to mOD
        else:
            print('You need to use a file that is either .dat or .csv.')
        

        if len(self.wavelengths) == 1:
            self.tshift = np.asarray(self.tshift)
            for i in range(len(self.wavelengths)):
                # select trace that matches desired wavelength
                wavelength_idx = (np.abs(w - self.wavelengths[i])).argmin()
                anew = a[:, wavelength_idx]
                # truncate time and absorbtion vectors to tstart --> tstop
                tstart_idx = find_nearest(t, self.tstart)[0]
                tstop_idx = find_nearest(t, self.tstop)[0]
                tnew = t[tstart_idx:tstop_idx]
                
                tnew = tnew - self.tshift[i]
                # if type(self.tshift) is None:
                #     return
                # if type(self.tshift) is int:
                #     tnew = tnew - self.tshift[i]
                # if type(self.tshift) is float:
                #     tnew = tnew - self.tshift[i]
                # else:
                #     self.tshift = None
                
                anew = anew[tstart_idx:tstop_idx]
                anew = anew / max(abs(anew)) # normalize the absorption vector, use abs(a) in case the signal is negative

                
                # if self.tshift == True:
                #     idx_max = find_nearest(abs(anew), max(abs(anew)))[0]
                #     time_rise = tnew[:idx_max]
                #     a_rise = abs(anew[:idx_max])
                #     a_TZ = 0.75 * max(abs(a_rise))
                #     idx_TZ = find_nearest(abs(a_rise), a_TZ)[0]
                #     tnew = tnew - tnew[idx_TZ]
                # else:
                #     pass
                
                trace = np.vstack((tnew, anew))
            return trace
            
        else:
            self.tshift = np.asarray(self.tshift)
            dataset = []
            for i in range(len(self.wavelengths)):
                # select trace that matches desired wavelength
                wavelength_idx = (np.abs(w - self.wavelengths[i])).argmin()
                anew = a[:, wavelength_idx]
                
                # truncate time and absorbtion vectors to tstart --> tstop
                tstart_idx = find_nearest(t, self.tstart)[0]
                tstop_idx = find_nearest(t, self.tstop)[0]
                tnew = t[tstart_idx:tstop_idx]
                
                tnew = tnew - self.tshift[i]
                # if type(self.tshift) is None:
                #     return
                # if type(self.tshift) is int:
                #     tnew = tnew - self.tshift[i]
                # if type(self.tshift) is float:
                #     tnew = tnew - self.tshift[i]
                # else:
                #     self.tshift = None
                    
                anew = anew[tstart_idx:tstop_idx]
                anew = anew #/ max(abs(anew)) # normalize the absorption vector, use abs(a) in case the signal is negative
                
                # if self.tshift == True:
                #     idx_max = find_nearest(abs(anew), max(abs(anew)))[0]
                #     time_rise = tnew[:idx_max]
                #     a_rise = abs(anew[:idx_max])
                #     a_TZ = 0.75  * max(abs(a_rise))
                #     idx_TZ = find_nearest(abs(a_rise), a_TZ)[0]
                #     tnew = tnew - tnew[idx_TZ]
                # else:
                #     pass
                
                trace = np.vstack((tnew, anew))
                dataset.append(trace)
                
            return dataset
    
class Fit(Data):
    def __init__(self, file, wavelengths, tstart, tstop, t0, Wpump, number_of_states, x0, bounds, tshift = None):
        self.file = file
        self.wavelengths = wavelengths
        self.tstart = tstart
        self.tstop = tstop
        self.t0 = t0
        self.Wpump = Wpump
        self.number_of_states = number_of_states
        self.x0 = x0
        self.bounds = bounds
        self.tshift = tshift
    
    def TA_Fit_Time_Zero(self):
        """
        For two-state model.
        Fit time zero wile keeping pump amplitude constant at A = 1
        There are no tshifts.
        """
        self.tshift = [0] * len(self.wavelengths)
        dataset = Data(self.file, self.wavelengths, self.tstart, self.tstop, self.tshift).initialize()
        
        """
        self.x0 should now be in this format:
            self.x0 = [t0_1, t0_2, t0_3, t1, t2, w11, w12, w21, w22, w31, w32]
        self.bounds should follow the same format, self.bounds = ([lower], [higher])
        """
        
        if len(self.wavelengths) == 1:
            def G(t, A, t0, Wpump):
                return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
                
            def model(y, t, t0, t1, t2):
                N1 = y[0]
                N2 = y[1]
                dN1_dt = G(t, 1, t0, self.Wpump) - N1/t1
                dN2_dt = N1/t1 - N2/t2
                return [dN1_dt, dN2_dt]
            
            def y(t, t0, t1, t2, w1, w2):
                y0 = [0, 0] # the states are initially unoccupied
                global result
                result = integrate.odeint(model, y0, tdata, args = (t0, t1, t2,))
                return w1*result[:, 0] + w2*result[:, 1]
                                      
            tdata = dataset[0]
            adata = dataset[1]

            popt, pcov = optimize.curve_fit(lambda t, t0, t1, t2, w1, w2: y(t, t0, t1, t2, w1, w2), tdata, adata, self.x0, bounds = self.bounds, maxfev = 1000000)
            
            t0, t1, t2, w1, w2 = popt
            perr = np.sqrt(np.diag(pcov))
            
            # print('##################################')
            # #print('num of iterations = ', itercount)
            # print('sum of weights = ', popt[3] + popt[4])
            # print("")
            # print('fitted parameters = ', popt)
            # print("")
            # print('covar matrix = ', pcov)
            # print("")
            # print('Errors = ', perr)
            # print("")
            
            sigma_squared = sum((adata - w1*result[:, 0] + w2*result[:, 1]) ** 2)
            print('sigma^2 = ', sigma_squared)
            
            plt.figure()
            plt.plot(tdata, G(tdata, popt[0], t0, self.Wpump), 'y-', label = 'Vpump')
            plt.plot(tdata, popt[3]*result[:,0], 'r-', label = 'w1*N1')
            plt.plot(tdata, popt[4]*result[:,1], 'b-', label = 'w2*N2')
            fit_result = popt[3]*result[:,0] + popt[4]*result[:,1]
            plt.plot(tdata, fit_result, 'k--', label = 'w1*N1 + w2*N2')
            plt.scatter(tdata, adata)
            plt.legend()
            
            wsum = popt[3] + popt[4]
            popt[3] = popt[3] / wsum
            popt[4] = popt[4] / wsum
            
            plt.title('t0 = %.2f fs, t1 = %.2f fs, t2 = %.2f fs, w1 = %.2f, w2 = %.2f' % tuple(popt))
            #plt.figtext(0.5, 0.01, wavelength, wrap=True, horizontalalignment='center', fontsize=12)
            plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
            plt.ylabel('ΔA (mOD)')
            plt.show()
            
            TA_return = [t0, t1, t2, w1, w2, result]
            
            
        if len(self.wavelengths) == 2:
            def G(t, A, t0, Wpump):
                return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
            
            def model(y, t, t0, t1, t2):
                N1 = y[0]
                N2 = y[1]
                dN1_dt = G(t, 1, t0, self.Wpump) - N1/t1
                dN2_dt = N1/t1 - N2/t2
                return [dN1_dt, dN2_dt]
            
            # def y1(t, A1, t1, t2, w11):
            #     w12 = 1 - w11
            def y1(t, t01, t1, t2, w11, w12):
                y0 = [0, 0] # the states are initially unoccupied
                global result1
                result1 = integrate.odeint(model, y0, t, args = (t01, t1, t2,))
                return w11*result1[:, 0] + w12*result1[:, 1]
    
            # def y2(t, A2, t1, t2, w21):
            #     w22 = 1 - w21
            def y2(t, t02, t1, t2, w21, w22):
                y0 = [0, 0]
                global result2
                result2 = integrate.odeint(model, y0, t, args = (t02, t1, t2,))
                return w21*result2[:, 0] + w22*result2[:, 1]
    
            # def comboFunc(comboData, A1, A2, t1, t2, w11, w21):
            #     result1 = y1(tdata1, A1, t1, t2, w11)
            #     result2 = y2(tdata2, A2, t1, t2, w21)
            def comboFunc(comboData, t01, t02, t1, t2, w11, w12, w21, w22):
                result1 = y1(tdata1, t01, t1, t2, w11, w12)
                result2 = y2(tdata2, t02, t1, t2, w21, w22)
                sigma_squared = sum((a_combo - np.append(result1, result2)) ** 2)
                print('sigma^2 = ', sigma_squared)
                return np.append(result1, result2)
            # some initial parameter values
        
            # curve fit the combined data to the combined function
            t_combo = []
            a_combo = []
            for i in dataset:
                t_combo = np.append(t_combo, i[0])
                a_combo = np.append(a_combo, i[1])
                
            data1 = dataset[0]
            data2 = dataset[1]
            
            tdata1, adata1 = data1
            tdata2, adata2 = data2
            
            fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, self.x0, bounds = self.bounds)
            
            # A1, A2, t1, t2, w11, w21 = fittedParameters
            # A1err, A2err, t1err, t2err, w11err, w21err = np.sqrt(np.diag(pcov))
            # print('A1err = %.2f, A2err = %.2f, t1err = %.2f, t2err = %.2f, w11err = %.2f, w21err = %.2f' % tuple((A1err, A2err, t1err, t2err, w11err, w21err)))
        
            # w12 = 1 - w11
            # w22 = 1 - w21
            t01, t02, t1, t2, w11, w12, w21, w22 = fittedParameters
            t01err, t02err, t1err, t2err, w11err, w12err, w21err, w22err = np.sqrt(np.diag(pcov))
            print('t01err = %.2f, t02err = %.2f, t1err = %.2f, t2err = %.2f, w11err = %.2f, w12err = %.2f, w21err = %.2f, w22err = %.2f' % tuple((t01err, t02err, t1err, t2err, w11err, w12err, w21err, w22)))
    
            plt.figure()
            g = G(tdata1, 1, t01, self.Wpump)
            plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
            plt.plot(tdata1, g*max(adata1)/max(g), color = '#b676f2', label = 'Vpump scaled')
            plt.axvline(x = t01, color = '#F1B929', label = 'Time-Zero')
            plt.plot(tdata1, w11*result1[:,0], 'r-', label = 'w1*N1')
            plt.plot(tdata1, w12*result1[:,1], 'b-', label = 'w2*N2')
            fit_result = w11*result1[:,0] + w12*result1[:,1]
            plt.plot(tdata1, fit_result, 'k--', label = 'w1*N1 + w2*N2')
            plt.scatter(tdata1, adata1)
            plt.legend()
            # wsum = w11 + w12
            # w11 = w11 / wsum
            # w12 = w12 / wsum
            plt.title('t0 = %.2f, t1 = %.2f fs, t2 = %.2f fs, w1 = %.2f, w2 = %.2f' % tuple((t01, t1, t2, w11, w12)))
            plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
            plt.ylabel('ΔA (mOD)')
            
            plt.figure()
            g = G(tdata2, 1, t02, self.Wpump)
            plt.plot(tdata2, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
            plt.plot(tdata2, g*max(adata2)/max(g), color = '#b676f2', label = 'Vpump scaled')
            plt.axvline(x = t02, color = '#F1B929', label = 'Time-Zero')
            plt.plot(tdata2, w21*result2[:, 0], 'r-', label = 'w1*N1')
            plt.plot(tdata2, w22*result2[:, 1], 'b-', label = 'w2*N2')
            fit_result = w21*result2[:,0] + w22*result2[:,1]
            plt.plot(tdata2, fit_result, 'k--', label = 'w1*N1 + w2*N2')
            plt.scatter(tdata2, adata2)
            plt.legend()
            # wsum = w21 + w22
            # w21 = w21 / wsum
            # w22 = w22 / wsum
            plt.title('t0 = %.2f, t1 = %.2f fs, t2 = %.2f fs, w1 = %.2f, w2 = %.2f' % tuple((t02, t1, t2, w21, w22)))
            plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[1])
            plt.ylabel('ΔA (mOD)')
            plt.show()
            
            return t01, t02, t1, t2, w11, w12, w21, w22, result1, result2
            
        if len(self.wavelengths) == 3:
            def G(t, A, t0, Wpump):
                return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
            
            def model(y, t, t0, t1, t2):
                N1 = y[0]
                N2 = y[1]
                dN1_dt = G(t, 1, t0, self.Wpump) - N1/t1
                dN2_dt = N1/t1 - N2/t2
                return [dN1_dt, dN2_dt]
            
            def y1(t, t01, t1, t2, w11, w12):
                y0 = [0, 0] # the states are initially unoccupied
                global result1
                result1 = integrate.odeint(model, y0, t, args = (t01, t1, t2,))
                return w11*result1[:, 0] + w12*result1[:, 1]
    
            def y2(t, t02, t1, t2, w21, w22):
                y0 = [0, 0]
                global result2
                result2 = integrate.odeint(model, y0, t, args = (t02, t1, t2,))
                return w21*result2[:, 0] + w22*result2[:, 1]
            
            def y3(t, t03, t1, t2, w31, w32):
                y0 = [0, 0]
                global result3
                result3 = integrate.odeint(model, y0, t, args = (t03, t1, t2,))
                return w31*result3[:, 0] + w32*result2[:, 1]
    
            def comboFunc(comboData, t01, t02, t03, t1, t2, w11, w12, w21, w22, w31, w32):
                result1 = y1(tdata1, t01, t1, t2, w11, w12)
                result2 = y2(tdata2, t02, t1, t2, w21, w22)
                result3 = y3(tdata3, t03, t1, t2, w31, w32)
                sigma_squared = sum((a_combo - np.append(np.append(result1, result2), result3)) ** 2)
                print('sigma^2 = ', sigma_squared)
                return np.append(  np.append(result1, result2), result3)
            # some initial parameter values
        
            # curve fit the combined data to the combined function
            t_combo = []
            a_combo = []
            for i in dataset:
                t_combo = np.append(t_combo, i[0])
                a_combo = np.append(a_combo, i[1])
                
            data1 = dataset[0]
            data2 = dataset[1]
            data3 = dataset[2]
            
            tdata1, adata1 = data1
            tdata2, adata2 = data2
            tdata3, adata3 = data3
                            
            fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, self.x0, bounds = self.bounds)
            
            t01, t02, t03, t1, t2, w11, w12, w21, w22, w31, w32 = fittedParameters
            t01err, t02err, t03err, t1err, t2err, w11err, w12err, w21err, w22err, w31err, w32err = np.sqrt(np.diag(pcov))
            print('t01err = %.2f, t02err = %.2f, t03err = %.2f, t1err = %.2f, t2err = %.2f, w11err = %.2f, w12err = %.2f, w21err = %.2f, w22err = %.2f, w31err = %.2f, w32err = %.2f' % tuple((A1err, A2err, A3err, t1err, t2err, w11err, w12err, w21err, w22err, w31err, w32err)))
    
            plt.figure()
            g = G(tdata1, 1, t01, self.Wpump)
            plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
            plt.plot(tdata1, g*max(adata1)/max(g), color = '#b676f2', label = 'Vpump scaled')
            plt.axvline(x = t01, color = '#F1B929', label = 'Time-Zero')
            plt.plot(tdata1, w11*result1[:,0], 'r-', label = 'w1*N1')
            plt.plot(tdata1, w12*result1[:,1], 'b-', label = 'w2*N2')
            fit_result = w11*result1[:,0] + w12*result1[:,1]
            plt.plot(tdata1, fit_result, 'k--', label = 'w1*N1 + w2*N2')
            plt.scatter(tdata1, adata1)
            plt.legend()
            wsum = w11 + w12
            w11 = w11 / wsum
            w12 = w12 / wsum
            plt.title('t0 = %.2f, t1 = %.2f fs, t2 = %.2f fs, w1 = %.2f, w2 = %.2f' % tuple((t01, t1, t2, w11, w12)))
            plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
            plt.ylabel('ΔA (mOD)')
            
            plt.figure()
            g = G(tdata2, 1, t02, self.Wpump)
            plt.plot(tdata2, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
            plt.plot(tdata2, g*max(adata2)/max(g), color = '#b676f2', label = 'Vpump scaled')
            plt.axvline(x = t02, color = '#F1B929', label = 'Time-Zero')
            plt.plot(tdata2, w21*result2[:, 0], 'r-', label = 'w1*N1')
            plt.plot(tdata2, w22*result2[:, 1], 'b-', label = 'w2*N2')
            fit_result = w21*result2[:,0] + w22*result2[:,1]
            plt.plot(tdata2, fit_result, 'k--', label = 'w1*N1 + w2*N2')
            plt.scatter(tdata2, adata2)
            plt.legend()
            wsum = w21 + w22
            w21 = w21 / wsum
            w22 = w22 / wsum
            plt.title('t0 = %.2f, t1 = %.2f fs, t2 = %.2f fs, w1 = %.2f, w2 = %.2f' % tuple((t02, t1, t2, w21, w22)))
            plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[1])
            plt.ylabel('ΔA (mOD)')
            
            plt.figure()
            g = G(tdata3, 1, t03, self.Wpump)
            plt.plot(tdata3, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
            plt.plot(tdata3, g*max(adata3)/max(g), color = '#b676f2', label = 'Vpump scaled')
            plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
            plt.plot(tdata3, w31*result3[:, 0], 'r-', label = 'w1*N1')
            plt.plot(tdata2, w32*result3[:, 1], 'b-', label = 'w2*N2')
            fit_result = w31*result3[:,0] + w32*result3[:,1]
            plt.plot(tdata3, fit_result, 'k--', label = 'w1*N1 + w2*N2')
            plt.scatter(tdata3, adata3)
            plt.legend()
            wsum = w31 + w32
            w31 = w31 / wsum
            w32 = w32 / wsum
            plt.title('t0 = %.2f, t1 = %.2f fs, t2 = %.2f fs, w1 = %.2f, w2 = %.2f' % tuple((t03, t1, t2, w31, w32)))
            plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[2])
            plt.ylabel('ΔA (mOD)')
            plt.show()
            
            return t01, t02, t03, t1, t2, w11, w12, w21, w22, w31, w32, result1, result2, result3
    
    def TA_parallel(self):
        dataset = Data(self.file, self.wavelengths, self.tstart, self.tstop, self.tshift).initialize()
        
        tdata = dataset[0]
        adata = dataset[1]
        
        if len(self.wavelengths) == 1: #parallel 1 wavelength
            def G(t, A, t0, Wpump):
                return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
            
            def model(y, t, A, t1, t2, t3):
                """
                Samantha
                """
                N1 = y[0]
                N2 = y[1]
                N3 = y[2]
                dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1 - N1/t3
                dN2_dt = N1/t1 - N2/t2
                dN3_dt = N1/t3 - N3/t2
                return [dN1_dt, dN2_dt, dN3_dt]
    
            def y(t, A, t1, t2, t3, w1, w2, w3):
                y0 = [0, 0, 0]
                global result
                result = integrate.odeint(model, y0, tdata, args = (A, t1, t2, t3,))
                sigma_squared = sum((a_combo - result) ** 2)
                print('sigma^2 = ', sigma_squared)
                return w1*result[:,0] + w2*result[:,1] + w3*result[:,2]
    
    
            popt, pcov = optimize.curve_fit(lambda t, A, t1, t2, t3, w1, w2, w3: y(t, A, t1, t2, t3, w1, w2, w3), tdata, adata, self.x0, bounds = self.bounds, maxfev = 1000000)
            A, t1, t2, t3, w1, w2, w3 = popt
            perr = np.sqrt(np.diag(pcov))
            
            # print('w1 + w2 + w3 = ', popt[4] + popt[5] + popt[6])
            # print('fitted parameters = ', popt)
            # print('covar matrix = ', pcov)
            # print('Errors = ', perr)
    
            fit_result = w1*result[:,0] + w2*result[:,1] + w3*result[:,2]
    
            sigma_squared = sum((adata - fit_result) ** 2)
            print('sigma^2 = ', sigma_squared)
               
            plt.figure()
            A = popt[0]
            g = G(tdata, A, self.t0, self.Wpump)
            plt.plot(tdata, g, color = '#b676f2', label = 'Vpump')
            plt.plot(tdata, g*max(adata)/max(g), color = '#b676f2', linestyle = 'dashed', label = 'Vpump scaled')
            plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
            plt.plot(tdata, w1*result[:,0], 'r-', label = 'w1*N1')
            plt.plot(tdata, w2*result[:,1], 'b-', label = 'w2*N2')
            plt.plot(tdata, w3*result[:,2], 'g-', label = 'w3*N3')
            fit_result = w1*result[:,0] + w2*result[:,1] + w3*result[:,2]
            plt.plot(tdata, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3')
            plt.scatter(tdata, adata)
            plt.legend()
            
            A, t1, t2, t3, w1, w2, w3 = popt
            print(w1, w2, w3)
            wsum = w1 + w2 + w3
            w1 = w1 / wsum
            w2 = w2 / wsum
            w3 = w3 / wsum
            
            print(w1, w2, w3)
            
            plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f fs, w1 = %.2f, w2 = %.2f, w3 = %.2f' % tuple((A, t1, t2, t3, w1, w2, w3)))
            #plt.figtext(0.5, 0.01, wavelength, wrap=True, horizontalalignment='center', fontsize=12)
            plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
            plt.ylabel('ΔA (mOD)')
            plt.show()
    
            TA_return = [A, t1, t2, t3, w1, w2, w3, result]
            
        if len(self.wavelengths) == 2: #parallel 2 wavelengths

            def G(t, A, t0, Wpump):
                return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
            
            def model(y, t, A, t1, t2, t3, t4):
                """
                Samantha
                """
                N1 = y[0]
                N2 = y[1]
                N3 = y[2]
                N4 = y[3]
                dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1 - N1/t3
                dN2_dt = N1/t1 - N2/t2
                dN3_dt = N1/t3 - N3/t4
                dN4_dt = N3/t4 + N2/t2
                return [dN1_dt, dN2_dt, dN3_dt, dN4_dt]
    
            def y1(t, A1, t1, t2, t3, t4, w11, w12, w13, w14):
                y0 = [0, 0, 0, 0] # the states are initially unoccupied
                global result1
                result1 = integrate.odeint(model, y0, t, args = (A1, t1, t2, t3, t4,))
                return w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2] + w14*result1[:, 3]
    
            def y2(t, A2, t1, t2, t3, t4, w21, w22, w23, w24):
                y0 = [0, 0, 0, 0]
                global result2
                result2 = integrate.odeint(model, y0, t, args = (A2, t1, t2, t3, t4,))
                return w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2] + w24*result1[:, 3]
    
            def comboFunc(comboData, A1, A2, t1, t2, t3, t4, w11, w12, w13, w14, w21, w22, w23, w24):
                result1 = y1(tdata1, A1, t1, t2, t3, t4, w11, w12, w13, w14)
                result2 = y2(tdata2, A2, t1, t2, t3, t4, w21, w22, w23, w24)
                sigma_squared = sum((a_combo - np.append(result1, result2)))
                print('sigma^2 = ', sigma_squared)
                return np.append(result1, result2)
            # some initial parameter values
        
            # curve fit the combined data to the combined function
            t_combo = []
            a_combo = []
            for i in dataset:
                t_combo = np.append(t_combo, i[0])
                a_combo = np.append(a_combo, i[1])
                
            data1 = dataset[0]
            data2 = dataset[1]
            
            tdata1, adata1 = data1
            tdata2, adata2 = data2                
            
            fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, self.x0, bounds = self.bounds)
            
            A1, A2, t1, t2, t3, t4, w11, w12, w13, w14, w21, w22, w23, w24 = fittedParameters
            A1err, A2err, t1err, t2err, t3err, t4err, w11err, w12err, w13err, w14err, w21err, w22err, w23err, w24err = np.sqrt(np.diag(pcov))
            print('A1err = %.2f, A2err = %.2f, t1err = %.2f, t2err = %.2f, t3err = %.2f, t4err = %.2f, w11err = %.2f, w12err = %.2f, w13err = %.2f, w14err = %.2f, w21err = %.2f, w22err = %.2f, w23err = %.2f, w24err = %.2f' % tuple((A1err, A2err, t1err, t2err, t3err, t4err, w11err, w12err, w13err, w14err, w21err, w22err, w23err, w24err)))
      
     
            plt.figure()
            g = G(tdata1, A1, self.t0, self.Wpump)
            plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
            plt.plot(tdata1, g*max(adata1)/max(g), color = '#b676f2', label = 'Vpump scaled')
            plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
            plt.plot(tdata1, w11*result1[:, 0], 'r-', label = 'w1*N1')
            plt.plot(tdata1, w12*result1[:, 1], 'b-', label = 'w2*N2')
            plt.plot(tdata1, w13*result1[:, 2], 'g-', label = 'w3*N3')
            fit_result = w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2] + w14*result1[:, 3]
            plt.plot(tdata1, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3')
            plt.scatter(tdata1, adata1)
            plt.legend()
            # wsum = w11 + w12 + w13
            # w11 = w11 / wsum
            # w12 = w12 / wsum
            # w13 = w13 / wsum
            plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f fs, t4 = %.2f fs, w1 = %.2f, w2 = %.2f, w3 = %.2f' % tuple((A1, t1, t2, t3, t4, w11, w12, w13)))
            plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
            plt.ylabel('ΔA (mOD)')
            
            plt.figure()
            g = G(tdata2, A2, self.t0, self.Wpump)
            plt.plot(tdata2, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
            plt.plot(tdata2, g*max(adata2)/max(g), color = '#b676f2', label = 'Vpump scaled')
            plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
            plt.plot(tdata2, w21*result2[:, 0], 'r-', label = 'w1*N1')
            plt.plot(tdata2, w22*result2[:, 1], 'b-', label = 'w2*N2')
            plt.plot(tdata2, w23*result2[:, 2], 'g-', label = 'w3*N3')
            fit_result = w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2] + w24*result2[:, 3]
            plt.plot(tdata2, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3')
            plt.scatter(tdata2, adata2)
            plt.legend()
            # wsum = w21 + w22 + w23
            # w21 = w21 / wsum
            # w22 = w22 / wsum
            # w23 = w23 / wsum
            plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f fs, t4 = %.2f fs, w1 = %.2f, w2 = %.2f, w3 = %.2f' % tuple((A1, t1, t2, t3, t4, w21, w22, w23)))
            plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[1])
            plt.ylabel('ΔA (mOD)')
            plt.show()
            
            return A1, A2, t1, t2, t3, t4, w11, w12, w13, w14, w21, w22, w23, w24, result1, result2

        if len(self.wavelengths) == 3: #parallel 3 wavelengths
            def G(t, A, t0, Wpump):
                return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
             
            def model(y, t, A, t1, t2, t3, t4):
                """
                Samantha
                """
                N1 = y[0]
                N2 = y[1]
                N3 = y[2]
                N4 = y[3]
                dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1 - N1/t3
                dN2_dt = N1/t1 - N2/t2
                dN3_dt = N1/t3 - N3/t4
                dN4_dt = N3/t4 + N2/t2
                return [dN1_dt, dN2_dt, dN3_dt, dN4_dt]
    
            # def y1(t, A1, t1, t2, t3, t4, w11, w12, w14): #w14 should be bound to 0 in input file since it is the ground state
                # w13 = 1 - w11 - w12 - w14
            def y1(t, A1, t1, t2, t3, t4, w11, w12, w13, w14): #w14 should be bound to 0 in input file since it is the ground state
                y0 = [0, 0, 0, 0] # the stat es are initially unoccupied
                global result1
                result1 = integrate.odeint(model, y0, t, args = (A1, t1, t2, t3, t4,))
                return w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2] + w14*result1[:, 3]
    
            # def y2(t, A2, t1, t2, t3, t4, w21, w22, w24):
            #     w23 = 1 - w21 - w22 - w24
            def y2(t, A2, t1, t2, t3, t4, w21, w22, w23, w24):
                y0 = [0, 0, 0, 0]
                global result2
                result2 = integrate.odeint(model, y0, t, args = (A2, t1, t2, t3, t4,))
                return w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2] + w24*result1[:, 3]
                
            # def y3(t, A3, t1, t2, t3, t4, w31, w32, w34):
            #     w33 = 1 - w31 - w32 - w34
            def y3(t, A3, t1, t2, t3, t4, w31, w32, w33, w34):
                y0 = [0, 0, 0, 0]
                global result3
                result3 = integrate.odeint(model, y0, t, args = (A3, t1, t2, t3, t4,))
                return w31*result3[:, 0] + w32*result3[:, 1] + w33*result3[:, 2] + w34*result1[:, 3]
    
            # def comboFunc(comboData, A1, A2, A3, t1, t2, t3, t4, w11, w12, w14, w21, w22, w24, w31, w32, w34):
            #     result1 = y1(tdata1, A1, t1, t2, t3, t4, w11, w12, w14)
            #     result2 = y2(tdata2, A2, t1, t2, t3, t4, w21, w22, w24)
            #     result3 = y3(tdata3, A3, t1, t2, t3, t4, w31, w32, w34)
            def comboFunc(comboData, A1, A2, A3, t1, t2, t3, t4, w11, w12, w13, w14, w21, w22, w23, w24, w31, w32, w33, w34):
                result1 = y1(tdata1, A1, t1, t2, t3, t4, w11, w12, w13, w14)
                result2 = y2(tdata2, A2, t1, t2, t3, t4, w21, w22, w23, w24)
                result3 = y3(tdata3, A3, t1, t2, t3, t4, w31, w32, w33, w34)
                sigma_squared = sum((a_combo - np.append(  np.append(result1, result2), result3)))
                print('sigma^2 = ', sigma_squared)
                return np.append(  np.append(result1, result2), result3)
            # some initial parameter values
        
            # curve fit the combined data to the combined function
            t_combo = []
            a_combo = []
            for i in dataset:
                t_combo = np.append(t_combo, i[0])
                a_combo = np.append(a_combo, i[1])
                
            data1 = dataset[0]
            data2 = dataset[1]
            data3 = dataset[2]
            
            tdata1, adata1 = data1
            tdata2, adata2 = data2
            tdata3, adata3 = data3
            
            if self.tshift == True:
                 t1_idx, tzero = find_nearest(tdata1, self.t0)
                 shift_height = adata1[t1_idx] / np.max(abs(adata1))
                 tdata1 = tdata1 - tzero
                
                 t2_idx, t2 = find_nearest(adata2, shift_height * np.max(abs(adata2)))
                 tdata2 = tdata2 - t2
                
                 t3_idx, t3 = find_nearest(adata3, shift_height * np.max(abs(adata3)))
                 tdata3 = tdata3 - t3
            else:
                pass
            
            fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, self.x0, bounds = self.bounds)
            
            # A1, A2, A3, t1, t2, t3, t4, w11, w12, w14, w21, w22, w24, w31, w32, w34 = fittedParameters
            # A1err, A2err, A3err, t1err, t2err, t3err, t4err, w11err, w12err, w14err, w21err, w22err,  w24err, w31err, w32err, w34err = np.sqrt(np.diag(pcov))
            # print('A1err = %.2f, A2err = %.2f, A3err = %.2f, t1err = %.2f, t2err = %.2f, t3err = %.2f, t4err = %.2f, w11err = %.2f, w12err = %.2f, w14err = %.2f, w21err = %.2f, w22err = %.2f, w24err = %.2f, w31err = %.2f, w32err = %.2f, w34err = %.2f' % tuple((A1err, A2err, A3err, t1err, t2err, t3err, t4err, w11err, w12err, w14err, w21err, w22err, w24err, w31err, w32err, w34err)))
    
            # w13 = 1 - w11 - w12 - w14
            # w23 = 1 - w21 - w22 - w24
            # w33 = 1 - w31 - w32 - w34
            
            A1, A2, A3, t1, t2, t3, t4, w11, w12, w13, w14, w21, w22, w23, w24, w31, w32, w33, w34 = fittedParameters
            A1err, A2err, A3err, t1err, t2err, t3err, t4err, w11err, w12err, w13err, w14err, w21err, w22err, w23err,  w24err, w31err, w32err, w33err, w34err = np.sqrt(np.diag(pcov))
            print('A1err = %.2f, A2err = %.2f, A3err = %.2f, t1err = %.2f, t2err = %.2f, t3err = %.2f, t4err = %.2f, w11err = %.2f, w12err = %.2f, w13err = %.2f, w14err = %.2f, w21err = %.2f, w22err = %.2f, w23err = %.2f, w24err = %.2f, w31err = %.2f, w32err = %.2f, w33err = %.2f, w34err = %.2f' % tuple((A1err, A2err, A3err, t1err, t2err, t3err, t4err, w11err, w12err, w13err, w14err, w21err, w22err, w23err, w24err, w31err, w32err, w33err, w34err)))
            
            
            plt.figure()
            g = G(tdata1, A1, self.t0, self.Wpump)
            plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
            plt.plot(tdata1, g*max(adata1)/max(g), color = '#b676f2', label = 'Vpump scaled')
            plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
            plt.plot(tdata1, w11*result1[:, 0], 'r-', label = 'w1*N1')
            plt.plot(tdata1, w12*result1[:, 1], 'b-', label = 'w2*N2')
            plt.plot(tdata1, w13*result1[:, 2], 'g-', label = 'w3*N3')
            fit_result = w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2] + w14*result1[:, 3]
            plt.plot(tdata1, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3')
            plt.scatter(tdata1, adata1)
            plt.legend()
            # wsum = w11 + w12 + w13 + w14
            # w11 = w11 / wsum
            # w12 = w12 / wsum
            # w13 = w13 / wsum
            # w14 = w14 / wsum
            plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f fs, t4 = %.2f fs w1 = %.2f, w2 = %.2f, w3 = %.2f' % tuple((A1, t1, t2, t3, t4, w11, w12, w13)))
            plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
            plt.ylabel('ΔA (mOD)')
            
            plt.figure()
            g = G(tdata2, A2, self.t0, self.Wpump)
            plt.plot(tdata2, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
            plt.plot(tdata2, g*max(adata2)/max(g), color = '#b676f2', label = 'Vpump scaled')
            plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
            plt.plot(tdata2, w21*result2[:, 0], 'r-', label = 'w1*N1')
            plt.plot(tdata2, w22*result2[:, 1], 'b-', label = 'w2*N2')
            plt.plot(tdata2, w23*result2[:, 2], 'g-', label = 'w3*N3')
            fit_result = w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2] + w24*result2[:, 3]
            plt.plot(tdata2, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3')
            plt.scatter(tdata2, adata2)
            plt.legend()
            # wsum = w21 + w22 + w23 + w24
            # w21 = w21 / wsum
            # w22 = w22 / wsum
            # w23 = w23 / wsum
            # w24 = w24 / wsum
            plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f fs w1 = %.2f, w2 = %.2f, w3 = %.2f' % tuple((A2, t1, t2, t3, t4, w21, w22, w23)))
            plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[1])
            plt.ylabel('ΔA (mOD)')
            
            plt.figure()
            g = G(tdata3, A3, self.t0, self.Wpump)
            plt.plot(tdata3, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
            plt.plot(tdata3, g*max(adata3)/max(g), color = '#b676f2', label = 'Vpump scaled')
            plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
            plt.plot(tdata3, w31*result3[:, 0], 'r-', label = 'w1*N1')
            plt.plot(tdata3, w32*result3[:, 1], 'b-', label = 'w2*N2')
            plt.plot(tdata3, w33*result3[:, 2], 'g-', label = 'w3*N3')
            fit_result = w31*result3[:, 0] + w32*result3[:, 1] + w33*result3[:, 2] + w34*result3[:, 3]
            plt.plot(tdata3, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3')
            plt.scatter(tdata3, adata3)
            plt.legend()
            # wsum = w31 + w32 + w33 + w34
            # w31 = w31 / wsum
            # w32 = w32 / wsum
            # w33 = w33 / wsum
            # w34 = w34 / wsum
            plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f fs w1 = %.2f, w2 = %.2f, w3 = %.2f' % tuple((A3, t1, t2, t3, t4, w31, w32, w33)))
            plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[2])
            plt.ylabel('ΔA (mOD)')
            plt.show()
            
            return A1, A2, A3, t1, t2, t3, t4, w11, w12, w13, w14, w21, w22, w23, w24, w31, w32, w33, w34, result1, result2, result3

    
    def TA(self): #sequential linear model
        dataset = Data(self.file, self.wavelengths, self.tstart, self.tstop, self.tshift).initialize()
        
        if len(self.wavelengths) == 1: #sequential linear 1 wavelength
            # fit single wavelength
            
            def G(t, A, t0, Wpump):
                return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)     
            
            tdata = dataset[0]
            adata = dataset[1]
            
            if self.number_of_states == 1: #sequential linear 1 wavelength 1 state
                def model(y, t, A, t1):
                    N1 = y[0]
                    dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                    return [dN1_dt]
        
                def y(t, A, t1):
                    y0 = [0]
                    global result
                    result = integrate.odeint(model, y0, tdata, args = (A, t1,))
                    return result[:,0]
                popt, pcov = optimize.curve_fit(lambda t, A, t1: y(t, A, t1), tdata, adata, self.x0, maxfev = 1000000)
                
                A, t1 = popt
                
                perr = np.sqrt(np.diag(pcov))
                print('fitted parameters = ', popt)
                print('Errors = ', perr)
                plt.figure()
                plt.plot(tdata, G(tdata, popt[0], self.t0, self.Wpump), 'y-', label = 'Vpump')
                fit_result = result[:,0]
                plt.plot(tdata, fit_result, 'k--', label = 'N1')
                plt.scatter(tdata, adata)
                plt.legend()
                popt[0] = 1 / popt[0]
                plt.title('A = %.2f, t1 = %.2f fs' % tuple(popt))
                #plt.figtext(0.5, 0.01, wavelength, wrap=True, horizontalalignment='center', fontsize=12)
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
                plt.ylabel('ΔA (mOD)')
                plt.show()
                
                TA_return = [A, t1, result]
            
            elif self.number_of_states == 2: #sequential linear 1 wavelength 2 states
                def model(y, t, A, t1, t2):
                    N1 = y[0]
                    N2 = y[1]
                    dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                    dN2_dt = N1/t1 - N2/t2
                    return [dN1_dt, dN2_dt]
                
                def y(t, A, t1, t2, w1, w2):
                    y0 = [0, 0] # the states are initially unoccupied
                    global result
                    result = integrate.odeint(model, y0, tdata, args = (A, t1, t2,))
                    return w1*result[:, 0] + w2*result[:, 1]

                popt, pcov = optimize.curve_fit(lambda t, A, t1, t2, w1, w2: y(t, A, t1, t2, w1, w2), tdata, adata, self.x0, bounds = self.bounds, maxfev = 1000000)
                
                A, t1, t2, w1, w2 = popt
                perr = np.sqrt(np.diag(pcov))
                
                # print('##################################')
                # #print('num of iterations = ', itercount)
                # print('sum of weights = ', popt[3] + popt[4])
                # print("")
                # print('fitted parameters = ', popt)
                # print("")
                # print('covar matrix = ', pcov)
                # print("")
                # print('Errors = ', perr)
                # print("")
                
                sigma_squared = sum((adata - w1*result[:, 0] + w2*result[:, 1]) ** 2)
                print('sigma^2 = ', sigma_squared)
                
                plt.figure()
                plt.plot(tdata, G(tdata, popt[0], self.t0, self.Wpump), 'y-', label = 'Vpump')
                plt.plot(tdata, popt[3]*result[:,0], 'r-', label = 'w1*N1')
                plt.plot(tdata, popt[4]*result[:,1], 'b-', label = 'w2*N2')
                fit_result = popt[3]*result[:,0] + popt[4]*result[:,1]
                plt.plot(tdata, fit_result, 'k--', label = 'w1*N1 + w2*N2')
                plt.scatter(tdata, adata)
                plt.legend()
                
                wsum = popt[3] + popt[4]
                popt[3] = popt[3] / wsum
                popt[4] = popt[4] / wsum
                
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, w1 = %.2f, w2 = %.2f' % tuple(popt))
                #plt.figtext(0.5, 0.01, wavelength, wrap=True, horizontalalignment='center', fontsize=12)
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
                plt.ylabel('ΔA (mOD)')
                plt.show()
                
                TA_return = [A, t1, t2, w1, w2, result]
                
            elif self.number_of_states == 3: #sequential linear 1 wavelength 3 states
                def model(y, t, A, t1, t2, t3):
                    N1 = y[0]
                    N2 = y[1]
                    N3 = y[2]
                    dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                    dN2_dt = N1/t1 - N2/t2
                    dN3_dt = N2/t2 - N3/t3
                    return [dN1_dt, dN2_dt, dN3_dt]
        
                def y(t, A, t1, t2, t3, w1, w2, w3):
                    y0 = [0, 0, 0]
                    global result
                    result = integrate.odeint(model, y0, tdata, args = (A, t1, t2, t3,))
                    return w1*result[:,0] + w2*result[:,1] + w3*result[:,2]
        
        
                popt, pcov = optimize.curve_fit(lambda t, A, t1, t2, t3, w1, w2, w3: y(t, A, t1, t2, t3, w1, w2, w3), tdata, adata, self.x0, bounds = self.bounds, maxfev = 1000000)
                A, t1, t2, t3, w1, w2, w3 = popt
                perr = np.sqrt(np.diag(pcov))
                
                # print('w1 + w2 + w3 = ', popt[4] + popt[5] + popt[6])
                # print('fitted parameters = ', popt)
                # print('covar matrix = ', pcov)
                # print('Errors = ', perr)
        
                fit_result = result[:,0]
        
                sigma_squared = sum((adata - fit_result) ** 2)
                print('sigma^2 = ', sigma_squared)
                   
                plt.figure()
                A = popt[0]
                g = G(tdata, A, self.t0, self.Wpump)
                plt.plot(tdata, g, color = '#b676f2', label = 'Vpump')
                plt.plot(tdata, g*max(adata)/max(g), color = '#b676f2', linestyle = 'dashed', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata, w1*result[:,0], 'r-', label = 'w1*N1')
                plt.plot(tdata, w2*result[:,1], 'b-', label = 'w2*N1')
                plt.plot(tdata, w3*result[:,2], 'g-', label = 'w3*N1')
                fit_result = w1*result[:,0] + w2*result[:,1] + w3*result[:,2]
                plt.plot(tdata, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3')
                plt.scatter(tdata, adata)
                plt.legend()
                
                A, t1, t2, t3, w1, w2, w3 = popt
                print(w1, w2, w3)
                wsum = w1 + w2 + w3
                w1 = w1 / wsum
                w2 = w2 / wsum
                w3 = w3 / wsum
                
                print(w1, w2, w3)
                
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f fs, w1 = %.2f, w2 = %.2f, w3 = %.2f' % tuple((A, t1, t2, t3, w1, w2, w3)))
                #plt.figtext(0.5, 0.01, wavelength, wrap=True, horizontalalignment='center', fontsize=12)
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
                plt.ylabel('ΔA (mOD)')
                plt.show()
        
                TA_return = [A, t1, t2, t3, w1, w2, w3, result]
        
        
            elif self.number_of_states == 4: #sequential linear 1 wavelength 4 states
                def model(y, t, A, t1, t2, t3, t4):
                    N1 = y[0]
                    N2 = y[1]
                    N3 = y[2]
                    N4 = y[3]
                    dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                    dN2_dt = N1/t1 - N2/t2
                    dN3_dt = N2/t2 - N3/t3
                    dN4_dt = N3/t3 - N4/t4
                    return [dN1_dt, dN2_dt, dN3_dt, dN4_dt]
                
                def y(t, A, t1, t2, t3, t4, w1, w2, w3, w4):
                    y0 = [0, 0, 0, 0]
                    global result
                    result = integrate.odeint(model, y0, tdata, args = (A, t1, t2, t3, t4,))
                    return w1*result[:,0] + w2*result[:,1] + w3*result[:,2] + w4*result[:,3]
        
                popt, pcov = optimize.curve_fit(lambda t, A, t1, t2, t3, t4, w1, w2, w3, w4: y(t, A, t1, t2, t3, t4, w1, w2, w3, w4), tdata, adata, self.x0, bounds = self.bounds, maxfev = 100000)
                A, t1, t2, t3, t4, w1, w2, w3, w4 = popt
                perr = np.sqrt(np.diag(pcov))
                
                print('w1 + w2 + w3 + w4 = ', w1 + w2 + w3 + w4)
                print('fitted parameters = ', popt)
                print('covar matrix = ', pcov)
                print('Errors = ', perr)
        
                fit_result = result[:,0]
                sigma_squared = sum((adata - fit_result) ** 2)
                print('sigma^2 = ', sigma_squared)
        
                plt.figure()
                A = popt[0]
                g = G(tdata, A, self.t0, self.Wpump)
                plt.plot(tdata, g, color = '#b676f2', label = 'Vpump')
                plt.plot(tdata, g*max(adata)/max(g), color = '#b676f2', linestyle = 'dashed', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata, w1*result[:,0], 'r-', label = 'w1*N1')
                plt.plot(tdata, w2*result[:,1], 'g-', label = 'w2*N2')
                plt.plot(tdata, w3*result[:,2], 'b-', label = 'w3*N3')
                plt.plot(tdata, w4*result[:,3], 'm-', label = 'w4*N4')
                fit_result = w1*result[:,0] + w2*result[:,1] + w3*result[:,2] + w4*result[:,3]
                plt.plot(tdata, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4')
                plt.scatter(tdata, adata)
                plt.legend()
                
                wsum = w1 + w2 + w3 + w4
                w1 = w1 / wsum
                w2 = w2 / wsum
                w3 = w3 / wsum
                w4 = w4 / wsum
                
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f' % tuple(popt))
                #plt.figtext(0.5, 0.01, wavelength, wrap=True, horizontalalignment='center', fontsize=12)
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
                plt.ylabel('ΔA (mOD)')
                plt.show()
                
                TA_return = [A, t1, t2, t3, t4, w1, w2, w3, w4, result]
        
            elif self.number_of_states == 5: #sequential linear 1 wavelength 5 states
                def model(y, t, A, t1, t2, t3, t4, t5):
                    N1 = y[0]
                    N2 = y[1]
                    N3 = y[2]
                    N4 = y[3]
                    N5 = y[4]
                    dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                    dN2_dt = N1/t1 - N2/t2
                    dN3_dt = N2/t2 - N3/t3
                    dN4_dt = N3/t3 - N4/t4
                    dN5_dt = N4/t4 - N5/t5
                    return [dN1_dt, dN2_dt, dN3_dt, dN4_dt, dN5_dt]
    
                def y(t, A, t1, t2, t3, t4, t5, w1, w2, w3, w4, w5):
                    y0 = [0, 0, 0, 0, 0]
                    global result
                    result = integrate.odeint(model, y0, tdata, args = (A, t1, t2, t3, t4, t5,))
                    return w1*result[:,0] + w2*result[:,1] + w3*result[:,2] + w4*result[:,3] + w5*result[:,4]
        
                popt, pcov = optimize.curve_fit(lambda t, A, t1, t2, t3, t4, t5, w1, w2, w3, w4, w5: y(t, A, t1, t2, t3, t4, t5, w1, w2, w3, w4, w5), tdata, adata, self.x0, maxfev = 100000)
                A, t1, t2, t3, t4, t5, w1, w2, w3, w4, w5 = popt
                perr = np.sqrt(np.diag(pcov))
                
                print('w1 + w2 + w3 + w4 + w5 = ', w1 + w2 + w3 + w4 + w5)
                print('fitted parameters = ', popt)
                print('covar matrix = ', pcov)
                print('Errors = ', perr)
        
                fit_result = result[:,0]
        
                sigma_squared = sum((adata - fit_result) ** 2)
                print('sigma^2 = ', sigma_squared)
        
                plt.figure()
                A = popt[0]
                g = G(tdata, A, self.t0, self.Wpump)
                plt.plot(tdata, g, color = '#b676f2', label = 'Vpump')
                plt.plot(tdata, g*max(adata)/max(g), color = '#b676f2', linestyle = 'dashed', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata, w1*result[:,0], 'r-', label = 'w1*N1')
                plt.plot(tdata, w2*result[:,1], 'g-', label = 'w2*N2')
                plt.plot(tdata, w3*result[:,2], 'b-', label = 'w3*N3')
                plt.plot(tdata, w4*result[:,3], 'm-', label = 'w4*N4')
                plt.plot(tdata, w5*result[:,4], 'c-', label = 'w5*N5')
                fit_result = w1*result[:,0] + w2*result[:,1] + w3*result[:,2] + w4*result[:,3] + w5*result[:,4]
                plt.plot(tdata, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4 + w5*N5')
                plt.scatter(tdata, adata)
                plt.legend()
                
                wsum = w1 + w2 + w3 + w4 + w5
                w1 = w1 / wsum
                w2 = w2 / wsum
                w3 = w3 / wsum
                w4 = w4 / wsum
                w5 = w5 / wsum
                
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, t5 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f, w5 = %.2f' % tuple(popt))
                #plt.figtext(0.5, 0.01, wavelength, wrap=True, horizontalalignment='center', fontsize=12)
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
                plt.ylabel('ΔA (mOD)')
                plt.show()
                
                TA_return = [A, t1, t2, t3, t4, t5, w1, w2, w3, w4, w5, result]
        
            else:
                print('Your number of states is not currently supported. Please choose a lower number of states.')
                print("", "")
                print("Your outputted units are:", "", "X-axis = fs, ", "Y-axis = mOD, ", "τ = fs, ", "and weights = unitless.")
    
            return TA_return
###############################################################################
            
        elif len(self.wavelengths) == 2: 
            # global fitting of two wavelengths
            if self.number_of_states == 1: #sequential linear 2 wavelengths 1 state
                def G(t, A, t0, Wpump):
                    return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
             
                def model(y, t, A, t1):
                   N1 = y[0]
                   dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                   return [dN1_dt]
            
                def y1(t, A1, t1):
                    y0 = [0] # the states are initially unoccupied
                    global result1
                    result1 = integrate.odeint(model, y0, t, args = (A1, t1,))
                    return result1[:, 0]
        
                def y2(t, A2, t1):
                    y0 = [0]
                    global result2
                    result2 = integrate.odeint(model, y0, t, args = (A2, t1,))
                    return result2[:, 0]
        
                def comboFunc(comboData, A1, A2, t1):
                    result1 = y1(tdata1, A1, t1)
                    result2 = y2(tdata2, A2, t1)
                    return np.append(result1, result2)
             
                t_combo = []
                a_combo = []
                for i in dataset:
                    t_combo = np.append(t_combo, i[0])
                    a_combo = np.append(a_combo, i[1])
                    
                data1 = dataset[0]
                data2 = dataset[1]
                
                tdata1, adata1 = data1
                tdata2, adata2 = data2
                
                fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, self.x0, bounds = self.bounds)
        
                A1, A2, t1 = fittedParameters
                A1err, A2err, t1err = np.sqrt(np.diag(pcov))
                print('A1err = %.2f, A2err = %.2f, t1err = %.2f' % tuple((A1err, A2err, t1err)))
        
                plt.figure()
                g = G(tdata1, A1, self.t0, self.Wpump)
                plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata1, g*max(adata1)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                fit_result = result1[:,0]
                plt.plot(tdata1, fit_result, 'k--', label = 'N1')
                plt.scatter(tdata1, adata1)
                plt.legend()
                plt.title('A = %.2f, t1 = %.2f fs' % tuple((A1, t1)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata2, A2, self.t0, self.Wpump)
                plt.plot(tdata2, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata2, g*max(adata2)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                fit_result = result2[:,0]
                plt.plot(tdata2, fit_result, 'k--', label = 'N1')
                plt.scatter(tdata2, adata2)
                plt.legend()
                plt.title('A = %.2f, t1 = %.2f fs' % tuple((A2, t1)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[1])
                plt.ylabel('ΔA (mOD)')
                plt.show()
                
                return A1, A2, t1, result1, result2  
                  
            elif self.number_of_states == 2: #sequential linear 2 wavelengths 2 states
                def G(t, A, t0, Wpump):
                    return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
                
                def model(y, t, A, t1, t2):
                    N1 = y[0]
                    N2 = y[1]
                    dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                    dN2_dt = N1/t1 - N2/t2
                    return [dN1_dt, dN2_dt]
                
                # def y1(t, A1, t1, t2, w11):
                #     w12 = 1 - w11
                def y1(t, A1, t1, t2, w11, w12):
                    y0 = [0, 0] # the states are initially unoccupied
                    global result1
                    result1 = integrate.odeint(model, y0, t, args = (A1, t1, t2,))
                    return w11*result1[:, 0] + w12*result1[:, 1]
        
                # def y2(t, A2, t1, t2, w21):
                #     w22 = 1 - w21
                def y2(t, A2, t1, t2, w21, w22):
                    y0 = [0, 0]
                    global result2
                    result2 = integrate.odeint(model, y0, t, args = (A2, t1, t2,))
                    return w21*result2[:, 0] + w22*result2[:, 1]
        
                # def comboFunc(comboData, A1, A2, t1, t2, w11, w21):
                #     result1 = y1(tdata1, A1, t1, t2, w11)
                #     result2 = y2(tdata2, A2, t1, t2, w21)
                def comboFunc(comboData, A1, A2, t1, t2, w11, w12, w21, w22):
                    result1 = y1(tdata1, A1, t1, t2, w11, w12)
                    result2 = y2(tdata2, A2, t1, t2, w21, w22)
                    sigma_squared = sum((a_combo - np.append(result1, result2)) ** 2)
                    print('sigma^2 = ', sigma_squared)
                    return np.append(result1, result2)
                # some initial parameter values
            
                # curve fit the combined data to the combined function
                t_combo = []
                a_combo = []
                for i in dataset:
                    t_combo = np.append(t_combo, i[0])
                    a_combo = np.append(a_combo, i[1])
                    
                data1 = dataset[0]
                data2 = dataset[1]
                
                tdata1, adata1 = data1
                tdata2, adata2 = data2
                
                fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, self.x0, bounds = self.bounds)
                
                # A1, A2, t1, t2, w11, w21 = fittedParameters
                # A1err, A2err, t1err, t2err, w11err, w21err = np.sqrt(np.diag(pcov))
                # print('A1err = %.2f, A2err = %.2f, t1err = %.2f, t2err = %.2f, w11err = %.2f, w21err = %.2f' % tuple((A1err, A2err, t1err, t2err, w11err, w21err)))
         
                # w12 = 1 - w11
                # w22 = 1 - w21
                A1, A2, t1, t2, w11, w12, w21, w22 = fittedParameters
                A1err, A2err, t1err, t2err, w11err, w12err, w21err, w22err = np.sqrt(np.diag(pcov))
                print('A1err = %.2f, A2err = %.2f, t1err = %.2f, t2err = %.2f, w11err = %.2f, w12err = %.2f, w21err = %.2f, w22err = %.2f' % tuple((A1err, A2err, t1err, t2err, w11err, w12err, w21err, w22)))
        
                plt.figure()
                g = G(tdata1, A1, self.t0, self.Wpump)
                plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata1, g*max(adata1)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata1, w11*result1[:,0], 'r-', label = 'w1*N1')
                plt.plot(tdata1, w12*result1[:,1], 'b-', label = 'w2*N2')
                fit_result = w11*result1[:,0] + w12*result1[:,1]
                plt.plot(tdata1, fit_result, 'k--', label = 'w1*N1 + w2*N2')
                plt.scatter(tdata1, adata1)
                plt.legend()
                # wsum = w11 + w12
                # w11 = w11 / wsum
                # w12 = w12 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, w1 = %.2f, w2 = %.2f' % tuple((A1, t1, t2, w11, w12)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata2, A2, self.t0, self.Wpump)
                plt.plot(tdata2, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata2, g*max(adata2)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata2, w21*result2[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata2, w22*result2[:, 1], 'b-', label = 'w2*N2')
                fit_result = w21*result2[:,0] + w22*result2[:,1]
                plt.plot(tdata2, fit_result, 'k--', label = 'w1*N1 + w2*N2')
                plt.scatter(tdata2, adata2)
                plt.legend()
                # wsum = w21 + w22
                # w21 = w21 / wsum
                # w22 = w22 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, w1 = %.2f, w2 = %.2f' % tuple((A2, t1, t2, w21, w22)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[1])
                plt.ylabel('ΔA (mOD)')
                plt.show()
                
                return A1, A2, t1, t2, w11, w12, w21, w22, result1, result2
     
            elif self.number_of_states == 3: #sequential linear 2 wavelengths 3 states
                def G(t, A, t0, Wpump):
                    return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
                
                def model(y, t, A, t1, t2, t3):
                    N1 = y[0]
                    N2 = y[1]
                    N3 = y[2]
                    dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                    dN2_dt = N1/t1 - N2/t2
                    dN3_dt = N2/t2 - N3/t3
                    return [dN1_dt, dN2_dt, dN3_dt]
        
                def y1(t, A1, t1, t2, t3, w11, w12, w13):
                    y0 = [0, 0, 0] # the states are initially unoccupied
                    global result1
                    result1 = integrate.odeint(model, y0, t, args = (A1, t1, t2, t3,))
                    return w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2]
        
                def y2(t, A2, t1, t2, t3, w21, w22, w23):
                    y0 = [0, 0, 0]
                    global result2
                    result2 = integrate.odeint(model, y0, t, args = (A2, t1, t2, t3,))
                    return w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2]
        
                def comboFunc(comboData, A1, A2, t1, t2, t3, w11, w12, w13, w21, w22, w23):
                    result1 = y1(tdata1, A1, t1, t2, t3, w11, w12, w13)
                    result2 = y2(tdata2, A2, t1, t2, t3, w21, w22, w23)
                    sigma_squared = sum((a_combo - np.append(result1, result2)) ** 2)
                    print('sigma^2 = ', sigma_squared)
                    return np.append(result1, result2)
                # some initial parameter values
            
                # curve fit the combined data to the combined function
                t_combo = []
                a_combo = []
                for i in dataset:
                    t_combo = np.append(t_combo, i[0])
                    a_combo = np.append(a_combo, i[1])
                    
                data1 = dataset[0]
                data2 = dataset[1]
                
                tdata1, adata1 = data1
                tdata2, adata2 = data2                
                
                fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, self.x0, bounds = self.bounds)
                
                A1, A2, t1, t2, t3, w11, w12, w13, w21, w22, w23 = fittedParameters
                A1err, A2err, t1err, t2err, t3err, w11err, w12err, w13err, w21err, w22err, w23err = np.sqrt(np.diag(pcov))
                print('A1err = %.2f, A2err = %.2f, t1err = %.2f, t2err = %.2f, t3err = %.2f, w11err = %.2f, w12err = %.2f, w13err = %.2f, w21err = %.2f, w22err = %.2f, w23err = %.2f' % tuple((A1err, A2err, t1err, t2err, t3err, w11err, w12err, w13err, w21err, w22err, w23err)))
        
                plt.figure()
                g = G(tdata1, A1, self.t0, self.Wpump)
                plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata1, g*max(adata1)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata1, w11*result1[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata1, w12*result1[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata1, w13*result1[:, 2], 'g-', label = 'w3*N3')
                fit_result = w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2]
                plt.plot(tdata1, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3')
                plt.scatter(tdata1, adata1)
                plt.legend()
                wsum = w11 + w12 + w13
                w11 = w11 / wsum
                w12 = w12 / wsum
                w13 = w13 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f fs, w1 = %.2f, w2 = %.2f, w3 = %.2f' % tuple((A1, t1, t2, t3, w11, w12, w13)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata2, A2, self.t0, self.Wpump)
                plt.plot(tdata2, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata2, g*max(adata2)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata2, w21*result2[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata2, w22*result2[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata2, w23*result2[:, 2], 'g-', label = 'w3*N3')
                fit_result = w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2]
                plt.plot(tdata2, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3')
                plt.scatter(tdata2, adata2)
                plt.legend()
                wsum = w21 + w22 + w23
                w21 = w21 / wsum
                w22 = w22 / wsum
                w23 = w23 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f' % tuple((A2, t1, t2, t3, w21, w22, w23)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[1])
                plt.ylabel('ΔA (mOD)')
                plt.show()
                
                return A1, A2, t1, t2, t3, w11, w12, w13, w21, w22, w23, result1, result2
     
            elif self.number_of_states == 4: #sequential linear 2 wavelengths 4 states
                def G(t, A, t0, Wpump):
                    return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
                
                def model(y, t, A, t1, t2, t3, t4):
                    N1 = y[0]
                    N2 = y[1]
                    N3 = y[2]
                    N4 = y[3]
                    dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                    dN2_dt = N1/t1 - N2/t2
                    dN3_dt = N2/t2 - N3/t3
                    dN4_dt = N3/t3 - N4/t4
                    return [dN1_dt, dN2_dt, dN3_dt, dN4_dt]
        
                def y1(t, A1, t1, t2, t3, t4, w11, w12, w13, w14):
                    y0 = [0, 0, 0, 0] # the states are initially unoccupied
                    global result1
                    result1 = integrate.odeint(model, y0, t, args = (A1, t1, t2, t3, t4,))
                    return w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2] + w14*result1[:, 3]
        
                def y2(t, A2, t1, t2, t3, t4, w21, w22, w23, w24):
                    y0 = [0, 0, 0, 0]
                    global result2
                    result2 = integrate.odeint(model, y0, t, args = (A2, t1, t2, t3, t4,))
                    return w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2] + w24*result2[:, 3]
        
                def comboFunc(comboData, A1, A2, t1, t2, t3, t4, w11, w12, w13, w14, w21, w22, w23, w24):
                    result1 = y1(tdata1, A1, t1, t2, t3, t4, w11, w12, w13, w14)
                    result2 = y2(tdata2, A2, t1, t2, t3, t4, w21, w22, w23, w24)
                    return np.append(result1, result2)
                # some initial parameter values
            
                # curve fit the combined data to the combined function
                t_combo = []
                a_combo = []
                for i in dataset:
                    t_combo = np.append(t_combo, i[0])
                    a_combo = np.append(a_combo, i[1])
                    
                data1 = dataset[0]
                data2 = dataset[1]
                
                tdata1, adata1 = data1
                tdata2, adata2 = data2
                                
                fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, self.x0, bounds = self.bounds)
                
                A1, A2, t1, t2, t3, t4, w11, w12, w13, w14, w21, w22, w23, w24 = fittedParameters
                A1err, A2err, t1err, t2err, t3err, t4err, w11err, w12err, w13err, w14err, w21err, w22err, w23err, w24err = np.sqrt(np.diag(pcov))
                print('A1err = %.2f, A2err = %.2f, t1err = %.2f, t2err = %.2f, t3err = %.2f, t4err = %.2f, w11err = %.2f, w12err = %.2f, w13err = %.2f, w14err = %.2f, w21err = %.2f, w22err = %.2f, w23err = %.2f, w24err = %.2f' % tuple((A1err, A2err, t1err, t2err, t3err, t4err, w11err, w12err, w13err, w14err, w21err, w22err, w23err, w24err)))
        
                plt.figure()
                g = G(tdata1, A1, self.t0, self.Wpump)
                plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata1, g*max(adata1)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata1, w11*result1[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata1, w12*result1[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata1, w13*result1[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata1, w14*result1[:, 3], 'm-', label = 'w4*N4')
                fit_result = w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2] + w14*result1[:, 3]
                plt.plot(tdata1, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4')
                plt.scatter(tdata1, adata1)
                plt.legend()
                wsum = w11 + w12 + w13 + w14
                w11 = w11 / wsum
                w12 = w12 / wsum
                w13 = w13 / wsum
                w14 = w14 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f fs, t4 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f' % tuple((A1, t1, t2, t3, t4, w11, w12, w13, w14)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata2, A2, self.t0, self.Wpump)
                plt.plot(tdata2, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata2, g*max(adata2)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata2, w21*result2[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata2, w22*result2[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata2, w23*result2[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata2, w24*result2[:, 3], 'm-', label = 'w4*N4')
                fit_result = w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2] + w24*result2[:, 3]
                plt.plot(tdata2, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4')
                plt.scatter(tdata2, adata2)
                plt.legend()
                wsum = w21 + w22 + w23 + w24
                w21 = w21 / wsum
                w22 = w22 / wsum
                w23 = w23 / wsum
                w24 = w24 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f' % tuple((A2, t1, t2, t3, t4, w21, w22, w23, w24)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[1])
                plt.ylabel('ΔA (mOD)')
                plt.show()
                
                return A1, A2, t1, t2, t3, t4, w11, w12, w13, w14, w21, w22, w23, w24, result1, result2
    
            elif self.number_of_states == 5: #sequential linear 2 wavelengths 5 states
                def G(t, A, t0, Wpump):
                    return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
                
                def model(y, t, A, t1, t2, t3, t4, t5):
                    N1 = y[0]
                    N2 = y[1]
                    N3 = y[2]
                    N4 = y[3]
                    N5 = y[4]
                    dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                    dN2_dt = N1/t1 - N2/t2
                    dN3_dt = N2/t2 - N3/t3
                    dN4_dt = N3/t3 - N4/t4
                    dN5_dt = N4/t4 - N5/t5
                    return [dN1_dt, dN2_dt, dN3_dt, dN4_dt, dN5_dt]
        
                def y1(t, A1, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15):
                    y0 = [0, 0, 0, 0, 0] # the states are initially unoccupied
                    global result1
                    result1 = integrate.odeint(model, y0, t, args = (A1, t1, t2, t3, t4, t5,))
                    return w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2] + w14*result1[:, 3] + w15*result1[:, 4]
        
                def y2(t, A2, t1, t2, t3, t4, t5, w21, w22, w23, w24, w25):
                    y0 = [0, 0, 0, 0, 0]
                    global result2
                    result2 = integrate.odeint(model, y0, t, args = (A2, t1, t2, t3, t4, t5,))
                    return w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2] + w24*result2[:, 3] + w25*result2[:, 4]
        
                def comboFunc(comboData, A1, A2, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15, w21, w22, w23, w24, w25):
                    result1 = y1(tdata1, A1, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15)
                    result2 = y2(tdata2, A2, t1, t2, t3, t4, t5, w21, w22, w23, w24, w25)
                    return np.append(result1, result2)
                # some initial parameter values
            
                # curve fit the combined data to the combined function
                t_combo = []
                a_combo = []
                for i in dataset:
                    t_combo = np.append(t_combo, i[0])
                    a_combo = np.append(a_combo, i[1])
                    
                data1 = dataset[0]
                data2 = dataset[1]
                
                tdata1, adata1 = data1
                tdata2, adata2 = data2
                             
                fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, self.x0, bounds = self.bounds)
                
                A1, A2, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15, w21, w22, w23, w24, w25 = fittedParameters
                A1err, A2err, t1err, t2err, t3err, t4err, t5err, w11err, w12err, w13err, w14err, w15err, w21err, w22err, w23err, w24err, w25err = np.sqrt(np.diag(pcov))
                print('A1err = %.2f, A2err = %.2f, t1err = %.2f, t2err = %.2f, t3err = %.2f, t4err = %.2f, t5err = %.2f, w11err = %.2f, w12err = %.2f, w13err = %.2f, w14err = %.2f, w15err = %.2f, w21err = %.2f, w22err = %.2f, w23err = %.2f, w24err = %.2f w25err = %.2f' % tuple((A1err, A2err, t1err, t2err, t3err, t4err, t5err, w11err, w12err, w13err, w14err, w15err, w21err, w22err, w23err, w24err, w25err)))
        
                plt.figure()
                g = G(tdata1, A1, self.t0, self.Wpump)
                plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata1, g*max(adata1)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata1, w11*result1[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata1, w12*result1[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata1, w13*result1[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata1, w14*result1[:, 3], 'm-', label = 'w4*N4')
                plt.plot(tdata1, w15*result1[:, 4], 'c-', label = 'w5*N5')
                fit_result = w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2] + w14*result1[:, 3] + w15*result1[:, 4]
                plt.plot(tdata1, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4 + w5*N5')
                plt.scatter(tdata1, adata1)
                plt.legend()
                wsum = w11 + w12 + w13 + w14 + w15
                w11 = w11 / wsum
                w12 = w12 / wsum
                w13 = w13 / wsum
                w14 = w14 / wsum
                w15 = w15 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f fs, t4 = %.2f, t5 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f, w5 = %.2f' % tuple((A1, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata2, A2, self.t0, self.Wpump)
                plt.plot(tdata2, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata2, g*max(adata2)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata2, w21*result2[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata2, w22*result2[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata2, w23*result2[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata2, w24*result2[:, 3], 'm-', label = 'w4*N4')
                plt.plot(tdata2, w25*result2[:, 4], 'c-', label = 'w5*N5')
                fit_result = w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2] + w24*result2[:, 3] + w25*result2[:, 4]
                plt.plot(tdata2, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4 + w5*N5')
                plt.scatter(tdata2, adata2)
                plt.legend()
                wsum = w21 + w22 + w23 + w24 + w25
                w21 = w21 / wsum
                w22 = w22 / wsum
                w23 = w23 / wsum
                w24 = w24 / wsum
                w25 = w25 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, t5 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f, w5 = %.2f' % tuple((A2, t1, t2, t3, t4, t5, w21, w22, w23, w24, w25)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[1])
                plt.ylabel('ΔA (mOD)')
                plt.show()
                
                return A1, A2, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15, w21, w22, w23, w24, w25, result1, result2
            
            else:
                print('Currently only 1-5 states are supported for global fitting. 5 states will be fitted by default instead.')
               
###############################################################################            
                
        elif len(self.wavelengths) == 3:
            # global fitting of three wavelengths
            if self.number_of_states == 1: #sequential linear 3 wavelengths 1 state
                def G(t, A, t0, Wpump):
                    return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
             
                def model(y, t, A, t1):
                   N1 = y[0]
                   dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                   return [dN1_dt]
            
                def y1(t, A1, t1):
                    y0 = [0] # the states are initially unoccupied
                    global result1
                    result1 = integrate.odeint(model, y0, t, args = (A1, t1,))
                    return result1[:, 0]
        
                def y2(t, A2, t1):
                    y0 = [0]
                    global result2
                    result2 = integrate.odeint(model, y0, t, args = (A2, t1,))
                    return result2[:, 0]
                
                def y3(t, A3, t1):
                    y0 = [0]
                    global result3
                    result3 = integrate.odeint(model, y0, t, args = (A3, t1,))
                    return result3[:, 0]
        
                def comboFunc(comboData, A1, A2, A3, t1):
                    result1 = y1(tdata1, A1, t1)
                    result2 = y2(tdata2, A2, t1)
                    result3 = y3(tdata3, A3, t1)
                    sigma_squared = sum((a_combo - np.append(  np.append(result1, result2), result3)**2))
                    print('sigma^2 = ', sigma_squared)
                    return np.append(  np.append(result1, result2), result3)
             
                t_combo = []
                a_combo = []
                for i in dataset:
                    t_combo = np.append(t_combo, i[0])
                    a_combo = np.append(a_combo, i[1])
                    
                data1 = dataset[0]
                data2 = dataset[1]
                data3 = dataset[2]
                
                tdata1, adata1 = data1
                tdata2, adata2 = data2
                tdata3, adata3 = data3
                           
                fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, self.x0, bounds = self.bounds)
        
                A1, A2, A3, t1 = fittedParameters
                A1err, A2err, A3err, t1err = np.sqrt(np.diag(pcov))
                print('A1err = %.2f, A2err = %.2f, A3err = %.2f, t1err = %.2f' % tuple((A1err, A2err, A3err, t1err)))
        
                plt.figure()
                g = G(tdata1, A1, self.t0, self.Wpump)
                plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata1, g*max(adata1)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                fit_result = result1[:,0]
                plt.plot(tdata1, fit_result, 'k--', label = 'N1')
                plt.scatter(tdata1, adata1)
                plt.legend()
                plt.title('A = %.2f, t1 = %.2f fs' % tuple((A1, t1)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata2, A2, self.t0, self.Wpump)
                plt.plot(tdata2, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata2, g*max(adata2)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                fit_result = result2[:,0]
                plt.plot(tdata2, fit_result, 'k--', label = 'N1')
                plt.scatter(tdata2, adata2)
                plt.legend()
                plt.title('A = %.2f, t1 = %.2f fs' % tuple((A2, t1)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[1])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata3, A3, self.t0, self.Wpump)
                plt.plot(tdata3, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata3, g*max(adata3)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                fit_result = result3[:,0]
                plt.plot(tdata3, fit_result, 'k--', label = 'N1')
                plt.scatter(tdata3, adata3)
                plt.legend()
                plt.title('A = %.2f, t1 = %.2f fs' % tuple((A3, t1)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[2])
                plt.ylabel('ΔA (mOD)')
                plt.show()
                
                return A1, A2, A3, t1, result1, result2, result3
                  
            elif self.number_of_states == 2: #sequential linear 3 wavelengths 2 states
                def G(t, A, t0, Wpump):
                    return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
                
                def model(y, t, A, t1, t2):
                    N1 = y[0]
                    N2 = y[1]
                    dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                    dN2_dt = N1/t1 - N2/t2
                    return [dN1_dt, dN2_dt]
                
                def y1(t, A1, t1, t2, w11, w12):
                    y0 = [0, 0] # the states are initially unoccupied
                    global result1
                    result1 = integrate.odeint(model, y0, t, args = (A1, t1, t2,))
                    return w11*result1[:, 0] + w12*result1[:, 1]
        
                def y2(t, A2, t1, t2, w21, w22):
                    y0 = [0, 0]
                    global result2
                    result2 = integrate.odeint(model, y0, t, args = (A2, t1, t2,))
                    return w21*result2[:, 0] + w22*result2[:, 1]
                
                def y3(t, A3, t1, t2, w31, w32):
                    y0 = [0, 0]
                    global result3
                    result3 = integrate.odeint(model, y0, t, args = (A3, t1, t2,))
                    return w31*result3[:, 0] + w32*result2[:, 1]
        
                def comboFunc(comboData, A1, A2, A3, t1, t2, w11, w12, w21, w22, w31, w32):
                    result1 = y1(tdata1, A1, t1, t2, w11, w12)
                    result2 = y2(tdata2, A2, t1, t2, w21, w22)
                    result3 = y3(tdata3, A3, t1, t2, w31, w32)
                    sigma_squared = sum((a_combo - np.append(np.append(result1, result2), result3)) ** 2)
                    print('sigma^2 = ', sigma_squared)
                    return np.append(  np.append(result1, result2), result3)
                # some initial parameter values
            
                # curve fit the combined data to the combined function
                t_combo = []
                a_combo = []
                for i in dataset:
                    t_combo = np.append(t_combo, i[0])
                    a_combo = np.append(a_combo, i[1])
                    
                data1 = dataset[0]
                data2 = dataset[1]
                data3 = dataset[2]
                
                tdata1, adata1 = data1
                tdata2, adata2 = data2
                tdata3, adata3 = data3
                              
                fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, self.x0, bounds = self.bounds)
                
                A1, A2, A3, t1, t2, w11, w12, w21, w22, w31, w32 = fittedParameters
                A1err, A2err, A3err, t1err, t2err, w11err, w12err, w21err, w22err, w31err, w32err = np.sqrt(np.diag(pcov))
                print('A1err = %.2f, A2err = %.2f, A3err = %.2f, t1err = %.2f, t2err = %.2f, w11err = %.2f, w12err = %.2f, w21err = %.2f, w22err = %.2f, w31err = %.2f, w32err = %.2f' % tuple((A1err, A2err, A3err, t1err, t2err, w11err, w12err, w21err, w22err, w31err, w32err)))
        
                plt.figure()
                g = G(tdata1, A1, self.t0, self.Wpump)
                plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata1, g*max(adata1)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata1, w11*result1[:,0], 'r-', label = 'w1*N1')
                plt.plot(tdata1, w12*result1[:,1], 'b-', label = 'w2*N2')
                fit_result = w11*result1[:,0] + w12*result1[:,1]
                plt.plot(tdata1, fit_result, 'k--', label = 'w1*N1 + w2*N2')
                plt.scatter(tdata1, adata1)
                plt.legend()
                wsum = w11 + w12
                w11 = w11 / wsum
                w12 = w12 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, w1 = %.2f, w2 = %.2f' % tuple((A1, t1, t2, w11, w12)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata2, A2, self.t0, self.Wpump)
                plt.plot(tdata2, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata2, g*max(adata2)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata2, w21*result2[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata2, w22*result2[:, 1], 'b-', label = 'w2*N2')
                fit_result = w21*result2[:,0] + w22*result2[:,1]
                plt.plot(tdata2, fit_result, 'k--', label = 'w1*N1 + w2*N2')
                plt.scatter(tdata2, adata2)
                plt.legend()
                wsum = w21 + w22
                w21 = w21 / wsum
                w22 = w22 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, w1 = %.2f, w2 = %.2f' % tuple((A2, t1, t2, w21, w22)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[1])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata3, A3, self.t0, self.Wpump)
                plt.plot(tdata3, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata3, g*max(adata3)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata3, w31*result3[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata2, w32*result3[:, 1], 'b-', label = 'w2*N2')
                fit_result = w31*result3[:,0] + w32*result3[:,1]
                plt.plot(tdata3, fit_result, 'k--', label = 'w1*N1 + w2*N2')
                plt.scatter(tdata3, adata3)
                plt.legend()
                wsum = w31 + w32
                w31 = w31 / wsum
                w32 = w32 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, w1 = %.2f, w2 = %.2f' % tuple((A3, t1, t2, w31, w32)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[2])
                plt.ylabel('ΔA (mOD)')
                plt.show()
                
                return A1, A2, A3, t1, t2, w11, w12, w21, w22, w31, w32, result1, result2, result3
     
            elif self.number_of_states == 3: #sequential linear 3 wavelengths 3 states
                def G(t, A, t0, Wpump):
                    return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
                
                def model(y, t, A, t1, t2, t3):
                    N1 = y[0]
                    N2 = y[1]
                    N3 = y[2]
                    dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                    dN2_dt = N1/t1 - N2/t2
                    dN3_dt = N2/t2 - N3/t3
                    return [dN1_dt, dN2_dt, dN3_dt]
        
                # def y1(t, A1, t1, t2, t3, w11, w12):
                    # w13 = 1 - w11 - w12
                def y1(t, A1, t1, t2, t3, w11, w12, w13):
                    y0 = [0, 0, 0] # the states are initially unoccupied
                    global result1
                    result1 = integrate.odeint(model, y0, t, args = (A1, t1, t2, t3,))
                    return w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2]
        
                # def y2(t, A2, t1, t2, t3, w21, w22):
                    # w23 = 1 - w21 - w22
                def y2(t, A2, t1, t2, t3, w21, w22, w23):
                    y0 = [0, 0, 0]
                    global result2
                    result2 = integrate.odeint(model, y0, t, args = (A2, t1, t2, t3,))
                    return w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2]
                    
                # def y3(t, A3, t1, t2, t3, w31, w32):
                    # w33 = 1 - w31 - w32
                def y3(t, A3, t1, t2, t3, w31, w32, w33):
                    y0 = [0, 0, 0]
                    global result3
                    result3 = integrate.odeint(model, y0, t, args = (A3, t1, t2, t3,))
                    return w31*result3[:, 0] + w32*result3[:, 1] + w33*result3[:, 2]
        
                # def comboFunc(comboData, A1, A2, A3, t1, t2, t3, w11, w12, w21, w22, w31, w32):
                #     result1 = y1(tdata1, A1, t1, t2, t3, w11, w12)
                #     result2 = y2(tdata2, A2, t1, t2, t3, w21, w22)
                #     result3 = y3(tdata3, A3, t1, t2, t3, w31, w32)
                def comboFunc(comboData, A1, A2, A3, t1, t2, t3, w11, w12, w13, w21, w22, w23, w31, w32, w33):
                    result1 = y1(tdata1, A1, t1, t2, t3, w11, w12, w13)
                    result2 = y2(tdata2, A2, t1, t2, t3, w21, w22, w23)
                    result3 = y3(tdata3, A3, t1, t2, t3, w31, w32, w33)
                    sigma_squared = sum((a_combo - np.append(np.append(result1, result2), result3)) ** 2)
                    print('sigma^2 = ', sigma_squared)
                    return np.append(  np.append(result1, result2), result3)
                # some initial parameter values
            
                # curve fit the combined data to the combined function
                t_combo = []
                a_combo = []
                for i in dataset:
                    t_combo = np.append(t_combo, i[0])
                    a_combo = np.append(a_combo, i[1])
                    
                data1 = dataset[0]
                data2 = dataset[1]
                data3 = dataset[2]
                
                tdata1, adata1 = data1
                tdata2, adata2 = data2
                tdata3, adata3 = data3
                
                if self.tshift == True:
                    t1_idx, tzero = find_nearest(tdata1, self.t0)
                    shift_height = adata1[t1_idx] / np.max(abs(adata1))
                    
                    tdata1 = tdata1 - tzero
                    
                    t2_idx, t2 = find_nearest(adata2, shift_height * np.max(abs(adata2)))
                    tdata2 = tdata2 - t2
                    
                    t3_idx, t3 = find_nearest(adata3, shift_height * np.max(abs(adata3)))
                    tdata3 = tdata3 - t3
                else:
                    pass
                
                fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, self.x0, bounds = self.bounds)
                
                # A1, A2, A3, t1, t2, t3, w11, w12, w21, w22, w31, w32 = fittedParameters
                # A1err, A2err, A3err, t1err, t2err, t3err, w11err, w12err, w21err, w22err, w31err, w32err = np.sqrt(np.diag(pcov))
                
                # w13 = 1 - w11 - w12
                # w23 = 1 - w21 - w22
                # w33 = 1 - w31 - w32
                # print('A1err = %.2f, A2err = %.2f, A3err = %.2f, t1err = %.2f, t2err = %.2f, t3err = %.2f, w11err = %.2f, w12err = %.2f, w21err = %.2f, w22err = %.2f, w31err = %.2f, w32err = %.2f' % tuple((A1err, A2err, A3err, t1err, t2err, t3err, w11err, w12err, w21err, w22err, w31err, w32err)))
                A1, A2, A3, t1, t2, t3, w11, w12, w13, w21, w22, w23, w31, w32, w33 = fittedParameters
                A1err, A2err, A3err, t1err, t2err, t3err, w11err, w12err, w13err, w21err, w22err, w23err, w31err, w32err, w33err = np.sqrt(np.diag(pcov))
                
                # w13 = 1 - w11 - w12
                # w23 = 1 - w21 - w22
                # w33 = 1 - w31 - w32
                # print('A1err = %.2f, A2err = %.2f, A3err = %.2f, t1err = %.2f, t2err = %.2f, t3err = %.2f, w11err = %.2f, w12err = %.2f, w21err = %.2f, w22err = %.2f, w31err = %.2f, w32err = %.2f' % tuple((A1err, A2err, A3err, t1err, t2err, t3err, w11err, w12err, w21err, w22err, w31err, w32err)))
                print('A1err = %.2f, A2err = %.2f, A3err = %.2f, t1err = %.2f, t2err = %.2f, t3err = %.2f, w11err = %.2f, w12err = %.2f, w13err = %.2f, w21err = %.2f, w22err = %.2f, w23err = %.2f, w31err = %.2f, w32err = %.2f, w33err = %.2f' % tuple((A1err, A2err, A3err, t1err, t2err, t3err, w11err, w12err, w13err, w21err, w22err, w23err, w31err, w32err, w33err)))
        
                plt.figure()
                g = G(tdata1, A1, self.t0, self.Wpump)
                plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata1, g*max(adata1)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata1, w11*result1[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata1, w12*result1[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata1, w13*result1[:, 2], 'g-', label = 'w3*N3')
                fit_result = w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2]
                plt.plot(tdata1, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3')
                plt.scatter(tdata1, adata1)
                plt.legend()
                # wsum = w11 + w12 + w13
                # w11 = w11 / wsum
                # w12 = w12 / wsum
                # w13 = w13 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f fs, w1 = %.2f, w2 = %.2f, w3 = %.2f' % tuple((A1, t1, t2, t3, w11, w12, w13)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata2, A2, self.t0, self.Wpump)
                plt.plot(tdata2, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata2, g*max(adata2)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata2, w21*result2[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata2, w22*result2[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata2, w23*result2[:, 2], 'g-', label = 'w3*N3')
                fit_result = w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2]
                plt.plot(tdata2, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3')
                plt.scatter(tdata2, adata2)
                plt.legend()
                # wsum = w21 + w22 + w23
                # w21 = w21 / wsum
                # w22 = w22 / wsum
                # w23 = w23 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f' % tuple((A2, t1, t2, t3, w21, w22, w23)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[1])
                plt.ylabel('ΔA (mOD)')

                plt.figure()
                g = G(tdata3, A3, self.t0, self.Wpump)
                plt.plot(tdata3, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata3, g*max(adata3)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata3, w31*result3[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata3, w32*result3[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata3, w33*result3[:, 2], 'g-', label = 'w3*N3')
                fit_result = w31*result3[:, 0] + w32*result3[:, 1] + w33*result3[:, 2]
                plt.plot(tdata3, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3')
                plt.scatter(tdata3, adata3)
                plt.legend()
                # wsum = w31 + w32 + w33
                # w31 = w31 / wsum
                # w32 = w32 / wsum
                # w33 = w33 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f' % tuple((A3, t1, t2, t3, w31, w32, w33)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[2])
                plt.ylabel('ΔA (mOD)')
                plt.show()
                
                return A1, A2, A3, t1, t2, t3, w11, w12, w13, w21, w22, w23, w31, w32, w33, result1, result2, result3
     
            elif self.number_of_states == 4: #sequential linear 3 wavelengths 4 states
                def G(t, A, t0, Wpump):
                    return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
                
                def model(y, t, A, t1, t2, t3, t4):
                    N1 = y[0]
                    N2 = y[1]
                    N3 = y[2]
                    N4 = y[3]
                    dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                    dN2_dt = N1/t1 - N2/t2
                    dN3_dt = N2/t2 - N3/t3
                    dN4_dt = N3/t3 - N4/t4
                    return [dN1_dt, dN2_dt, dN3_dt, dN4_dt]
        
                def y1(t, A1, t1, t2, t3, t4, w11, w12, w13, w14):
                    y0 = [0, 0, 0, 0] # the states are initially unoccupied
                    global result1
                    result1 = integrate.odeint(model, y0, t, args = (A1, t1, t2, t3, t4,))
                    return w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2] + w14*result1[:, 3]
        
                def y2(t, A2, t1, t2, t3, t4, w21, w22, w23, w24):
                    y0 = [0, 0, 0, 0]
                    global result2
                    result2 = integrate.odeint(model, y0, t, args = (A2, t1, t2, t3, t4,))
                    return w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2] + w24*result2[:, 3]
                    
                def y3(t, A3, t1, t2, t3, t4, w31, w32, w33, w34):
                    y0 = [0, 0, 0, 0]
                    global result3
                    result3 = integrate.odeint(model, y0, t, args = (A3, t1, t2, t3, t4,))
                    return w31*result3[:, 0] + w32*result3[:, 1] + w33*result3[:, 2] + w34*result3[:, 3]
                    
                def comboFunc(comboData, A1, A2, A3, t1, t2, t3, t4, w11, w12, w13, w14, w21, w22, w23, w24, w31, w32, w33, w34):
                    result1 = y1(tdata1, A1, t1, t2, t3, t4, w11, w12, w13, w14)
                    result2 = y2(tdata2, A2, t1, t2, t3, t4, w21, w22, w23, w24)
                    result3 = y3(tdata3, A3, t1, t2, t3, t4, w31, w32, w33, w34)
                    return np.append(  np.append(result1, result2), result3)
                # some initial parameter values
            
                # curve fit the combined data to the combined function
                t_combo = []
                a_combo = []
                for i in dataset:
                    t_combo = np.append(t_combo, i[0])
                    a_combo = np.append(a_combo, i[1])
                    
                data1 = dataset[0]
                data2 = dataset[1]
                data3 = dataset[2]
                
                tdata1, adata1 = data1
                tdata2, adata2 = data2
                tdata3, adata3 = data3
                               
                fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, self.x0, bounds = self.bounds)
                
                A1, A2, A3, t1, t2, t3, t4, w11, w12, w13, w14, w21, w22, w23, w24, w31, w32, w33, w34 = fittedParameters
                A1err, A2err, A3err, t1err, t2err, t3err, t4err, w11err, w12err, w13err, w14err, w21err, w22err, w23err, w24err, w31err, w32err, w33err, w34err = np.sqrt(np.diag(pcov))
                print('A1err = %.2f, A2err = %.2f, A3err = %.2f, t1err = %.2f, t2err = %.2f, t3err = %.2f, t4err = %.2f, w11err = %.2f, w12err = %.2f, w13err = %.2f, w14err = %.2f, w21err = %.2f, w22err = %.2f, w23err = %.2f, w24err = %.2f, w31err = %.2f, w32err = %.2f, w33err = %.2f, w34err = %.2f' % tuple((A1err, A2err, A3err, t1err, t2err, t3err, t4err, w11err, w12err, w13err, w14err, w21err, w22err, w23err, w24err, w31err, w32err, w33err, w34err)))
        
                plt.figure()
                g = G(tdata1, A1, self.t0, self.Wpump)
                plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata1, g*max(adata1)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata1, w11*result1[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata1, w12*result1[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata1, w13*result1[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata1, w14*result1[:, 3], 'm-', label = 'w4*N4')
                fit_result = w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2] + w14*result1[:, 3]
                plt.plot(tdata1, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4')
                plt.scatter(tdata1, adata1)
                plt.legend()
                wsum = w11 + w12 + w13 + w14
                w11 = w11 / wsum
                w12 = w12 / wsum
                w13 = w13 / wsum
                w14 = w14 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f fs, t4 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f' % tuple((A1, t1, t2, t3, t4, w11, w12, w13, w14)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata2, A2, self.t0, self.Wpump)
                plt.plot(tdata2, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata2, g*max(adata2)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata2, w21*result2[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata2, w22*result2[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata2, w23*result2[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata2, w24*result2[:, 3], 'm-', label = 'w4*N4')
                fit_result = w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2] + w24*result2[:, 3]
                plt.plot(tdata2, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4')
                plt.scatter(tdata2, adata2)
                plt.legend()
                wsum = w21 + w22 + w23 + w24
                w21 = w21 / wsum
                w22 = w22 / wsum
                w23 = w23 / wsum
                w24 = w24 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f' % tuple((A2, t1, t2, t3, t4, w21, w22, w23, w24)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[1])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata3, A3, self.t0, self.Wpump)
                plt.plot(tdata3, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata3, g*max(adata3)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata3, w31*result3[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata3, w32*result3[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata3, w33*result3[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata3, w34*result3[:, 3], 'm-', label = 'w4*N4')
                fit_result = w31*result3[:, 0] + w32*result3[:, 1] + w33*result3[:, 2] + w34*result3[:, 3]
                plt.plot(tdata3, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4')
                plt.scatter(tdata3, adata3)
                plt.legend()
                wsum = w31 + w32 + w33 + w34
                w31 = w31 / wsum
                w32 = w32 / wsum
                w33 = w33 / wsum
                w34 = w34 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f' % tuple((A3, t1, t2, t3, t4, w31, w32, w33, w34)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[2])
                plt.ylabel('ΔA (mOD)')
                plt.show()
                
                return A1, A2, A3, t1, t2, t3, t4, w11, w12, w13, w14, w21, w22, w23, w24, w31, w32, w33, w34, result1, result2, result3
    
            elif self.number_of_states == 5: #sequential linear 3 wavelengths 5 states
                def G(t, A, t0, Wpump):
                    return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
                
                def model(y, t, A, t1, t2, t3, t4, t5):
                    N1 = y[0]
                    N2 = y[1]
                    N3 = y[2]
                    N4 = y[3]
                    N5 = y[4]
                    dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                    dN2_dt = N1/t1 - N2/t2
                    dN3_dt = N2/t2 - N3/t3
                    dN4_dt = N3/t3 - N4/t4
                    dN5_dt = N4/t4 - N5/t5
                    return [dN1_dt, dN2_dt, dN3_dt, dN4_dt, dN5_dt]
        
                def y1(t, A1, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15):
                    y0 = [0, 0, 0, 0, 0] # the states are initially unoccupied
                    global result1
                    result1 = integrate.odeint(model, y0, t, args = (A1, t1, t2, t3, t4, t5,))
                    return w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2] + w14*result1[:, 3] + w15*result1[:, 4]
        
                def y2(t, A2, t1, t2, t3, t4, t5, w21, w22, w23, w24, w25):
                    y0 = [0, 0, 0, 0, 0]
                    global result2
                    result2 = integrate.odeint(model, y0, t, args = (A2, t1, t2, t3, t4, t5,))
                    return w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2] + w24*result2[:, 3] + w25*result2[:, 4]
                    
                def y3(t, A3, t1, t2, t3, t4, t5, w31, w32, w33, w34, w35):
                    y0 = [0, 0, 0, 0, 0]
                    global result3
                    result3 = integrate.odeint(model, y0, t, args = (A3, t1, t2, t3, t4, t5,))
                    return w31*result3[:, 0] + w32*result3[:, 1] + w33*result3[:, 2] + w34*result3[:, 3] + w35*result3[:, 4]
        
                def comboFunc(comboData, A1, A2, A3, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15, w21, w22, w23, w24, w25, w31, w32, w33, w34, w35):
                    result1 = y1(tdata1, A1, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15)
                    result2 = y2(tdata2, A2, t1, t2, t3, t4, t5, w21, w22, w23, w24, w25)
                    result3 = y3(tdata3, A3, t1, t2, t3, t4, t5, w31, w32, w33, w34, w35)
                    return np.append(  np.append(result1, result2), result3)
                # some initial parameter values
            
                # curve fit the combined data to the combined function
                t_combo = []
                a_combo = []
                for i in dataset:
                    t_combo = np.append(t_combo, i[0])
                    a_combo = np.append(a_combo, i[1])
                    
                data1 = dataset[0]
                data2 = dataset[1]
                data3 = dataset[2]
                
                tdata1, adata1 = data1
                tdata2, adata2 = data2
                tdata3, adata3 = data3
                              
                fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, self.x0, bounds = self.bounds)
                
                A1, A2, A3, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15, w21, w22, w23, w24, w25, w31, w32, w33, w34, w35 = fittedParameters
                A1err, A2err, A3err, t1err, t2err, t3err, t4err, t5err, w11err, w12err, w13err, w14err, w15err, w21err, w22err, w23err, w24err, w25err, w31err, w32err, w33err, w34err, w35err = np.sqrt(np.diag(pcov))
                print('A1err = %.2f, A2err = %.2f, A3err = %.2f, t1err = %.2f, t2err = %.2f, t3err = %.2f, t4err = %.2f, t5err = %.2f, w11err = %.2f, w12err = %.2f, w13err = %.2f, w14err = %.2f, w15err = %.2f, w21err = %.2f, w22err = %.2f, w23err = %.2f, w24err = %.2f w25err = %.2f, w31, w32err = %.2f, w33err = %.2f, w34err = %.2f, w35err = %.2f' % tuple((A1err, A2err, A3err, t1err, t2err, t3err, t4err, t5err, w11err, w12err, w13err, w14err, w15err, w21err, w22err, w23err, w24err, w25err, w31err, w32err, w33err, w34err, w35err)))
        
                plt.figure()
                g = G(tdata1, A1, self.t0, self.Wpump)
                plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata1, g*max(adata1)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata1, w11*result1[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata1, w12*result1[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata1, w13*result1[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata1, w14*result1[:, 3], 'm-', label = 'w4*N4')
                plt.plot(tdata1, w15*result1[:, 4], 'c-', label = 'w5*N5')
                fit_result = w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2] + w14*result1[:, 3] + w15*result1[:, 4]
                plt.plot(tdata1, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4 + w5*N5')
                plt.scatter(tdata1, adata1)
                plt.legend()
                wsum = w11 + w12 + w13 + w14 + w15
                w11 = w11 / wsum
                w12 = w12 / wsum
                w13 = w13 / wsum
                w14 = w14 / wsum
                w15 = w15 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f fs, t4 = %.2f, t5 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f, w5 = %.2f' % tuple((A1, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata2, A2, self.t0, self.Wpump)
                plt.plot(tdata2, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata2, g*max(adata2)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata2, w21*result2[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata2, w22*result2[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata2, w23*result2[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata2, w24*result2[:, 3], 'm-', label = 'w4*N4')
                plt.plot(tdata2, w25*result2[:, 4], 'c-', label = 'w5*N5')
                fit_result = w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2] + w24*result2[:, 3] + w25*result2[:, 4]
                plt.plot(tdata2, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4 + w5*N5')
                plt.scatter(tdata2, adata2)
                plt.legend()
                wsum = w21 + w22 + w23 + w24 + w25
                w21 = w21 / wsum
                w22 = w22 / wsum
                w23 = w23 / wsum
                w24 = w24 / wsum
                w25 = w25 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, t5 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f, w5 = %.2f' % tuple((A2, t1, t2, t3, t4, t5, w21, w22, w23, w24, w25)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[1])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata3, A3, self.t0, self.Wpump)
                plt.plot(tdata3, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata3, g*max(adata3)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata3, w31*result3[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata3, w32*result3[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata3, w33*result3[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata3, w34*result3[:, 3], 'm-', label = 'w4*N4')
                plt.plot(tdata3, w35*result3[:, 4], 'c-', label = 'w5*N5')
                fit_result = w31*result3[:, 0] + w32*result3[:, 1] + w33*result3[:, 2] + w34*result3[:, 3] + w35*result3[:, 4]
                plt.plot(tdata3, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4 + w5*N5')
                plt.scatter(tdata3, adata3)
                plt.legend()
                wsum = w31, w32, w33, w34, w35
                w31 = w31 / wsum
                w32 = w32 / wsum
                w33 = w33 / wsum
                w34 = w34 / wsum
                w35 = w35 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, t5 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f, w5 = %.2f' % tuple((A3, t1, t2, t3, t4, t5, w31, w32, w33, w34, w35)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[2])
                plt.ylabel('ΔA (mOD)')
                plt.show()
                
                return A1, A2, A3, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15, w21, w22, w23, w24, w25, w31, w32, w33, w34, w35, result1, result2, result3
            
            else:
                print('Currently only 1-5 states are supported for global fitting. 5 states will be fitted by default instead.')
               
###############################################################################                
                
        elif len(self.wavelengths) == 4:
            # global fitting of four wavelengths
            if self.number_of_states == 1:
                def G(t, A, t0, Wpump):
                    return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
             
                def model(y, t, A, t1):
                   N1 = y[0]
                   dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                   return [dN1_dt]
            
                def y1(t, A1, t1):
                    y0 = [0] # the states are initially unoccupied
                    global result1
                    result1 = integrate.odeint(model, y0, t, args = (A1, t1,))
                    return result1[:, 0]
        
                def y2(t, A2, t1):
                    y0 = [0]
                    global result2
                    result2 = integrate.odeint(model, y0, t, args = (A2, t1,))
                    return result2[:, 0]
                
                def y3(t, A3, t1):
                    y0 = [0]
                    global result3
                    result3 = integrate.odeint(model, y0, t, args = (A3, t1,))
                    return result3[:, 0]
                    
                def y4(t, A4, t1):
                    y0 = [0]
                    global result4
                    result4 = integrate.odeint(model, y0, t, args = (A4, t1,))  
                    return result4[:, 0]
                
                def comboFunc(comboData, A1, A2, A3, A4, t1):
                    result1 = y1(tdata1, A1, t1)
                    result2 = y2(tdata2, A2, t1)
                    result3 = y3(tdata3, A3, t1)
                    result4 = y4(tdata4, A4, t1)
                    return np.append(  np.append(  np.append(result1, result2), result3), result4)
             
                t_combo = []
                a_combo = []
                for i in dataset:
                    t_combo = np.append(t_combo, i[0])
                    a_combo = np.append(a_combo, i[1])
                    
                data1 = dataset[0]
                data2 = dataset[1]
                data3 = dataset[2]
                data4 = dataset[3]
                
                tdata1, adata1 = data1
                tdata2, adata2 = data2
                tdata3, adata3 = data3
                tdata4, adata4 = data4
    
                
                fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, self.x0, bounds = self.bounds)
        
                A1, A2, A3, A4, t1 = fittedParameters
                A1err, A2err, A3err, A4err, t1err = np.sqrt(np.diag(pcov))
                print('A1err = %.2f, A2err = %.2f, A3err = %.2f, A4err = %.2f, t1err = %.2f' % tuple((A1err, A2err, A3err, A4err, t1err)))
        
                plt.figure()
                g = G(tdata1, A1, self.t0, self.Wpump)
                plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata1, g*max(adata1)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                fit_result = result1[:,0]
                plt.plot(tdata1, fit_result, 'k--', label = 'N1')
                plt.scatter(tdata1, adata1)
                plt.legend()
                plt.title('A = %.2f, t1 = %.2f fs' % tuple((A1, t1)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata2, A2, self.t0, self.Wpump)
                plt.plot(tdata2, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata2, g*max(adata2)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                fit_result = result2[:,0]
                plt.plot(tdata2, fit_result, 'k--', label = 'N1')
                plt.scatter(tdata2, adata2)
                plt.legend()
                plt.title('A = %.2f, t1 = %.2f fs' % tuple((A2, t1)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[1])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata3, A3, self.t0, self.Wpump)
                plt.plot(tdata3, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata3, g*max(adata3)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                fit_result = result3[:,0]
                plt.plot(tdata3, fit_result, 'k--', label = 'N1')
                plt.scatter(tdata3, adata3)
                plt.legend()
                plt.title('A = %.2f, t1 = %.2f fs' % tuple((A3, t1)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[2])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata4, A4, self.t0, self.Wpump)
                plt.plot(tdata4, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata4, g*max(adata4)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                fit_result = result4[:,0]
                plt.plot(tdata4, fit_result, 'k--', label = 'N1')
                plt.scatter(tdata4, adata4)
                plt.legend()
                plt.title('A = %.2f, t1 = %.2f fs' % tuple((A4, t1)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[3])
                plt.ylabel('ΔA (mOD)')
                plt.show()
                
                return A1, A2, A3, A4, t1, result1, result2, result3, result4
                  
            elif self.number_of_states == 2:
                def G(t, A, t0, Wpump):
                    return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
                
                def model(y, t, A, t1, t2):
                    N1 = y[0]
                    N2 = y[1]
                    dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                    dN2_dt = N1/t1 - N2/t2
                    return [dN1_dt, dN2_dt]
                
                def y1(t, A1, t1, t2, w11, w12):
                    y0 = [0, 0] # the states are initially unoccupied
                    global result1
                    result1 = integrate.odeint(model, y0, t, args = (A1, t1, t2,))
                    return w11*result1[:, 0] + w12*result1[:, 1]
        
                def y2(t, A2, t1, t2, w21, w22):
                    y0 = [0, 0]
                    global result2
                    result2 = integrate.odeint(model, y0, t, args = (A2, t1, t2,))
                    return w21*result2[:, 0] + w22*result2[:, 1]
                
                def y3(t, A3, t1, t2, w31, w32):
                    y0 = [0, 0]
                    global result3
                    result3 = integrate.odeint(model, y0, t, args = (A3, t1, t2,))
                    return w31*result3[:, 0] + 2*result2[:, 1]
                    
                def y4(t, A4, t1, t2, w41, w42):
                    y0 = [0, 0]
                    global result4
                    result4 = integrate.odeint(model, y0, t, args = (A4, t1, t2,))
                    return w41*result3[:, 0] + w42*result4[:, 1]
        
                def comboFunc(comboData, A1, A2, A3, A4, t1, t2, w11, w12, w21, w22, w31, w32, w41, w42):
                    result1 = y1(tdata1, A1, t1, t2, w11, w12)
                    result2 = y2(tdata2, A2, t1, t2, w21, w22)
                    result3 = y3(tdata3, A3, t1, t2, w31, w32)
                    result4 = y4(tdata4, A4, t1, t2, w41, w42)
                    return np.append(  np.append( np.append(result1, result2), result3), result4)
                # some initial parameter values
            
                # curve fit the combined data to the combined function
                t_combo = []
                a_combo = []
                for i in dataset:
                    t_combo = np.append(t_combo, i[0])
                    a_combo = np.append(a_combo, i[1])
                    
                data1 = dataset[0]
                data2 = dataset[1]
                data3 = dataset[2]
                data4 = dataset[3]
                
                tdata1, adata1 = data1
                tdata2, adata2 = data2
                tdata3, adata3 = data3
                tdata4, adata4 = data4
                
                fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, self.x0, bounds = self.bounds)
                
                A1, A2, A3, A4, t1, t2, w11, w12, w21, w22, w31, w32, w41, w42 = fittedParameters
                A1err, A2err, A3err, A4err, t1err, t2err, w11err, w12err, w21err, w22err, w31err, w32err, w41err, w42err = np.sqrt(np.diag(pcov))
                print('A1err = %.2f, A2err = %.2f, A3err = %.2f, A4err = %.2f, t1err = %.2f, t2err = %.2f, w11err = %.2f, w12err = %.2f, w21err = %.2f, w22err = %.2f, w31err = %.2f, w32err = %.2f, w41err = %.2f, w42 = %.2f'
                      % tuple((A1err, A2err, A3err, A4err, t1err, t2err, w11err, w12err, w21err, w22err, w31err, w32err, w41err, w42err)))
        
                plt.figure()
                g = G(tdata1, A1, self.t0, self.Wpump)
                plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata1, g*max(adata1)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata1, w11*result1[:,0], 'r-', label = 'w1*N1')
                plt.plot(tdata1, w12*result1[:,1], 'b-', label = 'w2*N2')
                fit_result = w11*result1[:,0] + w12*result1[:,1]
                plt.plot(tdata1, fit_result, 'k--', label = 'w1*N1 + w2*N2')
                plt.scatter(tdata1, adata1)
                plt.legend()
                wsum = w11 + w12
                w11 = w11 / wsum
                w12 = w12 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, w1 = %.2f, w2 = %.2f' % tuple((A1, t1, t2, w11, w12)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata2, A2, self.t0, self.Wpump)
                plt.plot(tdata2, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata2, g*max(adata2)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata2, w21*result2[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata2, w22*result2[:, 1], 'b-', label = 'w2*N2')
                fit_result = w21*result2[:,0] + w22*result2[:,1]
                plt.plot(tdata2, fit_result, 'k--', label = 'w1*N1 + w2*N2')
                plt.scatter(tdata2, adata2)
                plt.legend()
                wsum = w21 + w22
                w21 = w21 / wsum
                w22 = w22 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, w1 = %.2f, w2 = %.2f' % tuple((A2, t1, t2, w21, w22)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[1])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata3, A3, self.t0, self.Wpump)
                plt.plot(tdata3, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata3, g*max(adata3)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata3, w31*result3[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata3, w32*result3[:, 1], 'b-', label = 'w2*N2')
                fit_result = w31*result3[:,0] + w32*result3[:,1]
                plt.plot(tdata3, fit_result, 'k--', label = 'w1*N1 + w2*N2')
                plt.scatter(tdata3, adata3)
                plt.legend()
                wsum = w31 + w32
                w31 = w31 / wsum
                w32 = w32 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, w1 = %.2f, w2 = %.2f' % tuple((A3, t1, t2, w31, w32)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[2])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata4, A4, self.t0, self.Wpump)
                plt.plot(tdata4, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata4, g*max(adata4)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata4, w41*result4[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata4, w42*result4[:, 1], 'b-', label = 'w2*N2')
                fit_result = w41*result4[:,0] + w42*result4[:,1]
                plt.plot(tdata4, fit_result, 'k--', label = 'w1*N1 + w2*N2')
                plt.scatter(tdata4, adata4)
                plt.legend()
                wsum = w41 + w42
                w41 = w41 / wsum
                w42 = w42 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, w1 = %.2f, w2 = %.2f' % tuple((A4, t1, t2, w41, w42)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[3])
                plt.ylabel('ΔA (mOD)')
                plt.show()
                
                return A1, A2, A3, t1, t2, w11, w12, w21, w22, w31, w32, result1, result2, result3
     
            elif self.number_of_states == 3:
                def G(t, A, t0, Wpump):
                    return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
                
                def model(y, t, A, t1, t2, t3):
                    N1 = y[0]
                    N2 = y[1]
                    N3 = y[2]
                    dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                    dN2_dt = N1/t1 - N2/t2
                    dN3_dt = N2/t2 - N3/t3
                    return [dN1_dt, dN2_dt, dN3_dt]
        
                def y1(t, A1, t1, t2, t3, w11, w12, w13):
                    y0 = [0, 0, 0] # the states are initially unoccupied
                    global result1
                    result1 = integrate.odeint(model, y0, t, args = (A1, t1, t2, t3,))
                    return w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2]
        
                def y2(t, A2, t1, t2, t3, w21, w22, w23):
                    y0 = [0, 0, 0]
                    global result2
                    result2 = integrate.odeint(model, y0, t, args = (A2, t1, t2, t3,))
                    return w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2]
                    
                def y3(t, A3, t1, t2, t3, w31, w32, w33):
                    y0 = [0, 0, 0]
                    global result3
                    result3 = integrate.odeint(model, y0, t, args = (A3, t1, t2, t3,))
                    return w31*result3[:, 0] + w32*result3[:, 1] + w33*result3[:, 2]
                    
                def y4(t, A4, t1, t2, t3, w41, w42, w43):
                    y0 = [0, 0, 0]
                    global result4
                    result4 = integrate.odeint(model, y0, t, args = (A4, t1, t2, t3,))
                    return w41*result4[:, 0] + w42*result4[:, 1] + w43*result4[:, 2]
        
                def comboFunc(comboData, A1, A2, A3, A4, t1, t2, t3, w11, w12, w13, w21, w22, w23, w31, w32, w33, w41, w42, w43):
                    result1 = y1(tdata1, A1, t1, t2, t3, w11, w12, w13)
                    result2 = y2(tdata2, A2, t1, t2, t3, w21, w22, w23)
                    result3 = y3(tdata3, A3, t1, t2, t3, w31, w32, w33)
                    result4 = y4(tdata4, A4, t1, t2, t3, w41, w42, w43)
                    return np.append(  np.append(np.append(result1, result2), result3), result4)
                # some initial parameter values
            
                # curve fit the combined data to the combined function
                t_combo = []
                a_combo = []
                for i in dataset:
                    t_combo = np.append(t_combo, i[0])
                    a_combo = np.append(a_combo, i[1])
                    
                data1 = dataset[0]
                data2 = dataset[1]
                data3 = dataset[2]
                data4 = dataset[3]
                
                tdata1, adata1 = data1
                tdata2, adata2 = data2
                tdata3, adata3 = data3
                tdata4, adata4 = data4
                
                fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, self.x0, bounds = self.bounds)
                
                A1, A2, A3, A4, t1, t2, t3, w11, w12, w13, w21, w22, w23, w31, w32, w33, w41, w42, w43 = fittedParameters
                A1err, A2err, A3err, A4err, t1err, t2err, t3err, w11err, w12err, w13err, w21err, w22err, w23err, w31err, w32err, w33err, w41err, w42err, w43err = np.sqrt(np.diag(pcov))
                print('A1err = %.2f, A2err = %.2f, A3err = %.2f, A4err = %.2f, t1err = %.2f, t2err = %.2f, t3err = %.2f, w11err = %.2f, w12err = %.2f, w13err = %.2f, w21err = %.2f, w22err = %.2f, w23err = %.2f, w31err = %.2f, w32err = %.2f, w33err = %.2f, w41err = %.2f, w42err = %.2f, w43 = %.2f'
                      % tuple((A1err, A2err, A3err, A4err, t1err, t2err, t3err, w11err, w12err, w13err, w21err, w22err, w23err, w31err, w32err, w33err, w41err, w42err, w43err)))
        
                plt.figure()
                g = G(tdata1, A1, self.t0, self.Wpump)
                plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata1, g*max(adata1)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata1, w11*result1[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata1, w12*result1[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata1, w13*result1[:, 2], 'g-', label = 'w3*N3')
                fit_result = w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2]
                plt.plot(tdata1, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3')
                plt.scatter(tdata1, adata1)
                plt.legend()
                wsum = w11 + w12 + w13
                w11 = w11 / wsum
                w12 = w12 / wsum
                w13 = w13 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f fs, w1 = %.2f, w2 = %.2f, w3 = %.2f' % tuple((A1, t1, t2, t3, w11, w12, w13)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata2, A2, self.t0, self.Wpump)
                plt.plot(tdata2, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata2, g*max(adata2)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata2, w21*result2[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata2, w22*result2[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata2, w23*result2[:, 2], 'g-', label = 'w3*N3')
                fit_result = w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2]
                plt.plot(tdata2, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3')
                plt.scatter(tdata2, adata2)
                plt.legend()
                wsum = w21 + w22 + w23
                w21 = w21 / wsum
                w22 = w22 / wsum
                w23 = w23 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f' % tuple((A2, t1, t2, t3, w21, w22, w23)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[1])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata3, A3, self.t0, self.Wpump)
                plt.plot(tdata3, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata3, g*max(adata3)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata3, w31*result3[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata3, w32*result3[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata3, w33*result3[:, 2], 'g-', label = 'w3*N3')
                fit_result = w31*result3[:, 0] + w32*result3[:, 1] + w33*result3[:, 2]
                plt.plot(tdata3, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3')
                plt.scatter(tdata3, adata3)
                plt.legend()
                wsum = w31 + w32 + w33
                w31 = w31 / wsum
                w32 = w32 / wsum
                w33 = w33 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f' % tuple((A3, t1, t2, t3, w31, w32, w33)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[2])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata4, A4, self.t0, self.Wpump)
                plt.plot(tdata4, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata4, g*max(adata4)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata4, w41*result4[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata4, w42*result4[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata4, w43*result4[:, 2], 'g-', label = 'w3*N3')
                fit_result = w41*result4[:, 0] + w42*result4[:, 1] + w43*result4[:, 2]
                plt.plot(tdata4, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3')
                plt.scatter(tdata4, adata4)
                plt.legend()
                wsum = w41 + w42 + w43
                w41 = w41 / wsum
                w42 = w42 / wsum
                w43 = w43 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f' % tuple((A4, t1, t2, t3, w41, w42, w43)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[3])
                plt.ylabel('ΔA (mOD)')
                plt.show()
                
                return A1, A2, A3, A4, t1, t2, t3, w11, w12, w13, w21, w22, w23, w31, w32, w33, w41, w42, w43, result1, result2, result3, result4
     
            elif self.number_of_states == 4:
                def G(t, A, t0, Wpump):
                    return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
                
                def model(y, t, A, t1, t2, t3, t4):
                    N1 = y[0]
                    N2 = y[1]
                    N3 = y[2]
                    N4 = y[3]
                    dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                    dN2_dt = N1/t1 - N2/t2
                    dN3_dt = N2/t2 - N3/t3
                    dN4_dt = N3/t3 - N4/t4
                    return [dN1_dt, dN2_dt, dN3_dt, dN4_dt]
        
                def y1(t, A1, t1, t2, t3, t4, w11, w12, w13, w14):
                    y0 = [0, 0, 0, 0] # the states are initially unoccupied
                    global result1
                    result1 = integrate.odeint(model, y0, t, args = (A1, t1, t2, t3, t4,))
                    return w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2] + w14*result1[:, 3]
        
                def y2(t, A2, t1, t2, t3, t4, w21, w22, w23, w24):
                    y0 = [0, 0, 0, 0]
                    global result2
                    result2 = integrate.odeint(model, y0, t, args = (A2, t1, t2, t3, t4,))
                    return w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2] + w24*result2[:, 3]
                    
                def y3(t, A3, t1, t2, t3, t4, w31, w32, w33, w34):
                    y0 = [0, 0, 0, 0]
                    global result3
                    result3 = integrate.odeint(model, y0, t, args = (A3, t1, t2, t3, t4,))
                    return w31*result3[:, 0] + w32*result3[:, 1] + w33*result3[:, 2] + w34*result3[:, 3]
                    
                def y4(t, A4, t1, t2, t3, t4, w41, w42, w43, w44):
                    y0 = [0, 0, 0, 0]
                    global result4
                    result4 = integrate.odeint(model, y0, t, args = (A4, t1, t2, t3, t4,))
                    return w41*result4[:, 0] + w42*result4[:, 1] + w43*result4[:, 2] + w44*result4[:, 3]
                    
                def comboFunc(comboData, A1, A2, A3, A4, t1, t2, t3, t4, w11, w12, w13, w14, w21, w22, w23, w24, w31, w32, w33, w34, w41, w42, w43, w44):
                    result1 = y1(tdata1, A1, t1, t2, t3, t4, w11, w12, w13, w14)
                    result2 = y2(tdata2, A2, t1, t2, t3, t4, w21, w22, w23, w24)
                    result3 = y3(tdata3, A3, t1, t2, t3, t4, w31, w32, w33, w34)
                    result4 = y4(tdata4, A4, t1, t2, t3, t4, w41, w42, w43, w44)
                    return np.append(  np.append(  np.append(result1, result2), result3), result4)
                # some initial parameter values
            
                # curve fit the combined data to the combined function
                t_combo = []
                a_combo = []
                for i in dataset:
                    t_combo = np.append(t_combo, i[0])
                    a_combo = np.append(a_combo, i[1])
                    
                data1 = dataset[0]
                data2 = dataset[1]
                data3 = dataset[2]
                data4 = dataset[3]
                
                tdata1, adata1 = data1
                tdata2, adata2 = data2
                tdata3, adata3 = data3
                tdata4, adata4 = data4
                
                fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, self.x0, bounds = self.bounds)
                
                A1, A2, A3, A4, t1, t2, t3, t4, w11, w12, w13, w14, w21, w22, w23, w24, w31, w32, w33, w34, w41, w42, w43, w44 = fittedParameters
                A1err, A2err, A3err, A4err, t1err, t2err, t3err, t4err, w11err, w12err, w13err, w14err, w21err, w22err, w23err, w24err, w31err, w32err, w33err, w34err, w41err, w42err, w43err, w44err = np.sqrt(np.diag(pcov))
                print('A1err = %.2f, A2err = %.2f, A3err = %.2f, A4err = %.2f, t1err = %.2f, t2err = %.2f, t3err = %.2f, t4err = %.2f, w11err = %.2f, w12err = %.2f, w13err = %.2f, w14err = %.2f, w21err = %.2f, w22err = %.2f, w23err = %.2f, w24err = %.2f, w31err = %.2f, w32err = %.2f, w33err = %.2f, w34err = %.2f, w41err = %.2f, w42err = %.2f, w43err = %.2f, w44err = %.2f'
                      % tuple((A1err, A2err, A3err, t1err, t2err, t3err, t4err, w11err, w12err, w13err, w14err, w21err, w22err, w23err, w24err, w31err, w32err, w33err, w34err, w41err, w42err, w43err, w44err)))
        
                plt.figure()
                g = G(tdata1, A1, self.t0, self.Wpump)
                plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata1, g*max(adata1)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata1, w11*result1[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata1, w12*result1[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata1, w13*result1[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata1, w14*result1[:, 3], 'm-', label = 'w4*N4')
                fit_result = w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2] + w14*result1[:, 3]
                plt.plot(tdata1, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4')
                plt.scatter(tdata1, adata1)
                plt.legend()
                wsum = w11 + w12 + w13 + w14
                w11 = w11 / wsum
                w12 = w12 / wsum
                w13 = w13 / wsum
                w14 = w14 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f fs, t4 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f' % tuple((A1, t1, t2, t3, t4, w11, w12, w13, w14)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
                plt.ylabel('ΔA (mOD)')

                plt.figure()
                g = G(tdata2, A2, self.t0, self.Wpump)
                plt.plot(tdata2, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata2, g*max(adata2)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata2, w21*result2[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata2, w22*result2[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata2, w23*result2[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata2, w24*result2[:, 3], 'm-', label = 'w4*N4')
                fit_result = w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2] + w24*result2[:, 3]
                plt.plot(tdata2, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4')
                plt.scatter(tdata2, adata2)
                plt.legend()
                wsum = w21 + w22 + w23 + w24
                w21 = w21 / wsum
                w22 = w22 / wsum
                w23 = w23 / wsum
                w24 = w24 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f' % tuple((A2, t1, t2, t3, t4, w21, w22, w23, w24)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[1])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata3, A3, self.t0, self.Wpump)
                plt.plot(tdata3, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata3, g*max(adata3)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata3, w31*result3[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata3, w32*result3[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata3, w33*result3[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata3, w34*result3[:, 3], 'm-', label = 'w4*N4')
                fit_result = w31*result3[:, 0] + w32*result3[:, 1] + w33*result3[:, 2] + w34*result3[:, 3]
                plt.plot(tdata3, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4')
                plt.scatter(tdata3, adata3)
                plt.legend()
                wsum = w31 + w32 + w33 + w34
                w31 = w31 / wsum
                w32 = w32 / wsum
                w33 = w33 / wsum
                w34 = w34 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f' % tuple((A3, t1, t2, t3, t4, w31, w32, w33, w34)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[2])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata4, A4, self.t0, self.Wpump)
                plt.plot(tdata4, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata4, g*max(adata4)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata4, w41*result4[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata4, w42*result4[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata4, w43*result4[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata4, w44*result4[:, 3], 'm-', label = 'w4*N4')
                fit_result = w41*result4[:, 0] + w42*result4[:, 1] + w43*result4[:, 2] + w44*result4[:, 3]
                plt.plot(tdata4, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4')
                plt.scatter(tdata4, adata4)
                plt.legend()
                wsum = w41 + w42 + w43 + w44
                w41 = w41 / wsum
                w42 = w42 / wsum
                w43 = w43 / wsum
                w44 = w44 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f' % tuple((A4, t1, t2, t3, t4, w41, w42, w43, w44)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[3])
                plt.ylabel('ΔA (mOD)')
                plt.show()
                
                return A1, A2, A3, A4, t1, t2, t3, t4, w11, w12, w13, w14, w21, w22, w23, w24, w31, w32, w33, w34, w41, w42, w43, w44, result1, result2, result3, result4
    
            elif self.number_of_states == 5:
                def G(t, A, t0, Wpump):
                    return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
                
                def model(y, t, A, t1, t2, t3, t4, t5):
                    N1 = y[0]
                    N2 = y[1]
                    N3 = y[2]
                    N4 = y[3]
                    N5 = y[4]
                    dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                    dN2_dt = N1/t1 - N2/t2
                    dN3_dt = N2/t2 - N3/t3
                    dN4_dt = N3/t3 - N4/t4
                    dN5_dt = N4/t4 - N5/t5
                    return [dN1_dt, dN2_dt, dN3_dt, dN4_dt, dN5_dt]
        
                def y1(t, A1, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15):
                    y0 = [0, 0, 0, 0, 0] # the states are initially unoccupied
                    global result1
                    result1 = integrate.odeint(model, y0, t, args = (A1, t1, t2, t3, t4, t5,))
                    return w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2] + w14*result1[:, 3] + w15*result1[:, 4]
        
                def y2(t, A2, t1, t2, t3, t4, t5, w21, w22, w23, w24, w25):
                    y0 = [0, 0, 0, 0, 0]
                    global result2
                    result2 = integrate.odeint(model, y0, t, args = (A2, t1, t2, t3, t4, t5,))
                    return w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2] + w24*result2[:, 3] + w25*result2[:, 4]
                    
                def y3(t, A3, t1, t2, t3, t4, t5, w31, w32, w33, w34, w35):
                    y0 = [0, 0, 0, 0, 0]
                    global result3
                    result3 = integrate.odeint(model, y0, t, args = (A3, t1, t2, t3, t4, t5,))
                    return w31*result3[:, 0] + w32*result3[:, 1] + w33*result3[:, 2] + w34*result3[:, 3] + w35*result3[:, 4]
                    
                def y4(t, A4, t1, t2, t3, t4, t5, w41, w42, w43, w44, w45):
                    y0 = [0, 0, 0, 0, 0]
                    global result4
                    result4 = integrate.odeint(model, y0, t, args = (A4, t1, t2, t3, t4, t5,))
                    return w41*result4[:, 0] + w42*result4[:, 1] + w43*result4[:, 2] + w44*result4[:, 3] + w45*result4[:, 4]
        
                def comboFunc(comboData, A1, A2, A3, A4, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15, w21, w22, w23, w24, w25, w31, w32, w33, w34, w35, w41, w42, w43, w44, w45):
                    result1 = y1(tdata1, A1, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15)
                    result2 = y2(tdata2, A2, t1, t2, t3, t4, t5, w21, w22, w23, w24, w25)
                    result3 = y3(tdata3, A3, t1, t2, t3, t4, t5, w31, w32, w33, w34, w35)
                    result4 = y4(tdata4, A4, t1, t2, t3, t4, t5, w41, w42, w43, w44, w45)
                    return np.append(  np.append(  np.append(result1, result2), result3), result4)
                # some initial parameter values
            
                # curve fit the combined data to the combined function
                t_combo = []
                a_combo = []
                for i in dataset:
                    t_combo = np.append(t_combo, i[0])
                    a_combo = np.append(a_combo, i[1])
                    
                data1 = dataset[0]
                data2 = dataset[1]
                data3 = dataset[2]
                data4 = dataset[3]
                
                tdata1, adata1 = data1
                tdata2, adata2 = data2
                tdata3, adata3 = data3
                tdata4, adata4 = data4
                
                fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, self.x0, bounds = self.bounds)
                
                A1, A2, A3, A4, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15, w21, w22, w23, w24, w25, w31, w32, w33, w34, w35, w41, w42, w43, w44, w45 = fittedParameters
                A1err, A2err, A3err, A4err, t1err, t2err, t3err, t4err, t5err, w11err, w12err, w13err, w14err, w15err, w21err, w22err, w23err, w24err, w25err, w31err, w32err, w33err, w34err, w35err, w41err, w42err, w43err, w44err, w45err = np.sqrt(np.diag(pcov))
                print('A1err = %.2f, A2err = %.2f, A3err = %.2f, A4err = %.2f, t1err = %.2f, t2err = %.2f, t3err = %.2f, t4err = %.2f, t5err = %.2f, w11err = %.2f, w12err = %.2f, w13err = %.2f, w14err = %.2f, w15err = %.2f, w21err = %.2f, w22err = %.2f, w23err = %.2f, w24err = %.2f w25err = %.2f, w31, w32err = %.2f, w33err = %.2f, w34err = %.2f, w35err = %.2f, w41err = %.2f, w42err = %.2f, w43err = %.2f, w44err = %.2f, w45err = %.2f'
                      % tuple((A1err, A2err, A3err, A4err, t1err, t2err, t3err, t4err, t5err, w11err, w12err, w13err, w14err, w15err, w21err, w22err, w23err, w24err, w25err, w31err, w32err, w33err, w34err, w35err, w41err, w42err, w43err, w44err, w45err)))
        
                plt.figure()
                g = G(tdata1, A1, self.t0, self.Wpump)
                plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata1, g*max(adata1)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata1, w11*result1[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata1, w12*result1[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata1, w13*result1[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata1, w14*result1[:, 3], 'm-', label = 'w4*N4')
                plt.plot(tdata1, w15*result1[:, 4], 'c-', label = 'w5*N5')
                fit_result = w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2] + w14*result1[:, 3] + w15*result1[:, 4]
                plt.plot(tdata1, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4 + w5*N5')
                plt.scatter(tdata1, adata1)
                plt.legend()
                wsum = w11 + w12 + w13 + w14 + w15
                w11 = w11 / wsum
                w12 = w12 / wsum
                w13 = w13 / wsum
                w14 = w14 / wsum
                w15 = w15 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f fs, t4 = %.2f, t5 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f, w5 = %.2f' % tuple((A1, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata2, A2, self.t0, self.Wpump)
                plt.plot(tdata2, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata2, g*max(adata2)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata2, w21*result2[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata2, w22*result2[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata2, w23*result2[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata2, w24*result2[:, 3], 'm-', label = 'w4*N4')
                plt.plot(tdata2, w25*result2[:, 4], 'c-', label = 'w5*N5')
                fit_result = w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2] + w24*result2[:, 3] + w25*result2[:, 4]
                plt.plot(tdata2, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4 + w5*N5')
                plt.scatter(tdata2, adata2)
                plt.legend()
                wsum = w21 + w22 + w23 + w24 + w25
                w21 = w21 / wsum
                w22 = w22 / wsum
                w23 = w23 / wsum
                w24 = w24 / wsum
                w25 = w25 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, t5 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f, w5 = %.2f' % tuple((A2, t1, t2, t3, t4, t5, w21, w22, w23, w24, w25)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[1])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata3, A3, self.t0, self.Wpump)
                plt.plot(tdata3, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata3, g*max(adata3)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata3, w31*result3[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata3, w32*result3[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata3, w33*result3[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata3, w34*result3[:, 3], 'm-', label = 'w4*N4')
                plt.plot(tdata3, w35*result3[:, 4], 'c-', label = 'w5*N5')
                fit_result = w31*result3[:, 0] + w32*result3[:, 1] + w33*result3[:, 2] + w34*result3[:, 3] + w35*result3[:, 4]
                plt.plot(tdata3, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4 + w5*N5')
                plt.scatter(tdata3, adata3)
                plt.legend()
                wsum = w31, w32, w33, w34, w35
                w31 = w31 / wsum
                w32 = w32 / wsum
                w33 = w33 / wsum
                w34 = w34 / wsum
                w35 = w35 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, t5 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f, w5 = %.2f' % tuple((A3, t1, t2, t3, t4, t5, w31, w32, w33, w34, w35)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[2])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata4, A4, self.t0, self.Wpump)
                plt.plot(tdata4, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata4, g*max(adata4)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata4, w41*result4[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata4, w42*result4[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata4, w43*result4[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata4, w44*result4[:, 3], 'm-', label = 'w4*N4')
                plt.plot(tdata4, w45*result4[:, 4], 'c-', label = 'w5*N5')
                fit_result = w41*result4[:, 0] + w42*result4[:, 1] + w43*result4[:, 2] + w44*result4[:, 3] + w45*result4[:, 4]
                plt.plot(tdata4, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4 + w5*N5')
                plt.scatter(tdata4, adata4)
                plt.legend()
                wsum = w41, w42, w43, w44, w45
                w41 = w41 / wsum
                w42 = w42 / wsum
                w43 = w43 / wsum
                w44 = w44 / wsum
                w45 = w45 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, t5 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f, w5 = %.2f' % tuple((A4, t1, t2, t3, t4, t5, w41, w42, w43, w44, w45)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[3])
                plt.ylabel('ΔA (mOD)')
                plt.show()
                
                return A1, A2, A3, A4, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15, w21, w22, w23, w24, w25, w31, w32, w33, w34, w35, w41, w42, w43, w44, w45, result1, result2, result3, result4
            
            else:
                print('Currently only 1-5 states are supported for global fitting. 5 states will be fitted by default instead.')
                def G(t, A, t0, Wpump):
                    return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
                
                def model(y, t, A, t1, t2, t3, t4, t5):
                    N1 = y[0]
                    N2 = y[1]
                    N3 = y[2]
                    N4 = y[3]
                    N5 = y[4]
                    dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                    dN2_dt = N1/t1 - N2/t2
                    dN3_dt = N2/t2 - N3/t3
                    dN4_dt = N3/t3 - N4/t4
                    dN5_dt = N4/t4 - N5/t5
                    return [dN1_dt, dN2_dt, dN3_dt, dN4_dt, dN5_dt]
        
                def y1(t, A1, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15):
                    y0 = [0, 0, 0, 0, 0] # the states are initially unoccupied
                    global result1
                    result1 = integrate.odeint(model, y0, t, args = (A1, t1, t2, t3, t4, t5,))
                    return w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2] + w14*result1[:, 3] + w15*result1[:, 4]
        
                def y2(t, A2, t1, t2, t3, t4, t5, w21, w22, w23, w24, w25):
                    y0 = [0, 0, 0, 0, 0]
                    global result2
                    result2 = integrate.odeint(model, y0, t, args = (A2, t1, t2, t3, t4, t5,))
                    return w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2] + w24*result2[:, 3] + w25*result2[:, 4]
                    
                def y3(t, A3, t1, t2, t3, t4, t5, w31, w32, w33, w34, w35):
                    y0 = [0, 0, 0, 0, 0]
                    global result3
                    result3 = integrate.odeint(model, y0, t, args = (A3, t1, t2, t3, t4, t5,))
                    return w31*result3[:, 0] + w32*result3[:, 1] + w33*result3[:, 2] + w34*result3[:, 3] + w35*result3[:, 4]
                    
                def y4(t, A4, t1, t2, t3, t4, t5, w41, w42, w43, w44, w45):
                    y0 = [0, 0, 0, 0, 0]
                    global result4
                    result4 = integrate.odeint(model, y0, t, args = (A4, t1, t2, t3, t4, t5,))
                    return w41*result4[:, 0] + w42*result4[:, 1] + w43*result4[:, 2] + w44*result4[:, 3] + w45*result4[:, 4]
        
                def comboFunc(comboData, A1, A2, A3, A4, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15, w21, w22, w23, w24, w25, w31, w32, w33, w34, w35, w41, w42, w43, w44, w45):
                    result1 = y1(tdata1, A1, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15)
                    result2 = y2(tdata2, A2, t1, t2, t3, t4, t5, w21, w22, w23, w24, w25)
                    result3 = y3(tdata3, A3, t1, t2, t3, t4, t5, w31, w32, w33, w34, w35)
                    result4 = y4(tdata4, A4, t1, t2, t3, t4, t5, w41, w42, w43, w44, w45)
                    return np.append(  np.append(  np.append(result1, result2), result3), result4)
                # some initial parameter values
            
                # curve fit the combined data to the combined function
                t_combo = []
                a_combo = []
                for i in dataset:
                    t_combo = np.append(t_combo, i[0])
                    a_combo = np.append(a_combo, i[1])
                    
                data1 = dataset[0]
                data2 = dataset[1]
                data3 = dataset[2]
                data4 = dataset[3]
                
                tdata1, adata1 = data1
                tdata2, adata2 = data2
                tdata3, adata3 = data3
                tdata4, adata4 = data4

                fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, self.x0, bounds = self.bounds)
                
                A1, A2, A3, A4, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15, w21, w22, w23, w24, w25, w31, w32, w33, w34, w35, w41, w42, w43, w44, w45 = fittedParameters
                A1err, A2err, A3err, A4err, t1err, t2err, t3err, t4err, t5err, w11err, w12err, w13err, w14err, w15err, w21err, w22err, w23err, w24err, w25err, w31err, w32err, w33err, w34err, w35err, w41err, w42err, w43err, w44err, w45err = np.sqrt(np.diag(pcov))
                print('A1err = %.2f, A2err = %.2f, A3err = %.2f, A4err = %.2f, t1err = %.2f, t2err = %.2f, t3err = %.2f, t4err = %.2f, t5err = %.2f, w11err = %.2f, w12err = %.2f, w13err = %.2f, w14err = %.2f, w15err = %.2f, w21err = %.2f, w22err = %.2f, w23err = %.2f, w24err = %.2f w25err = %.2f, w31, w32err = %.2f, w33err = %.2f, w34err = %.2f, w35err = %.2f, w41err = %.2f, w42err = %.2f, w43err = %.2f, w44err = %.2f, w45err = %.2f' % tuple((A1err, A2err, A3err, A4err, t1err, t2err, t3err, t4err, t5err, w11err, w12err, w13err, w14err, w15err, w21err, w22err, w23err, w24err, w25err, w31err, w32err, w33err, w34err, w35err, w41err, w42err, w43err, w44err, w45err)))
        
                plt.figure()
                g = G(tdata1, A1, self.t0, self.Wpump)
                plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata1, g*max(adata1)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata1, w11*result1[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata1, w12*result1[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata1, w13*result1[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata1, w14*result1[:, 3], 'm-', label = 'w4*N4')
                plt.plot(tdata1, w15*result1[:, 4], 'c-', label = 'w5*N5')
                fit_result = w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2] + w14*result1[:, 3] + w15*result1[:, 4]
                plt.plot(tdata1, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4 + w5*N5')
                plt.scatter(tdata1, adata1)
                plt.legend()
                wsum = w11 + w12 + w13 + w14 + w15
                w11 = w11 / wsum
                w12 = w12 / wsum
                w13 = w13 / wsum
                w14 = w14 / wsum
                w15 = w15 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f fs, t4 = %.2f, t5 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f, w5 = %.2f' % tuple((A1, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata2, A2, self.t0, self.Wpump)
                plt.plot(tdata2, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata2, g*max(adata2)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata2, w21*result2[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata2, w22*result2[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata2, w23*result2[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata2, w24*result2[:, 3], 'm-', label = 'w4*N4')
                plt.plot(tdata2, w25*result2[:, 4], 'c-', label = 'w5*N5')
                fit_result = w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2] + w24*result2[:, 3] + w25*result2[:, 4]
                plt.plot(tdata2, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4 + w5*N5')
                plt.scatter(tdata2, adata2)
                plt.legend()
                wsum = w21 + w22 + w23 + w24 + w25
                w21 = w21 / wsum
                w22 = w22 / wsum
                w23 = w23 / wsum
                w24 = w24 / wsum
                w25 = w25 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, t5 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f, w5 = %.2f' % tuple((A2, t1, t2, t3, t4, t5, w21, w22, w23, w24, w25)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[1])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata3, A3, self.t0, self.Wpump)
                plt.plot(tdata3, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata3, g*max(adata3)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata3, w31*result3[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata3, w32*result3[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata3, w33*result3[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata3, w34*result3[:, 3], 'm-', label = 'w4*N4')
                plt.plot(tdata3, w35*result3[:, 4], 'c-', label = 'w5*N5')
                fit_result = w31*result3[:, 0] + w32*result3[:, 1] + w33*result3[:, 2] + w34*result3[:, 3] + w35*result3[:, 4]
                plt.plot(tdata3, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4 + w5*N5')
                plt.scatter(tdata3, adata3)
                plt.legend()
                wsum = w31, w32, w33, w34, w35
                w31 = w31 / wsum
                w32 = w32 / wsum
                w33 = w33 / wsum
                w34 = w34 / wsum
                w35 = w35 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, t5 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f, w5 = %.2f' % tuple((A3, t1, t2, t3, t4, t5, w31, w32, w33, w34, w35)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[2])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata4, A4, self.t0, self.Wpump)
                plt.plot(tdata4, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata4, g*max(adata4)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata4, w41*result4[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata4, w42*result4[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata4, w43*result4[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata4, w44*result4[:, 3], 'm-', label = 'w4*N4')
                plt.plot(tdata4, w45*result4[:, 4], 'c-', label = 'w5*N5')
                fit_result = w41*result4[:, 0] + w42*result4[:, 1] + w43*result4[:, 2] + w44*result4[:, 3] + w45*result4[:, 4]
                plt.plot(tdata4, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4 + w5*N5')
                plt.scatter(tdata4, adata4)
                plt.legend()
                wsum = w41, w42, w43, w44, w45
                w41 = w41 / wsum
                w42 = w42 / wsum
                w43 = w43 / wsum
                w44 = w44 / wsum
                w45 = w45 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, t5 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f, w5 = %.2f' % tuple((A4, t1, t2, t3, t4, t5, w41, w42, w43, w44, w45)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[3])
                plt.ylabel('ΔA (mOD)')
                plt.show()
                
                return A1, A2, A3, A4, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15, w21, w22, w23, w24, w25, w31, w32, w33, w34, w35, w41, w42, w43, w44, w45, result1, result2, result3, result4
                
###############################################################################                
                
        elif len(self.wavelengths) == 5:
            # global fitting of five wavelengths
            if self.number_of_states == 1:
                def G(t, A, t0, Wpump):
                    return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
             
                def model(y, t, A, t1):
                   N1 = y[0]
                   dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                   return [dN1_dt]
            
                def y1(t, A1, t1):
                    y0 = [0] # the states are initially unoccupied
                    global result1
                    result1 = integrate.odeint(model, y0, t, args = (A1, t1,))
                    return result1[:, 0]
        
                def y2(t, A2, t1):
                    y0 = [0]
                    global result2
                    result2 = integrate.odeint(model, y0, t, args = (A2, t1,))
                    return result2[:, 0]
                
                def y3(t, A3, t1):
                    y0 = [0]
                    global result3
                    result3 = integrate.odeint(model, y0, t, args = (A3, t1,))
                    return result3[:, 0]
                    
                def y4(t, A4, t1):
                    y0 = [0]
                    global result4
                    result4 = integrate.odeint(model, y0, t, args = (A4, t1,))
                    return result4[:, 0]
                
                def y5(t, A5, t1):
                    y0 = [0]
                    global result5
                    result5 = integrate.odeint(model, y0, t, args = (A5, t1,))  
                    return result5[:, 0]
                
                def comboFunc(comboData, A1, A2, A3, A4, A5, t1):
                    result1 = y1(tdata1, A1, t1)
                    result2 = y2(tdata2, A2, t1)
                    result3 = y3(tdata3, A3, t1)
                    result4 = y4(tdata4, A4, t1)
                    result5 = y5(tdata5, A5, t1)
                    return np.append(  np.append(  np.append(  np.append(result1, result2), result3), result4), result5)
             
                t_combo = []
                a_combo = []
                for i in dataset:
                    t_combo = np.append(t_combo, i[0])
                    a_combo = np.append(a_combo, i[1])
                    
                data1 = dataset[0]
                data2 = dataset[1]
                data3 = dataset[2]
                data4 = dataset[3]
                data5 = dataset[4]
                
                tdata1, adata1 = data1
                tdata2, adata2 = data2
                tdata3, adata3 = data3
                tdata4, adata4 = data4
                tdata5, adata5 = data5
                
                fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, self.x0, bounds = self.bounds)
        
                A1, A2, A3, A4, A5, t1 = fittedParameters
                A1err, A2err, A3err, A4err, A5err, t1err = np.sqrt(np.diag(pcov))
                print('A1err = %.2f, A2err = %.2f, A3err = %.2f, A4err = %.2f, A5err = %.2f, t1err = %.2f' % tuple((A1err, A2err, A3err, A4err, A5err, t1err)))
        
                plt.figure()
                g = G(tdata1, A1, self.t0, self.Wpump)
                plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata1, g*max(adata1)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                fit_result = result1[:,0]
                plt.plot(tdata1, fit_result, 'k--', label = 'N1')
                plt.scatter(tdata1, adata1)
                plt.legend()
                plt.title('A = %.2f, t1 = %.2f fs' % tuple((A1, t1)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata2, A2, self.t0, self.Wpump)
                plt.plot(tdata2, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata2, g*max(adata2)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                fit_result = result2[:,0]
                plt.plot(tdata2, fit_result, 'k--', label = 'N1')
                plt.scatter(tdata2, adata2)
                plt.legend()
                plt.title('A = %.2f, t1 = %.2f fs' % tuple((A2, t1)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[1])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata3, A3, self.t0, self.Wpump)
                plt.plot(tdata3, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata3, g*max(adata3)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                fit_result = result3[:,0]
                plt.plot(tdata3, fit_result, 'k--', label = 'N1')
                plt.scatter(tdata3, adata3)
                plt.legend()
                plt.title('A = %.2f, t1 = %.2f fs' % tuple((A3, t1)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[2])
                plt.ylabel('ΔA (mOD)')

                plt.figure()
                g = G(tdata4, A4, self.t0, self.Wpump)
                plt.plot(tdata4, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                # plt.plot(tdata4, g*max(adata4)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                fit_result = result4[:,0]
                plt.plot(tdata4, fit_result, 'k--', label = 'N1')
                plt.scatter(tdata4, adata4)
                plt.legend()
                plt.title('A = %.2f, t1 = %.2f fs' % tuple((A4, t1)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[3])
                plt.ylabel('ΔA (mOD)')

                plt.figure()
                g = G(tdata5, A5, self.t0, self.Wpump)
                plt.plot(tdata5, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                # plt.plot(tdata5, g*max(adata5)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                fit_result = result5[:,0]
                plt.plot(tdata5, fit_result, 'k--', label = 'N1')
                plt.scatter(tdata5, adata5)
                plt.legend()
                plt.title('A = %.2f, t1 = %.2f fs' % tuple((A5, t1)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[4])
                plt.ylabel('ΔA (mOD)')
                plt.show()
                
                return A1, A2, A3, A4, A5, t1, result1, result2, result3, result4, result5
                  
            elif self.number_of_states == 2:
                def G(t, A, t0, Wpump):
                    return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
                
                def model(y, t, A, t1, t2):
                    N1 = y[0]
                    N2 = y[1]
                    dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                    dN2_dt = N1/t1 - N2/t2
                    return [dN1_dt, dN2_dt]
                
                def y1(t, A1, t1, t2, w11, w12):
                    y0 = [0, 0] # the states are initially unoccupied
                    global result1
                    result1 = integrate.odeint(model, y0, t, args = (A1, t1, t2,))
                    return w11*result1[:, 0] + w12*result1[:, 1]
        
                def y2(t, A2, t1, t2, w21, w22):
                    y0 = [0, 0]
                    global result2
                    result2 = integrate.odeint(model, y0, t, args = (A2, t1, t2,))
                    return w21*result2[:, 0] + w22*result2[:, 1]
                
                def y3(t, A3, t1, t2, w31, w32):
                    y0 = [0, 0]
                    global result3
                    result3 = integrate.odeint(model, y0, t, args = (A3, t1, t2,))
                    return w31*result3[:, 0] + w32*result2[:, 1]
                    
                def y4(t, A4, t1, t2, w41, w42):
                    y0 = [0, 0]
                    global result4
                    result4 = integrate.odeint(model, y0, t, args = (A4, t1, t2,))
                    return w41*result3[:, 0] + w42*result4[:, 1]
                
                def y5(t, A5, t1, t2, w51, w52):
                    y0 = [0, 0]
                    global result5
                    result5 = integrate.odeint(model, y0, t, args = (A5, t1, t2,))
                    return w51*result5[:, 0] + w52*result5[:, 1]
        
                def comboFunc(comboData, A1, A2, A3, A4, A5, t1, t2, w11, w12, w21, w22, w31, w32, w41, w42, w51, w52):
                    result1 = y1(tdata1, A1, t1, t2, w11, w12)
                    result2 = y2(tdata2, A2, t1, t2, w21, w22)
                    result3 = y3(tdata3, A3, t1, t2, w31, w32)
                    result4 = y4(tdata4, A4, t1, t2, w41, w42)
                    result5 = y5(tdata5, A5, t1, t2, w51, w52)
                    return np.append(  np.append(  np.append(  np.append(result1, result2), result3), result4), result5)
                # some initial parameter values
            
                # curve fit the combined data to the combined function
                t_combo = []
                a_combo = []
                for i in dataset:
                    t_combo = np.append(t_combo, i[0])
                    a_combo = np.append(a_combo, i[1])
                    
                data1 = dataset[0]
                data2 = dataset[1]
                data3 = dataset[2]
                data4 = dataset[3]
                data5 = dataset[4]
                
                tdata1, adata1 = data1
                tdata2, adata2 = data2
                tdata3, adata3 = data3
                tdata4, adata4 = data4
                tdata5, adata5 = data5
                
                fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, self.x0, bounds = self.bounds)
                
                A1, A2, A3, A4, A5, t1, t2, w11, w12, w21, w22, w31, w32, w41, w42, w51, w52 = fittedParameters
                A1err, A2err, A3err, A4err, A5err, t1err, t2err, w11err, w12err, w21err, w22err, w31err, w32err, w41err, w42err, w51err, w52err = np.sqrt(np.diag(pcov))
                print('A1err = %.2f, A2err = %.2f, A3err = %.2f, A43rr = %.2f, A5err = %.2f, t1err = %.2f, t2err = %.2f, w11err = %.2f, w12err = %.2f, w21err = %.2f, w22err = %.2f, w31err = %.2f, w32err = %.2f, w41err = %.2f, w42 = %.2f, w51 = %.2f, w52 = %.2f' % tuple((A1err, A2err, A3err, A4err, A5err, t1err, t2err, w11err, w12err, w21err, w22err, w31err, w32err, w41err, w42err, w51err, w52err)))
        
                plt.figure()
                if self.tshift == None:
                    g = G(tdata1, A1, self.t0, self.Wpump)
                    plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata1, g*max(abs(adata1))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero') 
                else:
                    g = G(tdata1, A1, self.t0 + self.tshift[0], self.Wpump)
                    plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata1, g*max(abs(adata1))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0 + self.tshift[0], color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata1, w11*result1[:,0], 'r-', label = 'w1*N1')
                plt.plot(tdata1, w12*result1[:,1], 'b-', label = 'w2*N2')
                fit_result = w11*result1[:,0] + w12*result1[:,1]
                plt.plot(tdata1, fit_result, 'k--', label = 'w1*N1 + w2*N2')
                plt.scatter(tdata1, adata1)
                plt.legend()
                wsum = w11 + w12
                w11 = w11 / wsum
                w12 = w12 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, w1 = %.2f, w2 = %.2f' % tuple((A1, t1, t2, w11, w12)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata2, A2, self.t0, self.Wpump)
                plt.plot(tdata2, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                # plt.plot(tdata2, g*max(adata2)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata2, w21*result2[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata2, w22*result2[:, 1], 'b-', label = 'w2*N2')
                fit_result = w21*result2[:,0] + w22*result2[:,1]
                plt.plot(tdata2, fit_result, 'k--', label = 'w1*N1 + w2*N2')
                plt.scatter(tdata2, adata2)
                plt.legend()
                wsum = w21 + w22
                w21 = w21 / wsum
                w22 = w22 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, w1 = %.2f, w2 = %.2f' % tuple((A2, t1, t2, w21, w22)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[1])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata3, A3, self.t0, self.Wpump)
                plt.plot(tdata3, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                # plt.plot(tdata3, g*max(adata3)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata3, w31*result3[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata3, w32*result3[:, 1], 'b-', label = 'w2*N2')
                fit_result = w31*result3[:,0] + w32*result3[:,1]
                plt.plot(tdata3, fit_result, 'k--', label = 'w1*N1 + w2*N2')
                plt.scatter(tdata3, adata3)
                plt.legend()
                wsum = w31 + w32
                w31 = w31 / wsum
                w32 = w32 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, w1 = %.2f, w2 = %.2f' % tuple((A3, t1, t2, w31, w32)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[2])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata4, A4, self.t0, self.Wpump)
                plt.plot(tdata4, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                # plt.plot(tdata4, g*max(adata4)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata4, w41*result4[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata4, w42*result4[:, 1], 'b-', label = 'w2*N2')
                fit_result = w41*result4[:,0] + w42*result4[:,1]
                plt.plot(tdata4, fit_result, 'k--', label = 'w1*N1 + w2*N2')
                plt.scatter(tdata4, adata4)
                plt.legend()
                wsum = w41 + w42
                w41 = w41 / wsum
                w42 = w42 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, w1 = %.2f, w2 = %.2f' % tuple((A4, t1, t2, w41, w42)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[3])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata5, A5, self.t0, self.Wpump)
                plt.plot(tdata5, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                # plt.plot(tdata5, g*max(adata5)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata5, w51*result5[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata5, w52*result5[:, 1], 'b-', label = 'w2*N2')
                fit_result = w51*result5[:,0] + w52*result5[:,1]
                plt.plot(tdata5, fit_result, 'k--', label = 'w1*N1 + w2*N2')
                plt.scatter(tdata5, adata5)
                plt.legend()
                wsum = w51 + w52
                w51 = w51 / wsum
                w52 = w52 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, w1 = %.2f, w2 = %.2f' % tuple((A5, t1, t2, w51, w52)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[4])
                plt.ylabel('ΔA (mOD)')
                plt.show()
                
                return A1, A2, A3, A4, A5, t1, t2, w11, w12, w21, w22, w31, w32, w41, w42, w51, w52, result1, result2, result3, result4, result5
     
            elif self.number_of_states == 3:
                def G(t, A, t0, Wpump):
                    return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
                
                def model(y, t, A, t1, t2, t3):
                    N1 = y[0]
                    N2 = y[1]
                    N3 = y[2]
                    dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                    dN2_dt = N1/t1 - N2/t2
                    dN3_dt = N2/t2 - N3/t3
                    return [dN1_dt, dN2_dt, dN3_dt]
        
                def y1(t, A1, t1, t2, t3, w11, w12, w13):
                    y0 = [0, 0, 0] # the states are initially unoccupied
                    global result1
                    result1 = integrate.odeint(model, y0, t, args = (A1, t1, t2, t3,))
                    return w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2]
        
                def y2(t, A2, t1, t2, t3, w21, w22, w23):
                    y0 = [0, 0, 0]
                    global result2
                    result2 = integrate.odeint(model, y0, t, args = (A2, t1, t2, t3,))
                    return w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2]
                    
                def y3(t, A3, t1, t2, t3, w31, w32, w33):
                    y0 = [0, 0, 0]
                    global result3
                    result3 = integrate.odeint(model, y0, t, args = (A3, t1, t2, t3,))
                    return w31*result3[:, 0] + w32*result3[:, 1] + w33*result3[:, 2]
                    
                def y4(t, A4, t1, t2, t3, w41, w42, w43):
                    y0 = [0, 0, 0]
                    global result4
                    result4 = integrate.odeint(model, y0, t, args = (A4, t1, t2, t3,))
                    return w41*result4[:, 0] + w42*result4[:, 1] + w43*result4[:, 2]
                    
                def y5(t, A5, t1, t2, t3, w51, w52, w53):
                    y0 = [0, 0, 0]
                    global result5
                    result5 = integrate.odeint(model, y0, t, args = (A5, t1, t2, t3,))
                    return w51*result5[:, 0] + w52*result5[:, 1] + w53*result5[:, 2]
        
                def comboFunc(comboData, A1, A2, A3, A4, A5, t1, t2, t3, w11, w12, w13, w21, w22, w23, w31, w32, w33, w41, w42, w43, w51, w52, w53):
                    result1 = y1(tdata1, A1, t1, t2, t3, w11, w12, w13)
                    result2 = y2(tdata2, A2, t1, t2, t3, w21, w22, w23)
                    result3 = y3(tdata3, A3, t1, t2, t3, w31, w32, w33)
                    result4 = y4(tdata4, A4, t1, t2, t3, w41, w42, w43)
                    result5 = y5(tdata5, A5, t1, t2, t3, w51, w52, w53)
                    return np.append(  np.append(  np.append(  np.append(result1, result2), result3), result4), result5)
                # some initial parameter values
            
                # curve fit the combined data to the combined function
                t_combo = []
                a_combo = []
                for i in dataset:
                    t_combo = np.append(t_combo, i[0])
                    a_combo = np.append(a_combo, i[1])
                    
                data1 = dataset[0]
                data2 = dataset[1]
                data3 = dataset[2]
                data4 = dataset[3]
                data5 = dataset[4]
                
                tdata1, adata1 = data1
                tdata2, adata2 = data2
                tdata3, adata3 = data3
                tdata4, adata4 = data4
                tdata5, adata5 = data5
                
                fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, self.x0, bounds = self.bounds)
                
                A1, A2, A3, A4, A5, t1, t2, t3, w11, w12, w13, w21, w22, w23, w31, w32, w33, w41, w42, w43, w51, w52, w53 = fittedParameters
                A1err, A2err, A3err, A4err, A5err, t1err, t2err, t3err, w11err, w12err, w13err, w21err, w22err, w23err, w31err, w32err, w33err, w41err, w42err, w43err, w51err, w52err, w53err = np.sqrt(np.diag(pcov))
                print('A1err = %.2f, A2err = %.2f, A3err = %.2f, A4err = %.2f, A5err = %.2f, t1err = %.2f, t2err = %.2f, t3err = %.2f, w11err = %.2f, w12err = %.2f, w13err = %.2f, w21err = %.2f, w22err = %.2f, w23err = %.2f, w31err = %.2f, w32err = %.2f, w33err = %.2f, w41err = %.2f, w42err = %.2f, w43 = %.2f, w51err = %.2f, w52err = %.2f, w53 = %.2f' % tuple((A1err, A2err, A3err, A4err, A5err, t1err, t2err, t3err, w11err, w12err, w13err, w21err, w22err, w23err, w31err, w32err, w33err, w41err, w42err, w43err, w51err, w52err, w53err)))
        
                plt.figure()
                if self.tshift == None:
                    g = G(tdata1, A1, self.t0, self.Wpump)
                    plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata1, g*max(abs(adata1))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero') 
                else:
                    g = G(tdata1, A1, self.t0 + self.tshift[0], self.Wpump)
                    plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata1, g*max(abs(adata1))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0 + self.tshift[0], color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata1, w11*result1[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata1, w12*result1[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata1, w13*result1[:, 2], 'g-', label = 'w3*N3')
                fit_result = w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2]
                plt.plot(tdata1, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3')
                plt.scatter(tdata1, adata1)
                plt.legend()
                wsum = w11 + w12 + w13
                w11 = w11 / wsum
                w12 = w12 / wsum
                w13 = w13 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f fs, w1 = %.2f, w2 = %.2f, w3 = %.2f' % tuple((A1, t1, t2, t3, w11, w12, w13)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                if self.tshift == None:
                    g = G(tdata2, A2, self.t0, self.Wpump)
                    plt.plot(tdata2, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata2, g*max(abs(adata2))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero') 
                else:
                    g = G(tdata2, A2, self.t0 + self.tshift[1], self.Wpump)
                    plt.plot(tdata2, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata2, g*max(abs(adata2))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0 + self.tshift[1], color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata2, w21*result2[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata2, w22*result2[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata2, w23*result2[:, 2], 'g-', label = 'w3*N3')
                fit_result = w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2]
                plt.plot(tdata2, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3')
                plt.scatter(tdata2, adata2)
                plt.legend()
                wsum = w21 + w22 + w23
                w21 = w21 / wsum
                w22 = w22 / wsum
                w23 = w23 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f' % tuple((A2, t1, t2, t3, w21, w22, w23)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[1])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                if self.tshift == None:
                    g = G(tdata3, A3, self.t0, self.Wpump)
                    plt.plot(tdata3, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata3, g*max(abs(adata3))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero') 
                else:
                    g = G(tdata3, A3, self.t0 + self.tshift[2], self.Wpump)
                    plt.plot(tdata3, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata3, g*max(abs(adata3))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0 + self.tshift[2], color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata3, w31*result3[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata3, w32*result3[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata3, w33*result3[:, 2], 'g-', label = 'w3*N3')
                fit_result = w31*result3[:, 0] + w32*result3[:, 1] + w33*result3[:, 2]
                plt.plot(tdata3, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3')
                plt.scatter(tdata3, adata3)
                plt.legend()
                wsum = w31 + w32 + w33
                w31 = w31 / wsum
                w32 = w32 / wsum
                w33 = w33 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f' % tuple((A3, t1, t2, t3, w31, w32, w33)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[2])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                if self.tshift == None:
                    g = G(tdata4, A4, self.t0, self.Wpump)
                    plt.plot(tdata4, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata4, g*max(abs(adata4))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero') 
                else:
                    g = G(tdata4, A4, self.t0 + self.tshift[3], self.Wpump)
                    plt.plot(tdata4, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata4, g*max(abs(adata4))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0 + self.tshift[3], color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata4, w41*result4[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata4, w42*result4[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata4, w43*result4[:, 2], 'g-', label = 'w3*N3')
                fit_result = w41*result4[:, 0] + w42*result4[:, 1] + w43*result4[:, 2]
                plt.plot(tdata4, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3')
                plt.scatter(tdata4, adata4)
                plt.legend()
                wsum = w41 + w42 + w43
                w41 = w41 / wsum
                w42 = w42 / wsum
                w43 = w43 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f' % tuple((A4, t1, t2, t3, w41, w42, w43)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[3])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                if self.tshift == None:
                    g = G(tdata5, A5, self.t0, self.Wpump)
                    plt.plot(tdata5, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata5, g*max(abs(adata5))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero') 
                else:
                    g = G(tdata5, A5, self.t0 + self.tshift[4], self.Wpump)
                    plt.plot(tdata5, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata5, g*max(abs(adata5))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0 + self.tshift[4], color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata5, w51*result5[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata5, w52*result5[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata5, w53*result5[:, 2], 'g-', label = 'w3*N3')
                fit_result = w51*result5[:, 0] + w52*result5[:, 1] + w53*result5[:, 2]
                plt.plot(tdata5, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3')
                plt.scatter(tdata5, adata5)
                plt.legend()
                wsum = w51 + w52 + w53
                w51 = w51 / wsum
                w52 = w52 / wsum
                w53 = w53 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f' % tuple((A5, t1, t2, t3, w51, w52, w53)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[4])
                plt.ylabel('ΔA (mOD)')
                plt.show()
                
                return A1, A2, A3, A4, A5, t1, t2, t3, w11, w12, w13, w21, w22, w23, w31, w32, w33, w41, w42, w43, w51, w52, w53, result1, result2, result3, result4, result5
     
            elif self.number_of_states == 4:
                def G(t, A, t0, Wpump):
                    return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
                
                def model(y, t, A, t1, t2, t3, t4):
                    N1 = y[0]
                    N2 = y[1]
                    N3 = y[2]
                    N4 = y[3]
                    dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                    dN2_dt = N1/t1 - N2/t2
                    dN3_dt = N2/t2 - N3/t3
                    dN4_dt = N3/t3 - N4/t4
                    return [dN1_dt, dN2_dt, dN3_dt, dN4_dt]
        
                def y1(t, A1, t1, t2, t3, t4, w11, w12, w13, w14):
                    y0 = [0, 0, 0, 0] # the states are initially unoccupied
                    global result1
                    result1 = integrate.odeint(model, y0, t, args = (A1, t1, t2, t3, t4,))
                    return w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2] + w14*result1[:, 3]
        
                def y2(t, A2, t1, t2, t3, t4, w21, w22, w23, w24):
                    y0 = [0, 0, 0, 0]
                    global result2
                    result2 = integrate.odeint(model, y0, t, args = (A2, t1, t2, t3, t4,))
                    return w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2] + w24*result2[:, 3]
                    
                def y3(t, A3, t1, t2, t3, t4, w31, w32, w33, w34):
                    y0 = [0, 0, 0, 0]
                    global result3
                    result3 = integrate.odeint(model, y0, t, args = (A3, t1, t2, t3, t4,))
                    return w31*result3[:, 0] + w32*result3[:, 1] + w33*result3[:, 2] + w34*result3[:, 3]
                    
                def y4(t, A4, t1, t2, t3, t4, w41, w42, w43, w44):
                    y0 = [0, 0, 0, 0]
                    global result4
                    result4 = integrate.odeint(model, y0, t, args = (A4, t1, t2, t3, t4,))
                    return w41*result4[:, 0] + w42*result4[:, 1] + w43*result4[:, 2] + w44*result4[:, 3]
                    
                def y5(t, A5, t1, t2, t3, t4, w51, w52, w53, w54):
                    y0 = [0, 0, 0, 0]
                    global result5
                    result5 = integrate.odeint(model, y0, t, args = (A5, t1, t2, t3, t4,))
                    return w51*result5[:, 0] + w52*result5[:, 1] + w53*result5[:, 2] + w54*result5[:, 3]
                    
                def comboFunc(comboData, A1, A2, A3, A4, A5, t1, t2, t3, t4, w11, w12, w13, w14, w21, w22, w23, w24, w31, w32, w33, w34, w41, w42, w43, w44, w51, w52, w53, w54):
                    result1 = y1(tdata1, A1, t1, t2, t3, t4, w11, w12, w13, w14)
                    result2 = y2(tdata2, A2, t1, t2, t3, t4, w21, w22, w23, w24)
                    result3 = y3(tdata3, A3, t1, t2, t3, t4, w31, w32, w33, w34)
                    result4 = y4(tdata4, A4, t1, t2, t3, t4, w41, w42, w43, w44)
                    result5 = y5(tdata5, A5, t1, t2, t3, t4, w51, w52, w53, w54)
                    return np.append(  np.append(  np.append(  np.append(result1, result2), result3), result4), result5)
                # some initial parameter values
            
                # curve fit the combined data to the combined function
                t_combo = []
                a_combo = []
                for i in dataset:
                    t_combo = np.append(t_combo, i[0])
                    a_combo = np.append(a_combo, i[1])
                    
                data1 = dataset[0]
                data2 = dataset[1]
                data3 = dataset[2]
                data4 = dataset[3]
                data5 = dataset[4]
                
                tdata1, adata1 = data1
                tdata2, adata2 = data2
                tdata3, adata3 = data3
                tdata4, adata4 = data4
                tdata5, adata5 = data5
                
                fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, self.x0, bounds = self.bounds)
                
                A1, A2, A3, A4, A5, t1, t2, t3, t4, w11, w12, w13, w14, w21, w22, w23, w24, w31, w32, w33, w34, w41, w42, w43, w44, w51, w52, w53, w54 = fittedParameters
                A1err, A2err, A3err, A4err, A5err, t1err, t2err, t3err, t4err, w11err, w12err, w13err, w14err, w21err, w22err, w23err, w24err, w31err, w32err, w33err, w34err, w41err, w42err, w43err, w44err, w51err, w52err, w53err, w54err = np.sqrt(np.diag(pcov))
                print('A1err = %.2f, A2err = %.2f, A3err = %.2f, A4err = %.2f, A5err = %.2f, t1err = %.2f, t2err = %.2f, t3err = %.2f, t4err = %.2f, w11err = %.2f, w12err = %.2f, w13err = %.2f, w14err = %.2f, w21err = %.2f, w22err = %.2f, w23err = %.2f, w24err = %.2f, w31err = %.2f, w32err = %.2f, w33err = %.2f, w34err = %.2f, w41err = %.2f, w42err = %.2f, w43err = %.2f, w44err = %.2f, w51err = %.2f, w52 = %.2f, w53 = %.2f, w54 = %.2f' % tuple((A1err, A2err, A3err, A4err, A5err, t1err, t2err, t3err, t4err, w11err, w12err, w13err, w14err, w21err, w22err, w23err, w24err, w31err, w32err, w33err, w34err, w41err, w42err, w43err, w44err, w51err, w52err, w53err, w54err)))
        
                plt.figure()
                g = G(tdata1, A1, self.t0, self.Wpump)
                plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata1, g*max(adata1)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata1, w11*result1[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata1, w12*result1[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata1, w13*result1[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata1, w14*result1[:, 3], 'm-', label = 'w4*N4')
                fit_result = w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2] + w14*result1[:, 3]
                plt.plot(tdata1, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4')
                plt.scatter(tdata1, adata1)
                plt.legend()
                wsum = w11 + w12 + w13 + w14
                w11 = w11 / wsum
                w12 = w12 / wsum
                w13 = w13 / wsum
                w14 = w14 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f fs, t4 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f' % tuple((A1, t1, t2, t3, t4, w11, w12, w13, w14)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata2, A2, self.t0, self.Wpump)
                plt.plot(tdata2, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata2, g*max(adata2)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata2, w21*result2[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata2, w22*result2[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata2, w23*result2[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata2, w24*result2[:, 3], 'm-', label = 'w4*N4')
                fit_result = w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2] + w24*result2[:, 3]
                plt.plot(tdata2, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4')
                plt.scatter(tdata2, adata2)
                plt.legend()
                wsum = w21 + w22 + w23 + w24
                w21 = w21 / wsum
                w22 = w22 / wsum
                w23 = w23 / wsum
                w24 = w24 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f' % tuple((A2, t1, t2, t3, t4, w21, w22, w23, w24)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[1])
                plt.ylabel('ΔA (mOD)')

                plt.figure()
                g = G(tdata3, A3, self.t0, self.Wpump)
                plt.plot(tdata3, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata3, g*max(adata3)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata3, w31*result3[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata3, w32*result3[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata3, w33*result3[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata3, w34*result3[:, 3], 'm-', label = 'w4*N4')
                fit_result = w31*result3[:, 0] + w32*result3[:, 1] + w33*result3[:, 2] + w34*result3[:, 3]
                plt.plot(tdata3, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4')
                plt.scatter(tdata3, adata3)
                plt.legend()
                wsum = w31 + w32 + w33 + w34
                w31 = w31 / wsum
                w32 = w32 / wsum
                w33 = w33 / wsum
                w34 = w34 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f' % tuple((A3, t1, t2, t3, t4, w31, w32, w33, w34)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[2])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata4, A4, self.t0, self.Wpump)
                plt.plot(tdata4, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata4, g*max(adata4)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata4, w41*result4[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata4, w42*result4[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata4, w43*result4[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata4, w44*result4[:, 3], 'm-', label = 'w4*N4')
                fit_result = w41*result4[:, 0] + w42*result4[:, 1] + w43*result4[:, 2] + w44*result4[:, 3]
                plt.plot(tdata4, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4')
                plt.scatter(tdata4, adata4)
                plt.legend()
                wsum = w41 + w42 + w43 + w44
                w41 = w41 / wsum
                w42 = w42 / wsum
                w43 = w43 / wsum
                w44 = w44 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f' % tuple((A4, t1, t2, t3, t4, w41, w42, w43, w44)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[3])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                g = G(tdata5, A5, self.t0, self.Wpump)
                plt.plot(tdata5, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata5, g*max(adata5)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata5, w51*result5[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata5, w52*result5[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata5, w53*result5[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata5, w54*result5[:, 3], 'm-', label = 'w4*N4')
                fit_result = w51*result5[:, 0] + w52*result5[:, 1] + w53*result5[:, 2] + w54*result5[:, 3]
                plt.plot(tdata5, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4')
                plt.scatter(tdata5, adata5)
                plt.legend()
                wsum = w51 + w52 + w53 + w54
                w51 = w51 / wsum
                w52 = w52 / wsum
                w53 = w53 / wsum
                w54 = w54 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f' % tuple((A5, t1, t2, t3, t4, w51, w52, w53, w54)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[4])
                plt.ylabel('ΔA (mOD)')
                plt.show()
                
                return A1, A2, A3, A4, A5, t1, t2, t3, t4, w11, w12, w13, w14, w21, w22, w23, w24, w31, w32, w33, w34, w41, w42, w43, w44, w51, w52, w53, w54, result1, result2, result3, result4, result5
    
            elif self.number_of_states == 5:
                def G(t, A, t0, Wpump):
                    return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
                
                def model(y, t, A, t1, t2, t3, t4, t5):
                    N1 = y[0]
                    N2 = y[1]
                    N3 = y[2]
                    N4 = y[3]
                    N5 = y[4]
                    dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                    dN2_dt = N1/t1 - N2/t2
                    dN3_dt = N2/t2 - N3/t3
                    dN4_dt = N3/t3 - N4/t4
                    dN5_dt = N4/t4 - N5/t5
                    return [dN1_dt, dN2_dt, dN3_dt, dN4_dt, dN5_dt]
        
                def y1(t, A1, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15):
                    y0 = [0, 0, 0, 0, 0] # the states are initially unoccupied
                    global result1
                    result1 = integrate.odeint(model, y0, t, args = (A1, t1, t2, t3, t4, t5,))
                    return w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2] + w14*result1[:, 3] + w15*result1[:, 4]
        
                def y2(t, A2, t1, t2, t3, t4, t5, w21, w22, w23, w24, w25):
                    y0 = [0, 0, 0, 0, 0]
                    global result2
                    result2 = integrate.odeint(model, y0, t, args = (A2, t1, t2, t3, t4, t5,))
                    return w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2] + w24*result2[:, 3] + w25*result2[:, 4]
                    
                def y3(t, A3, t1, t2, t3, t4, t5, w31, w32, w33, w34, w35):
                    y0 = [0, 0, 0, 0, 0]
                    global result3
                    result3 = integrate.odeint(model, y0, t, args = (A3, t1, t2, t3, t4, t5,))
                    return w31*result3[:, 0] + w32*result3[:, 1] + w33*result3[:, 2] + w34*result3[:, 3] + w35*result3[:, 4]
                    
                def y4(t, A4, t1, t2, t3, t4, t5, w41, w42, w43, w44, w45):
                    y0 = [0, 0, 0, 0, 0]
                    global result4
                    result4 = integrate.odeint(model, y0, t, args = (A4, t1, t2, t3, t4, t5,))
                    return w41*result4[:, 0] + w42*result4[:, 1] + w43*result4[:, 2] + w44*result4[:, 3] + w45*result4[:, 4]
                    
                def y5(t, A5, t1, t2, t3, t4, t5, w51, w52, w53, w54, w55):
                    y0 = [0, 0, 0, 0, 0]
                    global result5
                    result5 = integrate.odeint(model, y0, t, args = (A5, t1, t2, t3, t4, t5,))
                    return w51*result5[:, 0] + w52*result5[:, 1] + w53*result5[:, 2] + w54*result5[:, 3] + w55*result5[:, 4]
        
                def comboFunc(comboData, A1, A2, A3, A4, A5, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15, w21, w22, w23, w24, w25, w31, w32, w33, w34, w35, w41, w42, w43, w44, w45, w51, w52, w53, w54, w55):
                    result1 = y1(tdata1, A1, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15)
                    result2 = y2(tdata2, A2, t1, t2, t3, t4, t5, w21, w22, w23, w24, w25)
                    result3 = y3(tdata3, A3, t1, t2, t3, t4, t5, w31, w32, w33, w34, w35)
                    result4 = y4(tdata4, A4, t1, t2, t3, t4, t5, w41, w42, w43, w44, w45)
                    result5 = y5(tdata5, A5, t1, t2, t3, t4, t5, w51, w52, w53, w54, w55)
                    return np.append(  np.append(  np.append(  np.append(result1, result2), result3), result4), result5)
                # some initial parameter values
            
                # curve fit the combined data to the combined function
                t_combo = []
                a_combo = []
                for i in dataset:
                    t_combo = np.append(t_combo, i[0])
                    a_combo = np.append(a_combo, i[1])
                    
                data1 = dataset[0]
                data2 = dataset[1]
                data3 = dataset[2]
                data4 = dataset[3]
                data5 = dataset[4]
                
                tdata1, adata1 = data1
                tdata2, adata2 = data2
                tdata3, adata3 = data3
                tdata4, adata4 = data4
                tdata5, adata5 = data5
                
                fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, self.x0, bounds = self.bounds)
                
                A1, A2, A3, A4, A5, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15, w21, w22, w23, w24, w25, w31, w32, w33, w34, w35, w41, w42, w43, w44, w45, w51, w52, w53, w54, w55 = fittedParameters
                A1err, A2err, A3err, A4err, A5err, t1err, t2err, t3err, t4err, t5err, w11err, w12err, w13err, w14err, w15err, w21err, w22err, w23err, w24err, w25err, w31err, w32err, w33err, w34err, w35err, w41err, w42err, w43err, w44err, w45err, w51err, w52err, w53err, w54err, w55err = np.sqrt(np.diag(pcov))
                print('A1err = %.2f, A2err = %.2f, A3err = %.2f, A4err = %.2f, A5err = %.2f, t1err = %.2f, t2err = %.2f, t3err = %.2f, t4err = %.2f, t5err = %.2f, w11err = %.2f, w12err = %.2f, w13err = %.2f, w14err = %.2f, w15err = %.2f, w21err = %.2f, w22err = %.2f, w23err = %.2f, w24err = %.2f w25err = %.2f, w31 = %.2f, w32err = %.2f, w33err = %.2f, w34err = %.2f, w35err = %.2f, w41err = %.2f, w42err = %.2f, w43err = %.2f, w44err = %.2f, w45err = %.2f, w51err = %.2f, w52err = %.2f, w53err = %.2f, w54 = %.2f, w55err = %.2f' % tuple((A1err, A2err, A3err, A4err, A5err, t1err, t2err, t3err, t4err, t5err, w11err, w12err, w13err, w14err, w15err, w21err, w22err, w23err, w24err, w25err, w31err, w32err, w33err, w34err, w35err, w41err, w42err, w43err, w44err, w45err, w51err, w52err, w53err, w54err, w55err)))
        
                plt.figure()
                if self.tshift == None:
                    g = G(tdata1, A1, self.t0, self.Wpump)
                    plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata1, g*max(abs(adata1))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero') 
                else:
                    g = G(tdata1, A1, self.t0 + self.tshift[0], self.Wpump)
                    plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata1, g*max(abs(adata1))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0 + self.tshift[1], color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata1, w11*result1[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata1, w12*result1[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata1, w13*result1[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata1, w14*result1[:, 3], 'm-', label = 'w4*N4')
                plt.plot(tdata1, w15*result1[:, 4], 'c-', label = 'w5*N5')
                fit_result = w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2] + w14*result1[:, 3] + w15*result1[:, 4]
                plt.plot(tdata1, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4 + w5*N5')
                plt.scatter(tdata1, adata1)
                plt.legend()
                wsum = w11 + w12 + w13 + w14 + w15
                w11 = w11 / wsum
                w12 = w12 / wsum
                w13 = w13 / wsum
                w14 = w14 / wsum
                w15 = w15 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f fs, t4 = %.2f, t5 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f, w5 = %.2f' % tuple((A1, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                if self.tshift == None:
                    g = G(tdata2, A2, self.t0, self.Wpump)
                    plt.plot(tdata2, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata2, g*max(abs(adata2))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero') 
                else:
                    g = G(tdata2, A2, self.t0 + self.tshift[1], self.Wpump)
                    plt.plot(tdata2, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata2, g*max(abs(adata2))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0 + self.tshift[1], color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata2, w21*result2[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata2, w22*result2[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata2, w23*result2[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata2, w24*result2[:, 3], 'm-', label = 'w4*N4')
                plt.plot(tdata2, w25*result2[:, 4], 'c-', label = 'w5*N5')
                fit_result = w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2] + w24*result2[:, 3] + w25*result2[:, 4]
                plt.plot(tdata2, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4 + w5*N5')
                plt.scatter(tdata2, adata2)
                plt.legend()
                wsum = w21 + w22 + w23 + w24 + w25
                w21 = w21 / wsum
                w22 = w22 / wsum
                w23 = w23 / wsum
                w24 = w24 / wsum
                w25 = w25 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, t5 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f, w5 = %.2f' % tuple((A2, t1, t2, t3, t4, t5, w21, w22, w23, w24, w25)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[1])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                if self.tshift == None:
                    g = G(tdata3, A3, self.t0, self.Wpump)
                    plt.plot(tdata3, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata3, g*max(abs(adata3))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero') 
                else:
                    g = G(tdata3, A3, self.t0 + self.tshift[2], self.Wpump)
                    plt.plot(tdata3, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata3, g*max(abs(adata3))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0 + self.tshift[2], color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata3, w31*result3[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata3, w32*result3[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata3, w33*result3[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata3, w34*result3[:, 3], 'm-', label = 'w4*N4')
                plt.plot(tdata3, w35*result3[:, 4], 'c-', label = 'w5*N5')
                fit_result = w31*result3[:, 0] + w32*result3[:, 1] + w33*result3[:, 2] + w34*result3[:, 3] + w35*result3[:, 4]
                plt.plot(tdata3, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4 + w5*N5')
                plt.scatter(tdata3, adata3)
                plt.legend()
                wsum = w31 + w32 + w33 + w34 + w35
                w31 = w31 / wsum
                w32 = w32 / wsum
                w33 = w33 / wsum
                w34 = w34 / wsum
                w35 = w35 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, t5 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f, w5 = %.2f' % tuple((A3, t1, t2, t3, t4, t5, w31, w32, w33, w34, w35)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[2])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                if self.tshift == None:
                    g = G(tdata4, A4, self.t0, self.Wpump)
                    plt.plot(tdata4, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata4, g*max(abs(adata4))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero') 
                else:
                    g = G(tdata4, A4, self.t0 + self.tshift[3], self.Wpump)
                    plt.plot(tdata4, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata4, g*max(abs(adata4))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0 + self.tshift[3], color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata4, w41*result4[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata4, w42*result4[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata4, w43*result4[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata4, w44*result4[:, 3], 'm-', label = 'w4*N4')
                plt.plot(tdata4, w45*result4[:, 4], 'c-', label = 'w5*N5')
                fit_result = w41*result4[:, 0] + w42*result4[:, 1] + w43*result4[:, 2] + w44*result4[:, 3] + w45*result4[:, 4]
                plt.plot(tdata4, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4 + w5*N5')
                plt.scatter(tdata4, adata4)
                plt.legend()
                wsum = w41 + w42 + w43 + w44 + w45
                w41 = w41 / wsum
                w42 = w42 / wsum
                w43 = w43 / wsum
                w44 = w44 / wsum
                w45 = w45 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, t5 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f, w5 = %.2f' % tuple((A4, t1, t2, t3, t4, t5, w41, w42, w43, w44, w45)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[3])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                if self.tshift == None:
                    g = G(tdata5, A5, self.t0, self.Wpump)
                    plt.plot(tdata5, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata5, g*max(abs(adata5))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero') 
                else:
                    g = G(tdata5, A5, self.t0 + self.tshift[4], self.Wpump)
                    plt.plot(tdata5, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata5, g*max(abs(adata5))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0 + self.tshift[4], color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata5, w51*result5[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata5, w52*result5[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata5, w53*result5[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata5, w54*result5[:, 3], 'm-', label = 'w4*N4')
                plt.plot(tdata5, w55*result5[:, 4], 'c-', label = 'w5*N5')
                fit_result = w51*result5[:, 0] + w52*result5[:, 1] + w53*result5[:, 2] + w54*result5[:, 3] + w55*result5[:, 4]
                plt.plot(tdata5, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4 + w5*N5')
                plt.scatter(tdata5, adata5)
                plt.legend()
                wsum = w51 + w52 + w53 + w54 + w55
                w51 = w51 / wsum
                w52 = w52 / wsum
                w53 = w53 / wsum
                w54 = w54 / wsum
                w55 = w55 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, t5 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f, w5 = %.2f' % tuple((A5, t1, t2, t3, t4, t5, w51, w52, w53, w54, w55)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[4])
                plt.ylabel('ΔA (mOD)')
                plt.show()
                
                return A1, A2, A3, A4, A5, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15, w21, w22, w23, w24, w25, w31, w32, w33, w34, w35, w41, w42, w43, w44, w45, w51, w52, w53, w54, w55, result1, result2, result3, result4, result5
            
            else:
                def G(t, A, t0, Wpump):
                    return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
                
                def model(y, t, A, t1, t2, t3, t4, t5):
                    N1 = y[0]
                    N2 = y[1]
                    N3 = y[2]
                    N4 = y[3]
                    N5 = y[4]
                    dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                    dN2_dt = N1/t1 - N2/t2
                    dN3_dt = N2/t2 - N3/t3
                    dN4_dt = N3/t3 - N4/t4
                    dN5_dt = N4/t4 - N5/t5
                    return [dN1_dt, dN2_dt, dN3_dt, dN4_dt, dN5_dt]
        
                def y1(t, A1, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15):
                    y0 = [0, 0, 0, 0, 0] # the states are initially unoccupied
                    global result1
                    result1 = integrate.odeint(model, y0, t, args = (A1, t1, t2, t3, t4, t5,))
                    return w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2] + w14*result1[:, 3] + w15*result1[:, 4]
        
                def y2(t, A2, t1, t2, t3, t4, t5, w21, w22, w23, w24, w25):
                    y0 = [0, 0, 0, 0, 0]
                    global result2
                    result2 = integrate.odeint(model, y0, t, args = (A2, t1, t2, t3, t4, t5,))
                    return w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2] + w24*result2[:, 3] + w25*result2[:, 4]
                    
                def y3(t, A3, t1, t2, t3, t4, t5, w31, w32, w33, w34, w35):
                    y0 = [0, 0, 0, 0, 0]
                    global result3
                    result3 = integrate.odeint(model, y0, t, args = (A3, t1, t2, t3, t4, t5,))
                    return w31*result3[:, 0] + w32*result3[:, 1] + w33*result3[:, 2] + w34*result3[:, 3] + w35*result3[:, 4]
                    
                def y4(t, A4, t1, t2, t3, t4, t5, w41, w42, w43, w44, w45):
                    y0 = [0, 0, 0, 0, 0]
                    global result4
                    result4 = integrate.odeint(model, y0, t, args = (A4, t1, t2, t3, t4, t5,))
                    return w41*result4[:, 0] + w42*result4[:, 1] + w43*result4[:, 2] + w44*result4[:, 3] + w45*result4[:, 4]
                    
                def y5(t, A5, t1, t2, t3, t4, t5, w51, w52, w53, w54, w55):
                    y0 = [0, 0, 0, 0, 0]
                    global result5
                    result5 = integrate.odeint(model, y0, t, args = (A5, t1, t2, t3, t4, t5,))
                    return w51*result5[:, 0] + w52*result5[:, 1] + w53*result5[:, 2] + w54*result5[:, 3] + w55*result5[:, 4]
        
                def comboFunc(comboData, A1, A2, A3, A4, A5, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15, w21, w22, w23, w24, w25, w31, w32, w33, w34, w35, w41, w42, w43, w44, w45, w51, w52, w53, w54, w55):
                    result1 = y1(tdata1, A1, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15)
                    result2 = y2(tdata2, A2, t1, t2, t3, t4, t5, w21, w22, w23, w24, w25)
                    result3 = y3(tdata3, A3, t1, t2, t3, t4, t5, w31, w32, w33, w34, w35)
                    result4 = y4(tdata4, A4, t1, t2, t3, t4, t5, w41, w42, w43, w44, w45)
                    result5 = y5(tdata5, A5, t1, t2, t3, t4, t5, w51, w52, w53, w54, w55)
                    return np.append(  np.append(  np.append(  np.append(result1, result2), result3), result4), result5)
                # some initial parameter values
            
                # curve fit the combined data to the combined function
                t_combo = []
                a_combo = []
                for i in dataset:
                    t_combo = np.append(t_combo, i[0])
                    a_combo = np.append(a_combo, i[1])
                    
                data1 = dataset[0]
                data2 = dataset[1]
                data3 = dataset[2]
                data4 = dataset[3]
                data5 = dataset[4]
                
                tdata1, adata1 = data1
                tdata2, adata2 = data2
                tdata3, adata3 = data3
                tdata4, adata4 = data4
                tdata5, adata5 = data5
                
                fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, self.x0, bounds = self.bounds)
                
                A1, A2, A3, A4, A5, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15, w21, w22, w23, w24, w25, w31, w32, w33, w34, w35, w41, w42, w43, w44, w45, w51, w52, w53, w54, w55 = fittedParameters
                A1err, A2err, A3err, A4err, A5err, t1err, t2err, t3err, t4err, t5err, w11err, w12err, w13err, w14err, w15err, w21err, w22err, w23err, w24err, w25err, w31err, w32err, w33err, w34err, w35err, w41err, w42err, w43err, w44err, w45err, w51err, w52err, w53err, w54err, w55err = np.sqrt(np.diag(pcov))
                # print('A1err = %.2f, A2err = %.2f, A3err = %.2f, A4err = %.2f, A5err = %.2f, t1err = %.2f, t2err = %.2f, t3err = %.2f, t4err = %.2f, t5err = %.2f, w11err = %.2f, w12err = %.2f, w13err = %.2f, w14err = %.2f, w15err = %.2f, w21err = %.2f, w22err = %.2f, w23err = %.2f, w24err = %.2f w25err = %.2f, w31, w32err = %.2f, w33err = %.2f, w34err = %.2f, w35err = %.2f, w41err = %.2f, w42err = %.2f, w43err = %.2f, w44err = %.2f, w45err = %.2f, w51err = %.2f, w52err = %.2f, w53err = %.2f, w54 = %.2f, w55err = %.2f' % tuple((A1err, A2err, A3err, A4err, A5err, t1err, t2err, t3err, t4err, t5err, w11err, w12err, w13err, w14err, w15err, w21err, w22err, w23err, w24err, w25err, w31err, w32err, w33err, w34err, w35err, w41err, w42err, w43err, w44err, w45err, w51err, w52err, w53err, w54err, w55err)))
        
                plt.figure()
                if self.tshift == None:
                    g = G(tdata1, A1, self.t0, self.Wpump)
                    plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata1, g*max(abs(adata1))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero') 
                else:
                    g = G(tdata1, A1, self.t0 + self.tshift[0], self.Wpump)
                    plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata1, g*max(abs(adata1))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0 + self.tshift[1], color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata1, w11*result1[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata1, w12*result1[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata1, w13*result1[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata1, w14*result1[:, 3], 'm-', label = 'w4*N4')
                plt.plot(tdata1, w15*result1[:, 4], 'c-', label = 'w5*N5')
                fit_result = w11*result1[:, 0] + w12*result1[:, 1] + w13*result1[:, 2] + w14*result1[:, 3] + w15*result1[:, 4]
                plt.plot(tdata1, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4 + w5*N5')
                plt.scatter(tdata1, adata1)
                plt.legend()
                wsum = w11 + w12 + w13 + w14 + w15
                w11 = w11 / wsum
                w12 = w12 / wsum
                w13 = w13 / wsum
                w14 = w14 / wsum
                w15 = w15 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f fs, t4 = %.2f, t5 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f, w5 = %.2f' % tuple((A1, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[0])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                if self.tshift == None:
                    g = G(tdata2, A2, self.t0, self.Wpump)
                    plt.plot(tdata2, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata2, g*max(abs(adata2))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero') 
                else:
                    g = G(tdata2, A2, self.t0 + self.tshift[1], self.Wpump)
                    plt.plot(tdata2, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata2, g*max(abs(adata2))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0 + self.tshift[1], color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata2, w21*result2[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata2, w22*result2[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata2, w23*result2[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata2, w24*result2[:, 3], 'm-', label = 'w4*N4')
                plt.plot(tdata2, w25*result2[:, 4], 'c-', label = 'w5*N5')
                fit_result = w21*result2[:, 0] + w22*result2[:, 1] + w23*result2[:, 2] + w24*result2[:, 3] + w25*result2[:, 4]
                plt.plot(tdata2, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4 + w5*N5')
                plt.scatter(tdata2, adata2)
                plt.legend()
                wsum = w21 + w22 + w23 + w24 + w25
                w21 = w21 / wsum
                w22 = w22 / wsum
                w23 = w23 / wsum
                w24 = w24 / wsum
                w25 = w25 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, t5 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f, w5 = %.2f' % tuple((A2, t1, t2, t3, t4, t5, w21, w22, w23, w24, w25)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[1])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                if self.tshift == None:
                    g = G(tdata3, A3, self.t0, self.Wpump)
                    plt.plot(tdata3, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata3, g*max(abs(adata3))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero') 
                else:
                    g = G(tdata3, A3, self.t0 + self.tshift[2], self.Wpump)
                    plt.plot(tdata3, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata3, g*max(abs(adata3))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0 + self.tshift[2], color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata3, w31*result3[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata3, w32*result3[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata3, w33*result3[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata3, w34*result3[:, 3], 'm-', label = 'w4*N4')
                plt.plot(tdata3, w35*result3[:, 4], 'c-', label = 'w5*N5')
                fit_result = w31*result3[:, 0] + w32*result3[:, 1] + w33*result3[:, 2] + w34*result3[:, 3] + w35*result3[:, 4]
                plt.plot(tdata3, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4 + w5*N5')
                plt.scatter(tdata3, adata3)
                plt.legend()
                wsum = w31, w32, w33, w34, w35
                w31 = w31 / wsum
                w32 = w32 / wsum
                w33 = w33 / wsum
                w34 = w34 / wsum
                w35 = w35 / wsum
                # plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, t5 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f, w5 = %.2f' % tuple((A3, t1, t2, t3, t4, t5, w31, w32, w33, w34, w35)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[2])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                if self.tshift == None:
                    g = G(tdata4, A4, self.t0, self.Wpump)
                    plt.plot(tdata4, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata4, g*max(abs(adata4))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero') 
                else:
                    g = G(tdata4, A4, self.t0 + self.tshift[3], self.Wpump)
                    plt.plot(tdata4, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata4, g*max(abs(adata4))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0 + self.tshift[3], color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata4, w41*result4[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata4, w42*result4[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata4, w43*result4[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata4, w44*result4[:, 3], 'm-', label = 'w4*N4')
                plt.plot(tdata4, w45*result4[:, 4], 'c-', label = 'w5*N5')
                fit_result = w41*result4[:, 0] + w42*result4[:, 1] + w43*result4[:, 2] + w44*result4[:, 3] + w45*result4[:, 4]
                plt.plot(tdata4, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4 + w5*N5')
                plt.scatter(tdata4, adata4)
                plt.legend()
                wsum = w41, w42, w43, w44, w45
                w41 = w41 / wsum
                w42 = w42 / wsum
                w43 = w43 / wsum
                w44 = w44 / wsum
                w45 = w45 / wsum
                # plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, t5 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f, w5 = %.2f' % tuple((A4, t1, t2, t3, t4, t5, w41, w42, w43, w44, w45)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[3])
                plt.ylabel('ΔA (mOD)')
                
                plt.figure()
                if self.tshift == None:
                    g = G(tdata5, A5, self.t0, self.Wpump)
                    plt.plot(tdata5, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata5, g*max(abs(adata5))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero') 
                else:
                    g = G(tdata5, A5, self.t0 + self.tshift[4], self.Wpump)
                    plt.plot(tdata5, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                    plt.plot(tdata5, g*max(abs(adata5))/max(abs(g)), color = '#b676f2', label = 'Vpump scaled')
                    plt.axvline(x = self.t0 + self.tshift[4], color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata5, w51*result5[:, 0], 'r-', label = 'w1*N1')
                plt.plot(tdata5, w52*result5[:, 1], 'b-', label = 'w2*N2')
                plt.plot(tdata5, w53*result5[:, 2], 'g-', label = 'w3*N3')
                plt.plot(tdata5, w54*result5[:, 3], 'm-', label = 'w4*N4')
                plt.plot(tdata5, w55*result5[:, 4], 'c-', label = 'w5*N5')
                fit_result = w51*result5[:, 0] + w52*result5[:, 1] + w53*result5[:, 2] + w54*result5[:, 3] + w55*result5[:, 4]
                plt.plot(tdata5, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4 + w5*N5')
                plt.scatter(tdata5, adata5)
                plt.legend()
                wsum = w51, w52, w53, w54, w55
                w51 = w51 / wsum
                w52 = w52 / wsum
                w53 = w53 / wsum
                w54 = w54 / wsum
                w55 = w55 / wsum
                # plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, t5 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f, w5 = %.2f' % tuple((A5, t1, t2, t3, t4, t5, w51, w52, w53, w54, w55)))
                plt.xlabel('%.1f nm, Delay (fs)' % self.wavelengths[4])
                plt.ylabel('ΔA (mOD)')
                plt.show()
                
                return A1, A2, A3, A4, A5, t1, t2, t3, t4, t5, w11, w12, w13, w14, w15, w21, w22, w23, w24, w25, w31, w32, w33, w34, w35, w41, w42, w43, w44, w45, w51, w52, w53, w54, w55, result1, result2, result3, result4, result5
    
        else:
            print('Globally fitting more than five wavelengths is not currently supported. Please try again with 1-5 wavelengths.')
            
            
#%% Function that is not ready yet
"""
class Data:
    def __init__(self, file, wavelengths, tstart, tstop):
        self.file = file # string: file = r'filename'
        self.wavelengths = wavelengths # single value or a vector: wavelengths = 700, or wavelengths = [700, 570]
        self.tstart = tstart
        self.tstop = tstop
        
    def initialize(self):
        # import data from external file
        if '.dat' in self.file:
            data = np.array(pd.read_csv(self.file, header = None, delim_whitespace = True))
            t = data[:, 0]
            a = 1000 * data[:, 1] # convert to mOD
        if '.csv' in self.file:
            data = np.array(pd.read_csv(self.file, header = None, delim_whitespace = False))
            t = data[1:, 0]
            w = data[0, 1:]
            a = 1000 * data[1:, 1:] # convert to mOD
        else:
            print('You need to use a file that is either .dat or .csv.')
        

        if len(self.wavelengths) == 1:
            for i in self.wavelengths:
                # select trace that matches desired wavelength
                wavelength_idx = (np.abs(w - i)).argmin()
                anew = a[:, wavelength_idx]
                # truncate time and absorbtion vectors to tstart --> tstop
                tstart_idx = (np.abs(t - self.tstart)).argmin()
                tstop_idx = (np.abs(t - self.tstop)).argmin() + 1
                tnew = t[tstart_idx:tstop_idx]
                anew = anew[tstart_idx:tstop_idx]
                anew = anew / max(abs(anew)) # normalize the absorption vector, use abs(a) in case the signal is negative
                trace = np.vstack((tnew, anew))
            return trace
        else:
            dataset = []
            for i in self.wavelengths:
                # select trace that matches desired wavelength
                wavelength_idx = (np.abs(w - i)).argmin()
                anew = a[:, wavelength_idx]
                
                # truncate time and absorbtion vectors to tstart --> tstop
                tstart_idx = (np.abs(t - self.tstart)).argmin()
                tstop_idx = (np.abs(t - self.tstop)).argmin() + 1
                tnew = t[tstart_idx:tstop_idx]
                anew = anew[tstart_idx:tstop_idx]
                anew = anew / max(abs(anew)) # normalize the absorption vector, use abs(a) in case the signal is negative
                trace = np.vstack((tnew, anew))
                # dataset.append(trace)
                if i == self.wavelengths[0]:
                    dataset = trace
                else:
                    dataset = np.column_stack((dataset, trace))
            return dataset
        
class GlobalFit(Data):
    def __init__(self, file, wavelengths, tstart, tstop, t0, Wpump, number_of_states, x0, bounds):
        # x0 is initial guess for [t1, t2,...tN, w11, w21...wN1, w12...w2N, w1N...wNN]
        # bounds = ([lower bounds of same order as x0], [upper bounds of same order as x0])
        # Note that the order of weights for x0 and bounds are different than for Fit.TA_global(),
        #   since all w1's come first, rather than all weights of self.wavelengths[0] coming first.
        self.file = file
        self.wavelengths = wavelengths
        self.tstart = tstart
        self.tstop = tstop
        self.t0 = t0
        self.Wpump = Wpump
        self.number_of_states = number_of_states
        self.x0 = x0
        self.bounds = bounds

    def TA_globalX(self):
        # goal: we need to globally fit any number of wavelengths, for any of 1-5 states
        global dataset
        dataset = Data(self.file, self.wavelengths, self.tstart, self.tstop).initialize()

        if self.number_of_states == 1:
            def G(t, A, t0, Wpump):
                return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
         
            def model(y, t, A, t1):
               N1 = y[0]
               dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
               return [dN1_dt]
        
            def y(t, A, t1):
                y0 = [0]
                global result
                result = integrate.odeint(model, y0, t, args = (A, t1,))
                return result[:, 0]
    
            def comboFunc(comboData, *x):
                t1 = x[-1]
                A = x[:-1]
                global result_list
                result_list = []
                for i in range(len(self.wavelengths)):
                    result_list = np.append(result_list, y(tdata1, A[i], t1))
                # result1 = y1(tdata1, A1, t1)
                # result2 = y2(tdata2, A2, t1)
                print('hi', np.shape(result_list))
                
                return result_list
         
            # t_combo = []
            # a_combo = []
            # for i in range(len(self.wavelengths)):
            #     # loop over all wavelengths in wavelength list
            #     t_combo = np.append(t_combo, dataset[0, :])
            #     a_combo = np.append(a_combo, dataset[i, :, 1])
            t_combo = dataset[0, :]
            a_combo = dataset[1, :]
            # since the same time axis is used for all wavelengths, we can just use tdata1
            tdata1 = t_combo[: int(len(t_combo)/len(self.wavelengths))]
            print(len(tdata1))
            s = len(self.wavelengths)
            A0 = [1] * s
            A_low = [-1] * s
            A_high = [1] * s
            X0 = []
            # X0 is the A0 list made above and the inputed x0 guess for the Fit class
            for i in A0:
                X0.append(i)
            for i in self.x0:
                X0.append(i)
            low_bounds = []
            high_bounds = []
            for i in A_low:
                low_bounds.append(i)
            for i in self.bounds[0]:
                low_bounds.append(i)
            for i in A_high:
                high_bounds.append(i)
            for i in self.bounds[1]:
                high_bounds.append(i)
            Bounds = (low_bounds, high_bounds)
            print('yo', np.shape(a_combo))
            fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, X0, bounds = Bounds)
            perr = np.sqrt(np.diag(pcov))
            
            A = fittedParameters[:-1]
            t1 = fittedParameters[-1]
            Aerr = perr[:-1]
            t1 = perr[-1]
            for i in range(len(Aerr)):
                print('%.2f nm' % self.wavelengths[i], 'Aerr = %.2f' % Aerr[i])
            print('t1err = %.2f fs' % t1)
    
            for i in range(s):
                adata = a_combo[i*len(tdata1) : (i+1)*len(tdata1)]              
                plt.figure()
                g = G(tdata1, A[i], self.t0, self.Wpump)
                plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata1, g*max(adata)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                fit_result = result[:,0]
                plt.plot(tdata1, result_list[i*len(tdata1): (i+1)*len(tdata1)], 'k--', label = 'N1')
                plt.scatter(tdata1, adata)
                plt.legend()
                plt.title('A = %.2f, t1 = %.2f fs' % tuple((A[i], t1)))
                plt.xlabel('%.2f nm, Delay (fs)' % self.wavelengths[i])
                plt.ylabel('ΔA (mOD)')
                # plt.show()
        
            return A, t1, result_list 
            
        elif self.number_of_states == 2:
            def G(t, A, t0, Wpump):
                return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)

            def model(y, t, A, t1, t2):
                N1 = y[0]
                N2 = y[1]
                dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                dN2_dt = N1/t1 - N2/t2
                return [dN1_dt, dN2_dt]
            
            def y(t, A, t1, t2, w11, w12):
                y0 = [0, 0]
                global result
                result = integrate.odeint(model, y0, t, args = (A, t1, t2,))
                return w11*result[:, 0], w12*result[:, 1]
    
            def comboFunc(comboData, *x):
                global state1_list, state2_list, result_list
                state1_list = []
                state2_list = []
                result_list = []
                
                ss = len(self.wavelengths)
                A = x[:ss]
                t1 = x[ss]
                t2 = x[ss+1]
                w1 = x[ss+2 : 2*ss+2]
                w2 = x[2*ss+2:]
                
                for i in range(ss):
                    y_result = y(tdata1, A[i], t1, t2, w1[i], w2[i])
                    print(np.shape(y_result))
                    state1_list.append(y_result[0])
                    state2_list.append(y_result[1])
                    result_list = np.append(result_list, y_result[0]+y_result[1])
                return result_list
            # some initial parameter values
        
            # curve fit the combined data to the combined function
            t_combo = []
            a_combo = []
            for i in dataset:
                t_combo = np.append(t_combo, i[0])
                a_combo = np.append(a_combo, i[1])
            
            tdata1 = dataset[0][0]
            # s = len(self.wavelengths)
            ss = len(self.wavelengths)
            A0 = [1] * ss
            A_low = [-1] * ss
            A_high = [1] * ss
            X0 = []
            # X0 is the A0 list made above and the inputed x0 guess for the Fit class
            for i in A0:
                X0.append(i)
            for i in self.x0:
                X0.append(i)
            low_bounds = []
            high_bounds = []
            for i in A_low:
                low_bounds.append(i)
            for i in self.bounds[0]:
                low_bounds.append(i)
            for i in A_high:
                high_bounds.append(i)
            for i in self.bounds[1]:
                high_bounds.append(i)
            Bounds = (low_bounds, high_bounds)
            fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, X0, bounds = Bounds)
            perr = np.sqrt(np.diag(pcov))
            
            ss = len(self.wavelengths)
            A = fittedParameters[:ss]
            t1 = fittedParameters[ss]
            t2 = fittedParameters[ss+1]
            w1 = fittedParameters[ss+2 : 2*ss+2]
            w2 = fittedParameters[2*ss+2:]
            
            Aerr = perr[:ss]
            t1err = perr[ss]
            t2err = perr[ss+1]
            w1err = perr[ss+2 : 2*ss+2]
            w2err = perr[2*ss+2:]
            
            for i in range(len(Aerr)):
                print('%.2f nm' % self.wavelengths[i], 'Aerr = %.2f' % Aerr[i])

            print('t1err = %.2f fs' % t1)
            print('t2err = %.2f fs' % t2)
            
            for i in range(len(Aerr)):
                print('w1err = %.2f' % w1err[i])
                print('w2err = %.2f' % w2err[i])
            
            for i in range(len(self.wavelengths)):
                adata = a_combo[i*len(tdata1) : (i+1)*len(tdata1)]
                
                plt.figure()
                g = G(tdata1, A[i], self.t0, self.Wpump)
                plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata1, g*max(adata)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata1, state1_list[i], 'r-', label = 'w1*N1')
                plt.plot(tdata1, state2_list[i], 'b-', label = 'w2*N2')
                fit_result = result_list[i*len(tdata1) : (i+1)*len(tdata1)]
                plt.plot(tdata1, fit_result, 'k--', label = 'w1*N1 + w2*N2')
                plt.scatter(tdata1, adata)
                plt.legend()
                wsum = w1[i] + w2[i]
                w1[i] = w1[i] / wsum
                w2[i] = w2[i] / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, w1 = %.2f, w2 = %.2f' % tuple((A[i], t1, t2, w1[i], w2[i])))
                plt.xlabel('%.2f nm, Delay (fs)' % self.wavelengths[i])
                plt.ylabel('ΔA (mOD)')
                # plt.show()
            
            return A, t1, t2, w1, w2, result_list, state1_list, state2_list
 
        elif self.number_of_states == 3:
            def G(t, A, t0, Wpump):
                return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
            
            def model(y, t, A, t1, t2, t3):
                N1 = y[0]
                N2 = y[1]
                N3 = y[2]
                dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                dN2_dt = N1/t1 - N2/t2
                dN3_dt = N2/t2 - N3/t3
                return [dN1_dt, dN2_dt, dN3_dt]
    
            def y(t, A, t1, t2, t3, w1, w2, w3):
                y0 = [0, 0, 0] # the states are initially unoccupied
                global result
                result = integrate.odeint(model, y0, t, args = (A, t1, t2, t3,))
                return w1*result[:, 0], w2*result[:, 1], w3*result[:, 2]
                
            def comboFunc(comboData, *x):
                global state1_list, state2_list, state3_list, result_list
                state1_list = []
                state2_list = []
                state3_list = []
                result_list = []
                
                s = len(self.wavelengths)
                A = x[:s]
                t1 = x[s+1]
                t2 = x[s+2]
                t3 = x[s+3]
                w1 = x[s+4 : 2*s+4]
                w2 = x[2*s+5 : 3*s+5]
                w3 = x[3*s+6:]
                
                for i in range(s):
                    result = y(tdata1, A[i], t1, t2, t3, w1[i], w2[i], w3[i])
                    state1_list.append(result[0])
                    state2_list.append(result[1])
                    state3_list.append(result[2])
                    result_list.append(result[0] + result[1] + result[2])
                return result_list, state1_list, state2_list, state3_list
            # some initial parameter values
        
            # curve fit the combined data to the combined function
            t_combo = []
            a_combo = []
            for i in dataset:
                t_combo = np.append(t_combo, i[0])
                a_combo = np.append(a_combo, i[1])
            
            tdata1 = dataset[0][0]
            s = len(self.wavelengths)
            A0 = [1] * s
            X0 = [A0, self.x0]
            fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, X0, bounds = self.bounds)
            perr = np.sqrt(np.diag(pcov))
            
            A = fittedParameters[:s]
            t1 = fittedParameters[s+1]
            t2 = fittedParameters[s+2]
            t3 = fittedParameters[s+3]
            w1 = fittedParameters[s+4 : 2*s+4]
            w2 = fittedParameters[2*s+5 : 3*s+5]
            w3 = fittedParameters[3*s+6:]
            
            Aerr = perr[:s]
            t1err = perr[s+1]
            t2err = perr[s+2]
            t3err = perr[s+3]
            w1err = perr[s+4 : 2*s+4]
            w2err = perr[2*s+5 : 3*s+5]
            w3err = perr[3*s+6:]

            for i in range(len(Aerr)):
                print('%.2f nm' % self.wavelengths[i], 'Aerr = %.2f' % Aerr[i])
           
            print('t1err = %.2f fs' % t1)
            print('t2err = %.2f fs' % t2)
            print('t3err = %.2f fs' % t3)
            
            for i in range(len(Aerr)):
                print('w1err = %.2f' % w1err[i])
                print('w2err = %.2f' % w2err[i])
                print('w3err = %.2f' % w3err[i])
                
            for i in range(len(s)):
                adata = a_combo[i*s : (i+1)*s]
    
                plt.figure()
                g = G(tdata1, A[i], self.t0, self.Wpump)
                plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata1, g*max(adata)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata1, state1_list[i], 'r-', label = 'w1*N1')
                plt.plot(tdata1, state2_list[i], 'b-', label = 'w2*N2')
                plt.plot(tdata1, state3_list[i], 'g-', label = 'w3*N3')
                fit_result = result_list[i]
                plt.plot(tdata1, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3')
                plt.scatter(tdata1, adata)
                plt.legend()
                wsum = w1[i] + w2[i] + w3[i]
                w1[i] = w1 / wsum
                w2[i] = w2 / wsum
                w3[i] = w3 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f' % tuple((A[i], t1, t2, t3, w1[i], w2[i], w3[i])))
                plt.xlabel('%.2f nm, Delay (fs)' % self.wavelengths[i])
                plt.ylabel('ΔA (mOD)')
                # plt.show()
            
            return A, t1, t2, t3, w1, w2, w3, result_list, state1_list, state2_list, state3_list
 
        elif self.number_of_states == 4:
            def G(t, A, t0, Wpump):
                return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
            
            def model(y, t, A, t1, t2, t3, t4):
                N1 = y[0]
                N2 = y[1]
                N3 = y[2]
                N4 = y[3]
                dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                dN2_dt = N1/t1 - N2/t2
                dN3_dt = N2/t2 - N3/t3
                dN4_dt = N3/t3 - N4/t4
                return [dN1_dt, dN2_dt, dN3_dt, dN4_dt]
    
            def y(t, A1, t1, t2, t3, t4, w1, w2, w3, w4):
                y0 = [0, 0, 0, 0] # the states are initially unoccupied
                global result
                result = integrate.odeint(model, y0, t, args = (A, t1, t2, t3, t4,))
                return w1*result[:, 0], w2*result[:, 1], w3*result[:, 2], w4*result[:, 3]
    
            def comboFunc(comboData, *x):
                global state1_list, state2_list, state3_list, state4_list, result_list
                state1_list = []
                state2_list = []
                state3_list = []
                state4_list = []
                result_list = []
                
                s = len(self.wavelengths)
                A = x[:s]
                t1 = x[s+1]
                t2 = x[s+2]
                t3 = x[s+3]
                t4 = x[s+4]
                w1 = x[s+5 : 2*s+5]
                w2 = x[2*s+5 : 3*s+5]
                w3 = x[3*s+6 : 4*s+6]
                w4 = x[4*s+7 : 5*s+7]
                
                for i in range(s):
                    result = y(tdata1, A[i], t1, t2, t3, t4, w1[i], w2[i], w3[i], w4[i])
                    state1_list.append(result[0])
                    state2_list.append(result[1])
                    state3_list.append(result[2])
                    state4_list.append(result[3])
                    result_list.append(result[0] + result[1] + result[2] + result[3])
                return result_list, state1_list, state2_list, state3_list, state4_list
                
            # some initial parameter values
        
            # curve fit the combined data to the combined function
            t_combo = []
            a_combo = []
            for i in dataset:
                t_combo = np.append(t_combo, i[0])
                a_combo = np.append(a_combo, i[1])
                
            tdata1 = dataset[0][0]
            s = len(self.wavelengths)
            A0 = [1] * s
            X0 = [A0, self.x0]
            fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, X0, bounds = self.bounds)
            perr = np.sqrt(np.diag(pcov))
            
            A = fittedParameters[:s]
            t1 = fittedParameters[s+1]
            t2 = fittedParameters[s+2]
            t3 = fittedParameters[s+3]
            t4 = fittedParameters[s+4]
            w1 = fittedParameters[s+5 : 2*s+5]
            w2 = fittedParameters[2*s+6 : 3*s+6]
            w3 = fittedParameters[3*s+7 : 4*s+7]
            w4 = fittedParameters[4*s+8:]
            
            Aerr = perr[:s]
            t1err = perr[s+1]
            t2err = perr[s+2]
            t3err = perr[s+3]
            t4err = perr[s+4]
            w1err = perr[s+5 : 2*s+5]
            w2err = perr[2*s+6 : 3*s+6]
            w3err = perr[3*s+7 : 4*s+7]
            w4err = perr[4*s+8:]
            
            for i in range(len(Aerr)):
                print('%.2f nm' % self.wavelengths[i], 'Aerr = %.2f' % Aerr[i])
           
            print('t1err = %.2f fs' % t1)
            print('t2err = %.2f fs' % t2)
            print('t3err = %.2f fs' % t3)
            print('t4err = %.2f fs' % t4)
            
            for i in range(len(Aerr)):
                print('w1err = %.2f' % w1err[i])
                print('w2err = %.2f' % w2err[i])
                print('w3err = %.2f' % w3err[i])
                print('w4err = %.2f' % w4err[i])
            
            for i in range(len(s)):
                adata = a_combo[i*s : (i+1)*s]
    
                plt.figure()
                g = G(tdata1, A[i], self.t0, self.Wpump)
                plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata1, g*max(adata)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata1, state1_list[i], 'r-', label = 'w1*N1')
                plt.plot(tdata1, state2_list[i], 'b-', label = 'w2*N2')
                plt.plot(tdata1, state3_list[i], 'g-', label = 'w3*N3')
                plt.plot(tdata1, state4_list[i], 'm-', label = 'w4*N4')
                fit_result = result_list[i]
                plt.plot(tdata1, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4')
                plt.scatter(tdata1, adata)
                plt.legend()
                wsum = w1[i] + w2[i] + w3[i] + w4[i]
                w1[i] = w1 / wsum
                w2[i] = w2 / wsum
                w3[i] = w3 / wsum
                w4[i] = w4 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f' % tuple((A[i], t1, t2, t3, t4, w1[i], w2[i], w3[i], w4[i])))
                plt.xlabel('%.2f nm, Delay (fs)' % self.wavelengths[i])
                plt.ylabel('ΔA (mOD)')
                # plt.show()

            return A, t1, t2, t3, t4, w1, w2, w3, w4, result_list, state1_list, state2_list, state3_list, state4_list

        elif self.number_of_states == 5:
            def G(t, A, t0, Wpump):
                return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
            
            def model(y, t, A, t1, t2, t3, t4, t5):
                N1 = y[0]
                N2 = y[1]
                N3 = y[2]
                N4 = y[3]
                N5 = y[4]
                dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                dN2_dt = N1/t1 - N2/t2
                dN3_dt = N2/t2 - N3/t3
                dN4_dt = N3/t3 - N4/t4
                dN5_dt = N4/t4 - N5/t5
                return [dN1_dt, dN2_dt, dN3_dt, dN4_dt, dN5_dt]
    
            def y(t, A, t1, t2, t3, t4, t5, w1, w2, w3, w4, w5):
                y0 = [0, 0, 0, 0, 0] # the states are initially unoccupied
                global result
                result = integrate.odeint(model, y0, t, args = (A, t1, t2, t3, t4, t5,))
                return w1*result[:, 0], w2*result[:, 1], w3*result[:, 2], w4*result[:, 3], w5*result[:, 4]
                
            def comboFunc(comboData, *x):
                global state1_list, state2_list, state3_list, state4_list, state5_list, result_list
                state1_list = []
                state2_list = []
                state3_list = []
                state4_list = []
                state5_list = []
                result_list = []
                
                s = len(self.wavelengths)
                A = x[:s]
                t1 = x[s+1]
                t2 = x[s+2]
                t3 = x[s+3]
                t4 = x[s+4]
                t5 = x[s+5]
                w1 = x[s+6 : 2*s+6]
                w2 = x[2*s+7 : 3*s+7]
                w3 = x[3*s+8 : 4*s+8]
                w4 = x[4*s+9 : 5*s+9]
                w5 = x[5*s+10:]
                
                for i in range(s):
                    result = y(tdata1, A[i], t1, t2, t3, t4, w1[i], w2[i], w3[i], w4[i], w5[i])
                    state1_list.append(result[0])
                    state2_list.append(result[1])
                    state3_list.append(result[2])
                    state4_list.append(result[3])
                    state5_list.append(result[4])
                    result_list.append(result[0] + result[1] + result[2] + result[3] + result[4])
                return result_list, state1_list, state2_list, state3_list, state4_list

            # some initial parameter values
        
            # curve fit the combined data to the combined function
            t_combo = []
            a_combo = []
            for i in dataset:
                t_combo = np.append(t_combo, i[0])
                a_combo = np.append(a_combo, i[1])
            
            tdata1 = dataset[0][0]
            s = len(self.wavelengths)
            A0 = [1] * s
            X0 = [A0, self.x0]
            fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, X0, bounds = self.bounds)
            perr = np.sqrt(np.diag(pcov))
            
            A = fittedParameters[:s]
            t1 = fittedParameters[s+1]
            t2 = fittedParameters[s+2]
            t3 = fittedParameters[s+3]
            t4 = fittedParameters[s+4]
            t5 = fittedParameters[s+5]
            w1 = fittedParameters[s+6 : 2*s+6]
            w2 = fittedParameters[2*s+7 : 3*s+7]
            w3 = fittedParameters[3*s+8 : 4*s+8]
            w4 = fittedParameters[4*s+9 : 5*s+9]
            w5 = fittedParameters[5*s+10]
            
            Aerr = perr[:s]
            t1err = perr[s+1]
            t2err = perr[s+2]
            t3err = perr[s+3]
            t4err = perr[s+4]
            t5err = perr[s+5]
            w1err = perr[s+6 : 2*s+6]
            w2err = perr[2*s+7 : 3*s+7]
            w3err = perr[3*s+8 : 4*s+8]
            w4err = perr[4*s+9 : 5*s+9]
            w5err = perr[5*s+10:]
    
            for i in range(len(Aerr)):
                print('%.2f nm' % self.wavelengths[i], 'Aerr = %.2f' % Aerr[i])
           
            print('t1err = %.2f fs' % t1)
            print('t2err = %.2f fs' % t2)
            print('t3err = %.2f fs' % t3)
            print('t4err = %.2f fs' % t4)
            print('t5err = %.2f fs' % t5)
            
            for i in range(len(Aerr)):
                print('w1err = %.2f' % w1err[i])
                print('w2err = %.2f' % w2err[i])
                print('w3err = %.2f' % w3err[i])
                print('w4err = %.2f' % w4err[i])
                print('w5err = %.2f' % w5err[i])
            
            for i in range(len(s)):
                adata = a_combo[i*s : (i+1)*s]
    
                plt.figure()
                g = G(tdata1, A[i], self.t0, self.Wpump)
                plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata1, g*max(adata)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata1, state1_list[i], 'r-', label = 'w1*N1')
                plt.plot(tdata1, state2_list[i], 'b-', label = 'w2*N2')
                plt.plot(tdata1, state3_list[i], 'g-', label = 'w3*N3')
                plt.plot(tdata1, state4_list[i], 'm-', label = 'w4*N4')
                plt.plot(tdata1, state5_list[i], 'c-', label = 'w5*N5')
                fit_result = result_list[i]
                plt.plot(tdata1, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4 + w5*N5')
                plt.scatter(tdata1, adata)
                plt.legend()
                wsum = w1[i] + w2[i] + w3[i] + w4[i] + w5[i]
                w1[i] = w1 / wsum
                w2[i] = w2 / wsum
                w3[i] = w3 / wsum
                w4[i] = w4 / wsum
                w5[i] = w5 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, t5 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f, w5 = %.2f' % tuple((A[i], t1, t2, t3, t4, t5, w1[i], w2[i], w3[i], w4[i], w5[i])))
                plt.xlabel('%.2f nm, Delay (fs)' % self.wavelengths[i])
                plt.ylabel('ΔA (mOD)')
                # plt.show()

            return A, t1, t2, t3, t4, t5, w1, w2, w3, w4, w5, result_list, state1_list, state2_list, state3_list, state4_list, state5_list
        
        else:
            print('Currently only 1-5 states are supported for global fitting. 5 states will be fitted by default instead.')
            def G(t, A, t0, Wpump):
                return A * np.exp(-4*np.log(2) * ((t - t0)/Wpump)**2)
            
            def model(y, t, A, t1, t2, t3, t4, t5):
                N1 = y[0]
                N2 = y[1]
                N3 = y[2]
                N4 = y[3]
                N5 = y[4]
                dN1_dt = G(t, A, self.t0, self.Wpump) - N1/t1
                dN2_dt = N1/t1 - N2/t2
                dN3_dt = N2/t2 - N3/t3
                dN4_dt = N3/t3 - N4/t4
                dN5_dt = N4/t4 - N5/t5
                return [dN1_dt, dN2_dt, dN3_dt, dN4_dt, dN5_dt]
    
            def y(t, A, t1, t2, t3, t4, t5, w1, w2, w3, w4, w5):
                y0 = [0, 0, 0, 0, 0] # the states are initially unoccupied
                global result
                result = integrate.odeint(model, y0, t, args = (A, t1, t2, t3, t4, t5,))
                return w1*result[:, 0], w2*result[:, 1], w3*result[:, 2], w4*result[:, 3], w5*result[:, 4]
                
            def comboFunc(comboData, *x):
                global state1_list, state2_list, state3_list, state4_list, state5_list, result_list
                state1_list = []
                state2_list = []
                state3_list = []
                state4_list = []
                state5_list = []
                result_list = []
                
                s = len(self.wavelengths)
                A = x[:s]
                t1 = x[s+1]
                t2 = x[s+2]
                t3 = x[s+3]
                t4 = x[s+4]
                t5 = x[s+5]
                w1 = x[s+6 : 2*s+6]
                w2 = x[2*s+7 : 3*s+7]
                w3 = x[3*s+8 : 4*s+8]
                w4 = x[4*s+9 : 5*s+9]
                w5 = x[5*s+10:]
                
                for i in range(s):
                    result = y(tdata1, A[i], t1, t2, t3, t4, w1[i], w2[i], w3[i], w4[i], w5[i])
                    state1_list.append(result[0])
                    state2_list.append(result[1])
                    state3_list.append(result[2])
                    state4_list.append(result[3])
                    state5_list.append(result[4])
                    result_list.append(result[0] + result[1] + result[2] + result[3] + result[4])
                return result_list, state1_list, state2_list, state3_list, state4_list

            # some initial parameter values
        
            # curve fit the combined data to the combined function
            t_combo = []
            a_combo = []
            for i in dataset:
                t_combo = np.append(t_combo, i[0])
                a_combo = np.append(a_combo, i[1])
            
            tdata1 = dataset[0][0]
            s = len(self.wavelengths)
            A0 = [1] * s
            X0 = [A0, self.x0]
            fittedParameters, pcov = optimize.curve_fit(comboFunc, t_combo, a_combo, X0, bounds = self.bounds)
            perr = np.sqrt(np.diag(pcov))
            
            A = fittedParameters[:s]
            t1 = fittedParameters[s+1]
            t2 = fittedParameters[s+2]
            t3 = fittedParameters[s+3]
            t4 = fittedParameters[s+4]
            t5 = fittedParameter[s+5]
            w1 = fittedParameters[s+6 : 2*s+6]
            w2 = fittedParameters[2*s+7 : 3*s+7]
            w3 = fittedParameters[3*s+8 : 4*s+8]
            w4 = fittedParameters[4*s+9 : 5*s+9]
            w5 = fittedParameters[5*s+10]
            
            Aerr = perr[:s]
            t1err = perr[s+1]
            t2err = perr[s+2]
            t3err = perr[s+3]
            t4err = perr[s+4]
            t5err = perr[s+5]
            w1err = perr[s+6 : 2*s+6]
            w2err = perr[2*s+7 : 3*s+7]
            w3err = perr[3*s+8 : 4*s+8]
            w4err = perr[4*s+9 : 5*s+9]
            w5err = perr[5*s+10:]
    
            for i in range(len(Aerr)):
                print('%.2f nm' % self.wavelengths[i], 'Aerr = %.2f' % Aerr[i])
           
            print('t1err = %.2f fs' % t1)
            print('t2err = %.2f fs' % t2)
            print('t3err = %.2f fs' % t3)
            print('t4err = %.2f fs' % t4)
            print('t5err = %.2f fs' % t5)
            
            for i in range(len(Aerr)):
                print('w1err = %.2f' % w1err[i])
                print('w2err = %.2f' % w2err[i])
                print('w3err = %.2f' % w3err[i])
                print('w4err = %.2f' % w4err[i])
                print('w5err = %.2f' % w5err[i])
            
            for i in range(len(s)):
                adata = a_combo[i*s : (i+1)*s]
    
                plt.figure()
                g = G(tdata1, A[i], self.t0, self.Wpump)
                plt.plot(tdata1, g, color = '#b676f2', linestyle = 'dashed', label = 'Vpump')
                plt.plot(tdata1, g*max(adata)/max(g), color = '#b676f2', label = 'Vpump scaled')
                plt.axvline(x = self.t0, color = '#F1B929', label = 'Time-Zero')
                plt.plot(tdata1, state1_list[i], 'r-', label = 'w1*N1')
                plt.plot(tdata1, state2_list[i], 'b-', label = 'w2*N2')
                plt.plot(tdata1, state3_list[i], 'g-', label = 'w3*N3')
                plt.plot(tdata1, state4_list[i], 'm-', label = 'w4*N4')
                plt.plot(tdata1, state5_list[i], 'c-', label = 'w5*N5')
                fit_result = result_list[i]
                plt.plot(tdata1, fit_result, 'k--', label = 'w1*N1 + w2*N2 + w3*N3 + w4*N4 + w5*N5')
                plt.scatter(tdata1, adata)
                plt.legend()
                wsum = w1[i] + w2[i] + w3[i] + w4[i] + w5[i]
                w1[i] = w1 / wsum
                w2[i] = w2 / wsum
                w3[i] = w3 / wsum
                w4[i] = w4 / wsum
                w5[i] = w5 / wsum
                plt.title('A = %.2f, t1 = %.2f fs, t2 = %.2f fs, t3 = %.2f, t4 = %.2f, t5 = %.2f, w1 = %.2f, w2 = %.2f, w3 = %.2f, w4 = %.2f, w5 = %.2f' % tuple((A[i], t1, t2, t3, t4, t5, w1[i], w2[i], w3[i], w4[i], w5[i])))
                plt.xlabel('%.2f nm, Delay (fs)' % self.wavelengths[i])
                plt.ylabel('ΔA (mOD)')
                # plt.show()
                
            return A, t1, t2, t3, t4, t5, w1, w2, w3, w4, w5, result_list, state1_list, state2_list, state3_list, state4_list, state5_list
"""
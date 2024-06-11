# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 15:29:15 2022

@author: fbondu
"""

# Recomputes Spectrum.py with object-oriented style
# with the goal to not recompute the DPSS at each call !
#  - SpecrumPlot for plotting spectra in various units
#  - PSD2TD to compute time sequences from PSDs 
#  - SlepianSpectrum to compute spectra from time domain sequences with multi-tapers
#  - PSFW to compute DPSS better than scipy.signal.windows.dpss (does not fail for n>92282 !)
#     (uses expansion on Legendre polynomials)


# with my implementation of spectra, (spyder/anaconda/windows) calculations on 64 bits
# np.finfo(np.longdouble)
# finfo(resolution=1e-15, min=-1.7976931348623157e+308, max=1.7976931348623157e+308, dtype=float64)
# extended precision (128 bits) might be desirable for high dynamic signals
#
# scipy.signal.win.dpss breaks at n=92681

# libraries import
import numpy                as np
import matplotlib.pyplot    as plt

# single objects in libraries import
from   scipy.sparse.linalg  import eigsh
from   scipy.special        import eval_legendre
from   scipy.stats          import chi2
from   scipy.sparse         import diags
from   scipy.signal.windows import dpss
from   sys                  import float_info

# defines classes:
# - time data
# - frequency data
# - data (time data + frequency data)

class TData:
# time data class
# contains np.array of data values evenly spaced in time
# as well as sampling time, figure, initial time value if not zero
    def __init__(self,dataVec,TimeStep=1, startTime=0, 
                 xunit='s', yunit = 'a.u.', name= ''):
        self.dataVec = dataVec  # x(t) as a np.array
        self.Ts      = TimeStep # sampling period
        self.fig     = None     # not calculated unless explicitly requested
        self.T0      = startTime
        self.xunit   = 's'      # data used when constructing plot
        self.xlabel  = 'time'   # data used when constructing plot
        self.ylabel  = 'data'   # data used when constructing plot
        self.yunit   = yunit    # data used when constructing plot
        self.title   = name    
        
    @classmethod
    def plot(self, fighandle=None):
        if fighandle == None: # create a new figure
            fighandle = plt.figure()
        else:                 # add to current figure
            fighandle = self.fig
        N    = len(self.dataVec)
        time = np.arange(0,N*self.Ts,N) + self.T0
        fighandle.axes[0].plot(time, self.dataVec)
        label = self.xlabel+ ' ('+self.xunit+')'
        fighandle.axes[0].set_xlable(label)
        label = self.ylabel+ ' ('+self.yunit+')'
        fighandle.axes[0].set_ylabel
        fighandle.axes[0].title(self.title)
        fighandle.grid(which='both')
        self.fig = fighandle

class FData:
# frequency data class
# contains np.array of data values evenly spaced in frequency
    def __init__(self,FrequencyVec, Fstep=None):
        self.dataVec = FrequencyVec # S(f) as a np.array
        self.Fstep   = Fstep        # frequency resolution
        self.fig     = None
        
    @classmethod
    def plot(self, fighandle=None):
        if fighandle == None: # create a new figure
            fighandle = plt.figure()
        else:                 # add to current figure
            fighandle = self.fig
        N    = len(self.dataVec)
        time = np.arange(0,N*self.Ts,N) + self.T0
        fighandle.axes[0].loglog(time, self.dataVec) # loglog as default
        self.fig = fighandle

class Data:
    def __init__(self, TimeData: TData, FrequencyData: FData =None):
        self.TimeData      = TData
        self.FrequencyData = FData
    # functions to compute spectrum from timedata
    # or simulated time data from spectrum to be added

class SlepianWindowSet:
    def __init__(self, NW, kmax):
        self.NW      = NW
        self.kmax    = kmax
        self.wk      = 0 # array of windows
        self.lambdak = 0 # vector of eigenvalues
    # to be continued...

class WindowSet:
    def __init__(self):
        self.algo  = 'dpss' # or 'legendre_expansion'
        self.NW    = 0
        self.Nfilt = 0 # number of filters
        self.wk    = 0 # set of windows

class PSD_Out_forensic: # forensic data of the PSD computation process
    def __init__(self):
        self.adapt_count = 0 # number of loops in the adaptive process
        self.h_k = 0
        self.S_k = 0

    
class PSD_SlepianFactory:
    def __init__(self, NW=15, kmax=None):
        self.windowSetList = []; # list of SlepianWindowSet
        self.NW = NW
        if kmax == None:
            self.kmax = int(2*NW-1)
            
        # PSD computation parameters
        self.ctype = 'adapt' # or 'linear' or 'equal'
    
    @classmethod
    def ComputePSD(self, d = Data): # PSD from time domain data
        d.FData = 0 # to be completed

    def PSD_2_Time(self, d = Data):
        d.TData = 0 # to be completed

    def filterSearch(self, NW, kmax):
        for fil in self.windowSetList:
            if fil.NW == NW and kmax <= fil.kmax:
                return fil # found; loop stops
        return None # None if not found


        
        
# latex text in figures
plt.rcParams['text.usetex'] = True

def SlepianSpectrum(TimeSeq, Ts=1, NW=10, otype='adapt', \
                    N_filters=None, stab_rel_error=1e-4, \
                    return_stability=False, return_stab_loop_count=False):
# SlepianSpectrum(TimeVec, Ts=1, NW=50, type='linear', return_stability=False)
# Parameters:
#      Input:
#         TimeSeq: list or numpy array of real values evenly spaced  in time
#         Ts:      sampling frequency
#         NW:      noise equivalent bandwidth in frequency bins; 
#                     effective bandwidth is NW/(len(TimeSeq)*Ts)
#         otype: 'linear' | 'equal' | 'adapt' (default)
#                  'linear' best linear estimate
#                           cf. eq. (10) Sanchez mtpsd 2010
#                  'equal' multi-taper with unity coefficients, cf. eq. (8) Sanchez
#                  'adapt' adaptive method
#         N_filters: number of filters to use for 'linear`' or 'equal' otype
#                   None as default: uses (int(2*NW)-1) filters
#      Output:
#          freq: frequency vector
#          DSP:  DSP values, in 1/Hz units
#           (with return_stab_loop_count=True and otype=='adapt')
#          count:  : number of loops for stabilized method
#           (with return_stability=True)
#          N_filters: number of filters used
#          nu: approximate number of degrees of freedom of estimate
#                 frequency dependent if 'adapt' option
#
# written after "mtpsd documentation", C Antonio Sanchez, nov 1, 2010
# see also in "Spectrum estimation and harmonic analysis", David J Thomson 1982
# see also "spectrum analysis for physical applications", D. Percival and A. Walden 1993
# see also "multitaper spectrum estimation", G. Prieto, 2004
#
# Possible evolutions:
#  - keep in memory already computed filters
#  - include jacknife/stationarity test, cf. David J Thomson 2007 "jacknifing multitaper spectrum estimates"
#  - improve estimate of bias
#  - quadratic inverse theory?

    N = len(TimeSeq)
    maxcont = 100
    max_dynamic_range = 1e-12
        # criterium for evaluating the DSP change with 'adapt' algorithm,
        # given that dynamic_range = DSP/max(DSP)

    # sample variance for stabilized calculation
    sigma2 = np.var(TimeSeq)*N
       # eq (12) in J. Park 1987: would give N greater bias
       # incompatible by factor N with eq. before 5.2 in Thomson 1981
       # but less bias in strongly shaped noises

    # number of filters
    if N_filters == None:
        N_filters = int(2*NW)-1

    # get dpss functions and associated eigenvalues
    # dpss are normalized such that sum(w_k[i]**2) = 1: as it should be for a window
    #     lambda_k are eigenvalues
    win_k, lambda_k = PSWF(N, NW, kmax=N_filters)

    # calculation of spectra
    # cf. C. Antonio Sanchez, "mtpsd documentation", nov 2010 
        
    SimpleMean = np.mean(TimeSeq)
    TimeSeq = np.tile(TimeSeq, (N_filters,1))
    # construct an array of timesequences, 
    # with as many columns as the number of filters

    #
    # compute spectrum for each window
    #
    h_k = TimeSeq * win_k
    # mean removal. See Sanchez section 2.4
    for k in range(N_filters):
        if (k%2)==0: # even
            h_k[k,:] = h_k[k,:] - np.sum(h_k[k,:])/np.sum(win_k[k,:])
        else: # odd filters
            h_k[k,:] = h_k[k,:] - SimpleMean

    y_k = np.fft.rfft2(h_k, axes=[1]) # return values for positive frequencies
    S_k = abs(y_k)**2
    S_k = S_k[:,1:] #remove freq=DC values

    #
    # average spectra
    #
    if otype == 'linear':
        weight_k = lambda_k/sum(lambda_k)
    else:
        # else otype == 'equal'
        weight_k = lambda_k*0 +1/N_filters

    SpecDSP = np.zeros(np.ma.size(S_k,1)) # same length as fft
    
    for k in range(N_filters):
        SpecDSP = SpecDSP + weight_k[k] * S_k[k]

    del weight_k, y_k, h_k, TimeSeq
    # end spectrum calculation for 'equal' and 'linear

    #
    # construction of frequency vector    
    #
    F_delta = 1/(N*Ts) # frequency resolution
    N_pts   = len(SpecDSP)
    freq = np.linspace(F_delta,F_delta*N_pts,N_pts)

    #
    # continue calculations for stabilized version
    #
    if otype =='adapt': # Sanchez eq (11) and (12)
        # init with average of first two solutions
        SpecDSP = 0.5*(S_k[0,:]+S_k[1,:])
        count = 0
        loopCond = True
        while( loopCond ):
              count = count+1
              sumcoeffs = SpecDSP * 0
              bk = S_k * 0
              bias = S_k * 0
              for k in range(N_filters):
                  bk[k,:] = SpecDSP / (lambda_k[k]*SpecDSP + (1-lambda_k[k])*sigma2)
                  bias[k,:] = lambda_k[k]*SpecDSP / ((1-lambda_k[k])*sigma2)
                  sumcoeffs = sumcoeffs + lambda_k[k]*bk[k,:]**2
              NewSpec = 0
              for k in range(N_filters):
                  wk = lambda_k[k]*bk[k,:]**2/sumcoeffs
                  NewSpec = NewSpec + wk * S_k[k,:]
              spec_relative_val_change = abs(NewSpec-SpecDSP)/SpecDSP
              # do not include in optimization frequencies for which the dynamic range is too big
#              print(np.amax(spec_relative_val_change), spec_relative_val_change[3000])
              idxEliminate = ( (NewSpec/np.amax(NewSpec)) < max_dynamic_range)
#              print(NewSpec[3000]/np.amax(NewSpec), spec_relative_val_change[3000])
              spec_relative_val_change[idxEliminate] = 0
              loopCond =  ( np.amax(spec_relative_val_change) > stab_rel_error) \
                   and (count<maxcont)
              SpecDSP = NewSpec
              
    # Output, all methods 'linear', 'equal', 'adapt'
    out = freq, SpecDSP*2*Ts 
        # re-calibrate spectrum for Ts non unity
        # factor of 2 for monolateral PSD

        # return stabilized algorithm loop count
    if otype=='adapt' and return_stab_loop_count==True:
        out = *out, count

        # return stability vector see eq. 5.5 Thomson 1982 and Sanchez eq 11
        # nu is approximate number of degrees of freedom of the estimate
    if return_stability==True:
        if otype=='adapt':
            nu = 2*sumcoeffs
        elif otype=='linear':
            nu = N_filters 
        else: # 'equal'
            nu = sum(lambda_k)
        out = *out, N_filters, nu

    return out


def SpectrumPlot(freq, spectrum, in_spec_unit='LSD', in_spec_log='natural', \
                 plot_spec_unit='LSD', plot_spec_log='natural', \
                 meas_unit='1', integ_rms = False, \
                 plot_stability=False, stab_data = [], \
                 plot_confidence_interval=False, conf_int_length=0.9, nu=[], \
                 fighandle = 0):
    # 'PSD' for power spectral density (1/Hz)
    #     or 'LSD' for linear spectral density (1/rtHz)
    # ’natural’ for normal unit
    #     or 'dB’ for decibels
    # dB necessarily implies PSD
    # integ_rms: boolean. If true, displays the rms value, integrated from high frequencies
    spec_units = {'PSD','LSD'}
    spec_logs  = {'natural','dB'}

    if in_spec_unit not in spec_units:
        raise ValueError("SpectrumPlot: in_spec_unit must be one of %r." % spec_units)
    if plot_spec_unit not in spec_units:
        raise ValueError("SpectrumPlot: plot_spec_unit must be one of %r." % spec_units)
    if in_spec_log not in spec_logs:
        raise ValueError("SpectrumPlot: in_spec_log must be one of %r." % spec_logs)
    if plot_spec_log not in spec_logs:
        raise ValueError("SpectrumPlot: plot_spec_log must be one of %r." % spec_logs)

    # put back all spectra in LSD, natural units
    if in_spec_log == 'dB':
        # obviously PSD
        spectrum = 10**(spectrum/20)
    else:
        if in_spec_unit == 'PSD':
            spectrum = np.sqrt(spectrum)
            
    # now put data with display requests
    if plot_spec_log == 'dB':
        spectrumPlot = 20*np.log10(spectrum)
    else:
        if plot_spec_unit == 'PSD':
            spectrumPlot = spectrum**2
        else:
            spectrumPlot = spectrum

    if fighandle == 0:
        rdx0 = plt
        if plot_stability==True:
            fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig = plt.figure()
        if plot_spec_log == 'dB':
            unit2 = 'dB'
            if meas_unit != '1':
                unit2 = unit2+meas_unit
            label1 = 'Power'
            unit1  = 'Hz'
        else:
            unit2 = meas_unit
            if plot_spec_unit == 'LSD':
                label1 = 'Linear'
                unit1  = r'$\sqrt{\mathrm{Hz}}$'
            else:
                label1 = 'Power'
                unit1  = 'Hz'
        if plot_stability==False:
            plt.grid(which='both')
            plt.ylabel(label1+' spectral density ('+unit2+'/'+unit1+')')
            plt.xlabel('Fourier frequency (Hz)')
        else:
            ax[0].grid(which='both')
            ax[1].grid(which='both')
            ax[1].set_xlabel('Fourier frequency (Hz)')
            ax[0].set_ylabel(label1+' spectral density ('+unit2+'/'+unit1+')')
            ax[1].set_ylabel(r'stability $\chi^2_N/N$')
            ax[1].set_ylim([0,2.1])
            rdx1 = ax[1]
            rdx0 = ax[0]
    else:
        rdx0 = fighandle.axes[0]
        if plot_stability==True:
            rdx1 = fighandle.axes[1]
        fig = fighandle

    if plot_spec_log == 'dB':
        rdx0.semilogx(freq, spectrumPlot)
    else:
        rdx0.loglog(freq, spectrumPlot)

    if integ_rms==True:
        res = calc_integ_rms(freq, spectrum)
        if plot_spec_log == 'dB':
            res = 20*np.log10(res)
            rdx0.semilogx(freq[:-1],res)
        else:
            if plot_spec_unit == 'PSD':
                res = res**2
            rdx0.loglog(freq[:-1],res)

    if plot_stability==True:
        rdx1.semilogx(freq, stab_data)
        
    if plot_confidence_interval==True:
        if type(nu)=='int' or type(nu)=='float':
            nu = freq*0 + nu # Sanchez eq. 17
        q = (1-conf_int_length)/2.0
        lowlim  = spectrum**2 * nu / chi2.ppf(1-q, nu)
        highlim = spectrum**2 * nu / chi2.ppf(q,   nu)
        if plot_spec_unit == 'LSD':
            lowlim  = np.sqrt(lowlim)
            highlim = np.sqrt(highlim)
        print(lowlim[100],highlim[100])
        rdx0.fill_between(freq, lowlim, highlim, color='red', alpha=0.3)

    return fig


def calc_integ_rms(freq, spectrum):
    # internal. 
    # spectrum in 1/sqrt(Hz), no dB.
    spectrum = np.flip(spectrum**2)
    df       = np.flip(np.diff(freq))
    spectrum = spectrum[1:]
    res      = np.cumsum(spectrum*df)
    res      = np.flip(res)
    return np.sqrt(res)


def PSD2TD (xPSD, fmin):
    # PSD2TD (PSD, fmin)
    # generate time-domain data for gaussian noise with given PSD
    # input:
    #     PSD : an array of PSD values, equally spaced in frequency, with interval fmin
    #           array has length N
    #     fmin: first frequency (assumed other frequencies to be n*fmin)
    #            not zero!
    # output:
    #     t: time vector
    #     x:  data evenly spaced in time, with interval Ts
    # for input of size N, time data has size 2N+2

    v = xPSD*fmin/2 # variance for quadratures
    N = len(xPSD)

    vi = np.random.randn(N) # variance 1
    vq = np.random.randn(N)
    w = (vi + 1j*vq)*np.sqrt(v)

    # length of returned time sequence 
    N_return = 2*(len(w)+1)
    # sampling time of returned time sequence
    Ts = 1/(N_return*fmin)

    # compute time vector
    x = np.fft.irfft(np.concatenate(([0], w, [0])),N_return)
        # add 0 amplitude at 0 frequency
        # add 0 amplitude at semi-alternate value (exact Nyquist frequency)
    x = x*N_return/np.sqrt(2)
        # correct for inverse fft factor, 
        # and correct for rms values

    t  = np.linspace(0,Ts*(N_return-1),N_return)
        # prepare time vector for plots
    
    return t, x

def PSWF(Npoints, NW, kmax):
#    computes dpss hopefully also for large vectors,
#    provided that kmax in not too big
# input:
#    Npoints  (int):   requested sequence length
#    NW       (float): time-bandwidth product = c in [1]
#    kmax     (int):   number of filters to compute
# output:
#    psi_n:    kmax * Npoints array of DPSS windows
#    lambda_n: corresponding eigen values 
#
#  windows computed with Legendre expansion for Npoints > 92682
#  should be very efficient when NW is large (>50000) and NW moderate (10’s)
#  eigenvalues computed with scipiy.signal.windows.dpss
#
# [1] Osipov and Rokhlin, Apll Comput. Harmon. Anal 36 (2014) 108-142
# [2] Xiao, Rokhlin and Yarvin, Inverse Problems 17 (2001) 805-838
# [3] Wang, Math. of comp. 79 (2009) 807-827
#

    if Npoints < 92681:
        psi_n, lambda_n = dpss(Npoints, NW, kmax, sym=False, norm=2, return_ratios=True)
        # check that lambda_n < 1
        lambda_n[lambda_n>=1] = 1-float_info.epsilon
        return psi_n, lambda_n

    c = NW*np.pi # Slepian 1983, eq. 17: c = pi W T
    
    M = max(2*kmax + 30, 60) # [3] after eq. (2.19)
    
    if c > (np.pi/2)*(M+0.5): # [3] eq. (2.20) # 
        print('error: bandwith too large\n')
        print('or increase kmax',)
        # otherwise use other estimate for large c ?
        return -1

    k      = np.arange(M)
    Akk    = k*(k+1) + c**2 * ( 2*k*(k+1)-1 )/( (2*k+3)*(2*k-1) ) # [1] eq 61
    Akkp2  = c**2 * (k+2)*(k+1)/( (2*k+3) * np.sqrt((2*k+1)*(2*k+5)) )

    Akk_even   = Akk[::2]
    Akkp2_even = Akkp2[::2]
    A_even     = diags([Akk_even,Akkp2_even,Akkp2_even], [0,-1,1])

    Akk_odd    = Akk[1::2]
    Akkp2_odd  = Akkp2[1::2]
    A_odd      = diags([Akk_odd,Akkp2_odd,Akkp2_odd], [0,-1,1])

    n_even = kmax//2 if kmax%2==0 else kmax//2+1
    _, beta_even  = eigsh(A_even, k=n_even, which='SM')
    n_odd  = kmax//2 
    _, beta_odd   = eigsh(A_odd,  k=n_odd, which='SM')

    x        = np.linspace(-1,1,Npoints)

    psi_n    = np.zeros((kmax, Npoints))
    lambda_n = np.zeros(kmax)

    CalibLegendre = np.sqrt(np.arange(M)+0.5) # [2] eq. (58)

    # determination of eigenfunctions
    for q in range(M):
        
        Lq_x = CalibLegendre[q]*eval_legendre(q, x)
        # compute Legendre polynomial first, only once
        # NB legendre(q) does not work for q>68, but eval_legendre does
        
        if q%2 == 0:
            
            # even functions
            for k in range(0,kmax,2):
                psi_n[k,:] = psi_n[k,:] + beta_even[q//2,k//2]*Lq_x
                    
        else:
            
            # odd functions
            for k in range(1,kmax,2):
                psi_n[k,:] = psi_n[k,:] + beta_odd[q//2,k//2]*Lq_x
                    
    # determination of eigenvalues from [3] eqs. (2.21) and (2.22)
    # does not work well for low orders, and odd k’s
    # as NW does not depend on Ndata, use dpss instead with large n
    _, lambda_n = dpss(10000, NW, kmax, sym=False, norm=2, return_ratios=True)
    lambda_n[lambda_n>=1] = 1-float_info.epsilon

    # normalize dpss
    for k in range(kmax):
        s = sum(psi_n[k,:]**2)
        psi_n[k,:] = psi_n[k,:]/np.sqrt(s)
                # sum(w**2)=1 for a window
    
                
    
    return psi_n, lambda_n

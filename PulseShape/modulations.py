from functools import wraps
import numpy as np
from scipy.integrate import cumtrapz

AmplitudeModulations = {}
FrequencyModulations = {}
eps = np.finfo(float).eps

class ModFunc:
    def __init__(self, func):
        self._repr = func.__doc__
        self.func = func
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__

    def __repr__(self):
        return self._repr

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def am_func(func):
    """
    Decorator to add function to the AmplitdueModulation dictionary of amplitude modulation functions
    Parameters
    ----------
    func: function
        the function to be added to the AmplitdueModulation dictionary
    """
    AmplitudeModulations[func.__name__] = ModFunc(func)
    return func


def fm_func(func):
    """
    Decorator to add function to the FrequencyModulation dictionary of frequency modulation functions
    Parameters
    ----------
    func: function
        the function to be added to the FrequencyModulation dictionary
    """
    FrequencyModulations[func.__name__] = ModFunc(func)
    return func


# Amp mods
@am_func
def rectangular(Pulse):
    """Amplitude modulation function for a rectangular pulse.
    kwargs
    ------
    None"""

    return np.ones(len(Pulse.time))


@am_func
def gaussian(Pulse):
    """Amplitude modulation function for a gaussian pulse.
    kwargs
    ------
    tFWHM: float
        Full width half max of the gaussian in the time domain (us)
    trunc: float
        Truncation parameter (0-1)

    NOTE: If both tFWHM and trunc are passed tFWHM will take precedence"""


    if not hasattr(Pulse, 'tFWHM'):
        if not hasattr(Pulse, 'trunc'):
            raise AttributeError('Pulse object must have wither `tFWHM` or `trunc` defined in kwargs')
        else:
            Pulse.tFWHM = np.sqrt(-(Pulse.pulse_time**2)/np.log2(Pulse.trunc))

    if Pulse.tFWHM == 0:
        Pulse.tFWHM = Pulse.time_step / 2

    return np.exp(-(4 * np.log(2) * Pulse.ti ** 2) / Pulse.tFWHM ** 2)


@am_func
def sinc(Pulse):
    """Amplitude modulation function for a sinc pulse.
    kwargs
    ------
    zerocross: float
        Width between the zero-crossing points in the time domain (us)"""

    if not hasattr(Pulse, 'zerocross'):
        raise AttributeError('Pulse object must have zerocross defined in kwargs')

    x = 2 * np.pi * Pulse.ti / Pulse.zerocross
    amp = np.sin(x) / x
    amp[np.isnan(amp)] = 1
    amp /= amp.max()
    return amp


@am_func
def halfsin(Pulse):
    """Amplitude modulation function for a halfsin pulse.
    kwargs
    ------
    None"""

    return np.cos(np.pi * Pulse.ti / Pulse.pulse_time)


@am_func
def quartersin(Pulse):
    """Amplitude modulation function for a quartersin pulse.
    kwargs
    ------
    trise: float
        Rise time at pulse edges (us)"""

    if not hasattr(Pulse, 'trise'):
        raise AttributeError('Pulse object must have trise defined in kwargs')

    amp = np.ones(len(Pulse.time))
    if Pulse.trise != 0 and 2 * Pulse.trise < Pulse.pulse_time:
        tpartial = np.arange(0, Pulse.time_step + Pulse.trise - eps, Pulse.time_step)
        npts = len(tpartial)
        amp[:npts] = np.sin(tpartial * (np.pi / (2 * Pulse.trise)))
        amp[-npts:] = amp[npts-1::-1]

    return amp


@am_func
def sech(Pulse):
    """Amplitude modulation function for a sech pulse.
    kwargs
    ------
    n: int
        Secant function order. Asymmetric secant functions can be specified by passing an Array-like object with two
        values
    beta: float
        Truncation parameter"""

    if not all(hasattr(Pulse, param) for param in ['n', 'beta']):
        raise AttributeError('Pulse object must have both n and beta defined in kwargs')

    Pulse.n = np.atleast_1d(Pulse.n)

    if len(Pulse.n) == 1:
        if Pulse.n == 1:
            amp = 1 / np.cosh(Pulse.beta * Pulse.ti / Pulse.pulse_time)
        else:
            amp = 1 / np.cosh(Pulse.beta * 0.5 * (2 * Pulse.ti / Pulse.pulse_time) ** Pulse.n)
    elif len(Pulse.n) == 2:
        amp = np.empty_like(Pulse.ti)
        amp[Pulse.ti < 0] = 1 / np.cosh(Pulse.beta * 0.5 * (2 * Pulse.ti[Pulse.ti < 0] / Pulse.pulse_time) ** Pulse.n[0])
        amp[Pulse.ti >= 0] = 1 / np.cosh(Pulse.beta * 0.5 * (2 * Pulse.ti[Pulse.ti >= 0] / Pulse.pulse_time) ** Pulse.n[1])
    else:
        raise ValueError('sech `n` parameter must have at least one and no more than 2 elements')

    return amp


@am_func
def WURST(Pulse):
    """Amplitude modulation function for a WURST pulse.
    kwargs
    ------
    nwurst: int
        WURST n parameter to set the steepness of the AM function"""

    if not hasattr(Pulse, 'nwurst'):
        raise AttributeError('Pulse object must have nwurst defined in kwargs')

    amp = 1 - np.abs(np.sin(np.pi * Pulse.ti/Pulse.pulse_time))**Pulse.nwurst
    return amp


@am_func
def gaussian_cascade(Pulse):
    """Amplitude modulation function for a gaussian cascade pulse.
    kwargs
    ------
    A0: Array-like
        list of gaussian amplitudes
    x0: Array-like
        list of gaussian mean positions as a fraction of the pulse_time
    FWHM: Array-like
        list of gaussian FWHMs as a fraction of the pulse_time"""

    if not all(hasattr(Pulse, param) for param in ['A0', 'x0', 'FWHM']):
        raise AttributeError('Pulse object must have `A0`, `x0`, and `FWHM` defined in kwargs')

    Pulse.A0, Pulse.x0, Pulse.FWHM = np.atleast_1d(Pulse.A0), np.atleast_1d(Pulse.x0), np.atleast_1d(Pulse.FWHM)

    amp = np.zeros(len(Pulse.time))
    for a0, x, fwhm in zip(Pulse.A0, Pulse.x0, Pulse.FWHM):
        amp += a0 * np.exp(-(4 * np.log(2) / (fwhm * Pulse.pulse_time) ** 2) * (Pulse.time - x * Pulse.pulse_time)**2)
    amp /= max(amp)
    return amp


@am_func
def fourier_series(Pulse):
    """Amplitude modulation function for a fourier series pulse.
    kwargs
    ------
    A0: float
        Initial amplitude coefficient
    An: Array-like
        List of Fourier coefficients for cos
    Bn: Array-like
        List of Fourier coefficients for sin"""

    if not all(hasattr(Pulse, param) for param in ['An', 'Bn', 'A0']):
        raise AttributeError('Pulse object must have `An`, `Bn`, and `A0` defined in kwargs')

    amp = np.zeros(len(Pulse.time)) + Pulse.A0
    for i, (an, bn) in enumerate(zip(Pulse.An, Pulse.Bn)):
        j = i + 1
        amp += an * np.cos(j * 2 * np.pi * Pulse.time / Pulse.pulse_time) + \
               bn * np.sin(j * 2 * np.pi * Pulse.time / Pulse.pulse_time)

    amp /= max(amp)
    return amp


@am_func
def I_BURP1(Pulse):
    """Function to call fourier series amplitude modulation function with default I_BURP1 parameters.
    kwargs
    ------
    None"""

    Pulse.A0 = 0.5
    Pulse.An = [0.70, - 0.15, - 0.94, 0.11, -0.02, -0.04, 0.01, -0.02, -0.01]
    Pulse.Bn = [-1.54, 1.01, - 0.24, -0.04, 0.08, -0.04, -0.01, 0.01, -0.01]
    return fourier_series(Pulse)


@am_func
def I_BURP2(Pulse):
    """Function to call fourier series amplitude modulation function with default I_BURP2 parameters.
    kwargs
    ------
    None"""

    Pulse.A0 = 0.5
    Pulse.An = [0.81, 0.07, -1.25, -0.24, 0.07, 0.11, 0.05, -0.02, -0.03, -0.02, 0.00]
    Pulse.Bn = [-0.68, -1.38, 0.20, 0.45, 0.23, 0.05, -0.04, -0.04, 0.00, 0.01, 0.01]
    return fourier_series(Pulse)


@am_func
def SNOB_i2(Pulse):
    """Function to call fourier series amplitude modulation function with default SNOB_i2 parameters.
    kwargs
    ------
    None"""

    Pulse.A0 = 0.5
    Pulse.An = [-0.2687, -0.2972, 0.0989, -0.0010, -0.0168, 0.0009, -0.0017, -0.0013, -0.0014]
    Pulse.Bn = [-1.1461, 0.4016, 0.0736, -0.0307, 0.0079, 0.0062, 0.0003, -0.0002, 0.0009]
    return fourier_series(Pulse)


@am_func
def SNOB_i3(Pulse):
    """Function to call fourier series amplitude modulation function with default SNOB_i3 parameters.
    kwargs
    ------
    None"""

    Pulse.A0 = 0.5
    Pulse.An = [0.2801, -0.9995, 0.1928, 0.0967, -0.0480, -0.0148, 0.0088, -0.0002, -0.0030]
    Pulse.Bn = [-1.1990, 0.4893, 0.2439, -0.0816, -0.0409, 0.0234, 0.0036, -0.0042, 0.0001]
    return fourier_series(Pulse)


@am_func
def G3(Pulse):
    """Function to call gaussian cascade amplitude modulation function with default G3 parameters.
    kwargs
    ------
    None"""

    Pulse.x0 = [0.287, 0.508, 0.795]
    Pulse.A0 = [-1, 1.37, 0.49]
    Pulse.FWHM = [0.189, 0.183, 0.243]
    return gaussian_cascade(Pulse)


@am_func
def G4(Pulse):
    """Function to call gaussian cascade amplitude modulation function with default G4 parameters.
    kwargs
    ------
    None"""

    Pulse.x0 = [0.177, 0.492, 0.653, 0.892]
    Pulse.A0 = [0.62, 0.72, -0.91, -0.33]
    Pulse.FWHM = [0.172, 0.129, 0.119, 0.139]
    return gaussian_cascade(Pulse)


@am_func
def Q3(Pulse):
    """Function to call gaussian cascade amplitude modulation function with default Q3 parameters.
    kwargs
    ------
    None"""

    Pulse.x0 = [0.306, 0.545, 0.804]
    Pulse.A0 = [-4.39, 4.57, 2.60]
    Pulse.FWHM = [0.180, 0.183, 0.245]
    return gaussian_cascade(Pulse)


@am_func
def Q5(Pulse):
    """Function to call gaussian cascade amplitude modulation function with default Q5 parameters.
    kwargs
    ------
    None"""

    Pulse.x0 = [0.162, 0.307, 0.497, 0.525, 0.803]
    Pulse.A0 = [-1.48, -4.34, 7.33, -2.30, 5.66]
    Pulse.FWHM = [0.186, 0.139, 0.143, 0.290, 0.137]
    return gaussian_cascade(Pulse)


# Freq Mods
@fm_func
def none(Pulse):
    """Frequency modulation function for a pulse with no frequency modulation.
    kwargs
    ------
    None"""

    freq = np.zeros(len(Pulse.time))
    phase = np.zeros(len(Pulse.time))

    return freq, phase


@fm_func
def linear(Pulse):
    """Frequency modulation function for a pulse with linear frequency modulation.
    kwargs
    ------
    freq: Array-like
        Frequency range to sweep with respect to base frequency (MHz)"""

    if not hasattr(Pulse, 'freq'):
        raise AttributeError('Pulse object must have a `freq` parameter of length 2)')

    k = (Pulse.freq[1] - Pulse.freq[0]) / Pulse.pulse_time
    freq = k * Pulse.ti
    phase = 2 * np.pi * ((k /2) * Pulse.ti ** 2)
    return freq, phase


@fm_func
def tanh(Pulse):
    """Frequency modulation function for a pulse with hyperbolic tangent frequency modulation.
    kwargs
    ------
    beta: float
        Truncation parameter
    freq: Array-like
        Frequency range to sweep with respect to the base frequency (MHz)"""

    if not all(hasattr(Pulse, param) for param in ['beta', 'freq']):
        raise AttributeError('Pulse object must have `beta` parameter and `freq` parameter (length 2)')

    Pulse.BWinf = (Pulse.freq[1] - Pulse.freq[0]) / np.tanh(Pulse.beta / 2)
    freq = (Pulse.BWinf / 2) * np.tanh((Pulse.beta/Pulse.pulse_time)* Pulse.ti)
    phase = (Pulse.BWinf/2)*(Pulse.pulse_time/Pulse.beta) * np.log(np.cosh((Pulse.beta/Pulse.pulse_time)*Pulse.ti))
    phase = 2 * np.pi * phase

    return freq, phase


@fm_func
def uniformq(Pulse):
    """Frequency modulation function for a pulse with a uniform frequency.
    kwargs
    ------
    None"""

    freq = cumtrapz(Pulse.amplitude_modulation**2, Pulse.ti, initial=0) / np.trapz(Pulse.amplitude_modulation**2, Pulse.ti, )
    freq = (Pulse.freq[1] - Pulse.freq[0]) * (freq - 1/2)
    phase = 2 * np.pi * cumtrapz(freq, Pulse.ti, initial=0)
    phase += np.abs(min(phase))
    return freq, phase

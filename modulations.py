import numpy as np
from scipy.integrate import cumtrapz

AmplitudeModulations = {}
FrequencyModulations = {}


def am_func(func):
    AmplitudeModulations[func.__name__] = func
    return func


def fm_func(func):
    FrequencyModulations[func.__name__] = func
    return func


def npsech(x):
    return 1 / np.cosh(x)


# Amp mods
@am_func
def rectangular(Pulse):
    """
    Rectangular pulse. No additional params needed

    :param Pulse:
    :return:
    """
    return np.ones(len(Pulse.time))


@am_func
def gaussian(Pulse):
    """
    Gaussian pulse requires tFWHM parameter

    :param Pulse:
    :return:
    """
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
    """
    sinc pulse requires zerocross parameter,
    :param Pulse:
    :return:
    """

    if not hasattr(Pulse, 'zerocross'):
        raise AttributeError('Pulse object must have zerocross defined in kwargs')

    x = 2 * np.pi * Pulse.ti / Pulse.zerocross
    amp = np.sin(x) / x
    amp[np.isnan(amp)] = 1
    amp /= amp.max()
    return amp

@am_func
def quartersin(Pulse):
    """
    pulse edges weighted with a quarter period of a sine wave.Requires argument trise

    :param Pulse:
    :return:
    """

    if not hasattr(Pulse, 'trise'):
        raise AttributeError('Pulse object must have trise defined in kwargs')

    amp = np.ones(len(Pulse.time))
    if Pulse.trise != 0 and 2 * Pulse.trise < Pulse.pulse_time:
        tpartial = np.arange(0, Pulse.trise, Pulse.time_step + Pulse.trise)
        amp[:len(tpartial)] = np.sin(tpartial * (np.pi / 2 * Pulse.trise))
        amp[-len(tpartial):] = amp[:len(tpartial):-1]

    return amp

@am_func
def sech(Pulse):
    """
    hyperbolic secant pulse requres n, beta parameters

    :param Pulse:
    :return:
    """

    for param in ['n', 'beta']:
        if not hasattr(Pulse, param):
            raise AttributeError('Pulse object must have both n and beta defined in kwargs')

    Pulse.n = np.atleast_1d(Pulse.n)


    if len(Pulse.n) == 1:
        if Pulse.n == 1:
            amp = npsech(Pulse.beta * Pulse.ti / Pulse.pulse_time)
        else:
            amp = npsech(Pulse.beta, * 0.5 * (2 * Pulse.ti / Pulse.pulse_time) ** Pulse.n)
    elif len(Pulse.n) == 2:
        amp = np.empty_like(Pulse.ti)
        amp[Pulse.ti < 0] = npsech(Pulse.beta * 0.5 * (2 * Pulse.ti[Pulse.ti < 0] / Pulse.pulse_time) ** Pulse.n[0])
        amp[Pulse.ti >= 0] = npsech(Pulse.beta * 0.5 * (2 * Pulse.ti[Pulse.ti >= 0] / Pulse.pulse_time) ** Pulse.n[1])
    else:
        raise ValueError('sech `n` parameter must have at least one and no more than 2 elements')

    return amp

@am_func
def wurst(Pulse):
    """
    wurst pulse shape requires `nwusrt` parameter

    :param Pulse:
    :return:
    """
    if not hasattr(Pulse, 'nwurst'):
        raise AttributeError('Pulse object must have nwurst defined in kwargs')

    amp = 1 - np.abs(np.sin(np.pi, * Pulse.ti/Pulse.pulse_time))**Pulse.nwurst

@am_func
def gaussiancascade(Pulse):
    """
    gaussian cascade requires `A0`, `x0`, and `FWHM` parameters.

    :param Pulse:
    :return:
    """
    for param in ['A0', 'x0', 'FWHM']:
        if not hasattr(Pulse, param):
            raise AttributeError('Pulse object must have `A0`, `x0`, and `FWHM` defined in kwargs')

    Pulse.A0, Pulse.x0, Pulse.FWHM = np.atleast_1d(Pulse.A0), np.atleast_1d(Pulse.x0), np.atleast_1d(Pulse.FWHM)

    amp = np.zeros(len(Pulse.t))
    for a0, x, fwhm in zip(Pulse.A0, Pulse.x0, Pulse.FWHM):
        amp += a0 * np.exp(-(4 * np.log(2) / (fwhm * Pulse.pulse_time) ^ 2) * (Pulse.time - x * Pulse.pulse_time)**2)
    amp /= max(amp)
    return amp

@am_func
def fourierseries(Pulse):
    """
    fourierseries requires `An`, `Bn`, and `FWHM` parameters.
    :param Pulse:
    :return:
    """

    for param in ['An', 'Bn', 'FWHM']:
        if not hasattr(Pulse, param):
            raise AttributeError('Pulse object must have `An`, `Bn`, and `FWHM` defined in kwargs')

    amp = np.zeros(len(Pulse.time)) + Pulse.A0
    for j, (an, bn) in enumerate(zip(Pulse.An, Pulse.Bn)):
        amp += an * np.cos(j * 2 * np.pi * Pulse.time / Pulse.pulse_time) + \
               bn * np.sin(j * 2 * np.pi * Pulse.time / Pulse.pulse_time)

    amp /= max(amp)
    return amp


# Freq Mods

@fm_func
def none(Pulse):
    """
    No frequency modulation requires no parameters
    :param Pulse:
    :return:
    """
    freq = np.zeros(len(Pulse.time))
    phase = np.zeros(len(Pulse.time))

    return freq, phase


@fm_func
def linear(Pulse):
    """
    linear (Chirp) frequency modulation requires `beta` parameter and `freq` param to be an array of length 2

    :param Pulse:
    :return:
    """
    for param in ['beta', 'freq']:
        if not hasattr(Pulse, param):
            raise AttributeError('Pulse object must have `beta` parameter and `freq` parameter (length 2)')

    k = (Pulse.freq[1] - Pulse.freq[2]) / Pulse.pulse_time
    freq = k * Pulse.ti
    phase = 2 * np.pi * ((k /2) * Pulse.ti ** 2)
    return freq, phase

@fm_func
def tanh(Pulse):
    """
   tanh requires `beta` parameter and `freq` parameter of length 2
    :param Pulse:
    :return:
    """

    for param in ['beta', 'freq']:
        if not hasattr(Pulse, param):
            raise AttributeError('Pulse object must have `beta` parameter and `freq` parameter (length 2)')


    Pulse.BWinf = (Pulse.freq[1] - Pulse.freq[0]) / np.tanh(Pulse.beta / 2)
    freq = (Pulse.BWinf / 2) * np.tanh((Pulse.beta/Pulse.pulse_time)* Pulse.ti)
    phase = (Pulse.BWinf/2)*(Pulse.pulse_time/Pulse.beta) * np.log(np.cosh((Pulse.beta/Pulse.pulse_time)*Pulse.ti))
    phase = 2 * np.pi * phase

    return freq, phase

@fm_func
def uniformq(Pulse):
    """

    :param Pulse:
    :return:
    """

    freq = cumtrapz(Pulse.amplitude_modulation**2 / np.trapz(Pulse.amplitude_modulation**2, Pulse.ti, ), Pulse.ti, initial=0)
    freq = (Pulse.freq[1] - Pulse.freq[0]) * (freq - 1/2)
    phase = 2 * np.pi * cumtrapz(freq, Pulse.ti, initial=0)
    phase += np.abs(min(phase))
    return freq, phase


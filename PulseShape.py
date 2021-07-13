import numpy as np
from scipy.interpolate import interp1d, pchip_interpolate
from scipy.integrate import cumtrapz
from modulations import AmplitudeModulations, FrequencyModulations

class Pulse:
    """
    A pulse object contains everything that needs to be known about a pulse.
    """

    def __init__(self, pulse_time, time_step, flip=np.pi, mwFreq=33.80,
                 amp=None, Qcrit=None, freq=0, phase=0, type='rectangular', **kwargs):
        """

        :param pulse_time:
        :param time_step:
        :param flip:
        :param amp:
        :param Qcrit:
        :param freq:
        :param phase:
        :param type:
        :param kwargs:
        """
        self.pulse_time = pulse_time
        self.time_step = time_step
        self.flip = flip
        if self.flip > np.pi:
            raise ValueError("flip angle should be less than or equal to pi")
        self.amp = amp
        self.Qcrit = Qcrit
        self.freq = freq
        self.inp_phase = phase
        self.type = type
        self.n = kwargs.get('n', 1)
        self.mwFreq = mwFreq

        ntype = len(type.split('/'))
        if ntype == 2:
            am, fm = type.split('/')
            self.am_func, self.fm_func = AmplitudeModulations[am], FrequencyModulations[fm]
        elif ntype == 1:
            self.am_func, self.fm_func = AmplitudeModulations[type], FrequencyModulations['none']
        else:
            raise ValueError('Pulse object accepts only one amplitude modulation and one frequency modulation')

        self.__dict__.update(kwargs)
        if hasattr(self, 'profile'):
            self.profile = self.profile if len(self.profile) == 2 else self.profile.T

        self.time = np.arange(0, self.pulse_time + self.time_step, self.time_step)
        self.ti = self.time - self.pulse_time / 2

        self._shape()
        if hasattr(self, 'profile') and self.fm_func.__name__ != 'none':
            self.bw_comp()

        if self.amp is None:
            self._compute_flip_amp()

        self._compute_IQ()


    def _shape(self):
        self.amplitude_modulation = self.am_func(self)
        self.frequency_modulation, self.phase = self.fm_func(self)

    def bw_comp(self):
        nu0 = self.frequency_modulation.copy()
        A0 = self.amplitude_modulation.copy()
        newaxis = nu0 + np.mean(self.freq) + self.mwFreq * 1e3

        if hasattr(self, 'profile'):
            f = self.profile[0] * 1e3
            H = self.profile[1]

            if newaxis.min() < f.min() or newaxis.max() > f.max():
                raise ValueError("The Frequency swept width is greater than that of the resonator profile. Reduce the "
                                 "frequency sweep width of the pulse or increase the frequency sweep width of the "
                                 "resonator profile ")

            if not np.any(np.isreal(H)):
                H = np.abs(H)

            profile = interp1d(f, H)(newaxis)

        elif hasattr(self, 'resonator_frequency'):
            f0 = self.resonator_frequency * 1e3
            QL = self.resonator_QL
            profile = np.abs(1 / (1 + 1j * QL * (newaxis / f0 - f0 / newaxis)))

        else:
            raise AttributeError('Pulse object must have `resonator_frequency` or `profile` defined in kwargs')

        if self.fm_func.__name__ == 'uniformq' or self.type == 'sech/tanh':
            profile *= A0

        int = cumtrapz(profile ** -2, nu0, initial=0)
        tf = self.time[-1] * int / int[-1]
        nu_adapted = pchip_interpolate(tf, nu0, self.time)

        self.frequency_modulation = nu_adapted
        self.phase = 2 * np.pi * cumtrapz(self.frequency_modulation, self.time, initial=0)
        self.phase += np.abs(np.min(self.phase))

        if self.fm_func.__name__ == 'uniformq' or self.type == 'sech/tanh':
            self.amplitude_modulation = pchip_interpolate(nu0, A0, nu_adapted)

    def _compute_flip_amp(self):

        if self.fm_func.__name__ == 'none':
            self.amp = self.flip / (2 * np.pi * np.trapz(self.amplitude_modulation, self.time))

        else:
            if self.Qcrit is None:
                self.Qcrit = (2 / np.pi) * np.log(2 / (1 + np.cos(self.flip)))
                self.Qcrit = np.minimum(self.Qcrit, 5)

            if not hasattr(self, 'profile'):
                if self.fm_func.__name__ == 'linear':
                    sweeprate = np.abs(self.freq[1] - self.freq[0]) / self.pulse_time

                elif self.fm_func.__name__ == 'tanh':
                    sweeprate = self.beta * np.abs(self.BWinf) / (2 * self.pulse_time)

                elif self.fm_func.__name__ == 'uniformq':
                    idx = np.argmin(np.abs(self.ti))
                    dnu = np.abs(np.diff(2 * np.pi * self.frequency_modulation / (self.time[1] - self.time[0])))
                    sweeprate = dnu[idx] / (2 * np.pi * (self.frequency_modulation[idx])**2)

            else:
                idx = np.argmin(np.abs(self.ti))
                dnu = np.abs(np.diff(2 * np.pi * self.frequency_modulation / (self.time[1] - self.time[0])))
                sweeprate = dnu[idx] / (2 * np.pi * (self.amplitude_modulation[idx])**2)

            self.amp = np.sqrt(2 * np.pi * self.Qcrit * sweeprate) / (2 * np.pi)

    def _compute_IQ(self):
        self.amplitude_modulation = self.amp * self.amplitude_modulation
        self.frequency_modulation += np.mean(self.freq)
        self.phase = self.phase + 2 * np.pi * np.mean(self.freq) * self.time + self.inp_phase
        self.IQ = self.amplitude_modulation * np.exp(1j * self.phase)



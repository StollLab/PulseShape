import numpy as np
from scipy.interpolate import interp1d, pchip_interpolate
from scipy.integrate import cumtrapz
from .modulations import AmplitudeModulations, FrequencyModulations
from .utils import sop


def nextpow2(x):
    return 1 if x == 0 else int(np.ceil(np.log2(x)))


class Pulse:
    """
    A pulse object contains everything that needs to be known about a pulse.
    """

    def __init__(self, pulse_time, time_step=None, flip=np.pi, mwFreq=33.80,
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

        self.inp_kwargs = kwargs.copy()
        ntype = len(type.split('/'))
        if ntype == 2:
            am, fm = type.split('/')
            self.am_func, self.fm_func = AmplitudeModulations[am], FrequencyModulations[fm]
        elif ntype == 1:
            self.am_func, self.fm_func = AmplitudeModulations[type], FrequencyModulations['none']
        else:
            raise ValueError('Pulse object accepts only one amplitude modulation and one frequency modulation')

        self.flip = flip
        if self.flip > np.pi:
            raise ValueError("flip angle should be less than or equal to pi")
        self.amp = amp
        self.Qcrit = Qcrit

        self.inp_phase = phase
        self.type = type
        self.n = kwargs.get('n', 1)
        self.mwFreq = mwFreq
        self.freq = freq
        self.pulse_time = pulse_time

        self.time_step = time_step
        if self.time_step is None:
            self.oversample_factor = kwargs.get('oversample_factor', 10)
            self.estimate_timestep()

        self.__dict__.update(kwargs)
        if hasattr(self, 'profile'):
            self.profile = self.profile if len(self.profile) == 2 else self.profile.T

        self.time = np.arange(0, self.pulse_time + self.time_step, self.time_step)
        self.ti = self.time - self.pulse_time / 2

        self._shape()
        if (hasattr(self, 'profile') and self.fm_func.__name__ != 'none') or hasattr(self, 'resonator_frequency'):
            self.bw_comp()

        if self.amp is None:
            self._compute_flip_amp()

        self._compute_IQ()
        self._exciteprofile()

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
            QL = self.resonator_ql
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

    def estimate_timestep(self):
        if self.fm_func.__name__ == 'none':
            FM_BW = 0
        else:
            FM_BW = np.abs(self.freq[1] -self.freq[0])

        dt = 1e-4
        tpulse = Pulse(time_step=dt, pulse_time=self.pulse_time, flip=self.flip, amp=1,
                       mwFreq=self.mwFreq, Qcrit=self.Qcrit, freq=self.freq,
                       phase=self.inp_phase, type=self.type, **self.inp_kwargs)

        if nextpow2(len(tpulse.time)) < 10:
            zf = 2 ** 10
        else:
            zf = 4 * 2 ** nextpow2(len(tpulse.time))

        A0fft = np.abs(np.fft.fftshift(np.fft.fft(tpulse.amplitude_modulation, zf)))
        f = np.fft.fftshift(np.fft.fftfreq(zf, dt))
        intg = cumtrapz(A0fft, initial=0)
        idx = np.argmin(np.abs(intg - 0.5 * np.max(intg)))
        indbw = np.squeeze(np.argwhere(A0fft[idx:] > 0.1 * max(A0fft)))
        AM_BW = 2 * (f[idx + indbw[-1]] - f[idx])
        BW = max(FM_BW, AM_BW)

        maxFreq = max(abs(np.mean(self.freq) + np.array([-1, 1]) * BW / 2))
        if maxFreq != 0:
            nyquist_dt = 1 / (2 * maxFreq)
            self.time_step = nyquist_dt / self.oversample_factor
        else:
            self.time_step = 0.002

        if self.time_step > self.pulse_time:
            self.time_step = self.pulse_time

        self.time_step = self.pulse_time / np.rint(self.pulse_time / self.time_step)

    def save_bruker(self, filename, shape_number=10):
        if filename[-4:] != '.shp':
            filename += '.shp'
        fshort = filename.split('/')[-1]
        with open(filename, 'w') as f:
            f.write(f'begin shape{shape_number} "{fshort}"\n')
            for C in self.IQ:
                f.write(f'{C.real:1.5e},{C.imag:1.5e}\n')
            f.write(f'end shape{shape_number}\n')

    def _exciteprofile(self):

        if not hasattr(self, 'nOffsets'):
            self.nOffsets = 201


        if not hasattr(self, 'offsets'):
            if nextpow2(len(self.time)) < 10:
                zf = 2 ** 10
            else:
                zf = 4 * 2 **nextpow2(len(self.time))

            IQft = np.abs(np.fft.fftshift(np.fft.fft(self.IQ, zf)))
            f = np.fft.fftfreq(zf, self.time_step)
            indbw = np.argwhere(IQft > 0.5 * max(IQft))
            bw = abs(f[indbw[-1]] - f[indbw[0]])
            center_freq = np.mean([f[indbw[-1]], f[indbw[0]]])
            self.offsets = np.squeeze(np.linspace(-bw, bw, self.nOffsets) + center_freq)

        self.offsets = np.atleast_1d(self.offsets)
        npoints = len(self.time)
        noffsets = len(self.offsets)
        Sx, Sy, Sz = sop(0.5, ['x', 'y', 'z'])

        Density0 = -Sz

        Mag = np.zeros((3, noffsets))

        for iOs, offset in enumerate(self.offsets):

            Ham0 = offset * Sz

            if min(self.IQ) == max(self.IQ):

                Ham = self.IQ.real[0] * Sx + self.IQ.imag[0] * Sy + Ham0
                tp = (self.time[1] - self.time[0]) * (npoints - 1)
                M = -2j * np.pi * tp * Ham

                q = np.sqrt(M[0, 0] ** 2 - abs(M[0, 1]) ** 2)
                if abs(q) < 1e-10:
                    UPulse = np.eye(2) + M
                else:
                    UPulse = np.cosh(q) * np.eye(2) + (np.sinh(q) / q) * M

            else:
                eye2 = np.eye(2)
                UPulse = eye2

                for it in range(0, npoints-1):
                    Ham = self.IQ.real[it] * Sx + self.IQ.imag[it] * Sy + Ham0
                    M = -2j * np.pi * (self.time[1]-self.time[0]) * Ham
                    q = np.sqrt(M[0, 0] ** 2 - abs(M[0, 1]) ** 2)

                    if abs(q) < 1e-10:
                        dU = eye2 + M
                    else:
                        dU = np.cosh(q) * eye2 + (np.sinh(q) / q) * M

                    UPulse = dU * UPulse

            density = UPulse @ Density0 @ UPulse.H

            Mag[0, iOs] = -2 * (Sx * density.T).sum().real
            Mag[1, iOs] = -2 * (Sy * density.T).sum().real
            Mag[2, iOs] = -2 * (Sz * density.T).sum().real

        self.Mag = Mag



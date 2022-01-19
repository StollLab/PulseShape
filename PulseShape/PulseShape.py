import numpy as np
from scipy.interpolate import interp1d, pchip_interpolate
from scipy.integrate import cumtrapz
from .modulations import AmplitudeModulations, FrequencyModulations
from .utils import sop, pulse_propagation, transmitter


def nextpow2(x):
    """Clone of MATLAB's nextpow function"""
    return 1 if x == 0 else int(np.ceil(np.log2(x)))


eps = np.finfo(float).eps


class Pulse:
    """A pulse object contains everything that needs to be known about a pulse."""

    def __init__(self, pulse_time, time_step=None, flip=np.pi, type='rectangular',
                 mwFreq=33.80, amp=None, Qcrit=None, freq=0, phase=0,
                 exciteprofile=True, **kwargs):
        """

        Parameters
        ----------
        pulse_time : float
            Length of pulse in us
        time_step : float
            Time increment of pulse shape in us. Usually defined by the AWG time step. e.g. Bruker SpinJet time_step =
            0.000625 (625 ns)
        flip : float
            Pulse flip angle in radians.
        shape : str
            Name of pulse shape as a string with either 'Amplitude/frequency' or just 'Amplitude'.
        mwFreq : float
            Microwave source base frequency in GHz.
        amp : flaot
            Pulse maximum amplitude in MHz.
        Qcrit : float
            Critical adiabtacity.
        freq : float, Array-like
            Pulse frequency offset/bandwidth in MHz.
        phase : float
            Pulse Phase in radians.
        exciteprofile : Array-like
            Resonator excitation profile. Matrix with the first row corresponding to the source frequency and the
            second corresponding to
        **kwargs : dict
            Additional arguments necessary for Amplitude/Frequency modulation shapes. For a list of AM/FM functions and
            their parameters run ``
        """

        # Copy input args in case needed
        self.inp_kwargs = kwargs.copy()
        self.trajectory = kwargs.get('trajectory', False)
        self.M0 = kwargs.get('M0', [0, 0, 1])

        # Separate FM and AM shapes
        ntype = len(type.split('/'))

        # Check if user supplied IQ
        userIQ = True if any(x in kwargs for x in ['I', 'Q', 'IQ']) else False

        # if not, get the mod functions from dictionaries
        if not userIQ:
            if ntype == 2:
                am, fm = type.split('/')
                if len(am.split('*')) == 1:
                    self.am_func, self.fm_func = AmplitudeModulations[am], FrequencyModulations[fm]
                else:
                    ams = am.split('*')
                    def t_func(Pulse):
                        return np.prod([AmplitudeModulations[am](Pulse) for am in ams], axis=0)
                    self.am_func = t_func
                    self.fm_func = FrequencyModulations[fm]
            # If only one type passed assume it is an am_func
            elif ntype == 1:
                self.am_func, self.fm_func = AmplitudeModulations[type], FrequencyModulations['none']
            else:
                raise ValueError('Pulse object accepts only one amplitude modulation and one frequency modulation')

        # Ensure flip angle is between 0 and pi radians
        self.flip = flip
        if self.flip > np.pi:
            raise ValueError("flip angle should be less than or equal to pi")

        # Assign amplitude related variabels
        self.amp = amp
        self.Qcrit = Qcrit

        # Assign misc variables
        self.inp_phase = phase
        self.type = type
        self.n = kwargs.get('n', 1)
        self.mwFreq = mwFreq
        self.freq = freq
        self.pulse_time = pulse_time

        # Assign or calculate time step
        self.time_step = time_step
        if self.time_step is None and not userIQ:
            self.oversample_factor = kwargs.get('oversample_factor', 10)
            self.estimate_timestep()

        # Assign any remain variables passed by kwargs
        self.__dict__.update(kwargs)

        # If resonator profile is passed, make sure it is a 2xn array
        if hasattr(self, 'profile'):
            self.profile = self.profile if len(self.profile) == 2 else self.profile.T

        # Calculate time domain
        if not userIQ:
            self.time = np.arange(0, self.pulse_time + self.time_step - eps, self.time_step)
            self.ti = self.time - self.pulse_time / 2

        # Calculate shape
        self._shape()

        # Perform resonator compensation if profile is provided
        if (hasattr(self, 'profile') and self.fm_func.__name__ != 'none') or hasattr(self, 'resonator_frequency'):
            self.bw_comp()

        # Compute amplitude if it is no provided
        if self.amp is None:
            self._compute_flip_amp()

        # Compute IQ
        if not hasattr(self, 'IQ'):
            self._compute_IQ()

        if exciteprofile:
            self.exciteprofile()

    def _shape(self):
        """Calculate shape of amplitude and frequency modulations if IQ not provided by user, otherwise set the
        time domain"""

        # Check if IQ provided by user and determine if I, Q or IQ were passed
        if any(x in self.__dict__ for x in ['I', 'Q', 'IQ']):
            lenIQ = [len(self.__dict__[X]) for X in ['I', 'Q'] if X in self.__dict__][0]
            IQ = self.__dict__.get('IQ', None)
            if IQ is None:
                Q = self.__dict__.get('Q', np.zeros(lenIQ))
                I = self.__dict__.get('I', np.zeros(lenIQ))
                IQ = I + 1j * Q

            # Calculate time/time step from pulse time and IQ
            self.time = np.linspace(0, self.pulse_time, len(IQ))
            if self.time_step is not None:
                new_time = np.arange(0, self.pulse_time + self.time_step - eps, self.time_step)
                new_IQ = interp1d(self.time, IQ)(new_time)
                self.time = new_time
                self.IQ = new_IQ
            else:
                self.time_step = np.mean(np.diff(self.time))
                self.IQ = IQ

            # Convert amplitude if provided by user
            if self.amp is not None:
                self.IQ = self.amp * self.IQ
            else:
                self.amp = np.max(self.IQ)

            # Assign empty variabels if user provided IQ
            self.amplitude_modulation = []
            self.frequency_modulation = []
            self.phase = []

        # Calculate am and fm from functions
        else:
            self.amplitude_modulation = self.am_func(self)
            self.frequency_modulation, self.phase = self.fm_func(self)

    def bw_comp(self):
        """Calculate resonator profile compensation and apply to pulse shape"""

        nu0 = self.frequency_modulation.copy()
        A0 = self.amplitude_modulation.copy()
        newaxis = nu0 + np.mean(self.freq) + self.mwFreq * 1e3

        # Calculate for user supplied profile
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

        # Calculate for user specified resonator frequency
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
        """Compute flip angle if not supplied by user"""

        if self.fm_func.__name__ == 'none':
            self.amp = self.flip / (2 * np.pi * np.trapz(self.amplitude_modulation, self.time))
        else:
            if self.Qcrit is None:
                with np.errstate(divide='ignore'):
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
                    sweeprate = dnu[idx] / (2 * np.pi * (self.amplitude_modulation[idx])**2)

            else:
                idx = np.argmin(np.abs(self.ti))
                dnu = np.abs(np.diff(2 * np.pi * self.frequency_modulation / (self.time[1] - self.time[0])))
                sweeprate = dnu[idx] / (2 * np.pi * (self.amplitude_modulation[idx])**2)

            self.amp = np.sqrt(2 * np.pi * self.Qcrit * sweeprate) / (2 * np.pi)

    def _compute_IQ(self):
        """Compute IQ if not supplied by user"""
        self.amplitude_modulation = self.amp * self.amplitude_modulation
        self.frequency_modulation += np.mean(self.freq)
        self.phase = self.phase + 2 * np.pi * np.mean(self.freq) * self.time + self.inp_phase
        self.IQ = self.amplitude_modulation * np.exp(1j * self.phase)

        # If the pulse is told to be applied in the y axis
        if self.inp_kwargs.get('axis', 'x') == 'y':
            self.IQ = self.IQ.imag + 1j * self.IQ.real

    def estimate_timestep(self):
        """Estimate times step based off of Nyquist theorem"""
        if self.fm_func.__name__ == 'none':
            FM_BW = 0
        else:
            FM_BW = np.abs(self.freq[1] - self.freq[0])

        # Guess initial dt
        dt = 1e-4
        tpulse = Pulse(time_step=dt, pulse_time=self.pulse_time, flip=self.flip, amp=1,
                       mwFreq=self.mwFreq, Qcrit=self.Qcrit, freq=self.freq,
                       phase=self.inp_phase, type=self.type, exciteprofile=False, **self.inp_kwargs)

        if nextpow2(len(tpulse.time)) < 10:
            zf = 2 ** 10
        else:
            zf = 4 * 2 ** nextpow2(len(tpulse.time))

        # perform FFT and calculate bandwidth
        A0fft = np.abs(np.fft.fftshift(np.fft.fft(tpulse.amplitude_modulation, zf)))
        f = np.fft.fftshift(np.fft.fftfreq(zf, dt))
        intg = cumtrapz(A0fft, initial=0)
        idx = np.argmin(np.abs(intg - 0.5 * np.max(intg)))
        indbw = np.squeeze(np.argwhere(A0fft[idx:] > 0.1 * max(A0fft)))
        AM_BW = 2 * (f[idx + indbw[-1] + 1] - f[idx])
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
        """
        Parameters
        ----------
        filename : str
            name of file
        shape_number : int
            Shape number to save. Must be larger than 9
        """

        # Ensure file has correct prefix
        if filename[-4:] != '.shp':
            filename += '.shp'

        # Ensure shape number is acceptable
        if not 10 <= shape_number <=1_000_000:
            raise ValueError('Shape number must be between 10 and 6,000,000')


        # Normalize IQ
        IQ = self.IQ / self.IQ.max()

        # Write file
        fshort = filename.split('/')[-1]
        with open(filename, 'w') as f:
            f.write(f'begin shape{shape_number} "{fshort}"\n')
            for C in IQ:
                f.write(f'{C.real:1.5e},{C.imag:1.5e}\n')
            f.write(f'end shape{shape_number}\n')

    def exciteprofile(self, offsets=None):
        """
        Calculate the pulse excitation profile over the provided frequency offset range
        Parameters
        ----------
        offsets: ndarray
            Range of frequency offsets for calculation of the excitation profile. If not provided, it will be calculated
            over the estimated bandwidth of the pulse
        """
        # Assign self.offsets if user provides new offset when calling exciteprofile
        if offsets is not None:
            self.offsets = offsets

        # Default offsets to 201 elements
        if not hasattr(self, 'nOffsets'):
            self.nOffsets = 201

        # Calculate offsets from expected bandwidth if not provided
        if not hasattr(self, 'offsets'):
            if nextpow2(len(self.time)) < 10:
                zf = 2 ** 10
            else:
                zf = 4 * 2 **nextpow2(len(self.time))

            IQft = np.abs(np.fft.fftshift(np.fft.fft(self.IQ, zf)))
            f = np.fft.fftshift(np.fft.fftfreq(zf, self.time_step))
            indbw = np.argwhere(IQft > 0.5 * max(IQft))
            bw = abs(f[indbw[-1]] - f[indbw[0]])
            center_freq = np.mean([f[indbw[-1]], f[indbw[0]]])
            self.offsets = np.squeeze(np.linspace(-bw, bw, self.nOffsets) + center_freq)

        # Calculate excitation profile
        self.offsets = np.atleast_1d(self.offsets)
        self.M = pulse_propagation(self, M0=self.M0, trajectory=self.trajectory)
        self.Mx, self.My, self.Mz = np.moveaxis(self.M, -1, 0)

    def transmitter(self, Ain, Aout):
        self.IQ = transmitter(self.IQ, Ain, Aout, task='compensate')
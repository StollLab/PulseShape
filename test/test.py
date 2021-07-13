import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from PulseShape import Pulse

sigma2fwhm = 2.35482004503
fwhm2sigma = 1 / sigma2fwhm

def test_bwcomp():
    profile = np.loadtxt('data/Transferfunction.dat')
    pulse = Pulse(0.150, 0.000625, np.pi, freq=[40, 120], type='sech/tanh', beta=10, profile=profile)

    ans = np.genfromtxt("data/sechtanh.csv", delimiter=',')
    ans = ans[:, 0] + 1j * ans[:, 1]
    np.testing.assert_almost_equal(pulse.IQ, ans)


def test_gaussian_rd():
    profile = np.loadtxt('data/Transferfunction.dat').T

    pulse = Pulse(pulse_time=0.060,
                    time_step=0.000625,
                    flip=np.pi,
                    type='gaussian',
                    profile=profile,
                    trunc=0.1)

    ans = np.genfromtxt("data/gaussian.csv", delimiter=',', dtype=complex,
                        converters={0: lambda x: complex(x.decode('utf8').replace('i', 'j'))})
    np.testing.assert_almost_equal(pulse.IQ, ans)


def test_rectrangular():
    pulse = Pulse(pulse_time=0.03,
                  flip=np.pi,
                  time_step=0.001)

    Amp = (pulse.flip / pulse.pulse_time) / (2 * np.pi)
    t0 = np.arange(0, pulse.pulse_time + pulse.time_step, pulse.time_step)
    IQ0 = np.ones(len(t0)) * Amp

    np.testing.assert_almost_equal(pulse.IQ.real, IQ0)


def test_sechtanh():

    pulse = Pulse(pulse_time=0.200,
                  type='sech/tanh',
                  freq=[120, 0],
                  beta=10.6,
                  flip=np.pi,
                  time_step=0.0005)

    t0 = np.arange(0, pulse.pulse_time + pulse.time_step, pulse.time_step)
    dFreq = pulse.freq[1] - pulse.freq[0]
    dt = t0-pulse.pulse_time/2

    Qcrit=5
    BW = dFreq/np.tanh(pulse.beta/2)
    Amp = np.sqrt((pulse.beta * np.abs(BW)*Qcrit) / (2 * np.pi*2*pulse.pulse_time))
    A = 1/np.cosh((pulse.beta / pulse.pulse_time)*(t0 - pulse.pulse_time / 2))
    f = (dFreq / (2 * np.tanh(pulse.beta / 2)))*np.tanh((pulse.beta / pulse.pulse_time) * dt)

    phi = (dFreq/(2*np.tanh(pulse.beta/2)))*(pulse.pulse_time/pulse.beta) * np.log(np.cosh((pulse.beta/pulse.pulse_time) * dt))
    phi = 2 * np.pi * (phi + np.mean(pulse.freq) * t0)
    IQ0 = Amp * A * np.exp(1j*phi)

    np.testing.assert_almost_equal(pulse.IQ, IQ0)
    np.testing.assert_almost_equal(pulse.amplitude_modulation, Amp * A)
    np.testing.assert_almost_equal(pulse.frequency_modulation, f + np.mean(pulse.freq))
    np.testing.assert_almost_equal(phi, pulse.phase)

def test_gaussian():
    pulse = Pulse(pulse_time=0.200,
                  type='gaussian',
                  tFWHM=0.064,
                  amp=((np.pi / 0.064)/(2 * np.pi)),
                  freq=100,
                  time_step=0.0001)

    t0 = np.arange(0, pulse.pulse_time + pulse.time_step, pulse.time_step)
    A = norm(pulse.pulse_time / 2, pulse.tFWHM * fwhm2sigma).pdf(t0)
    A = pulse.amp * (A / max(A))
    f = np.cos(2 * np.pi * pulse.freq * t0) + 1j * np.sin(2 * np.pi * pulse.freq * t0)
    IQ0 = A*f

    np.testing.assert_almost_equal(pulse.IQ, IQ0)

def test_gaussian2():
    dt = 0.001
    t0 = np.arange(-0.300, 0.300 + dt, dt)
    A = norm(0, 0.100 * fwhm2sigma).pdf(t0)
    A = A / max(A)
    ind = np.argwhere(np.rint(np.abs(A-0.5)*1e5) / 1e5 == 0).flatten()
    t0 = t0[ind[0]:ind[1] + 1] - t0[ind[0]]
    IQ0 = A[ind[0]:ind[1] + 1]

    pulse = Pulse(pulse_time=np.round(t0[-1], 12),
                  type='gaussian',
                  trunc=0.5,
                  time_step=dt,
                  amp=1)

    np.testing.assert_almost_equal(pulse.IQ.real, IQ0)

import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d, pchip_interpolate
from scipy.integrate import cumtrapz
from PulseShape import Pulse, sop, transmitter
import pytest

sigma2fwhm = 2.35482004503
fwhm2sigma = 1 / sigma2fwhm


def test_bwcomp():
    profile = np.loadtxt('data/Transferfunction.dat')
    pulse = Pulse(pulse_time=0.150,
                  time_step=0.000625,
                  flip=np.pi,
                  freq=[40, 120],
                  type='sech/tanh',
                  beta=10,
                  profile=profile,
                  mwFreq=33.80,
                  exciteprofile=False)
    ans = np.genfromtxt("data/sechtanh.csv", delimiter=',')
    ans = ans[:, 0] + 1j * ans[:, 1]
    np.testing.assert_almost_equal(pulse.IQ, ans)


def test_bwcomp2():
    pulse = Pulse(pulse_time=0.128,
                  time_step=0.00001,
                  flip=np.pi,
                  freq=[-150, 150],
                  mwFreq=9.5,
                  amp=1,
                  type='quartersin/linear',
                  trise=0.030,
                  oversample_factor=10,
                  exciteprofile=False)

    pulse2 = Pulse(pulse_time=0.128,
                   time_step=0.00001,
                   flip=np.pi,
                   freq=[-150, 150],
                   mwFreq=9.5,
                   resonator_frequency=9.5,
                   resonator_ql=50,
                   amp=1,
                   type='quartersin/linear',
                   trise=0.030,
                   oversample_factor=10,
                  exciteprofile=False)

    f0 = np.arange(9, 10 + 1e-5, 1e-5)
    H = 1 / (1 + 1j * pulse2.resonator_ql * (f0 / pulse2.resonator_frequency - pulse2.resonator_frequency / f0))
    v1 = np.abs(H)

    t0 = np.arange(0, pulse.pulse_time + pulse.time_step, pulse.time_step)
    A = np.ones_like(t0)
    t_part = np.arange(0, pulse.trise + pulse.time_step, pulse.time_step)
    A[:len(t_part)] = np.sin((t_part) * (np.pi / (2 * pulse.trise)))
    A[-len(t_part):] = A[len(t_part) - 1::-1]

    BW = pulse.freq[1] - pulse.freq[0]
    f = -(BW / 2) + (BW / pulse.pulse_time) * t0

    phi = 2 * np.pi * cumtrapz(f, t0, initial=0)
    phi += np.abs(np.min(phi))

    v1_range = interp1d(f0 * 10 ** 3, v1, fill_value=0, bounds_error=False)(f + pulse.mwFreq * 10 ** 3)

    const = np.trapz(1 / v1_range ** 2) / t0[-1]
    t_f = cumtrapz((1 / const) * (1 / v1_range ** 2), initial=0)

    f_adapted = pchip_interpolate(t_f, f + pulse.mwFreq * 10 ** 3, t0)
    f_adapted -= pulse.mwFreq * 10 ** 3
    phi_adapted = 2 * np.pi * cumtrapz(f_adapted, t0, initial=0)
    phi_adapted += np.abs(np.min(phi_adapted))

    IQ0 = A * np.exp(1j * phi)
    IQ0_adapted = A * np.exp(1j * phi_adapted)

    np.testing.assert_almost_equal(IQ0, pulse.IQ)
    np.testing.assert_almost_equal(IQ0_adapted, pulse2.IQ)


def test_bwcomp3():
    pulse = Pulse(pulse_time=0.200,
                  time_step=0.00001,
                  type='sech/tanh',
                  beta=10,
                  freq=[-100, 100],
                  amp=1,
                  exciteprofile=False)

    QL = 60
    f0 = np.arange(9.2, 9.5 + 1e-2, 1e-2)
    dipfreq = 9.35
    v1 = np.abs(1 / (1 + 1j * QL * (f0 / dipfreq - dipfreq / f0)))

    pulse2 = Pulse(pulse_time=0.200,
                   time_step=0.00001,
                   type='sech/tanh',
                   beta=10,
                   freq=[-100, 100],
                   amp=1,
                   mwFreq=9.34,
                   profile=[f0, v1],
                   exciteprofile=False)

    f = np.fft.fftshift(np.fft.fftfreq(len(pulse.time), np.diff(pulse.time).mean()))

    t0 = np.arange(0, pulse.pulse_time + pulse.time_step, pulse.time_step)
    A = (1 / np.cosh(pulse.beta * ((t0 - pulse.pulse_time / 2) / pulse.pulse_time)))
    BWinf = (pulse.freq[1] - pulse.freq[0]) / np.tanh(pulse.beta / 2)

    f = (BWinf / 2) * np.tanh((pulse.beta / pulse.pulse_time) * (t0 - pulse.pulse_time / 2))
    phi = (BWinf / 2) * (pulse.pulse_time / pulse.beta) * \
          np.log(np.cosh((pulse.beta / pulse.pulse_time) * (t0 - pulse.pulse_time / 2)))
    phi = 2 * np.pi * phi

    v1_range = interp1d(f0 * 10 ** 3, v1)(f + pulse2.mwFreq * 10 ** 3)
    v1_range = A * v1_range

    const = np.trapz(1. / v1_range ** 2 / t0[-1], f)
    t_f = cumtrapz((1 / const) * (1. / v1_range ** 2), f, initial=0)

    f_adapted = pchip_interpolate(t_f, f + pulse2.mwFreq * 10 ** 3, t0)
    f_adapted = f_adapted - pulse2.mwFreq * 10 ** 3
    phi_adapted = 2 * np.pi * cumtrapz(f_adapted, t0, initial=0)

    phi_adapted = phi_adapted + abs(min(phi_adapted))
    A_adapted = pchip_interpolate(f, A, f_adapted)

    IQ0 = A * np.exp(1j * phi)
    IQ0_adapted = A_adapted * np.exp(1j * phi_adapted)

    np.testing.assert_almost_equal(pulse.IQ, IQ0)
    np.testing.assert_almost_equal(pulse2.IQ, IQ0_adapted)


def test_estimate_timestep():
    pulse = Pulse(pulse_time=0.128,
                  flip=np.pi,
                  freq=[-50, 50],
                  amp=20,
                  type='quartersin/linear',
                  trise=0.010,
                  oversample_factor=10)
    assert pulse.time_step == 1e-3


def test_linear_chirp():
    pulse = Pulse(pulse_time=0.064,
                  time_step=0.0001,
                  flip=np.pi / 2,
                  freq=[60, 180],
                  type='rectangular/linear')

    t0 = np.arange(0, pulse.pulse_time + pulse.time_step, pulse.time_step)

    BW = pulse.freq[1] - pulse.freq[0]
    Amp = np.sqrt(4 * np.log(2) * BW / pulse.pulse_time) / (2 * np.pi)
    f = -(BW / 2) + (BW / pulse.pulse_time) * t0

    phi = cumtrapz(f, t0, initial=0)
    phi = phi + abs(min(phi))
    print(t0.shape, phi.shape)
    IQ0 = Amp * np.exp(2j * np.pi * (phi + np.mean(pulse.freq) * t0))

    np.testing.assert_almost_equal(pulse.IQ, IQ0)


def test_quartersin_chirp():
    pulse = Pulse(pulse_time=0.128,
                  flip=np.pi,
                  freq=[-50, 50],
                  amp=20,
                  type='quartersin/linear',
                  trise=0.010,
                  oversample_factor=10)

    BW = pulse.freq[1] - pulse.freq[0]
    dt = 1 / (2 * pulse.oversample_factor * BW / 2)
    dt = pulse.pulse_time / (np.rint(pulse.pulse_time / dt))
    t0 = np.arange(0, pulse.pulse_time + dt, dt)

    t_part = np.arange(0, pulse.trise + dt, dt)
    A = np.ones(len(t0))
    A[:len(t_part)] = np.sin((np.pi * t_part) / (2 * pulse.trise))
    A[-len(t_part):] = A[len(t_part) - 1::-1]

    f = -(BW / 2) + (BW / pulse.pulse_time) * t0
    phi = cumtrapz(f, t0, initial=0)
    phi = phi + abs(min(phi))

    IQ0 = pulse.amp * A * np.exp(2j * np.pi * phi)

    np.testing.assert_almost_equal(pulse.IQ, IQ0)
    np.testing.assert_almost_equal(pulse.amp * A, pulse.amplitude_modulation)
    np.testing.assert_almost_equal(f, pulse.frequency_modulation)


def test_halfsin_chirp():
    pulse = Pulse(pulse_time=0.128,
                  flip=np.pi,
                  freq=[-50, 50],
                  amp=20,
                  type='halfsin/linear',
                  oversample_factor=10)

    BW = pulse.freq[1] - pulse.freq[0];
    dt = 1 / (2 * pulse.oversample_factor * BW / 2)
    dt = pulse.pulse_time / (np.rint(pulse.pulse_time / dt))
    t0 = np.arange(0, pulse.pulse_time + dt, dt)

    A = np.sin(np.pi * t0 / pulse.pulse_time);

    f = -(BW / 2) + (BW / pulse.pulse_time) * t0

    phi = cumtrapz(f, t0, initial=0)
    phi = phi + abs(min(phi))

    IQ0 = pulse.amp * A * np.exp(2j * np.pi * phi)

    np.testing.assert_almost_equal(pulse.IQ, IQ0)
    np.testing.assert_almost_equal(pulse.amplitude_modulation, pulse.amp * A)
    np.testing.assert_almost_equal(pulse.frequency_modulation, f)
    np.testing.assert_almost_equal(pulse.IQ, IQ0)


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
    dt = t0 - pulse.pulse_time / 2

    Qcrit = 5
    BW = dFreq / np.tanh(pulse.beta / 2)
    Amp = np.sqrt((pulse.beta * np.abs(BW) * Qcrit) / (2 * np.pi * 2 * pulse.pulse_time))
    A = 1 / np.cosh((pulse.beta / pulse.pulse_time) * (t0 - pulse.pulse_time / 2))
    f = (dFreq / (2 * np.tanh(pulse.beta / 2))) * np.tanh((pulse.beta / pulse.pulse_time) * dt)

    phi = (dFreq / (2 * np.tanh(pulse.beta / 2))) * (pulse.pulse_time / pulse.beta) * np.log(
        np.cosh((pulse.beta / pulse.pulse_time) * dt))
    phi = 2 * np.pi * (phi + np.mean(pulse.freq) * t0)
    IQ0 = Amp * A * np.exp(1j * phi)

    np.testing.assert_almost_equal(pulse.IQ, IQ0)
    np.testing.assert_almost_equal(pulse.amplitude_modulation, Amp * A)
    np.testing.assert_almost_equal(pulse.frequency_modulation, f + np.mean(pulse.freq))
    np.testing.assert_almost_equal(phi, pulse.phase)


def test_gaussian():
    pulse = Pulse(pulse_time=0.200,
                  type='gaussian',
                  tFWHM=0.064,
                  amp=((np.pi / 0.064) / (2 * np.pi)),
                  freq=100,
                  time_step=0.0001)

    t0 = np.arange(0, pulse.pulse_time + pulse.time_step, pulse.time_step)
    A = norm(pulse.pulse_time / 2, pulse.tFWHM * fwhm2sigma).pdf(t0)
    A = pulse.amp * (A / max(A))
    f = np.cos(2 * np.pi * pulse.freq * t0) + 1j * np.sin(2 * np.pi * pulse.freq * t0)
    IQ0 = A * f

    np.testing.assert_almost_equal(pulse.IQ, IQ0)


def test_gaussian2():
    dt = 0.001
    t0 = np.arange(-0.300, 0.300 + dt, dt)
    A = norm(0, 0.100 * fwhm2sigma).pdf(t0)
    A = A / max(A)
    ind = np.argwhere(np.rint(np.abs(A - 0.5) * 1e5) / 1e5 == 0).flatten()
    t0 = t0[ind[0]:ind[1] + 1] - t0[ind[0]]
    IQ0 = A[ind[0]:ind[1] + 1]

    pulse = Pulse(pulse_time=np.round(t0[-1], 12),
                  type='gaussian',
                  trunc=0.5,
                  time_step=dt,
                  amp=1)

    np.testing.assert_almost_equal(pulse.IQ.real, IQ0)


def test_WURST():
    pulse = Pulse(pulse_time=0.500,
                  type='WURST/linear',
                  freq=[-150, 350],
                  amp=15,
                  nwurst=15)

    A = 1 - np.abs((np.sin((np.pi * (pulse.time - pulse.pulse_time / 2)) / pulse.pulse_time)) ** pulse.nwurst)
    BW = np.diff(pulse.freq)[0]
    f = -(BW / 2) + (BW / pulse.pulse_time) * pulse.time
    phi = cumtrapz(f, pulse.time - pulse.pulse_time / 2, initial=0)
    phi += np.abs(min(phi)) + np.mean(pulse.freq) * pulse.time

    IQ0 = pulse.amp * A * np.exp(2j * np.pi * phi)

    np.testing.assert_almost_equal(pulse.IQ, IQ0)
    np.testing.assert_almost_equal(pulse.amplitude_modulation, pulse.amp * A)
    np.testing.assert_almost_equal(pulse.frequency_modulation, f + np.mean(pulse.freq))
    np.testing.assert_almost_equal(pulse.phase, 2 * np.pi * phi)


def test_higherordersech():
    pulse = Pulse(pulse_time=0.600,
                  time_step=0.0005,
                  type='sech/uniformq',
                  freq=[-100, 100],
                  beta=10.6,
                  n=8,
                  amp=20)

    t0 = np.arange(0, pulse.pulse_time + pulse.time_step, pulse.time_step)
    ti = t0 - pulse.pulse_time / 2
    A = 1 / np.cosh((pulse.beta) * (2 ** (pulse.n - 1)) * (ti / pulse.pulse_time) ** pulse.n)
    A = pulse.amp * A
    f = cumtrapz(A ** 2 / np.trapz(A ** 2, ti), ti, initial=0)
    BW = np.diff(pulse.freq)[0]
    f = BW * f - BW / 2

    phi = cumtrapz(f, ti, initial=0)
    phi = phi + abs(min(phi))
    IQ0 = A * np.exp(2j * np.pi * phi)

    np.testing.assert_almost_equal(pulse.IQ, IQ0)
    np.testing.assert_almost_equal(pulse.amplitude_modulation, A)
    np.testing.assert_almost_equal(pulse.frequency_modulation, f + np.mean(pulse.freq))
    np.testing.assert_almost_equal(pulse.phase, 2 * np.pi * phi)


def test_sinc():
    pulse = Pulse(pulse_time=0.200,
                  type='sinc',
                  zerocross=0.034,
                  time_step=0.001,
                  amp=1)

    t0 = np.arange(0, pulse.pulse_time + pulse.time_step, pulse.time_step)
    ti = t0 - pulse.pulse_time / 2
    x = 2 * np.pi * ti / pulse.zerocross
    A = np.sin(x) / x
    A[ti == 0] = 1
    IQ0 = A
    np.testing.assert_almost_equal(pulse.IQ, IQ0)


def test_asymmetric_sech():
    pulse = Pulse(pulse_time=0.100,
                  time_step=5e-4,
                  type='sech/uniformq',
                  freq=[-100, 100],
                  beta=10.4,
                  n=[6, 1],
                  amp=30)

    t0 = np.arange(0, pulse.pulse_time + pulse.time_step, pulse.time_step)
    npts = len(t0)
    ti = t0 - pulse.pulse_time / 2
    A = np.ones(len(t0))
    A[:int(np.rint(npts / 2)) - 1] = 1 / np.cosh((pulse.beta) * (2 ** (pulse.n[0] - 1)) *
                                                 (ti[:int(np.rint(npts / 2)) - 1] / pulse.pulse_time) ** pulse.n[0])
    A[int(np.rint(npts / 2)):] = 1 / np.cosh((pulse.beta) * (2 ** (pulse.n[1] - 1)) *
                                             (ti[int(np.rint(npts / 2)):] / pulse.pulse_time) ** pulse.n[1])

    A = A * pulse.amp
    f = cumtrapz(A ** 2 / np.trapz(A ** 2, ti), ti, initial=0)
    BW = pulse.freq[1] - pulse.freq[0]
    f = BW * f - BW / 2
    phi = cumtrapz(f, ti, initial=0)
    phi += abs(min(phi))
    IQ0 = A * np.exp(2j * np.pi * phi)
    np.testing.assert_almost_equal(pulse.IQ, IQ0)
    np.testing.assert_almost_equal(pulse.amplitude_modulation, A)
    np.testing.assert_almost_equal(pulse.frequency_modulation, f)
    np.testing.assert_almost_equal(2 * np.pi * phi, pulse.phase)


pulses1 = [{'pulse_time': 0.060, 'type': 'rectangular'},
           {'pulse_time': 0.200, 'tFWHM': 0.060, 'type': 'gaussian'},
           {'pulse_time': 0.200, 'zerocross': 0.050, 'type': 'sinc'},
           {'pulse_time': 0.100, 'trise': 0.020, 'type': 'quartersin'},
           {'pulse_time': 0.500, 'beta': 12, 'type': 'sech'},
           {'pulse_time': 0.300, 'nwurst': 20, 'type': 'WURST'}]


@pytest.mark.parametrize('pulse', pulses1)
def test_flip_am(pulse):
    offsets = 0
    tol = 1e-12

    p1 = Pulse(flip=np.pi / 2, offsets=offsets, **pulse)
    p2 = Pulse(flip=np.pi, offsets=offsets, **pulse)
    print(p1.Mz)
    assert np.all(p1.Mz < tol)
    assert np.all(p2.Mz > -1 - tol)
    print(p2.Mz - (-1 + tol))
    assert np.all(p2.Mz < -1 + tol)


pulses2 = [{'type': 'quartersin/linear', 'pulse_time': 0.200, 'trise': 0.050, 'freq': [-250, 250]},
           {'type': 'WURST/linear', 'pulse_time': 0.200, 'nwurst': 30, 'freq': [150, -150]},
           {'type': 'sech/tanh', 'pulse_time': 0.400, 'beta': 10, 'freq': [-35, 35]},
           {'type': 'sech*WURST/tanh', 'pulse_time': 0.500, 'beta': 4, 'nwurst': 8, 'freq': [60, -60]},
           {'type': 'sech/uniformq', 'pulse_time': 0.300, 'beta': 10, 'n': 4, 'freq': [-100, 100]},
           {'type': 'sech/uniformq', 'pulse_time': 0.400, 'beta': 7, 'n': 12, 'freq': [-125, 125]}]


@pytest.mark.parametrize('pulse', pulses2)
def test_flip_amfm(pulse):
    offsets = 0
    tol = 1e-2

    p1 = Pulse(flip=np.pi / 2, offsets=offsets, **pulse)
    p2 = Pulse(flip=np.pi, offsets=offsets, **pulse)

    print(p1.Mz)
    assert p1.Mz < tol
    assert p2.Mz > -1 - tol
    assert p2.Mz < -1 + tol


def test_G3():
    pulse = Pulse(type='G3', pulse_time=0.800, time_step=0.001, amp=1)
    t0 = np.arange(0, pulse.pulse_time + pulse.time_step, pulse.time_step)

    A0 = np.array([-1, 1.37, 0.49])
    x0 = np.array([0.287, 0.508, 0.795]) * pulse.pulse_time
    FWHM = np.array([0.189, 0.183, 0.243]) * pulse.pulse_time

    A = np.zeros_like(t0)
    for An, xn, Fn in zip(A0, x0, FWHM):
        A += An * np.exp(-(4 * np.log(2) / Fn ** 2) * (t0 - xn) ** 2)

    IQ0 = A / max(A)

    np.testing.assert_almost_equal(pulse.IQ, IQ0)


def test_G4():
    pulse = Pulse(type='G4', pulse_time=0.800, time_step=0.001, amp=1)
    t0 = np.arange(0, pulse.pulse_time + pulse.time_step, pulse.time_step)

    A0 = np.array([0.62, 0.72, -0.91, -0.33])
    x0 = np.array([0.177, 0.492, 0.653, 0.892]) * pulse.pulse_time
    FWHM = np.array([0.172, 0.129, 0.119, 0.139]) * pulse.pulse_time

    A = np.zeros_like(t0)
    for An, xn, Fn in zip(A0, x0, FWHM):
        A += An * np.exp(-(4 * np.log(2) / Fn ** 2) * (t0 - xn) ** 2)

    IQ0 = A / max(A)

    np.testing.assert_almost_equal(pulse.IQ, IQ0)


def test_Q3():
    pulse = Pulse(type='Q3', pulse_time=1.200, time_step=0.01, amp=1)
    t0 = np.arange(0, pulse.pulse_time + pulse.time_step, pulse.time_step)

    A0 = np.array([-4.39, 4.57, 2.60])
    x0 = np.array([0.306, 0.545, 0.804]) * pulse.pulse_time
    FWHM = np.array([0.180, 0.183, 0.245]) * pulse.pulse_time

    A = np.zeros_like(t0)
    for An, xn, Fn in zip(A0, x0, FWHM):
        A += An * np.exp(-(4 * np.log(2) / Fn ** 2) * (t0 - xn) ** 2)

    IQ0 = A / max(A)

    np.testing.assert_almost_equal(pulse.IQ, IQ0)


def test_Q5():
    pulse = Pulse(type='Q5', pulse_time=1.200, time_step=0.01, amp=1)
    t0 = np.arange(0, pulse.pulse_time + pulse.time_step, pulse.time_step)

    A0 = np.array([-1.48, -4.34, 7.33, -2.30, 5.66])
    x0 = np.array([0.162, 0.307, 0.497, 0.525, 0.803]) * pulse.pulse_time
    FWHM = np.array([0.186, 0.139, 0.143, 0.290, 0.137]) * pulse.pulse_time

    A = np.zeros_like(t0)
    for An, xn, Fn in zip(A0, x0, FWHM):
        A += An * np.exp(-(4 * np.log(2) / Fn ** 2) * (t0 - xn) ** 2)

    IQ0 = A / max(A)

    np.testing.assert_almost_equal(pulse.IQ, IQ0)


def test_gaussian_cascade():
    pulse = Pulse(type='gaussian_cascade',
                  pulse_time=1.200,
                  time_step=0.01,
                  amp=1,
                  A0=[-0.84, 1.53, -2.12, 5.21],
                  x0=[0.154, 0.358, 0.521, 0.766],
                  FWHM=[0.180, 0.183, 0.195, 0.235])

    t0 = np.arange(0, pulse.pulse_time + pulse.time_step, pulse.time_step)

    A0 = np.array([-0.84, 1.53, -2.12, 5.21])
    x0 = np.array([0.154, 0.358, 0.521, 0.766]) * pulse.pulse_time
    FWHM = np.array([0.180, 0.183, 0.195, 0.235]) * pulse.pulse_time

    A = np.zeros_like(t0)
    for An, xn, Fn in zip(A0, x0, FWHM):
        A += An * np.exp(-(4 * np.log(2) / Fn ** 2) * (t0 - xn) ** 2)

    IQ0 = A / max(A)

    np.testing.assert_almost_equal(pulse.IQ, IQ0)


def test_I_BURP1():
    pulse = Pulse(pulse_time=0.500, type='I_BURP1', time_step=0.001, amp=1)
    t0 = np.arange(0, pulse.pulse_time + pulse.time_step, pulse.time_step)
    A0 = 0.5
    An = [0.70, - 0.15, - 0.94, 0.11, -0.02, -0.04, 0.01, -0.02, -0.01]
    Bn = [-1.54, 1.01, - 0.24, -0.04, 0.08, -0.04, -0.01, 0.01, -0.01]

    A = np.zeros_like(t0) + A0
    for i, (an, bn) in enumerate(zip(An, Bn)):
        j = i + 1
        A += an * np.cos(j * 2 * np.pi * t0 / pulse.pulse_time) + \
             bn * np.sin(j * 2 * np.pi * t0 / pulse.pulse_time)

    A /= max([-min(A), max(A)])
    IQ0 = A / max(A)

    np.testing.assert_almost_equal(pulse.IQ, IQ0)


def test_I_BURP2():
    pulse = Pulse(pulse_time=0.700, type='I_BURP2', time_step=0.001, amp=1)
    t0 = np.arange(0, pulse.pulse_time + pulse.time_step, pulse.time_step)
    A0 = 0.5
    An = [0.81, 0.07, -1.25, -0.24, 0.07, 0.11, 0.05, -0.02, -0.03, -0.02, 0.00]
    Bn = [-0.68, -1.38, 0.20, 0.45, 0.23, 0.05, -0.04, -0.04, 0.00, 0.01, 0.01]

    A = np.zeros_like(t0) + A0
    for i, (an, bn) in enumerate(zip(An, Bn)):
        j = i + 1
        A += an * np.cos(j * 2 * np.pi * t0 / pulse.pulse_time) + \
             bn * np.sin(j * 2 * np.pi * t0 / pulse.pulse_time)

    A /= max([-min(A), max(A)])
    IQ0 = A / max(A)

    np.testing.assert_almost_equal(pulse.IQ, IQ0)


def test_SNOB_i2():
    pulse = Pulse(pulse_time=0.200, type='SNOB_i2', time_step=0.001, amp=1)
    t0 = np.arange(0, pulse.pulse_time + pulse.time_step, pulse.time_step)
    A0 = 0.5
    An = [-0.2687, -0.2972, 0.0989, -0.0010, -0.0168, 0.0009, -0.0017, -0.0013, -0.0014]
    Bn = [-1.1461, 0.4016, 0.0736, -0.0307, 0.0079, 0.0062, 0.0003, -0.0002, 0.0009]

    A = np.zeros_like(t0) + A0
    for i, (an, bn) in enumerate(zip(An, Bn)):
        j = i + 1
        A += an * np.cos(j * 2 * np.pi * t0 / pulse.pulse_time) + \
             bn * np.sin(j * 2 * np.pi * t0 / pulse.pulse_time)

    A /= max([-min(A), max(A)])
    IQ0 = A / max(A)

    np.testing.assert_almost_equal(pulse.IQ, IQ0)


def test_SNOB_i3():
    pulse = Pulse(pulse_time=0.200, type='SNOB_i3', time_step=0.001, amp=1)
    t0 = np.arange(0, pulse.pulse_time + pulse.time_step, pulse.time_step)
    A0 = 0.5
    An = [0.2801, -0.9995, 0.1928, 0.0967, -0.0480, -0.0148, 0.0088, -0.0002, -0.0030]
    Bn = [-1.1990, 0.4893, 0.2439, -0.0816, -0.0409, 0.0234, 0.0036, -0.0042, 0.0001]

    A = np.zeros_like(t0) + A0
    for i, (an, bn) in enumerate(zip(An, Bn)):
        j = i + 1
        A += an * np.cos(j * 2 * np.pi * t0 / pulse.pulse_time) + \
             bn * np.sin(j * 2 * np.pi * t0 / pulse.pulse_time)

    A /= max([-min(A), max(A)])
    IQ0 = A / max(A)

    np.testing.assert_almost_equal(pulse.IQ, IQ0)


def test_fourier_series():
    pulse = Pulse(pulse_time=2.00,
                  type='fourier_series',
                  time_step=0.001,
                  amp=1,
                  A0=0.308,
                  An=[1.017, -0.480, -1.033, 0.078, 0.103, 0.109, 0.027, -0.043, -0.018, 0.000, 0.005, 0.004],
                  Bn=[-0.384, -1.894, 0.574, 0.409, 0.098, 0.009, -0.079, -0.024, 0.014, 0.010, 0.003, -0.001])

    t0 = np.arange(0, pulse.pulse_time + pulse.time_step, pulse.time_step)
    A = np.zeros_like(t0) + pulse.A0
    for i, (an, bn) in enumerate(zip(pulse.An, pulse.Bn)):
        j = i + 1
        A += an * np.cos(j * 2 * np.pi * t0 / pulse.pulse_time) + \
             bn * np.sin(j * 2 * np.pi * t0 / pulse.pulse_time)

    A /= max([-min(A), max(A)])
    IQ0 = A / max(A)

    np.testing.assert_almost_equal(pulse.IQ, IQ0)


def test_userIQ():
    pulse = Pulse(type='sech/tanh', pulse_time=0.200, freq=[120, 0], beta=10.6, Qcrit=5, time_step=5e-4)

    dt = 0.0001
    t0 = np.arange(0, pulse.pulse_time + pulse.time_step, pulse.time_step)
    BW = (pulse.freq[1] - pulse.freq[0]) / np.tanh(pulse.beta / 2)
    Amplitude = np.sqrt((pulse.beta * abs(BW) * pulse.Qcrit) / (2 * np.pi * 2 * pulse.pulse_time))
    A = 1 / (np.cosh((pulse.beta / pulse.pulse_time) * (t0 - pulse.pulse_time / 2)))

    f = ((pulse.freq[1] - pulse.freq[0]) / (2 * np.tanh(pulse.beta / 2))) * \
        np.tanh((pulse.beta / pulse.pulse_time) * (t0 - pulse.pulse_time / 2))

    phi = ((pulse.freq[1] - pulse.freq[0]) / (2 * np.tanh(pulse.beta / 2))) * (pulse.pulse_time / pulse.beta) \
          * np.log(np.cosh((pulse.beta / pulse.pulse_time) * (t0 - pulse.pulse_time / 2)))

    phi = 2 * np.pi * (phi + np.mean(pulse.freq) * t0)
    IQ0 = Amplitude * A * np.exp(1j * phi)

    pulse2 = Pulse(pulse_time=0.200, I=IQ0.real, Q=IQ0.imag, time_step=pulse.time_step)
    np.testing.assert_almost_equal(pulse2.IQ, pulse.IQ)


def test_exciteprofile1():
    p1 = Pulse(flip=np.pi / 2, pulse_time=0.016, phase=np.pi / 2, exciteprofile=False)
    p2 = Pulse(flip=np.pi, pulse_time=0.032, phase=np.pi / 2, exciteprofile=False)
    offsets = np.arange(-70, 70.5, 0.5)
    p1.exciteprofile(offsets)
    p2.exciteprofile(offsets)

    for p in [p1, p2]:
        Sx, Sy, Sz = sop(0.5, ['x', 'y', 'z'])
        Amplitude = (p.flip / p.pulse_time) / (2 * np.pi)

        H = np.einsum('i,jk->ijk', offsets, Sz) + (Amplitude * Sy)[None, :]
        M = -2j * np.pi * H * p.pulse_time
        q = np.sqrt(M[:, 0, 0] ** 2 - np.abs(M[:, 0, 1]) ** 2)
        coshterm = np.einsum('i,jk->ijk', np.cosh(q), np.eye(2))

        U = coshterm + (np.sinh(q) / q)[:, None, None] * M
        rho = np.einsum('ijk,kl->ijl', U, -Sz)
        rho = np.einsum('ijk,ikl->ijl', rho, U.conj().transpose(0, 2, 1))

        Mx = -2 * np.trace(np.einsum('ij,ljk->lik', Sx, rho), axis1=1, axis2=2)
        My = -2 * np.trace(np.einsum('ij,ljk->lik', Sy, rho), axis1=1, axis2=2)
        Mz = -2 * np.trace(np.einsum('ij,ljk->lik', Sz, rho), axis1=1, axis2=2)

        np.testing.assert_almost_equal(Mx, p.Mx)
        np.testing.assert_almost_equal(My, p.My)
        np.testing.assert_almost_equal(Mz, p.Mz)


def test_exciteprofile2():
    p1 = Pulse(type='sinc', flip=np.pi, pulse_time=0.2, zerocross=0.05)
    offsets = np.arange(-100, 101)
    p1.exciteprofile(offsets)
    Mz = np.loadtxt('data/Mz.csv', delimiter=',')

    np.testing.assert_almost_equal(Mz, p1.Mz)


def test_exciteprofile3():
    p1 = Pulse(type='sech/tanh', flip=np.pi, pulse_time=0.2, time_step=5e-4, freq=[110, 10], beta=15)

    Mz = np.loadtxt('data/Mz2.csv', delimiter=',')

    np.testing.assert_almost_equal(Mz, p1.Mz, decimal=3)


def test_exciteprofile4():
    # TODO: Turn into a real test.
    p1 = Pulse(flip=np.pi, pulse_time=0.012, time_step=5e-4, M0=[0, 0, 1], trajectory=True)
    p2 = Pulse(flip=np.pi, pulse_time=0.012, time_step=5e-4, M0=[0, 0, 1], trajectory=True)

    p2.IQ = p2.IQ.imag + 1j * p2.IQ.real

    p1.exciteprofile(offsets=0)
    p2.exciteprofile(offsets=0)

    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure(figsize=(4, 4))
    #
    # ax = fig.add_subplot(111, projection='3d')
    #
    # ax.scatter(p1.Mx[0], p1.My[0], p1.Mz[0], c=p1.time, cmap='viridis')
    # ax.scatter(p2.Mx[0], p2.My[0], p2.Mz[0], c=p1.time, cmap='viridis')
    #
    # plt.show()
    #
    # plt.title('p1')
    # plt.plot(p1.IQ.real)
    # plt.plot(p1.IQ.imag)
    # plt.show()
    #
    # plt.title('p2')
    # plt.plot(p2.IQ.real)
    # plt.plot(p2.IQ.imag)
    # plt.show()


def test_excite_userIQ():
    dt = 1e-3
    tp = 8e-3
    v1 = ((np.pi / 2) / tp) / (2 * np.pi)
    I = np.ones(int(6 * tp / dt) + 1)
    I[int(tp / dt):int(3 * tp / dt)] *= -1
    pulse = Pulse(pulse_time=6 * tp, I=v1 * I)

    Sx, Sy, Sz = sop(0.5, ['x', 'y', 'z'])

    H1 = np.einsum('i,jk->ijk', pulse.offsets, Sz) + (v1 * Sx)[None, :]
    M1 = -2j * np.pi * tp * H1
    q1 = np.sqrt(M1[:, 0, 0] ** 2 - np.abs(M1[:, 0, 1]) ** 2)
    U1 = np.einsum('i,jk->ijk', np.cosh(q1), np.eye(2)) + (np.sinh(q1) / q1)[:, None, None] * M1

    H2 = np.einsum('i,jk->ijk', pulse.offsets, Sz) - (v1 * Sx)[None, :]
    M2 = -2j * np.pi * tp * 2 * H2
    q2 = np.sqrt(M2[:, 0, 0] ** 2 - np.abs(M2[:, 0, 1]) ** 2)
    U2 = np.einsum('i,jk->ijk', np.cosh(q2), np.eye(2)) + (np.sinh(q2) / q2)[:, None, None] * M2

    H3 = np.einsum('i,jk->ijk', pulse.offsets, Sz) + (v1 * Sx)[None, :]
    M3 = -2j * np.pi * tp * 3 * H3
    q3 = np.sqrt(M3[:, 0, 0] ** 2 - np.abs(M3[:, 0, 1]) ** 2)
    U3 = np.einsum('i,jk->ijk', np.cosh(q3), np.eye(2)) + (np.sinh(q3) / q3)[:, None, None] * M3

    U = np.einsum('ijk,ikl->ijl', U1, U2)
    U = np.einsum('ijk,ikl->ijl', U, U3)

    rho = np.einsum('ijk,kl->ijl', U, -Sz)
    rho = np.einsum('ijk,ikl->ijl', rho, U.conj().transpose(0, 2, 1))

    Mz = -2 * np.trace(np.einsum('ij,ljk->lik', Sz, rho), axis1=1, axis2=2)

    np.testing.assert_almost_equal(Mz, pulse.Mz)


def test_save_bruker():
    profile = np.loadtxt('data/Transferfunction.dat').T

    pulse = Pulse(0.150, 0.000625, np.pi, freq=[40, 120], type='sech/tanh', beta=10, profile=profile)

    pulse.save_bruker('data/istmp.shp')

def test_multiple_M0():
    offsets = np.linspace(-50, 50, 3)
    P1 = Pulse(pulse_time=0.016, flip=np.pi/2, offsets=offsets, trajectory=True)
    P1.exciteprofile(offsets=offsets)
    ans = np.load('data/multiple_M0.npy')
    np.testing.assert_almost_equal(P1.M, ans)

def test_multiple_M02():
    offsets = np.linspace(-50, 50, 10)
    P1 = Pulse(pulse_time=0.016, flip=np.pi/2, offsets=offsets, trajectory=True)
    P1.exciteprofile(offsets=offsets)

def test_accumulate_M():
    P1 = Pulse(pulse_time=0.300,
               time_step=1e-4,
               flip=np.pi,
               freq=[-40, 40],
               type='sech/tanh',
               beta=10,
               offsets=np.linspace(-50, 50, 201),
               trajectory=True)

    ans = np.load('data/test_accumulate.npy')
    np.testing.assert_allclose(P1.M, ans)

def test_transmitter_compression():
    nu1_max = 15
    P1 = Pulse(type='sech/none', time_step=2.5e-4, pulse_time=0.2, beta=10, amp=nu1_max)

    InputAmplitude = 0.75
    Ain = np.linspace(0, 1, 1001)
    Aout = nu1_max * (Ain - 0.30 * Ain**3)
    IQ_compressed = transmitter(InputAmplitude * P1.IQ / P1.IQ.max(), Ain, Aout, 'simulate')
    ans = np.load('data/IQcompressed.npy')

    np.testing.assert_allclose(IQ_compressed, ans)

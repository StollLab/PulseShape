import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d, pchip_interpolate
from scipy.integrate import cumtrapz
from PulseShape import Pulse

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
                  mwFreq=33.80)

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
                  oversample_factor=10)
    
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
                   oversample_factor=10)


    f0 = np.arange(9, 10 + 1e-5, 1e-5)
    H = 1/(1 + 1j * pulse2.resonator_ql * (f0 / pulse2.resonator_frequency - pulse2.resonator_frequency / f0))
    v1 = np.abs(H)


    t0 = np.arange(0, pulse.pulse_time + pulse.time_step, pulse.time_step)
    A = np.ones_like(t0)
    t_part = np.arange(0, pulse.trise + pulse.time_step, pulse.time_step)
    A[:len(t_part)] = np.sin((t_part) * (np.pi / (2 * pulse.trise)))
    A[-len(t_part):] = A[len(t_part) - 1::-1]

    BW = pulse.freq[1] - pulse.freq[0]
    f = -(BW/2) + (BW/pulse.pulse_time) * t0

    phi = 2 * np.pi * cumtrapz(f, t0, initial=0)
    phi += np.abs(np.min(phi))

    v1_range = interp1d(f0 * 10**3, v1, fill_value=0, bounds_error=False)(f + pulse.mwFreq * 10**3)

    const = np.trapz(1/v1_range**2) / t0[-1]
    t_f = cumtrapz((1 / const) * (1 / v1_range**2), initial=0)

    f_adapted = pchip_interpolate(t_f, f + pulse.mwFreq * 10**3, t0)
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
                  amp=1)

    QL = 60
    f0 = np.arange(9.2, 9.5 + 1e-2, 1e-2)
    dipfreq=9.35
    v1 = np.abs(1 / (1 + 1j * QL * (f0 / dipfreq - dipfreq / f0)))

    pulse2 = Pulse(pulse_time=0.200,
                   time_step=0.00001,
                   type='sech/tanh',
                   beta=10,
                   freq=[-100, 100],
                   amp=1,
                   mwFreq=9.34,
                   profile=[f0, v1])

    f = np.fft.fftshift(np.fft.fftfreq(len(pulse.time), np.diff(pulse.time).mean()))

    t0 = np.arange(0, pulse.pulse_time + pulse.time_step, pulse.time_step)
    A = (1 / np.cosh(pulse.beta * ((t0 - pulse.pulse_time / 2) / pulse.pulse_time)))
    BWinf = (pulse.freq[1] - pulse.freq[0]) / np.tanh(pulse.beta / 2)

    f = (BWinf / 2) * np.tanh((pulse.beta / pulse.pulse_time) * (t0 - pulse.pulse_time / 2))
    phi = (BWinf / 2) * (pulse.pulse_time / pulse.beta) * \
          np.log(np.cosh((pulse.beta / pulse.pulse_time) * (t0 - pulse.pulse_time / 2)))
    phi = 2 * np.pi * phi

    v1_range = interp1d(f0 * 10**3, v1)(f + pulse2.mwFreq * 10 ** 3)
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
                  flip=np.pi/2,
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

def test_WURST():
    pulse = Pulse(pulse_time=0.500,
                  type='WURST/linear',
                  freq=[-150, 350],
                  amp=15,
                  nwurst=15)

    A = 1 - np.abs((np.sin((np.pi * (pulse.time - pulse.pulse_time / 2)) / pulse.pulse_time)) ** pulse.nwurst)
    BW = np.diff(pulse.freq)[0]
    f = -(BW / 2) + (BW / pulse.pulse_time) * pulse.time
    phi = cumtrapz(f, pulse.time - pulse.pulse_time/2, initial=0)
    phi += np.abs(min(phi)) + np.mean(pulse.freq) * pulse.time

    IQ0 = pulse.amp * A * np.exp(2j * np.pi * phi)

    plt.plot(IQ0.real)
    plt.show()

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
    ti = t0 - pulse.pulse_time/2
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

def test_flip_am():
    pulse1 = {'pulse_time':0.060, 'type': 'rectangular'}
    pulse2 = {'pulse_time':0.200, 'tFWHM':0.060, 'type':'gaussian'}
    pulse3 = {'pulse_time':0.200, 'zerocross':0.050, 'type':'sinc'}
    pulse4 = {'pulse_time':0.100, 'trise':0.020, 'type':'quartersin'}
    pulse5 = {'pulse_time':0.500, 'beta':12, 'type':'sech'}
    pulse6 = {'pulse_time':0.300, 'nwurst':20, 'type':'WURST'}

    offsets = 0
    tol = 1e-12
    for pulse in [pulse1, pulse2, pulse3, pulse4, pulse5, pulse6]:
        p1 = Pulse(flip=np.pi/2, offsets=offsets, **pulse)
        p1.exciteprofile()
        p2 = Pulse(flip=np.pi, offsets=offsets,  **pulse)
        p2.exciteprofile()

        assert np.all(p1.Mag[2] < tol)
        assert np.all(p2.Mag[2] > -1 - tol)
        assert np.all(p2.Mag[2] < -1 + tol)

def test_save_bruker():
    profile = np.loadtxt('data/Transferfunction.dat').T

    pulse = Pulse(0.150, 0.000625, np.pi, freq=[40, 120], type='sech/tanh', beta=10, profile=profile)

    pulse.save_bruker('data/istmp.shp')
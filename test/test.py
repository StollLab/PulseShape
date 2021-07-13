import numpy as np
from PulseShape import Pulse


def test_construct():
    profile = np.loadtxt('data/Transferfunction.dat')
    myPulse = Pulse(0.150, 0.000625, np.pi, freq=[40, 120], type='sech/tanh', beta=10, profile=profile)

    ans = np.genfromtxt("data/sechtanh.csv", delimiter=',')
    ans = ans[:, 0] + 1j * ans[:, 1]
    np.testing.assert_almost_equal(myPulse.IQ, ans)


def test_gaussian():
    profile = np.loadtxt('data/Transferfunction.dat').T

    myPulse = Pulse(pulse_time=0.060,
                    time_step=0.000625,
                    flip=np.pi,
                    type='gaussian',
                    profile=profile,
                    trunc=0.1)

    ans = np.genfromtxt("data/gaussian.csv", delimiter=',', dtype=complex,
                        converters={0: lambda x: complex(x.decode('utf8').replace('i', 'j'))})
    np.testing.assert_almost_equal(myPulse.IQ, ans)


def test_rectrangular():
    pulse = Pulse(pulse_time=0.03,
                  flip=np.pi,
                  time_step=0.001)

    Amp = (pulse.flip / pulse.pulse_time) / (2 * np.pi)
    t0 = np.arange(0, pulse.pulse_time + pulse.time_step, pulse.time_step)
    IQ0 = np.ones(len(t0)) * Amp

    np.testing.assert_almost_equal(pulse.IQ.real, IQ0)
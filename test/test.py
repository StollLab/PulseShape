import numpy as np
import matplotlib.pyplot as  plt
from PulseShape import Pulse


def test_construct():

    profile = np.loadtxt('data/Transferfunction.dat').T
    myPulse = Pulse(0.150, 0.000625, np.pi, freq=[40, 120], type='sech/tanh', beta=10, profile=profile)


    ans = np.loadtxt('data/20210401_SechTanh_150ns.shp', delimiter=',')

    plt.plot(myPulse.IQ.real)
    plt.plot(ans[:, 0])
    plt.show()

    plt.plot(myPulse.IQ.imag)
    plt.plot(ans[:, 1])
    plt.show()


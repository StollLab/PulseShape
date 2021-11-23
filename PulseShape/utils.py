from functools import reduce
from itertools import accumulate
import numpy as np
from scipy.sparse import csr_matrix, kron
from scipy.interpolate import interp1d


def sop(spins, comps):
    spins = np.atleast_1d(spins)
    comps = np.atleast_1d(comps)

    Ops = []
    for spin in spins:
        for comp in comps:
            n = int(2 * spin + 1)
            Op = csr_matrix(1, (1, 1))
            if comp == 'x':
                m = np.arange(1, n)
                r = np.array([m, m+1])
                c = np.array([m+1, m])
                dia = 1 / 2 * np.sqrt(m * m[::-1])
                val = np.array([dia, dia])

            elif comp == 'y':
                m = np.arange(1, n)
                dia = -0.5j * np.sqrt(m * m[::-1])
                r = np.array([m, m+1])
                c = np.array([m+1, m])
                val = np.array([dia, -dia])

            elif comp == 'z':
                m = np.arange(1, n+1)
                r = m
                c = m
                val = spin + 1 - m

            else:
                raise NameError(f'{comp} is an unsupport SOP componant')
            r = np.squeeze(r.astype(int)) - 1
            c = np.squeeze(c.astype(int)) - 1
            val = np.squeeze(val)

            M_ = csr_matrix((val, (r, c)), shape=(n, n))
            Op = kron(Op, M_)
            Ops.append(Op)

    if len(Ops) == 1:
        return np.array(Ops[0].todense())
    else:
        return [np.array(Op.todense()) for Op in Ops]


def pulse_propagation(pulse, M0=[0, 0, 1], trajectory=False):
    """Vectorization of solution pulse propagation"""

    M0 = np.asarray(M0, dtype=float)
    if len(M0.shape) == 1:
        Mmag = np.linalg.norm(M0)
        M0 /= Mmag
        M0 = np.tile(M0, (len(pulse.offsets), 1))
        Mmag = np.array([Mmag for i in range(len(pulse.offsets))])
    else:
        Mmag = np.linalg.norm(M0, axis=1)
        M0 /= Mmag[:, None]


    Sx, Sy, Sz = sop(0.5, ['x', 'y', 'z'])
    density0 = 0.5 * np.array(([[1 + M0[:, 2], M0[:, 0] - 1j * M0[:, 1]],
                                [M0[:, 0] + 1j * M0[:, 1], 1 - M0[:, 2]]]))
    density0 = np.moveaxis(density0, 2, 0)

    dt = pulse.time[1] - pulse.time[0]

    H = pulse.offsets[:, None, None] * Sz
    H = H[:, None, :, :] + pulse.IQ.real[:, None, None] * Sx + pulse.IQ.imag[:, None, None] * Sy

    M = -2j * np.pi * dt * H
    q = np.sqrt(M[:, :, 0, 0]**2 - np.abs(M[:, :, 0, 1])**2)

    dUs = np.cosh(q)[:, :, None, None] * np.eye(2, dtype=complex) + (np.sinh(q) / q)[:, :, None, None] * M
    mask = np.abs(q) < 1e-10
    dUs[mask] = np.eye(2, dtype=complex) + M[mask]

    if not trajectory:
        Upulses = np.empty((len(dUs), 2, 2), dtype=complex)
        for i in range(len(dUs)):
            Upulses[i] = reduce(lambda x, y: y@x, dUs[i, :-1])


        density = np.einsum('ijk,ikl,ilm->ijm', Upulses, density0, Upulses.conj().transpose((0, 2, 1)))
        density = density.transpose((0, 2, 1))

        Mag = np.zeros((len(pulse.offsets), 3))
        Mag[..., 0] =  2 * density[..., 0, 1].real             # 2 * (Sx[None, :, :] * density).sum(axis=(1, 2)).real
        Mag[..., 1] = -2 * density[..., 1, 0].imag             # 2 * (Sy[None, :, :] * density).sum(axis=(1, 2)).real
        Mag[..., 2] =  density[..., 0, 0] - density[..., 1, 1] # 2 * (Sz[None, :, :] * density).sum(axis=(1, 2)).real
        return np.squeeze(Mag * Mmag[:, None])
    else:
        Upulses = np.empty((len(dUs), len(pulse.time), 2, 2), dtype=complex)
        for i in range(len(dUs)):
            Upulses[i] = [np.eye(2)] +  list((accumulate(dUs[i, :-1], lambda x, y: y @ x)))

        density = np.einsum('hijk,hkl,hilm->hijm', Upulses, density0, Upulses.conj().transpose((0, 1, 3, 2)))
        density = density.transpose((0, 1, 3, 2))

        Mag = np.zeros((len(pulse.offsets), len(pulse.time), 3))
        Mag[..., 0] = 2 * density[..., 0, 1].real # 2 * (Sx[None, None, :, :] * density).sum(axis=(2, 3)).real
        Mag[..., 1] = -2 * density[..., 1, 0].imag # 2 * (Sy[None, None, :, :] * density).sum(axis=(2, 3)).real
        Mag[..., 2] = density[..., 0, 0] - density[..., 1, 1] # 2 * (Sz[None, None, :, :] * density).sum(axis=(2, 3)).real

        return np.squeeze(Mag * Mmag[:, None, None])

def transmitter(signal, Ain, Aout, task='simulate', n=4):
    Ain, Aout = Ain.copy(), Aout.copy()

    # Fit data to get noiseless Aout
    V = [Ain]
    for i in range(n-2):
        V.insert(0, Ain * V[0])
    V = np.asarray(V).T

    coeff = np.linalg.lstsq(V, Aout)
    coeff = np.concatenate([coeff[0],  [0]])

    Aout = np.polyval(coeff, Ain)

    # Calculate nonlinearity
    if task.lower() == 'simulate':
        F = interp1d(Ain, Aout, kind='cubic')
    elif task.lower() == 'compensate':
        F = interp1d(Aout, Ain, kind='cubic')
    else:
        raise ValueError('`task` must be either simulate or compensate')

    signal = np.sign(signal.real) * F(np.abs(signal.real)) + \
             1j * np.sign(signal.imag) * F(np.abs(signal.imag))

    return signal

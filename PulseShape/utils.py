from functools import reduce
import numpy as np
from scipy.sparse import csr_matrix, kron
from numba import njit

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

@njit(cache=True)
def calc_mag(offsets, IQ, Sx, Sy, Sz, npoints, time, eye2, Density0):
    Mag = np.zeros((3, len(offsets)))

    for iOs, offset in enumerate(offsets):

        Ham0 = offset * Sz

        if IQ.min() == IQ.max():

            Ham = IQ.real[0] * Sx + IQ.imag[0] * Sy + Ham0
            tp = (time[1] - time[0]) * (npoints - 1)
            M = -2j * np.pi * tp * Ham

            q = np.sqrt(M[0, 0] ** 2 - abs(M[0, 1]) ** 2)
            if abs(q) < 1e-10:
                UPulse = eye2 + M
            else:
                UPulse = np.cosh(q) * eye2 + (np.sinh(q) / q) * M

        else:
            UPulse = eye2.copy()

            for it in range(0, npoints - 1):
                Ham = IQ.real[it] * Sx + IQ.imag[it] * Sy + Ham0
                M = -2j * np.pi * (time[1] - time[0]) * Ham
                q = np.sqrt(M[0, 0] ** 2 - abs(M[0, 1]) ** 2)

                if abs(q) < 1e-10:
                    dU = eye2 + M
                else:
                    dU = np.cosh(q) * eye2 + (np.sinh(q) / q) * M

                UPulse = dU @ UPulse


        density = UPulse @ Density0 @ UPulse.conjugate().T

        Mag[0, iOs] = -2 * (Sx * density.T).sum().real
        Mag[1, iOs] = -2 * (Sy * density.T).sum().real
        Mag[2, iOs] = -2 * (Sz * density.T).sum().real

    return Mag

def bloch(pulse):
    """Vectorization of solution bloch equations"""
    Sx, Sy, Sz = sop(0.5, ['x', 'y', 'z'])
    dt = pulse.time[1] - pulse.time[0]

    H = pulse.offsets[:, None, None] * Sz
    H = H[:, None, :, :] + pulse.IQ.real[:, None, None] * Sx + pulse.IQ.imag[:, None, None] * Sy

    M = -2j * np.pi * dt * H
    q = np.sqrt(M[:, :, 0, 0]**2 - np.abs(M[:, :, 0, 1])**2)

    dUs = np.cosh(q)[:, :, None, None] * np.eye(2, dtype=complex) + (np.sinh(q) / q)[:, :, None, None] * M
    mask = np.abs(q) < 1e-10
    dUs[mask] = np.eye(2, dtype=complex) + M[mask]

    Upulses = np.empty((len(dUs), 2, 2), dtype=complex)
    for i in range(len(dUs)):
        Upulses[i] = reduce(lambda x, y: y@x, dUs[i, :-1])

    density0 = -Sz.astype(complex)
    density = np.einsum('ijk,kl,ilm->ijm', Upulses, density0, Upulses.conj().transpose((0, 2, 1)))
    density = density.transpose((0, 2, 1))

    Mag = np.zeros((3, len(pulse.offsets)))
    Mag[0] = -2 * (Sx[None, :, :] * density).sum(axis=(1, 2)).real
    Mag[1] = -2 * (Sy[None, :, :] * density).sum(axis=(1, 2)).real
    Mag[2] = -2 * (Sz[None, :, :] * density).sum(axis=(1, 2)).real

    return Mag

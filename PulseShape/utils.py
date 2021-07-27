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
        return Ops[0].todense()
    else:
        return [Op.todense() for Op in Ops]

@njit
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


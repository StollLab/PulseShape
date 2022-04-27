from functools import reduce
from pathlib import Path
from itertools import accumulate
import numpy as np
from scipy.sparse import csr_matrix, kron
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

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
    Ainori, Aoutori = Ain, Aout
    Ain, Aout = Ain.copy(), Aout.copy()
    # Fit data to get noiseless Aout
    V = [Ain]
    for i in range(n-2):
        V.insert(0, Ain * V[0])
    V = np.asarray(V).T

    coeff = np.linalg.lstsq(V, Aout)
    coeff = np.concatenate([coeff[0],  [0]])
    Ain = np.linspace(0, 100, 256)
    Aout = np.polyval(coeff, Ain)

    # Calculate nonlinearity
    if task.lower() == 'simulate':
        F = interp1d(Ain, Aout, kind='cubic', fill_value='extrapolate')

    elif task.lower() == 'compensate':
        F = interp1d(Aout, Ain, kind='cubic', fill_value='extrapolate')
    else:
        raise ValueError('`task` must be either simulate or compensate')

    signal = np.sign(signal.real) * F(np.abs(signal.real)) + \
             1j * np.sign(signal.imag) * F(np.abs(signal.imag))

    return signal

def transmitter_profile(file_name):
    f = Path(file_name)

    if f.suffix in ('.DTA', '.DSC'):
        DTA_file = f.with_suffix('.DTA')
        DSC_file = f.with_suffix('.DSC')
    else:
        raise ValueError('file_name must be a Bruker DTA or DSC file')

    param_dict = read_param_file(str(DSC_file))

    # Calculate time axis data from experimental params
    x_points = int(param_dict['XPTS'][0])
    x_min = float(param_dict['XMIN'][0])
    x_width = float(param_dict['XWID'][0])
    x_max = x_min + x_width
    x_axis = np.linspace(x_min, x_max, x_points )

    y_points = int(param_dict['YPTS'][0])
    y_min = float(param_dict['YMIN'][0]) + 1
    y_width = float(param_dict['YWID'][0])
    y_max = y_min + y_width
    y_axis = np.linspace(y_min, y_max, y_points )

    # Read spec data
    data = np.fromfile(str(DTA_file), dtype='>d')

    # Reshape and form complex array
    data.shape = (-1, 2)
    data = data[:, 0] + 1j * data[:, 1]

    # Reshape to a 2D matrix
    data.shape = (y_points, -1)

    # Correct Phase
    data = opt_phase(data)

    tau = 200  # ns
    N = 2 ** 14  # Zero Padding
    ts, Vs = [], []
    ffts, fft_freqs = [], []
    freqs, nu = [], []

    for i, V in enumerate(data):

        V = V.copy() - V.mean()
        ts.append(x_axis * 1e3), Vs.append(V)

        # Apply window and padding
        window = np.exp(-x_axis / tau)
        nutation_win = window * V

        NWpad = np.zeros(N)
        NWpad[:len(nutation_win)] = nutation_win

        # Get FFT and frequency
        ft = np.fft.fftshift(np.fft.rfft(NWpad))
        dt = np.median(np.diff(x_axis))
        f = np.fft.fftshift(np.fft.rfftfreq(N, dt))

        ffts.append(ft.real)
        fft_freqs.append(f.real)

        # Return FeqMax
        idxmax = np.argmax(ft.real)
        nu.append(f[idxmax])

    # Convert Ghz to MHz
    nu = np.asarray(nu) * 1e3
    return y_axis, nu


def read_param_file(param_file):
    param_dict = {}
    with open(param_file, 'r', encoding='latin-1') as file:
        for line in file:
            # Skip blank lines and lines with comment chars
            if line.startswith(("*", "#", "\n")):
                continue

            # Add keywords to param_dict
            line = line.split()
            try:
                key = line.pop(0)
                val = [arg.strip() for arg in line]
            except IndexError:
                key = line
                val = None

            if key:
                param_dict[key] = val

    return param_dict


def get_imag_norms_squared(phi, V):
    V_imag = np.imag(V[:, None] * np.exp(1j * phi)[None, :, None])

    return (V_imag * V_imag).sum(-1)


def opt_phase(V, return_params=False):

    V = np.atleast_2d(V)

    # Calculate 3 points of cost function which should be a smooth continuous sine wave
    phis = np.array([0, np.pi / 2, np.pi]) / 2
    costs = get_imag_norms_squared(phis, V)

    # Calculate sine function fitting 3 points
    offset = (costs[:, 0] + costs[:, 2]) / 2
    phase_shift = np.arctan2(costs[:, 0] - offset, costs[:, 1] - offset)

    # Calculate phi by calculating the phase when the derivative of the sine function is 0 and using the second
    # derivative to ensure it is a minima and not a maxima
    possible_phis = np.array([(np.pi / 2 - phase_shift) / 2, (3 * np.pi / 2 - phase_shift) / 2]).T
    second_deriv = -np.sin(2 * possible_phis + phase_shift[:, None])
    opt_phase = possible_phis[second_deriv > 0]

    # Check to ensure the real component is positive
    temp_V = V * np.exp(1j * opt_phase)[:, None]
    opt_phase[temp_V.sum(axis=1) < 0] += np.pi
    V = V * np.exp(1j * opt_phase)[:, None]

    if return_params:
        return np.squeeze(V), np.squeeze(opt_phase)
    else:
        return np.squeeze(V)


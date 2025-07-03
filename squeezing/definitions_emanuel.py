import numpy as np
from scipy.integrate import quad, nquad, dblquad
from scipy.special import gamma, genlaguerre
from thewalrus.decompositions import blochmessiah, williamson
import pickle
import os
import csv

def hash_dict(params):
    """ TODO: Docstring

    """
    return hash(tuple(params.values()))

def is_pickled(params):
    """ Checks if Pickeled object with matching parameters exits

    :param name: Name of the file to look for (except hash value)
    :type name: str
    :param params: list of parameters
    :return: Pickeled object. If not found returns None
    :rtype: Object

    """
    hash_value = hash_dict(params)
    for filename in os.listdir("Pickled/"):
        if filename == str(hash_value):
            with open(f"Pickled/{filename}", 'rb') as file:
                object_tmp = pickle.load(file)
                print("I found a pickeled object called "
                        f"\"{filename}\" matching your description!")
                return object_tmp
    return None

def save_data(fname, header, data, comment=""):
    """ Saves data to CSV file in folder Data

    :param fname: name of file
    :type fname: str
    :param header: Header to put into CSV file
    :type header: str
    :param data: 2D Data array to be saved
    :type data: 2D array of floats
    :param comment: Comment to put into CSV file. Defaults to empty string.
    :type comment: str, optional

    """
    with open("Data/{}.csv".format(fname), 'w', newline='') as file:
        file.write("# "+comment+'\n')
        file.write("{}\n".format(header))
        writer = csv.writer(file, lineterminator='\n')
        for i in range(len(data)):
            writer.writerow(data[i])

# Mathematical stuff
Sym_mat = lambda n: np.block([
    [np.zeros((n, n)), np.eye(n)],
    [-np.eye(n), np.zeros((n, n))]])

def inner_prod(f, g):
    """ TODO: Docstring

    """
    integrand = lambda x: np.conj(f(x)) * g(x)
    return quad(integrand, 0, cutfreq)[0]

def symplectic_housholder(a, e):
    """ TODO: Docstring

    """
    lam = np.linalg.norm(a)
    if np.linalg.norm(a - lam * e) <= 1e-10:
        vec = 0 * e
    else:
        vec = (a - lam * e) / np.linalg.norm(a - lam * e)
    
    H = np.eye(3) - 2 * np.inner(vec, vec)
    H_sym = np.block([
        [H, np.zeros((3, 3))],
        [np.zeros((3, 3)), H]])
    return H_sym


def gbmd(A):
    """ Generalized Bloch Messiah decomposition
    TODO: Docstring

    """
    a = np.matrix(A[1,3:6]).T
    e = np.matrix([1, 0, 0]).T
    H_sym = symplectic_housholder(a, e)

    T = A @ Sym_mat(3) @ A.T
    if np.round(np.linalg.det(T), 10) == 0:
        print("Matrix not singular!")

    R1 = A @ H_sym.T
    sign = np.sign(R1[0, 0] * R1[1, 3])


    V = np.block([
        [sign * np.eye(3), np.zeros((3, 3))],
        [np.zeros((3, 3)), np.eye(3)]
        ])
    R = R1 @ V
    R11 = R[0, 0]
    R12 = R[0, 1:3]
    R13 = R[0, 3]
    R14 = R[0, 4:6]
    R23 = R[1, 3]

    s = np.sqrt(R23 * R11)
    
    S_tilde = np.zeros((6, 6)) 
    S_tilde[0, 0] = R23 / s 
    S_tilde[0, 1:3] = - R12 * R23 / s**2 
    S_tilde[0, 3] = - R13 / s 
    S_tilde[0, 4:6] = - R14 * R23 / s**2
    S_tilde[1:3, 1:3] = np.eye(2)
    S_tilde[1:3, 3:4] = -R14.T / s
    S_tilde[3, 3] = R11 / s
    S_tilde[4:6, 3:4] = R12.T / s
    S_tilde[4:6, 4:6] = np.eye(2)
    
    S_inv = H_sym.T @ S_tilde
    S = - Sym_mat(3) @ S_inv.T @ Sym_mat(3)

    glob_sign = np.sign(S[0, 0])

    squeezing_x = np.sqrt(sum([S[0, j]**2 for j in range(3)]))
    squeezing_p = np.sqrt(sum([S[3, j + 3]**2 for j in range(3)]))
    r_eff = np.log(squeezing_p / squeezing_x) / 2
    
    Sigma = glob_sign * np.matrix([
        [sign * s * np.exp(-r_eff), 0, 0, 0, 0, 0],
        [0, 0, 0, s * np.exp(r_eff), 0, 0]])

    Squeezing_eff = glob_sign * np.matrix([
        [np.exp(r_eff), 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, np.exp(-r_eff), 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]])

    S_eff = Squeezing_eff@S

    O, D, Q = blochmessiah(S_eff)
    #A_dec = Sigma @ S_eff
    return Sigma, S_eff 

# Numerical stuff
cutfreq = 2 * np.pi * 500

# Physical constants
# speed of light in m/ps
C = 3e8 * 1e-12
# reduced plancks constant in kg m^2 per ps
HBAR = 1.055e-34 * 1e-12
# vacuum permitivity in 10^{-48} F/m
EPSILON0 = 8.85e-12 * 1e48
# EO-coeff in ps^3 A / (kg m) or 10^{-36}m/V
R41 = 4 * 1e-12 * 1e36


# Scaled chi distribution
def f(omega, k, sigma):
    return (np.sqrt(2 / (sigma * gamma(k + 1 / 2)))
            * (abs(omega) / sigma)**k
            * np.exp(- 1 / 2 * (omega / sigma)**2)
            )

# Generalized Laguerre polynomilas
def p(omega, n, k, sigma):
    return np.sqrt(
            gamma(n + 1) * gamma(k + 1 / 2) / gamma(n + k + 1 / 2)
            ) * genlaguerre(n, k - 1 / 2)(omega**2 / sigma**2)

def LG(omega, n, k, sigma):
    return p(omega, n, k, sigma) * f(omega, k, sigma)

def eval_LG(n, k, sigma):
    return lambda omega: LG(omega, n, k, sigma)


# Nonlinear interaction
@np.vectorize
def n(omega):
    """ Refractive index

    :param omega:

    """
    a = 4.27
    b = 3.01
    gamma = 0.142 * 1e-12

    omega = abs(omega)
    if omega >= 2 * np.pi * 750:
        omega = 2 * np.pi * 750
    return np.real(np.sqrt(a + b * (2 * np.pi * C)**2 / ((2 * np.pi * C)**2 - omega**2 * gamma) + 0j))

L = 20e-6
r = 3e-6
phi_alpha = 0
omegap = 2 * np.pi * 200
sigmap = 2 * np.pi * 100
beamarea = np.pi * r**2
d = -n(omegap)**4 * R41 
lam = beamarea * EPSILON0 * d / 2

def f_p(omega, kp, sigmap):
    """Pump modes for generation and detection

    :param omega:

    """
    if abs(phi_alpha) == 0:
        return np.heaviside(omega, 1 / 2) * np.sqrt(abs(omega) / n(omega)) * f(omega, kp, sigmap)
    return np.exp(1j * phi_alpha) * np.heaviside(omega, 1 / 2) * np.sqrt(abs(omega) / n(omega)) * f(omega, kp, sigmap)

def delta_k(omega, Omega):
    return 1 / C * ((omega + Omega) * n(omega + Omega) - omega * n(omega) - Omega * n(Omega))

def crystal_fft(k):
        return lam * L * np.pi * np.sinc(L / (2 * np.pi) * k)

def kernel_S(omega, Omega, kp, sigmap):
    """JSA of the generation crystal V for squeezing

    :param omega:
    :param Omega:

    """
    prefactor = 1 / HBAR * (HBAR / (4 * np.pi * EPSILON0 * C * beamarea))**(3 / 2)
    return prefactor * np.sqrt(abs(omega) * abs(Omega) / (n(omega) * n(Omega))) \
            * f_p(omega + Omega, kp, sigmap) \
            * crystal_fft(delta_k(omega, Omega))

def kernel_B(omega, Omega, kp, sigmap):
    """JSA of the generation crystal W for beam splitting

    :param omega:
    :param Omega:

    """
    prefactor = 2 / HBAR * (HBAR / (4 * np.pi * EPSILON0 * C * beamarea))**(3 / 2)
    return prefactor * np.sqrt(abs(omega) * abs(Omega) / (n(omega) * n(Omega))) \
        * (f_p(omega - Omega, kp, sigmap) - np.conj(f_p(Omega - omega, kp, sigmap))) \
        * crystal_fft(delta_k(-omega, Omega))

# Functions to calculate the matricies A and B
def calc_thetas(v, us, kp, sigmap):
    """ TODO: Docstring

    """
    kernel = lambda x, y, z: v(x) * (
            (kernel_S(y, x, kp, sigmap) + np.conj(kernel_S(x, y, kp, sigmap))) * 
            (np.conj(kernel_S(y, z, kp, sigmap)) + kernel_S(z, y, kp, sigmap)) * np.conj(v(z))
            )
    theta_S = np.sqrt(nquad(kernel, [[0, cutfreq], [0, cutfreq], [0, cutfreq]])[0])
    kernel = lambda x, y, z: np.conj(v(x)) * np.conj(kernel_B(x, y, kp, sigmap)) * kernel_B(z, y, kp, sigmap) * v(z)
    theta_B = np.sqrt(nquad(kernel, [[0, cutfreq], [0, cutfreq],[0, cutfreq]])[0])
    return theta_B, theta_S

def calc_overlaps(v, us, kp, sigmap):
    """ TODO: Docstring

    """
    vBus = []
    uBvs = []
    uSvs = []
    vSus = []
    for u in us:
        kernel = lambda x, y: np.conj(v(x)) * kernel_B(y, x, kp, sigmap) * u(y)
        vBus.append(dblquad(kernel, 0, cutfreq, 0, cutfreq)[0])
        kernel = lambda x, y: np.conj(u(x)) * kernel_B(y, x, kp, sigmap) * v(y)
        uBvs.append(dblquad(kernel, 0, cutfreq, 0, cutfreq)[0])
        kernel = lambda x, y: np.conj(u(x)) * (np.conj(kernel_S(y, x, kp, sigmap)) + kernel_S(x, y, kp, sigmap)) * np.conj(v(y))
        uSvs.append(dblquad(kernel, 0, cutfreq, 0, cutfreq)[0])
        kernel = lambda x, y: np.conj(v(x)) * (np.conj(kernel_S(y, x, kp, sigmap)) + kernel_S(x, y, kp, sigmap)) * np.conj(u(y))
        vSus.append(dblquad(kernel, 0, cutfreq, 0, cutfreq)[0])

    kernel = lambda x, y: np.conj(v(x)) * kernel_B(y, x, kp, sigmap) * v(y)
    vBv = dblquad(kernel, 0, cutfreq, 0, cutfreq)[0]

    kernel = lambda x, y: np.conj(v(x)) * (np.conj(kernel_S(y, x, kp, sigmap)) + kernel_S(x, y, kp, sigmap)) * np.conj(v(y))
    vSv = dblquad(kernel, 0, cutfreq, 0, cutfreq)[0]

    kernel = lambda x, y, z: np.conj(v(x)) * np.conj(kernel_B(x, y, kp, sigmap)) * (np.conj(kernel_S(y, z, kp, sigmap)) + kernel_S(z, y, kp, sigmap)) * np.conj(v(z)) 
    vBSv = nquad(kernel, [[0, cutfreq], [0, cutfreq],[0, cutfreq]])[0]

    return [vBus, uBvs, vSus, uSvs, vBv, vSv, vBSv]

def calc_alpha_dep(mateles, alpha):
    """ TODO: Docstring

    """
    vBus = mateles[0].copy()
    uBvs = mateles[1].copy()
    vSus = mateles[2].copy()
    uSvs = mateles[3].copy()
    N = len(vBus)
    for i in range(N):
        vBus[i] *= alpha
        uBvs[i] *= alpha
        vSus[i] *= alpha
        uSvs[i] *= alpha
    vBv = alpha * mateles[4]
    vSv = alpha * mateles[5]
    vBSv = alpha**2 * mateles[6]
    return vBus, uBvs, vSus, uSvs, vBv, vSv, vBSv

def calc_A(theta, mu, nu, v, us, vBus, vSus):
    """ TODO: Docstring

    """
    matele_F = []
    matele_G = []
    N = len(vBus)
    for i in range(N):
        matele_F.append(mu * inner_prod(v, us[i]) - nu * vBus[i] / theta)
        matele_G.append(nu * vSus[i] / theta)
    
    A = []
    row = []
    for i in range(N):
        row.append(np.real(matele_F[i] + matele_G[i]))
    for i in range(N):
        row.append(np.imag(-matele_F[i] + matele_G[i]))
    A.append(row)
    row = []
    for i in range(N):
        row.append(np.imag(matele_F[i] + matele_G[i]))
    for i in range(N):
        row.append(np.real(matele_F[i] - matele_G[i]))
    A.append(row)
    return np.matrix(A)


def calc_B(theta, theta_B, theta_S, mu, nu, v, us, vBv, vSv, uSvs, uBvs, vBSv):
    """ TODO: Docstring

    """
    N = len(uSvs)
    xi = nu * theta_S / theta
    zeta1 = mu * nu * vBv / theta
    zeta2 = nu * theta_B / theta
    zeta = np.sqrt(mu**2 - zeta1 - np.conj(zeta1) + zeta2**2)
    
    ugs = []
    uvs = []
    ufs = []
    for i in range(N):
        ugs.append(nu * uSvs[i] / (xi * theta))
        uvs.append(inner_prod(us[i], v))
        ufs.append((mu * uvs[i] - nu * uBvs[i] / theta) / zeta)

    normx = np.sqrt(1 - sum([abs(ug)**2 for ug in ugs]))

    fg1 = mu * vSv 
    fg2 = nu * vBSv / theta
    fg = nu * (fg1 - fg2) / (zeta * xi * theta)

    h1f = np.conj(fg)
    for i in range(N):
        h1f -= np.conj(ugs[i]) * ufs[i]
    h1f /= normx
    normz = np.sqrt(1 - sum([abs(uf)**2 for uf in ufs]) - abs(h1f)**2)
    B = np.matrix([
        [np.real(np.conj(h1f)) + normx, normz, -np.imag(np.conj(h1f)), 0],
        [np.imag(np.conj(h1f)), 0, np.real(np.conj(h1f)) - normx, normz]])
    return B

def calc_theta(theta_B, theta_S, silent=True):
    """ TODO: Docstring

    """
    theta = np.sqrt(abs(theta_B**2 - theta_S**2))
    if theta_B >= theta_S:
        if not silent:
            print("Beam splitting regime!")
        mu = np.cos(theta)
        nu = np.sin(theta)
    else:
        if not silent:
            print("Squeezing regime!")
        mu = np.cosh(theta)
        nu = np.sinh(theta)
    return theta, mu, nu

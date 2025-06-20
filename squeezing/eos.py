import numpy as np
from scipy.special import gamma, genlaguerre
from scipy.integrate import quad, dblquad, nquad
import matplotlib.pyplot as plt
from thewalrus.decompositions import blochmessiah

# Physical constants
# speed of light in m/ps
C = 3e8 * 1e-12
# reduced plancks constant in kg m^2 per ps
HBAR = 1.055e-34 * 1e-12
# vacuum permitivity in 10^{-48} F/m
EPSILON0 = 8.85e-12 * 1e48
# EO-coeff in ps^3 A / (kg m) or 10^{-36}m/V
R41 = 4 * 1e-12 * 1e36

# Mathematical stuff
Sym_mat = lambda n: np.block([
    [np.zeros((n, n)), np.eye(n)],
    [-np.eye(n), np.zeros((n, n))]])

# Numerical stuff
cutfreq = 2 * np.pi * 500

# Scaled chi distribution
def f(omega, k, sigma):
    return (np.sqrt(2 / (sigma * gamma(k + 1 / 2)))
            * (abs(omega) / sigma)**k
            * np.exp(- 1 / 2 * (omega / sigma)**2)
            )

k0 = 0.5
sigma0 = 2 * np.pi * 30

# Generalized Laguerre polynomilas
def p(n, omega):
    return np.sqrt(
            gamma(n + 1) * gamma(k0 + 1 / 2) / gamma(n + k0 + 1 / 2)
            ) * genlaguerre(n, k0 - 1 / 2)(omega**2 / sigma0**2)

def LG(n, omega):
    return p(n, omega) * f(omega, k0, sigma0)

u0 = lambda omega: LG(0, omega)
u1 = lambda omega: LG(1, omega)
u2 = lambda omega: LG(2, omega)

us = [u0, u1, u2]

L = 20e-6
r = 3e-6
beamarea = np.pi * r**2
alpha = 0.01 * 1e6
omegap = 2 * np.pi * 200
sigmap = 2 * np.pi * 100
kp = (omegap / sigmap)**2

omega_out = 2 * np.pi * 150
sigma_out = 2 * np.pi * 50
k_out = (omega_out / sigma_out)**2

v = lambda omega: f(omega, k_out, sigma_out)

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
                                   
d = -n(omegap)**4 * R41 
lam = beamarea * EPSILON0 * d / 2

def f_p(omega):
    """Pump modes for generation and detection

    :param omega:

    """
    return np.heaviside(omega, 1 / 2) * np.sqrt(abs(omega) / n(omega)) * f(omega, kp, sigmap)


def delta_k(omega, Omega):
    return 1 / C * ((omega + Omega) * n(omega + Omega) - omega * n(omega) - Omega * n(Omega))

def crystal_fft(k):
        return lam * L * np.pi * np.sinc(L / (2 * np.pi) * k)

def kernel_S(omega, Omega):
    """JSA of the generation crystal V for squeezing

    :param omega:
    :param Omega:

    """
    prefactor = 1 / HBAR * (HBAR / (4 * np.pi * EPSILON0 * C * beamarea))**(3 / 2)
    return prefactor * alpha * np.sqrt(abs(omega) * abs(Omega) / (n(omega) * n(Omega))) \
            * f_p(omega + Omega) \
            * crystal_fft(delta_k(omega, Omega))

def kernel_B(omega, Omega):
    """JSA of the generation crystal W for beam splitting

    :param omega:
    :param Omega:

    """
    prefactor = 2 / HBAR * (HBAR / (4 * np.pi * EPSILON0 * C * beamarea))**(3 / 2)
    return prefactor * alpha * np.sqrt(abs(omega) * abs(Omega) / (n(omega) * n(Omega))) \
        * (f_p(omega - Omega) - np.conj(f_p(Omega - omega))) \
        * crystal_fft(delta_k(-omega, Omega))

fig, (ax1, ax2) = plt.subplots(1, 2)
x = np.linspace(0, cutfreq, 100)
X, Y = np.meshgrid(x, x)
ax1.pcolormesh(X / (2 * np.pi), Y / (2 * np.pi), kernel_S(X, Y))
ax2.pcolormesh(X / (2 * np.pi), Y / (2 * np.pi), kernel_B(X, Y))
plt.show()

def inner_prod(f, g):
    """ TODO: Docstring

    """
    integrand = lambda x: np.conj(f(x)) * g(x)
    return quad(integrand, 0, cutfreq)[0]

def calc_A():
    """ TODO: Docstring

    """
    matele_F = []
    for u in us:
        kernel = lambda omega,Omega: np.conj(v(omega)) * kernel_B(omega, Omega) * u(Omega)
        matele_F.append(inner_prod(v, u) - dblquad(kernel, 0, cutfreq, 0, cutfreq)[0])
    
    matele_G = []
    for u in us:
        kernel = lambda omega,Omega: np.conj(v(omega)) * (np.conj(kernel_S(Omega, omega)) + kernel_S(omega, Omega)) * np.conj(u(Omega))
        matele_G.append(dblquad(kernel, 0, cutfreq, 0, cutfreq)[0])
    
    A = []
    row = []
    for i in range(3):
        row.append(np.real(matele_F[i] + matele_G[i]))
    for i in range(3):
        row.append(np.imag(-matele_F[i] + matele_G[i]))
    A.append(row)
    row = []
    for i in range(3):
        row.append(np.imag(matele_F[i] + matele_G[i]))
    for i in range(3):
        row.append(np.real(matele_F[i] - matele_G[i]))
    A.append(row)
    return np.matrix(A)

#A = calc_A()
# alpha = 0.01 * 1e6
A = np.matrix([
    [-0.00301105, -0.0292298,   0.12081718,  0.      ,    0.        ,  0.        ],
    [ 0.        ,  0.       ,   0.        ,  0.034868,   -0.05813175,  0.13636899]
    ])

print("A: ", A)

def calc_B():
    """ TODO: Docstring

    """
    kernel = lambda x, y, z: v(x) * (kernel_S(y, x) + np.conj(kernel_S(x, y))) * (np.conj(kernel_S(y, z)) + kernel_S(z, y)) * np.conj(v(z))
    xi = np.sqrt(nquad(kernel, [[0, cutfreq], [0, cutfreq], [0, cutfreq]])[0])
    print("xi: ", xi)
    kernel = lambda x, y: np.conj(v(x)) * kernel_B(y, x) * v(y)
    zeta2 = dblquad(kernel, 0, cutfreq, 0, cutfreq)[0]
    kernel = lambda x, y, z: np.conj(v(x)) * np.conj(kernel_B(x, y)) * kernel_B(z, y) * v(z)
    zeta3 = nquad(kernel, [[0, cutfreq], [0, cutfreq],[0, cutfreq]])[0]
    zeta = np.sqrt(1 - zeta2 - np.conj(zeta2) + zeta3)
    print("zeta: ", zeta)
    
    ugs = []
    ufs = []
    uvs = []
    for u in us:
        kernel = lambda x, y: np.conj(u(y)) * (np.conj(kernel_S(y, x)) + kernel_S(x, y)) * np.conj(v(x))
        ugs.append(dblquad(kernel, 0, cutfreq, 0, cutfreq)[0] / xi)
        uv = quad(lambda x: np.conj(u(x)) * v(x), 0, cutfreq)[0]
        uvs.append(uv)
        kernel = lambda x, y: np.conj(u(y)) * kernel_B(x, y) * v(x)
        ufs.append((uv - dblquad(kernel, 0, cutfreq, 0, cutfreq)[0]) / zeta)

    x = np.sqrt(1 - sum([abs(ug)**2 for ug in ugs]))
    print("x: ", x)

    kernel = lambda x, y: np.conj(v(x)) * (np.conj(kernel_S(x, y)) + kernel_S(y, x)) * np.conj(v(y))
    fg1 = dblquad(kernel, 0, cutfreq, 0, cutfreq)[0]
    kernel = lambda x, y, z: np.conj(v(x)) * np.conj(kernel_B(x, y)) * (np.conj(kernel_S(y, z)) + kernel_S(z, y)) * np.conj(v(z))
    fg2 = nquad(kernel, [[0, cutfreq], [0, cutfreq],[0, cutfreq]])[0]
    fg = (fg1 - fg2) / (zeta * xi)
    print("fg: ", fg)

    h1f = np.conj(fg)
    for i in range(0,3):
        h1f -= np.conj(ugs[i]) * ufs[i]
        print(np.conj(ugs[i]) * ufs[i])
    h1f /= x
    print("h1f: ", h1f)
    try:
        z = np.sqrt(1 - sum([abs(uf)**2 for uf in ufs]) - abs(h1f)**2)
    except:
        print("Error while calculating z")
        print("ufs: ", sum([abs(uf)**2 for uf in ufs]))
        print("h1f: ", abs(h1f)**2)
        z = 0
    print("z: ", z)
    B = np.matrix([
        [np.real(np.conj(h1f)) + x, z, -np.imag(np.conj(h1f)), 0],
        [np.imag(np.conj(h1f)), 0, np.real(np.conj(h1f)) - x, z]])
    return B

#B = calc_B()
# Weak (alpha = 0.02 * 1e6)
B = np.matrix([
    [ 0.01716219,  0.97996923, -0.        ,  0.        ],
    [ 0.        ,  0.        , -0.30675994,  0.97996923]
    ])
        
print("B: ", B)
cov_th = B @ B.T
print("cov_th: ", cov_th)


def symplectic_housholder(a, e, sign=1):
    """ TODO: Docstring

    """
    lam = sign * np.linalg.norm(a)
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
    if np.round(np.linalg.det(T), 5) == 0:
        print("Matrix not singular!")


    R1 = A @ H_sym.T

    u, sv, vh = np.linalg.svd(np.matrix(R1[0,0:3]).T)
    U = np.block([
        [u, np.zeros((3, 3))],
        [np.zeros((3, 3)), u]])
    R = R1
    print("R: ", R)
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
    
    print("S: ", S)
    print("s: ", s)
    
    Sigma = np.matrix([
        [s, 0, 0, 0, 0, 0],
        [0, 0, 0, s, 0, 0]])

    O, D, Q = blochmessiah(S)
    #A_dec = Sigma @ S
    #print(A)
    #print(A_dec)
    return Sigma, S 


Sigma, S = gbmd(A)

def u_red_x(omega, i):
    """ TODO: Docstring

    """
    s = 0
    s_abs = 0
    for j in range(0, 3):
        s += S[i, j] * us[j](omega)
        s_abs += S[i, j]**2
    return s / np.sqrt(s_abs)

def u_red_p(omega, i):
    """ TODO: Docstring

    """
    s = 0
    s_abs = 0
    for j in range(0, 3):
        s += S[i, j + 3] * us[j](omega)
        s_abs += S[i, j + 3]**2
    return s / np.sqrt(s_abs)

u0_red_x = lambda omega: u_red_x(omega, 0)
u1_red_x = lambda omega: u_red_x(omega, 1)
u2_red_x = lambda omega: u_red_x(omega, 2)
u0_red_p = lambda omega: u_red_p(omega, 3)
u1_red_p = lambda omega: u_red_p(omega, 4)
u2_red_p = lambda omega: u_red_p(omega, 5)

omegas = np.linspace(0, cutfreq, 100)
fs = omegas / (2 * np.pi)
plt.plot(fs, v(omegas), color="black")
plt.plot(fs, u0(omegas), color="C0")
plt.plot(fs, u1(omegas), color="C1")
plt.plot(fs, u2(omegas), color="C2")
plt.plot(fs, u0_red_x(omegas), color="C0", linestyle="dashed")
plt.plot(fs, u1_red_x(omegas), color="C1", linestyle="dashed")
plt.plot(fs, u2_red_x(omegas), color="C2", linestyle="dashed")
plt.plot(fs, u0_red_p(omegas), color="C0", linestyle="dotted")
plt.plot(fs, u1_red_p(omegas), color="C1", linestyle="dotted")
plt.plot(fs, u2_red_p(omegas), color="C2", linestyle="dotted")
plt.show()

rs = [1.2, 0.8, 0.5]
cov = np.diag(
        [np.exp(2 * r) for r in rs] + [np.exp(-2 * r) for r in rs]
        )

print(cov)
cov_S = S @ cov @ S.T

x = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, x)
fig, axs = plt.subplots(2, 3)

# Input modes input
Ws = []
for i in range(3):
    cov_red = np.matrix([
        [cov[i,i], cov[i,i + 3]],
        [cov[i + 3,i], cov[i + 3,i + 3]]])
    cov_red_det = np.sqrt(np.linalg.det(cov_red))
    cov_inv = np.linalg.inv(cov_red)
    W = lambda x,p: 1 / (np.pi * np.sqrt(cov_red_det)) * np.exp(- x**2 * cov_inv[0, 0] - p**2 * cov_inv[1, 1] - x * p * (cov_inv[0, 1] + cov_inv[1, 0]))
    axs[0, i].pcolormesh(X, Y, W(X, Y))
    axs[0, i].set_title(f"Input mode {i+1}")

# First mode output
s = Sigma[0, 0]
cov_red = np.matrix([
    [cov_S[0,0], cov_S[0,3]],
    [cov_S[3,0], cov_S[3,3]]])
cov_red_det = np.sqrt(np.linalg.det(cov_red))
cov_inv = np.linalg.inv(cov_red)
W = lambda x,p: 1 / (np.pi * np.sqrt(cov_red_det)) * np.exp(- x**2 * cov_inv[0, 0] - p**2 * cov_inv[1, 1] - x * p * (cov_inv[0, 1] + cov_inv[1, 0]))
axs[1, 0].pcolormesh(X, Y, W(X, Y))
axs[1, 0].set_title("Output statistic (pure)")

# First mode output + thermal
s = Sigma[0, 0]
cov_red = s**2 * np.matrix([
    [cov_S[0,0], cov_S[0,3]],
    [cov_S[3,0], cov_S[3,3]]]) + cov_th
print(cov_red)
print("r1: ", np.log(cov_S[0,0]))
print("r2: ", np.log(cov_S[3,3]))
print(cov_S[0,0] * cov_S[3,3])
cov_red_det = np.sqrt(np.linalg.det(cov_red))
cov_inv = np.linalg.inv(cov_red)

W = lambda x,p: 1 / (np.pi * np.sqrt(cov_red_det)) * np.exp(- x**2 * cov_inv[0, 0] - p**2 * cov_inv[1, 1] - x * p * (cov_inv[0, 1] + cov_inv[1, 0]))
axs[1, 1].pcolormesh(X, Y, W(X, Y))
axs[1, 1].set_title("Output statistic (thermal)")

# First mode output + thermal
cov_red = np.matrix([
    [1, 0],
    [0, 1]])
cov_red_det = np.sqrt(np.linalg.det(cov_red))
cov_inv = np.linalg.inv(cov_red)

W = lambda x,p: 1 / (np.pi * np.sqrt(cov_red_det)) * np.exp(- x**2 * cov_inv[0, 0] - p**2 * cov_inv[1, 1] - x * p * (cov_inv[0, 1] + cov_inv[1, 0]))
axs[1, 2].set_title("Statistic of vauum")
axs[1, 2].pcolormesh(X, Y, W(X, Y))
plt.show()

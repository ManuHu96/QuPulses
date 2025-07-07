""" Module to calculate the input-output relation using the first-order
unitary approach introduced in
(https://doi.org/10.1103/PhysRevD.105.056023) and
(https://doi.org/10.1103/PhysRevX.14.041032).

"""

import numpy as np
from scipy.integrate import quad, nquad, dblquad
from scipy.special import gamma, genlaguerre

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


def f(omega, k, sigma):
    """ Fundaental pulsed mode given by the scaled Chi distribution,
    which interpolates between subcycle and multicycle pulse.

    Parameters
    ----------
    omega : float
        Frequency at wich the function is evaluated
    k : float
        Cycle parameter. Determines the number of optical cycle
        completed during the pulse.
    sigma : float
        Frequency scaling. Gives the Frequency range of the pulse.

    Returns
    -------
    : float
        Evaluates the function at omega

    """
    return (np.sqrt(2 / (sigma * gamma(k + 1 / 2)))
            * (abs(omega) / sigma)**k
            * np.exp(- 1 / 2 * (omega / sigma)**2)
            )


def p(omega, n, k, sigma):
    """ Scaled Laguerre polynomials to calculate orthonormal
    pulse basis.

    Parameters
    ----------
    omega : float
        Frequency at wich the function is evaluated
    n : int
        Order of polynomial
    k : float
        Cycle parameter. Determines the number of optical cycle
        completed during the pulse.
    sigma : float
        Frequency scaling. Gives the Frequency range of the pulse.

    Returns
    -------
        float: Evaluates the polynomial at omega
    """
    return np.sqrt(
            gamma(n + 1) * gamma(k + 1 / 2) / gamma(n + k + 1 / 2)
            ) * genlaguerre(n, k - 1 / 2)(omega**2 / sigma**2)


def LG(omega, n, k, sigma):
    """ Generalized Laguerre-Gauss polynomials

    Parameters
    ----------
        omega : float
            Frequency at wich the function is evaluated
        n : int
            Order of polynomial
        k : float
            Cycle parameter. Determines the number of optical cycle
            completed during the pulse.
        sigma : float
            Frequency scaling. Gives the Frequency range of the pulse.

    Returns
    -------
        float: Evaluates the generalized Laguerre-Gauss at omega

    """
    return p(omega, n, k, sigma) * f(omega, k, sigma)


def eval_LG(n, k, sigma):
    """ Gives Generalized Laguerre-Gauss polynomials as a function of
    omega with fixed parameters n, k and sigma

    Parameters
    ----------
    n : int
        Order of polynomial
    k : float
        Cycle parameter. Determines the number of optical cycle
        completed during the pulse.
    sigma : float
        Frequency scaling. Gives the Frequency range of the pulse.

    Returns
    -------
     : float
        Evaluates the generalized Laguerre-Gauss at omega

    """
    return lambda omega: LG(omega, n, k, sigma)


# Nonlinear interaction
@np.vectorize
def n(omega):
    """ Refractive index of Zinc-Tellurid
    (https://doi.org/10.1063/1.1713411)

    Parameters
    ----------
    omega : float
        Frequency at wich the refractive index is evaluated

    Returns
    -------
     : float
        Refractive index of Zinc-Tellurid at omega

    """
    a = 4.27
    b = 3.01
    gamma = 0.142 * 1e-12

    omega = abs(omega)
    if omega >= 2 * np.pi * 750:
        omega = 2 * np.pi * 750
    n = np.real(
            np.sqrt(
                a + b * (2 * np.pi * C)**2 /
                ((2 * np.pi * C)**2 - omega**2 * gamma) + 0j
                )
            )
    return n


L = 20e-6
r = 3e-6
phi_alpha = 0
omegap = 2 * np.pi * 200
sigmap = 2 * np.pi * 100
kp = (omegap / sigmap)**2
beamarea = np.pi * r**2
d = -n(omegap)**4 * R41
lam = beamarea * EPSILON0 * d / 2


def f_p(omega, kp, sigmap):
    """ Pulsed mode which is used to drive the nonlinear crystal

    Parameters
    ----------
    omega : float
        Frequency at wich the function is evaluated
    kp : float
        Cycle parameter. Determines the number of optical cycle
        completed during the pulse.
    sigmap : float
        Frequency scaling. Gives the Frequency range of the pulse.

    Returns
    -------
     : float
        Evaluates the function at omega

    """
    if abs(phi_alpha) == 0:
        evaluated = (
                np.heaviside(omega, 1 / 2) *
                np.sqrt(abs(omega) / n(omega)) * f(omega, kp, sigmap)
                )
        return evaluated
    evaluated = (
            np.exp(1j * phi_alpha) *
            np.heaviside(omega, 1 / 2) *
            np.sqrt(abs(omega) / n(omega)) * f(omega, kp, sigmap)
            )
    return evaluated


def delta_k(omega, Omega):
    """ Wave-vector mismatch between a wave vector at frequency omega
    and Omega in the nonlinear crystal

    Parameters
    ----------
    omega : float
        Frequency of first wave vector
    Omega : float
        Frequency of second wave vector

    Returns
    -------
     : float
        Wave-vector mismatch

    """
    dk = 1 / C * (
            (omega + Omega) * n(omega + Omega) -
            omega * n(omega) - Omega * n(Omega)
            )
    return dk


def crystal_fft(k):
    """ Phase matching function, i.e., the Fourier transform of the
    transversal profile of the nonlinear crystal. In free space the
    profile is rectangular and the Fourier-trasform thus a
    sinc function.

    Parameters
    ----------
    k : float
        Wave-vector mismatch

    Returns
    -------
     : float
        Phase matching function

    """
    return lam * L * np.pi * np.sinc(L / (2 * np.pi) * k)


def kernel_S(omega, Omega, kp, sigmap):
    """ Kernel determining the squeezing like nonlinear interaction
    between a photo at frequency omega and Omega.

    Parameters
    ----------
    omega : float
        Frequency of first photon
    Omega : float
        Frequency of second photon
    kp : float
        Cycle parameter. Determines the number of optical cycle
        completed during the pulse.
    sigmap : float
        Frequency scaling. Gives the Frequency range of the pulse.

    Returns
    -------
        complex: Kernel S(omega, Omega)

    """
    prefactor = 1 / HBAR * (
            HBAR / (4 * np.pi * EPSILON0 * C * beamarea)
            )**(3 / 2)
    return prefactor \
        * np.sqrt(abs(omega) / n(omega)) \
        * np.sqrt(abs(Omega) / n(Omega)) \
        * f_p(omega + Omega, kp, sigmap) \
        * crystal_fft(delta_k(omega, Omega))


def kernel_B(omega, Omega, kp, sigmap):
    """ Kernel determining the beam-splitter like nonlinear
    interaction between a photo at frequency omega and Omega.

    Parameters
    ----------
    omega : float
        Frequency of first photon
    Omega : float
        Frequency of second photon
    kp : float
        Cycle parameter. Determines the number of optical cycle
        completed during the pulse.
    sigmap : float
        Frequency scaling. Gives the Frequency range of the pulse.

    Returns
    -------
    B(omega, Omega) : complex
        Kernel B(omega, Omega)

    """
    prefactor = 2 / HBAR * (
        HBAR / (4 * np.pi * EPSILON0 * C * beamarea)
        )**(3 / 2)
    return prefactor \
        * np.sqrt(abs(omega) / n(omega)) \
        * np.sqrt(abs(Omega) / n(Omega)) \
        * (f_p(omega - Omega, kp, sigmap)
            - np.conj(f_p(Omega - omega, kp, sigmap))) \
        * crystal_fft(delta_k(-omega, Omega))


# Functions to calculate the matricies A and B
def squeezing_splitting_values(v, us, kp, sigmap):
    """ Calculates the norm of f_S(omega) = int v*(Omega)
    (S(Omega, omega) + S*(omega, Omega)) d Omega and
    f_B(omega) = int v*(Omega) B(Omega, omega) d Omega
    determining the amount of squeezing/beam splitting like
    interaction contributing to the input-output relation

    Parameters
    ----------
    v : single parameter function
        Output mode function
    us : list of single parameter function
        List of orthonormal input mode functions
    kp : float
        Cycle parameter. Determines the number of optical cycle
        completed during the pulse.
    sigmap : float
        Frequency scaling. Gives the Frequency range of the pulse.

    Returns
    -------
    squeezing_value : float
        Value determining the amount of squeezing
    splitting_value : float
        Value determining the amount of beam splitting

    """
    def kernel(x, y, z):
        return v(x) \
            * (kernel_S(y, x, kp, sigmap)
                + np.conj(kernel_S(x, y, kp, sigmap))) \
            * (np.conj(kernel_S(y, z, kp, sigmap))
                + kernel_S(z, y, kp, sigmap)) \
            * np.conj(v(z))
    squeezing_value = np.sqrt(nquad(
        kernel, [[0, cutfreq], [0, cutfreq], [0, cutfreq]]
        )[0])

    def kernel(x, y, z):
        return np.conj(v(x)) \
            * np.conj(kernel_B(x, y, kp, sigmap)) \
            * kernel_B(z, y, kp, sigmap) * v(z)
    splitting_value = np.sqrt(nquad(
        kernel, [[0, cutfreq], [0, cutfreq], [0, cutfreq]]
        )[0])
    return squeezing_value, splitting_value


def calc_overlaps(v, us, kp, sigmap):
    """ Calculates the matrix elements of the squeezing and
    beam splitting kernel for different combinations of input and
    output mode functions v, us

    Parameters
    ----------
    v : single parameter function
        Output mode function
    us : list of single parameter function
        List of orthonormal input mode functions
    kp : float
        Cycle parameter. Determines the number of optical
        cycle completed during the pulse.
    sigmap : float
        Frequency scaling. Gives the Frequency range of the pulse.

    Returns
    -------
     : list
        List containing the matrix elements
        vBus, uBvs, vSus, uSvs, vBv, vSv, vBSv

    """
    vBus = []
    uBvs = []
    uSvs = []
    vSus = []
    for u in us:
        def kernel(x, y):
            return np.conj(v(x)) * kernel_B(y, x, kp, sigmap) * u(y)
        vBus.append(dblquad(kernel, 0, cutfreq, 0, cutfreq)[0])

        def kernel(x, y):
            return np.conj(u(x)) * kernel_B(y, x, kp, sigmap) * v(y)
        uBvs.append(dblquad(kernel, 0, cutfreq, 0, cutfreq)[0])

        def kernel(x, y):
            return np.conj(u(x)) * (
                np.conj(kernel_S(y, x, kp, sigmap))
                + kernel_S(x, y, kp, sigmap)
                ) * np.conj(v(y))
        uSvs.append(dblquad(kernel, 0, cutfreq, 0, cutfreq)[0])

        def kernel(x, y):
            return np.conj(v(x)) * (
                np.conj(kernel_S(y, x, kp, sigmap))
                + kernel_S(x, y, kp, sigmap)
                ) * np.conj(u(y))
        vSus.append(dblquad(kernel, 0, cutfreq, 0, cutfreq)[0])

    def kernel(x, y):
        return np.conj(v(x)) * kernel_B(y, x, kp, sigmap) * v(y)
    vBv = dblquad(kernel, 0, cutfreq, 0, cutfreq)[0]

    def kernel(x, y):
        return np.conj(v(x)) * (
            np.conj(kernel_S(y, x, kp, sigmap))
            + kernel_S(x, y, kp, sigmap)
            ) * np.conj(v(y))
    vSv = dblquad(kernel, 0, cutfreq, 0, cutfreq)[0]

    def kernel(x, y, z):
        return np.conj(v(x)) * np.conj(kernel_B(x, y, kp, sigmap)) \
            * (np.conj(kernel_S(y, z, kp, sigmap))
                + kernel_S(z, y, kp, sigmap)) * np.conj(v(z))
    vBSv = nquad(kernel, [[0, cutfreq], [0, cutfreq], [0, cutfreq]])[0]

    return [vBus, uBvs, vSus, uSvs, vBv, vSv, vBSv]


def calc_alpha_dependence(mateles, alpha):
    """ Calculates the dependency of the matrix elements calculated
    using calc_overlaps() on the pump/probe amplitude alpha, driving
    the nonlinear interaction.

    Parameters
    ----------
    mateles : list
        List of all matrix elements,
        vBus, uBvs, vSus, uSvs, vBv, vSv, vBSv,
        calculated using calc_overlaps()
    alpha : float
        Pump/probe amplitude driving the nonlinear interaction

    Returns
    -------
    vBus : list
        List of scaled matrix elements vBus
    uBvs : list
        List of scaled matrix elements uBvs
    vSus : list
        List of scaled matrix elements vSus
    uSvs : list
        List of scaled matrix elements uSvs
    vBv : float
        Scaled matrix elements vBv
    vSv : float
        Scaled matrix elements vSv
    vBSv : float
        Scaled matrix elements vBS

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


def calc_theta(splitting_value, squeezing_value, alpha, silent=True):
    """ Calculates the angle theta determining the amount of
    squeezing/beam splitting in the nonlinear interaction

    Parameters
    ----------
    splitting_value : float
        Amount of beam splitting (see squeezing_splitting_values)
    squeezing_value : float
        Amount of squeezing (see squeezing_splitting_values)
    alpha : float
        Amplitude of probe/pump driving the nonlinear interaction
    silent : boolean, optional
        If True does not show plots

    Returns
    -------
    theta_B : float
        Parameter determining the amount of beam splitting
    theta_S : float
        Parameter determining the amount of squeeing
    theta : float
        Parameter difference between squeezing and beam spltting
    mu : float
        cosh(theta) for squeezing regime and cos(theta)
        for beam splitting
    nu : float
        sinh(theta) for squeezing regime and sin(theta)
        for beam splitting

    """
    theta_B = splitting_value * alpha
    theta_S = squeezing_value * alpha

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
    return theta_B, theta_S, theta, mu, nu


def calc_A(theta, mu, nu, v, us, vBus, vSus):
    """ Calculates the matrix A in the input-output relation
    gamma_out = A gamma_in + B gamma_vac.

    Parameters
    ----------
    theta : float
        Angle determining the amount of squeezing/beam splitting in
        the nonlinear interaction.
    mu : float
        For squeezing cosh(theta), for beam splitting cos(theta)
        (see calc_theta)
    nu : float
        For squeezing sinh(theta), for beam splitting sin(theta)
        (see calc_theta)
    v : single parameter function
        Output mode function
    us : list of single parameter function
        List of orthonormal input mode functions
    vBus : list
        List of scaled matrix elements vBus
    vSus : list
        List of scaled matrix elements vSus

    Returns
    -------
    A : numpy matrix

    """
    matele_F = []
    matele_G = []
    N = len(vBus)
    for i in range(N):
        def integrand(x):
            return np.conj(v(x)) * us[i](x)
        matele_F.append(
                mu * quad(integrand, 0, cutfreq)[0]
                - nu * vBus[i] / theta
                )
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


def calc_B(theta, theta_B, theta_S, mu, nu,
           v, us, vBv, vSv, uSvs, uBvs, vBSv):
    """ Calculates the Matrix B in the input-output relation
    gamma_out = A gamma_in + B gamma_vac

    Parameters
    ----------
    theta : float
        Angle determining the amount of squeezing/beam splitting in
        the nonlinear interaction.
    theta_B : float
        Determining the amount of beam splitting (see calc_theta)
    theta_S : float
        Determining the amount of squeeing (see calc_theta)
    mu : float
        For squeezing cosh(theta), for beam splitting cos(theta)
        (see calc_theta)
    nu : float
        For squeezing sinh(theta), for beam splitting sin(theta)
        (see calc_theta)
    v : single parameter function
        Output mode function
    us : list of single parameter function
        List of orthonormal input mode functions
    vBv : float
        Scaled matrix element vBv (see calc_alpha_dependence)
    vSv : float
        Scaled matrix element vSv (see calc_alpha_dependence)
    uSvs : list
        List of scaled matrix elements uSvs (see calc_alpha_dependence)
    uBvs : list
        List of scaled matrix elements uBvs (see calc_alpha_dependence)
    vBSv : float
        List of scaled matrix elements vBSv (see calc_alpha_dependence)

    Returns
    -------
        numpy matrix: Matrix B of the input-output relation

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

        def integrand(x):
            return np.conj(us[i](x)) * v(x)
        uvs.append(quad(integrand, 0, cutfreq)[0])
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
        [np.real(np.conj(h1f)) + normx,
         normz,
         -np.imag(np.conj(h1f)), 0],
        [np.imag(np.conj(h1f)),
         0,
         np.real(np.conj(h1f)) - normx, normz]])
    return B

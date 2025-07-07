""" Module to calculate the transformed Fock state.

"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
from theimpatientphysicist import save_data, is_pickled, hash_dict
from first_order_unitary import (
        genlaguerre, eval_LG, f, squeezing_splitting_values,
        omegap, sigmap, kp, calc_overlaps,
        cutfreq, kernel_S, kernel_B, calc_theta, calc_alpha_dependence,
        calc_A, calc_B)


# Model parameters
# Input mode
k0 = 0.5
sigma0 = 2 * np.pi * 30
N = 1
# Output mode
sigma_out = 2 * np.pi * 20

# Input mode
us = []
for i in range(N):
    us.append(eval_LG(i, k0, sigma0))

params_reduced = {
    "k0": k0,
    "sigma0": sigma0,
    "N": N,
    "sigma_out": sigma_out,
    "omegap": omegap,
    "sigmap": sigmap
}


def W(x, p, sigma_x=1, sigma_p=1, A_inv=np.eye(2), n_ph=3):
    """ Calculates the transformed Wigner function of a Fock state.

    Parameters
    ----------
    x : float
        The position parameter in phase space
    p : float
        The momentum parameter in phase space
    sigma_x : float, optional
        Thermalization in x direction. Defaults to 1.
    sigma_p : float, optional
        Thermalization in p direction. Defaults to 1.
    A_inv : numpy matrix, optional
        Inverse of the transformation matrix A.
        Does not have to be symplectic, but singular.
        Defaults to numpy.eye(2)
    n_ph : int, optional
        Photon number of the Fock state. Defaults to 3.

    Returns
    -------
    W(x, p) : float
        Transformed Wigner function at (x, p)

    """
    sum_lg = 0
    xx = A_inv[0, 0] * x + A_inv[0, 1] * p
    pp = A_inv[1, 0] * x + A_inv[1, 1] * p
    for m in range(n_ph + 1):
        lg1 = genlaguerre(m, - 1 / 2)(
            - 2 * xx**2 / (sigma_x**2 - 2 * sigma_x))
        lg2 = genlaguerre(n_ph - m, - 1 / 2)(
            - 2 * pp**2 / (sigma_p**2 - 2 * sigma_p))
        mon1 = (1 - 2 / sigma_x)**m
        mon2 = (1 - 2 / sigma_x)**(n_ph - m)
        sum_lg += mon1 * lg1 * mon2 * lg2
    det_A = np.linalg.det(A_inv)
    prefactor = abs(det_A) / (np.pi * np.sqrt(sigma_x * sigma_p))
    return prefactor \
        * np.exp(-xx**2 / sigma_x - pp**2 / sigma_p) * sum_lg


def calc_matrecies(f_out, splitting_value, squeezing_value, mateles):
    """ Calculates the matrices A, cov_th, Sigma, S determining the
    statistics of the output state from the input quantum state.

    Parameters
    ----------
    f_out : float
        Central frequency of the output mode function.
    splitting_value : float
        Amount of beam splitting (see squeezing_splitting_values)
    squeezing_value : float
        Amount of squeezing (see squeezing_splitting_values)
    mateles : list
        List of all matrix elements,
        [vBus, uBvs, vSus, uSvs, vBv, vSv, vBSv],
        calculated using calc_overlaps()

    Returns
    -------
    A : numpy matrix
        A nonsingular suqare matrix transforming the phase space
        statistics (usually symplectic, not necessary here)
    cov_th : numpy matrix
         Matrix determining the state independent thermalization

    """
    print("freq: ", f_out)
    # Output mode
    omega_out = f_out * 2 * np.pi
    k_out = (omega_out / sigma_out)**2

    def v(omega):
        return f(omega, k_out, sigma_out)

    print("Input num: ", len(mateles[0]))
    if splitting_value >= squeezing_value:
        alpha = (np.pi / 2) / np.sqrt(
            abs(splitting_value**2 - squeezing_value**2)
        )
    else:
        alpha = 1 * 1e6  # Real number, phase in definitions
    print("alpha: ", alpha)

    theta_B, theta_S, theta, mu, nu = calc_theta(
        splitting_value, squeezing_value, alpha, silent=False
    )
    print("theta: ", theta)

    vBu, uBv, vSu, uSv, vBv, vSv, vBSv = calc_alpha_dependence(
        mateles, alpha)

    A = calc_A(theta, mu, nu, v, us, vBu, vSu)

    print("A: ", A)

    B = calc_B(
        theta, theta_B, theta_S,
        mu, nu, v, us, vBv, vSv, uSv, uBv, vBSv
    )

    cov_th = B @ B.T
    print("cov_th: ", cov_th)

    return A, cov_th


def calc_frequency_dependency(f_out):
    """ Calculates all parameters dependent on f_out.

    Parameters
    ----------
    f_out : float
        Central frequency of the output mode function.

    Returns
    -------
    splitting_value : float
        Norm of f_S(omega), quantifying the amount of beam
        splitting in the nonlinear interaction (splitting_value)
    squeezing_value : float
        Norm of f_B(omega), quantifying the amount of squeezing
        in the nonlinear interaction (squeezing_value)
    mateles : list
        Matrix elements [vBus, uBvs, vSus, uSvs, vBv, vSv, vBSv]
        of S and B between us and v (see calc_overlaps).

    """
    print("freq: ", f_out)
    # Output mode
    omega_out = f_out * 2 * np.pi
    k_out = (omega_out / sigma_out)**2

    def v(omega):
        return f(omega, k_out, sigma_out)

    params = {
        "k0": k0,
        "sigma0": sigma0,
        "N": N,
        "omega_out": omega_out,
        "sigma_out": sigma_out,
        "omegap": omegap,
        "sigmap": sigmap
    }

    # Calculate Matrices
    pickled_params = is_pickled(params)
    if pickled_params is None:
        squeezing_value, splitting_value = \
            squeezing_splitting_values(v, us, kp, sigmap)
    else:
        splitting_value = pickled_params["splitting_value"]
        squeezing_value = pickled_params["squeezing_value"]

    if pickled_params is None:
        mateles = calc_overlaps(v, us, kp, sigmap)
        to_pickle = {
            "splitting_value": splitting_value,
            "squeezing_value": squeezing_value,
            "mateles": mateles
        }
        with open(f"Pickled/{hash_dict(params)}", 'wb') as file:
            pickle.dump(to_pickle, file)
    else:
        mateles = pickled_params["mateles"]
    return splitting_value, squeezing_value, mateles


def plot_frequency_conversion(f_out, save=False):
    """ Plots the kernel S(omega, Omega) and B(omega, Omega)
    determining the nonlinear interaction. Also plots the input and
    output modes.

    Parameters
    ----------
    f_out : float
        Central frequency of the output mode function.
    save : boolean, optional
        If True saves the plotted data to CSV.

    """
    print("freq: ", f_out)
    # Output mode
    omega_out = f_out * 2 * np.pi
    k_out = (omega_out / sigma_out)**2

    def v(omega):
        return f(omega, k_out, sigma_out)

    # Input, output modes and nonlinear interaction
    fig, axs = plt.subplots(1, 3)
    x = np.linspace(0, cutfreq, 100)
    X, Y = np.meshgrid(x, x)
    p1 = axs[0].pcolormesh(
        X / (2 * np.pi),
        Y / (2 * np.pi),
        kernel_S(X, Y, kp, sigmap)
    )
    fig.colorbar(p1)
    p2 = axs[1].pcolormesh(
        X / (2 * np.pi),
        Y / (2 * np.pi),
        kernel_B(X, Y, kp, sigmap)
    )
    fig.colorbar(p2)

    omegas = np.linspace(0, cutfreq, 100)
    fs = omegas / (2 * np.pi)
    axs[2].plot(fs, v(omegas), color="black", label="output")
    axs[2].plot(fs, us[0](omegas), color="C0", label="input")
    axs[2].legend()
    plt.show()
    if save:
        data = []
        for freq in fs:
            omega = 2 * np.pi * freq
            data.append([freq, v(omega), us[0](omega)])
        save_data(
            "onemodefunctions",
            "f,v,u0",
            data,
            comment=str(params_reduced)
        )


def plot_wigner_function(
        f_out, splitting_value, squeezing_value, mateles, save=False):
    """ Plots the Wigner function of the input Fock state,
    and the transformed output state as well as a cut at x=0.

    Parameters
    ----------
    f_out : float
        Central frequency of the output mode function.
    splitting_value : float
        Amount of beam splitting (see squeezing_splitting_values)
    squeezing_value : float
        Amount of squeezing (see squeezing_splitting_values)
    mateles : list
        List of all matrix elements,
        [vBus, uBvs, vSus, uSvs, vBv, vSv, vBSv],
        calculated using calc_overlaps()
    save : boolean, optional
        If True saves the plotted data to CSV

    """
    # Output mode
    omega_out = f_out * 2 * np.pi
    k_out = (omega_out / sigma_out)**2

    def v(omega):
        return f(omega, k_out, sigma_out)

    A, cov_th = calc_matrecies(f_out,
                               splitting_value,
                               squeezing_value,
                               mateles
                               )

    # Plot Wigner function
    A_inv = np.linalg.inv(A)
    cov_th_A = A_inv @ cov_th @ A_inv.T

    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    fig, axs = plt.subplots(2, 2)
    sigma_x = cov_th_A[0, 0] + 1
    sigma_p = cov_th_A[1, 1] + 1
    axs[0, 0].pcolormesh(
        X, Y, W(X, Y,
                sigma_x=sigma_x, sigma_p=sigma_p, A_inv=A_inv)
    )
    axs[1, 0].plot(
        x, W(x, 0, sigma_x=sigma_x, sigma_p=sigma_p, A_inv=A_inv)
    )
    axs[1, 0].plot(
        x, W(0, x, sigma_x=sigma_x, sigma_p=sigma_p, A_inv=A_inv)
    )
    sigma_x = cov_th_A[0, 0] + 1
    sigma_p = cov_th_A[1, 1] + 1
    axs[0, 1].pcolormesh(X, Y, W(X, Y))
    axs[1, 1].plot(x, W(x, 0))
    axs[1, 1].plot(x, W(0, x))
    axs[0, 0].set_title("f_out: " + str(f_out))
    plt.show()
    if save:
        data = []
        for i in x:
            for j in y:
                data.append(
                    [i, j,
                     W(i, j,
                       sigma_x=sigma_x, sigma_p=sigma_p, A_inv=A_inv),
                     W(i, j)]
                )
            save_data(
                "fock_state_wigner",
                "x,p,wOut,wIn",
                data,
                comment=str(params_reduced)
            )


def scalings(f_out, splitting_value, squeezing_value, mateles, alpha):
    """ Calculates the (quasi) symplectic transformation parameters
    A[0, 0] and A[1, 1]

    Parameters
    ----------
    f_out : float
        Central frequency of the output mode function.
    splitting_value : float
        Amount of beam splitting (see squeezing_splitting_values)
    squeezing_value : float
        Amount of squeezing (see squeezing_splitting_values)
    mateles : list
        List of all matrix elements,
        [vBus, uBvs, vSus, uSvs, vBv, vSv, vBSv],
        calculated using calc_overlaps()
    alpha : float
        Pump/probe amplitude driving the nonlinear interaction

    Returns
    -------
    A[0, 0]: float
        Scaling of the x quadrature of the phase space.
    A[1, 1]: float
        Scaling of the p quadrature of the phase space.

    """
    omega_out = f_out * 2 * np.pi
    k_out = (omega_out / sigma_out)**2

    def v(omega):
        return f(omega, k_out, sigma_out)

    theta_B, theta_S, theta, mu, nu = \
        calc_theta(splitting_value, squeezing_value, alpha)
    vBu, uBv, vSu, uSv, vBv, vSv, vBSv = \
        calc_alpha_dependence(mateles, alpha)

    A = calc_A(theta, mu, nu, v, us, vBu, vSu)
    return A[0, 0], A[1, 1]


def sigma_xp(f_out, splitting_value, squeezing_value, mateles, alpha):
    """ Calculates the thermalization in the x and p direction of the
    phase space.

    Parameters
    ----------
    f_out : float
        Central frequency of the output mode function.
    splitting_value : float
        Amount of beam splitting (see squeezing_splitting_values)
    squeezing_value : float
        Amount of squeezing (see squeezing_splitting_values)
    mateles : list
        List of all matrix elements,
        [vBus, uBvs, vSus, uSvs, vBv, vSv, vBSv],
        calculated using calc_overlaps()
    alpha : float
        Pump/probe amplitude driving the nonlinear interaction

    Returns
    -------
    therm_x : float
        The thermalization in x direction of the phase space (sigma_x)
        transforming the x quadrature of the
    therm_p : float
        The thermalization in p direction of the phase space (sigma_p)
        transforming the x quadrature of the
    therm_x * therm_p : float
        Product of thermalization

    """
    omega_out = f_out * 2 * np.pi
    k_out = (omega_out / sigma_out)**2

    def v(omega):
        return f(omega, k_out, sigma_out)

    theta_B, theta_S, theta, mu, nu = \
        calc_theta(splitting_value, squeezing_value, alpha)
    vBu, uBv, vSu, uSv, vBv, vSv, vBSv = \
        calc_alpha_dependence(mateles, alpha)
    B = calc_B(
        theta, theta_B, theta_S,
        mu, nu, v, us, vBv, vSv, uSv, uBv, vBSv
    )
    cov_th = B @ B.T
    therm_x = cov_th[0, 0] + 1
    therm_p = cov_th[1, 1] + 1
    return therm_x, therm_p, therm_x * therm_p


def calc_transformation_params():
    """ Calculate the transformation parameters A[0, 0], A[1, 1],
    sigma_x, sigma_p, describing the output quantum state after the
    transformation with the nonlinear interaction.

    """
    data = []
    f_outs = list(range(170, 231))
    alphas = np.linspace(1e5, 5e6, 100)
    for f_out in f_outs:
        splitting_value, squeezing_value, mateles = \
            calc_frequency_dependency(f_out)
        for alpha in alphas:
            scale_x, scale_p = scalings(
                f_out, splitting_value,
                squeezing_value, mateles, alpha)
            therm_x, therm_p, uncert = sigma_xp(
                f_out, splitting_value,
                squeezing_value, mateles, alpha)

            data.append(
                [alpha, f_out,
                    scale_x, scale_p, therm_x, therm_p]
            )

    save_data(
        "state_transformation11",
        "alpha,fOut,scalex,scalep,thermx,thermp",
        data, comment=str(params_reduced)
    )


if __name__ == "__main__":
    f_out = 173
    plot_frequency_conversion(f_out)
    splitting_value, squeezing_value, mateles = \
        calc_frequency_dependency(f_out)
    plot_wigner_function(
        f_out, splitting_value, squeezing_value, mateles)
    calc_transformation_params()

""" Module to calculate the transformed multimode squeezed state.

"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
from theimpatientphysicist import save_data, is_pickled, hash_dict
from generalized_bloch_messiah import gbmd
from first_order_unitary import (
        eval_LG, f, squeezing_splitting_values,
        omegap, sigmap, kp, calc_overlaps,
        cutfreq, kernel_S, kernel_B, calc_theta, calc_alpha_dependence,
        calc_A, calc_B)


# Model parameters
# Input mode
k0 = 0.5
sigma0 = 2 * np.pi * 30
N = 3
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


def W(x, p, cov=np.eye(2)):
    """ Calculates the Wigner function of a Gaussian state.

    Parameters
    ----------
    x : float
        The position parameter in phase space
    p : float
        The momentum parameter in phase space
    cov : numpy matrix, optional
        Covariance matrix of the Gaussian distribution

    Returns
    -------
    W(x, p) : float
        Gaussian Wigner function at (x, p)

    """
    cov_inv = np.linalg.inv(cov)
    return 1 / (np.pi * np.sqrt(np.linalg.det(cov))) \
        * np.exp(
                - x**2 * cov_inv[0, 0]
                - p**2 * cov_inv[1, 1]
                - x * p * (cov_inv[0, 1] + cov_inv[1, 0])
                )


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
    Sigma : numpy matrix
        Sigma = np.matrix([
        [Sigma_x, 0 , 0, 0],
        [0, 0 , Sigma_p, 0]])
        with the diagonal, generalized Bloch Messiah matrecies Sigma_x
        and Sigma_p.
    S : numpy matrix
        Symplectic matrix transforming into the generalized Bloch
        Messiah basis.

    """
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

    Sigma, S = gbmd(A)
    return A, cov_th, Sigma, S


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
    silent : boolean, optional
        If True does not show plots

    """
    params = {
        "k0": k0,
        "sigma0": sigma0,
        "N": N,
        "f_out": f_out,
        "sigma_out": sigma_out,
        "omegap": omegap,
        "sigmap": sigmap
    }

    # Output mode
    omega_out = f_out * 2 * np.pi
    k_out = (omega_out / sigma_out)**2

    splitting_value, squeezing_value, mateles = \
        calc_frequency_dependency(f_out)

    A, cov_th, Sigma, S = calc_matrecies(
            f_out, splitting_value, squeezing_value, mateles)

    def v(omega):
        return f(omega, k_out, sigma_out)

    def u_red_x(omega, i):
        """ Test

        """
        s = 0
        for j in range(N):
            s += S[i, j] * us[j](omega)
        return s

    def u_red_p(omega, i):
        """ TODO: Docstring

        """
        s = 0
        for j in range(N):
            s += S[i + N, j + N] * us[j](omega)
        return s

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
    axs[2].plot(fs, us[0](omegas), color="C0", label="input 1")
    axs[2].plot(fs, us[1](omegas), color="C1", label="input 2")
    axs[2].plot(fs, us[2](omegas), color="C2", label="input 3")
    axs[2].plot(fs,
                u_red_x(omegas, 0),
                color="C0",
                linestyle="dashed",
                label="Transformed x"
                )
    axs[2].plot(fs,
                u_red_p(omegas, 0),
                color="C0",
                linestyle="dotted",
                label="Transformed p"
                )
    axs[2].legend()
    plt.show()
    if save:
        data = []
        for freq in fs:
            omega = 2 * np.pi * freq
            data.append(
                    [freq,
                     v(omega),
                     us[0](omega),
                     us[1](omega),
                     us[2](omega),
                     u_red_x(omega, 0),
                     u_red_p(omega, 0)
                     ])
        save_data(
                "modefunctions",
                "f,v,u0,u1,u2,uredx,uredp",
                data,
                comment=str(params)
                )


def plot_wigner_function(f_out, cov, save=False):
    """ Plots the Wigner function of the input Fock state,
    and the transformed output state as well as a cut at x=0.

    Parameters
    ----------
    f_out : float
        Central frequency of the output mode function.
    cov : numpy matrix
        Covariace matrix of the Gaussian input state.
    save : boolean, optional
        If True saves the plotted data to CSV.

    """
    params = {
        "k0": k0,
        "sigma0": sigma0,
        "N": N,
        "f_out": f_out,
        "sigma_out": sigma_out,
        "omegap": omegap,
        "sigmap": sigmap
    }

    # Get matrices
    splitting_value, squeezing_value, mateles = \
        calc_frequency_dependency(f_out)

    A, cov_th, Sigma, S = calc_matrecies(
            f_out, splitting_value, squeezing_value, mateles)

    # Plot Wigner function
    cov_S = S @ cov @ S.T

    # Calculate reduced covariance matrices
    Sigma_x = Sigma[0, 0]
    Sigma_p = Sigma[1, 3]
    scale = np.sqrt(sum([S[0, j]**2 for j in range(3)]))

    cov_red1 = np.matrix([
        [Sigma_x**2 * cov_S[0, 0], cov_S[0, 3]],
        [cov_S[3, 0], Sigma_p**2 * cov_S[3, 3]]]) + cov_th

    cov_red2 = np.matrix([
        [Sigma_x**2 * cov_S[0, 0], cov_S[0, 3]],
        [cov_S[3, 0], Sigma_p**2 * cov_S[3, 3]]])

    cov_red3 = np.matrix([
        [cov_S[0, 0], cov_S[0, 3]],
        [cov_S[3, 0], cov_S[3, 3]]]) / scale**2

    sq_x = np.log(cov_red3[0, 0]) / 2
    sq_p = np.log(cov_red3[1, 1]) / 2
    print("Squeezing x: ", sq_x)
    print("Squeezing p: ", sq_p)

    x = np.linspace(-6, 6, 100)
    y = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x, y)
    fig, axs = plt.subplots(2, 3)

    # Input modes input
    cov_in = []
    for i in range(3):
        cov_red = np.matrix([
            [cov[i, i], cov[i, i + 3]],
            [cov[i + 3, i], cov[i + 3, i + 3]]])
        cov_in.append(cov_red)
        axs[0, i].pcolormesh(X, Y, W(X, Y, cov_red))
        axs[0, i].set_title(f"Input mode {i+1}")

    # First mode output + thermal
    axs[1, 0].pcolormesh(X, Y, W(X, Y, cov=cov_red1))
    axs[1, 0].set_title("Output statistic (thermal)")

    # First mode output
    scale = np.sqrt(sum([S[0, j]**2 for j in range(3)]))
    axs[1, 1].pcolormesh(X, Y, W(X, Y, cov=cov_red2))
    axs[1, 1].set_title("Output statistic (pure)")

    # First mode output + unsqueezed
    axs[1, 2].pcolormesh(X, Y, W(X, Y, cov=cov_red3))
    axs[1, 2].set_title("Output statistic (pure + unsqueezed)")
    plt.show()
    if save:
        data = []
        for i in x:
            for j in y:
                data.append(
                        [i, j,
                         W(i, j, cov=cov_in[0]),
                         W(i, j, cov=cov_in[1]),
                         W(i, j, cov=cov_in[2]),
                         W(i, j, cov=cov_red1),
                         W(i, j, cov=cov_red2),
                         W(i, j, cov=cov_red3)
                         ]
                        )
        save_data(
                "squeezed_state_wigner",
                "x,p,wIn1,wIn2,wIn3,wOut,wOutP,wOutS",
                data,
                comment=str(params)
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
    f_outs = list(range(170, 236))
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


def calc_input_bm_overlap(cov):
    """ Calculates the overlap between the input modes and the
    generalized Bloch Messiah modes

    Parameters
    ----------
    cov : numpy matrix
        Covariace matrix of the Gaussian input state.

    """
    S00 = []
    S01 = []
    S02 = []
    S10 = []
    S11 = []
    S12 = []
    sqs_x = []
    sqs_p = []
    data1 = []
    data2 = []
    f_outs = list(range(170, 236, 1))
    print(f_outs)
    for f_out in f_outs:
        splitting_value, squeezing_value, mateles = \
            calc_frequency_dependency(f_out)
        A, cov_th, Sigma, S = calc_matrecies(
                f_out, splitting_value, squeezing_value, mateles)

        # Effective squeezing of the "pure" state
        cov_S = S @ cov @ S.T
        scale = np.sqrt(sum([S[0, j]**2 for j in range(3)]))
        sq_x = np.log(cov_S[0, 0] / scale**2) / 2
        sq_p = np.log(cov_S[3, 3] / scale**2) / 2

        S /= scale
        S00.append(abs(S[0, 0]))
        S01.append(abs(S[0, 1]))
        S02.append(abs(S[0, 2]))
        S10.append(abs(S[3, 3]))
        S11.append(abs(S[3, 4]))
        S12.append(abs(S[3, 5]))
        sqs_x.append(sq_x)
        sqs_p.append(sq_p)
        data1.append([
            f_out,
            abs(S[0, 0]),
            abs(S[0, 1]),
            abs(S[0, 2]),
            abs(S[3, 3]),
            abs(S[3, 4]),
            abs(S[3, 5])
            ])
        data2.append([f_out, sq_x, sq_p])
    plt.plot(f_outs, S00, label="x1")
    plt.plot(f_outs, S01, label="x2")
    plt.plot(f_outs, S02, label="x3")
    plt.plot(f_outs, S10, linestyle="dashed", label="p1")
    plt.plot(f_outs, S11, linestyle="dashed", label="p2")
    plt.plot(f_outs, S12, linestyle="dashed", label="p3")
    plt.legend()
    plt.show()

    plt.plot(f_outs, sqs_x, label="sq x")
    plt.plot(f_outs, sqs_p, label="sq p")
    plt.legend()
    plt.show()
    save_data(
            "overlaps",
            "fOut,x1,x2,x3,p1,p2,p3",
            data1,
            comment=str(params_reduced)
            )
    save_data(
            "squeezing_params",
            "fOut,sqx,sqp",
            data2,
            comment=str(params_reduced)
            )


if __name__ == "__main__":
    f_out = 173
    # Define covariance matrix of Gaussian state
    rs = [-np.exp(0), -np.exp(-0.5), -np.exp(-1)]
    ths = [1, 1, 1]
    cov = np.diag(
            [ths[i] * np.exp(2 * rs[i]) for i in range(3)]
            + [ths[i] * np.exp(-2 * rs[i]) for i in range(3)]
            )

    plot_frequency_conversion(f_out, save=True)
    plot_wigner_function(f_out, cov, save=True)
    calc_input_bm_overlap(cov)
    calc_transformation_params()

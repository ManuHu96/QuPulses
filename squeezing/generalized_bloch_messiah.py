""" Module to calculate the generalized Bloch Messiah decomposition

"""
import numpy as np
from thewalrus.decompositions import blochmessiah


def Sym_mat(n):
    """ Matrix (Omega) defining the 2n x 2n symplectic matrices,
    i.e., M.T @ Omega @ M = Omega

    Parameters
    ----------
    n : int
        Half the dimension of phase space

    Returns
    -------
     : numpy matrix
        Matrix defining the symplectic structure

    """
    return np.block([
        [np.zeros((n, n)), np.eye(n)],
        [-np.eye(n), np.zeros((n, n))]])


def symplectic_housholder(a, e):
    """ Calculated the symplectic Housholder matrix which transforms
    the vector a to the vector e

    Parameters
    ----------
    a : numpy matrix
        3D-Column vector which gets transformed to e
    e : numpy matrix
        3D-Column vector which is the target

    Returns
    -------
     : numpy matrix
        symplectic Housholder matrix transforming a to e

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

    Parameters
    ----------
    A : numpy matrix
        Matrix to calculate the generalized Bloch Messiah decomposition

    Returns
    -------
    Sigma : numpy matrix
        Containes the generalized squeezing parameters
    S : numpy matrix
        Symplectic matrix

    """
    a = np.matrix(A[1, 3:6]).T
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
    # A_dec = Sigma @ S_eff
    return Sigma, S_eff

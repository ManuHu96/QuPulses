import numpy as np
from scipy.special import gamma
from scipy.integrate import quad, dblquad, nquad
import matplotlib.pyplot as plt
from thewalrus.decompositions import blochmessiah
from definitions_emanuel import *

# Model parameters
# Input mode
k0 = 0.5
sigma0 = 2 * np.pi * 30
N = 1
# Output mode
f_out = 230
sigma_out = 2 * np.pi * 20
# Nonlinear interaction
#alpha = np.pi / 2 / 1.669050403075085e-06 #230
alpha = np.pi / 2 / 1.2811282761865378e-06 #210
#alpha = np.pi / 2 / 3.718869038085251e-07 #190
print(alpha)

def calculate_fock_state(f_out, silent=False, save=False):
    """ TODO: Docstring

    """
    print("freq: ", f_out)
    omega_out = f_out * 2 * np.pi
    params = {
            "k0" : k0,
            "sigma0" : sigma0,
            "N" : N,
            "omega_out" : omega_out,
            "sigma_out" : sigma_out, 
            "omegap" : omegap,
            "sigmap" : sigmap
    }
    
    # Input mode
    us = []
    for i in range(N):
        us.append(eval_LG(i, k0, sigma0))
    
    # Output mode
    k_out = (omega_out / sigma_out)**2
    v = lambda omega: f(omega, k_out, sigma_out)
    
    # Nonlinear interaction
    kp = (omegap / sigmap)**2
    
    # Calculate Matrices
    pickled_params = is_pickled(params)
    if pickled_params == None:
        theta_B_num, theta_S_num = calc_thetas(v, us, kp, sigmap)
    else:
        theta_B_num = pickled_params["theta_B_num"]
        theta_S_num = pickled_params["theta_S_num"]
    
    theta_B = theta_B_num * alpha
    theta_S = theta_S_num * alpha
    print(theta_B_num, theta_S_num)
    
    theta, mu, nu = calc_theta(theta_B, theta_S, silent=False)
    print("theta: ", theta)
    
    if pickled_params == None:
        mateles = calc_overlaps(v, us, kp, sigmap)
        to_pickle = {
                "theta_B_num" : theta_B_num,
                "theta_S_num" : theta_S_num,
                "mateles" : mateles
                }
        with open(f"Pickled/{hash_dict(params)}",'wb') as file:
            pickle.dump(to_pickle, file)
    else:
        mateles = pickled_params["mateles"]
    
    vBu, uBv, vSu, uSv, vBv, vSv, vBSv = calc_alpha_dep(mateles, alpha)
    
    A = calc_A(theta, mu, nu, v, us, vBu, vSu)
    
    print("A: ", A)
    
    B = calc_B(theta, theta_B, theta_S, mu, nu, v, us, vBv, vSv, uSv, uBv, vBSv)
            
    cov_th = B @ B.T
    print("cov_th: ", cov_th)
    
    if not silent:
        fig, axs = plt.subplots(1, 3)
        x = np.linspace(0, cutfreq, 100)
        X, Y = np.meshgrid(x, x)
        p1 = axs[0].pcolormesh(X / (2 * np.pi), Y / (2 * np.pi), kernel_S(X, Y, kp, sigmap))
        fig.colorbar(p1)
        p2 = axs[1].pcolormesh(X / (2 * np.pi), Y / (2 * np.pi), kernel_B(X, Y, kp, sigmap))
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
        save_data("onemodefunctions", "f,v,u0", data, comment=str(params))
    
    A_inv = np.linalg.inv(A) 
    cov_th_A = A_inv@cov_th@A_inv.T
    det_A = np.linalg.det(A)
    
    n_ph = 3
    
    def W(x, p, sigma_x=1, sigma_p=1, A_inv=np.eye(2)):
        """ TODO: Docstring
    
        """
        sum_lg = 0
        xx = A_inv[0, 0] * x + A_inv[0, 1] * p
        pp = A_inv[1, 0] * x + A_inv[1, 1] * p
        for m in range(n_ph+1):
            sum_lg += (
                    (1 - 2 / sigma_x)**m * genlaguerre(m, - 1 / 2)(- 2 * xx**2 / (sigma_x**2 - 2 * sigma_x)) * 
                    (1 - 2 / sigma_x)**(n_ph - m) * genlaguerre(n_ph - m, - 1 / 2)(- 2 * pp**2 / (sigma_p**2 - 2 * sigma_p))
                    )
        prefactor = 1 / (np.pi * np.sqrt(sigma_x * sigma_p) * abs(det_A))
        return prefactor * np.exp(-xx**2 / sigma_x - pp**2 / sigma_p) * sum_lg
    
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    if not silent:
        X, Y = np.meshgrid(x, y)
        fig, axs = plt.subplots(2, 2)
        sigma_x = cov_th_A[0, 0] + 1
        sigma_p = cov_th_A[1, 1] + 1
        axs[0,0].pcolormesh(X, Y, W(X, Y, sigma_x=sigma_x, sigma_p=sigma_p, A_inv=A_inv))
        axs[1,0].plot(x, W(x, 0, sigma_x=sigma_x, sigma_p=sigma_p, A_inv=A_inv))
        axs[1,0].plot(x, W(0, x, sigma_x=sigma_x, sigma_p=sigma_p, A_inv=A_inv))
        sigma_x = cov_th_A[0, 0] + 1
        sigma_p = cov_th_A[1, 1] + 1
        axs[0,1].pcolormesh(X, Y, W(X, Y))
        axs[1,1].plot(x, W(x, 0))
        axs[1,1].plot(x, W(0, x))
        axs[0,1].set_title("omega_out: " + str(round(params["omega_out"] / (2 * np.pi), 2)))
        plt.show()
    if save:
        data = []
        for i in x:
            for j in y:
                data.append([i, j, W(i, j, sigma_x=sigma_x, sigma_p=sigma_p, A_inv=A_inv), W(i, j)])
            save_data("fock_state_wigner", "x,p,wOut,wIn", data, comment=str(params))

    @np.vectorize
    def scalings(alpha):
        """ TODO: Docstring
    
        """
        theta_B = theta_B_num * alpha
        theta_S = theta_S_num * alpha
        theta, mu, nu = calc_theta(theta_B, theta_S)
        vBu, uBv, vSu, uSv, vBv, vSv, vBSv = calc_alpha_dep(mateles, alpha)
        
        A = calc_A(theta, mu, nu, v, us, vBu, vSu)
        print(np.sqrt(A[0, 0]*A[1, 1]))
        return A[0, 0], A[1, 1]
    
    @np.vectorize
    def sigma_xp(alpha):
        """ TODO: Docstring
    
        """
        theta_B = theta_B_num * alpha
        theta_S = theta_S_num * alpha
        theta, mu, nu = calc_theta(theta_B, theta_S)
        vBu, uBv, vSu, uSv, vBv, vSv, vBSv = calc_alpha_dep(mateles, alpha)
        B = calc_B(theta, theta_B, theta_S, mu, nu, v, us, vBv, vSv, uSv, uBv, vBSv)
                
        cov_th = B @ B.T
        sigma_x = cov_th[0, 0] + 1
        sigma_p = cov_th[1, 1] + 1
        return sigma_x, sigma_p, sigma_x * sigma_p
    
    x = np.linspace(1e5, 5e6, 100)
    scale_x, scale_p = scalings(x)
    sigma_x, sigma_p, sigma = sigma_xp(x)
    if not silent:
        plt.plot(x, scale_x)
        plt.plot(x, scale_p)
        plt.plot(x, np.sqrt(scale_p * scale_x))
        plt.show()
        print(min(sigma))
        plt.plot(x, sigma_x)
        plt.plot(x, sigma_p)
        plt.plot(x, sigma)
        plt.show()
    return scale_x, scale_p, sigma_x, sigma_p, sigma

def calculate_transformation_params(params):
    """ TODO: Docstring

    """
    f_outs = list(range(170, 231))
    data = []
    x = np.linspace(1e5, 5e6, 100)
    for f_out in f_outs:
        scale_x, scale_p, therm_x, therm_p, sigma = calculate_fock_state(f_out, silent=True)
        
        for i in range(len(x)):
            data.append([x[i], f_out, scale_x[i], scale_p[i], therm_x[i],  therm_p[i]])
    
    save_data("state_transformation11", "alpha,fOut,scalex,scalep,thermx,thermp", data, comment=str(params))

calculate_fock_state(f_out, save=True)
params = {
        "k0" : k0,
        "sigma0" : sigma0,
        "N" : N,
        "sigma_out" : sigma_out, 
        "omegap" : omegap,
        "sigmap" : sigmap
}
calculate_transformation_params(params)


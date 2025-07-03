import numpy as np
from scipy.special import gamma, genlaguerre, genlaguerre
from scipy.integrate import quad, dblquad, nquad
import matplotlib.pyplot as plt
from definitions_emanuel import *

# Model parameters
# Input mode
k0 = 0.5
sigma0 = 2 * np.pi * 30
N = 3
# Output mode
f_out = 195
sigma_out = 2 * np.pi * 20
# Nonlinear interaction

def calculate_squeezed_state(f_out, silent=False, save=False):
    """ TODO:

    """
    print("freq: ", f_out)
    omega_out = 2 * np.pi * f_out
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
    
    if theta_B_num >= theta_S_num:
        alpha = np.pi / 2 * 1 / np.sqrt(abs(theta_B_num**2 - theta_S_num**2))
    else:
        alpha = 1 * 1e6 # Real number, phase in definitions
    print("alpha: ", alpha)

    theta_B = theta_B_num * alpha
    theta_S = theta_S_num * alpha
    
    theta, mu, nu = calc_theta(theta_B, theta_S)
    print("theta, ", theta)
    
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
    
    Sigma, S = gbmd(A)
    
    if not silent:
        fig, axs = plt.subplots(1, 3)
        x = np.linspace(0, cutfreq, 100)
        X, Y = np.meshgrid(x, x)
        p1 = axs[0].pcolormesh(X / (2 * np.pi), Y / (2 * np.pi), kernel_S(X, Y, kp, sigmap))
        fig.colorbar(p1)
        p2 = axs[1].pcolormesh(X / (2 * np.pi), Y / (2 * np.pi), kernel_B(X, Y, kp, sigmap))
        fig.colorbar(p2)
    if save:
        data = []
        for xi in x:
            for yi in x:
                data.append([xi / (2 * np.pi), yi / (2 * np.pi), kernel_S(xi, yi, kp, sigmap), kernel_B(xi, yi, kp, sigmap)])
        save_data("kernel", "f1,f2,S,B", data, comment=str(params))


    
    #O, D, Q = blochmessiah(S)
    #P = np.matrix([
    #    [1, 0, 0, 0, 0, 0],
    #    [0, 0, 0, 1, 0, 0]])
    #print(P @ O)
    #print(Q)

    def u_red_x(omega, i):
        """ TODO: Docstring
    
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
    
    
    u0_red_x = lambda omega: u_red_x(omega, 0)
    u1_red_x = lambda omega: u_red_x(omega, 1)
    u2_red_x = lambda omega: u_red_x(omega, 2)
    u0_red_p = lambda omega: u_red_p(omega, 0)
    u1_red_p = lambda omega: u_red_p(omega, 1)
    u2_red_p = lambda omega: u_red_p(omega, 2)
    
    omegas = np.linspace(0, cutfreq, 100)
    fs = omegas / (2 * np.pi)
    if not silent:
        axs[2].plot(fs, v(omegas), color="black", label="output")
        axs[2].plot(fs, us[0](omegas), color="C0", label="input 1")
        axs[2].plot(fs, us[1](omegas), color="C1", label="input 2")
        axs[2].plot(fs, us[2](omegas), color="C2", label="input 3")
        axs[2].plot(fs, u0_red_x(omegas), color="C0", linestyle="dashed", label="Transformed x")
        axs[2].plot(fs, u0_red_p(omegas), color="C0", linestyle="dotted", label="Transformed p")
        axs[2].legend()
        plt.show()
    if save:
        data = []
        for freq in fs:
            omega = 2 * np.pi * freq
            data.append([freq, v(omega), us[0](omega), us[1](omega), us[2](omega), u0_red_x(omega), u0_red_p(omega)])
        save_data("modefunctions", "f,v,u0,u1,u2,uredx,uredp", data, comment=str(params))

    
    rs = [-np.exp(0), -np.exp(-0.5), -np.exp(-1)]
    ths = [1, 1, 1]
    cov = np.diag(
            [ths[i] * np.exp(2 * rs[i]) for i in range(3)] + [ths[i] * np.exp(-2 * rs[i]) for i in range(3)]
            )
    
    cov_S = S @ cov @ S.T
    
    def W(x, p, cov=np.eye(2)):
        """ TODO: Docstring
    
        """
        cov_inv = np.linalg.inv(cov)
        return 1 / (np.pi * np.sqrt(np.linalg.det(cov))) * np.exp(- x**2 * cov_inv[0, 0] - p**2 * cov_inv[1, 1] - x * p * (cov_inv[0, 1] + cov_inv[1, 0]))

    scale = np.sqrt(sum([S[0, j]**2 for j in range(3)]))
    cov_red3 = np.matrix([
        [cov_S[0,0], cov_S[0,3]],
        [cov_S[3,0], cov_S[3,3]]]) / scale**2
    sq_x = np.log(cov_red3[0,0]) / 2
    sq_p = np.log(cov_red3[1,1]) / 2
    print("Squeezing x: ", sq_x)
    print("Squeezing p: ", sq_p)
    
    if not silent:
        x = np.linspace(-6, 6, 100)
        y = np.linspace(-6, 6, 100)
        X, Y = np.meshgrid(x, y)
        fig, axs = plt.subplots(2, 3)
        # Input modes input
        cov_in = []
        for i in range(3):
            cov_red = np.matrix([
                [cov[i,i], cov[i,i + 3]],
                [cov[i + 3,i], cov[i + 3,i + 3]]])
            cov_in.append(cov_red)
            axs[0, i].pcolormesh(X, Y, W(X, Y, cov_red))
            axs[0, i].set_title(f"Input mode {i+1}")
        
        # First mode output + thermal
        Sigma_x = Sigma[0, 0]
        Sigma_p = Sigma[1, 3]
        cov_red1 = np.matrix([
            [Sigma_x**2 * cov_S[0,0], cov_S[0,3]],
            [cov_S[3,0], Sigma_p**2 * cov_S[3,3]]]) + cov_th
        axs[1, 0].pcolormesh(X, Y, W(X, Y, cov=cov_red1))
        axs[1, 0].set_title("Output statistic (thermal)")
        
        # First mode output
        scale = np.sqrt(sum([S[0, j]**2 for j in range(3)]))
        cov_red2 = np.matrix([
            [Sigma_x**2 * cov_S[0,0], cov_S[0,3]],
            [cov_S[3,0], Sigma_p**2 * cov_S[3,3]]])
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
            save_data("squeezed_state_wigner", "x,p,wIn1,wIn2,wIn3,wOut,wOutP,wOutS", data, comment=str(params))
        
    @np.vectorize
    def scalings(alpha):
        """ TODO: Docstring
    
        """
        theta_B = theta_B_num * alpha
        theta_S = theta_S_num * alpha
        theta, mu, nu = calc_theta(theta_B, theta_S)
        vBu, uBv, vSu, uSv, vBv, vSv, vBSv = calc_alpha_dep(mateles, alpha)
        
        A = calc_A(theta, mu, nu, v, us, vBu, vSu)
        Sigma, S = gbmd(A)
        return Sigma[0, 0], Sigma[1, 3]
    
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
        plt.show()
        min_index = np.argmin(sigma)
        print(x[min_index], sigma[min_index])
        plt.plot(x, sigma_x)
        plt.plot(x, sigma_p)
        plt.plot(x, sigma)
        plt.show()
    return S, scale_x, scale_p, sigma_x, sigma_p, sigma, sq_x, sq_p

def calculate_transformation_params(params):
    """ TODO: Docstring

    """
    f_outs = list(range(170, 231, 1))
    data = []
    x = np.linspace(1e5, 5e6, 100)
    for f_out in f_outs:
        S, scale_x, scale_p, therm_x, therm_p, sigma, sq_x, sq_p = calculate_squeezed_state(f_out, silent=True)
        
        for i in range(len(x)):
            data.append([x[i], f_out, scale_x[i], scale_p[i], therm_x[i],  therm_p[i]])
    
    save_data("state_transformation31", "alpha,fOut,scalex,scalep,thermx,thermp", data, comment=str(params))

def calculate_overlap():
    """ TODO: Docstring

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
    f_outs = list(range(170, 231, 1))
    for f_out in f_outs:
        S, scale_x, scale_p, sigma_x, sigma_p, sigma, sq_x, sq_p = calculate_squeezed_state(f_out, silent=True)
        scale = np.sqrt(sum([S[0, j]**2 for j in range(3)]))
        S /= scale
        S00.append(abs(S[0,0]))
        S01.append(abs(S[0,1]))
        S02.append(abs(S[0,2]))
        S10.append(abs(S[3,3]))
        S11.append(abs(S[3,4]))
        S12.append(abs(S[3,5]))
        sqs_x.append(sq_x)
        sqs_p.append(sq_p)
        data1.append([
            f_out, 
            abs(S[0,0]), 
            abs(S[0,1]), 
            abs(S[0,2]), 
            abs(S[3,3]), 
            abs(S[3,4]), 
            abs(S[3,5])
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
    save_data("overlaps", "fOut,x1,x2,x3,p1,p2,p3", data1, comment=str(params))
    save_data("squeezing_params", "fOut,sqx,sqp", data2, comment=str(params))

calculate_squeezed_state(f_out, save=True)
params = {
        "k0" : k0,
        "sigma0" : sigma0,
        "N" : N,
        "sigma_out" : sigma_out, 
        "omegap" : omegap,
        "sigmap" : sigmap
}
#calculate_transformation_params(params)
calculate_overlap()

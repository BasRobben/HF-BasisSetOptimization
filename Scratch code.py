import numpy as np
from math import factorial
import matplotlib.pyplot as plt
from scipy import optimize


def optimise_primitive_gaussians(num_primitives, zeta, l, m, n, r):

    # calculate slater type orbital of 1S H2
    sto_1s = np.sqrt((zeta ** 3) / np.pi) * np.exp(-zeta * r)
    sto_2s = np.sqrt((zeta ** 5) / (3 * np.pi)) * r * np.exp(-zeta * r)
    sto_2px = np.sqrt((zeta ** 5) / np.pi) * r * np.exp(-zeta * r)

    # Initial guess matrix (1S, STO-3G, Hydrogen) [1st row: alphas, 2nd row: coefficients]
    # coeff = np.array([[0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00],
    #                   [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00]])

    # Initial guess matrix (1S, STO-3G Carbon)  [1st row: alphas, 2nd row: coefficients]
    # coeff = np.array([[0.7161683735E+02, 0.1304509632E+02, 0.3530512160E+01],
    #                   [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00]])

    # Initial guess matrix (2S and 2P, STO-3G Carbon)
    # [1st row: alphas,
    # 2nd row: coefficients 2P,
    # 3rd row: coefficients 2S]
    # coeff = np.array([[0.2941249355E+01, 0.6834830964E+00, 0.2222899159E+00],
    #                   [0.1559162750E+00, 0.6076837186E+00, 0.3919573931E+00]])
    #                  # [-0.9996722919E-01, 0.3995128261E+00, 0.7001154689E+00]])

    # Initial guess matrix (1S, STO-3G Oxygen)  [1st row: alphas, 2nd row: coefficients]
    # coeff = np.array([[0.1307093214E+03, 0.2380886605E+02, 0.6443608313E+01],
    #                   [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00]])

    # Initial guess matrix (2S and 2P, STO-3G Oxygen)
    # [1st row: alphas,
    # 2nd row: coefficients 2P,
    # 3rd row: coefficients 2S]
    # coeff = np.array([[0.5033151319E+01, 0.1169596125E+01, 0.3803889600E+00],
    #                   [0.1559162750E+00, 0.6076837186E+00, 0.3919573931E+00]])
    #                  # [-0.9996722919E-01, 0.3995128261E+00, 0.7001154689E+00]])

    # # Differential evolution (2S, STO-3G Carbon)
    # coeff = np.array([[2.92459811e+01,  2.81832737e+01,  3.53631668e-01],
    #                   [3.48458711e-02,  1.37978990e-02,  7.18862421e-01]])

    # Matrix for 1S, STO-3G Carbon optimized DOI:https://doi.org/10.1063/1.439203
    # coeff = np.array([[1.53396e2, 2.30596e1, 4.92985],
    #                   [7.07666e-2, 3.95356e-1, 6.63169e-1]])

    # Matrix for 2S, STO-3G Carbon optimized DOI:https://doi.org/10.1063/1.439203
    # coeff = np.array([[4.39495, 6.58157e-1, 1.98374e-1],
    #                   [-9.26218e-2, 3.40674e-1, 7.43772e-1]])

    # Matrix for 2P, STO-3G Carbon optimized DOI:https://doi.org/10.1063/1.439203
    # coeff = np.array([[4.23235, 8.61673e-1, 2.01326e-1],
    #                   [1.10281e-1, 4.62034e-1, 6.27665e-1]])

    # Matrix for 1S, STO-3G Oxygen optimized DOI:https://doi.org/10.1063/1.439203
    # coeff = np.array([[2.77555e2, 4.18923e1, 9.04437],
    #                   [7.03080e-2, 3.95853e-1, 6.61894e-1]])

    # Matrix for 2S, STO-3G Oxygen optimized DOI:https://doi.org/10.1063/1.439203
    # coeff = np.array([[9.28383, 1.14812, 3.53831e-1],
    #                   [-8.75258e-2, 4.07671e-1, 6.76432e-1]])

    # Matrix for 2P, STO-3G Oxygen optimized DOI:https://doi.org/10.1063/1.439203
    coeff = np.array([[8.09648, 1.67764, 3.74101e-1],
                      [1.23326e-1, 4.75101e-1, 6.15173e-1]])


    # calculate normalisation, returns matrix [1 x 3] of normalisation constants
    normalisation = ((((2 * coeff[0, :]) / np.pi) ** (3/4)) *
                     np.sqrt((((8 * coeff[0, :]) ** (l + m + n)) * factorial(l) * factorial(m) * factorial(n)) /
                             (factorial(2 * l) * factorial(2 * m) * factorial(2 * n))))

    # calculate primitive gaussian per alpha, normalisation and coefficient
    # store in matrix of size [n_primitives x i]
    primitive_gaussian_array = np.zeros([num_primitives, len(r)])

    # For S orbitals
    # for i in range(len(coeff[0, :])):
    #     primitive_gaussian = normalisation[i] * coeff[1, i] * np.exp(-coeff[0, i] * (r ** 2))
    #     primitive_gaussian_array[i] = primitive_gaussian

    # For P orbitals
    for i in range(len(coeff[0, :])):
        primitive_gaussian = normalisation[i] * coeff[1, i] * r * np.exp(-coeff[0, i] * (r ** 2))
        primitive_gaussian_array[i] = primitive_gaussian

    # construct the summed gaussian
    sto3g = np.sum(primitive_gaussian_array, axis=0)

    # Plot
    plt.ylim(0, 0.8)
    plt.xlim(0, 5)
    plt.plot(r, primitive_gaussian_array[0], label="Primitive 1")
    plt.plot(r, primitive_gaussian_array[1], label="Primitive 2")
    plt.plot(r, primitive_gaussian_array[2], label="Primitive 3")
    plt.plot(r, sto3g, label="STO-3G (Standard optimized)", linestyle="dashed")
    plt.plot(r, sto_2px, label="Target STO-2P", linestyle="dotted")
    # plt.plot(r, sto_2s, label="Target STO-2S", linestyle="dotted")
    plt.legend()
    plt.show()

    return


# zeta for hydrogen:
# 1S shell: 1.24

# zeta for carbon: DOI: https://doi.org/10.1063/1.1673374
# 1S shell: 5.67
# 2S and 2P shell: 1.72

# zeta for oxygen: DOI: https://doi.org/10.1063/1.1673374
# 1S shell: 7.66
# 2S and 2P shell: 2.25

print(optimise_primitive_gaussians(3, 2.25, 1, 0, 0, np.linspace(0, 6, 1000)))
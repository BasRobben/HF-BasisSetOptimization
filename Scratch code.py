import numpy as np
from math import factorial
import matplotlib.pyplot as plt
from scipy import optimize


def optimise_primitive_gaussians(num_primitives, zeta, l, m, n, r):

    # calculate slater type orbital of 1S H2
    sto_1s = np.sqrt((zeta ** 3) / np.pi) * np.exp(-zeta * r)

    # initial guess matrix (1S, STO-3G, Hydrogen) [1st row: alphas, 2nd row: coefficients]
    coeff = np.array([[0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00],
                      [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00]])

    # calculate normalisation, returns matrix [1 x 3] of normalisation constants
    normalisation = (((2 * coeff[0, :] / np.pi) ** (3/4)) *
                     np.sqrt((((8 * coeff[0, :]) ** (l + m + n)) * factorial(l) * factorial(m) * factorial(n)) /
                             (factorial(2 * l) * factorial(2 * m) * factorial(2 * n))))

    # calculate primitive gaussian per alpha, normalisation and coefficient
    # store in matrix of size [n_primitives x i]
    primitive_gaussian_array = np.zeros([num_primitives, len(r)])

    for i in range(len(coeff[0, :])):
        primitive_gaussian = normalisation[i] * coeff[1, i] * np.exp(-coeff[0, i] * (r ** 2))
        primitive_gaussian_array[i] = primitive_gaussian

    # optimize coeff matrix
    # optimized_coeff = optimize.minimize(sto_1s, coeff, args=(r,), method='BFGS')
    optimized_coeff = optimize.minimize(sto_1s, coeff, method='BFGS')

    # construct the summed gaussian
    sto3g = np.sum(primitive_gaussian_array, axis=0)

    # make plot of primitive gaussians
    plt.ylim(0, 1)
    plt.xlim(0, 6)
    plt.plot(r, primitive_gaussian_array[0])
    plt.plot(r, primitive_gaussian_array[1])
    plt.plot(r, primitive_gaussian_array[2])
    plt.plot(r, sto3g)
    plt.plot(r, sto_1s)
    plt.show()

    return primitive_gaussian_array, sto3g, optimized_coeff




print(optimise_primitive_gaussians(3, 1.24, 0, 0, 0, np.linspace(0, 6, 1000)))
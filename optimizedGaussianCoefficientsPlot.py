import numpy as np
from scipy.optimize import differential_evolution
from math import factorial
import matplotlib.pyplot as plt


# Change zeta parameter in print (at the bottom) for closest fit to that specific orbital of atom
# Also uncomment the parts of code that matter for the target orbital and for the primitive calculation
# And change in def objective the return to the uncommented target sto and the plot as well

# zeta for hydrogen:
# 1S orbital: 1.24

# zeta for carbon: DOI: https://doi.org/10.1063/1.1673374
# 1S orbital: 5.67
# 2S and 2P orbital: 1.72

# zeta for oxygen: DOI: https://doi.org/10.1063/1.1673374
# 1S orbital: 7.66
# 2S and 2P orbital: 2.25


def optimize_gaussians(orbital, num_primitives, zeta, l, m, n, r):

    if orbital == "1S":
        # Calculate STO-1S (target values)
        sto_1s = np.sqrt((zeta ** 3) / np.pi) * np.exp(-zeta * r)
    elif orbital == "2S":
        # Calculate STO-2S (target values)
        sto_2s = np.sqrt((zeta ** 5) / (3 * np.pi)) * r * np.exp(-zeta * r)
    elif orbital == "2P":
        # Calculate STO-2P (target values)
        sto_2p = np.sqrt((zeta ** 5) / np.pi) * r * np.exp(-zeta * r)

    # Objective function to minimize
    def objective(params):
        # Reshape params into coefficients matrix
        coeff = params.reshape(2, num_primitives)

        # Compute normalisation constants
        normalisation = ((((2 * coeff[0, :]) / np.pi) ** (3 / 4)) *
                         np.sqrt((((8 * coeff[0, :]) ** (l + m + n)) * factorial(l) * factorial(m) * factorial(n)) /
                                 (factorial(2 * l) * factorial(2 * m) * factorial(2 * n))))

        if orbital == "1S" or "2S" and n == 0 and m == 0 and l == 0:
            # Calculate the primitive Gaussians (S orbitals)
            primitive_gaussian_array = np.zeros([num_primitives, len(r)])
            for i in range(len(coeff[0, :])):
                primitive_gaussian = normalisation[i] * coeff[1, i] * np.exp(-coeff[0, i] * (r ** 2))
                primitive_gaussian_array[i] = primitive_gaussian
        elif orbital == "2P" and n or m or l == 1:
            # Calculate the primitive Gaussians (P orbitals)
            primitive_gaussian_array = np.zeros([num_primitives, len(r)])
            for i in range(len(coeff[0, :])):
                primitive_gaussian = normalisation[i] * coeff[1, i] * r * np.exp(-coeff[0, i] * (r ** 2))
                primitive_gaussian_array[i] = primitive_gaussian
        else: raise ValueError("Check if orbital is 1S, 2S or 2P in string or if the n, m or l values are correct")

        # Construct the summed Gaussian
        sto3g = np.sum(primitive_gaussian_array, axis=0)

        if orbital == "1S":
            SSE = np.sum((sto3g - sto_1s) ** 2)
        elif orbital == "2S":
            SSE = np.sum((sto3g - sto_2s) ** 2)
        elif orbital == "2P":
            SSE = np.sum((sto3g - sto_2p) ** 2)

        return SSE

    # Set up bounds for optimization
    # Typically, `alphas > 0` and `coefficients` are between -1 and 1
    bounds = [(0, 1000)] * num_primitives + [(-1, 1)] * num_primitives  # Adjust bounds if needed

    # Run differential evolution
    result = differential_evolution(objective, bounds, disp=True)

    # Extract optimized coeff matrix
    optimized_coeff = result.x.reshape(2, num_primitives)

    # Plot results with optimized coefficients
    normalisation = ((((2 * optimized_coeff[0, :]) / np.pi) ** (3 / 4)) *
                     np.sqrt(
                         (((8 * optimized_coeff[0, :]) ** (l + m + n)) * factorial(l) * factorial(m) * factorial(n)) /
                         (factorial(2 * l) * factorial(2 * m) * factorial(2 * n))))

    if orbital == "1S" or "2S" and n == 0 and m == 0 and l == 0:
        # For S orbitals
        primitive_gaussian_array = np.zeros([num_primitives, len(r)])
        for i in range(len(optimized_coeff[0, :])):
            primitive_gaussian = normalisation[i] * optimized_coeff[1, i] * np.exp(-optimized_coeff[0, i] * (r ** 2))
            primitive_gaussian_array[i] = primitive_gaussian
    elif orbital == "2P" and n or m or l == 1:
        # For P orbitals
        primitive_gaussian_array = np.zeros([num_primitives, len(r)])
        for i in range(len(optimized_coeff[0, :])):
            primitive_gaussian = normalisation[i] * optimized_coeff[1, i] * r * np.exp(-optimized_coeff[0, i] * (r ** 2))
            primitive_gaussian_array[i] = primitive_gaussian
    else:
        raise ValueError("Check if orbital is 1S, 2S or 2P in string or if the n, m or l values are correct")

    sto3g = np.sum(primitive_gaussian_array, axis=0)

    # Plot
    # plt.ylim(0, 0.8)
    plt.xlim(left=0, right=6)
    plt.xlabel("r (a.u.)")
    plt.ylabel("Ψ(r)")
    plt.title(f'STO fit for {orbital}')
    for i in range(num_primitives):
        plt.plot(r, primitive_gaussian_array[i], label=f"Primitive {i + 1}")
    plt.plot(r, sto3g, label="STO-3G (DE)", linestyle="dashed")
    if orbital == "1S":
        plt.plot(r, sto_1s, label="Target STO-1S", linestyle="dotted")
    elif orbital == "2S":
        plt.plot(r, sto_2s, label="Target STO-2S", linestyle="dotted")
    elif orbital == "2P":
        plt.plot(r, sto_2p, label="Target STO-2P", linestyle="dotted")
    plt.legend()
    plt.show()

    return optimized_coeff

print(optimize_gaussians("2P",3, 1.72, 1, 0, 0, np.linspace(0, 6, 1000)))
import sys
import numpy as np
import matplotlib.pyplot as plt
from pyqint import HF, PyQInt, MoleculeBuilder, cgf
from pytessel import PyTessel
from scipy.optimize import minimize

## Courtesy of ChatGPT

## Result STO-3G: -111.21925705796176 Hartrees

## Result:
# Optimized Energy: -111.21926208702752 Hartrees
# Optimized Parameters (coefficients and exponents): 
# [ 1.54329155e-01  5.35327822e-01  4.44634997e-01 -9.99669039e-02
#   3.99513292e-01  7.00115139e-01  1.55915860e-01  6.07684087e-01
#   3.91957098e-01  1.54329232e-01  5.35328171e-01  4.44635189e-01
#  -9.99667363e-02  3.99513065e-01  7.00115202e-01  1.55915966e-01
#   6.07684222e-01  3.91957105e-01  7.16168374e+01  1.30450963e+01
#   3.53051261e+00  2.94124910e+00  6.83483234e-01  2.22289937e-01
#   2.94124925e+00  6.83483228e-01  2.22290158e-01  1.30709320e+02
#   2.38088611e+01  6.44360814e+00  5.03315116e+00  1.16959626e+00
#   3.80389436e-01  5.03315112e+00  1.16959613e+00  3.80389215e-01]


def main():
    mol_co = MoleculeBuilder().from_name('CO')

    ## CO molecule
    # Define the atom positions 
    # (we can extract these from the MoleculeBuilder object)
    p_C = [0.000000,0.000000,-1.290265]
    p_O = [0.000000,0.000000,0.967698]

    # Define coefficients and exponents for the STO-3G basis set
    CO_c = [
        [0.154329, 0.535328, 0.444635],     # C 1s
        [-0.099967, 0.399513, 0.700115],    # C 2s
        [0.155916, 0.607684, 0.391957],     # C 2p
        [0.154329, 0.535328, 0.444635],     # O 1s
        [-0.099967, 0.399513, 0.700115],    # O 2s
        [0.155916, 0.607684, 0.391957],     # O 2p
    ]
    
    CO_a = [
        [71.616837, 13.045096, 3.530512],   # C 1s
        [2.941249, 0.683483, 0.22229],      # C 2s
        [2.941249, 0.683483, 0.22229],      # C 2p
        [130.70932, 23.808861, 6.443608],   # O 1s
        [5.033151, 1.169596, 0.380389],     # O 2s
        [5.033151, 1.169596, 0.380389],     # O 2p
    ]

    cgfs = []

    # Carbon atom (including 1s)
    c_1s = cgf(p_C)
    for coeff, alpha in zip(CO_c[0], CO_a[0]):
        c_1s.add_gto(coeff, alpha, 0, 0, 0)
    cgfs.append(c_1s)

    c_2s = cgf(p_C)
    for coeff, alpha in zip(CO_c[1], CO_a[1]):
        c_2s.add_gto(coeff, alpha, 0, 0, 0)
    cgfs.append(c_2s)

    for orientation in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:  # px, py, pz
        c_2p = cgf(p_C)
        for coeff, alpha in zip(CO_c[2], CO_a[2]):
            c_2p.add_gto(coeff, alpha, *orientation)
        cgfs.append(c_2p)

    # Oxygen atom (including 1s)
    o_1s = cgf(p_O)
    for coeff, alpha in zip(CO_c[3], CO_a[3]):
        o_1s.add_gto(coeff, alpha, 0, 0, 0)
    cgfs.append(o_1s)

    o_2s = cgf(p_O)
    for coeff, alpha in zip(CO_c[4], CO_a[4]):
        o_2s.add_gto(coeff, alpha, 0, 0, 0)
    cgfs.append(o_2s)

    for orientation in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:  # px, py, pz
        o_2p = cgf(p_O)
        for coeff, alpha in zip(CO_c[5], CO_a[5]):
            o_2p.add_gto(coeff, alpha, *orientation)
        cgfs.append(o_2p)

    # Optimization parameters: coefficients and exponents for C 1s, O 1s, C 2s, C 2p, O 2s, O 2p
    def objective_function(params):
        # Update coefficients and exponents (C 1s, O 1s, C 2s, C 2p, O 2s, O 2p)
        updated_CO_c = [
            [params[0], params[1], params[2]],  # C 1s coefficients
            [params[3], params[4], params[5]],  # C 2s coefficients
            [params[6], params[7], params[8]],  # C 2p coefficients
            [params[9], params[10], params[11]],  # O 1s coefficients
            [params[12], params[13], params[14]],  # O 2s coefficients
            [params[15], params[16], params[17]],  # O 2p coefficients
        ]
        updated_CO_a = [
            [params[18], params[19], params[20]],  # C 1s exponents
            [params[21], params[22], params[23]],  # C 2s exponents
            [params[24], params[25], params[26]],  # C 2p exponents
            [params[27], params[28], params[29]],  # O 1s exponents
            [params[30], params[31], params[32]],  # O 2s exponents
            [params[33], params[34], params[35]],  # O 2p exponents
        ]
        
        # Rebuild the CGFs based on updated coefficients and exponents
        cgfs_new = []
        # Carbon atom (1s, 2s, 2p)
        c_1s_new = cgf(p_C)
        for coeff, alpha in zip(updated_CO_c[0], updated_CO_a[0]):
            c_1s_new.add_gto(coeff, alpha, 0, 0, 0)
        cgfs_new.append(c_1s_new)

        c_2s_new = cgf(p_C)
        for coeff, alpha in zip(updated_CO_c[1], updated_CO_a[1]):
            c_2s_new.add_gto(coeff, alpha, 0, 0, 0)
        cgfs_new.append(c_2s_new)

        for orientation in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            c_2p_new = cgf(p_C)
            for coeff, alpha in zip(updated_CO_c[2], updated_CO_a[2]):
                c_2p_new.add_gto(coeff, alpha, *orientation)
            cgfs_new.append(c_2p_new)

        # Oxygen atom (1s, 2s, 2p)
        o_1s_new = cgf(p_O)
        for coeff, alpha in zip(updated_CO_c[3], updated_CO_a[3]):
            o_1s_new.add_gto(coeff, alpha, 0, 0, 0)
        cgfs_new.append(o_1s_new)

        o_2s_new = cgf(p_O)
        for coeff, alpha in zip(updated_CO_c[4], updated_CO_a[4]):
            o_2s_new.add_gto(coeff, alpha, 0, 0, 0)
        cgfs_new.append(o_2s_new)

        for orientation in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            o_2p_new = cgf(p_O)
            for coeff, alpha in zip(updated_CO_c[5], updated_CO_a[5]):
                o_2p_new.add_gto(coeff, alpha, *orientation)
            cgfs_new.append(o_2p_new)

        # Perform Hartree-Fock calculation and return the energy
        result_hf = HF().rhf(mol_co, cgfs_new)
        return result_hf['energy']

    # Initial guess for coefficients and exponents (same as default STO-3G values)
    initial_params = np.concatenate([np.array(CO_c).flatten(), np.array(CO_a).flatten()])

    # Perform optimization
    result = minimize(objective_function, initial_params, method='BFGS')

    optimized_energy = result.fun
    optimized_params = result.x

    print(f"Optimized Energy: {optimized_energy} Hartrees")
    print(f"Optimized Parameters (coefficients and exponents): {optimized_params}")

if __name__ == '__main__':
    main()
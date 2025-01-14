import numpy as np
import matplotlib.pyplot as plt
from pyqint import HF, MoleculeBuilder, cgf
from scipy.optimize import differential_evolution


def createCGFs(p_C, p_O, CO_c, CO_a):
    cgfs = []

    # Carbon atom
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

    # Oxygen atom
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

    return cgfs


# Objective function for optimization
def objective_function(params, p_C, p_O, CO_c, CO_a, mol_co):
    # # Optimizing C2p, O2p
    # CO_c[2] = params[:3]
    # CO_a[2] = params[3:6]
    # CO_c[5] = params[6:9]
    # CO_a[5] = params[9:12]

    # Optimizing C2p, C2s, O2p, O2s
    CO_c[1] = params[:3]
    CO_a[1] = params[3:6]
    CO_c[2] = params[6:9]
    CO_a[2] = params[9:12]
    CO_c[4] = params[12:15]
    CO_a[4] = params[15:18]
    CO_c[5] = params[18:21]
    CO_a[5] = params[21:24]

    cgfs = createCGFs(p_C, p_O, CO_c, CO_a)
    result_hf = HF().rhf(mol_co, cgfs)
    energy = result_hf['energy']
    # print(f"Energy: {energy}, Parameters: {params}")
    return energy


def main():
    mol_co = MoleculeBuilder().from_name('CO')

    # Define atom positions
    p_C = np.array([0.000000, 0.000000, -1.290265])
    p_O = np.array([0.000000, 0.000000, 0.967698])

    # Initial coefficients and exponents for the STO-3G basis set
    CO_c = np.array([
        [0.154329, 0.535328, 0.444635],  # C 1s
        [-0.099967, 0.399513, 0.700115],  # C 2s
        [0.155916, 0.607684, 0.391957],  # C 2p
        [0.154329, 0.535328, 0.444635],  # O 1s
        [-0.099967, 0.399513, 0.700115],  # O 2s
        [0.155916, 0.607684, 0.391957]  # O 2p
    ])

    CO_a = np.array([
        [71.616837, 13.045096, 3.530512],  # C 1s
        [2.941249, 0.683483, 0.22229],  # C 2s
        [2.941249, 0.683483, 0.22229],  # C 2p
        [130.70932, 23.808861, 6.443608],  # O 1s
        [5.033151, 1.169596, 0.380389],  # O 2s
        [5.033151, 1.169596, 0.380389]  # O 2p
    ])

    # CO_c = np.array([
    #     [0.154329, 0.535328, 0.444635],  # C 1s
    #     [-2.24375400e-02, 8.22376020e-03, 8.26486356e-01],  # C 2s
    #     [7.48515821e-02, 3.54129066e-01, -3.98189056e-02],  # C 2p
    #     [0.154329, 0.535328, 0.444635],  # O 1s
    #     [4.18255137e-02, -2.46163625e-02, 6.14996917e-01],  # O 2s
    #     [8.32616919e-01, 1.43363562e-01, 1.26846714e-01]  # O 2p
    # ])
    #
    # CO_a = np.array([
    #     [71.616837, 13.045096, 3.530512],  # C 1s
    #     [3.07949202e+01, 2.46902785e+01, 3.92726311e-01],  # C 2s
    #     [2.94027376e+00, 1.33687645e+00, 9.31394443e+00],  # C 2p
    #     [130.70932, 23.808861, 6.443608],  # O 1s
    #     [8.07313768e+01, 7.72316541e+01, 4.09519207e-01],  # O 2s
    #     [2.17983802e+00, 8.90412742e+00, 8.58923585e+00]  # O 2p
    # ])

    cgfs_opt = createCGFs(p_C, p_O, CO_c, CO_a)
    result_hf_opt = HF().rhf(mol_co, cgfs_opt)
    print(f"STO-3G Hartree-Fock energy: {result_hf_opt['energy']} Hartrees")

    # Define bounds
    bounds_c = [(-0.1, 1.0)] * 3  # Coefficients bounds
    bounds_a = [(0.1, 135)] * 3  # Exponents bounds
    # bounds_all = bounds_c + bounds_a + bounds_c + bounds_a  # For both C 2p and O 2p

    bounds_all = bounds_c + bounds_a + bounds_c + bounds_a + bounds_c + bounds_a + bounds_c + bounds_a  # For both C2p, C2s, O2p, O2s

    # Flatten the x0 array and include it as the first entry in the population
    # x0_flattened = np.array([
    #     [0.155916, 0.607684, 0.391957],
    #     [2.941249, 0.683483, 0.22229],
    #     [0.154329, 0.535328, 0.444635],
    #     [130.70932, 23.808861, 6.443608],
    #     [-0.099967, 0.399513, 0.700115],
    #     [5.033151, 1.169596, 0.380389],
    #     [0.155916, 0.607684, 0.391957],
    #     [5.033151, 1.169596, 0.380389],
    # ]).flatten(order='C')

    # Parent matrix of next matrix
    # x0_flattened = np.array([[3.48458711e-02,  1.37978990e-02,  7.18862421e-01],
    #                          [2.92459811e+01,  2.81832737e+01,  3.53631668e-01],
    #                          [-7.89412922e-02,  1.06596131e-01, -3.01338495e-02],
    #                          [1.00245092e-01,  1.16056289e-01,  1.53522916e-01],
    #                          [2.84428934e-02, -1.13723363e-02,  8.08955524e-01],
    #                          [7.70999466e+01,  7.76177549e+01,  4.92062590e-01],
    #                          [6.68965596e-01,  1.34278961e-01,  2.51793294e-01],
    #                          [2.70276272e+00,  1.20564633e+01,  8.57718510e-01]]).flatten(order='C')

    # Minimum energy: -115.81482284617323 Hartrees
    x0_flattened = np.array([[-2.24375400e-02, 8.22376020e-03, 8.26486356e-01],
                             [3.07949202e+01, 2.46902785e+01, 3.92726311e-01],
                             [7.48515821e-02, 3.54129066e-01, -3.98189056e-02],
                             [2.94027376e+00, 1.33687645e+00, 9.31394443e+00],
                             [4.18255137e-02, -2.46163625e-02, 6.14996917e-01],
                             [8.07313768e+01, 7.72316541e+01, 4.09519207e-01],
                             [8.32616919e-01, 1.43363562e-01, 1.26846714e-01],
                             [2.17983802e+00, 8.90412742e+00, 8.58923585e+00]]).flatten(order='C')




    # Define the number of other population members
    pop_size = 15
    dim = len(x0_flattened)  # Dimension of the problem

    # Generate random solutions within bounds for initial population
    random_population = np.random.uniform(
        low=[b[0] for b in bounds_all],
        high=[b[1] for b in bounds_all],
        size=(pop_size - 1, dim)
    )

    # Combine x0 and randomly initialized solutions
    initial_population = np.vstack([x0_flattened, random_population])

    # Perform optimization using differential evolution with custom initialization
    result = differential_evolution(
        objective_function,
        bounds=bounds_all,
        args=(p_C, p_O, CO_c, CO_a, mol_co),
        strategy='randtobest1bin',
        maxiter=1000,
        popsize=pop_size,
        tol=1e-6,
        mutation=(0.2, 0.6),
        recombination=0.7,
        updating='deferred',
        workers=-1,
        disp=True,
        init=initial_population, # Pass the custom population
        rng=np.random.default_rng(seed=1234)
    )

    # # Perform optimization using differential evolution
    # # These settings take 3 hours :) (please help)
    # result = differential_evolution(
    #     objective_function,
    #     bounds=bounds_all,
    #     args=(p_C, p_O, CO_c, CO_a, mol_co), # Pass the necessary arguments
    #     strategy='currenttobest1bin',
    #     maxiter=1000, # set iteration per population size
    #     popsize=15, # set population size
    #     tol=1e-6,
    #     mutation=(0.5, 1.0),
    #     recombination=0.7,
    #     updating='deferred',
    #     workers = -1, # allocate all cores
    #     disp=True, # Display progress
    #     x0 = np.array([
    #               [0.155916, 0.607684, 0.391957],
    #               [2.941249, 0.683483, 0.22229],
    #               [0.154329, 0.535328, 0.444635],
    #               [130.70932, 23.808861, 6.443608],
    #               [-0.099967, 0.399513, 0.700115],
    #               [5.033151, 1.169596, 0.380389],
    #               [0.155916, 0.607684, 0.391957],
    #               [5.033151, 1.169596, 0.380389]
    #               ]).flatten(order='C'),
    #     init='custom',
    #     callback=optimization_callback
    # )

    print(f"Optimized coefficients and exponents:\n {result.x}")
    print(f"Minimum energy: {result.fun} Hartrees")

    # # Final Hartree-Fock calculation with optimized parameters
    # CO_c[2], CO_a[2] = result.x[:3], result.x[3:6]
    # CO_c[5], CO_a[5] = result.x[6:9], result.x[9:12]
    # cgfs_opt = createCGFs(p_C, p_O, CO_c, CO_a)
    # result_hf_opt = HF().rhf(mol_co, cgfs_opt)
    # print(f"Final optimized Hartree-Fock energy: {result_hf_opt['energy']} Hartrees")


if __name__ == '__main__':
    main()

import numpy as np
import matplotlib.pyplot as plt
from pyqint import HF, MoleculeBuilder, cgf
from scipy.optimize import differential_evolution


def createCGFs(p_C, p_H1, p_H2, p_H3, p_H4, CH4_c, CH4_a):
    cgfs = []

    # Carbon atom
    c_1s = cgf(p_C)
    for coeff, alpha in zip(CH4_c[0], CH4_a[0]):
        c_1s.add_gto(coeff, alpha, 0, 0, 0)
    cgfs.append(c_1s)

    c_2s = cgf(p_C)
    for coeff, alpha in zip(CH4_c[1], CH4_a[1]):
        c_2s.add_gto(coeff, alpha, 0, 0, 0)
    cgfs.append(c_2s)

    for orientation in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:  # px, py, pz
        c_2p = cgf(p_C)
        for coeff, alpha in zip(CH4_c[2], CH4_a[2]):
            c_2p.add_gto(coeff, alpha, *orientation)
        cgfs.append(c_2p)

    # Hydrogen atom
    h1_1s = cgf(p_H1)
    for coeff, alpha in zip(CH4_c[3], CH4_a[3]):
        h1_1s.add_gto(coeff, alpha, 0, 0, 0)
    cgfs.append(h1_1s)

    h2_1s = cgf(p_H2)
    for coeff, alpha in zip(CH4_c[4], CH4_a[4]):
        h2_1s.add_gto(coeff, alpha, 0, 0, 0)
    cgfs.append(h2_1s)

    h3_1s = cgf(p_H3)
    for coeff, alpha in zip(CH4_c[5], CH4_a[5]):
        h3_1s.add_gto(coeff, alpha, 0, 0, 0)
    cgfs.append(h3_1s)

    h4_1s = cgf(p_H4)
    for coeff, alpha in zip(CH4_c[6], CH4_a[6]):
        h4_1s.add_gto(coeff, alpha, 0, 0, 0)
    cgfs.append(h4_1s)

    return cgfs


# Objective function for optimization
def objective_function(params, p_C, p_H1, p_H2, p_H3, p_H4, CH4_c, CH4_a, mol_ch4):
    # # Optimizing C2p, O2p
    # CO_c[2] = params[:3]
    # CO_a[2] = params[3:6]
    # CO_c[5] = params[6:9]
    # CO_a[5] = params[9:12]

    # Optimizing C2s, C2p, H1s x4
    CH4_c[1] = params[:3]
    CH4_a[1] = params[3:6]
    CH4_c[2] = params[6:9]
    CH4_a[2] = params[9:12]
    CH4_c[3] = params[12:15]
    CH4_a[3] = params[15:18]
    CH4_c[4] = params[18:21]
    CH4_a[4] = params[21:24]
    CH4_c[5] = params[24:27]
    CH4_a[5] = params[27:30]
    CH4_c[6] = params[30:33]
    CH4_a[6] = params[33:36]

    cgfs = createCGFs(p_C, p_H1, p_H2, p_H3, p_H4, CH4_c, CH4_a)
    result_hf = HF().rhf(mol_ch4, cgfs)
    energy = result_hf['energy']
    # print(f"Energy: {energy}, Parameters: {params}")
    return energy


def main():
    mol_ch4 = MoleculeBuilder().from_name('CH4')

    # Define atom positions
    p_C = np.array([0.000000, 0.000000, 0.000000])
    p_H1 = np.array([1.195756, 1.195756, 1.195756])
    p_H2 = np.array([-1.195756,-1.195756,1.195756])
    p_H3 = np.array([-1.195756,1.195756,-1.195756])
    p_H4 = np.array([1.195756,-1.195756,-1.195756])

    # Initial coefficients and exponents for the STO-3G basis set
    # CH4_c = np.array([
    #     [0.154329, 0.535328, 0.444635],  # C 1s
    #     [-0.099967, 0.399513, 0.700115],  # C 2s
    #     [0.155916, 0.607684, 0.391957],  # C 2p
    #     [0.154329, 0.535328, 0.444635],  # H 1s
    #     [0.154329, 0.535328, 0.444635],  # H 1s
    #     [0.154329, 0.535328, 0.444635],  # H 1s
    #     [0.154329, 0.535328, 0.444635],  # H 1s
    # ])
    #
    # CH4_a = np.array([
    #     [71.616837, 13.045096, 3.530512],  # C 1s
    #     [2.941249, 0.683483, 0.22229],  # C 2s
    #     [2.941249, 0.683483, 0.22229],  # C 2p
    #     [0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00],  # alpha H 1s
    #     [0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00],  # H 1s
    #     [0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00],  # H 1s
    #     [0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00],  # H 1s
    # ])

    # Matrices of STO-fit-3G
    CH4_c = np.array([
                        [0.154329, 0.535328, 0.444635],
                        [-0.09988706,  0.25192232,  0.62775783],
                        [0.15069385,  0.5283842,   0.44201152],
                        [0.14921661,  0.62512329,  0.46270869],
                        [0.14837439,  0.53940812,  0.50453552],
                        [3.63859021,  0.66118605,  0.15342265],
                        [3.99035767,  0.71129007,  0.16742802]
                    ])

    CH4_a = np.array([
                        [71.616837, 13.045096, 3.530512],
                        [4.74384002,  0.53875824,  0.33611506],
                        [3.35959523,  0.68907755,  0.18757393],
                        [4.29763716,  0.68158787,  0.14609496],
                        [3.840744,    0.63646773,  0.1418655],
                        [3.63859021, 0.66118605,  0.15342265],
                        [3.99035767,  0.71129007,  0.16742802]
                    ])


    cgfs_opt = createCGFs(p_C, p_H1, p_H2, p_H3, p_H4, CH4_c, CH4_a)
    result_hf_opt = HF().rhf(mol_ch4, cgfs_opt)
    print(f"STO-3G Hartree-Fock energy: {result_hf_opt['energy']} Hartrees")

    # Define bounds
    bounds_c = [(-0.1, 1.0)] * 3  # Coefficients bounds
    bounds_a = [(0.1, 135)] * 3  # Exponents bounds

    bounds_all = bounds_c + bounds_a + bounds_c + bounds_a + bounds_c + bounds_a + bounds_c + bounds_a + bounds_c + bounds_a + bounds_c + bounds_a# For both C2p, C2s, H1s x4

    # Flatten the x0 array and include it as the first entry in the population
    x0_flattened = np.array([
        [-0.099967, 0.399513, 0.700115], # coeff C 2S
        [2.941249, 0.683483, 0.22229], # alpha C 2S
        [0.155916, 0.607684, 0.391957], # coeff C 2S
        [2.941249, 0.683483, 0.22229],  # alpha C 2P
        [0.154329, 0.535328, 0.444635],  # coeff H 1s
        [0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00],  # alpha H 1s
        [0.154329, 0.535328, 0.444635],  # coeff H 1s
        [0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00],  # alpha H 1s
        [0.154329, 0.535328, 0.444635],  # coeff H 1s
        [0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00],  # alpha H 1s
        [0.154329, 0.535328, 0.444635],  # coeff H 1s
        [0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00],  # alpha H 1s
    ]).flatten(order='C')


    # Define the number of other population members
    pop_size = 24
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
        args=(p_C, p_H1, p_H2, p_H3, p_H4, CH4_c, CH4_a, mol_ch4),
        strategy='rand2bin',
        maxiter=1000,
        popsize=pop_size,
        tol=1e-6,
        mutation=(0.1, 1.2),
        recombination=0.7,
        updating='deferred',
        workers=-1,
        disp=True,
        init=initial_population, # Pass the custom population
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

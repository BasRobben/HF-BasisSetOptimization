import numpy as np
from pyqint import HF, MoleculeBuilder, cgf
from scipy.optimize import minimize

def main():
    mol_co = MoleculeBuilder().from_name('CO')

    # Define atom positions 
    p_C = [0.000000,0.000000,-1.290265]
    p_O = [0.000000,0.000000,0.967698]

    # Initial coefficients and exponents for the STO-3G basis set
    CO_c = [
        [0.154329, 0.535328, 0.444635],     # C 1s
        [-0.099967, 0.399513, 0.700115],    # C 2s
        [0.155916, 0.607684, 0.391957],     # C 2p
        [0.154329, 0.535328, 0.444635],     # O 1s
        [-0.099967, 0.399513, 0.700115],    # O 2s
        [0.155916, 0.607684, 0.391957]     # O 2p
    ]
    
    CO_a = [
        [71.616837, 13.045096, 3.530512],   # C 1s
        [2.941249, 0.683483, 0.22229],      # C 2s
        [2.941249, 0.683483, 0.22229],      # C 2p
        [130.70932, 23.808861, 6.443608],   # O 1s
        [5.033151, 1.169596, 0.380389],     # O 2s
        [5.033151, 1.169596, 0.380389]     # O 2p
    ]
    
    cgfs_opt = createCGFs(p_C, p_O, CO_c, CO_a)
    result_hf_opt = HF().rhf(mol_co, cgfs_opt)
    print(f"STO-3G Hartree-Fock energy: {result_hf_opt['energy']} Hartrees")

    # Initial guess for the optimization
    initial_guess = np.concatenate([CO_c[1], CO_a[1], CO_c[2], CO_a[2], CO_c[4], CO_a[4], CO_c[5], CO_a[5]])  # Initial guesses for C2p, C2s, O2p, O2s

    bounds_c = [(-1.0, 1.0)] * 3
    bounds_a = [(-100, 100)] * 3

    # Expand this for all required orbitals
    n_orbitals = int(np.size(initial_guess) / 6)
    bounds_all = (
        (bounds_c + bounds_a) * n_orbitals
    )

    # Optimization function
    def objective_function(params):

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
        
        return result_hf['energy']  # Return the energy as the objective to minimize



    result = minimize(objective_function, initial_guess, method='Nelder-Mead', bounds=bounds_all)

    print(f"Optimized parameters: {result.x}")
    print(f"Minimum energy: {result.fun} Hartrees")

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

if __name__ == '__main__':
    main()

import numpy as np
from pyqint import HF, MoleculeBuilder, cgf
from scipy.optimize import minimize

def main():
    mol_ch4 = MoleculeBuilder().from_name('CH4')

    # Define atom positions
    p_C = [0.000000, 0.000000, 0.000000]
    p_H1 = [0.6327670, 0.6327670, 0.6327670]
    p_H2 = [-0.6327670, -0.6327670, 0.6327670]
    p_H3 = [-0.6327670, 0.6327670, -0.6327670]
    p_H4 = [0.6327670, -0.6327670, -0.6327670]

    # Initial coefficients and exponents for the STO-3G basis set
    CH4_c = [
        [0.154329, 0.535328, 0.444635],  # C 1s
        [-0.099967, 0.399513, 0.700115],  # C 2s
        [0.155916, 0.607684, 0.391957],  # C 2p
        [0.154329, 0.535328, 0.444635],  # H 1s
        [0.154329, 0.535328, 0.444635],  # H 1s
        [0.154329, 0.535328, 0.444635],  # H 1s
        [0.154329, 0.535328, 0.444635],  # H 1s
    ]

    CH4_a = [
        [71.616837, 13.045096, 3.530512],   # C 1s
        [2.941249, 0.683483, 0.22229],      # C 2s
        [2.941249, 0.683483, 0.22229],      # C 2p
        [3.425251, 0.623914, 0.168855],     # H1 1s
        [3.425251, 0.623914, 0.168855],     # H2 1s
        [3.425251, 0.623914, 0.168855],     # H3 1s
        [3.425251, 0.623914, 0.168855],     # H4 1s
    ]
    
    cgfs_opt = createCGFs(p_C, p_H1, p_H2, p_H3, p_H4, CH4_c, CH4_a)
    result_hf_opt = HF().rhf(mol_ch4, cgfs_opt)
    print(f"STO-3G Hartree-Fock energy: {result_hf_opt['energy']} Hartrees")

    # Initial guess for the optimization
    initial_guess = np.concatenate([CH4_c[1], CH4_a[1], CH4_c[2], CH4_a[2], CH4_c[3], CH4_a[3], CH4_c[4], CH4_a[4], CH4_c[5], CH4_a[5], CH4_c[6], CH4_a[6]])  # Initial guesses for C2p, C2s, O2p, O2s

    bounds_c = [(-1.0, 1.0)] * 3
    bounds_a = [(-100, 100)] * 3

    # Expand this for all required orbitals
    n_orbitals = int(np.size(initial_guess) / 6)
    bounds_all = (
        (bounds_c + bounds_a) * n_orbitals
    )

    # Optimization function
    def objective_function(params):

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
        
        return result_hf['energy']  # Return the energy as the objective to minimize

    result = minimize(objective_function, initial_guess, method='Nelder-Mead', bounds=bounds_all)
    print(f"Optimized parameters: {result.x}")
    print(f"Minimum energy: {result.fun} Hartrees")

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

    pos = [p_H1, p_H2, p_H3, p_H4]

    for i in range(0, 4):
        h_1s = cgf(pos[i])
        for coeff, alpha in zip(CH4_c[i+3], CH4_a[i+3]):
            h_1s.add_gto(coeff, alpha, 0, 0, 0)
        cgfs.append(h_1s)

    return cgfs

if __name__ == '__main__':
    main()

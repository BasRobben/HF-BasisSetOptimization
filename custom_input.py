import sys
import numpy as np
import matplotlib.pyplot as plt
from pyqint import HF, PyQInt,  MoleculeBuilder, cgf
from pytessel import PyTessel


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

    # Perform Hartree-Fock calculations
    result_hf = HF().rhf(mol_co, cgfs)

    print(f"Total energy CO: {result_hf['energy']} Hartrees")
    print(f"Orbital energies CO: {result_hf['orbe']} Hartrees")
    return result_hf['cgfs'], result_hf['orbc']

if __name__ == '__main__':
    main()
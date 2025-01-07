import sys
import numpy as np
import matplotlib.pyplot as plt
from pyqint import HF, PyQInt,  MoleculeBuilder, cgf
from pytessel import PyTessel


def main():
    # Create CO molecule
    mol_co = MoleculeBuilder().from_name('CO')

    cgfs = []

    # Carbon, STO-3G
    p_C = [0.000000,0.000000,-1.290265]
    c_1s = cgf(p_C)  # Position of Carbon nucleus
    for coeff, alpha in zip([0.154329, 0.535328, 0.444635], [71.616837, 13.045096, 3.530512]):
        c_1s.add_gto(coeff, alpha, 0, 0, 0)
    cgfs.append(c_1s)

    c_2s = cgf(p_C)
    for coeff, alpha in zip([-0.099967, 0.399513, 0.700115], [2.941249, 0.683483, 0.22229]):
        c_2s.add_gto(coeff, alpha, 0, 0, 0)
    cgfs.append(c_2s)

    for orientation in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:  # px, py, pz
        c_2p = cgf(p_C)
        for coeff, alpha in zip([0.155916, 0.607684, 0.391957], [2.941249, 0.683483, 0.22229]):
            c_2p.add_gto(coeff, alpha, *orientation)
        cgfs.append(c_2p)

    # Oxygen, STO-3G
    p_O = [0.000000,0.000000,0.967698]
    o_1s = cgf(p_O)
    for coeff, alpha in zip([0.154329, 0.535328, 0.444635], [130.70932, 23.808861, 6.443608]):
        o_1s.add_gto(coeff, alpha, 0, 0, 0)
    cgfs.append(o_1s)

    o_2s = cgf(p_O)
    for coeff, alpha in zip([-0.099967, 0.399513, 0.700115], [5.033151, 1.169596, 0.380389]):
        o_2s.add_gto(coeff, alpha, 0, 0, 0)
    cgfs.append(o_2s)

    for orientation in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:  # px, py, pz
        o_2p = cgf(p_O)
        for coeff, alpha in zip([0.155916, 0.607684, 0.391957], [5.033151, 1.169596, 0.380389]):
            o_2p.add_gto(coeff, alpha, *orientation)
        cgfs.append(o_2p)

    # Perform Hartree-Fock calculations
    result_hf = HF().rhf(mol_co, cgfs)

    print(f"Total energy CO: {result_hf['energy']} Hartrees")
    print(f"Orbital energies CO: {result_hf['orbe']} Hartrees")
    return result_hf['cgfs'], result_hf['orbc']

if __name__ == '__main__':
    main()
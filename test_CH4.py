import sys
import numpy as np
import matplotlib.pyplot as plt
from pyqint import HF, PyQInt,  MoleculeBuilder, cgf, Molecule
from pytessel import PyTessel

mol = MoleculeBuilder().from_name('CH4')

pos_C = [0.0000000, 0.0000000, 0.0000000]
pos_H1 = [0.6327670, 0.6327670, 0.6327670]
pos_H2 = [-0.6327670, -0.6327670, 0.6327670]
pos_H3 = [-0.6327670, 0.6327670, -0.6327670]
pos_H4 = [0.6327670, -0.6327670, -0.6327670]

coefficients = [
    [0.154329, 0.535328, 0.444635],
    [-0.099967, 0.399513, 0.700115],
    [0.155916, 0.607684, 0.391957],
    [0.154329, 0.535328, 0.444635],
    [0.154329, 0.535328, 0.444635],
    [0.154329, 0.535328, 0.444635],
    [0.154329, 0.535328, 0.444635]
]

alphas = [
    [71.616837, 13.045096, 3.530512],
    [2.941249, 0.683483, 0.22229],
    [2.941249, 0.683483, 0.22229],
    [3.425251, 0.623914, 0.168855],
    [3.425251, 0.623914, 0.168855],
    [3.425251, 0.623914, 0.168855],
    [3.425251, 0.623914, 0.168855]
]

cgfs = []

c_1s = cgf(pos_C)
for coeff, alpha in zip(coefficients[0], alphas[0]):
    c_1s.add_gto(coeff, alpha, 0, 0, 0)
cgfs.append(c_1s)

c_2s = cgf(pos_C)
for coeff, alpha in zip(coefficients[1], alphas[1]):
    c_2s.add_gto(coeff, alpha, 0, 0, 0)
cgfs.append(c_2s)

for orientation in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
    c_2p = cgf(pos_C)
    for coeff, alpha in zip(coefficients[2], alphas[2]):
        c_2p.add_gto(coeff, alpha, *orientation)
    cgfs.append(c_2p)

h1_1s = cgf(pos_H1)
for coeff, alpha in zip(coefficients[3], alphas[3]):
    h1_1s.add_gto(coeff, alpha, 0, 0, 0)
cgfs.append(h1_1s)

h2_1s = cgf(pos_H2)
for coeff, alpha in zip(coefficients[4], alphas[4]):
    h2_1s.add_gto(coeff, alpha, 0, 0, 0)
cgfs.append(h2_1s)

h3_1s = cgf(pos_H3)
for coeff, alpha in zip(coefficients[5], alphas[5]):
    h3_1s.add_gto(coeff, alpha, 0, 0, 0)
cgfs.append(h3_1s)

h4_1s = cgf(pos_H4)
for coeff, alpha in zip(coefficients[6], alphas[6]):
    h4_1s.add_gto(coeff, alpha, 0, 0, 0)
cgfs.append(h4_1s)

result_hf = HF().rhf(mol, cgfs, verbose=False)
print(f"Total energy: {result_hf['energy']} Hartree's")
print(f"Orbital energies: {result_hf['orbe']} Hartree's")

result_hf = HF().rhf(mol, 'STO3G', verbose=False)
print(f"Total energy: {result_hf['energy']} Hartree's")
print(f"Orbital energies: {result_hf['orbe']} Hartree's")
import sys
import numpy as np
import matplotlib.pyplot as plt
from pyqint import HF, PyQInt,  MoleculeBuilder, cgf
from pytessel import PyTessel
import json

def main():
    molecule_name = 'CH4'  # Choose between 'CO' or 'CH4'
    set_name = 'STO-3G'    # Choose basis set

    positions, coefficients, alphas = read_json('basissets.json', molecule_name, set_name)  # Read JSON file for basis set

    cgfs = createCGFs(positions, coefficients, alphas)  # Create contracted Gaussian functions

    mol = MoleculeBuilder().from_name(molecule_name)    # Create molecule object based on choice
    
    # Perform Hartree-Fock calculations
    result_hf = HF().rhf(mol, cgfs)
    cgfs_res, orbc_res, energy_res, orbe_res = result_hf['cgfs'], result_hf['orbc'], result_hf['energy'], result_hf['orbe']

    print(f"Total energy: {energy_res} Hartrees")
    print(f"Orbital energies: {orbe_res} Hartrees")

    if molecule_name == 'CO':
        for i in range(0, 10):  # 10 MOs: 5 AOs each atom
            build_isosurface(f'co_{i}.ply', cgfs_res, orbc_res[:,i], 0.1)

    elif molecule_name == "CH4":
        for i in range(0, 9):   # 9 MOs: 5 AOs for C and 4 AOs for H4
            build_isosurface(f'ch4_{i}.ply', cgfs_res, orbc_res[:,i], 0.1)

    else:
        raise ValueError("Invalid molecule name")


def build_isosurface(filename, cgfs, coeff, isovalue):
    # generate some data
    sz = 100
    integrator = PyQInt()
    grid = integrator.build_rectgrid3d(-5, 5, sz)
    scalarfield = np.reshape(integrator.plot_wavefunction(grid, coeff, cgfs), (sz, sz, sz))
    unitcell = np.diag(np.ones(3) * 10.0)

    pytessel = PyTessel()
    vertices, normals, indices = pytessel.marching_cubes(scalarfield.flatten(),
                                                         scalarfield.shape,
                                                         unitcell.flatten(),
                                                         isovalue)
    pytessel.write_ply(filename + '_pos.ply', vertices, normals, indices)
    vertices, normals, indices = pytessel.marching_cubes(scalarfield.flatten(),
                                                         scalarfield.shape,
                                                         unitcell.flatten(),
                                                         -isovalue)
    pytessel.write_ply(filename + '_neg.ply', vertices, normals, indices)


def read_json(filename, molecule_name, set_name):

    with open(filename, 'r') as file:
        data = json.load(file)

    molecule_data = data['molecules'].get(molecule_name)

    if not molecule_data:
        raise ValueError(f"Molecule {molecule_name} not found in JSON file")

    positions = molecule_data['positions']
    basis_set = molecule_data['basis_sets'].get(set_name)

    if not basis_set:
        raise ValueError(f"Basis set {set_name} not found for molecule {molecule_name}")

    coefficients = basis_set['coefficients']
    alphas = basis_set['alphas']

    if len(coefficients) != len(alphas):
        raise ValueError("Coefficients and alphas do not have the same length")

    return positions, coefficients, alphas

def createCGFs(positions, coefficients, alphas):
    cgfs = []

    atom_names = list(positions.keys())

    c_1s = cgf(positions['C'])
    for coeff, alpha in zip(coefficients[0], alphas[0]):
        c_1s.add_gto(coeff, alpha, 0, 0, 0)
    cgfs.append(c_1s)

    c_2s = cgf(positions['C'])
    for coeff, alpha in zip(coefficients[1], alphas[1]):
        c_2s.add_gto(coeff, alpha, 0, 0, 0)
    cgfs.append(c_2s)

    for orientation in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:  # px, py, pz
        c_2p = cgf(positions['C'])
        for coeff, alpha in zip(coefficients[2], alphas[2]):
            c_2p.add_gto(coeff, alpha, *orientation)
        cgfs.append(c_2p)
    
    if 'H1' in atom_names:
        pos = [positions['H1'], positions['H2'], positions['H3'], positions['H4']]
        for i in range(0, 4):
            h_1s = cgf(pos[i])
            for coeff, alpha in zip(coefficients[i+3], alphas[i+3]):
                h_1s.add_gto(coeff, alpha, 0, 0, 0)
            cgfs.append(h_1s)
    else:
        o_1s = cgf(positions['O'])
        for coeff, alpha in zip(coefficients[3], alphas[3]):
            o_1s.add_gto(coeff, alpha, 0, 0, 0)
        cgfs.append(o_1s)

        o_2s = cgf(positions['O'])
        for coeff, alpha in zip(coefficients[4], alphas[4]):
            o_2s.add_gto(coeff, alpha, 0, 0, 0)
        cgfs.append(o_2s)

        for orientation in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:  # px, py, pz
            o_2p = cgf(positions['O'])
            for coeff, alpha in zip(coefficients[5], alphas[5]):
                o_2p.add_gto(coeff, alpha, *orientation)
            cgfs.append(o_2p)
    return cgfs

if __name__ == '__main__':
    main()
import sys
import numpy as np
import matplotlib.pyplot as plt
from pyqint import HF, PyQInt,  MoleculeBuilder
from pytessel import PyTessel


def main():
    # fetches the contracted gaussian functionals (CGF) and the orbital coefficients (orbc) from
    # calculate_co and calculate_ch4 and builds individual file isosurfaces
    # depending on the value of orbitals.
    num_atomic_orbitals = 9
    cgfs_co, orbc_co = calculate_co()
    cgfs_ch4, orbc_ch4 = calculate_ch4()
    # for number_orbitals in range(0, num_atomic_orbitals):
    #     build_isosurface(f'co_{number_orbitals}.ply', cgfs_co, orbc_co[:,number_orbitals], 0.1)
    #     build_isosurface(f'ch4_{number_orbitals}.ply', cgfs_ch4, orbc_ch4[:,number_orbitals], 0.1)


def calculate_co():
    # calculates hartree fock for CO in STO-3G basis set
    mol_co = MoleculeBuilder().from_name('CO')
    result_hf = HF().rhf(mol_co, 'sto3g')
    print(f"Total energy CO: {result_hf['energy']} Hartree's")
    print(f"Orbital energies CO: {result_hf['orbe']} Hartree's")
    return result_hf['cgfs'], result_hf['orbc']


def calculate_ch4():
    # calculates hartree fock for CH4 in STO-3G basis set
    mol_ch4 = MoleculeBuilder().from_name('CH4')
    result_hf = HF().rhf(mol_ch4, 'sto3g')
    print(f"Total energy CH4: {result_hf['energy']} Hartree's")
    print(f"Orbital energies CH4: {result_hf['orbe']} Hartree's")
    return result_hf['cgfs'], result_hf['orbc']


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


if __name__ == '__main__':
    main()

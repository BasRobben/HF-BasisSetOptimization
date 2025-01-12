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
    # Energy: -111.21925705796176 Hartrees
    # CO_c = [
    #     [0.154329, 0.535328, 0.444635],     # C 1s
    #     [-0.099967, 0.399513, 0.700115],    # C 2s
    #     [0.155916, 0.607684, 0.391957],     # C 2p
    #     [0.154329, 0.535328, 0.444635],     # O 1s
    #     [-0.099967, 0.399513, 0.700115],    # O 2s
    #     [0.155916, 0.607684, 0.391957]     # O 2p
    # ]
    #
    # CO_a = [
    #     [71.616837, 13.045096, 3.530512],   # C 1s
    #     [2.941249, 0.683483, 0.22229],      # C 2s
    #     [2.941249, 0.683483, 0.22229],      # C 2p
    #     [130.70932, 23.808861, 6.443608],   # O 1s
    #     [5.033151, 1.169596, 0.380389],     # O 2s
    #     [5.033151, 1.169596, 0.380389]     # O 2p
    # ]

    ## https://doi.org/10.1063/1.439203
    ## Energy: -111.9055772807634 Hartrees
    CO_c = [
        [7.07666e-2, 3.95356e-1, 6.63169e-1],  # C 1s
        [-9.26218e-2, 3.40674e-1, 7.43772e-1],  # C 2s
        [1.10281e-1, 4.62034e-1, 6.27665e-1],  # C 2p
        [7.03080e-2, 3.95853e-1, 6.61894e-1],  # O 1s
        [-8.75258e-2, 4.07671e-1, 6.76432e-1],  # O 2s
        [1.23326e-1, 4.75101e-1, 6.15173e-1]  # O 2p
    ]

    CO_a = [
        [1.53396e2, 2.30596e1, 4.92985], # C 1s
        [4.39495, 6.58157e-1, 1.98374e-1], # C 2s
        [4.23235, 8.61673e-1, 2.01326e-1], # C 2p
        [2.77555e2, 4.18923e1, 9.04437], # O 1s
        [9.28383, 1.14812, 3.53831e-1], # O 2s
        [8.09648, 1.67764, 3.74101e-1] # O 2p
    ]

    # Optimized for C 2p
    # Energy: -111.25619844054083 Hartrees
    # CO_c = [
    #     [0.154329, 0.535328, 0.444635],     # C 1s
    #     [-0.099967, 0.399513, 0.700115],    # C 2s
    #     [0.07764249, 0.33793903, 0.42185065],     # C 2p
    #     [0.154329, 0.535328, 0.444635],     # O 1s
    #     [-0.099967, 0.399513, 0.700115],    # O 2s
    #     [0.155916, 0.607684, 0.391957]     # O 2p
    # ]
    #
    # CO_a = [
    #     [71.616837, 13.045096, 3.530512],   # C 1s
    #     [2.941249, 0.683483, 0.22229],      # C 2s
    #     [4.63846892, 0.94085213, 0.21516849],      # C 2p
    #     [130.70932, 23.808861, 6.443608],   # O 1s
    #     [5.033151, 1.169596, 0.380389],     # O 2s
    #     [5.033151, 1.169596, 0.380389]     # O 2p
    # ]

    # Optimized for C 2p and O 2p
    # Energy: -111.42954794960906 Hartrees
    # CO_c = [
    #     [0.154329, 0.535328, 0.444635],     # C 1s
    #     [-0.099967, 0.399513, 0.700115],    # C 2s
    #     [0.05998836, 0.26271949, 0.2594085],     # C 2p
    #     [0.154329, 0.535328, 0.444635],     # O 1s
    #     [-0.099967, 0.399513, 0.700115],    # O 2s
    #     [0.09388762, 0.37233551, 0.56289999]     # O 2p
    # ]
    #
    # CO_a = [
    #     [71.616837, 13.045096, 3.530512],   # C 1s
    #     [2.941249, 0.683483, 0.22229],      # C 2s
    #     [4.52149079, 0.88832976, 0.20509071],      # C 2p
    #     [130.70932, 23.808861, 6.443608],   # O 1s
    #     [5.033151, 1.169596, 0.380389],     # O 2s
    #     [8.92575978, 1.88122069, 0.43250111]     # O 2p
    # ]

    # Optimized C2s, C2p, O2s, O2p
    # Energy: -111.45495843493816 Hartrees
    # CO_c = [
    #     [0.154329, 0.535328, 0.444635],     # C 1s
    #     [3.48458711e-02,  1.37978990e-02,  7.18862421e-01],    # C 2s
    #     [-7.89412922e-02,  1.06596131e-01, -3.01338495e-02],     # C 2p
    #     [0.154329, 0.535328, 0.444635],     # O 1s
    #     [2.84428934e-02, -1.13723363e-02,  8.08955524e-01],    # O 2s
    #     [6.68965596e-01,  1.34278961e-01,  2.51793294e-01]     # O 2p
    # ]
    #
    # CO_a = [
    #     [71.616837, 13.045096, 3.530512],   # C 1s
    #     [2.92459811e+01,  2.81832737e+01,  3.53631668e-01],      # C 2s
    #     [1.00245092e-01,  1.16056289e-01,  1.53522916e-01],      # C 2p
    #     [130.70932, 23.808861, 6.443608],   # O 1s
    #     [7.70999466e+01,  7.76177549e+01,  4.92062590e-01],     # O 2s
    #     [2.70276272e+00,  1.20564633e+01,  8.57718510e-01]     # O 2p
    # ]

    # Optimized fitted on STO using differential evolution C1s, C2s, C2p, O1s, O2s, O2p
    # Energy: -110.18976720235227 Hartrees
    # CO_c = [
    #     [7.90501396e-01, 2.99580421e-01, 2.3765731e-02],     # C 1s
    #     [-4.64346283e-02, -4.7765928e-03, 1.],    # C 2s
    #     [0.26729886, 0.92131106, 0.76196688],     # C 2p
    #     [7.87066231e-01, 3.04134081e-01, 2.36093454e-02],     # O 1s
    #     [1., -4.42063585e-02, -8.28748560e-03],    # O 2s
    #     [1., 1., 0.52778558]     # O 2p
    # ]
    #
    # CO_a = [
    #     [6.1719865e+00, 3.87322674e+01, 5.08470932e+02],   # C 1s
    #     [9.61816638e+00, 9.93808259e+01, 3.2908929e-01],      # C 2s
    #     [6.53905139, 1.19533484, 0.32417256],      # C 2p
    #     [1.11716037e+01, 6.99288842e+01, 9.52173569e+02],   # O 1s
    #     [5.65433587e-01, 1.44437535e+01, 9.97058596e+01],     # O 2s
    #     [0.58056914, 1.76628246, 7.22355583]     # O 2p
    # ]

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
    result_hf = HF().rhf(mol_co, cgfs, verbose=True)
    cgfs_co, orbc_co, energy_co, orbe_co = result_hf['cgfs'], result_hf['orbc'], result_hf['energy'], result_hf['orbe']

    print(f"Total energy CO: {energy_co} Hartrees")
    print(f"Orbital energies CO: {orbe_co} Hartrees")
    # for i in range(0, 10):  # 10 MOs since 5 AOs each atom
    #     build_isosurface(f'co_{i}.ply', cgfs_co, orbc_co[:,i], 0.1)
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
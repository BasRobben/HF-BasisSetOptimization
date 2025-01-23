import numpy as np
from pyqint import HF, PyQInt,  MoleculeBuilder, cgf
from pytessel import PyTessel
import json

def main():
    """
    Main function to perform Hartree-Fock calculations and build isosurfaces for molecular orbitals.
    
    Reads basis set parameters from JSON file for specified molecules and basis set.
    Change molecule_name and set_name to run calculations for different molecules and basis sets.
    """

    molecule_name = 'CO'  # Choose between 'CO' or 'CH4'
    set_name = 'STO-3G'   # Choose basis set

    positions, coefficients, alphas = read_json('basissets.json', molecule_name, set_name)  # Read JSON file for specified basis set

    cgfs = createCGFs(positions, coefficients, alphas)  # Create contracted Gaussian functions

    mol = MoleculeBuilder().from_name(molecule_name)    # Create molecule object based on specified molecule
    
    # Perform Hartree-Fock calculations
    result_hf = HF().rhf(mol, cgfs)
    cgfs_res, orbc_res, energy_res, orbe_res = result_hf['cgfs'], result_hf['orbc'], result_hf['energy'], result_hf['orbe']

    print(f"Total energy: {energy_res} Hartrees")
    print(f"Orbital energies: {orbe_res} Hartrees")

    # Build isosurfaces for molecular orbitals
    if molecule_name == 'CO':
        for i in range(0, 10):  # 10 MOs: 5 AOs each atom
            build_isosurface(f'co_{i}', cgfs_res, orbc_res[:,i], 0.1)

    elif molecule_name == 'CH4':
        for i in range(0, 9):   # 9 MOs: 5 AOs for C and 4 AOs for H4
            build_isosurface(f'ch4_{i}', cgfs_res, orbc_res[:,i], 0.1)

    else:
        raise ValueError("Unsupported molecule")


def build_isosurface(filename, cgfs, coeff, isovalue):
    """
    Build an isosurface for a molecular orbital and save it to a file.

    Parameters:
    filename (str): The name of the file to save the isosurface to.
    cgfs (list): List of contracted Gaussian functions.
    coeff (numpy.ndarray): Coefficients for the molecular orbital.
    isovalue (float): Isovalue for the isosurface.

    Returns:
    None
    """

    sz = 100
    integrator = PyQInt()
    grid = integrator.build_rectgrid3d(-5, 5, sz)
    scalarfield = np.reshape(integrator.plot_wavefunction(grid, coeff, cgfs), (sz, sz, sz)) # Compute scalar field
    unitcell = np.diag(np.ones(3) * 10.0)

    # Apply marching cubes algorithm to extract isosurface
    # Write to PLY file
    pytessel = PyTessel()
    vertices, normals, indices = pytessel.marching_cubes(scalarfield.flatten(),
                                                         scalarfield.shape,
                                                         unitcell.flatten(),
                                                         isovalue)
    pytessel.write_ply(filename + '_pos.ply', vertices, normals, indices)

    # Repeat for negative isovalue
    vertices, normals, indices = pytessel.marching_cubes(scalarfield.flatten(),
                                                         scalarfield.shape,
                                                         unitcell.flatten(),
                                                         -isovalue)
    pytessel.write_ply(filename + '_neg.ply', vertices, normals, indices)


def read_json(filename, molecule_name, set_name):
    """
    Read basis set parameters from a JSON file for a specified molecule and basis set.

    Parameters:
    filename (str): The name of the JSON file.
    molecule_name (str): The name of the molecule.
    set_name (str): The name of the basis set.

    Returns:
    tuple: A tuple containing positions (array), coefficients (array), and alphas (array).

    Raises:
    ValueError: If the molecule or basis set is not found in the JSON file, or if coefficients and alphas do not have the same length.
    """

    # Open JSON file and load data
    with open(filename, 'r') as file:
        data = json.load(file)

    molecule_data = data['molecules'].get(molecule_name)    # Get data for specified molecule

    # Check if molecule is in JSON file
    if not molecule_data:
        raise ValueError(f"Molecule {molecule_name} not found in JSON file")

    # Get positions and basis set data
    positions = molecule_data['positions']
    basis_set = molecule_data['basis_sets'].get(set_name)

    # Check if basis set is in JSON file
    if not basis_set:
        raise ValueError(f"Basis set {set_name} not found for molecule {molecule_name}")

    # Get coefficients and alphas from basis set data
    coefficients = basis_set['coefficients']
    alphas = basis_set['alphas']

    # Check if coefficients and alphas have the same length
    if len(coefficients) != len(alphas):
        raise ValueError("Coefficients and alphas do not have the same length")

    return positions, coefficients, alphas

def createCGFs(positions, coefficients, alphas):
    """
    Create contracted Gaussian functions (CGFs) for a molecule.

    Parameters:
    positions (array): Array of atomic positions.
    coefficients (array): Array of coefficients for the Gaussian functions.
    alphas (array): Array of exponents for the Gaussian functions.

    Returns:
    cgfs (array): Array of cgf objects.
    """

    cgfs = []

    atom_names = list(positions.keys())

    # Carbon atom
    c_1s = cgf(positions['C'])
    for coeff, alpha in zip(coefficients[0], alphas[0]):
        c_1s.add_gto(coeff, alpha, 0, 0, 0)
    cgfs.append(c_1s)

    c_2s = cgf(positions['C'])
    for coeff, alpha in zip(coefficients[1], alphas[1]):
        c_2s.add_gto(coeff, alpha, 0, 0, 0)
    cgfs.append(c_2s)

    for orientation in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:   # px, py, pz
        c_2p = cgf(positions['C'])
        for coeff, alpha in zip(coefficients[2], alphas[2]):
            c_2p.add_gto(coeff, alpha, *orientation)
        cgfs.append(c_2p)
    
    # Check for hydrogen atom in molecule
    # If present, add 1s orbitals for each hydrogen atom (CH4 case)
    # Otherwise, add orbitals for oxygen atom (CO case)
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
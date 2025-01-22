# Hartree-Fock Basis Set Optimization

This project performs Hartree-Fock calculations and builds isosurfaces for molecular orbitals using specified basis sets.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- PyQInt
- PyTessel

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/HF-BasisSetOptimization.git
    cd HF-BasisSetOptimization
    ```

2. Install the required packages:
    ```sh
    pip install numpy matplotlib pyqint pytessel
    ```

## Usage

1. Edit the [main.py](http://_vscodecontentref_/0) file to specify the molecule and basis set:
    ```python
    molecule_name = 'CO'  # Choose between 'CO' or 'CH4'
    set_name = 'STO-3G'   # Choose basis set
    ```

2. Run the main script:
    ```sh
    python main.py
    ```

## Functions

### [main()](http://_vscodecontentref_/1)

Main function to perform Hartree-Fock calculations and build isosurfaces for molecular orbitals.

### `build_isosurface(filename, cgfs, coeff, isovalue)`

Builds an isosurface for a molecular orbital and saves it to a file.

- **Parameters:**
  - `filename` (str): The name of the file to save the isosurface to.
  - [cgfs](http://_vscodecontentref_/2) (list): List of contracted Gaussian functions.
  - `coeff` (numpy.ndarray): Coefficients for the molecular orbital.
  - `isovalue` (float): Isovalue for the isosurface.

### [read_json(filename, molecule_name, set_name)](http://_vscodecontentref_/3)

Reads basis set parameters from a JSON file for a specified molecule and basis set.

- **Parameters:**
  - `filename` (str): The name of the JSON file.
  - [molecule_name](http://_vscodecontentref_/4) (str): The name of the molecule.
  - [set_name](http://_vscodecontentref_/5) (str): The name of the basis set.

- **Returns:**
  - `tuple`: A tuple containing positions, coefficients, and alphas.

### [createCGFs(positions, coefficients, alphas)](http://_vscodecontentref_/6)

Creates contracted Gaussian functions (CGFs) for a molecule.

- **Parameters:**
  - [positions](http://_vscodecontentref_/7) (dict): Dictionary of atomic positions.
  - [coefficients](http://_vscodecontentref_/8) (list): List of coefficients for the Gaussian functions.
  - [alphas](http://_vscodecontentref_/9) (list): List of exponents for the Gaussian functions.

- **Returns:**
  - `list`: A list of CGFs for the molecule.

## License

This project is licensed under the MIT License.
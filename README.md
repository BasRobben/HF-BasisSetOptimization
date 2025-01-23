# Hartree-Fock Basis Set Optimization

This project employs a Restricted Hartree-Fock (RHF) procedure to optimize basis sets for the CO and CH₄ molecules. By implementing three distinct optimization methods — direct fitting, Nelder-Mead (NM) minimization, and Differential Evolution (DE) — the primary goal is to achieve lower total electronic energies compared to the widely-used STO-3g basis set.  The latter two methods were particularly effective in achieving this.

In addition to analyzing the total electronic energy, the project explores the individual molecular orbital (MO) energies and their geometries relative to those obtained with the reference STO-3g basis set.

---

## Features

- **Optimization Techniques**: Three separate scripts implement direct fitting, Nelder-Mead minimization, and Differential Evolution to optimize basis set parameters.
- **Basis Set Storage**: Optimized basis sets are saved in the `basissets.json` file, categorized by molecule, including atom positions (in Angstroms).
- **RHF Calculations**: The `main.py` script performs RHF calculations using the optimized basis sets, outputs total electronic energy and orbital energies (in Hatrees), and generates PLY files for MO visualization.
- **Molecular Orbital Visualization**: PLY files of MO isosurfaces (positive and negative phases) are generated for easy import into 3D rendering software like Blender.

---

## Requirements

The following Python packages are required:

- **For RHF Calculations (main script)**:
  - `numpy`
  - `pyqint`
  - `pytessel`
  - `json`

- **For Optimization Scripts**:
  - `scipy` (in addition to the above)

---

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/BasRobben/HF-BasisSetOptimization.git
    cd HF-BasisSetOptimization
    ```

2. Install the required Python packages:
    ```sh
    pip install numpy pyqint pytessel json scipy
    ```

---
## Usage
### Running RHF Calculations

1. Open `main.py` and specify the desired molecule and basis set:
    ```python
    molecule_name = 'CO'  # Choose between 'CO' or 'CH4'
    set_name = 'STO-3G'   # Choose the desired basis set
    ```

2. Run the main script to perform an RHF calculation:
    ```sh
    python main.py
    ```

3. Outputs:
    - Total electronic energy and individual MO energies are printed in the console in Hartrees.
    - PLY files containing MO isosurfaces are generated in the working directory. Use the naming convention to identify them: `<molecule>_<MO_index>_<phase>.ply`.

### Running Optimization Scripts
To optimize basis set parameters, run the respective script for the desired molecule. For example, to perform Nelder-Mead optimization for CO:
    ```sh
    python nelder-mead-CO.py
    ```

---

## Examples
### Example Output (RHF Calculation)
For `molecule_name = 'CO'` and `set_name = STO-3G`:
  - Total energy: -111.21925705796176 Hartrees
  - Orbital energies: [-20.39137481 -11.09020628  -1.40474568  -0.68994908  -0.50938169
  -0.50938169  -0.44087481   0.28652536   0.28652536   0.92528128] Hartrees
  - Generated PLY Files: `co_0_neg.ply`, `co_0_pos.ply`, `co_1_neg.ply`, etc.

### Visualizing MOs:
Import the generated PLY files into Blender or any other 3D rendering software to view the MO isosurfaces. 
Each MO consists of two isosurfaces: positive (`pos`) and negative (`neg`)

---

## License
This project is licensed under the MIT License.

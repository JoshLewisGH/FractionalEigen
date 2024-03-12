# Fractional Schrödinger Eigen-Solver

## Introduction

This project provides a numerical solver for the eigenfunctions of the fractional Schrödinger equation within a specified potential. It aims to solve for the \(N\) lowest energy states while offering functionality to filter out certain types of degeneracies related to parity. The solver utilizes imaginary time evolution coupled with a 6th order Suzuki-Trotter decomposition to achieve high-precision results.

## Features

- **Eigenfunction Solver:** Calculates the \(N\) lowest energy states for the fractional Schrödinger equation.
- **Degeneracy Filtering:** Ability to identify and filter out specific types of degeneracies associated with parity.
- **High Order Method:** Employs a 6th order Suzuki-Trotter decomposition for imaginary time evolution, ensuring accurate and efficient computation.
- **Potential Flexibility:** Users can define and apply different potentials to study various physical systems.

## Installation

To install and run this solver, you will need Python 3.x installed on your system. Follow the steps below to set up the project:

1. Clone the repository to your local machine:

```bash
git clone https://github.com/JoshLewisGH/FractionalEigen.git
cd FractionalEigen
```

2. Ensure that `scriptHead.py` and `imaginaryEigen.py` are placed in the same directory. These files work together to solve for the eigenfunctions of the fractional Schrödinger equation, with `scriptHead.py` serving as the entry point for users to set parameters and initiate the computation.

3. When you run `scriptHead.py`, the script will automatically create a directory structure within the same directory to store the produced eigenfunctions, plots, and other miscellaneous data. This organization helps in managing the outputs and ensuring that all generated files are easily accessible.

To run the solver, navigate to the project directory and execute:

```bash
python scriptHead.py
```

Follow any additional setup or usage instructions provided in the [Usage](#usage) section to configure and start your computations.

## Usage

This solver is designed for ease of use, requiring minimal setup to start solving for the eigenfunctions of the fractional Schrödinger equation. The project comprises two primary Python files:

- `imaginaryEigen.py` - Contains the core functionality and algorithms for solving the equation. This file does all the heavy lifting but does not require direct modification.
- `scriptHead.py` - Designed for users to easily adjust parameters according to their specific requirements. This is the only file you need to modify to set up your problem.

**Note:** The default setup of `scriptHead.py` is configured to solve for the eigenfunctions of the quantum harmonic oscillator for the integer order Schrödinger equation. This serves as a basic example to demonstrate how the solver works and to provide a starting point for users to adjust the script for their specific needs.

To use the solver, follow these steps:

1. Open `scriptHead.py` in your preferred text editor or IDE.
2. Within `scriptHead.py`, you will find various parameters and settings that you can adjust to define your problem space, including the potential, number of energy states to solve for, and any degeneracy filtering criteria.
3. After setting your parameters in `scriptHead.py`, run this file. It will internally call functions from `imaginaryEigen.py` to perform the computation.

```bash
python scriptHead.py
```

## Requirements

This project is developed with Python 3.x. Ensure you have Python 3.x installed on your system before running this solver. Below is a list of required Python packages that are necessary to run the project successfully:

- NumPy: For efficient numerical computations.
- Matplotlib: For generating plots and animations of the eigenfunctions.
- SciPy: Provides algorithms for optimization, integration, interpolation, eigenvalue problems, algebraic equations, and other purposes. Specifically, `scipy.fft` is used for fast Fourier transforms.
- Ast: For parsing typed Python expressions.
- Warnings: Used to warn users of conditions in the program, where exception handling is not explicitly needed.

Additionally, the project uses standard Python libraries `time`, `os`, and `shutil` for managing files and directories, and `genericpath` for path operations.

To install the required packages, you can use the following command:

```bash
pip install numpy matplotlib scipy
```

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0) - see the [LICENSE](LICENSE) file for details.

The GNU GPL is a free, copyleft license for software and other kinds of works, offering the recipients the freedom to use, study, share (copy), and modify the software. Software that is licensed under GPL-3.0 is required to be made available to users with the source code, ensuring that all modifications and derived works are also covered by the same license.

For more information on the GNU GPL v3.0, please visit [GNU's General Public License](https://www.gnu.org/licenses/gpl-3.0.html).

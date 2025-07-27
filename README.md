
# NBGA Optimization

This project implements a **Neighborhood-Based Genetic Algorithm (NBGA)** for ligand optimization, inspired by the approach described in [An Evolutionary Approach to Drug-Design Using a Novel Neighbourhood Based Genetic Algorithm](https://arxiv.org/abs/1205.6412). The algorithm evolves ligand structures to minimize their interaction energy with a protein's active site, using a tree-based representation for ligands and a dynamic neighborhood topology for genetic operations.

## Features

- **NBGA implementation** for ligand-protein binding optimization.
- **TSP Algorithm Comparison**: Compare NBGA, SWAP_GATSP, OX_SIM, and MOC_SIM on classic TSPLIB datasets with interactive plots and tables.
- Ligand structure represented as a variable-length tree of functional groups.
- Interaction energy calculated using a modified Lennard-Jones potential.
- Streamlit integration for interactive parameter tuning and visualization.
- Reproducible results with fixed random seed.

## Algorithm Overview

- **Ligand Representation:** Each ligand is a chromosome (list) where each gene represents a functional group or an empty position.
- **Fitness Evaluation:** The interaction energy between the ligand and protein residues is minimized. Empty or mostly empty ligands are penalized.
- **Genetic Operators:** Single-point crossover, uniform random mutation, and trio tournament selection.
- **Neighborhood Topology:** Individuals are arranged in a ring, and neighborhoods are randomized each generation to maintain diversity.
- **Visualization:** The evolution of the best fitness (energy) is plotted and can be smoothed for trend analysis.

For more details, see the [original paper](https://arxiv.org/abs/1205.6412).

---

## Getting Started

> **Python version:**  
> This project uses Python **3.10.0** as specified in the `.python-version` file.  
> It is recommended to use [pyenv](https://github.com/pyenv/pyenv) or ensure your environment matches this version.

### 1. Clone the Repository

```bash
git clone https://github.com/AnishSarkar22/nbga-optimization.git
cd NBGA-TSP
```

### 2. Create and Activate a Virtual Environment

It is recommended to use a virtual environment to manage dependencies:

```bash
# Create a virtual environment named .venv
python3 -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run script.py
```

This will launch a web interface where you can adjust algorithm parameters and visualize the optimization process.

---

## Usage

- **NBGA Ligand Optimization**: Adjust parameters and run the genetic algorithm to optimize ligand binding.
- **TSP Algorithm Comparison**: Select the TSP Comparison from the sidebar, choose your dataset directory, and run the comparison to visualize and compare algorithm performance.
- View the best ligand found and its interaction energy.
- Analyze the energy evolution plot for convergence trends.

---

## File Structure

- `Home.py` — Main NBGA implementation and Streamlit interface.
- `pages/1_Ligand_Optimization.py` — Script for Ligand Optimization.
- `pages/2_TSP_Comparison.py` — Script for TSP algorithm Comparison.
- `requirements.txt` — Python dependencies.
- `algorithms/ligand.py` — NBGA algorithm and utilities for ligand optimization.
- `algorithms/tsp.py` — NBGA and other algorithms for TSP (original version).
- `algorithms/tsp_enhanced.py` — Enhanced NBGA and TSP algorithms.
- `algorithms/README.md` — [Comparison and documentation of algorithms](algorithms/README.md).
- `evaluation_dataset/` — Reference optimal TSP datasets and extraction scripts.
- `tsp_dataset/` — TSP datasets and extraction scripts.
- Other scripts and data files as needed.

---

## Reference

If you use this code or approach in your research, please cite:

> Arnab Ghosh, Avishek Ghosh, Arkabandhu Chowdhury, Amit Konar.  
> "An Evolutionary Approach to Drug-Design Using a Novel Neighbourhood Based Genetic Algorithm."  
> [arXiv:1205.6412](https://arxiv.org/abs/1205.6412)

---

## License

This project is licensed under the [MIT License](./LICENSE). See the original paper for further details.
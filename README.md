# Nanogel Simulation Project

This repository contains Python scripts for simulating nanogel structures using different dynamics and techniques. The simulations leverage the ESPResSo and pyiron frameworks to create, manipulate, and analyze nanogel structures.

## Directory Structure

- **DPD Lan Therm**
  - `DPD_Lan_Therm.py`: Builds polymer chains within a nanogel simulation using the ESPResSo package. Ensures random distribution of charges while avoiding consecutive charged beads.

- **DPD**
  - `DPD.py`: Creates a nanogel structure from previously simulated data using Dissipative Particle Dynamics (DPD) with ESPResSo. Reads particle and bond data from files and constructs the nanogel.

- **Langevin**
  - `Langevin.py`: Similar to `DPD_Lan_Therm.py`, but uses Langevin dynamics within the ESPResSo framework to construct polymer chains with specific bonded interactions.

- **nanogel**
  - `Nanogel.py`: Uses the pyiron framework to create and manipulate a nanogel structure. Constructs a diamond crystal structure of carbon atoms, adjusts atomic positions, and saves them for further simulation.

## Getting Started

### Prerequisites

Ensure you have the following libraries installed:
- `espressomd`
- `numpy`
- `matplotlib`
- `scipy`
- `pyiron`

### Installing

Clone the repository:
```bash
git clone https://github.com/yourusername/nanogel-simulation.git
cd nanogel-simulation



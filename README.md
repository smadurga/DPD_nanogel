# Nanogel Simulation Project

This repository contains Python scripts for simulating nanogel structures using different dynamics and techniques. The simulations leverage the ESPResSo and pyiron frameworks to create, manipulate, and analyze nanogel structures.

## Authors: Daniel Valero, Francesc Mas* and Sergio Madurga* 

Department of Material Science and Physical Chemistry & Institute of Theoretical and Computational Chemistry (IQTC), University of Barcelona (UB), Barcelona, Catalonia, Spain 

* Correspondence: fmas@ub.edu; s.madurga@ub.edu 

Keywords: Computer simulations, Dissipative particle dynamics, Thermosensitive nanogel, PNIPAM, Volume phase transition 


## Directory Structure

- **DPD Langevin Thermalisation**
  - `DPD_Lan_Therm.py`: Builds polymer chains within a nanogel simulation using the ESPResSo package. Saves particle and bond data off the constructed nanogel after thermalization

- **DPD**
  - `DPD.py`: Creates a nanogel structure from previously simulated data using Dissipative Particle Dynamics (DPD) with ESPResSo. Reads particle and bond data from files and constructs the nanogel.

- **Langevin**
  - `Langevin.py`: Similar to `DPD_Lan_Therm.py`, but uses Langevin dynamics within the ESPResSo framework to construct polymer chains with specific bonded interactions.

- **nanogel builder**
  - `Nanogel.py`: Uses the pyiron framework to create and manipulate a nanogel structure. Constructs a diamond crystal structure of carbon atoms, adjusts atomic positions, and saves them for further simulation.

## Getting Started

### Prerequisites

Ensure you have the following libraries installed:
- `espressomd`
- `numpy`
- `pyiron`

### Installing

Clone the repository:
```bash
gh repo clone smadurga/DPD_nanogel



import numpy as np
from pyiron import Project
pr = Project(path='structures')

a = 1*20.78460969
structure = pr.create_ase_bulk('C',crystalstructure="diamond",a=a,cubic=True)
structure.set_repeat([4,4,4])

positions = structure.positions
positions = positions-15*a/8

with open('positions.npy', 'wb') as f:
    np.save(f, positions)
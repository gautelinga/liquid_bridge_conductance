# Liquid bridge conductance

This repository contains the code for computing the conductance of liquid bridges in granular porous media, as described in our paper [Reis et al., Adv. Wat. Res. 198 (2025)](https://doi.org/10.1016/j.advwatres.2025.104914). The code takes surface meshes from [Surface Evolver](https://kenbrakke.com/evolver/evolver.html) as input, meshes the interior using [PyMeshLab](https://github.com/cnr-isti-vclab/PyMeshLab) and [TetGen](https://www.wias-berlin.de/software/tetgen/1.5/index.html), and solves Stokes flow in the resulting volumetric mesh using [FEniCS](https://fenicsproject.org/).

## Usage:

Fix mesh input from Surface Evolver:
```
mpirun -np 4 python3 mesh_fix.py --infolder Surface_Evolver_Bridges/ --outfolder Fixed_Meshes_3
```

Mesh interior:
```
mpirun -np 4 python3 mesh_interior.py --infolder Fixed_Meshes_3/ --outfolder volume_meshes_3/
```

Solve flow:
```
mpirun -np 4 python3 stokes_slip.py volume_meshes_3/ output_4/
```

Analyze geometry:
```
python geometry_analyze.py --datafolder output_4/
```

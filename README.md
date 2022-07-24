# anymal_trotting
This repository contains the codes for building the memory of motion used in MEMMO project.

## Dataset
The dataset used to train the memory of motion can be found in https://gepettoweb.laas.fr/memmo/


## Dependencies
To use this repository, the following packages are needed:
```python
pinocchio (https://github.com/stack-of-tasks/pinocchio)
crocoddyl (https://github.com/loco-3d/crocoddyl)
caracal (https://github.com/cmastalli/caracal)
SL1M (https://github.com/loco-3d/SL1M)
tensorflow (https://www.tensorflow.org/)
```

## Notebooks
### The following examples are provided:

For generating the contact sequence:
```python
notebooks/SL1M_generate_trot.ipynb
```

For generating the whole body motion data:
```python
notebooks/generate_data_grid_standard_noise.ipynb
```

For training the memory of motion:
```python
notebooks/learn_data_trot_pybullet_direct.ipynb (For the direct model)
notebooks/learn_data_trot_pybullet_refined.ipynb (For the hierarchical model)
```

---
marp: true
---

# FLARE: Fast Learning of Atomistic Rare Events(for 0 to xx).md
> Jonathan Vandermause ([jonathan\_vandermause@g.harvard.edu](mailto:jonathan_vandermause%40g.harvard.edu))

Not theory involved below!

[toc]

# 1.  Introduction

### 1. Objectives:** **
**Train 2+3-body Gaussian process models on *****ab initio***** force data. **

* Open-source
* Bayesian machine learning (Gaussian Process)
* Training fast potentials on accurate DFT data

![image](images/1ocZtyAphDX6___gBCOrW7J_79mjw3oWFg-kCIucdHo.png)

### 2. Installation
```python
!pip install --upgrade mir-flare

# for flare++
!pip install --upgrade flare_pp

# However, the automatically process failed in my laptop, so I have to install it manually:
git clone https://github.com/mir-group/flare_pp.git
cd flare_pp/
python setup.py install
# Work!
```
#### import
```python
from flare import gp, struc, output, predict, md, otf, env,\
  otf_parser
from flare.kernels import mc_simple
from flare.utils import md_helper

import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
from ase.visualize import view
```
### 3. First trial

* **build** `gp_model = gp.GaussianProcess(parameters...)` 
* **Input training data:**`gp_model.update_db(traing_x, traning_y)`
* **training:**`gp_model.set_L_alpha() # update covariance matrix`
* **prediction: **`gp_model.likelihood`
* Turning Hyperparameters to improve prediction:
   *  `gp_model.hyps = np.array([0.01, 1, 0.001, 1, 0.2])`
   * Correspond:  ** Signal variance of the 2-body kernel (in eV) \* Length scale of the 2-body kernel (in A) \* Signal variance of the 3-body kernel (in eV) \* Length scale of the 3-body kernel (in A) \* Noise hyperparameter (in eV/A)**
   * Human words: the hyps are noise level(less than errors scale 0.1eV/A) and length scale 1 A for **2-body **and **3-body**, respecitively.
   * Optimize these Hyps: `gp_model.train(print_progress=True)  # gradient descent approach`
### 4. Learning curve
* **Optimize** Hyps
* **Build** `gp_model = gp.GaussianProcess(parameters...)`
* **Training**: `for atom in mole: ; gp_model_update_db(training_x, training_y); gp_model.set_L_alpha()`
* **Prediction**: `pred_forces, stds = predict.predict_on_structure(validation_structure, gp_model)`
* **Calculate** **error**: ` mse = np.mean(np.abs(pred_forces - validation_forces))`
* **Save** data: **MSE**, **Time**, **number** **of** **atomic** **environment**
   * `validation_errors = np.zeros(n_atoms)`
   * `prediction_times = np.zeros(n_atoms)`
   * `validation_count = 0`
   * `...`
   * `validation_errors[validation_count] = mae`
   * `prediction_times[validation_count] = time1 - time0`
   * `validation_count += 1`
注意：每一个原子都是一个environment(而不是a molecular)
* **Plot**: `plt.plot(validation_errors), plt.ylabel(''), plt.xlabel(''), plt.show()`
   * `plt.plot(prediction_times), plt.ylabel(''), plt.xlabel(''), plt.show()`
   * **valication\_errors/prediction\_times** are lists contain error value of each atom. Its size equal to number of atoms

![image](images/fvvEQr2ncewSap2nknfQ2iRgZ7NzkiV-Nt9Z4nwlMl8.png)

![image](images/cJaIa__ZSCxlFFGiWwDlek3U8OV3PbUqMOw7L0w_smg.png)

### 5. Toward large scale applications: Pd/Ag

* How does palladium dissolve into a silver surface?
* Relevant to the selectivity/reactivity tradeoff in bimetallic catalysts
* Cool, as we can see below, the FF forces seems really uncertainty!

![image](images/gPKPOez9ar-kXNv5HfVXD1OwC19pzFiuvMYt5BH_Htg.png)



#### * Steps1:  Set up the initial structure.

#### * Steps2: Set up a GP model
   * **Build**: `gp_model = gp.GaussianProcess(parameters,...)`
#### * Steps3: Set up an OTF training object.
   * `otf_model = otf.OTF(parameters, ... ,dft_input = input_file_name)`
#### * Steps4: Perform the simulation!
   * `otf_model.run()`
#### * Parsing the OTF output file.
   * `output_file = 'otf_run.out'`
`otf_trajectory = otf_parser.OtfAnalysis(output_file)`
   * ...

![image](images/0UpMty302aISWu4fqa3j67sX5rrDUnUT2Nb3O90CAxw.png)



# 2. Prepare data

### 1. Date from ase
Transfer your data from ase `Atoms` to `Structure.from_ase_atoms()`

```python
import ase.io
from flare.struc import Structure

frames = ase.io.read('db.xyz@:')
trajectory = []
for atoms in frames:
    trajectory.append(Strucuture.from_ase_atoms(atoms))
```
# 
# 3. Training fisrt Gaussian process from AIMD
![image](images/vFiv7lRu22yYVbapjIPin8hbUIRZlYp9NEMK2y8rsok.png)

### 1. Setting up a Gaussain Process Object
```python
from flare.gp import GaussianProcess
```
* Important parameters for GP
   * `kernels=['twobody', 'threebody'] # 2-body and 3-body system`
   * `hyps = [0.01, 0.01, 0.01, 0.01, 0.01]  # initial guess hyps(can be optimized later), last hyps is noise variance`
   * `cutoffs =  {'twobody':7, 'threebody':3} # cutoff values for 2,3-body, repectively`

```python
gp = GaussianProcess(kernels=['twobody', 'threebody'],
hyps=[0.01, 0.01, 0.01, 0.01, 0.01],
cutoffs = {'twobody':7, 'threebody':3},
hyp_labels=['Two-Body Signal Variance','Two-Body Length Scale','Three-Body Signal Variance',
'Three-Body Length Scale', 'Noise Variance'])
```
### 2. Extracting the Frames from a previous AIMD run(Optional)

* I like to use ASE, So just transfer data from ase `Atoms` to `Structure.from_ase_atoms()`
### 3. Training your Gaussian Process
* Input arguments for training:
   * `rel_std_tolerance:The length scale for uncertainty expected in forces prediction.`
`rel_std_tolerance * hyps[-1] = cutoff value` for **the criteria of adding atoms to training set**.
   * `abs_std_tolerance`:  similar to `rel_std_tolerance`, but it's a absolute value and not respect to dataset.
default is 0. **WTF?**
* Pre-Training arguments

For using reasonable hyperparameters, and prevent test db for low diversity of atomic configuration.

   * `pre_train_on_skips`: Slice the input frames by `frames[::pre_train_on_skips]`
   * The results wil be stored in `gp_from_aimd.out.`
   * The resultant model will be stored in a `.json` file format which can be later loaded using the `GaussianProcess.from_dict()` method.

```python
from flare.gp_from_aimd import TrajectoryTrainer
TT = TrajectoryTrainer(frames=trajectory,
                    gp = gp,
                    rel_std_tolerance = 3,
                    abs_std_tolerance=0,
                    pre_train_on_skips=5)
TT.run()
print("Done!")
```


# 4. On-the-fly training using ASE
> "On the fly used to describe sth that is being changed while the process that the change affects is ongoing. 

> "In Colloquial used to mean sth that was not planned ahead, or changes while execution of same activity"

> ————wikipedia

### 1. Set up supercell with ASE


### 2. Set up FLARE calculator
set up Gaussian process model as introduced before

```python
from flare.gp import GaussianProcess
from flare.utils.parameter_helper import ParameterHelper

# set up GP hyperparameters
kernels = ['twobody', 'threebody'] # use 2+3 body kernel
parameters = {'cutoff_twobody': 5.0,
              'cutoff_threebody': 3.5}
pm = ParameterHelper(
    kernels = kernels,
    random = True,
    parameters=parameters
)

hm = pm.as_dict()
hyps = hm['hyps']
cut = hm['cutoffs']
print('hyps', hyps)

gp_model = GaussianProcess(
    kernels = kernels,
    component = 'mc', # If you are using ASE, please set to "mc" no matter for single-component or multi-component
    hyps = hyps,
    cutoffs = cut,
    hyp_labels = ['sig2','ls2','sig3','ls3','noise'],
    opt_algorithm = 'L-BFGS-B',
    n_cpus = 1
)
```
### 3. Set up DFT calculator

* we can use ASE calculators here

```python
for ase.calculators.aims import Aims
from ase.calculator.dftb import Dftb
dft_calc = Aims()
```
### 4. Set up On-The-Fly MD engine

1. Setup MD arguments

```python
from ase import units
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)

temperature = 500
MaxwellBoltzmannDistribution(super_cell, temperature * units.kB)
Stationary(super_cell)  # zero linear momentum
ZeroRotation(super_cell)  # zero angular momentum

md_engine = 'VelocityVerlet'
md_kwargs = {}
```
2. Set up otf parameters

```python
otf_params = {'init_atoms': [0, 1, 2, 3],
              'output_name': 'otf',
              'std_tolerance_factor': 2,
              'max_atoms_added' : 4,
              'freeze_hyps': 10,
              'write_model': 3} # If you will probably resume the training, please set to 3
```
3. Set up `ASE_OTF` training engine, and run

```python
from flare.ase.otf import ASE_OTF
test_otf = ASE_OTF(super_cell,
                   timestep = 1 * units.fs,
                   number_of_steps = 3,
                   dft_calc = lj_calc,
                   md_engine = md_engine,
                   md_kwargs = md_kwargs,
                   **otf_params)

test_otf.run()
```
4. check `otf.out` file
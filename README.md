# wheeledSim
## Overview
Maintainer: Sean J. Wang, sjw2@andrew.cmu.edu
## Installation
To install, first install dependencies. Clone repository and use pip to install

    git clone https://github.com/robomechanics/wheeledSim.git
    cd wheeledSim
    pip3 install .

### Required Dependencies
This package is built using Python 3. The following packages are required.
- [PyBullet](https://pybullet.org) Physics simulation engine
- [NumPy](https://numpy.org)
- [SciPy](https://scipy.org)
- [noise](https://pypi.org/project/noise)

### Optional Dependencies
The following packages are only necessary to run some examples.
- [PyTorch](pytorch.org)
- [matplotlib](https://matplotlib.org/)

## Examples
### scripts/simulation/run_simulator.py

Run the simulator with some default parameters. This will require roscore to be running. Will publish position, front-facing camera and a top-down heightmap

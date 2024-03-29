# TrajectoryInference

This small Python library implements basic functions to perform trajectory inference with Gaussian processes. It is still in development phase.

The core classes and functions are defined in the gp_code directory.

## Set up

We recommend using a virtual environment:


```
$ python3 -m venv ./venv
$ source ./venv/bin/activate
$ pip install -r requirements.txt
```

## Kernel visualization

You can visualize the effect of different kernels with the kernelVisualization script:

```
$ python3 examples/test_kernelVisualization.py
```

## Training the GP parameters in a given setup

```
$ python3 examples/test_train.py
```

## Running some inference examples
To display the mixture predictive distribution at a certain point, from past observations  
```
$ python3 examples/test_mixtureGP.py
```

To get an animation of the updates of one single trajectory conditioned on a fixed goal
```
$ python3 examples/test_animation_single_trajectory.py
```

To get an animation of the updates of the trajectories
```
$ python3 examples/test_animation_multi.py
```

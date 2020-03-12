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
$ python3 kernelVisualization.py
```

## Training the GP parameters

```
$ python3 train.py
```

## Running some prediction examples

```
$ python3 main.py
```

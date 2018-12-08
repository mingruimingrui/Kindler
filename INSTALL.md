# Installation guide

## Requirements

```
python=2.7 or 3.X
pytorch>=1.0
torchvision>=0.2.1
tqdm
requests
```

Kindler has been tested extensively on both `python2.7` and `python3.5`. Other `python3` version should work just as well.

The currently implementation requires `pytorch>=1.0` where the `extension.h` module is exposed which allows users to easily compile `.cpp` and `.cu` files for use in `pytorch`. At this moment of writing, `pytorch1.0` is still in beta so install `pytorch-nightly`.

```
# Random note, it is highly recommended that you use a virtual env for this
# No doubt you'd still want a pytorch-latest-stable at hand
conda install -c pytorch pytorch-nightly
```

## Installation

The easiest way to use **Kindler** is to install it as a package so do it as you would always do

```
git clone git@github.com:mingruimingrui/Kindler.git
cd Kindler
python setup.py build_ext install
```

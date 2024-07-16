<div align="center">

# Region-based Pose Tracking

</div>

## Description

Python bindings for the [M3T](https://github.com/DLR-RM/3DObjectTracking/tree/master/M3T) library are provided to perform real-time 3D object tracking.

## Installation

You first need to clone the repository main branch:

```bash
git clone https://github.com/TomRavaud/region_based_pose_tracking.git
cd region_based_pose_tracking
```

Then, you need to install the required dependencies. The project uses conda to manage the dependencies. You can create a new conda environment using the provided `environment.yaml` file:

```bash
conda env create -f environment.yaml
```

Finally, you can install our custom packages using the following commands:

```bash
pip install ./m3t_bindings
pip install ./m3t_ext
```

To remove the conda environment, you can use the following command:

```bash
conda remove -n pym3t --all
```

## Datasets

Datasets used in this work are [RBOT](https://www.mi.hs-rm.de/~schwan/research/RBOT/)
and [BCOT](https://ar3dv.github.io/BCOT-Benchmark/).
Our code assumes that the datasets are downloaded and extracted in the `data` folder.

## Usage

You can run the provided example (on RBOT) using the following command:

```bash
python -m pym3t_ext.scripts.track
```

Note that if you intend to run the scripts headless, you can use the tool `Xvfb`:

```bash
Xvfb :1 -screen 0 640x480x24 &
export DISPLAY=:1
```
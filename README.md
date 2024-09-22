<div align="center">

# Region-based Pose Tracking

</div>

## Description

Python bindings for the [M3T](https://github.com/DLR-RM/3DObjectTracking/tree/master/M3T) library are provided to perform real-time 3D object tracking.
Moreover, we provide an extension to the M3T library to replace the traditional histogram-based segmentation with deep learning-based alternatives (pixel-wise
segmentation using a MLP and line-wise segmentation using a 1D U-Net).

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

## Models' weights

Trained parameters for the segmentation models are
to be downloaded and extracted in the `weights` folder. You can find their latest version [here](https://drive.google.com/drive/folders/1zmQhv8lmP1kQ_x5EHczd-JnJsvo0vQPR?usp=drive_link).

In addition, our models make use of the MobileSAM pretrained model. You can download the weights from
the [MobileSAM repository](https://github.com/ChaoningZhang/MobileSAM).

## Usage

You can run the provided tracking example (on RBOT) using the following command:

```bash
python -m pym3t_ext.scripts.track
```

Other scripts are provided to run a tracking method
on a whole dedicated dataset (`pym3t_ext.scripts.evaluate`) and to compute performance scores at each frame (`pym3t_ext.scripts.compute_metrics`). Each script comes with its own set of parameters defined in the `configs` folder.

Note that if you intend to run the scripts headless, you can use the tool `Xvfb`:

```bash
Xvfb :1 -screen 0 640x480x24 &
export DISPLAY=:1
```
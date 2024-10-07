# DensePose Evaluation Metrics

This repository provides all the necessary code to evaluate DensePose metrics on the COCO 2014 minival dataset.

## Features
- **Evaluation Only**: This repository focuses solely on evaluation metrics. It does not include any scripts for data preprocessing or model training.
- **Submodule Integration**: The code is designed to be used as a submodule within other repositories. It is notably used as a submodule of [MeshPose](https://github.com/Snapchat/MeshPose/tree/main).

## Installation

To use this repository, clone it as a submodule into your project.

## Prerequisites

To download the required data files for evaluation, navigate to the `DensePoseData` directory and run the following commands:

```bash
cd DensePoseData/
bash get_densepose_uv.sh
bash get_eval_data.sh
cd ..
```

## Dependencies

This code requires the following Python packages to be installed:

- `scipy`
- `torch`
- `h5py`
- `pycocotools`

Make sure to install these dependencies before running the code. You can install them using `pip` by running the following command:

```bash
pip install scipy torch h5py pycocotools
```

This code is derived from [detectron2](https://github.com/facebookresearch/detectron2/tree/main). Please see the `NOTICE` file for additional details.

---

For more details, feel free to visit the [MeshPose repository](https://github.com/Snapchat/MeshPose/tree/main) or the original [detectron2 repository](https://github.com/facebookresearch/detectron2/tree/main).

# ai_utils
This package contains different wrappers for the AI algorithms used in the labs to speed up development and 
improve research quality.


<!-- TABLE OF CONTENTS -->

  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#description">Description</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#dependencies">Dependencies</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li>
      <a href="#utilities">Utilities</a>
      <ul>
          <li><a href="#find-package-path">Find package path</a></li>
          <li><a href="#run-tests">Run tests</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#authors">Authors</a></li>
  </ol>


## Description
The package contains the following algorith wrappers.
- **Image segmentation**
  - Yolact++ (Instance segmentation)
  - Yolact Edge (Instance segmentation for edge devices)
  - Mask2Former (Panoptic segmentation)
- **Classification**
  - MMT
- **Pose detection** 
  - Mediapipe (hand pose)

## Getting started

### Dependencies
Each wrapper depends on its own AI module package. (e.g., ```YolactInference``` depends on ```Yolact```). \
We generally use a setup with CUDA 11.3 and Nvidia drivers 470. If you have this setup install the following pytorch dependences:
``` commandline
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2 -f https://download.pytorch.org/whl/torch_stable.html
```
### Installation

To install ai_utils on your system clone the repo, open a terminal in the main directory and run the following command (source your virtual env if you use them):
```
python3 -m pip install .
```
Instead if you want to install the package in "editable" or "develop" mode (to prevent the uninstall/install of the package at each pkg update) you have can run the following command:

```
python3 -m pip install -e .
```
#### AI modules installation

To use the ai_utils library in your environment you can refer to the following documentation. \
First of all, be sure that your python build command is properly upgraded:
``` commandline
  pip install --upgrade build
```

- Algorihm installations documentation

  - [Yolact++ installation](docs/yolact/yolact_installation.md)
  - [Mask2Former installation](docs/mask2former/mask2former_installation.md)
  - [MMT installation](docs/mmt/mmt_installation.md)
  - [Mediapipe installation](docs/mediapipe/mediapipe_installation.md)
  - [MiDaS installation](docs/midas/midas_installation.md)
## Usage
To use the ai_utils library be sure to have installed the base algorithm as explained in [Install Algorithms](#install-algorithms).

- Algorithms setup documentation:

  - [Yolact++ setup](docs/yolact/yolact_setup.md)
  - [Mask2Former setup](docs/mask2former/mask2former_setup.md)
  - [MMT setup](docs/mmt/mmt_setup.md)
  - [Mediapipe setup](docs/mediapipe/mediapipe_setup.md)
  - [MiDaS setup](docs/midas/midas_setup.md)

## Utilities
### Find package path

If you are using a standard Cmake package configuration you should save your network weights or config files 
inside your package under a custom directory ```e.g., YOUR_PACKAGE_PATH/weights/YOUR_WEIGHT)```. \
You'll probably need your package path in your code to use the AI algorithms (e.g., to load neural network weights).
If the files are saved as has been declared above you can use the following command to retrieve the package path:
``` python
    import os
    pkg_dir_name = '/' + os.path.join(*os.path.abspath(__file__).split('/')[:-2])
```
NOTE: in ROS2 packages are installed in the install directory of your workspace so the command to use to retrieve weight
dir is:
```python
    import os
    pkg_dir_name = '/' + os.path.join(*__file__.split('/')[:-5]) + '/share/PACKAGE_NAME/'
```
Anyway the path depends on how you install your package, the above command follow the standard convention of ROS2.

### Run Tests
To run the tests in the ```tests``` directory you have to install pytest:
```
    sudo apt-get install pytest
```
Before running the tests you have to download the neural network weights from: 
- [Yolact weights](https://drive.google.com/file/d/1SOTmS1UMBOSmnGbeVVRooTAOXYrhBvZZ/view?usp=sharing)
- [Mask2former weights](https://drive.google.com/file/d/1jdU_F01kVx5fO2yqeqt8IOMZ3-IpV7TN/view?usp=sharing)
- [Mask2former net_config directory](https://drive.google.com/drive/folders/1nnJ4R2Hx3n4eZ7pGGpUbOBX4d6dLMKkE?usp=sharing)

Put the weights and the config files in directories and change the path of weights in the tests scripts. All these passages are explained in the documentation: [Mask2Former setup](docs/mask2former/mask2former_setup.md) and [Yolact++ setup](docs/yolact/yolact_setup.md). You can put directories directly in ai_utils package.

Finally, you can run tests from the terminal. Open a terimal in the ai_utils directory and type the following command:
```
    python -m pytest .
```

Warnings are not meaningful in general.

## License

Distributed under the ```GPLv3``` License. See [LICENSE](LICENSE) for more information.

## Authors

The package is provided by:

- [Federico Rollo](https://github.com/FedericoRollo) [Mantainer]
- [Andrea Zunino](https://github.com/andreazuna89)



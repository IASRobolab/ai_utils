# ai_utils
This package is aimed to contain different wrappers for the AI algorithms used in the labs to speed up development and 
improve research quality.

## Algorithms
- **Image segmentation**
  - Yolact++ (Instance segmentation)
  - Mask2Former (Panoptic segmentation)
- **Classification**
  - MMT
- **Pose detection** 
  - Mediapipe (hand pose)


## How to
- [Install ai_utils](#install-ai_utils-module)
- [Install Algorithms](#install-algorithms)
- [Setup and Run Algorithms](#setup-and-run-algorithms)
- [Find package directory path](#find-package-path)
- [Run Tests](#run-tests)

# Install ai_utils module 

To install ai_utils on your system clone the repo, open a terminal in the main directory and run the following command (source your virtual env if you use them):
```
python3 -m pip install .
```
Instead if you want to install the package in "editable" or "develop" mode (to prevent the uninstall/install of the package at each pkg update) you have can run the following command:

```
python3 -m pip install -e .
```


# Install Algorithms 
To use the ai_utils library in your local pip you can refer to the following documentation depending on what you want to install. \
Be sure that your python build command is upgraded:
``` commandline
  pip install --upgrade build
```
### Algorihm installations documentation:

- [Yolact++ installation](docs/yolact/yolact_installation.md)
- [Mask2Former installation](docs/mask2former/mask2former_installation.md)
- [MMT installation](docs/mmt/mmt_installation.md)
- [Mediapipe installation](docs/mediapipe/mediapipe_installation.md)

# Setup and Run Algorithms
To use the ai_utils library be sure to have installed the base algorithm as explained in [Install Algorithms](#install-algorithms).

### Algorithms setup documentation:

- [Yolact++ setup](docs/yolact/yolact_setup.md)
- [Mask2Former setup](docs/mask2former/mask2former_setup.md)
- [MMT setup](docs/mmt/mmt_setup.md)
- [Mediapipe setup](docs/mediapipe/mediapipe_setup.md)


# Find package path
If you are using a standard Cmake package configuration you should save your network weights or config files 
inside your package under a custom directory ```e.g., YOUR_PACKAGE_PATH/weights/YOUR_WEIGHT)```. \
You'll probably need your package path in your code to use the AI algorithms (e.g., to load neural network weights).
If the files are saved as has been declared above you can use the following command to retrieve the package path 
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

# Run Tests

To run the tests in the ```tests``` directory you have to install pytest:
```
    sudo apt-get install pytest
```
You can run tests from the terminal. Open a terimal in the ai_utils directory and type the following command:
```
    python -m pytest .
```

Warnings are not meaningful in general.





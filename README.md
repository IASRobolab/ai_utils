# ai_utils
This package is aimed to contain different wrapper for the AI algorithms used in the labs

## Algorithms
- Yolact++
- MMT
- Mask2Former

## Algoritm installations
If you want to setup the following algorithm in your local pip you can run the following commands in your bash depending on what you need to install.
Be sure that your pythonb build command is upgraded:
```
  pip install --upgrade build
```

### Yolact++
First of all, clone the custom yolact++ repository, which contains the setup.py file, in a chosen directory (this should not change after installation). This repository has been slightly modified by the Robolab Leonardo fellows. Python 3 is needed to run Yolact++.

To install Yolact++ on your pip environment, activate your environment (or install directly on system if you prefer) and run:
```
  cd YOUR_YOLACT_PATH/yolact
  python setup.py build develop
```

Moreover, to use Yolact++ you need to install DCNv2. There exists two versions of DCNv2 in the Yolact++ repository (DCNv2 and DCNv2_latest directories). Choose the version you need to use:
- DCNv2 is used for OLDER GPU architectures (compatible with older pytorch version)
- DCNv2_latest is used for NEWER GPU architectures (compatible with latest pytorch version)
please substitute YOUR_DCNv2_FOLDER with DCNv2 or DCNv2_latest in the following

To install it, run:
```
  cd YOUR_YOLACT_PATH/yolact/external/YOUR_DCNv2_FOLDER
  python setup.py build develop
```

### MMT
First of all, clone the custom MMT repository, which contains the setup.py file, in a chosen directory (this should not change after installation). This repository has been slightly modified by the Robolab Leonardo fellows.

To install MMT on your pip environment, activate your environment (or install directly on system if you prefer) and run:
```
  cd YOUR_MMT_PATH/MMT
  python setup.py install
```

### Mask2Former
First of all, clone the custom Mask2Former repository, which contains the setup.py file, in a chosen directory (this should not change after installation). This repository has been slightly modified by the Robolab Leonardo fellows.

To install Mask2Former on your pip environment, activate your environment (or install directly on system if you prefer). You need to clone also the detectron2 repository slightly modified by the Robolab Leonardo fellows for Mask2Former.

To setup your environment, run:
```
  # Detectron2 installation (you need to clone the Robolab repository first)
  cd YOUR_DETECTRON2_PATH/detectron2
  pip install -e .
  pip install git+https://github.com/cocodataset/panopticapi.git
  pip install git+https://github.com/mcordts/cityscapesScripts.git
  
  # Mask2Former installation (you need to clone the Robolab repository first)
  cd YOUR_Mask2Former_PATH/Mask2Former
  pip install -r requirements.txt
  cd mask2former/modeling/pixel_decoder/ops
  sh make.sh
  cd YOUR_Mask2Former_PATH/Mask2Former
  python setup.py build develop
```


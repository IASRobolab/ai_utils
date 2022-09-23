
# Mask2Former installation
First, clone the custom Mask2Former repository, which contains the setup.py file, in a chosen directory 
(this should not change after installation). This repository has been slightly modified by the Robolab Leonardo fellows.

To install Mask2Former on your pip environment, activate your environment (or install directly on system if you prefer).
You need to clone also the detectron2 repository slightly modified by the Robolab Leonardo fellows for Mask2Former.

To setup your environment, run:
```  commandline
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
  pip install -e .
```

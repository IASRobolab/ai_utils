# Yolact++ installation
First of all, clone the custom yolact++ repository, which contains the setup.py file, in a chosen directory (this should
not change after installation). This repository has been slightly modified by the Robolab Leonardo fellows. Python 3
is needed to run Yolact++.

Install the right pytorch dependences:
``` commandline
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2 -f https://download.pytorch.org/whl/torch_stable.html
```
To install Yolact++ on your pip environment, activate your environment (or install directly on system if you prefer)
and run:
``` commandline
  cd YOUR_YOLACT_PATH/yolact
  pip install -e .
```

Moreover, to use Yolact++ you need to install DCNv2. There exists two versions of DCNv2 in the Yolact++ repository 
(DCNv2 and DCNv2_latest directories). Choose the version you need to use:
- DCNv2 is used for OLDER GPU architectures (compatible with older pytorch version)
- DCNv2_latest is used for NEWER GPU architectures (compatible with latest pytorch version)

To install it, substitute YOUR_DCNv2_FOLDER with DCNv2 or DCNv2_latest in the following:
``` commandline
  cd YOUR_YOLACT_PATH/yolact/external/YOUR_DCNv2_FOLDER
  pip install -e .
```

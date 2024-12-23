# PatchHAR: A MLP-Like Architecture for Efficient Activity Recognition Using Wearables


This is the official implementation for  [PatchHAR: A MLP-Like Architecture for Efficient
Activity Recognition Using Wearables 🔗](https://ieeexplore.ieee.org/abstract/document/10400955) publised in IEEE Transactions on Biometrics, Behavior, and Identity Science (T-BIOM).


## Setup

**1. Installation** 

* Setup conda environment (recommended).
```bash
# Create a conda environment
conda create -y -n har python=3.10

# Activate the environment
conda activate har

# Please refer to https://pytorch.org/ if you need a different cuda version
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install pther dependencies
pip install -r requirements.txt
```

**2. Data preparation**

Please see `./data` for WISDM datasets.



## Quick Start

Please refer to ``./run`` for more info about our scripts. 

**1. Training & Evaluation** 

```bash
GPU_ID=1 # replace it with your GPU ID
bash script/wisdm.sh ${GPU_ID}
```




## Citation

If you find this useful in your research, please consider citing:
```
@article{wang2024patchhar,
  title={Patchhar: A mlp-like architecture for efficient activity recognition using wearables},
  author={Wang, Shuoyuan and Zhang, Lei and Wang, Xing and Huang, Wenbo and Wu, Hao and Song, Aiguo},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
  year={2024},
  publisher={IEEE}
}

```
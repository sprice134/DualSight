# DualSight: Multi-Stage Instance Segmentation Framework

### Highlights
 - A multi-stage segmentation framework that combines lightweight model predictions, prompt generation, and the precision of SAM segmenting.
 - One of the largest open-access datasets of labeled metallic powder particles for instance segmentation. 

![alt text](https://github.com/sprice134/DualSight/blob/master/manuscriptData/dualSightImprovement.png)

# Installation
```
# Create and activate a Python virtual environment
python3 -m venv env
source env/bin/activate

# Download Dependencies
pip install torch torchvision torchaudi
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib ipykernel ultralytics

# Aquire SAM Weights
mkdir -p models
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -P models/

git clone https://github.com/sprice134/DualSight.git
```

# Dataset
```
powders/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```
# Bibtex
```
@article{price2025Dualsight,
  title={{DualSight}: Multi-Stage Instance Segmentation Framework for Improved Precision},
  author={Price, Stephen and Judd, Kiran and Tsaknopoulos, Kyle and Neamtu, Rodica and Cote, Danielle L.},
  journal={Scientific Reports},
  year={2025},
}


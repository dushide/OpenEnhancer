
## OpenEnhancer
OpenEnhancer is a framework for open-set learning through multiple feature fusion that integrates feature-specific sparse learning module and multi-feature attention fusion module. This repository contains the code and datasets used for the experiments in our paper.

### Installation
1. **Clone the repository:**
- git clone https://github.com/dushide/OpenEnhancer
- cd OpenEnhancer
2. **Set up the environment:**
Use Conda to create an environment with the required dependencies:
- conda create -n OpenEnhancer python=3.7.2
- conda activate OpenEnhancer
- pip install -r requirements.txt

### Datasets Preparation
- For all datasets, please obtain them from the following links: <https://drive.google.com/drive/folders/1-1tdfHOFvp_ka7BidSpRUfiqnauW6Nhn?usp=drive_link>;
- Download datasets from the provided links.
- Place the datasets in the `/data` directory.

### Quick Running and Results Reproduction
Run  `test.py` for multi-feature open-set learning tasks to see the OSCR performance. 

### Notes
 - Ensure all dependencies are installed, as listed in the requirements.txt.   
 - The code is designed to run on both **CPU** and **GPU**.   
 - For custom datasets, modify the dataset loader in loadMatData.py.

### Contact
If you have any questions, please feel free to contact dushidems@gmail.com at any time. Thanks.

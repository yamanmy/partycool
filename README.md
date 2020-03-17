# Partycool <img src="/example_images/partycool_super_smaller.jpg">
## Particle Size Analysis Package for All Microscopy Images 
This package can be used to analyze the particle size from microscopy images such as FEM, SEM images and optical microscopy.
We will try to implement also the recignition of AFM images through Machine Learning.
In addition, there is a visualization tools that can be used to plot the size distribution of the particles.
#### Contributers: Muammer yaman, Ximin Hu, Margherita Taddei, Yangwei Shi
#### Release date:

## Organization of the project
The project has the following structure:
   
   
    partycool/
      |- README.md
      |- partycool/
         |- __init__.py
         |- core.py
         |- tests/
            |- __init__.py
            |- test_core.py
      |- doc/
         |- technology_review.pdf
      |- example_images/
         |- train/
         |- darkfield/
         |- optic/
         |- sem/
      |- trial/
         |- .gitignore
         |- Pytorch-tutorial.ipynb
         |- Shape_detection.ipynb
         |- cifar_net.pth
         |- dog.jpg
         |- region_of_interest_scalebar.ipynb
         |- shape.png 
      |- .travis.yml
      |- .gitignore
      |- environment.yml
      |- LICENSE
      |- function_list.py
      |- partycool.ipynb    
      |- partycool_my.ipynb
      |- requirements.txt
      |- test


### How to install
```
pip install partycool
```
### Software dependencies
* Python3
### Preview of app 

* Categories of nanoparticles
<img src="/example_images/pie.png" width="450">

* Statistics distribution
<img src="/example_images/noninteractive.png" width="400">

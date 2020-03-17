# Partycool <img src="/example_images/partycool_super_smaller.jpg">

[![Build Status](https://travis-ci.com/yamanmy/partycool.svg?branch=master)

## Particle Size Analysis Package for Scanning Electron Microscopy Images 
Partycool is a package used to analyze nanoparticles size in details directly from electron microscopy images!

To use it just import the partycool package and upload your SEM image and you will get a partycool summary and interactive plots showing the size feautures of your particles.

This is a very handy tools for scientists working with nanoparticles in different field of chemistry , physiscs and biology. Knowing the precise size and size distribution of the sample is crucial for nanomaterials application in optoelectroics, clean energy, biomedicine and much more.

Lets walk together through the step that will bring from your electron microscopy image to the size analysis of your particles.

#### Contributers: Muammer Yaman, Ximin Hu, Margherita Taddei, Yangwei Shi
#### Release date: 2020-03-16

## Organization of the project
The project has the following structure:
   
   
    partycool/
      |- README.md
      |- partycool/
         |- __init__.py
         |- partycool.py
         |- tests/
            |- __init__.py
            |- test_partycool.py
            |- partycool.py           
         |- trial/
            |- partycool.ipynb
            |- watershed.ipynb
            |- Pytorch-partycool.ipynb    
            |- *.ipynb
      |- doc/
         |- technology_review.pdf
         |- partycool.docx
         |- Presentation.pptx
      |- example_images/
         |- README.md
         |- watershed_trail/
         |- cut_images/ 
            |- zoom/         
         |- train/
            |- darkfield/
            |- optic/
            |- sem/
            |- afm/
      |- example/
         |- user_case.ipynb
         |- infonanoparticle.html
      |- .travis.yml
      |- environment.yml
      |- LICENSE
      |- setup.py
      |- requirements.txt

The module code is inside `partycool/partycool.py` and contains all the functions necessary to pass from your input image to the size analysis output. Information about the functions are shown below.

### Module code

We place the module code in a file called `partycool.py` in the directory called
`partycool`. 
The module code can be implemented by typing in your jupyter notebook `import partycool`. 
All the library used to create the functions inside `partycool.py` are listed in `__init__.py`.
In the module code all functions are defined in lines that precede the lines that
use that function. This helps readability of the code, because you know that if
you see some name, the definition of that name will appear earlier in the file,
either as a function/variable definition, or as an import from some other module
or package.
The `boundary_detection` function is used to to distinguish the scale bar background with particle backgroundand uses the OpenCV threshold operatio to distinguish the scale bar background with particle background in the SEM image.
The `corner_detection` function is used to find the length of each pixel in nm. It uses the `dilated_image` function to find the brighter feautures corrisponding to the particle via the gaussian filter of OpenCV.
The image is then treated with the `img_pread` function that gives out the dilated and boundary cutted image.

<img src="/example_images/img_pread_output.png" width="500">

Then, the `contour_capture` function captures the contours from the given imgage and gives the average perimeters of the particles through `peri_avg` function.
The `shape_radar` function unify the result from `contour_capture` to the filtered image and gives as an output the image showing the size distribution of the particles. Monomers are shown in white, dimers in red and polymers in green.

<img src="/example_images/countour_capture_output.png" width="500">

The `partycool_summary` gives as output the final dataframe containing all the size analysis of the particles in the input image.
The `partycool_plots` gives the size analysis information in form of interactive plots using histograms and pie plot from Plotly.

At the end, we define the `watershed` function that will be used to distinguish close nanoparticles into monomers, dimers or polymers

<img src="/example_images/watershed_output.png" width="500">

### Preview of app 

* Categories of nanoparticles
<img src="/example_images/pie.png" width="600">

* Statistics distribution
<img src="/example_images/noninteractivenew.png" width="600">


### Major dependencies
* Python >= 3.6
   - OpenCV
   - Skimage
   - Plotly

### How to install & import
```
pip install partycool
```

```
from partycool import *
```

### Project Data

The data used to the develop the code are SEM images of gold nanoparticles on protein substrate taken from the Ginger Lab at the University of Washington. In the `example_images` repository we provide a good collection of SEM, TEM, AFM, opticaland dark field images that can be used by the user.

### Testing

All the function in the `partycool` code were tested using nosetests. The testing files are found the `tests` repository and each function is tested with the `test_partycool.py` using assert statements.

"
.......
----------------------------------------------------------------------
Ran 7 tests in 64.733s

OK
"


### Continuous integration 

Travis CI is deployed for continuous integration, code style checked through flake8

### Licensing 

Our code is open to public utilization but to protect it we used the MIT license. You can read the conditions of the license in the `LICENSE` file.

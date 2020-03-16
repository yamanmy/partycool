import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="partycool", # Replace with your own username
    version="0.0.1",
    author="Muammer yaman, Ximin Hu, Margherita Taddei, Yangwei Shi",
    author_email="author@example.com",
    description="Particle Size Analysis Package for All Microscopy Images",
    long_description="This package can be used to analyze the particle size from microscopy images such as FEM, SEM images and optical microscopy. We will try to implement also the recignition of AFM images through Machine Learning. In addition, there is a visualization tools that can be used to plot the size distribution of the particles.",
    long_description_content_type="text/markdown",
    url="https://github.com/yamanmy/partycool",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
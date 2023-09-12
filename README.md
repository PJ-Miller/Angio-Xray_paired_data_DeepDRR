# Angio-Xray_paired_data_DeepDRR

# Creation of Digitally Reconstructed Radiographs with and without the vessel information pointed out in the segemtation maps
This is the code to create paired X-rays and Angiography data from CTA scans. 


Two versions exists. One is a google colab script to execute the code online. 
The other is the python code run using polyaxon # link later



## DEEP DRR
This is a method to create paired data using the DeepDRR model [link](https://github.com/arcadelab/DeepDRR).
The installation requirements are ginven as:

DeepDRR requires an NVIDIA GPU, preferably with >11 GB of memory.

-   Install CUDA. Version 11 is recommended, but DeepDRR has been used with 8.0
-   Make sure your C compiler is on the path. DeepDRR has been used with gcc 9.3.0
-   We recommend installing pycuda separately, as it may need to be built. If you are using Anaconda, run  
'''
    conda install -c conda-forge pycuda
'''

to install it in your environment.

-   You may also wish to install PyTorch separately, depending on your setup.
-   Install from PyPI



## DATA

The dataset used to test was the CTA scan taken from the dongyang hospital and the KiTs challenge [link](https://www.sciencedirect.com/science/article/pii/S2352340922000130)


Hereby an image of the CTA scan and the later results with and without the vessel information. 
By changing the different materials, the results change.


Here an example of the CTA scan of the dongyang dataset, visualized in the imfusion software by enhancing the vessel information.

<test_image src="https://github.com/PJ-Miller/Angio-Xray_paired_data_DeepDRR/blob/main/images/CTA_scan.png" width=50% height=50%>


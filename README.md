# Non-Parametric Data Synthesis
EEC 289A Assignment 2

## Overview
This project aims at synthesizing new images by using the distribution of neighboring pixels. Within a neighborhood of a target pixel, this neighborhood will be compared with known image patches from the example image. Windowing kernel, gaussian kernel, and similarity threshold may need to be tuned for different images.

Index for synthesized images:
* cimg are without alpha value.
* dimg and img are with alpha value.
* pimg are gaussian applied after squared l2-norm.
* zimg are without alpha, gaussian applied after squared norm, and with different distribution estimate.
* zzimg are with the final patch selection method.

## Requirements
* Numpy
* Jupyter Notebook
* Matplotlib
* Pytorch (optional)

## Results
Through several revisions of patch sections, the synthesized images were satisfactory for the project. Hyperparameters could be tuned, but the results would be difficult to compare.

## Acknowledgements
This project is a course assignment for an unsupervised learning class taught by Yubei Chen, Spring 2024. 

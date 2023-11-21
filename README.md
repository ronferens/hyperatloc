# HyperAtLoc

## Intro
HyperAtLoc is the extension of the [AtLoc](https://cdn.aaai.org/ojs/6608/6608-13-9836-1-10-20200520.pdf) camera pose regressor using the with hypernetwork based on the HyperPose method.

* This repository is based on the original [AtLoc repository](https://github.com/BingCS/AtLoc).


* The repository has *two* branchs:
  * **main** - supports training and testing the single-scene AtLoc and HyperAtLoc models
  * **mshyperpose** - a dedicated branch to test and train the multi-scene MSHyperPose model

## Running the code
Please refer to the detailed instruction in the original [AtLoc repository](https://github.com/BingCS/AtLoc).

## Results
The following table details the results achieved by the HyperAtLoc and MSHyperPose models with references to the AtLoc and [MapNet](https://openaccess.thecvf.com/content_cvpr_2018/papers/Brahmbhatt_Geometry-Aware_Learning_of_CVPR_2018_paper.pdf) methods.
For each scene, we quantify performance by calculating median positional (in meters) and rotational errors (in degrees):

| Algorithm  | Loop 1 (mean, median) | Loop 2 (mean, median) | Full 1 (mean, median) | Full 2 (mean, median) |
|------------|-----------------------|-----------------------|------------------------|------------------------|
| MapNet     | 8.76, 3.46            | 5.79, 1.54            | 9.84, 3.96             | 4.91, 1.67             |
| AtLoc      | 8.86, 4.67            | 5.05, 2.01            | 8.61, 4.58             | 5.68, 2.23             |
| HyperAtLoc | **8.42, 3.30**        | **4.91, 1.33**        | **8.10, 3.04**         | **4.73, 1.27**         |




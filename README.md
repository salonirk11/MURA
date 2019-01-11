# MURA
MURA (musculoskeletal radiographs) is a large dataset of bone X-rays.

Existing methods use DenseNet169 on each class of organs to generate inference about the normality/abnormality of X-ray studies.

Aim here is to get better results using a model with lesser number of layers, preferably by image pre-processing and hyperparameter tuning.

## Preprocessing

The following preprocessing steps have been used to help the nural model generate more features.

* resize image to 224x224 without compromising the aspect ratio of the x-ray.
* Histogram equalisation
* Edge enhancement using Gaussian Blur unsharp mask.
* future work: segmentation of multiple x-rays in an image.

# Radio Galaxy Morphoogy Classification with Mask R-CNN 

This is an application of [Mask R-CNN](https://arxiv.org/abs/1703.06870) to Radio Galaxy Morphology Classification using data from [Radio Galaxy Zoo: ClaRAN - A Deep Learning Classifier for Radio Morphologies](https://github.com/chenwuperth/rgz_rcnn).

# Paper
Presented at the [2020 2nd International Workshop on Signal Processing and Machine Learning (WSPML 2020)](http://www.wspml.org/), November 20-22, 2020.

Published in the proceeedings of the 2020 4th International Conference on Vision, Image, and Signal Processing, ICVISP 2020.

Paper: [Radio Galaxy Morphology Classification with Mask R-CNN](https://dl.acm.org/doi/abs/10.1145/3448823.3448881)

# Installation

* Clone repo
* Download and unzip images from [here](https://www.dropbox.com/s/jezo3jim08u8hmc/galaxy_data.tar_gz?dl=0)

# Dataset
The top image below is an original flux density image obtained from the open databased maintained by Space Telescope Science Institute (third.ucllnl.org) with linear scaling while the bottom image shows the derived image with log scaling as used in CLARAN and this paper.

![](images/Figure2.JPG)

![](images/Figure3.JPG)

# Results
Annotated Radio Galaxy images with bounding boxes used for training Mask R-CNN.

Upper left 1C_1P, upper right 1C_2P, lower left 2C_2P and lower right 3C_3P 

![](images/Figure6.JPG)

Classification results with Mask-RCNN for the above images.

![](images/Figure7.JPG)



















# Author

Emin Alp Arslan [eminalparslan@gmail.com](eminalparslan@gmail.com)

# SSL-FER

Pytorch Implementation of the BMVC 2022 Paper "Revisiting Self-Supervised Contrastive
Learning for Facial Expression Recognition" 

The project is bulit based on https://github.com/sthalles/SimCLR


## About

<img src=misc/Picture1.png>

The success of most advanced facial expression recognition works relies heavily on large-scale annotated datasets. However, it poses great challenges in acquiring clean and consistent annotations for facial expression datasets. On the other hand, self-supervised contrastive learning has gained great popularity due to its simple yet effective instance discrimination training strategy, which can potentially circumvent the annotation issue. Nevertheless, there remain inherent disadvantages of instance-level discrimination, which are even more challenging when faced with complicated facial representations. In this paper, we revisit the use of self-supervised contrastive learning and explore three core strategies to enforce expression-specific representations and to minimize the interference from other facial attributes, such as identity and face styling. Experimental results show that our proposed method outperforms the current state-of-the-art self-supervised learning methods, in terms of both categorical and dimensional facial expression recognition tasks.

## Highlights

We explored three main promising solutions in Self-supervised learning-based Facial Expression Recognition (FER).

### ·Positives with Same Expression

✔ TimeAug - We added additional temporal shifts to generate positive pairs.

✔ FaceSwap - We swapped between faces to get same expression on different faces with a low computational cost.

### ·Negatives with Same Identity

✔ HardNeg - We sampled with a larger time-interval to avoid identity-related shortcuts. 
 
### ·False Negatives Cancellation

✔ MaskFN - We minimized the negative effects caused by false negatives using mouth-eye descriptors. 

## Dataset

We pretrained on [VoxCeleb1](https://mm.kaist.ac.kr/datasets/voxceleb/) dataset and evaluated on two Facial eExpression Recognition datasets, [AffectNet](http://mohammadmahoor.com/affectnet/) and [FER2013](https://www.kaggle.com/datasets/msambare/fer2013). We further evaluated on a Face Recognition dataset, [LFW](http://vis-www.cs.umass.edu/lfw/). In our experiment, we used the released version provided by scikit-learn. The usage can be found [here](https://scikit-learn.org/0.19/datasets/labeled_faces.html).

## Requirements

To run the script you'll need to install dlib (http://dlib.net) including its Python bindings, and OpenCV. 

You'll also need to obtain the trained model from
dlib:  
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2  
    http://dlib.net/files/mmod_human_face_detector.dat.bz2


## Trained Model

Models can be downloaded from [here]()

### Results
| Pretrained Methods | Datasets | EXPR-F1 | EXPR-Acc | Valence-CCC | Valence-RMSE | Arousal-CCC | Arousal-RMSE |
| :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | 
| Supervised | ImageNet | 56.7% | 56.6% | 0.563 | 0.462 | 0.480 | 0.376 |
| [BYOL](https://proceedings.neurips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html) | VoxCeleb1 | 56.3% | 56.4% | 0.560 | 0.460 | 0.462 | 0.386 |
| [MoCo-v2](https://arxiv.org/abs/2003.04297) | VoxCeleb1 | 56.8% | 56.8% | 0.570 | 0.454 | 0.486 | 0.378 |
| [SimCLR](http://proceedings.mlr.press/v119/chen20j.html) | VoxCeleb1 | 57.5% | 57.7% | 0.594 | 0.431 | 0.451 | 0.387 |
| [CycleFace](https://openaccess.thecvf.com/content/ICCV2021/html/Chang_Learning_Facial_Representations_From_the_Cycle-Consistency_of_Face_ICCV_2021_paper.html) | VoxCeleb1,2 | 48.8% | 49.7% | 0.534 | 0.492 | 0.436 | 0.383 |
| - | - | - | - | - | - | - | - | 
| TimeAug | VoxCeleb1 | 57.8% | 57.9% | 0.583 | 0.448 | 0.500 | 0.374 |
| TimeAug+HardNeg | VoxCeleb1 | 58.1% | 58.3% | 0.594 | 0.437 | 0.500 | 0.373 |
| TimeAug+HardNeg+[CutMix](https://openaccess.thecvf.com/content_ICCV_2019/html/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.html) | VoxCeleb1 | 58.3% | 58.4% | 0.542 | 0.463 | 0.508 | 0.368 |
| TimeAug+FaceSwap+MaskFN | VoxCeleb1 | 58.6% | 58.7% | 0.568 | 0.444 | 0.502 | 0.369  |
| TimeAug+HardNeg+FaceSwap | VoxCeleb1 | 58.8% | 58.9% | 0.601 | 0.429 | 0.514 | 0.367 |
| TimeAug+HardNeg+MaskFN | VoxCeleb1 | 58.9% | 58.9% | 0.578 | 0.448 | 0.493 | 0.370 |
| TimeAug+HardNeg+FaceSwap+MaskFN | VoxCeleb1 | 59.3% | 59.3% | 0.595 | 0.435 | 0.502 | 0.372 |


## Citing

Please cite the following paper if you use our methods in your research:

```
@inproceedings{shu2022revisiting,
  title={Revisiting Self-Supervised Contrastive Learning for Facial Expression Recognition},
  author={Shu, Yuxuan and Gu, Xiao and Yang, Guang-Zhong and Lo, Benny},
  booktitle={BMVC},
  year={2022}
}
```

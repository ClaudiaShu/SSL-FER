# SSL-FER

Pytorch Implementation of the BMVC 2022 Paper "Revisiting Self-Supervised Contrastive
Learning for Facial Expression Recognition" 

The project is bulit based on https://github.com/sthalles/SimCLR


## About

<img src=pics/Picture1.png>

The success of most advanced facial expression recognition works relies heavily on large-scale annotated datasets. However, it poses great challenges in acquiring clean and consistent annotations for facial expression datasets. On the other hand, self-supervised contrastive learning has gained great popularity due to its simple yet effective instance discrimination training strategy, which can potentially circumvent the annotation issue. Nevertheless, there remain inherent disadvantages of instance-level discrimination, which are even more challenging when faced with complicated facial representations. In this paper, we revisit the use of self-supervised contrastive learning and explore three core strategies to enforce expression-specific representations and to minimize the interference from other facial attributes, such as identity and face styling. Experimental results show that our proposed method outperforms the current state-of-the-art self-supervised learning methods, in terms of both categorical and dimensional facial expression recognition tasks.

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
## Requirements

To run the script you'll need to install dlib (http://dlib.net) including its Python bindings, and OpenCV. 

You'll also need to obtain the trained model from
dlib:  
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2  
    http://dlib.net/files/mmod_human_face_detector.dat.bz2

# Slimmable Multi-task Image Compression for Human and Machine Vision
![image](https://github.com/May-Yao/slimmable_image_compression/blob/main/framework_02.png)
Jiangzhong Cao; Ximei Yao; Huan Zhang; Jian Jin; Yun Zhang; Bingo Wing-Kuen Ling  
Slimmable Multi-Task Image Compression for Human and Machine Vision  
[[paper](https://ieeexplore.ieee.org/document/10080941)]
# Abstract
In the Internet of Things (IoT) communications, visual data is frequently processed among intelligent devices using artificial intelligence algorithms, replacing humans for analyzing and
decision-making while only occasionally requiring human’s scrutiny. However, due to high redundancy of compressive encoders, existing image coding solutions for machine vision are not efficient at runtime. To balance the rate-accuracy performance and efficiency of image compression for machine vision while attaining high-quality reconstructed images for human vision, this paper introduces a novel slimmable multi-task compression framework for human and machine vision in visual IoT applications. Firstly, the image compression for human and machine vision under the constraint of bandwidth, latency, computational resources are modelled as a multi-task optimization problem. Secondly, slimmable encoders are employed to multiple human and machine vision tasks in which the parameters of the sub-encoder for machine vision tasks are shared among all tasks and jointly learned. Thirdly, to solve the feature match between latent representation and intermediate features of deep vision networks, feature transformation networks are introduced as decoders of machine vision feature compression. Finally, the proposed framework is successfully applied to human and machine vision tasks’ scenarios, e.g., object detection and image reconstruction. Experimental results show that the proposed method outperforms baselines and other image compression approaches on machine vision tasks with higher efficiency (shorter latency) in two vision tasks’ scenarios while retaining comparable quality on image reconstruction.  
# Installation
The codebases are built on top of [CompressAI](https://github.com/InterDigitalInc/CompressAI) , [Slimmabe Networks](https://github.com/JiahuiYu/slimmable_networks) and [YOLOv3](https://github.com/ultralytics/yolov3/tree/master).
### Requirements

### Steps
1. 

# Citing
If you use our work in your research or wish to refer to the baseline results published here, please use the following BibTeX entries:  
@ARTICLE{10080941,  
  author={Cao, Jiangzhong and Yao, Ximei and Zhang, Huan and Jin, Jian and Zhang, Yun and Ling, Bingo Wing-Kuen},  
  journal={IEEE Access},   
  title={Slimmable Multi-Task Image Compression for Human and Machine Vision},   
  year={2023},  
  volume={11},  
  pages={29946-29958},  
  doi={10.1109/ACCESS.2023.3261668}}  
# License
CC 4.0 Attribution-NonCommercial International  
The software is for educaitonal and academic research purpose only.
